#!/usr/bin/env python3
"""Robot-agnostic RViz view of an S3 run (visual_odometry_node + the online leg odometry). No ground truth needed.

Only subscribes to topics the S3 pipeline already publishes; nothing in the pipeline depends on this node.
Inputs (all robots):
  * S3 pose of every frame: /AirSLAM/LatestOdometry/backend (with the back-end correction) and /AirSLAM/LatestOdometry
    (front end), camera poses in the VO map frame,
  * back-end keyframe path /AirSLAM/odometry/backend (re-optimised after every solve),
  * online leg odometry /airslam/leg_odom (LegOdometry: foot positions in the base frame, contact state, contact
    segment; its stamps are the display clock) and its camera <- base extrinsic /airslam/leg_odom/extrinsic,
  * optional: joint angles of the robot (leg_odometry.yaml robot: leg_inputs, else the Go2 LowStateStamped) for the
    robot model; the back-end gravity estimate /airslam/backend/gravity_up if the VO publishes it.
Up direction without it: the VO map z if the map is gravity aligned (the VO uses the IMU: the first camera's image y
axis points along -z), else the first camera's -y.
Camera only: without leg odometry `leg_wait` s after the first S3 pose (visual_only S3, robots without legs) the S3
poses are the clock and the camera is drawn alone (map -> camera_optical; no robot, footprints or ground following).
Without back-end poses for `leg_wait` s of front-end poses (S3 without back end, e.g. visual_only) the front-end poses
are the S3 poses.
Outputs:
  * TF world -> map (static, gravity levelled, origin on the ground under the start pose), world -> ground (the grid,
    following the stance feet), map -> <base link> (smoothed S3 pose, `delay` s behind the newest leg sample),
    <base link> -> camera_optical (camera only: map -> camera_optical); in model mode `none` also <base link> ->
    foot_<leg>,
  * /joint_states for robot_state_publisher (model modes mesh / skeleton),
  * /robot_viz/path (Path), /robot_viz/trajectory (Marker, coloured by age), /robot_viz/footprints (MarkerArray, one
    footprint per stance of each foot), /robot_viz/skeleton (Marker: lines between the links / to the feet),
    /robot_viz/camera_frustum (Marker).
The sparse map points are the VO's own topic (/AirSLAM/map), shown by robot_viz.rviz (off by default).
"""
import math
import os
import sys
from collections import deque

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from lowstate_republisher.msg import LegOdometry

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from robot_viz_geometry import (interpolate_pose, blend, mat_to_pose_stamped, mat_to_transform,  # noqa: E402
                                nearest, pose_to_mat, quat_to_mat, sec_to_stamp, stamp_to_sec)
from robot_viz_model import RobotDescription, robot_settings  # noqa: E402

FOOT_ORDER = ('FL', 'FR', 'RL', 'RR')                   # LegOdometry foot_b / contact order
FOOT_COLORS = ((1.0, 0.35, 0.3), (0.35, 0.9, 0.35), (0.3, 0.6, 1.0), (1.0, 0.85, 0.2))
SENSOR_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT)


def camera_intrinsics(path):
    """(fx, fy, cx, cy), (width, height) of cam0 in an AirSLAM camera.yaml, or None."""
    try:
        with open(path) as f:
            cam = yaml.safe_load(''.join(line for line in f if not line.startswith('%YAML')))
        return [float(v) for v in cam['cam0']['intrinsics'][:4]], (int(cam['image_width']), int(cam['image_height']))
    except Exception:
        return None


class RobotViz(Node):
    def __init__(self):
        super().__init__('robot_viz')
        p = self.declare_parameter
        robot_dir = p('robot_dir', '').value
        if not robot_dir or not os.path.isdir(robot_dir):
            raise RuntimeError(f'robot_dir {robot_dir!r} is not a directory (robots/<robot>)')
        self.settings = robot_settings(robot_dir)
        self.model = RobotDescription(self.settings['model'], self.settings['name'], p('model_mode', 'auto').value)
        self.base_frame = self.model.root or 'base_link'
        self.foot_radius = float(p('foot_radius', self.settings['foot_radius']).value)
        self.delay = float(p('delay', 0.6).value)          # minimum display delay (s); raised to the S3 latency
        rate_hz = float(p('rate_hz', 50.0).value)
        self.ground_follow = bool(p('ground_follow', True).value)
        self.ground_tau = float(p('ground_tau', 0.5).value)
        # the robot is drawn at C_disp * T_front, C = T_s3 T_front^-1 the back-end correction of each frame; C_disp
        # follows it with time constant correction_tau (s): a back-end solve moves the robot smoothly (<= 0: jumps)
        self.correction_tau = float(p('correction_tau', 1.0).value)
        # path: 'backend' = back-end keyframe path (re-drawn after every solve) + the displayed tail, footprints move
        # with their keyframe's correction; 'trail' = the displayed robot path only, append-only, footprints fixed
        self.path_mode = p('path_mode', 'backend').value
        if self.path_mode not in ('trail', 'backend'):
            raise RuntimeError('path_mode must be trail or backend')
        self.footprint_min_samples = int(p('footprint_min_samples', 3).value)
        leg_topic = p('leg_topic', '/airslam/leg_odom').value
        self.leg_wait = float(p('leg_wait', 3.0).value)    # s of S3 poses without leg odometry -> camera only
        self.intrinsics = camera_intrinsics(p('camera_config', self.settings['camera_config']).value)

        self.tf = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.path_pub = self.create_publisher(Path, '/robot_viz/path', latched)
        self.path_marker_pub = self.create_publisher(Marker, '/robot_viz/trajectory', latched)
        self.footprint_pub = self.create_publisher(MarkerArray, '/robot_viz/footprints', latched)
        self.skeleton_pub = self.create_publisher(Marker, '/robot_viz/skeleton', 10)
        self.frustum_pub = self.create_publisher(Marker, '/robot_viz/camera_frustum', 10)
        self.create_subscription(Odometry, p('s3_odometry_topic', '/AirSLAM/LatestOdometry/backend').value,
                                 self.on_odometry, 100)
        self.create_subscription(Odometry, p('s3_front_odometry_topic', '/AirSLAM/LatestOdometry').value,
                                 self.on_front_odometry, 100)
        self.create_subscription(Path, p('s3_path_topic', '/AirSLAM/odometry/backend').value, self.on_backend_path, 10)
        self.create_subscription(Vector3Stamped, p('s3_gravity_topic', '/airslam/backend/gravity_up').value,
                                 self.on_gravity, 10)
        self.create_subscription(LegOdometry, leg_topic, self.on_leg, SENSOR_QOS)
        self.create_subscription(TransformStamped, leg_topic + '/extrinsic', self.on_extrinsic,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.joint_names = self.settings['joint_names']
        # movable joints of the model that are not leg joints (e.g. the ANYmal inspection payload pan / tilt): held at 0,
        # else robot_state_publisher publishes no TF below them and RViz reports "No transform from [<link>]"
        self.fixed_joints = [j['name'] for j in self.model.joints
                             if j['type'] in ('revolute', 'continuous', 'prismatic') and j['name'] not in self.joint_names]
        if self.model.mode != 'none':
            self.subscribe_joints(self.settings['joint_source'])
        self.create_timer(1.0 / rate_hz, self.on_timer)
        self.create_timer(0.2, self.on_slow_timer)
        self.T_cam_base = None        # T_C_B: base -> camera (leg node extrinsic)
        self.camera_only = self.front_only = False
        self.reset()
        self.get_logger().info(
            f'robot {self.settings["name"]}: model {os.path.basename(self.settings["model"]) or "-"} as {self.model.mode}'
            + (f' ({len(self.model.missing)} mesh files missing)' if self.model.missing else '')
            + (f', joints from {self.settings["joint_source"]["topic"]} ({self.settings["joint_source"]["message"]})'
               if self.model.mode != 'none' else ''))

    def reset(self):
        self.joints = deque()          # (t, q [n]) in leg-input order
        self.leg = deque()             # leg samples (t, foot_b [4, 3], contact [4], segment [4]) not yet displayed
        self.leg_recent = deque()      # the last ~5 s of leg samples (feet for the display time / world init)
        self.t_newest = None           # newest leg stamp: the display clock
        self.s3_t, self.s3_T = [], []  # S3 camera poses in map
        self.front_buf = deque()       # front-end poses until the first back-end pose (or front_only)
        self.pending_rt, self.pending_front = {}, {}
        self.f_t, self.f_T, self.c_T = [], [], []
        self.C_disp = None
        self.trail, self.trail_len = [], 0.0
        self.kf_target, self.kf_disp, self.kf_ps, self.kf_moving = {}, {}, {}, set()
        self.latency = deque(maxlen=50)
        self.stance = [None] * 4
        self.footprints = [[] for _ in range(4)]
        self.footprints_dirty = self.path_dirty = False
        self.delay_eff = self.delay_logged = self.delay
        self.T_world_map, self.up_backend, self.ground_z, self.last_pub_t = None, None, None, None
        self.publish_static()

    # ---------------------------------------------------------------- inputs
    def subscribe_joints(self, src):
        names = src['names']
        if src['message'] == 'lowstate':
            from lowstate_republisher.msg import LowStateStamped
            self.create_subscription(LowStateStamped, src['topic'],
                                     lambda m: self.add_joints(m.header.stamp, list(m.motor_q)), SENSOR_QOS)
            return
        if src['message'] == 'anymal_state':
            from anymal_msgs.msg import AnymalState as Msg
            get = lambda m: m.joints   # noqa: E731
        else:
            Msg, get = JointState, (lambda m: m)
        perm = {}

        def on_joints(msg):
            j = get(msg)
            src_names = list(j.name)
            key = tuple(src_names)
            if key not in perm:   # source joints in leg-input order (by name), else the model's names
                wanted = names or self.joint_names
                missing = [n for n in wanted if n not in src_names]
                if missing:
                    self.get_logger().warn(f'joint message lacks {missing}: robot model not animated', once=True)
                    perm[key] = None
                else:
                    perm[key] = [src_names.index(n) for n in wanted]
            if perm[key] is not None:
                self.add_joints(msg.header.stamp, [float(j.position[i]) for i in perm[key]])
        self.create_subscription(Msg, src['topic'], on_joints, SENSOR_QOS)

    def add_joints(self, stamp, q):
        t = stamp_to_sec(stamp)
        self.joints.append((t, np.asarray(q, dtype=float)))
        while self.joints and self.joints[0][0] < t - 5.0:
            self.joints.popleft()

    def on_extrinsic(self, msg):
        if self.camera_only:
            return
        tr, q = msg.transform.translation, msg.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [tr.x, tr.y, tr.z]
        self.T_cam_base = T
        self.publish_static()
        self.get_logger().info('camera <- base extrinsic received from the leg node')

    def on_leg(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        if self.t_newest is not None and t < self.t_newest - 1.0:
            self.get_logger().warn('leg odometry time jumped back (bag restarted): resetting the view')
            self.reset()
        if self.camera_only:
            return
        sample = (t, np.array(msg.foot_b, dtype=float).reshape(4, 3), list(msg.contact), list(msg.contact_segment))
        self.leg.append(sample)
        self.leg_recent.append(sample)
        while self.leg_recent and self.leg_recent[0][0] < t - 5.0:
            self.leg_recent.popleft()
        while len(self.leg) > 5000:
            self.leg.popleft()
        self.t_newest = t if self.t_newest is None else max(self.t_newest, t)
        if self.T_world_map is None:
            self.try_init_world()

    def on_odometry(self, msg):
        if not self.front_only:
            self.add_s3(msg.header.stamp, pose_to_mat(msg.pose.pose))

    def add_s3(self, stamp, T):
        """One S3 pose (back end; front end if the S3 run has no back end, e.g. visual_only)."""
        t = stamp_to_sec(stamp)
        if self.s3_t and t < self.s3_t[-1] - 1.0:   # bag restarted (camera only: no leg clock to notice it)
            self.get_logger().warn('S3 time jumped back (bag restarted): resetting the view')
            self.camera_only = self.front_only = False
            self.reset()
        if self.s3_t and t <= self.s3_t[-1]:
            return
        if self.t_newest is not None:
            self.latency.append(self.t_newest - t)
        self.s3_t.append(t)
        self.s3_T.append(T)
        self.path_dirty |= self.path_mode == 'backend'
        self.match_frame(stamp, rt=T)
        if self.T_world_map is None:
            self.try_init_world()

    def on_front_odometry(self, msg):
        T = pose_to_mat(msg.pose.pose)
        self.match_frame(msg.header.stamp, front=T)
        if self.front_only:
            self.add_s3(msg.header.stamp, T)
            return
        # no back-end pose for leg_wait s of front-end poses: an S3 run without back end, the front end is the estimate
        self.front_buf.append((msg.header.stamp, T))
        span = stamp_to_sec(self.front_buf[-1][0]) - stamp_to_sec(self.front_buf[0][0])
        if not self.s3_t and span > self.leg_wait:
            self.front_only = True
            self.get_logger().info(f'no back-end poses {self.leg_wait:g} s into the S3 run: showing the front end')
            for stamp, Tf in self.front_buf:
                self.add_s3(stamp, Tf)
        while len(self.front_buf) > 2000:
            self.front_buf.popleft()

    def match_frame(self, stamp, rt=None, front=None):
        """Pairs the S3 and the front-end pose of a frame: correction C = T_s3 T_front^-1."""
        key = stamp.sec * 1000000000 + stamp.nanosec
        if rt is not None:
            self.pending_rt[key] = rt
        if front is not None:
            self.pending_front[key] = front
        if key in self.pending_rt and key in self.pending_front:
            rt, front = self.pending_rt.pop(key), self.pending_front.pop(key)
            t = key * 1e-9
            if not self.f_t or t > self.f_t[-1]:
                self.f_t.append(t)
                self.f_T.append(front)
                self.c_T.append(rt @ np.linalg.inv(front))
        for d in (self.pending_rt, self.pending_front):
            while len(d) > 200:
                d.pop(min(d))

    def on_backend_path(self, msg):
        for ps in msg.poses:
            t = round(stamp_to_sec(ps.header.stamp), 6)
            self.kf_target[t] = pose_to_mat(ps.pose)
            self.kf_moving.add(t)
        self.path_dirty |= self.path_mode == 'backend'

    def on_gravity(self, msg):
        up = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        if not np.all(np.isfinite(up)) or np.linalg.norm(up) < 0.5:
            return
        self.up_backend = up / np.linalg.norm(up)
        if self.T_world_map is not None and self.up_backend @ self.T_world_map[2, :3] < math.cos(math.radians(0.05)):
            self.set_world(self.up_backend)

    # ---------------------------------------------------------------- displayed pose
    def update_correction(self, t, dt):
        if not self.f_t:
            return
        target = self.c_T[max(0, int(np.searchsorted(self.f_t, t, side='right')) - 1)]
        if self.C_disp is None or self.correction_tau <= 0:
            self.C_disp = target.copy()
        else:
            self.C_disp = blend(self.C_disp, target, 1.0 - math.exp(-max(0.0, dt) / self.correction_tau))

    def display_pose(self, t):
        """Displayed camera pose in map at t: smoothed correction * front-end pose (fallback: the S3 pose)."""
        if self.f_t and self.C_disp is not None:
            return self.C_disp @ interpolate_pose(self.f_t, self.f_T, t)
        return interpolate_pose(self.s3_t, self.s3_T, t)

    def front_at(self, t, max_dt=0.5):
        if not self.f_t:
            return None
        i = int(np.searchsorted(self.f_t, t))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.f_t)), key=lambda j: abs(self.f_t[j] - t))
        return self.f_T[best] if abs(self.f_t[best] - t) <= max_dt else None

    def update_nodes(self, dt):
        a = 1.0 if self.correction_tau <= 0 else 1.0 - math.exp(-dt / self.correction_tau)
        for t in list(self.kf_moving):
            front = self.front_at(t)
            if front is None:
                continue
            target = self.kf_target[t] @ np.linalg.inv(front)
            cur = self.kf_disp.get(t)
            if cur is None:
                cur = self.C_disp.copy() if self.C_disp is not None else target
            C = blend(cur, target, a)
            self.kf_disp[t] = C
            self.kf_ps[t] = mat_to_pose_stamped(C @ front, 'map', t)
            if np.abs(C - target).max() < 1e-5:
                self.kf_moving.discard(t)
            self.path_dirty = self.footprints_dirty = True

    # ---------------------------------------------------------------- footprints
    def update_footprints(self, t):
        """Leg samples up to the display time t: each contact segment of a foot becomes one footprint (mean foot
        position over the segment at the displayed base pose, minus the foot radius)."""
        while self.leg and self.leg[0][0] <= t:
            ts, feet, contact, segment = self.leg.popleft()
            T_map_base = self.display_pose(ts) @ self.T_cam_base
            front = interpolate_pose(self.f_t, self.f_T, ts) @ self.T_cam_base if self.f_t else T_map_base
            for k in range(4):
                open_ = self.stance[k]
                if open_ is not None and (not contact[k] or open_[0] != segment[k]):
                    self.close_stance(k)
                    open_ = None
                if contact[k]:
                    p = T_map_base[:3, :3] @ feet[k] + T_map_base[:3, 3]
                    pf = front[:3, :3] @ feet[k] + front[:3, 3]
                    if open_ is None:
                        self.stance[k] = [segment[k], p.copy(), 1, pf.copy(), ts]
                    else:
                        open_[1] += p
                        open_[2] += 1
                        open_[3] += pf
                        open_[4] += ts

    def up(self):
        return self.T_world_map[2, :3] if self.T_world_map is not None else np.array([0.0, 0.0, 1.0])

    def close_stance(self, k):
        seg, total, n, total_front, total_t = self.stance[k]
        self.stance[k] = None
        if n >= self.footprint_min_samples:
            self.footprints[k].append((total_t / n, total_front / n, total / n - self.foot_radius * self.up()))
            self.footprints_dirty = True

    def footprint_positions(self, k):
        fps = self.footprints[k]
        if not fps:
            return np.zeros((0, 3))
        if self.path_mode != 'backend':
            return np.array([f[2] for f in fps])
        ts = np.array([f[0] for f in fps])
        pf = np.array([f[1] for f in fps])
        out = np.empty_like(pf)
        nodes = sorted(self.kf_disp)
        C_now = self.C_disp if self.C_disp is not None else np.eye(4)
        if nodes:
            Cs = np.array([self.kf_disp[t] for t in nodes])
            i = np.clip(np.searchsorted(nodes, ts, side='right') - 1, 0, len(nodes) - 1)
            out[:] = np.einsum('nij,nj->ni', Cs[i, :3, :3], pf) + Cs[i, :3, 3]
            late = ts > nodes[-1]
        else:
            late = np.ones(len(ts), dtype=bool)
        if late.any():
            out[late] = pf[late] @ C_now[:3, :3].T + C_now[:3, 3]
        return out - self.foot_radius * self.up()

    # ---------------------------------------------------------------- world frame
    def try_init_world(self):
        """world: z up, x along the start heading, origin on the ground under the start pose (the leg odometry feet in
        contact around the first S3 pose; camera only: the first camera position)."""
        if not self.s3_T:
            return
        t0 = self.s3_t[0]
        if self.camera_only:
            self.T_map_base0, self.feet_base0 = self.s3_T[0], [np.zeros(3)]
        else:
            if self.T_cam_base is None or not self.leg_recent:
                return
            if self.leg_recent[-1][0] < t0 or self.leg_recent[0][0] > t0 + 1.0:
                return
            self.T_map_base0 = self.s3_T[0] @ self.T_cam_base
            _, feet, contact, _ = nearest(self.leg_recent, t0)
            self.feet_base0 = [feet[k] for k in range(4) if contact[k]] or list(feet)
        height = self.set_world(self.up_backend if self.up_backend is not None else self.map_up())
        self.get_logger().info(f'world frame: {"camera" if self.camera_only else "base"} {height:.3f} m above the ground '
                               'at the first S3 pose')

    def map_up(self):
        """Up in map coordinates: map z if the map is gravity aligned (the first camera's image y axis, pointing down,
        is close to -z), else the first camera's -y (the map is the first camera frame)."""
        y = self.s3_T[0][:3, 1]
        return np.array([0.0, 0.0, 1.0]) if abs(y[2]) > 0.7 else -y

    def set_world(self, up):
        up = np.asarray(up, dtype=float) / np.linalg.norm(up)
        R_map_base = self.T_map_base0[:3, :3]
        ahead = R_map_base[:, 2] if self.camera_only else R_map_base[:, 0]   # optical axis / base x
        fwd = ahead - np.dot(ahead, up) * up
        fwd /= np.linalg.norm(fwd)
        R_world_map = np.stack([fwd, np.cross(up, fwd), up])
        base_height = 0.0 if self.camera_only else \
            -(float(np.mean([up @ (R_map_base @ f) for f in self.feet_base0])) - self.foot_radius)
        T = np.eye(4)
        T[:3, :3] = R_world_map
        T[:3, 3] = -R_world_map @ self.T_map_base0[:3, 3] + np.array([0.0, 0.0, base_height])
        self.T_world_map = T
        self.publish_static()
        return base_height

    def publish_static(self):
        tfs = []
        if self.T_cam_base is not None and not self.camera_only:
            tfs.append(mat_to_transform(np.linalg.inv(self.T_cam_base), self.base_frame, 'camera_optical', None))
        if self.T_world_map is not None:
            tfs.append(mat_to_transform(self.T_world_map, 'world', 'map', None))
        if tfs:
            self.tf_static.sendTransform(tfs)

    # ---------------------------------------------------------------- outputs
    def check_camera_only(self):
        if (not self.camera_only and self.t_newest is None and self.T_cam_base is None and self.s3_t
                and self.s3_t[-1] - self.s3_t[0] > self.leg_wait):
            self.camera_only = True
            self.get_logger().info(f'no leg odometry {self.leg_wait:g} s into the S3 run: camera only')
            if self.T_world_map is None:
                self.try_init_world()

    def on_timer(self):
        self.check_camera_only()
        clock = self.s3_t[-1] if self.camera_only and self.s3_t else self.t_newest
        if clock is None or not self.s3_T or (self.T_cam_base is None and not self.camera_only):
            return
        if len(self.latency) >= 5 and len(self.s3_t) >= 6:   # display delay >= S3 latency + pose interval (p95)
            gaps = np.diff(self.s3_t[-50:])
            need = max(self.delay, float(np.percentile(self.latency, 95) + np.percentile(gaps, 95)) + 0.03)
            self.delay_eff = need if need > self.delay_eff else max(need, self.delay_eff - 0.001)
            if abs(self.delay_eff - self.delay_logged) > 0.05:
                self.delay_logged = self.delay_eff
                self.get_logger().info(f'display delay {self.delay_eff:.2f} s (S3 output latency)')
        t = clock - self.delay_eff
        if self.last_pub_t is not None and t <= self.last_pub_t:
            return
        dt = 0.0 if self.last_pub_t is None else t - self.last_pub_t
        self.last_pub_t = t
        self.update_correction(t, dt)
        T_map_cam = self.display_pose(t)
        if self.camera_only:
            T_map_base = T_map_cam
            transforms = [mat_to_transform(T_map_cam, 'map', 'camera_optical', t)]
        else:
            T_map_base = T_map_cam @ self.T_cam_base
            transforms = [mat_to_transform(T_map_base, 'map', self.base_frame, t)]
        if not self.trail or t - self.trail[-1][0] >= 0.1:
            if self.trail:
                self.trail_len += float(np.linalg.norm(T_map_cam[:3, 3] - self.trail[-1][2]))
            self.trail.append((t, mat_to_pose_stamped(T_map_cam, 'map', t), T_map_cam[:3, 3].copy()))
            self.path_dirty = True
            self.footprints_dirty |= self.path_mode == 'backend'
        feet_t = None
        if not self.camera_only:
            self.update_footprints(t)
            feet_t = nearest(self.leg_recent, t) if self.leg_recent else None
        q = None
        if self.model.mode != 'none' and self.joints and not self.camera_only:
            q = nearest(self.joints, t)[1]
            js = JointState()
            sec_to_stamp(t, js.header.stamp)
            names = list(self.joint_names[:len(q)])
            js.position = [float(v) for v in q[:len(names)]] + [0.0] * len(self.fixed_joints)
            js.name = names + self.fixed_joints
            self.joint_pub.publish(js)
        if self.model.mode == 'none' and feet_t is not None:   # TF of the feet in place of a model
            for k, name in enumerate(FOOT_ORDER):
                T = np.eye(4)
                T[:3, 3] = feet_t[1][k]
                transforms.append(mat_to_transform(T, self.base_frame, f'foot_{name}', t))
        if not self.camera_only:
            self.publish_skeleton(t, q, feet_t)
        if self.T_world_map is not None:
            ground = np.eye(4)
            if self.ground_follow and feet_t is not None:
                T_world_base = self.T_world_map @ T_map_base
                _, feet, contact, _ = feet_t
                z = sorted(float(T_world_base[2, :3] @ feet[k] + T_world_base[2, 3])
                           for k in range(4) if contact[k]) or \
                    sorted(float(T_world_base[2, :3] @ f + T_world_base[2, 3]) for f in feet)[:2]
                target = float(np.mean(z)) - self.foot_radius
                a = 1.0 if self.ground_z is None else min(1.0, max(0.0, dt) / max(self.ground_tau, 1e-3))
                self.ground_z = target if self.ground_z is None else self.ground_z + a * (target - self.ground_z)
                ground[2, 3] = self.ground_z
            transforms.append(mat_to_transform(ground, 'world', 'ground', t))
        self.tf.sendTransform(transforms)

    def publish_skeleton(self, t, q, feet_t):
        """Lines between the links (skeleton mode) or from the base to the feet (no model), in the base frame."""
        if self.model.mode == 'mesh':
            return
        m = Marker()
        m.header.frame_id = self.base_frame
        sec_to_stamp(t, m.header.stamp)
        m.ns, m.id, m.type, m.action = 'skeleton', 0, Marker.LINE_LIST, Marker.ADD
        m.frame_locked = True
        m.pose.orientation.w = 1.0
        m.scale.x = 0.015
        m.color = ColorRGBA(r=0.15, g=0.15, b=0.15, a=1.0)
        pt = lambda v: Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))   # noqa: E731
        if self.model.mode == 'skeleton':
            poses = self.model.link_poses(dict(zip(self.joint_names, q)) if q is not None else {})
            for a, b in self.model.skeleton_edges():
                m.points += [pt(poses[a][1]), pt(poses[b][1])]
        elif feet_t is not None:
            for f in feet_t[1]:
                m.points += [pt(np.zeros(3)), pt(f)]
        if m.points:
            self.skeleton_pub.publish(m)

    def on_slow_timer(self):
        if self.path_mode == 'backend' and self.kf_moving:
            self.update_nodes(0.2)
        if self.path_dirty and self.s3_T:
            self.publish_path()
            self.path_dirty = False
        if self.footprints_dirty:
            self.publish_footprints()
            self.footprints_dirty = False
        if self.last_pub_t is not None and self.intrinsics is not None:
            self.publish_frustum()

    def publish_path(self):
        if self.path_mode == 'trail':
            poses = [ps for _, ps, _ in self.trail]
        else:
            nodes = sorted(self.kf_ps)
            t_last = nodes[-1] if nodes else -math.inf
            poses = [self.kf_ps[t] for t in nodes] + [ps for t, ps, _ in self.trail if t > t_last]
        if not poses:
            return
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = poses[-1].header.stamp
        path.poses = poses
        self.path_pub.publish(path)
        if len(poses) < 2:
            return
        m = Marker()   # line coloured by age: newest deep blue, older lighter
        m.header = path.header
        m.ns, m.id, m.type, m.action = 'trajectory', 0, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.points = [ps.pose.position for ps in poses]
        new, old = np.array([0.0, 0.35, 0.95]), np.array([0.80, 0.90, 1.0])
        ages = np.array([stamp_to_sec(ps.header.stamp) for ps in poses])
        w = (ages - ages[0]) / max(ages[-1] - ages[0], 1e-6)
        m.colors = [ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=1.0) for c in old + np.outer(w, new - old)]
        self.path_marker_pub.publish(m)

    def publish_footprints(self):
        markers = MarkerArray()
        for k, name in enumerate(FOOT_ORDER):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = f'footprints_{name}', 0, Marker.SPHERE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = max(0.04, 2.5 * self.foot_radius)
            m.scale.z = 0.012
            m.color = ColorRGBA(r=FOOT_COLORS[k][0], g=FOOT_COLORS[k][1], b=FOOT_COLORS[k][2], a=0.9)
            m.points = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in self.footprint_positions(k)]
            markers.markers.append(m)
        self.footprint_pub.publish(markers)

    def publish_frustum(self):
        (fx, fy, cx, cy), (w, h) = self.intrinsics
        d = 0.35
        corners = [Point(x=(u - cx) / fx * d, y=(v - cy) / fy * d, z=d) for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
        m = Marker()
        m.header.frame_id = 'camera_optical'
        m.ns, m.id, m.type, m.action = 'frustum', 0, Marker.LINE_LIST, Marker.ADD
        m.frame_locked = True
        m.pose.orientation.w = 1.0
        m.scale.x = 0.012
        m.color = ColorRGBA(r=0.2, g=0.8, b=1.0, a=1.0)
        origin = Point()
        for i in range(4):
            m.points += [origin, corners[i], corners[i], corners[(i + 1) % 4]]
        self.frustum_pub.publish(m)


def main():
    rclpy.init()
    node = RobotViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
