"""S3 online leg odometry ROS 2 bridge.

ROS Humble's rclpy (system Python) and the Torch/MuJoCo environment are different
interpreters, so this node only does ROS I/O and drives TRAIL/online_worker.py in the
Torch environment through length-prefixed pickles on its stdin/stdout.

In:  LowStateStamped or separate joint / IMU topics (sources.py; decimated to the robot's network_rate_hz like the recorder), VO keyframe nav_msgs/Path.
Out: lowstate_republisher/LegOdometry on `leg_topic` (base frame B, see TRAIL common/frames.py), latched T_C_B
     (camera <- base) on `leg_topic`/extrinsic.
"""
from __future__ import annotations

import math
import pickle
import time
import struct
import subprocess
import threading
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from lowstate_republisher.msg import LegOdometry, LowStateStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data

from .recorder import StreamClock, find_trail_root, joint_key, load_robot, make_sampler
from .sources import LegInputs


def to_stamp(t: float):
    from builtin_interfaces.msg import Time
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class OnlineLegOdom(Node):
    def __init__(self) -> None:
        super().__init__("airslam_online_leg_odom")
        self.declare_parameter("mode", "online_adapt")  # online_adapt | fusion_only
        self.declare_parameter("robot", "go2")
        self.declare_parameter("config", "")  # empty -> AirSLAM2/robots/<robot>/leg_odometry.yaml (online:)
        config = str(self.get_parameter("config").value)
        trail = find_trail_root()
        config = config or str(trail.parents[2] / "robots" / str(self.get_parameter("robot").value) / "leg_odometry.yaml")
        self.declare_parameter("worker_python", "python3")
        self.declare_parameter("lowstate_topic", "/lowstate_stamped")
        self.declare_parameter("leg_topic", "/airslam/leg_odom")
        self.declare_parameter("path_topic", "/AirSLAM/odometry")
        self.declare_parameter("frame_id", "leg_body")  # the robot base frame B
        self.declare_parameter("max_restarts", 5)  # worker crashes recovered before giving up
        mode = str(self.get_parameter("mode").value)
        if mode not in ("online_adapt", "fusion_only"):
            raise ValueError(f"mode must be online_adapt or fusion_only, got {mode}")
        leg_topic = str(self.get_parameter("leg_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        command = [str(self.get_parameter("worker_python").value), str(trail / "online_worker.py"),
                   "--config", config, "--mode", mode]
        self.command, self.trail = command, trail
        self.write_lock = threading.Lock()
        self.closing = False
        self.restarts = 0
        self.max_restarts = int(self.get_parameter("max_restarts").value)
        self.start_worker()
        # network rate of the robot (leg_odometry.yaml (robot:) network_rate_hz); the worker reads the same descriptor
        robot = load_robot(str(self.get_parameter("robot").value))
        self.rate_hz = int(robot.get("network_rate_hz", 125))
        self.decimator = make_sampler(robot, self.rate_hz)   # leg_odometry.yaml (robot:) lowstate_sampling (slot | on_change)
        self.clock = StreamClock()
        self.counts = {"nonfinite": 0, "gaps": 0, "missing_samples": 0}
        self.odom_publisher = self.create_publisher(LegOdometry, leg_topic, 100)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.extrinsic_publisher = self.create_publisher(TransformStamped, leg_topic + "/extrinsic", latched)
        # inputs are subscribed only once the worker is ready (network loaded, TensorRT engine loaded or built and
        # cached): readiness checks (common/ros_ready.py) see this node's subscriptions only then
        self.wait_worker_ready()
        # LowStateStamped, or the robot's separate joint / IMU topics (leg_odometry.yaml robot: leg_inputs, sources.py)
        self.inputs = LegInputs(self, robot, self.on_lowstate, str(self.get_parameter("lowstate_topic").value))
        path_topic = str(self.get_parameter("path_topic").value)
        self.create_subscription(PathMsg, path_topic, self.on_path, 10)
        if path_topic.endswith("/backend"):  # back-end supervision: the front-end path feeds the quality gate
            self.create_subscription(PathMsg, path_topic[:-len("/backend")],
                                     lambda message: self.on_path(message, "vo_path"), 10)
        # the back-end's total gyro bias (visual_odometry_node, backend.gyro_bias_total) for the worker
        self.create_subscription(Vector3Stamped, "/airslam/backend/gyro_bias", self.on_backend_bias, 10)
        self.reader = threading.Thread(target=self.read_worker, daemon=True)
        self.reader.start()
        self.get_logger().info(f"S3 leg odometry ({mode}) {self.inputs.describe()} -> {leg_topic}; worker: {' '.join(command)}")

    def start_worker(self) -> None:
        with self.write_lock:
            self.worker = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           cwd=str(self.trail))

    def wait_worker_ready(self) -> None:
        """Blocks until the worker's 'init' message (forwarding its log lines); raises if it exits before."""
        t0 = time.monotonic()
        stream = self.worker.stdout
        while True:
            header = stream.read(4)
            if len(header) < 4:
                raise RuntimeError(f"S3 leg worker exited before it was ready (code {self.worker.poll()})")
            kind, *payload = pickle.loads(stream.read(struct.unpack("<I", header)[0]))
            if kind == "log":
                self.get_logger().info(payload[0])
            elif kind == "init":
                self.publish_extrinsic(payload[0])
                self.get_logger().info(f"S3 leg worker ready after {time.monotonic() - t0:.1f} s")
                return

    def send(self, message) -> None:
        data = pickle.dumps(message, protocol=4)
        with self.write_lock:
            try:
                self.worker.stdin.write(struct.pack("<I", len(data)) + data)
                self.worker.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    def on_lowstate(self, message: LowStateStamped) -> None:
        if (len(message.gyroscope) != 3 or len(message.accelerometer) != 3 or len(message.motor_q) != 12
                or len(message.motor_dq) != 12 or len(message.motor_tau_est) != 12):
            return
        values = (list(message.gyroscope) + list(message.accelerometer) + list(message.motor_q)
                  + list(message.motor_dq) + list(message.motor_tau_est))
        if not all(math.isfinite(v) for v in values):  # never feed NaN / inf into the network
            self.counts["nonfinite"] += 1
            if self.counts["nonfinite"] % 1000 == 1:
                self.get_logger().warn(f"dropped non-finite LowStateStamped ({self.counts['nonfinite']})")
            return
        stamp = self.clock.stamp(message, self.get_clock().now().nanoseconds)
        if stamp is None:
            if self.clock.rejected % 1000 == 1:
                self.get_logger().warn(f"LowStateStamped without header stamp rejected ({self.clock.rejected})")
            return
        missing = self.decimator.accept(stamp, joint_key(message))
        if missing is None:
            return
        if missing > 0:  # network-rate slots without a message
            self.counts["gaps"] += 1
            self.counts["missing_samples"] += missing
            if self.counts["gaps"] % 100 == 1:
                self.get_logger().warn(f"lowstate gaps: {self.counts['gaps']} ({self.counts['missing_samples']} samples at {self.rate_hz} Hz)")
        self.send(("lowstate", stamp * 1e-9, list(message.gyroscope), list(message.accelerometer),
                   list(message.motor_q), list(message.motor_dq), list(message.motor_tau_est)))

    def on_path(self, message: PathMsg, kind: str = "path") -> None:
        times, poses = [], []
        for item in message.poses:
            p, q = item.pose.position, item.pose.orientation
            times.append(item.header.stamp.sec + item.header.stamp.nanosec * 1e-9)
            poses.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        order = sorted(range(len(times)), key=times.__getitem__)
        self.send((kind, [times[i] for i in order], [poses[i] for i in order]))

    def on_backend_bias(self, message: Vector3Stamped) -> None:
        v = message.vector
        self.send(("gyro_bias", message.header.stamp.sec + message.header.stamp.nanosec * 1e-9, [v.x, v.y, v.z]))

    def read_worker(self) -> None:
        stream = self.worker.stdout
        while True:
            header = stream.read(4)
            if len(header) < 4:
                code = self.worker.poll()
                if self.closing or not rclpy.ok():
                    return
                if self.restarts >= self.max_restarts:
                    self.get_logger().error(f"S3 leg worker exited (code {code}); restart limit reached, "
                                            f"no leg odometry from now on")
                    return
                self.restarts += 1
                self.get_logger().error(f"S3 leg worker exited (code {code}); restarting ({self.restarts}/{self.max_restarts})")
                time.sleep(1.0)
                self.start_worker()
                stream = self.worker.stdout
                continue
            try:
                kind, *payload = pickle.loads(stream.read(struct.unpack("<I", header)[0]))
            except Exception as error:  # never let one bad message stop the bridge
                self.get_logger().error(f"S3 leg worker message dropped: {error}")
                continue
            if kind == "odom":
                for record in payload[0]:
                    self.publish(*record)
            elif kind == "init":
                self.publish_extrinsic(payload[0])
            elif kind == "log":
                self.get_logger().info(payload[0])

    def publish(self, t, twist, variance, bias, foot, pressure, contact, segment, version) -> None:
        message = LegOdometry()
        message.header.stamp = to_stamp(t)
        message.header.frame_id = self.frame_id
        message.twist, message.variance, message.gyro_bias = twist, variance, bias
        message.foot_b, message.pressure = [float(v) for v in foot], [float(v) for v in pressure]
        message.contact = [int(v) for v in contact]
        message.contact_segment = [int(v) for v in segment]
        message.model_version = int(version)
        self.odom_publisher.publish(message)

    def publish_extrinsic(self, T_C_B) -> None:
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_left_rect"
        message.child_frame_id = self.frame_id
        t, q = message.transform.translation, message.transform.rotation
        t.x, t.y, t.z, q.x, q.y, q.z, q.w = (float(v) for v in T_C_B)
        self.extrinsic_publisher.publish(message)
        self.get_logger().info(f"S3 T_C_B = {[round(float(v), 5) for v in T_C_B]}")

    def close(self) -> None:
        self.closing = True
        self.send(("stop",))
        try:
            self.worker.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.worker.kill()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = OnlineLegOdom()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
