"""Leg odometry inputs: joints and IMU as LowStateStamped-like samples for the recorder and the S3 online node.

Two input modes, chosen by the robot descriptor (leg_odometry.yaml robot: leg_inputs):

* lowstate (default, Go2): lowstate_republisher/LowStateStamped on one topic (recorder.lowstate_topic), which already
  carries the IMU and the 12 joints in the Unitree motor order.
* separate topics (e.g. ANYmal / GrandTour): a joint topic (anymal_msgs/AnymalState or sensor_msgs/JointState) and a
  sensor_msgs/Imu topic. The IMU is linearly interpolated at every joint stamp (the two streams need not be phase
  locked); a joint sample outside the received IMU span or between IMU samples more than max_gap_s apart is dropped.
  The joints are reordered to the Unitree order FR FL RR RL x hip thigh calf by leg_inputs.joints.names (the source
  joint names in that order). Gyro and accelerometer stay in the IMU frame (rotated into base axes by the readers,
  TRAIL common/frames.py, robot: imu_extrinsic).

    leg_inputs:
      joints: {topic: /anymal/state_estimator/anymal_state, message: anymal_state, names: [RF_HAA, ...]}
      imu: {topic: /anymal/imu, max_gap_s: 0.005, resampled_topic: /airslam/imu_resampled}

    resampled_topic (optional): imu_resampler.py republishes the IMU at the joint stamps (the samples the leg network
    sees) as sensor_msgs/Imu for the VO (camera yaml imu_topic); the launch and the S2 preparation start it.

Both modes hand the callback objects with the LowStateStamped fields the consumers read: header.stamp, gyroscope,
accelerometer, motor_q, motor_dq, motor_tau_est.
"""
from __future__ import annotations

import bisect
from collections import deque
from types import SimpleNamespace

from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# Best effort (matches reliable and best-effort publishers) with a deep queue: the sensor-data default (depth 5) drops
# messages when bag playback delivers them in bursts (disk stalls) or a callback is briefly slow.
LEG_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT)

JOINT_MESSAGES = ("anymal_state", "joint_state")


def stamp_ns_of(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def leg_inputs(robot: dict) -> dict | None:
    """The robot's separate-topic inputs (validated), or None for the LowStateStamped input."""
    cfg = robot.get("leg_inputs")
    if not cfg:
        return None
    joints, imu = cfg.get("joints") or {}, cfg.get("imu") or {}
    if not joints.get("topic") or not imu.get("topic"):
        raise ValueError("leg_inputs needs joints.topic and imu.topic")
    message = str(joints.get("message", "joint_state"))
    if message not in JOINT_MESSAGES:
        raise ValueError(f"leg_inputs.joints.message must be one of {JOINT_MESSAGES}, got {message}")
    names = [str(name) for name in joints.get("names") or []]
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError("leg_inputs.joints.names must list 12 unique source joints (order FR FL RR RL x hip thigh calf)")
    return dict(joint_topic=str(joints["topic"]), joint_message=message, names=names,
                imu_topic=str(imu["topic"]), max_gap_ns=int(round(float(imu.get("max_gap_s", 0.005)) * 1e9)),
                resampled_topic=str(imu.get("resampled_topic") or ""))


class ImuInterpolator:
    """Pairs joint samples with the IMU linearly interpolated at their stamps (the rule of the GrandTour conversion:
    inside the IMU span and bracketing samples <= max_gap_ns apart). Joint samples wait until an IMU sample at or
    after their stamp has arrived; emit(stamp_ns, gyro, acc, q, dq, tau) receives the paired ones in stamp order."""

    def __init__(self, emit, max_gap_ns: int, max_pending: int = 2000, keep_ns: int = 1_000_000_000) -> None:
        self.emit, self.max_gap_ns = emit, max_gap_ns
        self.max_pending, self.keep_ns = max_pending, keep_ns
        self.imu_t: list[int] = []
        self.imu_v: list[tuple] = []
        self.pending: deque = deque()
        self.counts = {"joints": 0, "imu": 0, "paired": 0, "without_imu": 0, "imu_out_of_order": 0,
                       "joints_out_of_order": 0, "pending_overflow": 0}

    def add_imu(self, stamp_ns: int, gyro, acc) -> None:
        self.counts["imu"] += 1
        if self.imu_t and stamp_ns <= self.imu_t[-1]:
            self.counts["imu_out_of_order"] += 1
            return
        self.imu_t.append(stamp_ns)
        self.imu_v.append((tuple(float(v) for v in gyro), tuple(float(v) for v in acc)))
        self._drain()

    def add_joints(self, stamp_ns: int, q, dq, tau) -> None:
        self.counts["joints"] += 1
        if self.pending and stamp_ns <= self.pending[-1][0]:
            self.counts["joints_out_of_order"] += 1
            return
        self.pending.append((stamp_ns, q, dq, tau))
        while len(self.pending) > self.max_pending:   # no IMU for a long time
            self.pending.popleft()
            self.counts["pending_overflow"] += 1
        self._drain()

    def _drain(self) -> None:
        while self.pending and self.imu_t and self.imu_t[-1] >= self.pending[0][0]:
            t, q, dq, tau = self.pending.popleft()
            values = self._interpolate(t)
            if values is None:
                self.counts["without_imu"] += 1
            else:
                self.counts["paired"] += 1
                self.emit(t, values[0], values[1], q, dq, tau)
        # keep the IMU samples a later joint stamp can still need (one before the oldest pending / last paired stamp)
        horizon = (self.pending[0][0] if self.pending else self.imu_t[-1] if self.imu_t else 0) - self.keep_ns
        cut = max(0, bisect.bisect_left(self.imu_t, horizon) - 1)
        if cut:
            del self.imu_t[:cut], self.imu_v[:cut]

    def _interpolate(self, t: int):
        ts = self.imu_t
        if not ts or t < ts[0] or t > ts[-1] or len(ts) < 2:
            return None
        j = min(max(bisect.bisect_left(ts, t), 1), len(ts) - 1)
        t0, t1 = ts[j - 1], ts[j]
        if t1 - t0 > self.max_gap_ns:
            return None
        w = (t - t0) / max(t1 - t0, 1)
        (g0, a0), (g1, a1) = self.imu_v[j - 1], self.imu_v[j]
        gyro = [(1 - w) * x0 + w * x1 for x0, x1 in zip(g0, g1)]
        acc = [(1 - w) * x0 + w * x1 for x0, x1 in zip(a0, a1)]
        return gyro, acc


def lowstate_like(stamp_ns: int, gyro, acc, q, dq, tau):
    stamp = SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp), gyroscope=list(gyro), accelerometer=list(acc),
                           motor_q=list(q), motor_dq=list(dq), motor_tau_est=list(tau))


class LegInputs:
    """Subscribes the robot's leg odometry inputs on `node` and calls callback(message) with LowStateStamped (lowstate
    mode) or LowStateStamped-like samples (separate topics)."""

    def __init__(self, node, robot: dict, callback, lowstate_topic: str) -> None:
        self.cfg = leg_inputs(robot)
        self.callback = callback
        if self.cfg is None:
            from lowstate_republisher.msg import LowStateStamped
            if not lowstate_topic:
                raise ValueError("lowstate_topic is required without leg_inputs")
            self.mode = "lowstate"
            self.topics = {"lowstate": lowstate_topic, "imu": None, "joints": None}
            node.create_subscription(LowStateStamped, lowstate_topic, callback, LEG_QOS)
            return
        from sensor_msgs.msg import Imu
        self.mode = "joints_imu"
        self.topics = {"lowstate": None, "imu": self.cfg["imu_topic"], "joints": self.cfg["joint_topic"]}
        self.perm = None
        self.pairer = ImuInterpolator(lambda *sample: callback(lowstate_like(*sample)), self.cfg["max_gap_ns"])
        if self.cfg["joint_message"] == "anymal_state":
            from anymal_msgs.msg import AnymalState as JointMessage
        else:
            from sensor_msgs.msg import JointState as JointMessage
        node.create_subscription(Imu, self.cfg["imu_topic"], self.on_imu, LEG_QOS)
        node.create_subscription(JointMessage, self.cfg["joint_topic"], self.on_joints, LEG_QOS)

    def describe(self) -> str:
        if self.mode == "lowstate":
            return self.topics["lowstate"]
        return f"{self.topics['joints']} ({self.cfg['joint_message']}) + {self.topics['imu']} (sensor_msgs/Imu)"

    def on_imu(self, message) -> None:
        v, a = message.angular_velocity, message.linear_acceleration
        self.pairer.add_imu(stamp_ns_of(message.header.stamp), (v.x, v.y, v.z), (a.x, a.y, a.z))

    def on_joints(self, message) -> None:
        joints = message.joints if self.cfg["joint_message"] == "anymal_state" else message
        names = list(joints.name)
        if self.perm is None or [names[i] if i < len(names) else None for i in self.perm] != self.cfg["names"]:
            missing = sorted(set(self.cfg["names"]) - set(names))
            if missing:
                raise ValueError(f"joint message on {self.cfg['joint_topic']} lacks {missing}")
            self.perm = [names.index(name) for name in self.cfg["names"]]
        p = self.perm
        q = [float(joints.position[i]) for i in p]
        dq = [float(joints.velocity[i]) for i in p]
        tau = [float(joints.effort[i]) for i in p]
        self.pairer.add_joints(stamp_ns_of(message.header.stamp), q, dq, tau)
