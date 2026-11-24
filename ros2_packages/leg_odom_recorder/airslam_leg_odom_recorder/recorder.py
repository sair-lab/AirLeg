from __future__ import annotations

import csv
import importlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import rclpy
from lowstate_republisher.msg import LowStateStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from .sources import LegInputs


DEFAULT_JOINTS = [
    f"{leg}_{joint}_joint"
    # Unitree LowState motor order of every supported robot (Go2, Go1): must match the simulation data and therefore
    # the order encoded by the checkpoint normalizer (= TRAIL common/leg_kinematics.LOWSTATE_LEG_ORDER).
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
LOWSTATE_PERIOD_NS = 8_000_000  # 125 Hz default; recorder / online node use leg_odometry.yaml (robot:) network_rate_hz


def stamp_ns(message, receive_ns: int) -> int:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return receive_ns
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else receive_ns


class StreamClock:
    """One time source per stream: header stamps if the first message has one, otherwise receive time.
    A later message with a missing header stamp (header mode) is rejected instead of silently switching clocks."""

    def __init__(self) -> None:
        self.use_header = None
        self.rejected = 0

    def stamp(self, message, receive_ns: int):
        value = stamp_ns(message, 0)
        if self.use_header is None:
            self.use_header = value > 0
        if not self.use_header:
            return receive_ns
        if value <= 0:
            self.rejected += 1
            return None
        return value


class Decimator:
    """Keeps the first message of every period slot (default LOWSTATE_PERIOD_NS: 500 Hz LowState -> 125 Hz)."""

    def __init__(self, period_ns: int = LOWSTATE_PERIOD_NS) -> None:
        self.period_ns = int(period_ns)
        self.next_ns = None

    def accept(self, stamp: int, key=None):
        """None if the slot is already taken, else the number of empty slots skipped before this message."""
        if self.next_ns is not None and stamp < self.next_ns:
            return None
        missing = 0 if self.next_ns is None else (stamp - self.next_ns) // self.period_ns
        if self.next_ns is None:
            self.next_ns = stamp
        self.next_ns += ((stamp - self.next_ns) // self.period_ns + 1) * self.period_ns
        return missing


class ChangeSampler:
    """Keeps a message when its joint values (key) differ from the previous message, i.e. at the instants the robot
    actually refreshed them (Go1: ~50 Hz values held in the 500 Hz lowstate stream); the stamp is the first appearance
    of the new values. Changes closer than min_period_ns to the last kept one are skipped; unchanged values (e.g. the
    robot standing still) are kept again after hold_period_ns so the stream has no gaps."""

    def __init__(self, period_ns: int, min_fraction: float = 0.5, hold_fraction: float = 1.5) -> None:
        self.period_ns = int(period_ns)
        self.min_period_ns = int(period_ns * min_fraction)
        self.hold_period_ns = int(period_ns * hold_fraction)
        self.last_key = None
        self.last_ns = None

    def accept(self, stamp: int, key=None):
        """None if the values did not change (or too soon), else the number of nominal periods skipped."""
        if key is not None and key == self.last_key:
            if self.last_ns is None or stamp - self.last_ns < self.hold_period_ns:
                return None
            self.last_ns = stamp                  # held value, re-emitted one period after the last kept sample
            return 0
        self.last_key = key
        if self.last_ns is not None and stamp - self.last_ns < self.min_period_ns:
            return None
        missing = 0 if self.last_ns is None else max(0, int(round((stamp - self.last_ns) / self.period_ns)) - 1)
        self.last_ns = stamp
        return missing


def make_sampler(robot: dict, rate_hz: int):
    """Lowstate sampler of leg_odometry.yaml (robot:) lowstate_sampling: 'slot' (default, fixed period slots) or 'on_change'."""
    mode = str(robot.get("lowstate_sampling", "slot"))
    period = 1_000_000_000 // int(rate_hz)
    if mode == "on_change":
        return ChangeSampler(period)
    if mode != "slot":
        raise ValueError(f"lowstate_sampling must be slot or on_change, got {mode}")
    return Decimator(period)


def joint_key(message) -> tuple:
    """Joint fields of a LowStateStamped (the IMU fields may be replaced, e.g. by an external IMU relay)."""
    return tuple(message.motor_q) + tuple(message.motor_dq) + tuple(message.motor_tau_est)


def find_trail_root() -> Path:
    """Locate AirSLAM2/core/3rdparty/TRAIL in source or install layouts."""
    for ancestor in Path(__file__).resolve().parents:
        for candidate in (
            ancestor / "core" / "3rdparty" / "TRAIL",
            ancestor / "src" / "AirSLAM2" / "core" / "3rdparty" / "TRAIL",
        ):
            if (candidate / "data_scripts" / "estimate_contacts" / "core.py").is_file():
                return candidate
    raise FileNotFoundError(
        "Cannot locate AirSLAM2/core/3rdparty/TRAIL from the recorder installation"
    )


def load_robot(robot: str) -> dict:
    """Robot descriptor AirSLAM2/robots/<robot>/leg_odometry.yaml (robot:) (or a leg_odometry.yaml (robot:) path) through TRAIL common/robot.py."""
    trail_root = find_trail_root()
    if str(trail_root.parent) not in sys.path:
        sys.path.insert(0, str(trail_root.parent))
    return importlib.import_module("TRAIL.common.robot").load_robot(robot or None)


class LegOdomRecorder(Node):
    def __init__(self) -> None:
        super().__init__("airslam_leg_odom_recorder")
        # robot: name under AirSLAM2/robots or a leg_odometry.yaml (robot:) path; the kinematics model, the Jacobian pressure
        # settings and the output rate default to its descriptor (contact_* parameters override them)
        self.declare_parameter("robot", "go2")
        self.robot = load_robot(str(self.get_parameter("robot").value).strip())
        pressure = self.robot["pressure"]
        # per-robot defaults of the parameters below: leg_odometry.yaml (robot:) `recorder:` (ROS parameters given at launch win)
        recorder = self.robot.get("recorder") or {}
        self.declare_parameter("map_root", "")
        self.declare_parameter("lowstate_topic", str(recorder.get("lowstate_topic", "/lowstate_stamped")))
        self.declare_parameter("estimate_contacts", bool(recorder.get("estimate_contacts", False)))
        self.declare_parameter("contact_estimation_method", str(pressure["method"]))
        self.declare_parameter("contact_model", "")
        self.declare_parameter("contact_pressure_scale", float(pressure["pressure_scale"]))
        self.declare_parameter("contact_pressure_mode", str(pressure["pressure_mode"]))
        self.declare_parameter("contact_normalize_sum", bool(pressure["normalize_sum"]))
        self.declare_parameter("contact_damping", float(pressure["damping"]))
        self.declare_parameter("contact_smooth_alpha", float(pressure["smooth_alpha"]))
        self.declare_parameter("contact_power", 1.0)
        self.declare_parameter("output_rate_hz", int(self.robot.get("network_rate_hz", 125)))
        self.declare_parameter("overwrite", bool(recorder.get("overwrite", False)))
        self.declare_parameter("queue_size", int(recorder.get("queue_size", 20000)))
        self.declare_parameter("flush_interval_sec", float(recorder.get("flush_interval_sec", 1.0)))
        self.declare_parameter("joint_names", DEFAULT_JOINTS)

        map_root_value = str(self.get_parameter("map_root").value).strip()
        if not map_root_value:
            raise ValueError("map_root is required")
        map_root = Path(map_root_value).expanduser()
        self.robot_dir = map_root / "leg_odom" / "robot"
        self.robot_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = bool(self.get_parameter("overwrite").value)
        self.joint_names = [str(value) for value in self.get_parameter("joint_names").value]
        if len(self.joint_names) != 12 or len(set(self.joint_names)) != 12:
            raise ValueError("joint_names must contain 12 unique joints (4 legs x hip/thigh/calf)")
        lowstate_topic = str(self.get_parameter("lowstate_topic").value).strip()
        if set(self.joint_names) != set(DEFAULT_JOINTS):
            raise ValueError(
                "the leg inputs deliver the 12 standard Unitree joint names (sources.py)"
            )
        self.output_rate_hz = int(self.get_parameter("output_rate_hz").value)

        self.records: queue.Queue = queue.Queue(
            maxsize=max(1, int(self.get_parameter("queue_size").value))
        )
        self.flush_interval = max(0.05, float(self.get_parameter("flush_interval_sec").value))
        self.stop_event = threading.Event()
        self.decimator = make_sampler(self.robot, self.output_rate_hz)
        self.lowstate_clock = StreamClock()
        self.counts = {
            "imu": 0, "joints": 0, "contacts": 0,
            "contact_errors": 0, "lowstate_errors": 0, "dropped": 0,
        }
        self.files = {}
        self.writers = {}
        self._open_csv("imu", [
            "sample", "stamp_ns", "time", "receive_ns",
            "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "orientation_x", "orientation_y", "orientation_z", "orientation_w",
            *[f"angular_velocity_covariance_{i}" for i in range(9)],
            *[f"linear_acceleration_covariance_{i}" for i in range(9)],
        ])
        joint_columns = ["sample", "stamp_ns", "time", "receive_ns"]
        for suffix in ("pos", "vel", "torque"):
            joint_columns.extend(f"{name}_{suffix}" for name in self.joint_names)
        self._open_csv("joints", joint_columns)

        estimate_contacts = bool(self.get_parameter("estimate_contacts").value)
        self.contact_estimator = None
        self.contact_leg_order = ()
        self.contact_joint_indices = ()
        if estimate_contacts:
            self.contact_estimator, self.contact_leg_order = self._make_contact_estimator()
            try:
                self.contact_joint_indices = tuple(
                    tuple(self.joint_names.index(f"{leg}_{joint}_joint")
                          for joint in ("hip", "thigh", "calf"))
                    for leg in self.contact_leg_order
                )
            except ValueError as exception:
                raise ValueError(
                    "joint_names does not contain the joints required by the contact estimator"
                ) from exception
        if self.contact_estimator is not None:
            self._open_csv("contacts", [
                "sample", "stamp_ns", "time", "receive_ns",
                "FL_pressure", "FR_pressure", "RL_pressure", "RR_pressure",
            ])
        # LowStateStamped on lowstate_topic, or the robot's separate joint / IMU topics (leg_odometry.yaml robot:
        # leg_inputs, sources.py) delivered as LowStateStamped-like samples
        self.inputs = LegInputs(self, self.robot, self.on_lowstate, lowstate_topic)
        self.get_logger().info(f"Recording {self.robot['name']} IMU and joints from {self.inputs.describe()} "
                               f"at {self.output_rate_hz} Hz")
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self._write_manifest(map_root, estimate_contacts)
        self.get_logger().info(f"Recording leg odometry data to {self.robot_dir}")

    def _make_contact_estimator(self):
        trail_root = find_trail_root()
        if str(trail_root.parent) not in sys.path:
            sys.path.insert(0, str(trail_root.parent))
        module = importlib.import_module("TRAIL.data_scripts.estimate_contacts")
        model_value = str(self.get_parameter("contact_model").value).strip()
        model_path = Path(model_value).expanduser() if model_value else Path(self.robot["model"])
        estimator = module.OnlineFootPressureEstimator(
            method=str(self.get_parameter("contact_estimation_method").value),
            model_path=model_path,
            pressure_scale=float(self.get_parameter("contact_pressure_scale").value),
            pressure_mode=str(self.get_parameter("contact_pressure_mode").value),
            normalize_sum=bool(self.get_parameter("contact_normalize_sum").value),
            damping=float(self.get_parameter("contact_damping").value),
            smooth_alpha=float(self.get_parameter("contact_smooth_alpha").value),
            power=float(self.get_parameter("contact_power").value),
        )
        self.get_logger().info(
            "Estimating foot pressure from joint torque with "
            f"method={self.get_parameter('contact_estimation_method').value}"
        )
        return estimator, tuple(module.LEG_ORDER)

    def _open_csv(self, name: str, header: list[str]) -> None:
        path = self.robot_dir / f"{name}.csv"
        mode = "w" if self.overwrite else "x"
        stream = path.open(mode, encoding="utf-8", newline="")
        writer = csv.writer(stream)
        writer.writerow(header)
        stream.flush()
        self.files[name] = stream
        self.writers[name] = writer

    def _write_manifest(self, map_root: Path, estimate_contacts: bool) -> None:
        manifest = {
            "format_version": 1,
            "created_unix_ns": time.time_ns(),
            "map_root": str(map_root.resolve()),
            "input_mode": self.inputs.mode,
            "topics": {**self.inputs.topics, "contacts": None},
            "contact_source": "estimated" if estimate_contacts else None,
            "contact_estimation_method": (
                str(self.get_parameter("contact_estimation_method").value)
                if estimate_contacts else None
            ),
            "contact_normalize_sum": bool(self.get_parameter("contact_normalize_sum").value),
            "contact_pressure_scale": float(self.get_parameter("contact_pressure_scale").value),
            "contact_pressure_mode": str(self.get_parameter("contact_pressure_mode").value),
            "contact_smooth_alpha": float(self.get_parameter("contact_smooth_alpha").value),
            "contact_damping": float(self.get_parameter("contact_damping").value),
            "robot": self.robot["name"],
            "joint_names": self.joint_names,
            "timestamp_policy": "header stamp; ROS receive time fallback when zero",
            "output_rate_hz": self.output_rate_hz,
            "lowstate_sampling": str(self.robot.get("lowstate_sampling", "slot")),
            # imu.csv holds the raw IMU (sensor frame); TRAIL common/frames.py rotates it into the base frame with
            # this extrinsic (robots/<robot>/leg_odometry.yaml (robot:) imu_extrinsic; {source: model} = the model's IMU site)
            "imu_frame": "sensor",
            "imu_extrinsic": dict(self.robot.get("imu_extrinsic") or {"source": "model"}),
        }
        path = self.robot_dir.parent / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _enqueue(self, name: str, row: list) -> None:
        try:
            self.records.put_nowait((name, row))
        except queue.Full:
            self.counts["dropped"] += 1
            if self.counts["dropped"] == 1 or self.counts["dropped"] % 100 == 0:
                self.get_logger().error(f"Recorder queue full; dropped={self.counts['dropped']}")

    def _record_imu(self, stamp: int, receive: int, gyroscope, accelerometer,
                    orientation, angular_covariance, acceleration_covariance) -> None:
        index = self.counts["imu"]
        self.counts["imu"] += 1
        self._enqueue("imu", [
            index, stamp, f"{stamp * 1e-9:.9f}", receive,
            *gyroscope, *accelerometer, *orientation,
            *angular_covariance, *acceleration_covariance,
        ])

    def _record_joints(self, stamp: int, receive: int, positions: list[float],
                       velocities: list[float], efforts: list[float]) -> None:
        index = self.counts["joints"]
        self.counts["joints"] += 1
        self._enqueue("joints", [
            index, stamp, f"{stamp * 1e-9:.9f}", receive,
            *positions, *velocities, *efforts,
        ])
        if self.contact_estimator is not None:
            try:
                q_leg = [
                    [positions[index] for index in indices]
                    for indices in self.contact_joint_indices
                ]
                tau_leg = [
                    [efforts[index] for index in indices]
                    for indices in self.contact_joint_indices
                ]
                pressure = self.contact_estimator.estimate(q_leg, tau_leg)
            except ValueError as exception:
                self.counts["contact_errors"] += 1
                if self.counts["contact_errors"] == 1:
                    self.get_logger().error(f"Cannot estimate foot pressure: {exception}")
                return
            contact_index = self.counts["contacts"]
            self.counts["contacts"] += 1
            self._enqueue("contacts", [
                contact_index, stamp, f"{stamp * 1e-9:.9f}", receive,
                *[float(value) for value in pressure],
            ])

    def on_lowstate(self, message: LowStateStamped) -> None:
        fields = (
            ("gyroscope", message.gyroscope, 3),
            ("accelerometer", message.accelerometer, 3),
            ("motor_q", message.motor_q, 12),
            ("motor_dq", message.motor_dq, 12),
            ("motor_tau_est", message.motor_tau_est, 12),
        )
        invalid = [name for name, values, size in fields if len(values) != size]
        if invalid:
            self.counts["lowstate_errors"] += 1
            if self.counts["lowstate_errors"] == 1:
                self.get_logger().error(
                    "Ignoring invalid LowStateStamped fields: " + ", ".join(invalid)
                )
            return

        receive = self.get_clock().now().nanoseconds
        stamp = self.lowstate_clock.stamp(message, receive)
        if stamp is None:
            if self.lowstate_clock.rejected % 1000 == 1:
                self.get_logger().warn(f"LowStateStamped without header stamp rejected ({self.lowstate_clock.rejected})")
            return
        if self.decimator.accept(stamp, joint_key(message)) is None:
            return
        self._record_imu(
            stamp, receive,
            [float(value) for value in message.gyroscope],
            [float(value) for value in message.accelerometer],
            (0.0, 0.0, 0.0, 1.0),
            (0.0,) * 9,
            (0.0,) * 9,
        )

        source_index = {name: index for index, name in enumerate(DEFAULT_JOINTS)}
        positions = [float(message.motor_q[source_index[name]]) for name in self.joint_names]
        velocities = [float(message.motor_dq[source_index[name]]) for name in self.joint_names]
        efforts = [float(message.motor_tau_est[source_index[name]]) for name in self.joint_names]
        self._record_joints(stamp, receive, positions, velocities, efforts)

    def _writer_loop(self) -> None:
        last_flush = time.monotonic()
        while not self.stop_event.is_set() or not self.records.empty():
            try:
                name, row = self.records.get(timeout=0.1)
                self.writers[name].writerow(row)
                self.records.task_done()
            except queue.Empty:
                pass
            if time.monotonic() - last_flush >= self.flush_interval:
                for stream in self.files.values():
                    stream.flush()
                last_flush = time.monotonic()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        deadline = time.monotonic() + 10.0
        while self.writer_thread.is_alive() and time.monotonic() < deadline:
            try:
                self.writer_thread.join(timeout=0.2)
            except KeyboardInterrupt:
                # ros2 launch may deliver another SIGINT while shutdown is in progress.
                continue
        for stream in self.files.values():
            stream.flush()
            stream.close()
        self.robot_dir.parent.mkdir(parents=True, exist_ok=True)
        status = self.robot_dir.parent / "recorder_status.json"
        pairer = getattr(self.inputs, "pairer", None)   # separate joint / IMU topics: pairing counts
        status.write_text(json.dumps({**self.counts, **({"inputs": pairer.counts} if pairer else {})}, indent=2),
                          encoding="utf-8")
        if rclpy.ok():
            self.get_logger().info(f"Recorder stopped: {self.counts}")

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LegOdomRecorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
