#!/usr/bin/env python3
from __future__ import annotations

import csv
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _trail_on_path() -> None:
    """core.py is also loaded by file path (ROS recorder): make TRAIL's common/ importable."""
    import sys
    trail = Path(__file__).resolve().parents[2]
    if str(trail) not in sys.path:
        sys.path.insert(0, str(trail))


_trail_on_path()
# single source of the leg order; DEFAULT_MODEL = model of the default robot (robots/go2/leg_odometry.yaml (robot:))
from common.leg_kinematics import DEFAULT_MODEL, LEG_ORDER  # noqa: E402

JOINTS_PER_LEG = ("hip", "thigh", "calf")


@dataclass
class ContactEstimate:
    sample: np.ndarray
    time: np.ndarray
    pressure: np.ndarray
    score: np.ndarray
    forces: np.ndarray | None = None


def read_joint_csv(path: Path) -> dict[str, np.ndarray]:
    """读取带 sample/time 的 joints.csv，返回按列组织的 numpy 数组。"""
    _trail_on_path()
    from common.csv_table import read_numeric_csv
    return read_numeric_csv(path)


def get_sample_time(joints: dict[str, np.ndarray], default_dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(next(iter(joints.values())))
    sample = joints.get("sample", np.arange(n, dtype=np.float64))
    time = joints.get("time", np.arange(n, dtype=np.float64) * default_dt)
    return sample, time


def matrix_from_joint_columns(joints: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    """读取 4 条腿 x 3 关节矩阵，输出形状 [N, 4, 3]。"""
    per_leg = []
    for leg in LEG_ORDER:
        cols = [f"{leg}_{joint}_joint_{suffix}" for joint in JOINTS_PER_LEG]
        missing = [col for col in cols if col not in joints]
        if missing:
            raise KeyError(f"Missing joint columns: {missing}")
        per_leg.append(np.stack([joints[col] for col in cols], axis=-1))
    return np.stack(per_leg, axis=1)


def ema_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """指数滑动平均，降低力估计尖峰。"""
    if len(x) == 0:
        return x
    alpha = float(np.clip(alpha, 0.0, 0.999))
    from scipy.signal import lfilter
    # y[i] = alpha y[i-1] + (1 - alpha) x[i], y[0] = x[0]: initial state alpha x[0]
    return lfilter([1.0 - alpha], [1.0, -alpha], x, axis=0, zi=alpha * x[:1])[0].astype(x.dtype, copy=False)


def normalize_pressure(pressure: np.ndarray, pressure_scale: float, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """把四足压力转为接触分数；可选把每帧四足压力和缩放为 pressure_scale。"""
    raw = np.maximum(pressure, 0.0)
    denom = np.sum(raw, axis=1, keepdims=True)
    uniform = np.full_like(raw, 1.0 / raw.shape[1])
    score = np.divide(raw, denom + eps)
    score = np.where(denom > eps, score, uniform)
    return score, score * float(pressure_scale)


def prepare_kinematic_mujoco_xml(model_path: Path) -> Path:
    """去掉视觉 mesh，生成只用于运动学/Jacobian 的临时 MuJoCo XML。"""
    tree = ET.parse(model_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)

    for parent in root.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            if "mesh" in child.attrib or child.attrib.get("class") == "visual":
                parent.remove(child)

    tmp = tempfile.NamedTemporaryFile(prefix="go2_kinematic_", suffix=".xml", delete=False)
    tmp.close()
    out = Path(tmp.name)
    tree.write(out, encoding="utf-8", xml_declaration=False)
    return out


class MujocoFootForceEstimator:
    """用 tau = J(q)^T F 从关节力矩反推足端力。"""

    def __init__(self, model_path: Path = DEFAULT_MODEL, pressure_mode: str = "normal_abs") -> None:
        try:
            import mujoco
        except Exception as exc:
            raise RuntimeError("Jacobian contact estimation requires the mujoco Python package.") from exc

        self.mujoco = mujoco
        temp_model_path = prepare_kinematic_mujoco_xml(model_path)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(temp_model_path))
        finally:
            temp_model_path.unlink(missing_ok=True)  # the model is fully loaded; do not leak one file per estimator
        self.data = mujoco.MjData(self.model)
        self.pressure_mode = pressure_mode

        self.leg_dof_ids: dict[str, np.ndarray] = {}
        self.leg_qpos_ids: dict[str, np.ndarray] = {}
        self.leg_body_ids: dict[str, int] = {}
        for leg in LEG_ORDER:
            joint_names = [f"{leg}_{joint}_joint" for joint in JOINTS_PER_LEG]
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            if any(jid < 0 for jid in joint_ids) or body_id < 0:
                raise KeyError(f"Missing MuJoCo joints/body for leg {leg}")
            self.leg_dof_ids[leg] = np.asarray([self.model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
            self.leg_qpos_ids[leg] = np.asarray([self.model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
            self.leg_body_ids[leg] = int(body_id)

        self.base_qpos = np.zeros(self.model.nq, dtype=np.float64)
        if self.model.nq >= 7:
            self.base_qpos[3] = 1.0

    def set_joint_positions(self, q_leg: np.ndarray) -> None:
        self.data.qpos[:] = self.base_qpos
        for leg_idx, leg in enumerate(LEG_ORDER):
            self.data.qpos[self.leg_qpos_ids[leg]] = q_leg[leg_idx]
        self.mujoco.mj_forward(self.model, self.data)

    def foot_jacobian(self, leg: str) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.leg_body_ids[leg])
        return jacp[:, self.leg_dof_ids[leg]]

    def estimate_one(self, q_leg: np.ndarray, tau_leg: np.ndarray, damping: float) -> tuple[np.ndarray, np.ndarray]:
        self.set_joint_positions(q_leg)
        forces = []
        pressure = []
        for leg_idx, leg in enumerate(LEG_ORDER):
            j_pos = self.foot_jacobian(leg)
            a = j_pos.T
            tau = tau_leg[leg_idx]
            if damping > 0.0:
                force = np.linalg.solve(a.T @ a + damping * damping * np.eye(3), a.T @ tau)
            else:
                force = np.linalg.lstsq(a, tau, rcond=None)[0]
            forces.append(force)
            if self.pressure_mode == "normal_positive":
                pressure.append(max(force[2], 0.0))
            elif self.pressure_mode == "normal_negative":
                pressure.append(max(-force[2], 0.0))
            elif self.pressure_mode == "force_norm":
                pressure.append(float(np.linalg.norm(force)))
            else:
                pressure.append(abs(force[2]))
        return np.asarray(forces, dtype=np.float64), np.asarray(pressure, dtype=np.float64)

    def estimate_sequence(self, q: np.ndarray, tau: np.ndarray, damping: float) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((len(q), len(LEG_ORDER), 3), dtype=np.float64)
        pressure = np.zeros((len(q), len(LEG_ORDER)), dtype=np.float64)
        for i in range(len(q)):
            forces[i], pressure[i] = self.estimate_one(q[i], tau[i], damping)
        return forces, pressure


class ClosedFormFootForceEstimator:
    """MujocoFootForceEstimator with the closed-form leg kinematics (common/leg_kinematics.py): same interface
    and results (< 1e-9), vectorised over samples x legs."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, pressure_mode: str = "normal_abs") -> None:
        _trail_on_path()
        from common.leg_kinematics import LegKinematics, foot_forces
        self.kin = LegKinematics(model_path)
        self.foot_forces = foot_forces
        self.pressure_mode = pressure_mode

    def estimate_one(self, q_leg: np.ndarray, tau_leg: np.ndarray, damping: float) -> tuple[np.ndarray, np.ndarray]:
        _, forces, pressure = self.kin.single(q_leg, tau_leg, damping, self.pressure_mode)
        return forces, pressure

    def estimate_sequence(self, q: np.ndarray, tau: np.ndarray, damping: float) -> tuple[np.ndarray, np.ndarray]:
        return self.foot_forces(self.kin.foot_jacobians(q), tau, damping, self.pressure_mode)

    def foot_positions(self, q_leg: np.ndarray, site: str | None = None) -> np.ndarray:
        """Foot centres [4, 3] in the base (trunk) frame = body frame B of the leg odometry (relative to a trunk
        site, trunk axes, if `site` is given)."""
        return self.kin.single(q_leg, site=site)[0]


class EmaState:
    """Streaming form of ema_filter: y = x for the first sample, then alpha y + (1 - alpha) x.
    ``value`` may be restored to undo a rejected sample."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.value: np.ndarray | None = None

    def update(self, x: np.ndarray) -> np.ndarray:
        self.value = x if self.value is None else self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value


class OnlineFootPressureEstimator:
    """Stateful single-sample estimator shared by ROS recording and offline tools."""

    def __init__(
        self,
        method: str = "torque_norm",
        model_path: Path = DEFAULT_MODEL,
        pressure_scale: float = 1.0,
        pressure_mode: str = "normal_abs",
        normalize_sum: bool = False,
        damping: float = 1e-4,
        smooth_alpha: float = 0.9,
        power: float = 1.0,
        kinematics: str = "closed_form",
    ) -> None:
        if method not in {"jacobian", "torque_norm"}:
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.pressure_scale = float(pressure_scale)
        self.normalize_sum = bool(normalize_sum)
        self.damping = float(damping)
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 0.999))
        self.power = float(power)
        self.ema = EmaState(self.smooth_alpha)
        if kinematics not in {"closed_form", "mujoco"}:
            raise ValueError(f"Unsupported kinematics: {kinematics}")
        estimator_type = ClosedFormFootForceEstimator if kinematics == "closed_form" else MujocoFootForceEstimator
        self.jacobian_estimator = estimator_type(model_path, pressure_mode) if method == "jacobian" else None
        self.last_q: np.ndarray | None = None

    def estimate(self, q_leg: np.ndarray, tau_leg: np.ndarray) -> np.ndarray:
        """Return pressure in ``LEG_ORDER`` for arrays shaped ``[4, 3]``."""
        q_leg = np.asarray(q_leg, dtype=np.float64)
        tau_leg = np.asarray(tau_leg, dtype=np.float64)
        if q_leg.shape != (4, 3) or tau_leg.shape != (4, 3):
            raise ValueError("q_leg and tau_leg must both have shape [4, 3]")
        if not np.isfinite(q_leg).all() or not np.isfinite(tau_leg).all():
            raise ValueError("joint position/effort contains a non-finite value")

        self.last_q = q_leg
        if self.jacobian_estimator is not None:
            _, raw_pressure = self.jacobian_estimator.estimate_one(
                q_leg, tau_leg, self.damping
            )
        else:
            raw_pressure = np.power(
                np.maximum(np.linalg.norm(tau_leg, axis=-1), 0.0), self.power
            )

        filtered = self.ema.update(raw_pressure)
        # Match the offline estimator exactly: torque-norm is a relative
        # contact distribution, whereas Jacobian pressure remains metric-like
        # unless normalization is explicitly requested.
        if self.method == "torque_norm" or self.normalize_sum:
            _, pressure = normalize_pressure(filtered[None], self.pressure_scale)
            return pressure[0]
        return filtered * self.pressure_scale


def estimate_torque_norm(
    tau: np.ndarray,
    pressure_scale: float,
    smooth_alpha: float,
    power: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """旧版备用方法：只用每条腿三个关节力矩范数估计接触强度。"""
    leg_torque = ema_filter(np.linalg.norm(tau, axis=-1), smooth_alpha)
    strength = np.power(np.maximum(leg_torque, 0.0), float(power))
    denom = np.sum(strength, axis=1, keepdims=True)
    uniform = np.full_like(strength, 1.0 / strength.shape[1])
    score = np.divide(strength, denom + eps)
    score = np.where(denom > eps, score, uniform)
    return score, score * float(pressure_scale)


def estimate_contacts_from_joint_columns(
    joints: dict[str, np.ndarray],
    estimator: MujocoFootForceEstimator | None = None,
    method: str = "jacobian",
    pressure_scale: float = 1.0,
    pressure_mode: str = "normal_abs",
    normalize_sum: bool = False,
    damping: float = 1e-4,
    smooth_alpha: float = 0.9,
    power: float = 1.0,
    default_dt: float = 0.0025,
    model_path: Path | None = None,
) -> ContactEstimate:
    """从带时间戳的关节数据估计足底压力（model_path：机器人运动学模型，默认 DEFAULT_MODEL）。"""
    sample, time = get_sample_time(joints, default_dt)
    tau = matrix_from_joint_columns(joints, "torque")
    forces = None

    if method == "jacobian":
        q = matrix_from_joint_columns(joints, "pos")
        if estimator is None:
            estimator = ClosedFormFootForceEstimator(model_path or DEFAULT_MODEL, pressure_mode)
        forces, pressure = estimator.estimate_sequence(q, tau, damping)
        pressure = ema_filter(pressure, smooth_alpha)
        if normalize_sum:
            score, pressure = normalize_pressure(pressure, pressure_scale)
        else:
            pressure = pressure * float(pressure_scale)
            score, _ = normalize_pressure(pressure, 1.0)
    elif method == "torque_norm":
        score, pressure = estimate_torque_norm(tau, pressure_scale, smooth_alpha, power)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return ContactEstimate(sample=sample, time=time, pressure=pressure, score=score, forces=forces)


def estimate_contacts_from_joint_csv(joints_csv: Path, **kwargs) -> ContactEstimate:
    return estimate_contacts_from_joint_columns(read_joint_csv(joints_csv), **kwargs)


def write_contacts_csv(path: Path, estimate: ContactEstimate) -> None:
    """写出 sample,time,FL_pressure,FR_pressure,RL_pressure,RR_pressure。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "time", "FL_pressure", "FR_pressure", "RL_pressure", "RR_pressure"])
        for i in range(len(estimate.sample)):
            writer.writerow(
                [
                    int(estimate.sample[i]),
                    f"{estimate.time[i]:.9f}",
                    f"{estimate.pressure[i, 0]:.9f}",
                    f"{estimate.pressure[i, 1]:.9f}",
                    f"{estimate.pressure[i, 2]:.9f}",
                    f"{estimate.pressure[i, 3]:.9f}",
                ]
            )
