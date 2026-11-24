from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.data import CONTACT_FEATURES, IMU_FEATURES, stack_columns


@dataclass(frozen=True)
class RealSequence:
    path: Path
    sequence_id: str


def discover_real_sequences(data_root: Path, limit: int | None = None) -> list[RealSequence]:
    def is_real_sequence(path: Path) -> bool:
        return (path / "robot" / "imu.csv").exists() and (path / "robot" / "joints.csv").exists()

    if is_real_sequence(data_root):
        seqs = [RealSequence(path=data_root, sequence_id=data_root.name)]
    else:
        seqs = []
        seen: set[Path] = set()
        stack = [data_root]
        while stack:
            root = stack.pop()
            try:
                children = sorted(root.iterdir(), key=lambda item: str(item))
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                try:
                    resolved = child.resolve()
                except OSError:
                    resolved = child
                if resolved in seen:
                    continue
                seen.add(resolved)
                if is_real_sequence(child):
                    seqs.append(RealSequence(path=child, sequence_id=child.relative_to(data_root).as_posix()))
                else:
                    stack.append(child)
        seqs = sorted(seqs, key=lambda item: str(item.path))
    return seqs[:limit] if limit is not None else seqs


def read_csv_real(path: Path) -> dict[str, np.ndarray]:
    from common.csv_table import read_numeric_csv
    return {name: values if name == "time" else values.astype(np.float32)
            for name, values in read_numeric_csv(path).items()}


def _time_range(table: dict[str, np.ndarray], path: Path) -> tuple[float, float]:
    if "time" not in table:
        raise KeyError(f"Missing time column in {path}")
    times = np.asarray(table["time"], dtype=np.float64)
    if len(times) == 0:
        raise ValueError(f"Empty time column in {path}")
    return float(np.min(times)), float(np.max(times))


def _stack_interp(table: dict[str, np.ndarray], names: list[str], target_time: np.ndarray, path: Path) -> np.ndarray:
    missing = [name for name in names if name not in table]
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}")
    source_time = np.asarray(table["time"], dtype=np.float64)
    order = np.argsort(source_time)
    source_time = source_time[order]
    unique_time, unique_idx = np.unique(source_time, return_index=True)
    if len(unique_time) < 2:
        raise ValueError(f"Need at least two unique timestamps for interpolation: {path}")
    cols = []
    for name in names:
        values = np.asarray(table[name], dtype=np.float64)[order][unique_idx]
        cols.append(np.interp(target_time, unique_time, values))
    return np.stack(cols, axis=-1).astype(np.float32)


def load_real_features(
    seq: RealSequence,
    cfg: dict,
    gyro_bias: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IMU columns in base axes (rotated with the IMU extrinsic of ``cfg``, common/frames.py); gyro columns are
    bias-corrected with ``gyro_bias`` or ``<sequence>/gyro_bias.csv`` (biases in base axes)."""
    robot_dir = seq.path / "robot"
    imu_path = robot_dir / "imu.csv"
    joints_path = robot_dir / "joints.csv"
    imu = read_csv_real(imu_path)
    joints = read_csv_real(joints_path)

    imu_time = np.asarray(imu["time"], dtype=np.float64)
    align_ranges = [_time_range(joints, joints_path)]

    joint_names = list(joints)
    joint_groups = []
    for suffix in cfg["data"].get("joint_suffixes", ["_pos", "_vel", "_torque"]):
        cols = [name for name in joint_names if name.endswith(suffix)]
        if cols:
            joint_groups.append(cols)

    contacts = None
    contacts_path = robot_dir / "contacts.csv"
    if cfg["data"].get("use_contacts", True):
        contacts = read_csv_real(contacts_path)
        missing = [name for name in CONTACT_FEATURES if name not in contacts]
        if missing:
            raise KeyError(f"Missing contact columns in {contacts_path}: {missing}")
        align_ranges.append(_time_range(contacts, contacts_path))

    start_time = max([float(np.min(imu_time))] + [item[0] for item in align_ranges])
    end_time = min([float(np.max(imu_time))] + [item[1] for item in align_ranges])
    mask = (imu_time >= start_time) & (imu_time <= end_time)
    if not np.any(mask):
        raise ValueError(f"No timestamp overlap among imu/joints/contacts in {seq.path}")

    target_time = imu_time[mask]
    from common.frames import config_imu_extrinsic, rotate_imu
    feature_parts = [rotate_imu(stack_columns(imu, IMU_FEATURES)[mask], config_imu_extrinsic(cfg).R_BI)]
    for cols in joint_groups:
        feature_parts.append(_stack_interp(joints, cols, target_time, joints_path))

    if contacts is not None:
        feature_parts.append(_stack_interp(contacts, CONTACT_FEATURES, target_time, contacts_path))

    sample_src = imu.get("sample", np.arange(len(imu_time), dtype=np.float32))
    x = np.concatenate(feature_parts, axis=-1).astype(np.float32)
    bias_path = Path(gyro_bias) if gyro_bias else seq.path / "gyro_bias.csv"
    if bias_path.is_file():
        from common import gyro_bias as bias_file
        x[:, :3] -= bias_file.at(bias_path, target_time).astype(np.float32)
    return x, sample_src[mask].astype(np.float64), target_time.astype(np.float64)

