from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypose as pp
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from common.frames import ImuExtrinsic, config_imu_extrinsic, rotate_imu
from common.leg_kinematics import LEG_ORDER as FOOT_ORDER
from common.kinematics import wxyz_to_xyzw
from common.pressure import enabled as pressure_enabled, normalize_features


IMU_FEATURES = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
CONTACT_FEATURES = [f"{leg}_pressure" for leg in FOOT_ORDER]
LINEAR_TARGETS = ["vx", "vy", "vz"]
ANGULAR_TARGETS = ["wx", "wy", "wz"]
FULL_VELOCITY_DIM = 6
LINEAR_VELOCITY_DIM = 3
UNCERTAINTY_DIM = 6


@dataclass(frozen=True)
class SequenceInfo:
    split: str
    path: Path
    sequence_id: str


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

    def apply(self, x: np.ndarray) -> np.ndarray:
        # 所有网络输入都使用训练集统计量做标准化。
        return (x - self.mean) / self.std

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        data = np.load(path)
        return cls(data["mean"], data["std"])


def target_dim(cfg: dict) -> int:
    # 完整速度用于评估和积分；角速度由 IMU gyro 提供，不由速度分支预测。
    return FULL_VELOCITY_DIM


def velocity_pred_dim(cfg: dict) -> int:
    # 速度分支只预测机体系线速度 vx/vy/vz。
    return LINEAR_VELOCITY_DIM


def uncertainty_dim(cfg: dict) -> int:
    # 不确定性仍保留 6 轴：线速度 3 轴 + gyro 角速度 3 轴。
    return UNCERTAINTY_DIM


def dataset_imu_extrinsic(data_root: str | Path, cfg: dict) -> ImuExtrinsic:
    """IMU extrinsic of a training set: <data_root>/manifest.json 'imu_extrinsic', else the configured robot's
    (robots/<robot>/leg_odometry.yaml (robot:), common/frames.py)."""
    manifest = Path(data_root) / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding="utf-8")).get("imu_extrinsic")
        if stored:
            return ImuExtrinsic.from_dict(stored)
    return config_imu_extrinsic({}, cfg.get("robot"))


def list_sequences(data_root: str | Path, split: str, limit: int | None = None) -> list[SequenceInfo]:
    # 数据目录约定为 <root>/<split>/data/sequence_xxxxxx。
    split_dir = Path(data_root) / split / "data"
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    seqs = [
        SequenceInfo(split=split, path=p, sequence_id=p.name)
        for p in sorted(split_dir.glob("sequence_*"))
        if p.is_dir()
    ]
    return seqs[:limit] if limit is not None else seqs


def split_sequences(data_root: str | Path, cfg: dict) -> tuple[list[SequenceInfo], list[SequenceInfo]]:
    train = list_sequences(data_root, "train", cfg["data"].get("max_train_sequences"))
    try:
        val = list_sequences(data_root, "val", cfg["data"].get("max_val_sequences"))
    except FileNotFoundError:
        val_limit = cfg["data"].get("max_val_sequences") or min(4, len(train))
        val = train[: int(val_limit)]
        print("[data] val split not found; using train subset for validation.", flush=True)
    return train, val


def read_csv(path: Path) -> dict[str, np.ndarray]:
    # 按表头读取 CSV，因此不依赖列的固定顺序。
    from common.csv_table import read_numeric_csv
    return {name: values if name in ('time', 'timestamp', 'time_s') else values.astype(np.float32)
            for name, values in read_numeric_csv(path).items()}


def stack_columns(table: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in table]
    if missing:
        raise KeyError(f"Missing columns {missing}")
    return np.stack([table[name] for name in names], axis=-1).astype(np.float32)


def world_to_body_velocity_pypose(quat_wxyz: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    # 如果数据没有直接给机体系速度，就用真值姿态从世界系转换。
    quat = torch.from_numpy(quat_wxyz.astype(np.float32))
    vel = torch.from_numpy(vel_world.astype(np.float32))
    with torch.no_grad():
        body = pp.SO3(wxyz_to_xyzw(quat)).Inv().Act(vel)
    return body.cpu().numpy().astype(np.float32)


def _cache_signature(cfg: dict) -> str:
    # 输入特征配置改变时，签名会变化并触发 sequence cache 重建。
    data_cfg = cfg.get("data", {})
    payload = {
        "version": 5,
        "imu_extrinsic": config_imu_extrinsic(cfg).to_dict(),
        "pressure_normalization": cfg.get("pressure_normalization", {}),
        "imu_features": IMU_FEATURES,
        "contact_features": CONTACT_FEATURES,
        "joint_suffixes": data_cfg.get("joint_suffixes", ["_pos", "_vel", "_torque"]),
        "use_contacts": bool(data_cfg.get("use_contacts", True)),
        "predict_angular_velocity": bool(data_cfg.get("predict_angular_velocity", True)),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sequence_cache_root(cfg: dict) -> Path:
    cache_dir = Path(cfg["data"].get("sequence_cache_dir", "sequence_cache"))
    if not cache_dir.is_absolute():
        config_dir = Path(cfg.get("_meta", {}).get("config_dir", "."))
        cache_dir = config_dir / cache_dir
    return Path(os.path.normpath(str(cache_dir)))


def _sequence_cache_dir(seq: SequenceInfo, cfg: dict) -> Path:
    return _sequence_cache_root(cfg) / seq.split / seq.sequence_id


def _load_sequence_arrays_from_csv(
    seq: SequenceInfo,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    # x 是网络输入，y 是速度监督。
    imu = read_csv(seq.path / "imu.csv")
    joints = read_csv(seq.path / "joints.csv")
    gt = read_csv(seq.path / "ground_truth.csv")
    # IMU samples in base axes (common/frames.py): the network input is the IMU measurement expressed in the base frame
    feature_parts = [rotate_imu(stack_columns(imu, IMU_FEATURES), config_imu_extrinsic(cfg).R_BI)]

    joint_names = list(joints)
    for suffix in cfg["data"].get("joint_suffixes", ["_pos", "_vel", "_torque"]):
        cols = [name for name in joint_names if name.endswith(suffix)]
        if cols:
            feature_parts.append(stack_columns(joints, cols))

    contacts_path = seq.path / "contacts.csv"
    if cfg["data"].get("use_contacts", True) and contacts_path.exists():
        contacts = read_csv(contacts_path)
        cols = [name for name in CONTACT_FEATURES if name in contacts]
        if pressure_enabled(cfg) and len(cols) != 4:
            raise ValueError(f'Pressure normalization requires all four pressure columns: {contacts_path}')
        if cols:
            feature_parts.append(stack_columns(contacts, cols))
    elif pressure_enabled(cfg):
        raise FileNotFoundError(f'Pressure normalization requires {contacts_path}')

    if all(name in gt for name in ("base_lin_vel_body_x", "base_lin_vel_body_y", "base_lin_vel_body_z")):
        linear = stack_columns(gt, ["base_lin_vel_body_x", "base_lin_vel_body_y", "base_lin_vel_body_z"])
    else:
        quat = stack_columns(gt, ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"])
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-6)
        vel_world = stack_columns(gt, ["base_lin_vel_world_x", "base_lin_vel_world_y", "base_lin_vel_world_z"])
        linear = world_to_body_velocity_pypose(quat, vel_world)

    y_parts = [linear]
    y_parts.append(stack_columns(gt, ["base_ang_vel_body_x", "base_ang_vel_body_y", "base_ang_vel_body_z"]))

    n = min([part.shape[0] for part in feature_parts] + [linear.shape[0], y_parts[-1].shape[0]])
    x = np.concatenate([part[:n] for part in feature_parts], axis=-1).astype(np.float32)
    if pressure_enabled(cfg):
        x = normalize_features(x, pressure_times(imu, n, cfg), cfg)
    y = np.concatenate([part[:n] for part in y_parts], axis=-1).astype(np.float32)
    return x, y


def pressure_times(imu, n, cfg):
    for key in ('time', 'timestamp', 'time_s'):
        if key in imu:
            return np.asarray(imu[key][:n], dtype=np.float64)
    dt = cfg.get('pressure_normalization', {}).get('sample_dt_s')
    if dt is None or float(dt) <= 0:
        raise ValueError('IMU has no seconds timestamp; set pressure_normalization.sample_dt_s')
    return np.arange(n, dtype=np.float64) * float(dt)


def save_sequence_cache(seq: SequenceInfo, cfg: dict, force: bool = False) -> Path:
    # 将一条 sequence 的 CSV 预处理成 npy，训练时用 mmap 快速读取。
    path = _sequence_cache_dir(seq, cfg)
    meta_path = path / "meta.json"
    signature = _cache_signature(cfg)
    if meta_path.exists() and not force:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                if json.load(f).get("signature") == signature:
                    return path
        except Exception:
            pass

    path.mkdir(parents=True, exist_ok=True)
    x, y = _load_sequence_arrays_from_csv(seq, cfg)
    tmp_prefix = path / f".tmp.{os.getpid()}"
    arrays = {"x": x, "y": y}
    for name, array in arrays.items():
        np.save(f"{tmp_prefix}.{name}.npy", array)
        Path(f"{tmp_prefix}.{name}.npy").replace(path / f"{name}.npy")
    meta = {
        "signature": signature,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
    }
    tmp_meta = path / f".tmp.{os.getpid()}.meta.json"
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_meta.replace(meta_path)
    return path


def load_sequence_arrays(
    seq: SequenceInfo,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = cfg.get("data", {})
    if not data_cfg.get("use_sequence_cache", False):
        return _load_sequence_arrays_from_csv(seq, cfg)

    path = _sequence_cache_dir(seq, cfg)
    meta_path = path / "meta.json"
    signature = _cache_signature(cfg)
    rebuild = bool(data_cfg.get("rebuild_sequence_cache", False))
    if rebuild or not meta_path.exists() or not (path / "x.npy").exists():
        save_sequence_cache(seq, cfg, force=True)

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            if json.load(f).get("signature") != signature:
                save_sequence_cache(seq, cfg, force=True)
        return (
            np.load(path / "x.npy", mmap_mode="r"),
            np.load(path / "y.npy", mmap_mode="r"),
        )
    except Exception:
        save_sequence_cache(seq, cfg, force=True)
        return (
            np.load(path / "x.npy", mmap_mode="r"),
            np.load(path / "y.npy", mmap_mode="r"),
        )


class SequenceCache:
    def __init__(self, cfg: dict, max_items: int = 8) -> None:
        self.cfg = cfg
        self.max_items = max_items
        self.cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, seq: SequenceInfo):
        # 简单 LRU，避免 DataLoader 频繁重复打开同一条 sequence。
        if seq.path in self.cache:
            item = self.cache.pop(seq.path)
            self.cache[seq.path] = item
            return item
        item = load_sequence_arrays(seq, self.cfg)
        self.cache[seq.path] = item
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return item


def fit_normalizer(sequences: list[SequenceInfo], cfg: dict) -> Normalizer:
    cache = SequenceCache(cfg, int(cfg["data"].get("sequence_lru_size", 8)))
    stride = int(cfg["data"].get("stats_stride", 20))
    log_interval = max(1, int(cfg.get("train", {}).get("log_interval", 20)))
    sum_x = None
    sum_x2 = None
    count = 0
    print(
        f"[normalizer] fitting from {len(sequences)} sequences, stats_stride={stride}",
        flush=True,
    )
    iterator = enumerate(sequences, start=1)
    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(sequences), desc="[normalizer]", unit="seq", dynamic_ncols=True)
    for i, seq in iterator:
        x, _ = cache.get(seq)
        xs = x[::stride]
        if sum_x is None:
            sum_x = np.zeros(xs.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(xs.shape[1], dtype=np.float64)
        sum_x += xs.sum(axis=0)
        sum_x2 += (xs.astype(np.float64) ** 2).sum(axis=0)
        count += xs.shape[0]
        if progress is not None:
            progress.set_postfix(sequence=seq.sequence_id, samples=count, refresh=False)
            progress.update(1)
        elif i == 1 or i == len(sequences) or i % log_interval == 0:
            print(
                f"[normalizer] {i}/{len(sequences)} {seq.sequence_id}, "
                f"used_samples={count}",
                flush=True,
            )
    if progress is not None:
        progress.close()
    if count == 0 or sum_x is None or sum_x2 is None:
        raise ValueError("No samples to fit normalizer.")
    mean = sum_x / count
    var = np.maximum(sum_x2 / count - mean * mean, 1e-12)
    return Normalizer(mean, np.sqrt(var))


def infer_input_dim(data_root: Path, cfg: dict) -> int:
    x, _ = load_sequence_arrays(list_sequences(data_root, "train", 1)[0], cfg)
    return int(x.shape[1])


class VelocityWindowDataset(Dataset):
    def __init__(self, sequences: list[SequenceInfo], cfg: dict, normalizer: Normalizer, training: bool) -> None:
        self.sequences = sequences
        self.cfg = cfg
        self.normalizer = normalizer
        self.window = int(cfg["data"]["window"])
        self.stride = int(cfg["data"]["train_stride" if training else "eval_stride"])
        self.cache = SequenceCache(cfg, int(cfg["data"].get("sequence_lru_size", 8)))
        self.index: list[tuple[int, int]] = []
        limit_key = "max_windows_per_sequence" if training else "max_eval_windows_per_sequence"
        max_windows = cfg["data"].get(limit_key)
        sequence_items = list(enumerate(sequences))
        if training and bool(cfg["data"].get("shuffle_train_sequences", True)):
            seed = int(cfg["data"].get("dataset_seed", 42))
            np.random.default_rng(seed).shuffle(sequence_items)

        for seq_i, seq in sequence_items:
            x, y = self.cache.get(seq)
            n = min(x.shape[0], y.shape[0])
            if n < self.window:
                continue
            # 每个 start 对应一个长度为 window 的输入窗口。
            starts = list(range(0, n - self.window + 1, self.stride))
            if max_windows is not None and len(starts) > int(max_windows):
                ids = np.linspace(0, len(starts) - 1, int(max_windows)).round().astype(int)
                starts = [starts[i] for i in ids]
            self.index.extend((seq_i, s) for s in starts)
        if not self.index:
            raise ValueError(
                f"No velocity windows found. Check data.window={self.window}, stride={self.stride}, "
                f"and sequence lengths under the selected split."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_i, start = self.index[idx]
        x, y = self.cache.get(self.sequences[seq_i])
        end = start + self.window
        # 速度标签取窗口最后一帧。
        target_i = min(end - 1, y.shape[0] - 1)
        xw = self.normalizer.apply(x[start:end])
        return {
            "x": torch.from_numpy(np.array(xw, dtype=np.float32, copy=True)),
            "y": torch.from_numpy(np.array(y[target_i], dtype=np.float32, copy=True)),
            "gyro": torch.from_numpy(np.array(x[target_i, :3], dtype=np.float32, copy=True)),
        }


def collate_velocity_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # 显式 stack，避开 DataLoader 默认 collate 在多 worker 共享内存中对部分 tensor storage resize 的问题。
    return {key: torch.stack([item[key].clone() for item in items], dim=0) for key in items[0]}
