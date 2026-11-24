#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pypose as pp
import torch
import torch.nn.functional as F
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_config, resolve_device
from common.data import Normalizer
from common.pressure import enabled as pressure_enabled, fit_pressure, normalize_features
from common.model import PoseEstimator, configure_velocity_lora
from common.early_stopping import TrajectoryEarlyStopping, FixedBATrajectoryValidation
from common.model import load_velocity_model
from common.holdout import pair_filter as holdout_pair_filter
from common.teacher_check import pair_filter as teacher_pair_filter
from common.teacher_select import pair_keep as teacher_select_keep
from eval.real_utils import (
    RealSequence,
    discover_real_sequences,
    load_real_features,
)
from common.stereo_geometry import build_stereo_geometry


@dataclass(frozen=True)
class VisualPair:
    sequence_id: str
    sequence_path: Path
    t0: float
    t1: float
    robot_i0: int
    robot_i1: int
    kp0_path: Path
    kp1_path: Path
    image_dt: float
    robot_dt: float


def select_windows(sequence_ids: list[str], offsets: list[int], step: int, total: int) -> list[int]:
    """Indices of the windows kept (in their order). Windows are tiered by the start pair index k within their
    sequence: k % step == 0 (non-overlapping), then k % (step // 2) == 0 (half overlap), then the rest. total > 0: the
    budget is split evenly over the sequences (water filling: sequences with fewer windows take all, the rest of the
    budget goes to the others); a sequence fills its share from the first tier, evenly spaced, then from the next
    tiers. total == 0: the first tier of every sequence (validation, each pair once)."""
    tiers: dict[str, list[list[int]]] = {}
    half = max(1, step // 2)
    for i, (sid, k) in enumerate(zip(sequence_ids, offsets)):
        tier = 0 if k % step == 0 else 1 if k % half == 0 else 2
        tiers.setdefault(sid, [[], [], []])[tier].append(i)
    if total == 0:
        return sorted(i for t in tiers.values() for i in t[0])
    available = {sid: sum(len(x) for x in t) for sid, t in tiers.items()}
    quota, remaining, budget = {}, dict(available), int(total)
    while remaining:
        share = budget / len(remaining)
        small = {sid: n for sid, n in remaining.items() if n <= share}
        if not small:
            base, extra = divmod(budget, len(remaining))
            for j, sid in enumerate(sorted(remaining)):
                quota[sid] = base + (1 if j < extra else 0)
            break
        for sid, n in small.items():
            quota[sid] = n
            budget -= n
            del remaining[sid]
    keep, counts = [], [0, 0, 0]
    for sid, t in tiers.items():
        need = quota[sid]
        for level, tier in enumerate(t):
            if need <= 0:
                break
            chosen = tier if len(tier) <= need else [tier[j] for j in np.unique(
                np.round(np.linspace(0, len(tier) - 1, need)).astype(int))]
            keep.extend(chosen)
            counts[level] += len(chosen)
            need -= len(chosen)
    print(f"[visual_windows] {len(keep)} of {len(sequence_ids)} windows in {len(tiers)} sequences "
          f"(budget {total}; non-overlapping {counts[0]}, half {counts[1]}, 3/4 {counts[2]})", flush=True)
    return sorted(keep)


class VisualPairDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cfg: dict,
        normalizer: Normalizer,
        input_dim: int,
        max_windows: int | None = None,
        non_overlapping: bool = False,
    ) -> None:
        """max_windows: training windows in total (select_windows); non_overlapping: every pair in one window only
        (validation). Both apply to multi-frame windows (visual.window_frames > 2)."""
        self.cfg = cfg
        self.normalizer = normalizer
        self.input_dim = int(input_dim)
        self.window = int(cfg["data"]["window"])
        self.sequences = discover_real_sequences(data_root)
        if not self.sequences:
            raise FileNotFoundError(f"No real sequences under {data_root}")
        self.sequence_cache: dict[str, dict[str, Any]] = {}
        self.pairs: list[VisualPair] = []
        self.use_frozen_pair_cache = False
        self.precomputed_matches: list[dict[str, torch.Tensor] | None] = []
        for seq in self.sequences:
            cache = self._load_sequence(seq)
            self.sequence_cache[seq.sequence_id] = cache
            self.pairs.extend(self._build_pairs(seq, cache["time"]))
        if not self.pairs:
            raise ValueError("No usable visual frame pairs. Check keypoints, timestamps, and data.window.")
        self.pair_weights = np.ones(len(self.pairs), dtype=np.float32)
        self.precomputed_matches = [None] * len(self.pairs)
        self.window_pairs = None
        frames = int(cfg.get("visual", {}).get("window_frames", 2))
        if frames < 2:
            raise ValueError("visual.window_frames must be at least 2")
        if frames > 2:
            max_seconds = float(cfg["visual"].get("window_max_seconds", 5.0))
            if not math.isfinite(max_seconds) or max_seconds <= 0:
                raise ValueError("visual.window_max_seconds must be finite and positive")
            self.window_pairs, starts = [], []
            for start in range(len(self.pairs) - frames + 2):
                group = self.pairs[start:start + frames - 1]
                if (group[-1].t1 - group[0].t0 <= max_seconds
                        and all(a.sequence_id == b.sequence_id and a.kp1_path == b.kp0_path
                                for a, b in zip(group, group[1:]))):
                    self.window_pairs.append(group)
                    starts.append(start)
            if not self.window_pairs:
                raise ValueError("No continuous visual windows; check frame count and duration limit")
            if max_windows is not None or non_overlapping:
                first = {}
                for index, pair in enumerate(self.pairs):
                    first.setdefault(pair.sequence_id, index)
                # start pair index within the sequence: windows with indices a multiple of frames-1 apart do not overlap
                offsets = [start - first[group[0].sequence_id] for start, group in zip(starts, self.window_pairs)]
                keep = select_windows([g[0].sequence_id for g in self.window_pairs], offsets, frames - 1,
                                      0 if non_overlapping else int(max_windows))
                self.window_pairs = [self.window_pairs[i] for i in keep]
            self.pairs = [group[0] for group in self.window_pairs]
            self.pair_weights = np.ones(len(self.pairs), dtype=np.float32)

    def _load_sequence(self, seq: RealSequence) -> dict[str, Any]:
        x, sample, time_col = load_real_features(seq, self.cfg)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"{seq.path}: input_dim={x.shape[-1]}, checkpoint expects {self.input_dim}")
        x_norm = self.normalizer.apply(normalize_features(x, time_col, self.cfg))
        geom = build_stereo_geometry(
            seq.path / "camera.yaml",
            prefer_projection=bool(self.cfg.get("visual", {}).get("prefer_projection", True)),
            baseline_sign=float(self.cfg.get("visual", {}).get("baseline_sign", 1.0)),
        )
        return {
            "x": np.asarray(x_norm, dtype=np.float32),
            "time": np.asarray(time_col, dtype=np.float64),
            "sample": sample,
            "k": geom.k0.astype(np.float32),
        }

    def _build_pairs(self, seq: RealSequence, robot_time: np.ndarray) -> list[VisualPair]:
        kp_dir = seq.path / "keypoints"
        files = sorted(kp_dir.glob("*.txt"))
        if len(files) < 2:
            return []
        raw_times = np.asarray([_timestamp_from_path(p) for p in files], dtype=np.float64)
        image_times = choose_image_time_axis(raw_times, robot_time)
        pairs = []
        visual_cfg = self.cfg.get("visual", {})
        min_robot_dt = float(visual_cfg.get("min_robot_dt", 0.0))
        max_robot_dt_cfg = visual_cfg.get("max_robot_dt")
        max_robot_dt = None if max_robot_dt_cfg is None else float(max_robot_dt_cfg)
        states = ({"lost"} if visual_cfg.get("skip_lost_keyframes", True) else set()) | (
            {"weak"} if visual_cfg.get("skip_weak_keyframes", False) else set())
        lost = unreliable_keyframes(seq.path, states)
        skipped_lost = 0
        holdout_keep = holdout_pair_filter(seq.path)   # training / validation part of a held-out sequence
        teacher_keep = teacher_pair_filter(seq.path)   # fused rounds: teacher windows inconsistent with the leg model
        skipped_teacher = 0
        selected = teacher_select_keep(seq.path)       # teacher selection replaces both filters above
        if selected is not None:
            lost, teacher_keep = set(), (lambda a, b: True)

        # Only consecutive keyframes form supervision intervals.
        for i in range(len(files) - 1):
            j = i + 1
            if selected is not None and not selected(files[i].stem):
                skipped_teacher += 1
                continue
            if files[i].stem in lost or files[j].stem in lost:
                skipped_lost += 1
                continue
            if not holdout_keep(float(raw_times[i]), float(raw_times[j])):
                continue
            if not teacher_keep(float(raw_times[i]), float(raw_times[j])):
                skipped_teacher += 1
                continue
            t0 = float(image_times[i])
            t1 = float(image_times[j])
            if not t1 > t0:
                continue
            # Bracket the exact image interval. Velocity at robot index k
            # advances the pose over [time[k], time[k + 1]), so retain the
            # sample immediately before t0 and the sample at/after t1.
            # The predictor later clips the first and last integration dt
            # to [t0, t1], keeping network training fully differentiable.
            i0 = int(np.searchsorted(robot_time, t0, side="right") - 1)
            i1 = int(np.searchsorted(robot_time, t1, side="left"))
            if i0 < self.window - 1 or i1 <= i0 or i1 >= len(robot_time):
                continue
            robot_dt = float(t1 - t0)
            if robot_dt < min_robot_dt or (max_robot_dt is not None and robot_dt > max_robot_dt):
                continue
            pairs.append(
                VisualPair(
                    sequence_id=seq.sequence_id,
                    sequence_path=seq.path,
                    t0=t0,
                    t1=t1,
                    robot_i0=i0,
                    robot_i1=i1,
                    kp0_path=files[i],
                    kp1_path=files[j],
                    image_dt=t1 - t0,
                    robot_dt=robot_dt,
                )
            )
        if skipped_lost:
            print(f"[visual_pairs] {seq.sequence_id}: {skipped_lost} pairs touching {len(lost)} "
                  f"{'/'.join(sorted(states))} keyframes skipped", flush=True)
        if skipped_teacher:
            print(f"[visual_pairs] {seq.sequence_id}: {skipped_teacher} pairs "
                  f"{'not selected (teacher_select)' if selected is not None else 'in teacher-check windows'} skipped", flush=True)
        max_pairs = visual_cfg.get("max_pairs_per_sequence")
        if max_pairs is not None:
            pairs = pairs[: int(max_pairs)]
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.window_pairs is not None:
            edges = [self._pair_item(pair, index) for pair in self.window_pairs[index]]
            return dict(edges[0], window_items=edges)
        return self._pair_item(self.pairs[index], index)

    def _pair_item(self, pair, index):
        cache = self.sequence_cache[pair.sequence_id]
        item = {
            "pair": pair,
            "pair_index": int(index),
            "k": torch.from_numpy(cache["k"]),
            "pair_weight": float(self.pair_weights[index]),
        }
        matches = self.precomputed_matches[index] if self.use_frozen_pair_cache else None
        if matches is not None:
            item["matches"] = matches
        else:
            item["kp0"] = torch.from_numpy(load_keypoints(pair.kp0_path))
            item["kp1"] = torch.from_numpy(load_keypoints(pair.kp1_path))
        if not self.use_frozen_pair_cache:
            start = pair.robot_i0 - self.window + 1
            end = pair.robot_i1 + 1
            item["x_context"] = torch.from_numpy(
                np.array(cache["x"][start:end], dtype=np.float32, copy=True)
            )
            item["time_context"] = torch.from_numpy(
                np.array(cache["time"][start:end], dtype=np.float64, copy=True)
            )
        return item


def velocity_model_fingerprint(model: PoseEstimator) -> str:
    """Deterministic identity used to invalidate frozen-motion caches."""
    digest = hashlib.sha256()
    for name, value in model.velocity_branch.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _safe_cache_name(sequence_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sequence_id)


def predict_sequence_motion_batched(
    model: PoseEstimator,
    x: torch.Tensor,
    time_col: np.ndarray,
    normalizer: Normalizer,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    """Batch all frozen-network windows and cumulatively integrate their poses."""
    window = int(cfg["data"]["window"])
    start_idx = window - 1
    steps = int(x.shape[0] - start_idx - 1)
    if steps < 1:
        raise ValueError(f"Sequence is too short: got {x.shape[0]}, need more than {window}")
    time_array = np.asarray(time_col, dtype=np.float64)
    dt_numpy = np.diff(time_array)[start_idx : start_idx + steps]
    if np.any(dt_numpy <= 0.0):
        raise ValueError("time column must be strictly increasing for motion caching")

    windows = x.unfold(0, window, 1).transpose(-1, -2).contiguous()[:steps]
    gyro = recover_gyro_sequence(x[start_idx : start_idx + steps], normalizer)
    cache_cfg = cfg.get("motion_cache", {})
    batch_size = max(1, int(cache_cfg.get("prediction_batch_size", 4096)))
    velocity_chunks = []
    prediction_time = time.perf_counter()
    for begin in range(0, steps, batch_size):
        end = min(begin + batch_size, steps)
        pred, refined_gyro, _, _ = model.velocity_branch.forward_with_gyro(
            windows[begin:end], gyro[begin:end]
        )
        velocity = torch.cat([pred[..., :3], refined_gyro], dim=-1)
        velocity_chunks.append(velocity)
    velocities = torch.cat(velocity_chunks, dim=0)
    prediction_seconds = time.perf_counter() - prediction_time

    integration_time = time.perf_counter()
    # PyPose cumprod defaults to left=True, i.e. x_i @ ... @ x_1, while motion
    # integration requires the chronological right product x_1 @ ... @ x_i.
    # Keep an explicit float64 scan here so the composition convention is
    # unambiguous and matches the original step-by-step integrator.
    velocity_numpy = velocities.detach().cpu().double().numpy()
    positions_numpy = np.zeros((steps + 1, 3), dtype=np.float64)
    quaternions_numpy = np.zeros((steps + 1, 4), dtype=np.float64)
    quaternions_numpy[0, 3] = 1.0
    position = positions_numpy[0].copy()
    quaternion = quaternions_numpy[0].copy()  # xyzw
    for index in range(steps):
        q_xyz = quaternion[:3]
        q_w = quaternion[3]
        linear = velocity_numpy[index, :3]
        rotated_linear = linear + 2.0 * (
            q_w * np.cross(q_xyz, linear) + np.cross(q_xyz, np.cross(q_xyz, linear))
        )
        position = position + rotated_linear * dt_numpy[index]
        rotvec = velocity_numpy[index, 3:6] * dt_numpy[index]
        angle = float(np.linalg.norm(rotvec))
        if angle > 1e-12:
            delta_xyz = rotvec * (math.sin(0.5 * angle) / angle)
            delta_w = math.cos(0.5 * angle)
        else:
            delta_xyz = 0.5 * rotvec
            delta_w = 1.0
        # Right multiplication: world_R_body(k+1) = R(k) * Exp(w_body dt).
        delta = np.concatenate([delta_xyz, np.asarray([delta_w])])
        a_xyz, a_w = quaternion[:3], quaternion[3]
        b_xyz, b_w = delta[:3], delta[3]
        quaternion = np.concatenate(
            [a_w * b_xyz + b_w * a_xyz + np.cross(a_xyz, b_xyz), [a_w * b_w - np.dot(a_xyz, b_xyz)]]
        )
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        positions_numpy[index + 1] = position
        quaternions_numpy[index + 1] = quaternion
    positions = torch.from_numpy(positions_numpy).to(dtype=x.dtype)
    quaternion_tensor = torch.from_numpy(quaternions_numpy).to(dtype=x.dtype)
    rotvecs = pp.SO3(quaternion_tensor).Log().tensor()
    integration_seconds = time.perf_counter() - integration_time
    timing = {
        "processing_time_s": prediction_seconds + integration_seconds,
        "velocity_time_s": prediction_seconds,
        "integrate_time_s": integration_seconds,
        "prediction_batch_size": batch_size,
    }
    return positions, rotvecs, velocities, start_idx, timing


def build_frozen_motion_cache(
    model: PoseEstimator,
    dataset: "VisualPairDataset",
    normalizer: Normalizer,
    cfg: dict,
    device: torch.device,
    cache_dir: Path,
) -> dict[str, Any]:
    """Predict and integrate each complete sequence once for a frozen stage."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = velocity_model_fingerprint(model)
    fingerprint = hashlib.sha256((fingerprint + json.dumps(
        cfg.get('pressure_normalization', {}), sort_keys=True
    )).encode() + normalizer.mean.tobytes() + normalizer.std.tobytes()).hexdigest()
    result: dict[str, dict[str, torch.Tensor]] = {}
    was_training = model.velocity_branch.training
    model.velocity_branch.eval()
    for seq in dataset.sequences:
        sequence = dataset.sequence_cache[seq.sequence_id]
        robot_time = np.asarray(sequence["time"], dtype=np.float64)
        # Gyro columns carry the bias correction, so they are part of the cache key.
        time_fingerprint = hashlib.sha256(
            robot_time.tobytes() + np.ascontiguousarray(sequence["x"][:, :3]).tobytes()).hexdigest()
        path = cache_dir / f"{_safe_cache_name(seq.sequence_id)}.pt"
        payload = None
        if path.exists():
            candidate = torch.load(path, map_location="cpu", weights_only=False)
            metadata = candidate.get("metadata", {})
            if (
                metadata.get("version") == 3
                and metadata.get("model_fingerprint") == fingerprint
                and metadata.get("sequence_id") == seq.sequence_id
                and metadata.get("time_fingerprint") == time_fingerprint
                and metadata.get("window") == int(cfg["data"]["window"])
            ):
                payload = candidate
                print(f"[motion_cache] hit sequence={seq.sequence_id} path={path}", flush=True)
        if payload is None:
            x = torch.from_numpy(np.array(sequence["x"], dtype=np.float32, copy=True)).to(device)
            with torch.no_grad():
                positions, rotvecs, velocities, start_idx, timing = predict_sequence_motion_batched(
                    model, x, robot_time, normalizer, cfg
                )
            quaternions = pp.so3(rotvecs).Exp().tensor()
            payload = {
                "metadata": {
                    "version": 3,
                    "sequence_id": seq.sequence_id,
                    "model_fingerprint": fingerprint,
                    "time_fingerprint": time_fingerprint,
                    "window": int(cfg["data"]["window"]),
                    "start_idx": int(start_idx),
                    "timing": timing,
                },
                "time": torch.from_numpy(robot_time[start_idx : start_idx + positions.shape[0]].copy()),
                "position": positions.detach().cpu(),
                "quaternion_xyzw": quaternions.detach().cpu(),
                "velocity": velocities.detach().cpu(),
            }
            torch.save(payload, path)
            print(f"[motion_cache] built sequence={seq.sequence_id} poses={positions.shape[0]} path={path}", flush=True)
        result[seq.sequence_id] = {
            "time": payload["time"],
            "position": payload["position"],
            "quaternion_xyzw": payload["quaternion_xyzw"],
        }
    if was_training:
        model.velocity_branch.train()
    # Convert exact image timestamps to relative poses once as well. Keeping
    # this dense tensor on CPU lets each training batch use one indexed GPU
    # transfer instead of many tiny pose interpolation transfers.
    relative_poses = []
    cpu = torch.device("cpu")
    pairs = dataset.pairs
    if dataset.window_pairs is not None:
        pairs = list({(p.sequence_id, p.t0, p.t1): p
                      for window in dataset.window_pairs for p in window}.values())
    for pair in pairs:
        sequence_cache = result[pair.sequence_id]
        pose0 = interpolate_cached_pose(sequence_cache, pair.t0, cpu)
        pose1 = interpolate_cached_pose(sequence_cache, pair.t1, cpu)
        relative_poses.append((pose0.Inv() * pose1).tensor())
    return {"sequences": result, "pair_transforms": torch.stack(relative_poses, dim=0),
            "pair_indices": {(p.sequence_id, p.t0, p.t1): i for i, p in enumerate(pairs)}}


def interpolate_cached_pose(cache: dict[str, torch.Tensor], timestamp: float, device: torch.device):
    times = cache["time"]
    t = torch.tensor(float(timestamp), dtype=times.dtype)
    right = int(torch.searchsorted(times, t, right=False).item())
    right = min(max(right, 1), int(times.numel()) - 1)
    left = right - 1
    alpha = ((t - times[left]) / (times[right] - times[left]).clamp_min(1e-12)).clamp(0.0, 1.0)
    alpha = alpha.to(dtype=torch.float32, device=device)
    p0 = cache["position"][left].to(device)
    p1 = cache["position"][right].to(device)
    r0 = pp.SO3(cache["quaternion_xyzw"][left].to(device))
    r1 = pp.SO3(cache["quaternion_xyzw"][right].to(device))
    rotation = r0 * pp.so3((r0.Inv() * r1).Log().tensor() * alpha).Exp()
    return pp.SE3(torch.cat([p0 + alpha * (p1 - p0), rotation.tensor()], dim=-1))


def cached_body_delta_batch(
    batch: list[dict[str, Any]], motion_cache: dict[str, Any], device: torch.device
):
    indices = torch.as_tensor([motion_cache["pair_indices"][
        (item["pair"].sequence_id, item["pair"].t0, item["pair"].t1)] for item in batch], dtype=torch.long)
    return pp.SE3(motion_cache["pair_transforms"].index_select(0, indices).to(device))


class VisualCalibration(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        ext = cfg.get("extrinsic", {})
        self.optimize_extrinsic = bool(ext.get("optimize_extrinsic", True))
        self.use_initial_extrinsic = bool(ext.get("use_initial_extrinsic", False))
        if self.use_initial_extrinsic:
            init_xyz = torch.as_tensor(ext.get("init_xyz", [0.0, 0.0, 0.0]), dtype=torch.float32)
            init_quat = torch.as_tensor(ext.get("init_quat_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=torch.float32)
        else:
            init_xyz = torch.zeros(3, dtype=torch.float32)
            init_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        init_quat = init_quat / init_quat.norm().clamp_min(1e-8)
        init_se3 = torch.cat([init_xyz, init_quat], dim=0)
        target_scale = max(float(ext.get("target_scale", ext.get("init_scale", 1.0))), 1e-6)
        scale_min = max(float(ext.get("scale_min", 0.1)), 1e-6)
        scale_max = max(float(ext.get("scale_max", 10.0)), scale_min)
        self.register_buffer("init_body_cam_tensor", init_se3)
        self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))
        self.register_buffer("log_scale_min", torch.tensor(math.log(scale_min), dtype=torch.float32))
        self.register_buffer("log_scale_max", torch.tensor(math.log(scale_max), dtype=torch.float32))
        self.delta_body_cam = nn.Parameter(
            torch.zeros(6, dtype=torch.float32), requires_grad=self.optimize_extrinsic
        )
        self.log_scale_delta = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def body_cam(self):
        init = pp.SE3(self.init_body_cam_tensor)
        return pp.se3(self.delta_body_cam).Exp() * init

    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale_delta.clamp(self.log_scale_min, self.log_scale_max))

    def clamp_scale_(self) -> None:
        with torch.no_grad():
            self.log_scale_delta.clamp_(self.log_scale_min, self.log_scale_max)

    def prior_loss(self) -> torch.Tensor:
        return self.delta_body_cam.square().mean()

    def scale_prior_loss(self) -> torch.Tensor:
        s = self.scale()
        return (s - self.target_scale).square()

    def summary(self) -> dict[str, Any]:
        with torch.no_grad():
            body_cam = self.body_cam()
            t_tensor = body_cam.tensor().detach()
            t = t_tensor.cpu().numpy()
            rotvec = pp.SO3(t_tensor[..., 3:]).Log().tensor().detach().cpu().numpy()
            return {
                "T_body_cam_xyz_quat_xyzw": [float(v) for v in t.tolist()],
                "translation_xyz": [float(v) for v in t[:3].tolist()],
                "rotation_quat_xyzw": [float(v) for v in t[3:].tolist()],
                "rotation_rotvec": [float(v) for v in rotvec.tolist()],
                "scale": float(self.scale().detach().cpu()),
                "target_scale": float(self.target_scale.detach().cpu()),
                "optimize_extrinsic": bool(self.optimize_extrinsic),
                "use_initial_extrinsic": bool(self.use_initial_extrinsic),
                "scale_min": float(torch.exp(self.log_scale_min).detach().cpu()),
                "scale_max": float(torch.exp(self.log_scale_max).detach().cpu()),
                "delta_body_cam": [float(v) for v in self.delta_body_cam.detach().cpu().tolist()],
            }


def _timestamp_from_path(path: Path) -> float:
    stem = path.stem
    try:
        if "." in stem:
            return float(stem)
        return float(int(stem)) * 1e-9
    except ValueError as exc:
        raise ValueError(f"Keypoint filename must be a numeric timestamp: {path}") from exc


def choose_image_time_axis(image_times: np.ndarray, robot_times: np.ndarray) -> np.ndarray:
    # Keypoint filenames and robot CSVs share one clock; only fall back to
    # aligning the first samples when the two streams do not overlap at all.
    overlaps = image_times[-1] >= robot_times[0] and image_times[0] <= robot_times[-1]
    return image_times if overlaps else image_times - image_times[0] + robot_times[0]


def unreliable_keyframes(sequence_path: Path, states: set[str]) -> set[str]:
    """Keypoint file stems of the keyframes whose S2 preparation VO tracking state is in `states` (keyframe_status.csv:
    lost = pose from a motion model, a tracking break; weak = few inliers); empty without the file (older caches)."""
    path = sequence_path / "keyframe_status.csv"
    if not states or not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["timestamp"] for row in csv.DictReader(stream) if row["track_state"] in states}


def load_keypoints(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        if len(values) != 9:
            raise ValueError(f"{path}: expected 9 columns, got {len(values)}")
        rows.append(values)
    if not rows:
        return np.zeros((0, 9), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def collate_visual_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def make_stage_train_loader(
    dataset: VisualPairDataset,
    cfg: dict,
    stage_cfg: dict,
    seed: int,
) -> tuple[DataLoader, int]:
    """Sample adjacent-keyframe pairs, optionally balancing sequence contributions."""
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    configured_batches = cfg.get("train", {}).get("batches_per_epoch")
    num_batches = int(configured_batches) if configured_batches is not None else math.ceil(len(dataset) / batch_size)
    if num_batches < 1:
        raise ValueError("batches_per_epoch must be positive")
    counts: dict[str, int] = {}
    for pair in dataset.pairs:
        counts[pair.sequence_id] = counts.get(pair.sequence_id, 0) + 1
    balance = bool(cfg.get("visual", {}).get("balance_sequences", True))
    weights = [1.0 / counts[p.sequence_id] if balance else 1.0 for p in dataset.pairs]
    sampler = WeightedRandomSampler(weights, num_batches * batch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    frozen_fast_path = (
        stage_cfg.get("train_network", False) is False
        and not bool(stage_cfg.get("use_velocity_smooth", False))
    )
    num_workers = (
        int(cfg.get("motion_cache", {}).get("num_workers", 0))
        if frozen_fast_path
        else int(cfg.get("train", {}).get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_visual_batch,
        persistent_workers=num_workers > 0,  # no worker re-spawn every epoch
    )
    return loader, num_batches


def pair_group_summary(dataset: VisualPairDataset) -> str:
    counts: dict[str, int] = {}
    for pair in dataset.pairs:
        counts[pair.sequence_id] = counts.get(pair.sequence_id, 0) + 1
    return ", ".join(f"{seq}={count}" for seq, count in sorted(counts.items()))


def merged_config(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged_config(out[key], value)
        else:
            out[key] = value
    return out


def set_network_train_mode(model: PoseEstimator, mode: str | bool, lora_cfg: dict | None = None) -> list[nn.Parameter]:
    for param in model.velocity_branch.parameters():
        param.requires_grad = False
    if mode is False or mode == "false" or mode == "none":
        model.velocity_branch.eval()
        return []
    if mode in ("head", "heads"):
        model.velocity_branch.eval()
        for name, param in model.velocity_branch.named_parameters():
            if name.startswith("velocity_head.") or name.startswith("gyro_head."):
                param.requires_grad = True
        return [p for p in model.velocity_branch.parameters() if p.requires_grad]
    if mode in ("lora", "lora_heads", "lora_velocity"):
        lora_params = configure_velocity_lora(model, lora_cfg)
        if not lora_params:
            raise ValueError(f"train_network={mode} requires a non-empty stage lora config")
        model.velocity_branch.train()
        for name, param in model.velocity_branch.named_parameters():
            is_lora = "lora_a" in name or "lora_b" in name
            is_head = (
                mode in ("lora_heads", "lora_velocity") and name.startswith("velocity_head.")
            ) or (mode == "lora_heads" and name.startswith("gyro_head."))
            param.requires_grad = is_lora or is_head
        return [p for p in model.velocity_branch.parameters() if p.requires_grad]
    raise ValueError(f"Unknown train_network mode: {mode}")


def integrate_variable_body_velocity(vel: torch.Tensor, dt: torch.Tensor, linear_scale=None):
    """p += R v dt, then R <- R Exp(w dt); linear_scale (the visual metric scale) multiplies v.

    The result is composed from tangent vectors (pure-translation and pure-rotation se3.Exp) instead of re-wrapping
    .tensor() slices: pypose's backward functions expect tangent-space gradients in the LieTensor slots, so re-wrapped
    slices bias the training gradient.
    """
    bsz, steps, dim = vel.shape
    rot = pp.identity_SO3(bsz, dtype=vel.dtype, device=vel.device)
    pos = vel.new_zeros(bsz, 3)
    linear = vel[..., :3] if linear_scale is None else vel[..., :3] * linear_scale
    angular = vel[..., 3:6] if dim >= 6 else vel.new_zeros(bsz, steps, 3)
    for i in range(steps):
        pos = pos + rot.Act(linear[:, i]) * dt[:, i : i + 1]
        rot = rot * pp.so3(angular[:, i] * dt[:, i : i + 1]).Exp()
    zero = vel.new_zeros(bsz, 3)
    return pp.se3(torch.cat([pos, zero], dim=-1)).Exp() * pp.se3(torch.cat([zero, rot.Log().tensor()], dim=-1)).Exp()


def scale_body_motion(T, scale):
    """Constant body motions (motion cache) with the metric scale on the translation, composed like
    integrate_variable_body_velocity so the scale gradient stays tangent-consistent."""
    tensor = T.tensor().detach()
    zero = tensor.new_zeros(tensor.shape[:-1] + (3,))
    rotation = pp.SO3(tensor[..., 3:]).Log().tensor()
    return pp.se3(torch.cat([tensor[..., :3] * scale, zero], dim=-1)).Exp() * pp.se3(torch.cat([zero, rotation], dim=-1)).Exp()


def exact_image_interval_dt(
    time_context: torch.Tensor,
    window: int,
    length: int,
    t0: float,
    t1: float,
) -> torch.Tensor:
    """Duration of each robot interval that overlaps the exact image interval."""
    interval_times = time_context[window - 1 : window + length]
    interval_start = interval_times[:-1]
    interval_end = interval_times[1:]
    image_start = interval_start.new_tensor(float(t0))
    image_end = interval_end.new_tensor(float(t1))
    overlap_start = torch.maximum(interval_start, image_start)
    overlap_end = torch.minimum(interval_end, image_end)
    return (overlap_end - overlap_start).clamp_min(0.0).float()


def frozen_feature_cache(model: PoseEstimator) -> dict | None:
    """The per-run encoder feature cache, if the encoder is frozen and in eval mode (heads training)."""
    cache = getattr(model, "frozen_feature_cache", None)
    branch = model.velocity_branch
    if cache is None or branch.training or any(p.requires_grad for p in branch.encoder.parameters()) \
            or any(p.requires_grad for p in branch.input.parameters()):
        return None
    return cache


def predict_body_delta_batch(
    model: PoseEstimator,
    batch: list[dict[str, Any]],
    normalizer: Normalizer,
    device: torch.device,
    unique_samples: bool = False,
    linear_scale=None,
) -> tuple[Any, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Predict and integrate every visual pair with one batched network call (linear_scale: metric scale of v).

    Each pair can span a different number of robot intervals. All fixed-size
    sensor windows are concatenated for the velocity network, then its outputs
    are split and zero-padded for a single batched pose integration. A zero dt
    makes padded integration steps exact identities.
    """
    if not batch:
        raise ValueError("Cannot predict an empty visual batch")

    window = int(model.cfg["data"]["window"])
    window_chunks: list[torch.Tensor] = []
    gyro_chunks: list[torch.Tensor] = []
    dt_chunks: list[torch.Tensor] = []
    lengths: list[int] = []
    sample_keys = []

    for item in batch:
        x_context = item["x_context"].to(device).float().unsqueeze(0)
        time_context = item["time_context"].to(device).double()
        pair = item["pair"]
        length = int(pair.robot_i1 - pair.robot_i0)
        if length < 1:
            raise ValueError("x_context is too short for velocity prediction")

        gyro = recover_gyro_sequence(x_context[:, window - 1 : window - 1 + length], normalizer).squeeze(0)
        windows = x_context.unfold(1, window, 1).transpose(-1, -2).contiguous()
        windows = windows[:, :length].reshape(length, window, model.input_dim)
        dt = exact_image_interval_dt(time_context, window, length, pair.t0, pair.t1)

        window_chunks.append(windows)
        gyro_chunks.append(gyro)
        dt_chunks.append(dt)
        lengths.append(length)
        if unique_samples:
            sample_keys.extend((pair.sequence_id, i) for i in range(pair.robot_i0, pair.robot_i1))

    all_windows = torch.cat(window_chunks, dim=0)
    all_gyro = torch.cat(gyro_chunks, dim=0)
    feature_cache = frozen_feature_cache(model)
    if feature_cache is not None:
        # Frozen encoder in eval mode: its feature of a sensor window never changes, so every distinct
        # window is encoded once per training run; only the heads see gradients.
        keys = [(item["pair"].sequence_id, i) for item in batch
                for i in range(item["pair"].robot_i0, item["pair"].robot_i1)]
        missing = [j for j, key in enumerate(keys) if key not in feature_cache]
        if missing:
            with torch.no_grad():
                encoded = model.velocity_branch.forward_features(all_windows[missing])
            for j, h in zip(missing, encoded):
                feature_cache.setdefault(keys[j], h)
        hidden = torch.stack([feature_cache[key] for key in keys])
        all_pred, all_refined_gyro, _, all_gyro_correction = model.velocity_branch.forward_heads(hidden, all_gyro)
    elif unique_samples:
        # Overlapping keyframe intervals/windows share sensor predictions, even
        # with dropout enabled. Each distinct sensor window is inferred once.
        positions, selected, inverse = {}, [], []
        for i, key in enumerate(sample_keys):
            if key not in positions:
                positions[key] = len(selected)
                selected.append(i)
            inverse.append(positions[key])
        selected = torch.tensor(selected, device=device)
        inverse = torch.tensor(inverse, device=device)
        pred, gyro, _, correction = model.velocity_branch.forward_with_gyro(
            all_windows[selected], all_gyro[selected])
        all_pred, all_refined_gyro, all_gyro_correction = pred[inverse], gyro[inverse], correction[inverse]
    else:
        all_pred, all_refined_gyro, _, all_gyro_correction = model.velocity_branch.forward_with_gyro(
            all_windows, all_gyro
        )
    all_vel = torch.cat([all_pred[..., :3], all_refined_gyro], dim=-1)

    velocity_chunks = list(torch.split(all_vel, lengths, dim=0))
    prediction_chunks = list(torch.split(all_pred, lengths, dim=0))
    gyro_correction_chunks = list(torch.split(all_gyro_correction, lengths, dim=0))
    padded_velocities = pad_sequence(velocity_chunks, batch_first=True, padding_value=0.0)
    padded_dt = pad_sequence(dt_chunks, batch_first=True, padding_value=0.0)
    transforms = integrate_variable_body_velocity(padded_velocities, padded_dt, linear_scale)
    return transforms, velocity_chunks, prediction_chunks, gyro_correction_chunks


def deadzone_squared_mean(value: torch.Tensor, deadzone: float) -> torch.Tensor:
    """Squared penalty on the portion of each value outside a symmetric deadzone."""
    if deadzone < 0.0:
        raise ValueError(f"deadzone must be non-negative, got {deadzone}")
    return F.relu(value.abs() - deadzone).square().mean()


def gyro_correction_smooth_loss(correction, times):
    """Time-weighted squared correction derivative, in (rad/s²)²."""
    if len(correction) < 2:
        return correction.sum() * 0.0
    dt = torch.diff(times).to(correction)
    valid = dt > 0
    if not valid.any():
        return correction.sum() * 0.0
    rate = torch.diff(correction, dim=0)[valid] / dt[valid, None]
    return (rate.square().mean(-1) * dt[valid]).sum() / dt[valid].sum()


def recover_gyro_sequence(x_eval: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    mean = torch.as_tensor(normalizer.mean[:3], dtype=x_eval.dtype, device=x_eval.device)
    std = torch.as_tensor(normalizer.std[:3], dtype=x_eval.dtype, device=x_eval.device)
    return x_eval[..., :3] * std + mean


def matched_keypoints(kp0: torch.Tensor, kp1: torch.Tensor) -> dict[str, torch.Tensor]:
    """Keypoints of frame 0 whose track id (column 0) also appears in frame 1, in frame-0 order; a repeated id in
    frame 1 matches its last occurrence."""
    empty = {"xy0": kp0.new_zeros((0, 2)), "xy1": kp0.new_zeros((0, 2)), "p0": kp0.new_zeros((0, 3)),
             "p1": kp0.new_zeros((0, 3)), "score": kp0.new_zeros((0,)),
             "valid0": kp0.new_zeros((0,), dtype=torch.bool), "valid1": kp0.new_zeros((0,), dtype=torch.bool)}
    if kp0.numel() == 0 or kp1.numel() == 0:
        return empty
    ids0 = kp0[:, 0].long()
    ids1, order = torch.sort(kp1[:, 0].long(), stable=True)
    j = torch.searchsorted(ids1, ids0, right=True) - 1           # last occurrence of each id in frame 1
    found = (j >= 0) & (ids1[j.clamp_min(0)] == ids0)
    if not bool(found.any()):
        return empty
    i0 = torch.nonzero(found).squeeze(1)
    i1 = order[j[found]]
    p0 = kp0[i0, 6:9]
    p1 = kp1[i1, 6:9]
    return {
        "xy0": kp0[i0, 2:4],
        "xy1": kp1[i1, 2:4],
        "p0": p0,
        "p1": p1,
        "score": 0.5 * (kp0[i0, 1] + kp1[i1, 1]).clamp_min(0.0),
        "valid0": torch.isfinite(p0).all(dim=-1),
        "valid1": torch.isfinite(p1).all(dim=-1),
    }


# Keypoint matches of trainable stages, memoized per (kp0 file, kp1 file) for the life of the process: the matches
# depend only on the keypoint files, which are fixed within one training run (visual_data of one round).
_MATCH_MEMO: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def memoized_matches(item: dict[str, Any]) -> dict[str, torch.Tensor]:
    """matched_keypoints of an item, reused across steps and epochs when the item names its keypoint files."""
    if "matches" in item:
        return item["matches"]
    key = item.get("match_key")
    if key is None and "pair" in item:
        key = (str(item["pair"].kp0_path), str(item["pair"].kp1_path))
    if key is None:
        return matched_keypoints(item["kp0"].float(), item["kp1"].float())
    matches = _MATCH_MEMO.get(key)
    if matches is None:
        matches = {name: value.contiguous() for name, value in
                   matched_keypoints(item["kp0"].float(), item["kp1"].float()).items()}
        _MATCH_MEMO[key] = matches
    return matches


def _visual_pair_cache_signature(dataset: VisualPairDataset, indices: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"matched-keypoints-v1")
    for index in indices:
        pair = dataset.pairs[index]
        digest.update(str(index).encode("ascii"))
        for path in (pair.kp0_path, pair.kp1_path):
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def prepare_frozen_visual_pair_cache(
    dataset: VisualPairDataset,
    indices: list[int],
    cache_path: Path,
) -> None:
    """Pre-match keypoints once so frozen stages avoid text I/O and Python ID matching."""
    unique_indices = sorted(set(int(index) for index in indices))
    signature = _visual_pair_cache_signature(dataset, unique_indices)
    cached = None
    if cache_path.exists():
        candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") == signature and candidate.get("indices") == unique_indices:
            cached = candidate.get("matches")
            print(f"[visual_pair_cache] hit pairs={len(unique_indices)} path={cache_path}", flush=True)
    if cached is None:
        started = time.time()
        cached = []
        for position, index in enumerate(unique_indices, start=1):
            pair = dataset.pairs[index]
            kp0 = torch.from_numpy(load_keypoints(pair.kp0_path))
            kp1 = torch.from_numpy(load_keypoints(pair.kp1_path))
            cached.append({key: value.cpu().contiguous() for key, value in matched_keypoints(kp0, kp1).items()})
            if position == 1 or position % 2000 == 0 or position == len(unique_indices):
                print(
                    f"[visual_pair_cache] build={position}/{len(unique_indices)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"version": 1, "signature": signature, "indices": unique_indices, "matches": cached},
            cache_path,
        )
    if len(cached) != len(unique_indices):
        raise ValueError(f"Invalid visual pair cache length in {cache_path}")
    for index, matches in zip(unique_indices, cached):
        dataset.precomputed_matches[index] = matches
    dataset.use_frozen_pair_cache = True


def _camera_motion_pnp_ransac(
    points0: np.ndarray,
    pixels1: np.ndarray,
    camera_matrix: np.ndarray,
    iterations: int,
    reprojection_threshold_px: float,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Estimate camera-0 to camera-1 motion without using noisy frame-1 depth."""
    if points0.shape[0] < 6:
        return None
    object_points = np.ascontiguousarray(points0, dtype=np.float64)
    image_points = np.ascontiguousarray(pixels1, dtype=np.float64)
    k = np.ascontiguousarray(camera_matrix, dtype=np.float64)
    success, rvec, translation, inlier_indices = cv2.solvePnPRansac(
        object_points,
        image_points,
        k,
        None,
        iterationsCount=max(1, int(iterations)),
        reprojectionError=float(reprojection_threshold_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inlier_indices is None or len(inlier_indices) < 6:
        return None
    inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, translation = cv2.solvePnPRefineLM(
            object_points[inlier_indices], image_points[inlier_indices], k, None, rvec, translation
        )
    inlier_mask = np.zeros(points0.shape[0], dtype=bool)
    inlier_mask[inlier_indices] = True
    return Rotation.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix(), np.asarray(translation).reshape(3), inlier_mask


def _se3_matrix(xyz_quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(xyz_quat_xyzw[3:]).as_matrix()
    matrix[:3, 3] = xyz_quat_xyzw[:3]
    return matrix


def geometric_initialization_candidate_indices(
    dataset: VisualPairDataset,
    motion_cache: dict[str, Any],
    cfg: dict,
) -> list[int]:
    """Cheap shortlist before parsing and matching keypoint text files."""
    init_cfg = cfg.get("extrinsic", {}).get("geometric_initialization", {})
    requested = max(6, int(init_cfg.get("num_pairs", 100)))
    multiplier = max(1, int(init_cfg.get("candidate_multiplier", 10)))
    transforms = motion_cache["pair_transforms"]
    candidates: list[tuple[float, int, str]] = []
    for index, pair in enumerate(dataset.pairs):
        motion = transforms[index]
        rotation = pp.SO3(motion[3:]).Log().tensor()
        richness = float(torch.linalg.norm(motion[:3]) + 0.25 * torch.linalg.norm(rotation))
        # Text size is a cheap proxy for feature count and avoids parsing every
        # candidate before the actual shortlist is known.
        feature_proxy = min(pair.kp0_path.stat().st_size, pair.kp1_path.stat().st_size)
        candidates.append((feature_proxy * max(richness, 1e-4), index, pair.sequence_id))
    limit = requested * multiplier
    buckets: dict[str, list[tuple[float, int]]] = {}
    for score, index, sequence_id in candidates:
        buckets.setdefault(sequence_id, []).append((score, index))
    per_bucket = max(1, math.ceil(limit / max(1, len(buckets))))
    selected: list[int] = []
    for rows in buckets.values():
        rows.sort(reverse=True)
    bucket_rows = list(buckets.values())
    for rank in range(per_bucket):
        for rows in bucket_rows:
            if rank < len(rows):
                selected.append(rows[rank][1])
                if len(selected) >= limit:
                    return selected
    if len(selected) < limit:
        selected_set = set(selected)
        remaining = sorted(candidates, reverse=True)
        selected.extend(index for _, index, _ in remaining if index not in selected_set)
    return selected[:limit]


def estimate_geometric_initial_extrinsic(
    dataset: VisualPairDataset,
    motion_cache: dict[str, Any],
    calibration: VisualCalibration,
    cfg: dict,
    output_dir: Path,
) -> bool:
    """Initialize body-camera extrinsic and odometry scale without network training."""
    init_cfg = cfg.get("extrinsic", {}).get("geometric_initialization", {})
    requested = max(6, int(init_cfg.get("num_pairs", 100)))
    candidate_multiplier = max(1, int(init_cfg.get("candidate_multiplier", 10)))
    min_matches = max(6, int(init_cfg.get("min_pnp_points", init_cfg.get("min_3d_matches", 20))))
    min_inliers = max(6, int(init_cfg.get("min_pnp_inliers", 20)))
    min_inlier_ratio = float(init_cfg.get("min_pnp_inlier_ratio", 0.15))
    min_depth = float(cfg.get("visual", {}).get("min_depth_m", 0.05))
    max_depth = float(cfg.get("visual", {}).get("max_depth_m", 80.0))
    transforms = motion_cache["pair_transforms"]
    candidates = []
    for index, pair in enumerate(dataset.pairs):
        matches = dataset.precomputed_matches[index]
        if matches is None:
            continue
        valid = (
            matches["valid0"]
            & (matches["p0"][:, 2] > min_depth)
            & (matches["p0"][:, 2] < max_depth)
            & torch.isfinite(matches["xy1"]).all(dim=-1)
        )
        count = int(valid.sum())
        if count < min_matches:
            continue
        motion = transforms[index]
        rotvec = pp.SO3(motion[3:]).Log().tensor()
        vector = torch.cat([motion[:3], rotvec]).double().numpy()
        richness = float(np.linalg.norm(vector[:3]) + 0.25 * np.linalg.norm(vector[3:]))
        candidates.append((index, count, richness, vector, valid))
    if len(candidates) < 6:
        print(f"[geometric_init] insufficient candidates={len(candidates)}; keep existing initialization", flush=True)
        return False

    candidates.sort(key=lambda row: row[1] * max(row[2], 1e-4), reverse=True)
    pool_limit = max(requested, requested * candidate_multiplier)
    candidate_buckets: dict[str, list[tuple[Any, ...]]] = {}
    for row in candidates:
        pair = dataset.pairs[row[0]]
        candidate_buckets.setdefault(pair.sequence_id, []).append(row)
    pool = []
    bucket_rows = list(candidate_buckets.values())
    for rank in range(max(len(rows) for rows in bucket_rows)):
        for rows in bucket_rows:
            if rank < len(rows):
                pool.append(rows[rank])
                if len(pool) >= pool_limit:
                    break
        if len(pool) >= pool_limit:
            break
    features = np.stack([row[3] for row in pool])
    scale = np.std(features, axis=0)
    normalized = (features - np.mean(features, axis=0)) / np.maximum(scale, 1e-4)
    quality = np.asarray([math.log1p(row[1]) * max(row[2], 1e-4) for row in pool])
    quality /= max(float(quality.max()), 1e-12)
    ordered_positions = [int(np.argmax(quality))]
    min_distance = np.linalg.norm(normalized - normalized[ordered_positions[0]], axis=1)
    while len(ordered_positions) < len(pool):
        score = min_distance * (0.25 + 0.75 * quality)
        score[ordered_positions] = -1.0
        chosen = int(np.argmax(score))
        ordered_positions.append(chosen)
        min_distance = np.minimum(min_distance, np.linalg.norm(normalized - normalized[chosen], axis=1))

    rng = np.random.default_rng(int(init_cfg.get("seed", cfg.get("train", {}).get("seed", 42))))
    cv2.setRNGSeed(int(init_cfg.get("seed", cfg.get("train", {}).get("seed", 42))))
    observations = []
    diagnostics = []
    sequence_names = {dataset.pairs[row[0]].sequence_id for row in pool}
    max_per_sequence = max(1, math.ceil(requested * float(init_cfg.get("max_sequence_fraction", 0.35))))
    sequence_counts: dict[str, int] = {}
    attempted = 0
    for position in ordered_positions:
        if len(observations) >= requested:
            break
        index, count, richness, _, valid = pool[position]
        pair = dataset.pairs[index]
        if sequence_counts.get(pair.sequence_id, 0) >= max_per_sequence:
            continue
        attempted += 1
        matches = dataset.precomputed_matches[index]
        points0 = matches["p0"][valid].double().numpy()
        pixels1 = matches["xy1"][valid].double().numpy()
        camera_matrix = dataset.sequence_cache[pair.sequence_id]["k"]
        estimate = _camera_motion_pnp_ransac(
            points0,
            pixels1,
            camera_matrix,
            int(init_cfg.get("pnp_ransac_iterations", 100)),
            float(init_cfg.get("pnp_reprojection_threshold_px", 3.0)),
            float(init_cfg.get("pnp_confidence", 0.999)),
        )
        if estimate is None:
            continue
        camera_rotation, camera_translation, inliers = estimate
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / max(1, count)
        if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
            continue
        body = _se3_matrix(transforms[index].double().numpy())
        camera = np.eye(4, dtype=np.float64)
        camera[:3, :3], camera[:3, 3] = camera_rotation, camera_translation
        observations.append((body, camera, math.sqrt(inlier_count)))
        sequence_counts[pair.sequence_id] = sequence_counts.get(pair.sequence_id, 0) + 1
        diagnostics.append(
            {
                "pair_index": index,
                "sequence_id": pair.sequence_id,
                "matches": count,
                "inliers": inlier_count,
                "inlier_ratio": inlier_ratio,
                "motion_richness": richness,
            }
        )
    min_usable = max(6, int(init_cfg.get("min_usable_motions", 50)))
    if len(observations) < min_usable or len(sequence_counts) < min(2, len(sequence_names)):
        print(
            f"[geometric_init] rejected usable={len(observations)}/{min_usable} "
            f"sequences={dict(sequence_counts)}; keep existing initialization",
            flush=True,
        )
        return False

    rotation_sigma = max(float(init_cfg.get("rotation_sigma_rad", 0.1)), 1e-6)
    translation_sigma = max(float(init_cfg.get("translation_sigma_m", 0.05)), 1e-6)

    def residual(parameters: np.ndarray) -> np.ndarray:
        translation, rotvec, log_scale = parameters[:3], parameters[3:6], parameters[6]
        x_matrix = np.eye(4, dtype=np.float64)
        x_matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        x_matrix[:3, 3] = translation
        motion_scale = math.exp(float(log_scale))
        rows = []
        for body, camera, weight in observations:
            body_scaled = body.copy()
            body_scaled[:3, 3] *= motion_scale
            predicted = np.linalg.inv(body_scaled @ x_matrix) @ x_matrix
            rotation_error = Rotation.from_matrix(predicted[:3, :3].T @ camera[:3, :3]).as_rotvec()
            translation_error = predicted[:3, 3] - camera[:3, 3]
            rows.append(weight * np.concatenate([translation_error / translation_sigma, rotation_error / rotation_sigma]))
        return np.concatenate(rows)

    ext_cfg = cfg.get("extrinsic", {})
    scale_min = float(ext_cfg.get("scale_min", 0.1))
    scale_max = float(ext_cfg.get("scale_max", 10.0))
    starts = [np.zeros(7, dtype=np.float64)]
    for rotation in Rotation.random(int(init_cfg.get("rotation_restarts", 12)), random_state=rng):
        start = np.zeros(7, dtype=np.float64)
        start[3:6] = rotation.as_rotvec()
        starts.append(start)
    solutions = [
        least_squares(
            residual,
            start,
            bounds=(np.r_[[-1.0] * 3, [-2.0 * math.pi] * 3, math.log(scale_min)], np.r_[[1.0] * 3, [2.0 * math.pi] * 3, math.log(scale_max)]),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(init_cfg.get("max_nfev", 500)),
        )
        for start in starts
    ]
    solution = min(solutions, key=lambda item: float(np.mean(residual(item.x) ** 2)))
    singular_values = np.linalg.svd(solution.jac, compute_uv=False)
    jacobian_condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
    max_condition = float(init_cfg.get("max_jacobian_condition", 1e6))
    if not np.isfinite(jacobian_condition) or jacobian_condition > max_condition:
        print(
            f"[geometric_init] rejected ill-conditioned solution condition={jacobian_condition:.3e} "
            f"limit={max_condition:.3e}; keep existing initialization",
            flush=True,
        )
        return False
    xyz = solution.x[:3]
    quaternion = Rotation.from_rotvec(solution.x[3:6]).as_quat()
    scale_value = math.exp(float(solution.x[6]))
    se3 = pp.SE3(torch.tensor(np.concatenate([xyz, quaternion]), dtype=calibration.delta_body_cam.dtype))
    with torch.no_grad():
        calibration.delta_body_cam.copy_(se3.Log().tensor().to(calibration.delta_body_cam.device))
        calibration.log_scale_delta.copy_(calibration.log_scale_delta.new_tensor(math.log(scale_value)))
        calibration.clamp_scale_()
    payload = {
        "method": "stereo_3d_to_2d_pnp_ransac_plus_robust_hand_eye",
        "num_candidates": len(candidates),
        "num_attempted": attempted,
        "num_usable_motions": len(observations),
        "usable_by_sequence": sequence_counts,
        "optimizer_success": bool(solution.success),
        "optimizer_cost": float(solution.cost),
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition": jacobian_condition,
        "initial_calibration": calibration.summary(),
        "selected_pairs": diagnostics,
    }
    path = output_dir / "geometric_initial_extrinsic.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[geometric_init] usable={len(observations)} xyz={xyz.tolist()} "
        f"quat_xyzw={quaternion.tolist()} scale={scale_value:.6f} path={path}",
        flush=True,
    )
    return True








def _segment_mean(values: torch.Tensor, segment: torch.Tensor, count: int) -> torch.Tensor:
    """Mean of values per segment id in [0, count); 0 for an empty segment."""
    total = values.new_zeros(count).index_add(0, segment, values)
    n = torch.bincount(segment, minlength=count).to(values.dtype)
    return torch.where(n > 0, total / n.clamp_min(1.0), total)


def visual_geometry_loss_batch(
    items: list[dict[str, Any]],
    T_b0_b1,
    calibration: VisualCalibration,
    cfg: dict,
    device: torch.device,
    T_body_cam=None,
) -> dict[str, torch.Tensor]:
    """Per-pair visual losses of all pairs in one pass: T_b0_b1 is a metric pp.SE3 batch [B]; returns [B] tensors.

    Per pair: Huber mean of the two-way reprojection residuals of the matches with depth, Huber mean of the
    Sampson distance of the matches without depth; pairs with too few matches or a too small predicted
    translation contribute zeros (num_matches = 0).
    """
    visual = cfg.get("visual", {})
    min_matches = int(visual.get("min_matches", 20))
    min_translation = float(visual.get("min_pred_translation_m", 0.0))
    min_depth = float(visual.get("min_depth_m", 0.05))
    max_depth = float(visual.get("max_depth_m", 80.0))
    reproj_delta = float(visual.get("reprojection_delta_px", 3.0))
    sampson_delta = float(visual.get("sampson_delta_px", 1.5))
    B = len(items)
    matches = [memoized_matches(item) for item in items]
    counts = torch.as_tensor([m["xy0"].shape[0] for m in matches], dtype=torch.long)
    T = T_b0_b1.tensor().reshape(B, 7)
    translation_norm = torch.linalg.norm(T[:, :3], dim=-1).detach().cpu()  # one sync for the whole batch
    usable = (counts >= min_matches) & (translation_norm >= min_translation)
    zero = T.new_zeros(B)
    if not bool(usable.any()):
        return {"reprojection": zero, "epipolar": zero, "num_matches": zero, "num_reprojection": zero}

    keep = torch.nonzero(usable).squeeze(1).tolist()
    cat = {key: torch.cat([matches[i][key] for i in keep]).to(device) for key in ("xy0", "xy1", "p0", "p1", "valid0", "valid1")}
    seg = torch.repeat_interleave(torch.arange(len(keep)), counts[keep]).to(device)   # position in `keep`
    k = torch.stack([items[i]["k"] for i in keep]).to(device).float()                  # [U, 3, 3]

    if T_body_cam is None:
        T_body_cam = calibration.body_cam()
    # LieTensor operations only (indexing, products, Act, rotation().matrix(), translation = Act(0)): see
    # integrate_variable_body_velocity
    T_u = T_b0_b1[keep]
    T_c1_c0 = (T_u * T_body_cam).Inv() * T_body_cam
    T_c0_c1 = T_c1_c0.Inv()

    p0, p1 = cat["p0"], cat["p1"]
    valid0 = cat["valid0"] & (p0[:, 2] > min_depth) & (p0[:, 2] < max_depth)
    valid1 = cat["valid1"] & (p1[:, 2] > min_depth) & (p1[:, 2] < max_depth)

    def projected(points, k_):
        z = points[:, 2:3].clamp_min(1e-6)
        norm = points[:, :2] / z
        return torch.stack([k_[:, 0, 0] * norm[:, 0] + k_[:, 0, 2], k_[:, 1, 1] * norm[:, 1] + k_[:, 1, 2]], dim=-1)

    residuals, owners = [], []
    for valid, T_ab, points, target in ((valid0, T_c1_c0, p0, cat["xy1"]), (valid1, T_c0_c1, p1, cat["xy0"])):
        idx = torch.nonzero(valid).squeeze(1)
        if idx.numel() == 0:
            continue
        s_ = seg[idx]
        pred = T_ab[s_].Act(points[idx])
        front = pred[:, 2] > min_depth
        idx, s_, pred = idx[front], s_[front], pred[front]
        if idx.numel() == 0:
            continue
        residuals.append((projected(pred, k[s_]) - target[idx]).reshape(-1))
        owners.append(s_.repeat_interleave(2))
    U = len(keep)
    if residuals:
        res = torch.cat(residuals)
        huber = F.huber_loss(res, torch.zeros_like(res), delta=reproj_delta, reduction="none")
        reproj_u = _segment_mean(huber, torch.cat(owners), U)
    else:
        reproj_u = T.new_zeros(U)

    epi = ~(valid0 | valid1)
    idx = torch.nonzero(epi).squeeze(1)
    if idx.numel() > 0:
        r = T_c1_c0.rotation().matrix()                                     # [U, 3, 3]
        t = T_c1_c0.Act(torch.zeros(len(keep), 3, dtype=r.dtype, device=r.device))
        tx = torch.zeros_like(r)
        tx[:, 0, 1], tx[:, 0, 2], tx[:, 1, 2] = -t[:, 2], t[:, 1], -t[:, 0]
        tx[:, 1, 0], tx[:, 2, 0], tx[:, 2, 1] = t[:, 2], -t[:, 1], t[:, 0]
        kinv = torch.linalg.inv(k)
        fmat = kinv.transpose(1, 2) @ tx @ r @ kinv                         # [U, 3, 3]
        s_ = seg[idx]
        one = torch.ones((idx.numel(), 1), dtype=p0.dtype, device=device)
        x0 = torch.cat([cat["xy0"][idx], one], dim=-1)
        x1 = torch.cat([cat["xy1"][idx], one], dim=-1)
        f = fmat[s_]
        fx0 = (f @ x0.unsqueeze(-1)).squeeze(-1)
        ftx1 = (f.transpose(1, 2) @ x1.unsqueeze(-1)).squeeze(-1)
        numer = (x1 * fx0).sum(dim=-1).square()
        denom = fx0[:, 0].square() + fx0[:, 1].square() + ftx1[:, 0].square() + ftx1[:, 1].square()
        dist = torch.sqrt((numer / denom.clamp_min(1e-8)).clamp_min(0.0) + 1e-8)
        huber = F.huber_loss(dist, torch.zeros_like(dist), delta=sampson_delta, reduction="none")
        epi_u = _segment_mean(huber, s_, U)
    else:
        epi_u = T.new_zeros(U)

    keep_t = torch.as_tensor(keep, device=device)
    n_reproj = (torch.bincount(seg[valid0], minlength=U) + torch.bincount(seg[valid1], minlength=U)).to(T.dtype)
    out = {
        "reprojection": zero.index_put((keep_t,), reproj_u),
        "epipolar": zero.index_put((keep_t,), epi_u),
        "num_matches": zero.index_put((keep_t,), counts[keep].to(device=device, dtype=T.dtype)),
        "num_reprojection": zero.index_put((keep_t,), n_reproj),
    }
    return out


def batch_loss(
    model: PoseEstimator,
    calibration: VisualCalibration,
    batch: list[dict[str, Any]],
    normalizer: Normalizer,
    cfg: dict,
    stage_cfg: dict,
    device: torch.device,
    motion_cache: dict[str, Any] | None = None,
    predictions=None,
) -> dict[str, torch.Tensor]:
    if batch and "window_items" in batch[0]:
        return window_batch_loss(model, calibration, batch, normalizer, cfg, stage_cfg, device, motion_cache)
    loss_cfg = cfg.get("loss", {})
    use_extrinsic_prior = bool(cfg.get("extrinsic", {}).get("use_extrinsic_prior", False))
    use_scale_prior = bool(stage_cfg.get("use_scale_prior", stage_cfg.get("name") != "extrinsic"))
    use_velocity_smooth = bool(stage_cfg.get("use_velocity_smooth", stage_cfg.get("name") != "extrinsic"))
    totals: dict[str, torch.Tensor] = {}
    if motion_cache is None:
        transforms, _, linear_velocities, gyro_corrections = (
            predictions if predictions is not None else predict_body_delta_batch(
                model, batch, normalizer, device, linear_scale=calibration.scale())
        )
    else:
        transforms = scale_body_motion(cached_body_delta_batch(batch, motion_cache, device), calibration.scale())
        linear_velocities = []
        gyro_corrections = []
    pair_weights = torch.as_tensor(
        [float(item.get("pair_weight", 1.0)) for item in batch],
        dtype=calibration.scale().dtype,
        device=device,
    )
    weight_denom = pair_weights.sum().clamp_min(1e-8)
    geom = visual_geometry_loss_batch(batch, transforms, calibration, cfg, device, calibration.body_cam())
    totals = {key: (pair_weights * value).sum() / weight_denom for key, value in geom.items()}
    smooth_terms = [
        ((v[1:] - v[:-1]).square().mean(), pair_weights[i])
        for i, v in enumerate(linear_velocities)
        if v.shape[0] > 1
    ]
    if smooth_terms:
        smooth_values = torch.stack([term for term, _ in smooth_terms])
        smooth_weights = torch.stack([weight for _, weight in smooth_terms])
        smooth = (smooth_values * smooth_weights).sum() / smooth_weights.sum().clamp_min(1e-8)
    else:
        smooth = calibration.scale().new_zeros(())
    gyro_deadzone = float(loss_cfg.get("gyro_correction_deadzone", 0.02))
    gyro_terms = [
        (deadzone_squared_mean(correction, gyro_deadzone), pair_weights[i])
        for i, correction in enumerate(gyro_corrections)
    ]
    if gyro_terms:
        gyro_values = torch.stack([term for term, _ in gyro_terms])
        gyro_weights = torch.stack([weight for _, weight in gyro_terms])
        gyro_correction_prior = (
            (gyro_values * gyro_weights).sum() / gyro_weights.sum().clamp_min(1e-8)
        )
    else:
        gyro_correction_prior = calibration.scale().new_zeros(())
    zero = calibration.scale().new_zeros(())
    totals["extrinsic_prior"] = calibration.prior_loss() if use_extrinsic_prior else zero
    totals["scale_prior"] = calibration.scale_prior_loss() if use_scale_prior else zero
    totals["velocity_smooth"] = smooth if use_velocity_smooth else zero
    totals["gyro_correction_prior"] = gyro_correction_prior
    # Bias belongs to BA; the head may only add a zero-mean, motion-related correction.
    by_sequence = {}
    for correction, item in zip(gyro_corrections, batch):
        by_sequence.setdefault(item["pair"].sequence_id, []).append(correction)
    totals["gyro_correction_mean"] = (
        torch.stack([torch.cat(parts).mean(0).square().sum() for parts in by_sequence.values()]).mean()
        if by_sequence else zero
    )
    window = int(model.cfg["data"]["window"])
    gyro_smooth = [gyro_correction_smooth_loss(
        correction, item["time_context"][window - 1:window - 1 + len(correction)]
    ) for correction, item in zip(gyro_corrections, batch)]
    totals["gyro_correction_smooth"] = (
        (torch.stack(gyro_smooth) * pair_weights).sum() / weight_denom
        if gyro_smooth else zero
    )
    totals["loss"] = (
        float(loss_cfg.get("reprojection_weight", 1.0)) * totals["reprojection"]
        + float(loss_cfg.get("epipolar_weight", 0.2)) * totals["epipolar"]
        + float(loss_cfg.get("extrinsic_prior_weight", 0.01)) * totals["extrinsic_prior"]
        + float(loss_cfg.get("scale_prior_weight", 0.1)) * totals["scale_prior"]
        + float(loss_cfg.get("smooth_velocity_weight", 0.001)) * totals["velocity_smooth"]
        + float(loss_cfg.get("gyro_correction_weight", 0.01)) * totals["gyro_correction_prior"]
        + float(loss_cfg.get("gyro_correction_smooth_weight", 0.0)) * totals["gyro_correction_smooth"]
        + float(loss_cfg.get("gyro_correction_mean_weight", 0.0)) * totals["gyro_correction_mean"]
        + 0.0 * calibration.log_scale_delta
    )
    return totals


def window_batch_loss(model, calibration, windows, normalizer, cfg, stage_cfg, device, motion_cache=None):
    edges = [edge for window in windows for edge in window["window_items"]]
    prediction = (predict_body_delta_batch(model, edges, normalizer, device, unique_samples=True,
                                           linear_scale=calibration.scale())
                  if motion_cache is None else (scale_body_motion(cached_body_delta_batch(edges, motion_cache, device),
                                                                  calibration.scale()), None, [], []))
    result = batch_loss(model, calibration, edges, normalizer, cfg, stage_cfg, device,
                        motion_cache=motion_cache, predictions=prediction)
    cross_items, cross_T = [], []
    for window in windows:
        items = window["window_items"]
        for j in range(1, len(items)):
            cross_items.append(dict(k=items[0]["k"], kp0=items[0]["kp0"], kp1=items[j]["kp1"],
                                    match_key=(str(items[0]["pair"].kp0_path), str(items[j]["pair"].kp1_path))))
    lengths = {len(window["window_items"]) for window in windows}
    if len(lengths) == 1 and lengths.pop() > 1:
        # Chronological composition T_0j = T_01 T_12 ... T_(j-1)j, never reset to the BA pose: every window has the
        # same number of edges, so compose all windows at once (same products in the same order as per window).
        edges_per_window = len(windows[0]["window_items"])
        relative = prediction[0].tensor().reshape(len(windows), edges_per_window, 7)
        cumulative = pp.SE3(relative[:, 0])
        for j in range(1, edges_per_window):
            cumulative = cumulative * pp.SE3(relative[:, j])
            cross_T.append(cumulative.tensor())
        cross_T = [torch.stack(cross_T, dim=1).reshape(-1, 7)]      # window-major, as cross_items
    else:
        offset = 0
        for window in windows:
            items = window["window_items"]
            cumulative = prediction[0][offset:offset + 1]
            for j in range(1, len(items)):
                cumulative = cumulative * prediction[0][offset + j:offset + j + 1]
                cross_T.append(cumulative.tensor())
            offset += len(items)
    geometry = (visual_geometry_loss_batch(cross_items, pp.SE3(torch.cat(cross_T)), calibration, cfg, device,
                                           calibration.body_cam()) if cross_items else None)
    used = geometry["num_matches"].detach() > 0 if geometry is not None else None
    n_cross = int(used.sum()) if used is not None else 0
    loss = cfg.get("loss", {})
    weight = float(cfg.get("visual", {}).get("cross_frame_weight", 1.0))
    for key in ("reprojection", "epipolar"):
        adjacent = result[key]
        distant = geometry[key][used].mean() if n_cross else adjacent * 0
        result["adjacent_" + key] = adjacent
        result["cross_" + key] = distant
        result[key] = adjacent + weight * distant
        result["loss"] = result["loss"] + weight * float(
            loss.get(key + "_weight", 1.0 if key == "reprojection" else 0.2)
        ) * distant
    result["cross_valid_edges"] = result["loss"].new_tensor(n_cross / len(windows))
    return result


def make_optimizer(
    model: PoseEstimator,
    calibration: VisualCalibration,
    stage_cfg: dict,
    train_params: list[nn.Parameter],
) -> torch.optim.Optimizer:
    groups = [
        {"params": [calibration.delta_body_cam], "lr": float(stage_cfg.get("extrinsic_lr", 1e-4))},
        {"params": [calibration.log_scale_delta], "lr": float(stage_cfg.get("log_scale_lr", stage_cfg.get("extrinsic_lr", 1e-4)))},
    ]
    network_lr = float(stage_cfg.get("network_lr", 0.0))
    if "lora_lr" in stage_cfg:
        # separate LoRA learning rate; the heads keep network_lr
        lora_params = [p for n, p in model.velocity_branch.named_parameters()
                       if p.requires_grad and ("lora_a" in n or "lora_b" in n)]
        lora_ids = {id(p) for p in lora_params}
        train_params = [p for p in train_params if id(p) not in lora_ids]
        if lora_params:
            groups.append({"params": lora_params, "lr": float(stage_cfg["lora_lr"])})
    if train_params and network_lr > 0.0:
        groups.append({"params": train_params, "lr": network_lr})
    return torch.optim.AdamW(groups, weight_decay=float(stage_cfg.get("weight_decay", 0.0)))


def run_stage(
    model: PoseEstimator,
    calibration: VisualCalibration,
    loader: DataLoader,
    eval_loader: DataLoader | None,
    normalizer: Normalizer,
    cfg: dict,
    stage_cfg: dict,
    device: torch.device,
    output_dir: Path,
    trajectory_dataset: VisualPairDataset | None = None,
    start_epoch: int = 1,
    previous_history: list[dict[str, Any]] | None = None,
    train_motion_cache: dict[str, Any] | None = None,
    eval_motion_cache: dict[str, Any] | None = None,
    implicit_term=None,
) -> list[dict[str, float]]:
    name = str(stage_cfg.get("name", "stage"))
    epochs = int(stage_cfg.get("epochs", 1))
    train_mode = stage_cfg.get("train_network", False)
    if train_mode in ("lora", "lora_heads", "lora_velocity"):
        configure_velocity_lora(model, stage_cfg.get("lora"))
    train_params = set_network_train_mode(model, train_mode, stage_cfg.get("lora"))
    # heads training: cache the frozen encoder features per sensor window (frozen_feature_cache: false disables)
    model.frozen_feature_cache = ({} if train_mode in ("head", "heads") and
                                  stage_cfg.get("frozen_feature_cache", cfg.get("train", {}).get("frozen_feature_cache", True))
                                  else None)
    calibration.log_scale_delta.requires_grad_(
        float(stage_cfg.get("log_scale_lr", stage_cfg.get("extrinsic_lr", 1e-4))) > 0
    )
    calibration.train()
    optimizer = make_optimizer(model, calibration, stage_cfg, train_params)
    history = []
    early_cfg = cfg.get("early_stopping", {})
    early = bool(early_cfg.get("enabled", False)) and (name == "scale" or train_mode not in (False, "false", "none"))
    stopper = validator = None
    best_path = output_dir / f"best_{_safe_cache_name(name)}.pt"
    if early:
        if trajectory_dataset is None:
            raise ValueError("Early stopping requires complete validation sequences")
        if start_epoch > 1:
            previous = [r for r in (previous_history or []) if r.get("stage") == name]
            if not previous or "early_stopping_state" not in previous[-1] or not best_path.exists():
                raise ValueError("Resume requires early-stopping state and best checkpoint in the same output directory")
            stopper = TrajectoryEarlyStopping(early_cfg, state=copy.deepcopy(previous[-1]["early_stopping_state"]))
        if stopper is None or (start_epoch <= epochs and not stopper.state["stopped"]):
            validator = FixedBATrajectoryValidation(
                model, normalizer, cfg, device, [s.sequence_id for s in trajectory_dataset.sequences])
        if stopper is None:
            baseline = validator.evaluate(model, calibration)
            stopper = TrajectoryEarlyStopping(early_cfg, baseline["trajectory_rmse"])
            save_visual_checkpoint(best_path, model, calibration, normalizer, previous_history or [])
            write_history(output_dir / f"trajectory_baseline_{_safe_cache_name(name)}.json", baseline)
    for epoch in range(start_epoch, epochs + 1):
        if stopper is not None and stopper.state["stopped"]:
            break
        # evaluate_visual_loader switches the module to eval mode; restore the
        # intended mode before every following training epoch.
        if stage_cfg.get("train_network", False) in ("lora", "lora_heads", "lora_velocity"):
            model.velocity_branch.train()
        else:
            model.velocity_branch.eval()
        # A derivative penalty should measure temporal changes, not independent
        # dropout masks. Keep one deterministic forward for all loss terms.
        if float(cfg.get("loss", {}).get("gyro_correction_smooth_weight", 0.0)) > 0:
            for module in model.velocity_branch.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = 0.0
        totals: dict[str, float] = {}
        count = 0
        t0 = time.time()
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            metrics = batch_loss(
                model, calibration, batch, normalizer, cfg, stage_cfg, device, train_motion_cache
            )
            if implicit_term is not None:   # implicit hypergradient through the fusion BA (common/implicit_gradient.py)
                implicit_value = implicit_term.loss()
                (metrics["loss"] + implicit_value.to(metrics["loss"].dtype)).backward()
                metrics["implicit_term"] = implicit_value.detach()
            else:
                metrics["loss"].backward()
            params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, float(cfg.get("train", {}).get("grad_clip", 1.0)))
            optimizer.step()
            calibration.clamp_scale_()
            row = {key: float(value.detach().cpu()) for key, value in metrics.items()}
            row["scale"] = float(calibration.scale().detach().cpu())
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            interval = int(cfg.get("train", {}).get("print_interval_steps", 10))
            if step == 1 or step % interval == 0 or step == len(loader):
                print(
                    f"[{name}] epoch={epoch}/{epochs} step={step}/{len(loader)} "
                    f"loss={row['loss']:.6f} reproj={row['reprojection']:.4f} "
                    f"epi={row['epipolar']:.4f} gyro_prior={row['gyro_correction_prior']:.6f} "
                    f"scale={row['scale']:.6f} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
        avg = {key: value / max(1, count) for key, value in totals.items()}
        avg["epoch"] = float(epoch)
        avg["stage"] = name
        # The visual validation loss is logged only (selection uses the trajectory validator below); a stage may
        # evaluate it every visual_eval_interval epochs, always at the last one.
        eval_interval = max(1, int(stage_cfg.get("visual_eval_interval", 1)))
        if eval_loader is not None and (epoch % eval_interval == 0 or epoch == epochs):
            eval_metrics = evaluate_visual_loader(
                model, calibration, eval_loader, normalizer, cfg, stage_cfg, device, eval_motion_cache
            )
            calibration_summary = calibration.summary()
            avg.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            avg["eval_translation_xyz"] = calibration_summary["translation_xyz"]
            avg["eval_rotation_quat_xyzw"] = calibration_summary["rotation_quat_xyzw"]
            avg["eval_rotation_rotvec"] = calibration_summary["rotation_rotvec"]
            avg["eval_scale"] = calibration_summary["scale"]
        if validator is not None:
            trajectory_metrics = validator.evaluate(model, calibration)
            avg.update({"eval_" + k: v for k, v in trajectory_metrics.items()})
        improved = False
        if stopper is not None:
            improved, avg["early_stopping"] = stopper.update(epoch, avg["eval_trajectory_rmse"])
            avg["early_stopping_state"] = copy.deepcopy(stopper.state)
        history.append(avg)
        all_history = [*(previous_history or []), *history]
        if improved:
            save_visual_checkpoint(best_path, model, calibration, normalizer, all_history)
        # Keep an epoch-level recovery point. Optimizer state is intentionally
        # recreated on resume, while model/calibration/history continue from
        # the latest fully completed epoch.
        save_visual_checkpoint(
            output_dir / "last_visual.pt",
            model,
            calibration,
            normalizer,
            all_history,
        )
        # Persist the complete summary immediately after every fully completed
        # epoch. Atomic replacement prevents an interrupted write from leaving
        # a truncated history.json.
        write_history(output_dir / "history.json", all_history)
        print(json.dumps(avg, indent=2), flush=True)
    if stopper is not None:
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"], strict=True)
        calibration.load_state_dict(best["visual_calibration"], strict=True)
        write_history(output_dir / f"early_stopping_{_safe_cache_name(name)}.json", dict(
            checkpoint=str(best_path.resolve()), metric=stopper.metric,
            baseline_score=stopper.state["baseline_score"], best_score=stopper.state["best_score"],
            best_epoch=stopper.state["best_epoch"], stopped=stopper.state["stopped"],
            completed_epoch=history[-1]["epoch"] if history else start_epoch - 1))
        print(f"[early_stopping] restored epoch={stopper.state['best_epoch']} checkpoint={best_path}", flush=True)
    return history


def evaluate_visual_loader(
    model: PoseEstimator,
    calibration: VisualCalibration,
    loader: DataLoader,
    normalizer: Normalizer,
    cfg: dict,
    stage_cfg: dict,
    device: torch.device,
    motion_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    was_training = model.velocity_branch.training
    model.velocity_branch.eval()
    calibration.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            metrics = batch_loss(
                model, calibration, batch, normalizer, cfg, stage_cfg, device, motion_cache
            )
            row = {key: float(value.detach().cpu()) for key, value in metrics.items()}
            row["scale"] = float(calibration.scale().detach().cpu())
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    if was_training:
        model.velocity_branch.train()
    calibration.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def save_visual_checkpoint(
    path: Path,
    model: PoseEstimator,
    calibration: VisualCalibration,
    normalizer: Normalizer,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": model.cfg,
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "visual_calibration": calibration.state_dict(),
        "visual_calibration_summary": calibration.summary(),
        "history": history,
    }
    lora_config = getattr(model.velocity_branch, "lora_config", None)
    if lora_config is not None:
        payload["lora_config"] = lora_config
    torch.save(payload, path)
    (path.parent / "visual_calibration.json").write_text(
        json.dumps(payload["visual_calibration_summary"], indent=2),
        encoding="utf-8",
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Finetune LearnVIL velocity with stereo feature geometry supervision.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--model",
        default=None,
        type=Path,
        help="Pretrained velocity checkpoint; optional when --resume or --init-visual is supplied.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default=None,
        type=Path,
        help="Resume model, visual calibration, and history from a visual checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-visual",
        default=None,
        type=Path,
        help=(
            "Initialize model, visual calibration, and scale from an existing visual checkpoint, "
            "but start every configured stage with a fresh history in a new output directory."
        ),
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Dataset root containing separate train/ and val/ real-sequence directories.",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fixed-calibration", action="store_true",
                        help="Keep checkpoint calibration and preprocessing fixed; train network parameters selected by the stage config.")
    args = parser.parse_args(argv)
    if args.model is None and args.resume is None and args.init_visual is None:
        parser.error("one of --model, --resume, or --init-visual is required")

    cfg = load_config(args.config, "sim_to_real")
    if not args.fixed_calibration:
        for stage in cfg.get("optim", {}).get("stages", []):
            if "lora" in stage or stage.get("train_network") in ("lora", "lora_heads", "lora_velocity"):
                parser.error("Initialization (without --fixed-calibration) does not support LoRA; configure alternating.lora for S2")
    # Supplying an extrinsic means using it as a fixed calibration in S2.
    ext = cfg.setdefault("extrinsic", {})
    ext["optimize_extrinsic"] = not bool(ext.get("use_initial_extrinsic", False))
    device = resolve_device(args.device or cfg.get("train", {}).get("device", "auto"))
    seed = int(cfg.get("train", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_path = args.resume or args.init_visual or args.model
    model, normalizer, checkpoint = load_velocity_model(checkpoint_path, device)
    if normalizer is None:
        normalizer_path = checkpoint_path.parent / "normalizer.npz"
        if not normalizer_path.exists() and args.model is not None:
            normalizer_path = args.model.parent / "normalizer.npz"
        normalizer = Normalizer.load(normalizer_path)
    model.to(device)
    calibration = VisualCalibration(cfg).to(device)
    configured_extrinsic = calibration.init_body_cam_tensor.detach().clone()
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        if "visual_calibration" not in checkpoint:
            raise ValueError(f"Resume checkpoint has no visual_calibration: {args.resume}")
        calibration.load_state_dict(checkpoint["visual_calibration"], strict=True)
        history = list(checkpoint.get("history", []))
        print(
            f"[resume] checkpoint={args.resume} completed_epochs={len(history)} "
            f"scale={float(calibration.scale().detach().cpu()):.6f}",
            flush=True,
        )
    elif args.init_visual is not None:
        if "visual_calibration" not in checkpoint:
            raise ValueError(f"Initialization checkpoint has no visual_calibration: {args.init_visual}")
        calibration.load_state_dict(checkpoint["visual_calibration"], strict=True)
        print(
            f"[init_visual] checkpoint={args.init_visual} history_reset=true "
            f"scale={float(calibration.scale().detach().cpu()):.6f}",
            flush=True,
        )
    if args.fixed_calibration:
        if args.init_visual is None and args.resume is None:
            parser.error("--fixed-calibration requires --init-visual or --resume")
        calibration.optimize_extrinsic = False
        calibration.use_initial_extrinsic = True
        calibration.delta_body_cam.requires_grad_(False)
        calibration.log_scale_delta.requires_grad_(False)
        cfg['optim']['stages'] = [dict(stage, extrinsic_lr=0.0)
                                 for stage in cfg['optim']['stages'] if stage['name'] != 'extrinsic']
    if not calibration.optimize_extrinsic and not args.fixed_calibration:
        with torch.no_grad():
            calibration.init_body_cam_tensor.copy_(configured_extrinsic)
            calibration.delta_body_cam.zero_()
    output_dir = args.output_dir or Path(cfg.get("train", {}).get("output_dir", "runs/visual_finetune"))
    if args.init_visual is not None and output_dir.resolve() == args.init_visual.parent.resolve():
        raise ValueError("--output-dir must differ from the --init-visual checkpoint directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.init_visual is not None:
        initialization = {
            "mode": "init_visual_fresh_history",
            "checkpoint": str(args.init_visual.resolve()),
            "source_history_epochs": len(checkpoint.get("history", [])),
            "initial_scale": float(calibration.scale().detach().cpu()),
            "initial_calibration": calibration.summary(),
        }
        (output_dir / "initialization.json").write_text(
            json.dumps(initialization, indent=2), encoding="utf-8"
        )

    runtime_cfg = merged_config(model.cfg, cfg)
    if args.fixed_calibration:
        runtime_cfg['pressure_normalization'] = copy.deepcopy(
            checkpoint['cfg'].get('pressure_normalization', {}))
    model.cfg = runtime_cfg
    train_root = args.data_root / "train"
    val_root = args.data_root / "val"
    missing_splits = [str(path) for path in (train_root, val_root) if not path.is_dir()]
    if missing_splits:
        raise FileNotFoundError(
            f"Expected train/ and val/ directories under {args.data_root}; "
            f"missing: {', '.join(missing_splits)}"
        )

    if args.evaluate_only:
        if "visual_calibration" not in checkpoint:
            raise ValueError("Visual evaluation requires a calibrated checkpoint")
        calibration.load_state_dict(checkpoint["visual_calibration"], strict=True)
        # Use the checkpoint's fitted preprocessing; validation must not refit it.
        eval_cfg = merged_config(runtime_cfg, {"pressure_normalization": checkpoint["cfg"].get("pressure_normalization", {})})
        dataset = VisualPairDataset(val_root, eval_cfg, normalizer, model.input_dim, non_overlapping=True)
        loader = DataLoader(dataset, batch_size=int(cfg.get("eval", {}).get("batch_size", 16)),
                            num_workers=0, collate_fn=collate_visual_batch)
        metrics = evaluate_visual_loader(model, calibration, loader, normalizer, eval_cfg,
                                         {}, device)
        if metrics.get("num_matches", 0) <= 0:
            raise ValueError("Validation has no usable visual matches")
        if dataset.window_pairs is not None and metrics.get("cross_valid_edges", 0) <= 0:
            raise ValueError("Validation windows have no usable cross-frame matches")
        metrics["visual_loss"] = (float(cfg["loss"].get("reprojection_weight", 1.0)) * metrics["reprojection"]
                                  + float(cfg["loss"].get("epipolar_weight", 0.2)) * metrics["epipolar"])
        if cfg.get("early_stopping", {}).get("enabled", False):
            validator = FixedBATrajectoryValidation(
                model, normalizer, eval_cfg, device, [s.sequence_id for s in dataset.sequences])
            metrics.update(validator.evaluate(model, calibration))
        (output_dir / "visual_metrics.json").write_text(json.dumps(metrics, indent=2))
        return

    if pressure_enabled(runtime_cfg) and not args.fixed_calibration:
        def raw_pressure_sequence(seq):
            x, _, t = load_real_features(seq, runtime_cfg)
            return x, t
        fit_pressure(discover_real_sequences(train_root), runtime_cfg, raw_pressure_sequence, normalizer)
    dataset = VisualPairDataset(train_root, runtime_cfg, normalizer, model.input_dim,
                                max_windows=runtime_cfg.get("visual", {}).get("max_windows"))
    eval_dataset = VisualPairDataset(val_root, runtime_cfg, normalizer, model.input_dim, non_overlapping=True)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg.get("eval", {}).get("batch_size", cfg.get("train", {}).get("batch_size", 4))),
        shuffle=False,
        num_workers=int(cfg.get("eval", {}).get("num_workers", 0)),
        collate_fn=collate_visual_batch,
        drop_last=False,
        persistent_workers=int(cfg.get("eval", {}).get("num_workers", 0)) > 0,
    )
    print(
        f"[setup] train_root={train_root} val_root={val_root} "
        f"train_sequences={len(dataset.sequences)} val_sequences={len(eval_dataset.sequences)} "
        f"train_pairs={len(dataset)} eval_pairs={len(eval_dataset)} device={device} "
        f"output_dir={output_dir} "
        f"initial_scale={float(calibration.scale().detach().cpu()):.6f}",
        flush=True,
    )
    print(f"[train_pair_groups] {pair_group_summary(dataset)}", flush=True)
    print(f"[eval_pair_groups] {pair_group_summary(eval_dataset)}", flush=True)

    if not history and args.init_visual is None and not calibration.use_initial_extrinsic:
        motion_cache = build_frozen_motion_cache(
            model, dataset, normalizer, runtime_cfg, device,
            output_dir / "motion_cache" / "train")
        indices = geometric_initialization_candidate_indices(dataset, motion_cache, runtime_cfg)
        prepare_frozen_visual_pair_cache(dataset, indices,
                                        output_dir / "visual_pair_cache" / "initialization.pt")
        if not estimate_geometric_initial_extrinsic(dataset, motion_cache, calibration, runtime_cfg, output_dir):
            raise RuntimeError("Extrinsic initialization failed; supply calibration or sequences with richer motion")
    for stage_cfg in cfg.get("optim", {}).get("stages", []):
        stage_name = str(stage_cfg.get("name", "stage"))
        if stage_name == "extrinsic" and calibration.use_initial_extrinsic:
            continue
        # Trainable-network stages require sensor windows and current raw
        # predictions, so disable the frozen-stage fast path by default.
        dataset.use_frozen_pair_cache = False
        eval_dataset.use_frozen_pair_cache = False
        completed_epoch = max(
            (int(row.get("epoch", 0)) for row in history if row.get("stage") == stage_name),
            default=0,
        )
        stage_epochs = int(stage_cfg.get("epochs", 1))
        stage_early = (cfg.get("early_stopping", {}).get("enabled", False)
                       and (stage_name == "scale" or stage_cfg.get("train_network", False) not in (False, "false", "none")))
        # Resume must still restore best and republish a possibly interrupted summary.
        if completed_epoch >= stage_epochs and not stage_early:
            print(
                f"[resume] skip completed stage={stage_name} epochs={completed_epoch}/{stage_epochs}",
                flush=True,
            )
            continue
        loader, stage_batches = make_stage_train_loader(
            dataset, runtime_cfg, stage_cfg, seed
        )
        # fused head rounds: implicit hypergradient through the fusion BA (alternating.implicit_gradient), heads with or
        # without LoRA adapters (lora_heads: sampled legs, common/implicit_gradient.py)
        implicit_term = None
        ig_cfg = runtime_cfg.get("implicit_gradient", {})
        if ig_cfg.get("enabled", False) and stage_cfg.get("train_network") in ("head", "heads", "lora_heads", "lora_velocity"):
            from common.implicit_gradient import ImplicitTerm
            ig_params = set_network_train_mode(model, stage_cfg.get("train_network"), stage_cfg.get("lora"))
            implicit_term = ImplicitTerm(model, normalizer, calibration, runtime_cfg, dataset, device, ig_params,
                                         train_root, ig_cfg)
        print(
            f"[stage_sampler] stage={stage_name} "
            f"batches_per_epoch={stage_batches}",
            flush=True,
        )
        train_motion_cache = None
        eval_motion_cache = None
        stage_eval_loader = eval_loader
        if (
            stage_cfg.get("train_network", False) is False
            and not bool(stage_cfg.get("use_velocity_smooth", False))
        ):
            cache_root = output_dir / "motion_cache"
            train_motion_cache = build_frozen_motion_cache(
                model, dataset, normalizer, runtime_cfg, device, cache_root / "train"
            )
            eval_motion_cache = build_frozen_motion_cache(
                model, eval_dataset, normalizer, runtime_cfg, device, cache_root / "val"
            )
            if dataset.window_pairs is None:
                train_indices = list(range(len(dataset.pairs)))
                prepare_frozen_visual_pair_cache(
                    dataset,
                    train_indices,
                    output_dir / "visual_pair_cache" / f"train_{stage_name}.pt",
                )
                prepare_frozen_visual_pair_cache(
                    eval_dataset,
                    list(range(len(eval_dataset.pairs))),
                    output_dir / "visual_pair_cache" / f"val_{stage_name}.pt",
                )
            # A cached item contains several small tensor views. Sending those
            # through many worker processes is substantially slower and uses
            # much more memory than direct in-process indexing.
            stage_eval_loader = DataLoader(
                eval_dataset,
                batch_size=int(cfg.get("eval", {}).get("batch_size", cfg.get("train", {}).get("batch_size", 4))),
                shuffle=False,
                num_workers=int(cfg.get("motion_cache", {}).get("num_workers", 0)),
                collate_fn=collate_visual_batch,
                drop_last=False,
            )
        stage_history = run_stage(
            model,
            calibration,
            loader,
            stage_eval_loader,
            normalizer,
            runtime_cfg,
            stage_cfg,
            device,
            output_dir,
            trajectory_dataset=eval_dataset,
            start_epoch=completed_epoch + 1,
            previous_history=history,
            train_motion_cache=train_motion_cache,
            eval_motion_cache=eval_motion_cache,
            implicit_term=implicit_term,
        )
        history.extend(stage_history)
        safe_stage_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stage_name)
        stage_checkpoint = output_dir / f"visual_{safe_stage_name}.pt"
        save_visual_checkpoint(stage_checkpoint, model, calibration, normalizer, history)
        print(
            f"[stage_done] stage={stage_name} checkpoint={stage_checkpoint} "
            f"next_stage_starts_from_completed_stage=true",
            flush=True,
        )

    save_visual_checkpoint(output_dir / "visual_finetuned.pt", model, calibration, normalizer, history)
    write_history(output_dir / "history.json", history)
    print(f"[done] checkpoint={output_dir / 'visual_finetuned.pt'}", flush=True)
    print(f"[done] calibration={output_dir / 'visual_calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
