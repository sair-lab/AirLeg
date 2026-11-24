#!/usr/bin/env python3
"""Leg-network inference and keyframe-pair losses for AirSLAM2.

The ``infer`` command emits a timestamped body-twist CSV consumed by g2o (S2 rounds, S3 offline replays). The
keyframe-pair losses below are shared with S3 online learning (common/online.py).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# The development tree already vendors PyPose at AirSLAM2/lib/pypose. Make the
# stage runnable from any Torch environment.
for ancestor in ROOT.parents:
    candidate = ancestor / "lib" / "pypose"
    if (candidate / "pypose" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from common.pressure import normalize_features, online_start, enabled as pressure_enabled
import pypose as pp
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from common.config import resolve_device
from common.model import load_velocity_model
from common import gyro_bias as bias_file
from eval.real_utils import RealSequence, load_real_features


def checkpoint_velocity_scale(payload: dict[str, Any]) -> float:
    """Return the visual-finetune metric scale stored in a checkpoint."""
    calibration = payload.get("visual_calibration")
    if isinstance(calibration, dict) and "log_scale_delta" in calibration:
        log_scale = float(torch.as_tensor(calibration["log_scale_delta"]).detach().cpu())
        if "log_scale_min" in calibration:
            log_scale = max(
                log_scale,
                float(torch.as_tensor(calibration["log_scale_min"]).detach().cpu()),
            )
        if "log_scale_max" in calibration:
            log_scale = min(
                log_scale,
                float(torch.as_tensor(calibration["log_scale_max"]).detach().cpu()),
            )
        scale = math.exp(log_scale)
    else:
        summary = payload.get("visual_calibration_summary", {})
        scale = float(summary.get("scale", 1.0)) if isinstance(summary, dict) else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Checkpoint contains an invalid velocity scale: {scale}")
    return scale


def apply_linear_scale(values: torch.Tensor, scale: float) -> torch.Tensor:
    if values.shape[-1] < 3:
        raise ValueError(f"Velocity tensor must have at least 3 channels, got {values.shape[-1]}")
    if scale == 1.0:
        return values
    return torch.cat([values[..., :3] * scale, values[..., 3:]], dim=-1)


def checkpoint_camera_from_body(payload: dict[str, Any]) -> list[float]:
    summary = payload.get("visual_calibration_summary", {})
    values = summary.get("T_body_cam_xyz_quat_xyzw") if isinstance(summary, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError("Checkpoint has no estimated T_body_cam extrinsic")
    return pp.SE3(torch.tensor(values, dtype=torch.float64)).Inv().tensor().tolist()


def load_stage_config(path: Path | None) -> dict[str, Any]:
    if path is None:   # inference settings (device, stationary, ...) all at their defaults
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    from common.yaml_base import load_yaml
    return load_yaml(path)


def raw_pressure_features(data_root, x, times, cfg):
    """Recover metric-like pressure for legacy recordings, without editing CSVs."""
    manifest = data_root / 'manifest.json'
    metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not pressure_enabled(cfg) or metadata.get('contact_estimation_method') != 'torque_norm':
        return x
    from common.pressure import channels
    from common.robot import robot_model
    from data_scripts.estimate_contacts.core import estimate_contacts_from_joint_csv
    model = robot_model(cfg.get('robot'))
    joints = data_root / 'robot/joints.csv'
    digest = hashlib.sha256()
    for path in (joints, model, ROOT / 'data_scripts/estimate_contacts/core.py'):
        digest.update(path.read_bytes())
    cache = data_root / 'tmp/raw_pressure' / (digest.hexdigest() + '.npz')
    if not cache.exists():
        estimate = estimate_contacts_from_joint_csv(joints, method='jacobian', normalize_sum=False, model_path=model)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, time=estimate.time, pressure=estimate.pressure)
        temporary.replace(cache)
    with np.load(cache) as data:
        pressure = np.column_stack([np.interp(times, data['time'], data['pressure'][:, i])
                                    for i in range(4)])
    result = x.copy()
    result[:, channels()] = pressure
    print(f'[stage4] legacy torque_norm replaced by cached raw Jacobian pressure: {cache}', flush=True)
    return result


def load_inputs(data_root: Path, checkpoint: Path, device: torch.device, bundle=None, gyro_bias=None):
    model, normalizer, payload = bundle or load_velocity_model(checkpoint, device)
    velocity_scale = checkpoint_velocity_scale(payload)
    if normalizer is None:
        raise ValueError(f"Checkpoint has no input normalizer: {checkpoint}")
    sequence = RealSequence(data_root, data_root.name)
    x, sample, times = load_real_features(sequence, model.cfg, gyro_bias)
    if x.shape[1] != model.input_dim:
        raise ValueError(f"Input dimension {x.shape[1]} does not match checkpoint {model.input_dim}")
    x = raw_pressure_features(data_root, x, times, model.cfg)
    x_normalized = torch.from_numpy(normalizer.apply(
        normalize_features(x, times, model.cfg, online=True)
    )).to(device=device, dtype=torch.float32)
    x_raw = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    model.to(device)
    return (model, payload, x_normalized, x_raw, sample,
            np.asarray(times, dtype=np.float64), velocity_scale)


def sliding_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.shape[0] < window:
        raise ValueError(f"Need at least {window} sensor samples, got {x.shape[0]}")
    return x.unfold(0, window, 1).transpose(1, 2)


def run_infer(args: argparse.Namespace) -> int:
    if args.data_root is None:
        raise ValueError("infer requires --data-root")
    cfg = load_stage_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    model, payload, x_norm, x_raw, sample, times, velocity_scale = load_inputs(
        args.data_root, args.checkpoint, device, gyro_bias=args.gyro_bias
    )
    if args.extrinsic_output is not None:
        values = checkpoint_camera_from_body(payload)
        args.extrinsic_output.parent.mkdir(parents=True, exist_ok=True)
        args.extrinsic_output.write_text(
            " ".join(f"{value:.17g}" for value in values) + "\n", encoding="utf-8"
        )
    model.eval()
    window = int(model.cfg["data"]["window"])
    windows = sliding_windows(x_norm, window)
    start = online_start(x_raw.cpu().numpy(), times, model.cfg)
    windows = windows[start - window + 1:]
    gyro = x_raw[start:, :3]
    batch_size = int(cfg.get("inference_batch_size", 4096))
    velocity_parts = []
    variance_parts = []
    correction_parts = []
    with torch.no_grad():
        for begin in range(0, len(windows), batch_size):
            end = min(len(windows), begin + batch_size)
            linear, refined_gyro, log_var, correction = model.velocity_branch.forward_with_gyro(
                windows[begin:end], gyro[begin:end]
            )
            linear = apply_linear_scale(linear, velocity_scale)
            velocity = torch.cat([linear[..., :3], refined_gyro], dim=-1)
            velocity_parts.append(velocity.cpu())
            variance_parts.append(
                apply_linear_scale(log_var.exp(), velocity_scale * velocity_scale).cpu()
            )
            correction_parts.append(correction.cpu())
    velocity = torch.cat(velocity_parts).numpy()
    variance = torch.cat(variance_parts).numpy()
    correction = torch.cat(correction_parts).numpy()
    from common.stationary import sequence_mask
    stationary = sequence_mask(args.data_root, times[start:], cfg)   # zero twist while standing (cfg stationary:)
    if stationary is not None:
        velocity[stationary] = 0.0
    bias = (bias_file.at(args.gyro_bias, times[start:]) if args.gyro_bias
            else np.zeros((len(velocity), 3)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample", "time", "vx", "vy", "vz", "wx", "wy", "wz",
            "var_vx", "var_vy", "var_vz", "var_wx", "var_wy", "var_wz",
            "gyro_correction_x", "gyro_correction_y", "gyro_correction_z",
            "bias_gx", "bias_gy", "bias_gz",
        ])
        for index in range(len(velocity)):
            writer.writerow([
                int(sample[start + index]), f"{times[start + index]:.9f}",
                *[f"{value:.9g}" for value in velocity[index]],
                *[f"{value:.9g}" for value in variance[index]],
                *[f"{value:.9g}" for value in correction[index]],
                *[f"{value:.9g}" for value in bias[index]],
            ])
    temporary.replace(args.output)
    print(json.dumps({
        "command": "infer", "samples": len(velocity), "output": str(args.output),
        "start_time": float(times[start]), "end_time": float(times[-1]),
        "device": str(device), "velocity_scale": velocity_scale,
        "stationary_samples": None if stationary is None else int(stationary.sum()),
    }, indent=2))
    return 0


def build_pairs(camera_times: np.ndarray, sensor_times: np.ndarray, window: int,
                delay: float, max_pairs: int, seed: int):
    pairs: list[tuple[int, int, int, int]] = []
    for first in range(len(camera_times) - 1):
        second = first + 1
        t0 = float(camera_times[first] + delay)
        t1 = float(camera_times[second] + delay)
        i0 = int(np.searchsorted(sensor_times, t0, side="right") - 1)
        i1 = int(np.searchsorted(sensor_times, t1, side="left"))
        if i0 >= window - 1 and i1 > i0 and i1 < len(sensor_times):
            pairs.append((first, second, i0, i1))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs] if max_pairs > 0 else pairs


def robust_square(value: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(absolute <= delta, 0.5 * value.square(),
                       delta * (absolute - 0.5 * delta)).mean()


def batch_pair_loss(model, windows, x_raw, sensor_times, pairs,
                    shifted_camera_times, camera_poses, extrinsic, cfg,
                    velocity_scale):
    """Evaluate a true pair batch with deduplicated network windows."""
    device = windows.device
    window = int(model.cfg["data"]["window"])
    sensor_indices = [torch.arange(p[2], p[3], device=device) for p in pairs]
    lengths = torch.tensor([len(v) for v in sensor_indices], device=device)
    unique_indices, inverse = torch.unique(
        torch.cat(sensor_indices), sorted=True, return_inverse=True
    )
    unique_windows = windows[unique_indices - (window - 1)]
    unique_raw = x_raw[unique_indices]
    linear, refined_gyro, _, correction = model.velocity_branch.forward_with_gyro(
        unique_windows, unique_raw[:, :3]
    )
    linear = apply_linear_scale(linear, velocity_scale)
    unique_velocity = torch.cat([linear[..., :3], refined_gyro], dim=-1)

    inverse_chunks = torch.split(inverse, lengths.tolist())
    velocity_chunks = [unique_velocity[index] for index in inverse_chunks]
    correction_chunks = [correction[index] for index in inverse_chunks]
    acceleration_chunks = [unique_raw[index, 3:6] for index in inverse_chunks]
    dt_chunks = []
    for camera_i, camera_j, sensor_i, sensor_j in pairs:
        starts = torch.as_tensor(sensor_times[sensor_i:sensor_j], device=device)
        ends = torch.as_tensor(sensor_times[sensor_i + 1:sensor_j + 1], device=device)
        t0 = starts.new_tensor(float(shifted_camera_times[camera_i]))
        t1 = starts.new_tensor(float(shifted_camera_times[camera_j]))
        dt_chunks.append(
            (torch.minimum(ends, t1) - torch.maximum(starts, t0)).clamp_min(0.0).float()
        )

    velocity = pad_sequence(velocity_chunks, batch_first=True)
    correction = pad_sequence(correction_chunks, batch_first=True)
    acceleration = pad_sequence(acceleration_chunks, batch_first=True)
    dt = pad_sequence(dt_chunks, batch_first=True)
    steps = torch.arange(velocity.shape[1], device=device).unsqueeze(0)
    valid = steps < lengths.unsqueeze(1)

    first_indices = torch.tensor([p[0] for p in pairs], device=device)
    second_indices = torch.tensor([p[1] for p in pairs], device=device)
    body_pose = camera_poses[first_indices] * extrinsic
    gravity = velocity.new_tensor([0.0, 0.0, -9.80665]).expand(len(pairs), -1)
    gravity_body0 = body_pose.rotation().Inv().Act(gravity)

    increments = pp.so3(velocity[:, :, 3:6] * dt.unsqueeze(-1)).Exp()
    cumulative = pp.cumprod(increments, dim=1, left=False)
    identity = pp.identity_SO3(
        len(pairs), dtype=velocity.dtype, device=device
    ).tensor().unsqueeze(1)
    rotations = pp.SO3(torch.cat([identity, cumulative.tensor()[:, :-1]], dim=1))
    rotated_linear = rotations.Act(velocity[:, :, :3])
    position = (rotated_linear * dt.unsqueeze(-1)).sum(1)
    imu_delta_v = (
        (rotations.Act(acceleration) + gravity_body0.unsqueeze(1)) * dt.unsqueeze(-1)
    ).sum(1)
    batch_index = torch.arange(len(pairs), device=device)
    final_rotation = cumulative[batch_index, lengths - 1]
    prediction = pp.SE3(torch.cat([position, final_rotation.tensor()], dim=-1))

    camera_delta = camera_poses[first_indices].Inv() * camera_poses[second_indices]
    body_target = extrinsic.Inv() * camera_delta * extrinsic
    pose_error = (body_target.Inv() * prediction).Log().tensor()
    pose_translation = robust_square(pose_error[:, :3], float(cfg.get("pose_huber", 0.2)))
    pose_rotation = robust_square(pose_error[:, 3:], float(cfg.get("rotation_huber", 0.2)))

    adjacent = valid[:, 1:] & valid[:, :-1]
    smooth_values = F.smooth_l1_loss(
        (velocity[:, 1:, :3] - velocity[:, :-1, :3]) /
        dt[:, :-1].clamp_min(1e-4).unsqueeze(-1),
        torch.zeros_like(velocity[:, :-1, :3]), beta=1.0, reduction="none"
    ).mean(-1)
    smooth = ((smooth_values * adjacent).sum(1) /
              adjacent.sum(1).clamp_min(1)).mean()

    # the accelerometer measures the IMU point: compare with the network velocity moved to it, v_I = v_B + w_B x r_BI
    # (base axes; x_raw IMU columns are already rotated into base axes, common/frames.py)
    from common.frames import config_imu_extrinsic, imu_point_velocity
    r_BI = config_imu_extrinsic(model.cfg).r_BI
    last = velocity[batch_index, lengths - 1]
    last_linear = imu_point_velocity(last[:, :3], last[:, 3:6], r_BI)
    first_linear = imu_point_velocity(velocity[:, 0, :3], velocity[:, 0, 3:6], r_BI)
    network_delta_v = final_rotation.Act(last_linear) - first_linear
    relative_velocity = F.smooth_l1_loss(network_delta_v, imu_delta_v, beta=0.5)
    deadzone = float(cfg.get("gyro_correction_deadzone", 0.02))
    gyro_values = F.relu(correction.abs() - deadzone).square().mean(-1)
    gyro_prior = ((gyro_values * valid).sum(1) / valid.sum(1)).mean()
    return {
        "pose_translation": pose_translation,
        "pose_rotation": pose_rotation,
        "velocity_smooth": smooth,
        "relative_velocity": relative_velocity,
        "gyro_correction": gyro_prior,
    }


def weighted_total(losses: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    weights = cfg.get("loss", {})
    return (
        float(weights.get("pose_translation", 1.0)) * losses["pose_translation"]
        + float(weights.get("pose_rotation", 1.0)) * losses["pose_rotation"]
        + float(weights.get("velocity_smooth", 1e-4)) * losses["velocity_smooth"]
        + float(weights.get("relative_velocity", 1e-3)) * losses["relative_velocity"]
        + float(weights.get("gyro_correction", 0.05)) * losses["gyro_correction"]
    )


def pair_batches(pairs, batch_size):
    ordered = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def evaluate(model, pairs, windows, x_raw, sensor_times, shifted_times,
             poses, extrinsic, cfg, velocity_scale) -> float:
    if not pairs:
        return math.inf
    values = []
    model.eval()
    with torch.no_grad():
        for batch in pair_batches(pairs, max(1, int(cfg.get("pair_batch_size", 16)))):
            losses = batch_pair_loss(
                model, windows, x_raw, sensor_times, batch, shifted_times,
                poses, extrinsic, cfg, velocity_scale
            )
            values.append(float(weighted_total(losses, cfg).cpu()))
    return float(np.mean(values))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path)
    common.add_argument("--checkpoint", required=True, type=Path)
    common.add_argument("--config", type=Path)
    common.add_argument("--device")

    infer = subparsers.add_parser("infer", parents=[common])
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--extrinsic-output", type=Path)
    infer.add_argument("--gyro-bias", type=Path, help="time,bgx,bgy,bgz CSV subtracted from raw gyro")
    infer.set_defaults(function=run_infer)

    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
