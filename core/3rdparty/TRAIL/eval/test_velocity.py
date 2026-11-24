#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_config, resolve_device
from common.data import ANGULAR_TARGETS, LINEAR_TARGETS, Normalizer, VelocityWindowDataset, collate_velocity_batch, list_sequences
from common.model import PoseEstimator


def velocity_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # 速度测试按各轴输出 RMSE，便于定位 vx/vy/vz 或角速度问题。
    err = pred.detach() - target.detach()
    rmse = torch.sqrt(torch.mean(err * err, dim=0))
    names = (LINEAR_TARGETS + ANGULAR_TARGETS)[: pred.shape[-1]]
    out = {f"rmse_{name}": float(rmse[i].cpu()) for i, name in enumerate(names)}
    out["rmse_mean"] = float(torch.mean(rmse).cpu())
    out["linear_rmse_mean"] = float(torch.mean(rmse[:3]).cpu())
    if pred.shape[-1] >= 6:
        out["angular_rmse_mean"] = float(torch.mean(rmse[3:6]).cpu())
    return out


def evaluate_loader(model: PoseEstimator, loader: DataLoader, device: torch.device) -> dict[str, float]:
    # 对一个 DataLoader 中的所有窗口求平均指标。
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            gyro = batch["gyro"].to(device).float()
            pred_linear, refined_gyro, log_var, _ = model.velocity_branch.forward_with_gyro(x, gyro)
            pred = model.compose_full_velocity(pred_linear, refined_gyro)
            row = velocity_metrics(pred, y)
            row["nll"] = float(torch.mean(0.5 * ((y - pred) ** 2 / torch.exp(log_var).clamp_min(1e-8) + log_var)).cpu())
            rows.append(row)
    if not rows:
        raise ValueError("No evaluation windows found.")
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S1 velocity estimation on a split of a simulation set, default test.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg["train"].get("device", "auto"))
    model, normalizer, payload = PoseEstimator.load_model(args.model, map_location=device)
    # pressure statistics fitted on the training data live in the checkpoint
    cfg["pressure_normalization"] = payload["cfg"].get("pressure_normalization", cfg.get("pressure_normalization", {}))
    # and so does the IMU -> base rotation (common/frames.py)
    if payload["cfg"].get("imu_extrinsic"):
        cfg["imu_extrinsic"] = payload["cfg"]["imu_extrinsic"]
    if normalizer is None:
        normalizer = Normalizer.load(Path(args.model).parent / "normalizer.npz")
    model.to(device).eval()
    seqs = list_sequences(args.data_root, args.split, cfg["data"].get(f"max_{args.split}_sequences"))
    loader = DataLoader(
        VelocityWindowDataset(seqs, cfg, normalizer, training=False),
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 0)),
        collate_fn=collate_velocity_batch,
    )
    summary = evaluate_loader(model, loader, device)
    report = {"split": args.split, "summary": summary}
    if cfg["eval"].get("per_sequence_metrics", True):
        # 单独评估每条 sequence，并保存最差序列列表用于排查数据场景。
        sequence_metrics = []
        skipped_sequences = []
        for seq in seqs:
            try:
                seq_dataset = VelocityWindowDataset([seq], cfg, normalizer, training=False)
            except ValueError as exc:
                skipped_sequences.append({"sequence_id": seq.sequence_id, "reason": str(exc)})
                continue
            seq_loader = DataLoader(
                seq_dataset,
                batch_size=int(cfg["eval"]["batch_size"]),
                shuffle=False,
                num_workers=int(cfg["eval"].get("num_workers", 0)),
                collate_fn=collate_velocity_batch,
            )
            row = evaluate_loader(model, seq_loader, device)
            row["sequence_id"] = seq.sequence_id
            sequence_metrics.append(row)
        sequence_metrics = sorted(sequence_metrics, key=lambda item: float(item["rmse_mean"]), reverse=True)
        top_k = int(cfg["eval"].get("worst_sequence_top_k", 20))
        report["worst_sequences"] = sequence_metrics[:top_k]
        report["all_sequences"] = sequence_metrics
        report["skipped_sequences"] = skipped_sequences
    console_report = {
        "split": report["split"],
        "summary": report["summary"],
    }
    if "worst_sequences" in report:
        console_report["worst_sequences"] = report["worst_sequences"]
    print(json.dumps(console_report, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
