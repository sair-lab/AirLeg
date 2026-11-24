#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_config, resolve_device
from common.data import (
    ANGULAR_TARGETS,
    LINEAR_TARGETS,
    Normalizer,
    VelocityWindowDataset,
    collate_velocity_batch,
    fit_normalizer,
    infer_input_dim,
    split_sequences,
)
from common.model import PoseEstimator


def make_velocity_loader(seqs, cfg, normalizer, training: bool, num_workers: int | None = None) -> DataLoader:
    # 速度训练使用滑动窗口数据集，窗口最后一帧作为监督目标。
    ds = VelocityWindowDataset(seqs, cfg, normalizer, training)
    batch_size = int(cfg["train"]["batch_size"] if training else cfg["eval"]["batch_size"])
    if num_workers is None:
        num_workers = int(cfg["train"].get("num_workers", 0) if training else cfg["eval"].get("num_workers", 0))
    shuffle = bool(training and cfg["data"].get("shuffle_train_windows", True))
    # Workers stay alive across epochs (spawning them is expensive); pinned batches copy asynchronously.
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_velocity_batch,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def velocity_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # 按速度各分量统计 RMSE，TensorBoard 和 json 报告共用这套指标。
    err = pred.detach() - target.detach()
    rmse = torch.sqrt(torch.mean(err * err, dim=0))
    names = (LINEAR_TARGETS + ANGULAR_TARGETS)[: pred.shape[-1]]
    out = {f"rmse_{name}": float(rmse[i].cpu()) for i, name in enumerate(names)}
    out["rmse_mean"] = float(torch.mean(rmse).cpu())
    out["linear_rmse_mean"] = float(torch.mean(rmse[:3]).cpu())
    if pred.shape[-1] >= 6:
        out["angular_rmse_mean"] = float(torch.mean(rmse[3:6]).cpu())
    return out


def run_velocity_epoch(
    model: PoseEstimator,
    loader,
    optimizer,
    cfg,
    device,
    train: bool,
    uncertainty: bool,
    writer=None,
    stage: str = "velocity",
    epoch: int = 0,
) -> dict[str, float]:
    model.velocity_branch.train(train)
    totals: dict[str, float] = {"loss": 0.0, "nll": 0.0}
    count = 0
    epoch_start = time.time()
    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        with torch.set_grad_enabled(train):
            gyro = batch["gyro"].to(device).float()
            pred_linear, refined_gyro, log_var, _ = model.velocity_branch.forward_with_gyro(x, gyro)
            pred_full = model.compose_full_velocity(pred_linear, refined_gyro)
            if uncertainty:
                # 第二阶段冻结均值预测，只用 NLL 训练不确定度 head。
                var = torch.exp(log_var).clamp_min(1e-8)
                loss = torch.mean(0.5 * ((y - pred_full.detach()) ** 2 / var + log_var))
            else:
                # 第一阶段只训练线速度 backbone/head，不预测也不监督角速度。
                loss = F.smooth_l1_loss(pred_linear, y[:, : model.velocity_dim])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params = [p for p in model.velocity_branch.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params, float(cfg["train"].get("grad_clip", 1.0)))
                optimizer.step()
        metrics = velocity_metrics(pred_full, y)
        totals["loss"] += float(loss.detach().cpu())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        nll = torch.mean(0.5 * ((y - pred_full.detach()) ** 2 / torch.exp(log_var).clamp_min(1e-8) + log_var))
        totals["nll"] += float(nll.detach().cpu())
        count += 1
        print_interval = int(cfg["train"].get("print_interval_steps", cfg["train"].get("log_interval", 20)))
        if writer is not None and train and (step == 1 or step % print_interval == 0):
            global_step = max(0, epoch - 1) * len(loader) + step
            writer.add_scalar(f"{stage}/batch_loss", float(loss.detach().cpu()), global_step)
        if step == 1 or step % print_interval == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            print(
                f"[{'train' if train else 'val'}] step={step}/{len(loader)} "
                f"loss={float(loss.detach().cpu()):.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return {k: v / max(1, count) for k, v in totals.items()}


def write_velocity_tensorboard(writer, stage: str, epoch: int, train_metrics: dict, val_metrics: dict | None) -> None:
    train_keys = ("loss", "nll", "rmse_mean", "linear_rmse_mean", "angular_rmse_mean")
    for name in train_keys:
        if name in train_metrics:
            writer.add_scalar(f"{stage}/train_{name}", train_metrics[name], epoch)

    if val_metrics is None:
        return
    for name, value in val_metrics.items():
        if name in ("loss", "nll") or name.startswith("rmse_") or name.endswith("_rmse_mean"):
            writer.add_scalar(f"{stage}/val_{name}", value, epoch)


def evaluate_velocity_loader(model: PoseEstimator, loader, device) -> dict[str, float]:
    model.velocity_branch.eval()
    totals: dict[str, float] = {"nll": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            gyro = batch["gyro"].to(device).float()
            pred_linear, refined_gyro, log_var, _ = model.velocity_branch.forward_with_gyro(x, gyro)
            pred_full = model.compose_full_velocity(pred_linear, refined_gyro)
            for name, value in velocity_metrics(pred_full, y).items():
                totals[name] = totals.get(name, 0.0) + value
            nll = torch.mean(0.5 * ((y - pred_full) ** 2 / torch.exp(log_var).clamp_min(1e-8) + log_var))
            totals["nll"] += float(nll.cpu())
            count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def write_val_sequence_report(
    model: PoseEstimator,
    val_seqs,
    cfg,
    normalizer,
    device,
    stage: str,
    epoch: int,
) -> dict:
    # 额外按 sequence 输出 val 指标，方便定位最差序列。
    rows = []
    log_interval = max(1, int(cfg["train"].get("log_interval", cfg["train"].get("print_interval_steps", 20))))
    print(f"[eval_seq] stage={stage}, epoch={epoch}, sequences={len(val_seqs)}", flush=True)
    for i, seq in enumerate(val_seqs, start=1):
        try:
            # one small sequence: in-process loading (a worker pool per sequence costs more than it saves)
            loader = make_velocity_loader([seq], cfg, normalizer, False, num_workers=0)
        except ValueError as exc:
            rows.append({"sequence_id": seq.sequence_id, "skipped": True, "reason": str(exc)})
            if i == 1 or i == len(val_seqs) or i % log_interval == 0:
                print(f"[eval_seq] {i}/{len(val_seqs)} {seq.sequence_id} skipped: {exc}", flush=True)
            continue
        metrics = evaluate_velocity_loader(model, loader, device)
        metrics["sequence_id"] = seq.sequence_id
        rows.append(metrics)
        if i == 1 or i == len(val_seqs) or i % log_interval == 0:
            print(
                f"[eval_seq] {i}/{len(val_seqs)} {seq.sequence_id} "
                f"rmse_mean={metrics.get('rmse_mean', 0.0):.6f}",
                flush=True,
            )
    valid_rows = [row for row in rows if not row.get("skipped")]
    skipped_rows = [row for row in rows if row.get("skipped")]
    valid_rows = sorted(valid_rows, key=lambda item: float(item.get("rmse_mean", 0.0)), reverse=True)
    top_k = int(cfg["eval"].get("worst_sequence_top_k", 20))
    report = {
        "stage": stage,
        "epoch": epoch,
        "split": "val",
        "num_sequences": len(rows),
        "num_valid_sequences": len(valid_rows),
        "num_skipped_sequences": len(skipped_rows),
        "worst_sequences": valid_rows[:top_k],
        "all_sequences": valid_rows,
        "skipped_sequences": skipped_rows,
    }
    return report


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_velocity_stage(
    model,
    train_seqs,
    val_seqs,
    cfg,
    normalizer,
    output_dir,
    device,
    epochs: int,
    uncertainty: bool,
    history: list[dict],
    writer=None,
) -> None:
    # 一个 stage 对应一段完整训练：速度均值阶段或不确定度阶段。
    train_loader = make_velocity_loader(train_seqs, cfg, normalizer, True)
    val_loader = make_velocity_loader(val_seqs, cfg, normalizer, False)
    params = [p for p in model.velocity_branch.parameters() if p.requires_grad]
    lr_key = "uncertainty_lr" if uncertainty else "lr"
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["train"].get(lr_key, cfg["train"]["lr"])),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    best = math.inf
    stage = "uncertainty" if uncertainty else "velocity"
    val_interval = max(1, int(cfg["train"].get("val_interval_epochs", 1)))
    for epoch in range(1, epochs + 1):
        print(f"[{stage}] epoch {epoch}/{epochs}", flush=True)
        train_metrics = run_velocity_epoch(
            model, train_loader, optimizer, cfg, device, True, uncertainty, writer, stage, epoch
        )
        should_validate = epoch == 1 or epoch == epochs or epoch % val_interval == 0
        val_metrics = None
        if should_validate:
            print(f"[{stage}] validation epoch {epoch}/{epochs}", flush=True)
            val_metrics = run_velocity_epoch(
                model, val_loader, optimizer, cfg, device, False, uncertainty, writer, stage, epoch
            )
        row = {"stage": stage, "epoch": epoch, "train": train_metrics, "val": val_metrics}
        if (
            val_metrics is not None
            and cfg["eval"].get("per_sequence_metrics", True)
            and cfg["eval"].get("per_sequence_metrics_during_train", True)
        ):
            row["val_sequences"] = write_val_sequence_report(
                model, val_seqs, cfg, normalizer, device, stage, epoch
            )
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        tensorboard_interval = int(cfg["train"].get("tensorboard_log_interval", 1))
        if writer is not None and tensorboard_interval > 0 and (epoch == 1 or epoch % tensorboard_interval == 0):
            write_velocity_tensorboard(writer, stage, epoch, train_metrics, val_metrics)
        model.save_model(output_dir / "last.pt", normalizer, {"stage": stage, "epoch": epoch})
        if val_metrics is not None:
            metric = val_metrics["nll"] if uncertainty else val_metrics["rmse_mean"]
            if metric < best:
                best = metric
                model.save_model(output_dir / "best.pt", normalizer, {"stage": stage, "epoch": epoch, "metric": metric})
        # Keep one complete, crash-safe training summary updated after every
        # fully completed epoch, across both velocity and uncertainty stages.
        write_history(output_dir / "history.json", history)


def prepare_normalizer(cfg: dict, data_root: Path, output_dir: Path, train_seqs) -> Normalizer:
    # normalizer 只由 train split 估计，并随 checkpoint 保存。
    normalizer_path = output_dir / "normalizer.npz"
    force_refit = bool(cfg["data"].get("force_refit_normalizer", False))
    if normalizer_path.exists() and not force_refit:
        try:
            normalizer = Normalizer.load(normalizer_path)
            expected_dim = infer_input_dim(data_root, cfg)
            if int(normalizer.mean.shape[0]) == expected_dim:
                print(f"[normalizer] loaded cached {normalizer_path}", flush=True)
                return normalizer
            print(
                f"[normalizer] cached dim mismatch: cached={normalizer.mean.shape[0]}, expected={expected_dim}; refit",
                flush=True,
            )
        except Exception as exc:
            print(f"[normalizer] failed to load cached normalizer ({exc}); refit", flush=True)
    elif force_refit:
        print("[normalizer] force_refit_normalizer=true; refit", flush=True)
    normalizer = fit_normalizer(train_seqs, cfg)
    normalizer.save(normalizer_path)
    print(f"[normalizer] saved {normalizer_path}", flush=True)
    return normalizer


def load_resume_model(
    resume_model: Path,
    input_dim: int,
    cfg: dict,
    device: torch.device,
) -> tuple[PoseEstimator, Normalizer | None]:
    print(f"[resume] loading velocity checkpoint: {resume_model}", flush=True)
    ckpt_model, ckpt_normalizer, ckpt = PoseEstimator.load_model(resume_model, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {})
    probe = PoseEstimator(input_dim, cfg)
    if int(ckpt.get("input_dim", ckpt_model.input_dim)) != int(input_dim):
        raise ValueError(
            f"Resume checkpoint input_dim mismatch: checkpoint={ckpt.get('input_dim', ckpt_model.input_dim)}, "
            f"current={input_dim}. Check data features/cache/config."
        )
    if int(ckpt.get("output_dim", ckpt_model.output_dim)) != int(probe.output_dim):
        raise ValueError(
            f"Resume checkpoint output_dim mismatch: checkpoint={ckpt.get('output_dim', ckpt_model.output_dim)}, "
            f"current={probe.output_dim}."
        )
    model = probe.to(device)
    model.velocity_branch.load_state_dict(ckpt_model.velocity_branch.state_dict(), strict=True)
    print(
        f"[resume] loaded stage={ckpt.get('extra', {}).get('stage', 'unknown')}, "
        f"epoch={ckpt.get('extra', {}).get('epoch', 'unknown')}, "
        f"checkpoint_cfg_model_keys={list(ckpt_cfg.get('model', {}).keys())}",
        flush=True,
    )
    if ckpt_normalizer is None:
        print("[resume] checkpoint has no normalizer; will fit/load one from the output directory.", flush=True)
    return model, ckpt_normalizer


def train_velocity_branch_resume(
    cfg: dict,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    resume_model: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # IMU pose in the base frame of this training set; stored in cfg (hence in every checkpoint) and used for all
    # later inference with this model (common/frames.py)
    from common.data import dataset_imu_extrinsic
    cfg['imu_extrinsic'] = dataset_imu_extrinsic(data_root, cfg).to_dict()
    print(f"[frames] IMU extrinsic T_B_I = {cfg['imu_extrinsic']}", flush=True)
    train_seqs, val_seqs = split_sequences(data_root, cfg)
    from copy import deepcopy
    from common.pressure import enabled as pressure_enabled, fit_pressure
    from common.data import _load_sequence_arrays_from_csv, read_csv, pressure_times
    if pressure_enabled(cfg):
        raw_cfg = deepcopy(cfg)
        raw_cfg['pressure_normalization']['enabled'] = False
        def raw_pressure_sequence(seq):
            x, _ = _load_sequence_arrays_from_csv(seq, raw_cfg)
            return x, pressure_times(read_csv(seq.path / 'imu.csv'), len(x), cfg)
        fit_pressure(train_seqs, cfg, raw_pressure_sequence)
        # Cached mean/std may describe raw pressure even with the same dimension.
        cfg['data']['force_refit_normalizer'] = True
    print(
        f"[setup] data_root={data_root}, output_dir={output_dir}, device={device}, "
        f"train_sequences={len(train_seqs)}, val_sequences={len(val_seqs)}",
        flush=True,
    )
    print("[model] inferring input dimension", flush=True)
    input_dim = infer_input_dim(data_root, cfg)
    if resume_model is not None:
        model, normalizer = load_resume_model(resume_model, input_dim, cfg, device)
        if normalizer is None:
            normalizer = prepare_normalizer(cfg, data_root, output_dir, train_seqs)
        elif pressure_enabled(cfg):
            fit_pressure(train_seqs, cfg, raw_pressure_sequence, normalizer)
        normalizer.save(output_dir / "normalizer.npz")
        print(f"[normalizer] saved resumed normalizer {output_dir / 'normalizer.npz'}", flush=True)
    else:
        normalizer = prepare_normalizer(cfg, data_root, output_dir, train_seqs)
        model = PoseEstimator(input_dim, cfg).to(device)
    print(
        f"[model] input_dim={model.input_dim}, velocity_dim={model.velocity_dim}, "
        f"uncertainty_dim={model.uncertainty_dim}, output_dim={model.output_dim}, "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    velocity_epochs = int(cfg["train"].get("velocity_epochs", cfg["train"].get("epochs", 20)))
    history: list[dict] = []
    writer = build_summary_writer(cfg, output_dir)
    best_path = output_dir / "best.pt"
    try:
        if velocity_epochs > 0:
            for param in model.velocity_branch.parameters():
                param.requires_grad = True
            run_velocity_stage(
                model,
                train_seqs,
                val_seqs,
                cfg,
                normalizer,
                output_dir,
                device,
                velocity_epochs,
                uncertainty=False,
                history=history,
                writer=writer,
            )

            if best_path.exists():
                best_model, _, _ = PoseEstimator.load_model(best_path, map_location=device)
                model.velocity_branch.load_state_dict(best_model.velocity_branch.state_dict(), strict=False)
        else:
            print("[velocity] velocity_epochs=0; skip velocity mean stage.", flush=True)

        if (
            not cfg["train"].get("skip_uncertainty", False)
            and cfg.get("model", {}).get("predict_uncertainty", True)
            and model.velocity_branch.uncertainty_head is not None
        ):
            for name, param in model.velocity_branch.named_parameters():
                param.requires_grad = name.startswith("uncertainty_head.")
            uncertainty_epochs = int(cfg["train"].get("uncertainty_epochs", max(1, velocity_epochs // 3)))
            if uncertainty_epochs > 0:
                run_velocity_stage(
                    model,
                    train_seqs,
                    val_seqs,
                    cfg,
                    normalizer,
                    output_dir,
                    device,
                    uncertainty_epochs,
                    uncertainty=True,
                    history=history,
                    writer=writer,
                )
            else:
                print("[uncertainty] uncertainty_epochs=0; skip uncertainty stage.", flush=True)
        else:
            print("[uncertainty] skip uncertainty stage.", flush=True)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    return best_path


def build_summary_writer(cfg: dict, output_dir: Path):
    tb_cfg = cfg["train"].get("tensorboard", {})
    if not tb_cfg.get("enabled", True):
        return None
    if SummaryWriter is None:
        print("[tensorboard] torch.utils.tensorboard is unavailable; skip TensorBoard logging.", flush=True)
        return None
    log_dir = tb_cfg.get("log_dir") or str(output_dir / "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[tensorboard] log_dir={log_dir}", flush=True)
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the velocity branch in two stages (S1, simulation).")
    parser.add_argument("--config", required=True, help="e.g. AirSLAM2/robots/anymal/leg_odometry.yaml (simulation: section)")
    parser.add_argument("--data-root", default=None, help="default: data.data_root of the config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-model", default=None, help="Continue training from an existing velocity checkpoint.")
    parser.add_argument("--skip-uncertainty", action="store_true", help="Only train/resume the velocity mean stage.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        cfg["train"]["velocity_epochs"] = args.epochs
    if args.skip_uncertainty:
        cfg["train"]["skip_uncertainty"] = True
    device = resolve_device(args.device or cfg["train"].get("device", "auto"))
    print(f"[setup] config={args.config}", flush=True)
    # paths in the config are relative to the config file, command line paths to the working directory
    config_dir = Path(cfg["_meta"]["config_dir"])
    data_root = Path(args.data_root) if args.data_root else config_dir / cfg["data"]["data_root"]
    print(f"[setup] robot={cfg.get('robot', 'go2')}, data_root={data_root}", flush=True)
    output_dir = Path(args.output_dir) if args.output_dir else config_dir / cfg["train"]["output_dir"]
    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] device={device}", flush=True)
    t0 = time.time()
    best = train_velocity_branch_resume(
        cfg,
        data_root,
        output_dir,
        device,
        Path(args.resume_model) if args.resume_model else None,
    )
    print(f"[done] velocity checkpoint: {best}", flush=True)
    print(f"[elapsed] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
