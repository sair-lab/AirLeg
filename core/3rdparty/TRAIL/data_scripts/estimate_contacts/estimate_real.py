#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .core import (ClosedFormFootForceEstimator, MujocoFootForceEstimator,
                       estimate_contacts_from_joint_csv, write_contacts_csv)
except ImportError:
    from core import (ClosedFormFootForceEstimator, MujocoFootForceEstimator,
                      estimate_contacts_from_joint_csv, write_contacts_csv)
from common.robot import load_robot  # noqa: E402  (core put TRAIL on sys.path)


def discover_real_sequences(data_root: Path) -> list[Path]:
    """查找 bag 转换后的序列目录：每个目录下应有 robot/imu.csv 和 robot/joints.csv。"""
    if (data_root / "robot" / "imu.csv").exists() and (data_root / "robot" / "joints.csv").exists():
        return [data_root]
    seqs = []
    for joints_path in data_root.rglob("robot/joints.csv"):
        seq_dir = joints_path.parent.parent
        if (seq_dir / "robot" / "imu.csv").exists():
            seqs.append(seq_dir)
    return sorted(set(seqs))


def main() -> None:
    # model and pressure defaults come from the robot descriptor (robots/<robot>/leg_odometry.yaml (robot:))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default=None, help="robot name under robots/ or a leg_odometry.yaml (robot:) (default go2)")
    robot = load_robot(pre.parse_known_args()[0].robot)
    pressure = robot["pressure"]
    parser = argparse.ArgumentParser(parents=[pre], description="Estimate contacts for real bag-extracted robot data.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path(robot["model"]))
    parser.add_argument("--method", choices=["jacobian", "torque_norm"], default=pressure["method"])
    parser.add_argument("--pressure-scale", type=float, default=pressure["pressure_scale"])
    parser.add_argument("--pressure-mode", choices=["normal_abs", "normal_positive", "normal_negative", "force_norm"],
                        default=pressure["pressure_mode"])
    parser.add_argument("--normalize-sum", action="store_true", default=pressure["normalize_sum"])
    parser.add_argument("--damping", type=float, default=pressure["damping"])
    parser.add_argument("--smooth-alpha", type=float, default=pressure["smooth_alpha"])
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--kinematics", choices=["closed_form", "mujoco"], default="closed_form",
                        help="closed-form leg kinematics (vectorised) or the MuJoCo reference (MuJoCo XML models only)")
    parser.add_argument("--default-dt", type=float, default=0.0025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    seqs = discover_real_sequences(args.data_root)
    if not seqs:
        raise FileNotFoundError(f"No real sequences with robot/imu.csv and robot/joints.csv under {args.data_root}")

    estimator = None
    if args.method == "jacobian":
        estimator_type = ClosedFormFootForceEstimator if args.kinematics == "closed_form" else MujocoFootForceEstimator
        estimator = estimator_type(args.model.resolve(), args.pressure_mode)
    print(f"[setup] real sequences={len(seqs)}, method={args.method}, output=robot/contacts.csv")

    for idx, seq in enumerate(seqs, start=1):
        output = seq / "robot" / "contacts.csv"
        if output.exists() and not args.overwrite:
            print(f"[skip] {idx}/{len(seqs)} exists: {output}")
            continue
        try:
            estimate = estimate_contacts_from_joint_csv(
                seq / "robot" / "joints.csv",
                estimator=estimator,
                method=args.method,
                pressure_scale=args.pressure_scale,
                pressure_mode=args.pressure_mode,
                normalize_sum=args.normalize_sum,
                damping=args.damping,
                smooth_alpha=args.smooth_alpha,
                power=args.power,
                default_dt=args.default_dt,
            )
            write_contacts_csv(output, estimate)
            print(f"[write] {idx}/{len(seqs)} {output} samples={len(estimate.sample)}")
        except Exception as exc:
            print(f"[error] {idx}/{len(seqs)} {seq}: {exc}")
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()

