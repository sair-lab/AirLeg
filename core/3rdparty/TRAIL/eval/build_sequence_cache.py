#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import load_config
from common.data import list_sequences, save_sequence_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Go2 sequence .npy caches from CSV logs.")
    parser.add_argument("--config", default=str(ROOT.parents[2] / "robots" / "go2" / "leg_odometry.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "simulation"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seqs = list_sequences(args.data_root, args.split, args.max_sequences)
    total = len(seqs)
    t0 = time.time()
    print(f"[cache] split={args.split}, sequences={total}, force={args.force}", flush=True)
    for i, seq in enumerate(seqs, start=1):
        path = save_sequence_cache(seq, cfg, force=args.force)
        if i == 1 or i == total or i % int(cfg.get("train", {}).get("print_interval_steps", 20)) == 0:
            print(
                f"[cache] {i}/{total}: {seq.sequence_id}, path={path}, "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
