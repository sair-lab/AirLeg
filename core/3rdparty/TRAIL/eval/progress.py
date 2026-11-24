"""Shared progress page for experiment runs.

A run keeps its item states in <run>/status/items.tsv (<item>\\t<fraction 0-1>\\t<note>) and a frozen copy of the
earlier progress page in <run>/status/previous_progress.md; the page is previous_progress + this run's section.

usage: python -m eval.progress <run_dir> <title> [--target PROGRESS.md] [--loop]
       (--loop refreshes every minute until <run>/status/STOP exists)
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

DEFAULT_TARGET = Path(__file__).resolve().parents[1] / 'runs' / 's3_v13_ext_timeoffset' / 'PROGRESS.md'


def bar(fraction: float, width: int = 30) -> str:
    fraction = max(0.0, min(1.0, fraction))
    n = int(round(fraction * width))
    return '█' * n + '░' * (width - n) + f' {100 * fraction:5.1f}%'


def render(run: Path, title: str) -> str:
    rows = [line.split('\t') for line in (run / 'status' / 'items.tsv').read_text().splitlines() if line.strip()]
    total = sum(float(r[1]) for r in rows) / max(1, len(rows))
    lines = ['', '---', '', f'# {title}', '',
             f'更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}，结果目录 `runs/{run.name}/`', '',
             f'**总进度** `{bar(total, 40)}`', '', '| 项目 | 进度 | 说明 |', '|---|---|---|']
    lines += [f'| {r[0]} | `{bar(float(r[1]))}` | {r[2] if len(r) > 2 else ""} |' for r in rows]
    return '\n'.join(lines + [''])


def update(run: Path, title: str, target: Path = DEFAULT_TARGET) -> None:
    previous = (run / 'status' / 'previous_progress.md').read_text().rstrip('\n')
    target.write_text(previous + '\n' + render(run, title))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('title')
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    while True:
        update(run, args.title, args.target)
        if not args.loop or (run / 'status' / 'STOP').exists():
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
