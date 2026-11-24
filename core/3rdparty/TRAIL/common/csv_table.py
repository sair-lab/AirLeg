"""Numeric CSV tables as {column: float64 array}.

Parsed with np.loadtxt (C parser since NumPy 1.23, same correctly rounded conversion as float()).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_numeric_csv(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    with path.open('r', encoding='utf-8', newline='') as f:
        header = next(csv.reader(f), None)
    if not header:
        raise ValueError(f'Empty CSV: {path}')
    values = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64, ndmin=2)
    if values.shape[0] == 0:
        raise ValueError(f'Empty CSV: {path}')
    if values.shape[1] != len(header):
        raise ValueError(f'{path}: {values.shape[1]} columns, header has {len(header)}')
    return {name: values[:, i].copy() for i, name in enumerate(header)}
