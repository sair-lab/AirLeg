from __future__ import annotations

import sys
from pathlib import Path

import torch

from common.yaml_base import load_section


def load_config(path: str | Path, section: str = 'simulation') -> dict:
    """YAML config with its `base:` chain merged (common/yaml_base.py); relative paths are relative to `path`. From a
    combined robots/<robot>/leg_odometry.yaml only `section` (S1 default: simulation); other files whole."""
    path = Path(path)
    cfg = load_section(path, section)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    cfg["_meta"]["config_dir"] = str(path.parent)
    return cfg


def resolve_device(name: str | None) -> torch.device:
    """"auto"/None: CUDA when available and this PyTorch build has kernels for the GPU architecture, else CPU."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            requested_arch = f"sm_{capability[0]}{capability[1]}"
            compiled_arches = set(torch.cuda.get_arch_list())
            if requested_arch in compiled_arches:
                return torch.device("cuda")
            print(f"CUDA device requires {requested_arch}, but this PyTorch build only contains "
                  f"{sorted(compiled_arches)}; using CPU.", file=sys.stderr, flush=True)
        return torch.device("cpu")
    return torch.device(name)
