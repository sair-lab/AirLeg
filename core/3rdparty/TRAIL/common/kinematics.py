from __future__ import annotations

import torch


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)

