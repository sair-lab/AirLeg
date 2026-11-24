from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from common.data import Normalizer, target_dim, uncertainty_dim, velocity_pred_dim


class LoRAWeight(nn.Module):
    """Add a trainable low-rank update to an existing 2-D weight tensor."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        # Dropout on the rank input dimension gives adapter dropout without
        # changing the frozen module's activation path.
        update = self.lora_b @ self.dropout(self.lora_a)
        return weight + self.scaling * update


def configure_velocity_lora(model: "PoseEstimator", config: dict | None) -> list[nn.Parameter]:
    """Inject LoRA into Transformer attention/FFN weights and return adapter parameters.

    ``nn.MultiheadAttention`` stores Q/K/V in one ``in_proj_weight``; it must
    be parametrized explicitly instead of relying on generic Linear wrapping.
    Calling this function repeatedly with the same model is safe.
    """
    if not config:
        return []
    branch = model.velocity_branch
    existing = getattr(branch, "lora_config", None)
    normalized = {
        "rank": int(config.get("rank", 4)),
        "alpha": float(config.get("alpha", 8.0)),
        "dropout": float(config.get("dropout", 0.05)),
        "target_attention": bool(config.get("target_attention", True)),
        "target_attention_output": bool(config.get("target_attention_output", False)),
        "target_ffn": bool(config.get("target_ffn", True)),
        # Apply adapters to the last N encoder layers; null means all layers.
        "num_layers": None if config.get("num_layers") is None else int(config["num_layers"]),
    }
    # Older checkpoints carry disabled adapters (alpha=0). Removing them
    # leaves the original weights unchanged and lets S2 enable fresh adapters.
    if existing is not None and existing != normalized and existing["alpha"] == 0.0:
        for module in list(branch.modules()):
            if parametrize.is_parametrized(module):
                for name, adapters in list(module.parametrizations.items()):
                    if len(adapters) == 1 and isinstance(adapters[0], LoRAWeight):
                        parametrize.remove_parametrizations(module, name, leave_parametrized=False)
        existing = None
    if existing is not None:
        if existing != normalized:
            raise ValueError(f"Model already has a different LoRA configuration: {existing}")
        return [p for n, p in branch.named_parameters() if "lora_a" in n or "lora_b" in n]

    def add(module: nn.Module, name: str) -> None:
        weight = getattr(module, name)
        if weight.ndim != 2:
            raise ValueError(f"LoRA only supports 2-D weights, got {tuple(weight.shape)}")
        adapter = LoRAWeight(
            weight.shape[0],
            weight.shape[1],
            normalized["rank"],
            normalized["alpha"],
            normalized["dropout"],
        ).to(device=weight.device, dtype=weight.dtype)
        parametrize.register_parametrization(
            module,
            name,
            adapter,
        )

    layers = branch.encoder.layers
    num_layers = normalized["num_layers"]
    if num_layers is not None and (num_layers <= 0 or num_layers > len(layers)):
        raise ValueError(f"LoRA num_layers must be in [1, {len(layers)}], got {num_layers}")
    selected_layers = layers if num_layers is None else layers[-num_layers:]
    for layer in selected_layers:
        if normalized["target_attention"]:
            add(layer.self_attn, "in_proj_weight")
        if normalized["target_attention_output"]:
            add(layer.self_attn.out_proj, "weight")
        if normalized["target_ffn"]:
            add(layer.linear1, "weight")
            add(layer.linear2, "weight")
    branch.lora_config = normalized
    return [p for n, p in branch.named_parameters() if "lora_a" in n or "lora_b" in n]


def checkpoint_lora_config(checkpoint: dict) -> dict | None:
    value = checkpoint.get("lora_config")
    return dict(value) if value else None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]，pe: [1, T, H]；输出仍为 [B, T, H]。
        # 给时间窗口中的每一帧加入固定位置编码，帮助 Transformer 感知时序顺序。
        return x + self.pe[:, : x.shape[1]]


class VelocityBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        velocity_dim: int,
        uncertainty_dim_: int,
        cfg: dict,
        predict_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        m = cfg["model"]
        hidden = int(m["hidden_dim"])
        self.output_dim = velocity_dim
        self.uncertainty_dim = uncertainty_dim_
        self.min_log_var = float(m.get("min_log_var", -8.0))
        self.max_log_var = float(m.get("max_log_var", 4.0))
        self.input = nn.Linear(input_dim, hidden)
        self.pos = SinusoidalPositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(m["transformer_heads"]),
            dim_feedforward=hidden * 4,
            dropout=float(m.get("dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(m["transformer_layers"]))
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(m.get("dropout", 0.1))),
            nn.Linear(hidden, velocity_dim),
        )
        # Gyro residual head: use the temporal feature and the current raw gyro
        # to predict a correction dg.  Zero initialization preserves the
        # original gyro-only behavior when loading an older checkpoint.
        self.gyro_head = nn.Sequential(
            nn.LayerNorm(hidden + 3),
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Dropout(float(m.get("dropout", 0.1))),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.gyro_head[-1].weight)
        nn.init.zeros_(self.gyro_head[-1].bias)
        self.uncertainty_head = None
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(m.get("dropout", 0.1))),
                nn.Linear(hidden, uncertainty_dim_),
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sensor window and return its last temporal feature."""
        z = self.encoder(self.pos(self.input(x)))
        return z[:, -1]

    def refine_gyro(self, hidden: torch.Tensor, gyro: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (gyro + dg, dg), where gyro is in physical rad/s units."""
        dg = self.gyro_head(torch.cat([hidden, gyro], dim=-1))
        return gyro + dg, dg

    def forward_with_gyro(
        self, x: torch.Tensor, gyro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict linear velocity, refined gyro, uncertainty and gyro residual."""
        return self.forward_heads(self.forward_features(x), gyro)

    def forward_heads(
        self, h: torch.Tensor, gyro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward_with_gyro from the encoder feature h (e.g. cached while the encoder is frozen)."""
        vel = self.velocity_head(h)
        refined_gyro, dg = self.refine_gyro(h, gyro)
        if self.uncertainty_head is None:
            log_var = vel.new_zeros(*vel.shape[:-1], self.uncertainty_dim)
        else:
            log_var = self.uncertainty_head(h).clamp(self.min_log_var, self.max_log_var)
        return vel, refined_gyro, log_var, dg

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入 x: [B, window, input_dim]。
        # input 映射后: [B, window, hidden]；Transformer 输出 z: [B, window, hidden]。
        h = self.forward_features(x)
        # vel: [B, 3]，只预测机体系线速度；log_var: [B, 6]，仍覆盖线速度+gyro角速度。
        vel = self.velocity_head(h)
        if self.uncertainty_head is None:
            return vel, vel.new_zeros(*vel.shape[:-1], self.uncertainty_dim)
        log_var = self.uncertainty_head(h).clamp(self.min_log_var, self.max_log_var)
        return vel, log_var


class PoseEstimator(nn.Module):
    def __init__(self, input_dim: int, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.output_dim = target_dim(cfg)
        self.velocity_dim = velocity_pred_dim(cfg)
        self.uncertainty_dim = uncertainty_dim(cfg)
        self.velocity_branch = VelocityBranch(
            input_dim,
            self.velocity_dim,
            self.uncertainty_dim,
            cfg,
            predict_uncertainty=bool(cfg.get("model", {}).get("predict_uncertainty", True)),
        )

    def compose_full_velocity(self, linear_vel: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        # linear_vel: [..., 3]，gyro: [..., 3] -> full velocity: [..., 6]。
        return torch.cat([linear_vel, gyro], dim=-1)

    def save_model(self, path: str | Path, normalizer: Normalizer | None = None, extra: dict | None = None) -> None:
        # checkpoint 同时保存模型结构配置和 normalizer，方便后续独立加载。
        payload = {
            "model": self.state_dict(),
            "cfg": self.cfg,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "extra": extra or {},
        }
        lora_config = getattr(self.velocity_branch, "lora_config", None)
        if lora_config is not None:
            payload["lora_config"] = lora_config
        if normalizer is not None:
            payload["normalizer_mean"] = normalizer.mean
            payload["normalizer_std"] = normalizer.std
        torch.save(payload, path)

    @classmethod
    def load_model(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["PoseEstimator", Normalizer | None, dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(int(ckpt["input_dim"]), ckpt["cfg"])
        configure_velocity_lora(model, checkpoint_lora_config(ckpt))
        model.load_state_dict(ckpt["model"], strict=False)
        normalizer = None
        if "normalizer_mean" in ckpt and "normalizer_std" in ckpt:
            normalizer = Normalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])
        return model, normalizer, ckpt

def velocity_head_output_dim(state_dict: dict[str, torch.Tensor]) -> int | None:
    for key in ("velocity_branch.velocity_head.4.weight", "velocity_branch.velocity_head.5.weight"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[0])
    return None


def load_velocity_model(path: Path, device: torch.device):
    """Velocity checkpoint -> (model, normalizer or None, payload); accepts 3- or 6-dim velocity heads."""
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload["model"]
    model = PoseEstimator(int(payload["input_dim"]), payload["cfg"])
    configure_velocity_lora(model, checkpoint_lora_config(payload))
    prediction_dim = velocity_head_output_dim(state_dict)
    if prediction_dim is not None and prediction_dim != model.velocity_branch.output_dim:
        final = model.velocity_branch.velocity_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Unexpected velocity head layout")
        model.velocity_branch.velocity_head[-1] = nn.Linear(final.in_features, prediction_dim)
        model.velocity_branch.output_dim = prediction_dim
        model.velocity_dim = prediction_dim
    model.load_state_dict(state_dict, strict=False)
    normalizer = None
    if "normalizer_mean" in payload and "normalizer_std" in payload:
        normalizer = Normalizer(payload["normalizer_mean"], payload["normalizer_std"])
    return model, normalizer, payload
