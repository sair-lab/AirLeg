"""S3 online leg odometry: streaming inference, online gyro bias and LoRA adaptation.

Everything here reuses the offline S1/S2/S4 code paths (``stage4`` loss/inference
helpers, ``common.pressure``, ``common.gyro_bias``, ``common.model`` LoRA) and only
adds in-memory buffering. The S2 checkpoint is never written: LoRA adapters live in
memory, start with ``lora_b = 0`` (output identical to S2) and are re-created per run.

Input feature row (matches ``eval.real_utils.load_real_features``):
``[gyro_xyz - bias, acc_xyz, q(12, FR FL RR RL x hip thigh calf), dq(12), pressure(FL FR RL RR)]``, IMU columns and
the bias in base axes (the worker rotates each IMU sample with the checkpoint extrinsic, common/frames.py).
"""
from __future__ import annotations

import copy
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parametrize

import stage4  # noqa: F401  (also puts the vendored PyPose on sys.path)
import pypose as pp
from stage4 import (apply_linear_scale, batch_pair_loss, build_pairs, checkpoint_velocity_scale,
                    evaluate, load_stage_config, sliding_windows, weighted_total)
from common.leg_kinematics import LEG_ORDER
from common.gyro_bias import gauss_newton_bias, keyframe_pairs
from common.model import LoRAWeight, configure_velocity_lora, load_velocity_model
from common.pressure import PressureWindow, enabled as pressure_enabled


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.velocity_branch.named_parameters() if 'lora_' in n}


def load_lora_state(model, state):
    params = dict(model.velocity_branch.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def linear_only_adapters(branch):
    """Backbone adapters act on the linear velocity only: the gyro and uncertainty heads read the S2 (adapter-free)
    encoder feature, so the angular-rate correction and the variances stay those of the S2 model."""
    loras = [m for m in branch.modules() if isinstance(m, LoRAWeight)]

    def forward_with_gyro(x, gyro):
        h = branch.forward_features(x)
        scales = [m.scaling for m in loras]
        try:
            for m in loras:
                m.scaling = 0.0
            with torch.no_grad():
                h0 = branch.forward_features(x)
        finally:
            for m, value in zip(loras, scales):
                m.scaling = value
        _, refined, log_var, dg = branch.forward_heads(h0, gyro)
        return branch.velocity_head(h), refined, log_var, dg

    branch.forward_with_gyro = forward_with_gyro


def _plain_layer(layer):
    """A TransformerEncoderLayer with the configuration of `layer` and no parametrizations (eval mode)."""
    attn = layer.self_attn
    plain = torch.nn.TransformerEncoderLayer(
        d_model=attn.embed_dim, nhead=attn.num_heads, dim_feedforward=layer.linear1.out_features, dropout=0.0,
        activation=layer.activation, layer_norm_eps=layer.norm1.eps, batch_first=attn.batch_first,
        norm_first=layer.norm_first, bias=layer.linear1.bias is not None)
    plain.activation_relu_or_gelu = layer.activation_relu_or_gelu
    p0 = next(layer.parameters())
    return plain.to(device=p0.device, dtype=p0.dtype).eval()


def _tensor_of(layer, path, merged):
    """Tensor `path` (state_dict key of a plain layer) of a possibly parametrized layer: the S2 weight
    (parametrizations[...].original) or the current merged weight (W + s b a)."""
    *mods, name = path.split('.')
    module = layer
    for m in mods:
        module = getattr(module, m)
    if parametrize.is_parametrized(module, name):
        return getattr(module, name) if merged else module.parametrizations[name].original
    return getattr(module, name)


class LinearOnlyInference(torch.nn.Module):
    """Inference of a linear_only LoRA branch with a shared prefix (latency plan B).

    linear_only_adapters runs the whole encoder twice (with / without adapters). The layers before the first
    adapted one are identical in both passes: they run once here; the adapted layers and all after them run twice,
    as plain copies with the S2 weights (base: gyro and uncertainty heads) and with the merged weights W + s b a
    (merged: linear velocity head). refresh() re-merges after a new LoRA state. Outputs equal
    linear_only_adapters' forward_with_gyro. Inference only (no gradients to the adapters)."""

    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        layers = list(branch.encoder.layers)
        adapted = [i for i, layer in enumerate(layers)
                   if any(parametrize.is_parametrized(m) for m in layer.modules())]
        if not adapted:
            raise ValueError('LinearOnlyInference needs LoRA adapters in the encoder')
        self.split = adapted[0]
        self.prefix = layers[:self.split]
        self.adapted = layers[self.split:]
        self.base = torch.nn.ModuleList(_plain_layer(layer) for layer in self.adapted)
        self.merged = torch.nn.ModuleList(_plain_layer(layer) for layer in self.adapted)
        self.norm = branch.encoder.norm
        with torch.no_grad():
            for src, base, merged in zip(self.adapted, self.base, self.merged):
                for key, value in base.state_dict().items():
                    value.copy_(_tensor_of(src, key, False))
                for key, value in merged.state_dict().items():
                    value.copy_(_tensor_of(src, key, True))

    def refittable(self):
        """{merged state_dict key: tensor} of the merged weights that change with the LoRA state."""
        out = {}
        for k, src in enumerate(self.adapted):
            for key in self.merged[k].state_dict():
                *mods, name = key.split('.')
                module = src
                for m in mods:
                    module = getattr(module, m)
                if parametrize.is_parametrized(module, name):
                    out[f'{k}.{key}'] = getattr(module, name).detach()
        return out

    def refresh(self):
        with torch.no_grad():
            for k, src in enumerate(self.adapted):
                merged = self.merged[k].state_dict()
                for key, value in merged.items():
                    value.copy_(_tensor_of(src, key, True))

    def forward(self, x, gyro):
        branch = self.branch
        z = branch.pos(branch.input(x))
        for layer in self.prefix:
            z = layer(z, src_mask=None, is_causal=False, src_key_padding_mask=None)
        z0, z1 = z, z
        for base, merged in zip(self.base, self.merged):
            z0 = base(z0, src_mask=None, is_causal=False, src_key_padding_mask=None)
            z1 = merged(z1, src_mask=None, is_causal=False, src_key_padding_mask=None)
        if self.norm is not None:
            z0, z1 = self.norm(z0), self.norm(z1)
        _, refined, log_var, dg = branch.forward_heads(z0[:, -1], gyro)
        return branch.velocity_head(z1[:, -1]), refined, log_var, dg


def select_parameters(model, groups):
    """Trainable parameter names (velocity_branch.named_parameters) of the online groups:
    lora (backbone adapters, lora config), velocity_head_last (last Linear of the linear-velocity head),
    heads_last (last Linear of the velocity and gyro heads)."""
    branch = model.velocity_branch
    named = dict(branch.named_parameters())
    last = lambda head: [n for n in named if n.startswith(f'{head}.{len(getattr(branch, head)) - 1}.')]
    choose = {'lora': [n for n in named if 'lora_a' in n or 'lora_b' in n],
              'velocity_head_last': last('velocity_head'),
              'heads_last': last('velocity_head') + last('gyro_head')}
    names = []
    for group in groups:
        if group not in choose:
            raise ValueError(f'unknown online train group {group!r}; known: {sorted(choose)}')
        names += [n for n in choose[group] if n not in names]
    if not names:
        raise ValueError(f'online train groups {groups} select no parameters')
    return names


def fresh_online_model(checkpoint, device, lora_cfg=None):
    """Load the S2 model; merge any checkpoint adapters in memory and optionally attach new ones.

    New adapters have zero ``lora_b``, so the output equals the S2 model bit for bit.
    """
    model, normalizer, payload = load_velocity_model(Path(checkpoint), device)
    branch = model.velocity_branch
    # Adapter dropout acts on lora_a; it must be off while the checkpoint adapters are merged into the weights.
    model.eval()
    if getattr(branch, 'lora_config', None) is not None:
        for module in list(branch.modules()):
            if parametrize.is_parametrized(module):
                for name, adapters in list(module.parametrizations.items()):
                    if len(adapters) == 1 and isinstance(adapters[0], LoRAWeight):
                        parametrize.remove_parametrizations(module, name, leave_parametrized=True)
        branch.lora_config = None
    for p in model.parameters():
        p.requires_grad_(False)
    if lora_cfg:
        for p in configure_velocity_lora(model, dict(lora_cfg)):
            p.requires_grad_(True)
    if normalizer is None:
        raise ValueError(f'Checkpoint has no input normalizer: {checkpoint}')
    suffixes = list(model.cfg['data'].get('joint_suffixes', []))
    expected = 6 + 12 * len(suffixes) + (4 if model.cfg['data'].get('use_contacts', True) else 0)
    if suffixes != ['_pos', '_vel'] or model.input_dim != expected:
        raise ValueError(f'Online input supports joint_suffixes [_pos, _vel] only; checkpoint has '
                         f'{suffixes} and input_dim {model.input_dim}')
    model.to(device).eval()
    return model, normalizer, payload


class FeatureRing:
    """Time-stamped feature rows; gyro is stored raw so a newer bias applies to all rows."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.t, self.rows = [], []
        self.lock = threading.Lock()

    def append(self, t, row):
        with self.lock:
            if self.t and t <= self.t[-1]:
                return False
            self.t.append(float(t))
            self.rows.append(row)
            if len(self.t) > 2 * self.capacity:
                del self.t[:-self.capacity], self.rows[:-self.capacity]
            return True

    def span(self, t0, t1, pad=0):
        """Rows with time in [t0, t1] plus `pad` earlier rows, as (times, features)."""
        with self.lock:
            times = np.asarray(self.t)
            i0 = max(0, int(np.searchsorted(times, t0, side='left')) - pad)
            i1 = int(np.searchsorted(times, t1, side='right'))
            return times[i0:i1].copy(), np.asarray(self.rows[i0:i1], dtype=np.float32)

    def tail(self, n):
        with self.lock:
            return np.asarray(self.t[-n:]), np.asarray(self.rows[-n:], dtype=np.float32), len(self.t)


def normalized(features, bias, normalizer):
    x = features.copy()
    x[:, :3] -= bias.astype(np.float32)
    return x, normalizer.apply(x).astype(np.float32)


class StreamingInference:
    """Causal per-sample inference identical to ``stage4.py infer`` (``online_start`` readiness)."""

    def __init__(self, checkpoint, device, lora_cfg=None, buffer_s=120.0, rate_hz=125.0, max_gap_s=0.05):
        self.device = torch.device(device)
        self.checkpoint, self.lora_cfg = checkpoint, lora_cfg
        self.model, self.normalizer, self.payload = fresh_online_model(checkpoint, self.device, lora_cfg)
        self.shared = None        # LinearOnlyInference (latency plan B), enable_linear_only
        self.trt = None           # common.leg_trt.TrtSwitch (latency plan C), enable_trt
        self.log = lambda text: print(text, file=sys.stderr, flush=True)
        self.cfg = self.model.cfg
        self.window = int(self.cfg['data']['window'])
        self.scale = checkpoint_velocity_scale(self.payload)
        self.pressure = PressureWindow(self.cfg['pressure_normalization']) if pressure_enabled(self.cfg) else None
        self.ring = FeatureRing(buffer_s * rate_hz)
        self.lock = threading.Lock()
        self.ready_index = None   # absolute index of the first predictable sample
        self.count = 0            # samples pushed (absolute index of the next one)
        self.emitted = 0          # absolute index of the next sample to predict
        self.torch_version = 0    # LoRA state of the PyTorch model
        # a sensor gap longer than max_gap_s restarts the readiness: no window may span uneven time
        self.max_gap_s = float(max_gap_s)
        self.gaps = 0
        self.resume_index = 0

    def push(self, t, gyro_raw, acc, q, dq, pressure_raw):
        """Append one sample (network rate, IMU in base axes); returns the normalised pressure (FL FR RL RR)."""
        p = np.asarray(pressure_raw, dtype=np.float32)
        if self.ring.t and t <= self.ring.t[-1]:
            return None  # out of order (PressureWindow requires increasing time)
        if self.ring.t and t - self.ring.t[-1] > self.max_gap_s:
            self.gaps += 1
            self.ready_index = None
            self.resume_index = self.count + self.window - 1  # first sample whose window lies after the gap
        ready = True
        if self.pressure is not None:
            p, ready = self.pressure.update(p, t)
        self.ring.append(t, np.concatenate([gyro_raw, acc, q, dq, p]).astype(np.float32))
        if ready and self.ready_index is None:
            self.ready_index = max(self.window - 1, self.count, self.resume_index)
            self.emitted = self.ready_index
        self.count += 1
        return p

    def step(self, bias):
        """Predict every pending sample: list of (time, twist[6], variance[6])."""
        if self.ready_index is None or self.emitted >= self.count:
            return []
        pending = min(self.count - self.emitted, self.ring.capacity)
        times, rows, _ = self.ring.tail(pending + self.window - 1)
        if len(rows) < pending + self.window - 1:
            pending = len(rows) - self.window + 1
            if pending <= 0:
                return []
        x, xn = normalized(rows, bias, self.normalizer)
        windows = sliding_windows(torch.from_numpy(xn).to(self.device), self.window)
        gyro = torch.from_numpy(x[self.window - 1:, :3]).to(self.device)
        with self.lock, torch.no_grad():
            if self.trt is not None and self.trt.active:
                linear, refined, log_var, _ = self.trt.infer(windows, gyro)
            elif self.shared is not None:
                linear, refined, log_var, _ = self.shared(windows, gyro)
            else:
                linear, refined, log_var, _ = self.model.velocity_branch.forward_with_gyro(windows, gyro)
        linear = apply_linear_scale(linear, self.scale)
        twist = torch.cat([linear[..., :3], refined], dim=-1).cpu().numpy()
        variance = apply_linear_scale(log_var.exp(), self.scale * self.scale).cpu().numpy()
        self.emitted = self.count
        return list(zip(times[self.window - 1:], twist, variance))

    @property
    def version(self):
        """LoRA state of the weights that produced the latest outputs (TensorRT swaps after its refit)."""
        return self.trt.version if self.trt is not None and self.trt.active else self.torch_version

    def enable_linear_only(self, shared_prefix=False):
        """Adapters change the linear velocity only. shared_prefix: LinearOnlyInference (plan B, same outputs)
        instead of linear_only_adapters' two full encoder passes."""
        branch = self.model.velocity_branch
        linear_only_adapters(branch)
        if shared_prefix and any(isinstance(m, LoRAWeight) for m in branch.modules()):
            self.shared = LinearOnlyInference(branch).eval()

    def enable_trt(self, cfg, linear_only):
        """TensorRT inference (plan C), built in the background; PyTorch until it passes the switch-over check.
        Any failure (no tensorrt, build, check) is logged and the run continues with PyTorch."""
        try:
            from common.leg_trt import TrtSwitch
            self.trt = TrtSwitch(self, dict(cfg), linear_only)
        except Exception as error:   # e.g. import tensorrt failed
            self.trt = None
            self.log(f'TensorRT leg inference unavailable, PyTorch kept: {error!r}')

    def load_adapter(self, state):
        with self.lock:
            load_lora_state(self.model, state)
            if self.shared is not None:
                self.shared.refresh()
            self.torch_version += 1
            version = self.torch_version
        if self.trt is not None:
            self.trt.request_refit(version)


class OnlineGyroBias:
    """Gyro bias: static initialisation, then visual-rotation Gauss-Newton on a growing window.

    The static test and the GN are ``common.gyro_bias.static_bias`` / ``visual_bias`` on
    in-memory arrays (same criteria, Jacobian d r/d b = -R_j^T (C_j - C_i), Huber IRLS).
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.bias = np.zeros(3)
        self.initialized = False
        self.samples = []      # (t, gyro, max |dq|) during initialisation
        self.info = {}

    def add_static_sample(self, t, gyro, joint_speed, q=None):
        if self.initialized:
            return
        self.samples.append((t, np.asarray(gyro, float), float(joint_speed), None if q is None else np.asarray(q, float)))
        if t - self.samples[0][0] < float(self.cfg.get('init_s', 2.0)):
            return
        t_arr = np.array([s[0] for s in self.samples])
        g = np.stack([s[1] for s in self.samples])
        speed = np.array([s[2] for s in self.samples])
        # joint_drift_max [rad/s]: the joints must really be frozen: per window, the
        # least-squares slope of every joint angle stays below it (joint positions, so the encoder-velocity noise that
        # sets joint_vel_max averages out). A slow posture change with the feet planted rotates the base, and its gyro
        # rate would otherwise be taken as bias. Off (None) = unchanged behaviour.
        drift_max = self.cfg.get('joint_drift_max')
        q = np.stack([s[3] for s in self.samples]) if drift_max is not None else None
        window, step = float(self.cfg.get('static_window_s', 0.5)), float(self.cfg.get('static_step_s', 0.25))
        means = []
        for begin in np.arange(t_arr[0], t_arr[-1] - window + 1e-9, step):
            inside = (t_arr >= begin) & (t_arr < begin + window)
            if (inside.sum() >= 0.8 * window / np.median(np.diff(t_arr))
                    and g[inside].std(axis=0).max() < float(self.cfg.get('gyro_std_max', 0.01))
                    and speed[inside].max() < float(self.cfg.get('joint_vel_max', 0.25))
                    and (q is None or self._drift(t_arr[inside], q[inside]) < float(drift_max))):
                means.append(g[inside].mean(axis=0))
        if len(means) >= int(self.cfg.get('static_min_windows', 2)):
            self.bias = np.median(means, axis=0)
            self.info = dict(source='static', windows=len(means))
        else:
            self.info = dict(source='none', reason='robot not stationary during initialisation')
        self.initialized = True
        self.samples = []

    @staticmethod
    def _drift(t, q):
        """Largest |least-squares slope| of the joint angles over the window, rad/s."""
        dt = t - t.mean()
        return float(np.abs((dt[:, None] * (q - q.mean(axis=0))).sum(axis=0) / max((dt ** 2).sum(), 1e-12)).max())

    def update_visual(self, t, gyro, kf_t, R_WC, R_CB):
        """t, gyro: gyro samples in base axes; kf_t: keyframe times on the leg clock; R_WC: [K,3,3]; R_CB: base -> camera."""
        span = kf_t[-1] - kf_t[0]
        if span < float(self.cfg.get('min_window_s', 6.0)):
            return False
        pair_s = min(float(self.cfg.get('pair_s', 20.0)), span / 3.0)
        pairs = keyframe_pairs(kf_t, pair_s)
        if len(pairs) < 3:
            return False
        bias, r = gauss_newton_bias(t, gyro, kf_t, R_WC @ R_CB, pairs, self.bias, float(self.cfg.get('huber_deg', 1.0)),
                                    int(self.cfg.get('iterations', 6)))
        rms = float(np.degrees(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))))
        step = float(np.abs(bias - self.bias).max())
        ok = (np.isfinite(bias).all() and rms <= float(self.cfg.get('max_rms_deg', 2.0))
              and step <= float(self.cfg.get('max_step', 0.01)))
        if ok:
            ema = float(self.cfg.get('ema', 0.5))
            self.bias = (1.0 - ema) * self.bias + ema * bias
        self.info = dict(source='visual', accepted=bool(ok), pairs=len(pairs), window_s=float(span),
                         residual_rms_deg=rms, step=step)
        return ok


class OnlineLoRATrainer:
    """Rounds of small LoRA updates supervised by matured BA keyframe poses.

    Trains a private copy of the model; accepted adapters are handed back for a hot swap.
    """

    def __init__(self, inference: StreamingInference, cfg):
        self.cfg = dict(cfg)
        self.stage_cfg = load_stage_config(None)
        self.stage_cfg.update(self.cfg.get('loss_config', {}))
        self.model = copy.deepcopy(inference.model)
        self.normalizer = inference.normalizer
        self.scale = inference.scale
        self.window = inference.window
        self.device = inference.device
        groups = self.cfg.get('params', ['lora'])
        groups = [groups] if isinstance(groups, str) else list(groups)
        named = dict(self.model.velocity_branch.named_parameters())
        self.names = select_parameters(self.model, groups)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = [named[n] for n in self.names]
        for p in self.params:
            p.requires_grad_(True)
        self.count = int(sum(p.numel() for p in self.params))
        if self.count > int(self.cfg.get('max_params', 10000)):
            raise ValueError(f'online train groups {groups}: {self.count} parameters > max_params')
        self.initial = [p.detach().clone() for p in self.params]
        self.optimizer = self._optimizer()
        self.bad_rounds = 0

    def _optimizer(self):
        return torch.optim.AdamW(self.params, lr=float(self.cfg.get('lr', 1e-4)),
                                 weight_decay=float(self.cfg.get('weight_decay', 0.0)))

    def state(self):
        return {n: p.detach().clone() for n, p in zip(self.names, self.params)}

    def load(self, values):
        with torch.no_grad():
            for p, v in zip(self.params, values):
                p.copy_(v)

    def reset(self):
        self.load(self.initial)  # the S2 model (adapters: lora_b = 0; heads: S2 weights)
        self.optimizer = self._optimizer()
        self.bad_rounds = 0

    def _loss(self, pairs, data):
        losses = batch_pair_loss(self.model, *data[:3], pairs, *data[3:], self.stage_cfg, self.scale)
        return weighted_total(losses, self.stage_cfg)

    def round(self, ring: FeatureRing, kf_t, kf_poses, T_C_B, delay, bias, good=None):
        """kf_t: camera times; kf_poses: [K, 7] T_W_C (xyz, qxyzw); T_C_B: [7]; good: [K-1] usable keyframe pairs
        (quality gate) or None. Returns stats or None."""
        times, rows = ring.span(kf_t[0] + delay, kf_t[-1] + delay + 0.05, pad=self.window)
        if len(rows) < self.window + 2:
            return None
        x, xn = normalized(rows, bias, self.normalizer)
        windows = sliding_windows(torch.from_numpy(xn).to(self.device), self.window)
        x_raw = torch.from_numpy(x).to(self.device)
        pairs = sorted(build_pairs(kf_t, times, self.window, delay, 0, 0))
        if good is not None:
            pairs = [p for p in pairs if good[p[0]]]
        validation = pairs[4::5]
        training = [p for p in pairs if p not in validation]
        if len(training) < 2 or not validation:
            return None
        poses = pp.SE3(torch.as_tensor(kf_poses, dtype=torch.float32, device=self.device))
        extrinsic = pp.SE3(torch.as_tensor(T_C_B, dtype=torch.float32, device=self.device))
        data = (windows, x_raw, times, kf_t + delay, poses, extrinsic)

        def val():
            return evaluate(self.model, validation, *data, self.stage_cfg, self.scale)

        before, snapshot = val(), [p.detach().clone() for p in self.params]
        self.model.eval()  # dropout off; only adapters receive gradients
        anchor = float(self.cfg.get('anchor', 1e-4))
        for _ in range(int(self.cfg.get('steps', 5))):
            loss = self._loss(training, data)
            loss = loss + anchor * sum(((p - p0) ** 2).sum() for p, p0 in zip(self.params, self.initial))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg.get('grad_clip', 1.0)))
            self.optimizer.step()
        after = val()
        accepted = math.isfinite(after) and after <= before * float(self.cfg.get('tol', 1.0))
        if not accepted:
            self.load(snapshot)
        # Compare with the plain S2 model; persistent degradation resets the trained parameters.
        current = [p.detach().clone() for p in self.params]
        self.load(self.initial)
        base = val()
        self.load(current)
        ratio = (after if accepted else before) / max(base, 1e-12)
        self.bad_rounds = self.bad_rounds + 1 if ratio > float(self.cfg.get('rollback_ratio', 1.2)) else 0
        reset = self.bad_rounds >= int(self.cfg.get('rollback_patience', 3))
        if reset:
            self.reset()
        changed = accepted or reset
        return dict(before=before, after=after, base=base, accepted=accepted, reset=reset,
                    pairs=len(training), state=self.state() if changed else None)


class _Span:
    """FeatureRing stand-in in the trainer process: the rows the parent cut with the same span() arguments."""

    def __init__(self, times, rows):
        self.times, self.rows = times, rows

    def span(self, t0, t1, pad=0):
        return self.times, self.rows


def _trainer_process(conn, checkpoint, device, lora_cfg, train_cfg, linear_only, initial_state):
    """LoRA rounds in a spawned process (latency plan A): no GIL shared with the inference loop."""
    os.dup2(2, 1)   # fd 1 of the worker is the protocol pipe to the ROS bridge: nothing may be written there
    sys.stdout = sys.stderr
    try:
        inference = StreamingInference(checkpoint, device, lora_cfg, buffer_s=1.0)
        trainer = OnlineLoRATrainer(inference, train_cfg)
        if linear_only:
            linear_only_adapters(trainer.model.velocity_branch)
        # the in-process trainer is a deepcopy of the inference model (same random lora_a): start from its state
        load_lora_state(trainer.model, {k: v.to(trainer.device) for k, v in initial_state.items()})
        trainer.initial = [p.detach().clone() for p in trainer.params]
        trainer.optimizer = trainer._optimizer()
        del inference
        conn.send(('ready', trainer.count))
    except Exception:
        conn.send(('error', traceback.format_exc()))
        return
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        if request is None:
            return
        try:
            times, rows, kf_t, kf_poses, T_C_B, delay, bias, good = request
            stats = trainer.round(_Span(times, rows), kf_t, kf_poses, T_C_B, delay, bias, good)
            if stats is not None and stats['state'] is not None:
                stats['state'] = {k: v.cpu() for k, v in stats['state'].items()}
            conn.send(('ok', stats))
        except Exception:
            conn.send(('error', traceback.format_exc()))


class RemoteTrainer:
    """OnlineLoRATrainer.round in a spawned process (spawn: the parent has initialised CUDA). Same round()
    signature and result; the parent cuts the ring with the same span() arguments. A lost process disables the
    trainer (round() -> None, logged once) instead of blocking every later round."""

    def __init__(self, inference, train_cfg, linear_only, log=None):
        import multiprocessing as mp
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self.window = inference.window
        self.dead = False
        ctx = mp.get_context('spawn')
        self.conn, child = ctx.Pipe()
        state = {k: v.cpu() for k, v in lora_state(inference.model).items()}
        self.process = ctx.Process(target=_trainer_process, daemon=True, name='leg_trainer',
                                   args=(child, str(inference.checkpoint), str(inference.device), inference.lora_cfg,
                                         dict(train_cfg), bool(linear_only), state))
        self.process.start()
        child.close()   # only the trainer holds that end: its exit gives EOFError here
        kind, value = self._recv(timeout=600.0)
        if kind != 'ready':
            raise RuntimeError(f'LoRA trainer process failed to start: {value}')
        self.count = int(value)

    def _recv(self, timeout=None):
        t0 = time.monotonic()
        while not self.conn.poll(0.5):
            if not self.process.is_alive():
                raise EOFError('LoRA trainer process exited')
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise TimeoutError('LoRA trainer process did not answer')
        return self.conn.recv()

    def round(self, ring, kf_t, kf_poses, T_C_B, delay, bias, good=None):
        if self.dead:
            return None
        times, rows = ring.span(kf_t[0] + delay, kf_t[-1] + delay + 0.05, pad=self.window)
        try:
            self.conn.send((times, rows, np.asarray(kf_t), np.asarray(kf_poses), np.asarray(T_C_B), delay,
                            np.asarray(bias), None if good is None else np.asarray(good)))
            kind, value = self._recv()
        except (EOFError, OSError, BrokenPipeError) as error:
            self.dead = True
            self.log(f'LoRA trainer process lost ({error!r}): online adaptation disabled for this run')
            return None
        if kind == 'error':
            raise RuntimeError('LoRA round in the trainer process failed: ' + value.strip().splitlines()[-1])
        return value

    def close(self):
        try:
            self.conn.send(None)
        except Exception:
            pass


def _relative(a, b):
    """Relative pose a^-1 b of [7] poses (xyz, qxyzw): translation in a, rotation matrix."""
    from scipy.spatial.transform import Rotation
    Ra, Rb = Rotation.from_quat(a[3:]), Rotation.from_quat(b[3:])
    return Ra.inv().apply(b[:3] - a[:3]), (Ra.inv() * Rb)


class BackendGate:
    """Quality of the back-end keyframe pairs, from the back-end and front-end keyframe paths only.

    A pair is bad when the keyframes are far apart in time (tracking gap), a keyframe is missing from the front-end
    path, or the back-end relative pose departs from the front-end one (there the back-end leans on the leg edges and
    would teach the network its own output). Thresholds are multiples of the running medians (no dataset constants
    beyond small absolute floors). A round trains only when enough pairs of its window are good.
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get('enabled', False))
        self.history = {'dt': [], 'dtrans': [], 'drot': []}
        self.last = -np.inf

    def _threshold(self, key, k, floor):
        """max(floor, k x running median); before 20 samples: no gap test, disagreement floor only."""
        values = self.history[key][-int(self.cfg.get('history', 500)):]
        if len(values) < 20:
            return np.inf if key == 'dt' else floor
        return max(floor, k * float(np.median(values)))

    def evaluate(self, kf_t, poses, vo_t, vo_poses):
        """Returns (good[K-1] bool, info) for the consecutive pairs of the window."""
        K = len(kf_t)
        dt = np.diff(kf_t)
        dtrans, drot, found = np.full(K - 1, np.nan), np.full(K - 1, np.nan), np.zeros(K - 1, bool)
        if vo_t is not None and len(vo_t):
            idx = np.clip(np.searchsorted(vo_t, kf_t), 0, len(vo_t) - 1)
            prev = np.clip(idx - 1, 0, len(vo_t) - 1)
            idx = np.where(np.abs(vo_t[prev] - kf_t) < np.abs(vo_t[idx] - kf_t), prev, idx)
            match = np.abs(vo_t[idx] - kf_t) < float(self.cfg.get('match_s', 1e-3))
            for k in range(K - 1):
                if not (match[k] and match[k + 1]):
                    continue
                t_be, R_be = _relative(poses[k], poses[k + 1])
                t_vo, R_vo = _relative(vo_poses[idx[k]], vo_poses[idx[k + 1]])
                dtrans[k] = float(np.linalg.norm(t_be - t_vo))
                drot[k] = float(np.degrees((R_be.inv() * R_vo).magnitude()))
                found[k] = True
        new = kf_t[:-1] > self.last
        for key, values in (('dt', dt), ('dtrans', dtrans), ('drot', drot)):
            self.history[key] += [float(v) for v in values[new] if np.isfinite(v)]
            del self.history[key][:-int(self.cfg.get('history', 500))]
        self.last = max(self.last, float(kf_t[-2])) if K > 1 else self.last
        # offline check: the gap is the informative part; the disagreement only marks outliers
        lim_dt = self._threshold('dt', float(self.cfg.get('k_gap', 3.0)), 0.0)
        lim_t = self._threshold('dtrans', float(self.cfg.get('k', 5.0)), float(self.cfg.get('floor_m', 0.002)))
        lim_r = self._threshold('drot', float(self.cfg.get('k', 5.0)), float(self.cfg.get('floor_deg', 0.05)))
        gap, missing = dt > lim_dt, ~found
        disagree = found & ((dtrans > lim_t) | (drot > lim_r))
        good = ~(gap | missing | disagree)
        info = dict(good=int(good.sum()), gap=int(gap.sum()), missing=int(missing.sum()), disagree=int(disagree.sum()),
                    lim_t=round(lim_t, 4), lim_r=round(lim_r, 3))
        return good, info


class ContactTracker:
    """Per-leg contact from normalised pressure with enter/exit hysteresis and a minimum stable time.

    ``segment`` increments at every touchdown, so one stance phase shares one id.
    """

    def __init__(self, enter=0.20, exit=0.10, min_stable_s=0.02):
        self.enter, self.exit, self.min_stable = float(enter), float(exit), float(min_stable_s)
        self.state = np.zeros(4, dtype=np.uint8)
        self.candidate = np.zeros(4, dtype=np.uint8)
        self.since = np.zeros(4)
        self.segment = np.zeros(4, dtype=np.uint32)

    def update(self, t, pressure):
        for leg, p in enumerate(pressure):
            desired = 1 if p > self.enter else 0 if p < self.exit else self.state[leg]
            if desired == self.state[leg]:
                self.candidate[leg] = desired
            elif self.candidate[leg] != desired:
                self.candidate[leg], self.since[leg] = desired, t
            elif t - self.since[leg] >= self.min_stable:
                self.state[leg] = desired
                self.segment[leg] += int(desired)
        return self.state.copy(), self.segment.copy()


def foot_positions(estimator):
    """Foot-sphere centres (FL FR RL RR) in the base (trunk) frame B after ``estimator.estimate``.

    MuJoCo keeps the base at the identity, so body xpos is in the trunk frame.
    """
    mj = estimator.jacobian_estimator
    if hasattr(mj, 'kin'):  # closed-form kinematics: from the joint angles of the last estimate
        return mj.foot_positions(estimator.last_q)
    return np.stack([mj.data.xpos[mj.leg_body_ids[leg]] for leg in LEG_ORDER])
