#!/usr/bin/env python3
"""S3 online leg-odometry worker (Torch environment), driven by the ROS bridge over pipes.

stdin  (length-prefixed pickles): ('lowstate', t, gyro3, acc3, q12, dq12, tau12)  [robot network_rate_hz,
        IMU frame, joint order FR FL RR RL x hip thigh calf], ('path', times[K], poses[K,7] T_W_C),
        ('gyro_bias', t, bias3) [back-end total gyro bias, B; used with bias.source: backend], ('stop',)
The IMU sample is rotated into base axes on arrival (checkpoint IMU extrinsic, common/frames.py); the emitted twist,
gyro bias and foot positions are in the base frame B, T_C_B maps B into the left camera.
stdout (length-prefixed pickles): ('init', T_C_B[7]), ('odom', [records]), ('log', text)

Modes: online_adapt (inference + gyro bias + LoRA), fusion_only (inference + gyro bias).
The S2 checkpoint is only read; adapters start at zero and are rebuilt every run.
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import queue
import struct
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from common.frames import config_imu_extrinsic
from common.leg_kinematics import LOWSTATE_LEG_ORDER
from common.online import (BackendGate, ContactTracker, OnlineGyroBias, OnlineLoRATrainer, RemoteTrainer,
                           StreamingInference, foot_positions, linear_only_adapters)
from common.robot import load_robot
from common.stationary import Detector, settings as stationary_settings
from common.yaml_base import load_section
from data_scripts.estimate_contacts.core import LEG_ORDER, EmaState, OnlineFootPressureEstimator
from stage4 import checkpoint_camera_from_body, resolve_device

RECORDER_JOINTS = [f'{leg}_{joint}_joint' for leg in LOWSTATE_LEG_ORDER for joint in ('hip', 'thigh', 'calf')]
LEG_JOINTS = [[RECORDER_JOINTS.index(f'{leg}_{joint}_joint') for joint in ('hip', 'thigh', 'calf')]
              for leg in LEG_ORDER]


class Channel:
    """Length-prefixed pickles of plain Python objects (the bridge may use another NumPy)."""

    def __init__(self):
        self.lock = threading.Lock()
        # the protocol gets its own descriptor and fd 1 becomes stderr: stray writes of native libraries (TensorRT
        # prints to fd 1) or of child processes can no longer corrupt the stream
        sys.stdout.flush()
        self.out = os.fdopen(os.dup(1), 'wb', buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr  # keep stray prints off the protocol stream

    def send(self, message):
        data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        with self.lock:
            self.out.write(struct.pack('<I', len(data)) + data)
            self.out.flush()

    def log(self, text):
        self.send(('log', text))


def torch_checkpoint_cfg(path):
    """Training config stored in a checkpoint (robot, IMU extrinsic, window)."""
    import torch
    return torch.load(path, map_location='cpu', weights_only=False).get('cfg') or {}


def read_messages(stream, inbox):
    """A broken or truncated message ends the worker cleanly (the bridge restarts it) instead of hanging."""
    try:
        while True:
            header = stream.read(4)
            if len(header) < 4:
                break
            inbox.put(pickle.loads(stream.read(struct.unpack('<I', header)[0])))
    except Exception as error:
        inbox.put(('error', f'input stream failed: {error!r}'))
    inbox.put(('stop',))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['online_adapt', 'fusion_only'], required=True)
    args = parser.parse_args()
    cfg = load_section(args.config, 'online')   # online: of leg_odometry.yaml, with its base: chain
    base = args.config.resolve().parent
    checkpoint = Path(cfg['checkpoint'])
    checkpoint = checkpoint if checkpoint.is_absolute() else base / checkpoint
    device = resolve_device(cfg.get('device', 'auto'))
    channel = Channel()
    adapt = args.mode == 'online_adapt'

    # robot: this config, else the one the checkpoint was trained for (S1 robot:), else go2
    payload_cfg = torch_checkpoint_cfg(checkpoint)
    robot = load_robot(cfg.get('robot') or payload_cfg.get('robot'))
    rate_hz = float(robot.get('network_rate_hz', 125.0))
    # a gap longer than max_gap_s restarts the window (default max(0.05 s, 2.5 sample periods))
    max_gap_s = float(cfg['max_gap_s']) if cfg.get('max_gap_s') is not None else max(0.05, 2.5 / rate_hz)
    train_cfg = cfg.get('train', {})
    groups = train_cfg.get('params', ['lora'])
    groups = [groups] if isinstance(groups, str) else list(groups)
    inference = StreamingInference(checkpoint, device, cfg.get('lora') if adapt and 'lora' in groups else None,
                                   buffer_s=float(cfg.get('buffer_s', 120.0)), rate_hz=rate_hz, max_gap_s=max_gap_s)
    R_BI = config_imu_extrinsic(inference.cfg).R_BI.astype(np.float32)   # IMU -> base axes (checkpoint extrinsic)
    reported_gaps = 0
    extrinsic = cfg.get('extrinsic', {}).get('T_C_B') or checkpoint_camera_from_body(inference.payload)
    T_C_B = np.asarray(extrinsic, dtype=np.float64)
    R_CB = Rotation.from_quat(T_C_B[3:]).as_matrix()
    delay = float(cfg.get('time_delay', 0.0))
    pressure = robot['pressure']
    contact_cfg = cfg.get('contact', {})
    model = Path(contact_cfg['model']) if contact_cfg.get('model') else Path(robot['model'])
    model = model if model.is_absolute() else base / model
    # Unsmoothed Jacobian pressure for contact timing; the network input keeps the offline EMA.
    estimator = OnlineFootPressureEstimator(
        method='jacobian', model_path=model,
        pressure_mode=contact_cfg.get('pressure_mode', pressure['pressure_mode']), normalize_sum=False,
        damping=float(contact_cfg.get('damping', pressure['damping'])), smooth_alpha=0.0)
    ema = EmaState(float(np.clip(contact_cfg.get('smooth_alpha', pressure['smooth_alpha']), 0.0, 0.999)))
    contacts = ContactTracker(**{k: contact_cfg[k] for k in ('enter', 'exit', 'min_stable_s') if k in contact_cfg})
    bias = OnlineGyroBias(cfg.get('bias', {}))
    still_cfg = stationary_settings(cfg)   # zero twist while standing, as stage4 infer (common/stationary.py)
    still = Detector(**still_cfg) if still_cfg else None
    inference.log = channel.log
    linear_only = bool(train_cfg.get('linear_only', False))
    # latency plan A: LoRA rounds in a spawned process; B: shared-prefix linear_only inference; C: TensorRT
    separate = adapt and bool(train_cfg.get('separate_process', False))
    shared_prefix = bool(train_cfg.get('shared_prefix', False))
    trt_cfg = cfg.get('trt') or {}
    trainer = None
    if adapt:
        trainer = (RemoteTrainer(inference, train_cfg, linear_only, log=channel.log) if separate
                   else OnlineLoRATrainer(inference, train_cfg))
        if linear_only:   # adapters change the linear velocity only
            inference.enable_linear_only(shared_prefix)
            if not separate:
                linear_only_adapters(trainer.model.velocity_branch)
    if trt_cfg.get('enabled', False):
        inference.enable_trt(trt_cfg, adapt and linear_only)
    gate = BackendGate(train_cfg.get('gate')) if adapt else None
    vo_path = {}   # latest front-end keyframe path (quality gate)
    backend_bias = bias.cfg.get('source', 'visual') == 'backend'   # gyro bias from the back-end (gyro_bias_total)
    feedback = dict(n=0)   # back-end gyro bias messages
    mature_lag = int(train_cfg.get('mature_lag', 10))
    channel.send(('init', T_C_B.tolist()))
    channel.log(f'S3 leg worker ready: mode={args.mode}, device={device}, robot={robot["name"]}, '
                f'checkpoint={checkpoint}, rate={rate_hz:g} Hz, imu_extrinsic={config_imu_extrinsic(inference.cfg).to_dict()}, '
                f'stationary={still_cfg}'
                + f', bias source={"backend" if backend_bias else "visual"}'
                + (f', train groups={groups} ({trainer.count} parameters), lr={train_cfg.get("lr", 1e-4)}, '
                   f'linear_only={bool(train_cfg.get("linear_only", False))}, '
                   f'gate={gate.cfg if gate.enabled else "off"}' if trainer else '')
                + f', separate_process={separate}, shared_prefix={inference.shared is not None}, '
                  f'trt={dict(trt_cfg) if trt_cfg.get("enabled", False) else "off"}')

    feet = {}            # sample time -> (foot12, pressure4, contact4, segment4)
    # (time, foot positions in B (4x3)) history for gate.passive_m (no-step keyframe pairs, kinematic test)
    steps_hist = collections.deque(maxlen=120000)
    steps_lock = threading.Lock()
    stationary = {}      # sample time -> stationary flag
    paths = queue.Queue(maxsize=1)
    stop = threading.Event()

    def high_level():
        """Gyro bias (both modes) and LoRA rounds (online_adapt) on matured keyframes. An exception in one round is
        logged and the loop continues."""
        state = dict(trained_until=-np.inf, bias_at=-np.inf)
        while not stop.is_set():
            try:
                times, poses = paths.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                high_level_round(times, poses, state)
            except Exception:
                import traceback
                channel.log('high-level round failed (skipped): ' + traceback.format_exc(limit=3).replace('\n', ' | '))

    def high_level_round(times, poses, state):
        trained_until, bias_at = state['trained_until'], state['bias_at']
        if len(times) <= mature_lag + 2:
            return
        times, poses = times[:-mature_lag], poses[:-mature_lag]
        window_s = float(bias.cfg.get('max_window_s', 120.0))
        keep = times >= times[-1] - window_s
        # the growing-window GN is pure Python: throttle it so it never starves inference
        if not backend_bias and bias.initialized and times[-1] - bias_at >= float(bias.cfg.get('update_every_s', 5.0)):
            bias_at = times[-1]
            t_gyro, rows = inference.ring.span(times[keep][0] + delay, times[-1] + delay)
            if len(t_gyro) > 10:
                R_WC = Rotation.from_quat(poses[keep, 3:]).as_matrix()
                bias.update_visual(t_gyro, rows[:, :3].astype(np.float64), times[keep] + delay, R_WC, R_CB)
                channel.log(f'gyro bias {np.round(bias.bias, 6).tolist()} {bias.info}')
        if trainer is None:
            state['bias_at'] = bias_at
            return
        fresh = np.count_nonzero(times > trained_until)
        if fresh < int(train_cfg.get('train_every', 3)):
            state['bias_at'] = bias_at
            return
        window = int(train_cfg.get('kf_window', 10))
        good, gate_info = None, {}
        if gate.enabled:
            vo_t, vo_poses = vo_path.get('latest', (None, None))
            good, gate_info = gate.evaluate(times[-window:], poses[-window:], vo_t, vo_poses)
        passive_m = float(train_cfg.get('gate', {}).get('passive_m', 0.0))
        if passive_m > 0.0:   # a pair without a step but with back-end motion is carried, not walked: not trained on
            with steps_lock:
                hist = list(steps_hist)
            if hist:
                ht = np.array([h[0] for h in hist])
                foot_mm = float(train_cfg.get('gate', {}).get('passive_foot_m', 0.01))
                def no_step(ta, tb):   # every foot within passive_foot_m of its position at ta, sampled ~every 0.1 s
                    i0, i1 = np.searchsorted(ht, ta + delay), np.searchsorted(ht, tb + delay)
                    if i0 >= len(ht) or i1 <= i0 or abs(ht[i0] - ta - delay) > 0.05:
                        return False
                    step = max(1, (i1 - i0) // max(5, int((tb - ta) / 0.1)))
                    f0 = hist[i0][1]
                    return all(np.linalg.norm(hist[i][1] - f0, axis=1).max() < foot_mm for i in range(i0, i1, step))
                kt, kp = times[-window:], poses[-window:]
                if good is None:
                    good = np.ones(len(kt) - 1, bool)
                n_passive = 0
                for k in range(len(kt) - 1):
                    if np.linalg.norm(kp[k + 1, :3] - kp[k, :3]) > passive_m and no_step(kt[k], kt[k + 1]):
                        good[k] = False
                        n_passive += 1
                gate_info['passive'] = n_passive
        trained_until = times[-1]
        state.update(trained_until=trained_until, bias_at=bias_at)
        if good is not None and good.sum() < int(gate.cfg.get('min_good', 6) if gate.enabled else 1):
            channel.log('LoRA round: gated, ' + ', '.join(f'{k}={v}' for k, v in gate_info.items()))
            return
        stats = trainer.round(inference.ring, times[-window:], poses[-window:], T_C_B, delay, bias.bias.copy(), good)
        if stats is None:
            return
        stats.update(gate_info)
        if stats['state'] is not None:
            inference.load_adapter(stats['state'])
        channel.log('LoRA round: ' + ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                                               for k, v in stats.items() if k != 'state'))

    worker = threading.Thread(target=high_level, daemon=True)
    worker.start()
    inbox = queue.Queue()
    threading.Thread(target=read_messages, args=(sys.stdin.buffer, inbox), daemon=True).start()

    while True:
        message = inbox.get()
        batch = [message]
        while not inbox.empty() and len(batch) < 256:
            batch.append(inbox.get_nowait())
        for item in batch:
            if item[0] == 'error':
                channel.log(item[1])
                continue
            if item[0] == 'stop':
                stop.set()
                if isinstance(trainer, RemoteTrainer):
                    trainer.close()
                return
            if item[0] == 'gyro_bias':   # back-end total gyro bias (bias.source: backend)
                if backend_bias and np.all(np.isfinite(item[2])):
                    new = np.asarray(item[2], np.float64)
                    if not bias.initialized or np.abs(new - bias.bias).max() > 2e-4 or feedback['n'] % 50 == 0:
                        channel.log(f'gyro bias from back-end {np.round(new, 6).tolist()} (was {np.round(bias.bias, 6).tolist()})')
                    bias.bias = new   # one atomic swap (the inference reads a copy per batch)
                    bias.initialized, bias.samples = True, []
                    bias.info = dict(source='backend', t=float(item[1]))
                    feedback['n'] += 1
                continue
            if item[0] == 'vo_path':
                vo_path['latest'] = (np.asarray(item[1], np.float64), np.asarray(item[2], np.float64))  # one atomic swap
                continue
            if item[0] == 'path':
                try:
                    paths.get_nowait()
                except queue.Empty:
                    pass
                paths.put((np.asarray(item[1], np.float64), np.asarray(item[2], np.float64)))
                continue
            _, t, gyro, acc, q, dq, tau = item
            q, dq, tau = (np.asarray(v, np.float64) for v in (q, dq, tau))
            if not (np.isfinite(t) and np.all(np.isfinite(gyro)) and np.all(np.isfinite(acc)) and np.all(np.isfinite(q))
                    and np.all(np.isfinite(dq)) and np.all(np.isfinite(tau))):
                continue  # the bridge already drops these; never feed non-finite values into the network
            try:
                raw = estimator.estimate(q[LEG_JOINTS], tau[LEG_JOINTS])
            except ValueError as error:
                channel.log(f'pressure estimation failed: {error}')
                continue
            previous = ema.value
            smoothed = ema.update(raw)
            gyro = R_BI @ np.asarray(gyro, np.float32)        # base axes from here on
            acc = R_BI @ np.asarray(acc, np.float32)
            pressure = inference.push(t, gyro, acc,
                                      q.astype(np.float32), dq.astype(np.float32), smoothed)
            if pressure is None:
                ema.value = previous  # rejected sample: undo the EMA step
                continue
            bias.add_static_sample(t, gyro, np.abs(dq).max(), q)   # joint angles: bias.joint_drift_max
            if still is not None:
                stationary[t] = still.update(t, gyro, np.abs(dq).max())
            # same load normalisation as the network input, applied to the unsmoothed pressure
            scale = float(np.sum(pressure)) / max(float(np.sum(smoothed)), 1e-9)
            contact, segment = contacts.update(t, raw * scale)
            feet[t] = (foot_positions(estimator).reshape(-1), pressure, contact, segment)
            with steps_lock:
                steps_hist.append((t, np.asarray(foot_positions(estimator), np.float32).reshape(4, 3)))
        if inference.gaps != reported_gaps:
            reported_gaps = inference.gaps
            channel.log(f'sensor gap > {inference.max_gap_s:.3f} s ({reported_gaps} so far): leg odometry resumes after one window')
        records = []
        applied_bias = bias.bias.copy()  # one read: the background thread may update the bias meanwhile
        for t, twist, variance in inference.step(applied_bias):
            foot, pressure, contact, segment = feet.pop(t, (np.zeros(12), np.zeros(4), np.zeros(4), np.zeros(4)))
            if stationary.pop(t, False):
                twist = np.zeros_like(twist)
            # plain Python types only: the ROS bridge runs another NumPy version
            records.append((float(t), twist.tolist(), variance.tolist(), applied_bias.tolist(), foot.tolist(),
                            np.asarray(pressure).tolist(), np.asarray(contact).tolist(), np.asarray(segment).tolist(),
                            int(inference.version)))
        for table in (feet, stationary):
            for stale in [k for k in table if records and k < records[-1][0]]:
                table.pop(stale)
            while len(table) > 2000:  # inference not ready yet (e.g. robot lifted): keep the newest only
                table.pop(next(iter(table)))
        if records:
            channel.send(('odom', records))


if __name__ == '__main__':
    main()
