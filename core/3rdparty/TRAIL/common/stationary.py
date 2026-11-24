"""Stationary detection of the leg odometry (S2 infer / validation and the S3 online worker, config `stationary:`).

A sample is stationary when, over the trailing `window_s` (causal: identical online and offline), every joint speed
stays below `joint_vel_max` and every gyro axis (base axes) has a standard deviation below `gyro_std_max` (the static
test of common/gyro_bias.py). The network twist of a stationary sample is set to zero. Without it a leg model that
drifts while the robot stands keeps triggering degraded-mode probe keyframes and re-link loops in the S3 front
end and moves the fused S2 teacher.
gyro_mean_max (optional): also the trailing mean angular rate (norm) must stay below it; a slow steady
body rotation on planted feet has a small gyro std and would otherwise be zeroed as standing.
"""
from collections import deque
from pathlib import Path

import numpy as np

DEFAULTS = dict(enabled=False, window_s=0.5, gyro_std_max=0.01, joint_vel_max=0.25, gyro_mean_max=None)


def settings(cfg):
    """Detector keyword arguments of cfg['stationary'], or None when disabled."""
    s = dict(DEFAULTS, **((cfg or {}).get('stationary') or {}))
    if not s.pop('enabled'):
        return None
    if s['window_s'] <= 0 or s['gyro_std_max'] <= 0 or s['joint_vel_max'] <= 0 or (s['gyro_mean_max'] is not None and s['gyro_mean_max'] <= 0):
        raise ValueError('stationary needs positive window_s, gyro_std_max and joint_vel_max')
    return s


class Detector:
    """Causal stationary test, one sample at a time (time in s, gyro rad/s in base axes, max joint speed rad/s)."""

    def __init__(self, window_s, gyro_std_max, joint_vel_max, gyro_mean_max=None):
        self.window_s, self.gyro_std_max, self.joint_vel_max = float(window_s), float(gyro_std_max), float(joint_vel_max)
        self.gyro_mean_max = None if gyro_mean_max is None else float(gyro_mean_max)
        self.samples = deque()
        self.sum, self.square = np.zeros(3), np.zeros(3)
        self.last_moving = -np.inf

    def update(self, t, gyro, joint_speed):
        g = np.asarray(gyro, dtype=np.float64)
        if not joint_speed < self.joint_vel_max:
            self.last_moving = t
        self.samples.append((t, g))
        self.sum += g
        self.square += g * g
        while t - self.samples[0][0] > self.window_s:
            _, old = self.samples.popleft()
            self.sum -= old
            self.square -= old * old
        n = len(self.samples)
        variance = np.maximum(self.square / n - (self.sum / n) ** 2, 0.0)
        return bool(t - self.samples[0][0] >= 0.9 * self.window_s and t - self.last_moving >= self.window_s
                    and np.sqrt(variance.max()) < self.gyro_std_max
                    and (self.gyro_mean_max is None or np.linalg.norm(self.sum / n) < self.gyro_mean_max))


def sequence_mask(data_root, times, cfg):
    """Stationary flag at each of `times` (the robot sample times of a sequence), or None when disabled."""
    s = settings(cfg)
    if s is None:
        return None
    from common.gyro_bias import read_gyro
    from eval.real_utils import read_csv_real

    t, gyro = read_gyro(data_root)
    joints = read_csv_real(Path(data_root) / 'robot' / 'joints.csv')
    joint_time = np.asarray(joints['time'], dtype=np.float64)
    speed = np.max(np.abs(np.column_stack([np.interp(t, joint_time, np.asarray(joints[k], dtype=np.float64))
                                           for k in joints if k.endswith('_vel')])), axis=1)
    detector = Detector(**s)
    flags = np.array([detector.update(ti, gi, si) for ti, gi, si in zip(t, gyro, speed)], dtype=bool)
    index = np.searchsorted(t, np.asarray(times, dtype=np.float64), side='right') - 1
    return np.where(index >= 0, flags[np.clip(index, 0, len(t) - 1)], False)
