"""Load-adaptive foot pressure preprocessing; all times are in seconds."""
from collections import deque

import numpy as np


def enabled(cfg):
    return cfg.get('data', {}).get('use_contacts', True) and cfg.get('pressure_normalization', {}).get('enabled', False)


def channels():
    # Pressure occupies the last four input channels, after IMU and joints.
    return slice(-4, None)


class PressureWindow:
    """Keep the latest window_s of VALID sample duration, bridging invalid spans.

    update returns (normalized_pressure, ready). Invalid samples are still output.
    Reset by constructing a new instance at each sequence/model boundary.
    """
    def __init__(self, settings):
        self.window = float(settings.get('online_window_s', 3.0))
        self.init = float(settings.get('online_init_s', 1.0))
        mean = float(settings['train_mean_total_pressure'])
        ratio = float(settings.get('min_total_pressure_ratio', 0.5))
        if not np.isfinite([self.init, self.window, mean, ratio]).all() or not 0 < self.init <= self.window or mean <= 0 or ratio <= 0:
            raise ValueError('Pressure windows, training mean and floor ratio must be positive; init <= window')
        self.floor = max(mean * ratio, 1e-6)
        self.samples = deque()
        self.duration = self.integral = 0.0
        self.previous_time = None

    def update(self, pressure, timestamp):
        p = np.asarray(pressure, dtype=np.float64)
        if p.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(timestamp):
            raise ValueError('Expected four finite pressures and a finite timestamp')
        dt = 0.0 if self.previous_time is None else float(timestamp - self.previous_time)
        if self.previous_time is not None and dt <= 0:
            raise ValueError('Pressure timestamps must increase strictly')
        self.previous_time = float(timestamp)
        p = np.maximum(p, 0.0)
        total = float(p.sum())
        if total >= self.floor and dt > 0:
            self.samples.append([dt, total])
            self.duration += dt
            self.integral += dt * total
            while self.duration > self.window + 1e-12:
                excess = min(self.duration - self.window, self.samples[0][0])
                self.integral -= excess * self.samples[0][1]
                self.duration -= excess
                self.samples[0][0] -= excess
                if self.samples[0][0] <= 1e-12:
                    self.samples.popleft()
        denominator = max(self.floor, self.integral / self.duration if self.duration else self.floor)
        return (p / denominator).astype(np.float32), self.duration + 1e-9 >= self.init


def normalize_features(x, times, cfg, online=False):
    if not enabled(cfg):
        return x
    settings = dict(cfg['pressure_normalization'])
    if not online:
        settings['online_window_s'] = settings.get('offline_window_s', 3.0)
        settings['online_init_s'] = settings['online_window_s']
    state = PressureWindow(settings)
    result = np.array(x, dtype=np.float32, copy=True)
    cols = channels()
    p, total, dt = pressure_samples(x, times, cfg)
    valid = (total >= state.floor) & (dt > 0)
    elapsed = np.cumsum(np.where(valid, dt, 0.0))
    if not valid.any():
        result[:, cols] = p / state.floor
        return result
    # Prefix integrals on the valid-time axis; invalid spans consume no window.
    edges = np.r_[0., elapsed[valid]]
    integrals = np.r_[0., np.cumsum(total[valid] * dt[valid])]
    right = elapsed if online else np.maximum(elapsed, min(state.window, elapsed[-1]))
    left = np.maximum(0., right - state.window)
    integral = np.interp(right, edges, integrals) - np.interp(left, edges, integrals)
    mean = np.divide(integral, right - left, out=np.full(len(x), state.floor), where=right > left)
    result[:, cols] = p / np.maximum(mean, state.floor)[:, None]
    return result


def pressure_samples(x, times, cfg):
    p = np.maximum(np.asarray(x[:, channels()], dtype=np.float64), 0.)
    times = np.asarray(times, dtype=np.float64)
    if len(times) != len(p) or not np.isfinite(times).all() or not np.isfinite(p).all():
        raise ValueError('Invalid pressure samples/timestamps')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Pressure timestamps must increase strictly')
    dt = np.r_[0., np.diff(times)] if len(times) else np.zeros(0)
    return p, p.sum(1), dt


def online_start(x, times, cfg):
    """First prediction index: enough sensor history AND valid pressure duration."""
    start = int(cfg['data']['window']) - 1
    if enabled(cfg):
        state = PressureWindow(cfg['pressure_normalization'])
        _, total, dt = pressure_samples(x, times, cfg)
        ready = np.flatnonzero(np.cumsum(np.where(total >= state.floor, dt, 0.)) + 1e-9 >= state.init)
        if len(ready):
            return max(start, int(ready[0]))
        raise ValueError('Sequence has insufficient valid pressure for online initialization')
    return start


def fit_pressure(sequences, cfg, loader, normalizer=None, online=False):
    """Two streaming passes over raw TRAIN data, retaining only four-channel stats.

    loader(sequence) returns raw features and timestamps. Stored in cfg, hence
    carried by existing checkpoint APIs without changing their signatures.
    """
    if not enabled(cfg):
        return
    cols = channels()
    total = count = 0
    for seq in sequences:
        x, _ = loader(seq)
        p = np.maximum(x[:, cols], 0.0)
        if not np.isfinite(p).all():
            raise ValueError('Non-finite training pressure')
        total += float(p.sum(dtype=np.float64))
        count += len(p)
    if not count or total <= 0:
        raise ValueError('Pressure normalization requires positive raw training pressure')
    cfg['pressure_normalization']['train_mean_total_pressure'] = total / count
    if normalizer is not None:
        sums = np.zeros(4); squares = np.zeros(4); count = 0
        for seq in sequences:
            x, t = loader(seq)
            p = normalize_features(x, t, cfg, online=online)[:, cols].astype(np.float64)
            sums += p.sum(0); squares += (p * p).sum(0); count += len(p)
        mean = sums / count
        normalizer.mean[cols] = mean
        normalizer.std[cols] = np.sqrt(np.maximum(squares / count - mean * mean, 1e-12))
