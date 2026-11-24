"""Teacher check of the fused S2 rounds (alternating.teacher_check).

Every training sequence is cut into `window_s` windows of its pure visual teacher (prepare BA, camera poses T_w_C, as
body positions with the round's T_C_B). Per window: teacher path length / path length of the round's leg velocities
|v| dt. Windows whose ratio deviates from the median of all training windows by more than `tolerance` are left out of
the round's training pairs (excluded_intervals.json in the visual dataset directory); windows with a leg path below
`min_leg_path_m` are not judged. A global scale error of the leg model is shared by all windows and cancels in the
median; what remains are teacher segments whose visual scale disagrees with the rest. The fused
teacher already contains the leg edges, so the check uses the pure visual one.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

EXCLUDED_FILE = 'excluded_intervals.json'


def _teacher_positions(trajectory, camera_from_body):
    rows = np.loadtxt(trajectory, ndmin=2)
    rows = rows[np.argsort(rows[:, 0], kind='stable')]
    t_cb = np.asarray(camera_from_body[:3], dtype=float)
    rotation = Rotation.from_quat(rows[:, 4:8])
    return rows[:, 0], rows[:, 1:4] + rotation.apply(t_cb)


def _leg_path(velocity_csv):
    """(times, cumulative |v| dt) of the network velocity output of stage4 infer."""
    v = np.genfromtxt(velocity_csv, delimiter=',', names=True)
    speed = np.sqrt(v['vx'] ** 2 + v['vy'] ** 2 + v['vz'] ** 2)
    return v['time'], np.r_[0.0, np.cumsum(speed[:-1] * np.diff(v['time']))]


def _windows(times, positions, leg_times, leg_cum, window_s, min_leg_path):
    """[(start, end, ratio or None)] per window of the teacher trajectory."""
    teacher_cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    out, start = [], times[0]
    while start < times[-1]:
        end = min(start + window_s, times[-1])
        i0, i1 = np.searchsorted(times, [start, end])
        i1 = min(i1, len(times) - 1)
        leg = np.interp(times[i1], leg_times, leg_cum) - np.interp(times[i0], leg_times, leg_cum)
        ratio = (teacher_cum[i1] - teacher_cum[i0]) / leg if leg >= min_leg_path and i1 > i0 else None
        out.append((float(start), float(end), ratio))
        start += window_s
    return out


def check(round_dir, teachers, extrinsics, cfg):
    """{sequence id: [(start, end)]} excluded windows of the training sequences (teachers: id -> pure visual teacher
    trajectory, extrinsics: id -> T_C_B); writes round_dir/teacher_check.json."""
    window_s, tolerance = float(cfg.get('window_s', 10.0)), float(cfg.get('tolerance', 0.2))
    min_leg_path = float(cfg.get('min_leg_path_m', 1.0))
    if window_s <= 0 or tolerance <= 0 or min_leg_path < 0:
        raise ValueError('teacher_check needs window_s > 0, tolerance > 0 and min_leg_path_m >= 0')
    windows = {}
    for sid, teacher in teachers.items():
        times, positions = _teacher_positions(teacher, extrinsics[sid])
        leg_times, leg_cum = _leg_path(Path(round_dir) / sid / 'leg_odom/tmp/network_velocity.csv')
        windows[sid] = _windows(times, positions, leg_times, leg_cum, window_s, min_leg_path)
    ratios = [r for w in windows.values() for _, _, r in w if r is not None]
    if not ratios:
        return {}
    median = float(np.median(ratios))
    excluded, report = {}, dict(median_ratio=median, tolerance=tolerance, window_s=window_s, sequences={})
    for sid, w in windows.items():
        bad = [(a, b) for a, b, r in w if r is not None and abs(r / median - 1.0) > tolerance]
        judged = [r for _, _, r in w if r is not None]
        excluded[sid] = bad
        report['sequences'][sid] = dict(
            windows=len(w), judged=len(judged), excluded=len(bad), excluded_s=float(sum(b - a for a, b in bad)),
            median_ratio=float(np.median(judged) / median) if judged else None)
    total = sum(len(w) for w in windows.values())
    report['excluded_windows'] = sum(len(b) for b in excluded.values())
    report['excluded_fraction'] = report['excluded_windows'] / max(total, 1)
    (Path(round_dir) / 'teacher_check.json').write_text(json.dumps(report, indent=1))
    print(f"[teacher_check] median teacher/leg path {median:.3f}: {report['excluded_windows']} of {total} windows "
          f"excluded ({100 * report['excluded_fraction']:.1f} %)", flush=True)
    return excluded


def write_excluded(directory, intervals):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / EXCLUDED_FILE).write_text(json.dumps([list(i) for i in intervals]))


def pair_filter(directory):
    """Predicate (t0, t1) -> bool: the pair does not overlap an excluded window (always true without the file)."""
    path = Path(directory) / EXCLUDED_FILE
    if not path.is_file():
        return lambda t0, t1: True
    intervals = [tuple(i) for i in json.loads(path.read_text())]
    return lambda t0, t1: not any(t0 <= b and t1 >= a for a, b in intervals)
