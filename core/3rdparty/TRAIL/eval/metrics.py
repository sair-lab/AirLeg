"""Evaluation against the lidar SLAM reference (evaluation only; never used for estimation).

usage: python -m eval.metrics [robot]  (prints the up direction of every reference world)

Sequences, reference trajectories and robot IMU files come from the dataset registry of the robot descriptor
(robots/<robot>/leg_odometry.yaml (robot:) `dataset`, default go2); SEQS / COLS are the default robot's.

The reference world is the first lidar frame, not gravity-aligned on several sequences (20-27 deg), so its z is
not vertical. The up direction of the reference world is estimated from the robot accelerometer: specific force
rotated into the reference world with the reference attitude and averaged over the sequence (motion accelerations
cancel). Metrics after the usual SE3 alignment of the estimate onto the reference (estimate timestamps):
  ate      RMS position error (m)
  z        RMS error along the reference z (partly non-vertical)
  vert     RMS error along the gravity up direction
  horiz    RMS horizontal error
  rpe5     mean translation error of 5 s relative motions (m), alignment-free
Options of evaluate(): alignment (se3 / sim3), camera-lidar time offset, reference gap check, uniform time grid.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.robot import DEFAULT_ROBOT, load_robot  # noqa: E402

# up direction per sequence (derived from the robot IMU); generated files live under runs/ (git-ignored)
CACHE = ROOT / 'runs' / '_cache' / 'gravity'


def dataset(robot=None):
    """Dataset registry of `robot` (leg_odometry.yaml (robot:) `dataset`: sequences, labels, splits, reference / imu templates)."""
    return load_robot(robot)['dataset']


SEQS = list(dataset()['sequences'])
COLS = list(dataset()['labels'])


@lru_cache(None)
def reference(seq, robot=None):
    """Reference trajectory [N, 8] (TUM, sorted) and the up direction of its world for sequence `seq`."""
    data = dataset(robot)
    ref = np.loadtxt(data['reference'].format(seq=seq), comments='#')
    ref = ref[np.argsort(ref[:, 0])]
    name = load_robot(robot)['name']
    cache = CACHE / (f'gravity_{seq}.json' if name == DEFAULT_ROBOT else f'gravity_{name}_{seq}.json')
    if cache.exists():
        up = np.array(json.loads(cache.read_text())['up'])
    else:
        # robot IMU (sample,time,gyro_xyz,acc_xyz) in the IMU frame; rotated into base axes (common/frames.py), the
        # axes of the reference attitude (identity for the Unitree robots, the inverted ANYmal IMU flips x and z)
        from common.frames import imu_extrinsic, rotate_vectors
        imu = np.loadtxt(data['imu'].format(seq=seq), delimiter=',', skiprows=1)
        t = imu[:, 1]
        m = (t > ref[0, 0]) & (t < ref[-1, 0])
        R = Slerp(ref[:, 0], Rotation.from_quat(ref[:, 4:8]))(t[m])
        f = R.apply(rotate_vectors(imu[m, 5:8], imu_extrinsic(robot).R_BI)).mean(axis=0)
        up = f / np.linalg.norm(f)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'up': up.tolist(), 'tilt_deg': float(np.degrees(np.arccos(up[2])))}))
    return ref, up


def _interp_with_gaps(t, times, values, max_gap):
    """Linear interpolation of values [N, k] at t; False where the bracketing samples are > max_gap apart."""
    j = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    ok = (times[j] - times[j - 1]) <= max_gap
    w = ((t - times[j - 1]) / np.maximum(times[j] - times[j - 1], 1e-12))[:, None]
    return (1 - w) * values[j - 1] + w * values[j], ok


def _align(est, g, align):
    """est aligned onto g: 'se3' (rotation + translation) or 'sim3' (with scale, Umeyama). Returns (aligned, scale)."""
    mp, mg = est.mean(0), g.mean(0)
    U, S, Vt = np.linalg.svd((g - mg).T @ (est - mp))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    scale = 1.0
    if align == 'sim3':
        scale = float(np.trace(np.diag(S) @ D) / np.sum((est - mp) ** 2))
    elif align != 'se3':
        raise ValueError(f'align must be se3 or sim3: {align}')
    return scale * (est - mp) @ (U @ D @ Vt).T + mg, scale


def evaluate(path, seq, align='se3', time_offset=0.0, max_gap=0.1, grid_hz=None, max_est_gap=1.0, robot=None):
    """Metrics of a TUM trajectory against the reference (keys: see the module docstring, plus n and scale).

    align: 'se3' (default) or 'sim3' (see _align), time_offset (s):
    t_reference = t_estimate + time_offset (camera-lidar offset, see estimate_time_offset); max_gap: estimate
    samples whose bracketing reference samples are further apart are dropped; grid_hz: evaluate on a uniform time
    grid (linear interpolation of the estimate, skipping estimate gaps > max_est_gap) so that keyframe-rate and
    frame-rate outputs are weighted alike; None keeps the estimate timestamps; robot: dataset
    registry of robots/<robot>/leg_odometry.yaml (robot:) (default go2).
    """
    p = Path(path)
    if not p.exists():
        return None
    d = np.loadtxt(p, ndmin=2)
    if d.size == 0 or len(d) < 10:
        return None
    d = d[np.argsort(d[:, 0])]
    ref, up = reference(seq, robot)
    t, est = d[:, 0] + time_offset, d[:, 1:4]
    if grid_hz:
        grid = np.arange(t[0], t[-1], 1.0 / grid_hz)
        est, ok = _interp_with_gaps(grid, t, est, max_est_gap)
        t, est = grid[ok], est[ok]
    m = (t > ref[0, 0]) & (t < ref[-1, 0])
    t, est = t[m], est[m]
    g, ok = _interp_with_gaps(t, ref[:, 0], ref[:, 1:4], max_gap)
    t, est, g = t[ok], est[ok], g[ok]
    if len(t) < 10:
        return None
    al, scale = _align(est, g, align)
    r = al - g
    v = r @ up
    h = r - np.outer(v, up)
    # 5 s relative motions (alignment-free norms of the difference of displacement vectors after rotation fit)
    j = np.searchsorted(t, t + 5.0)
    ok = j < len(t)
    i, j = np.nonzero(ok)[0], j[ok]
    rpe = np.linalg.norm((al[j] - al[i]) - (g[j] - g[i]), axis=1).mean() if len(i) else float('nan')
    rms = lambda x: float(np.sqrt(np.mean(np.sum(np.atleast_2d(x.T).T ** 2, axis=-1) if x.ndim > 1 else x ** 2)))
    return dict(ate=rms(r), z=rms(r[:, 2]), vert=rms(v), horiz=rms(h), rpe5=float(rpe), n=int(len(t)), scale=scale)


def estimate_time_offset(path, seq, search=0.1, step=0.002, **kwargs):
    """Camera-lidar time offset (s) minimising the ATE over [-search, search] (evaluation diagnostic only)."""
    offsets = np.arange(-search, search + 1e-12, step)
    ates = [(evaluate(path, seq, time_offset=o, **kwargs) or {'ate': np.inf})['ate'] for o in offsets]
    k = int(np.argmin(ates))
    return float(offsets[k]), float(ates[k])


def table(rows, keys=('ate', 'vert', 'rpe5'), base=None, robot=None, seqs=None):
    """rows: {label: {seq: metrics}} -> markdown (per sequence 'ate / vert', means of every key); seqs: subset of the
    robot's sequences (default all)."""
    data = dataset(robot)
    label_of = dict(zip(data['sequences'], data['labels']))
    seqs = list(seqs or data['sequences'])
    cols = [label_of[s] for s in seqs]
    out = ['| 变体 / 运行 | ' + ' | '.join(cols) + ' | ' + ' | '.join(f'均值 {k}' for k in keys) + ' |',
           '|' + '---|' * (len(cols) + 1 + len(keys))]
    for label, per in rows.items():
        ok = all(per.get(s) for s in seqs)
        cells = [f'{per[s]["ate"]:.3f} / {per[s]["vert"]:.3f}' if per.get(s) else '—' for s in seqs]
        means = [f'{np.mean([per[s][k] for s in seqs]):.4f}' if ok else '—' for k in keys]
        out.append(f'| {label} | ' + ' | '.join(cells) + ' | ' + ' | '.join(means) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    robot = sys.argv[1] if len(sys.argv) > 1 else None
    for s, c in zip(dataset(robot)['sequences'], dataset(robot)['labels']):
        ref, up = reference(s, robot)
        print(c, 'up in reference world', np.round(up, 3), 'tilt from ref z %.1f deg' % np.degrees(np.arccos(up[2])))
