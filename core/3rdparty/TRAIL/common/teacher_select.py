"""Teacher selection of the S2 rounds (sim_to_real.alternating.teacher_select).

Every consecutive keyframe pair (i, j) of a training sequence (visual dataset directory: keypoints with the round's
teacher points, robot/imu.csv, gyro_bias.csv) gets four errors of its teacher:
  px   median reprojection residual (px) of the points of i moved by the teacher's relative pose into j
  out  outlier fraction (residual >= inlier_px)
  rot  angle (deg) between the teacher's relative rotation and the gyro integral (gyro - bias, rotated into the camera
       frame by a per-sequence rotation fitted to all pairs)
  dev  |teacher path / leg path of its window_s window, relative to the median of all windows - 1| (as teacher_check)
Pairs with fewer than min_matches common points are never selected. Per sequence at least floor x the pairs of the old
rule (lost / weak keyframes skipped, teacher_check windows excluded) are kept, the best-scoring ones (diversity). The
selected pairs replace the old rule's in the round's training set (train_visual_finetune.py, teacher_select.json).

Thresholds:
  adaptive (recommended, {visual_tau, fused_tau}): from the round's own data. Per term z = (x - median) /
      max(1.4826 MAD, floor) over all training pairs, score = max z, a pair passes if score <= tau (visual_tau in the
      pure visual rounds, fused_tau in the fused rounds). No absolute numbers: a robot whose sensors are noisier overall
      is not rejected wholesale. Floors: visual_floor x the old rule in the pure visual rounds, the old rule in the fused.
  absolute: score = max(px / px_max, out / (1 - inlier_min), rot / rot_max_deg, dev / tolerance) (each optionally
      loosened by noise_ref: x max(1, median here / reference median)), pass if score <= s; s = visual_strictness in the
      pure visual rounds; fused rounds: mode pool (s = 1) or tighten (s = shrink ** fused round index, never looser).
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SELECT_FILE = 'teacher_select.json'
DEFAULTS = dict(mode='pool', px_max=2.0, inlier_px=3.0, inlier_min=0.5, rot_max_deg=1.0, tolerance=0.15,
                window_s=10.0, min_leg_path_m=1.0, min_matches=20, shrink=0.8, min_depth_m=0.2)


def _intrinsics(camera_yaml):
    p = yaml.safe_load(Path(camera_yaml).read_text())['cam0']['p']
    return float(p[0]), float(p[5]), float(p[2]), float(p[6])


def _keypoints(path):
    """{track id: (u, v, X, Y, Z)} of a keypoint file (9 columns: id, score, u, v, ..., X, Y, Z camera frame)."""
    try:
        d = np.loadtxt(path, ndmin=2)
    except ValueError:
        return {}
    if d.size == 0 or d.shape[1] != 9:
        return {}
    return {int(r[0]): r[[2, 3, 6, 7, 8]] for r in d}


def _poses(trajectory):
    rows = np.loadtxt(trajectory, ndmin=2)
    out = {}
    for r in rows:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(r[4:8]).as_matrix()
        T[:3, 3] = r[1:4]
        out[round(float(r[0]), 6)] = T
    return out


def _gyro(seq_dir):
    t, w = [], []
    with (Path(seq_dir) / 'robot/imu.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            t.append(float(row['time']))
            w.append((float(row['gyro_x']), float(row['gyro_y']), float(row['gyro_z'])))
    t, w = np.asarray(t), np.asarray(w)
    bias_file = Path(seq_dir) / 'gyro_bias.csv'
    if bias_file.is_file():
        b = np.genfromtxt(bias_file, delimiter=',', names=True)
        w = w - np.array([float(np.atleast_1d(b[k])[0]) for k in ('bgx', 'bgy', 'bgz')])
    order = np.argsort(t)
    return t[order], w[order]


def _integrate(t, w, t0, t1):
    """Rotation over [t0, t1] of the piecewise constant angular velocity w (sample k holds over [t_k, t_k+1))."""
    k0 = max(int(np.searchsorted(t, t0, side='right')) - 1, 0)
    k1 = int(np.searchsorted(t, t1, side='left'))
    R = Rotation.identity()
    for k in range(k0, min(k1, len(t) - 1) + 1):
        a, b = max(t[k], t0), min(t[k + 1] if k + 1 < len(t) else t1, t1)
        if b > a:
            R = R * Rotation.from_rotvec(w[k] * (b - a))
    return R


def _wahba(a, b):
    """Rotation R minimising sum |a_k - R b_k|^2."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def sequence_pairs(seq_dir, trajectory, cfg):
    """Per consecutive keyframe pair of seq_dir/keypoints: dict(stem, t0, t1, n, px, out, rot)."""
    seq_dir = Path(seq_dir)
    fx, fy, cx, cy = _intrinsics(seq_dir / 'camera.yaml')
    poses = _poses(trajectory)
    files = sorted((seq_dir / 'keypoints').glob('*.txt'))
    tg, wg = _gyro(seq_dir)
    pairs, prev = [], None
    for f in files:
        t = float(f.stem)
        cur = (f.stem, t, _keypoints(f), poses.get(round(t, 6)))
        if prev is not None:
            s0, t0, k0, T0 = prev
            _, t1, k1, T1 = cur
            rec = dict(stem=s0, stem1=f.stem, t0=t0, t1=t1, n=0, px=np.inf, out=1.0, rot_teacher=None, rot_gyro=None)
            if T0 is not None and T1 is not None:
                T10 = np.linalg.inv(T1) @ T0
                ids = [i for i in k0 if i in k1 and np.isfinite(k0[i][2:]).all()]
                if ids:
                    X = np.array([k0[i][2:] for i in ids])
                    uv = np.array([k1[i][:2] for i in ids])
                    Y = X @ T10[:3, :3].T + T10[:3, 3]
                    ok = Y[:, 2] > cfg['min_depth_m']
                    if ok.any():
                        e = np.hypot(fx * Y[ok, 0] / Y[ok, 2] + cx - uv[ok, 0], fy * Y[ok, 1] / Y[ok, 2] + cy - uv[ok, 1])
                        rec.update(n=int(ok.sum()), px=float(np.median(e)), out=float((e >= cfg['inlier_px']).mean()))
                rec['rot_teacher'] = Rotation.from_matrix((T0[:3, :3].T @ T1[:3, :3]))
                rec['rot_gyro'] = _integrate(tg, wg, t0, t1)
            pairs.append(rec)
        prev = cur
    # camera <- IMU rotation fitted to the rotation vectors of the pairs that turn by more than 1 deg
    a = np.array([p['rot_teacher'].as_rotvec() for p in pairs if p['rot_teacher'] is not None])
    b = np.array([p['rot_gyro'].as_rotvec() for p in pairs if p['rot_teacher'] is not None])
    big = np.linalg.norm(a, axis=1) > np.radians(1.0) if len(a) else np.zeros(0, bool)
    R_ci = _wahba(a[big], b[big]) if big.sum() >= 10 else None
    R_ci_rot = Rotation.from_matrix(R_ci) if R_ci is not None else None
    for p in pairs:
        if p['rot_teacher'] is None or R_ci_rot is None:
            p['rot'] = np.inf if p['rot_teacher'] is None else 0.0
        else:
            g = R_ci_rot * p['rot_gyro'] * R_ci_rot.inv()
            p['rot'] = float(np.degrees((p['rot_teacher'].inv() * g).magnitude()))
        del p['rot_teacher'], p['rot_gyro']
    return pairs


def _window_ratios(pairs, trajectory, velocity_csv, extr_t, cfg):
    """Teacher / leg path ratio of the window_s window of every pair (None if not judged)."""
    rows = np.loadtxt(trajectory, ndmin=2)
    rows = rows[np.argsort(rows[:, 0], kind='stable')]
    pos = rows[:, 1:4] + Rotation.from_quat(rows[:, 4:8]).apply(np.asarray(extr_t, float))
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    v = np.genfromtxt(velocity_csv, delimiter=',', names=True)
    speed = np.sqrt(v['vx'] ** 2 + v['vy'] ** 2 + v['vz'] ** 2)
    leg_cum = np.r_[0.0, np.cumsum(speed[:-1] * np.diff(v['time']))]
    times = rows[:, 0]
    ratios = []
    start = times[0]
    while start < times[-1]:
        end = min(start + cfg['window_s'], times[-1])
        i0, i1 = np.searchsorted(times, [start, end])
        i1 = min(i1, len(times) - 1)
        leg = np.interp(times[i1], v['time'], leg_cum) - np.interp(times[i0], v['time'], leg_cum)
        ratios.append((start, end, (cum[i1] - cum[i0]) / leg if leg >= cfg['min_leg_path_m'] and i1 > i0 else None))
        start += cfg['window_s']
    for p in pairs:
        p['ratio'] = next((r for a, b, r in ratios if a <= p['t0'] < b), None)


def old_rule_keep(seq_dir, pairs, excluded):
    """Pairs the old rule keeps: neither keyframe lost / weak in the preparation VO, outside the teacher-check windows."""
    bad = set()
    status = Path(seq_dir) / 'keyframe_status.csv'
    if status.is_file():
        with status.open(newline='') as f:
            bad = {r['timestamp'] for r in csv.DictReader(f) if r['track_state'] in ('lost', 'weak')}
    return [p['stem'] not in bad and p['stem1'] not in bad
            and not any(a <= p['t0'] < b or a < p['t1'] <= b for a, b in excluded) for p in pairs]


def select(round_dir, dataset_dir, train, excluded, fused_index, previous_s, config, write=True, pure_visual=False):
    """train: id -> dict(trajectory, extrinsic); excluded: id -> old teacher-check windows; fused_index: 0 for the first
    fused round. Writes the selection files; returns the strictness s used."""
    cfg = dict(DEFAULTS, **(config or {}))
    data = {}
    for sid, info in train.items():
        seq_dir = Path(dataset_dir) / 'train' / sid
        pairs = sequence_pairs(seq_dir, info['trajectory'], cfg)
        _window_ratios(pairs, info['trajectory'], Path(round_dir) / sid / 'leg_odom/tmp/network_velocity.csv',
                       info['extrinsic'][:3], cfg)
        data[sid] = (seq_dir, pairs, old_rule_keep(seq_dir, pairs, excluded.get(sid, [])))
    ratios = [p['ratio'] for _, ps, _ in data.values() for p in ps if p['ratio'] is not None]
    median = float(np.median(ratios)) if ratios else 1.0
    for _, ps, _ in data.values():
        for p in ps:
            p['dev'] = abs(p['ratio'] / median - 1.0) if p['ratio'] is not None else 0.0
    # noise_ref: {px, rot, dev} medians of the data the thresholds were set on (ANYmal pure
    # visual teacher); a term whose median on this data is higher is loosened by that ratio (never tightened), so the
    # thresholds follow the sensor noise floor instead of rejecting a robot's whole data for it.
    scale = {'px': 1.0, 'rot': 1.0, 'dev': 1.0}
    for k, ref in (cfg.get('noise_ref') or {}).items():
        v = np.array([p[k] for _, ps, _ in data.values() for p in ps], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) and ref > 0:
            scale[k] = max(1.0, float(np.median(v)) / float(ref))
    adaptive = cfg.get('adaptive')
    if adaptive:
        # data-driven thresholds. Per term over all training pairs of this round:
        # z = (x - median) / max(1.4826 MAD, floor); score = max z; pass if score <= tau (visual_tau / fused_tau).
        # No absolute numbers: a robot whose sensors are noisier overall is not rejected wholesale.
        floors = dict(px=0.05, out=0.02, rot=0.02, dev=0.005, **(adaptive.get('floors') or {}))
        allp = [p for _, ps, _ in data.values() for p in ps if p['n'] >= cfg['min_matches']]
        stats = {}
        for k in ('px', 'out', 'rot', 'dev'):
            v = np.array([p[k] for p in allp], dtype=float)
            v = v[np.isfinite(v)]
            m = float(np.median(v)) if len(v) else 0.0
            stats[k] = (m, max(1.4826 * float(np.median(np.abs(v - m))) if len(v) else 0.0, floors[k]))
        for _, ps, _ in data.values():
            for p in ps:
                p['score'] = np.inf if p['n'] < cfg['min_matches'] else max(
                    (p[k] - stats[k][0]) / stats[k][1] if np.isfinite(p[k]) else np.inf for k in stats)
        scale = {k: dict(median=v[0], sigma=v[1]) for k, v in stats.items()}
    else:
        for _, ps, _ in data.values():
            for p in ps:
                p['score'] = np.inf if p['n'] < cfg['min_matches'] else max(
                    p['px'] / (cfg['px_max'] * scale['px']), p['out'] / (1.0 - cfg['inlier_min']),
                    p['rot'] / (cfg['rot_max_deg'] * scale['rot']), p['dev'] / (cfg['tolerance'] * scale['dev']))
    floor_fraction = 1.0
    if adaptive:         # strictness = tau in robust sigmas; floors visual_floor x old rule / 1 x old rule
        s = float(adaptive['visual_tau'] if pure_visual else adaptive['fused_tau'])
        if pure_visual:
            floor_fraction = float(cfg.get('visual_floor', 0.5))
    elif pure_visual:      # strict in the pure visual rounds, floor = visual_floor x the old rule's pairs
        s, floor_fraction = float(cfg['visual_strictness']), float(cfg.get('visual_floor', 0.5))
    elif cfg['mode'] == 'pool':
        s = 1.0
    elif cfg['mode'] == 'tighten':
        s = cfg['shrink'] ** fused_index
        if previous_s is not None:
            s = min(s, previous_s)
    else:
        raise ValueError(f"teacher_select.mode must be pool or tighten, got {cfg['mode']}")
    summary = dict(mode=cfg['mode'], strictness=s, fused_index=fused_index, median_ratio=median,
                   pure_visual=bool(pure_visual), floor_fraction=floor_fraction, noise_scale=scale,
                   thresholds={k: cfg[k] for k in ('px_max', 'inlier_px', 'inlier_min', 'rot_max_deg', 'tolerance')},
                   sequences={})
    tot = dict(pairs=0, old=0, passed=0, floor_added=0, selected=0, old_and_selected=0)
    for sid, (seq_dir, ps, old) in data.items():
        scores = np.array([p['score'] for p in ps])
        passed = scores <= s
        n_old = int(sum(old))
        chosen = passed.copy()
        need = int(np.ceil(floor_fraction * n_old)) - int(chosen.sum())
        added = 0
        if need > 0:   # diversity floor: the sequence's best-scoring remaining pairs (finite score only)
            rest = [k for k in np.argsort(scores, kind='stable') if not chosen[k] and np.isfinite(scores[k])]
            for k in rest[:need]:
                chosen[k] = True
                added += 1
        stems = [p['stem'] for p, c in zip(ps, chosen) if c]
        if write:
            (seq_dir / SELECT_FILE).write_text(json.dumps(stems))
        rec = dict(pairs=len(ps), old=n_old, passed=int(passed.sum()), floor_added=added, selected=len(stems),
                   old_and_selected=int(sum(c and o for c, o in zip(chosen, old))),
                   median_px=float(np.median([p['px'] for p in ps if np.isfinite(p['px'])] or [np.nan])),
                   median_rot=float(np.median([p['rot'] for p in ps if np.isfinite(p['rot'])] or [np.nan])))
        summary['sequences'][sid] = rec
        for k in tot:
            tot[k] += rec[k]
    summary.update(tot)
    if write:
        (Path(round_dir) / SELECT_FILE).write_text(json.dumps(summary, indent=1))
    select.last = (summary, data)
    print(f"[teacher_select] {'adaptive (median, sigma)' if adaptive else 'noise scale'} {scale}", flush=True)
    print(f"[teacher_select] {'pure visual' if pure_visual else cfg['mode']} s={s:.3f}: {tot['selected']} pairs selected of {tot['pairs']} "
          f"(old rule {tot['old']}, passed {tot['passed']}, floor added {tot['floor_added']}, "
          f"old pairs dropped {tot['old'] - tot['old_and_selected']})", flush=True)
    return s


def pair_keep(directory):
    """Predicate stem -> bool, or None without a selection file (old rule)."""
    path = Path(directory) / SELECT_FILE
    if not path.is_file():
        return None
    stems = set(json.loads(path.read_text()))
    return lambda stem: stem in stems
