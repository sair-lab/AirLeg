#!/usr/bin/env python3
"""Start-to-end drift on the Go2 test sequences.

Each Go2 test sequence starts and ends with the robot standing still in front of an AprilGrid board. The ground truth
(gt_v3: <seq>_traj.txt, T_camstart_cam of the left infra1 optical frame; <seq>.json with the start / end standing windows
t_first..t_last) gives the camera pose in both windows. The estimate (TUM, left camera poses T_w_C, image timestamps) is

  1. averaged over each window (all its poses inside; if none, e.g. keyframe-only outputs that add no keyframe while
     standing, the nearest pose within --hold-s seconds, or further if --legs-root shows the robot stood still);
  2. aligned by the rigid transform mapping the start-window estimate onto the start-window ground truth;
  3. compared with the end-window ground truth: translation error (m) and rotation error (deg);
  4. drift % = translation error / the FIXED reference path length of the sequence (REF_PATH_M below, the median of
     the path lengths estimated by all evaluated methods, so every method is divided by the same number).

The estimate's own path length between the windows is reported too (path_m) but not used for the percentage.

usage:
  eval_go2_drift.py RUN_DIR [RUN_DIR ...] [--source online] [--csv out.csv]
      RUN_DIR/<seq>/ of an S3 run: --source online (default) = the per-frame online back-end trajectory saved by the VO
      (<seq>/vo/trajectory_frame.txt: every frame's pose with the latest back-end correction); runs made before the VO
      saved it fall back to the poses recorded from its topics (frames/frame_backend.tum, before the first back-end
      pose frames/frame_frontend.tum), marked "recorded" in the output. Any other value is a file under <seq>/vo/
      (e.g. trajectory_backend.txt, the final keyframe back-end trajectory).
  eval_go2_drift.py --traj test_stairs_02=/path/a.tum test_night_06=/path/b.tum [...]
      single trajectories (left camera T_w_C, TUM).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

GT_DIR = Path('/path/to/go2_data/gt_v3')

# Fixed reference path length per test sequence (m): median of the start-to-end path lengths of all evaluated methods.
# Only these sequences are evaluated.
REF_PATH_M = {
    'test_flat_pebble_04': 33.402,
    'test_densefoliage_01': 55.528,
    'test_flat_lowtexture_03': 92.250,
    'test_flat_grass_03': 157.443,
    'test_night_06': 165.356,
    'test_mixed_hybrid_05': 181.129,
    'test_stairs_02': 139.230,
    'test_slope_asphalt_01': 335.172,
    'test_flat_asphalt_multilap_02': 521.125,
    'test_mall_crowded_07': 123.632,
}


def load_tum(path):
    d = np.loadtxt(path, comments='#', ndmin=2)
    d = d[np.isfinite(d).all(1)]
    return d[np.argsort(d[:, 0], kind='stable')]


def to_T(rows):
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = Rot.from_quat(rows[:, 4:8]).as_matrix()
    T[:, :3, 3] = rows[:, 1:4]
    return T


def mean_pose(T):
    U, _, Vt = np.linalg.svd(T[:, :3, :3].sum(0))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = T[:, :3, 3].mean(0)
    return out


def standing(legs_root, seq, t0, t1, max_dq=0.25):
    """True if the 95th percentile of the largest joint speed over [t0, t1] is below max_dq rad/s (legs.npz: t ns, dq)."""
    if legs_root is None:
        return False
    legs = np.load(Path(legs_root) / seq / 'legs.npz')
    t, v = legs['t'] * 1e-9, np.abs(legs['dq']).max(1)
    m = (t >= t0) & (t <= t1)
    return bool(m.any()) and float(np.percentile(v[m], 95)) < max_dq


def drift(est, seq, gt_dir=GT_DIR, ref_path=REF_PATH_M, hold_s=2.0, legs_root=None):
    """Drift of the estimate est (TUM rows, left camera T_w_C) on sequence seq; dict with the metrics or 'error'."""
    if seq not in ref_path:
        return dict(error=f'no reference path length for {seq}')
    info = json.loads((gt_dir / f'{seq}.json').read_text())
    gt = load_tum(gt_dir / f'{seq}_traj.txt')
    if len(est) == 0:
        return dict(error='empty trajectory')
    poses, counts, held = {}, {}, []
    for w in ('start', 'end'):
        t_a, t_b = info[w]['t_first'], info[w]['t_last']
        m = (gt[:, 0] >= t_a - 1e-6) & (gt[:, 0] <= t_b + 1e-6)
        Tg = mean_pose(to_T(gt[m]))
        inside = (est[:, 0] >= t_a) & (est[:, 0] <= t_b)
        if inside.any():
            Te, n = mean_pose(to_T(est[inside])), int(inside.sum())
        else:
            d = np.where(est[:, 0] < t_a, t_a - est[:, 0], est[:, 0] - t_b)
            k = int(np.argmin(d))
            lo, hi = (est[k, 0], t_a) if est[k, 0] < t_a else (t_b, est[k, 0])
            if d[k] > hold_s and not standing(legs_root, seq, lo, hi):
                return dict(error=f'no estimate in the {w} window (nearest pose {d[k]:.1f} s away)')
            Te, n = to_T(est[k:k + 1])[0], 0
            held.append(f'{w} {d[k]:.1f}s')
        poses[w], counts[w] = (Tg, Te), n
    Tg0, Te0 = poses['start']
    Tg1, Te1 = poses['end']
    Ta = Tg0 @ np.linalg.inv(Te0) @ Te1                      # end estimate in the start-camera (ground-truth) frame
    dp = Ta[:3, 3] - Tg1[:3, 3]
    rot = np.degrees(np.linalg.norm(Rot.from_matrix(Tg1[:3, :3].T @ Ta[:3, :3]).as_rotvec()))
    seg = est[(est[:, 0] >= info['start']['t_last']) & (est[:, 0] <= info['end']['t_first']), 1:4]
    path = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else float('nan')
    trans = float(np.linalg.norm(dp))
    return dict(trans_m=trans, rot_deg=float(rot), drift_pct=100 * trans / ref_path[seq], ref_path_m=ref_path[seq],
                path_m=path, dx=float(dp[0]), dy=float(dp[1]), dz=float(dp[2]),
                n_start=counts['start'], n_end=counts['end'], held=';'.join(held))


def run_trajectory(seqdir, source):
    """(trajectory, note) of one sequence of an S3 run directory; trajectory None if missing."""
    has = lambda p: p.exists() and p.stat().st_size > 0
    if source == 'online':
        p = seqdir / 'vo' / 'trajectory_frame.txt'
        if has(p):
            return load_tum(p), ''
        est = run_trajectory(seqdir, 'frames_online')[0]
        return est, 'recorded' if est is not None else ''
    if source == 'frames_online':
        fb, ff = seqdir / 'frames/frame_backend.tum', seqdir / 'frames/frame_frontend.tum'
        if has(fb):              # before the first back-end solution the online output is the front end
            b = load_tum(fb)
            f = load_tum(ff) if has(ff) else b[:0]
            return np.r_[f[f[:, 0] < b[0, 0]], b], ''
        return (load_tum(ff) if has(ff) else None), ''   # visual_only: no back end
    p = seqdir / 'vo' / source
    return (load_tum(p) if has(p) else None), ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='*', type=Path, help='S3 run directories (<run>/<seq>/...)')
    ap.add_argument('--traj', nargs='+', default=[], metavar='SEQ=FILE', help='single TUM trajectories')
    ap.add_argument('--source', default='online',
                    help='online (vo/trajectory_frame.txt), frames_online (recorded topics) or a file under <seq>/vo/')
    ap.add_argument('--gt-dir', type=Path, default=GT_DIR)
    ap.add_argument('--ref-path', type=Path, help='JSON {seq: reference path length m} replacing the built-in lengths')
    ap.add_argument('--hold-s', type=float, default=2.0, help='max distance of the nearest pose to a window without poses')
    ap.add_argument('--legs-root', type=Path, help='<root>/<seq>/legs.npz: accept a further pose if the robot stood still')
    ap.add_argument('--csv', type=Path)
    a = ap.parse_args()
    if not a.runs and not a.traj:
        ap.error('give run directories or --traj SEQ=FILE')
    ref = json.loads(a.ref_path.read_text()) if a.ref_path else REF_PATH_M
    jobs = []                                                 # (run name, seq, trajectory or None)
    for item in a.traj:
        seq, _, f = item.partition('=')
        jobs.append(('traj', seq, load_tum(f) if Path(f).is_file() and Path(f).stat().st_size else None, ''))
    for run in a.runs:
        for seqdir in sorted(p for p in run.iterdir()
                             if p.is_dir() and p.name in ref and (a.gt_dir / f'{p.name}.json').exists()):
            jobs.append((run.name, seqdir.name, *run_trajectory(seqdir, a.source)))
    rows = []
    for run, seq, est, note in jobs:
        r = dict(error='no trajectory') if est is None else drift(est, seq, a.gt_dir, ref, a.hold_s, a.legs_root)
        rows.append(dict(run=run, seq=seq, **r, **({"note": note} if note else {})))
        print(f'{run:20} {seq:30} ' + (r['error'] if 'error' in r else
              f'trans {r["trans_m"]:6.3f} m  rot {r["rot_deg"]:5.2f} deg  drift {r["drift_pct"]:5.2f} % of '
              f'{r["ref_path_m"]:6.1f} m  (own path {r["path_m"]:6.1f} m{", held " + r["held"] if r["held"] else ""}'
              f'{", " + note if note else ""})'))
    print()
    runs = list(dict.fromkeys(r['run'] for r in rows))
    for run in runs:
        ok = [r for r in rows if r['run'] == run and 'error' not in r]
        n = sum(r['run'] == run for r in rows)
        if ok:
            print(f'{run:20} mean over {len(ok)}/{n}: trans {np.mean([r["trans_m"] for r in ok]):.3f} m  '
                  f'rot {np.mean([r["rot_deg"] for r in ok]):.2f} deg  drift {np.mean([r["drift_pct"] for r in ok]):.2f} %')
    if len(runs) > 1:      # per sequence mean over the runs, then over the sequences evaluated in every run
        seqs = [s for s in dict.fromkeys(r['seq'] for r in rows)
                if all(any(r['run'] == run and r['seq'] == s and 'error' not in r for r in rows) for run in runs)]
        if seqs:
            per = [np.mean([r['drift_pct'] for r in rows if r['seq'] == s and 'error' not in r]) for s in seqs]
            print(f'{"all runs":20} mean over {len(seqs)} sequences evaluated in every run: drift {np.mean(per):.2f} %')
    if a.csv and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with a.csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 0 if all('error' not in r for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
