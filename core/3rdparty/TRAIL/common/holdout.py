"""Validation segments held out of every training sequence (S2 `holdout:`).

From the first keyframe of a training sequence, the last `length_s` seconds of every `period_s` seconds are
validation time. Training pairs whose interval touches a segment, or starts within `margin_s` after its end (the
network input window still reaches into it), are dropped; the segments form the pseudo validation sequence
<id>__holdout (same robot data and keypoints, fixed-BA trajectory of the segments only), whose trajectory error is
evaluated segment by segment. A single failed sequence then weighs as little in the validation as in training.
"""
import json
from pathlib import Path

import numpy as np

SUFFIX = '__holdout'
SEGMENTS_FILE = 'holdout_segments.json'   # in a visual dataset directory: which pairs of it may be used


def settings(config):
    """(period_s, length_s, margin_s) of an S2 config's holdout section, or None when disabled."""
    holdout = (config or {}).get('holdout') or {}
    if not holdout.get('enabled', False):
        return None
    period, length, margin = (float(holdout['period_s']), float(holdout['length_s']),
                              float(holdout.get('margin_s', 1.0)))
    if not 0.0 < length < period or margin < 0.0:
        raise ValueError('holdout requires 0 < length_s < period_s and margin_s >= 0')
    return period, length, margin


def _stamp(path):
    return float(Path(path).stem)


def windows(keypoints, period_s, length_s):
    """[(start, end)] validation segments of the sequence whose keyframe files are in `keypoints`."""
    times = sorted(_stamp(p) for p in Path(keypoints).glob('*.txt'))
    if not times:
        raise ValueError(f'holdout needs keyframes: {keypoints}')
    out, k = [], 0
    while times[0] + k * period_s + period_s - length_s < times[-1]:
        start = times[0] + k * period_s + period_s - length_s
        out.append((start, start + length_s))
        k += 1
    return out


def inside(t, segments, margin=0.0):
    """t in a segment, or in the `margin` seconds after its end (input windows reaching into it)."""
    return any(a <= t <= b + margin for a, b in segments)


def split_files(keypoints, segments, keep_inside, margin=0.0):
    """Keyframe files of the training (keep_inside False) or validation (True) part."""
    files = sorted(Path(keypoints).glob('*.txt'))
    if keep_inside:
        return [f for f in files if inside(_stamp(f), segments)]
    return [f for f in files if not inside(_stamp(f), segments, margin)]


def prune(keypoints, keep):
    """Removes the keyframe files of a generated keypoint directory that are not in `keep` (names)."""
    names = {Path(k).name for k in keep}
    for f in Path(keypoints).glob('*.txt'):
        if f.name not in names:
            f.unlink()


def write_segments(directory, segments, keep_inside, margin=0.0):
    """Marks a visual dataset directory as the training (outside) or validation (inside) part."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / SEGMENTS_FILE).write_text(json.dumps(
        dict(segments=[list(s) for s in segments], keep_inside=bool(keep_inside), margin_s=float(margin))))


def pair_filter(directory):
    """Predicate (t0, t1) -> bool for the keyframe pairs of a visual dataset directory (always true without marks).
    Training part: the interval and the margin after it stay clear of every segment, so no pair spans one.
    Validation part: the interval lies within one segment."""
    path = Path(directory) / SEGMENTS_FILE
    if not path.is_file():
        return lambda t0, t1: True
    marks = json.loads(path.read_text())
    segments, margin = [tuple(s) for s in marks['segments']], float(marks['margin_s'])
    if marks['keep_inside']:
        return lambda t0, t1: any(a <= t0 and t1 <= b for a, b in segments)
    return lambda t0, t1: not any(t0 <= b + margin and t1 >= a for a, b in segments)


def trajectory(source, segments, output):
    """Fixed-BA trajectory rows (TUM) inside the validation segments."""
    rows = np.loadtxt(source, ndmin=2)
    rows = rows[[inside(r[0], segments) for r in rows]]
    rows = rows[np.argsort(rows[:, 0], kind='stable')]
    rows = rows[np.r_[True, np.diff(rows[:, 0]) > 0]]   # the validation needs strictly increasing times
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, rows, fmt='%.9f')
    return str(output)


def runs(times, segments):
    """Index arrays of the `times` (sorted) inside each segment, segments with fewer than 3 samples left out."""
    times = np.asarray(times)
    out = [np.nonzero((times >= a) & (times <= b))[0] for a, b in segments]
    return [i for i in out if len(i) >= 3]


def usable(path, segments):
    """The held-out trajectory has a segment with at least 3 poses (what the segmented evaluation needs)."""
    rows = np.loadtxt(path, ndmin=2)
    return len(rows) > 0 and bool(runs(rows[:, 0], segments))
