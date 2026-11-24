"""Stereo disparity-offset self-calibration of the S2 bags from images only (prepare.stereo_selfcal).

A calibration error in the right camera's yaw adds a constant offset delta to every rectified disparity: measured
d = f b / Z + delta. Unlike a baseline (pure scale) error it compresses far points more than near ones, so the stereo
depths of one scene seen from two places disagree with any rigid motion. For frame pairs ~1-2 s apart, every pair's
relative pose, every matched point (frame-1 coordinates) and one global delta are estimated jointly by robust
Gauss-Newton with a Schur complement from the stereo observations (u, v, d) of both frames. No ground truth and no leg
odometry. delta > 0: measured disparities too large, depths too short, visual scale short (about delta / d). A baseline
or focal length error scales every depth alike, stays a rigid motion and is not observable here.

calibrate() runs it on every bag before the S2 preparation. Bags with the same camera config recorded on the same UTC
day form a group (one rig calibration state). A bag counts only with at least min_pairs frame pairs, a median
disparity <= max_median_disparity_px (near points barely see delta; other model errors would be absorbed) and a time-block
spread <= max_block_std_deg. A group is corrected when its valid bags
agree within group_tolerance_deg and their median |delta| (as yaw, delta / f) reaches apply_min_deg: cam1 is rotated
about its y axis, re-estimated on the group's best bags and refined once with the measured gain (about 0.75-0.95 deg
of estimate per deg of rotation).

usage (one bag): python -m common.stereo_selfcal --bag <ros2 bag dir> --camera <camera yaml> [--out result.json]
"""
import argparse
import ctypes
import hashlib
import json
import re
import sqlite3
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

DEFAULTS = dict(enabled=False, anchors=220, offsets=[1.0, 2.0], min_translation_m=0.3, max_translation_m=3.0,
                blocks=5, workers=8, min_pairs=80, max_median_disparity_px=30.0, max_block_std_deg=0.03,
                group_tolerance_deg=0.1, apply_min_deg=0.05, initial_gain=0.9, refine_bags=3)

_zstd = None


def parse_image(b, compressed=False):
    """sensor_msgs/Image or CompressedImage CDR (rosbag2 per-message zstd allowed) -> (stamp ns, grey image)."""
    global _zstd
    if b[:4] == b'(\xb5/\xfd':
        if _zstd is None:
            _zstd = ctypes.CDLL('libzstd.so.1')
            _zstd.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
            _zstd.ZSTD_decompress.restype = ctypes.c_size_t
        n = _zstd.ZSTD_getFrameContentSize(b, len(b))
        out = ctypes.create_string_buffer(n)
        if _zstd.ZSTD_decompress(out, n, b, len(b)) != n:
            raise ValueError('zstd decompression failed')
        b = out.raw
    o = 4
    sec, nsec = struct.unpack_from('<iI', b, o); o += 8
    n = struct.unpack_from('<I', b, o)[0]; o += 4 + n; o = (o + 3) & ~3
    if compressed:   # string format, uint8[] data
        n = struct.unpack_from('<I', b, o)[0]; o += 4 + n; o = (o + 3) & ~3
        n = struct.unpack_from('<I', b, o)[0]; o += 4
        img = cv2.imdecode(np.frombuffer(b[o:o + n], np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('cannot decode compressed image')
        return sec * 1_000_000_000 + nsec, img
    h, w = struct.unpack_from('<II', b, o); o += 8
    n = struct.unpack_from('<I', b, o)[0]; encoding = b[o + 4:o + 3 + n].decode(); o += 4 + n; o += 1; o = (o + 3) & ~3
    step = struct.unpack_from('<I', b, o)[0]; o += 4
    n = struct.unpack_from('<I', b, o)[0]; o += 4
    img = np.frombuffer(b[o:o + n], np.uint8).reshape(h, step)
    if encoding in ('mono8', '8UC1'):
        img = img[:, :w]
    elif encoding in ('bgr8', 'rgb8'):
        img = cv2.cvtColor(img[:, :3 * w].reshape(h, w, 3), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f'unsupported image encoding {encoding}')
    return sec * 10 ** 9 + nsec, img


def read_camera(path):
    return yaml.safe_load(Path(path).read_text().replace('%YAML:1.0', ''))


class Rectifier:
    """Rectification of an AirSLAM camera yaml (distortion_type 0 none, 1 radtan, 2 equidistant)."""

    def __init__(self, path):
        cfg = read_camera(path)
        self.size = (int(cfg['image_width']), int(cfg['image_height']))
        self.topics = cfg['left_image_topic'], cfg['right_image_topic']
        # as the VO node (visual_odometry_node.cpp ImageInput): compressed topics, resize to the image size
        message = cfg.get('image_message') or ''
        self.compressed = tuple((message or ('compressed' if t.endswith('/compressed') else 'image')) == 'compressed'
                                for t in self.topics)
        kind = int(cfg.get('distortion_type', 1))

        def cam(name):
            i = cfg[name]['intrinsics']
            K = np.array([[i[0], 0, i[2]], [0, i[1], i[3]], [0, 0, 1.0]])
            d = np.array(cfg[name].get('distortion_coeffs', [0] * 5), float)
            return K, (d[:4].reshape(4, 1) if kind == 2 else (d[:5] if kind == 1 else np.zeros(5)))

        K0, D0 = cam('cam0')
        K1, D1 = cam('cam1')
        T10 = np.linalg.inv(np.array(cfg['cam1']['T'], float))   # T: pose of cam1 in cam0
        if kind == 2:
            R0, R1, P0, P1, _ = cv2.fisheye.stereoRectify(K0, D0, K1, D1, self.size, T10[:3, :3], T10[:3, 3],
                                                           flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=self.size,
                                                           balance=0.0, fov_scale=1.0)
            init = cv2.fisheye.initUndistortRectifyMap
        else:
            R0, R1, P0, P1, *_ = cv2.stereoRectify(K0, D0, K1, D1, self.size, T10[:3, :3], T10[:3, 3],
                                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
            init = cv2.initUndistortRectifyMap
        self.m0 = init(K0, D0, R0, P0, self.size, cv2.CV_32FC1)
        self.m1 = init(K1, D1, R1, P1, self.size, cv2.CV_32FC1)
        self.f, self.cx, self.cy, self.fb = P0[0, 0], P0[0, 2], P0[1, 2], abs(P1[0, 3])

    def __call__(self, left, right):
        return (cv2.remap(left, self.m0[0], self.m0[1], cv2.INTER_LINEAR),
                cv2.remap(right, self.m1[0], self.m1[1], cv2.INTER_LINEAR))


SIFT = cv2.SIFT_create(2000)
BF = cv2.BFMatcher(cv2.NORM_L2)


def stereo_features(a, b, max_dy=2.5, min_d=1.0, max_d=150.0):
    """Left keypoints with a right match on (about) the same row: (uv N x 2, d N, descriptors N x 128)."""
    k0, d0 = SIFT.detectAndCompute(a, None)
    k1, d1 = SIFT.detectAndCompute(b, None)
    if d0 is None or d1 is None or len(d0) < 10 or len(d1) < 10:
        return None
    uv, disp, desc = [], [], []
    for p in BF.knnMatch(d0, d1, k=2):
        if len(p) < 2 or p[0].distance > 0.8 * p[1].distance:
            continue
        (u0, v0), (u1, v1) = k0[p[0].queryIdx].pt, k1[p[0].trainIdx].pt
        if abs(v1 - v0) < max_dy and min_d < u0 - u1 < max_d:
            uv.append((u0, v0)); disp.append(u0 - u1); desc.append(d0[p[0].queryIdx])
    if len(uv) < 30:
        return None
    return np.array(uv), np.array(disp), np.array(desc, np.float32)


def camera_image(image, size):
    """A decoded image at the camera yaml size (INTER_AREA resize, as the VO node)."""
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    return image


def read_raw(bag, rect, anchors, offsets):
    """Raw stereo images at `anchors` evenly spread times and at each time + offset (s): [(left, right) or None]."""
    topics = rect.topics
    files = sorted(Path(bag).glob('*.db3'))
    if not files:
        raise ValueError(f'only sqlite3 rosbag2 storage is supported: {bag}')
    db = sqlite3.connect(str(files[0]))
    ids = {n: i for i, n in db.execute('select id, name from topics')}
    L, R = ids[topics[0]], ids[topics[1]]
    stamps = np.array([r[0] for r in db.execute('select timestamp from messages where topic_id=? order by timestamp', (L,))])
    picks = stamps[np.linspace(len(stamps) * 0.02, len(stamps) * 0.98 - 1, anchors).astype(int)]
    images = {}

    def frame(ts):
        i = int(np.clip(np.searchsorted(stamps, ts), 0, len(stamps) - 1))
        ts = int(stamps[i])
        if ts not in images:
            sl, left = parse_image(db.execute('select data from messages where topic_id=? and timestamp=?', (L, ts)).fetchone()[0],
                                   rect.compressed[0])
            left = camera_image(left, rect.size)
            row = db.execute('select data from messages where topic_id=? and timestamp between ? and ? order by '
                             'abs(timestamp-?) limit 1', (R, ts - 30_000_000, ts + 30_000_000, ts)).fetchone()
            images[ts] = None
            if row is not None:
                sr, right = parse_image(row[0], rect.compressed[1])
                right = camera_image(right, rect.size)
                if abs(sr - sl) < 1_000_000:   # same exposure
                    images[ts] = (left, right)
        return ts

    pairs = []
    for ts in picks:
        t0 = frame(int(ts))
        pairs.extend((t0, frame(int(ts + off * 1e9))) for off in offsets)
    return images, pairs


def project(X, rect, delta):
    z = X[:, 2]
    return np.stack([rect.f * X[:, 0] / z + rect.cx, rect.f * X[:, 1] / z + rect.cy, rect.fb / z + delta], 1)


def proj_jac(X, rect):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    J = np.zeros((len(X), 3, 3))
    J[:, 0, 0] = rect.f / z; J[:, 0, 2] = -rect.f * x / z ** 2
    J[:, 1, 1] = rect.f / z; J[:, 1, 2] = -rect.f * y / z ** 2
    J[:, 2, 2] = -rect.fb / z ** 2
    return J


def skew(v):
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return S - S.transpose(0, 2, 1)


def initial_pair(f0, f1, rect, min_t, max_t):
    """Matched stereo observations of a frame pair and the PnP pose (frame-1 -> frame-2), or None."""
    if f0 is None or f1 is None:
        return None
    uv0, d0, s0 = f0
    uv1, d1, s1 = f1
    idx = [(p[0].queryIdx, p[0].trainIdx) for p in BF.knnMatch(s0, s1, k=2)
           if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(idx) < 30:
        return None
    a, b = np.array(idx).T
    o1 = np.column_stack([uv0[a], d0[a]])
    o2 = np.column_stack([uv1[b], d1[b]])
    z = rect.fb / o1[:, 2]
    X = np.column_stack([(o1[:, 0] - rect.cx) * z / rect.f, (o1[:, 1] - rect.cy) * z / rect.f, z])
    K = np.array([[rect.f, 0, rect.cx], [0, rect.f, rect.cy], [0, 0, 1.0]])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, o2[:, :2], K, None, reprojectionError=2.0, iterationsCount=300,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 30:
        return None
    t = tvec.ravel()
    if not min_t <= np.linalg.norm(t) <= max_t:
        return None
    inl = inl.ravel()
    return dict(R=cv2.Rodrigues(rvec)[0], t=t, X=X[inl], o1=o1[inl], o2=o2[inl])


def solve(pairs, rect, delta=0.0, iterations=15, sigma=0.7, huber=2.0):
    """Robust Gauss-Newton over all pairs (poses, points) and the global delta; returns (delta, sigma_delta, info)."""
    lam = 1e-3
    for it in range(iterations):
        H_dd, g_d, chi2, nres, steps = 0.0, 0.0, 0.0, 0, []
        for p in pairs:
            X, R, t = p['X'], p['R'], p['t']
            Xc = X @ R.T + t
            r1 = p['o1'] - project(X, rect, delta)
            r2 = p['o2'] - project(Xc, rect, delta)
            # per observation Huber weights on the 3-vector residual (pixels)
            n1, n2 = np.linalg.norm(r1, axis=1) / sigma, np.linalg.norm(r2, axis=1) / sigma
            w1, w2 = np.minimum(1, huber / np.maximum(n1, 1e-9)), np.minimum(1, huber / np.maximum(n2, 1e-9))
            chi2 += float((w1 * n1 ** 2).sum() + (w2 * n2 ** 2).sum()); nres += 6 * len(X)
            P1, P2 = proj_jac(X, rect), proj_jac(Xc, rect)
            J1X = -P1
            J2X = -P2 @ R
            J2pose = -P2 @ np.concatenate([-skew(Xc), np.broadcast_to(np.eye(3), (len(X), 3, 3))], 2)  # (N,3,6)
            Jd = np.array([0.0, 0.0, -1.0])
            W1, W2 = w1[:, None, None], w2[:, None, None]
            Hpp = (W1 * J1X.transpose(0, 2, 1) @ J1X + W2 * J2X.transpose(0, 2, 1) @ J2X) + lam * np.eye(3)
            J2c = np.concatenate([J2pose, np.broadcast_to(Jd[None, :, None], (len(X), 3, 1))], 2)       # (N,3,7)
            J1c = np.concatenate([np.zeros((len(X), 3, 6)), np.broadcast_to(Jd[None, :, None], (len(X), 3, 1))], 2)
            Hpc = W2 * J2X.transpose(0, 2, 1) @ J2c + W1 * J1X.transpose(0, 2, 1) @ J1c                  # (N,3,7)
            Hcc = (W2 * J2c.transpose(0, 2, 1) @ J2c + W1 * J1c.transpose(0, 2, 1) @ J1c).sum(0)
            gp = -(W1 * J1X.transpose(0, 2, 1) @ r1[:, :, None] + W2 * J2X.transpose(0, 2, 1) @ r2[:, :, None])[:, :, 0]
            gc = -(W2 * J2c.transpose(0, 2, 1) @ r2[:, :, None] + W1 * J1c.transpose(0, 2, 1) @ r1[:, :, None]).sum(0)[:, 0]
            Hinv = np.linalg.inv(Hpp)
            S = Hcc - (Hpc.transpose(0, 2, 1) @ Hinv @ Hpc).sum(0)
            s = gc - (Hpc.transpose(0, 2, 1) @ Hinv @ gp[:, :, None]).sum(0)[:, 0]
            A = S[:6, :6] + lam * np.eye(6)
            B, c, a, b = S[:6, 6], S[6, 6], s[:6], s[6]
            Ainv = np.linalg.inv(A)
            H_dd += c - B @ Ainv @ B
            g_d += b - B @ Ainv @ a
            steps.append((Ainv, a, B, Hinv, gp, Hpc))
        dd = g_d / H_dd
        for p, (Ainv, a, B, Hinv, gp, Hpc) in zip(pairs, steps):
            dpose = Ainv @ (a - B * dd)
            dc = np.concatenate([dpose, [dd]])
            dX = (Hinv @ (gp - (Hpc @ dc))[:, :, None])[:, :, 0]
            dR = cv2.Rodrigues(dpose[:3])[0]
            p['R'] = dR @ p['R']; p['t'] = dR @ p['t'] + dpose[3:]
            p['X'] = p['X'] + dX
            keep = p['X'][:, 2] > 0.05
            if not keep.all():
                for k in ('X', 'o1', 'o2'):
                    p[k] = p[k][keep]
        delta += dd
        if abs(dd) < 1e-4 and it > 3:
            break
    sigma_delta = float(np.sqrt(chi2 / max(nres - 1, 1) / H_dd)) if H_dd > 0 else float('inf')
    return float(delta), sigma_delta, dict(chi2_per_res=chi2 / max(nres, 1), iterations=it + 1)


def estimate(bag, camera, cfg=None):
    """Disparity offset of one bag with one camera yaml: dict with delta_px, yaw_equiv_deg, pairs, ... or error."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    start = time.time()
    rect = Rectifier(camera)
    images, raw = read_raw(bag, rect, int(cfg['anchors']), cfg['offsets'])
    features = {ts: None if im is None else stereo_features(*rect(*im)) for ts, im in images.items()}
    pairs = [p for p in (initial_pair(features[a], features[b], rect, cfg['min_translation_m'], cfg['max_translation_m'])
                         for a, b in raw) if p is not None]
    result = dict(bag=str(bag), camera=str(camera), pairs=len(pairs))
    if len(pairs) < 10:
        return dict(result, error='too few usable pairs', seconds=round(time.time() - start, 1))
    delta, sd, info = solve(pairs, rect)
    blocks = []
    for chunk in np.array_split(np.arange(len(pairs)), int(cfg['blocks'])):
        sub = [dict(R=pairs[i]['R'].copy(), t=pairs[i]['t'].copy(), X=pairs[i]['X'].copy(), o1=pairs[i]['o1'],
                    o2=pairs[i]['o2']) for i in chunk]
        blocks.append(solve(sub, rect, delta, iterations=8)[0])
    disp = np.concatenate([p['o1'][:, 2] for p in pairs])
    return dict(result, points=int(sum(len(p['X']) for p in pairs)), delta_px=delta, sigma_px=sd,
                block_delta_px=blocks, median_disparity_px=float(np.median(disp)), f=rect.f, fb=rect.fb,
                yaw_equiv_deg=float(np.degrees(delta / rect.f)),
                block_std_deg=float(np.degrees(np.std(blocks, ddof=1) / rect.f)),
                seconds=round(time.time() - start, 1), **info)


def rotated_camera(source, yaw_deg, out, comment):
    """Camera yaml with cam1 rotated about its y axis (T rotation <- R Rot_y); distortion_type 0 becomes 1 with zero
    coefficients, because AirSLAM skips the rectification (and so the cam1 rotation) of type 0 inputs."""
    text = Path(source).read_text()
    cfg = read_camera(source)
    T = np.array(cfg['cam1']['T'], float)
    T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec([0.0, np.radians(yaw_deg), 0.0]).as_matrix()
    head, tail = text.split('\ncam1:', 1)
    rows = re.findall(r'\n(\s*- \[[^\]]*\])', tail)[:3]
    for old, r in zip(rows, T[:3]):
        tail = tail.replace(old, f"{old[:old.index('-')]}- [{r[0]:.12f}, {r[1]:.12f}, {r[2]:.12f}, {r[3]:.12f}]", 1)
    text = head + '\ncam1:' + tail
    if int(cfg.get('distortion_type', 1)) == 0:
        text = re.sub(r'(?m)^distortion_type:\s*0\b.*$',
                      'distortion_type: 1  # radial-tangential with zero coefficients: rectify with the cam1 rotation', text)
    first, rest = text.split('\n', 1)
    Path(out).write_text(first + '\n' + comment + '\n' + rest)


def _estimate_job(job):
    bag, camera, cfg, out = job
    out = Path(out)
    if out.is_file():
        return json.loads(out.read_text())
    try:
        result = estimate(bag, camera, cfg)
    except Exception as error:   # one unreadable bag must not stop the preparation
        result = dict(bag=str(bag), camera=str(camera), error=f'{type(error).__name__}: {error}')
    out.write_text(json.dumps(result, indent=1))
    return result


def _valid(r, cfg):
    return ('error' not in r and r['pairs'] >= cfg['min_pairs'] and r['median_disparity_px'] <= cfg['max_median_disparity_px']
            and r['block_std_deg'] <= cfg['max_block_std_deg'])


def recording_day(bag):
    meta = yaml.safe_load((Path(bag) / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    ns = int(meta['starting_time']['nanoseconds_since_epoch'])
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime('%Y-%m-%d')


def calibrate(bags, cameras, directory, cfg):
    """bags: name -> bag dir, cameras: name -> camera yaml. Returns name -> corrected camera yaml for the corrected
    groups; writes the estimates, decisions and cameras to `directory` (report.json). Estimates are cached by file."""
    cfg = dict(DEFAULTS, **cfg)
    directory = Path(directory)
    (directory / 'estimates').mkdir(parents=True, exist_ok=True)

    def key(name, camera):   # bag and camera content (estimate settings are fixed per directory)
        return f"{Path(str(bags[name])).name}__{hashlib.sha256(Path(camera).read_bytes()).hexdigest()[:10]}"

    def run(jobs):
        jobs = [(str(bags[n]), str(c), cfg, str(directory / 'estimates' / f'{key(n, c)}.json')) for n, c in jobs]
        with ProcessPoolExecutor(max_workers=int(cfg['workers'])) as pool:
            return list(pool.map(_estimate_job, jobs))

    names = sorted(bags)
    print(f'[stereo_selfcal] estimating {len(names)} bags', flush=True)
    first = dict(zip(names, run([(n, cameras[n]) for n in names])))
    groups = {}
    for n in names:
        groups.setdefault((str(cameras[n]), recording_day(bags[n])), []).append(n)
    corrected, report = {}, dict(settings=cfg, groups=[])
    for index, ((camera, day), members) in enumerate(sorted(groups.items())):
        valid = [n for n in members if _valid(first[n], cfg)]
        yaws = [first[n]['yaw_equiv_deg'] for n in valid]
        entry = dict(camera=camera, day=day, bags={n: {k: first[n].get(k) for k in (
            'pairs', 'median_disparity_px', 'yaw_equiv_deg', 'block_std_deg', 'error')} | dict(valid=n in valid)
            for n in members})
        if not valid:
            entry.update(decision='no valid estimate')
        elif max(yaws) - min(yaws) > cfg['group_tolerance_deg']:
            entry.update(estimate_deg=float(np.median(yaws)), decision='bags disagree, not corrected')
        elif abs(np.median(yaws)) < cfg['apply_min_deg']:
            entry.update(estimate_deg=float(np.median(yaws)), decision='within calibration accuracy')
        else:
            estimate0 = float(np.median(yaws))
            probe = sorted(valid, key=lambda n: -first[n]['pairs'])[:int(cfg['refine_bags'])]
            out = directory / f'camera_{index:02d}_{day}.yaml'
            history = [(0.0, estimate0)]
            rotation = estimate0 / float(cfg['initial_gain'])
            for step in range(2):   # rotate, re-estimate on the best bags, one secant refinement
                trial = directory / f'camera_{index:02d}_{day}_yaw{rotation:+.4f}.yaml'   # names the cached estimates
                rotated_camera(camera, rotation, trial, f'# stereo_selfcal trial: cam1 yaw {rotation:+.4f} deg')
                residual = float(np.median([r['yaw_equiv_deg'] for r in run([(n, trial) for n in probe])
                                            if 'error' not in r]))
                history.append((rotation, residual))
                if step == 0:
                    gain = (estimate0 - residual) / rotation
                    if not 0.3 < gain < 2.0:
                        break
                    rotation += residual / gain
            best = min(history[1:], key=lambda h: abs(h[1]))
            rotated_camera(camera, best[0], out,
                           f'# stereo_selfcal (S2 prepare, {day}): cam1 rotated by yaw {best[0]:+.4f} deg about its y axis; '
                           f'disparity offset {estimate0:+.4f} deg yaw-equivalent before, {best[1]:+.4f} after '
                           f'(common/stereo_selfcal.py)')
            entry.update(estimate_deg=estimate0, decision='corrected', yaw_deg=best[0], residual_deg=best[1],
                         history=history, corrected_camera=str(out))
            corrected.update({n: out for n in members})
        print(f"[stereo_selfcal] {day} {Path(camera).name}: {len(valid)}/{len(members)} valid, "
              f"{entry.get('estimate_deg', float('nan')):+.3f} deg -> {entry['decision']}"
              + (f" (yaw {entry['yaw_deg']:+.3f}, residual {entry['residual_deg']:+.3f})" if 'yaw_deg' in entry else ''),
              flush=True)
        report['groups'].append(entry)
    (directory / 'report.json').write_text(json.dumps(report, indent=1))
    return corrected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--anchors', type=int, default=DEFAULTS['anchors'])
    ap.add_argument('--out')
    a = ap.parse_args()
    text = json.dumps(estimate(a.bag, a.camera, dict(anchors=a.anchors)), indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
