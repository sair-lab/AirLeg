#!/usr/bin/env python3
"""Convert GrandTour missions (ANYmal D) into AirSLAM2 bags and evaluation files (robots/anymal).

Input (--source): one directory per mission, in either form (detected per mission):
  * the official GrandTour release (ROS1): <source>/[release_]<mission>/<mission>_<name>.bag. Only these 6 of the ~34
    bags of a mission are read (download only them): anymal_state, anymal_imu, alphasense, tf_minimal (required),
    cpt7_ie_tc, ap20_prism_position (ground truth, optional; absent from the official benchmark missions).
    The ROS1 messages are converted to ROS2 (CDR) with rosbags, as rosbags-convert does.
  * per-mission ROS2 bags holding these topics: <source>/<mission>/metadata.yaml.
Output (--output):
  bags/<mission>/   ROS2 bag (sqlite3, no compression) with the recorded messages of the topics below, byte for byte
                    and with their recorded timestamps (no decoding, resampling or renaming), except the repaired
                    front_left images of the missions recorded upside down (left_rotated_180: 2024-11-18 except
                    13-22-14 and 17-31-36): their JPEGs are rotated by 180 deg losslessly in the DCT domain
                    (jpeg_rotate180.c, as jpegtran -rotate 180 -perfect; header, format and camera_info unchanged)
      /anymal/state_estimator/anymal_state          anymal_msgs/AnymalState (400 Hz; the leg odometry reads the joints,
                                                    ros2_packages/anymal_msgs, robot: leg_inputs)
      /anymal/imu                                   sensor_msgs/Imu (400 Hz, IMU frame anymal_imu)
      /boxi/alphasense/front_{left,right}/image_raw/compressed, .../camera_info
                                                    Alphasense front stereo, 1440x1080 grey JPEG (10 Hz); the VO node
                                                    decodes them and resizes them to the camera yaml (720x540)
      /tf_static                                    static transforms
      /boxi/inertial_explorer/tc/{odometry,navsatfix}, /boxi/ap20/prism_position
                                                    ground truth (CPT7 Inertial Explorer tightly coupled solution, AP20
                                                    prism), for evaluation only
  seq/<mission>/robot/   evaluation files (robots/anymal/leg_odometry.yaml (robot:) dataset registry):
      imu.csv         sample,time,gyro_xyz,acc_xyz of /anymal/imu (400 Hz, IMU frame)
      traj.txt        TUM pose of the base in the ENU frame of the CPT7 Inertial Explorer tightly coupled solution
                      (T_enu_base = T_enu_cpt7imu T_cpt7imu_base, /tf_static), only samples with a navsatfix position
                      sigma <= --max-sigma (forest / indoor parts reach metres); absent without CPT7
      traj_sigma.txt  time, 3D position sigma (m) of every CPT7 sample (before the cut)
      prism.txt       time x y z of the AP20 prism in the Leica total station frame (position only), if present
  s2_bags/            symlinks of the train + val missions of robots/anymal/leg_odometry.yaml (robot:) (S2 prepare uses every bag)
Every mission writes conversion.json (message counts per topic, left_rotated_180 check, repaired images). Resumable: finished missions
are skipped; the bag is written to bags/.tmp/<mission> and renamed when complete.

Requires: rosbags (tested 0.11.7), opencv-python-headless, numpy, scipy, pyyaml (any Python; ROS is not needed), and
gcc + libjpeg headers (libjpeg-dev) for the lossless rotation of the upside-down images.

usage: convert_grandtour_to_ros2.py --source DIR --output DIR [--only MISSION ...] [--max-seconds S] [--max-sigma M]
                                    [--workers N]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from rosbags.convert.converter import create_connections_converters
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
STATE, IMU = '/anymal/state_estimator/anymal_state', '/anymal/imu'
LEFT, RIGHT = '/boxi/alphasense/front_left/image_raw/compressed', '/boxi/alphasense/front_right/image_raw/compressed'
LEFT_INFO, RIGHT_INFO = '/boxi/alphasense/front_left/camera_info', '/boxi/alphasense/front_right/camera_info'
CPT7_ODOM, CPT7_FIX = '/boxi/inertial_explorer/tc/odometry', '/boxi/inertial_explorer/tc/navsatfix'
PRISM, TF_STATIC = '/boxi/ap20/prism_position', '/tf_static'
ROTATE_SRC = Path(__file__).resolve().with_name('jpeg_rotate180.c')
ROTATE_LIB = Path.home() / '.cache' / 'airslam' / 'libjpeg_rotate180.so'
BAG_TOPICS = (STATE, IMU, LEFT, RIGHT, LEFT_INFO, RIGHT_INFO, TF_STATIC, CPT7_ODOM, CPT7_FIX, PRISM)
# official release (ROS1): the bags of a mission that hold BAG_TOPICS (tf_model also has a /tf_static, a different
# static tree for the depth cameras: not read); the first four are required
RELEASE_BAGS = ('anymal_state', 'anymal_imu', 'alphasense', 'tf_minimal', 'cpt7_ie_tc', 'ap20_prism_position')


def find_missions(source: Path) -> dict:
    """mission -> list of bag paths: [<dir>] for a ROS2 bag directory, the RELEASE_BAGS files for a ROS1 release."""
    missions = {}
    for d in sorted(p for p in source.iterdir() if p.is_dir()):
        if (d / 'metadata.yaml').exists():
            missions[d.name] = [str(d)]
            continue
        name = d.name.removeprefix('release_')
        bags = [d / f'{name}_{b}.bag' for b in RELEASE_BAGS]
        if bags[0].exists():
            missing = [b.name for b in bags[:4] if not b.exists()]
            if missing:
                raise FileNotFoundError(f'{d}: missing {missing}')
            missions[name] = [str(b) for b in bags if b.exists()]
    return missions


def jpeg_rotate180():
    """The lossless JPEG rotation of jpeg_rotate180.c (compiled on first use): bytes -> bytes."""
    import ctypes
    import subprocess
    if not ROTATE_LIB.exists() or ROTATE_LIB.stat().st_mtime < ROTATE_SRC.stat().st_mtime:
        ROTATE_LIB.parent.mkdir(parents=True, exist_ok=True)
        tmp = ROTATE_LIB.with_suffix(f'.{np.random.randint(1 << 30)}.so')
        subprocess.run(['gcc', '-O2', '-shared', '-fPIC', str(ROTATE_SRC), '-o', str(tmp), '-ljpeg'], check=True)
        tmp.replace(ROTATE_LIB)                             # atomic: parallel workers
    lib = ctypes.CDLL(str(ROTATE_LIB))
    lib.jpeg_rotate180.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                                   ctypes.POINTER(ctypes.c_ulong)]
    lib.jpeg_rotate180_free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

    def rotate(data: bytes) -> bytes:
        out, n = ctypes.POINTER(ctypes.c_ubyte)(), ctypes.c_ulong()
        status = lib.jpeg_rotate180(data, len(data), ctypes.byref(out), ctypes.byref(n))
        if status:
            raise ValueError(f'lossless JPEG rotation failed ({status}: 1 libjpeg error, 2 partial MCU)')
        try:
            return ctypes.string_at(out, n.value)
        finally:
            lib.jpeg_rotate180_free(out)
    return rotate


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def matrix(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
    T[:3, 3] = [translation.x, translation.y, translation.z]
    return T


def static_chain(edges, child, root):
    """T_root_child from /tf_static edges {child: (parent, T_parent_child)}."""
    T = np.eye(4)
    while child != root:
        parent, A = edges[child]
        T = A @ T
        child = parent
    return T


def read_sensors(paths, max_ns):
    """The inputs of the evaluation files: IMU, CPT7, prism, static tf."""
    out = dict(imu_t=[], gyro=[], acc=[], imu_frame=None, odom=[], fix=[], prism=[], tf={})
    with AnyReader([Path(p) for p in paths]) as reader:
        topics = {IMU, CPT7_ODOM, CPT7_FIX, PRISM, TF_STATIC}
        conns = [c for c in reader.connections if c.topic in topics]
        start = None
        for conn, t, raw in reader.messages(connections=conns):
            start = t if start is None else start
            if max_ns and t - start > max_ns:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == IMU:
                out['imu_frame'] = out['imu_frame'] or msg.header.frame_id
                out['imu_t'].append(stamp_ns(msg.header))
                out['gyro'].append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                out['acc'].append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            elif conn.topic == CPT7_ODOM:
                p, o = msg.pose.pose.position, msg.pose.pose.orientation
                out['odom'].append((stamp_ns(msg.header), msg.child_frame_id, p, o))
            elif conn.topic == CPT7_FIX:
                out['fix'].append((stamp_ns(msg.header),
                                   float(np.sqrt(np.trace(np.asarray(msg.position_covariance).reshape(3, 3))))))
            elif conn.topic == PRISM:
                out['prism'].append((stamp_ns(msg.header), msg.point.x, msg.point.y, msg.point.z))
            elif conn.topic == TF_STATIC:
                for tr in msg.transforms:
                    out['tf'][tr.child_frame_id] = (tr.header.frame_id, matrix(tr.transform.translation, tr.transform.rotation))
    out['imu_t'] = np.asarray(out['imu_t'], dtype=np.int64)
    for k in ('gyro', 'acc'):
        out[k] = np.asarray(out[k], dtype=np.float64)
    return out


def write_eval(seq_dir: Path, s, counts):
    seq_dir.mkdir(parents=True, exist_ok=True)
    t = s['imu_t'] * 1e-9
    np.savetxt(seq_dir / 'imu.csv', np.column_stack([np.arange(len(t)), t, s['gyro'], s['acc']]), delimiter=',',
               header='sample,time,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z', comments='', fmt=['%d'] + ['%.9f'] + ['%.9g'] * 6)
    if s['prism']:
        np.savetxt(seq_dir / 'prism.txt', np.array([(p[0] * 1e-9, *p[1:]) for p in s['prism']]), fmt='%.9f %.6f %.6f %.6f',
                   header='time x y z (AP20 prism, Leica total station frame)')
    counts['prism'] = len(s['prism'])
    if not s['odom']:
        counts['cpt7'] = 0
        return
    child = s['odom'][0][1]
    T_child_base = np.linalg.inv(static_chain(s['tf'], child, 'base'))
    fix_t = np.array([f[0] for f in s['fix']], dtype=np.int64)
    fix_s = np.array([f[1] for f in s['fix']])
    rows, sig = [], []
    for ns, frame, p, o in s['odom']:
        if frame != child:
            raise ValueError(f'CPT7 child frame changed: {frame} != {child}')
        k = int(np.clip(np.searchsorted(fix_t, ns), 0, len(fix_t) - 1)) if len(fix_t) else -1
        sigma = fix_s[k] if k >= 0 and abs(fix_t[k] - ns) < 10_000_000 else np.inf
        sig.append((ns * 1e-9, sigma))
        if sigma > counts['max_sigma']:
            continue
        T = matrix(p, o) @ T_child_base
        rows.append((ns * 1e-9, *T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat()))
    np.savetxt(seq_dir / 'traj_sigma.txt', np.array(sig), fmt='%.9f %.4f', header='time position_sigma_m (CPT7 navsatfix)')
    if rows:
        np.savetxt(seq_dir / 'traj.txt', np.array(rows), fmt='%.9f' + ' %.6f' * 3 + ' %.9f' * 4,
                   header=f'TUM T_enu_base (CPT7 IE tc, {child} -> base via /tf_static, sigma <= {counts["max_sigma"]} m)')
    counts['cpt7'] = len(s['odom'])
    counts['cpt7_kept'] = len(rows)
    counts['cpt7_sigma_median'] = float(np.median([x[1] for x in sig]))


def left_rotated_180(paths, pairs=8, spacing_ns=10_000_000_000):
    """Whether the front_left images of a mission are rotated by 180 deg relative to front_right.

    On 2024-11-18 (ARC-1, ARC-3..ARC-6) the recorded front_left JPEGs are upside down while /tf_static and camera_info
    are unchanged; rotated back, they match front_right on the same rows as on every other mission. Counts ORB matches
    with |dy| < 3 px on `pairs` image pairs (one every `spacing_ns`), upright and rotated; returns (rotated, counts)."""
    orb, matcher = cv2.ORB_create(1500), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def same_row(a, b):
        k1, d1 = orb.detectAndCompute(a, None)
        k2, d2 = orb.detectAndCompute(b, None)
        if d1 is None or d2 is None:
            return 0
        m = matcher.match(d1, d2)
        y1 = np.float32([k1[x.queryIdx].pt[1] for x in m]); y2 = np.float32([k2[x.trainIdx].pt[1] for x in m])
        return int(np.sum(np.abs(y1 - y2) < 3))

    upright = rotated = 0
    with AnyReader([Path(p) for p in paths]) as reader:
        conns = [c for c in reader.connections if c.topic in (LEFT, RIGHT)]
        pending, next_ns, done = {}, None, 0
        for c, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, c.msgtype)
            ns = stamp_ns(msg.header)
            if next_ns is None:
                next_ns = ns
            if ns < next_ns:
                continue
            pending.setdefault(ns, {})['L' if c.topic == LEFT else 'R'] = bytes(msg.data)
            if len(pending[ns]) == 2:
                a, b = (cv2.imdecode(np.frombuffer(pending[ns][k], np.uint8), cv2.IMREAD_GRAYSCALE) for k in 'LR')
                upright += same_row(a, b)
                rotated += same_row(cv2.rotate(a, cv2.ROTATE_180), b)
                pending, next_ns, done = {}, ns + spacing_ns, done + 1
                if done >= pairs:
                    break
    return rotated > 3 * max(upright, 1), dict(upright=upright, rotated=rotated, pairs=done)


def convert(job):
    name, source, out, max_seconds, max_sigma = job
    dst = Path(out) / 'bags' / name
    seq_dir = Path(out) / 'seq' / name / 'robot'
    if dst.exists() and (seq_dir.parent / 'conversion.json').exists():
        return name, 'exists'
    src = source   # bag paths of the mission (find_missions)
    max_ns = int(max_seconds * 1e9) if max_seconds else 0
    counts = dict(max_sigma=max_sigma)
    s = read_sensors(src, max_ns)
    if s['imu_frame'] != 'anymal_imu':
        raise ValueError(f'{name}: unexpected IMU frame {s["imu_frame"]}')
    flip_left, flip_check = left_rotated_180(src)
    counts.update(left_rotated_180=bool(flip_left), left_flip_check=flip_check)

    tmp = dst.parent / '.tmp' / name
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    messages = {}
    rotate = jpeg_rotate180() if flip_left else None
    with AnyReader([Path(p) for p in src]) as reader, Writer(tmp, version=8) as writer:
        conns = [c for c in reader.connections if c.topic in BAG_TOPICS]
        missing = sorted(set(BAG_TOPICS[:7]) - {c.topic for c in conns})   # the ground truth topics may be absent
        if missing:
            raise ValueError(f'{name}: missing topics {missing}')
        if reader.is2:   # ROS2 source: the CDR messages are copied as they are
            out_conn = {}
            for c in conns:
                if c.topic not in out_conn:
                    qos = getattr(c.ext, 'offered_qos_profiles', ()) or ()
                    out_conn[c.topic] = writer.add_connection(c.topic, c.msgtype, typestore=reader.typestore,
                                                              offered_qos_profiles=qos)
            connection = lambda c: out_conn[c.topic]
            to_cdr = lambda c, raw: raw
        else:            # ROS1 release: converted to CDR as rosbags-convert does
            connmap, convmap = create_connections_converters(conns, None, reader, writer)
            connection = lambda c: connmap[(c.id, c.owner)]
            to_cdr = lambda c, raw: convmap[c.msgtype](raw)
        start = None
        for c, t, raw in reader.messages(connections=conns):
            start = t if start is None else start
            if max_ns and t - start > max_ns:
                break
            if rotate is not None and c.topic == LEFT:   # upside-down front_left: repaired in the bag
                msg = reader.deserialize(raw, c.msgtype)
                fixed = dataclasses.replace(msg, data=np.frombuffer(rotate(bytes(msg.data)), np.uint8))
                serialize = reader.typestore.serialize_cdr if reader.is2 else reader.typestore.serialize_ros1
                raw = serialize(fixed, c.msgtype)
                counts['left_images_rotated'] = counts.get('left_images_rotated', 0) + 1
            writer.write(connection(c), t, to_cdr(c, raw))
            messages[c.topic] = messages.get(c.topic, 0) + 1
    counts['messages'] = messages
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    write_eval(seq_dir, s, counts)
    (seq_dir.parent / 'conversion.json').write_text(json.dumps(dict(source=src, counts=counts), indent=2))
    return name, counts


def link_s2_bags(out: Path):
    """s2_bags/<mission>/ = the train + val missions of robots/anymal/leg_odometry.yaml (robot:): real directories whose files are
    symlinks (S2 prepare finds bags with Path.rglob('metadata.yaml'), which does not enter symlinked directories)."""
    import yaml
    reg = yaml.safe_load((REPO / 'robots/anymal/leg_odometry.yaml').read_text())['robot']['dataset']['splits']
    for m in reg['train'] + reg['val']:
        target = out / 'bags' / m
        if not target.exists():
            continue
        link_dir = out / 's2_bags' / m
        if link_dir.is_symlink():
            link_dir.unlink()
        link_dir.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            if not (link_dir / f.name).exists():
                (link_dir / f.name).symlink_to(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True,
                        help='GrandTour release root (<mission>/ or release_<mission>/ with ROS1 bags) or ROS2 bags')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*', help='missions to convert (default: every mission of --source)')
    parser.add_argument('--max-seconds', type=float, default=0.0, help='convert only the first S seconds (tests)')
    parser.add_argument('--max-sigma', type=float, default=0.1, help='CPT7 position sigma (m) kept in traj.txt')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    missions = find_missions(args.source)
    unknown = sorted(set(args.only or ()) - set(missions))
    if unknown or not missions:
        raise SystemExit(f'no GrandTour mission found in {args.source}' if not missions else f'not in {args.source}: {unknown}')
    names = args.only or sorted(missions)
    jobs = [(n, missions[n], str(args.output), args.max_seconds, args.max_sigma) for n in names]
    failed = []
    with Pool(min(args.workers, len(jobs))) as pool:
        for name, counts in pool.imap_unordered(_safe, jobs):
            print(f'[grandtour] {name}: {counts}', flush=True)
            if isinstance(counts, str) and counts.startswith('FAILED'):
                failed.append(name)
    if not args.max_seconds:
        link_s2_bags(args.output)
    print(f'[grandtour] done, failed: {failed}', flush=True)


def _safe(job):
    try:
        return convert(job)
    except Exception as e:  # keep the other missions going; the failure is reported
        return job[0], f'FAILED {e!r}'


if __name__ == '__main__':
    sys.exit(main())
