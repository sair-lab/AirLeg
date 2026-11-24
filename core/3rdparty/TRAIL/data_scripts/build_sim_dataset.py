#!/usr/bin/env python3
"""Organise an IsaacLab simulation collection into the S1 training layout (--robot anymal | go2).

Sources (<source>/<subset>/<group>/sequence_*, collected by the same IsaacLab driver):
  anymal  quadruped/sim/Anymal/Anymal_orin, 200 Hz -> stride 2 -> 100 Hz (Anymal_reshape_100hz)
  go2     quadruped/sim/Go2 (Flat, Gap, Hurdle, Parkour, Slope, Step, gaitways), 250 Hz -> stride 2 -> 125 Hz
          (Go2_reshape_125hz); the Go2 joint columns already carry the Unitree names (no renaming)

Same conventions as the other S1 dataset builders (split base / IMU frames, common/frames.py): <out>/<split>/data/sequence_*/
with imu / joints / ground_truth / commands decimated by --stride (2: 200 -> 100 Hz; every stride-th row from --trim-s
on, original sample and time kept):
  imu.csv           as recorded: inverted ANYmal IMU frame, specific force at the IMU point
  ground_truth.csv  base_* (base link origin, base frame; S1 labels base_lin_vel_body / base_ang_vel_body) and
                    imu_* (IMU point, IMU frame) columns as recorded
  joints.csv        columns renamed to the Unitree names the loader reads: {RF,LF,RH,LH}_{HAA,HFE,KFE}_* ->
                    {FR,FL,RR,RL}_{hip,thigh,calf}_joint_* (the column order is unchanged and already FR FL RR RL)
  contacts.csv      foot pressure estimated from the decimated joint torques with the Jacobian method and the pressure
                    settings of robots/anymal/leg_odometry.yaml (robot:) (normal_negative, closed-form kinematics of the robot model)
  contacts_sim.csv  decimated simulator normal foot force (reference only, not for training)
  metadata.json     copied
plus the root manifest.json (rates, IMU extrinsic, velocity reference point, trim, pressure settings, splits, exclusions).

--trim-s drops the first seconds of every sequence (the spawn impact: the accelerometer jumps by +-10 m/s^2 in the
first 0.1 s). Selection: every sequence except those whose base tilt exceeds --max-tilt-deg (falls / flips); a
stratum is the group directory plus the gait when the metadata has one; split 90 / 5 / 5 inside every stratum.
Resumable: finished sequences are skipped, each one is written to a temporary directory and renamed when complete; the
selection is fixed in <output>/selection.json on the first run.

The IMU extrinsic is taken from the source metadata (noise.imu_frame_offset_pos_in_base / _quat_wxyz_in_base); every
sequence must carry the same value and it must match the robot model (imu_extrinsic of robots/anymal/leg_odometry.yaml (robot:)).

usage: build_sim_dataset.py --robot anymal|go2 [--source DIR] [--output DIR] [--stride 2] [--trim-s 0.2] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TRAIL = Path(__file__).resolve().parents[1]
if str(TRAIL) not in sys.path:
    sys.path.insert(0, str(TRAIL))

from common.frames import ImuExtrinsic, imu_extrinsic  # noqa: E402
from common.robot import pressure_settings, robot_model  # noqa: E402
from data_scripts.estimate_contacts.core import (  # noqa: E402
    ClosedFormFootForceEstimator, estimate_contacts_from_joint_columns, read_joint_csv, write_contacts_csv)

SIM = Path('/path/to/isaaclab_sim')
DEFAULTS = {   # robot: (source, output, source rate Hz)
    'anymal': (SIM / 'Anymal' / 'Anymal_orin', SIM / 'Anymal' / 'Anymal_reshape_100hz', 200.0),
    'go2': (SIM / 'Go2', SIM / 'Go2_reshape_125hz', 250.0),
}
ROBOT = 'anymal'   # set by main (--robot) before the worker pool starts
SPLITS = (('train', 0.90), ('val', 0.05), ('test', 0.05))
DECIMATED = ('imu.csv', 'joints.csv', 'ground_truth.csv', 'commands.csv')
PRESSURE = None   # robots/<robot>/leg_odometry.yaml (robot:) pressure (the same settings as the recorder), set by main
LEGS = {'RF': 'FR', 'LF': 'FL', 'RH': 'RR', 'LH': 'RL'}
JOINTS = {'HAA': 'hip', 'HFE': 'thigh', 'KFE': 'calf'}


def joint_header(header: str) -> str:
    """ANYmal joint columns -> Unitree names ({leg}_{joint}_joint_{pos,vel,torque}); other columns unchanged."""
    out = []
    for name in header.split(','):
        parts = name.split('_')
        if len(parts) == 3 and parts[0] in LEGS and parts[1] in JOINTS:
            name = f'{LEGS[parts[0]]}_{JOINTS[parts[1]]}_joint_{parts[2]}'
        out.append(name)
    return ','.join(out)


def scan(seq: str):
    """(sequence, gait, level, IMU extrinsic of the metadata, max base tilt in degrees)."""
    p = Path(seq)
    meta = json.loads((p / 'metadata.json').read_text())
    noise = meta.get('noise', {})
    with open(p / 'ground_truth.csv') as f:
        names = f.readline().strip().split(',')
    a = np.loadtxt(p / 'ground_truth.csv', delimiter=',', skiprows=1, ndmin=2,
                   usecols=[names.index('base_roll'), names.index('base_pitch')])
    tilt = float(np.degrees(np.arccos(np.clip(np.cos(a[:, 0]) * np.cos(a[:, 1]), -1, 1))).max())
    ext = (tuple(noise.get('imu_frame_offset_pos_in_base') or ()),
           tuple(noise.get('imu_frame_offset_quat_wxyz_in_base') or ()))
    return seq, meta.get('gait'), meta.get('terrain_level'), ext, tilt


def select(source: Path, seed: int, max_tilt: float, workers: int):
    """Selected sequences with their split (shuffled, numbered), the excluded ones and the common IMU extrinsic."""
    seqs = sorted(str(p) for p in source.glob('*/*/sequence_*') if p.is_dir())
    with Pool(workers) as pool:
        info = pool.map(scan, seqs, chunksize=32)
    extrinsics = {i[3] for i in info}
    if len(extrinsics) != 1:
        raise ValueError(f'sequences carry different IMU extrinsics: {sorted(extrinsics)[:4]}')
    groups, excluded = defaultdict(list), []
    for seq, gait, level, _, tilt in info:
        p = Path(seq)
        subset, group = p.parent.parent.name, p.parent.name
        e = dict(source_path=seq, subset=subset, stratum=f'{group}_{gait}' if gait else group, gait=gait, level=level,
                 max_tilt_deg=round(tilt, 2))
        (excluded if tilt > max_tilt else groups[(subset, e['stratum'])]).append(e)
    rng = random.Random(seed)
    entries = []
    for key in sorted(groups):
        chosen = sorted(groups[key], key=lambda e: e['source_path'])
        rng.shuffle(chosen)
        n_val, n_test = round(len(chosen) * SPLITS[1][1]), round(len(chosen) * SPLITS[2][1])
        for i, e in enumerate(chosen):
            e['split'] = 'val' if i < n_val else 'test' if i < n_val + n_test else 'train'
            entries.append(e)
    rng.shuffle(entries)
    for idx, e in enumerate(entries):
        e['sequence_id'] = f"sequence_{idx:06d}_{ROBOT}_{e['subset']}_{e['stratum']}_{Path(e['source_path']).name}"
    return entries, excluded, [list(x) for x in next(iter(extrinsics))]


def source_extrinsic(ext) -> ImuExtrinsic:
    """IMU extrinsic of the simulation (metadata), checked against the robot model."""
    xyz, (w, x, y, z) = ext
    sim = ImuExtrinsic.from_dict({'xyz': list(xyz), 'quat_xyzw': [x, y, z, w]})
    model = imu_extrinsic(ROBOT)
    if not (np.allclose(sim.r_BI, model.r_BI, atol=1e-6) and np.allclose(sim.R_BI, model.R_BI, atol=1e-6)):
        raise ValueError(f'simulation IMU extrinsic {sim.to_dict()} differs from the robot model {model.to_dict()}')
    return sim


_ESTIMATOR = None


def _init(model: str):
    global _ESTIMATOR
    _ESTIMATOR = ClosedFormFootForceEstimator(Path(model), PRESSURE['pressure_mode'])


def build_one(job):
    entry, output, stride, trim_s = job
    final = Path(output) / entry['split'] / 'data' / entry['sequence_id']
    if final.exists():
        return entry['sequence_id'], json.loads((final / '.rows.json').read_text())
    tmp = final.parent / ('.tmp_' + final.name)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = Path(entry['source_path'])
    rows, first = {}, None
    for name, target in [(n, n) for n in DECIMATED] + [('contacts.csv', 'contacts_sim.csv')]:
        lines = (src / name).read_text().splitlines()
        if first is None:                                  # first data row at or after trim_s (same rows in every file)
            times = [float(line.split(',', 2)[1]) for line in lines[1:]]
            first = next(i for i, t in enumerate(times) if t >= trim_s - 1e-9)
        header = joint_header(lines[0]) if name == 'joints.csv' else lines[0]
        kept = [header] + lines[1 + first::stride]         # header + data rows first, first + stride, ... (text as is)
        (tmp / target).write_text('\n'.join(kept) + '\n')
        rows[target] = len(kept) - 1
    estimate = estimate_contacts_from_joint_columns(
        read_joint_csv(tmp / 'joints.csv'), estimator=_ESTIMATOR, method=PRESSURE['method'],
        pressure_mode=PRESSURE['pressure_mode'], normalize_sum=PRESSURE['normalize_sum'],
        damping=PRESSURE['damping'], smooth_alpha=PRESSURE['smooth_alpha'], pressure_scale=PRESSURE['pressure_scale'])
    write_contacts_csv(tmp / 'contacts.csv', estimate)
    rows['contacts.csv'] = len(estimate.sample)
    if len(set(rows.values())) != 1:
        raise ValueError(f'row counts differ in {src}: {rows}')
    shutil.copy2(src / 'metadata.json', tmp / 'metadata.json')
    (tmp / '.rows.json').write_text(json.dumps(rows))
    tmp.rename(final)
    return entry['sequence_id'], rows


def main():
    global ROBOT, PRESSURE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', choices=sorted(DEFAULTS), default='anymal')
    parser.add_argument('--source', type=Path, help='default per robot (DEFAULTS)')
    parser.add_argument('--output', type=Path, help='default per robot (DEFAULTS)')
    parser.add_argument('--stride', type=int, default=2, help='decimation of the source (2: 200 -> 100 Hz, 250 -> 125 Hz)')
    parser.add_argument('--source-hz', type=float, help='default per robot (DEFAULTS)')
    parser.add_argument('--trim-s', type=float, default=0.2, help='drop the first seconds (spawn impact)')
    parser.add_argument('--max-tilt-deg', type=float, default=60.0)
    parser.add_argument('--model', type=Path, help='default: the robot model')
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    ROBOT = args.robot
    PRESSURE = pressure_settings(ROBOT)
    source, output, source_hz = DEFAULTS[ROBOT]
    args.source, args.output = args.source or source, args.output or output
    args.source_hz, args.model = args.source_hz or source_hz, args.model or robot_model(ROBOT)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = args.output / 'selection.json'      # fixed on the first run, so a resumed run builds the same set
    if selection.exists():
        sel = json.loads(selection.read_text())
    else:
        entries, excluded, ext = select(args.source, args.seed, args.max_tilt_deg, args.workers)
        sel = dict(entries=entries, excluded=excluded, imu_extrinsic_metadata=ext)
        selection.write_text(json.dumps(sel, indent=1))
    entries = sel['entries']
    extrinsic = source_extrinsic(sel['imu_extrinsic_metadata'])
    print(f"[{ROBOT}] {len(entries)} sequences, {len(sel['excluded'])} excluded (tilt > {args.max_tilt_deg} deg); "
          f"IMU extrinsic {extrinsic.to_dict()}", flush=True)
    for split, _ in SPLITS:
        (args.output / split / 'data').mkdir(parents=True, exist_ok=True)
    jobs = [(e, str(args.output), args.stride, args.trim_s) for e in entries]
    rows = {}
    with Pool(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        for i, (sid, r) in enumerate(pool.imap_unordered(build_one, jobs, chunksize=8), 1):
            rows[sid] = r
            if i % 1000 == 0 or i == len(jobs):
                print(f'[{ROBOT}] {i}/{len(jobs)}', flush=True)
    manifest = dict(
        data_root=str(args.source), output_root=str(args.output), target_hz=args.source_hz / args.stride,
        source_hz=args.source_hz, stride=args.stride, trim_s=args.trim_s, seed=args.seed, robot=ROBOT,
        max_tilt_deg=args.max_tilt_deg, imu_extrinsic=extrinsic.to_dict(), velocity_reference='base',
        frames='imu.csv in the IMU frame; ground_truth base_* = base link origin / base frame, imu_* = IMU point / '
               'IMU frame; S1 labels base_lin_vel_body, base_ang_vel_body',
        joint_columns='Unitree names (ANYmal renamed: RF->FR, LF->FL, RH->RR, LH->RL; HAA->hip, HFE->thigh, KFE->calf), order FR FL RR RL',
        pressure=dict(PRESSURE, source='joints.csv torques after decimation', kinematics='closed form',
                      model=str(args.model), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest()),
        excluded=sel['excluded'], splits={split: [] for split, _ in SPLITS})
    for e in entries:
        manifest['splits'][e['split']].append(dict(e, dataset=args.source.name, stride=args.stride,
                                                    files=rows[e['sequence_id']]))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[{ROBOT}] done:', {s: len(v) for s, v in manifest['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()
