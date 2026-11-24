#!/usr/bin/env python3
"""S2 and S4 orchestration; paths in YAML are relative to the YAML file that defines them (configs with `base:`,
common/yaml_base.py). Default config: S2 the sim_to_real: section of AirSLAM2/robots/<robot>/leg_odometry.yaml, S4 the global_backend: section of AirSLAM2/robots/<robot>/algorithm.yaml."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from common import holdout as ho
from common.alternating import ROOT, link, run, run_alternating, write_json
from common.preparation import prepare_bags
from common.yaml_base import load_section, load_yaml

REPO = ROOT.parents[2]


def resolve_paths(config, base):
    path_keys = {'bag_root', 'cache_root', 'algorithm', 'recorder_config', 'camera_config',
                 'checkpoint', 'output_dir', 'map_root', 'data_root', 'model_dir', 'voc_path',
                 'backend', 'config_file'}
    for key, value in list(config.items()):
        if key == 'camera_configs' and isinstance(value, dict):   # bag name -> camera yaml
            config[key] = {name: str((base / Path(path).expanduser()).resolve()) for name, path in value.items()}
        elif isinstance(value, dict):
            resolve_paths(value, base)
        elif key in path_keys and value:
            path = Path(value).expanduser()
            config[key] = str((base / path).resolve()) if not path.is_absolute() else str(path)
    return config


def backend_path():
    candidates = [REPO.parents[1] / 'build/air_slam_ros2/map_optimization_backend',
                  REPO.parents[1] / 'install/air_slam_ros2/lib/air_slam_ros2/map_optimization_backend',
                  REPO / 'build/air_slam_ros2/map_optimization_backend',
                  REPO / 'install/air_slam_ros2/lib/air_slam_ros2/map_optimization_backend']
    candidates.extend(Path(prefix) / 'lib/air_slam_ros2/map_optimization_backend'
                      for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if prefix)
    executable = shutil.which('map_optimization_backend')
    if executable:
        return executable
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError('Build map_optimization_backend or set alternating.backend explicitly')


def run_s2(cfg, phase):
    preparation = cfg['prepare']
    cache = Path(preparation['cache_root'])
    overrides = preparation.pop('algorithm_overrides', None)
    if overrides:   # prepare.algorithm with these keys replaced (deep merge), for the preparation VO and the BA rounds
        cache.mkdir(parents=True, exist_ok=True)
        merged = cache / 'algorithm.yaml'
        merged.write_text(yaml.safe_dump({'base': str(Path(preparation['algorithm']).resolve()), **overrides},
                                         sort_keys=False))
        preparation['algorithm'] = str(merged)
    output = Path(cfg['train']['output_dir'])
    if phase in ('all', 'prepare'):
        sequences = prepare_bags(preparation, cache)
    else:
        sequences = json.loads((cache / 'sequences.json').read_text())
    if phase == 'prepare':
        return dict(cache_root=str(cache), sequences=len(sequences))
    output.mkdir(parents=True, exist_ok=True)
    visual_cfg = {key: value for key, value in cfg.items() if key not in ('prepare', 'alternating', 'checkpoint')}
    snapshot = output / 'visual.yaml'
    text = yaml.safe_dump(visual_cfg, sort_keys=False)
    if snapshot.exists():
        saved_visual = yaml.safe_load(snapshot.read_text())
        expected_visual = copy.deepcopy(visual_cfg)
        # Reusing initialize in a new experiment directory does not change training.
        for settings in (saved_visual, expected_visual):
            settings.get('train', {}).pop('output_dir', None)
        if saved_visual != expected_visual:
            raise ValueError(f'Configuration changed; choose a new train.output_dir: {output}')
    if not snapshot.exists():
        snapshot.write_text(text)
    initialization = output / 'initialize'
    checkpoint = initialization / 'visual_finetuned.pt'
    source = Path(cfg['checkpoint'])
    identity = dict(checkpoint=(str(source), source.stat().st_size, source.stat().st_mtime_ns),
                    sequences=sequences, cache_root=str(cache), visual=visual_cfg)
    for seq in sequences:
        marker = Path(seq['map_root']) / 'complete.json'
        identity.setdefault('cache_signatures', []).append(
            json.loads(marker.read_text())['signature'] if marker.exists() else None)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state = output / 'initialize_inputs.json'
    if state.exists() and json.loads(state.read_text())['signature'] != signature:
        raise ValueError('Initialization inputs changed; choose a new train.output_dir')
    write_json(state, dict(signature=signature, inputs=identity))
    if not (initialization / 'complete.json').exists():
        init_cfg = copy.deepcopy(visual_cfg)
        # Fit real-data preprocessing and establish calibration without head training.
        # Without supplied extrinsics, retain only the calibration stage.
        init_cfg['optim']['stages'] = [s for s in init_cfg['optim']['stages']
                                      if s['name'] == 'extrinsic']
        initialization_config = output / 'initialize.yaml'
        initialization_config.write_text(yaml.safe_dump(init_cfg, sort_keys=False))
        data_root = cache
        holdout = ho.settings(visual_cfg)
        if holdout:   # the initialization trains without the held-out segments and validates on them
            data_root = output / 'holdout_init'
            for seq in sequences:
                if seq.get('split', 'train') == 'val':
                    continue
                src = Path(seq['data_root'])
                segments = ho.windows(src / 'keypoints', holdout[0], holdout[1])
                for split, sid, keep_inside in (('train', seq['id'], False), ('val', seq['id'] + ho.SUFFIX, True)):
                    dst = data_root / split / sid
                    (dst / 'keypoints').mkdir(parents=True, exist_ok=True)
                    for name in ('robot', 'camera.yaml', 'keyframe_status.csv'):
                        if (src / name).exists() and not (dst / name).exists():
                            (dst / name).symlink_to((src / name).resolve())
                    for f in ho.split_files(src / 'keypoints', segments, keep_inside, holdout[2]):
                        if not (dst / 'keypoints' / f.name).exists():
                            (dst / 'keypoints' / f.name).symlink_to(f.resolve())
                    ho.write_segments(dst, segments, keep_inside, holdout[2])
            for seq in sequences:   # validation bags as they are
                if seq.get('split') == 'val':
                    dst = data_root / 'val' / seq['id']
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.symlink_to(Path(seq['data_root']).resolve())
        command = [sys.executable, ROOT / 'train_visual_finetune.py', '--config', initialization_config,
                   '--data-root', data_root, '--output-dir', initialization]
        last = initialization / 'last_visual.pt'
        command += ['--resume', last] if last.exists() else ['--model', cfg['checkpoint']]
        run(command, output / 'initialize.log')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        write_json(initialization / 'complete.json', dict(checkpoint=str(checkpoint)))
    alternating = copy.deepcopy(cfg.get('alternating', {}))
    if not alternating.get('enabled', True):
        return dict(final_checkpoint=str(checkpoint))
    alternating.setdefault('algorithm', preparation['algorithm'])
    if not alternating.get('backend'):
        alternating['backend'] = backend_path()
    alternating.setdefault('ba', {})
    for key in ('optimize_extrinsic', 'optimize_time_delay', 'use_bias'):
        alternating['ba'].setdefault(key, False)
    if alternating['ba']['optimize_extrinsic'] or alternating['ba']['optimize_time_delay']:
        raise ValueError('S2 uses one shared calibration; keep BA extrinsic/time delay fixed')
    # 's22' is the historical directory name of the S2 alternating results.
    return run_alternating(sequences, checkpoint, alternating, output / 's22',
                           visual_validation=dict(config=str(snapshot), data_root=str(cache)))


def run_s4_global(cfg, args):
    """S4 global back-end (default):
    1. prepare: loop detection, pose graph and visual global BA of the S3 map; only the accepted loops (loops.txt) and
       the merged map points are used;
    2. global: the S3 back-end graph (visual / leg / foot edges, gyro bias, slip, degeneracy weighting; dump of the
       online run) over the whole trajectory with the loop edges and iterated VCE, initialised from step 1;
    3. structure: the prepared map moved to these poses, then only the map points are optimized.
    The network is not fine-tuned."""
    alternating = copy.deepcopy(cfg.get('alternating', {}))
    for key in ('algorithm', 'backend', 'model_dir', 'voc_path'):
        if getattr(args, key, None):
            alternating[key] = str(Path(getattr(args, key)).resolve())
    backend = alternating.get('backend') or backend_path()
    algorithm_path = Path(alternating['algorithm']).resolve()
    algorithm = load_yaml(algorithm_path)
    base = algorithm_path.parent
    model_dir = str((base / (alternating.get('model_dir') or algorithm['model_dir'])).resolve())
    vocabulary = str((base / (alternating.get('voc_path') or algorithm['voc_path'])).resolve())
    map_root = Path(args.map_root or cfg['map_root']).resolve()
    output = Path(args.output_dir or cfg.get('output_dir', map_root / 'leg_odom/tmp/stage4')).resolve()
    missing = [f for f in ('AirSLAM_mapv0.bin', 'backend_nodes.csv', 'backend_feet.csv') if not (map_root / f).exists()]
    if missing:
        raise FileNotFoundError(f'S4 global_backend needs a map saved by an S3 online run with the back-end '
                                f'(visual_odometry.online.backend.enabled); missing in {map_root}: {missing}')
    settings = copy.deepcopy(cfg.get('global_backend', {}))
    backend_cfg = copy.deepcopy(((algorithm.get('visual_odometry') or {}).get('online') or {}).get('backend') or {})
    backend_cfg.update(settings.get('backend', {}))
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'global_backend.json'
    identity = dict(map_root=str(map_root), algorithm=str(algorithm_path), backend=backend_cfg)
    if snapshot.exists() and json.loads(snapshot.read_text()) != identity:
        raise ValueError(f'S4 inputs or settings changed; choose a new output directory: {output}')
    write_json(snapshot, identity)

    def step(name, mode, files):
        directory = output / name
        if (directory / 'complete.json').exists():
            return directory
        directory.mkdir(parents=True, exist_ok=True)
        for dest, src in files.items():
            if callable(src):
                src(directory / dest)
            else:
                link(src, directory / dest)
        (directory / 'ba_rejected').unlink(missing_ok=True)
        run([backend, mode, algorithm_path, model_dir, directory, vocabulary], directory / f'{name}.log')
        ba_rejected = (directory / 'ba_rejected').exists()  # prepare kept the map before its rejected global BA
        if ba_rejected:
            print(f'[s4] WARNING: {name} global BA rejected; continuing with the map before it', flush=True)
        write_json(directory / 'complete.json', dict(mode=mode, ba_rejected=ba_rejected))
        return directory

    prepare_files = {'AirSLAM_mapv0.bin': map_root / 'AirSLAM_mapv0.bin'}
    if (map_root / 'trajectory_backend.txt').exists():  # S3 back-end poses: loop-group conflict resolution
        prepare_files['trajectory_backend.txt'] = map_root / 'trajectory_backend.txt'
    prepare = step('prepare', 'prepare', prepare_files)
    loops = prepare / 'loops.txt'
    if not loops.exists():
        loops.write_text('')
    write_yaml = lambda path: path.write_text(yaml.safe_dump(backend_cfg, sort_keys=False))
    # the S3 dump: nodes / feet and, when the online run exported them (backend.export_extras), the sensor meta, IMU
    # samples and map point observations (IMU factors, landmarks, VCE). Copied, not linked: the global solve saves its
    # backend_*.csv into its directory, which would overwrite the S3 map's files through a link.
    dump = {name: (lambda path, src=map_root / name: shutil.copy2(src, path))
            for name in ('backend_nodes.csv', 'backend_feet.csv', 'backend_meta.csv', 'backend_imu.csv', 'backend_points.csv')
            if (map_root / name).exists()}
    glob = step('global', 'global', {'global.yaml': write_yaml, **dump,
                                     'prepare_trajectory.txt': prepare / 'trajectory_v1.txt', 'loops.txt': loops})
    structure = step('structure', 'structure', {'AirSLAM_mapv0.bin': prepare / 'AirSLAM_mapv1.bin',
                                                'init_trajectory.txt': glob / 'trajectory_v1.txt'})
    for source, name in [(structure / 'AirSLAM_mapv1.bin', 'AirSLAM_mapv1.bin'), (glob / 'trajectory_v1.txt', 'trajectory_v1.txt')]:
        temporary = map_root / (name + '.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(map_root / name)
    result = dict(method='global_backend', map=str(structure / 'AirSLAM_mapv1.bin'),
                  trajectory=str(glob / 'trajectory_v1.txt'), loops=sum(1 for _ in loops.open()),
                  prepare_ba_rejected=(prepare / 'ba_rejected').exists())
    write_json(output / 'result.json', result)
    return result


def run_s4(cfg, args):
    # the alternating S4 (network inference -> leg BA -> head fine-tuning) was removed
    if cfg.get('method', 'global_backend') != 'global_backend':
        raise ValueError(f"S4 method {cfg.get('method')} was removed; only global_backend remains")
    return run_s4_global(cfg, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['s2', 's4'])
    parser.add_argument('--robot', default='go2', help='S2: robots/<robot>/leg_odometry.yaml (sim_to_real:); S4: robots/<robot>/algorithm.yaml (global_backend:)')
    parser.add_argument('--config', type=Path)
    # 's22' is accepted as a legacy alias of 'train'.
    parser.add_argument('--phase', choices=['all', 'prepare', 'train'], default='all',
                        type=lambda phase: 'train' if phase == 's22' else phase,
                        help='S2: all includes preparation; prepare only builds the cache; train reuses the cache')
    for key in ('algorithm', 'backend', 'model-dir', 'voc-path', 'map-root', 'output-dir'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    robot = REPO / 'robots' / args.robot
    default = robot / ('leg_odometry.yaml' if args.stage == 's2' else 'algorithm.yaml')
    path = (args.config or default).resolve()
    cfg = resolve_paths(load_section(path, 'sim_to_real') if args.stage == 's2' else load_yaml(path), path.parent)
    if args.stage == 's4':   # global_backend: of the robot's algorithm.yaml; the same file is the S4 VO / back-end config
        cfg.setdefault('alternating', {}).setdefault('algorithm', str(robot / 'algorithm.yaml'))
    result = run_s2(cfg, args.phase) if args.stage == 's2' else run_s4(cfg, args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
