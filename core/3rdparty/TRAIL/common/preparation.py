"""Build immutable S2 caches with the existing ROS2 VO and sensor recorder."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common.alternating import ROOT, run, write_json
from common.robot import load_robot
from common.yaml_base import load_yaml


def log_tail(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 6000))
            return stream.read().decode('utf-8', errors='replace')
    except OSError as error:
        return f'Cannot read log: {error}'


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError('ROS process did not finish saving within 120 seconds')


def bag_topics(bag: Path) -> set:
    info = yaml.safe_load((bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    return {t['topic_metadata']['name'] for t in info.get('topics_with_message_count') or []}


def run_topics(camera: Path, robot: dict, recorder_params: dict) -> tuple[list, list]:
    """(required, optional) topics a preparation run consumes: the camera yaml images (required) and IMU (optional: the
    VO may not use it), and the recorder's leg inputs (robot leg_inputs joints + IMU, else its LowStateStamped topic)."""
    cam = yaml.safe_load(camera.read_text().replace('%YAML:1.0', ''))
    required = [cam.get('left_image_topic'), cam.get('right_image_topic')]
    inputs = robot.get('leg_inputs')
    if inputs:
        required += [inputs['joints']['topic'], inputs['imu']['topic']]
    else:
        required.append(recorder_params.get('lowstate_topic') or (robot.get('recorder') or {}).get('lowstate_topic', '/lowstate_stamped'))
    required = list(dict.fromkeys(str(t) for t in required if t))
    optional = [str(cam['imu_topic'])] if cam.get('imu_topic') and str(cam['imu_topic']) not in required else []
    return required, optional


def playback_options(config, bag: Path, camera: Path, robot: dict, recorder_params: dict, log_dir: Path) -> list:
    """`ros2 bag play` options of a preparation run:
    - prepare.playback_topics: auto (default; the topics the VO and the recorder consume, see run_topics) | all | list.
      Fewer topics, less disk reading and no starved player bursts.
    - prepare.playback_qos_depth (default 100; 0 = rosbag2 defaults): publisher queue depth of the played topics.
      Bags without recorded QoS are played with keep_last 10, which drops a burst released after a disk stall."""
    present = bag_topics(bag)
    selection = config.get('playback_topics', 'auto')
    if selection == 'all':
        topics = sorted(present)
    elif selection == 'auto':
        required, optional = run_topics(camera, robot, recorder_params)
        missing = [t for t in required if t not in present]
        if missing:
            raise ValueError(f'{bag}: topics needed by the preparation run are missing: {missing}')
        topics = required + [t for t in optional if t in present]
    else:
        topics = [str(t) for t in selection]
        if set(topics) - present:
            raise ValueError(f'{bag}: prepare.playback_topics not in the bag: {sorted(set(topics) - present)}')
    options = [] if selection == 'all' else ['--topics', *topics]
    depth = int(config.get('playback_qos_depth', 100))
    if depth > 0:
        qos = log_dir / 'playback_qos.yaml'
        qos.write_text(yaml.safe_dump({t: dict(history='keep_last', depth=depth, reliability='reliable',
                                               durability='volatile') for t in topics}, sort_keys=True))
        options += ['--qos-profile-overrides-path', str(qos)]
    return options


def prepare_bags(config, cache_root):
    readiness_timeout = int(config.get('readiness_timeout_s', 3600))
    if readiness_timeout <= 0:
        raise ValueError('prepare.readiness_timeout_s must be positive')
    root = Path(config['bag_root']).resolve()
    cache = Path(cache_root).resolve()
    bags = sorted(p.parent for p in root.rglob('metadata.yaml'))
    if (root / 'metadata.yaml').exists():
        bags = [root]
    if not bags:
        raise ValueError(f'No ROS2 bags with metadata.yaml under {root}')
    names = {bag: (bag.relative_to(root).as_posix() if bag != root else bag.name) for bag in bags}
    # optional whole validation bags; S2 validates on segments held out of every training bag (holdout:)
    validation = set(config.get('validation_sequences') or [])
    if validation - set(names.values()):
        raise ValueError('prepare.validation_sequences must name existing bags relative to bag_root')
    if len(validation) == len(bags):
        raise ValueError('At least one training bag is required')
    algorithm = Path(config['algorithm']).resolve()
    # recorder ROS parameters: an explicit recorder_config file, else prepare.robot (its leg_odometry.yaml (robot:) recorder: defaults)
    recorder_file = Path(config['recorder_config']).resolve() if config.get('recorder_config') else None
    recorder_params = (yaml.safe_load(recorder_file.read_text())['/**']['ros__parameters'] if recorder_file
                       else {'robot': str(config.get('robot') or 'go2')})
    recorder_dir = recorder_file.parent if recorder_file else Path.cwd()
    recorder_robot = str(recorder_params.get('robot') or 'go2')
    if recorder_robot.endswith(('.yaml', '.yml')):
        recorder_robot = str((recorder_dir / recorder_robot).resolve())
    robot_descriptor = load_robot(recorder_robot)
    recorder_signature = json.dumps([recorder_params, robot_descriptor.get('recorder')], sort_keys=True)
    algorithm_cfg = load_yaml(algorithm)   # with its base: chain
    camera = (algorithm.parent / (config.get('camera_config') or algorithm_cfg['camera_config_path'])).resolve()
    model_dir = str((algorithm.parent / (config.get('model_dir') or algorithm_cfg['model_dir'])).resolve())
    # per-bag camera configs (bag name -> camera yaml), e.g. a site with another rig calibration
    cameras = {name: Path(path).resolve() for name, path in (config.get('camera_configs') or {}).items()}
    if set(cameras) - set(names.values()):
        raise ValueError(f'prepare.camera_configs names unknown bags: {sorted(set(cameras) - set(names.values()))}')
    selfcal = config.get('stereo_selfcal') or {}
    if selfcal.get('enabled', False):
        # stereo disparity-offset self-calibration (common/stereo_selfcal.py): corrected cameras of the bag groups whose
        # right camera yaw is off; estimates, decisions and cameras in <cache_root>/stereo_selfcal (report.json)
        from common.stereo_selfcal import calibrate
        cameras.update(calibrate({name: bag for bag, name in names.items()},
                                 {name: cameras.get(name, camera) for name in names.values()},
                                 cache / 'stereo_selfcal', selfcal))
    # Isolate bag playback from other running robots/nodes.
    env = dict(os.environ, ROS_DOMAIN_ID=str(config.get('ros_domain_id', 100 + os.getpid() % 100)))
    sequences = []
    for bag, name in names.items():
        sid = hashlib.sha256(name.encode()).hexdigest()[:10] + '_' + bag.name
        split = 'val' if name in validation else 'train'
        bag_camera = cameras.get(name, camera)
        destination = cache / split / sid
        completion = destination / 'complete.json'
        bag_files = sorted(p for p in bag.rglob('*') if p.is_file())
        # the bag's own camera enters by its text: other bags' camera_configs or self-calibration keep this cache
        bag_config = {key: value for key, value in config.items() if key not in ('camera_configs', 'stereo_selfcal')}
        signature = hashlib.sha256(json.dumps(dict(
            point_export_version=2, config=bag_config, files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in bag_files],
            algorithm=yaml.safe_dump(algorithm_cfg, sort_keys=True), camera=bag_camera.read_text(), recorder=recorder_signature,
            preparation=Path(__file__).read_text(),
        ), sort_keys=True).encode()).hexdigest()
        entry = dict(id=sid, split=split, map_root=str(destination),
                     data_root=str(destination / 'leg_odom'))
        if name in cameras:
            entry['camera'] = str(bag_camera)
        sequences.append(entry)
        if completion.exists():
            if json.loads(completion.read_text())['signature'] != signature:
                raise ValueError(f'Cache inputs changed; choose a new cache_root: {destination}')
            required = ['AirSLAM_mapv0.bin', 'leg_odom/camera.yaml', 'leg_odom/robot/contacts.csv']
            if not all((destination / p).is_file() for p in required):
                raise FileNotFoundError(f'Incomplete cache: {destination}')
            continue
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f'Incomplete previous recording: remove or move {destination} before retrying')
        print(f'[prepare] {name} -> {destination}', flush=True)
        # VO clears the map directory on startup, preserving only leg_odom.
        log_dir = destination / 'leg_odom' / 'preparation'
        log_dir.mkdir(parents=True, exist_ok=True)
        params = dict(recorder_params, map_root=str(destination), estimate_contacts=False, overwrite=False)
        recorder_cfg = {'/**': {'ros__parameters': params}}
        recorder_path = log_dir / 'recorder.yaml'
        recorder_path.write_text(yaml.safe_dump(recorder_cfg))
        vo_command = ['ros2', 'run', 'air_slam_ros2', 'visual_odometry_node', '--ros-args',
                      '-p', f'config_path:={algorithm}', '-p', f'model_dir:={model_dir}',
                      '-p', f'camera_config_path:={bag_camera}', '-p', f'saving_dir:={destination}',
                      # visual-only VO: no leg node runs during preparation, a fusion mode would wait for it
                      '-p', 'online_mode:=visual_only']
        for key, value in config.get('vo_parameters', {}).items():
            vo_command += ['-p', f'{key}:={value}']
        recorder_command = ['ros2', 'run', 'airslam_leg_odom_recorder', 'recorder',
                            '--ros-args', '--params-file', str(recorder_path)]
        processes, logs = [], []
        try:
            commands = [(vo_command, 'vo.log'), (recorder_command, 'recorder.log')]
            # the VO IMU at the joint stamps (leg_odometry.yaml robot: leg_inputs.imu.resampled_topic, imu_resampler.py)
            if ((robot_descriptor.get('leg_inputs') or {}).get('imu') or {}).get('resampled_topic'):
                commands.append((['ros2', 'run', 'airslam_leg_odom_recorder', 'imu_resampler', '--ros-args',
                                  '-p', f'robot:={recorder_robot}'], 'imu_resampler.log'))
            for command, filename in commands:
                log = (log_dir / filename).open('w')
                logs.append(log)
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True))
            # Query graph readiness with the ROS Python interpreter, independent of Torch's env.
            ready = ROOT / 'common' / 'ros_ready.py'
            ready_log = log_dir / 'readiness.log'
            with ready_log.open('w') as log:
                readiness = subprocess.Popen(
                    [config.get('ros_python', '/usr/bin/python3'), str(ready), str(readiness_timeout)],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + readiness_timeout + 30
                    while readiness.poll() is None:
                        if any(p.poll() is not None for p in processes):
                            raise RuntimeError('VO/recorder exited before playback')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('ROS readiness check timed out')
                        time.sleep(0.2)
                    if readiness.returncode:
                        raise RuntimeError('VO/recorder subscriptions are not ready')
                except Exception as error:
                    details = '\n'.join(
                        f'--- {name} ---\n{log_tail(log_dir / name)}'
                        for name in ('vo.log', 'recorder.log', 'readiness.log'))
                    raise RuntimeError(f'{error}; logs: {log_dir}\n{details}') from error
                finally:
                    if readiness.poll() is None:
                        readiness.terminate()
                        try:
                            readiness.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            readiness.kill()
                            readiness.wait()
            playback = ['ros2', 'bag', 'play', str(bag), '--rate', str(config.get('playback_rate', 0.5)),
                        '--delay', '2', *playback_options(config, bag, bag_camera, robot_descriptor, recorder_params, log_dir)]
            with (log_dir / 'playback.log').open('w') as log:
                subprocess.run(playback, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            time.sleep(float(config.get('drain_seconds', 5.0)))
            if any(p.poll() is not None for p in processes):
                raise RuntimeError(f'VO/recorder exited during playback; see {destination}')
        finally:
            errors = []
            for process in processes:
                try:
                    stop(process)
                except RuntimeError as error:
                    errors.append(error)
            for log in logs:
                log.close()
            if errors:
                raise errors[0]
        data = destination / 'leg_odom'
        status = json.loads((data / 'recorder_status.json').read_text())
        if status.get('dropped', 0) or status.get('lowstate_errors', 0) or not status.get('imu') or not status.get('joints'):
            raise RuntimeError(f'Incomplete sensor recording: {status}')
        if not (destination / 'AirSLAM_mapv0.bin').is_file() or len(list((data / 'keypoints').glob('*.txt'))) < 2:
            raise RuntimeError(f'VO did not export a usable map: {destination}')
        # robot of the recorder config: kinematics model and pressure settings (robots/<robot>/leg_odometry.yaml (robot:))
        robot = recorder_robot
        pressure_command = [sys.executable, ROOT / 'data_scripts/estimate_contacts/estimate_real.py',
                            '--data-root', data, '--robot', robot, '--method', 'jacobian', '--overwrite']
        for key in ('pressure_scale', 'pressure_mode', 'damping', 'smooth_alpha', 'power'):
            if 'contact_' + key in params:
                pressure_command += ['--' + key.replace('_', '-'), str(params['contact_' + key])]
        if params.get('contact_model'):
            pressure_command += ['--model', str((recorder_dir / params['contact_model']).resolve())]
        run(pressure_command, log_dir / 'pressure.log')
        metadata = json.loads((data / 'manifest.json').read_text())
        metadata.update(contact_source='estimated', contact_estimation_method='jacobian',
                        contact_normalize_sum=False)
        write_json(data / 'manifest.json', metadata)
        write_json(completion, dict(signature=signature, bag=str(bag), **entry))
    write_json(cache / 'sequences.json', sequences)
    return sequences
