"""File-based S2 alternating optimization (BA teacher maps <-> visual head training)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from common.yaml_base import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def run(command, log):
    """No shell expansion; propagate failure and retain the complete diagnostic log."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[pipeline]', ' '.join(map(str, command)), flush=True)
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to replace {destination}')
    destination.symlink_to(source)


def visual_teacher(prepared, keypoints, directory, calibration, reason):
    """Fallback of a failed fused (VCE) teacher: the round uses the sequence's pure visual teacher (prepare map,
    trajectory, keypoints), like the pure visual rounds and the validation sequences; S2 continues."""
    print(f'[alternating] WARNING {Path(directory).name}: fused teacher failed ({reason}); '
          'using the pure visual teacher', flush=True)
    directory = Path(directory)
    link(prepared['map'], directory / 'AirSLAM_mapv1.bin')
    link(prepared['trajectory'], directory / 'trajectory_v1.txt')
    link(keypoints, directory / 'leg_odom/keypoints')
    result = dict(map=str(directory / 'AirSLAM_mapv1.bin'), trajectory=str(directory / 'trajectory_v1.txt'),
                  extrinsic=calibration['extrinsic'], time_delay=calibration['time_delay'],
                  teacher='visual', fallback=reason)
    write_json(directory / 'complete.json', result)
    return result


def run_alternating(sequences, checkpoint, config, output_dir, visual_validation):
    """Each BA starts from the fixed visual map with theta_k, then updates theta.

    sequences: records with id, map_root, data_root, optional split=train|val.
    S2 starts with visual-only supervision before adding leg constraints.
    Validation maps use visual-only BA and never contribute training samples.
    visual_validation: dict(config=S2 visual config, data_root=cache) (pipeline.py run_s2).
    """
    if not visual_validation:
        raise ValueError('run_alternating is the S2 alternating loop and needs visual_validation')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    ids = [s['id'] for s in sequences]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Sequence IDs must be nonempty and unique')
    if any(Path(i).name != i or i in ('.', '..') for i in ids):
        raise ValueError('Sequence IDs must be simple directory names')
    counts = config.get('rounds', [1, 2, 3])
    if (not isinstance(counts, (list, tuple)) or len(counts) != 3
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError('alternating.rounds requires [scale_rounds, adaptation_rounds, head_rounds], nonnegative integers')
    scale_rounds, adaptation_rounds, head_rounds = counts
    rounds = sum(counts)
    early = (yaml.safe_load(Path(visual_validation['config']).read_text()).get('early_stopping', {})
             if visual_validation else {})
    trajectory_stopping = bool(early.get('enabled', False))
    # validation segments held out of every training sequence (common/holdout.py), next to the validation bags
    from common import holdout as ho
    holdout = ho.settings(yaml.safe_load(Path(visual_validation['config']).read_text())) if visual_validation else None
    if holdout:
        for s in sequences:
            if s.get('split', 'train') != 'val':
                s['holdout'] = ho.windows(Path(s['data_root']) / 'keypoints', holdout[0], holdout[1])
                s['holdout_margin'] = holdout[2]
    round_stopper = None
    if trajectory_stopping:
        from common.early_stopping import TrajectoryEarlyStopping
        round_early = dict(early, min_epochs=1, patience=early.get('round_patience', 2))
        # Validate settings before launching any BA or training work.
        TrajectoryEarlyStopping(round_early, 0.0)
        if not holdout and not any(s.get('split') == 'val' for s in sequences):
            raise ValueError('Trajectory early stopping requires validation bags or holdout segments')
    algorithm = load_yaml(config['algorithm'])   # with its base: chain; written in full for the BA runs
    ba = copy.deepcopy(algorithm)
    leg = ba['map_refinement'].setdefault('leg_odom', {})
    leg.update(config.get('ba', {}))
    leg.update(enabled=True)
    backend = str(config['backend'])
    base = Path(config['algorithm']).resolve().parent
    model_dir = str((base / (config.get('model_dir') or algorithm['model_dir'])).resolve())
    vocabulary = str((base / (config.get('voc_path') or algorithm['voc_path'])).resolve())
    python = config.get('python', sys.executable)
    vce = config.get('vce', {})
    if vce.get('enabled', False):
        if Path(backend).name != 'map_optimization_backend':
            raise ValueError('VCE requires map_optimization_backend')
        if any(leg.get(key, False) for key in ('use_bias', 'optimize_extrinsic', 'optimize_time_delay')):
            raise ValueError('VCE requires fixed extrinsic/time delay and disabled bias optimization')
    bias_cfg = config.get('gyro_bias', {})
    use_bias = bool(bias_cfg.get('enabled', False))
    # False keeps every sequence's initial bias fixed: BA-estimated biases can drift far from the stationary
    # values on long sequences.
    estimate_bias = use_bias and bool(bias_cfg.get('estimate', False))
    if estimate_bias and not vce.get('enabled', False):
        raise ValueError('alternating.gyro_bias.estimate requires alternating.vce.enabled')
    # A changed input/config must not silently reuse an old optimization result.
    sources = [checkpoint, Path(config['algorithm']), Path(backend), ROOT / 'stage4.py', Path(__file__),
               ROOT / 'common/stationary.py']
    # camera of a sequence: its own (prepare.camera_configs, e.g. another rig calibration) or the algorithm's
    camera_of = lambda seq: Path(seq['camera']) if seq.get('camera') else (base / algorithm['camera_config_path']).resolve()
    if vce.get('enabled', False):
        sources.extend([*([Path(vce['config_file'])] if vce.get('config_file') else []), ROOT / 'common/vce.py',
                        *sorted({camera_of(seq) for seq in sequences})])
    if vce.get('enabled', False) or use_bias:  # the initial bias estimate lives there too
        sources.append(ROOT / 'common/gyro_bias.py')
    for seq in sequences:
        sources.append(Path(seq['map_root']) / 'AirSLAM_mapv0.bin')
        sources.extend(sorted((Path(seq['data_root']) / 'robot').glob('*.csv')))
        if visual_validation:
            sources.append(Path(seq['data_root']) / 'camera.yaml')
            sources.extend(sorted((Path(seq['data_root']) / 'keypoints').glob('*.txt')))
            sources.extend(sorted(Path(seq['data_root']).glob('keyframe_status.csv')))
    if visual_validation:
        sources.extend([Path(visual_validation['config']), ROOT / 'train_visual_finetune.py',
                        ROOT / 'common/visual_alternating.py', ROOT / 'common/early_stopping.py',
                        ROOT / 'common/implicit_gradient.py', ROOT / 'common/holdout.py',
                        ROOT / 'common/teacher_check.py'])
    identity = dict(config=config, sequences=sequences, checkpoint=str(checkpoint),
                    visual_validation=visual_validation,
                    sources=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources])
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = output / 'run.json'
    if state_path.exists():
        if json.loads(state_path.read_text())['signature'] != signature:
            raise ValueError(f'Inputs changed; choose a new output directory: {output}')
    else:
        write_json(state_path, dict(signature=signature, inputs=identity))
    training_config = output / 'finetune.json'
    finetune = dict(config.get('finetune', {}))
    if visual_validation:   # stage4 infer zeroes the twist while standing like the validation (S2 stationary:)
        finetune.setdefault('stationary', yaml.safe_load(Path(visual_validation['config']).read_text()).get('stationary'))
    write_json(training_config, finetune)

    def optimize(seq, previous, directory, mode, calibration=None, bias=None):
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / 'complete.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            for name in ('map', 'trajectory'):
                if not Path(result[name]).is_file():
                    raise FileNotFoundError(result[name])
            return result
        link(previous, directory / 'AirSLAM_mapv0.bin')
        current = copy.deepcopy(ba)
        settings = current['map_refinement']['leg_odom']
        if calibration:
            settings.update(extrinsic_xyz=calibration['extrinsic'][:3],
                            extrinsic_quat_xyzw=calibration['extrinsic'][3:],
                            time_delay=calibration['time_delay'])
        settings['enabled'] = mode == 'ba' and seq.get('split', 'train') != 'val'
        config_path = directory / 'algorithm.yaml'
        config_path.write_text(yaml.safe_dump(current, sort_keys=False))
        command = [backend, mode, config_path, model_dir, directory, vocabulary]
        selected = None
        if vce.get('enabled', False) and settings['enabled']:
            from common.vce import optimize_vce
            try:
                selected = optimize_vce(backend, vce, previous,
                    directory / 'leg_odom/tmp/network_velocity.csv',
                    camera_of(seq), calibration, directory, run, link,
                    gyro_bias=dict(enabled=True, initial_csv=str(bias),
                        random_walk_sigma=float(bias_cfg.get('random_walk_sigma', 1e-3)),
                        prior_sigma=float(bias_cfg.get('prior_sigma', 0.02))) if bias else None)
            except RuntimeError as error:
                # e.g. a global VCE converging with points behind the camera: this
                # sequence keeps its pure visual teacher in this round instead of stopping S2
                return visual_teacher(initial_maps[seq['id']], output / 'prepare' / seq['id'] / 'leg_odom/keypoints',
                                      directory, calibration, str(error))
        else:
            run(command, directory / 'ba.log')
        result = dict(map=str(directory / 'AirSLAM_mapv1.bin'),
                      trajectory=str(directory / 'trajectory_v1.txt'),
                      **yaml.safe_load((directory / 'calibration.yaml').read_text()))
        if not Path(result['map']).is_file() or not Path(result['trajectory']).is_file():
            raise RuntimeError(f'BA did not produce complete outputs: {directory}')
        if selected is not None:
            result['fusion'] = selected
        write_json(result_file, result)
        return result

    # Sequences are independent within a step (inference, BA, VCE): run them in parallel_sequences threads, each
    # driving its own subprocesses (sequential ~1-2 h per BA round with 30 ANYmal sequences). Results keep the
    # sequence order; the first failure is raised.
    workers = max(1, int(config.get('parallel_sequences', 1)))

    def for_sequences(function):
        if workers == 1 or len(sequences) < 2:
            return [function(seq) for seq in sequences]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(function, sequences))

    # Freeze the loop-closed, visual-only BA result as every round's map input.
    initial_maps = dict(zip(ids, for_sequences(lambda s: optimize(
        s, Path(s['map_root']) / 'AirSLAM_mapv0.bin', output / 'prepare' / s['id'], 'prepare'))))
    maps = initial_maps.copy()
    validation_sequences = [dict(id=s['id'], data_root=s['data_root'],
        trajectory=initial_maps[s['id']]['trajectory'], time_delay=float(leg.get('time_delay', 0.0)))
        for s in sequences if s.get('split') == 'val']
    for s in sequences:
        if 'holdout' in s:   # one pseudo sequence per training sequence, evaluated segment by segment
            path = ho.trajectory(initial_maps[s['id']]['trajectory'], s['holdout'], output / 'holdout' / f"{s['id']}.txt")
            if not ho.usable(path, s['holdout']):   # shorter than one period, or no segment with 3 fixed-BA poses
                print(f"[holdout] {s['id']}: no usable validation segment", flush=True)
                continue
            validation_sequences.append(dict(id=s['id'] + ho.SUFFIX, bias_id=s['id'], data_root=s['data_root'],
                segments=[list(w) for w in s['holdout']], trajectory=path, time_delay=float(leg.get('time_delay', 0.0))))
    if holdout:
        write_json(output / 'holdout' / 'segments.json', {s['id']: s['holdout'] for s in sequences if 'holdout' in s})

    # Gyro bias: the head starts without a constant correction; the bias comes from outside the network.
    from common import gyro_bias as bias_file
    biases = {}
    init_cfg = bias_cfg.get('init', {})

    def camera_from_body(model):
        import torch
        from stage4 import checkpoint_camera_from_body
        return checkpoint_camera_from_body(torch.load(model, map_location='cpu', weights_only=False))

    def imu_extrinsic_of(model):
        import torch
        from common.frames import config_imu_extrinsic
        return config_imu_extrinsic(torch.load(model, map_location='cpu', weights_only=False).get('cfg'))

    def initial_bias(seq, original):
        directory = output / 'prepare' / seq['id']
        path = directory / 'gyro_bias_init.csv'
        if path.exists():
            return path
        mode = init_cfg.get('mode', 'mixed')
        bias, info = bias_file.initial_bias_for(
            mode, seq['data_root'], trajectory=initial_maps[seq['id']]['trajectory'],
            T_C_B=camera_from_body(original) if mode in ('mixed', 'static', 'visual') else None,
            static=init_cfg.get('static'), visual=init_cfg.get('visual'),
            initial_dir=init_cfg.get('initial_dir'), sequence_id=seq['id'], extrinsic=imu_extrinsic_of(original))
        if bias is None:  # network_mean
            velocity = directory / 'network_velocity_orig.csv'
            run([python, ROOT / 'stage4.py', 'infer', '--data-root', seq['data_root'], '--checkpoint', original,
                 '--output', velocity, '--config', training_config], directory / 'infer_orig.log')
            bias = bias_file.network_mean(velocity)
        difference = np.abs(info.get('visual_minus_static_mrad_s', [0.0]))
        if difference.max() > float(init_cfg.get('warn_difference_mrad_s', 1.0)):
            print(f"[gyro_bias] WARNING {seq['id']}: visual and static bias differ by "
                  f"{np.round(info['visual_minus_static_mrad_s'], 2)} mrad/s", flush=True)
        print(f"[gyro_bias] {seq['id']} initial ({mode}, {info.get('source')}): {bias}", flush=True)
        write_json(directory / 'gyro_bias_init.json', dict(mode=mode, bias=bias.tolist(), **info))
        bias_file.write(path, 0.0, bias)
        return path

    def update_biases(directory):
        """Train sequences take their BA estimate; validation uses the train time mean."""
        train = [s['id'] for s in sequences if s.get('split', 'train') != 'val']
        for sid in train:
            if (directory / sid / 'gyro_bias.csv').is_file():
                biases[sid] = directory / sid / 'gyro_bias.csv'
        mean = directory / 'gyro_bias_val.csv'
        if not mean.exists():
            bias_file.write(mean, 0.0, bias_file.time_mean([biases[sid] for sid in train]))
        biases.update({s['id']: mean for s in sequences if s.get('split') == 'val'})

    if use_bias:
        original = checkpoint
        if bias_cfg.get('reset_gyro_head', True):
            checkpoint = output / 'prepare' / 'reset_checkpoint.pt'
            if not checkpoint.exists():
                bias_file.reset_gyro_head(original, checkpoint)
        biases.update({s['id']: initial_bias(s, original) for s in sequences})

    def visual_score(model, directory, data_root, visual_config):
        directory.mkdir(parents=True, exist_ok=True)
        run([python, ROOT / 'train_visual_finetune.py', '--config', visual_config,
             '--init-visual', model, '--fixed-calibration', '--data-root', data_root,
             '--output-dir', directory, '--evaluate-only'],
            directory / 'evaluation.log')
        return float(json.loads((directory / 'visual_metrics.json').read_text())['visual_loss'])

    def repeats_after_rejection(update):
        """With a fixed bias, a rejected head round leaves the checkpoint, BA teacher and seed unchanged, so the
        next round would recompute the same result: stop and run the final BA instead."""
        return (bool(visual_validation) and not estimate_bias and update.get('phase') == 'head'
                and not update.get('accepted', True))

    stopped_after_rejection = False
    # Include one final BA so the output maps correspond to the accepted final model.
    for iteration in range(rounds + 1):
        phase = ('scale' if iteration < scale_rounds else
                 'adaptation' if iteration < scale_rounds + adaptation_rounds else 'head')
        pure_visual = bool(visual_validation) and iteration < scale_rounds + adaptation_rounds
        final_round = (iteration == rounds or stopped_after_rejection
                       or (round_stopper is not None and round_stopper.state['stopped']))
        if final_round and round_stopper is not None:
            checkpoint = Path(round_stopper.state['best_checkpoint'])
        directory = output / f'round_{iteration:03d}'
        directory.mkdir(parents=True, exist_ok=True)

        def round_sequence(seq):
            """Inference and teacher map of one sequence for this round; returns its manifest entry."""
            sid = seq['id']
            work = directory / sid
            work.mkdir(parents=True, exist_ok=True)
            if not (work / 'complete.json').exists():
                velocity = work / 'leg_odom/tmp/network_velocity.csv'
                calibration_file = work / 'model_extrinsic.txt'
                run([python, ROOT / 'stage4.py', 'infer', '--data-root', seq['data_root'],
                     '--checkpoint', checkpoint, '--output', velocity, '--config', training_config,
                     '--extrinsic-output', calibration_file,
                     *(['--gyro-bias', biases[sid]] if use_bias else [])], work / 'infer.log')
                calibration = maps[sid] if iteration else dict(
                    extrinsic=list(map(float, calibration_file.read_text().split())),
                    time_delay=float(leg.get('time_delay', 0.0)))
            else:
                calibration = None
            if pure_visual:
                # Calibration and adaptation share the immutable pure visual teacher.
                # Inference above supplies calibration and a baseline trajectory only.
                if not (work / 'complete.json').exists():
                    link(initial_maps[sid]['map'], work / 'AirSLAM_mapv1.bin')
                    link(initial_maps[sid]['trajectory'], work / 'trajectory_v1.txt')
                    link(output / 'prepare' / sid / 'leg_odom/keypoints',
                         work / 'leg_odom/keypoints')
                    write_json(work / 'complete.json', dict(
                        map=str(work / 'AirSLAM_mapv1.bin'),
                        trajectory=str(work / 'trajectory_v1.txt'),
                        extrinsic=calibration['extrinsic'], time_delay=calibration['time_delay']))
                maps[sid] = json.loads((work / 'complete.json').read_text())
                for name in ('map', 'trajectory'):
                    if not Path(maps[sid][name]).is_file():
                        raise FileNotFoundError(maps[sid][name])
            else:
                maps[sid] = optimize(seq, initial_maps[sid]['map'], work, 'ba', calibration,
                                     biases.get(sid) if estimate_bias else None)
            return dict(data_root=seq['data_root'], trajectory=maps[sid]['trajectory'],
                        extrinsic=maps[sid]['extrinsic'], time_delay=maps[sid]['time_delay'],
                        split=seq.get('split', 'train'))

        manifest = for_sequences(round_sequence)
        if estimate_bias:
            update_biases(directory)
        if final_round:
            break
        record = directory / 'update.json'
        if record.exists():
            saved = json.loads(record.read_text())
            checkpoint = Path(saved['checkpoint'])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if trajectory_stopping and saved.get('early_stopping_state'):
                round_stopper = TrajectoryEarlyStopping(round_early, state=saved['early_stopping_state'])
            stopped_after_rejection = repeats_after_rejection(saved)
            continue
        manifest_path = directory / 'sequences.json'
        write_json(manifest_path, manifest)
        from common.visual_alternating import prepare_visual_round, finetune_visual
        excluded = {}
        check_cfg = config.get('teacher_check', {})
        if check_cfg.get('enabled', False) and not pure_visual:   # fused rounds: teacher vs this round's leg model
            from common import teacher_check
            train_ids = [s['id'] for s in sequences if s.get('split', 'train') != 'val']
            excluded = teacher_check.check(directory, {sid: initial_maps[sid]['trajectory'] for sid in train_ids},
                                           {sid: maps[sid]['extrinsic'] for sid in train_ids}, check_cfg)
        select_cfg = config.get('teacher_select', {})
        select_on = select_cfg.get('enabled', False) and (
            not pure_visual or select_cfg.get('adaptive') or select_cfg.get('visual_strictness') is not None)
        dataset = prepare_visual_round(sequences, directory, fixed_validation=trajectory_stopping,
                                       filter_observations=vce.get('enabled', False) and not pure_visual,
                                       biases=biases, excluded={} if select_on else excluded)
        if select_on:   # teacher selection on this round's teacher (common/teacher_select.py)
            from common import teacher_select
            earlier = [json.loads(p.read_text()) for p in sorted(output.glob('round_*/teacher_select.json'))
                       if p.parent.name < directory.name]
            earlier = [e for e in earlier if not e.get('pure_visual', False)]   # fused rounds only
            previous_s = earlier[-1]['strictness'] if earlier else None
            train_ids = [s['id'] for s in sequences if s.get('split', 'train') != 'val']
            teacher_select.select(directory, dataset, {sid: dict(trajectory=maps[sid]['trajectory'],
                                                                 extrinsic=maps[sid]['extrinsic']) for sid in train_ids},
                                  excluded, len(earlier), previous_s, select_cfg, pure_visual=pure_visual)
        validation = [dict(v, gyro_bias=str(biases[v.get('bias_id', v['id'])])) if use_bias else v
                      for v in validation_sequences]
        # fused head rounds: implicit hypergradient through the fusion BA of the training sequences
        ig = config.get('implicit_gradient', {})
        implicit = None
        if ig.get('enabled', False) and phase == 'head' and not pure_visual:
            if vce.get('enabled', False):
                fusion_runs = {}
                for seq in sequences:
                    if seq.get('split', 'train') == 'val' or maps[seq['id']].get('teacher') == 'visual':
                        continue   # no fusion BA (validation, or a failed VCE replaced by the visual teacher)
                    fusion = (directory / seq['id'] / 'AirSLAM_mapv1.bin').resolve().parent
                    fusion_runs[seq['id']] = dict(fusion_dir=str(fusion), fusion_yaml=str(fusion.parent / 'fusion.yaml'))
                implicit = dict(ig, output_dir=str(directory / 'implicit'), backend=str(backend), sequences=fusion_runs)
            else:
                print('[alternating] implicit_gradient skipped: needs alternating.vce.enabled', flush=True)
        candidate = finetune_visual(checkpoint, visual_validation['config'],
                                   dict(config, python=python), directory, dataset, validation, phase=phase,
                                   implicit=implicit)
        if phase != 'head':
            report = dict(accepted=True, selection='last_epoch', epochs=config.get('epochs', 30))
        elif trajectory_stopping:
            selected = json.loads((directory / 'visual_head' /
                                   f'early_stopping_{phase}.json').read_text())
            if selected['metric'] != TrajectoryEarlyStopping.metric:
                raise ValueError('Unexpected model-selection metric')
            before, after = selected['baseline_score'], selected['best_score']
            TrajectoryEarlyStopping.validate(before)
            TrajectoryEarlyStopping.validate(after)
            report = dict(accepted=after < before, metric=selected['metric'],
                          trajectory_before=before, trajectory_after=after,
                          best_epoch=selected['best_epoch'])
            if round_stopper is None and phase == 'head':
                round_stopper = TrajectoryEarlyStopping(round_early, before)
                round_stopper.state['best_checkpoint'] = str(checkpoint)
        else:
            # Legacy visual-loss acceptance when trajectory stopping is disabled.
            before = visual_score(checkpoint, directory / 'visual_before', dataset, directory / 'visual.yaml')
            after = visual_score(candidate, directory / 'visual_after', dataset, directory / 'visual.yaml')
            report = dict(accepted=math.isfinite(after) and after <= before *
                          float(config.get('visual_tolerance', 1.02)),
                          visual_before=before, visual_after=after)
        if report['accepted']:
            checkpoint = candidate
        if round_stopper is not None:
            improved, _ = round_stopper.update(iteration + 1, after if report['accepted'] else before)
            if improved:
                round_stopper.state['best_checkpoint'] = str(checkpoint)
            report['early_stopping_state'] = copy.deepcopy(round_stopper.state)
            print(f"[round_stopping] round={iteration} best_score={round_stopper.state['best_score']:.6f} "
                  f"stopped={round_stopper.state['stopped']}", flush=True)
        write_json(record, dict(report, phase=phase, checkpoint=str(checkpoint)))
        stopped_after_rejection = repeats_after_rejection(dict(report, phase=phase))
        if stopped_after_rejection:
            print(f'[round_stopping] round={iteration} rejected with a fixed gyro bias: '
                  'the next round would repeat it; running the final BA', flush=True)
    result = dict(final_checkpoint=str(checkpoint), maps=maps, rounds=iteration, max_rounds=rounds,
                  stopped_after_rejection=stopped_after_rejection)
    if round_stopper is not None:
        result.update(early_stopped=round_stopper.state['stopped'],
                      trajectory_score=round_stopper.state['best_score'],
                      best_round=round_stopper.state['best_epoch'] - 1)
    write_json(output / 'result.json', result)
    return result
