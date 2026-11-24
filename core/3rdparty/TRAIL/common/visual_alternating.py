"""Build immutable visual supervision for one BA/network-update round."""
import copy
import math
from pathlib import Path

import numpy as np
import yaml

from common.alternating import link, run, ROOT


def update_points(original, optimized, output, filter_observations=False):
    """Keep original observations/track IDs; refresh only camera-frame XYZ.

    Map merging can renumber track IDs, so identify observations by their fixed
    left-image coordinates within the same timestamped keyframe.
    """
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted(original.glob('*.txt')):
        rows = np.loadtxt(source, ndmin=2)
        if rows.size == 0:
            (output / source.name).write_text('')
            continue
        if rows.shape[1] != 9:
            raise ValueError(f'Expected nine keypoint columns: {source}')
        updated = optimized / source.name
        points = {}
        if updated.exists() and updated.stat().st_size:
            values = np.loadtxt(updated, ndmin=2)
            if values.shape[1] != 9:
                raise ValueError(f'Expected nine keypoint columns: {updated}')
            for row in values:
                key = tuple(row[2:4])
                if key in points:
                    raise ValueError(f'Ambiguous observation {key}: {updated}')
                points[key] = row[6:9]
        if filter_observations:
            rows = rows[[tuple(row[2:4]) in points for row in rows]]
        for row in rows:
            row[6:9] = points.get(tuple(row[2:4]), np.full(3, np.nan))
        np.savetxt(output / source.name, rows, fmt='%.12g')


def prepare_visual_round(sequences, directory, fixed_validation=False, filter_observations=False,
                         biases=None, excluded=None):
    """excluded: sequence id -> teacher windows left out of its training pairs (common/teacher_check.py)."""
    dataset = directory / 'visual_data'
    for seq in sequences:
        source = Path(seq['data_root'])
        teacher = directory.parent / 'prepare' if fixed_validation and seq.get('split') == 'val' else directory
        optimized = teacher / seq['id'] / 'leg_odom' / 'keypoints'
        if not optimized.is_dir():
            raise FileNotFoundError(f'BA did not export keypoints: {optimized}')
        dest = dataset / seq.get('split', 'train') / seq['id']
        link(source / 'robot', dest / 'robot')
        link(source / 'camera.yaml', dest / 'camera.yaml')
        if (source / 'keyframe_status.csv').is_file():   # VO tracking state of the preparation
            link(source / 'keyframe_status.csv', dest / 'keyframe_status.csv')
        if biases:
            link(biases[seq['id']], dest / 'gyro_bias.csv')
        update_points(source / 'keypoints', optimized, dest / 'keypoints',
                      filter_observations=filter_observations and seq.get('split') != 'val')
        if (excluded or {}).get(seq['id']):
            from common.teacher_check import write_excluded
            write_excluded(dest, excluded[seq['id']])
        if seq.get('holdout'):   # validation segments out of the training part, into <id>__holdout
            from common import holdout as ho
            margin = seq.get('holdout_margin', 1.0)
            ho.prune(dest / 'keypoints', ho.split_files(source / 'keypoints', seq['holdout'], False, margin))
            ho.write_segments(dest, seq['holdout'], False, margin)
            val_teacher = directory.parent / 'prepare' if fixed_validation else directory
            vdest = dataset / 'val' / (seq['id'] + ho.SUFFIX)
            link(source / 'robot', vdest / 'robot')
            link(source / 'camera.yaml', vdest / 'camera.yaml')
            if (source / 'keyframe_status.csv').is_file():
                link(source / 'keyframe_status.csv', vdest / 'keyframe_status.csv')
            if biases:
                link(biases[seq['id']], vdest / 'gyro_bias.csv')
            update_points(source / 'keypoints', val_teacher / seq['id'] / 'leg_odom' / 'keypoints', vdest / 'keypoints')
            ho.prune(vdest / 'keypoints', ho.split_files(source / 'keypoints', seq['holdout'], True))
            ho.write_segments(vdest, seq['holdout'], True)
    return dataset


def finetune_visual(checkpoint, config_path, config, directory, dataset, validation_sequences=None, phase="head",
                    implicit=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg.get('early_stopping', {}).get('enabled', False):
        cfg['early_stopping']['validation_sequences'] = validation_sequences or []
    if phase not in ('scale', 'adaptation', 'head'):
        raise ValueError(f'Unknown alternating phase: {phase}')
    if phase != 'head':
        cfg.setdefault('early_stopping', {})['enabled'] = False
    epochs = config.get('epochs', 30)
    if type(epochs) is not int or epochs <= 0:
        raise ValueError('alternating.epochs must be a positive integer')
    scale_only_lr = float(config.get('scale_lr', 1e-3))
    scale_lr = float(config.get('log_scale_lr', 1e-6))
    if (not math.isfinite(scale_only_lr) or scale_only_lr <= 0
            or not math.isfinite(scale_lr) or scale_lr < 0):
        raise ValueError('scale_lr must be positive and log_scale_lr nonnegative; both finite')
    # LoRA adapters on the frozen backbone, trained with both heads in the listed phases ('adaptation' = pure visual
    # rounds, 'head' = fused rounds); other phases keep the adapters (of an earlier round) frozen.
    lora = dict(config.get('lora', {}))
    use_lora = bool(lora.pop('enabled', False)) and phase in lora.pop('phases', ['adaptation', 'head'])
    lora_lr = float(lora.pop('lr', 1e-4))
    scale_only = phase == 'scale'
    stage = copy.deepcopy(next(s for s in cfg['optim']['stages'] if s['name'] == 'head'))
    stage.pop('lora', None)
    stage.update(name=phase, epochs=epochs, train_network=False if scale_only else 'heads',
                 extrinsic_lr=0.0,
                 log_scale_lr=scale_only_lr if scale_only else scale_lr)
    if scale_only:
        stage.update(network_lr=0.0, use_velocity_smooth=False)
    elif use_lora:
        stage.update(train_network='lora_heads', lora=lora, lora_lr=lora_lr)
    if phase != 'head':
        # pure visual rounds keep the last epoch: their per-epoch visual validation is only logged
        stage['visual_eval_interval'] = int(config.get('adaptation_eval_interval', 1))
    cfg['optim']['stages'] = [stage]
    window = config.get('window', {})
    frames = window.get('frames', 5)
    if type(frames) is not int or frames < 3:
        raise ValueError('alternating.window.frames must be an integer >= 3')
    duration = float(window.get('max_seconds', 5.0))
    weight = float(window.get('cross_frame_weight', 1.0))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Window duration and cross-frame weight must be finite and positive')
    cfg.setdefault('visual', {}).update(window_frames=frames, window_max_seconds=duration,
                                        cross_frame_weight=weight)
    # Keep approximately the previous number of adjacent edges per GPU batch.
    for section in ('train', 'eval'):
        settings = cfg.setdefault(section, {})
        settings['batch_size'] = max(1, int(settings.get('batch_size', 4)) // (frames - 1))
    if implicit:  # fused head round: implicit hypergradient (common/implicit_gradient.py)
        cfg['implicit_gradient'] = implicit
    path = directory / 'visual.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    output = directory / 'visual_head'
    last = output / 'last_visual.pt'
    initialization = ['--resume', last] if last.exists() else ['--init-visual', checkpoint]
    run([config.get('python', 'python'), ROOT / 'train_visual_finetune.py',
         '--config', path, *initialization,
         '--fixed-calibration',
         '--data-root', dataset, '--output-dir', output], directory / 'finetune.log')
    candidate = output / 'visual_finetuned.pt'
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate
