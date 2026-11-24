"""No-GT model selection against immutable, visual-only BA trajectories."""
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class TrajectoryEarlyStopping:
    metric = 'fixed_ba_se3_rmse'

    def __init__(self, config, baseline=None, state=None):
        self.minimum = config.get('min_epochs', 20)
        self.patience = config.get('patience', 15)
        self.delta = float(config.get('min_delta_relative', 0.005))
        if any(type(x) is not int or x < 1 for x in (self.minimum, self.patience)):
            raise ValueError('Early stopping epochs and patience must be positive integers')
        if not math.isfinite(self.delta) or not 0 <= self.delta < 1:
            raise ValueError('early_stopping.min_delta_relative must be in [0, 1)')
        if state is None:
            self.validate(baseline)
            state = dict(metric=self.metric, baseline_score=baseline, best_score=baseline,
                         best_epoch=0, trend_best=baseline, last_improved=0, stopped=False)
        if state.get('metric') != self.metric:
            raise ValueError('Incompatible early-stopping state; use a new output directory')
        self.validate(state['best_score'])
        self.state = state

    @staticmethod
    def validate(score):
        if score is None or not math.isfinite(score) or score < 0:
            raise ValueError(f'Invalid fixed-BA trajectory score: {score}')

    def update(self, epoch, score):
        self.validate(score)
        state = self.state
        improved = score < state['best_score']
        if improved:
            state.update(best_score=score, best_epoch=epoch)
        # Always retain the best actual model; only substantial progress resets patience.
        if score < state['trend_best'] * (1 - self.delta):
            state.update(trend_best=score, last_improved=epoch)
        state['stopped'] = epoch >= self.minimum and epoch - state['last_improved'] >= self.patience
        return improved, dict(score=score, best_epoch=state['best_epoch'],
                              last_improved=state['last_improved'], stopped=state['stopped'])


def aligned_rmse(position, reference):
    """Position RMSE after proper rotation/translation alignment, without rescaling."""
    x, y = position - position.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    error = x @ rotation.T - y
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def integrate_camera(times, velocity, camera, extrinsic, time_delay=0.0):
    """Continuous left-held body twist integration at camera timestamps, T_C_B extrinsic."""
    ct = camera[:, 0] + time_delay
    clock = np.unique(np.r_[ct, times[(times > ct[0]) & (times < ct[-1])]])
    ids = np.searchsorted(times, clock[:-1], side='right') - 1
    dt = np.diff(clock)
    delta = Rotation.from_rotvec(velocity[ids, 3:] * dt[:, None]).as_matrix()
    # Parallel prefix products avoid a Python loop over every sensor sample.
    step = 1
    while step < len(delta):
        delta[step:] = delta[:-step] @ delta[step:]
        step *= 2
    re = Rotation.from_quat(extrinsic[3:]).as_matrix()
    rc = Rotation.from_quat(camera[0, 4:8]).as_matrix()
    rotation = (rc @ re)[None] @ np.concatenate([np.eye(3)[None], delta])
    displacement = np.einsum('nij,nj->ni', rotation[:-1], velocity[ids, :3]) * dt[:, None]
    position = np.vstack([np.zeros((1, 3)), np.cumsum(displacement, axis=0)])
    position += camera[0, 1:4] + rc @ extrinsic[:3]
    idx = np.searchsorted(clock, ct)
    return position[idx] - np.einsum('nij,j->ni', rotation[idx] @ re.T, extrinsic[:3])


class FixedBATrajectoryValidation:
    def __init__(self, model, normalizer, cfg, device, sequence_ids):
        from stage4 import load_inputs, online_start, sliding_windows

        settings = cfg['early_stopping']
        sources = settings.get('validation_sequences', [])
        if (not sources or len({s['id'] for s in sources}) != len(sources)
                or {s['id'] for s in sources} != set(sequence_ids)):
            raise ValueError('Trajectory stopping requires fixed BA references for every validation '
                             'sequence; run pipeline.py s2 --phase train')
        self.batch_size = settings.get('inference_batch_size', 2048)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('early_stopping.inference_batch_size must be a positive integer')
        self.device = device
        self.sequences = []
        branch = model.velocity_branch
        frozen = not any(p.requires_grad for name, p in branch.named_parameters()
                         if not name.startswith(('velocity_head.', 'gyro_head.', 'uncertainty_head.')))
        was_training = branch.training
        branch.eval()
        try:
            with torch.no_grad():
                for source in sources:
                    _, _, x, raw, _, times, _ = load_inputs(
                        Path(source['data_root']), None, device, (model, normalizer, {}),
                        gyro_bias=source.get('gyro_bias'))
                    start = online_start(raw.cpu().numpy(), times, model.cfg)
                    window = int(model.cfg['data']['window'])
                    windows = sliding_windows(x, window)[start - window + 1:]
                    times = times[start:]
                    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                        raise ValueError(f'Invalid robot timestamps: {source["id"]}')
                    camera = np.loadtxt(source['trajectory'], ndmin=2)
                    delay = float(source.get('time_delay', 0.0))
                    if (camera.shape[1] != 8 or not np.isfinite(camera).all()
                            or np.any(np.diff(camera[:, 0]) <= 0) or not math.isfinite(delay)):
                        raise ValueError(f'Invalid fixed BA trajectory: {source["trajectory"]}')
                    camera = camera[(camera[:, 0] + delay >= times[0]) &
                                    (camera[:, 0] + delay <= times[-1])]
                    if len(camera) < 3:
                        raise ValueError(f'Insufficient fixed BA/sensor overlap: {source["id"]}')
                    hidden = None
                    if frozen:
                        hidden = torch.cat([branch.forward_features(windows[b:b + self.batch_size]).cpu()
                                            for b in range(0, len(windows), self.batch_size)])
                    from common.stationary import sequence_mask
                    self.sequences.append(dict(source=source, camera=camera, times=times,
                        gyro=raw[start:, :3].cpu(), hidden=hidden, window=window, start=start,
                        x=x.cpu() if hidden is None else None, time_delay=delay,
                        stationary=sequence_mask(source['data_root'], times, cfg)))   # as stage4 infer
        finally:
            branch.train(was_training)

    def evaluate(self, model, calibration):
        from stage4 import checkpoint_camera_from_body, checkpoint_velocity_scale, sliding_windows

        # Match the exported checkpoint's scale and extrinsic conversion exactly.
        payload = dict(visual_calibration=calibration.state_dict(),
                       visual_calibration_summary=calibration.summary())
        extrinsic = np.asarray(checkpoint_camera_from_body(payload))
        scale = checkpoint_velocity_scale(payload)
        branch = model.velocity_branch
        was_training = branch.training
        branch.eval()
        rows = {}
        try:
            with torch.no_grad():
                for seq in self.sequences:
                    parts = []
                    windows = None if seq['hidden'] is not None else sliding_windows(
                        seq['x'], seq['window'])[seq['start'] - seq['window'] + 1:]
                    for b in range(0, len(seq['times']), self.batch_size):
                        sl = slice(b, b + self.batch_size)
                        hidden = (seq['hidden'][sl].to(self.device) if windows is None
                                  else branch.forward_features(windows[sl].to(self.device)))
                        linear = branch.velocity_head(hidden)[..., :3] * scale
                        angular, _ = branch.refine_gyro(hidden, seq['gyro'][sl].to(self.device))
                        parts.append(torch.cat([linear, angular], dim=-1).cpu().numpy())
                    velocity = np.concatenate(parts)
                    if seq['stationary'] is not None:
                        velocity[seq['stationary']] = 0.0
                    if not np.isfinite(velocity).all():
                        raise ValueError(f'Nonfinite validation velocity: {seq["source"]["id"]}')
                    camera = seq['camera']
                    if seq['source'].get('segments'):   # held-out segments (common/holdout.py), one integration each
                        from common.holdout import runs
                        errors = [aligned_rmse(integrate_camera(seq['times'], velocity, camera[i], extrinsic,
                                                                seq['time_delay']), camera[i, 1:4])
                                  for i in runs(camera[:, 0], seq['source']['segments'])]
                        if not errors:
                            raise ValueError(f'No usable held-out segment: {seq["source"]["id"]}')
                        rmse = float(np.mean(errors))
                    else:
                        rmse = aligned_rmse(integrate_camera(seq['times'], velocity, camera, extrinsic,
                                                             seq['time_delay']), camera[:, 1:4])
                    rows[seq['source']['id']] = dict(
                        se3_rmse_m=rmse,
                        samples=len(camera), start_time=float(camera[0, 0]), end_time=float(camera[-1, 0]),
                        reference=seq['source']['trajectory'])
        finally:
            branch.train(was_training)
        result = dict(trajectory_rmse=float(np.mean([r['se3_rmse_m'] for r in rows.values()])),
                      trajectory_sequences=rows)
        TrajectoryEarlyStopping.validate(result['trajectory_rmse'])
        return result
