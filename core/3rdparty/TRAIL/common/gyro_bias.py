"""Per-keyframe gyro bias files: ``time,bgx,bgy,bgz`` (rad/s, base axes, subtracted from the gyro in base axes,
i.e. after the IMU -> base rotation of common/frames.py).

A single row means a constant bias. Between keyframes the bias is linearly
interpolated and held at the end values outside the covered interval.
"""
from pathlib import Path

import numpy as np

HEADER = 'time,bgx,bgy,bgz'


def load(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if data.shape[1] != 4 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f'Invalid gyro bias CSV: {path}')
    return data[:, 0], data[:, 1:]


def at(path, times):
    keyframe_times, bias = load(path)
    return np.stack([np.interp(times, keyframe_times, bias[:, i]) for i in range(3)], axis=-1)


def write(path, times, bias):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([np.atleast_1d(times), np.atleast_2d(bias)])
    np.savetxt(path, rows, delimiter=',', header=HEADER, comments='', fmt='%.17g')


def time_mean(paths):
    """Time-weighted mean over several bias files; single-row files weigh one second."""
    total, weight = np.zeros(3), 0.0
    for path in paths:
        times, bias = load(path)
        duration = times[-1] - times[0]
        if duration > 0:
            total += np.trapz(bias, times, axis=0)
            weight += duration
        else:
            total += bias[0]
            weight += 1.0
    return total / weight


def recording_extrinsic(data_root):
    """IMU extrinsic of a recording (<data_root>/manifest.json): an explicit imu_extrinsic {xyz, quat_xyzw} (e.g. an
    external IMU written into the recording), else the robot's, else the default robot's."""
    import json

    from common.frames import ImuExtrinsic, imu_extrinsic
    manifest = Path(data_root) / 'manifest.json'
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    stored = info.get('imu_extrinsic') or {}
    if 'xyz' in stored:
        return ImuExtrinsic.from_dict(stored)
    return imu_extrinsic(info.get('robot'))


def read_gyro(data_root, extrinsic=None):
    """Gyro in base axes (time, N x 3 rad/s) from <data_root>/robot/imu.csv; extrinsic = common.frames.ImuExtrinsic
    (default: recording_extrinsic)."""
    from common.frames import rotate_vectors
    from eval.real_utils import read_csv_real

    imu = read_csv_real(Path(data_root) / 'robot' / 'imu.csv')
    gyro = np.column_stack([np.asarray(imu[f'gyro_{a}'], dtype=np.float64) for a in 'xyz'])
    extrinsic = extrinsic if extrinsic is not None else recording_extrinsic(data_root)
    return np.asarray(imu['time'], dtype=np.float64), rotate_vectors(gyro, extrinsic.R_BI)


def static_bias(data_root, window_s=1.0, step_s=0.5, gyro_std_max=0.01, joint_vel_max=0.25, min_windows=3,
                min_fill=0.8, extrinsic=None):
    """Median gyro over every stationary window of the whole sequence, or (None, info) with too few windows.

    A window is stationary when every gyro axis has std < gyro_std_max and all joint speeds stay below
    joint_vel_max (noise floors ~0.0065 rad/s and ~0.14 rad/s), and it holds at least min_fill of the samples
    expected at the median IMU rate.
    """
    from eval.real_utils import read_csv_real

    t, gyro = read_gyro(data_root, extrinsic)
    joints = read_csv_real(Path(data_root) / 'robot' / 'joints.csv')
    joint_time = np.asarray(joints['time'], dtype=np.float64)
    joint_speed = np.max(np.abs(np.column_stack(
        [joints[k] for k in joints if k.endswith('_vel')])), axis=1)
    needed = min_fill * window_s / np.median(np.diff(t))
    means, starts = [], []
    for begin in np.arange(t[0], t[-1] - window_s, step_s):
        inside = (t >= begin) & (t < begin + window_s)
        speed = joint_speed[(joint_time >= begin) & (joint_time < begin + window_s)]
        g = gyro[inside]
        if (len(g) >= needed and g.std(axis=0).max() < gyro_std_max
                and len(speed) and speed.max() < joint_vel_max):
            means.append(g.mean(axis=0)); starts.append(float(begin - t[0]))
    info = dict(windows=len(means), window_starts_s=starts)
    return (np.median(means, axis=0) if len(means) >= min_windows else None), info


def _attitudes(t, w, bias):
    """Gyro attitude R_k (R_0 = I) at every sample with omega = w - bias, and C = sum R_{k+1} dt."""
    from scipy.spatial.transform import Rotation

    dt = np.diff(t)
    steps = Rotation.from_rotvec((w[:-1] - bias) * dt[:, None]).as_matrix()
    R = np.empty((len(t), 3, 3)); R[0] = np.eye(3)
    for i in range(len(dt)):
        R[i + 1] = R[i] @ steps[i]
    C = np.zeros((len(t), 3, 3))
    C[1:] = np.cumsum(R[1:] * dt[:, None, None], axis=0)
    return R, C


def visual_bias(data_root, trajectory, T_C_B, window_s=20.0, huber_deg=1.0, iterations=6, extrinsic=None):
    """One constant gyro bias from long-window relative rotations of a visual trajectory.

    trajectory: TUM file of T_W_C (the loop-closed, visual-only BA); T_C_B: [tx ty tz qx qy qz qw].
    For keyframe pairs ~window_s apart, minimises sum rho(|Log(R_vis(i,j)^T R_gyro(i,j; b))|) by Gauss-Newton
    (Huber weights at huber_deg) with the preintegration Jacobian d r / d b = -R_j^T (C_j - C_i).
    Returns (bias, info).
    """
    from scipy.spatial.transform import Rotation

    t, w = read_gyro(data_root, extrinsic)
    visual = np.loadtxt(trajectory, ndmin=2)
    visual = visual[(visual[:, 0] > t[0]) & (visual[:, 0] < t[-1])]
    kf_t = visual[:, 0]
    R_vis = Rotation.from_quat(visual[:, 4:8]).as_matrix() @ Rotation.from_quat(T_C_B[3:]).as_matrix()
    pairs = keyframe_pairs(kf_t, window_s)
    if len(pairs) < 3:
        raise ValueError(f'Too few {window_s:g} s keyframe windows for a visual gyro bias: {data_root}')
    bias, r = gauss_newton_bias(t, w, kf_t, R_vis, pairs, np.zeros(3), huber_deg, iterations)
    rms = float(np.degrees(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))))
    return bias, dict(windows=len(pairs), residual_rms_deg=rms)


def keyframe_pairs(kf_t, window_s):
    """(i, j) keyframe pairs about window_s apart (the first keyframe after start + window_s, within 1.5 window_s)."""
    pairs = []
    for i, start in enumerate(kf_t):
        j = int(np.searchsorted(kf_t, start + window_s))
        if j < len(kf_t) and kf_t[j] - start <= 1.5 * window_s:
            pairs.append((i, j))
    return pairs


def gauss_newton_bias(t, w, kf_t, R_vis, pairs, bias, huber_deg, iterations):
    """Gyro bias minimising sum rho(|Log(R_vis(i,j)^T R_gyro(i,j; b))|) over the keyframe pairs, from `bias`.

    R_vis: [K,3,3] body attitudes at the keyframe times kf_t; Huber IRLS at huber_deg; preintegration Jacobian
    d r / d b = -R_j^T (C_j - C_i). Returns (bias, residual rotation vectors of the last iteration).
    Shared by the offline estimate (visual_bias) and the online one (online.OnlineGyroBias.update_visual).
    """
    from scipy.spatial.transform import Rotation

    i, j = np.array(pairs).T
    rel_v = np.einsum('nba,nbc->nac', R_vis[i], R_vis[j])
    bias = np.array(bias, dtype=np.float64)
    for _ in range(iterations):
        R, C = _attitudes(t, w, bias)
        idx = np.clip(np.searchsorted(t, kf_t, side='right') - 1, 0, len(t) - 1)  # partial last step to kf_t
        tau = kf_t - t[idx]
        Rk = R[idx] @ Rotation.from_rotvec((w[idx] - bias) * tau[:, None]).as_matrix()
        Ck = C[idx] + Rk * tau[:, None, None]
        rel_g = np.einsum('nba,nbc->nac', Rk[i], Rk[j])
        r = Rotation.from_matrix(np.einsum('nba,nbc->nac', rel_v, rel_g)).as_rotvec()
        J = -np.transpose(Rk[j], (0, 2, 1)) @ (Ck[j] - Ck[i])
        norm = np.degrees(np.linalg.norm(r, axis=1))
        weight = np.where(norm <= huber_deg, 1.0, huber_deg / np.maximum(norm, 1e-12))
        delta = -np.linalg.solve(np.einsum('n,nra,nrb->ab', weight, J, J), np.einsum('n,nra,nr->a', weight, J, r))
        bias = bias + delta
        if np.abs(delta).max() < 1e-7:
            break
    return bias, r


def initial_bias_for(mode, data_root, trajectory=None, T_C_B=None, static=None, visual=None, initial_dir=None,
                     sequence_id=None, extrinsic=None):
    """Initial constant bias for one sequence; returns (bias, info) or (None, info) for mode 'network_mean'.

    mixed (default): x, y from static_bias and z from visual_bias (static z reads low);
                     all three axes from visual_bias when the sequence has no stationary windows.
    static:          static_bias, visual_bias when there are no stationary windows.
    visual:          visual_bias.
    file:            <initial_dir>/<sequence_id>.csv (first row).
    network_mean:    left to the caller (needs network inference).
    """
    if mode not in ('mixed', 'static', 'visual', 'file', 'network_mean'):
        raise ValueError(f'Unknown alternating.gyro_bias.init.mode: {mode}')
    if mode == 'network_mean':
        return None, dict(source='network_mean')
    if mode == 'file':
        path = Path(initial_dir) / f'{sequence_id}.csv'
        return load(path)[1][0], dict(source='file', file=str(path))
    info = {}
    if mode in ('mixed', 'static'):
        static_value, info['static'] = static_bias(data_root, extrinsic=extrinsic, **(static or {}))
        if static_value is not None:
            info['static']['bias'] = static_value.tolist()
    else:
        static_value = None
    need_visual = mode == 'visual' or static_value is None or mode == 'mixed'
    if need_visual:
        if trajectory is None or T_C_B is None:
            raise ValueError('visual gyro bias needs the visual trajectory and T_C_B')
        visual_value, info['visual'] = visual_bias(data_root, trajectory, T_C_B, extrinsic=extrinsic, **(visual or {}))
        info['visual']['bias'] = visual_value.tolist()
    if static_value is None:
        bias, axes = visual_value, ['visual'] * 3
    elif mode == 'static':
        bias, axes = static_value, ['static'] * 3
    else:  # mixed
        bias, axes = np.r_[static_value[:2], visual_value[2]], ['static', 'static', 'visual']
    info['source'] = dict(zip('xyz', axes))
    if static_value is not None and need_visual:
        info['visual_minus_static_mrad_s'] = ((visual_value - static_value) * 1e3).tolist()
    return bias, info


def network_mean(velocity_csv):
    """Bias implied by a network's gyro correction: minus its time-weighted mean."""
    data = np.genfromtxt(velocity_csv, delimiter=',', names=True)
    dt = np.diff(data['time'])
    correction = np.column_stack([data[f'gyro_correction_{a}'] for a in 'xyz'])[:-1]
    return -(correction * dt[:, None]).sum(axis=0) / dt.sum()


def reset_gyro_head(source, destination):
    """Zero the final gyro-head layer so the head starts with no constant correction."""
    import torch

    payload = torch.load(source, map_location='cpu', weights_only=False)
    state = payload['model']
    last = max(int(k.split('.')[2]) for k in state if k.startswith('velocity_branch.gyro_head.'))
    for name in ('weight', 'bias'):
        state[f'velocity_branch.gyro_head.{last}.{name}'].zero_()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
