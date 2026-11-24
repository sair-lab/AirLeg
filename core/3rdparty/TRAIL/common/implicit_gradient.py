"""Implicit hypergradient of the S2 visual loss through the fusion BA (alternating.implicit_gradient).

The fused rounds train the velocity heads on the visual loss U of the BA map mu*, while mu* itself was solved with the
leg-odometry edges M_e(theta) of the previous model. The implicit part of dU/dtheta goes through mu*:
    dU/dtheta |_implicit = sum_e g_e^T d delta_e / d theta,   g_e = dU/d delta_e = -J_z^T Omega J_mu w,  H w = dU/dmu,
with the measurement chart M <- [R_M Exp(phi), t_M + R_M rho], delta = [phi; rho] (map_optimization_backend
vce-adjoint, core/include/vce/adjoint.h).

U_full is the expected training visual loss under the training sampler (balance_sequences: window w of sequence s drawn
with p_w = 1 / (S N_s)); per batch the adjacent terms are a mean over the window edges and the cross terms a mean over
the used cross pairs (window_batch_loss), so in expectation
    U = rw (A_r + cw C_r) + ew (A_e + cw C_e),  A = sum_w p_w sum_{e in w} t_e / sum_w p_w n_adj(w),
    C = sum_w p_w sum_{c in w, used} t_c / sum_w p_w n_used(w)
(t = per-pair reprojection / Sampson Huber means of visual_geometry_loss_batch). Only these terms depend on mu*.

Per fused head round (ImplicitTerm): dU/dX^k per BA observation (upper_gradient) -> vce-adjoint per training sequence
(in parallel) -> g_e. Training then adds the linear term sum_e <g_e, delta_e(theta)>, linearised once per round in the
head outputs u = [v, refined gyro] of every robot step (like the bias Jacobians of IMU preintegration):
    sum_{e,k} <a_{e,k}, u_{e,k}(theta)>,  a_{e,k} = d<g_e, delta_e>/du_{e,k} at the round's first step,
which has the exact gradient there and needs no pose integration per step.
With a frozen encoder (heads) the term is summed exactly over the cached encoder features. With trainable encoder
parameters (LoRA adapters, train_network lora_heads) the features move with theta: every step evaluates the term on
lora_sample_legs random legs (features recomputed with the graph) scaled by legs / sample, an unbiased estimate whose
gradient reaches the adapters and the heads.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pypose as pp
import torch
import yaml

import train_visual_finetune as tvf


# ----------------------------------------------------------------------------- U_full and dU/dX^k
class LeafKeypoints:
    """Keypoint rows of every keyframe file on the device with the XYZ columns as leaf tensors (requires_grad)."""

    def __init__(self, device, requires_grad=True):
        self.device, self.requires_grad = device, requires_grad
        self.rows, self.xyz = {}, {}

    def get(self, path):
        key = str(path)
        if key not in self.rows:
            rows = torch.from_numpy(tvf.load_keypoints(Path(path))).to(self.device)
            self.rows[key] = rows
            self.xyz[key] = rows[:, 6:9].clone().requires_grad_(self.requires_grad)
        rows = self.rows[key]
        return torch.cat([rows[:, :6], self.xyz[key]], dim=1)

    def gradients(self):
        """(path, rows, grad) for every file whose leaf received a gradient."""
        for key, leaf in self.xyz.items():
            if leaf.grad is not None:
                yield key, self.rows[key], leaf.grad


def _with_leaf_keypoints(item, leaves):
    """Copy of a dataset window item whose kp0 / kp1 come from the leaf keypoints."""
    edges = []
    for edge in item['window_items']:
        e = dict(edge)
        e['kp0'] = leaves.get(edge['pair'].kp0_path)
        e['kp1'] = leaves.get(edge['pair'].kp1_path)
        e.pop('matches', None)
        edges.append(e)
    return dict(edges[0], window_items=edges)


def _window_terms(setup, windows, theta_grad):
    """Per-pair adjacent / cross visual terms of a list of window items (window_batch_loss without regularizers)."""
    edges = [e for w in windows for e in w['window_items']]
    with torch.set_grad_enabled(theta_grad):
        T = tvf.predict_body_delta_batch(setup.model, edges, setup.normalizer, setup.device, unique_samples=True,
                                         linear_scale=setup.calibration.scale())[0]
    T_bc = setup.calibration.body_cam()
    adjacent = tvf.visual_geometry_loss_batch(edges, T, setup.calibration, setup.cfg, setup.device, T_bc)
    cross_items, cross_T, owner = [], [], []
    offset = 0
    for k, w in enumerate(windows):
        items = w['window_items']
        cumulative = T[offset:offset + 1]
        for j in range(1, len(items)):
            cumulative = cumulative * T[offset + j:offset + j + 1]
            cross_items.append(dict(k=items[0]['k'], kp0=items[0]['kp0'], kp1=items[j]['kp1']))
            cross_T.append(cumulative)
            owner.append(k)
        offset += len(items)
    cross = (tvf.visual_geometry_loss_batch(cross_items, torch.cat(cross_T), setup.calibration, setup.cfg, setup.device,
                                            T_bc) if cross_items else None)
    return edges, adjacent, cross, owner


def sequence_weights(dataset):
    counts = {}
    for pair in dataset.pairs:
        counts[pair.sequence_id] = counts.get(pair.sequence_id, 0) + 1
    balance = bool(dataset.cfg.get('visual', {}).get('balance_sequences', True))
    return [1.0 / counts[p.sequence_id] if balance else 1.0 for p in dataset.pairs]


def visual_loss(setup, leaves=None, theta_grad=False, chunk=32, backward=False):
    """U_full (float) and its components. backward: accumulate dU/dX into the leaf keypoints; theta_grad: also return
    dU/dtheta (totals['theta_grad'], via autograd.grad). Chunk by chunk with the exact final normalisation (two passes).
    setup: model, normalizer, calibration, cfg, dataset (windows), device, params."""
    ds = setup.dataset
    weights = sequence_weights(ds)
    loss_cfg = setup.cfg.get('loss', {})
    rw, ew = float(loss_cfg.get('reprojection_weight', 1.0)), float(loss_cfg.get('epipolar_weight', 0.2))
    cw = float(setup.cfg.get('visual', {}).get('cross_frame_weight', 1.0))
    leaves = leaves or LeafKeypoints(setup.device, requires_grad=False)
    order = list(range(len(ds)))
    # pass 1: denominators over all windows (no gradients)
    den_a = den_c = 0.0
    with torch.no_grad():
        for start in range(0, len(order), chunk):
            idx = order[start:start + chunk]
            windows = [_with_leaf_keypoints(ds[i], leaves) for i in idx]
            _, adjacent, cross, owner = _window_terms(setup, windows, False)
            for k, i in enumerate(idx):
                den_a += weights[i] * len(windows[k]['window_items'])
            if cross is not None:
                used = (cross['num_matches'] > 0).cpu().numpy()
                for c, k in enumerate(owner):
                    den_c += weights[idx[k]] * float(used[c])
    totals = dict(A_r=0.0, A_e=0.0, C_r=0.0, C_e=0.0)
    theta_out = [torch.zeros_like(p) for p in setup.params]
    for start in range(0, len(order), chunk):
        idx = order[start:start + chunk]
        windows = [_with_leaf_keypoints(ds[i], leaves) for i in idx]
        with torch.set_grad_enabled(backward or theta_grad):
            edges, adjacent, cross, owner = _window_terms(setup, windows, theta_grad)
            w_edge = torch.as_tensor([weights[i] for k, i in enumerate(idx) for _ in windows[k]['window_items']],
                                     device=setup.device, dtype=adjacent['reprojection'].dtype)
            parts = dict(A_r=(w_edge * adjacent['reprojection']).sum() / den_a,
                         A_e=(w_edge * adjacent['epipolar']).sum() / den_a)
            if cross is not None and den_c > 0:
                used = (cross['num_matches'] > 0).to(adjacent['reprojection'].dtype)
                w_cross = torch.as_tensor([weights[idx[k]] for k in owner], device=setup.device, dtype=used.dtype) * used
                parts['C_r'] = (w_cross * cross['reprojection']).sum() / den_c
                parts['C_e'] = (w_cross * cross['epipolar']).sum() / den_c
            chunk_u = rw * (parts['A_r'] + cw * parts.get('C_r', 0.0)) + ew * (parts['A_e'] + cw * parts.get('C_e', 0.0))
            if theta_grad and chunk_u.requires_grad:   # dU/dtheta via autograd.grad (parameters' .grad untouched)
                grads = torch.autograd.grad(chunk_u, setup.params, retain_graph=backward, allow_unused=True)
                for acc, g in zip(theta_out, grads):
                    if g is not None:
                        acc += g
            if backward and chunk_u.requires_grad:
                chunk_u.backward()
        for k, v in parts.items():
            totals[k] += float(v)
    totals['U'] = rw * (totals['A_r'] + cw * totals['C_r']) + ew * (totals['A_e'] + cw * totals['C_e'])
    totals.update(den_adjacent=den_a, den_cross=den_c, windows=len(ds), sequences=len(set(p.sequence_id for p in ds.pairs)))
    if theta_grad:
        totals['theta_grad'] = theta_out
    return totals


def upper_gradient(setup, out_dir, chunk=32):
    """dU_full/dX^k per sequence: <out_dir>/<sequence id>/upper_grad.csv; returns the visual loss summary."""
    leaves = LeafKeypoints(setup.device, requires_grad=True)
    totals = visual_loss(setup, leaves, theta_grad=False, chunk=chunk, backward=True)
    rows_by_seq = {}
    for path, rows, grad in leaves.gradients():
        p = Path(path)
        seq = p.parent.parent.name                     # <seq>/keypoints/<time>.txt
        g = grad.detach().cpu().numpy()
        r = rows.detach().cpu().numpy()
        ok = np.isfinite(r[:, 6:9]).all(1) & (np.abs(g).sum(1) > 0)
        for row, gi in zip(r[ok], g[ok]):
            rows_by_seq.setdefault(seq, []).append((p.stem, row[2], row[3], *gi))
    out_dir = Path(out_dir)
    for seq, rows in rows_by_seq.items():
        d = out_dir / seq
        d.mkdir(parents=True, exist_ok=True)
        with (d / 'upper_grad.csv').open('w') as f:
            f.write('kf_time,uL,vL,gx,gy,gz\n')
            for t, u, v, gx, gy, gz in rows:
                f.write(f'{t},{u:.9g},{v:.9g},{gx:.12g},{gy:.12g},{gz:.12g}\n')
    totals['upper_rows'] = {s: len(r) for s, r in rows_by_seq.items()}
    return totals


# ----------------------------------------------------------------------------- leg edges and the linear term
def camera_motion(T, calibration):
    """Camera relative motion M = T_C_B T_B0B1 T_C_B^-1 of metric body motions (pypose SE3 LieTensor)."""
    T_bc = calibration.body_cam()
    return T_bc.Inv() * T * T_bc


def leg_delta(M, legs, device):
    """delta = [Log(R0^T R); R0^T (t - t0)] of M0^-1 M (M0 constant, from the vce-adjoint leg edges)."""
    M0 = pp.SE3(torch.as_tensor(np.stack([np.concatenate([l['t0'], l['q0']]) for l in legs]), dtype=M.dtype, device=device))
    d = M0.Inv() * M
    rho = d.Act(torch.zeros(d.shape[:-1] + (3,), dtype=d.dtype, device=device))
    return torch.cat([d.rotation().Log().tensor(), rho], dim=-1)


def load_legs(path):
    """vce-adjoint outputs: implicit_grad.csv (g) joined with leg_edges.csv (M0) by edge_id."""
    path = Path(path)
    g = np.genfromtxt(path / 'implicit_grad.csv', delimiter=',', names=True, skip_header=1)  # line 1: '# ...' chart note
    m = np.genfromtxt(path / 'leg_edges.csv', delimiter=',', names=True)
    gm = {int(r['edge_id']): r for r in np.atleast_1d(g)}
    legs = []
    for r in np.atleast_1d(m):
        gr = gm[int(r['edge_id'])]
        legs.append(dict(edge=int(r['edge_id']), t_i=float(r['t_i']), t_j=float(r['t_j']),
                         t0=np.array([r['tx'], r['ty'], r['tz']]), q0=np.array([r['qx'], r['qy'], r['qz'], r['qw']]),
                         g=np.array([gr['g_rx'], gr['g_ry'], gr['g_rz'], gr['g_tx'], gr['g_ty'], gr['g_tz']])))
    return legs


class _View:
    """Setup-like view of the live training objects (no copies)."""

    def __init__(self, model, normalizer, calibration, cfg, dataset, device, params):
        self.model, self.normalizer, self.calibration, self.cfg = model, normalizer, calibration, cfg
        self.dataset, self.device, self.params = dataset, device, params


class ImplicitTerm:
    """The implicit term of one fused head round.

    ig_cfg (common/alternating.py): enabled, stationarity_tol, max_reconverge_iterations, output_dir, backend,
    sequences {id: {fusion_dir, fusion_yaml}}. Construction computes dU/dX^k with the live model / dataset / config
    (identical normalisation to training) and g_e per training sequence (vce-adjoint, sequences in parallel; a sequence
    whose adjoint fails is skipped with a warning). loss() is the linearised term (see the module docstring).
    """

    def __init__(self, model, normalizer, calibration, cfg, dataset, device, params, train_root, ig_cfg):
        out = Path(ig_cfg['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        view = _View(model, normalizer, calibration, cfg, dataset, device, params)
        done = out / 'complete.json'
        if not done.exists():
            was_cache, was_training = getattr(model, 'frozen_feature_cache', None), model.velocity_branch.training
            model.frozen_feature_cache = None          # no side effects on the training cache or RNG
            model.velocity_branch.eval()
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                totals = upper_gradient(view, out / 'upper')
            model.frozen_feature_cache = was_cache
            model.velocity_branch.train(was_training)
            jobs, skipped = {}, {}
            for sid, s in ig_cfg['sequences'].items():
                adj = out / 'adjoint' / sid
                if (adj / 'implicit_grad.csv').exists():
                    continue
                if not (out / 'upper' / sid / 'upper_grad.csv').exists():
                    skipped[sid] = 'no upper gradient rows'
                    continue
                acfg = yaml.safe_load(Path(s['fusion_yaml']).read_text())
                acfg['adjoint'] = dict(fusion_dir=str(s['fusion_dir']), upper_grad_csv=str(out / 'upper' / sid / 'upper_grad.csv'),
                                       output_dir=str(adj), stationarity_tol=float(ig_cfg.get('stationarity_tol', 1e-6)),
                                       max_reconverge_iterations=int(ig_cfg.get('max_reconverge_iterations', 20)))
                path = out / f'adjoint_{sid}.yaml'
                path.write_text(yaml.safe_dump(acfg, sort_keys=False))
                log = (out / f'adjoint_{sid}.log').open('w')
                jobs[sid] = (subprocess.Popen([str(ig_cfg['backend']), 'vce-adjoint', str(path)], stdout=log,
                                              stderr=subprocess.STDOUT), log)
            for sid, (job, log) in jobs.items():
                rc = job.wait()
                log.close()
                if rc:
                    skipped[sid] = f'vce-adjoint failed ({rc}), see adjoint_{sid}.log'
            for sid, reason in skipped.items():
                print(f'[implicit_gradient] WARNING: {sid} skipped: {reason}', flush=True)
            done.write_text(json.dumps(dict(U=totals, sequences=list(ig_cfg['sequences']), skipped=skipped),
                                       indent=1, default=float))
        pair_cfg = copy.deepcopy(cfg)
        pair_cfg['visual']['window_frames'] = 2
        self.pairs = tvf.VisualPairDataset(Path(train_root), pair_cfg, normalizer, model.input_dim)
        by_seq = {}
        for i, p in enumerate(self.pairs.pairs):
            by_seq.setdefault(p.sequence_id, []).append((p.t0, p.t1, i))
        self.items, self.legs, self.unmatched = [], [], 0
        for sid in ig_cfg['sequences']:
            if not (out / 'adjoint' / sid / 'implicit_grad.csv').exists():
                continue
            for leg in load_legs(out / 'adjoint' / sid):
                k = [i for t0, t1, i in by_seq.get(sid, []) if abs(t0 - leg['t_i']) < 1e-6 and abs(t1 - leg['t_j']) < 1e-6]
                if not k:
                    self.unmatched += 1
                    continue
                self.items.append(k[0])
                self.legs.append(leg)
        self.view = _View(model, normalizer, calibration, cfg, self.pairs, device, params)
        self.g = torch.as_tensor(np.stack([l['g'] for l in self.legs]) if self.legs else np.zeros((0, 6)), device=device)
        self.coef = None
        branch = model.velocity_branch
        self.encoder_trainable = any(p.requires_grad for p in branch.encoder.parameters()) or \
            any(p.requires_grad for p in branch.input.parameters())
        self.sample = int(ig_cfg.get('lora_sample_legs', 64))
        self.rng = torch.Generator().manual_seed(int(ig_cfg.get('seed', 0)))   # leaves the training RNG untouched
        print(f'[implicit_gradient] legs={len(self.legs)} unmatched={self.unmatched} |g|={float(self.g.norm()):.4g}'
              + (f' encoder trainable: {min(self.sample, len(self.legs))} sampled legs per step' if self.encoder_trainable else ''),
              flush=True)

    def loss(self, chunk=512):
        """sum_{e,k} <a_{e,k}, u_{e,k}(theta)> + <dL/dc, c> (c: trainable calibration leaves), linearised at the first
        call of the round; float64 scalar with the graph to the heads and the calibration."""
        v = self.view
        if not self.legs:
            return torch.zeros((), dtype=torch.float64, device=v.device)
        if self.coef is None:
            self._linearize(chunk)
        if self.encoder_trainable:
            total = self._sampled()
        else:
            pred, gyro, _, _ = v.model.velocity_branch.forward_heads(self.hidden, self.gyro)
            total = (self.coef * torch.cat([pred[..., :3], gyro], dim=-1).double()).sum()
        for c, p in zip(self.cal_coef, self.cal_params):
            total = total + (c * p.double()).sum()
        return total

    def _sampled(self):
        """legs / K x the linear term of K random legs, encoder features recomputed with the graph (trainable encoder)."""
        v = self.view
        n = len(self.legs)
        sel = torch.randperm(n, generator=self.rng)[:min(self.sample, n)].tolist()
        window = int(v.model.cfg['data']['window'])
        windows, rows = [], []
        for k in sel:
            item = self.pairs[self.items[k]]
            x = item['x_context'].to(v.device).float().unsqueeze(0)
            length = int(self.offsets[k + 1] - self.offsets[k])
            windows.append(x.unfold(1, window, 1).transpose(-1, -2).contiguous()[:, :length].reshape(length, window, -1))
            rows.append(torch.arange(int(self.offsets[k]), int(self.offsets[k + 1]), device=v.device))
        rows = torch.cat(rows)
        branch = v.model.velocity_branch
        pred, gyro, _, _ = branch.forward_heads(branch.forward_features(torch.cat(windows)), self.gyro[rows])
        return (n / len(sel)) * (self.coef[rows] * torch.cat([pred[..., :3], gyro], dim=-1).double()).sum()

    def _exact(self, sel):
        """sum over the legs `sel` of <g_e, delta_e(theta)> with the integrated motions and the head outputs."""
        v = self.view
        items = [self.pairs[self.items[k]] for k in sel]
        T, velocities, _, _ = tvf.predict_body_delta_batch(v.model, items, v.normalizer, v.device,
                                                           linear_scale=v.calibration.scale())
        M = camera_motion(T, v.calibration)
        return (self.g[sel] * leg_delta(M, [self.legs[k] for k in sel], v.device).double()).sum(), items, velocities

    def _linearize(self, chunk):
        """a = d<g, delta>/du per robot step (u = [v, refined gyro], the integration inputs), the gradient of the
        trainable calibration leaves (scale, extrinsic), and the gyro inputs and (frozen encoder) encoder features of every
        step; step rows of leg k: offsets[k]:offsets[k + 1]."""
        v = self.view
        self.cal_params = [p for p in v.calibration.parameters() if p.requires_grad]
        cal_coef = [torch.zeros_like(p, dtype=torch.float64) for p in self.cal_params]
        window = int(v.model.cfg['data']['window'])
        cache = tvf.frozen_feature_cache(v.model)
        coef, hidden, gyro, lengths = [], [], [], []
        branch = v.model.velocity_branch
        was_training = branch.training
        # only d/du is needed: no adapter dropout, and no graph through a trainable encoder (heads-sized graph)
        frozen = [p for p in list(branch.encoder.parameters()) + list(branch.input.parameters()) if p.requires_grad]
        branch.eval()
        for p in frozen:
            p.requires_grad_(False)
        try:
            self._linearize_chunks(chunk, window, cache, cal_coef, coef, hidden, gyro, lengths)
        finally:
            branch.train(was_training)
            for p in frozen:
                p.requires_grad_(True)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.coef, self.gyro, self.cal_coef = torch.cat(coef), torch.cat(gyro), cal_coef
        self.hidden = torch.cat(hidden) if hidden else None

    def _linearize_chunks(self, chunk, window, cache, cal_coef, coef, hidden, gyro, lengths):
        v = self.view
        for start in range(0, len(self.items), chunk):
            value, items, velocities = self._exact(list(range(start, min(len(self.items), start + chunk))))
            grads = torch.autograd.grad(value, list(velocities) + self.cal_params, allow_unused=True)
            coef.extend(g.detach().double() for g in grads[:len(velocities)])
            for acc, g in zip(cal_coef, grads[len(velocities):]):
                if g is not None:
                    acc += g.double()
            with torch.no_grad():   # head inputs of the same steps, in predict_body_delta_batch order
                for item in items:
                    x = item['x_context'].to(v.device).float().unsqueeze(0)
                    pair = item['pair']
                    length = int(pair.robot_i1 - pair.robot_i0)
                    lengths.append(length)
                    gyro.append(tvf.recover_gyro_sequence(x[:, window - 1: window - 1 + length], v.normalizer).squeeze(0))
                    if self.encoder_trainable:   # features recomputed per step (_sampled)
                        continue
                    keys = [(pair.sequence_id, i) for i in range(pair.robot_i0, pair.robot_i1)]
                    if cache is not None and all(k in cache for k in keys):
                        hidden.append(torch.stack([cache[k] for k in keys]))
                    else:
                        w = x.unfold(1, window, 1).transpose(-1, -2).contiguous()[:, :length].reshape(length, window, -1)
                        hidden.append(v.model.velocity_branch.forward_features(w))
