"""TensorRT inference of the S3 leg network (latency plan C), with refit for the online LoRA updates.

The network is exported to ONNX from a CPU copy of the S2 model (MHA fast path off: its fused op has no ONNX export):
fusion_only exports forward_with_gyro; online_adapt with linear_only exports common.online.LinearOnlyInference (shared
prefix, merged-weight copies of the adapted layers). The merged weights are found in the exported graph by value (they
are set to unique random values for the export) and renamed lora_merged.<key>, so a refit can address them; MatMul
weights may come out transposed (W^T), recorded per weight. Engines are cached by checkpoint / config / TensorRT / GPU.

TrtSwitch loads (or builds and caches) the engine before the worker reports ready, then, in the background while PyTorch
LoRA state, compares it with PyTorch on the newest windows (max |linear velocity| difference <= check_tol) and only
then takes over. Two engine instances alternate: a new LoRA state is refitted into the idle one in the background and
swapped in, so a refit (tens to hundreds of ms) never blocks the inference. Any failure is logged and PyTorch stays.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parametrize

import tensorrt as trt   # ImportError -> StreamingInference.enable_trt keeps PyTorch

from common.model import LoRAWeight
from common.online import LinearOnlyInference, fresh_online_model, linear_only_adapters

OUTPUTS = ('linear', 'refined', 'log_var', 'dg')
CODE_VERSION = 'leg_trt-1'


class _Logger(trt.ILogger):
    """TensorRT messages to stderr (never stdout: the worker's protocol stream)."""

    def __init__(self, level=trt.ILogger.WARNING):
        trt.ILogger.__init__(self)
        self.level = level

    def log(self, severity, msg):
        if severity <= self.level:
            print(f'[TensorRT] {msg}', file=sys.stderr, flush=True)


class _Plain(torch.nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch

    def forward(self, x, gyro):
        return self.branch.forward_with_gyro(x, gyro)


def adapted_keys(branch):
    """[(key, module, name)] of the parametrized (LoRA) weights from the first adapted encoder layer on; key =
    '<k>.<module path>.<name>' with k counted from that layer (LinearOnlyInference.merged[k] state_dict keys)."""
    layers = list(branch.encoder.layers)
    first = next(i for i, layer in enumerate(layers) if any(parametrize.is_parametrized(m) for m in layer.modules()))
    out = []
    for k, layer in enumerate(layers[first:]):
        for path, module in layer.named_modules():
            if parametrize.is_parametrized(module):
                for name in module.parametrizations:
                    out.append((f'{k}.{path}.{name}' if path else f'{k}.{name}', module, name))
    return out


def export_onnx(inference, lora, path):
    """ONNX of the inference network; returns {onnx initializer name: (key, transposed)} of the refittable weights."""
    import onnx
    from onnx import numpy_helper
    model, _, _ = fresh_online_model(inference.checkpoint, torch.device('cpu'), inference.lora_cfg if lora else None)
    branch = model.velocity_branch
    marks = {}
    if lora:
        linear_only_adapters(branch)
        module = LinearOnlyInference(branch).eval()
        gen = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for key, _, _ in adapted_keys(branch):
                k, rest = key.split('.', 1)
                target = module.merged[int(k)].state_dict()[rest]
                real = target.clone()
                target.copy_(torch.rand(target.shape, generator=gen) * 2 - 1)   # unique values to find it after export
                marks[key] = (target.numpy().copy(), real.numpy())
    else:
        module = _Plain(branch).eval()
    x = torch.randn(2, inference.window, model.input_dim)
    gyro = torch.randn(2, 3)
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(module, (x, gyro), str(path), input_names=['x', 'gyro'], output_names=list(OUTPUTS),
                              dynamic_axes={n: {0: 'batch'} for n in ('x', 'gyro', *OUTPUTS)}, opset_version=17,
                              do_constant_folding=True, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    refit = {}
    if marks:
        graph = onnx.load(str(path))
        inits = {init.name: init for init in graph.graph.initializer}
        values = {name: numpy_helper.to_array(init) for name, init in inits.items()}
        for key, (mark, real) in marks.items():
            hit = [(n, False) for n, v in values.items() if v.shape == mark.shape and np.array_equal(v, mark)]
            hit += [(n, True) for n, v in values.items() if v.shape == mark.T.shape and np.array_equal(v, mark.T)]
            if len(hit) != 1:
                raise RuntimeError(f'refittable weight {key} not found once in the ONNX graph ({len(hit)} matches)')
            old, transposed = hit[0]
            new = f'lora_merged.{key}'
            inits[old].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(real.T if transposed else real), new))
            for node in graph.graph.node:
                node.input[:] = [new if i == old else i for i in node.input]
            refit[new] = (key, transposed)
        onnx.save(graph, str(path))
    return refit


def build_engine(onnx_path, precision, max_batch, window, input_dim, refit):
    logger = _Logger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)   # fp32 means fp32 (TF32 matmuls differ by ~1e-3)
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    if refit:
        config.set_flag(trt.BuilderFlag.REFIT)
    profile = builder.create_optimization_profile()
    opt = min(4, max_batch)
    profile.set_shape('x', (1, window, input_dim), (opt, window, input_dim), (max_batch, window, input_dim))
    profile.set_shape('gyro', (1, 3), (opt, 3), (max_batch, 3))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT engine build failed')
    return bytes(plan)


class TrtSwitch:
    def __init__(self, inference, cfg, linear_only):
        self.inf = inference
        self.precision = str(cfg.get('precision', 'fp32'))
        self.max_batch = int(cfg.get('max_batch', 64))
        self.cache_dir = Path(os.path.expanduser(str(cfg.get('cache_dir', '~/.cache/trail_leg_trt'))))
        self.check_tol = float(cfg.get('check_tol', 5.0e-3))
        branch = inference.model.velocity_branch
        self.lora = any(isinstance(m, LoRAWeight) for m in branch.modules())
        if self.lora and not linear_only:
            raise ValueError('TensorRT leg inference supports LoRA with train.linear_only only')
        self.active = False
        self.version = 0                  # LoRA state of the active engine
        self.lock = threading.Lock()      # active engine index / version
        self.engines, self.contexts, self.elocks, self.evers = [], [], [], []
        self.active_index = 0
        self.refit_map = {}
        self.pending = 0
        self.cond = threading.Condition()
        self.stats = dict(refits=0, refit_ms=[], swaps=0)
        self.check_err = float('nan')
        self.logger = _Logger()
        # the engine is loaded (or built and cached) here, before the worker reports ready: no build during a run;
        # only the switch-over check (it needs live windows) runs in the background
        self.setup_s, self.source = self._load()
        threading.Thread(target=self._activate, daemon=True, name='leg_trt_activate').start()

    # ------------------------------------------------------------------ setup
    def _key(self):
        h = hashlib.sha256()
        h.update(Path(self.inf.checkpoint).read_bytes())
        h.update(json.dumps(dict(lora=self.inf.lora_cfg if self.lora else None, precision=self.precision,
                                 max_batch=self.max_batch, trt=trt.__version__, code=CODE_VERSION,
                                 gpu=torch.cuda.get_device_name(self.inf.device)), sort_keys=True).encode())
        return h.hexdigest()[:24]

    def _load(self):
        """Deserialise the cached engine of this checkpoint / config / TensorRT / GPU, building and caching it first if
        missing; returns (seconds, 'cache' | 'built'). Raises on failure (the caller keeps PyTorch)."""
        t0 = time.perf_counter()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._key()
        plan_path, meta_path = self.cache_dir / f'{key}.plan', self.cache_dir / f'{key}.json'
        source = 'cache'
        if plan_path.exists() and meta_path.exists():
            plan = plan_path.read_bytes()
            self.refit_map = {n: tuple(v) for n, v in json.loads(meta_path.read_text()).items()}
        else:
            source = 'built'
            with tempfile.TemporaryDirectory() as tmp:
                onnx_path = Path(tmp) / 'leg.onnx'
                self.refit_map = export_onnx(self.inf, self.lora, onnx_path)
                plan = build_engine(onnx_path, self.precision, self.max_batch, self.inf.window,
                                    self.inf.model.input_dim, self.lora)
            for path, data in ((meta_path, json.dumps(self.refit_map).encode()), (plan_path, plan)):
                fd, tmp = tempfile.mkstemp(dir=self.cache_dir, prefix=path.name + '.')
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                os.replace(tmp, path)   # atomic: concurrent workers may build the same engine
        runtime = trt.Runtime(self.logger)
        for _ in range(2 if self.lora else 1):
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError('engine deserialisation failed')
            self.engines.append(engine)
            self.contexts.append(engine.create_execution_context())
            self.elocks.append(threading.Lock())
            self.evers.append(0)
        self.runtime = runtime
        return time.perf_counter() - t0, source

    def _activate(self):
        setup_s, source = self.setup_s, self.source
        try:
            err = self._switch_over()
            if err is None:
                return
            self.inf.log(f'TensorRT leg inference ({self.precision}, engine {source} in {setup_s:.1f} s): '
                         f'switch-over check failed ({err}), PyTorch kept for this run')
            return
        except Exception as error:
            import traceback
            self.inf.log('TensorRT leg inference failed, PyTorch kept: '
                         + traceback.format_exc(limit=4).replace('\n', ' | '))
            return
        finally:
            if self.active:
                self.inf.log(f'TensorRT leg inference active: {self.precision}, engine {source}, '
                             f'setup {setup_s:.1f} s, max |linear| diff {self.check_err:.2e}, '
                             f'refittable weights {len(self.refit_map)}, LoRA state {self.version}')
                if self.lora:
                    threading.Thread(target=self._refit_loop, daemon=True, name='leg_trt_refit').start()

    def _switch_over(self):
        """Refit both engines to the current LoRA state, compare with PyTorch on the newest windows, activate.
        Returns None on success, else the reason."""
        window = self.inf.window
        while True:   # wait for enough samples
            times, rows, _ = self.inf.ring.tail(window + 31)
            if len(rows) >= window + 31:
                break
            time.sleep(0.5)
        for _ in range(20):
            v = self.inf.torch_version
            if self.lora:
                weights = self._weights()
                if weights is None:
                    continue
                v, arrays = weights
                for k in range(len(self.engines)):
                    with self.elocks[k]:
                        self._refit(k, arrays)
                        self.evers[k] = v
            times, rows, _ = self.inf.ring.tail(window + 31)
            from common.online import normalized
            from stage4 import sliding_windows
            x, xn = normalized(rows, np.zeros(3, np.float32), self.inf.normalizer)
            windows = sliding_windows(torch.from_numpy(xn).to(self.inf.device), window)
            gyro = torch.from_numpy(x[window - 1:, :3]).to(self.inf.device)
            with self.inf.lock, torch.no_grad():
                if self.inf.torch_version != v:
                    continue   # a new LoRA state arrived meanwhile: refit again
                if self.inf.shared is not None:
                    ref = self.inf.shared(windows, gyro)
                else:
                    ref = self.inf.model.velocity_branch.forward_with_gyro(windows, gyro)
                out = self._infer(0, windows, gyro)
                self.check_err = float((out[0] - ref[0]).abs().max())
                if not np.isfinite(self.check_err) or self.check_err > self.check_tol:
                    return f'max |linear| diff {self.check_err:.2e} > {self.check_tol:g}'
                self.version = v
                self.active_index = 0
                self.active = True
                return None
        return 'LoRA state changed during every check'

    # ------------------------------------------------------------------ refit
    def _weights(self):
        """(torch_version, {initializer name: float32 array}) of the current merged weights."""
        keys = {key: (module, name) for key, module, name in adapted_keys(self.inf.model.velocity_branch)}
        with self.inf.lock, torch.no_grad():
            v = self.inf.torch_version
            tensors = {n: getattr(*keys[key]).detach().float().cpu() for n, (key, _) in self.refit_map.items()}
        arrays = {}
        for n, t in tensors.items():
            a = t.numpy()
            arrays[n] = np.ascontiguousarray(a.T if self.refit_map[n][1] else a)
        return v, arrays

    def _refit(self, k, arrays):
        t0 = time.perf_counter()
        refitter = trt.Refitter(self.engines[k], self.logger)
        for name, array in arrays.items():
            if not refitter.set_named_weights(name, trt.Weights(array)):
                raise RuntimeError(f'refit: set_named_weights({name}) failed')
        missing = refitter.get_missing_weights()
        if missing:
            raise RuntimeError(f'refit: missing weights {list(missing)[:5]}')
        if not refitter.refit_cuda_engine():
            raise RuntimeError('refit_cuda_engine failed')
        self.stats['refits'] += 1
        self.stats['refit_ms'].append(1000.0 * (time.perf_counter() - t0))

    def request_refit(self, version):
        with self.cond:
            self.pending = max(self.pending, int(version))
            self.cond.notify()

    def _refit_loop(self):
        while True:
            with self.cond:
                while self.pending <= self.version:
                    self.cond.wait()
            try:
                v, arrays = self._weights()   # newest state only (intermediate ones are skipped)
                idle = 1 - self.active_index
                with self.elocks[idle]:
                    self._refit(idle, arrays)
                    self.evers[idle] = v
                with self.lock:
                    self.active_index, self.version = idle, v
                    self.stats['swaps'] += 1
                other = 1 - idle
                with self.elocks[other]:   # waits for an inference still running on it
                    self._refit(other, arrays)
                    self.evers[other] = v
            except Exception as error:
                self.active = False
                self.inf.log(f'TensorRT refit failed ({error!r}): PyTorch for the rest of the run')
                return

    # ------------------------------------------------------------------ inference
    def _infer(self, k, x, gyro):
        ctx = self.contexts[k]
        if getattr(self, 'stream', None) is None:
            self.stream = torch.cuda.Stream(self.inf.device)   # not the default stream (TensorRT would add syncs)
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.inf.device))   # inputs prepared on the current stream
        parts = {n: [] for n in OUTPUTS}
        for i in range(0, len(x), self.max_batch):
            xc = x[i:i + self.max_batch].contiguous()
            gc = gyro[i:i + self.max_batch].contiguous()
            ctx.set_input_shape('x', tuple(xc.shape))
            ctx.set_input_shape('gyro', tuple(gc.shape))
            ctx.set_tensor_address('x', xc.data_ptr())
            ctx.set_tensor_address('gyro', gc.data_ptr())
            outs = {}
            for n in OUTPUTS:
                outs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=torch.float32, device=self.inf.device)
                ctx.set_tensor_address(n, outs[n].data_ptr())
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            for n in OUTPUTS:
                parts[n].append(outs[n])
            keep = (xc, gc)   # inputs alive until the stream finished
        stream.synchronize()   # outputs ready for the current stream / the copy-out
        del keep
        return tuple(torch.cat(parts[n]) if len(parts[n]) > 1 else parts[n][0] for n in OUTPUTS)

    def infer(self, x, gyro):
        with self.lock:
            k = self.active_index
        with self.elocks[k]:
            return self._infer(k, x, gyro)
