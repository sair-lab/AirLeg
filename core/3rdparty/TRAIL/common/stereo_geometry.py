"""Stereo geometry (intrinsics, cam0 -> cam1 extrinsic, fundamental matrix) from a recorded camera.yaml."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StereoGeometry:
    k0: np.ndarray
    k1: np.ndarray
    baseline_m: float
    r_10: np.ndarray            # Rotation from cam0 to cam1.
    t_10: np.ndarray            # Translation from cam0 to cam1, expressed in cam1/cam0 convention used for F.
    f_10: np.ndarray


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("camera/config yaml needs pyyaml installed.") from exc
        return yaml.safe_load(text) or {}


def camera_block(config: dict[str, Any], key: str) -> dict[str, Any]:
    if key in config:
        return config[key]
    legacy = {"cam0": "left_camera", "cam1": "right_camera"}[key]
    if legacy in config:
        return config[legacy]
    raise KeyError(f"Cannot find {key} in camera.yaml")


def camera_matrix(cam: dict[str, Any], prefer_projection: bool = True) -> np.ndarray:
    p = np.asarray(cam.get("p", []), dtype=np.float64)
    if prefer_projection and p.size >= 12 and abs(float(p[0])) > 1e-12:
        p = p.reshape(3, 4)
        return p[:, :3].copy()
    k = np.asarray(cam.get("k", []), dtype=np.float64)
    if k.size >= 9:
        return k.reshape(3, 3).copy()
    raise KeyError("camera block must contain CameraInfo k or p.")


def _as_transform(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 12:
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr.reshape(3, 4)
        return out
    return None


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def build_stereo_geometry(camera_yaml: Path, prefer_projection: bool = True, baseline_sign: float = 1.0) -> StereoGeometry:
    config = load_yaml_or_json(camera_yaml)
    cam0 = camera_block(config, "cam0")
    cam1 = camera_block(config, "cam1")
    k0 = camera_matrix(cam0, prefer_projection=prefer_projection)
    k1 = camera_matrix(cam1, prefer_projection=prefer_projection)

    stereo = config.get("stereo", {}) or {}
    direct_candidates = [
        stereo.get("T_cam1_cam0"),
        stereo.get("T_right_left"),
        config.get("T_cam1_cam0"),
        config.get("T_right_left"),
    ]
    inverse_candidates = [
        stereo.get("T_cam0_cam1"),
        stereo.get("T_left_right"),
        config.get("T_cam0_cam1"),
        config.get("T_left_right"),
    ]
    transform = next((t for t in (_as_transform(v) for v in direct_candidates) if t is not None), None)
    inverse_transform = next((t for t in (_as_transform(v) for v in inverse_candidates) if t is not None), None)
    if transform is not None:
        r_10 = transform[:3, :3]
        t_10 = transform[:3, 3]
        baseline_m = float(np.linalg.norm(t_10))
    elif inverse_transform is not None:
        transform = np.linalg.inv(inverse_transform)
        r_10 = transform[:3, :3]
        t_10 = transform[:3, 3]
        baseline_m = float(np.linalg.norm(t_10))
    else:
        baseline = stereo.get("baseline_m")
        if baseline is None:
            p1 = np.asarray(cam1.get("p", []), dtype=np.float64)
            if p1.size >= 12 and abs(float(p1[0])) > 1e-12:
                baseline = abs(float(-p1.reshape(3, 4)[0, 3] / p1.reshape(3, 4)[0, 0]))
        if baseline is None:
            raise KeyError("Cannot infer stereo extrinsic from T_cam1_cam0/T_cam0_cam1, stereo.baseline_m or cam1.p.")
        r_10 = np.eye(3, dtype=np.float64)
        baseline_m = abs(float(baseline))
        t_10 = np.array([float(baseline) * float(baseline_sign), 0.0, 0.0], dtype=np.float64)

    f_10 = np.linalg.inv(k1).T @ _skew(t_10) @ r_10 @ np.linalg.inv(k0)
    return StereoGeometry(k0=k0, k1=k1, baseline_m=baseline_m, r_10=r_10, t_10=t_10, f_10=f_10)
