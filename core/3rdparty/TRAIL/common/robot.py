"""Robot descriptors: the robot: section of AirSLAM2/robots/<name>/leg_odometry.yaml (model, rates, pressure settings,
recorder defaults, dataset registry).

load_robot('anymal') or load_robot('/path/to/leg_odometry.yaml') (or a stand-alone robot yaml) -> dict with absolute paths ('model', dataset 'bag_root',
'imu', relative ones resolved against the descriptor) and '_dir' (the robot directory). Pure yaml (no torch).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

try:
    from common.yaml_base import is_combined
except ImportError:           # imported as TRAIL.common.robot (recorder)
    from .yaml_base import is_combined

ROBOTS_DIR = Path(__file__).resolve().parents[4] / 'robots'
DEFAULT_ROBOT = 'go2'


def robot_file(robot: str | Path | None = None) -> Path:
    robot = DEFAULT_ROBOT if robot in (None, '') else robot
    path = Path(robot).expanduser()
    if path.suffix in ('.yaml', '.yml'):
        return path.resolve()
    return ROBOTS_DIR / str(robot) / 'leg_odometry.yaml'


@lru_cache(None)
def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if is_combined(cfg):
        cfg = cfg['robot']
    base = path.parent

    def absolute(value):        # templates such as .../{seq}/traj.txt stay templates
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (base / p).resolve())

    cfg['_dir'] = str(base)
    cfg['_file'] = str(path)
    cfg['model'] = absolute(cfg['model'])
    dataset = cfg.get('dataset') or {}
    for key in ('bag_root', 'reference', 'imu'):
        if dataset.get(key):
            dataset[key] = absolute(dataset[key])
    dataset['cameras'] = {seq: absolute(path) for seq, path in (dataset.get('cameras') or {}).items()}
    return cfg


def load_robot(robot: str | Path | None = None) -> dict:
    """Descriptor of `robot` (name under robots/ or a leg_odometry.yaml path; None = go2). Do not modify the result."""
    return _load(robot_file(robot))


def robot_model(robot: str | Path | None = None) -> Path:
    """Leg kinematics model (MuJoCo XML or URDF) of `robot`."""
    return Path(load_robot(robot)['model'])


def pressure_settings(robot: str | Path | None = None) -> dict:
    """Jacobian foot pressure settings of `robot` (method, pressure_mode, normalize_sum, damping, smooth_alpha,
    pressure_scale)."""
    return dict(load_robot(robot)['pressure'])
