"""Shared helpers of the air_slam_ros2 launch files: robot directories and algorithm configs with `base:`.

robots/ is used from the AirSLAM2 source tree (located like the recorder locates TRAIL: from this file's real path,
in the source, a symlink install or a copy install next to src/AirSLAM2), so relative paths inside the robot
configs (e.g. model_dir: models in robots/common/algorithm.yaml) stay valid.
"""
import importlib.util
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        for candidate in (ancestor, ancestor / "src" / "AirSLAM2"):
            if (candidate / "robots").is_dir() and (candidate / "core" / "3rdparty" / "TRAIL").is_dir():
                return candidate
    raise FileNotFoundError("Cannot locate the AirSLAM2 repository (robots/, core/3rdparty/TRAIL) from " + str(here))


def robot_dir(robot: str) -> Path:
    """AirSLAM2/robots/<robot>/ (a path containing '/' is taken as the robot directory itself)."""
    return Path(robot).expanduser().resolve() if "/" in robot else repo_root() / "robots" / robot


def robot_descriptor(robot: str) -> dict:
    """The robot: section of AirSLAM2/robots/<robot>/leg_odometry.yaml (TRAIL common/robot.py)."""
    import sys
    trail = repo_root() / "core" / "3rdparty" / "TRAIL"
    if str(trail) not in sys.path:
        sys.path.insert(0, str(trail))
    from common.robot import load_robot
    return load_robot(robot or None)


def load_algorithm(path: str) -> dict:
    """Algorithm YAML with its base: chain merged (TRAIL common/yaml_base.py, same rules as the C++ loader)."""
    module = repo_root() / "core" / "3rdparty" / "TRAIL" / "common" / "yaml_base.py"
    spec = importlib.util.spec_from_file_location("airslam_yaml_base", module)
    yaml_base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(yaml_base)
    return yaml_base.load_yaml(path) or {}
