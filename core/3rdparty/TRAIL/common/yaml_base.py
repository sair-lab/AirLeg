"""YAML configs with inheritance: a file may name a parent with `base: <path>` (relative to the file).

The parent is loaded first (recursively) and the file's keys are deep-merged over it: mappings merge key by key,
everything else (scalars, lists) is replaced. Path values coming from a parent are resolved against the parent's own
directory and become absolute, so a robot overlay can live anywhere; the top file's own values are returned
unchanged (callers resolve them against the top file as before), so a file without `base:` loads exactly as with
yaml.safe_load. Config files are taken by their real path (symlinks of an install space resolved).

Same rules as the C++ loader LoadYamlWithBase (core/include/read_configs.h). Pure yaml: also imported by the ROS
launch files.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

# keys whose string values are file system paths (C++ algorithm config + TRAIL stage configs)
PATH_KEYS = frozenset({
    'model_dir', 'camera_config_path', 'voc_path', 'ffs_yaml',                      # algorithm (C++)
    'bag_root', 'cache_root', 'algorithm', 'recorder_config', 'camera_config', 'checkpoint', 'output_dir',
    'map_root', 'data_root', 'backend', 'config_file', 'sequence_cache_dir', 'model', 'robot_file',
})


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path_values(node, directory: Path, keys=PATH_KEYS):
    """Path values (by key name, at any depth) relative to `directory` -> absolute; in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'camera_configs' and isinstance(value, dict):   # bag name -> camera yaml (S2 prepare)
                for bag, path in value.items():
                    if isinstance(path, str) and path:
                        p = Path(path).expanduser()
                        value[bag] = str(p if p.is_absolute() else (directory / p).resolve())
            elif isinstance(value, (dict, list)):
                resolve_path_values(value, directory, keys)
            elif key in keys and isinstance(value, str) and value:
                path = Path(value).expanduser()
                node[key] = str(path if path.is_absolute() else (directory / path).resolve())
    elif isinstance(node, list):
        for value in node:
            resolve_path_values(value, directory, keys)
    return node


def load_yaml(path: str | Path, keys=PATH_KEYS, _seen=()) -> dict:
    """Load `path` with its `base:` chain merged (see module doc). The `base` key is removed."""
    path = Path(path).expanduser().resolve()
    if path in _seen:
        raise ValueError(f'base cycle: {" -> ".join(map(str, (*_seen, path)))}')
    cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(cfg, dict) or 'base' not in cfg:
        return cfg
    parent_path = (path.parent / Path(cfg.pop('base')).expanduser()).resolve()
    parent = resolve_path_values(load_yaml(parent_path, keys, (*_seen, path)), parent_path.parent, keys)
    return deep_merge(parent, cfg)


# sections of robots/<robot>/leg_odometry.yaml (and robots/common/leg_odometry.yaml, its base): the robot descriptor
# and the S1 / S2 / S3 configs of the leg odometry
SECTIONS = frozenset({'robot', 'simulation', 'sim_to_real', 'online'})


def is_combined(cfg) -> bool:
    """A combined leg_odometry.yaml: every top-level key (besides base) is a section."""
    return isinstance(cfg, dict) and bool(set(cfg) - {'base'}) and set(cfg) - {'base'} <= SECTIONS


def load_section(path: str | Path, section: str, keys=PATH_KEYS) -> dict:
    """`section` of a combined leg_odometry.yaml, its base: chain merged over the whole file (paths of the file itself
    stay relative to it, as in load_yaml). Any other file (a single-stage config, a run snapshot) is returned whole."""
    cfg = load_yaml(path, keys)
    if not is_combined(cfg):
        return cfg
    if section not in cfg:
        raise KeyError(f'{path} has no {section}: section')
    return cfg[section]
