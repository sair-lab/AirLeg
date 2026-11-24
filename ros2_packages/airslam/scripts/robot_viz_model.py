#!/usr/bin/env python3
"""Robot description for robot_viz (shared by robot_viz.launch.py and robot_viz.py).

The model of robots/<robot>/: robots/<robot>/viz.yaml `model:` if given, else the robot: model: of leg_odometry.yaml
(URDF, or MJCF converted by mjcf_to_urdf.py). Mesh files: file://, relative to the model, or package://<pkg>/... (the
directory <pkg> next to the model, else the installed ROS package). Display mode:
  mesh      the model with its meshes (robot_state_publisher + RViz RobotModel),
  skeleton  the kinematic tree only: TF frames of every link and lines between them (no meshes needed),
  none      no model: TF frames of the base, the camera and the feet (from the leg odometry), lines base -> feet,
  auto      mesh if every mesh file of the model resolves, else skeleton (else none without a model).
Joint angles: the leg odometry input of the robot (leg_odometry.yaml robot: leg_inputs, else the Go2 LowStateStamped),
reordered to FR FL RR RL x hip thigh calf and named {leg}_{hip,thigh,calf}_joint (the joint names of the robot models
in robots/; viz.yaml `joint_names:` overrides them).
"""
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

LEGS = ('FR', 'FL', 'RR', 'RL')                     # leg-input order (recorder sources.py, TRAIL leg_kinematics.py)
DEFAULT_JOINT_NAMES = [f'{leg}_{j}_joint' for leg in LEGS for j in ('hip', 'thigh', 'calf')]


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml_with_base(path):
    """YAML with its `base:` chain (relative to the file), like TRAIL common/yaml_base.py."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop('base', None)
    if base:
        data = deep_merge(load_yaml_with_base((path.parent / base).resolve()), data)
    return data


def robot_settings(robot_dir):
    """robot: section of leg_odometry.yaml merged with the optional viz.yaml of the robot directory."""
    robot_dir = Path(robot_dir)
    leg = load_yaml_with_base(robot_dir / 'leg_odometry.yaml') if (robot_dir / 'leg_odometry.yaml').is_file() else {}
    robot = dict(leg.get('robot') or {})
    viz = {}
    if (robot_dir / 'viz.yaml').is_file():
        viz = yaml.safe_load((robot_dir / 'viz.yaml').read_text()) or {}
    model = viz.get('model') or robot.get('model')
    joints = (robot.get('leg_inputs') or {}).get('joints') or {}
    return dict(
        name=robot.get('name') or robot_dir.name,
        model=str((robot_dir / model).resolve()) if model else '',
        foot_radius=float(viz.get('foot_radius', robot.get('foot_radius', 0.02))),
        joint_names=list(viz.get('joint_names') or DEFAULT_JOINT_NAMES),
        # joint source: separate joint topic (anymal_state | joint_state, names = source joints in leg-input order)
        # or the Go2 LowStateStamped (motor_q already in leg-input order)
        joint_source=dict(message=str(joints.get('message', 'joint_state')), topic=str(joints['topic']),
                          names=[str(n) for n in joints.get('names') or []]) if joints.get('topic') else
        dict(message='lowstate', topic=str(viz.get('lowstate_topic', '/lowstate_stamped')), names=[]),
        camera_config=str(robot_dir / viz.get('camera_config', 'camera.yaml')),
    )


def _mjcf_to_urdf(path, name):
    here = os.path.dirname(os.path.realpath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from mjcf_to_urdf import MjcfModel
    return MjcfModel(path).to_urdf(name=name)


def resolve_mesh(filename, base_dir):
    """Local path of a URDF mesh filename (file://, package://, relative), or None if it does not resolve."""
    if filename.startswith('file://'):
        path = filename[len('file://'):]
    elif filename.startswith('package://'):
        pkg, _, rel = filename[len('package://'):].partition('/')
        path = os.path.join(base_dir, pkg, rel)   # the package directory next to the model (robots/<robot>/model)
        if not os.path.isfile(path):
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory(pkg), rel)
            except Exception:   # package not installed
                return None
    else:
        path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)
    return path if os.path.isfile(path) else None


def _rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class RobotDescription:
    """URDF (text) of the robot model with its kinematic tree; forward kinematics for the skeleton display."""

    def __init__(self, model_path, name='robot', mode='auto'):
        self.mode, self.urdf, self.root, self.joints, self.missing = 'none', '', None, [], []
        if not model_path or mode == 'none':
            return
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        text = (_mjcf_to_urdf(model_path, name) if model_path.endswith('.xml') and '<mujoco' in open(model_path).read()
                else open(model_path).read())
        root = ET.fromstring(text)
        base_dir = os.path.dirname(model_path)
        meshes = [m.get('filename') for m in root.iter('mesh') if m.get('filename')]
        self.missing = [m for m in meshes if resolve_mesh(m, base_dir) is None]
        if mode == 'auto':
            mode = 'mesh' if meshes and not self.missing else 'skeleton'
        if mode == 'mesh' and self.missing:
            raise FileNotFoundError(f'{len(self.missing)} mesh files do not resolve, e.g. {self.missing[0]}')
        if mode == 'skeleton':   # drop the geometry: robot_state_publisher / RViz must not look for missing meshes
            for link in root.iter('link'):
                for tag in ('visual', 'collision'):
                    for el in link.findall(tag):
                        link.remove(el)
        else:                    # relative / package:// mesh paths -> absolute file:// (RViz has no base dir and may
            for m in root.iter('mesh'):   # not find the package)
                f = m.get('filename', '')
                if f and not f.startswith('file://'):
                    m.set('filename', 'file://' + resolve_mesh(f, base_dir))
        self.mode = mode
        self.urdf = '<?xml version="1.0"?>\n' + ET.tostring(root, encoding='unicode')
        children = set()
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            ax = j.find('axis')
            self.joints.append(dict(name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                                    child=j.find('child').get('link'), p=np.array(xyz), R=_rpy_to_matrix(*rpy),
                                    axis=[float(v) for v in ax.get('xyz').split()] if ax is not None else [1, 0, 0]))
            children.add(self.joints[-1]['child'])
        links = [l.get('name') for l in root.findall('link')]
        roots = [l for l in links if l not in children]
        self.root = roots[0] if roots else (links[0] if links else None)
        # parents before children
        order, known = [], {self.root}
        pending = list(self.joints)
        while pending:
            ready = [j for j in pending if j['parent'] in known]
            if not ready:
                break
            for j in ready:
                order.append(j)
                known.add(j['child'])
                pending.remove(j)
        self.joints = order

    def link_poses(self, q):
        """{link: (R, p)} in the root link frame for joint angles q {joint name: angle} (missing joints: 0)."""
        poses = {self.root: (np.eye(3), np.zeros(3))}
        for j in self.joints:
            Rp, pp = poses[j['parent']]
            R, p = Rp @ j['R'], pp + Rp @ j['p']
            if j['type'] in ('revolute', 'continuous'):
                R = R @ _axis_angle(j['axis'], q.get(j['name'], 0.0))
            elif j['type'] == 'prismatic':
                p = p + R @ (np.asarray(j['axis']) * q.get(j['name'], 0.0))
            poses[j['child']] = (R, p)
        return poses

    def skeleton_edges(self):
        """(parent link, child link) of every joint whose child is not at the parent origin (lines of the skeleton)."""
        return [(j['parent'], j['child']) for j in self.joints if np.linalg.norm(j['p']) > 1e-4 or j['type'] != 'fixed']
