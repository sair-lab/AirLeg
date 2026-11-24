#!/usr/bin/env python3
"""MuJoCo MJCF (robots/<robot>/model/*.xml) -> URDF for RViz, and the forward kinematics of its body tree.

Only what the visualisation needs: the body tree (links), hinge joints (revolute, axis / range from the default
classes), the visual mesh geoms with their material colours and the sites (fixed links, e.g. the IMU). Inertias and
collision geoms are left out. Meshes are referenced as file:// URIs next to the model (compiler meshdir).

  python3 mjcf_to_urdf.py robots/go2/model/model_go2.xml > go2.urdf
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _vec(text, n, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def _quat_to_matrix(q):
    """MuJoCo quaternion (w, x, y, z), not necessarily normalised."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_rpy(R):
    """URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def _axis_angle(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class MjcfModel:
    """Body tree of an MJCF file with the default classes resolved."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        root = ET.parse(self.path).getroot()
        compiler = root.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        self.meshdir = os.path.join(os.path.dirname(self.path), meshdir)
        if compiler is not None and compiler.get('angle', 'degree') != 'radian':
            raise ValueError('only angle="radian" models are supported')
        # default classes: class -> {tag: attributes}, children inherit from their parent class
        self.defaults = {}
        top = root.find('default')
        if top is not None:
            self._read_defaults(top, None, {})
        self.materials = {m.get('name'): _vec(m.get('rgba'), 4, [0.7, 0.7, 0.7, 1.0])
                          for m in root.iter('material')}
        self.meshes = {}
        for m in root.iter('mesh'):
            if m.get('file'):
                name = m.get('name') or os.path.splitext(os.path.basename(m.get('file')))[0]
                self.meshes[name] = os.path.join(self.meshdir, m.get('file'))
        world = root.find('worldbody')
        bodies = world.findall('body')
        if len(bodies) != 1:
            raise ValueError('expected one root body under worldbody')
        self.links = []    # (name, parent or None, pos, R, joint or None, visuals, sites)
        self._read_body(bodies[0], None, 'main')
        self.root = self.links[0]['name']

    def _read_defaults(self, node, parent, inherited):
        name = node.get('class', 'main')
        merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in node:
            if child.tag == 'default':
                continue
            merged.setdefault(child.tag, {}).update(child.attrib)
        self.defaults[name] = merged
        for child in node.findall('default'):
            self._read_defaults(child, name, merged)

    def _attrs(self, element, cls):
        attrs = dict(self.defaults.get(element.get('class', cls), {}).get(element.tag, {}))
        attrs.update(element.attrib)
        return attrs

    def _read_body(self, body, parent, cls):
        cls = body.get('childclass', cls)
        joints = body.findall('joint')
        joint = None
        if joints:
            if len(joints) > 1:
                raise ValueError(f'body {body.get("name")}: more than one joint')
            a = self._attrs(joints[0], cls)
            if a.get('type', 'hinge') != 'hinge':
                raise ValueError(f'joint {a.get("name")}: only hinge joints are supported')
            lim = a.get('range')
            joint = dict(name=a['name'], axis=_vec(a.get('axis'), 3, [0, 0, 1]),
                         range=_vec(lim, 2, [0, 0]) if lim else None, pos=_vec(a.get('pos'), 3, [0, 0, 0]))
        visuals = []
        for g in body.findall('geom'):
            a = self._attrs(g, cls)
            mesh = a.get('mesh')
            if a.get('type') != 'mesh' or mesh is None or a.get('group') == '3':
                continue
            rgba = self.materials.get(a.get('material'), _vec(a.get('rgba'), 4, [0.7, 0.7, 0.7, 1.0]))
            visuals.append(dict(mesh=self.meshes[mesh], pos=_vec(a.get('pos'), 3, [0, 0, 0]),
                                R=_quat_to_matrix(_vec(a.get('quat'), 4, [1, 0, 0, 0])),
                                material=a.get('material') or f'rgba_{len(visuals)}', rgba=rgba))
        sites = [dict(name=s.get('name'), pos=_vec(s.get('pos'), 3, [0, 0, 0]),
                      R=_quat_to_matrix(_vec(s.get('quat'), 4, [1, 0, 0, 0])))
                 for s in body.findall('site') if s.get('name')]
        self.links.append(dict(name=body.get('name'), parent=parent,
                               pos=_vec(body.get('pos'), 3, [0, 0, 0]) if parent else np.zeros(3),
                               R=_quat_to_matrix(_vec(body.get('quat'), 4, [1, 0, 0, 0])) if parent else np.eye(3),
                               joint=joint, visuals=visuals, sites=sites))
        for child in body.findall('body'):
            self._read_body(child, body.get('name'), cls)

    def site(self, name):
        """(link, position, rotation) of a site."""
        for link in self.links:
            for s in link['sites']:
                if s['name'] == name:
                    return link['name'], s['pos'], s['R']
        raise KeyError(name)

    def link_poses(self, q):
        """Pose (R, p) of every link in the root link frame for joint angles q {joint name: angle}."""
        poses = {}
        for link in self.links:   # parents come first (depth-first order)
            if link['parent'] is None:
                poses[link['name']] = (np.eye(3), np.zeros(3))
                continue
            Rp, pp = poses[link['parent']]
            R, p = Rp @ link['R'], pp + Rp @ link['pos']
            j = link['joint']
            if j is not None:
                Rj = _axis_angle(j['axis'], q.get(j['name'], 0.0))
                p = p + R @ (j['pos'] - Rj @ j['pos'])   # rotation about the joint anchor
                R = R @ Rj
            poses[link['name']] = (R, p)
        return poses

    def to_urdf(self, name=None):
        robot = ET.Element('robot', name=name or os.path.splitext(os.path.basename(self.path))[0])
        used = {}
        for link in self.links:
            for v in link['visuals']:
                used.setdefault(v['material'], v['rgba'])
        for mat, rgba in used.items():
            m = ET.SubElement(robot, 'material', name=mat)
            ET.SubElement(m, 'color', rgba=' '.join(f'{c:g}' for c in rgba))

        def origin(parent, pos, R):
            ET.SubElement(parent, 'origin', xyz=' '.join(f'{c:.6g}' for c in pos),
                          rpy=' '.join(f'{c:.9g}' for c in _matrix_to_rpy(R)))

        for link in self.links:
            el = ET.SubElement(robot, 'link', name=link['name'])
            for v in link['visuals']:
                vis = ET.SubElement(el, 'visual')
                origin(vis, v['pos'], v['R'])
                geo = ET.SubElement(vis, 'geometry')
                ET.SubElement(geo, 'mesh', filename='file://' + os.path.abspath(v['mesh']))
                ET.SubElement(vis, 'material', name=v['material'])
            if link['parent'] is not None:
                j = link['joint']
                if j is None:
                    joint = ET.SubElement(robot, 'joint', name=f'{link["name"]}_fixed', type='fixed')
                    origin(joint, link['pos'], link['R'])
                else:
                    joint = ET.SubElement(robot, 'joint', name=j['name'], type='revolute')
                    # URDF rotates about the child frame origin: put the child frame at the joint anchor
                    if np.any(j['pos']):
                        raise ValueError(f'joint {j["name"]}: joint pos offsets are not supported')
                    origin(joint, link['pos'], link['R'])
                    ET.SubElement(joint, 'axis', xyz=' '.join(f'{c:g}' for c in j['axis']))
                    lo, hi = j['range'] if j['range'] is not None else (-math.pi, math.pi)
                    ET.SubElement(joint, 'limit', lower=f'{lo:g}', upper=f'{hi:g}', effort='100', velocity='100')
                ET.SubElement(joint, 'parent', link=link['parent'])
                ET.SubElement(joint, 'child', link=link['name'])
            for s in link['sites']:
                ET.SubElement(robot, 'link', name=s['name'])
                joint = ET.SubElement(robot, 'joint', name=f'{s["name"]}_fixed', type='fixed')
                origin(joint, s['pos'], s['R'])
                ET.SubElement(joint, 'parent', link=link['name'])
                ET.SubElement(joint, 'child', link=s['name'])
        ET.indent(robot)
        return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding='unicode') + '\n'


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.stdout.write(MjcfModel(sys.argv[1]).to_urdf())


if __name__ == '__main__':
    main()
