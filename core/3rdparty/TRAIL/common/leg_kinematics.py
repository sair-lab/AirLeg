"""Closed-form Unitree leg kinematics (foot position, 3x3 foot Jacobian), vectorised over samples x legs.
Model: a MuJoCo XML (Go2, default-class inheritance resolved) or a URDF with Unitree joint names (Go1); the
default is the model of the default robot (robots/go2/leg_odometry.yaml (robot:), see common/robot.py).

Replaces the per-sample MuJoCo mj_forward / mj_jacBody calls. Link offsets, body orientations and joint
axes are read from the model file, so there is one source of the geometry. Frame: the trunk body frame at the identity (as MuJoCo with the base at the
origin), i.e. MuJoCo's world frame of the existing estimators.

Usage:
    kin = LegKinematics()                       # or LegKinematics(robot_model('anymal'))
    p = kin.foot_positions(q)                   # q [..., 4, 3] -> [..., 4, 3]
    J = kin.foot_jacobians(q)                   # -> [..., 4, 3, 3], J[..., leg, :, joint] = d p / d q
    f, pressure = foot_forces(J, tau, damping)  # tau = J^T f (damped least squares), as MujocoFootForceEstimator
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

LEG_ORDER = ('FL', 'FR', 'RL', 'RR')
# Unitree LowState motor order, = recorder.DEFAULT_JOINTS (ROS side) and the joints.csv column order
LOWSTATE_LEG_ORDER = ('FR', 'FL', 'RR', 'RL')
JOINTS = ('hip', 'thigh', 'calf')

try:
    from common.robot import robot_model
except ImportError:                                    # imported as TRAIL.common.leg_kinematics
    from .robot import robot_model
DEFAULT_MODEL = robot_model()


def _vec(text, default):
    return np.array([float(x) for x in text.split()], dtype=np.float64) if text else np.array(default, dtype=np.float64)


def _quat_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _defaults(root):
    """class name -> joint attributes, with nested <default> inheritance (MuJoCo semantics)."""
    out = {}

    def walk(node, inherited):
        joint = dict(inherited)
        j = node.find('joint')
        if j is not None:
            joint.update(j.attrib)
        out[node.get('class', 'main')] = joint
        for child in node.findall('default'):
            walk(child, joint)
    for top in root.findall('default'):
        walk(top, {})
    return out


def _rpy_matrix(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class LegKinematics:
    """Leg kinematics of a Unitree quadruped from a MuJoCo XML (Go2) or a URDF (Go1)."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, base_body: str = 'base_link'):
        root = ET.parse(str(model_path)).getroot()
        if str(model_path).lower().endswith('.urdf'):
            self._load_urdf(root)
            return
        if root.find('compiler') is not None and root.find('compiler').get('angle', 'degree') != 'radian':
            raise ValueError('LegKinematics expects <compiler angle="radian">')
        defaults = _defaults(root)
        bodies = {b.get('name'): b for b in root.iter('body')}
        parent = {c.get('name'): p.get('name') for p in root.iter('body') for c in p.findall('body')}
        base = bodies[base_body]

        def childclass(name):
            while name is not None:
                cls = bodies[name].get('childclass')
                if cls:
                    return cls
                name = parent.get(name)
            return 'main'

        self.sites = {s.get('name'): _vec(s.get('pos'), [0, 0, 0]) for s in base.findall('site')}
        self.site_rotations = {}                 # site name -> rotation of the site frame in the base frame
        for site in base.findall('site'):
            if site.get('quat'):
                self.site_rotations[site.get('name')] = _quat_matrix(_vec(site.get('quat'), [1, 0, 0, 0]))
            elif site.get('euler') or site.get('axisangle') or site.get('xyaxes') or site.get('zaxis'):
                raise ValueError(f'unsupported site orientation attribute in {site.get("name")}')
            else:
                self.site_rotations[site.get('name')] = np.eye(3)
        # per leg and joint: offset of the body in its parent (pos, R), joint anchor (in the body) and axis
        self.offset = np.zeros((4, 4, 3))        # [leg, hip/thigh/calf/foot, xyz]
        self.rotation = np.tile(np.eye(3), (4, 4, 1, 1))
        self.anchor = np.zeros((4, 3, 3))
        self.axis = np.zeros((4, 3, 3))
        for i, leg in enumerate(LEG_ORDER):
            names = [f'{leg}_{j}' for j in JOINTS] + [f'{leg}_foot']
            for k, name in enumerate(names):
                body = bodies[name]
                self.offset[i, k] = _vec(body.get('pos'), [0, 0, 0])
                if body.get('quat'):
                    self.rotation[i, k] = _quat_matrix(_vec(body.get('quat'), [1, 0, 0, 0]))
                elif body.get('euler') or body.get('axisangle') or body.get('xyaxes') or body.get('zaxis'):
                    raise ValueError(f'unsupported body orientation attribute in {name}')
                if k == 3:
                    continue
                joints = body.findall('joint')
                if len(joints) != 1:
                    raise ValueError(f'{name}: expected one hinge joint')
                j = joints[0]
                attrs = dict(defaults.get(j.get('class') or childclass(name), {}))
                attrs.update(j.attrib)
                if attrs.get('type', 'hinge') != 'hinge':
                    raise ValueError(f'{name}: only hinge joints are supported')
                a = _vec(attrs.get('axis'), [0, 0, 1])
                self.axis[i, k] = a / np.linalg.norm(a)
                self.anchor[i, k] = _vec(attrs.get('pos'), [0, 0, 0])
        if parent.get(f'{LEG_ORDER[0]}_hip') != base_body:
            raise ValueError('leg chains must start at the base body')
        # Single-sample fast path (plain floats; NumPy call overhead dominates one 4x3 sample): valid for the
        # Go2 structure, hip about x, thigh and calf about y, no body rotations, joints at the body origins.
        self._finish()

    def _finish(self):
        principal = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
        self.fast = (np.allclose(self.rotation, np.eye(3)) and not self.anchor.any()
                     and all(np.allclose(self.axis[i], principal) for i in range(4)))
        self._offsets = self.offset.tolist()

    def _load_urdf(self, root):
        """URDF chain trunk -> {leg}_hip_joint -> thigh_joint -> calf_joint -> {leg}_foot_fixed (Unitree naming):
        joint origins are the child body offsets in the parent (xyz, rpy), joints rotate about `axis` at the child
        origin; the IMU site is the imu_joint origin (position and rotation in the trunk frame)."""
        joints = {j.get('name'): j for j in root.findall('joint')}

        def origin(name):
            o = joints[name].find('origin')
            xyz = _vec(o.get('xyz') if o is not None else None, [0, 0, 0])
            rpy = _vec(o.get('rpy') if o is not None else None, [0, 0, 0])
            return xyz, _rpy_matrix(rpy)
        imu = origin('imu_joint') if 'imu_joint' in joints else None
        self.sites = {'imu': imu[0]} if imu else {}
        self.site_rotations = {'imu': imu[1]} if imu else {}
        self.offset = np.zeros((4, 4, 3))
        self.rotation = np.tile(np.eye(3), (4, 4, 1, 1))
        self.anchor = np.zeros((4, 3, 3))
        self.axis = np.zeros((4, 3, 3))
        for i, leg in enumerate(LEG_ORDER):
            names = [f'{leg}_{j}_joint' for j in JOINTS] + [f'{leg}_foot_fixed']
            for k, name in enumerate(names):
                if name not in joints:
                    raise ValueError(f'URDF has no joint {name}')
                self.offset[i, k], self.rotation[i, k] = origin(name)
                if k == 3:
                    continue
                if joints[name].get('type') not in ('revolute', 'continuous'):
                    raise ValueError(f'{name}: only revolute joints are supported')
                axis = joints[name].find('axis')
                a = _vec(axis.get('xyz') if axis is not None else None, [1, 0, 0])
                self.axis[i, k] = a / np.linalg.norm(a)
        self._finish()

    @staticmethod
    def _rot(axis, angle):
        """Rodrigues, axis [..., 3] (unit), angle [...] -> [..., 3, 3]."""
        c, s = np.cos(angle)[..., None, None], np.sin(angle)[..., None, None]
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        zero = np.zeros_like(x)
        K = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)
        outer = axis[..., :, None] * axis[..., None, :]
        return c * np.eye(3) + s * K + (1 - c) * outer

    def _chain(self, q):
        """World (trunk) rotation of each joint body, joint anchors / axes and the foot position."""
        q = np.asarray(q, dtype=np.float64)
        shape = q.shape[:-2]
        R = np.broadcast_to(np.eye(3), shape + (4, 3, 3))
        p = np.zeros(shape + (4, 3))
        anchors, axes = [], []
        for k in range(3):
            p = p + np.einsum('...ij,...j->...i', R, self.offset[:, k])
            R = R @ self.rotation[:, k]
            anchors.append(p + np.einsum('...ij,...j->...i', R, self.anchor[:, k]))
            axes.append(np.einsum('...ij,...j->...i', R, self.axis[:, k]))
            # rotation about the joint anchor: the anchor is at the body origin unless joint pos is set
            R = R @ self._rot(np.broadcast_to(self.axis[:, k], shape + (4, 3)), q[..., k])
            if np.any(self.anchor[:, k]):  # keep the anchor point fixed: origin = anchor - R_after * anchor
                p = anchors[-1] - np.einsum('...ij,...j->...i', R, self.anchor[:, k])
        foot = p + np.einsum('...ij,...j->...i', R, self.offset[:, 3])
        return anchors, axes, foot

    def foot_positions(self, q):
        """q [..., 4, 3] (LEG_ORDER x hip/thigh/calf) -> foot positions [..., 4, 3] in the trunk frame."""
        return self._chain(q)[2]

    def foot_jacobians(self, q):
        """d foot / d q for each leg: [..., 4, 3, 3] (rows xyz, columns hip/thigh/calf)."""
        anchors, axes, foot = self._chain(q)
        return np.stack([np.cross(axes[k], foot - anchors[k]) for k in range(3)], axis=-1)

    def single(self, q, tau=None, damping=1e-4, pressure_mode='normal_abs', site=None):
        """One sample (q, tau [4, 3]): foot positions [4, 3] (relative to `site` if given) and, with tau, forces
        [4, 3] and pressure [4] exactly as foot_forces (damped least squares). Plain floats when self.fast."""
        if not self.fast:
            p = self.foot_positions(q) - (self.sites.get(site, np.zeros(3)) if site else 0.0)
            if tau is None:
                return p, None, None
            f, pressure = foot_forces(self.foot_jacobians(q), tau, damping, pressure_mode)
            return p, f, pressure
        q = np.asarray(q, dtype=np.float64).tolist()
        t = None if tau is None else np.asarray(tau, dtype=np.float64).tolist()
        origin = self.sites.get(site, np.zeros(3)).tolist() if site else (0.0, 0.0, 0.0)
        d2 = damping * damping
        positions, forces, pressure = [], [], []
        for l in range(4):
            q0, q1, q2 = q[l]
            c0, s0 = math.cos(q0), math.sin(q0)
            c1, s1 = math.cos(q1), math.sin(q1)
            c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
            (hx, hy, hz), (tx, ty, tz), (kx, ky, kz), (fx, fy, fz) = self._offsets[l]
            # hip frame (before Rx(q0)): calf origin c, foot f
            cx, cz = tx + c1 * kx + s1 * kz, tz - s1 * kx + c1 * kz
            ex, ez = cx + c12 * fx + s12 * fz, cz - s12 * fx + c12 * fz
            ey = ty + ky + fy
            px, py, pz = hx + ex, hy + c0 * ey - s0 * ez, hz + s0 * ey + c0 * ez
            positions.append((px - origin[0], py - origin[1], pz - origin[2]))
            if t is None:
                continue
            # columns: hip  x × (p - hip); thigh / calf  (Rx y) × (p - anchor), anchors thigh = hip + Rx t, calf = hip + Rx c
            dy, dz = py - hy, pz - hz
            j0 = (0.0, -dz, dy)
            ay, az = c0, s0                        # Rx(q0) * y
            vx, vy, vz = ex - tx, c0 * (ey - ty) - s0 * (ez - tz), s0 * (ey - ty) + c0 * (ez - tz)
            j1 = (ay * vz - az * vy, az * vx, -ay * vx)
            wx, wy, wz = ex - cx, c0 * (ey - ty - ky) - s0 * (ez - cz), s0 * (ey - ty - ky) + c0 * (ez - cz)
            j2 = (ay * wz - az * wy, az * wx, -ay * wx)
            J = (j0, j1, j2)                       # J[k] = column k (d p / d q_k)
            # (J J^T + d^2 I) f = J tau
            M = [[sum(J[k][r] * J[k][c] for k in range(3)) + (d2 if r == c else 0.0) for c in range(3)] for r in range(3)]
            b = [sum(J[k][r] * t[l][k] for k in range(3)) for r in range(3)]
            det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                   + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
            f = []
            for col in range(3):
                A = [row[:] for row in M]
                for r in range(3):
                    A[r][col] = b[r]
                f.append((A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                          + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det)
            forces.append(f)
            pressure.append(max(f[2], 0.0) if pressure_mode == 'normal_positive'
                            else max(-f[2], 0.0) if pressure_mode == 'normal_negative'
                            else math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2) if pressure_mode == 'force_norm' else abs(f[2]))
        if t is None:
            return np.array(positions), None, None
        return np.array(positions), np.array(forces), np.array(pressure)

    def site_pose(self, name):
        """(R, t) of a site frame in the trunk (base) frame, e.g. the IMU: p_base = R p_site + t."""
        if name not in self.sites:
            raise KeyError(f'model has no site {name!r}')
        return self.site_rotations.get(name, np.eye(3)), self.sites[name]


def foot_forces(jacobians, tau, damping=1e-4, pressure_mode='normal_abs'):
    """tau = J^T f per leg: damped least squares (lstsq when damping == 0), as MujocoFootForceEstimator.
    jacobians [..., 4, 3, 3], tau [..., 4, 3] -> forces [..., 4, 3], pressure [..., 4].
    pressure_mode: normal_abs |f_z|, normal_positive max(f_z, 0), normal_negative max(-f_z, 0) (ANYmal: stance f_z < 0,
    the swing-leg inertia torques give no pressure), force_norm |f|."""
    a = np.swapaxes(jacobians, -1, -2)               # A = J^T
    tau = np.asarray(tau, dtype=np.float64)
    if damping > 0.0:
        lhs = np.swapaxes(a, -1, -2) @ a + damping * damping * np.eye(3)
        rhs = np.einsum('...ji,...j->...i', a, tau)
        forces = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    else:
        forces = np.linalg.solve(a, tau[..., None])[..., 0]
    if pressure_mode == 'normal_positive':
        pressure = np.maximum(forces[..., 2], 0.0)
    elif pressure_mode == 'normal_negative':
        pressure = np.maximum(-forces[..., 2], 0.0)
    elif pressure_mode == 'force_norm':
        pressure = np.linalg.norm(forces, axis=-1)
    else:
        pressure = np.abs(forces[..., 2])
    return forces, pressure
