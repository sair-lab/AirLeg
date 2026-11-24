"""IMU / base frame conventions of the leg odometry.

B is the base (trunk) frame of the robot model, I the IMU frame; the extrinsic T_B_I = (R_BI, r_BI) is the IMU pose
in the base frame (p_B = R_BI p_I + r_BI). The network output (linear velocity + uncertainty), the integrated pose,
foot positions, the camera extrinsic T_C_B and every angular rate / gyro bias are in B. Raw IMU data (CSV, ROS) stay in
I; the single conversion point is rotate_imu when IMU samples are read: gyro_B = R_BI gyro_I, acc_B = R_BI acc_I
(acc stays the specific force at the IMU point, the lever arm is not compensated). Where the IMU-point velocity is
needed: v_I = v_B + w_B x r_BI (base axes).

Configuration (robots/<robot>/leg_odometry.yaml (robot:)):
  imu_extrinsic: {source: model}                         # IMU site of the model (URDF imu_joint / MJCF site 'imu')
  imu_extrinsic: {xyz: [x, y, z], quat_xyzw: [...]}      # manual, e.g. an external IMU
A missing key means {source: model}. Checkpoints and dataset manifests store the resolved value ({xyz, quat_xyzw}).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ImuExtrinsic:
    """IMU pose in the base frame: rotation R_BI [3, 3] and position r_BI [3] (m)."""
    R_BI: np.ndarray = field(default_factory=lambda: np.eye(3))
    r_BI: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_dict(self) -> dict:
        return {'xyz': [float(v) for v in self.r_BI],
                'quat_xyzw': [float(v) for v in Rotation.from_matrix(self.R_BI).as_quat()]}

    @staticmethod
    def from_dict(d: dict) -> 'ImuExtrinsic':
        R = Rotation.from_quat(d.get('quat_xyzw', [0, 0, 0, 1])).as_matrix()
        return ImuExtrinsic(R, np.asarray(d.get('xyz', [0, 0, 0]), dtype=np.float64))

@lru_cache(None)
def imu_extrinsic(robot=None) -> ImuExtrinsic:
    """Resolved IMU extrinsic of `robot` (name under robots/ or leg_odometry.yaml (robot:) path) from its imu_extrinsic key."""
    from common.robot import load_robot, robot_model
    cfg = load_robot(robot).get('imu_extrinsic') or {'source': 'model'}
    if cfg.get('source', 'model' if 'xyz' not in cfg else 'manual') == 'model':
        from common.leg_kinematics import LegKinematics
        R, t = LegKinematics(robot_model(robot)).site_pose(cfg.get('site', 'imu'))
        return ImuExtrinsic(np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))
    return ImuExtrinsic.from_dict(cfg)


def config_imu_extrinsic(cfg: dict | None, robot=None) -> ImuExtrinsic:
    """Extrinsic stored in a training / checkpoint config (key 'imu_extrinsic'), else the robot's."""
    stored = (cfg or {}).get('imu_extrinsic')
    if stored:
        return ImuExtrinsic.from_dict(stored)
    return imu_extrinsic(robot if robot is not None else (cfg or {}).get('robot'))


def rotate_imu(imu, R_BI):
    """IMU columns [..., 6] = gyro xyz, acc xyz in I -> base axes (numpy array or torch tensor; copy)."""
    R = np.asarray(R_BI, dtype=np.float64)
    if np.allclose(R, np.eye(3), atol=1e-12):
        return imu.clone() if hasattr(imu, 'clone') else np.array(imu, dtype=np.float64, copy=True)
    if hasattr(imu, 'clone'):                          # torch
        import torch
        Rt = torch.as_tensor(R.T, dtype=imu.dtype, device=imu.device)
        return torch.cat([imu[..., :3] @ Rt, imu[..., 3:6] @ Rt, imu[..., 6:]], dim=-1)
    imu = np.asarray(imu, dtype=np.float64)
    return np.concatenate([imu[..., :3] @ R.T, imu[..., 3:6] @ R.T, imu[..., 6:]], axis=-1)


def rotate_vectors(v, R_BI):
    """[..., 3] vectors in I -> base axes."""
    return np.asarray(v, dtype=np.float64) @ np.asarray(R_BI, dtype=np.float64).T


def imu_point_velocity(v_B, w_B, r_BI):
    """Velocity of the IMU point in base axes: v_B + w_B x r_BI (numpy or torch, [..., 3])."""
    if hasattr(v_B, 'clone'):
        import torch
        r = torch.as_tensor(np.asarray(r_BI), dtype=v_B.dtype, device=v_B.device).expand_as(w_B)
        return v_B + torch.cross(w_B, r, dim=-1)
    return np.asarray(v_B) + np.cross(w_B, np.asarray(r_BI))
