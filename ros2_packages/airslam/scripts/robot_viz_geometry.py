#!/usr/bin/env python3
"""Pose helpers of robot_viz (4x4 matrices, quaternions (x, y, z, w), ROS stamps / messages)."""
import math

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(t, msg_stamp):
    msg_stamp.sec = int(math.floor(t))
    msg_stamp.nanosec = int(round((t - math.floor(t)) * 1e9)) % 1000000000
    return msg_stamp


def quat_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    """(x, y, z, w)"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def blend(T0, T1, a):
    T = np.eye(4)
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    T[:3, :3] = quat_to_mat(*slerp(mat_to_quat(T0[:3, :3]), mat_to_quat(T1[:3, :3]), a))
    return T


def pose_to_mat(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def mat_to_pose_stamped(T, frame, t):
    ps = PoseStamped()
    ps.header.frame_id = frame
    sec_to_stamp(t, ps.header.stamp)
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in T[:3, 3])
    q = mat_to_quat(T[:3, :3])
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (float(v) for v in q)
    return ps


def mat_to_transform(T, parent, child, t):
    msg = TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    if t is not None:
        sec_to_stamp(t, msg.header.stamp)
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (float(v) for v in T[:3, 3])
    q = mat_to_quat(T[:3, :3])
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = (
        float(v) for v in q)
    return msg


def interpolate_pose(times, poses, t):
    """Pose at t from sorted (times, 4x4 poses); clamped at the ends."""
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return poses[0]
    if i >= len(times):
        return poses[-1]
    t0, t1 = times[i - 1], times[i]
    return blend(poses[i - 1], poses[i], 0.0 if t1 <= t0 else (t - t0) / (t1 - t0))


def nearest(samples, t):
    """Sample (tuple starting with its time) of a time-sorted deque nearest to t."""
    times = [s[0] for s in samples]
    i = min(int(np.searchsorted(times, t)), len(times) - 1)
    if i > 0 and abs(times[i - 1] - t) < abs(times[i] - t):
        i -= 1
    return samples[i]
