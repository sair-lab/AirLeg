"""Run with ROS's Python: wait for the nodes of a run to create their input subscriptions.

usage: ros_ready.py [timeout_s] [--online]

VO: two image subscriptions (sensor_msgs/Image or CompressedImage). Recorder: its LowStateStamped subscription, or
both of its separate joint and IMU subscriptions (leg_odometry.yaml robot: leg_inputs). --online (S3): also the S3
leg odometry node (online_leg_odom in the launch, airslam_online_leg_odom standalone) with the same inputs. Every node subscribes only after its TensorRT engines
are loaded or built (VO: SuperPoint / LightGlue / FFS / YOLO in the constructor; online leg node: after the worker's
init), so the playback started after this check never runs while an engine is being built."""
import time
import sys
import rclpy

IMAGE_TYPES = ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage')
LOWSTATE, IMU = 'lowstate_republisher/msg/LowStateStamped', 'sensor_msgs/msg/Imu'
JOINT_TYPES = {'anymal_msgs/msg/AnymalState', 'sensor_msgs/msg/JointState'}

rclpy.init()
node = rclpy.create_node('s2_readiness')
try:
    args = [a for a in sys.argv[1:] if a != '--online']
    online = '--online' in sys.argv[1:]
    deadline = time.monotonic() + (float(args[0]) if args else 3600)
    subscriptions = {}
    nodes = []
    while time.monotonic() < deadline:
        subscriptions = {}
        nodes = node.get_node_names_and_namespaces()
        for name, namespace in nodes:
            if name in ('visual_odometry', 'leg_odom_recorder', 'airslam_leg_odom_recorder', 'online_leg_odom', 'airslam_online_leg_odom'):
                subscriptions[name] = node.get_subscriber_names_and_types_by_node(name, namespace)
        vo = subscriptions.get('visual_odometry', [])
        recorder = subscriptions.get('leg_odom_recorder', subscriptions.get('airslam_leg_odom_recorder', []))
        images = sum(any(t in types for t in IMAGE_TYPES) for _, types in vo)
        def inputs_ready(subs):
            kinds = {t for _, types in subs for t in types}
            return LOWSTATE in kinds or (IMU in kinds and bool(kinds & JOINT_TYPES))
        recorder_ready = inputs_ready(recorder)
        online_ready = not online or inputs_ready(subscriptions.get('online_leg_odom', subscriptions.get('airslam_online_leg_odom', [])))
        if images >= 2 and recorder_ready and online_ready:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    else:
        raise TimeoutError(f'VO and recorder subscriptions are not ready; '
                           f'discovered nodes={nodes}; subscriptions={subscriptions}')
finally:
    node.destroy_node()
    rclpy.shutdown()
