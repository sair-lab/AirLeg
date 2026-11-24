"""IMU resampler: the robot's IMU linearly interpolated at the joint stamps, as sensor_msgs/Imu, for the VO.

For robots with separate joint / IMU topics (leg_odometry.yaml robot: leg_inputs, sources.py): the samples are exactly
the IMU the leg odometry uses (ImuInterpolator: inside the IMU span, bracketing samples <= max_gap_s apart), stamped
with the joint stamps, published on leg_inputs.imu.resampled_topic. The VO then integrates the IMU on the same time
base as the leg factors of the back end. Started by visual_odometry.launch.py and the S2 preparation when
resampled_topic is set (camera yaml imu_topic: the same topic).
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu

from .recorder import load_robot
from .sources import LegInputs, leg_inputs


class ImuResampler(Node):
    def __init__(self) -> None:
        super().__init__("airslam_imu_resampler")
        self.declare_parameter("robot", "go2")
        robot = load_robot(str(self.get_parameter("robot").value))
        cfg = leg_inputs(robot)
        if not cfg or not cfg["resampled_topic"]:
            raise ValueError("imu_resampler needs leg_odometry.yaml robot: leg_inputs.imu.resampled_topic")
        self.publisher = self.create_publisher(Imu, cfg["resampled_topic"], 500)
        self.count = 0
        self.inputs = LegInputs(self, robot, self.on_sample, "")
        self.get_logger().info(f"IMU of {self.inputs.describe()} at the joint stamps -> {cfg['resampled_topic']}")

    def on_sample(self, sample) -> None:
        imu = Imu()
        imu.header.stamp.sec, imu.header.stamp.nanosec = int(sample.header.stamp.sec), int(sample.header.stamp.nanosec)
        imu.header.frame_id = "imu"
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in sample.gyroscope)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(v) for v in sample.accelerometer)
        imu.orientation_covariance[0] = -1.0
        self.publisher.publish(imu)
        self.count += 1

    def destroy_node(self) -> None:
        self.get_logger().info(f"IMU resampler: {self.count} samples, inputs {self.inputs.pairer.counts}")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuResampler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
