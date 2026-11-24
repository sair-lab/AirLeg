from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def recorder(context):
    # empty recorder_config -> the robot's defaults (AirSLAM2/robots/<robot>/leg_odometry.yaml (robot:) recorder:)
    config = LaunchConfiguration("recorder_config").perform(context)
    params = [config] if config else [{"robot": LaunchConfiguration("robot").perform(context)}]
    return [Node(
        package="airslam_leg_odom_recorder",
        executable="recorder",
        name="leg_odom_recorder",
        output="screen",
        parameters=params + [{"map_root": LaunchConfiguration("map_root")}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_root", default_value=""),
        DeclareLaunchArgument("robot", default_value="go2"),
        DeclareLaunchArgument("recorder_config", default_value=""),
        OpaqueFunction(function=recorder),
    ])
