import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_config import robot_dir  # noqa: E402


def main_node(context):
    """The node with config_path resolved (robots/<robot>/algorithm.yaml when empty)."""
    config_path = (LaunchConfiguration("config_path").perform(context)
                   or str(robot_dir(LaunchConfiguration("robot").perform(context)) / "algorithm.yaml"))
    camera_config_path = LaunchConfiguration("camera_config_path")
    dataroot = LaunchConfiguration("dataroot")
    map_root = LaunchConfiguration("map_root")
    model_dir = LaunchConfiguration("model_dir")
    traj_path = LaunchConfiguration("traj_path")
    voc_path = LaunchConfiguration("voc_path")
    return [
        Node(
            package="air_slam_ros2",
            executable="relocalization_node",
            name="relocalization",
            output="screen",
            parameters=[
                {
                    "config_path": config_path,
                    "model_dir": model_dir,
                    "map_root": map_root,
                    "voc_path": voc_path,
                    "traj_path": traj_path,
                    "dataroot": dataroot,
                    "camera_config_path": camera_config_path,
                },
            ],
        ),
    ]


def generate_launch_description():
    package_share = Path(get_package_share_directory("air_slam_ros2"))

    use_rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription([
        # robot: robots/<robot>/algorithm.yaml (base: robots/common) unless config_path is given
        DeclareLaunchArgument("robot", default_value="go2"),
        DeclareLaunchArgument("config_path", default_value=""),
        DeclareLaunchArgument(
            "model_dir",
            default_value="",
        ),
        DeclareLaunchArgument("map_root", default_value=""),
        DeclareLaunchArgument(
            "voc_path",
            default_value="",
        ),
        DeclareLaunchArgument("traj_path", default_value=""),
        DeclareLaunchArgument("dataroot", default_value=""),
        DeclareLaunchArgument(
            "camera_config_path",
            default_value="",
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(package_share / "rviz" / "relocalization.rviz"),
        ),
        OpaqueFunction(function=main_node),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
            condition=IfCondition(use_rviz),
        ),
    ])
