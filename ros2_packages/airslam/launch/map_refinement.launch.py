import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_config import robot_dir  # noqa: E402


def main_node(context):
    """The node with config_path resolved (robots/<robot>/algorithm.yaml when empty)."""
    config_path = (LaunchConfiguration("config_path").perform(context)
                   or str(robot_dir(LaunchConfiguration("robot").perform(context)) / "algorithm.yaml"))
    breakpoint = LaunchConfiguration("breakpoint")
    map_root = LaunchConfiguration("map_root")
    mode = LaunchConfiguration("mode")
    model_dir = LaunchConfiguration("model_dir")
    voc_path = LaunchConfiguration("voc_path")
    return [
        Node(
            package="air_slam_ros2",
            executable="map_refinement_node",
            name="map_refinement",
            output="screen",
            parameters=[
                {
                    "breakpoint": ParameterValue(breakpoint, value_type=int),
                    "mode": mode,
                    "config_path": config_path,
                    "model_dir": model_dir,
                    "map_root": map_root,
                    "voc_path": voc_path,
                },
            ],
        ),
    ]


def generate_launch_description():
    package_share = Path(get_package_share_directory("air_slam_ros2"))

    use_rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription([
        DeclareLaunchArgument("breakpoint", default_value="0"),
        # map_refinement_node modes (apps.cc): full (with map_refinement.leg_odom.enabled: S4 = TRAIL pipeline.py s4,
        # prepare -> global -> structure on a map saved by S3 with the back-end; else visual-only) | prepare | ba |
        # structure | global
        DeclareLaunchArgument("mode", default_value="full"),
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
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(package_share / "rviz" / "map_optimization.rviz"),
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
