import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_config import load_algorithm, robot_descriptor, robot_dir  # noqa: E402


def algorithm_path(context):
    """config_path, or robots/<robot>/algorithm.yaml when empty."""
    return (LaunchConfiguration("config_path").perform(context)
            or str(robot_dir(LaunchConfiguration("robot").perform(context)) / "algorithm.yaml"))


def imu_resampler(context):
    """The robot's IMU at the joint stamps for the VO (leg_odometry.yaml robot: leg_inputs.imu.resampled_topic; the camera
    yaml imu_topic is that topic), in every mode: the VO uses the IMU also without the leg fusion."""
    robot = LaunchConfiguration("robot").perform(context)
    inputs = robot_descriptor(robot).get("leg_inputs") or {}
    if not (inputs.get("imu") or {}).get("resampled_topic"):
        return []
    return [Node(package="airslam_leg_odom_recorder", executable="imu_resampler", name="imu_resampler",
                 output="screen", parameters=[{"robot": robot}])]


def online_leg_odometry(context):
    """S3: start the online leg-odometry node unless the resolved mode is visual_only."""
    config = load_algorithm(algorithm_path(context))
    vo = config.get("visual_odometry", {})
    online = vo.get("online", {}) or {}
    cli_mode = LaunchConfiguration("online_mode").perform(context)
    mode = cli_mode or online.get("mode", "visual_only")
    if mode == "visual_only":
        return []
    publisher = config.get("ros_publisher") or vo.get("ros_publisher", {})
    path_topic = publisher.get("path_topic", "/AirSLAM/odometry")
    # online.leg_lora: online fine-tuning supervised by the back-end keyframe path (<path_topic>/backend).
    # An explicit online_mode:=fusion_only on the command line wins (no fine-tuning).
    if online.get("leg_lora", False) and cli_mode != "fusion_only":
        mode = "online_adapt"
        if (online.get("backend") or {}).get("enabled", False):
            path_topic += "/backend"
        else:
            print("[online_leg_odometry] online.leg_lora without online.backend.enabled: "
                  "fine-tuning supervised by the front-end path")
    parameters = {
        "mode": mode,
        "robot": LaunchConfiguration("robot").perform(context),
        "worker_python": LaunchConfiguration("leg_worker_python").perform(context),
        "leg_topic": online.get("leg_topic", "/airslam/leg_odom"),
        "path_topic": path_topic,
    }
    leg_config = LaunchConfiguration("leg_config").perform(context)
    if leg_config:
        parameters["config"] = leg_config
    return [Node(
        package="airslam_leg_odom_recorder",
        executable="online_leg_odom",
        name="online_leg_odom",
        output="screen",
        parameters=[parameters],
    )]


def nodes(context):
    """VO node and recorder with the resolved robot configs."""
    robot = LaunchConfiguration("robot").perform(context)
    # recorder parameters: an explicit recorder_config file, else the robot (its leg_odometry.yaml (robot:) recorder: defaults)
    recorder_config = LaunchConfiguration("recorder_config").perform(context)
    recorder_params = [recorder_config] if recorder_config else [
        {"robot": str(robot_dir(robot) / "leg_odometry.yaml") if "/" in robot else robot}]
    return [
        Node(
            package="air_slam_ros2",
            executable="visual_odometry_node",
            name="visual_odometry",
            output="screen",
            parameters=[
                {
                    "config_path": algorithm_path(context),
                    "model_dir": LaunchConfiguration("model_dir"),
                    "camera_config_path": LaunchConfiguration("camera_config_path"),
                    "saving_dir": LaunchConfiguration("saving_dir"),
                    "left_image_topic": LaunchConfiguration("left_image_topic"),
                    "right_image_topic": LaunchConfiguration("right_image_topic"),
                    "imu_topic": LaunchConfiguration("imu_topic"),
                    "imu_message": LaunchConfiguration("imu_message"),
                    # a string parameter of the node ("true" / "false" / ""): launch_ros would pass "true" as a bool
                    "use_imu": ParameterValue(LaunchConfiguration("use_imu"), value_type=str),
                    "sync_tolerance_sec": LaunchConfiguration("sync_tolerance_sec"),
                    "online_mode": LaunchConfiguration("online_mode"),
                    "dynamic_mask_dir": LaunchConfiguration("dynamic_mask_dir"),
                },
            ],
        ),
        Node(
            package="airslam_leg_odom_recorder",
            executable="recorder",
            name="leg_odom_recorder",
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_leg_odom")),
            parameters=recorder_params + [{"map_root": LaunchConfiguration("saving_dir")}],
        ),
    ]


def viewer(context):
    """RViz: robot_viz.launch.py (listens only), or plain RViz with rviz_config when given."""
    if LaunchConfiguration("rviz").perform(context) != "true":
        return []
    config = LaunchConfiguration("rviz_config").perform(context)
    if config:
        return [Node(package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", config], output="screen")]
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / "robot_viz.launch.py")),
        launch_arguments={"robot": LaunchConfiguration("robot").perform(context),
                          "show_cloud": LaunchConfiguration("show_cloud").perform(context),
                          "s3": "false"}.items())]


def generate_launch_description():
    package_share = Path(get_package_share_directory("air_slam_ros2"))

    return LaunchDescription([
        # robot: robots/<robot>/ supplies the algorithm config (algorithm.yaml, base: robots/common), the camera,
        # the recorder config and the S3 leg config (leg_odometry.yaml (online:)); explicit paths below override them.
        DeclareLaunchArgument("robot", default_value="go2"),
        DeclareLaunchArgument("config_path", default_value=""),
        DeclareLaunchArgument("model_dir", default_value=""),
        DeclareLaunchArgument("camera_config_path", default_value=""),
        DeclareLaunchArgument("saving_dir", default_value=""),
        DeclareLaunchArgument("left_image_topic", default_value=""),
        DeclareLaunchArgument("right_image_topic", default_value=""),
        DeclareLaunchArgument("imu_topic", default_value=""),
        DeclareLaunchArgument("use_imu", default_value=""),       # true | false; empty: algorithm visual_odometry.use_imu
        DeclareLaunchArgument("imu_message", default_value=""),   # imu | lowstate; empty: camera yaml imu_message
        DeclareLaunchArgument("sync_tolerance_sec", default_value="0.01"),
        DeclareLaunchArgument("record_leg_odom", default_value="true"),
        DeclareLaunchArgument("recorder_config", default_value=""),
        # S3: online_adapt | fusion_only | visual_only; empty -> visual_odometry.online.mode.
        DeclareLaunchArgument("online_mode", default_value=""),
        # YOLO dynamic-object masks of the played bag (<dir>/{left,right}/<stamp_ns>.png); empty disables
        DeclareLaunchArgument("dynamic_mask_dir", default_value=""),
        DeclareLaunchArgument("leg_config", default_value=""),
        DeclareLaunchArgument(
            "leg_worker_python",
            default_value=os.environ.get("AIRSLAM_TORCH_PYTHON", "python3"),
        ),
        # rviz:=true starts the robot-agnostic view robot_viz.launch.py (robot model / skeleton, smoothed S3 estimate,
        # trajectory, footprints; map points with show_cloud:=true); rviz_config:=<file> plain RViz with that config
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        DeclareLaunchArgument("show_cloud", default_value="false"),
        OpaqueFunction(function=nodes),
        OpaqueFunction(function=imu_resampler),
        OpaqueFunction(function=online_leg_odometry),
        OpaqueFunction(function=viewer),
    ])
