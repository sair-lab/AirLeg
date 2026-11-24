"""Robot-agnostic RViz view of an S3 run: robot model (or TF skeleton), smoothed S3 estimate, trajectory, footprints;
sparse map points optional. Runs next to the usual S3 launch (visual_odometry.launch.py / a bag) and only listens.

  ros2 launch air_slam_ros2 robot_viz.launch.py robot:=go2
  ros2 launch air_slam_ros2 robot_viz.launch.py robot:=anymal show_cloud:=true
  ... model:=auto|mesh|skeleton|none   # auto: meshes if all mesh files of the robot model resolve, else skeleton
  ... robot_dir:=<dir>                 # a robot directory other than robots/<robot> (leg_odometry.yaml [+ viz.yaml])
  ... rviz:=false                      # only the TF / marker publishers (e.g. on the robot, RViz elsewhere)
  ... s3:=true [bag:=<dir> play_delay:=15 rate:=1.0]   # also start S3 (visual_odometry.launch.py, its RViz off) and
                                       # play the bag after play_delay s (TensorRT engines, leg worker);
                                       # record_leg_odom:=true saving_dir:=<dir> also records the leg odometry

The robot model is robots/<robot>/viz.yaml `model:` if given, else leg_odometry.yaml robot: model: (URDF or MJCF);
meshes go next to it (e.g. robots/go2/model/assets). Without meshes the links are drawn as TF frames and lines.
"""
import os
import sys
import tempfile
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_config import robot_dir as find_robot_dir  # noqa: E402


def arg(context, name):
    return LaunchConfiguration(name).perform(context)


def scripts_dir():
    """Installed lib/air_slam_ros2 (robot_viz.py, robot_viz_model.py), else the source scripts/ next to launch/."""
    try:
        d = Path(get_package_prefix('air_slam_ros2')) / 'lib' / 'air_slam_ros2'
        if (d / 'robot_viz_model.py').is_file():
            return d
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent / 'scripts'


def rviz_config(mode, show_cloud, base_frame):
    """robot_viz.rviz with the displays of the model mode and the map clouds switched on / off (temporary file)."""
    try:
        src = Path(get_package_share_directory('air_slam_ros2')) / 'rviz' / 'robot_viz.rviz'
    except Exception:
        src = Path(__file__).resolve().parent.parent / 'rviz' / 'robot_viz.rviz'
    cfg = yaml.safe_load(src.read_text())
    vm = cfg['Visualization Manager']
    enabled = {'Robot model': mode == 'mesh', 'Skeleton': mode != 'mesh', 'TF': mode != 'mesh',
               'Map points': show_cloud}
    for disp in vm['Displays']:
        if disp.get('Name') in enabled:
            disp['Enabled'] = disp['Value'] = enabled[disp['Name']]
    vm['Views']['Current']['Target Frame'] = base_frame
    out = tempfile.NamedTemporaryFile('w', prefix='robot_viz_', suffix='.rviz', delete=False)
    yaml.safe_dump(cfg, out, sort_keys=False, allow_unicode=True)
    out.close()
    return out.name


def view(context):
    sdir = scripts_dir()
    if str(sdir) not in sys.path:
        sys.path.insert(0, str(sdir))
    from robot_viz_model import RobotDescription, robot_settings
    rdir = arg(context, 'robot_dir') or str(find_robot_dir(arg(context, 'robot')))
    settings = robot_settings(rdir)
    desc = RobotDescription(settings['model'], settings['name'], arg(context, 'model'))
    base_frame = desc.root or 'base_link'
    actions = [LogInfo(msg=f'[robot_viz] {settings["name"]}: model {os.path.basename(settings["model"]) or "-"} as '
                           f'{desc.mode}' + (f' ({len(desc.missing)} mesh files missing)' if desc.missing else ''))]
    if desc.mode != 'none':
        actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher',
                            name='robot_viz_state_publisher', output='log',
                            parameters=[{'robot_description': desc.urdf, 'publish_frequency': 100.0}]))
    params = {'robot_dir': rdir, 'model_mode': desc.mode, 'delay': float(arg(context, 'delay')),
              'path_mode': arg(context, 'path_mode'), 'correction_tau': float(arg(context, 'correction_tau')),
              'ground_follow': arg(context, 'ground_follow') == 'true'}
    actions.append(Node(package='air_slam_ros2', executable='robot_viz.py', name='robot_viz', output='screen',
                        parameters=[params]))
    if arg(context, 'rviz') == 'true':
        actions.append(Node(package='rviz2', executable='rviz2', name='robot_viz_rviz', output='log',
                            arguments=['-d', rviz_config(desc.mode, arg(context, 'show_cloud') == 'true', base_frame)]))
    bag = arg(context, 'bag')
    if bag:
        if not os.path.exists(bag):
            raise RuntimeError(f'bag {bag} does not exist')
        actions.append(TimerAction(period=float(arg(context, 'play_delay')), actions=[
            LogInfo(msg=f'[robot_viz] playing {bag}'),
            ExecuteProcess(cmd=['ros2', 'bag', 'play', bag, '--rate', arg(context, 'rate')], output='screen')]))
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory('air_slam_ros2'))
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='go2'),
        DeclareLaunchArgument('robot_dir', default_value=''),
        DeclareLaunchArgument('model', default_value='auto'),          # auto | mesh | skeleton | none
        DeclareLaunchArgument('show_cloud', default_value='false'),    # sparse map points
        DeclareLaunchArgument('rviz', default_value='true'),
        # the robot is drawn >= delay s behind the newest leg odometry sample, with interpolated S3 poses; the node
        # raises it to the measured S3 output latency + pose interval
        DeclareLaunchArgument('delay', default_value='0.6'),
        DeclareLaunchArgument('path_mode', default_value='backend'),   # backend | trail
        DeclareLaunchArgument('correction_tau', default_value='1.0'),
        DeclareLaunchArgument('ground_follow', default_value='true'),
        DeclareLaunchArgument('s3', default_value='false'),           # also start visual_odometry.launch.py
        DeclareLaunchArgument('config_path', default_value=''),        # S3 options passed to it (empty: defaults)
        DeclareLaunchArgument('camera_config_path', default_value=''),
        DeclareLaunchArgument('online_mode', default_value=''),
        # the S3 recorder needs saving_dir (its map_root); off unless asked for
        DeclareLaunchArgument('record_leg_odom', default_value='false'),
        DeclareLaunchArgument('saving_dir', default_value=''),
        DeclareLaunchArgument('bag', default_value=''),                # play this bag after play_delay s
        DeclareLaunchArgument('play_delay', default_value='15.0'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        # scoped: the included file's launch configurations (its rviz) must not leak into this one
        GroupAction(scoped=True, condition=IfCondition(LaunchConfiguration('s3')), actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch' / 'visual_odometry.launch.py')),
            launch_arguments={'robot': LaunchConfiguration('robot'), 'config_path': LaunchConfiguration('config_path'),
                              'camera_config_path': LaunchConfiguration('camera_config_path'),
                              'online_mode': LaunchConfiguration('online_mode'),
                              'record_leg_odom': LaunchConfiguration('record_leg_odom'),
                              'saving_dir': LaunchConfiguration('saving_dir'), 'rviz': 'false'}.items())]),
        OpaqueFunction(function=view),
    ])
