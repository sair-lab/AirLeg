#!/usr/bin/env python3
"""Writes the RViz configs of air_slam_ros2 in one style (white background, one palette, the same display names):

  rviz/robot_viz.rviz         S3 (robot_viz.launch.py, also visual_odometry.launch.py rviz:=true): fixed frame world
  rviz/map_optimization.rviz  map refinement (map_refinement.launch.py): fixed frame map
  rviz/relocalization.rviz    relocalization (relocalization.launch.py): fixed frame map

robot_viz.launch.py switches displays on / off per run (model mode, show_cloud) and sets the followed frame.
usage: make_rviz_configs.py [<rviz dir>]   (default: ../rviz next to this script)
"""
import sys
from pathlib import Path

import yaml

# palette (R; G; B)
BLUE, GRAY, LIGHT, RED, GREEN, ORANGE = '31; 95; 208', '150; 150; 150', '190; 190; 190', '220; 50; 50', '40; 170; 70', \
    '240; 140; 20'


def topic(name, latched=False, reliable=True, marker=False):
    t = {'Depth': 5, 'Durability Policy': 'Transient Local' if latched else 'Volatile', 'History Policy': 'Keep Last',
         'Reliability Policy': 'Reliable' if reliable else 'Best Effort', 'Value': name}
    if marker:
        t['Filter size'] = 10
    return t


def grid(frame):
    return {'Alpha': 0.25, 'Cell Size': 1, 'Class': 'rviz_default_plugins/Grid', 'Color': '160; 160; 164',
            'Enabled': True, 'Line Style': {'Line Width': 0.03, 'Value': 'Lines'}, 'Name': 'Grid',
            'Normal Cell Count': 0, 'Offset': {'X': 0, 'Y': 0, 'Z': 0}, 'Plane': 'XY', 'Plane Cell Count': 150,
            'Reference Frame': frame, 'Value': True}


def cloud(name, topic_name, enabled, color=GRAY, size=0.03, cls='PointCloud', latched=False):
    return {'Alpha': 0.8, 'Autocompute Intensity Bounds': True,
            'Autocompute Value Bounds': {'Max Value': 10, 'Min Value': -10, 'Value': True}, 'Axis': 'Z',
            'Channel Name': 'intensity', 'Class': f'rviz_default_plugins/{cls}', 'Color': color,
            'Color Transformer': 'FlatColor', 'Decay Time': 0, 'Enabled': enabled, 'Invert Rainbow': False,
            'Max Color': '255; 255; 255', 'Max Intensity': 4096, 'Min Color': '0; 0; 0', 'Min Intensity': 0,
            'Name': name, 'Position Transformer': 'XYZ', 'Selectable': True, 'Size (Pixels)': 3, 'Size (m)': size,
            'Style': 'Flat Squares', 'Topic': topic(topic_name, latched=latched), 'Use Fixed Frame': True,
            'Use rainbow': True, 'Value': enabled}


def path(name, topic_name, enabled, color=BLUE, width=0.03):
    return {'Alpha': 1, 'Buffer Length': 1, 'Class': 'rviz_default_plugins/Path', 'Color': color, 'Enabled': enabled,
            'Head Diameter': 0.3, 'Head Length': 0.2, 'Length': 0.3, 'Line Style': 'Billboards', 'Line Width': width,
            'Name': name, 'Offset': {'X': 0, 'Y': 0, 'Z': 0}, 'Pose Color': color, 'Pose Style': 'None',
            'Radius': 0.03, 'Shaft Diameter': 0.1, 'Shaft Length': 0.1, 'Topic': topic(topic_name), 'Value': enabled}


def poses(name, topic_name, enabled, color=BLUE, cls='PoseArray'):
    d = {'Alpha': 1, 'Axes Length': 0.15, 'Axes Radius': 0.015, 'Class': f'rviz_default_plugins/{cls}', 'Color': color,
         'Enabled': enabled, 'Head Length': 0.07, 'Head Radius': 0.03, 'Name': name, 'Shaft Length': 0.23,
         'Shaft Radius': 0.01, 'Shape': 'Arrow (Flat)' if cls == 'PoseArray' else 'Axes', 'Topic': topic(topic_name),
         'Value': enabled}
    if cls == 'PoseArray':
        d.update({'Arrow Length': 0.3, 'Arrow2D Color': color, 'Arrow3D Head Length': 0.07,
                  'Arrow3D Head Radius': 0.03, 'Arrow3D Shaft Length': 0.23, 'Arrow3D Shaft Radius': 0.01})
    return d


def marker(name, topic_name, enabled, latched=False, array=False):
    return {'Class': 'rviz_default_plugins/' + ('MarkerArray' if array else 'Marker'), 'Enabled': enabled,
            'Name': name, 'Namespaces': {}, 'Topic': topic(topic_name, latched=latched, marker=not array),
            'Value': enabled}


def image(name, topic_name, enabled):
    return {'Class': 'rviz_default_plugins/Image', 'Enabled': enabled, 'Max Value': 1, 'Median window': 5,
            'Min Value': 0, 'Name': name, 'Normalize Range': True, 'Topic': topic(topic_name, reliable=False),
            'Value': enabled}


def robot_model(enabled):
    return {'Alpha': 1, 'Class': 'rviz_default_plugins/RobotModel', 'Collision Enabled': False,
            'Description File': '', 'Description Source': 'Topic',
            'Description Topic': topic('/robot_description', latched=True), 'Enabled': enabled,
            'Links': {'All Links Enabled': True, 'Expand Joint Details': False, 'Expand Link Details': False,
                      'Expand Tree': False, 'Link Tree Style': 'Links in Alphabetic Order'},
            'Mass Properties': {'Inertia': False, 'Mass': False}, 'Name': 'Robot model', 'TF Prefix': '',
            'Update Interval': 0, 'Value': enabled, 'Visual Enabled': True}


def tf(enabled):
    return {'Class': 'rviz_default_plugins/TF', 'Enabled': enabled, 'Frame Timeout': 15,
            'Frames': {'All Enabled': True}, 'Marker Scale': 0.5, 'Name': 'TF', 'Show Arrows': False,
            'Show Axes': True, 'Show Names': False, 'Tree': {}, 'Update Interval': 0, 'Value': enabled}


def view(follow_frame=None):
    v = {'Distance': 12.0, 'Enable Stereo Rendering': {'Stereo Eye Separation': 0.06, 'Stereo Focal Distance': 1,
                                                        'Swap Stereo Eyes': False, 'Value': False},
         'Focal Point': {'X': 0, 'Y': 0, 'Z': 0}, 'Focal Shape Fixed Size': True, 'Focal Shape Size': 0.05,
         'Invert Z Axis': False, 'Name': 'Current View', 'Near Clip Distance': 0.01, 'Pitch': 0.55, 'Yaw': 3.9}
    if follow_frame:
        v.update({'Class': 'rviz_default_plugins/ThirdPersonFollower', 'Target Frame': follow_frame,
                  'Value': 'ThirdPersonFollower (rviz_default_plugins)'})
    else:
        v.update({'Class': 'rviz_default_plugins/Orbit', 'Target Frame': '<Fixed Frame>',
                  'Value': 'Orbit (rviz_default_plugins)'})
    return v


WIDTH, HEIGHT = 1900, 1100   # RViz window (px)
LEFT_DOCK_WIDTH = 420        # px: Displays above, the feature tracking image (4:3) below; the rest is the 3D view


def window_state(images):
    """QMainWindow State (hex) of the RViz window: a LEFT_DOCK_WIDTH px left dock with Displays above the image panels
    (`images`: display name -> enabled; disabled ones stay hidden), Views right, Time at the bottom. RViz restores dock
    widgets by name (panel / display name). Without a saved state Qt sized the left dock by the image panel (~1000 of
    1900 px) and left the 3D view ~800 px. Computed with Qt offscreen (PyQt5, the Qt version RViz uses)."""
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
    app = QApplication.instance() or QApplication([])
    win = QMainWindow()
    win.setCentralWidget(QWidget())
    docks = {}
    for name, area in [('Displays', Qt.LeftDockWidgetArea)] + [(n, Qt.LeftDockWidgetArea) for n in images] + \
            [('Views', Qt.RightDockWidgetArea), ('Time', Qt.BottomDockWidgetArea)]:
        dock = QDockWidget(name)
        dock.setObjectName(name)
        dock.setWidget(QWidget())
        win.addDockWidget(area, dock)
        docks[name] = dock
    for name in images:
        win.splitDockWidget(docks['Displays'], docks[name], Qt.Vertical)
    win.resize(WIDTH, HEIGHT)
    win.show()
    app.processEvents()
    win.resizeDocks([docks['Displays']], [LEFT_DOCK_WIDTH], Qt.Horizontal)
    shown = [name for name, enabled in images.items() if enabled]
    if shown:   # image height for a 4:3 image at the dock width, plus the dock title bar
        win.resizeDocks([docks[name] for name in shown], [LEFT_DOCK_WIDTH * 3 // 4 + 30] * len(shown), Qt.Vertical)
    for name, enabled in images.items():
        docks[name].setVisible(enabled)
    docks['Views'].setVisible(False)   # Hide Right Dock
    app.processEvents()
    return bytes(win.saveState().toHex()).decode()


def config(displays, fixed_frame, follow_frame=None):
    images = {d['Name']: d['Enabled'] for d in displays if d['Class'] == 'rviz_default_plugins/Image'}
    return {
        'Panels': [
            {'Class': 'rviz_common/Displays', 'Help Height': 78, 'Name': 'Displays',
             'Property Tree Widget': {'Expanded': ['/Global Options1'], 'Splitter Ratio': 0.5}, 'Tree Height': 400},
            {'Class': 'rviz_common/Views', 'Expanded': ['/Current View1'], 'Name': 'Views', 'Splitter Ratio': 0.5},
            {'Class': 'rviz_common/Time', 'Experimental': False, 'Name': 'Time', 'SyncMode': 0, 'SyncSource': ''}],
        'Visualization Manager': {
            'Class': '', 'Displays': displays, 'Enabled': True,
            'Global Options': {'Background Color': '255; 255; 255', 'Fixed Frame': fixed_frame, 'Frame Rate': 30},
            'Name': 'root',
            'Tools': [{'Class': 'rviz_default_plugins/Interact', 'Hide Inactive Objects': True},
                      {'Class': 'rviz_default_plugins/MoveCamera'}, {'Class': 'rviz_default_plugins/Select'},
                      {'Class': 'rviz_default_plugins/FocusCamera'},
                      {'Class': 'rviz_default_plugins/Measure', 'Line color': '128; 128; 0'}],
            'Transformation': {'Current': {'Class': 'rviz_default_plugins/TF'}}, 'Value': True,
            'Views': {'Current': view(follow_frame), 'Saved': None}},
        'Window Geometry': {'Displays': {'collapsed': False}, 'Height': HEIGHT, 'Hide Left Dock': False,
                            'Hide Right Dock': True, 'QMainWindow State': window_state(images), 'Width': WIDTH,
                            'X': 60, 'Y': 30},
    }


def write(path_, cfg, header):
    path_.write_text(''.join(f'# {line}\n' for line in header) + yaml.safe_dump(cfg, sort_keys=False))
    print(path_)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'rviz'
    note = 'Generated by scripts/make_rviz_configs.py (edit the script, not this file).'
    write(out / 'robot_viz.rviz', config([
        grid('ground'),
        robot_model(True),
        marker('Skeleton', '/robot_viz/skeleton', False),
        tf(False),
        marker('Trajectory', '/robot_viz/trajectory', True, latched=True),
        marker('Footprints', '/robot_viz/footprints', True, latched=True, array=True),
        marker('Camera', '/robot_viz/camera_frustum', True),
        cloud('Map points', '/AirSLAM/map', False),
        poses('Front-end keyframes', '/AirSLAM/keyframe', False, color=LIGHT),
        path('Front-end path', '/AirSLAM/odometry', False, color=LIGHT),
        poses('Front-end pose', '/AirSLAM/frame_pose', False, color=ORANGE, cls='Pose'),
        image('Feature tracking', '/AirSLAM/feature', True),
    ], 'world', 'base'), [note, 'S3 view (robot_viz.launch.py; visual_odometry.launch.py rviz:=true). The launch file',
                          'switches Robot model / Skeleton / TF for the model mode and Map points for show_cloud, and',
                          'sets the followed frame.'])
    write(out / 'map_optimization.rviz', config([
        grid('<Fixed Frame>'),
        cloud('Map points', '/AirSLAM/map', True),
        poses('Keyframes', '/AirSLAM/keyframe', True),
        path('Trajectory', '/AirSLAM/odometry', True),
        poses('Current pose', '/AirSLAM/frame_pose', False, color=ORANGE, cls='Pose'),
        image('Feature tracking', '/AirSLAM/feature', False),
    ], 'map'), [note, 'Map refinement (map_refinement.launch.py).'])
    write(out / 'relocalization.rviz', config([
        grid('<Fixed Frame>'),
        cloud('Map points', '/AirSLAM/map', True),
        marker('Relocalized trajectory', '/AirSLAM/reloc/raw_trajectory', True),
        poses('Relocalized pose', '/AirSLAM/reloc/pose', True, color=RED, cls='Pose'),
        marker('Matches', '/AirSLAM/reloc/matches', True),
        image('Feature tracking', '/AirSLAM/feature', True),
    ], 'map'), [note, 'Relocalization (relocalization.launch.py).'])


if __name__ == '__main__':
    main()
