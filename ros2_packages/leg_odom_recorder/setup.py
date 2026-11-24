from setuptools import find_packages, setup

package_name = "airslam_leg_odom_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/record_leg_odom.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "recorder = airslam_leg_odom_recorder.recorder:main",
        "online_leg_odom = airslam_leg_odom_recorder.online_node:main",
        "imu_resampler = airslam_leg_odom_recorder.imu_resampler:main",
    ]},
)
