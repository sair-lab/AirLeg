# Adding a new robot

All code (`core/`, `ros2_packages/`, `core/3rdparty/TRAIL/`) is shared by all robots. A robot is described only by its
directory `robots/<name>/`, which `robot:=<name>` (launch) and `--robot <name>` (training scripts) select. This guide
walks through the files to create. [`robots/go2/`](../robots/go2/) is the reference for a Unitree-style robot that
publishes one joint + IMU message. [`robots/anymal/`](../robots/anymal/) is the reference for a robot with separate
joint and IMU topics.

## The `robots/` directory

```
robots/
├── common/                  shared by all robots
│   ├── algorithm.yaml       SLAM parameters: front end, keyframes, optimization gates, online fusion / back end, loops, S4
│   ├── leg_odometry.yaml    leg odometry: simulation: (S1), sim_to_real: (S2), online: (S3)
│   └── models/              SuperPoint / LightGlue / Fast-FoundationStereo ONNX (TensorRT engines are built here), vocabulary
├── go2/                     Unitree Go2 + RealSense D455 (infrared stereo 640x480)
├── anymal/                  ANYmal D + Alphasense front stereo (GrandTour; 1440x1080 fisheye resized to 720x540)
└── euroc/                   vision-only AirSLAM config (algorithm.yaml + camera.yaml, no legs)
```

Who reads what:

| File / section | Read by |
|---|---|
| `leg_odometry.yaml` `robot:` | TRAIL `common/robot.py` (S1 data, S2 pressure, S3 worker, S4, evaluation) and the recorder |
| `leg_odometry.yaml` `simulation:` | `train_velocity.py --config` |
| `leg_odometry.yaml` `sim_to_real:` | `pipeline.py s2 --robot <name>` |
| `leg_odometry.yaml` `online:` | `online_worker.py` (launch `robot:=<name>`) |
| `algorithm.yaml` | the C++ nodes, the launch files, TRAIL; its `global_backend:` section is read by `pipeline.py s4` |
| `camera.yaml` | the VO (launch `camera_config_path:=`), S2 preparation |
| `model/` | `common/leg_kinematics.py` (and robot_viz) |
| `checkpoints/` | S2 initialization (`leg_odometry_sim.pt`), S3 (`leg_odometry.pt`) |

Any file can still be given explicitly: `config_path:=`, `camera_config_path:=`, `leg_config:=`, `recorder_config:=`
(a ROS parameter file that overrides `robot: recorder`) at launch, `--config` for the training scripts.

## Requirements of the current code

The following conventions are not generalized yet. A new robot has to map its data onto them:

- **Legs and joints.** 4 legs × 3 joints, in the internal order FR, FL, RR, RL × hip, thigh, calf. The joint names in
  the kinematic model are `{FR,FL,RR,RL}_{hip,thigh,calf}_joint`. Foot pressures and feet use the order FL, FR, RL,
  RR.
- **Network input.** 34 values per sample: IMU (gyro 3 + accelerometer 3), joint positions 12, joint velocities 12,
  foot pressures 4. The sample rate is `robot: network_rate_hz` (Go2: 125 Hz, ANYmal: 100 Hz). The window is 50
  samples.
- **Stereo camera.** The robot needs a stereo camera with images on ROS 2 topics. The first camera pose should be
  roughly level, with the image y axis pointing down: the back end uses it to find the vertical direction when there
  is no IMU.
- **New network.** A new robot needs its own S1 model. Normalization statistics and the IMU extrinsic are stored in
  the checkpoint and cannot be reused across robots.

## Files of a robot directory

```
robots/<name>/
├── leg_odometry.yaml      robot description + S1 / S2 / S3 settings of the leg odometry (base: ../common/leg_odometry.yaml)
├── algorithm.yaml         SLAM settings that differ from robots/common/algorithm.yaml (base: ../common/algorithm.yaml)
├── camera.yaml            stereo intrinsics / extrinsics, image and IMU topics
├── model/                 kinematic model (MuJoCo XML or URDF), optional meshes for visualization
├── viz.yaml               optional: settings of the RViz view robot_viz
└── checkpoints/
    ├── leg_odometry_sim.pt   S1 model (initial model of S2)
    └── leg_odometry.pt       S2 model (used by S3)
```

**Config inheritance.** Every YAML file may start with `base: <path>`. The parent file is loaded first and then deep
merged:

- mappings merge key by key, and the child wins;
- scalars and lists are replaced as a whole.

Paths are relative to the file that defines them. C++ (`LoadYamlWithBase`) and Python (`TRAIL/common/yaml_base.py`)
follow the same rules. Write only the keys that differ from `robots/common/`.

## Step 1: kinematic model (`model/`)

The leg kinematics (`TRAIL/common/leg_kinematics.py`) computes the foot positions and the Jacobian foot pressure
`F = J^-T τ` in closed form. It reads the following from the model, which is either MuJoCo XML or URDF:

- the revolute joints `{FL,FR,RL,RR}_{hip,thigh,calf}_joint`, each with its origin in the parent link and its axis;
- the foot point `{leg}_foot_fixed`. Use the center of the foot collision sphere, as in the Unitree models;
- optionally, the IMU pose: URDF `imu_joint`, or MJCF `<site name="imu">`.

If the official model uses other names or has fixed frames between the actuated joints, convert it once.
`TRAIL/data_scripts/make_kinematic_urdf.py` is the ANYmal D example:

- it renames LF/RF/LH/RH × HAA/HFE/KFE to FL/FR/RL/RR × hip/thigh/calf;
- it folds the fixed frames into the next revolute joint.

It checks that the forward kinematics are unchanged. Keep the original model for visualization (step 6).

## Step 2: `leg_odometry.yaml`, section `robot:`

```yaml
base: ../common/leg_odometry.yaml

robot:
  name: myrobot
  model: model/myrobot_kinematic.urdf   # step 1
  network_rate_hz: 100                  # rate of the whole leg chain: recorder, network, S3 worker
  lowstate_sampling: slot               # slot: fixed time slots (default); on_change: sample when the joint values
                                        # refresh (robots whose state refreshes slower than the message rate)
  foot_radius: 0.03                     # m, foot sphere radius (foot-terrain constraint, visualization)
  imu_extrinsic: {source: model}        # IMU pose in the base frame, see "Frames" below
  pressure:                             # Jacobian foot pressure; one definition for S1 data, S2, S3 and S4
    method: jacobian
    pressure_mode: normal_abs           # normal_abs: |Fz|; normal_negative: max(-Fz, 0), for heavy legs (ANYmal)
    normalize_sum: false
    damping: 1.0e-4
    smooth_alpha: 0.9                   # EMA per sample: rescale it when network_rate_hz changes
    pressure_scale: 1.0
  recorder:                             # defaults of the leg-data recorder (S2 data, S4)
    lowstate_topic: /lowstate_stamped
    estimate_contacts: true
    overwrite: true
```

**Pressure mode.** With heavy legs, the swing-phase inertia torques show up as large `|Fz|`. ANYmal therefore uses
`normal_negative` with `smooth_alpha: 0.5`. Check the mode against the simulator foot forces or the robot's contact
flags.

### Joint and IMU input

There are two input modes, implemented in `ros2_packages/leg_odom_recorder/airslam_leg_odom_recorder/sources.py`:

1. **One message (default, Go2).** `lowstate_republisher/LowStateStamped` on `recorder.lowstate_topic`. It carries
   the IMU (gyroscope, accelerometer) and 12 motors (`motor_q`, `motor_dq`, `motor_tau_est`) in the order FR, FL, RR,
   RL × hip, thigh, calf. For a Unitree robot, republish its low state as this message.
2. **Separate topics (ANYmal).** A joint topic (`sensor_msgs/JointState` or `anymal_msgs/AnymalState`) and a
   `sensor_msgs/Imu` topic. The IMU is interpolated at every joint stamp. The joints are reordered by `names`, which
   lists the source joint names in the internal order:

   ```yaml
   robot:
     leg_inputs:
       joints:
         topic: /myrobot/joint_states
         message: joint_state          # joint_state | anymal_state
         names: [RF_HAA, RF_HFE, RF_KFE, LF_HAA, LF_HFE, LF_KFE, RH_HAA, RH_HFE, RH_KFE, LH_HAA, LH_HFE, LH_KFE]
       imu:
         topic: /myrobot/imu
         max_gap_s: 0.005              # joint samples between IMU samples further apart are dropped
         resampled_topic: /airslam/imu_resampled   # optional: IMU at the joint stamps, for the VO (camera.yaml imu_topic)
   ```

   The joint message must contain position, velocity and effort.

### Frames

- **B** is the base (trunk) frame of the kinematic model. **I** is the IMU frame.
- `imu_extrinsic` is the IMU pose in B:
  - `{source: model}` takes it from the model's `imu_joint` or `imu` site;
  - `{xyz: [x, y, z], quat_xyzw: [qx, qy, qz, qw]}` gives it explicitly (external IMU, or no IMU in the model).
  - Go2: (-0.02557, 0, 0.04232), no rotation.
  - ANYmal: (-0.25565, 0.00255, 0.07672), mounted upside down: R_BI = diag(-1, 1, -1). The simulation and the real
    `/anymal/imu` are both in this IMU frame.
- Raw IMU data stays in I. It is rotated into the B axes in one place, when it is read: S1 data, recording, the S3
  worker. No lever-arm compensation is applied, so the accelerometer stays the specific force at the IMU point.
- The network outputs the velocity of the B origin in B. The angular velocity, the gyro bias and the gyro correction
  are also in the B axes, and integration gives the pose of B.
- The foot positions, the camera ← base extrinsic `T_C_B` and the back-end gyro bias are in B. Only the S2
  accelerometer preintegration loss needs the IMU-point velocity `v_I = v_B + ω_B × r_BI`.
- S1 writes the extrinsic into the checkpoint (`cfg.imu_extrinsic`, also in the simulation `manifest.json`). S2 to S4
  use the value from the checkpoint.

A rotation error δθ in `imu_extrinsic` tilts the velocity by δθ: 1° gives about 1.7 % vertical drift.

## Step 3: `camera.yaml`

Copy `robots/go2/camera.yaml` (undistorted, rectified pinhole stereo) or `robots/anymal/camera.yaml` (fisheye images
that AirSLAM undistorts), then edit:

| Key | Meaning |
|---|---|
| `image_width`, `image_height` | Input size of the VO (images are resized to it) |
| `left_image_topic`, `right_image_topic` | Raw or `.../compressed` image topics |
| `distortion_type` | 0: undistorted; 1: radial-tangential `[k1 k2 p1 p2 k3]`; 2: equidistant `[k1 k2 k3 k4 0]` |
| `cam0`, `cam1` | `intrinsics` `[fx fy cx cy]`, `distortion_coeffs`, `T` (`T_type` 0: body ← camera, EuRoC; 1: camera ← body, Kalibr) |
| `imu_topic`, `imu_message` | IMU of the VO and the back end: `lowstate` (the IMU fields of LowStateStamped) or `imu` (`sensor_msgs/Imu`) |
| `T_imu_cam0` | Pose of cam0 in that IMU frame |
| `gyroscope_noise_density`, `accelerometer_noise_density`, `*_random_walk`, `rate_hz` | IMU noise model |

A robot can have several camera calibrations, e.g. per recording site or day (ANYmal: `camera_2024-11-18.yaml`).
Assign them per bag in `sim_to_real: prepare.camera_configs` (bag name → camera yaml). Online, pass the right one with
`camera_config_path:=`.

The camera ← base extrinsic `T_C_B` of the leg odometry is **not** read from here. S2 estimates it and stores it in
the S2 checkpoint, and S3 publishes it from there.

## Step 4: `algorithm.yaml`

```yaml
base: ../common/algorithm.yaml
camera_config_path: camera.yaml

visual_odometry:
  use_imu: true                # IMU in the front-end attitude filter and the back-end preintegration
  online:
    leg:
      time_delay: 0.0          # t_leg = t_camera + time_delay (s); equal to leg_odometry.yaml online: time_delay
    surface:
      foot_radius: 0.03        # same as robot: foot_radius
```

**IMU sources.**

- **Leg odometry.** The leg odometry always uses the IMU of its joint input: the `LowStateStamped` IMU fields, or
  `leg_inputs.imu`.
- **VO and back end.** They use the IMU of `camera.yaml` (`imu_topic`, `imu_message`).
  - `lowstate` reads the IMU fields of `LowStateStamped`, the same IMU as the leg odometry (Go2).
  - `imu` reads a `sensor_msgs/Imu` topic. ANYmal uses `/airslam/imu_resampled`: the leg IMU interpolated at the
    joint stamps (`leg_inputs.imu.resampled_topic`). A different IMU (e.g. the camera's own) needs its own
    `T_imu_cam0` and noise.
- **`visual_odometry.use_imu`.** This master switch controls only the VO and the back end. In the front end, the
  camera yaml `imu_mode` runs an attitude filter that aligns the first frame with gravity. In the back end, it adds
  the IMU preintegration factors. The launch argument `use_imu:=true|false` overrides it.
  - The common default is false; Go2 and ANYmal set it to true.
  - The Go2 body IMU needs a varying gyro bias (random-walk floor `imu_bg_rw_min`); see `robots/go2/algorithm.yaml`.

Start with `use_imu: false` if the IMU noise and `T_imu_cam0` are not calibrated yet.

## Step 5: training sections

Add the S1 / S2 / S3 sections to `leg_odometry.yaml`. Each section only overrides `robots/common/leg_odometry.yaml`:

```yaml
simulation:                     # S1
  robot: myrobot
  data:
    data_root: /data/sim/myrobot_100hz
  train:
    output_dir: ../../core/3rdparty/TRAIL/runs/s1_myrobot

sim_to_real:                    # S2
  checkpoint: checkpoints/leg_odometry_sim.pt
  train:
    output_dir: ../../core/3rdparty/TRAIL/runs/s2_myrobot
  prepare:
    bag_root: /data/myrobot/s2_bags          # every bag in this directory is used for training
    cache_root: /data/myrobot/s2_cache
    algorithm: algorithm.yaml
    robot: myrobot

online:                         # S3
  robot: myrobot
  checkpoint: checkpoints/leg_odometry.pt
  time_delay: 0.0
```

Then train the robot as described in [TRAINING.md](TRAINING.md).

## Step 6 (optional): visualization (`viz.yaml`)

`robot_viz` works without any extra file. It uses the kinematic model of `robot: model`:

- with all mesh files found, it shows the meshes;
- otherwise it shows a skeleton of TF frames and links.

Use `viz.yaml` to show a different (mesh) model:

```yaml
model: model/myrobot.urdf       # URDF or MJCF (converted on the fly); meshes via file://, relative paths or package://<pkg>/
joint_names: [RF_HAA, ...]      # joint names of this model in the order FR FL RR RL x hip thigh calf (if not Unitree names)
# optional: foot_radius, camera_config, lowstate_topic
```

`package://<pkg>/...` is resolved first in a `<pkg>` directory next to the model, then in the installed ROS packages.
Other movable joints of the model are held at 0.

## Checklist

1. Build and run S3 in `visual_only` mode first. This checks the camera config:

   ```bash
   ros2 launch air_slam_ros2 visual_odometry.launch.py robot:=myrobot online_mode:=visual_only saving_dir:=/tmp/vo
   ```

2. Record a short bag and check the leg data. Run with `record_leg_odom:=true` and look at
   `<saving_dir>/leg_odom/recorder_status.json`:
   - the counts are > 0;
   - there are no `lowstate_errors` / `contact_errors`;
   - the CSV files cover the map time span.
3. Check the kinematics and the pressure. While the robot stands, the feet computed from the recorded joints should
   not move. The pressure should be high in stance and low in swing.
4. Build the simulation data, then train S1 and S2 ([TRAINING.md](TRAINING.md)). Copy the checkpoints to
   `checkpoints/`.
5. Run S3 in `fusion_only` mode, then in `online_adapt` mode.
