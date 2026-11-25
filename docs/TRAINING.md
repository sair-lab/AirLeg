# Training from scratch

AirLeg trains its leg-odometry network in two offline stages. The network then keeps adapting online:

| Stage | What | Supervision | Entry point | Output |
|---|---|---|---|---|
| S1 | Pre-train the network in simulation | Simulator base velocity | `train_velocity.py` | `checkpoints/leg_odometry_sim.pt` |
| S2 | Adapt it to the real robot | Multi-sensor BA on unlabeled real bags (no ground truth) | `pipeline.py s2` | `checkpoints/leg_odometry.pt` |
| S3 | Online fusion and online adaptation | Fused back-end keyframes, at run time | `visual_odometry.launch.py` | Trajectory and map (the adapted weights stay in memory) |
| S4 (optional) | Loop closure and full-sequence optimization of an S3 map | none | `pipeline.py s4` | `trajectory_v1.txt`, `AirSLAM_mapv1.bin` |

All stages of one robot read `robots/<robot>/leg_odometry.yaml`:

- S1 reads the `simulation:` section;
- S2 reads the `sim_to_real:` section;
- S3 reads the `online:` section;
- every stage reads the `robot:` section for the kinematics, pressure and IMU extrinsic.

Each section inherits `robots/common/leg_odometry.yaml`, whose comments explain every option. For a new robot,
create these sections first ([ADD_ROBOT.md](ADD_ROBOT.md)).

## Environment

The commands below use the Python environment with PyTorch from the [main README](../README.md#python-environment-for-the-leg-network)
(`requirements_cuda.txt`) and run from `core/3rdparty/TRAIL`:

```bash
conda activate airleg
cd <repo>/core/3rdparty/TRAIL
```

- **S1** needs only this environment.
- **S2 and S4** also start ROS 2 nodes, the VO and the map back end of the built workspace. Source ROS and the
  workspace in the same shell before you activate the environment:

  ```bash
  source /opt/ros/humble/setup.bash
  source ~/airleg_ws/install/setup.bash
  conda activate airleg
  ```

## The leg-odometry network

The network estimates the linear velocity of the base from a fixed-length window of onboard measurements.

**Input.** A window of `data.window` samples (default 50) at `robot: network_rate_hz`. Each sample has 34 values:

- the IMU (gyro 3, accelerometer 3), rotated into the base axes with `robot: imu_extrinsic`;
- the joint positions (12) and velocities (12);
- the foot pressures (4), in the order FL, FR, RL, RR.

The inputs are standardized with the mean / std of the training set (`Normalizer`). They are saved in the checkpoint
and reused by S2, S3 and S4.

**Architecture** (`simulation: model`). It is a Transformer encoder:

1. A linear layer maps every sample to `hidden_dim` (128).
2. A sinusoidal position encoding is added.
3. `transformer_layers` (4) encoder layers with `transformer_heads` (4) heads follow.
4. The feature of the last time step is the current state. Three heads read it:
   - `velocity_head`: base linear velocity `v = [vx, vy, vz]`;
   - `gyro_head`: a small correction of the measured angular velocity;
   - `uncertainty_head` (optional): `log_var` of the velocity.

The body twist used for integration is the network linear velocity together with the (corrected) gyro:
`V_body = [vx, vy, vz, ωx, ωy, ωz]`.

**S1 losses.**

- Velocity mean: `L_vel = SmoothL1(v_pred, v_gt)`.
- Uncertainty head, with the backbone and the velocity head frozen:
  `L_unc = mean(0.5 · ((v_gt − stopgrad(v_pred))² / exp(log_var) + log_var))`.

### Foot-pressure normalization

The foot pressure is the Jacobian estimate `F = J^-T τ` from the joint torques (`robot: pressure`). Its scale depends
on the robot's load, so it is normalized by the recent total load (`pressure_normalization`, on by default):

| Key | Default | Meaning |
|---|---|---|
| `offline_window_s` | 3.0 | Window of valid load time for offline normalization |
| `online_init_s` | 1.0 | Valid load time collected online before the first output |
| `online_window_s` | 3.0 | Online window of valid load time |
| `min_total_pressure_ratio` | 0.5 | Floor of the total pressure, as a fraction of the training mean total pressure |

How the normalization works:

- **Shared denominator.** Every foot pressure is divided by one denominator: the time-weighted mean of the valid
  total pressure over the latest window. It is protected by the floor.
- **Low-load samples.** Samples below the floor (e.g. a jump) neither enter the mean nor consume window time, but stay
  network inputs. The window is therefore the latest 3 s of *valid* data, not 3 s of wall-clock time.
- **Start of a sequence.** Online, the network outputs nothing until 1 s of valid load is collected, then the window
  grows to 3 s. Offline, the start of a sequence is filled with the first full valid window.
- **Training statistics.** The training mean total pressure is stored in the checkpoint
  (`cfg.pressure_normalization.train_mean_total_pressure`).
- **Where it is fitted.** S1 fits it on the training data; S2 refits it in its initialize step.
- **Same in every stage.** The causal online implementation (`PressureWindow.update`) and the batched offline one give
  identical results, so S1, S2, S3 and S4 see the same input.

The recorder stores the raw (un-normalized) Jacobian pressure. For old maps whose manifest says `torque_norm`,
`stage4.py infer` rebuilds the raw pressure from `robot/joints.csv` with the closed-form kinematics
(`common/leg_kinematics.py`). It caches the result under `leg_odom/tmp/raw_pressure` and does not change the CSV.

## S1: simulation pre-training

### Simulation data

S1 reads a dataset with this layout. The sample rate is `robot: network_rate_hz`. The CSV files of one sequence share
the `sample,time` columns.

```
<sim_root>/
├── manifest.json                  rates, IMU extrinsic, pressure settings, splits
├── train/data/sequence_*/
│   ├── imu.csv                    sample,time,gyro_{x,y,z},acc_{x,y,z}           (IMU frame, specific force)
│   ├── joints.csv                 sample,time,{FR,FL,RR,RL}_{hip,thigh,calf}_joint_{pos,vel,...}
│   ├── contacts.csv               sample,time,FL_pressure,FR_pressure,RL_pressure,RR_pressure
│   ├── ground_truth.csv           sample,time,base_pos_*,base_quat_*,base_lin_vel_body_*,base_ang_vel_body_*,...
│   └── metadata.json
├── val/data/sequence_*/ ...
└── test/data/sequence_*/ ...
```

- **Labels.** The labels are `base_lin_vel_body_{x,y,z}`: the velocity of the base origin in the base frame.
- **Foot pressure.** `contacts.csv` is **not** the simulator contact force. It is the Jacobian foot pressure that the
  real robot will also use. It is computed from the joint torques with the `robot: pressure` settings of the robot.

We collected the Go2 and ANYmal D data with IsaacLab: rough-terrain locomotion policies on flat ground, slopes, stairs,
gaps, hurdles and soft ground, with randomized commands and gaits. Download: TODO_SIM_DATA_URL. The IsaacLab
collection driver is not part of this repository.

`data_scripts/build_sim_dataset.py` turns a raw IsaacLab collection of Go2 or ANYmal into the layout above. It:

- decimates the data (`--stride`, e.g. Go2 250 Hz → 125 Hz);
- drops the spawn impact (`--trim-s`);
- renames the joints to the internal names;
- estimates the Jacobian foot pressure;
- removes sequences in which the robot falls;
- splits the sequences 90 / 5 / 5 within every terrain group.

```bash
python data_scripts/build_sim_dataset.py --robot go2 --source <isaaclab_collection> --output <sim_root> --workers 16
```

For another robot, write the same layout. For example, add the robot to `DEFAULTS` in `build_sim_dataset.py`, or
convert your own simulator logs. Estimate `contacts.csv` from the joint torques with `data_scripts/estimate_contacts`
(`core.py`, as `build_sim_dataset.py` does) and the pressure settings of the new robot.

### Train

Set `simulation: data.data_root` in the robot config (or pass `--data-root`), then:

```bash
python train_velocity.py --config ../../../robots/go2/leg_odometry.yaml [--data-root <sim_root>] [--output-dir runs/s1_go2]
```

Training has two parts:

1. The velocity mean, with a SmoothL1 loss on the body linear velocity.
2. The uncertainty head. The velocity network is frozen, and the head is trained with a Gaussian NLL loss.
   `--skip-uncertainty` skips this part.

The output directory contains:

| File | Content |
|---|---|
| `best.pt` | Model with the best validation loss (normalizer, IMU extrinsic and config included) |
| `last.pt` | Last model |
| `normalizer.npz` | Normalization statistics |
| `history.json` | Loss history |
| `train.log`, `tensorboard/` | Logs |

The first epoch builds `.npy` caches of the sequences. To build them in advance:

```bash
python eval/build_sequence_cache.py --config ../../../robots/go2/leg_odometry.yaml --data-root <sim_root> --split train
```

### Test and install

```bash
python eval/test_velocity.py --model runs/s1_go2/best.pt --config ../../../robots/go2/leg_odometry.yaml \
  --data-root <sim_root> --split test --output runs/s1_go2/test_metrics.json
cp runs/s1_go2/best.pt ../../../robots/go2/checkpoints/leg_odometry_sim.pt
```

## S2: sim-to-real adaptation (no ground truth)

### Real data

Record ROS 2 bags on the real robot. Each bag needs:

- the stereo image topics of `camera.yaml`;
- the joint / IMU input of the robot: `/lowstate_stamped`, or the `robot: leg_inputs` topics.

No ground truth is needed. Put all training bags into one directory, which is `sim_to_real: prepare.bag_root`; every
bag in it is used for training. Validation uses segments held out of every training bag (`sim_to_real: holdout`).

On Go2, 0.5 h of walking over varied terrain is enough: the released model was trained on 8 bags. Keep the test
bags out of `bag_root`.

For GrandTour, convert the official release with `data_scripts/convert_grandtour_to_ros2.py`
([main README](../README.md#preparing-grandtour)). It creates `<grandtour_airleg>/s2_bags` with the 28 training
missions; set `sim_to_real: prepare.bag_root` of `robots/anymal/leg_odometry.yaml` to it.

### Run

```bash
python pipeline.py s2 --robot go2                     # = --phase all: prepare, then train
python pipeline.py s2 --robot go2 --phase prepare     # only build the cache
python pipeline.py s2 --robot go2 --phase train       # train from an existing cache
python pipeline.py s2 --config <my_s2.yaml>           # another config (base: ../robots/go2/leg_odometry.yaml, section sim_to_real:)
```

**Prepare** (`cache_root`) plays every bag at `playback_rate` (default 0.5) through the AirSLAM VO and the leg-data
recorder. It stores the following per bag:

- the visual keyframes and map;
- the recorded joints, IMU and foot pressure.

It also optionally estimates a stereo disparity bias (`prepare.stereo_selfcal`). It starts ROS 2 processes with
`prepare.ros_domain_id`. When that is not set, it picks a random domain between 100 and 199; set it explicitly when
several ROS jobs run on the same machine. The cache signature includes `algorithm.yaml`, so a config change triggers a
new preparation.

**Train** (`train.output_dir`) has three steps:

1. **Initialize.** It fits the foot-pressure normalization on the training data. It also estimates the camera ← base
   extrinsic `T_C_B`, unless `extrinsic.init_*` with `use_initial_extrinsic: true` fixes it.
2. **Alternating rounds** of bundle adjustment (BA) and network training. `alternating.rounds: [scale, visual, fused]`
   sets the number of rounds of each kind. The default is `[0, 3, 4]`; ANYmal uses `[0, 2, 4]`.
   - *Pure visual rounds.* The teacher is the visual-only BA map. The network heads and the scale adapt to it.
   - *Fused rounds.* BA fuses vision with the current leg odometry, and VCE weights the visual and leg terms. The
     refined map supervises the network through reprojection losses. An implicit gradient passes through the BA
     solution.
   - *Rejection.* A round whose validation trajectory score gets worse is rejected. Early stopping ends the training.
3. **Final BA** with the best model.

Rounds and sequences that are already finished are reused when you run the command again. An interrupted S2 therefore
resumes where it stopped.

**Resources:**

- one GPU with about 22 GB of free memory;
- `alternating.parallel_sequences` (default 12) BA processes at a time, with 1.5–3 GB of RAM each.

The 28 GrandTour training missions take about 30 min per epoch. The full S2 takes many hours.

### How S2 supervises the network

S2 trains the network with the geometry of the visual map; the code is in `train_visual_finetune.py`.

**Keyframe pairs.**

- For two neighbouring keyframes at `t0` and `t1`, the leg samples between them are `i0` (first sample ≥ `t0`) to
  `i1` (last sample ≤ `t1`).
- Each prediction needs the previous `window` samples, so the network reads samples `i0 − window + 1 … i1`.
- It predicts a velocity at every sample, and the velocities are integrated with the real sample intervals (PyPose):
  `R_{k+1} = R_k Exp(ω_k dt)`, `p_{k+1} = p_k + R_k v_k dt`.
- This gives the relative base pose `T_b0_b1`. S2 also uses chains of consecutive keyframes (from the first frame of
  a window to every later frame), the same in training and validation.

**From base motion to camera motion.**

- The translation is multiplied by a learnable scale `s = exp(log_scale)`. It is kept within
  `[scale_min, scale_max]` (0.9–1.1), and a prior pulls it towards `target_scale = 1`.
- The camera motion is then `T_c1_c0 = (T_b0_b1_scaled · T_body_cam)^-1 · T_body_cam`.
- `T_body_cam` is the camera → base extrinsic. Two settings control it:
  - `extrinsic.use_initial_extrinsic: true` fixes it to the given value.
  - Otherwise it is estimated in initialize as `Exp(δ) · T_init` (or `Exp(δ)`), optionally with a prior on δ
    (`extrinsic.use_extrinsic_prior`).

**Visual losses.** The teacher is the BA map of the current round: map points in each keyframe's camera frame, and
stereo depth where a map point is missing.

- **Reprojection.** Points with depth in either keyframe are reprojected into the other keyframe in both directions
  (`p1 = π(T_c1_c0 · P0)`, `p0 = π(T_c0_c1 · P1)`), with a Huber loss.
- **Epipolar.** Matches without depth in both keyframes use the Sampson error of `F = K^-T [t]× R K^-1`, with a robust
  loss.

**Total loss** (weights in `sim_to_real: loss`):

```
L = w_repr L_reprojection + w_epi L_epipolar + w_ext L_extrinsic_prior + w_scale (s − target_scale)²
  + w_vel L_velocity_smooth + w_gyro L_gyro_correction + w_gyro_smooth L_gyro_correction_smooth
```

- `L_velocity_smooth` acts on the predicted linear velocity only.
- `L_gyro_correction` penalizes the gyro-head correction beyond a dead zone.
- `L_gyro_correction_smooth` penalizes its rate of change. When it is on, dropout is disabled during training.

**Optimizer stages** (`sim_to_real: optim.stages`).

1. `extrinsic`: runs only in initialize, and only when no extrinsic is given. The network is frozen; only the
   extrinsic and the scale are optimized, with the reprojection and epipolar losses (plus the extrinsic prior if
   enabled).
2. `head`: the template of every alternating round. It trains the network with a small learning rate and keeps the
   extrinsic fixed. The trainable groups, epochs and learning rates of each round come from `alternating`, e.g.
   LoRA adapters (`alternating.lora`) on the frozen backbone together with the velocity and gyro heads.

**Outputs of a training round** (`round_XXX/visual_head/`):

| File | Content |
|---|---|
| `visual_finetuned.pt` | Selected model: network, normalizer, visual calibration, history |
| `last_visual.pt` | Last model |
| `visual_calibration.json` | `T_body_cam` (xyz + quaternion xyzw), `scale`, `target_scale`, scale limits |
| `history.json`, `visual_eval_*.json` | Loss history and visual validation |

**Debugging a new robot.**

- Run only initialize first (`alternating.enabled: false`), with few pairs (`visual.max_pairs_per_sequence: 100`).
  Check that:
  - the reprojection and epipolar losses decrease;
  - `scale` stays near 1;
  - the extrinsic does not diverge;
  - the numbers of valid matches and points are large enough.
- If the scale drifts, increase `loss.scale_prior_weight`. If the extrinsic drifts, increase
  `loss.extrinsic_prior_weight` or give an initial extrinsic.
- If the visual loss does not decrease, check:
  - the time alignment of images and leg data (`time_delay`);
  - that the intrinsics in `camera.yaml` match the images;
  - the direction of the extrinsic (camera → base).

### S2 reference

**Preparation cache** (`prepare.cache_root`).

- **How bags are played.** Preparation waits until the VO and the recorder subscribe. It plays each bag at
  `playback_rate`, lets the queues drain (`drain_seconds`), stops the nodes and checks the outputs before it marks the
  sequence complete. The readiness check runs in ROS's Python (`prepare.ros_python`, default `/usr/bin/python3`).
- **What is stored per sequence.**
  - The VO map.
  - `leg_odom/{camera.yaml, keypoints/, robot/}`. No images are saved.
  - Feature files `keypoints/<t>.txt` with one line per feature: `id score uL vL uR vR x y z`. The pixels are in the
    rectified left image. `x y z` is the VO map point in the left camera frame, or stereo depth when there is no map
    point, and NaN without depth. Features with the same `id` in consecutive files are the same point.
  - `robot/*.csv` with the raw Jacobian pressure; normalization happens in training.
- **Reuse and failures.** A complete cache rejects changed inputs. A failed or incomplete recording needs its sequence
  directory removed before a retry. Raw bags are never modified.
- **Validation.** The segments held out of every training bag (`holdout`). Bags listed in
  `prepare.validation_sequences` are excluded from training and validated as a whole. Validation data uses
  visual-only BA, and it adds no training samples or pressure statistics.

**Rounds** (`alternating`).

- **Frozen parts.** The extrinsic, the preprocessing and the pressure normalization come from initialize and stay
  fixed. The backbone and the uncertainty head stay frozen, unless LoRA adapters are enabled
  (`alternating.lora`, phases `adaptation` / `head`).
- **Phase 1, scale** (`rounds[0]`). Only the scale is trained.
- **Phase 2, pure visual** (`rounds[1]`). The velocity and gyro heads are trained, and the scale with a small learning
  rate.
  - Phases 1 and 2 use the fixed visual-only BA map as teacher.
  - They train the full `epochs` and keep the last epoch, without rejection.
- **Phase 3, fused** (`rounds[2]`). Every round starts again from the visual-only map:
  1. It runs inference on all sequences with one model.
  2. It runs the fused BA of each sequence (`alternating.vce`).
  3. It trains one model on all training sequences.
  4. The candidate and the input model are evaluated on the same validation points. Early stopping
     (`early_stopping`) selects the round; the phase-2 output is also a candidate.
  5. A final BA exports the maps of the selected model.
- **Teachers.** Each round's `visual_data/{train,val}/<id>` keeps the original keypoint ids, scores and pixels and
  replaces only `x y z` with the BA points. Observations are matched by timestamp and left-image pixel. Points lost
  in BA become NaN and fall back to the epipolar loss. No BA relative-pose loss is used.
- **Teacher filtering.**
  - `teacher_check` (used when `teacher_select` is off): removes the 10 s segments whose visual path length disagrees
    with the leg path length.
  - `teacher_select` (on by default; replaces `teacher_check`): selects keyframe pairs by robust statistics of the
    round's teacher (reprojection residual, outlier rate, rotation against the gyro, path-ratio deviation).
- **Keyframe windows** (`alternating.window`, default `frames: 5`, `max_seconds: 10`). Training and validation use
  windows of consecutive keyframes.
  - Each window adds the motion from its first frame to every later frame, weighted by `cross_frame_weight`, to the
    adjacent pairs.
  - Windows never cross a filtered pair.
  - The batch size counts windows; it is the per-pair batch size divided by `frames − 1`.
  - `train.batches_per_epoch` fixes the number of batches per epoch.

**Fused teacher (VCE).** With `alternating.vce.enabled: true`, every fused round builds the teacher in two stages:

1. Window VCE estimates the variance scales of the visual and leg terms.
2. A global fusion BA runs with these scales over the visual-only map, using the latest model's velocities and
   variances.

Numerical settings are in `alternating.vce.parameters`, or in a YAML file with a `vce:` section
(`alternating.vce.config_file`).

The results are in `s22/round_NNN/<sequence>/vce/{windows,fusion}`; failed attempts are kept as `vce_attempt_NNN`.

- **Failed windows.** Windows that fail or do not converge use α = 1.
- **Failed global VCE.** If it does not converge, the fixed-weight solution is used. If that also fails, the round
  stops instead of training on an invalid teacher.
- **Record of the choice.** `fusion/global_report.yaml` records the selected solution and the fallback reason.

The fused map keeps only the valid observations of the fusion BA, so rejected observations do not return as epipolar
supervision.

**Restarts.** Finished BA rounds and model updates are reused. Initialize and the training rounds resume at epoch
boundaries (the optimizers restart). After changing the rounds, the fusion or the window settings, use a new
`train.output_dir`.

### Outputs and install

```
<output_dir>/
├── initialize/              pressure normalization, extrinsic
├── s22/
│   ├── prepare/, holdout/   visual teacher and held-out validation segments
│   ├── round_000/ ...       per round: BA results per sequence, visual_head/ (trained model), update.json
│   └── result.json          final_checkpoint, best_round, early_stopped, trajectory_score
└── visual.yaml              config snapshot
```

Install the final model:

```bash
cp $(python -c "import json; print(json.load(open('<output_dir>/s22/result.json'))['final_checkpoint'])") \
   ../../../robots/go2/checkpoints/leg_odometry.pt
```

To check the network alone (dead reckoning on a recorded `leg_odom/` directory):

```bash
python stage4.py infer --data-root <map_root>/leg_odom --checkpoint ../../../robots/go2/checkpoints/leg_odometry.pt \
  --output <out>/network_trajectory.txt
```

## S3: online fusion and adaptation

S3 needs no separate training run. It loads the S2 checkpoint of `online: checkpoint` and adapts the network while it
runs. Run it as in the [main README](../README.md#running-airleg-online-s3):

```bash
ros2 launch air_slam_ros2 visual_odometry.launch.py robot:=go2 saving_dir:=<run_dir> online_mode:=online_adapt
```

The online adaptation is configured in the `online:` section (`robots/common/leg_odometry.yaml`) and in the
`visual_odometry.online` section (`robots/common/algorithm.yaml`). By default:

- **Trainable parameters.** A rank-1 LoRA adapter on the last feed-forward layer of the backbone (1280 parameters)
  changes only the linear-velocity output. The adapter starts at zero, so the output equals S2 until the first update.
- **Supervision.** The matured keyframes of the fused back end. The newest 10 keyframes are skipped because they are
  still being optimized. A quality gate removes keyframe pairs on which the front end and the back end disagree.
- **Schedule.** One round every 3 matured keyframes, with a 10-keyframe window and 5 steps at learning rate 3e-4. If
  the adapted model stays worse than S2, it is reset to S2.
- **Gyro bias.** The bias is estimated by the back end and fed to the leg network (`online: bias.source: backend`).

The adapted weights exist only in memory: every run starts again from the S2 checkpoint.

| `online_mode` | Leg odometry in the fusion | Online adaptation |
|---|---|---|
| `online_adapt` (default) | yes | yes |
| `fusion_only` | yes | no (fixed S2 network) |
| `visual_only` | no | no |

## S4 (optional): global back end

```bash
python pipeline.py s4 --robot go2 --map-root <run_dir> [--output-dir <dir>]
```

S4 needs a run saved with `record_leg_odom:=true`. It runs three steps:

1. *prepare*: loop detection and a pose graph;
2. *global*: optimization of the whole S3 back-end graph, with the loop edges and iterated VCE;
3. *structure*: map points only, with the poses fixed.

Settings are in the `global_backend:` section of `robots/<robot>/algorithm.yaml`. The steps run in
`<output_dir>/{prepare,global,structure}/` (default `<run_dir>/leg_odom/tmp/stage4/`); finished steps are reused, and
each step writes `<step>.log`. The final `trajectory_v1.txt` (global poses) and `AirSLAM_mapv1.bin` (structure map)
are copied back to the run directory; `result.json` records the number of loops and whether the global BA was
rejected (`structure` exit code 5). The network is not changed.

## Evaluation

All trajectories are in TUM format (`time tx ty tz qx qy qz qw`, camera pose `T_W_C`). The on-line result is
`trajectory_backend_rt.txt`. Compare it with ground truth using any TUM tool, e.g. [evo](https://github.com/MichaelGrupp/evo):

```bash
evo_ape tum <ground_truth.txt> <run_dir>/trajectory_backend_rt.txt -a
```

The camera ← base extrinsic `T_C_B` is published on `/airslam/leg_odom/extrinsic` and stored in the S2 checkpoint.
You need it when the ground truth is a base or IMU trajectory.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named 'torch'` in S2 / S4 / S3 worker | Wrong interpreter: activate the training environment; for S3 set `AIRSLAM_TORCH_PYTHON` |
| S2 prepare hangs at the start | The first run builds the TensorRT engines (minutes). `prepare.readiness_timeout_s` allows up to 1 h |
| S2 prepare loses messages | Bag on a slow disk: keep `playback_topics: auto` and `playback_qos_depth: 100`, or lower `playback_rate` |
| `Build map_optimization_backend or set alternating.backend explicitly` | Build the workspace and source `install/setup.bash` before running S2 |
| S3 log shows `foot=0/0` | The Fast-FoundationStereo engine is missing: build the engines once (main README, Build) |
