#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "utils.h"

// Path-valued keys of the algorithm config (at any depth), relative to the file that defines them.
inline bool IsConfigPathKey(const std::string& key){
  return key == "model_dir" || key == "camera_config_path" || key == "voc_path" || key == "ffs_yaml" ||
         key == "checkpoint";
}

inline void ResolveConfigPaths(YAML::Node node, const std::filesystem::path& dir){
  if(!node.IsMap()) return;
  for(auto it = node.begin(); it != node.end(); ++it){
    const std::string key = it->first.as<std::string>();
    if(it->second.IsMap()){
      ResolveConfigPaths(it->second, dir);
    }else if(it->second.IsScalar() && IsConfigPathKey(key)){
      const auto value = it->second.as<std::string>();
      if(!value.empty()) it->second = (dir / value).lexically_normal().string();
    }
  }
}

// Mappings merge key by key (override wins), anything else is replaced.
inline YAML::Node MergeConfig(const YAML::Node& base, const YAML::Node& override_node){
  if(!base.IsMap() || !override_node.IsMap()) return YAML::Clone(override_node);
  YAML::Node out = YAML::Clone(base);
  for(auto it = override_node.begin(); it != override_node.end(); ++it){
    const std::string key = it->first.as<std::string>();
    out[key] = out[key] && out[key].IsMap() && it->second.IsMap() ? MergeConfig(out[key], it->second)
                                                                   : YAML::Clone(it->second);
  }
  return out;
}

// YAML file with inheritance: `base: <path>` (relative to the file) names a parent loaded first (recursively);
// the file's keys are deep-merged over it. Path keys coming from a parent are resolved against the parent's
// directory (absolute); the file's own values are returned unchanged. Files are taken by their real path (install
// space symlinks resolved). Same rules as TRAIL common/yaml_base.py.
inline YAML::Node LoadYamlWithBase(const std::string& config_file, int depth = 0){
  if(depth > 16) throw std::runtime_error("config base chain too deep (cycle?): " + config_file);
  const auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(config_file));
  YAML::Node root = YAML::LoadFile(path.string());
  if(!root.IsMap() || !root["base"]) return root;
  const auto parent_path = std::filesystem::weakly_canonical(path.parent_path() / root["base"].as<std::string>());
  root.remove("base");
  YAML::Node parent = LoadYamlWithBase(parent_path.string(), depth + 1);
  ResolveConfigPaths(parent, parent_path.parent_path());
  return MergeConfig(parent, root);
}

// Resolve algorithm-owned paths independently of the process working directory.
// Explicit ROS parameter overrides retain their usual path semantics.
inline YAML::Node LoadAlgorithmConfig(const std::string& config_file){
  YAML::Node root = LoadYamlWithBase(config_file);
  const auto base = std::filesystem::weakly_canonical(std::filesystem::absolute(config_file)).parent_path();
  for(const auto* key : {"model_dir", "camera_config_path", "voc_path"}){
    if(!root[key]) continue;
    const auto value = root[key].as<std::string>();
    if(value.empty()) continue;
    root[key] = (base / value).lexically_normal().string();
  }
  return root;
}

inline YAML::Node RequireConfigSection(const YAML::Node& root, const std::string& section){
  if(!root[section]){
    throw std::runtime_error("missing required algorithm config section: " + section);
  }
  return root[section];
}

inline YAML::Node RequireRelocalizationSection(const YAML::Node& root){
  if(!root["relocalization"]){
    throw std::runtime_error("missing required algorithm config section: relocalization");
  }
  return root["relocalization"];
}


struct SuperPointConfig {
  SuperPointConfig() {}
  // network input size; 0 = the image size (default: no 512x512 stretching)
  int input_width = 0, input_height = 0;
  // descriptor sampling at the official SuperPoint cell centres ((x - s/2 + 0.5) / (w s - s/2 - 0.5) (w - 1), ~(x - 3.5) / 8)
  // instead of x / 8
  bool official_desc_sampling = false;
  void Load(const YAML::Node& superpoint_node){
    if(superpoint_node["official_desc_sampling"]) official_desc_sampling = superpoint_node["official_desc_sampling"].as<bool>();
    max_keypoints = superpoint_node["max_keypoints"].as<int>();
    keypoint_threshold = superpoint_node["keypoint_threshold"].as<float>();
    remove_borders = superpoint_node["remove_borders"].as<int>();
    if(superpoint_node["input_width"]) input_width = superpoint_node["input_width"].as<int>();
    if(superpoint_node["input_height"]) input_height = superpoint_node["input_height"].as<int>();
    dla_core = superpoint_node["dla_core"].as<int>();
    const YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_input_tensor_names; i++){
      input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_output_tensor_names; i++){
      output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
    }
    onnx_file = superpoint_node["onnx_file"].as<std::string>();
    engine_file= superpoint_node["engine_file"].as<std::string>();

    grid_filter = superpoint_node["grid_filter"]
        ? superpoint_node["grid_filter"].as<bool>() : false;
    grid_cols = superpoint_node["grid_cols"]
        ? superpoint_node["grid_cols"].as<int>() : 1;
    grid_rows = superpoint_node["grid_rows"]
        ? superpoint_node["grid_rows"].as<int>() : 1;
    h_thr = superpoint_node["h_thr"]
        ? superpoint_node["h_thr"].as<float>() : keypoint_threshold;
    l_thr = superpoint_node["l_thr"]
        ? superpoint_node["l_thr"].as<float>() : keypoint_threshold;
    if(grid_filter && (grid_cols <= 0 || grid_rows <= 0 ||
        max_keypoints < grid_cols * grid_rows || l_thr < 0.0f || h_thr < l_thr)){
      throw std::runtime_error(
          "invalid SuperPoint grid filter: require positive grid size, "
          "max_keypoints >= grid_cols*grid_rows and 0 <= l_thr <= h_thr");
    }
  }

  int max_keypoints;
  float keypoint_threshold;
  bool grid_filter = false;
  int grid_cols = 1;
  int grid_rows = 1;
  float h_thr = 0.0f;
  float l_thr = 0.0f;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct PointMatcherConfig {
  PointMatcherConfig() {}
  void Load(const YAML::Node& point_matcher_node){
    matcher = point_matcher_node["matcher"].as<int>();
    // image size of the LightGlue keypoint normalisation; absent / 0 = the camera's (camera yaml image_width /
    // image_height, filled in by WithImageSize where the camera is known)
    image_width = point_matcher_node["image_width"] ? point_matcher_node["image_width"].as<int>() : 0;
    image_height = point_matcher_node["image_height"] ? point_matcher_node["image_height"].as<int>() : 0;
    onnx_file = point_matcher_node["onnx_file"].as<std::string>();
    engine_file = point_matcher_node["engine_file"].as<std::string>();
  }

  // copy with the image size taken from the camera where the config leaves it at 0
  PointMatcherConfig WithImageSize(int width, int height) const {
    PointMatcherConfig config = *this;
    if(config.image_width <= 0) config.image_width = width;
    if(config.image_height <= 0) config.image_height = height;
    return config;
  }

  int matcher;
  int image_width = 0;
  int image_height = 0;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct KeyframeConfig {
  KeyframeConfig() {}
  void Load(const YAML::Node& keyframe_node){
    min_init_stereo_feature = keyframe_node["min_init_stereo_feature"].as<int>();
    lost_num_match = keyframe_node["lost_num_match"].as<int>();
    min_num_match = keyframe_node["min_num_match"].as<int>();
    max_num_match = keyframe_node["max_num_match"].as<int>();
    tracking_point_rate = keyframe_node["tracking_point_rate"].as<float>();
    tracking_parallax_rate = keyframe_node["tracking_parallax_rate"].as<double>();
    if(keyframe_node["lost_handling"]) lost_handling = keyframe_node["lost_handling"].as<bool>();
    if(keyframe_node["lost_prefer_leg"]) lost_prefer_leg = keyframe_node["lost_prefer_leg"].as<bool>();
  }

  // Lost frames: motion-model pose (leg > IMU > constant velocity), weak motion prior of break keyframes,
  // re-link after recovery, lost flag for the back-end. false = previous behaviour (last tracked pose).
  bool lost_handling = true;
  // a lost frame takes the motion-model pose (leg > IMU > constant velocity) even when its initial guess came from the
  // IMU preintegration (otherwise IMU dead reckoning is kept)
  bool lost_prefer_leg = false;
  int min_init_stereo_feature;
  int lost_num_match;
  int min_num_match;
  int max_num_match;
  float tracking_point_rate;
  double tracking_parallax_rate;
};

struct OptimizationConfig{
  OptimizationConfig() {}
  void Load(const YAML::Node& optimization_node){
    mono_point = optimization_node["mono_point"].as<double>();
    stereo_point = optimization_node["stereo_point"].as<double>();
  }

  double mono_point;
  double stereo_point;
};

// Configuration for the scene-specific leg-odometry refinement stage.  The
// defaults deliberately preserve the original visual-only map refinement.
template<class T> inline T ReadOr(const YAML::Node& node, const char* key, T fallback){
  return (node && node[key]) ? node[key].as<T>() : fallback;
}

struct LegOdomConfig{
  bool enabled = false;
  bool use_bias = false;
  bool optimize_extrinsic = false;
  bool use_extrinsic_prior = true;
  bool optimize_time_delay = false;

  std::string python_executable = "python3";  // ROS mode full forks TRAIL pipeline.py s4 with it

  bool has_extrinsic = false;  // extrinsic given explicitly in this config (else only the leg node's latched topic)
  Eigen::Vector3d extrinsic_xyz = Eigen::Vector3d::Zero();
  Eigen::Vector4d extrinsic_quat_xyzw = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  double time_delay = 0.0;
  double time_delay_min = -0.2;
  double time_delay_max = 0.2;
  double rotation_sigma = 0.08;
  double translation_sigma = 0.08;
  double bias_prior_sigma = 0.25;
  double bias_random_walk_sigma = 0.05;
  double extrinsic_rotation_prior_sigma = 0.20;
  double extrinsic_translation_prior_sigma = 0.20;
  double time_delay_prior_sigma = 0.03;
  double huber_delta = 2.5;
  double min_keyframe_dt = 1e-3;

  void Load(const YAML::Node& node){
    if(!node) return;
    enabled = ReadOr(node, "enabled", enabled);
    use_bias = ReadOr(node, "use_bias", use_bias);
    optimize_extrinsic = ReadOr(node, "optimize_extrinsic", optimize_extrinsic);
    use_extrinsic_prior = ReadOr(node, "use_extrinsic_prior", use_extrinsic_prior);
    optimize_time_delay = ReadOr(node, "optimize_time_delay", optimize_time_delay);
    python_executable = ReadOr(node, "python_executable", python_executable);
    if(node["extrinsic_xyz"]){
      for(int i = 0; i < 3; ++i) extrinsic_xyz[i] = node["extrinsic_xyz"][i].as<double>();
      has_extrinsic = true;
    }
    if(node["extrinsic_quat_xyzw"]){
      for(int i = 0; i < 4; ++i) extrinsic_quat_xyzw[i] = node["extrinsic_quat_xyzw"][i].as<double>();
      if(extrinsic_quat_xyzw.norm() < 1e-8) throw std::invalid_argument("Invalid leg extrinsic quaternion");
      extrinsic_quat_xyzw.normalize();
      has_extrinsic = true;
    }
    time_delay = ReadOr(node, "time_delay", time_delay);
    time_delay_min = ReadOr(node, "time_delay_min", time_delay_min);
    time_delay_max = ReadOr(node, "time_delay_max", time_delay_max);
    rotation_sigma = ReadOr(node, "rotation_sigma", rotation_sigma);
    translation_sigma = ReadOr(node, "translation_sigma", translation_sigma);
    bias_prior_sigma = ReadOr(node, "bias_prior_sigma", bias_prior_sigma);
    bias_random_walk_sigma = ReadOr(node, "bias_random_walk_sigma", bias_random_walk_sigma);
    extrinsic_rotation_prior_sigma = ReadOr(node, "extrinsic_rotation_prior_sigma", extrinsic_rotation_prior_sigma);
    extrinsic_translation_prior_sigma = ReadOr(node, "extrinsic_translation_prior_sigma", extrinsic_translation_prior_sigma);
    time_delay_prior_sigma = ReadOr(node, "time_delay_prior_sigma", time_delay_prior_sigma);
    huber_delta = ReadOr(node, "huber_delta", huber_delta);
    min_keyframe_dt = ReadOr(node, "min_keyframe_dt", min_keyframe_dt);
  }
};

// S3 superframe dense depth, terrain map and foot-surface factor settings (SI units).
struct SurfaceConfig{
  bool enabled = true;
  std::string ffs_yaml;
  // superframe selection
  double min_dt = 0.5, max_dt = 3.0, trans = 0.6, rot_deg = 20.0;
  double min_foot_coverage = 0.5;
  int max_depth_queue = 2;
  int max_superframes = 40;
  // conservative depth validity
  double min_depth = 0.3, max_depth = 3.5, min_disp = 2.0, grad_rel = 0.05;
  int border = 16, stride = 2;
  // anisotropic voxels in the gravity-aligned superframe frame
  double voxel_xy = 0.02, voxel_z = 0.01;
  // plane fit and step detection
  double radius = 0.12, layer_band = 0.05, step_band = 0.30;
  double edge_margin = 0.03, edge_sigma_k = 2.0, pose_sigma_xy = 0.02;
  double step_min = 0.03, max_height_span = 0.06;
  int min_pts = 30, min_quadrant_pts = 3;
  double min_inlier_ratio = 0.8, max_rmse = 0.01, planarity = 0.02, min_extent_ratio = 0.3;
  double max_slope_deg = 5.0;  // = the back-end foot gate
  // foot factor
  // foot factor (the edges and the contact offset are estimated in the back-end only, see BackendConfig)
  double foot_radius = 0.022, sigma = 0.01, depth_sigma_coeff = 0.002;
  // largest |contact offset| allowed by the foot-plane verification gate (= max(-sink_min, sink_max) of the back-end)
  double contact_offset_margin = 0.08;
  int max_obs_per_segment = 4, num_anchor_superframes = 3, max_anchor_trials = 3;
  // Foot-position correction applied to the leg-kinematics foot before the foot-surface association,
  // f' = Ry(pitch) Rx(roll) f + [0, 0, dz] in B (per robot). Fitted from the foot-terrain residuals: the FFS ground agrees
  // with the stereo landmarks, so the offset is in the camera-base-leg chain (ANYmal roll ~1.5 deg, dz ~17 mm;
  // Go2 dz ~ -59 mm).
  // old_plane: elevation-map anchors. Every superframe keeps its good planes (RANSAC on the 5 x 5 x 2 cm cloud: fit rmse,
  // inliers, extent, tilt <= plane_tilt_deg, slopes count). A foot associated as usual also gets one edge against the
  // oldest good plane under it of a superframe >= old_plane_dt s earlier (inlier within old_plane_near of the projected
  // foot, foot within foot_radius + old_plane_gate of the plane): drift-free plane of one superframe, long baseline (revisits).
  double foot_corr_roll_deg = 0.0, foot_corr_pitch_deg = 0.0, foot_corr_dz = 0.0;
  bool old_plane = false;
  double old_plane_dt = 30.0, old_plane_near = 0.10, old_plane_gate = 0.08;
  double plane_ransac_thr = 0.02, plane_rmse = 0.01, plane_tilt_deg = 25.0, plane_min_extent = 0.5;
  int plane_min_inliers = 150, plane_ransac_iter = 150, plane_max = 4;

  void Load(const YAML::Node& n, const std::string& base_dir){
    if(!n) return;
    enabled = ReadOr(n, "enabled", enabled);
    ffs_yaml = ReadOr(n, "ffs_yaml", ffs_yaml);
    if(!ffs_yaml.empty() && ffs_yaml.front() != '/') ffs_yaml = base_dir + "/" + ffs_yaml;
#define AIRSLAM_READ(key) key = ReadOr(n, #key, key)
    AIRSLAM_READ(min_dt); AIRSLAM_READ(max_dt); AIRSLAM_READ(trans); AIRSLAM_READ(rot_deg);
    AIRSLAM_READ(min_foot_coverage); AIRSLAM_READ(max_depth_queue); AIRSLAM_READ(max_superframes);
    AIRSLAM_READ(min_depth); AIRSLAM_READ(max_depth); AIRSLAM_READ(min_disp); AIRSLAM_READ(grad_rel);
    AIRSLAM_READ(border); AIRSLAM_READ(stride); AIRSLAM_READ(voxel_xy); AIRSLAM_READ(voxel_z);
    AIRSLAM_READ(radius); AIRSLAM_READ(layer_band); AIRSLAM_READ(step_band);
    AIRSLAM_READ(edge_margin); AIRSLAM_READ(edge_sigma_k); AIRSLAM_READ(pose_sigma_xy);
    AIRSLAM_READ(step_min); AIRSLAM_READ(max_height_span); AIRSLAM_READ(min_pts); AIRSLAM_READ(min_quadrant_pts);
    AIRSLAM_READ(foot_corr_roll_deg); AIRSLAM_READ(foot_corr_pitch_deg); AIRSLAM_READ(foot_corr_dz);
    AIRSLAM_READ(old_plane); AIRSLAM_READ(old_plane_dt); AIRSLAM_READ(old_plane_near); AIRSLAM_READ(old_plane_gate);
    AIRSLAM_READ(plane_ransac_thr); AIRSLAM_READ(plane_rmse); AIRSLAM_READ(plane_tilt_deg); AIRSLAM_READ(plane_min_extent);
    AIRSLAM_READ(plane_min_inliers); AIRSLAM_READ(plane_ransac_iter); AIRSLAM_READ(plane_max);
    AIRSLAM_READ(min_inlier_ratio); AIRSLAM_READ(max_rmse); AIRSLAM_READ(planarity); AIRSLAM_READ(min_extent_ratio);
    AIRSLAM_READ(max_slope_deg); AIRSLAM_READ(foot_radius); AIRSLAM_READ(sigma); AIRSLAM_READ(depth_sigma_coeff);
    AIRSLAM_READ(max_obs_per_segment); AIRSLAM_READ(num_anchor_superframes); AIRSLAM_READ(max_anchor_trials);
    AIRSLAM_READ(contact_offset_margin);
#undef AIRSLAM_READ
    if(!(voxel_xy > 0.0 && voxel_z > 0.0 && radius > 0.0 && min_depth > 0.0 && max_depth > min_depth))
      throw std::invalid_argument("invalid surface voxel/depth settings");
  }
};

// S3 light-weight fusion back-end: pose-only graph over finalized keyframes (keyframes that left the
// front-end local BA window). Visual relative poses are compressed into one edge per keyframe pair.
struct BackendConfig{
  bool enabled = true;
  bool write_back = false;   // apply back-end poses to the front-end keyframes and map points
  bool foot = true;          // foot-surface edges (front-end associations)
  bool vce = true;           // online VCE of the leg (rot / horizontal / vertical) and foot groups, bounded
  bool degen = true;         // per-edge visual down-weighting by covisible support
  // solve (and back-end VCE) once per finalized superframe node; "stride": every `stride` nodes. With
  // "superframe", `stride` is the fallback when no superframe arrives (standing still, no dense depth).
  std::string trigger = "superframe";
  int window = 50, stride = 10, iterations = 10;
  double huber = 3.0;        // on the whitened edge norm
  // base sigmas (rad; m in the camera frame, y ~ vertical), measured against lidar
  std::vector<double> vo_sigma = {0.002, 0.05, 0.02, 0.05};
  std::vector<double> leg_sigma = {0.0094, 0.017, 0.035, 0.017};
  // foot edges: contact offset delta shared by blocks of `sink_block` nodes, random walk between blocks
  int sink_block = 10;
  double foot_sigma_min = 0.01, sink_rw = 0.01, sink_prior = 0.05, sink_min = -0.06, sink_max = 0.08;
  double foot_max_slope_deg = 5.0, foot_max_rmse = 0.01;  // <= 0 disables the gate
  // visual edge information *= clamp(min(n/n0,1) * min(spread/spread0,1), min_scale, 1);
  // n = covisible map points of the keyframe pair, spread = their image spread (uniform = 1)
  // leg translation variance x clamp(slip / running median of slip, 1, slip_max_factor); slip = drift of the
  // stance feet (same contact segment at both keyframes) under the leg relative motion
  bool leg_slip = true;
  int slip_window = 200;
  double slip_max_factor = 10.0;
  // residual angular-rate (gyro) bias of the leg rotation (base frame B, on top of the leg node's bias), one state per block of
  // gyro_bias_block nodes with a random walk between blocks; absolute prior only on the first block
  bool gyro_bias = true;
  int gyro_bias_block = 10;
  double gyro_bias_prior = 0.005, gyro_bias_rw = 0.001;  // rad/s
  // gyro bias unification:
  //  gyro_bias_total: the leg rate keeps the bias the worker subtracted (LegOdometry gyro_bias), so the leg gyro bias
  //    blocks are the total bias (first-block prior centred on the worker's bias); after every solve the newest block is
  //    published on /airslam/backend/gyro_bias for the worker (TRAIL s3.yaml bias.source: backend: its only gyro bias)
  //  net_gyro: every raw IMU gyro sample (front-end attitude filter without its integral term, back-end IMU
  //    preintegration and ZUPT) gets the network correction of the nearest network sample (network rate - R_BI x the raw
  //    gyro at that sample = gyro-head correction - worker bias; one correction for its ~4 raw samples at 400 / 100 Hz).
  //    Needs the network samples on the IMU clock (GrandTour: both lowstate). net_gyro_R_BI (row-major) = the worker's
  //    IMU extrinsic. The back-end IMU gyro bias stays a separate state (tying it to the leg bias, or replacing the raw
  //    gyro by the 100 Hz network rate, was worse)
  bool gyro_bias_total = false, net_gyro = false;
  std::vector<double> net_gyro_R_BI = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  // visual translation scale: the visual relative translations of a block of vo_scale_block nodes are scaled by
  // (1 + v_b), v_b with a prior N(0, vo_scale_prior^2) on every block and a random walk vo_scale_rw between blocks; the
  // leg edges observe it (a stereo depth scale error). Off: the plain visual edge.
  bool vo_scale = false;
  int vo_scale_block = 10;
  double vo_scale_prior = 0.05, vo_scale_rw = 0.01;
  // front-end export of the extra back-end inputs (OnlineFusion::Finalize): raw IMU samples and every point
  // observation of the node (also dumped for backend_replay)
  bool export_extras = false;
  // visual constraint: visual_marg none (the fixed-sigma vo edge) | win_pair (every solve marginalises the window's
  // point observations, linearised at the front-end poses and points, into the information of each consecutive pair,
  // marginal.h; it replaces the fixed sigmas and the covisibility weighting)
  std::string visual_marg = "none";
  double marg_pixel_sigma = 1.0, marg_huber = 2.5;
  // IMU preintegration edges between nodes (edge_imu_backend.h; needs imu_mode > 0 and export_extras): velocity per
  // node, gyro / acc bias blocks of imu_bias_block nodes (prior on the first block, random walk from the IMU noise
  // model, floored by imu_bg_rw_min / imu_ba_rw_min; a block beyond imu_bg_max / imu_ba_max restarts at zero), world
  // gravity direction (imu_gravity_free: free with a prior of imu_gravity_prior_deg, else fixed), camera-IMU rotation
  // correction (imu_ext_rot, prior imu_ext_prior_deg), constant time offset (t_camera = t_imu + offset).
  // imu_bias_block 1 keeps the biases near zero (ANYmal: ~1e-3 rad/s), robust to a start-up attitude / extrinsic error.
  bool imu = false;
  int imu_bias_block = 10;
  double imu_bg_prior = 0.01, imu_ba_prior = 0.2, imu_time_offset = 0.0;
  bool imu_ext_rot = false, imu_gravity_free = false;
  double imu_ext_prior_deg = 3.0, imu_gravity_prior_deg = 2.0;
  double imu_bg_rw_min = 0.0, imu_ba_rw_min = 0.0, imu_bg_max = 0.0, imu_ba_max = 0.0;
  // speed: the online VCE uses only the leg / foot / plain visual factors and the states they touch (the full-
  // information factors -- marginalised visual, IMU and its priors -- belong to known groups anyway)
  bool vce_skip_full = false;
  // selected landmarks in the window (visual_mode landmarks; landmarks.h): <= landmarks_max points observed by
  // >= landmarks_min_nodes nodes, one per landmarks_voxel, reprojection edges with landmark_pixel_sigma; a node with
  // >= landmarks_min_per_node of them and no marginalised factor needs no visual edge. lm_lambda_init > 0 fixes the
  // initial LM damping (needed with the IMU factors). land_fallback: an outlier ratio above it drops the solve's
  // landmarks. land_analytic: analytic Jacobians of the scaled stereo edges; vce_skip_landmarks: VCE without them.
  // vce_landmarks > 0 (with vce_skip_landmarks false): only the edges of the vce_landmarks lowest-id landmarks enter VCE
  std::string visual_mode = "relpose";
  int landmarks_max = 100, landmarks_min_nodes = 3, landmarks_min_per_node = 10;
  double landmarks_voxel = 0.5, landmark_pixel_sigma = 1.0, lm_lambda_init = 0.0, land_fallback = 0.0;
  bool land_analytic = false, vce_skip_landmarks = false;
  int vce_landmarks = 0;
  // VCE leaves out the landmarks whose used observations do not determine the point (ill-conditioned 3x3 information:
  // otherwise the VCE normal system is rank deficient; S4 joint solve)
  bool vce_point_check = false;
  // solve sanity checks (0 = off): camera-IMU rotation correction (deg), |visual scale offset|
  double sanity_ext_deg = 0.0, sanity_vo_scale = 0.0;
  // zero-velocity factor when the robot stands (IMU gyro peak, acc std and leg translation below the thresholds)
  bool zupt = false;
  double zupt_gyro = 0.05, zupt_acc_std = 0.1, zupt_leg = 0.01, zupt_sigma_deg = 0.05, zupt_sigma_t = 0.002;
  // tilt_prior: at every node that passes the zupt stationarity test, an absolute roll / pitch
  // factor: the corrected IMU z axis seen in the world, (Rwb Exp(d))^T e_z, = the bias-corrected mean accelerometer
  // direction, sigma tilt_sigma_deg; e_z = the fixed world vertical (the frame the gravity-direction prior refers to)
  bool tilt_prior = false;
  double tilt_sigma_deg = 1.0;
  double win_leg_chi2 = 0.0;   // > 0: visual-leg translation chi2 gate of the win_pair factors (fallback: fixed-sigma edge)
  double land_leg_chi2 = 0.0;  // > 0: the same gate breaks the selected landmarks' tracks at the node
  // passive motion: a node at which the robot did not step since the previous one (every foot within passive_foot_m of
  // its position in B) but whose visual translation exceeds passive_vo_t (m, <= 0: off), visually tracked (not lost,
  // covisible >= degen_n0), is carried by an external motion (pushed, sliding, on a platform): the stance-foot assumption
  // fails, so no leg factor, no zero-velocity factor and no visual-leg gate at it. imu_max_dt (s, <= 0: off): no IMU
  // preintegration factor over an interval longer than imu_max_dt unless the robot stands (leg translation <
  // imu_max_dt_leg): such a factor is dominated by the bias and velocity errors (camera dropouts); the leg bridges it
  double passive_vo_t = 0.0, passive_foot_m = 0.05, imu_max_dt = 0.0, imu_max_dt_leg = 0.05;
  // 3-D linear-velocity bias of the leg translation (base frame B, m/s) per block = the keyframes between two
  // superframes (leg_corr_superframe = 1, else leg_vbias_block nodes), random walk between blocks, prior on the first:
  // body translation t - 0.5 (I + R) v_b dt. Needs gyro_bias. Pulls the leg scale / drift towards the visual one: helps
  // when vision is accurate, hurts when it is biased. Off by default.
  bool leg_vbias = false;
  int leg_corr_superframe = 1, leg_vbias_block = 20;
  double leg_vbias_prior = 0.05, leg_vbias_rw = 0.01;  // m/s
  int degen_n0 = 100, degen_lost = 15;
  double degen_spread0 = 0.3, degen_min_scale = 0.01;
  // learned leg reliability: leg translation variance x exp(theta^T (f - mean f)), clamped to
  // [leg_rel_lo, leg_rel_hi]. f = interval features / leg_rel_feature_scale: log slip, mean |v|, mean |omega|,
  // log mean network linear variance, std of |acc| (robot accelerometer, online.accel_topic). theta = ridge
  // regression of the log whitened leg translation residual of nodes that stayed leg_rel_lag nodes in the window
  // (against the back-end poses) on the centred features; refitted after every solve once leg_rel_min samples exist.
  // global back-end (S4): loop-closure relative-pose edges (AddLoop), sigmas {rot, x, y, z}; VCE + re-solve rounds
  // per solve (a single global solve needs several: each VCE update is limited to a factor vce_step)
  std::vector<double> loop_sigma = {0.01, 0.05, 0.05, 0.05};
  bool loop_robust = true;
  int vce_rounds = 1;
  bool leg_reliability = false;
  int leg_rel_min = 50, leg_rel_history = 2000, leg_rel_lag = 40;
  double leg_rel_ridge = 1.0, leg_rel_lo = 0.3, leg_rel_hi = 10.0;
  std::vector<double> leg_rel_feature_scale = {0.715, 0.412, 0.331, 0.384, 1.264};  // pooled std
  double vce_damping = 0.5, vce_lower = 0.5, vce_upper = 2.0, vce_step = 2.0, vce_min_redundancy = 5.0;

  void Load(const YAML::Node& n){
    if(!n) return;
#define AIRSLAM_READ(key) key = ReadOr(n, #key, key)
    AIRSLAM_READ(enabled); AIRSLAM_READ(write_back); AIRSLAM_READ(foot); AIRSLAM_READ(vce); AIRSLAM_READ(degen);
    AIRSLAM_READ(trigger); AIRSLAM_READ(window); AIRSLAM_READ(stride); AIRSLAM_READ(iterations); AIRSLAM_READ(huber);
    AIRSLAM_READ(vo_sigma); AIRSLAM_READ(leg_sigma); AIRSLAM_READ(sink_block); AIRSLAM_READ(foot_sigma_min);
    AIRSLAM_READ(sink_rw); AIRSLAM_READ(sink_prior); AIRSLAM_READ(sink_min); AIRSLAM_READ(sink_max);
    AIRSLAM_READ(foot_max_slope_deg); AIRSLAM_READ(foot_max_rmse); AIRSLAM_READ(degen_n0); AIRSLAM_READ(degen_lost);
    AIRSLAM_READ(leg_slip); AIRSLAM_READ(slip_window); AIRSLAM_READ(slip_max_factor);
    AIRSLAM_READ(gyro_bias); AIRSLAM_READ(gyro_bias_block); AIRSLAM_READ(gyro_bias_prior); AIRSLAM_READ(gyro_bias_rw);
    AIRSLAM_READ(gyro_bias_total); AIRSLAM_READ(net_gyro); AIRSLAM_READ(net_gyro_R_BI);
    AIRSLAM_READ(vo_scale); AIRSLAM_READ(vo_scale_block); AIRSLAM_READ(vo_scale_prior); AIRSLAM_READ(vo_scale_rw);
    AIRSLAM_READ(export_extras); AIRSLAM_READ(visual_marg); AIRSLAM_READ(marg_pixel_sigma); AIRSLAM_READ(marg_huber);
    AIRSLAM_READ(imu); AIRSLAM_READ(imu_bias_block); AIRSLAM_READ(imu_bg_prior); AIRSLAM_READ(imu_ba_prior);
    AIRSLAM_READ(imu_time_offset); AIRSLAM_READ(imu_ext_rot); AIRSLAM_READ(imu_gravity_free);
    AIRSLAM_READ(imu_ext_prior_deg); AIRSLAM_READ(imu_gravity_prior_deg); AIRSLAM_READ(imu_bg_rw_min);
    AIRSLAM_READ(imu_ba_rw_min); AIRSLAM_READ(imu_bg_max); AIRSLAM_READ(imu_ba_max); AIRSLAM_READ(vce_skip_full);
    AIRSLAM_READ(visual_mode); AIRSLAM_READ(landmarks_max); AIRSLAM_READ(landmarks_min_nodes);
    AIRSLAM_READ(landmarks_min_per_node); AIRSLAM_READ(landmarks_voxel); AIRSLAM_READ(landmark_pixel_sigma);
    AIRSLAM_READ(lm_lambda_init); AIRSLAM_READ(land_fallback); AIRSLAM_READ(land_analytic); AIRSLAM_READ(vce_skip_landmarks);
    AIRSLAM_READ(vce_landmarks); AIRSLAM_READ(vce_point_check);
    AIRSLAM_READ(sanity_ext_deg); AIRSLAM_READ(sanity_vo_scale); AIRSLAM_READ(zupt); AIRSLAM_READ(zupt_gyro);
    AIRSLAM_READ(zupt_acc_std); AIRSLAM_READ(zupt_leg); AIRSLAM_READ(zupt_sigma_deg); AIRSLAM_READ(zupt_sigma_t); AIRSLAM_READ(tilt_prior); AIRSLAM_READ(tilt_sigma_deg);
    AIRSLAM_READ(win_leg_chi2); AIRSLAM_READ(land_leg_chi2);
    AIRSLAM_READ(passive_vo_t); AIRSLAM_READ(passive_foot_m); AIRSLAM_READ(imu_max_dt); AIRSLAM_READ(imu_max_dt_leg);
    AIRSLAM_READ(leg_vbias); AIRSLAM_READ(leg_corr_superframe); AIRSLAM_READ(leg_vbias_block); AIRSLAM_READ(leg_vbias_prior);
    AIRSLAM_READ(leg_vbias_rw);
    AIRSLAM_READ(degen_spread0); AIRSLAM_READ(degen_min_scale); AIRSLAM_READ(vce_damping); AIRSLAM_READ(vce_lower);
    AIRSLAM_READ(vce_upper); AIRSLAM_READ(vce_step); AIRSLAM_READ(vce_min_redundancy);
    AIRSLAM_READ(loop_sigma); AIRSLAM_READ(loop_robust); AIRSLAM_READ(vce_rounds);
    AIRSLAM_READ(leg_reliability); AIRSLAM_READ(leg_rel_min); AIRSLAM_READ(leg_rel_history); AIRSLAM_READ(leg_rel_lag);
    AIRSLAM_READ(leg_rel_ridge); AIRSLAM_READ(leg_rel_lo); AIRSLAM_READ(leg_rel_hi); AIRSLAM_READ(leg_rel_feature_scale);
#undef AIRSLAM_READ
    if(vo_sigma.size() != 4 || leg_sigma.size() != 4) throw std::invalid_argument("backend.vo_sigma/leg_sigma need 4 values");
    if(trigger != "superframe" && trigger != "stride")
      throw std::invalid_argument("backend.trigger must be superframe|stride");
    if(window < 2 || stride < 1) throw std::invalid_argument("backend.window >= 2 and backend.stride >= 1 required");
    if(loop_sigma.size() != 4) throw std::invalid_argument("backend.loop_sigma needs 4 values");
    if(leg_rel_feature_scale.size() != 5) throw std::invalid_argument("backend.leg_rel_feature_scale needs 5 values");
  }
};

// S3 online fusion. Mode visual_only (default) keeps the original pipeline untouched.
struct OnlineFusionConfig{
  std::string mode = "visual_only";  // online_adapt | fusion_only | visual_only
  std::string leg_topic = "/airslam/leg_odom";
  std::string accel_topic = "/lowstate_stamped";  // robot accelerometer, subscribed only for backend.leg_reliability
  double buffer_s = 60.0;
  bool frame_leg_prior = true;
  LegOdomConfig leg;  // T_C_B (overridden by the leg node's latched extrinsic), sigmas, huber, time_delay
  SurfaceConfig surface;
  BackendConfig backend;
  bool leg_init_guess = true;   // tracking initial pose = reference keyframe * leg relative motion
  // Late leg data: the tracking-time leg queries (initial guess, frame prior) wait up to this long for the
  // leg stream to cover the frame (network latency); 0 = no wait.
  double leg_wait_ms = 20.0;
  // leg_init_guess sanity checks (optional, off by default): PnP is solved as well; with >= min_num_match PnP
  // inliers, a leg guess further than these from the PnP pose is replaced by it; a guess further than
  // leg_init_max_jump (m) from the last reliable pose is not used. They stay disabled: enabled, they made the
  // back-end trajectory worse.
  double leg_init_check_translation = std::numeric_limits<double>::infinity();
  double leg_init_check_rotation_deg = std::numeric_limits<double>::infinity();
  double leg_init_max_jump = std::numeric_limits<double>::infinity();

  bool UseLeg() const { return mode == "online_adapt" || mode == "fusion_only"; }
  void Load(const YAML::Node& root, const YAML::Node& n, const std::string& base_dir){
    (void)root;  // online.leg does not inherit map_refinement.leg_odom (S4 settings)
    if(!n) return;
    mode = ReadOr(n, "mode", mode);
    leg_topic = ReadOr(n, "leg_topic", leg_topic);
    accel_topic = ReadOr(n, "accel_topic", accel_topic);
    buffer_s = ReadOr(n, "buffer_s", buffer_s);
    frame_leg_prior = ReadOr(n, "frame_leg_prior", frame_leg_prior);
    leg.Load(n["leg"]);
    surface.Load(n["surface"], base_dir);
    backend.Load(n["backend"]);
    leg_init_guess = ReadOr(n, "leg_init_guess", leg_init_guess);
    leg_wait_ms = ReadOr(n, "leg_wait_ms", leg_wait_ms);
    leg_init_check_translation = ReadOr(n, "leg_init_check_translation", leg_init_check_translation);
    leg_init_check_rotation_deg = ReadOr(n, "leg_init_check_rotation_deg", leg_init_check_rotation_deg);
    leg_init_max_jump = ReadOr(n, "leg_init_max_jump", leg_init_max_jump);
    SetMode(mode);
  }
  void SetMode(const std::string& m){
    if(m != "online_adapt" && m != "fusion_only" && m != "visual_only")
      throw std::invalid_argument("online.mode must be online_adapt|fusion_only|visual_only, got " + m);
    mode = m;
  }
};

struct RosPublisherConfig{
  RosPublisherConfig() {}
  void Load(const YAML::Node& ros_publisher_node){
    feature = ros_publisher_node["feature"].as<int>();
    frame_pose = ros_publisher_node["frame_pose"].as<int>();
    keyframe = ros_publisher_node["keyframe"].as<int>();
    map = ros_publisher_node["map"].as<int>();
    reloc = ros_publisher_node["reloc"].as<int>();

    if(ros_publisher_node["feature_topic"]){
      LoadTopics(ros_publisher_node);
    }
  }

  void LoadTopics(const YAML::Node& ros_publisher_topics_node){
    feature_topic = ros_publisher_topics_node["feature_topic"].as<std::string>();
    frame_pose_topic = ros_publisher_topics_node["frame_pose_topic"].as<std::string>();
    frame_odometry_topic = ros_publisher_topics_node["frame_odometry_topic"].as<std::string>();
    keyframe_topic = ros_publisher_topics_node["keyframe_topic"].as<std::string>();
    path_topic = ros_publisher_topics_node["path_topic"].as<std::string>();
    map_topic = ros_publisher_topics_node["map_topic"].as<std::string>();
    reloc_topic = ros_publisher_topics_node["reloc_topic"].as<std::string>();
  }

  void Load(const YAML::Node& ros_publisher_topics_node,
            const YAML::Node& ros_publisher_switches_node){
    Load(ros_publisher_switches_node);
    LoadTopics(ros_publisher_topics_node);
  }

  int feature;
  std::string feature_topic;
  int frame_pose;
  std::string frame_pose_topic;
  std::string frame_odometry_topic;
  int keyframe;
  std::string keyframe_topic;
  std::string path_topic;
  int map;
  std::string map_topic;
  int reloc;
  std::string reloc_topic;
};

inline YAML::Node SelectPointMatcherSection(const YAML::Node& root,
                                            const YAML::Node& module_node){
  if(root["point_matcher"]) return root["point_matcher"];
  return module_node["point_matcher"];
}

inline void LoadPointMatcherConfig(PointMatcherConfig* config,
                                   const YAML::Node& root,
                                   const YAML::Node& module_node,
                                   const std::string& model_dir){
  config->Load(SelectPointMatcherSection(root, module_node));
  config->onnx_file = ConcatenateFolderAndFileName(model_dir, config->onnx_file);
  config->engine_file = ConcatenateFolderAndFileName(model_dir, config->engine_file);
}

inline void LoadRosPublisherConfig(RosPublisherConfig* config,
                                   const YAML::Node& root,
                                   const YAML::Node& module_node){
  if(root["ros_publisher"]){
    config->Load(root["ros_publisher"], module_node["ros_publisher"]);
  }else{
    config->Load(module_node["ros_publisher"]);
  }
}

// Voxel feature management (after Voxel-SVIO's voxel map): a keyframe creates no new map point in a 3D voxel
// (voxel_size m) that already holds max_per_voxel points of the keyframe (tracked or new, best SuperPoint score first).
// Stereo outliers are rejected only by the epipolar check of the rectified pair (camera max_y_diff,
// Frame::AddRightFeatures).
struct VoxelCapConfig {
  bool enabled = false;
  double voxel_size = 0.3;
  int max_per_voxel = 5;
  void Load(const YAML::Node& node){
    if(!node) return;
    enabled = ReadOr(node, "enabled", enabled);
    voxel_size = ReadOr(node, "voxel_size", voxel_size);
    max_per_voxel = ReadOr(node, "max_per_voxel", max_per_voxel);
  }
};

// Online dynamic-object masks (DynamicSegmenter): YOLO11n-seg instance segmentation (TensorRT FP16) of the rectified
// LEFT image; left features on a mask are dropped before stereo matching and tracking. Off by default.
struct DynamicMaskConfig {
  bool enabled = false;
  std::string onnx_file = "yolo11n_seg.onnx";   // dynamic input size; one engine per input size is cached next to it
  int input_size = 640;     // long side of the network input (the image is downscaled, multiples of 32)
  float confidence = 0.25f;
  float iou = 0.7f;         // NMS (per class)
  int max_detections = 100;
  int dilate_px = 7;        // mask dilation in image pixels
  std::vector<int> classes{0, 1, 2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19};   // COCO: person, vehicles, animals
  void Load(const YAML::Node& node){
    if(!node) return;
    enabled = ReadOr(node, "enabled", enabled);
    onnx_file = ReadOr(node, "onnx_file", onnx_file);
    input_size = ReadOr(node, "input_size", input_size);
    confidence = ReadOr(node, "confidence", confidence);
    iou = ReadOr(node, "iou", iou);
    max_detections = ReadOr(node, "max_detections", max_detections);
    dilate_px = ReadOr(node, "dilate_px", dilate_px);
    classes = ReadOr(node, "classes", classes);
    if(input_size < 32 || max_detections < 1 || dilate_px < 0 || classes.empty() || !(confidence > 0 && confidence < 1) ||
       !(iou > 0 && iou <= 1)) throw std::runtime_error("invalid dynamic_mask settings");
  }
};

// Leg gate (after the Mahalanobis gating of Voxel-SVIO / MSCKF with the leg odometry as the motion prior): every
// observation is tested against its projection with the leg-predicted pose (reference keyframe * leg relative motion),
// covariance = J (cov_scale * leg covariance, floored by min_sigma_r_deg / min_sigma_t) J^T + px_sigma^2;
// chi2 > obs_chi2: not used. The optimised pose is then tested against the leg prediction (6-dof Mahalanobis >
// frame_chi2): the frame is treated as lost (leg pose).
struct LegGateConfig {
  bool enabled = false;
  double obs_chi2 = 9.21, frame_chi2 = 22.46, cov_scale = 4.0, min_sigma_r_deg = 0.5, min_sigma_t = 0.02, px_sigma = 1.5;
  // slip: the stance-foot drift d (m) between the frames adds (slip_k * d)^2 to the translation variance and
  // (slip_k * d / 1 m)^2 rad^2 to the rotation variance (slip_k <= 0: off). strong visual: an optimised frame with
  // >= strong_inliers inliers and an inlier ratio >= strong_ratio is not rejected by the frame test (<= 0: off)
  double slip_k = 0.0, strong_ratio = 0.0;
  int strong_inliers = 0;
  // passive motion: when no foot moved by passive_foot_m in B over the last passive_window_s seconds (at most back to
  // the reference keyframe; contact flags are not used), the robot did not step but may still be carried (pushed, sliding
  // on ice, a moving platform). With passive_vis_t > 0 (m) such a frame enters the passive mode only when its PnP-RANSAC
  // pose is more than passive_vis_t from the leg-predicted pose (the camera visibly translated while the feet did not);
  // the mode then holds until the robot steps again, and a robot really standing keeps the leg prior and the gate.
  // A passive frame is tracked visually without the leg prior and the gate (weak visual: the usual lost handling);
  // pnp_init: it starts from the PnP-RANSAC pose instead of the leg prediction, which is far off for a carried robot;
  // passive_kf_s (<= 0: off): while passive, a frame more than passive_kf_s after the last keyframe becomes a keyframe
  // candidate (far matches with an old keyframe barely observe the forward translation)
  bool passive = false, pnp_init = false;
  double passive_foot_m = 0.05, passive_window_s = 0.5, passive_kf_s = 0.0, passive_vis_t = 0.0;
  void Load(const YAML::Node& node){
    if(!node) return;
    enabled = ReadOr(node, "enabled", enabled);
    obs_chi2 = ReadOr(node, "obs_chi2", obs_chi2);
    frame_chi2 = ReadOr(node, "frame_chi2", frame_chi2);
    cov_scale = ReadOr(node, "cov_scale", cov_scale);
    min_sigma_r_deg = ReadOr(node, "min_sigma_r_deg", min_sigma_r_deg);
    min_sigma_t = ReadOr(node, "min_sigma_t", min_sigma_t);
    px_sigma = ReadOr(node, "px_sigma", px_sigma);
    slip_k = ReadOr(node, "slip_k", slip_k);
    strong_ratio = ReadOr(node, "strong_ratio", strong_ratio);
    strong_inliers = ReadOr(node, "strong_inliers", strong_inliers);
    passive = ReadOr(node, "passive", passive);
    passive_foot_m = ReadOr(node, "passive_foot_m", passive_foot_m);
    passive_window_s = ReadOr(node, "passive_window_s", passive_window_s);
    pnp_init = ReadOr(node, "pnp_init", pnp_init);
    passive_kf_s = ReadOr(node, "passive_kf_s", passive_kf_s);
    passive_vis_t = ReadOr(node, "passive_vis_t", passive_vis_t);
  }
};

// Visual-degradation mode: after enter_lost consecutive lost frames the front-end is degraded: lost frames take the
// motion-model (leg) pose, regular keyframes are suppressed and only probe keyframes are inserted, when the pose moved
// more than probe_dist (m) or probe_deg from the last keyframe (the next frames match them, which detects the visual
// recovery). After recover_frames consecutive normally tracked frames a recovery keyframe is inserted and the probe
// keyframes of the degraded segment that were lost when inserted or share fewer than prune_covis map points with the
// recovery keyframe are deleted from the map (the leg factors bridge the segment).
struct DegradedConfig {
  bool enabled = false;
  int enter_lost = 2, recover_frames = 3, prune_covis = 15;
  double probe_dist = 0.5, probe_deg = 15.0;
  double probe_max_dt = 0.0;   // > 0: also a probe keyframe when the last keyframe is older (s), so the leg / IMU
                               // prediction never has to span more than the leg buffer (robot standing while lost)
  void Load(const YAML::Node& node){
    if(!node) return;
    enabled = ReadOr(node, "enabled", enabled);
    enter_lost = ReadOr(node, "enter_lost", enter_lost);
    recover_frames = ReadOr(node, "recover_frames", recover_frames);
    prune_covis = ReadOr(node, "prune_covis", prune_covis);
    probe_dist = ReadOr(node, "probe_dist", probe_dist);
    probe_deg = ReadOr(node, "probe_deg", probe_deg);
    probe_max_dt = ReadOr(node, "probe_max_dt", probe_max_dt);
  }
};

struct VisualOdometryConfigs{
  std::string dataroot;
  std::string camera_config_path;
  std::string model_dir;
  std::string saving_dir;

  SuperPointConfig superpoint_config;
  PointMatcherConfig point_matcher_config;
  KeyframeConfig keyframe_config;
  OptimizationConfig tracking_optimization_config;
  OptimizationConfig backend_optimization_config;
  RosPublisherConfig ros_publisher_config;
  OnlineFusionConfig online;
  DegradedConfig degraded;
  LegGateConfig leg_gate;
  VoxelCapConfig voxel_cap;
  DynamicMaskConfig dynamic_mask;
  // IMU in the VO front-end (camera yaml imu_mode: attitude filter, gravity-aligned start) and the S3 back-end
  // (online.backend.imu); false ignores the camera IMU settings. The leg odometry always uses its own IMU (lowstate).
  bool use_imu = false;

  VisualOdometryConfigs() {}

  VisualOdometryConfigs(const std::string& config_file_, const std::string& model_dir_){
    model_dir = model_dir_;

    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      throw std::runtime_error("config file doesn't exist: " + config_file_);
    }
    YAML::Node root = LoadYamlWithBase(config_file_);
    YAML::Node file_node = RequireConfigSection(root, "visual_odometry");

    superpoint_config.Load(file_node["superpoint"]);
    superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, superpoint_config.onnx_file);
    superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, superpoint_config.engine_file);

    LoadPointMatcherConfig(&point_matcher_config, root, file_node, model_dir);

    keyframe_config.Load(file_node["keyframe"]);
    degraded.Load(file_node["degraded"]);
    leg_gate.Load(file_node["leg_gate"]);
    voxel_cap.Load(file_node["voxel_cap"]);
    dynamic_mask.Load(file_node["dynamic_mask"]);
    dynamic_mask.onnx_file = ConcatenateFolderAndFileName(model_dir, dynamic_mask.onnx_file);
    tracking_optimization_config.Load(file_node["optimization"]["tracking"]);
    backend_optimization_config.Load(file_node["optimization"]["backend"]);
    LoadRosPublisherConfig(&ros_publisher_config, root, file_node);
    online.Load(root, file_node["online"],
                std::filesystem::weakly_canonical(std::filesystem::absolute(config_file_)).parent_path().string());
    if(file_node["use_imu"]) use_imu = file_node["use_imu"].as<bool>();
    backend_imu_config = online.backend.imu;
    SetUseImu(use_imu);
  }

  void SetUseImu(bool on){
    use_imu = on;
    online.backend.imu = on && backend_imu_config;
  }
  bool backend_imu_config = false;   // online.backend.imu as configured
};

struct MapRefinementConfigs{
  struct LoopVerificationConfig{
    int min_group_size = 3;
    int min_covisibility_weight = 1;
    double translation_ratio = 0.1;
    double min_translation_threshold = 0.5;
    double rotation_ratio = 0.1;
    double min_rotation_threshold_deg = 30.0;
    // Conflicting loop clusters inside one group (perceptual aliasing): keep the cluster whose implied correction of
    // the odometry poses (file in the map root, TUM, e.g. the S3 back-end trajectory) is smallest; reject the whole
    // group when the two best clusters differ by less than ambiguity_margin metres. Empty file: keep the old greedy
    // removal only.
    std::string odometry_poses = "";
    double ambiguity_margin = 1.0;

    void Load(const YAML::Node& node){
      if(!node) return;
      if(node["odometry_poses"]) odometry_poses = node["odometry_poses"].as<std::string>();
      if(node["ambiguity_margin"]) ambiguity_margin = node["ambiguity_margin"].as<double>();
      if(node["min_group_size"]) min_group_size = node["min_group_size"].as<int>();
      if(node["min_covisibility_weight"]){
        min_covisibility_weight = node["min_covisibility_weight"].as<int>();
      }
      if(node["translation_ratio"]){
        translation_ratio = node["translation_ratio"].as<double>();
      }
      if(node["min_translation_threshold"]){
        min_translation_threshold = node["min_translation_threshold"].as<double>();
      }
      if(node["rotation_ratio"]) rotation_ratio = node["rotation_ratio"].as<double>();
      if(node["min_rotation_threshold_deg"]){
        min_rotation_threshold_deg = node["min_rotation_threshold_deg"].as<double>();
      }
    }
  };

  PointMatcherConfig point_matcher_config;
  OptimizationConfig map_optimization_config;
  RosPublisherConfig ros_publisher_config;
  LoopVerificationConfig loop_verification_config;
  LegOdomConfig leg_odom_config;

  MapRefinementConfigs() {}

  MapRefinementConfigs(const std::string& config_file_, const std::string& model_dir_){
    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      throw std::runtime_error("config file doesn't exist: " + config_file_);
    }
    YAML::Node root = LoadYamlWithBase(config_file_);
    YAML::Node file_node = RequireConfigSection(root, "map_refinement");

    LoadPointMatcherConfig(&point_matcher_config, root, file_node, model_dir_);
    map_optimization_config.Load(file_node["optimization"]);
    loop_verification_config.Load(file_node["loop_verification"]);
    leg_odom_config.Load(file_node["leg_odom"]);
    LoadRosPublisherConfig(&ros_publisher_config, root, file_node);
  }
};

struct RelocalizationConfigs{
  std::string dataroot;
  std::string camera_config_path;
  std::string model_dir;
  std::string saving_dir;

  int min_inlier;
  int pose_refinement;
  SuperPointConfig superpoint_config;
  PointMatcherConfig point_matcher_config;
  OptimizationConfig pose_estimation_config;
  RosPublisherConfig ros_publisher_config;

  RelocalizationConfigs() {}

  RelocalizationConfigs(const std::string& config_file_, const std::string& model_dir_){
    model_dir = model_dir_;

    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      throw std::runtime_error("config file doesn't exist: " + config_file_);
    }

    YAML::Node root = LoadYamlWithBase(config_file_);
    YAML::Node file_node = RequireRelocalizationSection(root);

    min_inlier = file_node["min_inlier_num"].as<int>();
    pose_refinement = file_node["pose_refinement"].as<int>();

    superpoint_config.Load(file_node["superpoint"]);
    superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, superpoint_config.onnx_file);
    superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, superpoint_config.engine_file);

    LoadPointMatcherConfig(&point_matcher_config, root, file_node, model_dir);

    pose_estimation_config.Load(file_node["pose_estimation"]);
    LoadRosPublisherConfig(&ros_publisher_config, root, file_node);
  }
};

// backend.net_gyro_R_BI (row-major) as a rotation matrix
inline Eigen::Matrix3d NetGyroRBI(const BackendConfig& c){
  if(c.net_gyro_R_BI.size() != 9) throw std::invalid_argument("backend.net_gyro_R_BI needs 9 values");
  Eigen::Matrix3d R;
  for(int i = 0; i < 9; ++i) R(i / 3, i % 3) = c.net_gyro_R_BI[i];
  return R;
}


#endif  // READ_CONFIGS_H_
