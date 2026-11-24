#ifndef AIR_SLAM_FUSION_BACKEND_H_
#define AIR_SLAM_FUSION_BACKEND_H_

// S3 light-weight fusion back-end. A pose-only sliding-window graph over finalized keyframes
// (keyframes that left the front-end local BA window), solved in its own thread:
//  * visual edge: the front-end relative pose of consecutive keyframes, compressed into one edge with
//    per-axis sigmas, down-weighted per edge by its covisible support (dropped when tracking was lost),
//  * leg edge: leg-odometry relative pose of the same pair,
//  * foot edge: foot contact against the terrain plane of an earlier anchor keyframe, with a contact
//    offset shared by blocks of nodes and a random walk between blocks (slope / plane-RMSE gated),
//  * online VCE of the leg (rotation / horizontal / vertical) and foot groups. The visual groups stay
//    fixed: on a chain, a visual and a leg edge of the same pair only identify the SUM of their variances.
// Poses are T_W_C; the camera y axis is treated as vertical for the per-axis sigmas.

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "imu.h"
#include "read_configs.h"
#include "g2o_optimization/types.h"

// Map point observation of a node (window marginalisation, visual_marg win_pair).
struct LandmarkObs {
  long point = -1;                                          // map point id
  Eigen::Vector3d kp = Eigen::Vector3d::Zero();             // u, v, u_right (-1: mono) in this node's image
  Eigen::Vector3d pw = Eigen::Vector3d::Zero();             // front-end world position when finalized
  int n_obs = 0, ba_obs = 0, ba_outliers = 0;               // observers, local-BA constraints / rejections
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::vector<LandmarkObs, Eigen::aligned_allocator<LandmarkObs>> LandmarkObsList;

// Camera / IMU parameters of the run (window marginalisation, IMU factors).
struct SensorMeta {
  bool valid = false;
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0, bf = 0.0;  // rectified left camera
  bool has_imu = false;
  Eigen::Isometry3d T_i_c = Eigen::Isometry3d::Identity();  // left camera pose in the IMU frame (camera T_imu_cam0)
  double gyr_noise = 0.0, acc_noise = 0.0, gyr_walk = 0.0, acc_walk = 0.0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BackendNode {
  int id = -1;
  double time = 0.0;
  Eigen::Isometry3d T_wc = Eigen::Isometry3d::Identity();   // front-end pose when finalized
  Eigen::Isometry3d vo = Eigen::Isometry3d::Identity();     // front-end T_Cprev_C (first node: unused)
  Eigen::Isometry3d T_c_b = Eigen::Isometry3d::Identity();
  int covisible = 0;                                        // map points shared with the previous node
  double spread = 0.0;                                      // their image spread, uniform = 1
  bool has_leg = false;
  Eigen::Isometry3d leg = Eigen::Isometry3d::Identity();    // leg T_Cprev_C
  VectorOfFootConstraints feet;                             // id_pose == id, anchors are earlier nodes
  bool superframe = false;                                  // selected for dense depth (solve trigger)
  double slip_mean = -1.0;                                  // stance-foot drift under the leg motion (m), -1 = none
  double slip_factor = 1.0;                                 // derived in the back-end (running median)
  std::vector<double> rel_feat;                             // reliability features (leg_reliability), empty = none
  double rel_factor = 1.0;                                  // learned leg translation variance scale
  bool lost = false;                                        // front-end tracking lost: pose from a motion model
  bool standing = false;                                    // no step since the previous node (backend.passive_vo_t)
  bool has_init = false;                                    // initial pose instead of the VO chain (global back-end)
  Eigen::Isometry3d T_init = Eigen::Isometry3d::Identity();
  // export_extras: raw IMU samples bracketing (previous node, node] (IMU frame), map point observations, sensor meta
  ImuDataList imu;
  LandmarkObsList obs;
  SensorMeta meta;
  // worker gyro bias at the node (B; backend.gyro_bias_total); IMU samples with / without the network correction
  // (backend.net_gyro)
  Eigen::Vector3d gyro_bias_applied = Eigen::Vector3d::Zero();
  int imu_net = 0, imu_raw = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Reads backend_nodes.csv / backend_feet.csv written by FusionBackend::Save (offline replay, global back-end).
bool LoadBackendDump(const std::string& dir, std::vector<BackendNode>* nodes, std::string* error = nullptr);
// backend_meta.csv / backend_imu.csv / backend_points.csv (export_extras; written by Save, read by LoadBackendDump)
void SaveBackendExtras(const std::string& dir, const std::vector<BackendNode>& nodes);
void LoadBackendExtras(const std::string& dir, std::vector<BackendNode>* nodes);

class FusionBackend {
 public:
  explicit FusionBackend(const BackendConfig& cfg);
  ~FusionBackend();  // solves the remaining nodes, then joins
  const BackendConfig& Config() const { return cfg_; }

  void Push(const BackendNode& node);
  // Real-time output T = C * T_front, C = T_back * T_front^-1 of the newest solved node.
  Eigen::Isometry3d Correction() const;
  void RecordRealtime(double time, const Eigen::Isometry3d& T_front);
  // write_back: back-end poses of the nodes solved since the last call and the newest solved id.
  bool TakeWriteBack(std::map<int, Eigen::Isometry3d>* poses, int* newest);
  void RecordWriteBack(double ms);
  // trajectory_backend.txt (final window poses), trajectory_backend_rt.txt, backend_stats.txt,
  // backend_nodes.csv / backend_feet.csv (inputs, for offline replay).
  void Save(const std::string& dir);
  // Solves the queued nodes and stops the thread; the accessors below are valid afterwards.
  void Stop();
  std::vector<std::pair<double, Eigen::Isometry3d>> Trajectory() const;
  Eigen::Vector4d Alpha() const { return alpha_; }
  size_t Solves() const;
  Eigen::Vector3d LatestGyroBias() const { return gyro_bias_.empty() ? Eigen::Vector3d::Zero() : gyro_bias_.back(); }
  // Called from the back-end thread after every solve with the current back-end keyframe trajectory
  // (e.g. published as the supervision path of the online LoRA).
  typedef std::function<void(const std::vector<std::pair<double, Eigen::Isometry3d>>&)> PathCallback;
  void SetPathCallback(PathCallback cb);
  // Loop-closure relative pose T_i^-1 T_j between node ids; add before pushing the nodes.
  void AddLoop(int id_i, int id_j, const Eigen::Isometry3d& T_ij);

 private:
  void Loop();
  void Prepare(BackendNode* node);
  Eigen::Isometry3d InitialEstimate(size_t n) const;
  void Solve(size_t end);
  void LearnReliability(size_t end);

  BackendConfig cfg_;
  // input queue (any thread)
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<BackendNode> queue_;
  bool stop_ = false;
  // graph state (back-end thread)
  std::vector<BackendNode> nodes_;
  std::vector<Eigen::Isometry3d> est_;
  // dropped non-finite input nodes / rejected non-finite solves
  std::atomic<long> rejected_inputs_{0}, rejected_solves_{0};
  std::vector<double> sink_;                 // per block of cfg_.sink_block nodes
  std::map<int, size_t> index_;
  std::deque<double> slip_hist_;
  // learned leg reliability: history of (features, log residual) and the current fit
  std::deque<std::vector<double>> rel_X_;
  std::deque<double> rel_y_;
  Eigen::VectorXd rel_theta_, rel_mean_;
  size_t rel_done_ = 1;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> gyro_bias_;  // per block
  std::vector<double> vo_scale_;   // visual translation scale offset v per block (vo_scale)
  std::vector<long> landmark_sel_;  // landmark selection of the last solve (visual_mode landmarks)
  // IMU states (imu): velocity per node index, bias blocks, camera-IMU rotation correction, world gravity direction
  std::map<size_t, Eigen::Vector3d, std::less<size_t>, Eigen::aligned_allocator<std::pair<const size_t, Eigen::Vector3d>>> imu_vel_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> imu_bg_, imu_ba_;
  Eigen::Vector3d imu_ext_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d imu_Rwg_ = Eigen::Matrix3d::Identity();
  std::map<std::string, long> counters_;   // event counts (backend_stats.txt)
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> leg_vbias_;  // per leg velocity-bias block (leg_vbias)
  size_t solved_ = 0;
  Eigen::Vector4d alpha_ = Eigen::Vector4d::Ones();  // leg rot / horizontal / vertical, foot
  // outputs
  mutable std::mutex out_mutex_;
  Eigen::Isometry3d correction_ = Eigen::Isometry3d::Identity();
  std::map<int, Eigen::Isometry3d> write_back_;
  int newest_ = -1;
  std::vector<std::pair<double, Eigen::Isometry3d>> realtime_;
  std::vector<double> solve_ms_, vce_ms_, write_back_ms_;
  PathCallback path_cb_;
  struct LoopEdge { int i, j; Eigen::Isometry3d T; EIGEN_MAKE_ALIGNED_OPERATOR_NEW };
  std::vector<LoopEdge, Eigen::aligned_allocator<LoopEdge>> loops_;
  std::vector<std::string> log_;
  std::thread thread_;
};

#endif  // AIR_SLAM_FUSION_BACKEND_H_
