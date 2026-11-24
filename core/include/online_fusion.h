#ifndef AIR_SLAM_ONLINE_FUSION_H_
#define AIR_SLAM_ONLINE_FUSION_H_

// S3 online fusion of the live VO with the online leg-odometry node:
//  * leg relative-pose factors for local BA and frame tracking,
//  * superframe dense depth (FFS) -> per-superframe terrain submaps,
//  * foot-contact-to-surface factors against fixed, recent anchor superframes,
//  * a low-rate background window VCE that rescales the leg factor group.
// Conventions: Frame poses are T_W_C; T_C_B maps the robot base frame B (URDF base, not the IMU) into the
// rectified left camera C; world z is up (VO init pose / IMU gravity alignment).

#include <array>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <map>
#include <set>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include "utils.h"
#include "camera.h"
#include "frame.h"
#include "leg_odom.h"
#include "stage_times.h"
#include "depth_kernels.h"
#include "read_configs.h"
#include "g2o_optimization/types.h"
#include "fusion_backend.h"

#ifdef AIRSLAM_WITH_FFS
namespace ffs_infer { class FFSInfer; }
#endif

// Anisotropic voxel key (pattern of navigation chery_mapping frame_evidence.hpp).
struct VoxelKey {
  int x, y, z;
  bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelKeyHash {
  size_t operator()(const VoxelKey& k) const {
    return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^
           (static_cast<size_t>(k.z) * 83492791u);
  }
};

// Dense submap of one superframe, stored in its gravity-aligned frame G (origin at C,
// p_C = R_CG p_G, R_CG frozen at selection). Rigid with C, so pose updates never re-index.
struct SuperframeRecord {
  int frame_id = -1;
  FramePtr frame;
  Eigen::Matrix3d R_CG = Eigen::Matrix3d::Identity();
  std::vector<Eigen::Vector3f> points;  // in G, one per anisotropic (voxel_xy, voxel_z) voxel
  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> columns;  // (ix, iy, 0) -> point ids
};
typedef std::shared_ptr<const SuperframeRecord> SuperframeRecordPtr;

enum class SurfaceReject {
  NO_CONTACT, CONTACT_TRANSIENT, NO_LEG_DATA, NO_ANCHOR, NO_COVERAGE, STEP_REGION, EDGE_OR_HOLE,
  NON_PLANAR, ASSOCIATION_TOO_FAR, SEGMENT_INCONSISTENT, BUDGET, COUNT
};

class SuperframeMap {
 public:
  SuperframeMap(const SurfaceConfig& cfg, CameraPtr camera);
  ~SuperframeMap();
  bool DepthAvailable() const { return depth_available_; }

  // Tracking thread, before Map::InsertKeyframe. Decides and queues superframe depth.
  void OnKeyframe(const FramePtr& frame, const cv::Mat& left, const cv::Mat& right,
                  const LegOdomStream& legs, const Eigen::Isometry3d& T_C_B, double time_delay);
  // Builds a submap from a metric depth image (left rectified camera). Used by the depth
  // thread and by tests.
  void InsertDepth(const FramePtr& frame, const Eigen::Matrix3d& R_CG, const cv::Mat& depth);
  // CPU reference of the FFS disparity -> depth step (DisparityToDepth with this map's settings).
  cv::Mat DepthFromDisparity(const std::vector<float>& disparity, int width, int height) const;
  // CPU reference of the superframe points: strided back-projection into G, first point per voxel in scan order;
  // xyz / keys flattened (x y z / ix iy iz per point).
  void DepthPoints(const Eigen::Matrix3d& R_CG, const cv::Mat& depth, std::vector<float>* xyz, std::vector<int>* keys) const;
  // Same from the disparity on the GPU (depth_kernels.cu, bit-identical); false = use the CPU path.
  bool DisparityPointsGpu(const std::vector<float>& disparity, int width, int height, const Eigen::Matrix3d& R_CG,
                          std::vector<float>* xyz, std::vector<int>* keys);
  void InsertPoints(const FramePtr& frame, const Eigen::Matrix3d& R_CG, const std::vector<float>& xyz,
                    const std::vector<int>& keys);

  // Latest `num_anchor_superframes` submaps whose frames are alive and fixed in this BA.
  std::vector<SuperframeRecordPtr> Anchors(const std::function<bool(int)>& usable) const;
  // Foot-surface constraints for the free window frames against the given anchors.
  void Associate(const std::vector<FramePtr>& free_frames, const std::vector<SuperframeRecordPtr>& anchors,
                 const LegOdomStream& legs, const Eigen::Isometry3d& T_C_B, double time_delay,
                 VectorOfFootConstraints* out);
  std::string RejectSummary();
  // Whether the keyframe was selected as a superframe (queued for dense depth).
  bool IsSuperframe(int frame_id) const;
  // foot in B after the configured foot-position correction (foot_corr_*)
  Eigen::Vector3d CorrectedFoot(const Eigen::Vector3d& foot_b) const;

  // Plane of the foot neighbourhood in G; exposed for tests.
  struct Plane { Eigen::Vector3d q, n; double rmse = 0; SurfaceReject reject = SurfaceReject::COUNT; };
  Plane FitPlane(const SuperframeRecord& record, const Eigen::Vector3d& foot_g) const;

 private:
  struct Job { FramePtr frame; Eigen::Matrix3d R_CG; cv::Mat left, right; };
  struct SegmentPlane { int anchor_id; Eigen::Vector3d q_c, n_c; double sigma, rmse; bool bad = false; };
  void DepthLoop();
  bool Covered(const Eigen::Vector3d& p_w) const;
  // Point ids within horizontal radius r and |dz| <= band of p_g.
  std::vector<int> Near(const SuperframeRecord& record, const Eigen::Vector3d& p_g, double r, double band) const;
  bool Verify(const SegmentPlane& plane, const SuperframeRecord& anchor, const Eigen::Vector3d& p_w) const;
  void Reject(SurfaceReject r) { ++rejects_[static_cast<size_t>(r)]; }

  SurfaceConfig cfg_;
  CameraPtr camera_;
  std::atomic<bool> depth_available_{false};
  // submaps
  mutable std::shared_mutex records_mutex_;
  std::deque<SuperframeRecordPtr> records_;
  // selection state (tracking thread)
  FramePtr last_superframe_;
  mutable std::mutex selected_mutex_;
  std::set<int> selected_;  // recent superframe ids
  // depth worker
  std::mutex job_mutex_;
  std::condition_variable job_cv_;
  std::deque<Job> jobs_;
  bool stop_ = false;
  std::thread depth_thread_;
#ifdef AIRSLAM_WITH_FFS
  std::unique_ptr<ffs_infer::FFSInfer> ffs_;   // loaded (or built) in the constructor, used by the depth thread
#endif
  std::unique_ptr<depth_cuda::Voxelizer> voxelizer_;  // GPU post-processing (depth thread)
  bool depth_on_cpu_ = std::getenv("AIRSLAM_DEPTH_CPU") != nullptr;     // CPU reference path
  bool depth_verify_ = std::getenv("AIRSLAM_DEPTH_VERIFY") != nullptr;  // run both and count mismatches
  size_t depth_checked_ = 0, depth_mismatch_ = 0;
  // association state (tracking thread)
  std::map<std::pair<uint32_t, int>, SegmentPlane> segments_;
  std::array<size_t, static_cast<size_t>(SurfaceReject::COUNT)> rejects_{};
  // elevation-map anchors (surface.old_plane): good planes of every superframe
  struct OldPlane {
    int frame_id = -1; FramePtr frame; double time = 0.0;
    Eigen::Vector3d q_c, n_c, c_c; double radius = 0.0, rmse = 0.0;   // anchor camera frame; n towards the camera
    std::vector<Eigen::Vector3f> pts_c;                                 // inliers (capped)
  };
  void DetectPlanes(const SuperframeRecord& record);
  // Edge of (frame, leg) against the oldest good plane under the foot, searched once per (frame, leg) and cached.
  bool OldPlaneConstraint(const FramePtr& frame, int leg, const Eigen::Vector3d& foot_b, const Eigen::Vector3d& p_w,
                          FootSurfaceConstraint* c);
  std::mutex old_mutex_;
  std::vector<OldPlane> old_planes_;
  std::map<std::pair<int, int>, std::pair<bool, FootSurfaceConstraint>> old_cache_;
  size_t old_edges_ = 0, old_planes_kept_ = 0, old_superframes_ = 0, old_superframes_empty_ = 0;

 public:
  StageTimes times;  // depth thread: ffs_infer / depth_post / voxelize
};

class OnlineFusion {
 public:
  OnlineFusion(const OnlineFusionConfig& cfg, CameraPtr camera);
  ~OnlineFusion();
  LegOdomStream& Stream() { return stream_; }
  SuperframeMap* Superframes() { return superframes_.get(); }

  // Leg relative pose between camera frames f1 (older) and f2; false if not covered. wait_s: how long to
  // wait for late leg data (tracking queries). Successful results are cached per frame pair.
  bool LegConstraint(const FramePtr& f1, const FramePtr& f2, LegRelativeConstraint* c, double wait_s = 0.0) const;
  // Counters of the leg queries (ok / ok after waiting / late / not covered / cached), for the logs.
  std::string LegStats() const;
  // Mean drift (m) of the stance feet that keep their contact segment between f1 and f2 under the leg relative motion
  // c (as the back-end slip_mean); false without such a foot (leg_gate slip inflation)
  bool LegSlip(const FramePtr& f1, const FramePtr& f2, const LegRelativeConstraint& c, double* slip) const;
  // No step between camera times t1 < t2: at samples every <= 0.1 s over [t1, t2] (>= 5), every foot stays within
  // max_foot_m of its position in B at t1. Contact flags are not used (a lightly loaded foot of a standing robot
  // flickers around the contact threshold). False without foot data.
  bool Standing(double t1, double t2, double max_foot_m) const;
  // Fills the leg inputs of the window (newest first; free frames are not fixed in `poses`) and associates
  // the free frames' stance feet with superframe surfaces for the back-end (foot edges are back-end only).
  void PrepareLocalBA(const std::vector<FramePtr>& window, MapOfPoses& poses,
                      const std::function<FramePtr(int)>& keyframe, LocalBAExtras* extras);
  // Diagnostics after a local BA.
  void FinishLocalBA(const std::vector<FramePtr>& window, const LocalBAExtras& extras);
  void OnKeyframe(const FramePtr& frame, const cv::Mat& left, const cv::Mat& right);
  // After the keyframe poses of a local BA are written (window newest first): pushes the keyframes optimized for
  // the last time (`finished`, oldest first, decided by the local BA window) to the back-end, records the
  // real-time output.
  void Finalize(const std::vector<FramePtr>& window, const std::vector<FramePtr>& finished);
  FusionBackend* Backend() { return backend_.get(); }
  // Robot accelerometer (sensor clock, specific force), for the back-end leg reliability features.
  void PushAccel(double t, const Eigen::Vector3d& f_b);
  // Raw IMU samples (camera yaml imu_mode > 0), for the back-end extras (backend.export_extras).
  void PushImu(const ImuDataList& samples);
  // backend.net_gyro: adds to each raw gyro sample the network correction of the nearest network sample (network rate -
  // R_BI x the raw gyro at that sample, IMU frame); returns the number of corrected samples
  int NetGyroCorrect(ImuDataList* samples);

 private:
  Eigen::Isometry3d Extrinsic() const;
  // Leg data are used only once T_C_B is known (latched topic of the leg node, or given in the config);
  // before that the extrinsic would be identity and every leg prior / foot position wrong.
  bool LegReady() const;
  // Leg reliability features of the leg interval [t0, t1] (sensor clock), divided by backend.leg_rel_feature_scale:
  // log slip, mean |v|, mean |omega|, log mean network linear variance, std of |acc|. False if not covered.
  bool ReliabilityFeatures(double t0, double t1, double slip, std::vector<double>* f) const;

  OnlineFusionConfig cfg_;
  CameraPtr camera_;
  LegOdomStream stream_;
  std::unique_ptr<SuperframeMap> superframes_;
  std::unique_ptr<FusionBackend> backend_;
  std::map<int, VectorOfFootConstraints> feet_;  // latest accepted associations per keyframe, for the back-end
  FramePtr last_final_;
  mutable std::atomic<bool> warned_extrinsic_{false};
  // leg constraint cache: (id1, id2) -> motion and the extrinsic it was computed with
  struct CachedLeg {
    Eigen::Isometry3d T_C_B, body;
    Eigen::Matrix<double, 6, 6> covariance;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  mutable std::mutex leg_cache_mutex_;
  mutable std::map<std::pair<int, int>, CachedLeg, std::less<std::pair<int, int>>,
                   Eigen::aligned_allocator<std::pair<const std::pair<int, int>, CachedLeg>>> leg_cache_;
  mutable std::atomic<long> leg_ok_{0}, leg_waited_{0}, leg_late_{0}, leg_uncovered_{0}, leg_cached_{0};
  mutable std::mutex accel_mutex_;
  std::deque<std::pair<double, double>> accel_norm_;  // (time, |acc|)
  // back-end extras (backend.export_extras): sensor meta, IMU of the node interval, map point observations
  void ExportExtras(const FramePtr& frame, BackendNode* node);
  int NetGyroCorrectLocked(ImuDataList* samples);   // imu_mutex_ held
  std::mutex imu_mutex_;
  std::deque<ImuData> imu_buf_;
};
typedef std::shared_ptr<OnlineFusion> OnlineFusionPtr;

#endif  // AIR_SLAM_ONLINE_FUSION_H_
