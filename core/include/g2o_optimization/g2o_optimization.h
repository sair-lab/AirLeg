#ifndef G2O_OPTIMIZATION_H_
#define G2O_OPTIMIZATION_H_

#include <string>
#include <vector>

#include "read_configs.h"
#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "map.h"
#include "g2o_optimization/types.h"

namespace g2o { class SparseOptimizer; }

// Observation of point id_point in frame id_pose (camera 0, pixel sigma 0.8): stereo when keypoint(2) > 0 (right u),
// else mono. Returns true for a stereo constraint.
bool AddPointConstraint(int id_pose, int id_point, const Eigen::Vector3d& keypoint,
                        VectorOfMonoPointConstraints& mono, VectorOfStereoPointConstraints& stereo);
void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, bool fix_this_frame);
void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, MapOfVelocity& velocities, MapOfBias& biases, 
    VectorOfIMUConstraints& imu_constraints, bool fix_this_frame, bool add_imu_constraint, bool use_updated_bias=false);

// pose, velocaity and bias of same frame have same key, i.e. id.
void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg,
    LocalBAExtras* extras = nullptr);

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg,
    const VectorOfLegConstraints* leg_constraints = nullptr, double leg_huber = 2.5);

// S3 helpers shared by local BA and frame tracking.
void AddLegEdges(g2o::SparseOptimizer& optimizer, const VectorOfLegConstraints& legs, double huber);

bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg);

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, Eigen::Matrix4d& pose,
    std::vector<int>& inliers, double reprojection_error = 20.0);

// frames: [n, n-1, n-2....]
bool ComputeGyrBias(std::vector<FramePtr>& frames, Eigen::Vector3d& dbg);
bool ComputeVelocity(std::vector<FramePtr>& frames, Eigen::Vector3d& gw);

// frames: [n, n-1, n-2....], for debugging


// for offline map refinement
void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list, VectorOfRelativePoseConstraints& relative_pose_constraints);
bool GlobalBA(MapPtr map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    int first_iterations, int second_iterations, LegOdomConfig* leg_cfg = nullptr,
    const std::string& leg_velocity_path = "", bool structure_only = false);

#endif  // G2O_OPTIMIZATION_H_
