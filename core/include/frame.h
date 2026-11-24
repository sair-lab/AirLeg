#ifndef FRAME_H_
#define FRAME_H_

#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "utils.h"
#include "mappoint.h"
#include "camera.h"
#include "imu.h"
#include "3rdparty/DBoW2/include/DBoW2/TemplatedVocabulary.h"

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

// Tracking outcome of a frame: Normal (> min_num_match inliers), Weak, Lost (<= lost_num_match).
enum class TrackState { Normal = 0, Weak = 1, Lost = 2 };
// Origin of the frame pose: visual tracking, or a motion model when tracking is lost.
enum class PoseSource { Visual = 0, Leg = 1, Imu = 2, ConstantVelocity = 3, Stationary = 4 };
const char* PoseSourceName(PoseSource source);

class Frame{
public:
  Frame();
  Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp);
  // Frame& operator=(const Frame& other);

  int GetFrameId();
  double GetTimestamp();
  void SetPoseFixed(bool pose_fixed);
  void SetPose(const Eigen::Matrix4d& pose);
  Eigen::Matrix4d& GetPose();

  // point features
  bool FindGrid(float& x, float& y, int& grid_x, int& grid_y);
  void AddLeftFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left);
  int AddRightFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right,
      std::vector<cv::DMatch>& stereo_matches);

  Eigen::Matrix<float, 259, Eigen::Dynamic>& GetAllFeatures();

  size_t FeatureNum();

  bool GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos);
  std::vector<cv::KeyPoint>& GetAllKeypoints();
  cv::KeyPoint& GetKeypoint(size_t idx);
  int GetInlierFlag(std::vector<bool>& inliers_feature_message);

  double GetRightPosition(size_t idx);

  bool GetDescriptor(size_t idx, Eigen::Matrix<float, 256, 1>& descriptor) const;

  std::vector<double>& GetAllDepth();

  std::vector<int>& GetAllTrackIds();
  void SetTrackId(size_t idx, int track_id);
  int GetTrackId(size_t idx);

  MappointPtr GetMappoint(size_t idx);
  std::vector<MappointPtr>& GetAllMappoints();
  void InsertMappoint(size_t idx, MappointPtr mappoint);

  bool BackProjectPoint(size_t idx, Eigen::Vector3d& p3D);
  CameraPtr GetCamera();

  
  void RemoveMappoint(MappointPtr mappoint);
  void RemoveMappoint(int idx);

  // for IMU
  Eigen::Matrix4d IMUPose();
  void SetIMUPose(const Eigen::Matrix4d& pose);
  void SetIMUPreinteration(const Preinteration& preinteration);
  PreinterationPtr GetIMUPreinteration();
  bool VelocityIsInitialized();
  void SetVelocaity(const Eigen::Vector3d& velocity);
  Eigen::Vector3d GetVelocity();
  void SetPreviousFrame(const std::shared_ptr<Frame> previous_frame);
  std::shared_ptr<Frame> PreviousFrame();
  void SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate = true);
  void UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias);
  void GetBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias);
  void Repropagate();


  
public:
  int tracking_frame_id;
  int local_map_optimization_frame_id;
  int local_map_optimization_fix_frame_id;
  // Front-end tracking state (not serialized; loaded maps read as Normal / Visual).
  TrackState track_state = TrackState::Normal;
  PoseSource pose_source = PoseSource::Visual;
  // Weak motion prior of a lost keyframe whose pose has no leg / IMU edge to its previous keyframe:
  // measured T_prev^-1 T_this at insertion, information growing with the time gap (local BA keeps it connected).
  bool has_motion_prior = false;
  // Gravity direction (unit, down) in this frame's camera frame from the IMU attitude filter (imu_mode > 0).
  bool has_gravity = false;
  Eigen::Vector3d gravity_c = Eigen::Vector3d::Zero();
  Eigen::Matrix4d motion_prior = Eigen::Matrix4d::Identity();
  double motion_prior_dt = 0.0;


private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _frame_id;
    ar & _timestamp;
    ar & _pose_fixed;
    ar & boost::serialization::make_array(_pose.data(), _pose.size());

    SerializeFeatures(ar, _features, version);
    SerializeKeypoints(ar, _keypoints, version);
    ar & _feature_grid;
    ar & _grid_width_inv;
    ar & _grid_height_inv;
    ar & _u_right;
    ar & _depth;
    ar & _track_ids;
    ar & _mappoints;

    ar & _camera;

    ar & boost::serialization::make_array(_imu_pose.data(), _imu_pose.size());
    ar & _init_v;
    ar & boost::serialization::make_array(_velocity.data(), _velocity.size());
    ar & _preinteration;

  }

private:
  int _frame_id;
  double _timestamp;
  bool _pose_fixed;
  Eigen::Matrix4d _pose;

  // point features
  Eigen::Matrix<float, 259, Eigen::Dynamic> _features;
  std::vector<cv::KeyPoint> _keypoints;
  std::vector<int> _feature_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
  double _grid_width_inv;
  double _grid_height_inv;
  std::vector<double> _u_right;
  std::vector<double> _depth;
  std::vector<int> _track_ids;
  std::vector<MappointPtr> _mappoints;

  // camera
  CameraPtr _camera;

  // for imu
  Eigen::Matrix4d _imu_pose;
  bool _init_v;   
  Eigen::Vector3d _velocity;
  PreinterationPtr _preinteration;
  std::shared_ptr<Frame> _previous_frame;

};

typedef std::shared_ptr<Frame> FramePtr;

#endif  // FRAME_H_
