#include "frame.h"
#include <atomic>

const char* PoseSourceName(PoseSource source) {
  switch (source) {
    case PoseSource::Visual: return "visual";
    case PoseSource::Leg: return "leg";
    case PoseSource::Imu: return "imu";
    case PoseSource::ConstantVelocity: return "constant_velocity";
    case PoseSource::Stationary: return "stationary";
  }
  return "unknown";
}
#include <assert.h>


Frame::Frame(){
}

Frame::Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp):
    tracking_frame_id(-1), local_map_optimization_frame_id(-1), local_map_optimization_fix_frame_id(-1),
    _frame_id(frame_id), _pose_fixed(pose_fixed), _camera(camera), _timestamp(timestamp), _init_v(false){
  _grid_width_inv = static_cast<double>(FRAME_GRID_COLS)/static_cast<double>(_camera->ImageWidth());
  _grid_height_inv = static_cast<double>(FRAME_GRID_ROWS)/static_cast<double>(_camera->ImageHeight());
}

int Frame::GetFrameId(){
  return _frame_id;
}

double Frame::GetTimestamp(){
  return _timestamp;
}

void Frame::SetPoseFixed(bool pose_fixed){
  _pose_fixed = pose_fixed;
}

std::atomic<long> g_pose_reorthonormalized{0};

void Frame::SetPose(const Eigen::Matrix4d& pose){
  _pose = pose;
  // chains of motion-model poses (prev * leg relative motion) slowly lose the rotation's orthonormality; re-project
  // only when it is off by > 1e-6 (exact no-op for normal poses)
  const Eigen::Matrix3d R = _pose.block<3, 3>(0, 0);
  if((R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-6 && R.allFinite()){
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d Rn = svd.matrixU() * svd.matrixV().transpose();
    if(Rn.determinant() < 0){ Eigen::Matrix3d U = svd.matrixU(); U.col(2) *= -1; Rn = U * svd.matrixV().transpose(); }
    _pose.block<3, 3>(0, 0) = Rn;
    ++g_pose_reorthonormalized;
  }
  Eigen::Matrix4d Tcb = _camera->BodyToCamera();
  _imu_pose = _pose * Tcb;
}

Eigen::Matrix4d& Frame::GetPose(){
  return _pose;
}

bool Frame::FindGrid(float& x, float& y, int& grid_x, int& grid_y){
  grid_x = std::round(x * _grid_width_inv);
  grid_y = std::round(y * _grid_height_inv);

  grid_x = std::min(std::max(0, grid_x), (FRAME_GRID_COLS-1));
  grid_y = std::min(std::max(0, grid_y), (FRAME_GRID_ROWS-1));

  return !(grid_x < 0 || grid_x >= FRAME_GRID_COLS || grid_y < 0 || grid_y >= FRAME_GRID_ROWS);
}

void Frame::AddLeftFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left){
  _features = features_left;
  const size_t feature_count = _features.cols();
  for(size_t i = 0; i < feature_count; ++i){
    const float score = _features(0, i);
    float x = _features(1, i);
    float y = _features(2, i);
    _keypoints.emplace_back(x, y, 8, -1, score);
    int grid_x, grid_y;
    const bool found = FindGrid(x, y, grid_x, grid_y);
    assert(found);
    _feature_grid[grid_x][grid_y].push_back(i);
  }
  _u_right.assign(feature_count, -1);
  _depth.assign(feature_count, -1);
  _track_ids.assign(feature_count, -1);
  _mappoints.assign(feature_count, nullptr);
}

int Frame::AddRightFeatures(
    Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right,
    std::vector<cv::DMatch>& stereo_matches){
  int good_stereo_points = 0;
  for(const cv::DMatch& match : stereo_matches){
    const int left = match.queryIdx;
    const int right = match.trainIdx;
    const double dx = std::abs(_features(1, left) - features_right(1, right));
    const double dy = std::abs(_features(2, left) - features_right(2, right));
    if(dx <= _camera->MinXDiff() || dx >= _camera->MaxXDiff() ||
       dy > _camera->MaxYDiff()) continue;
    const double parallax = _features(1, left) - features_right(1, right);
    if(parallax > _camera->MinXDiff() && parallax < _camera->MaxXDiff()){
      _u_right[left] = features_right(1, right);
      _depth[left] = _camera->BF() / parallax;
      ++good_stereo_points;
    }
  }
  return good_stereo_points;
}

Eigen::Matrix<float, 259, Eigen::Dynamic>& Frame::GetAllFeatures(){
  return _features;
}

size_t Frame::FeatureNum(){
  return _features.cols();
}

bool Frame::GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos){
  if(idx >= static_cast<size_t>(_features.cols())) return false;
  keypoint_pos.head(2) = _features.block<2, 1>(1, idx).cast<double>();
  keypoint_pos(2) = _u_right[idx];
  return true;
}

std::vector<cv::KeyPoint>& Frame::GetAllKeypoints(){
  return _keypoints;
}

cv::KeyPoint& Frame::GetKeypoint(size_t idx){
  assert(idx < _keypoints.size());
  return _keypoints[idx];
}

int Frame::GetInlierFlag(std::vector<bool>& inliers_feature_message){
  int num_inliers = 0;
  inliers_feature_message.resize(_mappoints.size());
  for(size_t i = 0; i < _mappoints.size(); i++){
    if(_mappoints[i] && !_mappoints[i]->IsBad()){
      inliers_feature_message[i] = true;
      num_inliers++;
    }else{
      inliers_feature_message[i] = false;
    }
  }
  return num_inliers;
}

double Frame::GetRightPosition(size_t idx){
  assert(idx < _u_right.size());
  return _u_right[idx];
}

bool Frame::GetDescriptor(size_t idx, Eigen::Matrix<float, 256, 1>& descriptor) const{
  if(idx >= static_cast<size_t>(_features.cols())) return false;
  descriptor = _features.block<256, 1>(3, idx);
  return true;
}

std::vector<double>& Frame::GetAllDepth(){
  return _depth;
}

std::vector<int>& Frame::GetAllTrackIds(){
  return _track_ids;
}

void Frame::SetTrackId(size_t idx, int track_id){
  _track_ids[idx] = track_id;
}

int Frame::GetTrackId(size_t idx){
  assert(idx < _track_ids.size());
  return _track_ids[idx];
}

MappointPtr Frame::GetMappoint(size_t idx){
  assert(idx < _mappoints.size());
  return _mappoints[idx];
}

std::vector<MappointPtr>& Frame::GetAllMappoints(){
  return _mappoints;
}

void Frame::InsertMappoint(size_t idx, MappointPtr mappoint){
  assert(idx < FeatureNum());
  _mappoints[idx] = mappoint;
}

bool Frame::BackProjectPoint(size_t idx, Eigen::Vector3d& p3D){
  if(idx >= _depth.size() || _depth[idx] <= 0) return false;
  Eigen::Vector3d p2D;
  if(!GetKeypointPosition(idx, p2D)) return false;
  if(!_camera->BackProjectStereo(p2D, p3D)) return false;
  return true;
}

CameraPtr Frame::GetCamera(){
  return _camera;
}

void Frame::RemoveMappoint(MappointPtr mappoint){
  RemoveMappoint(mappoint->GetKeypointIdx(_frame_id));
}

void Frame::RemoveMappoint(int idx){
  if(idx < _mappoints.size() && idx >=0){
    _mappoints[idx] = nullptr;
  }
}

Eigen::Matrix4d Frame::IMUPose(){
  return _imu_pose;
}

void Frame::SetIMUPose(const Eigen::Matrix4d& pose){
  _imu_pose = pose;
  Eigen::Matrix4d Tbc = _camera->CameraToBody();
  _pose = _imu_pose * Tbc;
}

void Frame::SetIMUPreinteration(const Preinteration& preinteration){
  Preinteration frame_preinteration = preinteration;
  _preinteration = std::make_shared<Preinteration>(frame_preinteration);
}

PreinterationPtr Frame::GetIMUPreinteration(){
  return _preinteration;
}

bool Frame::VelocityIsInitialized(){
  return _init_v;
}

void Frame::SetVelocaity(const Eigen::Vector3d& velocity){
  _velocity = velocity;
  _init_v = true;
}

Eigen::Vector3d Frame::GetVelocity(){
  return _velocity;
}

void Frame::SetPreviousFrame(std::shared_ptr<Frame> previous_frame){
  _previous_frame = previous_frame;
}

std::shared_ptr<Frame> Frame::PreviousFrame(){
  return _previous_frame;
}

void Frame::SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate){
  _preinteration->SetBias(gyr_bias, acc_bias, to_repropagate);
}

void Frame::UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias){
  _preinteration->UpdateBias(gyr_bias, acc_bias);
}

void Frame::GetBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias){
  _preinteration->GetUpdatedBias(gyr_bias, acc_bias);
}

void Frame::Repropagate(){
  _preinteration->Repropagate();
}
