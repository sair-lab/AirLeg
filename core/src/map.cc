#include <set>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath> 
#include <math.h>
#include <limits>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "map.h"
#include <tuple>
#include "utils.h"
#include "frame.h"
#include "g2o_optimization/g2o_optimization.h"
#include "g2o_optimization/types.h"
#include "online_fusion.h"

Map::Map(): _publisher(MakeNullPublisher()), _imu_init(false), imu_init_stage(0){
}

Map::Map(OptimizationConfig& backend_optimization_config, CameraPtr camera, PublisherPtr publisher):
    _backend_optimization_config(backend_optimization_config), _camera(camera),
    _publisher(publisher ? std::move(publisher) : MakeNullPublisher()), _imu_init(false), imu_init_stage(0){
}

void Map::InsertKeyframe(FramePtr frame){
  // insert keyframe to map
  int frame_id = frame->GetFrameId();
  _keyframes[frame_id] = frame;
  _keyframe_ids.push_back(frame_id);

  // update mappoints
  std::vector<MappointPtr> new_mappoints;
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  std::vector<double>& depth = frame->GetAllDepth();
  std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  Eigen::Matrix4d& Twf = frame->GetPose();
  Eigen::Matrix3d Rwf = Twf.block<3, 3>(0, 0);
  Eigen::Vector3d twf = Twf.block<3, 1>(0, 3);
  // voxel feature management: count the keyframe's tracked points per voxel, new points by decreasing score
  std::map<std::tuple<long, long, long>, int> voxel_count;
  std::vector<size_t> order(frame->FeatureNum());
  for(size_t i = 0; i < order.size(); ++i) order[i] = i;
  auto voxel_of = [this](const Eigen::Vector3d& p){
    return std::make_tuple(static_cast<long>(std::floor(p.x() / _voxel_size)), static_cast<long>(std::floor(p.y() / _voxel_size)),
                           static_cast<long>(std::floor(p.z() / _voxel_size)));
  };
  const bool voxel = _voxel_size > 0.0 && _voxel_max > 0 && _keyframes.size() > 1;
  if(voxel){
    for(size_t i = 0; i < mappoints.size(); ++i)
      if(mappoints[i] && mappoints[i]->IsValid()) ++voxel_count[voxel_of(mappoints[i]->GetPosition())];
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& f = frame->GetAllFeatures();
    std::stable_sort(order.begin(), order.end(), [&f](size_t a, size_t b){ return f(0, a) > f(0, b); });
  }
  for(size_t i : order){
    MappointPtr mpt = mappoints[i];
    if(!mpt && voxel){
      Eigen::Vector3d pf;
      if(frame->BackProjectPoint(i, pf)){
        int& n = voxel_count[voxel_of(Rwf * pf + twf)];
        if(n >= _voxel_max){ ++voxel_skipped; continue; }
        ++n;
      }
    }
    if(!mpt){
      if(track_ids[i] < 0) continue;  // would not happen normally
      mpt = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i]));
      Eigen::Matrix<float, 256, 1> descriptor;
      if(frame->GetDescriptor(i, descriptor)){
        mpt->SetDescriptor(descriptor);
      }
      
      Eigen::Vector3d pf;
      if(frame->BackProjectPoint(i, pf)){
        Eigen::Vector3d pw = Rwf * pf + twf;
        mpt->SetPosition(pw);
      }
      frame->InsertMappoint(i, mpt);
      new_mappoints.push_back(mpt);
    }
    mpt->AddObverser(frame_id, i);
    if(mpt->GetType() == Mappoint::Type::UnTriangulated && mpt->ObverserNum() > 2){
      TriangulateMappoint(mpt);
    }
  }

  // add new mappoints to map
  for(MappointPtr mpt:new_mappoints){
    InsertMappoint(mpt);
  }

  // optimization
  if(_keyframes.size() < 2){
    imu_init_frame = frame;
  }else{
    LocalMapOptimization(frame);
    if(!IMUInit() && _camera->UseIMU()){
      InitializeIMU(frame);
    }
  }

}

void Map::CheckAndDeleteMappoint(MappointPtr mpt){
  if(mpt->ObverserNum() < 1){
    _mappoints.erase(mpt->GetId());
  }else if(mpt->ObverserNum() == 1){
    const std::map<int, int>& obversers = mpt->GetAllObversers();
    assert((obversers.size() == 1));
    int obverser_id = obversers.begin()->first;
    int keypoint_idx = obversers.begin()->second;
    std::map<int, FramePtr>::iterator iter = _keyframes.find(obverser_id);
    if(iter == _keyframes.end() || !iter->second){
      _mappoints.erase(mpt->GetId());
    }else if(iter->second->GetRightPosition(keypoint_idx) < 0){
      mpt->SetType(Mappoint::Type::UnTriangulated);
    }
  }
}

void Map::DeleteKeyframe(FramePtr frame){
  int frame_id = frame->GetFrameId();
  _keyframes.erase(frame_id);

  std::vector<int>::iterator keyframe_id_position = std::find(_keyframe_ids.begin(), _keyframe_ids.end(), frame_id);
  if (keyframe_id_position != _keyframe_ids.end()){
    _keyframe_ids.erase(keyframe_id_position);
  }

  std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  for(MappointPtr& mpt : mappoints){
    if(mpt){
      mpt->RemoveObverser(frame_id);
      CheckAndDeleteMappoint(mpt);
    }
  }
}

void Map::InsertMappoint(MappointPtr mappoint){
  int mappoint_id = mappoint->GetId();
  _mappoints[mappoint_id] = mappoint;
}

CameraPtr Map::GetCameraPtr(){
  return _camera;
}

FramePtr Map::GetFramePtr(int frame_id){
  if(_keyframes.count(frame_id) == 0){
    return nullptr;
  }
  return _keyframes[frame_id];
}

bool Map::TriangulateMappoint(MappointPtr mappoint){
  const std::map<int, int> obversers = mappoint->GetAllObversers();
  Eigen::Matrix3Xd G_bearing_vectors;
  Eigen::Matrix3Xd p_G_C_vector;
  G_bearing_vectors.resize(Eigen::NoChange, obversers.size());
  p_G_C_vector.resize(Eigen::NoChange, obversers.size());
  int num_valid_obversers = 0;
  for(const auto kv : obversers){
    int frame_id = kv.first;
    int keypoint_id = kv.second;
    if(_keyframes.count(frame_id) == 0) continue;
    if(keypoint_id < 0) continue;
    // if(!_keyframes[frame_id]->IsValid()) continue;
    Eigen::Vector3d keypoint_pos;
    if(!_keyframes[frame_id]->GetKeypointPosition(keypoint_id, keypoint_pos)) continue;

    Eigen::Vector3d backprojected_pos;
    _camera->BackProjectMono(keypoint_pos.head(2), backprojected_pos);
    Eigen::Matrix4d frame_pose = _keyframes[frame_id]->GetPose();
    Eigen::Matrix3d frame_R = frame_pose.block<3, 3>(0, 0);
    Eigen::Vector3d frame_p = frame_pose.block<3, 1>(0, 3);

    p_G_C_vector.col(num_valid_obversers) = frame_p;
    G_bearing_vectors.col(num_valid_obversers) = frame_R * backprojected_pos;
    num_valid_obversers++;
  }
  if(num_valid_obversers < 2) return false;

  Eigen::Matrix3Xd t_G_bv = G_bearing_vectors.leftCols(num_valid_obversers);
  Eigen::Matrix3Xd p_G_C = p_G_C_vector.leftCols(num_valid_obversers);

  const Eigen::MatrixXd BiD = t_G_bv *
      t_G_bv.colwise().squaredNorm().asDiagonal().inverse();
  const Eigen::Matrix3d AxtAx = num_valid_obversers * Eigen::Matrix3d::Identity() -
      BiD * t_G_bv.transpose();
  const Eigen::Vector3d Axtbx = p_G_C.rowwise().sum() - BiD *
      t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose();

  Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 1e-5;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if(rank < 3) return false;
  
  Eigen::Vector3d p_G_P = qr.solve(Axtbx);
  mappoint->SetPosition(p_G_P);
  return true;
}

void Map::LocalMapOptimization(FramePtr new_frame){
  int new_frame_id = new_frame->GetFrameId();  
  ApplyBackendWriteBack();

  MapOfPoses poses;
  MapOfPoints3d points;
  MapOfVelocity velocities;
  MapOfBias biases;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;
  VectorOfIMUConstraints imu_constraints;
  Eigen::Matrix3d Rwg = _Rwg;

  // camera
  camera_list.emplace_back(_camera);

  // select frames to optimize
  const size_t MaxFrameNumber = 5;
  size_t fixed_frame_num = 0;
  std::vector<FramePtr> neighbor_frames;
  size_t frame_num = std::min(MaxFrameNumber, _keyframes.size());
  neighbor_frames.push_back(new_frame);
  FramePtr last_frame = new_frame;

  while(neighbor_frames.size() < frame_num){
    last_frame = last_frame->PreviousFrame();
    if(!last_frame) break;   // chain shorter than the window
    neighbor_frames.push_back(last_frame);
  }

  // convert frame to vertex
  for(size_t i = 0; i < neighbor_frames.size(); ++i){
    FramePtr frame = neighbor_frames[i];
    bool fix_this_frame = ((frame->GetFrameId() == _keyframes.begin()->first) || (i == neighbor_frames.size()-1));
    if(fix_this_frame) fixed_frame_num++;

    if(IMUInit()){
      AddFrameVertex(frame, poses, 0, velocities, biases, imu_constraints, fix_this_frame, !fix_this_frame, fix_this_frame);
    }else{
      AddFrameVertex(frame, poses, 0, fix_this_frame);
    }
    frame->local_map_optimization_frame_id = new_frame_id;
  }

  // select mappoints and add fixed frames 
  std::map<FramePtr, int> fixed_frames;
  std::vector<MappointPtr> mappoints;
  for(auto neighbor_frame : neighbor_frames){
    std::vector<MappointPtr>& neighbor_mappoints = neighbor_frame->GetAllMappoints();
    for(MappointPtr mpt : neighbor_mappoints){
      if(!mpt || !mpt->IsValid() || mpt->local_map_optimization_frame_id == new_frame_id) continue;
      mpt->local_map_optimization_frame_id = new_frame_id;
      mappoints.push_back(mpt);

      const std::map<int, int>& obversers = mpt->GetAllObversers();
      for(auto& kv : obversers){
        FramePtr kf = GetFramePtr(kv.first);
        if(kf && kf->local_map_optimization_frame_id != new_frame_id){
          fixed_frames[kf]++;
        }
      }
    }

  }

  const size_t max_fixed_frame_num = SIZE_MAX;
  if(fixed_frames.size() > 0 && max_fixed_frame_num > fixed_frame_num){
    std::set<std::pair<int, FramePtr>> ordered_fixed_frames;
    for(auto& kv : fixed_frames){
      ordered_fixed_frames.insert(std::pair<int, FramePtr>(kv.second, kv.first));
    }

    size_t to_add_fixed_num = std::min((max_fixed_frame_num-fixed_frame_num), ordered_fixed_frames.size());
    for(std::set<std::pair<int, FramePtr>>::reverse_iterator rit = ordered_fixed_frames.rbegin(); to_add_fixed_num > 0; to_add_fixed_num--, rit++){
      rit->second->local_map_optimization_fix_frame_id = new_frame_id;
      AddFrameVertex(rit->second, poses, 0, true);
    }
    fixed_frame_num += to_add_fixed_num;
  }

  // add point constraint
  for(auto& mpt : mappoints){
    if(!mpt || !mpt->IsValid()) continue;

    const std::map<int, int> obversers = mpt->GetAllObversers();

    // vertex
    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = false;

    // constraints
    VectorOfMonoPointConstraints tmp_mono_point_constraints;
    VectorOfStereoPointConstraints tmp_stereo_point_constraints;
    for(auto& kv : obversers){
      FramePtr kf = GetFramePtr(kv.first);
      if(!kf || (kf->local_map_optimization_frame_id != new_frame_id && kf->local_map_optimization_fix_frame_id != new_frame_id)) continue;

      Eigen::Vector3d keypoint; 
      if(!kf->GetKeypointPosition(kv.second, keypoint)) continue;
      // visual constraint
      AddPointConstraint(kv.first, mpt_id, keypoint, tmp_mono_point_constraints, tmp_stereo_point_constraints);
    }

    // add to optimization
    if(tmp_stereo_point_constraints.size() > 0 || tmp_mono_point_constraints.size() > 1){
      points.insert(std::pair<int, Position3d>(mpt_id, point));
      mono_point_constraints.insert(mono_point_constraints.end(),
          tmp_mono_point_constraints.begin(), tmp_mono_point_constraints.end());
      stereo_point_constraints.insert(stereo_point_constraints.end(),
          tmp_stereo_point_constraints.begin(), tmp_stereo_point_constraints.end());
    }
  }

  // S3: leg relative-pose and foot-surface factors (no-op without online fusion)
  LocalBAExtras extras;
  LocalBAExtras* extras_ptr = nullptr;
  if(_online){
    _online->PrepareLocalBA(neighbor_frames, poses, [this](int id){ return GetFramePtr(id); }, &extras);
    extras_ptr = &extras;
  }
  // Weak motion priors of lost keyframes without a leg / IMU edge (tracking break): sigma grows with the gap,
  // 0.1 m and 2 deg per second (at least 1 cm / 0.2 deg).
  for(const FramePtr& frame : neighbor_frames){
    FramePtr previous = frame->PreviousFrame();
    if(!frame->has_motion_prior || !previous || !poses.count(previous->GetFrameId())) continue;
    LegRelativeConstraint c;
    c.id_pose1 = previous->GetFrameId();
    c.id_pose2 = frame->GetFrameId();
    c.T_c1_c2.matrix() = frame->motion_prior;
    const double dt = std::max(frame->motion_prior_dt, 0.0);
    const double sigma_r = std::max(2.0 * M_PI / 180.0 * dt, 0.2 * M_PI / 180.0), sigma_t = std::max(0.1 * dt, 0.01);
    c.information.setZero();
    c.information.diagonal() << Eigen::Vector3d::Constant(1.0 / (sigma_r * sigma_r)), Eigen::Vector3d::Constant(1.0 / (sigma_t * sigma_t));
    extras.legs.push_back(c);
    extras_ptr = &extras;
  }
  LocalmapOptimization(poses, points, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, imu_constraints, Rwg,
      _backend_optimization_config, extras_ptr);
  if(_online) _online->FinishLocalBA(neighbor_frames, extras);

  // erase point outliers
  std::vector<std::pair<FramePtr, MappointPtr>> outliers;
  for(auto& mono_point_constraint : mono_point_constraints){
    if(!mono_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(mono_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(mono_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }

  for(auto& stereo_point_constraint : stereo_point_constraints){
    if(!stereo_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(stereo_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(stereo_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }
  RemoveOutliers(outliers);

  for(auto& kv : poses){
    int frame_id = kv.first;
    Pose3d pose = kv.second;
    if(pose.fixed || _keyframes.count(frame_id) == 0) continue;
    Eigen::Matrix4d pose_eigen = Eigen::Matrix4d::Identity();
    pose_eigen.block<3, 3>(0, 0) = pose.R;
    pose_eigen.block<3, 1>(0, 3) = pose.p;
    _keyframes[frame_id]->SetPose(pose_eigen);

    if(velocities.count(frame_id) > 0){
      _keyframes[frame_id]->SetVelocaity(velocities[frame_id].velocity);
    }
  
    if(biases.count(frame_id) > 0){
      _keyframes[frame_id]->UpdateBias(biases[frame_id].gyr_bias, biases[frame_id].acc_bias);
    }

  }

  for(auto& kv : points){
    int mpt_id = kv.first;
    Position3d position = kv.second;
    if(_mappoints.count(mpt_id) == 0) continue;
    _mappoints[mpt_id]->SetPosition(position.p);
  }

  if(_online){
    // Optimized for the last time: the fixed last window frame and, in a full window, the one before it (it becomes
    // the fixed last frame of the next BA).
    std::vector<FramePtr> finished{neighbor_frames.back()};
    if(neighbor_frames.size() == MaxFrameNumber && neighbor_frames.size() >= 2)
      finished.push_back(neighbor_frames[neighbor_frames.size() - 2]);
    _online->Finalize(neighbor_frames, finished);
  }

  if(!_camera->UseIMU() || IMUInit()){
    PublishWindow(new_frame->GetTimestamp(), neighbor_frames);
  }
}

// S3 back-end write-back, like a loop correction: the solved keyframes take the back-end poses, newer
// keyframes follow the newest one rigidly, and map points move with their first observer.
bool MoveWithFirstCorrectedObserver(const MappointPtr& mpt, const std::map<int, Eigen::Matrix4d>& corrections){
  for(const auto& observer : mpt->GetAllObversers()){
    auto c = corrections.find(observer.first);
    if(c == corrections.end()) continue;
    mpt->SetPosition(c->second.topLeftCorner<3, 3>() * mpt->GetPosition() + c->second.topRightCorner<3, 1>());
    return true;
  }
  return false;
}

void Map::ApplyBackendWriteBack(){
  FusionBackend* backend = _online ? _online->Backend() : nullptr;
  if(!backend || !backend->Config().write_back) return;
  std::map<int, Eigen::Isometry3d> poses;
  int newest = -1;
  if(!backend->TakeWriteBack(&poses, &newest)) return;
  const auto t0 = std::chrono::steady_clock::now();
  std::map<int, Eigen::Matrix4d> corrections;
  for(const auto& kv : poses){
    auto it = _keyframes.find(kv.first);
    if(it != _keyframes.end() && it->second) corrections[kv.first] = kv.second.matrix() * it->second->GetPose().inverse();
  }
  if(corrections.count(newest)){
    const Eigen::Matrix4d C = corrections[newest];
    for(auto it = _keyframes.upper_bound(newest); it != _keyframes.end(); ++it) corrections[it->first] = C;
  }
  std::set<int> moved;
  for(const auto& kv : corrections){
    for(const MappointPtr& mpt : _keyframes[kv.first]->GetAllMappoints()){
      if(!mpt || !mpt->IsValid() || !moved.insert(mpt->GetId()).second) continue;
      MoveWithFirstCorrectedObserver(mpt, corrections);
    }
  }
  for(const auto& kv : corrections){
    FramePtr frame = _keyframes[kv.first];
    frame->SetPose(kv.second * frame->GetPose());
  }
  backend->RecordWriteBack(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Map::RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers){
  for(auto& kv : outliers){
    FramePtr frame = kv.first;
    MappointPtr mpt = kv.second;
    if(!frame || !mpt || mpt->IsBad()) continue;
    to_update_track_id.emplace_back(std::make_pair(frame, mpt->GetKeypointIdx(frame->GetFrameId())));

    frame->RemoveMappoint(mpt);
    mpt->RemoveObverser(frame->GetFrameId());
    CheckAndDeleteMappoint(mpt);
  }
}

int Map::LinkMappoints(FramePtr frame, const std::vector<std::pair<int, MappointPtr>>& links){
  const int frame_id = frame->GetFrameId();
  int linked = 0;
  for(const auto& link : links){
    const int idx = link.first;
    MappointPtr old_mpt = link.second;
    if(!old_mpt || !old_mpt->IsValid()) continue;
    if(old_mpt->GetKeypointIdx(frame_id) >= 0) continue;  // already observed by this frame (here or elsewhere)
    MappointPtr current = frame->GetMappoint(idx);
    if(current == old_mpt) continue;
    if(current && current->IsValid()){
      // merge `current` into `old_mpt`: every observer must stay one keypoint per point
      const std::map<int, int> observers = current->GetAllObversers();
      bool compatible = true;
      for(const auto& o : observers){
        const int k = old_mpt->GetKeypointIdx(o.first);
        if(k >= 0 && k != o.second){ compatible = false; break; }
      }
      if(!compatible) continue;
      for(const auto& o : observers){
        FramePtr observer = GetFramePtr(o.first);
        if(!observer) continue;
        observer->InsertMappoint(o.second, old_mpt);
        observer->SetTrackId(o.second, old_mpt->GetId());
        old_mpt->AddObverser(o.first, o.second);
      }
      current->SetBad();
      _mappoints.erase(current->GetId());
    }else{
      if(current){   // an invalid point leaves the slot: detach this frame's observation, else it keeps observing the
        current->RemoveObverser(frame_id);   // frame after its deletion (stale observer)
        CheckAndDeleteMappoint(current);
      }
      frame->InsertMappoint(idx, old_mpt);
      frame->SetTrackId(idx, old_mpt->GetId());
      old_mpt->AddObverser(frame_id, idx);
    }
    ++linked;
  }
  return linked;
}

int Map::UpdateFrameTrackIds(int track_id){
  for(auto& pair : to_update_track_id){
    FramePtr frame = pair.first;
    int kpt_id = pair.second;

    frame->SetTrackId(kpt_id, track_id);
    MappointPtr mpt = std::shared_ptr<Mappoint>(new Mappoint(track_id));
    track_id++;

    Eigen::Matrix<float, 256, 1> descriptor;
    if(frame->GetDescriptor(kpt_id, descriptor)){
      mpt->SetDescriptor(descriptor);
    }
      
    Eigen::Vector3d pf;
    if(frame->BackProjectPoint(kpt_id, pf)){
      Eigen::Matrix4d& Twf = frame->GetPose();
      Eigen::Matrix3d Rwf = Twf.block<3, 3>(0, 0);
      Eigen::Vector3d twf = Twf.block<3, 1>(0, 3);
      Eigen::Vector3d pw = Rwf * pf + twf;
      mpt->SetPosition(pw);
    }
    frame->InsertMappoint(kpt_id, mpt);
    mpt->AddObverser(frame->GetFrameId(), kpt_id);
    InsertMappoint(mpt);
  }

  to_update_track_id.clear();
  return track_id;
}

void Map::SaveKeyframeTrajectory(std::string file_path){
  std::vector<std::pair<double, Eigen::Matrix4d>> trajectory;
  for(auto& kv : _keyframes){
    FramePtr kf = kv.second;
    trajectory.emplace_back(std::make_pair(kf->GetTimestamp(), kf->GetPose()));
  }

  SaveTumTrajectoryToFile(file_path, trajectory);
}

bool Map::InitializeIMU(FramePtr frame){
  if(frame->GetTimestamp() - imu_init_frame->GetTimestamp() < 3 || _keyframes.size() < 10) return false;
  FramePtr last_frame = frame->PreviousFrame();
  std::vector<FramePtr> frame_list;
  frame_list.push_back(frame);
  while(last_frame && last_frame->GetTimestamp() >= imu_init_frame->GetTimestamp()){
    frame_list.push_back(last_frame);
    last_frame = last_frame->PreviousFrame();
  }
  if(frame_list.size() < 10) return false;

  for(size_t i = 0; i < frame_list.size()-1; i++){
    double dist = (frame_list[i]->GetPose().block<3, 1>(0, 3) - frame_list[i+1]->GetPose().block<3, 1>(0, 3)).norm();
    if(dist < 0.005){
      imu_init_frame = frame_list[i];
      std::cout << "Not enough motion" << std::endl;
      return false;
    }
  }


  // std::cout << "--------------before computing gyr bias--------------------" << std::endl;
  Eigen::Vector3d dbg = Eigen::Vector3d::Zero();
  if(ComputeGyrBias(frame_list, dbg)){
    for(size_t i = 0; i < frame_list.size(); i++){
      if(!frame_list[i]->GetIMUPreinteration()) continue;
      Eigen::Vector3d gyr_bias, acc_bias;
      frame_list[i]->GetBias(gyr_bias, acc_bias);
      frame_list[i]->SetBias(gyr_bias+dbg, acc_bias);
    }
  }
  // std::cout << "--------------after computing gyr bias--------------------" << std::endl;

  Eigen::Vector3d gw;
  ComputeVelocity(frame_list, gw);
  gw.normalize();

  std::cout << "Start initialize imu" << std::endl;
 
  MapOfPoses poses;
  std::vector<CameraPtr> camera_list;
  MapOfVelocity velocities;
  Bias bias;
  VectorOfIMUConstraints imu_constraints;
  
  camera_list.emplace_back(_camera);

  Eigen::Vector3d prior_gyr_bias, prior_acc_bias;
  frame_list[0]->GetBias(prior_gyr_bias, prior_acc_bias);
  bias.gyr_bias = prior_gyr_bias;
  bias.acc_bias = prior_acc_bias;
  bias.fixed = false;

  Eigen::Vector3d gravity_direction;
  for(size_t i = 0; i < frame_list.size(); i++){
    FramePtr frame = frame_list[i];
    FramePtr last_frame = frame->PreviousFrame();
    PreinterationPtr preinteration = frame->GetIMUPreinteration();

    AddFrameVertex(frame, poses, 0, true);

    if(i > 0){
      double dd = (frame->IMUPose().block<3, 1>(0, 3) - frame_list[i-1]->IMUPose().block<3, 1>(0, 3)).norm();
      std::cout << "frame_id = " << frame->GetFrameId() << ", dt = " << preinteration->dT << ", d = " << dd << ", v = " << (dd/preinteration->dT) << std::endl;
    }
    // std::cout << "i = " << i << ", twb = " << frame->IMUPose().block<3, 1>(0, 3).transpose() << std::endl;

    Velocity velocity;
    velocity.velocity = frame->GetVelocity();
    velocity.fixed = false;
    velocities.insert(std::pair<int, Velocity>(frame->GetFrameId(), velocity)); 

    if(last_frame != nullptr && i < frame_list.size()-1){
      IMUConstraintPtr imu_constraint = std::shared_ptr<ImuConstraint>(new ImuConstraint()); 
      imu_constraint->id_pose1 = last_frame->GetFrameId();
      imu_constraint->id_pose2 = frame->GetFrameId();
      imu_constraint->id_camera1 = 0;
      imu_constraint->id_camera2 = 0;
      imu_constraint->preinteration = preinteration;
      imu_constraints.emplace_back(imu_constraint);
    }
  }

  Eigen::Matrix3d Rwg;
  gravity_direction.normalize();
  Eigen::Vector3d gI;
  gI << 0.0, 0.0, -1.0;
  Eigen::Vector3d axis = gI.cross(gw);;
  double angle = acos(gI.dot(gw));
  Eigen::Vector3d rotation_vector = axis.normalized() * angle;
  SO3Exp(rotation_vector, Rwg);

  // std::cout << "-------------------Before initialization----------------------" << std::endl;
  if(!IMUInitialization(poses, velocities, bias, camera_list, imu_constraints, Rwg)) return false;

  // std::cout << "-------------------After initialization----------------------" << std::endl;

  // Change the map
  // 1. Update velocities and bias
  for(size_t i = 0; i < frame_list.size(); i++){
    frame_list[i]->SetBias(bias.gyr_bias, bias.acc_bias);
    frame_list[i]->SetVelocaity(velocities[frame_list[i]->GetFrameId()].velocity);
  }

  // 2. Delete keyframes before imu_init_frame and related mappoints
  int init_frame_id = imu_init_frame->GetFrameId();
  for(auto& kv : _keyframes){
    FramePtr kf = kv.second;
    if(kf->GetFrameId() < init_frame_id){
      DeleteKeyframe(kf);
    }
  }

  Eigen::Matrix4d Tgw = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d Rgw = Rwg.transpose();
  Eigen::Vector3d tgw = -Rgw * imu_init_frame->IMUPose().block<3, 1>(0, 3);
  Tgw.block<3, 3>(0, 0) = Rgw;
  Tgw.block<3, 1>(0, 3) = tgw;
  for(auto& kv : _keyframes){
    FramePtr kf = kv.second;
    kf->SetIMUPose(Tgw*kf->IMUPose());
    kf->SetVelocaity((Rgw*kf->GetVelocity()));
  }
  for(auto& kv : _mappoints){
    MappointPtr mpt = kv.second;
    if(mpt->IsValid()){
      mpt->SetPosition((Rgw*mpt->GetPosition()+tgw));
    }
  }

  // Set first frame fixed and reset its preinteration
  imu_init_frame->SetPoseFixed(true);
  imu_init_frame->GetIMUPreinteration()->Reset();

  SetRwg(Eigen::Matrix3d::Identity());
  SetIMUInit(true);
  // std::cout << "-------------------After rotating----------------------" << std::endl;

  return true;
}

void Map::SetRwg(const Eigen::Matrix3d& Rwg){
  _Rwg = Rwg;
}

Eigen::Matrix3d Map::GetRwg(){
  return _Rwg;
}

void Map::SetIMUInit(bool imu_init){
  _imu_init = imu_init;
}

bool Map::IMUInit(){
  return _imu_init;
}

void Map::SetPublisher(PublisherPtr publisher){
  _publisher = publisher ? std::move(publisher) : MakeNullPublisher();
}

void Map::Publish(double time, bool clear_old_message){
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);

  keyframe_message->time = time;
  for(auto&kv : _keyframes){
    const Eigen::Matrix4d& pose_eigen = kv.second->GetPose();
    keyframe_message->times.push_back(kv.second->GetTimestamp());
    keyframe_message->ids.push_back(kv.first);
    keyframe_message->poses.emplace_back(pose_eigen);
  }

  map_message->time = time;
  for(auto& kv : _mappoints){
    if(kv.second->IsValid()){
      map_message->ids.push_back(kv.first);
      map_message->points.push_back(kv.second->GetPosition());
    }
  }

  if(clear_old_message){
    _publisher->Clear();
  }

  _publisher->PublisheKeyframe(keyframe_message);
  _publisher->PublishMap(map_message);
}

void Map::PublishWindow(double time, const std::vector<FramePtr>& window){
  if(_window_publishes++ % 50 == 0){
    Publish(time, _window_publishes > 1);
    return;
  }
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);
  keyframe_message->time = time;
  map_message->time = time;
  std::set<int> seen;
  for(const FramePtr& frame : window){
    keyframe_message->times.push_back(frame->GetTimestamp());
    keyframe_message->ids.push_back(frame->GetFrameId());
    keyframe_message->poses.emplace_back(frame->GetPose());
    for(const MappointPtr& mpt : frame->GetAllMappoints()){
      if(!mpt || !mpt->IsValid() || !seen.insert(mpt->GetId()).second) continue;
      map_message->ids.push_back(mpt->GetId());
      map_message->points.push_back(mpt->GetPosition());
    }
  }
  _publisher->PublisheKeyframe(keyframe_message);
  _publisher->PublishMap(map_message);
}

std::map<int, MappointPtr>& Map::GetAllMappoints(){
  return _mappoints;
}

std::map<int, FramePtr>& Map::GetAllKeyframes(){
  return _keyframes;
}

int Map::RemoveInValidMappoints(){
  int mappoint_size = _mappoints.size();
  int stale_observers = 0;
  std::map<int, MappointPtr>::iterator it_mpt = _mappoints.begin();
  for(; it_mpt != _mappoints.end(); ){
    int mpt_id = it_mpt->first;
    MappointPtr mpt = it_mpt->second;

    if(!mpt){
      _mappoints.erase(it_mpt++);
    }else if(mpt->IsValid()){
      it_mpt++;
    }else{
      std::map<int, int>& obversers = mpt->GetAllObversers();
      for(auto& obverser : obversers){
        int frame_id = obverser.first;
        int kpt_idx = obverser.second;
        auto it_kf = _keyframes.find(frame_id);   // an observer whose keyframe was deleted: nothing to detach
        if(it_kf == _keyframes.end() || !it_kf->second){ ++stale_observers; continue; }
        it_kf->second->RemoveMappoint(kpt_idx);
      }
      _mappoints.erase(it_mpt++);
    }
  }
  if(stale_observers > 0) std::cout << "RemoveInValidMappoints: stale observers = " << stale_observers << std::endl;

  return (mappoint_size - _mappoints.size());
}

void Map::UpdateCovisibilityGraph(){
  _covisibile_frames.clear();
  for(const auto& kv : _keyframes){
    UpdateFrameCovisibility(kv.second);
  }
}

void Map::UpdateFrameCovisibility(FramePtr frame){
  if(_covisibile_frames.find(frame) != _covisibile_frames.end()){  
    _covisibile_frames[frame].clear();
  }

  std::map<int, FramePtr>::iterator frame_iter;
  std::map<FramePtr, int>::iterator covi_iter;
  const std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  for(const MappointPtr mpt : mappoints){
    if(!mpt || !mpt->IsValid()) continue;

    const std::map<int, int>& obversers = mpt->GetAllObversers();
    for(const auto& kv : obversers){
      const int covi_frame_id = kv.first;
      frame_iter = _keyframes.find(covi_frame_id);
      if(frame_iter == _keyframes.end()) continue;

      FramePtr covi_frame = frame_iter->second;
      covi_iter = _covisibile_frames[frame].find(covi_frame);
      if(covi_iter == _covisibile_frames[frame].end()){
        _covisibile_frames[frame][covi_frame] = 1;
      }else{
        covi_iter->second++;
      }    
    }
  }
}

void Map::GetConnectedFrames(FramePtr frame, std::map<FramePtr, int>& covi_frames){
  std::map<FramePtr, std::map<FramePtr, int>>::iterator it = _covisibile_frames.find(frame);
  if(it != _covisibile_frames.end()){
    covi_frames = it->second;
  }
}

double Map::MapScale(){
  std::vector<double> xs, ys, zs;
  for(auto& kv : _mappoints){
    MappointPtr mpt = kv.second;
    if(mpt->IsValid()){
      Eigen::Vector3d p = mpt->GetPosition();
      xs.push_back(p(0));
      ys.push_back(p(1));
      zs.push_back(p(2));
    }
  }

  double std_dev_x = CalculateStdDev(xs);
  double std_dev_y = CalculateStdDev(ys);
  double std_dev_z = CalculateStdDev(zs);

  double max_std_dev = std::max(std_dev_x, std::max(std_dev_y, std_dev_z));
  return 3 * max_std_dev;
}

void Map::CheckMap(){
  std::cout << "check map" << std::endl;
  std::cout << "_mappoints = " << _mappoints.size() << std::endl;
  std::cout << "_keyframes = " << _keyframes.size() << std::endl;
  
  // check map
  for(auto kv : _keyframes){
    FramePtr frame = kv.second;
    std::vector<MappointPtr>& mpts = frame->GetAllMappoints();
    for(size_t i = 0; i < mpts.size(); i++){
      MappointPtr mpt = mpts[i];
      if(!mpt){
        std::cout << "nullptr, frame_id = " << kv.first << ", kpt id = " << i << std::endl;
        continue;
      }

      int mpt_id_ = mpt->GetId();
      if(_mappoints.find(mpt_id_) == _mappoints.end()){
        std::cout << "frame_id = " << kv.first << ", kpt id = " << i << ", mpt_id_ = " << mpt_id_ << ", mpt type = " << mpt->GetType() << std::endl;
      }
    }

  }
}

// Export rectified stereo observations in the existing S2 nine-column format.
void Map::ExportTrainingData(const std::string& map_root, bool stereo_fallback){
  const auto root = std::filesystem::path(map_root) / "leg_odom";
  std::filesystem::create_directories(root / "keypoints");
  auto camera = GetCameraPtr();
  std::ofstream calibration(root / "camera.yaml");
  calibration.exceptions(std::ios::failbit | std::ios::badbit);
  calibration << std::setprecision(17);
  for(int side = 0; side < 2; ++side){
    calibration << "cam" << side << ":\n  p: [" << camera->Fx() << ", 0, "
      << camera->Cx() << ", " << (side ? -camera->BF() : 0.0)
      << ", 0, " << camera->Fy() << ", " << camera->Cy()
      << ", 0, 0, 0, 1, 0]\n";
  }
  for(const auto& item : _keyframes){
    const auto& frame = item.second;
    std::ostringstream name;
    name << std::fixed << std::setprecision(9) << frame->GetTimestamp() << ".txt";
    std::ofstream output(root / "keypoints" / name.str());
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << std::setprecision(12);
    for(size_t i = 0; i < frame->FeatureNum(); ++i){
      const int id = frame->GetTrackId(i);
      if(id < 0) continue;
      const auto& keypoint = frame->GetKeypoint(i);
      const double right = frame->GetRightPosition(i);
      Eigen::Vector3d point = Eigen::Vector3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
      // Map points are in world coordinates; supervision uses the observing camera.
      const auto mappoint = frame->GetMappoint(i);
      if(mappoint && mappoint->IsValid() && !mappoint->IsBad()){
        const auto& pose = frame->GetPose();  // T_W_C
        point = pose.block<3, 3>(0, 0).transpose() *
            (mappoint->GetPosition() - pose.block<3, 1>(0, 3));
      }
      if(!point.allFinite() || point.z() <= 0){
        point.setConstant(std::numeric_limits<double>::quiet_NaN());
      }
      if(stereo_fallback && !point.allFinite() && right >= 0 && keypoint.pt.x > right){
        camera->BackProjectStereo(Eigen::Vector3d(keypoint.pt.x, keypoint.pt.y, right), point);
      }
      output << id << ' ' << keypoint.response << ' ' << keypoint.pt.x << ' '
        << keypoint.pt.y << ' ' << right << ' ' << keypoint.pt.y << ' '
        << point.x() << ' ' << point.y() << ' ' << point.z() << '\n';
    }
  }
}

void Map::ExportKeyframeStatus(const std::string& map_root){
  std::ofstream output(std::filesystem::path(map_root) / "leg_odom" / "keyframe_status.csv");
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output << "timestamp,frame_id,track_state,pose_source\n";
  for(const auto& item : _keyframes){
    const auto& frame = item.second;
    const char* state = frame->track_state == TrackState::Normal ? "normal"
        : frame->track_state == TrackState::Weak ? "weak" : "lost";
    output << std::fixed << std::setprecision(9) << frame->GetTimestamp() << ',' << item.first << ',' << state
           << ',' << PoseSourceName(frame->pose_source) << '\n';
  }
}
