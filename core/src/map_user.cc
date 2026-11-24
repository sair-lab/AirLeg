#include <stdexcept>
#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <boost/serialization/serialization.hpp>

#include "map_refiner.h"
#include "read_configs.h"
#include "imu.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"
#include "g2o_optimization/edge_relative_pose.h"
#include "map_user.h"
#include "runtime_utils.h"


MapUser::MapUser(){
}

MapUser::MapUser(RelocalizationConfigs& configs, PublisherPtr publisher): _configs(configs), _stop(false){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.superpoint_config));
  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config.WithImageSize(
      static_cast<int>(_camera->ImageWidth()), static_cast<int>(_camera->ImageHeight()))));
  _map = std::shared_ptr<Map>(new Map());
  _publisher = publisher ? std::move(publisher) : MakeNullPublisher();

  _reloc_message = std::shared_ptr<RelocMessage>(new RelocMessage);
}

void MapUser::PubMap(){
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);

  for(auto&kv : _map->_keyframes){
    const Eigen::Matrix4d& pose_eigen = kv.second->GetPose();
    keyframe_message->times.push_back(kv.second->GetTimestamp());
    keyframe_message->ids.push_back(kv.first);
    keyframe_message->poses.emplace_back(pose_eigen);
  }

  for(auto& kv : _map->_mappoints){
    if(kv.second->IsValid()){
      map_message->ids.push_back(kv.first);
      map_message->points.push_back(kv.second->GetPosition());
    }
  }

  while(!_stop){
    double current_time = CurrentUnixTimeSec();
    keyframe_message->time = current_time;
    map_message->time = current_time;

    _publisher->PublisheKeyframe(keyframe_message);
    _publisher->PublishMap(map_message);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void MapUser::StopVisualization(){
  _stop = true;
  _publisher->ShutDown();
  if(_visualization_thread.joinable()){
    _visualization_thread.join();
  }
}

void MapUser::LoadMap(const std::string& map_root){
  std::string map_v1_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv1.bin");
  std::ifstream ifs(map_v1_path, std::ios::binary);
  if(!ifs) throw std::runtime_error("Cannot open map file: " + map_v1_path);
  boost::archive::binary_iarchive ia(ifs);
  ia >> _map;
  if(!_map || _map->_keyframes.empty()) throw std::runtime_error("Map has no keyframes: " + map_v1_path);

  new_frame_id = _map->_keyframes.rbegin()->first;
  _database = _map->_database;


  _reloc_message->map_scale = _map->MapScale() / 80;
  
  _visualization_thread = std::thread(boost::bind(&MapUser::PubMap, this));
}
  
void MapUser::LoadVocabulary(const std::string voc_path){
  _database->LoadVocabulary(voc_path);
}

bool MapUser::Relocalization(cv::Mat& image, Eigen::Matrix4d& pose){
  cv::Mat image_rect;
  _camera->UndistortImage(image, image_rect);
  Eigen::Matrix<float, 259, Eigen::Dynamic> features;
  _feature_detector->Detect(image_rect, features);

  FramePtr frame = std::shared_ptr<Frame>(new Frame(new_frame_id++, false, _camera, 0));
  int frame_id = frame->GetFrameId();
  frame->AddLeftFeatures(features);

  const double now = CurrentUnixTimeSec();
  if(_configs.ros_publisher_config.feature){
    FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
    feature_message->time = now;
    feature_message->image = image_rect;
    feature_message->keypoints = frame->GetAllKeypoints();
    feature_message->fm_type = FeatureMessgaeType::RelocFeature;
    _publisher->PublishFeature(feature_message);
  }

  DBoW2::WordIdToFeatures word_features;
  DBoW2::BowVector bow_vector;
  std::vector<DBoW2::WordId> word_of_features;
  _database->FrameToBow(features, word_features, bow_vector, word_of_features);

  // query
  std::map<FramePtr, int> frame_sharing_words;
  _database->Query(bow_vector, frame_sharing_words);

  if(frame_sharing_words.empty()) return false;

  // filtration
  int max_sharing_words = 0;
  for(const auto& kv : frame_sharing_words){
    max_sharing_words = kv.second > max_sharing_words ? kv.second : max_sharing_words;
  }
  int sharing_wrods_num_thr = std::max(static_cast<int>(max_sharing_words * 0.3f), 8);

  std::map<FramePtr, int>::iterator fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end();){
    FramePtr fsw = fsw_it->first;
    if(fsw_it->second < sharing_wrods_num_thr){
      fsw_it = frame_sharing_words.erase(fsw_it);
    }else{
      fsw_it++;
    }
  }

  if(frame_sharing_words.empty()) return false;

  // scoring
  std::map<FramePtr, double> frame_scores;
  fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end(); fsw_it++){
    FramePtr fsw = fsw_it->first;
    frame_scores[fsw] = _database->Score(_database->_frame_bow_vectors[fsw], bow_vector);
  }

  int print_debug_info = 0;
  if(print_debug_info){
    std::cout << "======================= frame_sharing_words scoring =================  " << std::endl;
    for(auto& kv : frame_scores){
      double frame_time = kv.first->GetTimestamp();
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time << ", score = " << kv.second << std::endl;
    }
  }

  // grouping
  std::map<FramePtr, RelocalizationGroupCandidate> group_candidates;
  FramePtr best_deputy;
  double best_group_score = -1; 
  std::map<FramePtr, RelocalizationGroupCandidate>::iterator relocalization_group_iter;
  std::map<FramePtr, double>::iterator fs_it = frame_scores.begin();
  for(; fs_it != frame_scores.end(); fs_it++){
    FramePtr fsw = fs_it->first;
    FramePtr deputy_of_group = fsw;
    double deputy_score = fs_it->second;

    RelocalizationGroupCandidate group_candidate;
    group_candidate.group_frames.insert(fsw);
    group_candidate.group_score += deputy_score;

    std::map<FramePtr, int> fsw_covi_frames;
    _map->GetConnectedFrames(fsw, fsw_covi_frames);
    for(auto& kv : fsw_covi_frames){
      FramePtr fsw_covi_frame = kv.first;
      if(kv.second > 10 && frame_scores.count(fsw_covi_frame)){
        double fsw_covi_score = frame_scores[fsw_covi_frame];
        group_candidate.group_frames.insert(fsw_covi_frame);
        group_candidate.group_score += fsw_covi_score;

        if(fsw_covi_score > deputy_score){
          deputy_of_group = fsw_covi_frame;
          deputy_score = fsw_covi_score;
        }
      }
    }

    relocalization_group_iter = group_candidates.find(deputy_of_group);
    if(relocalization_group_iter == group_candidates.end() || relocalization_group_iter->second.group_score < group_candidate.group_score){
      group_candidates[deputy_of_group] = group_candidate;

      if(group_candidate.group_score > best_group_score){
        best_group_score = group_candidate.group_score;
        best_deputy = deputy_of_group;
      }
    }   
  }

  if(best_group_score < 0) return false;

  // only accumulate the scores of top 5 frames for each group
  const size_t CoviFrameScoreNum = 5;
  best_group_score = 0.0;
  for(auto& kv : group_candidates){
    std::vector<double> group_scores;
    for(FramePtr f : kv.second.group_frames){
      group_scores.push_back(frame_scores[f]);
    }

    if(group_scores.size() > CoviFrameScoreNum){
      std::sort(group_scores.rbegin(), group_scores.rend()); 
    }

    double sum_group_score = 0;
    int N = std::min(CoviFrameScoreNum, group_scores.size());
    for(size_t i = 0; i < N; i++){
      sum_group_score += group_scores[i];
    }

    kv.second.group_score = sum_group_score;
    best_group_score = std::max(best_group_score, sum_group_score);
  }

  if(print_debug_info){
    std::cout << "======================= frame_sharing_words grouping =================  " << std::endl;
    std::cout << "best_group_score = " << best_group_score << std::endl;
    for(auto& kv : group_candidates){
      double frame_time = kv.first->GetTimestamp();
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time << ", group_score = " << kv.second.group_score << std::endl;
    }
  }

  // filtration of group
  if(group_candidates.size() > 3){
    double group_score_thr = best_group_score * 0.5;
    auto it = group_candidates.begin();
    for(; it != group_candidates.end();){
      if(it->second.group_score < group_score_thr){  
        it = group_candidates.erase(it);
      }else{
        it++;
      }
    }
  }

  // sorting of group
  std::vector<std::pair<FramePtr, RelocalizationGroupCandidate>> group_vector(group_candidates.begin(), group_candidates.end());
  std::sort(group_vector.begin(), group_vector.end(), [](const auto &a, const auto &b) {
      return a.second.group_score > b.second.group_score;
  });

  // feature matching
  std::vector<cv::DMatch> relocalization_matches;
  FramePtr relocalization_frame;
  const size_t GoodCandidateNum = std::min((size_t)3, group_vector.size());    
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = frame->GetAllFeatures();
  for(size_t i = 0; i < GoodCandidateNum; i++){
    FramePtr good_candidate = group_vector[i].first;
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& good_candidate_features = good_candidate->GetAllFeatures();
    std::vector<cv::DMatch> matches;
    _point_matcher->MatchingPoints(query_features, good_candidate_features, matches, true);
    if(matches.size() > relocalization_matches.size()){
      relocalization_matches = matches;
      relocalization_frame = good_candidate;
    }

    // if(relocalization_matches.size() > 50) break;
  }
  if(relocalization_matches.size() < _configs.min_inlier) return false;

  // std::map<int, MappointPtr> matched_mappoints;
  std::vector<MappointPtr> matched_mappoints(frame->FeatureNum(), nullptr);
  for(const auto& match : relocalization_matches){
    MappointPtr mpt = relocalization_frame->GetMappoint(match.trainIdx);
    matched_mappoints[match.queryIdx] = mpt;
  }

  std::vector<int> cv_inliers;
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  int num_inliers = SolvePnPWithCV(frame, matched_mappoints, Twc, cv_inliers);
  frame->SetPose(Twc);
  pose = Twc;

  if(_configs.pose_refinement){
    // pose estimation
    MapOfPoses poses; 
    MapOfPoints3d points;
    MapOfVelocity velocities;
    MapOfBias biases;
    std::vector<CameraPtr> camera_list;
    VectorOfMonoPointConstraints mono_point_constraints;
    VectorOfStereoPointConstraints stereo_point_constraints;
    VectorOfIMUConstraints imu_constraints;
    Eigen::Matrix3d Rwg = _map->GetRwg();

    camera_list.emplace_back(_camera);
    AddFrameVertex(frame, poses, 0, false);

    std::vector<size_t> mono_indexes;
    std::vector<size_t> stereo_indexes;
    for(std::size_t i = 0; i < matched_mappoints.size(); ++i){
      int keypoint_idx = i;
      MappointPtr mpt = matched_mappoints[i];
      if(mpt == nullptr || !mpt->IsValid()) continue;
      Eigen::Vector3d keypoint; 
      if(!frame->GetKeypointPosition(keypoint_idx, keypoint)) continue;

      int mpt_id = mpt->GetId();
      Position3d point;
      point.p = mpt->GetPosition();
      point.fixed = true;
      points.insert(std::pair<int, Position3d>(mpt_id, point));

      (AddPointConstraint(frame_id, mpt_id, keypoint, mono_point_constraints, stereo_point_constraints) ? stereo_indexes : mono_indexes).push_back(keypoint_idx);
    }

    if(points.size() < _configs.min_inlier) return false;

    num_inliers = FrameOptimization(poses, points, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, imu_constraints, Rwg,
      _configs.pose_estimation_config);

    pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = poses.begin()->second.R;
    pose.block<3, 1>(0, 3) = poses.begin()->second.p;
  }


  if(num_inliers < _configs.min_inlier) return false;

  // visualization
  if(_configs.ros_publisher_config.reloc){
    FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);
    frame_pose_message->time = now;
    frame_pose_message->pose = pose;

    _reloc_message->times.push_back(now);
    _reloc_message->poses.push_back(pose);
    _reloc_message->mappoints.clear();
    _reloc_message->mappoints.reserve(num_inliers);
    for(std::size_t i = 0; i < matched_mappoints.size(); ++i){
      if(i < cv_inliers.size() && cv_inliers[i] >= 0){
        _reloc_message->mappoints.push_back(matched_mappoints[i]->GetPosition()); 
      }
    }

    _publisher->PublishFramePose(frame_pose_message);
    _publisher->PubRelocResults(_reloc_message);
  }

  return true;
}

Eigen::Matrix4d MapUser::GetBaseFramePose(){
  return _map->_keyframes.begin()->second->GetPose();
}

double MapUser::GetBaseFrameTimestamp(){
  return _map->_keyframes.begin()->second->GetTimestamp();
}
