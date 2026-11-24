#include <stdexcept>
#include <iomanip>
#include "map_builder.h"

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream> 
#include <numeric>
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
#include "runtime_utils.h"

MapRefiner::MapRefiner(){
}

MapRefiner::MapRefiner(MapRefinementConfigs& configs, PublisherPtr publisher): odometry_length(0), 
    _configs(configs), _stop(false), _stopped(false), _map_ready(false),
    _refinement_valid(true){
  // the point matcher is created in LoadMap: its image size comes from the map's camera
  _publisher = publisher ? std::move(publisher) : MakeNullPublisher();
  _visualization_thread = std::thread(boost::bind(&MapRefiner::PubMap, this));
}

void MapRefiner::LoadMap(const std::string& map_root){
  _map_mutex.lock();
  std::string map_v0_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv0.bin");

  std::cout << "map_v0_path = " << map_v0_path << std::endl;

  _map = std::shared_ptr<Map>(new Map());
  _map_root = map_root;
  std::ifstream ifs(map_v0_path, std::ios::binary);
  if(!ifs){
    _map_mutex.unlock();
    throw std::runtime_error("Cannot open map file: " + map_v0_path);
  }
  boost::archive::binary_iarchive ia(ifs);
  ia >> _map;
  _database = _map->_database;  // Preserve the prepared place-recognition index across BA rounds.

  _map->SetIMUInit(false);

  _map->SetPublisher(_publisher);
  _camera = _map->GetCameraPtr();
  const PointMatcherConfig matcher_config = _configs.point_matcher_config.WithImageSize(
      static_cast<int>(_camera->ImageWidth()), static_cast<int>(_camera->ImageHeight()));
  if(!_point_matcher || _point_matcher_size != std::make_pair(matcher_config.image_width, matcher_config.image_height)){
    _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(matcher_config));
    _point_matcher_size = {matcher_config.image_width, matcher_config.image_height};
  }
  _map_ready = true;

  _map->CheckMap();
  _map_mutex.unlock();
}
  
void MapRefiner::LoadVocabulary(const std::string voc_path){
  _database = std::shared_ptr<Database>(new Database(voc_path));
}

void MapRefiner::UpdateCovisibilityGraph(){
  _map_mutex.lock();
  _map->UpdateCovisibilityGraph();
  _map_mutex.unlock();
}

int MapRefiner::LoopDetection(){
  odometry_length = 0.0;
  loop_frame_pairs.clear();
  merged_mappoints.clear();

  Eigen::Vector3d last_position, current_position;
  int num_frame = 0;
  _map_mutex.lock();
  for(const auto& kv : _map->_keyframes){
    FramePtr frame = kv.second;
    if(num_frame == 0){
      last_position = frame->GetPose().block<3, 1>(0, 3);
      num_frame++;
    }else{
      current_position = frame->GetPose().block<3, 1>(0, 3);
      double distance = (current_position - last_position).norm();
      odometry_length += distance;

      last_position = current_position;
      num_frame++;
    }

    DBoW2::WordIdToFeatures word_features; 
    DBoW2::BowVector bow_vector;
    std::vector<DBoW2::WordId> word_of_features;
    _database->FrameToBow(frame, word_features, bow_vector, word_of_features);
    LoopDetection(frame, word_features, bow_vector);
    _database->AddFrame(frame, word_features, bow_vector);
  }
  FilterLoopGroups();
  _map_mutex.unlock();
  return loop_frame_pairs.size();
}

void MapRefiner::LoopDetection(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
  int frame_id = frame->GetFrameId();
  // query
  std::map<FramePtr, int> frame_sharing_words;
  _database->Query(bow_vector, frame_sharing_words);

  if(frame_sharing_words.empty()) return;

  // filtration
  int max_sharing_words = 0;
  for(const auto& kv : frame_sharing_words){
    max_sharing_words = kv.second > max_sharing_words ? kv.second : max_sharing_words;
  }
  int sharing_wrods_num_thr = std::max(static_cast<int>(max_sharing_words * 0.5f), 8);

  std::map<FramePtr, int> covi_frames;
  _map->GetConnectedFrames(frame, covi_frames);
  std::map<FramePtr, int>::iterator fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end();){
    FramePtr fsw = fsw_it->first;
    if(fsw->GetFrameId() >= frame_id || fsw_it->second < sharing_wrods_num_thr || covi_frames.count(fsw)){
      fsw_it = frame_sharing_words.erase(fsw_it);
    }else{
      fsw_it++;
    }
  }

  if(frame_sharing_words.empty()) return;

  // scoring
  std::map<FramePtr, double> frame_scores;
  fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end(); fsw_it++){
    FramePtr fsw = fsw_it->first;
    frame_scores[fsw] = _database->Score(_database->_frame_bow_vectors[fsw], bow_vector);
  }

  // grouping
  std::map<FramePtr, LoopGroupCandidate> group_candidates;
  FramePtr best_deputy;
  double best_group_score = -1; 
  std::map<FramePtr, LoopGroupCandidate>::iterator loop_group_iter;
  std::map<FramePtr, double>::iterator fs_it = frame_scores.begin();
  for(; fs_it != frame_scores.end(); fs_it++){
    FramePtr fsw = fs_it->first;
    FramePtr deputy_of_group = fsw;
    double deputy_score = fs_it->second;

    LoopGroupCandidate group_candidate;
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

    loop_group_iter = group_candidates.find(deputy_of_group);
    if(loop_group_iter == group_candidates.end() || loop_group_iter->second.group_score < group_candidate.group_score){
      group_candidates[deputy_of_group] = group_candidate;

      if(group_candidate.group_score > best_group_score){
        best_group_score = group_candidate.group_score;
        best_deputy = deputy_of_group;
      }
    }
  }

  if(best_group_score < 0) return;

  // filtration of group by geometric verification
  Eigen::Vector3d current_position = frame->GetPose().block<3, 1>(0, 3);
  Eigen::Vector3d loop_position;
  double loop_distance_thr = odometry_length * 0.03;
  loop_group_iter = group_candidates.begin();
  for(; loop_group_iter != group_candidates.end();){
    FramePtr fsw = loop_group_iter->first;
    loop_position = fsw->GetPose().block<3, 1>(0, 3);
    double distance =  (current_position - loop_position).norm();
    if(distance > loop_distance_thr){  
      loop_group_iter = group_candidates.erase(loop_group_iter);
    }else{
      loop_group_iter++;
    }

  }

  // filtration of group by group score
  if(group_candidates.size() > 3){
    double group_score_thr = best_group_score * 0.5;
    loop_group_iter = group_candidates.begin();
    for(; loop_group_iter != group_candidates.end();){
      if(loop_group_iter->second.group_score < group_score_thr){  
        loop_group_iter = group_candidates.erase(loop_group_iter);
      }else{
        loop_group_iter++;
      }
    }
  }

  // sorting of group
  std::vector<std::pair<FramePtr, LoopGroupCandidate>> group_vector(group_candidates.begin(), group_candidates.end());
  std::sort(group_vector.begin(), group_vector.end(), [](const auto &a, const auto &b) {
      return a.second.group_score > b.second.group_score;
  });

  // feature matching
  const int GoodCandidateNum = group_vector.size() <= 5 ? group_vector.size() : 5; 
  std::vector<cv::DMatch> best_matches;
  FramePtr best_candidate;
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = frame->GetAllFeatures();
  for(int i = 0; i < GoodCandidateNum; i++){
    FramePtr good_candidate = group_vector[i].first;
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& good_candidate_features = good_candidate->GetAllFeatures();
    std::vector<cv::DMatch> matches;
    _point_matcher->MatchingPoints(query_features, good_candidate_features, matches, true);
    // if(matches.size() > 50){
    //   RelativatePoseEstimation(frame, word_features, good_candidate, matches, group_candidates);
    // }
    if(matches.size() > best_matches.size()){
      best_matches = matches;
      best_candidate = good_candidate;
    }
  }

  if(best_matches.size() > 50){
    RelativatePoseEstimation(frame, word_features, best_candidate, best_matches, group_candidates);
  }
}

void MapRefiner::RelativatePoseEstimation(FramePtr frame, DBoW2::WordIdToFeatures& word_features, 
    FramePtr loop_frame, std::vector<cv::DMatch>& loop_matches, std::map<FramePtr, LoopGroupCandidate>& group_candidates){
  int frame_id = frame->GetFrameId();

  std::vector<MappointPtr> matched_mappoints(frame->FeatureNum(), nullptr);
  for(const auto& match : loop_matches){
    MappointPtr mpt = loop_frame->GetMappoint(match.trainIdx);
    matched_mappoints[match.queryIdx] = mpt;
  }

  int valid_match_num = 0;
  for(const MappointPtr& mpt : matched_mappoints){
    if(mpt && mpt->IsValid()) valid_match_num++;
  }

  Eigen::Matrix4d pnp_pose = Eigen::Matrix4d::Identity();
  std::vector<int> pnp_inliers;
  const int pnp_inlier_num = SolvePnPWithCV(
      frame, matched_mappoints, pnp_pose, pnp_inliers, 3.0);
  const double pnp_inlier_ratio = valid_match_num > 0
      ? static_cast<double>(pnp_inlier_num) / valid_match_num : 0.0;

  std::set<int> occupied_cells;
  for(size_t i = 0; i < pnp_inliers.size(); i++){
    if(pnp_inliers[i] < 0) continue;
    Eigen::Vector3d keypoint;
    if(!frame->GetKeypointPosition(i, keypoint)) continue;
    const int grid_x = std::min(3, std::max(0,
        static_cast<int>(4.0 * keypoint(0) / _camera->ImageWidth())));
    const int grid_y = std::min(2, std::max(0,
        static_cast<int>(3.0 * keypoint(1) / _camera->ImageHeight())));
    occupied_cells.insert(grid_y * 4 + grid_x);
  }

  if(pnp_inlier_num < 50 || pnp_inlier_ratio < 0.35 || occupied_cells.size() < 6){
    std::cout << "Reject loop: query=" << frame_id
              << ", loop=" << loop_frame->GetFrameId()
              << ", matches=" << valid_match_num
              << ", pnp_inliers=" << pnp_inlier_num
              << ", ratio=" << pnp_inlier_ratio
              << ", occupied_cells=" << occupied_cells.size() << std::endl;
    return;
  }

  for(size_t i = 0; i < matched_mappoints.size(); i++){
    if(i >= pnp_inliers.size() || pnp_inliers[i] < 0){
      matched_mappoints[i] = nullptr;
    }
  }

  // initial pose estimation
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
  poses[frame_id].R = pnp_pose.block<3, 3>(0, 0);
  poses[frame_id].p = pnp_pose.block<3, 1>(0, 3);

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
  if(points.size() < 50) return;

  int num_inliers = FrameOptimization(poses, points, velocities, biases, camera_list,
    mono_point_constraints, stereo_point_constraints, imu_constraints, Rwg,
    _configs.map_optimization_config);

  if(num_inliers < 50){
    std::cout << "Reject loop after pose refinement: query=" << frame_id
              << ", loop=" << loop_frame->GetFrameId()
              << ", refined_inliers=" << num_inliers << std::endl;
    return;
  }

  std::vector<bool> outlier_mappoints(frame->FeatureNum(), false);
  for(size_t i = 0; i < mono_point_constraints.size(); i++){
    if(!mono_point_constraints[i]->inlier){
      outlier_mappoints[mono_indexes[i]] = true;
    }
  }
  for(size_t i = 0; i < stereo_point_constraints.size(); i++){
    if(!stereo_point_constraints[i]->inlier){
      outlier_mappoints[stereo_indexes[i]] = true;
    }
  }

  // find more matches
  int loop_frame_id = loop_frame->GetFrameId();
  std::set<FramePtr> loop_group_frames = group_candidates[loop_frame].group_frames;
  loop_group_frames.erase(loop_frame);

  Eigen::Matrix3d Rwq = poses.begin()->second.R;
  Eigen::Vector3d twq = poses.begin()->second.p;
  Eigen::Matrix4d Twl = loop_frame->GetPose();
  Eigen::Matrix3d Rwl = Twl.block<3, 3>(0, 0);
  Eigen::Vector3d twl = Twl.block<3, 1>(0, 3);
  Eigen::Matrix3d Rlq = Rwl.transpose() * Rwq;
  Eigen::Vector3d tlq = Rwl.transpose() * (twq - twl);

  std::function<bool(const int&, const MappointPtr)> check_reprojection_error = [&](const int& idx, const MappointPtr mpt){
    if(!mpt || !mpt->IsValid()) return false;
    Eigen::Vector3d p_on_query_frame;
    if(!frame->GetKeypointPosition(idx, p_on_query_frame)) return false;

    Eigen::Vector3d pw = mpt->GetPosition();
    Eigen::Vector3d pc = Rwq.transpose() * (pw - twq);
    Eigen::Vector3d p_2d;
    if(!_camera->StereoProject(p_2d, pc)) return false;

    Eigen::Vector3d diff = p_on_query_frame - p_2d;
    if(p_on_query_frame(2) < 0){
      double chi2 = diff.head<2>().transpose() * diff.head<2>();
      return (chi2 < _configs.map_optimization_config.mono_point);
    }else{
      double chi2 = diff.transpose() * diff;
      return (chi2 < _configs.map_optimization_config.stereo_point);
    }
  };

  std::function<MappointPtr(const int&, const DBoW2::WordId&)> find_more_matches_in_group = [&](const int& idx, const DBoW2::WordId& word_id){
    
    Eigen::Matrix<float, 256, 1> query_descriptor, candidate_descriptor;
    if(!frame->GetDescriptor(idx, query_descriptor)){
      return std::shared_ptr<Mappoint>(nullptr);
    }
    FramePtr best_match_frame;
    int best_match_idx = -1;
    float best_distance = 5;

    for(const auto& kv : _database->_inverted_file[word_id]){
      FramePtr f = kv.first;
      if(loop_group_frames.find(f) == loop_group_frames.end()) continue;
      for(const int& macth_candidate_idx : kv.second){
        if(!f->GetDescriptor(macth_candidate_idx, candidate_descriptor)) continue;
        float distance = DescriptorDistance(query_descriptor, candidate_descriptor);
        if(distance < best_distance){
          best_distance = distance;
          best_match_frame = f;
          best_match_idx = macth_candidate_idx;
        }
      }
    }
    if(best_match_idx < 0){
      return std::shared_ptr<Mappoint>(nullptr);
    }

    MappointPtr bast_match_mpt = best_match_frame->GetMappoint(best_match_idx);
    if(check_reprojection_error(idx, bast_match_mpt)){
      return bast_match_mpt;
    }
    return std::shared_ptr<Mappoint>(nullptr);
  };

  for(const auto& kv : word_features){
    const DBoW2::WordId word_id = kv.first;
    for(const auto& idx : kv.second){
      MappointPtr mpt = matched_mappoints[idx];
      if(!mpt || mpt->IsBad() || outlier_mappoints[idx]){ // find match from other frames in the loop group
        MappointPtr match_mpt = find_more_matches_in_group(idx, word_id);
        matched_mappoints[idx] = match_mpt;
      }else if(!mpt->IsValid()){
        // Loop detection must not mutate observations before the pose graph has
        // passed validation. Try another already-triangulated group point.
        matched_mappoints[idx] = find_more_matches_in_group(idx, word_id);
      }
    }
  }
  //*/

  // merge mappoints
  // save loop results
  LoopFramePair loop_frame_pair;
  loop_frame_pair.query_frame = frame;
  loop_frame_pair.loop_frame = loop_frame;
  loop_frame_pair.Rlq = Rlq;
  loop_frame_pair.tlq = tlq;
  loop_frame_pair.inlier_num = num_inliers;

  std::cout << "Accept loop: query=" << frame_id
            << ", loop=" << loop_frame_id
            << ", matches=" << valid_match_num
            << ", pnp_inliers=" << pnp_inlier_num
            << ", refined_inliers=" << num_inliers
            << ", ratio=" << pnp_inlier_ratio
            << ", occupied_cells=" << occupied_cells.size()
            << ", translation=" << tlq.norm() << std::endl;

  for(size_t i = 0; i < matched_mappoints.size(); i++){
    if(!matched_mappoints[i] || !matched_mappoints[i]->IsValid()) continue;
    MappointPtr mpt_on_query_frame = frame->GetMappoint(i);
    if(!mpt_on_query_frame || !mpt_on_query_frame->IsValid()) continue;
    if(mpt_on_query_frame == matched_mappoints[i]) continue;
    loop_frame_pair.mappoint_pairs.emplace_back(
        mpt_on_query_frame, matched_mappoints[i]);
  }
  loop_frame_pairs.emplace_back(std::move(loop_frame_pair));
}

void MapRefiner::FilterLoopGroups(){
  const int loop_num = static_cast<int>(loop_frame_pairs.size());
  if(loop_num == 0) return;

  const auto& cfg = _configs.loop_verification_config;
  std::vector<int> parent(loop_num);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&](int index){
    int root = index;
    while(parent[root] != root) root = parent[root];
    while(parent[index] != index){
      const int next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };
  auto unite = [&](int lhs, int rhs){
    lhs = find_root(lhs);
    rhs = find_root(rhs);
    if(lhs != rhs) parent[rhs] = lhs;
  };

  // Connected components of loop-query frames in the covisibility graph form
  // independent loop events.
  std::map<FramePtr, int> query_to_loop;
  for(int i = 0; i < loop_num; ++i){
    query_to_loop[loop_frame_pairs[i].query_frame] = i;
  }
  for(int i = 0; i < loop_num; ++i){
    std::map<FramePtr, int> connected_frames;
    _map->GetConnectedFrames(loop_frame_pairs[i].query_frame, connected_frames);
    for(const auto& connected : connected_frames){
      auto loop_it = query_to_loop.find(connected.first);
      if(loop_it != query_to_loop.end() &&
          connected.second >= cfg.min_covisibility_weight){
        unite(i, loop_it->second);
      }
    }
  }

  std::map<int, std::vector<int>> groups;
  for(int i = 0; i < loop_num; ++i) groups[find_root(i)].push_back(i);

  auto make_transform = [](const Eigen::Matrix3d& rotation,
                           const Eigen::Vector3d& translation){
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    return transform;
  };
  auto relative_pose = [](const FramePtr& first, const FramePtr& second){
    return first->GetPose().inverse() * second->GetPose();
  };
  auto rotation_angle = [](const Eigen::Matrix4d& transform){
    return std::abs(Eigen::AngleAxisd(
        transform.block<3, 3>(0, 0)).angle());
  };
  auto consistent = [&](int first_index, int second_index){
    const LoopFramePair& first = loop_frame_pairs[first_index];
    const LoopFramePair& second = loop_frame_pairs[second_index];
    const Eigen::Matrix4d K12 = relative_pose(
        first.query_frame, second.query_frame);
    const Eigen::Matrix4d L12 = relative_pose(
        first.loop_frame, second.loop_frame);
    const Eigen::Matrix4d T11 = make_transform(first.Rlq, first.tlq);
    const Eigen::Matrix4d T22 = make_transform(second.Rlq, second.tlq);
    const Eigen::Matrix4d error =
        T11 * K12 * T22.inverse() * L12.inverse();

    const double translation_threshold = std::max({
        cfg.translation_ratio * K12.block<3, 1>(0, 3).norm(),
        cfg.translation_ratio * L12.block<3, 1>(0, 3).norm(),
        cfg.min_translation_threshold});
    const double rotation_threshold = std::max({
        cfg.rotation_ratio * rotation_angle(K12),
        cfg.rotation_ratio * rotation_angle(L12),
        cfg.min_rotation_threshold_deg * M_PI / 180.0});
    return error.block<3, 1>(0, 3).norm() <= translation_threshold &&
        rotation_angle(error) <= rotation_threshold;
  };

  // Odometry poses (camera in world, TUM) for resolving conflicting loop clusters.
  TumTrajectory odometry;
  if(!cfg.odometry_poses.empty()){
    const std::string path = ConcatenateFolderAndFileName(_map_root, cfg.odometry_poses);
    odometry = ReadTumTrajectory(path);
    std::cout << "Loop verification odometry poses: " << odometry.size() << " from " << path << std::endl;
  }
  auto odometry_pose = [&](const FramePtr& frame, Eigen::Matrix4d& pose){
    const auto* p = NearestPose(odometry, frame->GetTimestamp(), 0.02);
    if(!p) return false;
    pose = p->second;
    return true;
  };
  // |p(query) - p(loop) ⊕ T_lq| in odometry poses: the correction the loop asks of the odometry trajectory.
  std::vector<double> correction(loop_num, -1.0);
  for(int i = 0; i < loop_num; ++i){
    Eigen::Matrix4d pq, pl;
    const LoopFramePair& loop = loop_frame_pairs[i];
    if(!odometry_pose(loop.query_frame, pq) || !odometry_pose(loop.loop_frame, pl)) continue;
    correction[i] = (pq.block<3, 1>(0, 3) - (pl * make_transform(loop.Rlq, loop.tlq)).block<3, 1>(0, 3)).norm();
    std::cout << "Loop correction: query=" << loop.query_frame->GetFrameId()
              << ", loop=" << loop.loop_frame->GetFrameId() << ", correction=" << correction[i] << std::endl;
  }
  auto median = [](std::vector<double> values){
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
  };

  std::vector<int> retained_indices;
  int group_id = 0;
  for(auto& group_entry : groups){
    std::vector<int> active = group_entry.second;
    const int original_size = static_cast<int>(active.size());
    if(original_size < cfg.min_group_size){
      std::cout << "Reject loop group " << group_id++
                << ": size=" << original_size
                << " < " << cfg.min_group_size << std::endl;
      continue;
    }

    // Perceptual aliasing: the group splits into clusters that are each consistent but contradict each other.
    // Keep the cluster asking the smallest correction of the odometry, unless the choice is ambiguous.
    if(!odometry.empty()){
      std::vector<int> label(active.size(), -1);
      int clusters = 0;
      for(size_t seed = 0; seed < active.size(); ++seed){
        if(label[seed] >= 0) continue;
        std::vector<size_t> stack{seed};
        label[seed] = clusters;
        while(!stack.empty()){
          const size_t i = stack.back(); stack.pop_back();
          for(size_t j = 0; j < active.size(); ++j){
            if(label[j] < 0 && consistent(active[i], active[j])){ label[j] = clusters; stack.push_back(j); }
          }
        }
        ++clusters;
      }
      if(clusters > 1){
        const size_t min_size = std::max<size_t>(2, cfg.min_group_size);
        std::vector<std::pair<double, int>> scores;  // (median correction, cluster)
        for(int c = 0; c < clusters; ++c){
          std::vector<double> values;
          for(size_t i = 0; i < active.size(); ++i)
            if(label[i] == c && correction[active[i]] >= 0) values.push_back(correction[active[i]]);
          const size_t size = std::count(label.begin(), label.end(), c);
          if(!values.empty() && size >= min_size) scores.emplace_back(median(values), c);
          std::cout << "Loop cluster " << c << " of group " << group_id << ": size=" << size
                    << ", median correction=" << (values.empty() ? -1.0 : median(values)) << std::endl;
        }
        std::sort(scores.begin(), scores.end());
        if(scores.size() >= 2 && scores[1].first - scores[0].first < cfg.ambiguity_margin){
          std::cout << "Reject ambiguous loop group " << group_id++ << ": corrections " << scores[0].first
                    << " vs " << scores[1].first << std::endl;
          continue;
        }
        if(!scores.empty()){
          std::vector<int> kept;
          for(size_t i = 0; i < active.size(); ++i) if(label[i] == scores[0].second) kept.push_back(active[i]);
          std::cout << "Keep loop cluster " << scores[0].second << " (correction " << scores[0].first
                    << "), drop " << active.size() - kept.size() << " loops" << std::endl;
          active = kept;
        }
      }
    }

    // Remove the member conflicting with the most peers until the remaining
    // set is pairwise consistent. Prefer keeping the loop with more inliers.
    while(static_cast<int>(active.size()) >= cfg.min_group_size){
      std::vector<int> conflict_count(active.size(), 0);
      int conflict_pairs = 0;
      for(size_t i = 0; i < active.size(); ++i){
        for(size_t j = i + 1; j < active.size(); ++j){
          if(consistent(active[i], active[j])) continue;
          ++conflict_count[i];
          ++conflict_count[j];
          ++conflict_pairs;
        }
      }
      if(conflict_pairs == 0) break;

      size_t worst = 0;
      for(size_t i = 1; i < active.size(); ++i){
        if(conflict_count[i] > conflict_count[worst] ||
            (conflict_count[i] == conflict_count[worst] &&
             loop_frame_pairs[active[i]].inlier_num <
                 loop_frame_pairs[active[worst]].inlier_num)){
          worst = i;
        }
      }
      const LoopFramePair& rejected = loop_frame_pairs[active[worst]];
      std::cout << "Reject inconsistent loop: query="
                << rejected.query_frame->GetFrameId()
                << ", loop=" << rejected.loop_frame->GetFrameId()
                << ", conflicts=" << conflict_count[worst] << std::endl;
      active.erase(active.begin() + worst);
    }

    if(static_cast<int>(active.size()) < cfg.min_group_size){
      std::cout << "Reject loop group " << group_id++
                << " after consistency check: original=" << original_size
                << ", retained=" << active.size() << std::endl;
      continue;
    }

    std::cout << "Accept loop group " << group_id++
              << ": original=" << original_size
              << ", retained=" << active.size() << std::endl;
    retained_indices.insert(retained_indices.end(), active.begin(), active.end());
  }

  std::sort(retained_indices.begin(), retained_indices.end());
  std::vector<LoopFramePair> retained_loops;
  retained_loops.reserve(retained_indices.size());
  merged_mappoints.clear();
  for(const int index : retained_indices){
    retained_loops.emplace_back(std::move(loop_frame_pairs[index]));
    for(const auto& mappoint_pair : retained_loops.back().mappoint_pairs){
      merged_mappoints[mappoint_pair.first].insert(mappoint_pair.second);
    }
  }
  loop_frame_pairs = std::move(retained_loops);
  std::cout << "Loop group verification: detected=" << loop_num
            << ", retained=" << loop_frame_pairs.size() << std::endl;
}

void MapRefiner::PoseGraphRefinement(){
  if(_map->_mappoints.size() < 800) return; // only for large map

  std::lock_guard<std::mutex> lock(_map_mutex);
  if(_map->_keyframes.size() < 2 || loop_frame_pairs.empty()) return;

  // 1. add frame vertexes and constraints between adjacent frames
  std::vector<CameraPtr> camera_list;
  camera_list.emplace_back(_camera);

  MapOfPoses poses;
  VectorOfRelativePoseConstraints relative_pose_constraints;
  std::map<int, FramePtr>::iterator it_frame = _map->_keyframes.begin();
  const int first_frame_id = it_frame->first;
  std::map<int, FramePtr>::iterator it_next_frame = it_frame;
  it_next_frame++;
  std::map<int, FramePtr>::iterator it_frame_end = _map->_keyframes.end();
  for(;it_next_frame != it_frame_end; it_frame++, it_next_frame++){
    const int frame_id = it_frame->first;
    FramePtr frame = it_frame->second;
    bool fix_this_frame = (frame_id == first_frame_id);
    AddFrameVertex(frame, poses, 0, fix_this_frame);

    Eigen::Matrix4d Twc1 = frame->GetPose();
    Eigen::Matrix4d Twc2 = it_next_frame->second->GetPose();
    Eigen::Matrix3d Rwc1 = Twc1.block<3, 3>(0, 0);
    Eigen::Vector3d twc1 = Twc1.block<3, 1>(0, 3);
    Eigen::Matrix3d Rwc2 = Twc2.block<3, 3>(0, 0);
    Eigen::Vector3d twc2 = Twc2.block<3, 1>(0, 3);
    Eigen::Matrix3d Rc1c2 = Rwc1.transpose() * Rwc2;
    Eigen::Vector3d tc1c2 = Rwc1.transpose() * (twc2 - twc1);

    RelativePoseConstraintPtr rpc = std::shared_ptr<RelativePoseConstraint>(new RelativePoseConstraint());
    rpc->id_pose1 = frame_id;
    rpc->id_pose2 = it_next_frame->first;
    rpc->id_camera1 = 0;
    rpc->id_camera2 = 0;
    rpc->Rc1c2 = Rc1c2;
    rpc->tc1c2 = tc1c2;
    rpc->information_scale = 100.0;
    relative_pose_constraints.push_back(rpc);
  }
  AddFrameVertex(it_frame->second, poses, 0, 0);

  // 2. add loop constraints
  for(const LoopFramePair& loop_frame_pair : loop_frame_pairs){
    RelativePoseConstraintPtr rpc = std::shared_ptr<RelativePoseConstraint>(new RelativePoseConstraint());
    rpc->id_pose1 = loop_frame_pair.loop_frame->GetFrameId();
    rpc->id_pose2 = loop_frame_pair.query_frame->GetFrameId();
    rpc->id_camera1 = 0;
    rpc->id_camera2 = 0;
    rpc->Rc1c2 = loop_frame_pair.Rlq;
    rpc->tc1c2 = loop_frame_pair.tlq;
    rpc->robust = true;
    rpc->information_scale = 1.0;
    relative_pose_constraints.push_back(rpc);
  }

  PoseGraphOptimization(poses, camera_list, relative_pose_constraints);

  bool rejected_any_loop = false;
  for(const RelativePoseConstraintPtr& rpc : relative_pose_constraints){
    if(rpc->robust && !rpc->active){
      rejected_any_loop = true;
      break;
    }
  }
  if(rejected_any_loop){
    // Merge proposals currently share one container and do not retain their
    // originating loop. Discard all proposals rather than risk applying a
    // proposal created by a rejected loop edge.
    merged_mappoints.clear();
    std::cout << "Mappoint merging disabled because at least one loop edge was rejected."
              << std::endl;
  }

  bool pose_graph_healthy = true;
  it_frame = _map->_keyframes.begin();
  it_next_frame = it_frame;
  it_next_frame++;
  for(; it_next_frame != _map->_keyframes.end(); it_frame++, it_next_frame++){
    const Eigen::Matrix4d& old_pose1 = it_frame->second->GetPose();
    const Eigen::Matrix4d& old_pose2 = it_next_frame->second->GetPose();
    const double old_step = (old_pose2.block<3, 1>(0, 3) -
        old_pose1.block<3, 1>(0, 3)).norm();

    const Pose3d& new_pose1 = poses.at(it_frame->first);
    const Pose3d& new_pose2 = poses.at(it_next_frame->first);
    const double new_step = (new_pose2.p - new_pose1.p).norm();
    const double step_limit = std::max(2.0, old_step * 3.0);

    const Eigen::Matrix3d old_relative_rotation =
        old_pose1.block<3, 3>(0, 0).transpose() * old_pose2.block<3, 3>(0, 0);
    const Eigen::Matrix3d new_relative_rotation =
        new_pose1.R.transpose() * new_pose2.R;
    Eigen::AngleAxisd rotation_difference(
        old_relative_rotation.transpose() * new_relative_rotation);
    const double rotation_change = std::abs(rotation_difference.angle());

    if(!std::isfinite(new_step) || new_step > step_limit ||
        !std::isfinite(rotation_change) || rotation_change > 0.523599){
      std::cout << "Reject pose graph result: adjacent frames "
                << it_frame->first << " -> " << it_next_frame->first
                << ", old_step=" << old_step
                << ", new_step=" << new_step
                << ", rotation_change=" << rotation_change << std::endl;
      pose_graph_healthy = false;
      break;
    }
  }

  if(!pose_graph_healthy){
    loop_frame_pairs.clear();
    merged_mappoints.clear();
    return;
  }
  accepted_loops.clear();
  for(const RelativePoseConstraintPtr& rpc : relative_pose_constraints){
    if(rpc->robust && rpc->active) accepted_loops.push_back({rpc->id_pose1, rpc->id_pose2, rpc->Rc1c2, rpc->tc1c2});
  }

  std::map<int, Eigen::Matrix4d> corrections;  // keyframe id -> new pose * old pose^-1
  // update frame poses
  for(auto& kv : poses){
    int frame_id = kv.first;
    FramePtr frame = _map->_keyframes[frame_id];
    Pose3d pose = kv.second;
    if(pose.fixed) continue;
    Eigen::Matrix4d pose_eigen = Eigen::Matrix4d::Identity();
    pose_eigen.block<3, 3>(0, 0) = pose.R;
    pose_eigen.block<3, 1>(0, 3) = pose.p;

    Eigen::Matrix4d Two = frame->GetPose();
    Eigen::Matrix3d Rwo = Two.block<3, 3>(0, 0);
    Eigen::Vector3d two = Two.block<3, 1>(0, 3);

    // to check again
    Eigen::Matrix3d tr = pose.R * Rwo.transpose();
    Eigen::Vector3d tt = -tr * two + pose.p;
    Eigen::Matrix4d correction = Eigen::Matrix4d::Identity();
    correction.topLeftCorner<3, 3>() = tr;
    correction.topRightCorner<3, 1>() = tt;
    corrections[frame_id] = correction;
    frame->SetPose(pose_eigen);
  }

  // update mappoint position
  for(auto& kv : _map->_mappoints){
    MappointPtr mpt = kv.second;
    if(mpt && mpt->IsValid()) MoveWithFirstCorrectedObserver(mpt, corrections);
  }

}

void MapRefiner::MergeMap(){
  std::lock_guard<std::mutex> lock(_map_mutex);
  MergeMappoints();
  merged_mappoints.clear();
  if(!GlobalBA(_map, _configs.map_optimization_config, false, 30, 0)){
    std::cout << "Initial global BA was rejected by the map health check."
              << std::endl;
  }
}

void MapRefiner::MergeMappoints(){
  if(merged_mappoints.empty()){
    return;
  }

  for(auto& kv : merged_mappoints){
    kv.second.insert(kv.first);
  }

  // group mappoints to be merged, each group can be merged into one mappoint
  std::map<int, int> mpt_id_to_group_id;
  std::map<int, std::set<int>> mappoint_groups;

  std::map<int, int>::iterator it_group;
  int new_group_id = 0;
  for(const auto& kv : merged_mappoints){
    std::set<int> found_groups;
    for(const MappointPtr& mpt : kv.second){
      it_group = mpt_id_to_group_id.find(mpt->GetId());
      if(it_group != mpt_id_to_group_id.end()){
        found_groups.insert(it_group->second);
      }
    }

    int best_group_id = -1;
    if(found_groups.size() == 0){
      best_group_id = new_group_id;
      new_group_id++;
    }else if(found_groups.size() == 1){
      best_group_id = *(found_groups.begin());
    }else{
      best_group_id = *(found_groups.begin());
      for(auto group_id : found_groups){
        if(group_id == best_group_id) continue;

        for(int mpt_in_group : mappoint_groups[group_id]){
          mpt_id_to_group_id[mpt_in_group] = best_group_id;
          mappoint_groups[best_group_id].insert(mpt_in_group);
        }
        mappoint_groups.erase(group_id);
      }
    }

    for(const MappointPtr& mpt : kv.second){
      int mpt_id = mpt->GetId();
      mpt_id_to_group_id[mpt_id] = best_group_id;
      mappoint_groups[best_group_id].insert(mpt_id);
    }
  }

  std::cout << "Before merging, mappoint size: " << _map->_mappoints.size() << std::endl;

  // merge mappoint group
  int keep_num = mappoint_groups.size();
  int sum_num = 0;
  int removed_num = 0;
  for(const auto& kv : mappoint_groups){
    sum_num += kv.second.size();
    removed_num += MergeMappointGroup(kv.second);
  }
  std::cout << "After merging, mappoint size: " << _map->_mappoints.size() << std::endl;

  std::cout << "sum_num = " << sum_num << ", keep_num = " << keep_num << std::endl;
  std::cout << "Remove " << removed_num << " validated mappoints." << std::endl;
}

int MapRefiner::MergeMappointGroup(const std::set<int>& mappoint_group){
  if(mappoint_group.size() < 2){
    return 0;
  }

  // 1. Keep the valid point with the largest number of observations.
  int best_mpt_id = -1;
  int best_observer_num = -1;
  for(const int& mpt_id : mappoint_group){
    auto mpt_it = _map->_mappoints.find(mpt_id);
    if(mpt_it == _map->_mappoints.end() || !mpt_it->second ||
        !mpt_it->second->IsValid()) continue;
    const int observer_num = mpt_it->second->ObverserNum();
    if(observer_num > best_observer_num){
      best_mpt_id = mpt_id;
      best_observer_num = observer_num;
    }
  }
  if(best_mpt_id < 0) return 0;

  MappointPtr best_mpt = _map->_mappoints[best_mpt_id];
  std::map<int, int>& obversers_of_best_mpt = best_mpt->GetAllObversers();
  std::map<int, int> claimed_observers = obversers_of_best_mpt;

  auto reprojection_is_valid = [&](const FramePtr& observer, int keypoint_idx,
                                   const Eigen::Vector3d& world_point){
    if(!observer || keypoint_idx < 0) return false;
    Eigen::Vector3d measurement;
    if(!observer->GetKeypointPosition(keypoint_idx, measurement)) return false;
    const Eigen::Matrix4d& Twc = observer->GetPose();
    Eigen::Vector3d camera_point = Twc.block<3, 3>(0, 0).transpose() *
        (world_point - Twc.block<3, 1>(0, 3));
    Eigen::Vector3d projection;
    if(!_camera->StereoProject(projection, camera_point)) return false;
    const Eigen::Vector3d difference = measurement - projection;
    if(measurement(2) < 0){
      return difference.head<2>().squaredNorm() <
          _configs.map_optimization_config.mono_point;
    }
    return difference.squaredNorm() <
        _configs.map_optimization_config.stereo_point;
  };

  std::vector<int> compatible_mappoints;
  for(const int& mpt_id : mappoint_group){
    if(mpt_id == best_mpt_id) continue;
    auto mpt_it = _map->_mappoints.find(mpt_id);
    if(mpt_it == _map->_mappoints.end() || !mpt_it->second ||
        !mpt_it->second->IsValid()) continue;
    MappointPtr mpt = mpt_it->second;

    bool compatible = (best_mpt->GetPosition() - mpt->GetPosition()).norm() <= 1.0;
    const std::map<int, int>& obversers = mpt->GetAllObversers();

    // A physical point cannot occupy two distinct keypoints in one image.
    for(const auto& observer : obversers){
      auto best_observer_it = claimed_observers.find(observer.first);
      if(best_observer_it != claimed_observers.end() &&
          best_observer_it->second != observer.second){
        compatible = false;
        break;
      }
    }

    if(compatible){
      for(const auto& observer : obversers){
        auto frame_it = _map->_keyframes.find(observer.first);
        if(frame_it == _map->_keyframes.end() ||
            !reprojection_is_valid(frame_it->second, observer.second,
                                   best_mpt->GetPosition())){
          compatible = false;
          break;
        }
      }
    }

    if(compatible){
      for(const auto& observer : obversers_of_best_mpt){
        auto frame_it = _map->_keyframes.find(observer.first);
        if(frame_it == _map->_keyframes.end() ||
            !reprojection_is_valid(frame_it->second, observer.second,
                                   mpt->GetPosition())){
          compatible = false;
          break;
        }
      }
    }

    if(compatible){
      compatible_mappoints.push_back(mpt_id);
      for(const auto& observer : obversers){
        claimed_observers[observer.first] = observer.second;
      }
    }
  }

  // 2. Transfer every compatible observation before invalidating old points.
  for(const int& mpt_id : compatible_mappoints){
    MappointPtr mpt = _map->_mappoints[mpt_id];
    const std::map<int, int> obversers = mpt->GetAllObversers();
    for(const auto& kv : obversers){
      int frame_id = kv.first;
      int kpt_id = kv.second;
      if(obversers_of_best_mpt.find(frame_id) == obversers_of_best_mpt.end()){
        best_mpt->AddObverser(frame_id, kpt_id);
        obversers_of_best_mpt[frame_id] = kpt_id;
      }
      auto frame_it = _map->_keyframes.find(frame_id);
      if(frame_it == _map->_keyframes.end()) continue;
      frame_it->second->InsertMappoint(kpt_id, best_mpt);
      frame_it->second->SetTrackId(kpt_id, best_mpt_id);
    }
    mpt->SetBad();
    _map->_mappoints.erase(mpt_id);
  }
  return compatible_mappoints.size();
}

void MapRefiner::SaveTrajectory(std::string save_path){
  if(!_refinement_valid){
    std::cout << "Skip saving refined trajectory because map validation failed."
              << std::endl;
    return;
  }
  _map_mutex.lock();
  _map->SaveKeyframeTrajectory(save_path);
  _map_mutex.unlock();
}

bool MapRefiner::GlobalMapOptimization(bool use_leg_odom, bool structure_only){
  std::lock_guard<std::mutex> lock(_map_mutex);
  LegOdomConfig* leg_cfg = nullptr;
  std::string leg_velocity_path;
  if(use_leg_odom && _configs.leg_odom_config.enabled){
    leg_cfg = &_configs.leg_odom_config;
    leg_velocity_path = ConcatenateFolderAndFileName(
        _map_root, "leg_odom/tmp/network_velocity.csv");
  }
  if(!GlobalBA(_map, _configs.map_optimization_config, true, 50, 40, leg_cfg,
      leg_velocity_path, structure_only)){
    std::cout << "Final global BA was rejected; keeping the last healthy map state."
              << std::endl;
    _refinement_valid = false;
    return false;
  }
  return true;
}

void MapRefiner::SaveFinalMap(std::string map_root){
  if(!_refinement_valid){
    std::cout << "Skip saving AirSLAM_mapv1.bin because map validation failed."
              << std::endl;
    return;
  }
  _map_mutex.lock();

  // Remove invalid map points before saving.
  int invalid_mappoint_num = _map->RemoveInValidMappoints();

  _map->_database = _database;

  std::string map_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv1.bin");
  std::ofstream ofs(map_path, std::ios::binary);
  std::cout << "map_path = " << map_path << std::endl;
  boost::archive::binary_oarchive oa(ofs);
  oa << _map;
  _map->ExportTrainingData(map_root, false);
  _map_mutex.unlock();
}

void MapRefiner::PubMap(){
  while(!_stop){
    _map_mutex.lock();
    if(_map_ready){
      _map->Publish(CurrentUnixTimeSec(), true);
    }
    _map_mutex.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cout << "PubMap is over" << std::endl;
  _stopped = true;
}

void MapRefiner::StopVisualization(){
  _stop = true;
  while(!_stopped){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if(_visualization_thread.joinable()){
    _visualization_thread.join();
  }

  std::cout << "_visualization_thread is stopped" << std::endl;
  _publisher->ShutDown();
}

void MapRefiner::Wait(int breakpoint){
  if(!breakpoint) return;

  std::cout << "Please to press c for the next step..." << std::endl;
  char input;
  while(1){
    std::cin >> input;
    if(input == 'c'){
      break;
    }
  }
}

void MapRefiner::SaveLoops(const std::string& path) const{
  std::ofstream f(path);
  f << std::setprecision(12);
  for(const AcceptedLoop& l : accepted_loops){
    const Eigen::Quaterniond q(l.R);
    auto kl = _map->_keyframes.find(l.loop_id), kq = _map->_keyframes.find(l.query_id);
    if(kl == _map->_keyframes.end() || kq == _map->_keyframes.end()) continue;
    f << l.loop_id << " " << l.query_id << " " << std::fixed << std::setprecision(9) << kl->second->GetTimestamp() << " "
      << kq->second->GetTimestamp() << " " << std::defaultfloat << std::setprecision(12) << l.t.x() << " " << l.t.y() << " " << l.t.z() << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
}

int MapRefiner::InitFromTrajectory(const std::string& path){
  std::lock_guard<std::mutex> lock(_map_mutex);
  const TumTrajectory traj = ReadTumTrajectory(path);
  std::map<int, FramePtr>& keyframes = _map->GetAllKeyframes();
  std::map<int, Eigen::Matrix4d> corrections;
  for(auto& kv : keyframes){
    if(const auto* p = NearestPose(traj, kv.second->GetTimestamp(), 6e-3))
      corrections[kv.first] = p->second * kv.second->GetPose().inverse();
  }
  const int matched = corrections.size();
  if(corrections.empty()) return 0;
  for(auto& kv : keyframes){
    if(corrections.count(kv.first)) continue;
    auto it = corrections.lower_bound(kv.first);
    if(it == corrections.end()) --it;
    else if(it != corrections.begin() && kv.first - std::prev(it)->first < it->first - kv.first) --it;
    corrections[kv.first] = it->second;
  }
  for(auto& kv : _map->GetAllMappoints()){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;
    MoveWithFirstCorrectedObserver(mpt, corrections);
  }
  for(auto& kv : keyframes) kv.second->SetPose(corrections[kv.first] * kv.second->GetPose());
  std::cout << "Keyframes initialised from " << path << ": matched " << matched << "/" << keyframes.size() << std::endl;
  return matched;
}
