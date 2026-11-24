#ifndef MAP_REFINER_H_
#define MAP_REFINER_H_

#include <iostream>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "super_point.h"
#include "read_configs.h"
#include "imu.h"
#include "utils.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "map.h"
#include "publisher.h"
#include "g2o_optimization/types.h"
#include "bow/database.h"

struct LoopGroupCandidate{
  LoopGroupCandidate(): group_score(0) {}

  std::set<FramePtr> group_frames;
  double group_score;
};

struct LoopFramePair{
  LoopFramePair() {}

  FramePtr query_frame;
  FramePtr loop_frame;
  Eigen::Matrix3d Rlq;
  Eigen::Vector3d tlq;
  int inlier_num = 0;
  std::vector<std::pair<MappointPtr, MappointPtr>> mappoint_pairs;
};


class MapRefiner{
public:
  MapRefiner();
  MapRefiner(MapRefinementConfigs& configs, PublisherPtr publisher = nullptr);

  void LoadMap(const std::string& map_root);

  // for loop closure
  void LoadVocabulary(const std::string voc_path);

  void UpdateCovisibilityGraph();

  int LoopDetection();
  void LoopDetection(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector);
  void RelativatePoseEstimation(FramePtr frame, DBoW2::WordIdToFeatures& word_features, 
      FramePtr loop_frame, std::vector<cv::DMatch>& loop_matches, std::map<FramePtr, LoopGroupCandidate>& group_candidates);
  void FilterLoopGroups();

  void PoseGraphRefinement();

  void MergeMap();
  void MergeMappoints();
  int MergeMappointGroup(const std::set<int>& mappoint_group);
  // structure_only: all keyframe poses fixed, only the map points are optimized (points re-solved on the poses first)
  bool GlobalMapOptimization(bool use_leg_odom = true, bool structure_only = false);
  // After a rejected global BA: save the last healthy map state instead of nothing.
  void AcceptLastHealthyMap() { _refinement_valid = true; }
  // Loops accepted by the pose graph: "loop_id query_id t_loop t_query tx ty tz qx qy qz qw" (T_loop_query).
  void SaveLoops(const std::string& path) const;
  // Moves the keyframes (and their map points rigidly with their first observer) to the poses of a TUM trajectory,
  // matched by timestamp (6 ms); unmatched keyframes take the correction of the nearest matched one. Returns matches.
  int InitFromTrajectory(const std::string& path);
  const LegOdomConfig& GetLegOdomConfig() const { return _configs.leg_odom_config; }

  // for saving
  void SaveTrajectory(std::string save_path);
  void SaveFinalMap(std::string map_root);

  // for visualization
  void PubMap();
  void StopVisualization();
  void Wait(int breakpoint);

private:
  // tmp
  double odometry_length;
  std::vector<LoopFramePair> loop_frame_pairs;
  struct AcceptedLoop { int loop_id, query_id; Eigen::Matrix3d R; Eigen::Vector3d t; EIGEN_MAKE_ALIGNED_OPERATOR_NEW };
  std::vector<AcceptedLoop, Eigen::aligned_allocator<AcceptedLoop>> accepted_loops;
  std::map<MappointPtr, std::set<MappointPtr>> merged_mappoints;


private:
  // class
  MapRefinementConfigs _configs;
  std::string _map_root;
  PointMatcherPtr _point_matcher;
  std::pair<int, int> _point_matcher_size{0, 0};   // image size the matcher was built with (camera of the map)
  MapPtr _map;
  CameraPtr _camera;

  // for loop closure
  DatabasePtr _database;

  // for visualization
  PublisherPtr _publisher;
  std::mutex _map_mutex;
  bool _stop;
  bool _stopped;
  bool _map_ready;
  bool _refinement_valid = true;
  std::thread _visualization_thread;
};

#endif  // MAP_REFINER_H_
