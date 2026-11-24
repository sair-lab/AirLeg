#ifndef MAP_H_
#define MAP_H_

#include <algorithm>
#include <opencv2/highgui/highgui.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include "read_configs.h"
#include "camera.h"
#include "mappoint.h"
#include "frame.h"
#include "g2o_optimization/types.h"
#include "publisher.h"
#include "bow/database.h"

// Moves a map point rigidly with the correction of its first (lowest id) observer that has one; false if none has.
// Shared by the back-end write-back, InitFromTrajectory (every keyframe corrected) and PoseGraphRefinement.
bool MoveWithFirstCorrectedObserver(const MappointPtr& mpt, const std::map<int, Eigen::Matrix4d>& corrections);

class MapRefiner;
class OnlineFusion;

class Map{
public:
  Map();
  Map(OptimizationConfig& backend_optimization_config, CameraPtr camera, PublisherPtr publisher);
  void InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);

  void CheckAndDeleteMappoint(MappointPtr mpt);
  void DeleteKeyframe(FramePtr frame);

  CameraPtr GetCameraPtr();
  FramePtr GetFramePtr(int frame_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  void LocalMapOptimization(FramePtr new_frame);
  void ApplyBackendWriteBack();
  void RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers);
  int UpdateFrameTrackIds(int track_id);
  // Re-link after a tracking break: keypoint idx of `frame` observes the existing point. A new point already at
  // idx is merged into it (all observers move) unless an observer would see it at two keypoints. Returns the
  // number of keypoints linked.
  int LinkMappoints(FramePtr frame, const std::vector<std::pair<int, MappointPtr>>& links);

  void SaveKeyframeTrajectory(std::string save_root);

  bool InitializeIMU(FramePtr frame);
  // voxel feature management (voxel_cap): <= max_per_voxel points of a keyframe per 3D voxel
  void SetPointVoxel(double size, int max_per_voxel){ _voxel_size = size; _voxel_max = max_per_voxel; }
  long voxel_skipped = 0;   // new stereo points not created (voxel full)
  void SetRwg(const Eigen::Matrix3d& Rwg);
  Eigen::Matrix3d GetRwg();
  void SetIMUInit(bool imu_init);
  bool IMUInit();

  void ExportTrainingData(const std::string& map_root, bool stereo_fallback = true);
  // VO only (track_state / pose_source are not in the serialized map): leg_odom/keyframe_status.csv, one row per
  // keyframe, timestamp as in the keypoint file names; S2 skips the keyframe pairs touching a lost keyframe.
  void ExportKeyframeStatus(const std::string& map_root);

  void SetPublisher(PublisherPtr publisher);
  // S3: null (default, visual_only) keeps the original local BA unchanged.
  void SetOnlineFusion(std::shared_ptr<OnlineFusion> online){ _online = std::move(online); }
  void Publish(double time, bool clear_old_message = false);
  // Incremental: the keyframes of a local BA window and the points they observe; every 50th call a full
  // refresh (cleared, so removed points disappear).
  void PublishWindow(double time, const std::vector<FramePtr>& window);


  // for offline optimization
  std::map<int, MappointPtr>& GetAllMappoints();
  std::map<int, FramePtr>& GetAllKeyframes();
  int RemoveInValidMappoints();

  void UpdateCovisibilityGraph();
  void UpdateFrameCovisibility(FramePtr frame);

  void GetConnectedFrames(FramePtr frame, std::map<FramePtr, int>& covi_frames);

  // visualization
  double MapScale();

  // debug
  void CheckMap();

public:
  // tmp parameters
  std::vector<std::pair<FramePtr, int>> to_update_track_id;

  // for imu
  FramePtr last_keyframe;

  double imu_init_time;
  FramePtr imu_init_frame;
  int imu_init_stage;  


private:
  int _window_publishes = 0;
  friend class MapRefiner;
  friend class MapUser;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _camera;
    ar & _mappoints;
    ar & _keyframes;
    // ar & _keyframe_ids;
    ar & _imu_init;
    ar & boost::serialization::make_array(_Rwg.data(), _Rwg.size());

    ar & _covisibile_frames;
    ar & _database;
  }

private:
  OptimizationConfig _backend_optimization_config;
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
  PublisherPtr _publisher;

  // for imu
  bool _imu_init;
  double _voxel_size = 0.0;
  int _voxel_max = 0;
  Eigen::Matrix3d _Rwg;

  // S3 online fusion (not serialized)
  std::shared_ptr<OnlineFusion> _online;

  // for loop detection adn relocalization
  std::map<FramePtr, std::map<FramePtr, int>> _covisibile_frames;
  DatabasePtr _database;

};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_
