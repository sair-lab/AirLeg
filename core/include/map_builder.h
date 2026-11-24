#ifndef MAP_BUILDER_H_
#define MAP_BUILDER_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "read_configs.h"
#include "imu.h"
#include "utils.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "feature_detector.h"
#include "map.h"
#include "publisher.h"
#include "g2o_optimization/types.h"
#include "online_fusion.h"
#include "stage_times.h"
#include "imu_attitude.h"

struct InputData{
  size_t index = 0;
  double time = 0.0;
  cv::Mat image_left;
  cv::Mat image_right;
  // dynamic-object masks (non-zero = dynamic), raw image coordinates; empty = none
  cv::Mat mask_left;
  cv::Mat mask_right;
  ImuDataList batch_imu_data;
  bool imu_data_valid = true;

  InputData() {}
  InputData& operator =(InputData& other){
		index = other.index;
		time = other.time;
		image_left = other.image_left.clone();
		image_right = other.image_right.clone();
		mask_left = other.mask_left.clone();
		mask_right = other.mask_right.clone();
		batch_imu_data = other.batch_imu_data;
		imu_data_valid = other.imu_data_valid;
		return *this;
	}
};
typedef std::shared_ptr<InputData> InputDataPtr;

enum FrameType {
  NormalFrame = 0,
  KeyFrame = 1,
  InitializationFrame = 2,
};

struct TrackingData{
  FramePtr frame;
  FrameType frame_type;
  FramePtr ref_keyframe;
  std::vector<cv::DMatch> matches;
  InputDataPtr input_data;

  TrackingData() {}
  TrackingData& operator =(TrackingData& other){
		frame = other.frame;
		ref_keyframe = other.ref_keyframe;
		matches = other.matches;
		input_data = other.input_data;
		return *this;
	}
};
typedef std::shared_ptr<TrackingData> TrackingDataPtr;

class DynamicSegmenter;

class MapBuilder{
public:
  MapBuilder(VisualOdometryConfigs& configs, PublisherPtr publisher = nullptr);
  bool UseIMU();
  bool UseImuData();   // the IMU stream is needed (VIO or the attitude filter, imu_mode > 0)
  void AddInput(InputDataPtr data);
  void ExtractFeatureThread();
  void TrackingThread();

  int TrackFrame(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches, Preinteration& _preinteration);
  // degraded: state machine of the visual-degradation mode (after the keyframe handling of a frame)
  void DegradedUpdate(FramePtr frame, bool keyframe);

  int FramePoseOptimization(FramePtr frame0, FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, 
      Preinteration& preinteration);
  int AddKeyframeCheck(FramePtr ref_keyframe, FramePtr current_frame, const std::vector<cv::DMatch>&);
  void InsertKeyframe(FramePtr frame);

  void PublishFrame(FramePtr frame, cv::Mat& image, FrameType frame_type, std::vector<cv::DMatch>& matches);
  // Pose of a frame whose tracking is lost, by priority leg > IMU > constant velocity (> last pose).
  void MotionModelPose(FramePtr frame0, FramePtr frame1, Preinteration& preinteration, Eigen::Matrix4d* Twc,
                       PoseSource* source);
  // After a break: match the keyframe against the last normal keyframes before it (LightGlue + PnP) and link
  // its observations to their map points. True if linked.
  bool TryRelink(FramePtr keyframe);
  void SaveTrajectory();
  void SaveTrajectory(std::string file_path);
  void SaveMap(const std::string& map_root);

  void Stop();
  bool IsStopped();
  // S3: null in visual_only mode.
  OnlineFusionPtr Online() { return _online; }


private:
  // left feature extraction and tracking thread
  std::mutex _buffer_mutex;
  std::condition_variable _buffer_cv;  // data pushed / popped / shutdown
  std::queue<InputDataPtr> _data_buffer;
  std::thread _feature_thread;

  // pose estimation thread
  std::mutex _tracking_mutex;
  std::condition_variable _tracking_cv;  // tracking data pushed / popped / feature thread done
  std::queue<TrackingDataPtr> _tracking_data_buffer;
  std::thread _tracking_thread;

  std::atomic<bool> _shutdown;
  std::atomic<bool> _feature_thread_stop;
  std::atomic<bool> _tracking_trhead_stop;

  // tmp 
  bool _init;
  bool _insert_next_keyframe;
  int _track_id;
  FramePtr _last_keyframe_feature;
  FramePtr _last_keyframe_tracking;
  FramePtr _last_tracked_frame;
  cv::Mat _last_keyimage;

  cv::Mat key_image_pub;
  int key_image_id_pub;
  std::vector<cv::KeyPoint> keyframe_keypoints_pub;

  // for imu
  Preinteration _preinteration_keyframe;
  bool _preintegration_imu_valid;

private:
  // class
  VisualOdometryConfigs _configs;
  CameraPtr _camera;
  PointMatcherPtr _point_matcher;
  FeatureDetectorPtr _feature_detector;
  std::shared_ptr<DynamicSegmenter> _dynamic_segmenter;   // online left-image dynamic masks (dynamic_mask.enabled)
  PublisherPtr _publisher;
  MapPtr _map;
  OnlineFusionPtr _online;
  StageTimes _times;  // per-stage front-end durations -> frontend_stats.txt
  ImuAttitude _attitude;  // gravity direction of the IMU frame (imu_mode > 0)
  // lost-frame handling (tracking thread)
  FramePtr _cv_prev, _cv_last;               // last two visually tracked frames (constant velocity model)
  std::deque<FramePtr> _normal_keyframes;    // latest normally tracked keyframes (re-link candidates)
  // visual-degradation mode (degraded): set by the tracking thread, read by the feature thread
  std::atomic<bool> _deg_active{false}, _deg_probe_request{false};
  bool _deg_probe_pending = false;            // feature thread: the next keyframe is a requested one
  bool _deg_recovering = false;               // tracking thread: the next keyframe is the recovery keyframe
  int _deg_lost_streak = 0, _deg_good_streak = 0;
  std::vector<FramePtr> _deg_probes;          // probe keyframes of the current degraded segment
  std::vector<FramePtr> _relink_candidates;  // frozen at a break
  int _relink_budget = 0;                    // keyframes after a break that still try to re-link
  bool _in_break = false;
  std::map<std::string, long> _counters;     // tracking states, pose sources, breaks, re-links, leg guess checks
  std::map<std::string, long> _fe_counters;  // feature thread: degraded keyframe requests
  // dynamic-object masking (feature thread): masked frames, features seen / removed
  std::atomic<long> _dynamic_frames{0}, _dynamic_features{0}, _dynamic_removed{0};
  // passive motion (leg_gate.passive): last tracked frame passive (tracking -> feature thread, passive_kf_s), mode
  // entered by the visual test (tracking thread, passive_vis_t), keyframe candidates forced while passive
  std::atomic<bool> _last_frame_passive{false};
  bool _passive_latched = false;
  std::atomic<long> _passive_kf_forced{0};
  // per-frame online output (PublishFrame): the frame pose with the latest back-end correction (front-end pose in
  // visual_only mode), saved as trajectory_frame.txt next to the keyframe trajectory
  std::mutex _frame_trajectory_mutex;
  std::vector<std::pair<double, Eigen::Matrix4d>> _frame_trajectory;
};

#endif  // MAP_BUILDER_H_
