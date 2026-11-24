#include <stdexcept>
#include "map_builder.h"
#include "dynamic_segmenter.h"

#include <algorithm>
#include <filesystem>

#include <assert.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "super_point.h"
#include "read_configs.h"
#include "imu.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"

extern std::atomic<long> g_pose_reorthonormalized;   // frame.cc

namespace {
// Drop feature columns (row 1 = x, row 2 = y, rectified pixels) that fall on a dynamic-object mask.
long RemoveMaskedFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features, const cv::Mat& mask){
  if(mask.empty() || features.cols() == 0) return 0;
  std::vector<int> keep;
  keep.reserve(features.cols());
  for(int i = 0; i < features.cols(); ++i){
    const int x = static_cast<int>(std::lround(features(1, i)));
    const int y = static_cast<int>(std::lround(features(2, i)));
    const bool dynamic = x >= 0 && y >= 0 && x < mask.cols && y < mask.rows && mask.at<uchar>(y, x) != 0;
    if(!dynamic) keep.push_back(i);
  }
  const long removed = features.cols() - static_cast<long>(keep.size());
  if(removed == 0) return 0;
  Eigen::Matrix<float, 259, Eigen::Dynamic> kept(259, keep.size());
  for(size_t j = 0; j < keep.size(); ++j) kept.col(j) = features.col(keep[j]);
  features = kept;
  return removed;
}
}  // namespace

MapBuilder::MapBuilder(VisualOdometryConfigs& configs, PublisherPtr publisher): _shutdown(false), _feature_thread_stop(false), 
    _tracking_trhead_stop(false), _init(false), _insert_next_keyframe(false), _track_id(0),
    _preintegration_imu_valid(true), _configs(configs){
  ClearDir(_configs.saving_dir, "leg_odom");
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  if(!configs.use_imu) _camera->DisableImu();
  std::cout << "[imu] use_imu=" << configs.use_imu << " imu_mode=" << _camera->ImuMode()
            << " backend.imu=" << configs.online.backend.imu << std::endl;
  {
    ImuAttitude::Params attitude;
    attitude.kp = _camera->AttitudeKp();
    attitude.ki = _camera->AttitudeKi();
    if(configs.online.UseLeg() && configs.online.backend.net_gyro) attitude.ki = 0.0;   // rate already bias-corrected
    _attitude.SetParams(attitude);
  }
  _preinteration_keyframe.SetNoiseAndWalk(_camera->GyrNoise(), _camera->AccNoise(), _camera->GyrWalk(), _camera->AccWalk());
  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config.WithImageSize(
      static_cast<int>(_camera->ImageWidth()), static_cast<int>(_camera->ImageHeight()))));
  _feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.superpoint_config));
  if(configs.dynamic_mask.enabled)
    _dynamic_segmenter = std::make_shared<DynamicSegmenter>(configs.dynamic_mask,
        static_cast<int>(_camera->ImageWidth()), static_cast<int>(_camera->ImageHeight()));
  _publisher = publisher ? std::move(publisher) : MakeNullPublisher();
  _map = std::shared_ptr<Map>(new Map(_configs.backend_optimization_config, _camera, _publisher));
  if(_configs.voxel_cap.enabled) _map->SetPointVoxel(_configs.voxel_cap.voxel_size, _configs.voxel_cap.max_per_voxel);
  if(_configs.online.UseLeg()){
    _online = std::make_shared<OnlineFusion>(_configs.online, _camera);
    _map->SetOnlineFusion(_online);
  }

  _feature_thread = std::thread(boost::bind(&MapBuilder::ExtractFeatureThread, this));
  _tracking_thread = std::thread(boost::bind(&MapBuilder::TrackingThread, this));
}

bool MapBuilder::UseImuData(){
  return _camera->ImuMode() > 0;
}

bool MapBuilder::UseIMU(){
  return _camera->UseIMU();
}

void MapBuilder::AddInput(InputDataPtr data){
  cv::Mat image_left_rect, image_right_rect;
  _camera->UndistortImage(data->image_left, data->image_right, image_left_rect, image_right_rect);
  data->image_left = image_left_rect;
  data->image_right = image_right_rect;
  if(!data->mask_left.empty() || !data->mask_right.empty()){
    cv::Mat mask_left_rect, mask_right_rect;
    _camera->UndistortMask(data->mask_left, mask_left_rect, false);
    _camera->UndistortMask(data->mask_right, mask_right_rect, true);
    data->mask_left = mask_left_rect;
    data->mask_right = mask_right_rect;
  }

  {
    std::unique_lock<std::mutex> lock(_buffer_mutex);
    _buffer_cv.wait(lock, [&] { return _data_buffer.size() <= 3 || _shutdown; });
    _data_buffer.push(data);
  }
  _buffer_cv.notify_all();
}

void MapBuilder::ExtractFeatureThread(){
  while(true){
    InputDataPtr input_data;
    {
      std::unique_lock<std::mutex> lock(_buffer_mutex);
      _buffer_cv.wait(lock, [&] { return !_data_buffer.empty() || _shutdown; });
      if(_data_buffer.empty()) break;  // shut down and drained
      input_data = _data_buffer.front();
      _data_buffer.pop();
    }
    _buffer_cv.notify_all();
    auto lap = std::chrono::steady_clock::now();

    int frame_id = input_data->index;
    double timestamp = input_data->time;
    // online dynamic masks of the left image (no mask given with the input): the network runs on its own stream
    // while SuperPoint extracts the features; collected before the left features are filtered
    bool segmenting = false;
    if(_dynamic_segmenter && input_data->mask_left.empty()){
      _dynamic_segmenter->Launch(input_data->image_left);
      segmenting = true;
    }
    // dynamic-object masks: drop features on dynamic objects (left counts once per frame)
    auto FilterDynamic = [&](Eigen::Matrix<float, 259, Eigen::Dynamic>& features, const cv::Mat& mask, bool left){
      if(left && segmenting){
        segmenting = false;
        input_data->mask_left = _dynamic_segmenter->Collect();
        lap = _times.Lap("dynamic_mask_wait", lap);   // the part of the segmentation not hidden behind SuperPoint
      }
      if(mask.empty()) return;
      if(left){
        ++_dynamic_frames;
        _dynamic_features += features.cols();
        _dynamic_removed += RemoveMaskedFeatures(features, mask);
      }else{
        RemoveMaskedFeatures(features, mask);
      }
    };
    // read-only downstream: no deep copies
    cv::Mat image_left_rect = input_data->image_left;
    cv::Mat image_right_rect = input_data->image_right;

    // construct frame
    FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera, timestamp));

    Eigen::Matrix<float, 259, Eigen::Dynamic> left_features, right_features; 
    std::vector<cv::DMatch> matches, stereo_matches;
    int good_stereo_point = 0;
    FrameType frame_type;
    if(!_init || _insert_next_keyframe){
      if(!_feature_detector->Detect(image_left_rect, image_right_rect, left_features, right_features)){
        std::cerr << "Feature extraction failed at t=" << std::fixed << std::setprecision(6) << timestamp << "; frame skipped" << std::endl;
        continue;
      }
      lap = _times.Lap("detect_stereo", lap);
      FilterDynamic(left_features, input_data->mask_left, true);
      FilterDynamic(right_features, input_data->mask_right, false);
      _point_matcher->MatchingPoints(left_features, right_features, stereo_matches, false);
      lap = _times.Lap("match_stereo", lap);
      frame->AddLeftFeatures(left_features);
      good_stereo_point = frame->AddRightFeatures(right_features, stereo_matches);
      frame_type = _init ? FrameType::KeyFrame : FrameType::InitializationFrame;

    }else{
      if(!_feature_detector->Detect(image_left_rect, left_features)){
        std::cerr << "Feature extraction failed at t=" << std::fixed << std::setprecision(6) << timestamp << "; frame skipped" << std::endl;
        continue;
      }
      lap = _times.Lap("detect_left", lap);
      FilterDynamic(left_features, input_data->mask_left, true);
      frame->AddLeftFeatures(left_features);
      frame_type = FrameType::NormalFrame;
    }

    if(_init){
      const Eigen::Matrix<float, 259, Eigen::Dynamic> features_last_keyframe = _last_keyframe_feature->GetAllFeatures();
      _point_matcher->MatchingPoints(features_last_keyframe, left_features, matches, true);
      lap = _times.Lap("match_keyframe", lap);
      int enough_match = AddKeyframeCheck(_last_keyframe_feature, frame, matches);
      if(enough_match != 0 && _configs.leg_gate.passive_kf_s > 0.0 && _last_frame_passive &&
         timestamp - _last_keyframe_feature->GetTimestamp() > _configs.leg_gate.passive_kf_s){
        enough_match = 0;   // carried robot: refresh the keyframe
        ++_passive_kf_forced;
      }

      if(enough_match == 0){  // try to insert this frame as keyframe
        if(frame_type == FrameType::NormalFrame){
          if(!_feature_detector->Detect(image_right_rect, right_features)) right_features.resize(259, 0);
          lap = _times.Lap("detect_right_late", lap);
          FilterDynamic(right_features, input_data->mask_right, false);
          _point_matcher->MatchingPoints(left_features, right_features, stereo_matches, false);
          lap = _times.Lap("match_stereo", lap);
          good_stereo_point = frame->AddRightFeatures(right_features, stereo_matches);
        }

        if(good_stereo_point < 10){
          _insert_next_keyframe = true;
          frame_type = FrameType::NormalFrame;
        }else{
          frame_type = FrameType::KeyFrame;
          _insert_next_keyframe = false;
        }
      }else{
        _insert_next_keyframe = (enough_match == 1) && (frame_type == FrameType::NormalFrame);
      }
    }else{
      if(good_stereo_point < _configs.keyframe_config.min_init_stereo_feature){
        std::cout << "good_stereo_point = " << good_stereo_point << std::endl;
        std::cout << "Not enough stereo points to initialize!" << std::endl;
        continue;
      }else{
        std::cout << "Initialization is done!" << std::endl;
        _init = true;
      }
    }

    // visual-degradation mode: no regular keyframes, only the requested (probe / recovery) ones
    if(_configs.degraded.enabled && _init && frame_type != FrameType::InitializationFrame){
      if(_deg_probe_request.exchange(false)){
        if(frame_type == FrameType::KeyFrame){
          ++_fe_counters["deg_keyframe_requested"];
        }else{
          _insert_next_keyframe = true;   // the next frame is detected in stereo and becomes the keyframe
          _deg_probe_pending = true;
        }
      }else if(frame_type == FrameType::KeyFrame && _deg_probe_pending){
        _deg_probe_pending = false;
        ++_fe_counters["deg_keyframe_requested"];
      }else if(_deg_active){
        if(frame_type == FrameType::KeyFrame){
          frame_type = FrameType::NormalFrame;   // suppressed (its stereo features are simply unused)
          ++_fe_counters["deg_keyframe_suppressed"];
        }
        _insert_next_keyframe = false;
      }
    }

    TrackingDataPtr tracking_data = std::shared_ptr<TrackingData>(new TrackingData());
    tracking_data->frame = frame;
    tracking_data->frame_type = frame_type;
    tracking_data->ref_keyframe = _last_keyframe_feature;
    tracking_data->matches = matches;
    tracking_data->input_data = input_data;

    if(frame_type != FrameType::NormalFrame){
      _last_keyframe_feature = frame;
    }

    {
      std::unique_lock<std::mutex> lock(_tracking_mutex);
      _tracking_cv.wait(lock, [&] { return _tracking_data_buffer.size() <= 5; });
      _tracking_data_buffer.push(tracking_data);
    }
    _tracking_cv.notify_all();
  }

  {
    std::lock_guard<std::mutex> lock(_tracking_mutex);
    _feature_thread_stop = true;
  }
  _tracking_cv.notify_all();
}

void MapBuilder::TrackingThread(){ 
  while(true){
    TrackingDataPtr tracking_data;
    {
      std::unique_lock<std::mutex> lock(_tracking_mutex);
      _tracking_cv.wait(lock, [&] { return !_tracking_data_buffer.empty() || _feature_thread_stop; });
      if(_tracking_data_buffer.empty()) break;  // the feature thread finished and everything is tracked
      tracking_data = _tracking_data_buffer.front();
      _tracking_data_buffer.pop();
    }
    _tracking_cv.notify_all();
    const auto frame_start = std::chrono::steady_clock::now();

    FramePtr frame = tracking_data->frame;
    FrameType frame_type = tracking_data->frame_type;
    FramePtr ref_keyframe = tracking_data->ref_keyframe;
    InputDataPtr input_data = tracking_data->input_data;
    std::vector<cv::DMatch> matches = tracking_data->matches;

    double timestamp = input_data->time;
    cv::Mat image_left_rect = input_data->image_left;
    cv::Mat image_right_rect = input_data->image_right;
    ImuDataList batch_imu_data = input_data->batch_imu_data;

    // IMU attitude filter (imu_mode > 0): gravity direction of this frame in its camera frame; the raw samples also go
    // to the back-end IMU factors (backend.export_extras)
    if(_camera->ImuMode() > 0){
      if(_online && _configs.online.backend.export_extras) _online->PushImu(batch_imu_data);
      if(_online && _configs.online.backend.net_gyro){   // raw gyro + network correction (raw samples pushed above)
        ImuDataList net = batch_imu_data;
        _online->NetGyroCorrect(&net);
        _attitude.Update(net);
      }else{
        _attitude.Update(batch_imu_data);
      }
      if(_attitude.Valid()){
        frame->has_gravity = true;
        frame->gravity_c = _camera->RCameraImu() * _attitude.Down();
      }
    }

    if(frame_type == FrameType::InitializationFrame){
      Eigen::Matrix4d init_pose;
      init_pose << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 1;
      // init_pose = Eigen::Matrix4d::Identity();
      if(frame->has_gravity){
        // gravity-aligned world: the measured gravity of the first frame becomes -z (yaw of the default pose kept)
        const Eigen::Matrix3d R0 = init_pose.block<3, 3>(0, 0);
        init_pose.block<3, 3>(0, 0) = Eigen::Quaterniond::FromTwoVectors(R0 * frame->gravity_c,
            -Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
        std::cout << "[imu] gravity-aligned initial pose, camera gravity " << frame->gravity_c.transpose() << std::endl;
      }else if(_camera->ImuMode() > 0){
        std::cerr << "[imu] no IMU attitude at the initial frame: world not gravity aligned" << std::endl;
      }
      frame->SetPose(init_pose);
      frame->SetPoseFixed(true);
      frame->SetVelocaity(Eigen::Vector3d::Zero());

      _preinteration_keyframe.SetBias(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), false);
      frame->SetIMUPreinteration(_preinteration_keyframe);

      if(_online) _online->OnKeyframe(frame, image_left_rect, image_right_rect);
      InsertKeyframe(frame);
      _last_keyframe_tracking = frame;
      _last_tracked_frame = frame;
      _last_keyimage = image_left_rect;

      PublishFrame(frame, image_left_rect, frame_type, matches);
      continue;
    }


    // IMU preinteration
    if(UseIMU()){
      const double interval_start = _preinteration_keyframe.Valid()
          ? _preinteration_keyframe.end_time
          : ref_keyframe->GetTimestamp();
      const bool integrated = _preintegration_imu_valid && input_data->imu_data_valid &&
          _preinteration_keyframe.AddBatchData(batch_imu_data, interval_start, timestamp);
      if(!integrated){
        std::cerr << "Disable IMU preintegration at frame " << frame->GetFrameId()
                  << ": interval=[" << std::fixed << std::setprecision(9)
                  << interval_start << ", " << timestamp << "], samples="
                  << batch_imu_data.size() << std::endl;
        _preintegration_imu_valid = false;
        _preinteration_keyframe.Reset();
      }
    }
    frame->SetIMUPreinteration(_preinteration_keyframe);

    auto lap = std::chrono::steady_clock::now();
    int track_inliers = TrackFrame(ref_keyframe, frame, matches, _preinteration_keyframe);
    lap = _times.Lap("track", lap);

    frame->SetPreviousFrame(ref_keyframe);

    // Tracking state; a lost frame carries its motion-model pose (FramePoseOptimization), so it becomes the
    // reference of the next plausibility checks as well.
    const KeyframeConfig& kc = _configs.keyframe_config;
    frame->track_state = track_inliers > kc.min_num_match ? TrackState::Normal
        : (track_inliers > kc.lost_num_match ? TrackState::Weak : TrackState::Lost);
    ++_counters[frame->track_state == TrackState::Normal ? "frames_normal"
                : frame->track_state == TrackState::Weak ? "frames_weak" : "frames_lost"];
    if(frame->track_state == TrackState::Lost) ++_counters[std::string("lost_pose_") + PoseSourceName(frame->pose_source)];
    const bool lost_handling = kc.lost_handling;
    if(lost_handling || track_inliers > kc.lost_num_match) _last_tracked_frame = frame;
    if(frame->track_state != TrackState::Lost){
      _cv_prev = _cv_last;
      _cv_last = frame;
    }

    if(frame_type == FrameType::KeyFrame){
      std::cout << "insert keyframe, id = " << frame->GetFrameId() << std::endl;
      const bool lost = lost_handling && frame->track_state == TrackState::Lost;
      if(lost){
        // Break: no shared map points with the previous keyframe. Without a leg / IMU edge, a weak motion
        // prior keeps the local BA connected.
        ++_counters["break_keyframes"];
        if(frame->pose_source == PoseSource::ConstantVelocity || frame->pose_source == PoseSource::Stationary){
          frame->has_motion_prior = true;
          frame->motion_prior = ref_keyframe->GetPose().inverse() * frame->GetPose();
          frame->motion_prior_dt = frame->GetTimestamp() - ref_keyframe->GetTimestamp();
        }
        std::cout << "[tracking] break: keyframe " << frame->GetFrameId() << " lost (inliers=" << track_inliers
                  << "), pose from " << PoseSourceName(frame->pose_source) << std::endl;
        if(!_in_break){
          _relink_candidates.assign(_normal_keyframes.begin(), _normal_keyframes.end());
          _in_break = true;
        }
        _relink_budget = 3;
      }
      if(_online) _online->OnKeyframe(frame, image_left_rect, image_right_rect);
      lap = _times.Lap("s3_on_keyframe", lap);
      InsertKeyframe(frame);
      lap = _times.Lap("insert_keyframe", lap);
      if(!lost && _in_break && _relink_budget > 0){
        // tracking recovered: try to attach the new keyframes to the map before the break
        --_relink_budget;
        ++_counters["relink_attempts"];
        if(TryRelink(frame)){
          ++_counters["relink_success"];
          _relink_budget = 0;
        }
        lap = _times.Lap("relink", lap);
      }
      if(!lost && _relink_budget == 0) _in_break = false;
      if(frame->track_state == TrackState::Normal){
        _normal_keyframes.push_back(frame);
        if(_normal_keyframes.size() > 3) _normal_keyframes.pop_front();
      }
      _last_keyframe_tracking = frame;
      _last_keyimage = image_left_rect;
    }

    if(_configs.degraded.enabled) DegradedUpdate(frame, frame_type == FrameType::KeyFrame);
    PublishFrame(frame, image_left_rect, frame_type, matches);
    _times.Lap("publish", lap);
    _times.Lap(frame_type == FrameType::KeyFrame ? "tracking_total_keyframe" : "tracking_total_frame", frame_start);
  }  

  _tracking_trhead_stop = true;
}

int MapBuilder::TrackFrame(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches, Preinteration& _preinteration){
  std::vector<int> inliers(current_frame->FeatureNum(), -1);
  std::vector<MappointPtr> matched_mappoints(current_frame->FeatureNum(), nullptr);
  std::vector<MappointPtr>& ref_frame_mappoints = ref_frame->GetAllMappoints();
  for(auto& match : matches){
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = ref_frame_mappoints[idx0];
    inliers[idx1] = ref_frame->GetTrackId(idx0);
  }

  int num_inliers = FramePoseOptimization(ref_frame, current_frame, matched_mappoints, inliers, _preinteration);

  // update track id
  if(num_inliers > _configs.keyframe_config.lost_num_match){
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();){
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if(inliers[idx1] > 0){
        current_frame->SetTrackId(idx1, ref_frame->GetTrackId(idx0));
        current_frame->InsertMappoint(idx1, ref_frame_mappoints[idx0]);
      }

      if(inliers[idx1] > 0){
        it++;
      }else{
        it = matches.erase(it);
      }
    }

  }

  return num_inliers;
}

int MapBuilder::FramePoseOptimization(FramePtr frame0, FramePtr frame1, std::vector<MappointPtr>& mappoints, 
    std::vector<int>& inliers, Preinteration& preinteration){

  // get initial pose
  bool imu_init = _map->IMUInit();
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  Eigen::Vector3d vwb = Eigen::Vector3d::Zero();
  int frame_id1 = frame1->GetFrameId();
  
  bool predict_by_pnp = true;
  PoseSource guess_source = PoseSource::Visual;  // visual = PnP / last pose
  if(imu_init && preinteration.Valid() && preinteration.dT < 2.0){
    Eigen::Matrix4d Twb1;
    Eigen::Matrix4d Twb0 = frame0->IMUPose();
    Eigen::Vector3d vwb0 = frame0->GetVelocity();
    preinteration.Predict(Twb0, vwb0, Twb1, vwb);
    Twc = Twb1 * frame1->GetCamera()->CameraToBody();
    Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3);
    if(check_dp.norm() < 1.0){
      predict_by_pnp = false;
      guess_source = PoseSource::Imu;
    }
  }

  const double leg_wait_s = _configs.online.leg_wait_ms * 1e-3;
  if(predict_by_pnp && _online && _configs.online.leg_init_guess){  // S3: leg-predicted initial pose
    LegRelativeConstraint c;
    if(_online->LegConstraint(frame0, frame1, &c, leg_wait_s)){
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.topLeftCorner<3, 3>() = c.T_c1_c2.linear();
      T.topRightCorner<3, 1>() = c.T_c1_c2.translation();
      const Eigen::Matrix4d Twc_leg = frame0->GetPose() * T;
      if((Twc_leg.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3)).norm() >
         _configs.online.leg_init_max_jump){
        ++_counters["leg_guess_rejected_jump"];  // implausible jump from the last reliable pose: use PnP
      }else{
        Twc = Twc_leg;
        predict_by_pnp = false;
        guess_source = PoseSource::Leg;
        // Cross-check with PnP when the map supports it: a confident PnP far from the leg guess wins.
        int valid_points = 0;
        for(const MappointPtr& mpt : mappoints) valid_points += mpt && mpt->IsValid();
        const bool cross_check = std::isfinite(_configs.online.leg_init_check_translation) ||
                                 std::isfinite(_configs.online.leg_init_check_rotation_deg);
        if(cross_check && valid_points >= _configs.keyframe_config.min_num_match){
          Eigen::Matrix4d Twc_pnp = Twc_leg;
          std::vector<int> pnp_inliers;
          const int n_pnp = SolvePnPWithCV(frame1, mappoints, Twc_pnp, pnp_inliers);
          if(n_pnp >= _configs.keyframe_config.min_num_match){
            const double dt = (Twc_pnp.block<3, 1>(0, 3) - Twc_leg.block<3, 1>(0, 3)).norm();
            const Eigen::AngleAxisd dR(Eigen::Matrix3d(Twc_pnp.block<3, 3>(0, 0).transpose() * Twc_leg.block<3, 3>(0, 0)));
            if(dt > _configs.online.leg_init_check_translation ||
               dR.angle() * 180.0 / M_PI > _configs.online.leg_init_check_rotation_deg){
              Twc = Twc_pnp;
              guess_source = PoseSource::Visual;
              ++_counters["leg_guess_replaced_by_pnp"];
            }
          }
        }
      }
    }
  }
  if(predict_by_pnp){
    // solve PnP using opencv to get initial pose
    std::vector<int> cv_inliers;
    int num_cv_inliers = SolvePnPWithCV(frame1, mappoints, Twc, cv_inliers);
    Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3);
    if(check_dp.norm() > 1.0 || num_cv_inliers < _configs.keyframe_config.lost_num_match ){
      Twc = _last_tracked_frame->GetPose();
    }
  }

  frame1->SetPose(Twc);
  frame1->SetVelocaity(vwb);

  // Second, optimization
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

  // map of poses
  if(imu_init && preinteration.Valid()){
    AddFrameVertex(frame0, poses, 0, velocities, biases, imu_constraints, true, false, true);
    AddFrameVertex(frame1, poses, 0, velocities, biases, imu_constraints, false, false);

    // imu constraint
    IMUConstraintPtr imu_constraint = std::shared_ptr<ImuConstraint>(new ImuConstraint()); 
    imu_constraint->id_pose1 = frame0->GetFrameId();
    imu_constraint->id_pose2 = frame1->GetFrameId();;
    imu_constraint->id_camera1 = 0;
    imu_constraint->id_camera2 = 0;
    imu_constraint->preinteration = std::make_shared<Preinteration>(preinteration);;
    imu_constraints.emplace_back(imu_constraint);
  }else{
    AddFrameVertex(frame1, poses, 0, false);
  }

  // S3 leg prior: reference keyframe (fixed) -> current frame, like the VIO IMU edge.
  VectorOfLegConstraints leg_constraints;
  LegRelativeConstraint leg_constraint;
  if(_online && _configs.online.frame_leg_prior && _online->LegConstraint(frame0, frame1, &leg_constraint, leg_wait_s)){
    if(!poses.count(frame0->GetFrameId())) AddFrameVertex(frame0, poses, 0, true);
    leg_constraints.push_back(leg_constraint);
  }

  // passive motion: the robot did not step (OnlineFusion::Standing), yet it may be carried (pushed, sliding, a moving
  // platform): with a visible camera translation (passive_vis_t), track this frame visually from the PnP-RANSAC pose
  // (pnp_init), without the leg prior and the leg gate (weak visual: the usual lost handling)
  bool passive = false, have_vis_pnp = false;
  Eigen::Matrix4d Twc_vis_pnp = Eigen::Matrix4d::Identity();
  if(_configs.leg_gate.passive && _online &&
     _online->Standing(std::max(frame0->GetTimestamp(), frame1->GetTimestamp() - _configs.leg_gate.passive_window_s),
                       frame1->GetTimestamp(), _configs.leg_gate.passive_foot_m)){
    passive = true;
    if(_configs.leg_gate.passive_vis_t > 0.0 && !_passive_latched){   // did the camera visibly translate?
      passive = false;
      ++_counters["passive_vis_checks"];
      LegRelativeConstraint c;
      Eigen::Matrix4d T = frame1->GetPose();
      std::vector<int> pin;
      if(_online->LegConstraint(frame0, frame1, &c, leg_wait_s) &&
         SolvePnPWithCV(frame1, mappoints, T, pin) >= _configs.keyframe_config.min_num_match && T.allFinite()){
        const Eigen::Vector3d p_leg = frame0->GetPose().block<3, 3>(0, 0) * c.T_c1_c2.translation() +
                                      frame0->GetPose().block<3, 1>(0, 3);
        passive = (T.block<3, 1>(0, 3) - p_leg).norm() > _configs.leg_gate.passive_vis_t;
        have_vis_pnp = true;
        Twc_vis_pnp = T;
      }else{
        ++_counters["passive_vis_unavailable"];
      }
      if(!passive) ++_counters["passive_vis_kept_leg"];
    }
  }
  if(passive){
    leg_constraints.clear();
    ++_counters["leg_passive_frames"];
    if(_configs.leg_gate.pnp_init && poses.count(frame_id1)){
      Eigen::Matrix4d T = have_vis_pnp ? Twc_vis_pnp : frame1->GetPose();   // reuse the test's PnP pose
      std::vector<int> pin;
      if(have_vis_pnp ||
         (SolvePnPWithCV(frame1, mappoints, T, pin) >= _configs.keyframe_config.min_num_match && T.allFinite())){
        poses[frame_id1].R = T.block<3, 3>(0, 0);
        poses[frame_id1].p = T.block<3, 1>(0, 3);
        ++_counters["pnp_init_used"];
      }
    }
  }
  _passive_latched = passive;
  _last_frame_passive = passive;

  // visual constraint construction
  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;
  // leg gate: leg-predicted pose and its (inflated, floored) covariance [rotation, translation] in the camera frame
  const LegGateConfig& lg = _configs.leg_gate;
  bool gate = false;
  Eigen::Matrix4d Twc_leg_gate = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double, 6, 6> gate_cov = Eigen::Matrix<double, 6, 6>::Zero();
  if(lg.enabled && _online && !passive){
    LegRelativeConstraint c;
    if(_online->LegConstraint(frame0, frame1, &c, leg_wait_s)){
      Eigen::Matrix<double, 6, 6> cov = c.information.inverse();
      if(cov.allFinite()){
        cov *= lg.cov_scale;
        const double r2 = std::pow(lg.min_sigma_r_deg * M_PI / 180.0, 2), t2 = lg.min_sigma_t * lg.min_sigma_t;
        for(int k = 0; k < 3; ++k) cov(k, k) = std::max(cov(k, k), r2);
        for(int k = 3; k < 6; ++k) cov(k, k) = std::max(cov(k, k), t2);
        double slip = 0.0;
        if(lg.slip_k > 0.0 && _online->LegSlip(frame0, frame1, c, &slip)){   // slipping feet: trust the leg less
          const double s2 = std::pow(lg.slip_k * slip, 2);
          for(int k = 0; k < 3; ++k) cov(k, k) += s2;
          for(int k = 3; k < 6; ++k) cov(k, k) += s2;
          _counters["leg_gate_slip_mm"] += static_cast<long>(slip * 1000.0);
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3, 3>() = c.T_c1_c2.linear();
        T.topRightCorner<3, 1>() = c.T_c1_c2.translation();
        Twc_leg_gate = frame0->GetPose() * T;
        gate_cov = cov;
        gate = true;
        ++_counters["leg_gate_frames"];
      }
    }
  }
  const Eigen::Matrix4d Tcw_leg_gate = Twc_leg_gate.inverse();
  const double gate_px2 = lg.px_sigma * lg.px_sigma;
  long gate_rejected = 0;
  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame1->GetKeypointPosition(i, keypoint)) continue;
    if(gate){   // Mahalanobis test against the leg-predicted projection
      const Eigen::Vector3d pc = Tcw_leg_gate.block<3, 3>(0, 0) * mpt->GetPosition() + Tcw_leg_gate.block<3, 1>(0, 3);
      if(pc.z() < 0.05){ ++gate_rejected; inliers[i] = -1; continue; }
      const double fx = _camera->Fx(), fy = _camera->Fy(), iz = 1.0 / pc.z();
      Eigen::Matrix<double, 2, 3> Jp;
      Jp << fx * iz, 0.0, -fx * pc.x() * iz * iz, 0.0, fy * iz, -fy * pc.y() * iz * iz;
      Eigen::Matrix<double, 3, 6> Jx;
      Jx.leftCols<3>() << 0.0, -pc.z(), pc.y(), pc.z(), 0.0, -pc.x(), -pc.y(), pc.x(), 0.0;   // [pc]x
      Jx.rightCols<3>() = -Eigen::Matrix3d::Identity();
      const Eigen::Matrix<double, 2, 6> J = Jp * Jx;
      const Eigen::Matrix2d S = J * gate_cov * J.transpose() + gate_px2 * Eigen::Matrix2d::Identity();
      const Eigen::Vector2d r(fx * pc.x() * iz + _camera->Cx() - keypoint.x(), fy * pc.y() * iz + _camera->Cy() - keypoint.y());
      if(r.dot(S.ldlt().solve(r)) > lg.obs_chi2){ ++gate_rejected; inliers[i] = -1; continue; }
    }

    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    // visual constraint
    (AddPointConstraint(frame_id1, mpt_id, keypoint, mono_point_constraints, stereo_point_constraints) ? stereo_indexes : mono_indexes).push_back(i);
  }

  int num_inliers = FrameOptimization(poses, points, velocities, biases, camera_list,
    mono_point_constraints, stereo_point_constraints, imu_constraints, Rwg,
    _configs.tracking_optimization_config, leg_constraints.empty() ? nullptr : &leg_constraints,
    _configs.online.leg.huber_delta);

  if(gate){
    _counters["leg_gate_obs_rejected"] += gate_rejected;
    const int n_obs = static_cast<int>(mono_point_constraints.size() + stereo_point_constraints.size());
    const bool strong = lg.strong_inliers > 0 && num_inliers >= lg.strong_inliers &&
                        n_obs > 0 && num_inliers >= lg.strong_ratio * n_obs;
    if(strong) ++_counters["leg_gate_strong_visual"];
    if(!strong && num_inliers > _configs.keyframe_config.lost_num_match){   // optimised pose vs the leg prediction
      const Eigen::Matrix3d Rl = Twc_leg_gate.block<3, 3>(0, 0), Rv = poses[frame_id1].R;
      const Eigen::AngleAxisd aa(Eigen::Matrix3d(Rl.transpose() * Rv));
      Eigen::Matrix<double, 6, 1> e;
      e.head<3>() = aa.angle() * aa.axis();
      e.tail<3>() = Rl.transpose() * (poses[frame_id1].p - Twc_leg_gate.block<3, 1>(0, 3));
      if(e.allFinite() && e.dot(gate_cov.ldlt().solve(e)) > lg.frame_chi2){
        ++_counters["leg_gate_frame_rejected"];
        num_inliers = 0;   // visual not trusted: lost handling (motion model, leg first)
      }
    }
  }

  if(num_inliers <= _configs.keyframe_config.lost_num_match){
    // Lost: no visual optimization of this frame; a leg / IMU guess is kept, otherwise a motion model.
    frame1->pose_source = _configs.keyframe_config.lost_handling ? guess_source : PoseSource::Visual;
    if(_configs.keyframe_config.lost_handling &&
       (guess_source == PoseSource::Visual || (_configs.keyframe_config.lost_prefer_leg && guess_source == PoseSource::Imu))){
      Eigen::Matrix4d Twc_model = Twc;
      MotionModelPose(frame0, frame1, preinteration, &Twc_model, &frame1->pose_source);
      frame1->SetPose(Twc_model);
    }
  }else{
    frame1->pose_source = PoseSource::Visual;
  }

  if(num_inliers > _configs.keyframe_config.lost_num_match ){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses[frame_id1].R;
    frame_pose.block<3, 1>(0, 3) = poses[frame_id1].p;
    frame1->SetPose(frame_pose);

    // update tracked mappoints
    for(size_t i = 0; i < mono_point_constraints.size(); i++){
      size_t idx = mono_indexes[i];
      if(!mono_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

    for(size_t i = 0; i < stereo_point_constraints.size(); i++){
      size_t idx = stereo_indexes[i];
      if(!stereo_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

    if(velocities.size() > 0){
      frame1->SetVelocaity(velocities[frame_id1].velocity);
      frame1->UpdateBias(biases[frame_id1].gyr_bias, biases[frame_id1].acc_bias);
    }
  }

  return num_inliers;
}

void MapBuilder::DegradedUpdate(FramePtr frame, bool keyframe){
  const DegradedConfig& dg = _configs.degraded;
  const bool lost = frame->track_state == TrackState::Lost;
  if(!_deg_active){
    _deg_lost_streak = lost ? _deg_lost_streak + 1 : 0;
    if(_deg_lost_streak >= dg.enter_lost){
      _deg_active = true;
      _deg_probes.clear();
      _deg_good_streak = 0;
      _deg_recovering = false;
      ++_counters["deg_segments"];
    }
    return;
  }
  ++_counters["deg_frames"];
  if(keyframe){
    if(_deg_recovering){   // recovery keyframe: prune the probes of the segment, leave the degraded mode
      int pruned = 0;
      for(const FramePtr& p : _deg_probes){
        if(!p || p == frame) continue;
        int covis = 0;
        for(const MappointPtr& m : p->GetAllMappoints())
          if(m && m->IsValid() && m->GetAllObversers().count(frame->GetFrameId())) ++covis;
        if(p->track_state == TrackState::Lost || covis < dg.prune_covis){
          _map->DeleteKeyframe(p);
          for(auto it = _normal_keyframes.begin(); it != _normal_keyframes.end();) it = (*it == p) ? _normal_keyframes.erase(it) : it + 1;
          _relink_candidates.erase(std::remove(_relink_candidates.begin(), _relink_candidates.end(), p), _relink_candidates.end());
          ++pruned;
        }
      }
      _counters["deg_probes_pruned"] += pruned;
      _counters["deg_probes_kept"] += static_cast<long>(_deg_probes.size()) - pruned;
      ++_counters["deg_recovered"];
      std::cout << "[degraded] recovered at keyframe " << frame->GetFrameId() << ": " << _deg_probes.size()
                << " probe keyframes, " << pruned << " pruned" << std::endl;
      _deg_probes.clear();
      _deg_active = false;
      _deg_recovering = false;
      _deg_lost_streak = 0;
      return;
    }
    _deg_probes.push_back(frame);
    ++_counters["deg_probes"];
  }
  _deg_good_streak = frame->track_state == TrackState::Normal ? _deg_good_streak + 1 : 0;
  if(!_deg_recovering && _deg_good_streak >= dg.recover_frames){
    _deg_recovering = true;
    _deg_probe_request = true;   // the recovery keyframe
    return;
  }
  // probe keyframe when the (leg) pose moved enough from the last keyframe
  if(!keyframe && !_deg_recovering && !_deg_probe_request && _last_keyframe_tracking){
    const Eigen::Matrix4d T = _last_keyframe_tracking->GetPose().inverse() * frame->GetPose();
    const double angle = Eigen::AngleAxisd(Eigen::Matrix3d(T.block<3, 3>(0, 0))).angle() * 180.0 / M_PI;
    const bool stale = dg.probe_max_dt > 0.0 &&
        frame->GetTimestamp() - _last_keyframe_tracking->GetTimestamp() > dg.probe_max_dt;
    if(T.block<3, 1>(0, 3).norm() > dg.probe_dist || angle > dg.probe_deg || stale){
      _deg_probe_request = true;
      if(stale) ++_counters["deg_probes_stale"];
    }
  }
}

void MapBuilder::MotionModelPose(FramePtr frame0, FramePtr frame1, Preinteration& preinteration, Eigen::Matrix4d* Twc,
                                 PoseSource* source){
  // 1. leg: reference keyframe pose * leg relative motion
  LegRelativeConstraint c;
  if(_online && _online->LegConstraint(frame0, frame1, &c, _configs.online.leg_wait_ms * 1e-3)){
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = c.T_c1_c2.linear();
    T.topRightCorner<3, 1>() = c.T_c1_c2.translation();
    const Eigen::Matrix4d Twc_leg = frame0->GetPose() * T;
    if((Twc_leg.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3)).norm() <=
       _configs.online.leg_init_max_jump){
      *Twc = Twc_leg;
      *source = PoseSource::Leg;
      return;
    }
  }
  // 2. IMU preintegration from the reference keyframe
  if(_map->IMUInit() && preinteration.Valid() && preinteration.dT < 2.0){
    Eigen::Matrix4d Twb1;
    Eigen::Vector3d vwb1;
    preinteration.Predict(frame0->IMUPose(), frame0->GetVelocity(), Twb1, vwb1);
    *Twc = Twb1 * frame1->GetCamera()->CameraToBody();
    *source = PoseSource::Imu;
    return;
  }
  // 3. constant velocity of the last two visually tracked frames, extrapolated to this frame's time
  if(_cv_prev && _cv_last && _cv_last->GetTimestamp() > _cv_prev->GetTimestamp()){
    const Eigen::Matrix4d& T0 = _cv_prev->GetPose();
    const Eigen::Matrix4d& T1 = _cv_last->GetPose();
    const Eigen::Matrix4d step = T0.inverse() * T1;
    const double ratio = (frame1->GetTimestamp() - _cv_last->GetTimestamp()) /
                         (_cv_last->GetTimestamp() - _cv_prev->GetTimestamp());
    if(std::isfinite(ratio) && ratio >= 0.0 && ratio < 30.0){
      const Eigen::AngleAxisd rotation(Eigen::Matrix3d(step.block<3, 3>(0, 0)));
      Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
      delta.block<3, 3>(0, 0) = Eigen::AngleAxisd(rotation.angle() * ratio, rotation.axis()).toRotationMatrix();
      delta.block<3, 1>(0, 3) = step.block<3, 1>(0, 3) * ratio;
      *Twc = T1 * delta;
      *source = PoseSource::ConstantVelocity;
      return;
    }
  }
  // 4. no model: the last pose
  *Twc = _last_tracked_frame->GetPose();
  *source = PoseSource::Stationary;
}

bool MapBuilder::TryRelink(FramePtr keyframe){
  for(auto it = _relink_candidates.rbegin(); it != _relink_candidates.rend(); ++it){
    FramePtr candidate = _map->GetFramePtr((*it)->GetFrameId());
    if(!candidate) continue;
    std::vector<cv::DMatch> matches;
    _point_matcher->MatchingPoints(candidate->GetAllFeatures(), keyframe->GetAllFeatures(), matches, true);
    std::vector<MappointPtr> mappoints(keyframe->FeatureNum(), nullptr);
    int valid = 0;
    for(const auto& m : matches){
      MappointPtr mpt = candidate->GetMappoint(m.queryIdx);
      if(mpt && mpt->IsValid()){ mappoints[m.trainIdx] = mpt; ++valid; }
    }
    if(valid < _configs.keyframe_config.min_num_match) continue;
    Eigen::Matrix4d Twc = keyframe->GetPose();
    std::vector<int> inliers;
    const int n = SolvePnPWithCV(keyframe, mappoints, Twc, inliers);
    if(n < _configs.keyframe_config.min_num_match) continue;
    std::vector<std::pair<int, MappointPtr>> links;
    for(size_t i = 0; i < inliers.size(); ++i) if(inliers[i] > 0 && mappoints[i]) links.emplace_back(i, mappoints[i]);
    const int linked = _map->LinkMappoints(keyframe, links);
    const Eigen::Vector3d dp = Twc.block<3, 1>(0, 3) - keyframe->GetPose().block<3, 1>(0, 3);
    std::cout << "[tracking] re-linked keyframe " << keyframe->GetFrameId() << " to pre-break keyframe "
              << candidate->GetFrameId() << ": pnp_inliers=" << n << " linked=" << linked
              << " pnp_vs_current=" << dp.norm() << " m" << std::endl;
    return linked > 0;
  }
  return false;
}

// return value: 0 : select this frame as keyframe, 1 : select next frame as keyframe, 2 : not select keyframe
int MapBuilder::AddKeyframeCheck(FramePtr ref_keyframe, FramePtr current_frame, const std::vector<cv::DMatch>& matches){
  int match_num = matches.size();
  if(match_num < _configs.keyframe_config.min_num_match) return 0;

  const Eigen::Matrix<float, 259, Eigen::Dynamic>& ref_features = ref_keyframe->GetAllFeatures();
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& current_features = current_frame->GetAllFeatures();

  float feature_tracking_thr = _configs.keyframe_config.tracking_point_rate;
  double ration_thr = _configs.keyframe_config.tracking_parallax_rate;
  if(UseIMU() && !_map->IMUInit()){
    feature_tracking_thr *= 1.1;
    ration_thr *= 0.7;
  }

  if((float)match_num/ref_features.cols() < feature_tracking_thr || (float)match_num/current_features.cols() < feature_tracking_thr || match_num < _configs.keyframe_config.max_num_match){
    return 1;
  }

  Eigen::Matrix2Xf ref_keypoints(2, match_num), current_keypoints(2, match_num);
  for(int i = 0; i < match_num; i++){
    int idx0 = matches[i].queryIdx;
    int idx1 = matches[i].trainIdx;

    ref_keypoints.col(i) = ref_features.block<2, 1>(1, idx0);
    current_keypoints.col(i) = current_features.block<2, 1>(1, idx1);
  }


  Eigen::Matrix2Xf parallax = ref_keypoints - current_keypoints;
  double average_parallax = (double)((parallax * parallax.transpose()).sum()) / match_num;
  double image_size = _camera->ImageHeight() * _camera->ImageWidth();
  
  if(average_parallax > image_size * ration_thr * ration_thr){
    return 1;
  }

  return 2;
}

void MapBuilder::InsertKeyframe(FramePtr frame){
  // create new track id
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  for(size_t i = 0; i < track_ids.size(); i++){
    if(track_ids[i] < 0){
      frame->SetTrackId(i, _track_id++);
    }
  }

  // insert keyframe to map
  _map->InsertKeyframe(frame); 

  _track_id = _map->UpdateFrameTrackIds(_track_id);

  Eigen::Vector3d gyr_bias, acc_bias;
  frame->GetBias(gyr_bias, acc_bias);
  _preinteration_keyframe.Reset();
  _preinteration_keyframe.SetBias(gyr_bias, acc_bias, false);
  _preintegration_imu_valid = true;
}

void MapBuilder::PublishFrame(FramePtr frame, cv::Mat& image, FrameType frame_type, std::vector<cv::DMatch>& matches){

  double timestamp = frame->GetTimestamp();
  const std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  std::vector<bool> inliers_feature_message;
  frame->GetInlierFlag(inliers_feature_message);
  const Eigen::Matrix4d& pose = frame->GetPose();

  if(frame_type==InitializationFrame){
    // std::vector<bool> inliers(keypoints.size(), true);
    // key_image_id_pub = frame->GetFrameId();
    key_image_pub = DrawFeatures(image, keypoints);
    key_image_id_pub = 1;
    keyframe_keypoints_pub = frame->GetAllKeypoints();
    return;
  } 

  FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
  FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);

  feature_message->time = timestamp;
  feature_message->image = image;
  feature_message->key_image = key_image_pub;
  feature_message->frame_id = frame->GetFrameId();
  feature_message->keyfrmae_id = key_image_id_pub;
  feature_message->keypoints = keypoints;
  feature_message->keyframe_keypoints = keyframe_keypoints_pub;
  feature_message->matches = matches;
  feature_message->fm_type = FeatureMessgaeType::VOFeature;
  feature_message->inliers = inliers_feature_message;
  
  frame_pose_message->time = timestamp;
  frame_pose_message->pose = pose;
  if(_online && _online->Backend()){
    frame_pose_message->has_backend_pose = true;
    frame_pose_message->backend_pose = _online->Backend()->Correction().matrix() * pose;
  }
  {
    std::lock_guard<std::mutex> lock(_frame_trajectory_mutex);
    _frame_trajectory.emplace_back(timestamp, frame_pose_message->has_backend_pose ? frame_pose_message->backend_pose : pose);
  }

  _publisher->PublishFeature(feature_message);
  _publisher->PublishFramePose(frame_pose_message);

  if(frame_type==KeyFrame){
    std::vector<bool> inliers(keypoints.size(), true);
    key_image_pub = DrawFeatures(image, keypoints);
    // key_image_id_pub = frame->GetFrameId();
    key_image_id_pub++;
    keyframe_keypoints_pub = frame->GetAllKeypoints();
  }
}

void MapBuilder::SaveTrajectory(){
  std::string file_path = ConcatenateFolderAndFileName(_configs.saving_dir, "keyframe_trajectory.txt");
  _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveTrajectory(std::string file_path){
  _map->SaveKeyframeTrajectory(file_path);
  {   // per-frame online trajectory (left camera T_w_C, image timestamps): the evaluated S3 output
    std::lock_guard<std::mutex> lock(_frame_trajectory_mutex);
    SaveTumTrajectoryToFile((std::filesystem::path(file_path).parent_path() / "trajectory_frame.txt").string(),
                            _frame_trajectory);
  }
  const std::string stats_path = (std::filesystem::path(file_path).parent_path() / "frontend_stats.txt").string();
  _times.Save(stats_path);
  std::ofstream stats(stats_path, std::ios::app);
  std::ostringstream counters;
  for(const auto& kv : _counters) counters << ' ' << kv.first << '=' << kv.second;
  if(_passive_kf_forced > 0) counters << " passive_kf_forced=" << _passive_kf_forced;
  if(_dynamic_frames > 0)
    counters << " dynamic_frames=" << _dynamic_frames << " dynamic_features=" << _dynamic_features
             << " dynamic_removed=" << _dynamic_removed;
  for(const auto& kv : _fe_counters) counters << ' ' << kv.first << '=' << kv.second;
  if(g_pose_reorthonormalized > 0) counters << " pose_reorthonormalized=" << g_pose_reorthonormalized;
  if(_map && _map->voxel_skipped > 0) counters << " voxel_skipped=" << _map->voxel_skipped;
  if(_online) counters << ' ' << _online->LegStats();
  if(_online && _online->Superframes())
    _online->Superframes()->times.Save((std::filesystem::path(file_path).parent_path() / "depth_stats.txt").string());
  stats << "counters" << counters.str() << '\n';
  std::cout << "[tracking] counters" << counters.str() << std::endl;
  if(_online && _online->Backend()) _online->Backend()->Save(std::filesystem::path(file_path).parent_path().string());
}

void MapBuilder::SaveMap(const std::string& map_root){
  std::string map_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv0.bin");
  std::ofstream ofs(map_path, std::ios::binary);
  std::cout << "map_path = " << map_path << std::endl;
  if(!ofs) throw std::runtime_error("Cannot write map file: " + map_path);
  boost::archive::binary_oarchive oa(ofs);

  _map->CheckMap();

  std::cout << "Map saveing..... " << std::endl;
  oa << _map;
  _map->ExportTrainingData(map_root);
  _map->ExportKeyframeStatus(map_root);
  std::cout << "Map saveing done! " << std::endl;

}

void MapBuilder::Stop(){
  {
    std::lock_guard<std::mutex> lock(_buffer_mutex);
    _shutdown = true;
  }
  _buffer_cv.notify_all();
  _publisher->ShutDown();
  _feature_thread.join();
  _tracking_thread.join();
}

bool MapBuilder::IsStopped(){
  bool have_stopped = (_feature_thread_stop && _tracking_trhead_stop);
  return have_stopped;
}
