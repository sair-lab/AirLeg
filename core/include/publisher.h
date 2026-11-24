#ifndef PUBLISHER_H_
#define PUBLISHER_H_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "read_configs.h"
#include "thread_publisher.h"
#include "utils.h"

enum FeatureMessgaeType {
  VOFeature = 0,
  RelocFeature = 1
};

struct FeatureMessgae{
  double time;
  cv::Mat image;
  cv::Mat key_image;
  int frame_id;
  int keyfrmae_id;
  std::vector<bool> inliers;
  std::vector<cv::KeyPoint> keyframe_keypoints;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::DMatch> matches;
  FeatureMessgaeType fm_type;
};
typedef std::shared_ptr<FeatureMessgae> FeatureMessgaePtr;
typedef std::shared_ptr<const FeatureMessgae> FeatureMessgaeConstPtr;

struct FramePoseMessage{
  double time;
  Eigen::Matrix4d pose;
  bool has_backend_pose = false;  // S3 fusion back-end: C * pose with the latest back-end correction C
  Eigen::Matrix4d backend_pose = Eigen::Matrix4d::Identity();
};
typedef std::shared_ptr<FramePoseMessage> FramePoseMessagePtr;
typedef std::shared_ptr<const FramePoseMessage> FramePoseMessageConstPtr;

struct KeyframeMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  std::vector<double> times;
  std::vector<int> ids;
  std::vector<Eigen::Matrix4d> poses;
};
typedef std::shared_ptr<KeyframeMessage> KeyframeMessagePtr;
typedef std::shared_ptr<const KeyframeMessage> KeyframeMessageConstPtr;

struct MapMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  bool reset;
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> points;
};
typedef std::shared_ptr<MapMessage> MapMessagePtr;
typedef std::shared_ptr<const MapMessage> MapMessageConstPtr;

struct RelocMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double map_scale;
  std::vector<double> times;
  std::vector<Eigen::Matrix4d> poses;
  std::vector<Eigen::Vector3d> mappoints;
};
typedef std::shared_ptr<RelocMessage> RelocMessagePtr;
typedef std::shared_ptr<const RelocMessage> RelocMessageConstPtr;

class Publisher{
public:
  virtual ~Publisher() = default;

  virtual void PublishFeature(FeatureMessgaePtr feature_message) = 0;
  virtual void PublishFramePose(FramePoseMessagePtr frame_pose_message) = 0;
  virtual void PublisheKeyframe(KeyframeMessagePtr keyframe_message) = 0;
  virtual void PublishMap(MapMessagePtr map_message) = 0;
  virtual void PubRelocResults(RelocMessagePtr reloc_message) = 0;

  virtual void Clear() = 0;
  virtual void ShutDown() = 0;
};

typedef std::shared_ptr<Publisher> PublisherPtr;

class NullPublisher : public Publisher{
public:
  void PublishFeature(FeatureMessgaePtr) override {}
  void PublishFramePose(FramePoseMessagePtr) override {}
  void PublisheKeyframe(KeyframeMessagePtr) override {}
  void PublishMap(MapMessagePtr) override {}
  void PubRelocResults(RelocMessagePtr) override {}
  void Clear() override {}
  void ShutDown() override {}
};

PublisherPtr MakeNullPublisher();

#endif  // PUBLISHER_H_
