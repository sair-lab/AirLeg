#ifndef ROS2_PUBLISHER_H_
#define ROS2_PUBLISHER_H_

#include <mutex>
#include <memory>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include "publisher.h"

class Ros2Publisher : public Publisher{
public:
  Ros2Publisher(const RosPublisherConfig& ros_publisher_config, const rclcpp::Node::SharedPtr& node);

  void PublishFeature(FeatureMessgaePtr feature_message) override;
  void PublishFramePose(FramePoseMessagePtr frame_pose_message) override;
  void PublisheKeyframe(KeyframeMessagePtr keyframe_message) override;
  void PublishMap(MapMessagePtr map_message) override;
  void PubRelocResults(RelocMessagePtr reloc_message) override;

  void Clear() override;
  void ShutDown() override;

private:
  RosPublisherConfig _config;
  rclcpp::Node::SharedPtr _node;
  std::shared_ptr<tf2_ros::TransformBroadcaster> _tf_broadcaster;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _ros_feature_pub;
  ThreadPublisher<FeatureMessgae> _feature_publisher;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _ros_frame_pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pub_latest_odometry;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _pub_backend_odometry;  // <frame_odometry_topic>/backend
  ThreadPublisher<FramePoseMessage> _frame_pose_publisher;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr _ros_keyframe_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr _ros_path_pub;
  std::mutex _keyframe_mutex;  // keyframe array / path: publisher thread vs Clear()
  std::map<int, int> _keyframe_id_to_index;
  geometry_msgs::msg::PoseArray _ros_keyframe_array;
  nav_msgs::msg::Path _ros_path;
  ThreadPublisher<KeyframeMessage> _keyframe_publisher;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr _ros_map_pub;
  std::mutex _map_mutex;  // map points: publisher thread vs Clear()
  std::unordered_map<int, int> _mappoint_id_to_index;
  sensor_msgs::msg::PointCloud _ros_mappoints;
  ThreadPublisher<MapMessage> _map_publisher;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _ros_reloc_raw_traj_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _ros_reloc_pose_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr _ros_reloc_mpts_pub;
  visualization_msgs::msg::Marker _ros_reloc_raw_traj;
  ThreadPublisher<RelocMessage> _reloc_traj_publisher;
  ThreadPublisher<RelocMessage> _reloc_pose_publisher;
  ThreadPublisher<RelocMessage> _reloc_mpts_publisher;
};

#endif  // ROS2_PUBLISHER_H_
