#include "ros2_publisher.h"

#include <algorithm>

#include <Eigen/Geometry>

#include "utils.h"

namespace {

rclcpp::Time ToRosTime(double seconds) {
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
}

geometry_msgs::msg::Pose ToPoseMessage(const Eigen::Matrix4d& pose_matrix) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = pose_matrix(0, 3);
  pose.position.y = pose_matrix(1, 3);
  pose.position.z = pose_matrix(2, 3);
  Eigen::Quaterniond q(pose_matrix.block<3, 3>(0, 0));
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

}  // namespace

Ros2Publisher::Ros2Publisher(const RosPublisherConfig& ros_publisher_config,
    const rclcpp::Node::SharedPtr& node)
    : _config(ros_publisher_config), _node(node) {
  if(_config.feature){
    _ros_feature_pub = _node->create_publisher<sensor_msgs::msg::Image>(_config.feature_topic, 10);
    std::function<void(const FeatureMessgaeConstPtr&)> publish_feature_function =
        [&](const FeatureMessgaeConstPtr& feature_message){
      if(_ros_feature_pub->get_subscription_count() == 0) return;  // drawing is only for viewers
      cv::Mat result;
      if(feature_message->fm_type == FeatureMessgaeType::RelocFeature){
        result = DrawFeatures(feature_message->image, feature_message->keypoints);
      }else{
        cv::Mat drawed_image = DrawFeatures(feature_message->image, feature_message->keypoints);

        result = DrawMatches(feature_message->key_image, drawed_image,
            feature_message->keyframe_keypoints, feature_message->keypoints, feature_message->matches);

        cv::copyMakeBorder(result, result, 50, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        cv::putText(result, "Feature Detection", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        std::string keyframe_info = "Keyframe Num : " + std::to_string(feature_message->keyfrmae_id);
        cv::putText(result, keyframe_info, cv::Point(10+drawed_image.cols, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        std::string frame_info = "Current Frame Id: " + std::to_string(feature_message->frame_id);
        cv::putText(result, frame_info, cv::Point(10+drawed_image.cols*2, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      }

      auto ros_feature_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result).toImageMsg();
      ros_feature_msg->header.stamp = ToRosTime(feature_message->time);
      _ros_feature_pub->publish(*ros_feature_msg);
    };
    _feature_publisher.Register(publish_feature_function);
    _feature_publisher.Start();
  }

  if(_config.frame_pose){
    _ros_frame_pose_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(_config.frame_pose_topic, 10);
    _pub_latest_odometry = _node->create_publisher<nav_msgs::msg::Odometry>(_config.frame_odometry_topic, 1000);
    _pub_backend_odometry = _node->create_publisher<nav_msgs::msg::Odometry>(_config.frame_odometry_topic + "/backend", 1000);
    _tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(_node);

    std::function<void(const FramePoseMessageConstPtr&)> publish_frame_pose_function =
        [&](const FramePoseMessageConstPtr& frame_pose_message){
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ToRosTime(frame_pose_message->time);
      pose_stamped.header.frame_id = "map";
      pose_stamped.pose = ToPoseMessage(frame_pose_message->pose);
      _ros_frame_pose_pub->publish(pose_stamped);

      nav_msgs::msg::Odometry odometry;
      odometry.header = pose_stamped.header;
      odometry.pose.pose = pose_stamped.pose;
      _pub_latest_odometry->publish(odometry);
      if(frame_pose_message->has_backend_pose){
        odometry.pose.pose = ToPoseMessage(frame_pose_message->backend_pose);
        _pub_backend_odometry->publish(odometry);
      }

      geometry_msgs::msg::TransformStamped transform;
      transform.header = pose_stamped.header;
      transform.child_frame_id = "camera";
      transform.transform.translation.x = frame_pose_message->pose(0, 3);
      transform.transform.translation.y = frame_pose_message->pose(1, 3);
      transform.transform.translation.z = frame_pose_message->pose(2, 3);
      transform.transform.rotation = pose_stamped.pose.orientation;
      _tf_broadcaster->sendTransform(transform);
    };
    _frame_pose_publisher.Register(publish_frame_pose_function);
    _frame_pose_publisher.Start();

  }

  if(_config.keyframe){
    _ros_keyframe_pub = _node->create_publisher<geometry_msgs::msg::PoseArray>(_config.keyframe_topic, 10);
    _ros_keyframe_array.header.frame_id = "map";

    _ros_path_pub = _node->create_publisher<nav_msgs::msg::Path>(_config.path_topic, 10);
    _ros_path.header.frame_id = "map";

    std::function<void(const KeyframeMessageConstPtr&)> publish_keyframe_function =
        [&](const KeyframeMessageConstPtr& keyframe_message){
      std::lock_guard<std::mutex> lock(_keyframe_mutex);
      _ros_keyframe_array.header.stamp = ToRosTime(keyframe_message->time);
      _ros_path.header.stamp = ToRosTime(keyframe_message->time);

      std::map<int, int>::iterator it;
      for(std::size_t i = 0; i < keyframe_message->ids.size(); ++i){
        const int keyframe_id = keyframe_message->ids[i];
        geometry_msgs::msg::Pose pose = ToPoseMessage(keyframe_message->poses[i]);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ToRosTime(keyframe_message->times[i]);
        pose_stamped.header.frame_id = "map";
        pose_stamped.pose = pose;

        it = _keyframe_id_to_index.find(keyframe_id);
        if(it == _keyframe_id_to_index.end()){
          _ros_keyframe_array.poses.push_back(pose);
          _ros_path.poses.push_back(pose_stamped);
          _keyframe_id_to_index[keyframe_id] = _ros_keyframe_array.poses.size()-1;
        }else{
          const int idx = it->second;
          _ros_keyframe_array.poses[idx] = pose;
          _ros_path.poses[idx] = pose_stamped;
        }
      }
      // the state above is always kept; serialising the whole array only for viewers
      if(_ros_keyframe_pub->get_subscription_count() > 0) _ros_keyframe_pub->publish(_ros_keyframe_array);
      if(_ros_path_pub->get_subscription_count() > 0) _ros_path_pub->publish(_ros_path);
    };
    _keyframe_publisher.Register(publish_keyframe_function);
    _keyframe_publisher.Start();
  }

  if(_config.map){
    _ros_map_pub = _node->create_publisher<sensor_msgs::msg::PointCloud>(_config.map_topic, 1);
    _ros_mappoints.header.stamp = _node->now();
    _ros_mappoints.header.frame_id = "map";

    std::function<void(const MapMessageConstPtr&)> publish_map_function =
        [&](const MapMessageConstPtr& map_message){
      std::lock_guard<std::mutex> lock(_map_mutex);
      _ros_mappoints.header.stamp = ToRosTime(map_message->time);

      std::unordered_map<int, int>::iterator it;
      for(std::size_t i = 0; i < map_message->ids.size(); ++i){
        const int mappoint_id = map_message->ids[i];
        it = _mappoint_id_to_index.find(mappoint_id);
        if(it == _mappoint_id_to_index.end()){
          geometry_msgs::msg::Point32 point;
          point.x = map_message->points[i](0);
          point.y = map_message->points[i](1);
          point.z = map_message->points[i](2);
          _ros_mappoints.points.push_back(point);
          _mappoint_id_to_index[mappoint_id] = _ros_mappoints.points.size()-1;
        }else{
          const int idx = it->second;
          _ros_mappoints.points[idx].x = map_message->points[i](0);
          _ros_mappoints.points[idx].y = map_message->points[i](1);
          _ros_mappoints.points[idx].z = map_message->points[i](2);
        }
      }
      if(_ros_map_pub->get_subscription_count() > 0) _ros_map_pub->publish(_ros_mappoints);
    };
    _map_publisher.Register(publish_map_function);
    _map_publisher.Start();

  }

  if(_config.reloc){
    _ros_reloc_raw_traj_pub =
        _node->create_publisher<visualization_msgs::msg::Marker>(_config.reloc_topic+"/raw_trajectory", 1);

    _ros_reloc_raw_traj.header.stamp = _node->now();
    _ros_reloc_raw_traj.header.frame_id = "map";
    _ros_reloc_raw_traj.pose.orientation.w = 1.0;
    _ros_reloc_raw_traj.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    _ros_reloc_raw_traj.scale.x = 0.25;
    _ros_reloc_raw_traj.scale.y = 0.25;
    _ros_reloc_raw_traj.scale.z = 0.25;
    _ros_reloc_raw_traj.color.a = 1.0;
    _ros_reloc_raw_traj.color.g = 1.0;

    std::function<void(const RelocMessageConstPtr&)> publish_reloc_traj_function =
        [&](const RelocMessageConstPtr& reloc_message){
      if(reloc_message->poses.empty()){
        return;
      }
      visualization_msgs::msg::Marker& traj = _ros_reloc_raw_traj;
      const std::size_t idx = reloc_message->poses.size() - 1;
      geometry_msgs::msg::Point p;
      p.x = reloc_message->poses[idx](0, 3);
      p.y = reloc_message->poses[idx](1, 3);
      p.z = reloc_message->poses[idx](2, 3);
      traj.header.stamp = ToRosTime(reloc_message->times[idx]);
      traj.points.push_back(p);

      _ros_reloc_raw_traj_pub->publish(_ros_reloc_raw_traj);
    };
    _reloc_traj_publisher.Register(publish_reloc_traj_function);
    _reloc_traj_publisher.Start();

    _ros_reloc_pose_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(_config.reloc_topic+"/pose", 10);
    std::function<void(const RelocMessageConstPtr&)> publish_reloc_pose_function =
        [&](const RelocMessageConstPtr& reloc_message){
      const std::size_t idx = reloc_message->times.size() - 1;
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ToRosTime(reloc_message->times[idx]);
      pose_stamped.header.frame_id = "map";
      pose_stamped.pose = ToPoseMessage(reloc_message->poses[idx]);
      _ros_reloc_pose_pub->publish(pose_stamped);
    };
    _reloc_pose_publisher.Register(publish_reloc_pose_function);
    _reloc_pose_publisher.Start();

    _ros_reloc_mpts_pub = _node->create_publisher<visualization_msgs::msg::Marker>(_config.reloc_topic+"/matches", 1);
    std::function<void(const RelocMessageConstPtr&)> publish_reloc_mpts_function =
        [&](const RelocMessageConstPtr& reloc_message){
      const std::size_t idx = reloc_message->times.size() - 1;
      visualization_msgs::msg::Marker ros_reloc_mpts;
      ros_reloc_mpts.header.stamp = ToRosTime(reloc_message->times[idx]);
      ros_reloc_mpts.header.frame_id = "map";
      ros_reloc_mpts.pose.orientation.w = 1.0;
      ros_reloc_mpts.type = visualization_msgs::msg::Marker::LINE_LIST;
      ros_reloc_mpts.scale.x = 0.1 * reloc_message->map_scale;
      ros_reloc_mpts.color.b = 1.0;
      ros_reloc_mpts.color.a = 0.5;

      geometry_msgs::msg::Point point1;
      point1.x = reloc_message->poses[idx](0, 3);
      point1.y = reloc_message->poses[idx](1, 3);
      point1.z = reloc_message->poses[idx](2, 3);

      for(std::size_t i = 0; i < reloc_message->mappoints.size(); ++i){
        geometry_msgs::msg::Point point2;
        point2.x = reloc_message->mappoints[i](0);
        point2.y = reloc_message->mappoints[i](1);
        point2.z = reloc_message->mappoints[i](2);
        ros_reloc_mpts.points.push_back(point1);
        ros_reloc_mpts.points.push_back(point2);

        std_msgs::msg::ColorRGBA color;
        color.g = 1.0;
        color.a = 1.0;
        ros_reloc_mpts.colors.push_back(color);
        ros_reloc_mpts.colors.push_back(color);
      }
      _ros_reloc_mpts_pub->publish(ros_reloc_mpts);
    };
    _reloc_mpts_publisher.Register(publish_reloc_mpts_function);
    _reloc_mpts_publisher.Start();
  }
}

void Ros2Publisher::PublishFeature(FeatureMessgaePtr feature_message){
  _feature_publisher.Publish(feature_message);
}

void Ros2Publisher::PublishFramePose(FramePoseMessagePtr frame_pose_message){
  _frame_pose_publisher.Publish(frame_pose_message);
}

void Ros2Publisher::PublisheKeyframe(KeyframeMessagePtr keyframe_message){
  _keyframe_publisher.Publish(keyframe_message);
}

void Ros2Publisher::PublishMap(MapMessagePtr map_message){
  _map_publisher.Publish(map_message);
}

void Ros2Publisher::PubRelocResults(RelocMessagePtr reloc_message){
  _reloc_traj_publisher.Publish(reloc_message);
  _reloc_pose_publisher.Publish(reloc_message);
  _reloc_mpts_publisher.Publish(reloc_message);
}

void Ros2Publisher::Clear(){
  {
    std::lock_guard<std::mutex> lock(_keyframe_mutex);
    _keyframe_id_to_index.clear();
    _ros_keyframe_array.poses.clear();
    _ros_path.poses.clear();
  }
  std::lock_guard<std::mutex> lock(_map_mutex);
  _mappoint_id_to_index.clear();
  _ros_mappoints.points.clear();

}

void Ros2Publisher::ShutDown(){
  if(_config.feature){
    _feature_publisher.ShutDown();
  }

  if(_config.frame_pose){
    _frame_pose_publisher.ShutDown();
  }

  if(_config.keyframe){
    _keyframe_publisher.ShutDown();
  }

  if(_config.map){
    _map_publisher.ShutDown();
  }

  if(_config.reloc){
    _reloc_traj_publisher.ShutDown();
    _reloc_pose_publisher.ShutDown();
    _reloc_mpts_publisher.ShutDown();
  }
}
