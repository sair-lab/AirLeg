#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "apps.h"
#include "read_configs.h"
#include "node_params.h"
#include "ros2_publisher.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("air_slam");

  node->declare_parameter<std::string>("config_path", "");
  node->declare_parameter<std::string>("model_dir", "");
  node->declare_parameter<std::string>("map_root", "");
  node->declare_parameter<std::string>("voc_path", "");
  node->declare_parameter<std::string>("traj_path", "");
  node->declare_parameter<std::string>("dataroot", "");
  node->declare_parameter<std::string>("camera_config_path", "");

  RelocalizationAppOptions options;
  options.config_path = node->get_parameter("config_path").as_string();
  const YAML::Node root = LoadAlgorithmConfig(options.config_path);
  options.model_dir = PreferParamOrConfig(node->get_parameter("model_dir").as_string(), root, "model_dir");
  options.map_root = node->get_parameter("map_root").as_string();
  if (options.map_root.empty()) {
    RCLCPP_ERROR(node->get_logger(), "map_root is required; pass --ros-args -p map_root:=/path/to/map");
    rclcpp::shutdown();
    return 1;
  }
  options.voc_path = PreferParamOrConfig(node->get_parameter("voc_path").as_string(), root, "voc_path");
  options.traj_path = node->get_parameter("traj_path").as_string();
  options.dataroot = node->get_parameter("dataroot").as_string();
  options.camera_config_path =
      PreferParamOrConfig(node->get_parameter("camera_config_path").as_string(), root, "camera_config_path");
  if (options.traj_path.empty() && !options.map_root.empty()) {
    options.traj_path = options.map_root + "/relocalization.txt";
  }

  RelocalizationConfigs configs(options.config_path, options.model_dir);
  if (!options.dataroot.empty()) {
    configs.dataroot = options.dataroot;
  }
  if (!options.camera_config_path.empty()) {
    configs.camera_config_path = options.camera_config_path;
  }

  PublisherPtr publisher = std::make_shared<Ros2Publisher>(configs.ros_publisher_config, node);
  const int result = RunRelocalizationApp(options, publisher, []() { return rclcpp::ok(); });
  rclcpp::shutdown();
  return result;
}
