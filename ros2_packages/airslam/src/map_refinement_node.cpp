#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "apps.h"
#include "read_configs.h"
#include "node_params.h"
#include "ros2_publisher.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("air_slam");

  node->declare_parameter<std::string>("mode", "full");
  node->declare_parameter<int>("breakpoint", 0);
  node->declare_parameter<std::string>("config_path", "");
  node->declare_parameter<std::string>("model_dir", "");
  node->declare_parameter<std::string>("map_root", "");
  node->declare_parameter<std::string>("voc_path", "");

  MapRefinementAppOptions options;
  options.mode = node->get_parameter("mode").as_string();
  options.breakpoint = node->get_parameter("breakpoint").as_int();
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

  MapRefinementConfigs configs(options.config_path, options.model_dir);
  PublisherPtr publisher = std::make_shared<Ros2Publisher>(configs.ros_publisher_config, node);
  const int result = RunMapRefinementApp(options, publisher);
  rclcpp::shutdown();
  return result;
}
