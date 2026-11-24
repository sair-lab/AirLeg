#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

// ROS parameter if set, else the algorithm-config value, else default_value (shared by the three nodes).
inline std::string PreferParamOrConfig(const std::string& param_value, const YAML::Node& root, const std::string& key,
                                       const std::string& default_value = "") {
  if (!param_value.empty()) return param_value;
  if (root[key]) return root[key].as<std::string>();
  return default_value;
}
