#pragma once
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include "utils.h"
#include "camera.h"
#include "frame.h"
#include "leg_odom.h"

namespace vce {
// adjoint: vce-adjoint mode (implicit hypergradient of the S2 visual loss through a vce-global run), implies global.
int RunExperiment(const std::string& config_path, bool global = false, bool adjoint = false);

}  // namespace vce
