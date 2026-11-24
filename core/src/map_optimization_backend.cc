// ROS-independent file boundary used by both S2 and S4 schedulers.
#include "apps.h"
#include "vce/experiment.h"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  if (argc == 3 && (std::string(argv[1]) == "vce" || std::string(argv[1]) == "vce-global" ||
                    std::string(argv[1]) == "vce-adjoint")) {
    try { return vce::RunExperiment(argv[2], std::string(argv[1]) == "vce-global", std::string(argv[1]) == "vce-adjoint"); }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
  }
  if(argc != 6){
    std::cerr << "Usage: map_optimization_backend prepare|ba|global|structure CONFIG MODEL_DIR MAP_ROOT VOC\n       map_optimization_backend vce|vce-global EXPERIMENT.yaml\n";
    return 2;
  }
  MapRefinementAppOptions options;
  options.mode = argv[1];
  if(options.mode != "prepare" && options.mode != "ba" && options.mode != "global" && options.mode != "structure") return 2;
  options.config_path = argv[2];
  options.model_dir = argv[3];
  options.map_root = argv[4];
  options.voc_path = argv[5];
  try {
    return RunMapRefinementApp(options, nullptr);
  } catch(const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
