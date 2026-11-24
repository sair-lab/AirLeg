// Replays the back-end inputs dumped by an online run (backend_nodes.csv, backend_feet.csv)
// through FusionBackend with overridden settings. Deterministic: solves are triggered as online (by default
// on every finalized superframe, `trigger: superframe`; `trigger: stride` solves every `stride` nodes). Usage: backend_replay <run_dir> <out_dir> [key=value ...] (keys of online.backend)
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "camera.h"
#include "fusion_backend.h"


int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: backend_replay <run_dir> <out_dir> [key=value ...]" << std::endl;
    return 2;
  }
  YAML::Node node;
  node["enabled"] = true;
  for (int i = 3; i < argc; ++i) {
    const std::string kv = argv[i];
    const size_t eq = kv.find('=');
    node[kv.substr(0, eq)] = YAML::Load(kv.substr(eq + 1));
  }
  // g_value=<m/s^2>: the camera yaml g_value of the online run (Camera::IMU_G_VALUE, default 9.81; IMU factors)
  if (node["g_value"]) Camera::IMU_G_VALUE = node["g_value"].as<double>();
  BackendConfig cfg;
  cfg.Load(node);
  cfg.write_back = false;

  std::vector<BackendNode> nodes;
  std::string error;
  if (!LoadBackendDump(argv[1], &nodes, &error)) {
    std::cerr << error << std::endl;
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  FusionBackend backend(cfg);
  for (const auto& n : nodes) backend.Push(n);
  std::filesystem::create_directories(argv[2]);
  backend.Save(argv[2]);
  std::cout << "replay done in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
            << " s, alpha " << backend.Alpha().transpose() << std::endl;
  return 0;
}
