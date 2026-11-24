#include "apps.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "fusion_backend.h"
#include "map_builder.h"
#include "map_refiner.h"
#include "map_user.h"
#include "read_configs.h"
#include "utils.h"

namespace {

std::filesystem::path CoreDirectory() {
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

int RunProcess(const std::vector<std::string>& arguments,
    const std::filesystem::path& log_path = {}) {
  if (arguments.empty()) return -1;
  const pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    if(!log_path.empty()){
      const int log_fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if(log_fd < 0) _exit(126);
      if(dup2(log_fd, STDOUT_FILENO) < 0 || dup2(log_fd, STDERR_FILENO) < 0){
        close(log_fd);
        _exit(126);
      }
      close(log_fd);
    }
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}


}  // namespace

namespace {
// S4 global back-end. Inputs in `dir`: global.yaml (online.backend keys), backend_nodes.csv / backend_feet.csv (dump of
// the S3 online back-end), prepare_trajectory.txt (loop-closed visual keyframe poses: initial poses, exact keyframe
// times) and loops.txt (accepted loops of the preparation). One window over the whole trajectory, solved once.
// Output: trajectory_v1.txt (exact keyframe times) and the usual back-end files.
int RunGlobalBackend(const std::string& dir) {
  const std::filesystem::path root(dir);
  BackendConfig cfg;
  cfg.Load(YAML::LoadFile((root / "global.yaml").string()));
  cfg.enabled = true;
  cfg.write_back = false;
  std::vector<BackendNode> nodes;
  std::string error;
  if (!LoadBackendDump(dir, &nodes, &error)) { std::cerr << error << std::endl; return 3; }
  const TumTrajectory traj = ReadTumTrajectory((root / "prepare_trajectory.txt").string());
  auto nearest = [&traj](double t) { return NearestPose(traj, t, 6e-3); };  // 6 ms (old dumps: 10 ms times)
  std::map<int, double> exact_time;
  std::vector<std::pair<double, int>> node_times;  // exact time -> node id, for the loop remapping
  int n_init = 0;
  for (auto& n : nodes) {
    if (const auto* p = nearest(n.time)) {
      n.has_init = true;
      n.T_init = Eigen::Isometry3d(p->second);
      exact_time[n.id] = p->first;
      ++n_init;
    }
    node_times.emplace_back(exact_time.count(n.id) ? exact_time[n.id] : n.time, n.id);
  }
  std::sort(node_times.begin(), node_times.end());
  FusionBackend backend(cfg);
  int n_loops = 0;
  {
    // A loop keyframe that is not a back-end node (the last keyframes of a run are never finalized) is moved to the
    // nearest earlier node; the measurement is composed with their relative pose in the prepared trajectory.
    auto node_at = [&](double t, int* id, Eigen::Isometry3d* T_kf_node) {
      auto it = std::upper_bound(node_times.begin(), node_times.end(), std::make_pair(t + 1e-6, 1 << 30));
      if (it == node_times.begin()) return false;
      --it;
      const auto* pk = nearest(t);
      const auto* pn = nearest(it->first);
      if (!pk || !pn || t - it->first > 3.0) return false;
      *id = it->second;
      *T_kf_node = Eigen::Isometry3d(pk->second).inverse() * Eigen::Isometry3d(pn->second);
      return true;
    };
    std::ifstream f((root / "loops.txt").string());
    int il, iq;
    double tl, tq, x, y, z, qx, qy, qz, qw;
    while (f >> il >> iq >> tl >> tq >> x >> y >> z >> qx >> qy >> qz >> qw) {
      Eigen::Isometry3d T_lq = Eigen::Isometry3d::Identity();
      T_lq.translation() = Eigen::Vector3d(x, y, z);
      T_lq.linear() = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
      int nl, nq;
      Eigen::Isometry3d T_l_nl, T_q_nq;
      if (!node_at(tl, &nl, &T_l_nl) || !node_at(tq, &nq, &T_q_nq) || nl == nq) continue;
      backend.AddLoop(nl, nq, T_l_nl.inverse() * T_lq * T_q_nq);
      ++n_loops;
    }
  }
  std::cout << "[S4 global back-end] nodes=" << nodes.size() << " initial poses=" << n_init << " loops=" << n_loops << std::endl;
  for (const auto& n : nodes) backend.Push(n);
  backend.Stop();
  backend.Save(dir);
  const auto result = backend.Trajectory();
  std::vector<std::pair<double, Eigen::Matrix4d>> out;
  for (size_t k = 0; k < result.size() && k < nodes.size(); ++k)
    out.emplace_back(exact_time.count(nodes[k].id) ? exact_time[nodes[k].id] : result[k].first, result[k].second.matrix());
  SaveTumTrajectoryToFile((root / "trajectory_v1.txt").string(), out);
  return 0;
}
}  // namespace

int RunMapRefinementApp(const MapRefinementAppOptions& options,
    PublisherPtr publisher) {
  // S4 global back-end: no map needed
  if(options.mode == "global") return RunGlobalBackend(options.map_root);
  MapRefinementConfigs configs(options.config_path, options.model_dir);
  if(options.mode != "full" && options.mode != "prepare" && options.mode != "ba" && options.mode != "structure"){
    throw std::invalid_argument("Unknown map refinement mode: " + options.mode);
  }
  // ROS entry point of S4: the global back-end of TRAIL pipeline.py s4 (prepare -> global -> structure), whose steps
  // call map_optimization_backend (built next to this executable) with positional arguments.
  if(options.mode == "full" && configs.leg_odom_config.enabled){
    std::vector<std::string> command{configs.leg_odom_config.python_executable,
        (CoreDirectory() / "3rdparty/TRAIL/pipeline.py").string(), "s4",
        "--algorithm", options.config_path, "--map-root", options.map_root,
        "--model-dir", options.model_dir, "--voc-path", options.voc_path,
        "--output-dir", (std::filesystem::path(options.map_root) / "leg_odom/tmp/stage4").string()};
    const auto backend = std::filesystem::canonical("/proc/self/exe").parent_path() / "map_optimization_backend";
    if(std::filesystem::exists(backend)){
      command.insert(command.end(), {"--backend", backend.string()});
    }
    return RunProcess(command);
  }
  MapRefiner refiner(configs, std::move(publisher));
  try {
    refiner.LoadMap(options.map_root);
  } catch (...) {
    refiner.StopVisualization();  // join the visualization thread before the refiner is destroyed
    throw;
  }
  if(options.mode == "prepare" || options.mode == "full"){
    refiner.LoadVocabulary(options.voc_path);
    refiner.Wait(options.breakpoint);
    refiner.UpdateCovisibilityGraph();
    refiner.LoopDetection();
    refiner.PoseGraphRefinement();
    refiner.SaveLoops(ConcatenateFolderAndFileName(options.map_root, "loops.txt"));
    refiner.MergeMap();
  }
  // structure: keyframes moved to init_trajectory.txt (S4 global back-end poses), then only the points are optimized
  const bool structure_only = options.mode == "structure";
  if(structure_only &&
      refiner.InitFromTrajectory(ConcatenateFolderAndFileName(options.map_root, "init_trajectory.txt")) == 0){
    std::cerr << "structure: no keyframe matched init_trajectory.txt" << std::endl;
    refiner.StopVisualization();
    return 4;
  }
  const bool use_leg = options.mode == "ba" && configs.leg_odom_config.enabled;
  if(!refiner.GlobalMapOptimization(use_leg, structure_only)){
    if(options.mode != "prepare"){
      refiner.StopVisualization();
      return 5;
    }
    // prepare: the rejected BA left the last healthy map (loops detected and merged); the S4 global back-end only
    // needs the loops and the keyframe times, and structure re-optimizes the points on the back-end poses.
    std::cerr << "WARNING: prepare global BA rejected; continuing with the map before it (marked ba_rejected)" << std::endl;
    refiner.AcceptLastHealthyMap();
    std::ofstream(ConcatenateFolderAndFileName(options.map_root, "ba_rejected")) << "global BA rejected\n";
  }
  refiner.UpdateCovisibilityGraph();
  refiner.SaveTrajectory(ConcatenateFolderAndFileName(options.map_root, "trajectory_v1.txt"));
  refiner.SaveFinalMap(options.map_root);
  const auto& calibration = refiner.GetLegOdomConfig();
  YAML::Emitter result;
  result.SetDoublePrecision(17);
  result << YAML::BeginMap << YAML::Key << "extrinsic" << YAML::Value << YAML::Flow
         << YAML::BeginSeq << calibration.extrinsic_xyz.x()
         << calibration.extrinsic_xyz.y() << calibration.extrinsic_xyz.z()
         << calibration.extrinsic_quat_xyzw.x() << calibration.extrinsic_quat_xyzw.y()
         << calibration.extrinsic_quat_xyzw.z() << calibration.extrinsic_quat_xyzw.w()
         << YAML::EndSeq << YAML::Key << "time_delay" << YAML::Value
         << calibration.time_delay << YAML::EndMap;
  std::ofstream(ConcatenateFolderAndFileName(options.map_root, "calibration.yaml"))
      << result.c_str();
  refiner.StopVisualization();
  return 0;
}

int RunRelocalizationApp(const RelocalizationAppOptions& options,
    PublisherPtr publisher,
    const std::function<bool()>& keep_running) {
  RelocalizationConfigs configs(options.config_path, options.model_dir);
  if (!options.dataroot.empty()) {
    configs.dataroot = options.dataroot;
  }
  if (!options.camera_config_path.empty()) {
    configs.camera_config_path = options.camera_config_path;
  }

  MapUser map_user(configs, std::move(publisher));
  map_user.LoadMap(options.map_root);
  map_user.LoadVocabulary(options.voc_path);

  std::vector<std::string> image_names;
  GetFileNames(configs.dataroot, image_names);
  std::sort(image_names.begin(), image_names.end());
  const size_t dataset_length = image_names.size();

  std::vector<std::pair<std::string, Eigen::Matrix4d>> trajectory;
  const Eigen::Matrix4d base_frame_pose = map_user.GetBaseFramePose();
  const double base_frame_time = map_user.GetBaseFrameTimestamp();
  trajectory.emplace_back(
      std::make_pair("base " + DoubleTimeToString(base_frame_time), base_frame_pose));

  int success_num = 0;
  for (size_t i = 0; i < dataset_length && keep_running(); ++i) {
    std::cout << "i ====== " << i << std::endl;

    cv::Mat image = cv::imread(
        ConcatenateFolderAndFileName(configs.dataroot, image_names[i]), 0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    const size_t pos = image_names[i].find_last_of('.');
    std::string image_idx = "fail " + image_names[i].substr(0, pos);

    const auto before_infer = std::chrono::high_resolution_clock::now();
    if (map_user.Relocalization(image, pose)) {
      image_idx = "success " + image_names[i].substr(0, pos);
      success_num++;
    }
    const auto after_infer = std::chrono::high_resolution_clock::now();
    const auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_infer - before_infer).count();
    std::cout << "One Frame Processinh Time: " << cost_time << " ms." << std::endl;

    trajectory.emplace_back(std::make_pair(image_idx, pose));
  }

  SaveTumTrajectoryToFile(options.traj_path, trajectory);
  const float recall = dataset_length > 0
      ? static_cast<float>(success_num) / dataset_length
      : 0.0f;
  std::cout << "sum_num = " << dataset_length
            << ", success_num = " << success_num
            << ", recall = " << recall
            << std::endl;

  map_user.StopVisualization();
  return 0;
}
