#ifndef APPS_H_
#define APPS_H_

#include <functional>
#include <string>

#include "publisher.h"

struct MapRefinementAppOptions {
  std::string mode = "full";  // full, prepare, ba, global (S4 global back-end) or structure (points on fixed poses)
  int breakpoint = 0;
  std::string config_path;
  std::string model_dir;
  std::string map_root;
  std::string voc_path;
};

struct RelocalizationAppOptions {
  std::string config_path;
  std::string model_dir;
  std::string map_root;
  std::string voc_path;
  std::string traj_path;
  std::string dataroot;
  std::string camera_config_path;
};

int RunMapRefinementApp(const MapRefinementAppOptions& options,
    PublisherPtr publisher);

int RunRelocalizationApp(const RelocalizationAppOptions& options,
    PublisherPtr publisher,
    const std::function<bool()>& keep_running = []() { return true; });

#endif  // APPS_H_
