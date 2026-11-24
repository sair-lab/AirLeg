#include "landmarks.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace {
std::tuple<long, long, long> Voxel(const Eigen::Vector3d& p, double size) {
  return std::make_tuple(static_cast<long>(std::floor(p.x() / size)), static_cast<long>(std::floor(p.y() / size)),
                         static_cast<long>(std::floor(p.z() / size)));
}
}  // namespace

std::vector<long> SelectWindow(const std::vector<BackendNode>& nodes, size_t lo, size_t end, const WindowSettings& s,
                               const std::vector<long>& previous) {
  struct Track { int count = 0; size_t first = 0, last = 0; Eigen::Vector3d pw = Eigen::Vector3d::Zero(); bool stereo = false; };
  std::map<long, Track> tracks;
  for (size_t n = lo; n < end; ++n) {
    if (nodes[n].lost) continue;
    for (const LandmarkObs& o : nodes[n].obs) {
      Track& t = tracks[o.point];
      if (t.count == 0) t.first = n;
      t.last = n;
      ++t.count;
      t.pw = o.pw;
      t.stereo = t.stereo || o.kp.z() >= 0.0;
    }
  }
  std::vector<std::pair<double, long>> ranked;
  for (const auto& kv : tracks) {
    const Track& t = kv.second;
    if (t.count < s.min_nodes || !t.stereo) continue;
    ranked.emplace_back(t.count + 0.5 * static_cast<double>(t.last - t.first), kv.first);   // track length and span
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<long> out;
  std::set<std::tuple<long, long, long>> voxels;
  std::set<long> chosen;
  auto take = [&](long id) {
    if (static_cast<int>(out.size()) >= s.max_points || chosen.count(id)) return;
    auto it = tracks.find(id);
    if (it == tracks.end() || it->second.count < s.min_nodes || !it->second.stereo) return;
    const auto v = Voxel(it->second.pw, s.voxel_size);
    if (voxels.count(v)) return;
    voxels.insert(v);
    chosen.insert(id);
    out.push_back(id);
  };
  for (long id : previous) take(id);              // continuity across solves
  for (const auto& r : ranked) take(r.second);
  return out;
}
