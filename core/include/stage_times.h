#ifndef AIR_SLAM_STAGE_TIMES_H_
#define AIR_SLAM_STAGE_TIMES_H_

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Per-stage wall-clock durations (ms) of a pipeline, summarised to a text file (frontend_stats.txt).
class StageTimes {
 public:
  void Add(const std::string& stage, double ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    times_[stage].push_back(ms);
  }
  // Adds the time since `start` under `stage` and returns now (chains consecutive stages).
  std::chrono::steady_clock::time_point Lap(const std::string& stage, std::chrono::steady_clock::time_point start) {
    const auto now = std::chrono::steady_clock::now();
    Add(stage, std::chrono::duration<double, std::milli>(now - start).count());
    return now;
  }
  void Save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(path);
    out << std::fixed << std::setprecision(3) << "stage n mean_ms p50_ms p95_ms max_ms total_s\n";
    for (auto kv : times_) {
      auto& v = kv.second;
      if (v.empty()) continue;
      std::sort(v.begin(), v.end());
      double sum = 0.0;
      for (double x : v) sum += x;
      auto q = [&v](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; };
      out << kv.first << ' ' << v.size() << ' ' << sum / v.size() << ' ' << q(0.5) << ' ' << q(0.95) << ' ' << v.back()
          << ' ' << sum / 1e3 << '\n';
    }
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::vector<double>> times_;
};

#endif  // AIR_SLAM_STAGE_TIMES_H_
