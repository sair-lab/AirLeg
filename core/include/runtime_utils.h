#ifndef RUNTIME_UTILS_H_
#define RUNTIME_UTILS_H_

#include <chrono>

inline double CurrentUnixTimeSec() {
  const auto now = std::chrono::system_clock::now();
  const auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

#endif  // RUNTIME_UTILS_H_
