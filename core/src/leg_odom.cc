#include "leg_odom.h"
#include "geometry_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace {

std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

Eigen::Isometry3d ExpBodyTwist(const Eigen::Vector3d& linear,
                               const Eigen::Vector3d& angular,
                               double dt) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  const double speed = angular.norm();
  const Eigen::Matrix3d omega = Skew(angular);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  if (speed > 1e-9) {
    const double theta = speed * dt;
    rotation += std::sin(theta) / speed * omega +
        (1.0 - std::cos(theta)) / (speed * speed) * omega * omega;
  } else {
    rotation += omega * dt + 0.5 * omega * omega * dt * dt;
  }
  result.linear() = rotation;
  // Match LearnVIL's differentiable integrator: translate with the pose at the
  // beginning of the interval, then update rotation.
  result.translation() = linear * dt;
  return result;
}

int Column(const std::unordered_map<std::string, int>& columns,
           const std::string& name) {
  const auto it = columns.find(name);
  return it == columns.end() ? -1 : it->second;
}

}  // namespace

bool LegOdomBuffer::LoadCsv(const std::string& path, std::string* error) {
  samples_.clear();
  std::ifstream input(path);
  if (!input.is_open()) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::string line;
  if (!std::getline(input, line)) {
    if (error) *error = "empty velocity CSV: " + path;
    return false;
  }
  const auto header = SplitCsv(line);
  std::unordered_map<std::string, int> columns;
  for (size_t i = 0; i < header.size(); ++i) columns[header[i]] = static_cast<int>(i);
  const int time_col = Column(columns, "time");
  const int value_cols[6] = {Column(columns, "vx"), Column(columns, "vy"),
      Column(columns, "vz"), Column(columns, "wx"), Column(columns, "wy"),
      Column(columns, "wz")};
  for (int index : value_cols) {
    if (index < 0 || time_col < 0) {
      if (error) *error = "velocity CSV requires time,vx,vy,vz,wx,wy,wz";
      return false;
    }
  }
  const int variance_cols[6] = {Column(columns, "var_vx"), Column(columns, "var_vy"),
      Column(columns, "var_vz"), Column(columns, "var_wx"), Column(columns, "var_wy"),
      Column(columns, "var_wz")};
  size_t line_number = 1;
  try {
    while (std::getline(input, line)) {
      ++line_number;
      if (line.empty()) continue;
      const auto fields = SplitCsv(line);
      LegVelocitySample sample;
      sample.time = std::stod(fields.at(time_col));
      for (int axis = 0; axis < 3; ++axis) {
        sample.linear[axis] = std::stod(fields.at(value_cols[axis]));
        sample.angular[axis] = std::stod(fields.at(value_cols[axis + 3]));
      }
      for (int axis = 0; axis < 6; ++axis) {
        if (variance_cols[axis] >= 0) {
          sample.variance[axis] = std::max(1e-12, std::stod(fields.at(variance_cols[axis])));
        }
      }
      if (!samples_.empty() && sample.time <= samples_.back().time) continue;
      samples_.push_back(sample);
    }
  } catch (const std::exception& exception) {
    samples_.clear();
    if (error) *error = "invalid velocity CSV at line " + std::to_string(line_number) +
        ": " + exception.what();
    return false;
  }
  if (samples_.size() < 2) {
    if (error) *error = "velocity CSV contains fewer than two ordered samples";
    return false;
  }
  return true;
}

bool LegOdomBuffer::Integrate(double start_time, double end_time,
    const Eigen::Matrix<double, 6, 1>& bias, Eigen::Isometry3d* delta,
    Eigen::Matrix<double, 6, 1>* variance) const {
  if (!delta || samples_.size() < 2 || !(end_time > start_time) ||
      start_time < samples_.front().time || end_time > samples_.back().time) {
    return false;
  }
  *delta = Eigen::Isometry3d::Identity();
  if (variance) variance->setZero();
  auto upper = std::upper_bound(samples_.begin(), samples_.end(), start_time,
      [](double time, const LegVelocitySample& sample) { return time < sample.time; });
  size_t index = upper == samples_.begin() ? 0 : static_cast<size_t>(upper - samples_.begin() - 1);
  double covered_until = start_time;
  while (index + 1 < samples_.size() && covered_until < end_time) {
    const double segment_start = std::max(start_time, samples_[index].time);
    const double segment_end = std::min(end_time, samples_[index + 1].time);
    if (segment_end > segment_start) {
      const double dt = segment_end - segment_start;
      const Eigen::Vector3d linear = samples_[index].linear - bias.head<3>();
      const Eigen::Vector3d angular = samples_[index].angular - bias.tail<3>();
      *delta = *delta * ExpBodyTwist(linear, angular, dt);
      if (variance) *variance += samples_[index].variance * dt * dt;
      covered_until = segment_end;
    }
    ++index;
  }
  return covered_until >= end_time - 1e-9;
}

void LegOdomStream::Push(const LegVelocitySample& sample, const FootState& foot) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!items_.empty() && sample.time <= items_.back().sample.time) return;
    items_.push_back({sample, foot});
    while (items_.size() > 2 && items_.front().sample.time < sample.time - buffer_s_) items_.pop_front();
  }
  pushed_.notify_all();
}

bool LegOdomStream::WaitUntil(double t, double timeout_s) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return pushed_.wait_for(lock, std::chrono::duration<double>(std::max(0.0, timeout_s)),
                          [&] { return !items_.empty() && items_.back().sample.time >= t; });
}

bool LegOdomStream::Slice(double t0, double t1, LegOdomBuffer* out) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!(t1 > t0) || items_.size() < 2 || items_.front().sample.time > t0 || items_.back().sample.time < t1)
    return false;
  auto begin = std::upper_bound(items_.begin(), items_.end(), t0,
      [](double t, const Item& item) { return t < item.sample.time; }) - 1;
  auto end = std::lower_bound(items_.begin(), items_.end(), t1,
      [](const Item& item, double t) { return item.sample.time < t; }) + 1;
  LegOdomBuffer::Samples samples;
  for (auto it = begin; it != end; ++it) samples.push_back(it->sample);
  *out = LegOdomBuffer(std::move(samples));
  return true;
}

bool LegOdomStream::FootAt(double t, FootState* foot, std::array<bool, 4>* stable, double window_s) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (items_.empty() || t < items_.front().sample.time || t > items_.back().sample.time) return false;
  auto it = std::lower_bound(items_.begin(), items_.end(), t,
      [](const Item& item, double time) { return item.sample.time < time; });
  if (it != items_.begin() && (it == items_.end() || t - (it - 1)->sample.time < it->sample.time - t)) --it;
  *foot = it->foot;
  stable->fill(true);
  auto a = std::lower_bound(items_.begin(), items_.end(), t - window_s,
      [](const Item& item, double time) { return item.sample.time < time; });
  for (; a != items_.end() && a->sample.time <= t + window_s; ++a)
    for (int leg = 0; leg < 4; ++leg) (*stable)[leg] = (*stable)[leg] && a->foot.contact[leg] == it->foot.contact[leg];
  return true;
}

bool LegOdomStream::NearestAngular(double t, double max_dt, double* tk, Eigen::Vector3d* w) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (items_.empty()) return false;
  auto it = std::lower_bound(items_.begin(), items_.end(), t,
      [](const Item& item, double time) { return item.sample.time < time; });
  if (it == items_.end() || (it != items_.begin() && t - (it - 1)->sample.time < it->sample.time - t)) --it;
  if (std::abs(it->sample.time - t) > max_dt) return false;
  *tk = it->sample.time;
  *w = it->sample.angular;
  return true;
}

Eigen::Vector3d LegOdomStream::LatestBias() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return items_.empty() ? Eigen::Vector3d::Zero() : items_.back().sample.gyro_bias;
}

double LegOdomStream::EndTime() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return items_.empty() ? -INFINITY : items_.back().sample.time;
}

void LegOdomStream::SetExtrinsic(const Eigen::Isometry3d& T_C_B) {
  std::lock_guard<std::mutex> lock(mutex_);
  T_C_B_ = T_C_B;
  has_extrinsic_ = true;
}

bool LegOdomStream::Extrinsic(Eigen::Isometry3d* T_C_B) const {
  std::lock_guard<std::mutex> lock(mutex_);
  *T_C_B = T_C_B_;
  return has_extrinsic_;
}
