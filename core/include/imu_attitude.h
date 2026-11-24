#ifndef IMU_ATTITUDE_H_
#define IMU_ATTITUDE_H_

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "imu.h"

// Gravity direction of the IMU frame from a Mahony explicit complementary filter.
// Initialisation: mean specific force of the first init_samples samples (a static start, as the OpenVINS static
// initialiser); then gyro integration with a slow accelerometer correction (kp, rad/s per unit direction error) and
// an integral gyro bias (ki). Samples whose |acc| differs from g by more than acc_gate (m/s^2, legged impacts)
// do not correct. Down(): unit gravity direction (pointing down) in the IMU frame.
class ImuAttitude {
 public:
  struct Params {
    double kp = 0.3, ki = 0.003;     // steady state
    double kp_init = 3.0;            // first init_time seconds (fast convergence)
    double init_time = 1.0;
    double acc_gate = 2.0;
    double g = 9.81;
    double max_dt = 0.1;             // larger gaps: no integration
    int init_samples = 8;               // 20 ms at 400 Hz (the first image may come early)
  };

  ImuAttitude() = default;
  explicit ImuAttitude(const Params& p) : p_(p) {}
  void SetParams(const Params& p) { p_ = p; }

  // Samples in time order; samples at or before the last processed one are ignored.
  void Update(const ImuDataList& batch) {
    for (const ImuData& m : batch) {
      if (has_time_ && m.timestamp <= t_) continue;
      if (!init_) {
        acc_sum_ += m.acc;
        if (++n_init_ >= p_.init_samples && acc_sum_.norm() > 1e-6) {
          // R_world_imu with the world z axis along the measured specific force (up), yaw arbitrary
          q_ = Eigen::Quaterniond::FromTwoVectors(acc_sum_.normalized(), Eigen::Vector3d::UnitZ());
          init_ = true;
          t0_ = m.timestamp;
        }
        t_ = m.timestamp;
        has_time_ = true;
        continue;
      }
      const double dt = m.timestamp - t_;
      t_ = m.timestamp;
      if (dt <= 0.0 || dt > p_.max_dt) continue;
      const bool settling = t_ - t0_ < p_.init_time;
      const Eigen::Vector3d v_hat = q_.conjugate() * Eigen::Vector3d::UnitZ();   // estimated up, IMU frame
      Eigen::Vector3d e = Eigen::Vector3d::Zero();
      const double an = m.acc.norm();
      if (std::abs(an - p_.g) < p_.acc_gate) e = (m.acc / an).cross(v_hat);
      if (!settling) integral_ += p_.ki * e * dt;
      const Eigen::Vector3d w = m.gyr + (settling ? p_.kp_init : p_.kp) * e + integral_;
      const double angle = w.norm() * dt;
      if (angle > 1e-12) q_ = (q_ * Eigen::Quaterniond(Eigen::AngleAxisd(angle, w.normalized()))).normalized();
    }
  }

  bool Valid() const { return init_; }
  bool Settled() const { return init_ && t_ - t0_ >= p_.init_time; }
  double Time() const { return t_; }
  Eigen::Vector3d Down() const { return -(q_.conjugate() * Eigen::Vector3d::UnitZ()); }
  Eigen::Vector3d GyroBias() const { return -integral_; }

 private:
  Params p_;
  Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d integral_ = Eigen::Vector3d::Zero(), acc_sum_ = Eigen::Vector3d::Zero();
  double t_ = 0.0, t0_ = 0.0;
  bool has_time_ = false, init_ = false;
  int n_init_ = 0;
};

#endif  // IMU_ATTITUDE_H_
