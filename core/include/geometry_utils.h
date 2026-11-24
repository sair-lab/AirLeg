#pragma once
#include <Eigen/Core>

// [v]x, the cross-product matrix (single definition for the IMU, leg odometry, VCE and edge code).
inline Eigen::Matrix3d Skew(const Eigen::Vector3d& v){
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}
