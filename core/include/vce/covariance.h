#pragma once
#include "leg_odom.h"
#include "geometry_utils.h"

namespace vce {
using Matrix6 = Eigen::Matrix<double,6,6>;
using Vector6 = Eigen::Matrix<double,6,1>;
using ::Skew;
Eigen::Matrix3d Exp(const Eigen::Vector3d& v);
// State perturbation: right-local rotation, additive B0-frame translation.
// Noise columns: linear velocity, angular velocity (CSV order).
void StepJacobians(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& linear,
                   const Eigen::Vector3d& angular, double dt, Matrix6& F, Matrix6& G);
Matrix6 ResidualJacobian(const Eigen::Isometry3d& body_delta,
                        const Eigen::Isometry3d& T_C_B);
struct Motion {
  Eigen::Isometry3d body = Eigen::Isometry3d::Identity();
  Matrix6 body_covariance = Matrix6::Zero();
  Matrix6 residual_covariance = Matrix6::Zero();
  // d[right-local rotation, B0 translation]/d(gyro bias) for angular = w - b.
  Eigen::Matrix<double,6,3> bias_jacobian = Eigen::Matrix<double,6,3>::Zero();
};
// Rejects missing variances, nonfinite/negative values and unordered timestamps.
LegOdomBuffer LoadVelocity(const std::string& path);
Motion Integrate(const LegOdomBuffer& buffer, double begin, double end,
                 const Eigen::Isometry3d& T_C_B, double rotation_sigma,
                 double translation_sigma);
}  // namespace vce
