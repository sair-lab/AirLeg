#include "g2o_optimization/edge_leg_odom.h"

#include <algorithm>
#include <cmath>

#include "imu.h"
#include "geometry_utils.h"

namespace {

Eigen::Matrix3d ExpRotation(const Eigen::Vector3d& value) {
  const double angle = value.norm();
  if (angle < 1e-12) return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, value / angle).toRotationMatrix();
}

Vector6d TransformError(const Eigen::Isometry3d& measurement,
                        const Eigen::Isometry3d& prediction) {
  Vector6d error;
  Eigen::Vector3d rotation_error;
  SO3Log(measurement.linear().transpose() * prediction.linear(), rotation_error);
  error.head<3>() = rotation_error;
  error.tail<3>() = measurement.linear().transpose() *
      (prediction.translation() - measurement.translation());
  return error;
}

// d SO3Log(R Exp(d)) / d d at d = 0, for both branches of SO3Log (small angle: vee of the
// antisymmetric part; otherwise the exact logarithm, whose derivative is Jr^-1).
Eigen::Matrix3d LogJacobian(const Eigen::Matrix3d& R) {
  const double d = 0.5 * (R.trace() - 1);
  if (std::abs(d) > 0.99999) return 0.5 * (R.trace() * Eigen::Matrix3d::Identity() - R.transpose());
  Eigen::Vector3d phi;
  SO3Log(R, phi);
  const double theta = phi.norm();
  const Eigen::Matrix3d W = Skew(phi);
  return Eigen::Matrix3d::Identity() + 0.5 * W +
         (1.0 / (theta * theta) - (1 + std::cos(theta)) / (2 * theta * std::sin(theta))) * W * W;
}

// Right Jacobian of the exact exponential (ExpRotation).
Eigen::Matrix3d RightJacobian(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  const Eigen::Matrix3d W = Skew(phi);
  if (theta < 1e-6) return Eigen::Matrix3d::Identity() - 0.5 * W + W * W / 6.0;
  return Eigen::Matrix3d::Identity() - (1 - std::cos(theta)) / (theta * theta) * W +
         (theta - std::sin(theta)) / (theta * theta * theta) * W * W;
}

// Jacobians of TransformError(M, T_ci_w T_cj_w^-1) w.r.t. the VIPose updates (dr, dt) of poses i and j.
// With Rcw = Rcb Rwb^T, tcw = tcb - Rcw twb and the right update twb += Rwb dt, Rwb = Rwb Exp(dr):
// Rp <- Exp(-Rcb dri) Rp, Rp <- Rp Exp(Rcb drj), tp = tcb - Rp tcb + Rcw_i (twb_j - twb_i).
void RelativePoseJacobians(const Eigen::Isometry3d& M, const VIPose& pi, const VIPose& pj,
                           Eigen::Matrix<double, 6, 6>* ji, Eigen::Matrix<double, 6, 6>* jj) {
  const Eigen::Matrix3d Rp = pi.Rcw * pj.Rcw.transpose();
  const Eigen::Matrix3d Rmt = M.linear().transpose();
  const Eigen::Matrix3d L = LogJacobian(Rmt * Rp);
  const Eigen::Matrix3d& Rcb = pi.Rcb;
  ji->block<3, 3>(0, 0) = -L * Rp.transpose() * Rcb;
  ji->block<3, 3>(0, 3).setZero();
  ji->block<3, 3>(3, 0) = Rmt * (-Skew(Rp * pi.tcb) * Rcb + Rcb * Skew(pi.Rwb.transpose() * (pj.twb - pi.twb)));
  ji->block<3, 3>(3, 3) = -Rmt * Rcb;
  jj->block<3, 3>(0, 0) = L * pj.Rcb;
  jj->block<3, 3>(0, 3).setZero();
  jj->block<3, 3>(3, 0) = Rmt * Rp * Skew(pj.tcb) * pj.Rcb;
  jj->block<3, 3>(3, 3) = Rmt * pi.Rcw * pj.Rwb;
}

Eigen::Isometry3d CameraRelativePose(const VIPose& pose_i,
                                     const VIPose& pose_j) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = pose_i.Rcw * pose_j.Rcw.transpose();
  result.translation() = pose_i.tcw - result.linear() * pose_j.tcw;
  return result;
}

}  // namespace

void VertexLegExtrinsic::oplusImpl(const double* update) {
  const Eigen::Map<const Vector6d> increment(update);
  Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
  delta.linear() = ExpRotation(increment.head<3>());
  delta.translation() = increment.tail<3>();
  _estimate = delta * _estimate;
}

void VertexLegTimeDelay::oplusImpl(const double* update) {
  _estimate = std::min(upper_, std::max(lower_, _estimate + update[0]));
}

double VertexLegTimeDelay::OplusSlope() const {
  constexpr double step = 1e-9;
  if (_estimate - step > lower_ && _estimate + step < upper_) return 1.0;
  const double plus = std::min(upper_, std::max(lower_, _estimate + step));
  const double minus = std::min(upper_, std::max(lower_, _estimate - step));
  return (plus - minus) / (2 * step);
}

EdgeLegOdom::EdgeLegOdom(std::shared_ptr<const LegOdomBuffer> buffer,
                         double camera_time_i, double camera_time_j)
    : buffer_(std::move(buffer)), camera_time_i_(camera_time_i),
      camera_time_j_(camera_time_j) {
  resize(5);
}

void EdgeLegOdom::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  const auto* extrinsic = static_cast<const VertexLegExtrinsic*>(_vertices[2]);
  const auto* delay = static_cast<const VertexLegTimeDelay*>(_vertices[3]);
  const auto* bias = static_cast<const VertexLegBias*>(_vertices[4]);
  Eigen::Isometry3d body_delta = fixed_body_delta_;
  if (!has_fixed_body_delta_ && (!buffer_ ||
      !buffer_->Integrate(camera_time_i_ + delay->estimate(),
          camera_time_j_ + delay->estimate(), bias->estimate(), &body_delta))) {
    _error.setConstant(1e3);
    return;
  }
  const Eigen::Isometry3d& camera_from_body = extrinsic->estimate();
  const Eigen::Isometry3d camera_delta =
      camera_from_body * body_delta * camera_from_body.inverse();
  const Eigen::Isometry3d visual_delta =
      CameraRelativePose(pose_i->estimate(), pose_j->estimate());
  _error = TransformError(camera_delta, visual_delta);
}

EdgeLegOdomGyrBias::EdgeLegOdomGyrBias(const Eigen::Isometry3d& T_C_B,
                                       const Eigen::Isometry3d& body_delta,
                                       const Eigen::Matrix<double, 6, 3>& bias_jacobian,
                                       const Eigen::Vector3d& linearization_bias)
    : T_C_B_(T_C_B), body_delta_(body_delta), bias_jacobian_(bias_jacobian),
      linearization_bias_(linearization_bias) {
  resize(4);
}

void EdgeLegOdomGyrBias::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  const auto* bias_i = static_cast<const VertexGyrBias*>(_vertices[2]);
  const auto* bias_j = static_cast<const VertexGyrBias*>(_vertices[3]);
  const Eigen::Vector3d delta_bias =
      0.5 * (bias_i->estimate() + bias_j->estimate()) - linearization_bias_;
  const Vector6d correction = bias_jacobian_ * delta_bias;
  Eigen::Isometry3d body_delta = body_delta_;
  body_delta.linear() = body_delta_.linear() * ExpRotation(correction.head<3>());
  body_delta.translation() += correction.tail<3>();
  const Eigen::Isometry3d camera_delta = T_C_B_ * body_delta * T_C_B_.inverse();
  _error = TransformError(camera_delta,
                          CameraRelativePose(pose_i->estimate(), pose_j->estimate()));
}

void EdgeGyrBiasPrior::computeError() {
  _error = static_cast<const VertexGyrBias*>(_vertices[0])->estimate() - _measurement;
}

void EdgeLegBiasRandomWalk::computeError() {
  const auto* first = static_cast<const VertexLegBias*>(_vertices[0]);
  const auto* second = static_cast<const VertexLegBias*>(_vertices[1]);
  _error = second->estimate() - first->estimate() - _measurement;
}

void EdgeLegBiasPrior::computeError() {
  const auto* vertex = static_cast<const VertexLegBias*>(_vertices[0]);
  _error = vertex->estimate() - _measurement;
}

void EdgeLegExtrinsicPrior::computeError() {
  const auto* vertex = static_cast<const VertexLegExtrinsic*>(_vertices[0]);
  _error = TransformError(_measurement, vertex->estimate());
}

void EdgeLegTimeDelayPrior::computeError() {
  const auto* vertex = static_cast<const VertexLegTimeDelay*>(_vertices[0]);
  _error[0] = vertex->estimate() - _measurement;
}

void EdgeLegGyroBias::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  const Eigen::Vector3d b = static_cast<const VertexVec<3>*>(_vertices[2])->estimate();
  Eigen::Isometry3d body = _measurement;
  body.linear() = _measurement.linear() * ExpRotation(-b * dt_);
  _error = TransformError(T_C_B_ * body * T_C_B_.inverse(), CameraRelativePose(pose_i->estimate(), pose_j->estimate()));
}

// Bias: Exp(-(b + d) dt) = Exp(-b dt) Exp(phi), phi = -dt Jr(-b dt) d, so R_M <- R_M Exp(R_L phi) and
// t_M <- t_M + R_M [t_L]x R_L phi with M = T_C_B (R_b Exp(-b dt), t_b) T_C_B^-1.
void EdgeLegGyroBias::linearizeOplus() {
  const VIPose& pi = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
  const VIPose& pj = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
  const Eigen::Vector3d b = static_cast<const VertexVec<3>*>(_vertices[2])->estimate();
  Eigen::Isometry3d body = _measurement;
  body.linear() = _measurement.linear() * ExpRotation(-b * dt_);
  const Eigen::Isometry3d M = T_C_B_ * body * T_C_B_.inverse();
  Eigen::Matrix<double, 6, 6> ji, jj;
  RelativePoseJacobians(M, pi, pj, &ji, &jj);
  _jacobianOplus[0] = ji;
  _jacobianOplus[1] = jj;
  const Eigen::Matrix3d Rp = pi.Rcw * pj.Rcw.transpose();
  const Eigen::Matrix3d E = M.linear().transpose() * Rp;
  const Eigen::Vector3d et = M.linear().transpose() * (pi.tcw - Rp * pj.tcw - M.translation());
  const Eigen::Matrix3d RL = T_C_B_.linear();
  const Eigen::Matrix3d dphi = -dt_ * RightJacobian(-b * dt_);
  Eigen::Matrix<double, 6, 3> jb;
  jb.topRows<3>() = -LogJacobian(E) * E.transpose() * RL * dphi;
  jb.bottomRows<3>() = (Skew(et) - Skew(T_C_B_.translation())) * RL * dphi;
  _jacobianOplus[2] = jb;
}

void EdgeScaledRelativePose::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  Eigen::Isometry3d M = _measurement;
  M.translation() *= 1.0 + static_cast<const VertexVec<1>*>(_vertices[2])->estimate()[0];
  _error = TransformError(M, CameraRelativePose(pose_i->estimate(), pose_j->estimate()));
}

// The scale only enters the translation residual R_m^T (t_pred - (1 + v) t_m): d / dv = -R_m^T t_m.
void EdgeScaledRelativePose::linearizeOplus() {
  Eigen::Isometry3d M = _measurement;
  M.translation() *= 1.0 + static_cast<const VertexVec<1>*>(_vertices[2])->estimate()[0];
  Eigen::Matrix<double, 6, 6> ji, jj;
  RelativePoseJacobians(M, static_cast<const VertexVIPose*>(_vertices[0])->estimate(),
                        static_cast<const VertexVIPose*>(_vertices[1])->estimate(), &ji, &jj);
  _jacobianOplus[0] = ji;
  _jacobianOplus[1] = jj;
  Eigen::Matrix<double, 6, 1> jv = Eigen::Matrix<double, 6, 1>::Zero();
  jv.tail<3>() = -_measurement.linear().transpose() * _measurement.translation();
  _jacobianOplus[2] = jv;
}

Eigen::Isometry3d EdgeLegGyroBiasVel::Measured(const Eigen::Vector3d& vb) const {
  const Eigen::Vector3d b = static_cast<const VertexVec<3>*>(_vertices[2])->estimate();
  Eigen::Isometry3d body = _measurement;
  body.linear() = _measurement.linear() * ExpRotation(-b * dt_);
  body.translation() = _measurement.translation() -
                       0.5 * (Eigen::Matrix3d::Identity() + _measurement.linear()) * vb * dt_;
  return T_C_B_ * body * T_C_B_.inverse();
}

void EdgeLegGyroBiasVel::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  const Eigen::Vector3d vb = static_cast<const VertexVec<3>*>(_vertices[3])->estimate();
  _error = TransformError(Measured(vb), CameraRelativePose(pose_i->estimate(), pose_j->estimate()));
}

void EdgeLegGyroBiasVel::linearizeOplus() {
  const VIPose& pi = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
  const VIPose& pj = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
  const Eigen::Vector3d b = static_cast<const VertexVec<3>*>(_vertices[2])->estimate();
  const Eigen::Vector3d vb = static_cast<const VertexVec<3>*>(_vertices[3])->estimate();
  const Eigen::Isometry3d M = Measured(vb);
  Eigen::Matrix<double, 6, 6> ji, jj;
  RelativePoseJacobians(M, pi, pj, &ji, &jj);
  _jacobianOplus[0] = ji;
  _jacobianOplus[1] = jj;
  const Eigen::Matrix3d Rp = pi.Rcw * pj.Rcw.transpose();
  const Eigen::Matrix3d E = M.linear().transpose() * Rp;
  const Eigen::Vector3d et = M.linear().transpose() * (pi.tcw - Rp * pj.tcw - M.translation());
  const Eigen::Matrix3d RL = T_C_B_.linear();
  const Eigen::Matrix3d dphi = -dt_ * RightJacobian(-b * dt_);
  Eigen::Matrix<double, 6, 3> jb;
  jb.topRows<3>() = -LogJacobian(E) * E.transpose() * RL * dphi;
  jb.bottomRows<3>() = (Skew(et) - Skew(T_C_B_.translation())) * RL * dphi;
  _jacobianOplus[2] = jb;
  // velocity bias: central differences (translation only, the rotation does not depend on it)
  const auto rel = CameraRelativePose(pi, pj);
  const double h = 1e-6;
  Eigen::Matrix<double, 6, 3> jv;
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d d = Eigen::Vector3d::Zero();
    d[k] = h;
    jv.col(k) = (TransformError(Measured(vb + d), rel) - TransformError(Measured(vb - d), rel)) / (2.0 * h);
  }
  _jacobianOplus[3] = jv;
}

void EdgeLegRelativePose::computeError() {
  const auto* pose_i = static_cast<const VertexVIPose*>(_vertices[0]);
  const auto* pose_j = static_cast<const VertexVIPose*>(_vertices[1]);
  _error = TransformError(_measurement, CameraRelativePose(pose_i->estimate(), pose_j->estimate()));
}

void EdgeLegRelativePose::linearizeOplus() {
  Eigen::Matrix<double, 6, 6> ji, jj;
  RelativePoseJacobians(_measurement, static_cast<const VertexVIPose*>(_vertices[0])->estimate(),
                        static_cast<const VertexVIPose*>(_vertices[1])->estimate(), &ji, &jj);
  _jacobianOplusXi = ji;
  _jacobianOplusXj = jj;
}

EdgeFootSurface::EdgeFootSurface(const Eigen::Isometry3d& T_C_B, const Eigen::Vector3d& foot_b,
                                 const Eigen::Vector3d& q_c, const Eigen::Vector3d& n_c, double radius)
    : T_C_B_(T_C_B), foot_b_(foot_b), q_c_(q_c), n_c_(n_c.normalized()), radius_(radius) {
  resize(3);
}

Eigen::Vector3d EdgeFootSurface::FootInAnchor() const {
  const VIPose& pose_i = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
  const VIPose& anchor = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
  const Eigen::Vector3d p_ci = T_C_B_ * foot_b_;
  const Eigen::Vector3d p_w = pose_i.Rcw.transpose() * (p_ci - pose_i.tcw);
  return anchor.Rcw * p_w + anchor.tcw;
}

void EdgeFootSurface::computeError() {
  const double delta = static_cast<const VertexLegTimeDelay*>(_vertices[2])->estimate();
  _error[0] = n_c_.dot(FootInAnchor() - q_c_) - radius_ + delta;
}

// p_w = Rwb_i p_b + twb_i with p_b = Rcb^T (T_C_B f - tcb); y = Rcb Rwb_j^T (p_w - twb_j) + tcb; r = n^T y.
void EdgeFootSurface::linearizeOplus() {
  const VIPose& pi = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
  const VIPose& pj = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
  const Eigen::Vector3d p_ci = T_C_B_ * foot_b_;
  const Eigen::Vector3d p_b = pi.Rbc * (p_ci - pi.tcb);
  const Eigen::Vector3d p_w = pi.Rcw.transpose() * (p_ci - pi.tcw);
  const Eigen::RowVector3d nj = n_c_.transpose() * pj.Rcw;
  Eigen::Matrix<double, 1, 6> ji, jj;
  ji << -nj * pi.Rwb * Skew(p_b), nj * pi.Rwb;
  jj << n_c_.transpose() * pj.Rcb * Skew(pj.Rwb.transpose() * (p_w - pj.twb)), -n_c_.transpose() * pj.Rcb;
  _jacobianOplus[0] = ji;
  _jacobianOplus[1] = jj;
  _jacobianOplus[2](0, 0) = static_cast<const VertexLegTimeDelay*>(_vertices[2])->OplusSlope();
}
