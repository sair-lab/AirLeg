#ifndef AIR_SLAM_EDGE_LEG_ODOM_H_
#define AIR_SLAM_EDGE_LEG_ODOM_H_

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>

#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "leg_odom.h"
#include "utils.h"

class VertexLegExtrinsic : public g2o::BaseVertex<6, Eigen::Isometry3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void setToOriginImpl() override { _estimate.setIdentity(); }
  void oplusImpl(const double* update) override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class VertexLegTimeDelay : public g2o::BaseVertex<1, double> {
 public:
  explicit VertexLegTimeDelay(double lower = -0.2, double upper = 0.2)
      : lower_(lower), upper_(upper) {}
  void setToOriginImpl() override { _estimate = 0.0; }
  void oplusImpl(const double* update) override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  // d estimate / d update of the clamped oplus, as g2o's central difference (step 1e-9) sees it.
  double OplusSlope() const;

 private:
  double lower_;
  double upper_;
};

// Additive body-twist bias ordered as [vx, vy, vz, wx, wy, wz].
class VertexLegBias : public g2o::BaseVertex<6, Vector6d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void setToOriginImpl() override { _estimate.setZero(); }
  void oplusImpl(const double* update) override {
    _estimate += Eigen::Map<const Vector6d>(update);
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// Vertices: camera pose i, camera pose j, T_C_B, time delay, interval bias.
class EdgeLegOdom : public g2o::BaseMultiEdge<6, Vector6d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLegOdom(std::shared_ptr<const LegOdomBuffer> buffer,
              double camera_time_i, double camera_time_j);
  void SetFixedBodyDelta(const Eigen::Isometry3d& delta) {
    fixed_body_delta_ = delta;
    has_fixed_body_delta_ = true;
  }
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 private:
  std::shared_ptr<const LegOdomBuffer> buffer_;
  double camera_time_i_;
  double camera_time_j_;
  bool has_fixed_body_delta_ = false;
  Eigen::Isometry3d fixed_body_delta_ = Eigen::Isometry3d::Identity();
};

// Vertices: camera pose i, camera pose j, gyro bias i, gyro bias j.
// First-order bias update of a fixed integration: the interval bias is the
// endpoint mean, linearized at the bias the velocity CSV was computed with.
class EdgeLegOdomGyrBias : public g2o::BaseMultiEdge<6, Vector6d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLegOdomGyrBias(const Eigen::Isometry3d& T_C_B, const Eigen::Isometry3d& body_delta,
                     const Eigen::Matrix<double, 6, 3>& bias_jacobian,
                     const Eigen::Vector3d& linearization_bias);
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 private:
  Eigen::Isometry3d T_C_B_, body_delta_;
  Eigen::Matrix<double, 6, 3> bias_jacobian_;
  Eigen::Vector3d linearization_bias_;
};

class EdgeGyrBiasPrior
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexGyrBias> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class EdgeLegBiasRandomWalk
    : public g2o::BaseBinaryEdge<6, Vector6d, VertexLegBias, VertexLegBias> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class EdgeLegBiasPrior
    : public g2o::BaseUnaryEdge<6, Vector6d, VertexLegBias> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class EdgeLegExtrinsicPrior
    : public g2o::BaseUnaryEdge<6, Eigen::Isometry3d, VertexLegExtrinsic> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class EdgeLegTimeDelayPrior
    : public g2o::BaseUnaryEdge<1, double, VertexLegTimeDelay> {
 public:
  void computeError() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// Random walk between two scalar states (e.g. consecutive foot contact offsets): r = x_j - x_i - z.
class EdgeScalarRandomWalk
    : public g2o::BaseBinaryEdge<1, double, VertexLegTimeDelay, VertexLegTimeDelay> {
 public:
  void computeError() override {
    _error[0] = static_cast<const VertexLegTimeDelay*>(_vertices[1])->estimate() -
                static_cast<const VertexLegTimeDelay*>(_vertices[0])->estimate() - _measurement;
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// Additive vector state (e.g. back-end gyro bias blocks) with prior and random-walk edges.
template <int N>
class VertexVec : public g2o::BaseVertex<N, Eigen::Matrix<double, N, 1>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void setToOriginImpl() override { this->_estimate.setZero(); }
  void oplusImpl(const double* update) override { this->_estimate += Eigen::Map<const Eigen::Matrix<double, N, 1>>(update); }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};
template <int N>
class EdgeVecPrior : public g2o::BaseUnaryEdge<N, Eigen::Matrix<double, N, 1>, VertexVec<N>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override { this->_error = static_cast<const VertexVec<N>*>(this->_vertices[0])->estimate() - this->_measurement; }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};
template <int N>
class EdgeVecRandomWalk : public g2o::BaseBinaryEdge<N, Eigen::Matrix<double, N, 1>, VertexVec<N>, VertexVec<N>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override {
    this->_error = static_cast<const VertexVec<N>*>(this->_vertices[1])->estimate() -
                   static_cast<const VertexVec<N>*>(this->_vertices[0])->estimate() - this->_measurement;
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// S3 back-end: leg relative pose with a residual gyro bias b (rad/s, base frame B) on top of the leg node's
// own bias. Vertices: pose i, pose j, bias block. Measurement: body motion T_Bi_Bj over dt;
// prediction T_C_B (R_b Exp(-b dt), t_b) T_C_B^-1, residual as EdgeLegRelativePose.
class EdgeLegGyroBias : public g2o::BaseMultiEdge<6, Eigen::Isometry3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLegGyroBias(const Eigen::Isometry3d& T_C_B, double dt) : T_C_B_(T_C_B), dt_(dt) { resize(3); }
  void computeError() override;
  using g2o::BaseMultiEdge<6, Eigen::Isometry3d>::linearizeOplus;
  void linearizeOplus() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 private:
  Eigen::Isometry3d T_C_B_;
  double dt_;
};

// S3 back-end: visual relative pose whose translation is scaled by (1 + v), v = the visual translation
// scale offset of a block of nodes (VertexVec<1>). Vertices: pose i, pose j, scale block. Residual as
// EdgeLegRelativePose with the measurement (R_m, (1 + v) t_m).
class EdgeScaledRelativePose : public g2o::BaseMultiEdge<6, Eigen::Isometry3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeScaledRelativePose() { resize(3); }
  void computeError() override;
  using g2o::BaseMultiEdge<6, Eigen::Isometry3d>::linearizeOplus;
  void linearizeOplus() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// EdgeLegGyroBias with a fourth vertex (backend leg_vbias), a linear-velocity bias v_b (base frame, m/s, constant
// over the interval): body motion (R_b Exp(-b dt), t_b - 0.5 (I + R_b) v_b dt). Pose / gyro-bias Jacobians as
// EdgeLegGyroBias, velocity-bias Jacobian numeric.
class EdgeLegGyroBiasVel : public g2o::BaseMultiEdge<6, Eigen::Isometry3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeLegGyroBiasVel(const Eigen::Isometry3d& T_C_B, double dt) : T_C_B_(T_C_B), dt_(dt) { resize(4); }
  void computeError() override;
  using g2o::BaseMultiEdge<6, Eigen::Isometry3d>::linearizeOplus;
  void linearizeOplus() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 private:
  Eigen::Isometry3d Measured(const Eigen::Vector3d& vb) const;
  Eigen::Isometry3d T_C_B_;
  double dt_;
};

// S3: fixed leg relative-pose measurement between camera poses i and j.
// Measurement T_Ci_Cj = T_C_B * T_Bi_Bj * T_C_B^-1; residual convention matches
// vce::Integrate's residual_covariance, so information = covariance^-1.
class EdgeLegRelativePose
    : public g2o::BaseBinaryEdge<6, Eigen::Isometry3d, VertexVIPose, VertexVIPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void computeError() override;
  using g2o::BaseBinaryEdge<6, Eigen::Isometry3d, VertexVIPose, VertexVIPose>::linearizeOplus;
  void linearizeOplus() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

// S3 foot-surface factor. Vertices: pose i (constrained), anchor superframe j, contact offset delta.
// r = n^T (T_Cj_W T_W_Ci T_C_B f_B - q) - a + delta, with the plane (q, n) in the anchor camera frame Cj.
// delta > 0 is sinkage into the surface, delta < 0 a foot resting above the fitted mean plane.
// Observability: on level ground r ~ z_foot - z_ground + delta, so lowering the trajectory and
// increasing delta cancel; delta is a shared, bounded, prior-regularised "effective normal offset".
class EdgeFootSurface : public g2o::BaseMultiEdge<1, double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeFootSurface(const Eigen::Isometry3d& T_C_B, const Eigen::Vector3d& foot_b,
                  const Eigen::Vector3d& q_c, const Eigen::Vector3d& n_c, double radius);
  void computeError() override;
  using g2o::BaseMultiEdge<1, double>::linearizeOplus;
  void linearizeOplus() override;
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  // Foot point expressed in the anchor camera frame (diagnostics/support checks).
  Eigen::Vector3d FootInAnchor() const;

 private:
  Eigen::Isometry3d T_C_B_;
  Eigen::Vector3d foot_b_, q_c_, n_c_;
  double radius_;
};

#endif  // AIR_SLAM_EDGE_LEG_ODOM_H_
