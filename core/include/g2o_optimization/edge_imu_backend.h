#ifndef EDGE_IMU_BACKEND_H_
#define EDGE_IMU_BACKEND_H_

// IMU preintegration factor of the S3 online back-end (BackendConfig imu).
//
// Vertices: [0] VertexVIPose i, [1] VertexVelocity i, [2] VertexGyrBias, [3] VertexAccBias (the bias block of the
// interval), [4] VertexVIPose j, [5] VertexVelocity j, [6] VertexGDirection (world gravity direction), [7] VertexVec<3>
// d (small rotation of the IMU axes relative to the configured camera-IMU extrinsic, rad; edge_leg_odom.h).
// The VIPose body frame is the IMU of the configured extrinsic (Rcb / tcb of the pose vertices). The corrected IMU frame
// is B' = B Exp(d): measured increments are expressed in B', so the prediction uses
//   R' = Rwb Exp(d),  dR' = Exp(d)^T dR Exp(d) ... i.e. the residual of EdgeIMU (edge_imu.cc) with Rwb_i, Rwb_j replaced
// by Rwb_i Exp(d), Rwb_j Exp(d) (positions / velocities unchanged: the lever arm is neglected for the small d).
// Error [er, ev, ep] (9), information = the PSD-clipped inverse of the preintegration covariance Cov(0:9) (as EdgeIMU).
// Analytic Jacobians (linearizeOplus; checked against central differences of the vertex oplus).

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <g2o/core/base_multi_edge.h>

#include "imu.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/edge_leg_odom.h"

class EdgeIMUBackend : public g2o::BaseMultiEdge<9, Vector9d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit EdgeIMUBackend(PreinterationPtr preintegration);
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  void computeError() override;
  using g2o::BaseMultiEdge<9, Vector9d>::linearizeOplus;   // keep linearizeOplus(JacobianWorkspace&) visible
  void linearizeOplus() override;
  PreinterationPtr preintegration;
  Eigen::Vector3d g;   // (0, 0, -IMU_G_VALUE)
};

struct ImuNoise {
  double gyr_noise = 0.0, acc_noise = 0.0, gyr_walk = 0.0, acc_walk = 0.0;   // as Camera::GyrNoise() etc. (discretised at the IMU rate)
};

// Preintegration of the IMU samples over [t0, t1] (sample times shifted by time_offset: t_imu + time_offset is the camera
// time), linearised at the biases (bg, ba). The samples must bracket [t0, t1] (Preinteration::AddBatchData
// interpolates at the ends). nullptr when they do not cover the interval or the integration fails.
PreinterationPtr BuildPreintegration(const std::vector<ImuData>& samples, double t0, double t1, double time_offset,
                                     const ImuNoise& noise, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

#endif  // EDGE_IMU_BACKEND_H_
