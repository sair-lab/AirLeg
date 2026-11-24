#ifndef MARGINAL_H_
#define MARGINAL_H_

// Visual information of map points marginalised onto keyframe poses : the S3 back-end
// replaces its fixed-sigma visual relative-pose edge by these factors (BackendConfig visual_marg).
//
// Pose error convention = TransformError of edge_leg_odom.cc (EdgeLegRelativePose): for a measured relative pose M =
// T_ci_cj (pose of camera j in camera i) and a predicted P, e = [Log(R_M^T R_P), R_M^T (t_P - t_M)] (rotation first).
// So P = M * (Exp(dr), dt) has e = (dr, dt). All poses are camera-to-world T_w_c (Eigen::Isometry3d).
//
// Pixel model: reprojection residuals in pixels, isotropic sigma `pixel_sigma`; a stereo observation (kp(2) >= 0, the
// right u) adds the right-image u residual (u_right = u - bf / z). Huber IRLS weight at the linearisation point
// (whitened residual norm > huber: weight huber / norm); points behind a camera or with a whitened residual beyond
// `max_chi2` are skipped. Points are eliminated by the Schur complement (FEJ: everything linearised at the given poses
// and point positions).

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

struct MargCamera {
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0, bf = 0.0;
};

struct MargObservation {
  int slot = 0;                // index into the pose list of the call
  Eigen::Vector3d kp;          // u, v, u_right (-1: mono)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MargPoint {
  Eigen::Vector3d pw;          // world position (linearisation point)
  std::vector<MargObservation, Eigen::aligned_allocator<MargObservation>> obs;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::vector<MargPoint, Eigen::aligned_allocator<MargPoint>> MargPoints;

struct MargSettings {
  double pixel_sigma = 1.0;
  double huber = 2.5;          // on the whitened residual norm of one observation
  double max_chi2 = 100.0;     // observation skipped beyond this whitened squared residual
  int min_points = 10;         // fewer usable points: no factor (return false)
};

typedef Eigen::Matrix<double, 6, 6> Matrix6d;

// Information (6x6, in the TransformError coordinates of T_ci_cj) of the relative pose between pose 0 (slot 0) and
// pose 1 (slot 1) from the points (each with observations in slot 0 and/or 1; points seen in one slot only add
// nothing). Pose 0 is the gauge (held fixed). False when fewer than settings.min_points points constrain both.
bool PairMarginal(const Eigen::Isometry3d& T_w_c0, const Eigen::Isometry3d& T_w_c1, const MargPoints& points,
                  const MargCamera& camera, const MargSettings& settings, Matrix6d* info, int* used_points = nullptr);

#endif  // MARGINAL_H_
