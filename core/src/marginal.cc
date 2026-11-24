#include "marginal.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace {
typedef Eigen::Matrix<double, 6, 3> Matrix63d;
typedef Eigen::Matrix<double, 3, 6> Matrix36d;

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// One whitened, Huber-weighted observation linearised at (T_w_c, pw). Pose update = right perturbation
// T_w_c (Exp(dr), dt), which equals T_w_c0 M (Exp(dr), dt) for any anchor c0, M = T_c0_c.
struct Linearised {
  int dim = 0;
  Eigen::Matrix<double, 3, 6> Jd;  // rows beyond dim are zero
  Eigen::Matrix3d Jx;
};

bool Linearise(const Eigen::Isometry3d& T_w_c, const Eigen::Vector3d& pw, const Eigen::Vector3d& kp,
               const MargCamera& cam, const MargSettings& s, Linearised* out) {
  const Eigen::Matrix3d Rt = T_w_c.linear().transpose();
  const Eigen::Vector3d pc = Rt * (pw - T_w_c.translation());
  const double z = pc.z();
  if (z <= 0.05) return false;
  const double iz = 1.0 / z, iz2 = iz * iz;
  const bool stereo = kp(2) >= 0.0;
  const int dim = stereo ? 3 : 2;

  Eigen::Vector3d r = Eigen::Vector3d::Zero();
  Eigen::Matrix3d Jp = Eigen::Matrix3d::Zero();  // d residual / d pc
  r(0) = cam.fx * pc.x() * iz + cam.cx - kp(0);
  r(1) = cam.fy * pc.y() * iz + cam.cy - kp(1);
  Jp.row(0) << cam.fx * iz, 0.0, -cam.fx * pc.x() * iz2;
  Jp.row(1) << 0.0, cam.fy * iz, -cam.fy * pc.y() * iz2;
  if (stereo) {
    r(2) = cam.fx * pc.x() * iz + cam.cx - cam.bf * iz - kp(2);
    Jp.row(2) << cam.fx * iz, 0.0, (cam.bf - cam.fx * pc.x()) * iz2;
  }

  const double chi2 = r.squaredNorm() / (s.pixel_sigma * s.pixel_sigma);
  if (chi2 > s.max_chi2) return false;
  const double norm = std::sqrt(chi2);
  const double w = norm <= s.huber ? 1.0 : s.huber / norm;
  const double scale = std::sqrt(w) / s.pixel_sigma;  // whitening and IRLS weight

  // pc(d) = Exp(-dr) (pc - dt): d pc / d dr = [pc]x, d pc / d dt = -I; d pc / d pw = R^T.
  Matrix36d Jpd;
  Jpd.leftCols<3>() = Skew(pc);
  Jpd.rightCols<3>() = -Eigen::Matrix3d::Identity();
  out->dim = dim;
  out->Jd = scale * Jp * Jpd;
  out->Jx = scale * Jp * Rt;
  return true;
}

// Point-wise Schur elimination onto the poses 1..m-1 (pose 0 = gauge). With `pair`, a point must be observed in
// both slot 0 and slot 1 (others ignored); otherwise in >= 2 distinct slots.
bool Marginalise(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& T_w_c,
                 const MargPoints& points, const MargCamera& cam, const MargSettings& s, bool pair,
                 Eigen::MatrixXd* info, int* used_points) {
  const int m = static_cast<int>(T_w_c.size());
  const int n = 6 * (m - 1);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  int used = 0;

  std::vector<Linearised, Eigen::aligned_allocator<Linearised>> lin;
  std::vector<int> slot;
  std::vector<Matrix63d, Eigen::aligned_allocator<Matrix63d>> Hdx(m);
  std::vector<int> touched;
  for (const MargPoint& p : points) {
    lin.clear();
    slot.clear();
    std::vector<bool> seen(m, false);
    for (const MargObservation& o : p.obs) {
      if (o.slot < 0 || o.slot >= m) continue;
      Linearised l;
      if (!Linearise(T_w_c[o.slot], p.pw, o.kp, cam, s, &l)) continue;
      lin.push_back(l);
      slot.push_back(o.slot);
      seen[o.slot] = true;
    }
    int distinct = 0;
    for (int k = 0; k < m; ++k) distinct += seen[k];
    if (lin.size() < 2 || distinct < 2 || (pair && !(seen[0] && seen[1]))) continue;

    // Point block and the pose-point coupling.
    Eigen::Matrix3d Hxx = Eigen::Matrix3d::Zero();
    touched.clear();
    for (int k = 1; k < m; ++k) Hdx[k].setZero();
    for (size_t i = 0; i < lin.size(); ++i) {
      const int d = lin[i].dim;
      const Eigen::Matrix3d Jx = lin[i].Jx.topRows(d).transpose() * lin[i].Jx.topRows(d);
      Hxx += Jx;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Hxx, Eigen::EigenvaluesOnly);
    if (es.eigenvalues()(0) < 1e-9) continue;
    for (size_t i = 0; i < lin.size(); ++i) {
      const int k = slot[i];
      if (k == 0) continue;  // anchor: depends on the point only
      const int d = lin[i].dim;
      const auto Jd = lin[i].Jd.topRows(d);
      const auto Jx = lin[i].Jx.topRows(d);
      H.block<6, 6>(6 * (k - 1), 6 * (k - 1)) += Jd.transpose() * Jd;
      Hdx[k] += Jd.transpose() * Jx;
      if (std::find(touched.begin(), touched.end(), k) == touched.end()) touched.push_back(k);
    }

    // H -= H_dX H_XX^-1 H_Xd.
    const Eigen::Matrix3d Hxx_inv = Hxx.inverse();
    for (int a : touched) {
      const Matrix63d G = Hdx[a] * Hxx_inv;
      for (int b : touched) H.block<6, 6>(6 * (a - 1), 6 * (b - 1)) -= G * Hdx[b].transpose();
    }
    ++used;
  }

  if (used_points) *used_points = used;
  *info = 0.5 * (H + H.transpose());
  return used >= s.min_points;
}
}  // namespace

bool PairMarginal(const Eigen::Isometry3d& T_w_c0, const Eigen::Isometry3d& T_w_c1, const MargPoints& points,
                  const MargCamera& camera, const MargSettings& settings, Matrix6d* info, int* used_points) {
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> T_w_c = {T_w_c0, T_w_c1};
  Eigen::MatrixXd H;
  const bool ok = Marginalise(T_w_c, points, camera, settings, true, &H, used_points);
  *info = H;
  return ok;
}
