#include "g2o_optimization/edge_imu_backend.h"

#include <Eigen/Eigenvalues>

#include "camera.h"

namespace {
Eigen::Matrix3d Sk(const Eigen::Vector3d& v) { Eigen::Matrix3d m; Hat(m, v); return m; }
// right Jacobian of SO3 and its inverse
Eigen::Matrix3d Jr(const Eigen::Vector3d& v) {
  const double t2 = v.squaredNorm(), t = std::sqrt(t2);
  const Eigen::Matrix3d W = Sk(v);
  if (t < 1e-5) return Eigen::Matrix3d::Identity() - 0.5 * W + W * W / 6;
  return Eigen::Matrix3d::Identity() - (1 - std::cos(t)) / t2 * W + (t - std::sin(t)) / (t2 * t) * W * W;
}
Eigen::Matrix3d JrInv(const Eigen::Vector3d& v) {
  const double t2 = v.squaredNorm(), t = std::sqrt(t2);
  const Eigen::Matrix3d W = Sk(v);
  if (t < 1e-5) return Eigen::Matrix3d::Identity() + 0.5 * W + W * W / 12;
  return Eigen::Matrix3d::Identity() + 0.5 * W + (1 / t2 - (1 + std::cos(t)) / (2 * t * std::sin(t))) * W * W;
}
}  // namespace

EdgeIMUBackend::EdgeIMUBackend(PreinterationPtr preintegration_) : preintegration(preintegration_) {
  resize(8);
  g << 0, 0, -Camera::IMU_G_VALUE;
  // PSD-clipped inverse of the preintegration covariance (as EdgeIMU)
  Matrix9d info = preintegration->Cov.block<9, 9>(0, 0).cast<double>().inverse();
  info = (info + info.transpose()) / 2;
  Eigen::SelfAdjointEigenSolver<Matrix9d> es(info);
  Vector9d eigs = es.eigenvalues();
  for (int i = 0; i < 9; i++) if (eigs[i] < 1e-12) eigs[i] = 0;
  info = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
  setInformation(info);
}

void EdgeIMUBackend::computeError() {
  const VertexVIPose* vp1 = static_cast<const VertexVIPose*>(_vertices[0]);
  const VertexVelocity* vv1 = static_cast<const VertexVelocity*>(_vertices[1]);
  const VertexGyrBias* vbg = static_cast<const VertexGyrBias*>(_vertices[2]);
  const VertexAccBias* vba = static_cast<const VertexAccBias*>(_vertices[3]);
  const VertexVIPose* vp2 = static_cast<const VertexVIPose*>(_vertices[4]);
  const VertexVelocity* vv2 = static_cast<const VertexVelocity*>(_vertices[5]);
  const VertexGDirection* vg = static_cast<const VertexGDirection*>(_vertices[6]);
  const VertexVec<3>* vd = static_cast<const VertexVec<3>*>(_vertices[7]);

  const Eigen::Vector3d bg = vbg->estimate(), ba = vba->estimate();
  const Eigen::Matrix3d dR = preintegration->GetDeltaRotation(bg);
  const Eigen::Vector3d dV = preintegration->GetDeltaVelocity(bg, ba);
  const Eigen::Vector3d dP = preintegration->GetDeltaPosition(bg, ba);

  // corrected IMU axes B' = B Exp(d)
  Eigen::Matrix3d Rd;
  SO3Exp(vd->estimate(), Rd);
  const Eigen::Matrix3d Ri = vp1->estimate().Rwb * Rd, Rj = vp2->estimate().Rwb * Rd;

  const Eigen::Vector3d gw = vg->estimate().Rwg * g;
  const double dt = preintegration->dT;
  Eigen::Vector3d er;
  SO3Log(dR.transpose() * Ri.transpose() * Rj, er);
  const Eigen::Vector3d ev = Ri.transpose() * (vv2->estimate() - vv1->estimate() - gw * dt) - dV;
  const Eigen::Vector3d ep =
      Ri.transpose() * (vp2->estimate().twb - vp1->estimate().twb - vv1->estimate() * dt - gw * dt * dt / 2) - dP;
  _error << er, ev, ep;
}

// Pose: Rwb <- Rwb Exp(dr), twb <- twb + Rwb dt; d: Exp(d + x) = Exp(d) Exp(Jr(d) x); g: Rwg <- Rwg Exp([x0, x1, 0]).
// With E = dR^T Ri^T Rj = Exp(er), a perturbation Ri <- Ri Exp(a) gives er += -Jr^-1(er) Rj^T Ri a, Rj <- Rj Exp(a)
// gives er += Jr^-1(er) a, and Ri^T x <- Ri^T x + [Ri^T x]x a.
void EdgeIMUBackend::linearizeOplus() {
  const VIPose& pi = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
  const VIPose& pj = static_cast<const VertexVIPose*>(_vertices[4])->estimate();
  const Eigen::Vector3d& vi = static_cast<const VertexVelocity*>(_vertices[1])->estimate();
  const Eigen::Vector3d& vj = static_cast<const VertexVelocity*>(_vertices[5])->estimate();
  const Eigen::Vector3d bg = static_cast<const VertexGyrBias*>(_vertices[2])->estimate();
  const Eigen::Matrix3d& Rwg = static_cast<const VertexGDirection*>(_vertices[6])->estimate().Rwg;
  const Eigen::Vector3d& d = static_cast<const VertexVec<3>*>(_vertices[7])->estimate();
  Preinteration& p = *preintegration;
  const double dt = p.dT;

  Eigen::Matrix3d Rd;
  SO3Exp(d, Rd);
  const Eigen::Matrix3d Ri = pi.Rwb * Rd, Rj = pj.Rwb * Rd, RiT = Ri.transpose();
  const Eigen::Matrix3d RjTRi = Rj.transpose() * Ri;
  const Eigen::Vector3d gw = Rwg * g;
  const Eigen::Matrix3d dR = p.GetDeltaRotation(bg);
  const Eigen::Matrix3d E = dR.transpose() * RiT * Rj;
  Eigen::Vector3d er;
  SO3Log(E, er);
  const Eigen::Matrix3d Jri = JrInv(er), Jd = Jr(d), RdT = Rd.transpose();
  const Eigen::Vector3d u = RiT * (vj - vi - gw * dt);                                // ev + dV
  const Eigen::Vector3d w = RiT * (pj.twb - pi.twb - vi * dt - gw * dt * dt / 2);     // ep + dP
  const Eigen::Matrix3d Su = Sk(u), Sw = Sk(w);
  const Eigen::Matrix3d Gg = RiT * Rwg * Sk(g);                                      // d(-Ri^T gw)/dx

  auto J = [&](int k) -> JacobianType& { _jacobianOplus[k].setZero(); return _jacobianOplus[k]; };
  auto active = [&](int k) { return !static_cast<const g2o::OptimizableGraph::Vertex*>(_vertices[k])->fixed(); };
  if (active(0)) {   // pose i [dr, dt]
    JacobianType& A = J(0);
    A.block<3, 3>(0, 0) = -Jri * RjTRi * RdT;
    A.block<3, 3>(3, 0) = Su * RdT;
    A.block<3, 3>(6, 0) = Sw * RdT;
    A.block<3, 3>(6, 3) = -RdT;
  }
  if (active(1)) {   // v i
    JacobianType& A = J(1);
    A.block<3, 3>(3, 0) = -RiT;
    A.block<3, 3>(6, 0) = -RiT * dt;
  }
  if (active(2)) {   // bg: dR(bg) = dR0 Exp(JRg (bg - bg0))
    JacobianType& A = J(2);
    A.block<3, 3>(0, 0) = -Jri * E.transpose() * Jr(p.JRg * (bg - p.bg)) * p.JRg;
    A.block<3, 3>(3, 0) = -p.JVg;
    A.block<3, 3>(6, 0) = -p.JPg;
  }
  if (active(3)) {   // ba
    JacobianType& A = J(3);
    A.block<3, 3>(3, 0) = -p.JVa;
    A.block<3, 3>(6, 0) = -p.JPa;
  }
  if (active(4)) {   // pose j [dr, dt]
    JacobianType& A = J(4);
    A.block<3, 3>(0, 0) = Jri * RdT;
    A.block<3, 3>(6, 3) = RiT * pj.Rwb;
  }
  if (active(5)) J(5).block<3, 3>(3, 0) = RiT;   // v j
  if (active(6)) {   // gravity direction (2)
    JacobianType& A = J(6);
    A.block<3, 2>(3, 0) = dt * Gg.leftCols<2>();
    A.block<3, 2>(6, 0) = dt * dt / 2 * Gg.leftCols<2>();
  }
  if (active(7)) {   // d: Ri, Rj <- Ri Exp(b), Rj Exp(b), b = Jr(d) x
    JacobianType& A = J(7);
    A.block<3, 3>(0, 0) = Jri * (Eigen::Matrix3d::Identity() - RjTRi) * Jd;
    A.block<3, 3>(3, 0) = Su * Jd;
    A.block<3, 3>(6, 0) = Sw * Jd;
  }
}

PreinterationPtr BuildPreintegration(const std::vector<ImuData>& samples, double t0, double t1, double time_offset,
                                     const ImuNoise& noise, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
  if (!(t1 > t0) || samples.size() < 2) return nullptr;
  // samples on the camera clock; AddBatchData needs strictly increasing stamps: drop duplicates / out-of-order
  ImuDataList shifted;
  shifted.reserve(samples.size());
  for (const ImuData& s : samples) {
    ImuData m = s;
    m.timestamp += time_offset;
    if (shifted.empty() || m.timestamp > shifted.back().timestamp) shifted.push_back(m);
  }
  // fresh object (end_time = -1: no continuity check), noises passed as the front-end does (Camera getters,
  // SetNoiseAndWalk squares them), linearisation bias set without repropagation (nothing integrated yet)
  PreinterationPtr p = std::make_shared<Preinteration>();
  p->SetNoiseAndWalk(noise.gyr_noise, noise.acc_noise, noise.gyr_walk, noise.acc_walk);
  p->SetBias(bg, ba, false);
  if (!p->AddBatchData(shifted, t0, t1) || !p->Valid() || p->dT <= 0) return nullptr;
  return p;
}
