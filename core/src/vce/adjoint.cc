// Implicit hypergradient through the fusion BA: see vce/adjoint.h.
#include "vce/adjoint.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "vce/covariance.h"

namespace vce {
namespace {
Eigen::VectorXd ErrorOf(g2o::OptimizableGraph::Edge* edge) {
  edge->computeError();
  return Eigen::Map<const Eigen::VectorXd>(edge->errorData(), edge->dimension());
}
Eigen::MatrixXd InformationOf(g2o::OptimizableGraph::Edge* edge) {
  const int d = edge->dimension();
  return Eigen::Map<const Eigen::MatrixXd>(edge->informationData(), d, d);
}
}  // namespace

Eigen::MatrixXd NumericJacobian(g2o::OptimizableGraph::Edge* edge, g2o::OptimizableGraph::Vertex* vertex, double eps) {
  const int d = edge->dimension(), n = vertex->dimension();
  Eigen::MatrixXd J(d, n);
  for (int c = 0; c < n; ++c) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx[c] = eps;
    vertex->push(); vertex->oplus(dx.data());
    const Eigen::VectorXd plus = ErrorOf(edge);
    vertex->pop();
    dx[c] = -eps;
    vertex->push(); vertex->oplus(dx.data());
    const Eigen::VectorXd minus = ErrorOf(edge);
    vertex->pop();
    J.col(c) = (plus - minus) / (2 * eps);
  }
  edge->computeError();
  return J;
}

double AdjointVector::Norm() const {
  double s = 0;
  for (const auto& v : pose) s += v.squaredNorm();
  for (const auto& v : point) s += v.squaredNorm();
  return std::sqrt(s);
}

AdjointSystem::AdjointSystem(const AdjointGraph& graph, bool exact) : graph_(graph), exact_(exact) {
  pose_index_.assign(graph.poses.size(), -1);
  for (size_t i = 0; i < graph.poses.size(); ++i)
    if (!graph.poses[i]->fixed()) pose_index_[i] = free_poses_++;
  for (const auto* p : graph.points)
    if (p->fixed()) throw std::runtime_error("adjoint: fixed points are not supported");
}

int AdjointSystem::PoseIndex(const VertexVIPose* v) const {
  for (size_t i = 0; i < graph_.poses.size(); ++i) if (graph_.poses[i] == v) return pose_index_[i];
  return -1;
}

AdjointVector AdjointSystem::Zero() const {
  AdjointVector v;
  v.pose.assign(free_poses_, Vec6::Zero());
  v.point.assign(graph_.points.size(), Eigen::Vector3d::Zero());
  return v;
}

void AdjointSystem::Linearize() {
  // vertex -> (kind 0 pose / 1 point, index); fixed vertices are absent
  std::unordered_map<const g2o::OptimizableGraph::Vertex*, std::pair<int, int>> index;
  for (size_t i = 0; i < graph_.poses.size(); ++i)
    if (pose_index_[i] >= 0) index[graph_.poses[i]] = {0, pose_index_[i]};
  for (size_t i = 0; i < graph_.points.size(); ++i) index[graph_.points[i]] = {1, static_cast<int>(i)};
  Hpp_ = Eigen::MatrixXd::Zero(6 * free_poses_, 6 * free_poses_);
  Hll_.assign(graph_.points.size(), Eigen::Matrix3d::Zero());
  Hpl_.assign(graph_.points.size(), {});
  gradient_ = Zero();
  AdjointVector colnorm = Zero();
  double cost = 0;
  std::vector<g2o::OptimizableGraph::Edge*> edges(graph_.visual);
  for (const auto& leg : graph_.legs) edges.push_back(leg.edge);
  edges.insert(edges.end(), graph_.others.begin(), graph_.others.end());
  for (auto* edge : edges) {
    const Eigen::VectorXd e = ErrorOf(edge);
    const Eigen::MatrixXd info = InformationOf(edge);
    cost += e.dot(info * e);
    Eigen::LLT<Eigen::MatrixXd> llt(info);
    if (llt.info() != Eigen::Success) throw std::runtime_error("adjoint: non-positive edge information");
    const Eigen::MatrixXd W = llt.matrixU();
    struct Block { int kind, idx; Eigen::MatrixXd J; };
    std::vector<Block> blocks;
    for (auto* raw : edge->vertices()) {
      auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(raw);
      if (v->fixed()) continue;
      auto it = index.find(v);
      if (it == index.end()) throw std::runtime_error("adjoint: edge has a free vertex outside the graph");
      blocks.push_back({it->second.first, it->second.second, NumericJacobian(edge, v)});
    }
    // exact: Hessian columns of this edge's cost by central differences of its gradient [J_b^T Omega e]_b
    std::vector<std::vector<Eigen::MatrixXd>> exact_h;   // [a][b]: d(grad_b)/d(x_a)^T, i.e. H_ba
    if (exact_) {
      std::vector<g2o::OptimizableGraph::Vertex*> verts;
      for (auto* raw : edge->vertices()) {
        auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(raw);
        if (!v->fixed()) verts.push_back(v);
      }
      auto stacked = [&]() {
        std::vector<Eigen::VectorXd> out;
        const Eigen::VectorXd ee = ErrorOf(edge);
        for (auto* v : verts) out.push_back(NumericJacobian(edge, v).transpose() * (info * ee));
        return out;
      };
      constexpr double h2 = 1e-5;
      exact_h.assign(verts.size(), std::vector<Eigen::MatrixXd>(verts.size()));
      for (size_t a = 0; a < verts.size(); ++a) {
        const int n = verts[a]->dimension();
        for (size_t b = 0; b < verts.size(); ++b) exact_h[a][b] = Eigen::MatrixXd::Zero(verts[b]->dimension(), n);
        for (int col = 0; col < n; ++col) {
          Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
          dx[col] = h2;
          verts[a]->push(); verts[a]->oplus(dx.data());
          const auto plus = stacked();
          verts[a]->pop();
          dx[col] = -h2;
          verts[a]->push(); verts[a]->oplus(dx.data());
          const auto minus = stacked();
          verts[a]->pop();
          for (size_t b = 0; b < verts.size(); ++b) exact_h[a][b].col(col) = (plus[b] - minus[b]) / (2 * h2);
        }
      }
      edge->computeError();
    }
    for (size_t ia = 0; ia < blocks.size(); ++ia) {
      const auto& a = blocks[ia];
      const Eigen::VectorXd ga = a.J.transpose() * (info * e);
      const Eigen::VectorXd ca = (W * a.J).colwise().squaredNorm().transpose();
      if (a.kind == 0) { gradient_.pose[a.idx] += ga; colnorm.pose[a.idx] += ca; }
      else { gradient_.point[a.idx] += ga; colnorm.point[a.idx] += ca; }
      for (size_t ib = 0; ib < blocks.size(); ++ib) {
        const auto& b = blocks[ib];
        // H_ab (rows a, columns b); exact: symmetrised numeric block
        const Eigen::MatrixXd h = exact_ ? Eigen::MatrixXd(0.5 * (exact_h[ib][ia] + exact_h[ia][ib].transpose()))
                                         : Eigen::MatrixXd(a.J.transpose() * info * b.J);
        if (a.kind == 0 && b.kind == 0) Hpp_.block(6 * a.idx, 6 * b.idx, 6, 6) += h;
        else if (a.kind == 1 && b.kind == 1) {
          if (a.idx != b.idx) throw std::runtime_error("adjoint: point-point coupling unsupported");
          Hll_[a.idx] += h;
        } else if (a.kind == 0 && b.kind == 1) {
          auto& list = Hpl_[b.idx];
          auto c = std::find_if(list.begin(), list.end(), [&](const Coupling& x) { return x.pose == a.idx; });
          if (c == list.end()) list.push_back({a.idx, h});
          else c->block += h;
        }
      }
    }
  }
  // exact mode: a few weakly observed points with large residuals can have a slightly indefinite exact block;
  // clip their eigenvalues to 1e-3 of the largest (counted).
  clipped_points_ = 0;
  for (auto& h : Hll_) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(0.5 * (h + h.transpose()));
    const Eigen::Vector3d ev = es.eigenvalues();
    if (ev[0] > 1e-9 * std::max(1.0, ev[2])) continue;
    ++clipped_points_;
    const Eigen::Vector3d fixed = ev.cwiseMax(1e-3 * std::max(ev[2], 1e-12));
    h = es.eigenvectors() * fixed.asDiagonal() * es.eigenvectors().transpose();
  }
  stationarity_.cost = cost;
  stationarity_.gradient_norm = gradient_.Norm();
  double scaled = 0;
  auto update = [&](const Eigen::VectorXd& g, const Eigen::VectorXd& c) {
    for (int i = 0; i < g.size(); ++i) scaled = std::max(scaled, std::abs(g[i]) / (std::sqrt(c[i]) + 1e-30));
  };
  for (int i = 0; i < free_poses_; ++i) update(gradient_.pose[i], colnorm.pose[i]);
  for (size_t i = 0; i < graph_.points.size(); ++i) update(gradient_.point[i], colnorm.point[i]);
  stationarity_.scaled_max = scaled / std::max(1.0, std::sqrt(cost));
}

AdjointVector AdjointSystem::Solve(const AdjointVector& rhs) const {
  const int n = 6 * free_poses_;
  Eigen::MatrixXd S = Hpp_;
  Eigen::VectorXd r(n);
  for (int i = 0; i < free_poses_; ++i) r.segment<6>(6 * i) = rhs.pose[i];
  std::vector<Eigen::Matrix3d> inv(graph_.points.size());
  for (size_t l = 0; l < graph_.points.size(); ++l) {
    Eigen::LDLT<Eigen::Matrix3d> ldlt(Hll_[l]);
    inv[l] = ldlt.solve(Eigen::Matrix3d::Identity());
    for (const auto& a : Hpl_[l]) {
      const Eigen::Matrix<double, 6, 3> BaHinv = a.block * inv[l];
      r.segment<6>(6 * a.pose) -= BaHinv * rhs.point[l];
      for (const auto& b : Hpl_[l]) S.block(6 * a.pose, 6 * b.pose, 6, 6) -= BaHinv * b.block.transpose();
    }
  }
  AdjointVector x = Zero();
  if (n > 0) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (S + S.transpose()));
    if (ldlt.info() != Eigen::Success) throw std::runtime_error("adjoint: reduced system factorization failed");
    const Eigen::VectorXd xp = ldlt.solve(r);
    for (int i = 0; i < free_poses_; ++i) x.pose[i] = xp.segment<6>(6 * i);
  }
  for (size_t l = 0; l < graph_.points.size(); ++l) {
    Eigen::Vector3d t = rhs.point[l];
    for (const auto& a : Hpl_[l]) t -= a.block.transpose() * x.pose[a.pose];
    x.point[l] = inv[l] * t;
  }
  return x;
}


void AdjointSystem::Apply(const AdjointVector& step) {
  for (size_t i = 0; i < graph_.poses.size(); ++i)
    if (pose_index_[i] >= 0) graph_.poses[i]->oplus(step.pose[pose_index_[i]].data());
  for (size_t l = 0; l < graph_.points.size(); ++l) graph_.points[l]->oplus(step.point[l].data());
}

Stationarity AdjointSystem::Refine(int max_iterations, double tol) {
  Linearize();
  for (int it = 0; it < max_iterations && stationarity_.scaled_max >= tol; ++it) {
    AdjointVector minus = gradient_;
    for (auto& v : minus.pose) v = -v;
    for (auto& v : minus.point) v = -v;
    const double before = stationarity_.cost;
    AdjointVector step = Solve(minus);
    // Backtracking: far from the quadratic regime the full Gauss-Newton step can raise the robust cost; halve it (up
    // to 2^-10) instead of stopping at the first rejected step, which can leave the solve at the fusion solution.
    bool accepted = false;
    for (int half = 0; half <= 10 && !accepted; ++half) {
      for (auto* v : graph_.poses) if (!v->fixed()) v->push();
      for (auto* v : graph_.points) v->push();
      Apply(step);
      Linearize();
      accepted = stationarity_.cost <= before * (1 + 1e-12) + 1e-12;
      for (auto* v : graph_.poses) if (!v->fixed()) { if (accepted) v->discardTop(); else v->pop(); }
      for (auto* v : graph_.points) { if (accepted) v->discardTop(); else v->pop(); }
      if (!accepted) {
        for (auto& v : step.pose) v *= 0.5;
        for (auto& v : step.point) v *= 0.5;
      }
    }
    if (!accepted) { Linearize(); break; }
  }
  return stationarity_;
}

Mat6 MeasurementJacobian(const AdjointLeg& leg, double eps) {
  Mat6 J;
  auto error_at = [&](const Vec6& delta) {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    P.linear() = Exp(delta.head<3>());
    P.translation() = delta.tail<3>();
    leg.edge->SetFixedBodyDelta(leg.body * (leg.T_C_B.inverse() * P * leg.T_C_B));  // M <- M P
    return ErrorOf(leg.edge);
  };
  for (int c = 0; c < 6; ++c) {
    Vec6 d = Vec6::Zero();
    d[c] = eps;
    const Eigen::VectorXd plus = error_at(d);
    d[c] = -eps;
    J.col(c) = (plus - error_at(d)) / (2 * eps);
  }
  leg.edge->SetFixedBodyDelta(leg.body);
  leg.edge->computeError();
  return J;
}

std::vector<Vec6> AdjointSystem::LegHypergradients(const AdjointVector& w) const {
  std::vector<Vec6> out;
  if (exact_) {
    // g = -(dG/dz)^T w_e with G = [J_i^T Omega e; J_j^T Omega e] the edge gradient w.r.t. its poses (numeric in z)
    for (const auto& leg : graph_.legs) {
      std::vector<std::pair<VertexVIPose*, int>> poses;
      for (int k = 0; k < 2; ++k) {
        auto* pose = static_cast<VertexVIPose*>(leg.edge->vertex(k));
        const int i = PoseIndex(pose);
        if (i >= 0) poses.push_back({pose, i});
      }
      const Eigen::MatrixXd info = InformationOf(leg.edge);
      auto gradient_at = [&](const Vec6& delta) {
        Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
        P.linear() = Exp(delta.head<3>());
        P.translation() = delta.tail<3>();
        leg.edge->SetFixedBodyDelta(leg.body * (leg.T_C_B.inverse() * P * leg.T_C_B));
        const Eigen::VectorXd e = ErrorOf(leg.edge);
        Eigen::VectorXd G(6 * poses.size());
        for (size_t k = 0; k < poses.size(); ++k) G.segment<6>(6 * k) = NumericJacobian(leg.edge, poses[k].first).transpose() * (info * e);
        return G;
      };
      Eigen::MatrixXd dG(6 * poses.size(), 6);
      constexpr double h2 = 1e-5;
      for (int c = 0; c < 6; ++c) {
        Vec6 d = Vec6::Zero();
        d[c] = h2;
        const Eigen::VectorXd plus = gradient_at(d);
        d[c] = -h2;
        dG.col(c) = (plus - gradient_at(d)) / (2 * h2);
      }
      leg.edge->SetFixedBodyDelta(leg.body);
      leg.edge->computeError();
      Eigen::VectorXd we(6 * poses.size());
      for (size_t k = 0; k < poses.size(); ++k) we.segment<6>(6 * k) = w.pose[poses[k].second];
      out.push_back(-dG.transpose() * we);
    }
    return out;
  }
  for (const auto& leg : graph_.legs) {
    Vec6 v = Vec6::Zero();
    for (int k = 0; k < 2; ++k) {
      auto* pose = static_cast<VertexVIPose*>(leg.edge->vertex(k));
      const int i = PoseIndex(pose);
      if (i >= 0) v += NumericJacobian(leg.edge, pose) * w.pose[i];
    }
    const Mat6 Jz = MeasurementJacobian(leg);
    out.push_back(-Jz.transpose() * (InformationOf(leg.edge) * v));
  }
  return out;
}

Eigen::Matrix<double, 3, 6> CameraPointPoseJacobian(VertexVIPose* pose, const Eigen::Vector3d& X, double eps) {
  Eigen::Matrix<double, 3, 6> J;
  for (int c = 0; c < 6; ++c) {
    Vec6 dx = Vec6::Zero();
    dx[c] = eps;
    pose->push(); pose->oplus(dx.data());
    const Eigen::Vector3d plus = pose->estimate().Rcw * X + pose->estimate().tcw;
    pose->pop();
    dx[c] = -eps;
    pose->push(); pose->oplus(dx.data());
    const Eigen::Vector3d minus = pose->estimate().Rcw * X + pose->estimate().tcw;
    pose->pop();
    J.col(c) = (plus - minus) / (2 * eps);
  }
  return J;
}
}  // namespace vce
