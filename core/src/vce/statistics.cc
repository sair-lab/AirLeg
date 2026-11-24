#include "vce/statistics.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace vce {
namespace {
using Sparse = Eigen::SparseMatrix<double>;
using SparseRows = Eigen::SparseMatrix<double, Eigen::RowMajor>;

Sparse Jacobian(const Group& g) {
  if (g.sparse_jacobian.rows() > 0) return g.sparse_jacobian;
  return g.jacobian.sparseView(0.0, 0.0);  // drops exact zeros only
}

// Entries of N^-1 on the pattern of the filled factor (Takahashi recurrence on P N P^T = L D L^T).
// The pattern of every row block of the observation Jacobian is a clique of N, so it lies on that pattern.
class SelectedInverse {
 public:
  bool Compute(const Eigen::SimplicialLDLT<Sparse>& ldlt) {
    const Sparse& factor = ldlt.matrixL().nestedExpression();
    const Eigen::VectorXd d = ldlt.vectorD();
    const int n = factor.cols();
    perm_ = ldlt.permutationP().indices();
    start_.assign(n + 1, 0);
    for (int j = 0; j < n; ++j) {
      start_[j + 1] = start_[j];
      for (Sparse::InnerIterator it(factor, j); it; ++it) if (it.row() > j) ++start_[j + 1];
    }
    rows_.resize(start_[n]);
    l_.resize(start_[n]);
    z_.assign(start_[n], 0.0);
    diag_.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
      int p = start_[j];
      for (Sparse::InnerIterator it(factor, j); it; ++it)
        if (it.row() > j) { rows_[p] = it.row(); l_[p++] = it.value(); }
      if (!std::is_sorted(rows_.begin() + start_[j], rows_.begin() + start_[j + 1])) return false;
    }
    for (int j = n - 1; j >= 0; --j) {
      const int b = start_[j], e = start_[j + 1];
      for (int a = b; a < e; ++a) {
        double sum = 0.0;
        for (int c = b; c < e; ++c) sum += Permuted(rows_[a], rows_[c]) * l_[c];
        z_[a] = -sum;
      }
      double sum = 0.0;
      for (int a = b; a < e; ++a) sum += l_[a] * z_[a];
      diag_[j] = 1.0 / d[j] - sum;
    }
    return std::all_of(z_.begin(), z_.end(), [](double v) { return std::isfinite(v); }) &&
           std::all_of(diag_.begin(), diag_.end(), [](double v) { return std::isfinite(v); });
  }
  // (N^-1)_ij in the original ordering.
  double operator()(int i, int j) const { return Permuted(perm_[i], perm_[j]); }

 private:
  double Permuted(int i, int j) const {
    if (i == j) return diag_[i];
    if (i < j) std::swap(i, j);
    const auto b = rows_.begin() + start_[j], e = rows_.begin() + start_[j + 1];
    const auto it = std::lower_bound(b, e, i);
    if (it == e || *it != i) throw std::logic_error("selected inverse entry outside the factor pattern");
    return z_[it - rows_.begin()];
  }
  Eigen::VectorXi perm_;
  std::vector<int> start_, rows_;
  std::vector<double> l_, z_, diag_;
};

// tr(N^-1 B^T B) = sum over rows r of b_r N^-1 b_r^T.
double Trace(const SparseRows& b, const SelectedInverse& inverse) {
  double trace = 0.0;
  for (int r = 0; r < b.outerSize(); ++r)
    for (SparseRows::InnerIterator x(b, r); x; ++x)
      for (SparseRows::InnerIterator y(b, r); y; ++y) trace += x.value() * y.value() * inverse(x.col(), y.col());
  return trace;
}
}  // namespace

// Sparse implementation of the moment equations. With B_s the column-equilibrated group Jacobians and
// N = sum_s B_s^T B_s: tr(N^-1 H_s) from the selected inverse, and S(s,t) = tr(N^-1 H_s N^-1 H_t)
// = ||B_t N^-1 B_s^T||_F^2 from column blocks of N^-1 B_s^T (rows s of the estimated groups only).
Statistics Estimate(const std::vector<Group>& groups, const Limits& limits) {
  Statistics out;
  auto fail = [&](const char* why) { out.reason = why; return out; };
  const int k = groups.size();
  if (!k) return fail("empty groups");
  if (!std::isfinite(limits.rank_tolerance) || limits.rank_tolerance<=0 || limits.rank_tolerance>=1 ||
      !std::isfinite(limits.min_redundancy) || limits.min_redundancy<0 ||
      !std::isfinite(limits.max_condition) || limits.max_condition<1) return fail("invalid statistical limits");
  std::vector<Sparse> j(k);
  for (int s = 0; s < k; ++s) j[s] = Jacobian(groups[s]);
  const int n = j.front().cols();
  out.states = n;
  out.q = Eigen::VectorXd::Zero(k);
  out.nu = Eigen::VectorXd::Zero(k);
  out.S = Eigen::MatrixXd::Zero(k, k);
  out.beta = Eigen::VectorXd::Ones(k);
  Eigen::VectorXd norm2 = Eigen::VectorXd::Zero(n);
  for (int s = 0; s < k; ++s) {
    const auto& g = groups[s];
    const Eigen::Map<const Eigen::VectorXd> values(j[s].valuePtr(), j[s].nonZeros());
    if (j[s].cols() != n || j[s].rows() != g.residual.size() || !values.allFinite() || !g.residual.allFinite())
      return fail("invalid group");
    out.observations += g.residual.size();
    out.q[s] = g.residual.squaredNorm();
    for (int c = 0; c < n; ++c)
      for (Sparse::InnerIterator it(j[s], c); it; ++it) norm2[c] += it.value() * it.value();
  }
  if (!n || out.observations <= n || !norm2.allFinite() || !out.q.allFinite() || norm2.minCoeff() <= 0)
    return fail("no residual degrees of freedom or unobserved state");
  const Eigen::VectorXd scale = norm2.array().rsqrt();
  std::vector<Sparse> b(k);
  std::vector<SparseRows> rows(k);
  Sparse h(n, n);
  for (int s = 0; s < k; ++s) {
    b[s] = j[s] * scale.asDiagonal();
    rows[s] = b[s];
    h += Sparse(b[s].transpose() * b[s]);
  }
  // Rank of the complete, column-equilibrated normal system (unit diagonal), before damping: LDLT pivots.
  Eigen::SimplicialLDLT<Sparse> solver(h);
  if (solver.info() != Eigen::Success) return fail("normal factorization failed");
  const Eigen::VectorXd pivots = solver.vectorD();
  if (!pivots.allFinite() || pivots.maxCoeff() <= 0) return fail("normal factorization failed");
  out.rank = (pivots.array() > limits.rank_tolerance * pivots.maxCoeff()).count();
  if (out.rank != n) return fail("rank deficient normal system");
  SelectedInverse inverse;
  Eigen::VectorXd trace(k);
  try {
    if (!inverse.Compute(solver)) return fail("normal solve failed");
    for (int s = 0; s < k; ++s) trace[s] = Trace(rows[s], inverse);
  } catch (const std::logic_error&) { return fail("normal solve failed"); }
  if (!trace.allFinite()) return fail("normal solve failed");
  for (int s = 0; s < k; ++s) out.nu[s] = groups[s].residual.size() - trace[s];
  out.redundancy_error = std::abs(out.nu.sum() - (out.observations - n));
  if (out.redundancy_error > 1e-6 * std::max(1, out.observations - n))
    return fail("redundancy identity failed");
  constexpr int kBlock = 64;
  for (int s = 0; s < k; ++s) {
    if (groups[s].fixed && !limits.complete_moments) continue;
    const Sparse bt = b[s].transpose();
    for (int c0 = 0; c0 < bt.cols(); c0 += kBlock) {
      const int w = std::min<int>(kBlock, bt.cols() - c0);
      const Eigen::MatrixXd y = solver.solve(Eigen::MatrixXd(bt.middleCols(c0, w)));
      if (solver.info() != Eigen::Success || !y.allFinite()) return fail("normal solve failed");
      for (int t = 0; t < k; ++t) out.S(s,t) += (rows[t] * y).squaredNorm();
    }
    out.S(s,s) += groups[s].residual.size() - 2 * trace[s];
  }
  for (int s = 0; s < k; ++s)  // fixed rows from the symmetric estimated rows
    if (groups[s].fixed && !limits.complete_moments)
      for (int t = 0; t < k; ++t) if (!groups[t].fixed) out.S(s,t) = out.S(t,s);
  std::vector<int> unknown;
  for (int s = 0; s < k; ++s) if (!groups[s].fixed) unknown.push_back(s);
  if (unknown.empty()) return fail("no estimated group");
  const int u = unknown.size();
  Eigen::MatrixXd a(u,u);
  Eigen::VectorXd rhs(u);
  for (int i = 0; i < u; ++i) {
    const int s = unknown[i];
    if (out.nu[s] < limits.min_redundancy) return fail("insufficient group redundancy");
    rhs[i] = out.q[s];
    for (int t = 0; t < k; ++t) if (groups[t].fixed) rhs[i] -= out.S(s,t);
    for (int j = 0; j < u; ++j) a(i,j) = out.S(s,unknown[j]);
  }
  if (a.diagonal().minCoeff() <= 0) return fail("nonpositive moment diagonal");
  const Eigen::VectorXd d = a.diagonal().array().rsqrt();
  const Eigen::MatrixXd normalized = d.asDiagonal() * a * d.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(normalized);
  if (spectrum.info() != Eigen::Success) return fail("moment spectrum failed");
  const auto values = spectrum.eigenvalues();
  out.condition = values.minCoeff() > 0 ? values.maxCoeff()/values.minCoeff() : INFINITY;
  if (!std::isfinite(out.condition) || out.condition > limits.max_condition)
    return fail("unidentifiable variance components");
  const Eigen::VectorXd beta = d.asDiagonal() * normalized.ldlt().solve(d.asDiagonal() * rhs);
  for (int i = 0; i < u; ++i) out.beta[unknown[i]] = beta[i];
  if (!beta.allFinite() || beta.minCoeff() <= 0) return fail("nonpositive variance solution");
  out.fit_residual = (a * beta - rhs).norm() / std::max(1.0, rhs.norm());
  if (out.fit_residual > 1e-6) return fail("moment solve residual too large");
  out.valid = true;
  return out;
}
ScaleUpdate UpdateScales(const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta,
                        double damping, double lower, double upper, double max_factor) {
  if (alpha.size() == 0 || alpha.size() != beta.size() || !alpha.allFinite() || !beta.allFinite() ||
      alpha.minCoeff() <= 0 || beta.minCoeff() <= 0 || damping <= 0 || damping > 1 ||
      lower <= 0 || upper < lower || max_factor <= 1 || !std::isfinite(damping) ||
      !std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(max_factor))
    throw std::invalid_argument("invalid scale update");
  ScaleUpdate out;
  out.alpha.resize(alpha.size());
  for (int i = 0; i < alpha.size(); ++i) {
    const double raw_step = damping * std::log(beta[i]);
    const double step = std::clamp(raw_step, -std::log(max_factor), std::log(max_factor));
    const double raw = alpha[i] * std::exp(step);
    out.alpha[i] = std::clamp(raw, lower, upper);
    out.limited |= step != raw_step || out.alpha[i] != raw;
    out.log_change = std::max(out.log_change, std::abs(std::log(out.alpha[i]/alpha[i])));
  }
  return out;
}
}  // namespace vce
