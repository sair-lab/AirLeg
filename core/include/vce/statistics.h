#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>
#include <vector>

namespace vce {
struct Limits {
  double rank_tolerance = 1e-10;
  double min_redundancy = 5;
  double max_condition = 1e8;
  bool complete_moments = false;  // also fill S between two fixed groups (never used by the solve)
};
// Already whitened with CURRENT covariance. Columns include every free pose
// and landmark coordinate; no gauge columns, priors, damping or Schur reduction.
struct Group {
  Eigen::MatrixXd jacobian;
  Eigen::SparseMatrix<double> sparse_jacobian;  // used instead of `jacobian` when it has rows
  Eigen::VectorXd residual;
  bool fixed = false;  // Known covariance: beta=1; subtract its S column from RHS.
};
struct Statistics {
  bool valid = false;
  std::string reason;
  int rank = 0, states = 0, observations = 0;
  double condition = 0, fit_residual = 0, redundancy_error = 0;
  Eigen::VectorXd q, nu, beta;
  Eigen::MatrixXd S;
};
Statistics Estimate(const std::vector<Group>& groups, const Limits& limits = {});
struct ScaleUpdate {
  Eigen::VectorXd alpha;
  bool limited = false;
  double log_change = 0;
};
ScaleUpdate UpdateScales(const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta,
                        double damping, double lower, double upper, double max_factor);
}  // namespace vce
