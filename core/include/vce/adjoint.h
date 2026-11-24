#pragma once
// Implicit hypergradient of an upper-level loss U (S2 visual loss) through the S2 global fusion BA (vce-global). At the BA optimum mu*, dmu*/dz = -H^-1 d2L/dmu dz; with the adjoint H w = dU/dmu^T
// the derivative of U w.r.t. the camera measurement z_e = M_e of leg edge e is g_e = -J_z^T Omega J_mu w_e (Gauss-Newton).
// H = sum J^T Omega J over the graph's edges at their current information (already divided by the VCE scale), points
// eliminated by the Schur complement. All Jacobians are central differences through the vertices' own oplus, i.e. in the
// retraction coordinates g2o uses; the measurement chart is M <- [R_M Exp(phi), t_M + R_M rho], delta = [phi; rho].
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/optimizable_graph.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include "g2o_optimization/edge_leg_odom.h"
#include "g2o_optimization/vertex_vi_pose.h"

namespace vce {
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

struct AdjointLeg {
  EdgeLegOdom* edge;           // vertex(0) = pose i, vertex(1) = pose j; extrinsic / delay / bias vertices fixed
  Eigen::Isometry3d body;      // the fixed body delta T_Bi_Bj set on the edge
  Eigen::Isometry3d T_C_B;     // body -> camera (the edge's extrinsic vertex)
};

// The variables and edges of one BA problem. Fixed poses (gauge) take no part in H; points are all free.
struct AdjointGraph {
  std::vector<VertexVIPose*> poses;
  std::vector<g2o::VertexPointXYZ*> points;
  std::vector<g2o::OptimizableGraph::Edge*> visual;   // vertex(0) = point, vertex(1) = pose
  std::vector<AdjointLeg> legs;
  std::vector<g2o::OptimizableGraph::Edge*> others;   // any further edges between poses (included in H / gradient)
};

// Numeric Jacobian of the edge error w.r.t. a vertex (central differences through oplus, step eps).
Eigen::MatrixXd NumericJacobian(g2o::OptimizableGraph::Edge* edge, g2o::OptimizableGraph::Vertex* vertex,
                                double eps = 1e-6);

struct Stationarity {
  double cost = 0;          // sum e^T Omega e
  double gradient_norm = 0; // |J^T Omega e| over the free variables
  double scaled_max = 0;    // max_i |g_i| / |W J_:,i| / max(1, sqrt(cost)) (same measure as Graph::Optimize windows)
};

// Linear system over the free poses (6 each) and points (3 each).
struct AdjointVector {
  std::vector<Vec6> pose;
  std::vector<Eigen::Vector3d> point;
  double Norm() const;
};

class AdjointSystem {
 public:
  // exact = false: Gauss-Newton H = sum J^T Omega J and mixed term J_mu^T Omega J_z (exact for consistent measurements);
  // exact = true: full second derivatives of L (adds the residual-curvature terms, numeric differences of the edge
  // gradients), needed when the measurements are mutually inconsistent.
  explicit AdjointSystem(const AdjointGraph& graph, bool exact = false);
  // Linearize at the current estimates: H blocks, gradient, cost.
  void Linearize();
  const AdjointVector& Gradient() const { return gradient_; }
  Stationarity Check() const { return stationarity_; }
  // Solve H x = rhs (Schur complement on points, dense LDLT on the reduced pose system).
  AdjointVector Solve(const AdjointVector& rhs) const;
  AdjointVector Zero() const;
  // Gauss-Newton steps x <- x - H^-1 g until scaled_max < tol (returns the final check).
  Stationarity Refine(int max_iterations, double tol);
  // g_e = -J_z^T Omega J_mu w_e for every leg (in graph.legs order).
  std::vector<Vec6> LegHypergradients(const AdjointVector& w) const;
  int PoseIndex(const VertexVIPose* v) const;   // -1 when fixed / unknown
  const AdjointGraph& graph() const { return graph_; }
  int ClippedPoints() const { return clipped_points_; }

 private:
  void Apply(const AdjointVector& step);
  const AdjointGraph& graph_;
  bool exact_ = false;
  int clipped_points_ = 0;
  std::vector<int> pose_index_;                 // graph.poses position -> free pose index or -1
  int free_poses_ = 0;
  Eigen::MatrixXd Hpp_;                         // 6F x 6F
  std::vector<Eigen::Matrix3d> Hll_;
  struct Coupling { int pose; Eigen::Matrix<double, 6, 3> block; };
  std::vector<std::vector<Coupling>> Hpl_;      // per point
  AdjointVector gradient_;
  Stationarity stationarity_;
};

// Right-perturbation Jacobian of a leg residual w.r.t. its camera measurement, delta = [phi; rho].
Mat6 MeasurementJacobian(const AdjointLeg& leg, double eps = 1e-6);
// d X^k / d (pose oplus) for X^k = R_cw X + t_cw (3x6, numeric) and d X^k / d X = R_cw.
Eigen::Matrix<double, 3, 6> CameraPointPoseJacobian(VertexVIPose* pose, const Eigen::Vector3d& X, double eps = 1e-6);
}  // namespace vce
