#include "fusion_backend.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#include <g2o/core/block_solver.h>
#include <g2o/core/jacobian_workspace.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/slam3d/vertex_pointxyz.h>

#include "camera.h"
#include "g2o_optimization/edge_leg_odom.h"
#include "g2o_optimization/edge_imu_backend.h"
#include "g2o_optimization/edge_imu.h"
#include "g2o_optimization/edge_project_point.h"
#include "landmarks.h"
#include "marginal.h"
#include "vce/statistics.h"

namespace {
typedef std::chrono::steady_clock Clock;
double Ms(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

Eigen::Isometry3d FromVIPose(const VIPose& p) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = p.Rcw.transpose();
  T.translation() = -p.Rcw.transpose() * p.tcw;
  return T;
}

// VCE groups. Rows of a relative-pose residual [rot(3), t_x, t_y, t_z] map to rot / horizontal / vertical.
enum Group { VISUAL, PRIOR, LEG_ROT, LEG_H, LEG_V, FOOT, NUM_GROUPS };
constexpr int kAxis[6] = {0, 0, 0, 1, 2, 1};
// KIND_FULL: a fixed full information matrix (marginalised visual factors, IMU), known (visual) VCE group
enum Kind { KIND_VISUAL, KIND_PRIOR, KIND_LEG, KIND_FOOT, KIND_FULL };
struct Factor {
  g2o::OptimizableGraph::Edge* edge;
  Kind kind;
  Eigen::VectorXd base;  // base inverse variances (diagonal information)
};
int RowGroup(Kind kind, int row) {
  switch (kind) {
    case KIND_VISUAL: return VISUAL;
    case KIND_FULL: return VISUAL;
    case KIND_PRIOR: return PRIOR;
    case KIND_LEG: return LEG_ROT + kAxis[row];
    default: return FOOT;
  }
}
// Prior of the world gravity direction (VertexGDirection, update Rwg Exp([d0, d1, 0])): error = x, y of Log(Rwg).
class EdgeGDirPrior : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexGDirection> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  void computeError() override {
    Eigen::Vector3d r;
    SO3Log(static_cast<const VertexGDirection*>(_vertices[0])->estimate().Rwg, r);
    _error = r.head<2>();
  }
};

// Absolute tilt at a stationary node (backend.tilt_prior).
// Why: with imu_gravity_free the gravity direction follows the already-fixed poses, so a pitch error picked up during
// an outage (e.g. camera occluded, IMU factors dropped) persists and shows as vertical drift.
// Vertices: [0] VertexVIPose, [1] VertexVec<3> d (camera-IMU
// rotation correction, as EdgeIMUBackend: R' = Rwb Exp(d)). Measurement = unit bias-corrected mean accelerometer (IMU
// frame, specific force points up at rest); error = R'^T e_z - measurement (numeric Jacobians).
class EdgeStaticTilt : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexVIPose, VertexVec<3>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  void computeError() override {
    const VIPose& p = static_cast<const VertexVIPose*>(_vertices[0])->estimate();
    Eigen::Matrix3d Ed;
    SO3Exp(static_cast<const VertexVec<3>*>(_vertices[1])->estimate(), Ed);
    _error = (p.Rwb * Ed).transpose() * Eigen::Vector3d::UnitZ() - _measurement;
  }
};

// Stereo reprojection with the depth scale (1 + v) of the node's visual scale block (vo_scale): the stereo baseline
// term bf becomes bf (1 + v), so a stereo depth scale error is absorbed like the scaled relative-pose edge.
// Vertices: point, pose, VertexVec<1> v. Error = observation - projection (as EdgeSE3ProjectStereoPoint).
class EdgeStereoPointScaled : public g2o::BaseMultiEdge<3, Eigen::Vector3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EdgeStereoPointScaled() { resize(3); }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  void computeError() override {
    const Eigen::Vector3d& pw = static_cast<const g2o::VertexPointXYZ*>(_vertices[0])->estimate();
    const VIPose& p = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
    const double s = 1.0 + static_cast<const VertexVec<1>*>(_vertices[2])->estimate()[0];
    const Eigen::Vector3d pc = p.Rcw * pw + p.tcw;
    const double iz = 1.0 / pc.z(), u = fx * pc.x() * iz + cx;
    _error = _measurement - Eigen::Vector3d(u, fy * pc.y() * iz + cy, u - bf * s * iz);
  }
  bool isDepthPositive() const {
    const VIPose& p = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
    return (p.Rcw * static_cast<const g2o::VertexPointXYZ*>(_vertices[0])->estimate() + p.tcw).z() > 0.0;
  }
  // analytic (land_analytic): EdgeSE3ProjectStereoPoint's Jacobians with bf -> bf s, plus d e / d v = (0, 0, bf / z)
  using g2o::BaseMultiEdge<3, Eigen::Vector3d>::linearizeOplus;
  void linearizeOplus() override {
    if (!analytic) { g2o::BaseMultiEdge<3, Eigen::Vector3d>::linearizeOplus(); return; }
    const Eigen::Vector3d& pw = static_cast<const g2o::VertexPointXYZ*>(_vertices[0])->estimate();
    const VIPose& p = static_cast<const VertexVIPose*>(_vertices[1])->estimate();
    const double s = 1.0 + static_cast<const VertexVec<1>*>(_vertices[2])->estimate()[0];
    const Eigen::Vector3d Xc = p.Rcw * pw + p.tcw, Xb = p.Rbc * Xc + p.tbc;
    const double iz = 1.0 / Xc.z(), iz2 = iz * iz;
    Eigen::Matrix3d P;   // d (u, v, u_r) / d Xc
    P << fx * iz, 0.0, -fx * Xc.x() * iz2,
         0.0, fy * iz, -fy * Xc.y() * iz2,
         fx * iz, 0.0, -fx * Xc.x() * iz2 + bf * s * iz2;
    _jacobianOplus[0] = -P * p.Rcw;
    Eigen::Matrix<double, 3, 6> D;
    D << 0.0, Xb.z(), -Xb.y(), 1.0, 0.0, 0.0,
         -Xb.z(), 0.0, Xb.x(), 0.0, 1.0, 0.0,
         Xb.y(), -Xb.x(), 0.0, 0.0, 0.0, 1.0;
    _jacobianOplus[1] = P * p.Rcb * D;
    _jacobianOplus[2] = Eigen::Vector3d(0.0, 0.0, bf * iz);
  }
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0, bf = 0.0;
  bool analytic = false;
};

Eigen::VectorXd PoseInformation(const std::vector<double>& sigma) {  // {rot, x, y, z}
  Eigen::VectorXd info(6);
  for (int r = 0; r < 6; ++r) {
    const double s = r < 3 ? sigma[0] : sigma[r - 2];
    info[r] = 1.0 / (s * s);
  }
  return info;
}
}  // namespace

FusionBackend::FusionBackend(const BackendConfig& cfg) : cfg_(cfg), thread_(&FusionBackend::Loop, this) {}

FusionBackend::~FusionBackend() { Stop(); }

void FusionBackend::Stop() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  queue_cv_.notify_one();
  if (thread_.joinable()) thread_.join();
}

size_t FusionBackend::Solves() const {
  std::lock_guard<std::mutex> lock(out_mutex_);
  return solve_ms_.size();
}

void FusionBackend::AddLoop(int id_i, int id_j, const Eigen::Isometry3d& T_ij) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  loops_.push_back({id_i, id_j, T_ij});
}

void FusionBackend::SetPathCallback(PathCallback cb) {
  std::lock_guard<std::mutex> lock(out_mutex_);
  path_cb_ = std::move(cb);
}

std::vector<std::pair<double, Eigen::Isometry3d>> FusionBackend::Trajectory() const {
  std::vector<std::pair<double, Eigen::Isometry3d>> out;
  for (size_t n = 0; n < nodes_.size(); ++n) out.emplace_back(nodes_[n].time, est_[n]);
  return out;
}

namespace {
// re-orthonormalise a rotation that drifted by > 1e-6; false if not finite
bool Orthonormalize(Eigen::Isometry3d* T) {
  if (!T->matrix().allFinite()) return false;
  const Eigen::Matrix3d R = T->linear();
  if ((R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-6)
    T->linear() = Eigen::Quaterniond(R).normalized().toRotationMatrix();
  return true;
}
}  // namespace

void FusionBackend::Push(const BackendNode& input) {
  BackendNode node = input;
  if (!Orthonormalize(&node.T_wc) || !Orthonormalize(&node.vo)) {
    ++rejected_inputs_;
    return;   // a non-finite front-end pose: the node is dropped
  }
  if (node.has_leg && !Orthonormalize(&node.leg)) node.has_leg = false;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push_back(node);
  }
  queue_cv_.notify_one();
}

Eigen::Isometry3d FusionBackend::Correction() const {
  std::lock_guard<std::mutex> lock(out_mutex_);
  return correction_;
}

void FusionBackend::RecordRealtime(double time, const Eigen::Isometry3d& T_front) {
  std::lock_guard<std::mutex> lock(out_mutex_);
  realtime_.emplace_back(time, correction_ * T_front);
}

bool FusionBackend::TakeWriteBack(std::map<int, Eigen::Isometry3d>* poses, int* newest) {
  std::lock_guard<std::mutex> lock(out_mutex_);
  if (write_back_.empty()) return false;
  poses->swap(write_back_);
  write_back_.clear();
  *newest = newest_;
  correction_.setIdentity();  // the front-end now carries the back-end poses
  return true;
}

void FusionBackend::RecordWriteBack(double ms) {
  std::lock_guard<std::mutex> lock(out_mutex_);
  write_back_ms_.push_back(ms);
}

void FusionBackend::Loop() {
  for (;;) {
    std::deque<BackendNode> batch;
    bool stop;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      batch.swap(queue_);
      stop = stop_;
    }
    // Solve at each superframe node (trigger "superframe") or every `stride` new nodes; decided per node,
    // so the result does not depend on how the queue was batched. On stop, once more for the remainder.
    for (auto& node : batch) {
      if (!nodes_.empty() && node.id <= nodes_.back().id) continue;
      index_[node.id] = nodes_.size();
      Prepare(&node);
      const bool superframe = node.superframe;
      nodes_.push_back(std::move(node));
      est_.emplace_back();
      est_.back() = InitialEstimate(nodes_.size() - 1);
      if ((cfg_.trigger == "superframe" && superframe) || nodes_.size() >= solved_ + static_cast<size_t>(cfg_.stride))
        Solve(nodes_.size());
    }
    if (stop) {
      if (nodes_.size() > solved_ && nodes_.size() > 1) Solve(nodes_.size());
      return;
    }
  }
}

// Initial pose of node n: the external initialisation if any, else the visual chain on top of est_[n - 1].
Eigen::Isometry3d FusionBackend::InitialEstimate(size_t n) const {
  if (nodes_[n].has_init) return nodes_[n].T_init;
  return n == 0 ? nodes_[n].T_wc : est_[n - 1] * nodes_[n].vo;
}

// Causal statistics of the arriving node: slip relative to the running median of the recent slips.
void FusionBackend::Prepare(BackendNode* node) {
  if (cfg_.leg_reliability && rel_theta_.size() > 0 && rel_theta_.size() == static_cast<int>(node->rel_feat.size())) {
    const Eigen::VectorXd f = Eigen::Map<const Eigen::VectorXd>(node->rel_feat.data(), node->rel_feat.size());
    node->rel_factor = std::clamp(std::exp(rel_theta_.dot(f - rel_mean_)), cfg_.leg_rel_lo, cfg_.leg_rel_hi);
  }
  if (node->slip_mean < 0.0) return;
  slip_hist_.push_back(node->slip_mean);
  while (static_cast<int>(slip_hist_.size()) > cfg_.slip_window) slip_hist_.pop_front();
  std::vector<double> v(slip_hist_.begin(), slip_hist_.end());
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  node->slip_factor = std::clamp(node->slip_mean / std::max(1e-4, v[v.size() / 2]), 1.0, cfg_.slip_max_factor);
}

void FusionBackend::Solve(size_t end) {
  if (end < 2) { solved_ = end; return; }
  const auto tick = Clock::now();
  const size_t W = cfg_.window, B = cfg_.sink_block;
  const size_t lo = end > W + 1 ? end - W - 1 : 0;  // fixed predecessor (or the first node)
  // New nodes start from the visual chain on top of the current estimate.
  for (size_t n = std::max<size_t>(solved_, 1); n < end; ++n) est_[n] = InitialEstimate(n);

  g2o::SparseOptimizer optimizer;
  auto linear = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linear)));
  algorithm->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt
  // lm_lambda_init: fixed initial LM damping; the g2o default (1e-5 of the largest Hessian diagonal) is raised a lot by
  // the IMU factors and leaves the landmark points unconverged within `iterations`
  if (cfg_.lm_lambda_init > 0.0) algorithm->setUserLambdaInit(cfg_.lm_lambda_init);
  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(false);

  // IMU factors need the real camera-IMU extrinsic in the pose vertices (body = IMU)
  const SensorMeta& meta = nodes_[end - 1].meta;
  const bool use_imu = cfg_.imu && meta.valid && meta.has_imu;
  const Eigen::Isometry3d T_c_i = meta.T_i_c.inverse();
  // passive motion: no step since the previous node, yet a clear visually tracked translation (carried robot)
  auto passive = [&](const BackendNode& node) {
    return cfg_.passive_vo_t > 0.0 && node.standing && node.has_leg && !node.lost && node.covisible >= cfg_.degen_n0 &&
           node.vo.translation().norm() > cfg_.passive_vo_t;
  };
  int n_passive = 0, n_imu_long = 0;
  int vid = 0;
  std::map<size_t, VertexVIPose*> poses;
  std::map<g2o::OptimizableGraph::Vertex*, int> offsets;
  int states = 0;
  auto add_vertex = [&](g2o::OptimizableGraph::Vertex* v, bool fixed) {
    v->setId(vid++);
    v->setFixed(fixed);
    optimizer.addVertex(v);
    if (!fixed) { offsets[v] = states; states += v->dimension(); }
  };
  auto pose_vertex = [&](size_t n) {
    auto it = poses.find(n);
    if (it != poses.end()) return it->second;
    auto* v = new VertexVIPose;
    if (use_imu) v->setEstimate(ToVIPose(est_[n].linear(), est_[n].translation(), T_c_i.linear(), T_c_i.translation()));
    else v->setEstimate(ToVIPose(est_[n].linear(), est_[n].translation()));
    add_vertex(v, n <= lo);
    return poses[n] = v;
  };
  for (size_t n = lo; n < end; ++n) pose_vertex(n);

  std::vector<Factor> factors;
  auto add_factor = [&](g2o::OptimizableGraph::Edge* e, Kind kind, const Eigen::VectorXd& base, bool robust) {
    if (robust) {
      auto* huber = new g2o::RobustKernelHuber;
      huber->setDelta(cfg_.huber);
      e->setRobustKernel(huber);
    }
    optimizer.addEdge(e);
    factors.push_back({e, kind, base});
  };
  auto weights = [&]() {
    for (auto& f : factors) {
      const int d = f.edge->dimension();
      if (f.kind == KIND_FULL) continue;   // own full information
      Eigen::VectorXd info = f.base;
      if (f.kind == KIND_LEG) for (int r = 0; r < d; ++r) info[r] /= alpha_[kAxis[r]];
      if (f.kind == KIND_FOOT) info /= alpha_[3];
      Eigen::Map<Eigen::MatrixXd>(f.edge->informationData(), d, d) = info.asDiagonal();
    }
  };

  // residual gyro bias blocks (blocks without free nodes stay fixed and carry the history)
  const size_t GB = cfg_.gyro_bias_block;
  // gyro_bias_total: the blocks are the total bias; the first block starts at (and its prior is centred on) the worker's
  // bias at the end of block 0
  const Eigen::Vector3d bias0 = cfg_.gyro_bias_total
      ? nodes_[std::min(GB - 1, nodes_.size() - 1)].gyro_bias_applied : Eigen::Vector3d::Zero();
  std::map<size_t, VertexVec<3>*> biases;
  auto bias_vertex = [&](size_t b) {
    auto it = biases.find(b);
    if (it != biases.end()) return it->second;
    while (gyro_bias_.size() <= b) gyro_bias_.push_back(gyro_bias_.empty() ? bias0 : gyro_bias_.back());
    auto* v = new VertexVec<3>;
    v->setEstimate(gyro_bias_[b]);
    add_vertex(v, b * GB + GB - 1 <= lo);
    return biases[b] = v;
  };

  // visual translation scale blocks (vo_scale; blocks without free nodes stay fixed and carry the history)
  const size_t SB = std::max(1, cfg_.vo_scale_block);
  std::map<size_t, VertexVec<1>*> vscales;
  auto vscale_vertex = [&](size_t b) {
    auto it = vscales.find(b);
    if (it != vscales.end()) return it->second;
    while (vo_scale_.size() <= b) vo_scale_.push_back(vo_scale_.empty() ? 0.0 : vo_scale_.back());
    auto* v = new VertexVec<1>;
    v->setEstimate(Eigen::Matrix<double, 1, 1>::Constant(vo_scale_[b]));
    add_vertex(v, b * SB + SB - 1 <= lo);
    return vscales[b] = v;
  };

  // visual-leg consistency of a node's front-end translation (win_leg_chi2, land_leg_chi2)
  auto vo_leg_chi2 = [&](const BackendNode& node, double scale, size_t n) {   // fixed-sigma visual vs leg translation
    const double s = 1.0 + (cfg_.vo_scale && n / SB < vo_scale_.size() ? vo_scale_[n / SB] : 0.0);
    const Eigen::Vector3d d = node.vo.translation() * s - node.leg.translation();   // camera frame
    double chi2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double sv = cfg_.vo_sigma[1 + k], sl = cfg_.leg_sigma[1 + k];
      const double var = sv * sv / scale + sl * sl * alpha_[kAxis[3 + k]] * (cfg_.leg_slip ? node.slip_factor : 1.0);
      chi2 += d[k] * d[k] / var;
    }
    return chi2;
  };

  // selected landmarks (visual_mode landmarks): the window's best tracked points (landmarks.h) and their observations
  const bool landmarks_on = cfg_.visual_mode == "landmarks";
  std::vector<long> selection;
  std::map<long, std::vector<std::pair<size_t, const LandmarkObs*>>> land_obs;
  std::vector<int> land_count(end, 0);
  std::vector<bool> land_gated(end, false);
  int n_land_gated = 0;
  if (landmarks_on) {
    WindowSettings ws;
    ws.max_points = cfg_.landmarks_max;
    ws.min_nodes = cfg_.landmarks_min_nodes;
    ws.voxel_size = cfg_.landmarks_voxel;
    selection = SelectWindow(nodes_, lo, end, ws, landmark_sel_);
    const std::set<long> sel(selection.begin(), selection.end());
    // land_leg_chi2: a node whose front-end translation contradicts the leg (fixed-sigma chi2, as win_leg_chi2) breaks
    // the landmark tracks there; each landmark keeps only its observations of the longest segment between such nodes
    // (repetitive ground textures: the front end re-associates the points at those nodes and keeps tracking them).
    // Not after a lost node: its vo is the relocalisation against the motion-model pose, not a tracked motion.
    std::vector<int> seg(end, 0);
    for (size_t n = lo + 1; n < end; ++n) {
      const BackendNode& node = nodes_[n];
      land_gated[n] = cfg_.land_leg_chi2 > 0.0 && !node.lost && !nodes_[n - 1].lost && node.has_leg && !passive(node) &&
                      vo_leg_chi2(node, 1.0, n) > cfg_.land_leg_chi2;
      n_land_gated += land_gated[n];
      seg[n] = seg[n - 1] + land_gated[n];
    }
    for (size_t n = lo; n < end; ++n) {
      if (nodes_[n].lost) continue;
      for (const LandmarkObs& o : nodes_[n].obs)
        if (sel.count(o.point)) { land_obs[o.point].emplace_back(n, &o); ++land_count[n]; }
    }
    if (n_land_gated > 0) {
      std::fill(land_count.begin(), land_count.end(), 0);
      for (auto& [id, obs] : land_obs) {
        std::map<int, int> per_seg;
        for (const auto& ob : obs) ++per_seg[seg[ob.first]];
        int best = per_seg.begin()->first;
        for (const auto& [k, c] : per_seg) if (c > per_seg[best]) best = k;
        obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const auto& ob) { return seg[ob.first] != best; }), obs.end());
        for (const auto& ob : obs) ++land_count[ob.first];
      }
    }
  }

  // window marginalisation (visual_marg win_pair): the points observed by the window's nodes, linearised at the
  // front-end poses (nodes_[n].T_wc) and points, eliminated by the Schur complement into one relative-pose factor per
  // consecutive pair (marginal.h); it replaces the fixed-sigma visual edge. Selected landmarks are not marginalised.
  std::vector<bool> has_win(end, false);
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> win_meas(end, Eigen::Isometry3d::Identity());
  std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> win_info(end, Matrix6d::Zero());
  double win_ms = 0.0;
  if (cfg_.visual_marg == "win_pair" && meta.valid) {
    const auto t_win = Clock::now();
    const std::set<long> sel(selection.begin(), selection.end());
    MargCamera cam{meta.fx, meta.fy, meta.cx, meta.cy, meta.bf};
    MargSettings ms;
    ms.pixel_sigma = cfg_.marg_pixel_sigma;
    ms.huber = cfg_.marg_huber;
    // observations per point over the window nodes (slot = n - lo; lost nodes carry no visual information)
    std::map<long, MargPoint> pts;
    for (size_t n = lo; n < end; ++n) {
      if (nodes_[n].lost) continue;
      for (const LandmarkObs& o : nodes_[n].obs) {
        if (sel.count(o.point)) continue;
        MargPoint& p = pts[o.point];
        if (p.obs.empty()) p.pw = o.pw;
        MargObservation ob;
        ob.slot = static_cast<int>(n - lo);
        ob.kp = o.kp;
        p.obs.push_back(ob);
      }
    }
    for (size_t n = lo + 1; n < end; ++n) {
      MargPoints pair;
      for (const auto& kv : pts) {
        MargPoint q;
        q.pw = kv.second.pw;
        for (const MargObservation& ob : kv.second.obs)
          if (ob.slot == static_cast<int>(n - lo) - 1 || ob.slot == static_cast<int>(n - lo)) {
            MargObservation o2 = ob;
            o2.slot = ob.slot - static_cast<int>(n - lo) + 1;
            q.obs.push_back(o2);
          }
        if (q.obs.size() >= 2) pair.push_back(q);
      }
      if (PairMarginal(nodes_[n - 1].T_wc, nodes_[n].T_wc, pair, cam, ms, &win_info[n])) {
        has_win[n] = true;
        win_meas[n] = nodes_[n - 1].T_wc.inverse() * nodes_[n].T_wc;
      }
    }
    win_ms = Ms(t_win);
  }

  // IMU: preintegration of every interval (n-1, n] at the current bias blocks
  const size_t IB = std::max(1, cfg_.imu_bias_block);
  auto block_bias = [&](size_t b, bool gyro) {
    auto& v = gyro ? imu_bg_ : imu_ba_;
    while (v.size() <= b) v.push_back(v.empty() ? Eigen::Vector3d::Zero() : v.back());
    return v[b];
  };
  if (use_imu && (cfg_.imu_bg_max > 0.0 || cfg_.imu_ba_max > 0.0))   // physically implausible bias: restart it
    for (size_t b = lo / IB; b * IB < end; ++b) {
      const bool bad = (cfg_.imu_bg_max > 0.0 && block_bias(b, true).norm() > cfg_.imu_bg_max) ||
                       (cfg_.imu_ba_max > 0.0 && block_bias(b, false).norm() > cfg_.imu_ba_max);
      if (!bad) continue;
      imu_bg_[b].setZero();
      imu_ba_[b].setZero();
      for (size_t n = b * IB; n < (b + 1) * IB; ++n) imu_vel_.erase(n);
      ++counters_["imu_bias_reset"];
    }
  std::vector<PreinterationPtr> preint(end);
  const ImuNoise noise{meta.gyr_noise, meta.acc_noise, meta.gyr_walk, meta.acc_walk};
  if (use_imu) {
    const double td = cfg_.imu_time_offset;
    // the window's IMU samples on one time line: a time offset moves an interval beyond the node's own samples
    ImuDataList line;
    for (size_t n = lo; n < end; ++n)
      for (const ImuData& s : nodes_[n].imu)
        if (line.empty() || s.timestamp > line.back().timestamp) line.push_back(s);
    auto at_or_before = [&](double t) {   // index of the last sample <= t (0 if none)
      const size_t k = std::upper_bound(line.begin(), line.end(), t, [](double v, const ImuData& s) { return v < s.timestamp; }) - line.begin();
      return k > 0 ? k - 1 : 0;
    };
    for (size_t n = lo + 1; n < end; ++n) {
      if (nodes_[n].imu.empty()) continue;
      const double a = nodes_[n - 1].time - td, b = nodes_[n].time - td;   // interval on the IMU clock
      const size_t i0 = at_or_before(a), i1 = std::min(line.size() - 1, at_or_before(b) + 1);
      if (line.empty() || i1 <= i0) continue;
      const ImuDataList slice(line.begin() + i0, line.begin() + i1 + 1);
      preint[n] = BuildPreintegration(slice, nodes_[n - 1].time, nodes_[n].time, td, noise, block_bias(n / IB, true),
                                      block_bias(n / IB, false));
    }
  }

  // leg velocity-bias blocks (leg_vbias): node n's interval belongs to block leg_block[n] = number of
  // superframe nodes before n (leg_corr_superframe) or n / leg_vbias_block; a block whose last node is <= lo is fixed
  const size_t VB = std::max(1, cfg_.leg_vbias_block);
  std::vector<size_t> leg_block(nodes_.size(), 0);
  std::map<size_t, size_t> block_last;
  for (size_t n = 0, sf = 0; n < nodes_.size(); ++n) {
    leg_block[n] = cfg_.leg_corr_superframe ? sf : n / VB;
    block_last[leg_block[n]] = n;
    if (nodes_[n].superframe) ++sf;
  }
  auto block_fixed = [&](size_t b) { auto it = block_last.find(b); return it == block_last.end() || it->second <= lo; };
  std::map<size_t, VertexVec<3>*> vbiases;
  auto vbias_vertex = [&](size_t b) {
    auto it = vbiases.find(b);
    if (it != vbiases.end()) return it->second;
    while (leg_vbias_.size() <= b) leg_vbias_.push_back(leg_vbias_.empty() ? Eigen::Vector3d::Zero() : leg_vbias_.back());
    auto* v = new VertexVec<3>;
    v->setEstimate(leg_vbias_[b]);
    add_vertex(v, block_fixed(b));
    return vbiases[b] = v;
  };

  // visual + leg edges
  const Eigen::VectorXd vo_info = PoseInformation(cfg_.vo_sigma), leg_info = PoseInformation(cfg_.leg_sigma);
  int n_visual = 0, n_lost = 0, n_leg = 0, n_marg = 0, n_vo_replaced = 0, n_win_gated = 0;
  double min_scale = 1.0;
  for (size_t n = lo + 1; n < end; ++n) {
    const BackendNode& node = nodes_[n];
    double scale = 1.0;
    if (cfg_.degen) {
      scale = std::clamp(std::min(1.0, node.covisible / static_cast<double>(cfg_.degen_n0)) *
                         std::min(1.0, node.spread / cfg_.degen_spread0), cfg_.degen_min_scale, 1.0);
      if (node.covisible < cfg_.degen_lost) scale = node.has_leg ? 0.0 : cfg_.degen_min_scale;  // keep the chain connected
    }
    if (node.lost) scale = node.has_leg ? 0.0 : cfg_.degen_min_scale;  // explicit front-end flag: visual edge from a motion model
    // the marginalised information replaces the covisibility weighting; win_leg_chi2: a marginalised factor whose
    // translation contradicts the leg (fixed-sigma chi2 at the covisibility scale) falls back to the fixed-sigma edge
    // (repetitive ground textures: near-zero visual motion with many consistent matches)
    bool use_win = has_win[n] && !node.lost;
    const bool leg_on = node.has_leg && !passive(node);   // passive: no leg factor, no visual-leg gate
    n_passive += node.has_leg && !leg_on;
    if (use_win && cfg_.win_leg_chi2 > 0.0 && leg_on && scale > 0.0 && vo_leg_chi2(node, scale, n) > cfg_.win_leg_chi2) {
      use_win = false;
      ++n_win_gated;
    }
    if (use_win) scale = 1.0;
    // a node with enough selected landmark observations needs no visual edge (only without a marginalised factor and
    // not at a land_leg_chi2 break: there the fixed-sigma edge, as for win_leg_chi2)
    const bool skip_vo = !node.lost && !use_win && !land_gated[n] && landmarks_on && land_count[n] >= cfg_.landmarks_min_per_node;
    if (skip_vo) {
      ++n_vo_replaced;
    } else if (scale > 0.0) {
      g2o::OptimizableGraph::Edge* e;
      if (cfg_.vo_scale) {   // visual translation scaled by (1 + v) of the node's block
        auto* es = new EdgeScaledRelativePose;
        es->setVertex(2, vscale_vertex(n / SB));
        es->setMeasurement(use_win ? win_meas[n] : node.vo);
        e = es;
      } else {
        auto* el = new EdgeLegRelativePose;
        el->setMeasurement(use_win ? win_meas[n] : node.vo);
        e = el;
      }
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, pose_vertex(n));
      if (use_win) {
        Eigen::Map<Eigen::MatrixXd>(e->informationData(), 6, 6) = win_info[n];
        add_factor(e, KIND_FULL, Eigen::VectorXd(), true);
        ++n_marg;
      } else {
        add_factor(e, KIND_VISUAL, vo_info * scale, true);
      }
      ++n_visual;
      min_scale = std::min(min_scale, scale);
    } else {
      ++n_lost;
    }
    if (leg_on && cfg_.gyro_bias && cfg_.leg_vbias) {
      auto* e = new EdgeLegGyroBiasVel(node.T_c_b, node.time - nodes_[n - 1].time);
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, pose_vertex(n));
      e->setVertex(2, bias_vertex(n / GB));
      e->setVertex(3, vbias_vertex(leg_block[n]));
      e->setMeasurement(node.T_c_b.inverse() * node.leg * node.T_c_b);  // body motion
      Eigen::VectorXd info = leg_info;
      if (cfg_.leg_slip) info.tail<3>() /= node.slip_factor;
      info.tail<3>() /= node.rel_factor;
      add_factor(e, KIND_LEG, info, true);
      ++n_leg;
    } else if (leg_on && cfg_.gyro_bias) {
      auto* e = new EdgeLegGyroBias(node.T_c_b, node.time - nodes_[n - 1].time);
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, pose_vertex(n));
      e->setVertex(2, bias_vertex(n / GB));
      e->setMeasurement(node.T_c_b.inverse() * node.leg * node.T_c_b);  // body motion
      Eigen::VectorXd info = leg_info;
      if (cfg_.leg_slip) info.tail<3>() /= node.slip_factor;
      info.tail<3>() /= node.rel_factor;
      add_factor(e, KIND_LEG, info, true);
      ++n_leg;
    } else if (leg_on) {
      auto* e = new EdgeLegRelativePose;
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, pose_vertex(n));
      e->setMeasurement(node.leg);
      Eigen::VectorXd info = leg_info;
      if (cfg_.leg_slip) info.tail<3>() /= node.slip_factor;  // slipping stance feet: trust the leg translation less
      info.tail<3>() /= node.rel_factor;
      add_factor(e, KIND_LEG, info, true);
      ++n_leg;
    }
  }

  // zupt: stationary intervals (IMU gyro peak < zupt_gyro, per-axis acc std < zupt_acc_std, leg translation < zupt_leg)
  // get a zero relative-motion factor (sigmas zupt_sigma_deg / zupt_sigma_t), known group
  int n_zupt = 0;
  if (cfg_.zupt)
    for (size_t n = lo + 1; n < end; ++n) {
      const BackendNode& node = nodes_[n];
      if (!node.has_leg || node.leg.translation().norm() >= cfg_.zupt_leg || node.imu.size() < 5 || passive(node)) continue;
      double gmax = 0.0;
      Eigen::Vector3d ma = Eigen::Vector3d::Zero(), sa = Eigen::Vector3d::Zero();
      int k = 0;
      for (const ImuData& d : node.imu) {
        if (d.timestamp < nodes_[n - 1].time || d.timestamp > node.time) continue;
        gmax = std::max(gmax, d.gyr.norm());
        ma += d.acc; sa += d.acc.cwiseAbs2(); ++k;
      }
      if (k < 5) continue;
      ma /= k;
      const double astd = (sa / k - ma.cwiseAbs2()).cwiseMax(0.0).cwiseSqrt().maxCoeff();
      if (gmax >= cfg_.zupt_gyro || astd >= cfg_.zupt_acc_std) continue;
      auto* e = new EdgeLegRelativePose;
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, pose_vertex(n));
      e->setMeasurement(Eigen::Isometry3d::Identity());
      Eigen::Matrix<double, 6, 1> info;
      const double sr = cfg_.zupt_sigma_deg * M_PI / 180.0, st = cfg_.zupt_sigma_t;
      info << Eigen::Vector3d::Constant(1.0 / (sr * sr)), Eigen::Vector3d::Constant(1.0 / (st * st));
      e->setInformation(info.asDiagonal());
      add_factor(e, KIND_FULL, Eigen::VectorXd(), true);
      ++n_zupt;
    }

  // selected landmarks: marginalised point vertices with reprojection edges, initialised from their first stereo
  // observation in the window with the current back-end pose; stereo depth scaled by the node's visual scale (vo_scale)
  int n_land = 0, n_land_obs = 0;
  std::vector<std::pair<g2o::OptimizableGraph::Edge*, bool>> land_edges;   // (edge, stereo)
  std::set<g2o::OptimizableGraph::Edge*> land_set;                          // vce_skip_landmarks
  std::vector<long> used_selection;
  for (long id : selection) {
    auto it = land_obs.find(id);
    if (it == land_obs.end() || it->second.size() < 2) continue;
    Eigen::Vector3d pw;
    bool init = false;
    for (const auto& [n, o] : it->second) {
      const double disp = o->kp.x() - o->kp.z();
      if (o->kp.z() < 0.0 || disp < 0.5) continue;
      const double z = meta.bf / disp;
      pw = est_[n] * Eigen::Vector3d((o->kp.x() - meta.cx) * z / meta.fx, (o->kp.y() - meta.cy) * z / meta.fy, z);
      init = true;
      break;
    }
    if (!init) continue;
    auto* vp = new g2o::VertexPointXYZ;
    vp->setEstimate(pw);
    vp->setMarginalized(true);
    add_vertex(vp, false);
    const double info = 1.0 / (cfg_.landmark_pixel_sigma * cfg_.landmark_pixel_sigma);
    for (const auto& [n, o] : it->second) {
      g2o::OptimizableGraph::Edge* edge;
      if (o->kp.z() >= 0.0 && cfg_.vo_scale) {
        auto* e = new EdgeStereoPointScaled;
        e->fx = meta.fx; e->fy = meta.fy; e->cx = meta.cx; e->cy = meta.cy; e->bf = meta.bf;
        e->analytic = cfg_.land_analytic;
        e->setVertex(0, vp);
        e->setVertex(1, pose_vertex(n));
        e->setVertex(2, vscale_vertex(n / SB));
        e->setMeasurement(o->kp);
        add_factor(e, KIND_VISUAL, Eigen::VectorXd::Constant(3, info), true);
        land_edges.emplace_back(e, true);
        edge = e;
      } else if (o->kp.z() >= 0.0) {
        auto* e = new EdgeSE3ProjectStereoPoint;
        e->fx = meta.fx; e->fy = meta.fy; e->cx = meta.cx; e->cy = meta.cy; e->bf = meta.bf;
        e->setVertex(0, vp);
        e->setVertex(1, pose_vertex(n));
        e->setMeasurement(o->kp);
        add_factor(e, KIND_VISUAL, Eigen::VectorXd::Constant(3, info), true);
        land_edges.emplace_back(e, true);
        edge = e;
      } else {
        auto* e = new EdgeSE3ProjectPoint;
        e->fx = meta.fx; e->fy = meta.fy; e->cx = meta.cx; e->cy = meta.cy;
        e->setVertex(0, vp);
        e->setVertex(1, pose_vertex(n));
        e->setMeasurement(o->kp.head<2>());
        add_factor(e, KIND_VISUAL, Eigen::VectorXd::Constant(2, info), true);
        land_edges.emplace_back(e, false);
        edge = e;
      }
      land_set.insert(edge);
      ++n_land_obs;
    }
    used_selection.push_back(id);
    ++n_land;
  }

  // IMU preintegration edges, velocities, bias blocks (random walk from the preintegrated walk covariance, prior on
  // block 0), world gravity direction (fixed, or free with a prior), optional camera-IMU rotation correction
  int n_imu = 0;
  int n_tilt = 0;   // static tilt factors (tilt_prior)
  std::map<size_t, VertexVelocity*> vels;
  std::map<size_t, VertexGyrBias*> bgs;
  std::map<size_t, VertexAccBias*> bas;
  VertexVec<3>* ext = nullptr;
  VertexGDirection* gdir = nullptr;
  if (use_imu) {
    auto vel_vertex = [&](size_t n) {
      auto it = vels.find(n);
      if (it != vels.end()) return it->second;
      Eigen::Vector3d v0 = Eigen::Vector3d::Zero();
      auto pv = imu_vel_.find(n);
      if (pv != imu_vel_.end()) v0 = pv->second;
      else if (n > 0) {   // finite difference of the IMU positions
        const double dt = nodes_[n].time - nodes_[n - 1].time;
        if (dt > 1e-3) v0 = ((est_[n] * T_c_i).translation() - (est_[n - 1] * T_c_i).translation()) / dt;
      }
      auto* v = new VertexVelocity(v0);
      add_vertex(v, n <= lo);
      return vels[n] = v;
    };
    auto bg_vertex = [&](size_t b) {
      auto it = bgs.find(b);
      if (it != bgs.end()) return it->second;
      auto* v = new VertexGyrBias(block_bias(b, true));
      add_vertex(v, b * IB + IB - 1 <= lo);
      return bgs[b] = v;
    };
    auto ba_vertex = [&](size_t b) {
      auto it = bas.find(b);
      if (it != bas.end()) return it->second;
      auto* v = new VertexAccBias(block_bias(b, false));
      add_vertex(v, b * IB + IB - 1 <= lo);
      return bas[b] = v;
    };
    gdir = new VertexGDirection(imu_Rwg_);
    add_vertex(gdir, !cfg_.imu_gravity_free);
    if (cfg_.imu_gravity_free) {
      const double s = cfg_.imu_gravity_prior_deg * M_PI / 180.0;
      auto* gp = new EdgeGDirPrior;
      gp->setVertex(0, gdir);
      gp->setInformation(Eigen::Matrix2d::Identity() / (s * s));
      add_factor(gp, KIND_FULL, Eigen::VectorXd(), false);
    }
    ext = new VertexVec<3>;
    ext->setEstimate(imu_ext_);
    add_vertex(ext, !cfg_.imu_ext_rot);
    std::map<size_t, Eigen::Matrix<double, 6, 6>> walk;   // accumulated walk covariance (gyro, acc) per block
    if (cfg_.tilt_prior)   // absolute tilt at stationary nodes (same test as zupt)
      for (size_t n = lo + 1; n < end; ++n) {
        const BackendNode& node = nodes_[n];
        if (!node.has_leg || node.leg.translation().norm() >= cfg_.zupt_leg || node.imu.size() < 5 || passive(node)) continue;
        double gmax = 0.0;
        Eigen::Vector3d ma = Eigen::Vector3d::Zero(), sa = Eigen::Vector3d::Zero();
        int k = 0;
        for (const ImuData& d : node.imu) {
          if (d.timestamp < nodes_[n - 1].time || d.timestamp > node.time) continue;
          gmax = std::max(gmax, d.gyr.norm());
          ma += d.acc; sa += d.acc.cwiseAbs2(); ++k;
        }
        if (k < 5) continue;
        ma /= k;
        const double astd = (sa / k - ma.cwiseAbs2()).cwiseMax(0.0).cwiseSqrt().maxCoeff();
        if (gmax >= cfg_.zupt_gyro || astd >= cfg_.zupt_acc_std) continue;
        const size_t b = n / IB;
        if (b < imu_ba_.size() && imu_ba_[b].allFinite()) ma -= imu_ba_[b];
        if (ma.norm() < 1.0) continue;
        auto* e = new EdgeStaticTilt;
        e->setVertex(0, pose_vertex(n));
        e->setVertex(1, ext);
        e->setMeasurement(ma.normalized());
        const double st = cfg_.tilt_sigma_deg * M_PI / 180.0;
        e->setInformation(Eigen::Matrix3d::Identity() / (st * st));
        add_factor(e, KIND_FULL, Eigen::VectorXd(), false);
        ++n_tilt;
      }
    for (size_t n = lo + 1; n < end; ++n) {
      if (!preint[n]) continue;
      if (cfg_.imu_max_dt > 0.0 && nodes_[n].time - nodes_[n - 1].time > cfg_.imu_max_dt &&
          !(nodes_[n].has_leg && nodes_[n].leg.translation().norm() < cfg_.imu_max_dt_leg)) {   // long interval, moving
        ++n_imu_long;
        continue;
      }
      const size_t b = n / IB;
      auto* e = new EdgeIMUBackend(preint[n]);
      e->setVertex(0, pose_vertex(n - 1));
      e->setVertex(1, vel_vertex(n - 1));
      e->setVertex(2, bg_vertex(b));
      e->setVertex(3, ba_vertex(b));
      e->setVertex(4, pose_vertex(n));
      e->setVertex(5, vel_vertex(n));
      e->setVertex(6, gdir);
      e->setVertex(7, ext);
      add_factor(e, KIND_FULL, Eigen::VectorXd(), true);
      walk.emplace(b, Eigen::Matrix<double, 6, 6>::Zero()).first->second += preint[n]->Cov.block<6, 6>(9, 9);
      ++n_imu;
    }
    for (const auto& kv : bgs) {   // random walk to the previous block, prior on the first block
      const size_t b = kv.first;
      if (kv.second->fixed()) continue;
      Eigen::Matrix<double, 6, 6> W = walk.count(b) ? walk[b] : Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
      W.diagonal() = W.diagonal().cwiseMax(1e-12);
      for (int k = 0; k < 3; ++k) {   // random-walk floor: early wrong estimates can be forgotten
        W(k, k) = std::max(W(k, k), cfg_.imu_bg_rw_min * cfg_.imu_bg_rw_min);
        W(k + 3, k + 3) = std::max(W(k + 3, k + 3), cfg_.imu_ba_rw_min * cfg_.imu_ba_rw_min);
      }
      if (b > 0) {
        auto* eg = new EdgeGyr;
        eg->setVertex(0, bg_vertex(b - 1));
        eg->setVertex(1, kv.second);
        eg->setInformation(W.block<3, 3>(0, 0).inverse());
        add_factor(eg, KIND_FULL, Eigen::VectorXd(), false);
        auto* ea = new EdgeAcc;
        ea->setVertex(0, ba_vertex(b - 1));
        ea->setVertex(1, ba_vertex(b));
        ea->setInformation(W.block<3, 3>(3, 3).inverse());
        add_factor(ea, KIND_FULL, Eigen::VectorXd(), false);
      }
      if (b == 0) {   // prior on the first block: fixed zero copies
        auto* g0 = new VertexGyrBias(Eigen::Vector3d::Zero());
        add_vertex(g0, true);
        auto* a0 = new VertexAccBias(Eigen::Vector3d::Zero());
        add_vertex(a0, true);
        auto* eg = new EdgeGyr;
        eg->setVertex(0, g0);
        eg->setVertex(1, kv.second);
        eg->setInformation(Eigen::Matrix3d::Identity() / (cfg_.imu_bg_prior * cfg_.imu_bg_prior));
        add_factor(eg, KIND_FULL, Eigen::VectorXd(), false);
        auto* ea = new EdgeAcc;
        ea->setVertex(0, a0);
        ea->setVertex(1, ba_vertex(b));
        ea->setInformation(Eigen::Matrix3d::Identity() / (cfg_.imu_ba_prior * cfg_.imu_ba_prior));
        add_factor(ea, KIND_FULL, Eigen::VectorXd(), false);
      }
    }
    if (cfg_.imu_ext_rot) {
      const double s = cfg_.imu_ext_prior_deg * M_PI / 180.0;
      auto* p = new EdgeVecPrior<3>;
      p->setVertex(0, ext);
      p->setMeasurement(Eigen::Vector3d::Zero());
      add_factor(p, KIND_PRIOR, Eigen::VectorXd::Constant(3, 1.0 / (s * s)), false);
    }
  }

  // loop-closure edges between nodes in the window (fixed group, like the visual edges)
  int n_loop = 0;
  {
    const Eigen::VectorXd loop_info = PoseInformation(cfg_.loop_sigma);
    std::vector<LoopEdge, Eigen::aligned_allocator<LoopEdge>> loops;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      loops = loops_;
    }
    for (const auto& l : loops) {
      auto a = index_.find(l.i), b = index_.find(l.j);
      if (a == index_.end() || b == index_.end() || a->second < lo || b->second < lo || a->second >= end || b->second >= end) continue;
      if (a->second <= lo && b->second <= lo) continue;
      auto* e = new EdgeLegRelativePose;
      e->setVertex(0, pose_vertex(a->second));
      e->setVertex(1, pose_vertex(b->second));
      e->setMeasurement(l.T);
      add_factor(e, KIND_VISUAL, loop_info, cfg_.loop_robust);
      ++n_loop;
    }
  }

  // foot edges, contact offset blocks
  std::map<size_t, VertexLegTimeDelay*> sinks;
  int n_foot = 0, gated_slope = 0, gated_rmse = 0;
  if (cfg_.foot) {
    auto sink_vertex = [&](size_t b) {
      auto it = sinks.find(b);
      if (it != sinks.end()) return it->second;
      while (sink_.size() <= b) sink_.push_back(sink_.empty() ? 0.0 : sink_.back());
      auto* v = new VertexLegTimeDelay(cfg_.sink_min, cfg_.sink_max);
      v->setEstimate(sink_[b]);
      add_vertex(v, b * B + B - 1 <= lo);  // blocks without free nodes stay fixed
      return sinks[b] = v;
    };
    const double max_slope = cfg_.foot_max_slope_deg * M_PI / 180.0;
    for (size_t n = lo + 1; n < end; ++n) {
      for (const auto& c : nodes_[n].feet) {
        auto it = index_.find(c.id_anchor);
        if (it == index_.end() || it->second >= n) continue;
        if (cfg_.foot_max_rmse > 0.0 && c.rmse > cfg_.foot_max_rmse) { ++gated_rmse; continue; }
        const Eigen::Vector3d n_w = est_[it->second].linear() * c.n_c.normalized();
        if (cfg_.foot_max_slope_deg > 0.0 && std::acos(std::min(1.0, std::abs(n_w.z()))) > max_slope) {
          ++gated_slope;
          continue;
        }
        auto* e = new EdgeFootSurface(nodes_[n].T_c_b, c.foot_b, c.q_c, c.n_c, c.radius);
        e->setVertex(0, pose_vertex(n));
        e->setVertex(1, pose_vertex(it->second));  // anchors older than the window join as fixed vertices
        e->setVertex(2, sink_vertex(n / B));
        e->setMeasurement(0.0);
        const double sigma = std::max(c.sigma, cfg_.foot_sigma_min);
        add_factor(e, KIND_FOOT, Eigen::VectorXd::Constant(1, 1.0 / (sigma * sigma)), true);
        ++n_foot;
      }
    }
    if (!sinks.empty()) {
      const size_t first = sinks.begin()->first, last = sinks.rbegin()->first;
      if (first > 0) sink_vertex(first - 1);
      for (size_t b = first; b <= last; ++b) {
        auto* v = sink_vertex(b);
        if (v->fixed()) continue;
        auto* prior = new EdgeLegTimeDelayPrior;
        prior->setVertex(0, v);
        prior->setMeasurement(0.0);
        add_factor(prior, KIND_PRIOR, Eigen::VectorXd::Constant(1, 1.0 / (cfg_.sink_prior * cfg_.sink_prior)), false);
        if (b == 0) continue;
        auto* walk = new EdgeScalarRandomWalk;
        walk->setVertex(0, sink_vertex(b - 1));
        walk->setVertex(1, v);
        walk->setMeasurement(0.0);
        add_factor(walk, KIND_PRIOR, Eigen::VectorXd::Constant(1, 1.0 / (cfg_.sink_rw * cfg_.sink_rw)), false);
      }
    }
  }

  if (!biases.empty()) {
    const size_t first = biases.begin()->first, last = biases.rbegin()->first;
    if (first > 0) bias_vertex(first - 1);
    for (size_t b = first; b <= last; ++b) {
      auto* v = bias_vertex(b);
      if (v->fixed()) continue;
      if (b == 0) {  // later blocks inherit the prior through the random walk
        auto* prior = new EdgeVecPrior<3>;
        prior->setVertex(0, v);
        prior->setMeasurement(bias0);
        add_factor(prior, KIND_PRIOR, Eigen::VectorXd::Constant(3, 1.0 / (cfg_.gyro_bias_prior * cfg_.gyro_bias_prior)), false);
        continue;
      }
      auto* walk = new EdgeVecRandomWalk<3>;
      walk->setVertex(0, bias_vertex(b - 1));
      walk->setVertex(1, v);
      walk->setMeasurement(Eigen::Vector3d::Zero());
      add_factor(walk, KIND_PRIOR, Eigen::VectorXd::Constant(3, 1.0 / (cfg_.gyro_bias_rw * cfg_.gyro_bias_rw)), false);
    }
  }

  for (auto& kv : vscales) {   // prior on every free block, random walk to the previous block
    if (kv.second->fixed()) continue;
    auto* prior = new EdgeVecPrior<1>;
    prior->setVertex(0, kv.second);
    prior->setMeasurement(Eigen::Matrix<double, 1, 1>::Zero());
    add_factor(prior, KIND_PRIOR, Eigen::VectorXd::Constant(1, 1.0 / (cfg_.vo_scale_prior * cfg_.vo_scale_prior)), false);
    if (kv.first == 0) continue;
    auto* walk = new EdgeVecRandomWalk<1>;
    walk->setVertex(0, vscale_vertex(kv.first - 1));
    walk->setVertex(1, kv.second);
    walk->setMeasurement(Eigen::Matrix<double, 1, 1>::Zero());
    add_factor(walk, KIND_PRIOR, Eigen::VectorXd::Constant(1, 1.0 / (cfg_.vo_scale_rw * cfg_.vo_scale_rw)), false);
  }

  if (!vbiases.empty()) {  // leg velocity bias: prior on the first block, random walk between blocks
    const size_t first = vbiases.begin()->first, last = vbiases.rbegin()->first;
    if (first > 0) vbias_vertex(first - 1);
    for (size_t b = first; b <= last; ++b) {
      auto* v = vbias_vertex(b);
      if (v->fixed()) continue;
      if (b == 0) {
        auto* prior = new EdgeVecPrior<3>;
        prior->setVertex(0, v);
        prior->setMeasurement(Eigen::Vector3d::Zero());
        add_factor(prior, KIND_PRIOR, Eigen::VectorXd::Constant(3, 1.0 / (cfg_.leg_vbias_prior * cfg_.leg_vbias_prior)), false);
        continue;
      }
      auto* walk = new EdgeVecRandomWalk<3>;
      walk->setVertex(0, vbias_vertex(b - 1));
      walk->setVertex(1, v);
      walk->setMeasurement(Eigen::Vector3d::Zero());
      add_factor(walk, KIND_PRIOR, Eigen::VectorXd::Constant(3, 1.0 / (cfg_.leg_vbias_rw * cfg_.leg_vbias_rw)), false);
    }
  }

  auto optimize = [&]() {
    weights();
    optimizer.initializeOptimization(0);
    optimizer.optimize(cfg_.iterations);
  };
  optimize();
  // Second round without the foot edges beyond the Huber threshold (3 sigma).
  int foot_outliers = 0;
  for (auto& f : factors) {
    if (f.kind != KIND_FOOT) continue;
    f.edge->computeError();
    if (f.edge->chi2() > cfg_.huber * cfg_.huber) { f.edge->setLevel(1); ++foot_outliers; }
  }
  // landmark observations beyond 3 sigma (pixel_sigma whitened) or behind the camera
  int land_outliers = 0;
  for (auto& le : land_edges) {
    le.first->computeError();
    bool behind;
    if (auto* sc = dynamic_cast<EdgeStereoPointScaled*>(le.first)) behind = !sc->isDepthPositive();
    else behind = le.second ? !static_cast<EdgeSE3ProjectStereoPoint*>(le.first)->isDepthPositive()
                            : !static_cast<EdgeSE3ProjectPoint*>(le.first)->isDepthPositive();
    if (behind || le.first->chi2() > 9.0 * le.first->dimension() / 2.0) { le.first->setLevel(1); ++land_outliers; }
  }
  // land_fallback: an outlier ratio above land_fallback means the landmarks contradict the fused poses as a whole
  // (e.g. a visual scale error on anonymised ground): this solve drops all its landmarks
  bool land_dropped = false;
  if (cfg_.land_fallback > 0.0 && n_land_obs > 0 && land_outliers > cfg_.land_fallback * n_land_obs) {
    for (auto& le : land_edges) le.first->setLevel(1);
    land_dropped = true;
    ++counters_["land_fallback"];
  }
  if (foot_outliers || land_outliers) optimize();
  const double solve_ms = Ms(tick);

  // Online VCE on the converged window: visual and prior groups known, leg and foot groups estimated.
  std::string vce_status = "off";
  double vce_ms = 0.0;
  // vce_landmarks: the landmarks kept for VCE (lowest vertex ids)
  std::set<g2o::HyperGraph::Vertex*> land_keep;
  if (cfg_.vce_landmarks > 0) {
    std::vector<g2o::HyperGraph::Vertex*> pts;
    for (auto& le : land_edges) pts.push_back(le.first->vertices()[0]);
    std::sort(pts.begin(), pts.end(), [](g2o::HyperGraph::Vertex* x, g2o::HyperGraph::Vertex* y) { return x->id() < y->id(); });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    if (static_cast<int>(pts.size()) > cfg_.vce_landmarks) pts.resize(cfg_.vce_landmarks);
    land_keep.insert(pts.begin(), pts.end());
  }
  for (int vce_round = 0; cfg_.vce && states > 0 && vce_round < cfg_.vce_rounds; ++vce_round) {
    const auto t_vce = Clock::now();
    std::vector<int> rows(NUM_GROUPS, 0);
    std::vector<const Factor*> used;
    for (const auto& f : factors) {
      if (f.edge->level() != 0) continue;  // robust down-weighted edges stay: dropping them biases the variances low
      if (cfg_.vce_skip_full && f.kind == KIND_FULL) continue;
      if (cfg_.vce_skip_landmarks && land_set.count(f.edge)) continue;   // fixed visual group: only their redundancy
      if (cfg_.vce_landmarks > 0 && land_set.count(f.edge) && !land_keep.count(f.edge->vertices()[0])) continue;
      used.push_back(&f);
    }
    // vce_point_check: a landmark whose used observations leave its 3x3 information rank deficient or ill-conditioned
    // (only mono / outlier-reduced observations) makes the VCE normal system singular: its edges stay out
    if (cfg_.vce_point_check) {
      std::map<g2o::HyperGraph::Vertex*, Eigen::Matrix3d> point_info;
      for (const Factor* f : used) {
        if (!land_set.count(f->edge)) continue;
        auto* edge = f->edge;
        const int d = edge->dimension();
        edge->computeError();
        g2o::JacobianWorkspace workspace;
        workspace.updateSize(edge);
        workspace.allocate();
        edge->linearizeOplus(workspace);
        const Eigen::MatrixXd jp = Eigen::Map<const Eigen::MatrixXd>(workspace.workspaceForVertex(0), d, 3);
        const Eigen::MatrixXd info = Eigen::Map<const Eigen::MatrixXd>(edge->informationData(), d, d);
        auto it = point_info.emplace(edge->vertices()[0], Eigen::Matrix3d::Zero()).first;
        it->second += jp.transpose() * info * jp;
      }
      std::set<g2o::HyperGraph::Vertex*> bad;
      for (const auto& [v, h] : point_info) {
        const Eigen::Vector3d ev = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(h).eigenvalues();
        if (!(ev.minCoeff() > 1e-6 * ev.maxCoeff())) bad.insert(v);
      }
      used.erase(std::remove_if(used.begin(), used.end(), [&](const Factor* f) {
        return land_set.count(f->edge) && bad.count(f->edge->vertices()[0]); }), used.end());
    }
    for (const Factor* f : used)
      for (int r = 0; r < f->edge->dimension(); ++r) ++rows[RowGroup(f->kind, r)];
    // state columns: every free vertex, or (vce_skip_full) only those of the used factors
    std::map<g2o::OptimizableGraph::Vertex*, int> cols;
    int ncols = states;
    if (cfg_.vce_skip_full) {
      ncols = 0;
      for (const Factor* f : used)
        for (auto* hv : f->edge->vertices()) {
          auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(hv);
          if (!v->fixed() && !cols.count(v)) { cols[v] = ncols; ncols += v->dimension(); }
        }
    }
    auto col_of = [&](g2o::OptimizableGraph::Vertex* v) { return cfg_.vce_skip_full ? cols.at(v) : offsets.at(v); };
    std::vector<vce::Group> groups(NUM_GROUPS);
    std::vector<std::vector<Eigen::Triplet<double>>> triplets(NUM_GROUPS);
    for (int g = 0; g < NUM_GROUPS; ++g) {
      groups[g].residual = Eigen::VectorXd::Zero(rows[g]);
      groups[g].fixed = g == VISUAL || g == PRIOR || (g == FOOT && rows[g] < 10);
    }
    std::vector<int> start(NUM_GROUPS, 0);
    for (const Factor* f : used) {
      auto* edge = f->edge;
      const int d = edge->dimension();
      // whitening W with W^T W = information (diagonal: the square roots; full matrices: Cholesky)
      const Eigen::MatrixXd info = Eigen::Map<const Eigen::MatrixXd>(edge->informationData(), d, d);
      Eigen::MatrixXd Wt = info.diagonal().cwiseSqrt().asDiagonal();
      if (f->kind == KIND_FULL) {
        Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (info + info.transpose()) + 1e-12 * Eigen::MatrixXd::Identity(d, d));
        if (llt.info() == Eigen::Success) Wt = llt.matrixL().transpose();
      }
      edge->computeError();
      const Eigen::VectorXd e0 = Wt * Eigen::Map<const Eigen::VectorXd>(edge->errorData(), d);
      std::vector<int> row(d);
      for (int r = 0; r < d; ++r) {
        const int g = RowGroup(f->kind, r);
        row[r] = start[g]++;
        groups[g].residual[row[r]] = e0[r];
      }
      // the edge's own linearization (analytic for the leg / foot / relative-pose edges, g2o numeric otherwise)
      g2o::JacobianWorkspace workspace;
      workspace.updateSize(edge);
      workspace.allocate();
      edge->linearizeOplus(workspace);
      for (size_t i = 0; i < edge->vertices().size(); ++i) {
        auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(edge->vertices()[i]);
        if (v->fixed()) continue;
        const Eigen::MatrixXd jv = Wt * Eigen::Map<const Eigen::MatrixXd>(workspace.workspaceForVertex(i), d, v->dimension());
        for (int c = 0; c < v->dimension(); ++c)
          for (int r = 0; r < d; ++r)
            if (jv(r, c) != 0.0) triplets[RowGroup(f->kind, r)].emplace_back(row[r], col_of(v) + c, jv(r, c));
      }
      edge->computeError();
    }
    for (int g = 0; g < NUM_GROUPS; ++g) {
      groups[g].sparse_jacobian.resize(rows[g], ncols);
      groups[g].sparse_jacobian.setFromTriplets(triplets[g].begin(), triplets[g].end());
    }
    // Drop empty groups (vce::Estimate needs consistent, non-empty blocks).
    std::vector<vce::Group> present;
    std::vector<int> ids;
    for (int g = 0; g < NUM_GROUPS; ++g)
      if (rows[g] > 0) { present.push_back(groups[g]); ids.push_back(g); }
    vce::Limits limits;
    limits.min_redundancy = cfg_.vce_min_redundancy;
    // A weakly redundant group can give a nonpositive (or unsupported) component: keep it at its current
    // scale and re-estimate the others.
    vce::Statistics stats;
    for (size_t attempt = 0; attempt < present.size(); ++attempt) {
      stats = vce::Estimate(present, limits);
      if (stats.valid || stats.beta.size() != static_cast<int>(present.size())) break;
      bool changed = false;
      for (size_t k = 0; k < present.size(); ++k) {
        if (present[k].fixed || (stats.beta[k] > 0 && stats.nu[k] >= limits.min_redundancy)) continue;
        present[k].fixed = changed = true;
      }
      if (!changed) break;
    }
    if (stats.valid) {
      Eigen::Vector4d alpha = alpha_;
      for (size_t k = 0; k < ids.size(); ++k) {
        if (present[k].fixed) continue;
        const int a = ids[k] - LEG_ROT;  // LEG_ROT..FOOT -> 0..3
        Eigen::VectorXd one(1), beta(1);
        one << alpha_[a];
        beta << stats.beta[k];
        alpha[a] = vce::UpdateScales(one, beta, cfg_.vce_damping, cfg_.vce_lower, cfg_.vce_upper, cfg_.vce_step).alpha[0];
      }
      alpha_ = alpha;
      optimize();
      vce_status = "accepted:";
      for (size_t k = 0; k < ids.size(); ++k)
        if (ids[k] >= LEG_ROT) vce_status += present[k].fixed ? '-' : "rhvf"[ids[k] - LEG_ROT];
    } else {
      vce_status = stats.reason;
    }
    vce_ms = Ms(t_vce);
  }

  // a solve with non-finite poses is rejected (the previous estimates are kept)
  bool finite = true;
  for (auto& kv : poses) if (kv.first > lo && !FromVIPose(kv.second->estimate()).matrix().allFinite()) finite = false;
  if (!finite) {
    ++rejected_solves_;
    std::cout << "[S3 backend] non-finite solve rejected (nodes " << lo << '-' << end << ")" << std::endl;
  }
  // sanity of the auxiliary states (sanity_*; 0 = off): a solve whose camera-IMU rotation correction or visual scale
  // offset is far outside its normal range (otherwise kept by every later solve) is rejected like a non-finite one:
  // nothing of it is kept
  if (finite) {
    std::string why;
    if (cfg_.sanity_ext_deg > 0.0 && ext && ext->estimate().norm() * 180.0 / M_PI > cfg_.sanity_ext_deg) why = "ext";
    for (auto& kv : vscales)
      if (cfg_.sanity_vo_scale > 0.0 && std::abs(kv.second->estimate()[0]) > cfg_.sanity_vo_scale) why = "vo_scale";
    if (!why.empty()) {
      finite = false;
      ++rejected_solves_;
      ++counters_["rejected_insane_" + why];
      std::cout << "[S3 backend] solve rejected by the " << why << " sanity check (nodes " << lo << '-' << end << ")" << std::endl;
    }
  }
  const bool keep = finite || (cfg_.sanity_ext_deg <= 0.0 && cfg_.sanity_vo_scale <= 0.0);
  if (finite) for (auto& kv : poses) if (kv.first > lo) est_[kv.first] = FromVIPose(kv.second->estimate());
  if (keep) for (auto& kv : sinks) sink_[kv.first] = kv.second->estimate();
  if (keep) for (auto& kv : biases) gyro_bias_[kv.first] = kv.second->estimate();
  if (keep) for (auto& kv : vscales) vo_scale_[kv.first] = kv.second->estimate()[0];
  if (keep) for (auto& kv : vbiases) leg_vbias_[kv.first] = kv.second->estimate();
  if (finite) {
    for (auto& kv : vels) if (kv.second->estimate().allFinite()) imu_vel_[kv.first] = kv.second->estimate();
    for (auto& kv : bgs) if (!kv.second->fixed()) imu_bg_[kv.first] = kv.second->estimate();
    for (auto& kv : bas) if (!kv.second->fixed()) imu_ba_[kv.first] = kv.second->estimate();
    if (ext && cfg_.imu_ext_rot) imu_ext_ = ext->estimate();
    if (gdir && cfg_.imu_gravity_free) imu_Rwg_ = gdir->estimate().Rwg;
    while (!imu_vel_.empty() && imu_vel_.begin()->first + 2 * W < lo) imu_vel_.erase(imu_vel_.begin());
  }
  if (keep) landmark_sel_ = used_selection;
  if (land_dropped) landmark_sel_.clear();   // a fresh selection next time
  solved_ = end;
  if (cfg_.leg_reliability) LearnReliability(end);

  std::ostringstream log;
  log << std::fixed << std::setprecision(4) << "[S3 backend] nodes=" << lo << '-' << end << " visual=" << n_visual
      << " loops=" << n_loop << " lost=" << n_lost << " min_scale=" << min_scale << " leg=" << n_leg << " foot=" << n_foot - foot_outliers << '/'
      << n_foot << " gated(slope/rmse)=" << gated_slope << '/' << gated_rmse << " alpha(leg r/h/v, foot)="
      << alpha_.transpose() << " gyro_bias=" << LatestGyroBias().transpose()
      << " leg_vbias=" << (leg_vbias_.empty() ? Eigen::Vector3d::Zero() : leg_vbias_.back()).transpose() << " vce=" << vce_status
      << " marg=" << n_marg << " win_gated=" << n_win_gated << " land_gated=" << n_land_gated << " win_ms=" << win_ms << " vo_replaced=" << n_vo_replaced
      << " land=" << n_land << '/' << n_land_obs << '/' << land_outliers << " land_dropped=" << land_dropped
      << " zupt=" << n_zupt << " tilt=" << n_tilt << " gdir_deg=" << Eigen::AngleAxisd(imu_Rwg_).angle() * 180.0 / M_PI << " imu=" << n_imu << " imu_long=" << n_imu_long << " passive=" << n_passive
      << " bg=" << (imu_bg_.empty() ? Eigen::Vector3d::Zero() : imu_bg_.back()).transpose()
      << " ba=" << (imu_ba_.empty() ? Eigen::Vector3d::Zero() : imu_ba_.back()).transpose()
      << " ext_deg=" << imu_ext_.norm() * 180.0 / M_PI << " vo_scale=" << (vo_scale_.empty() ? 0.0 : vo_scale_.back())
      << " sink=" << (sink_.empty() ? 0.0 : sink_.back())
      << std::setprecision(2) << " solve_ms=" << solve_ms << " vce_ms=" << vce_ms;
  if (rel_theta_.size()) log << std::setprecision(4) << " rel_theta=" << rel_theta_.transpose().format(Eigen::IOFormat(4, 0, ",", ","));
  std::cout << log.str() << std::endl;

  PathCallback cb;
  {
    std::lock_guard<std::mutex> lock(out_mutex_);
    cb = path_cb_;
  }
  if (cb) {  // the recent part only (the LoRA supervision uses at most the last 120 s)
    std::vector<std::pair<double, Eigen::Isometry3d>> path;
    const double t0 = nodes_[end - 1].time - 150.0;  // node times increase: start by time, not at node 0
    size_t n = std::lower_bound(nodes_.begin(), nodes_.begin() + end, t0,
                                [](const BackendNode& a, double t) { return a.time < t; }) - nodes_.begin();
    for (; n < end; ++n) path.emplace_back(nodes_[n].time, est_[n]);
    cb(path);
  }

  std::lock_guard<std::mutex> lock(out_mutex_);
  log_.push_back(log.str());
  solve_ms_.push_back(solve_ms);
  vce_ms_.push_back(vce_ms);
  correction_ = est_[end - 1] * nodes_[end - 1].T_wc.inverse();
  if (cfg_.write_back) {
    for (size_t n = lo + 1; n < end; ++n) {
      write_back_[nodes_[n].id] = est_[n];
      nodes_[n].T_wc = est_[n];  // what the front-end will carry once the write-back is applied
    }
    newest_ = nodes_[end - 1].id;
  }
}

void FusionBackend::Save(const std::string& dir) {
  Stop();
  std::vector<std::pair<double, Eigen::Matrix4d>> final_poses, realtime;
  for (const auto& p : Trajectory()) final_poses.emplace_back(p.first, p.second.matrix());
  std::lock_guard<std::mutex> lock(out_mutex_);
  for (const auto& r : realtime_) realtime.emplace_back(r.first, r.second.matrix());
  SaveTumTrajectoryToFile((std::filesystem::path(dir) / "trajectory_backend.txt").string(), final_poses);
  SaveTumTrajectoryToFile((std::filesystem::path(dir) / "trajectory_backend_rt.txt").string(), realtime);

  auto summary = [](std::vector<double> v) {
    std::ostringstream s;
    if (v.empty()) return std::string("n=0");
    std::sort(v.begin(), v.end());
    double mean = 0.0;
    for (double x : v) mean += x;
    s << std::fixed << std::setprecision(3) << "n=" << v.size() << " mean=" << mean / v.size()
      << " p95=" << v[std::min(v.size() - 1, static_cast<size_t>(0.95 * v.size()))] << " max=" << v.back();
    return s.str();
  };
  std::ofstream stats((std::filesystem::path(dir) / "backend_stats.txt").string());
  stats << "solve_ms " << summary(solve_ms_) << "\nvce_ms " << summary(vce_ms_) << "\nwrite_back_ms "
        << summary(write_back_ms_) << "\nalpha " << alpha_.transpose() << "\nrejected_inputs " << rejected_inputs_
        << "\nrejected_solves " << rejected_solves_ << "\n";
  for (const auto& kv : counters_) stats << kv.first << ' ' << kv.second << '\n';
  for (const auto& l : log_) stats << l << "\n";

  {   // gyro bias states per node (B: worker, leg block; IMU frame: IMU block) and the IMU gyro sources
    std::ofstream bias((std::filesystem::path(dir) / "backend_bias.csv").string());
    bias << "id,time,bw_x,bw_y,bw_z,bleg_x,bleg_y,bleg_z,bimu_x,bimu_y,bimu_z,imu_net,imu_raw\n";
    const size_t GB = std::max(1, cfg_.gyro_bias_block), IB = std::max(1, cfg_.imu_bias_block);
    const Eigen::Vector3d nan3 = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    for (size_t n = 0; n < nodes_.size(); ++n) {
      const BackendNode& node = nodes_[n];
      const Eigen::Vector3d bl = n / GB < gyro_bias_.size() ? gyro_bias_[n / GB] : nan3;
      const Eigen::Vector3d bi = n / IB < imu_bg_.size() ? imu_bg_[n / IB] : nan3;
      bias << node.id << ',' << std::fixed << std::setprecision(9) << node.time << std::defaultfloat << std::setprecision(9);
      for (const Eigen::Vector3d* v : {&node.gyro_bias_applied, &bl, &bi}) bias << ',' << (*v)[0] << ',' << (*v)[1] << ',' << (*v)[2];
      bias << ',' << node.imu_net << ',' << node.imu_raw << '\n';
    }
  }
  std::ofstream nodes((std::filesystem::path(dir) / "backend_nodes.csv").string());
  std::ofstream feet((std::filesystem::path(dir) / "backend_feet.csv").string());
  nodes << std::setprecision(12) << "id,time,covisible,spread,has_leg,"
        << "vo_tx,vo_ty,vo_tz,vo_qx,vo_qy,vo_qz,vo_qw,leg_tx,leg_ty,leg_tz,leg_qx,leg_qy,leg_qz,leg_qw,"
        << "f_tx,f_ty,f_tz,f_qx,f_qy,f_qz,f_qw,cb_tx,cb_ty,cb_tz,cb_qx,cb_qy,cb_qz,cb_qw,superframe,slip_mean,rel_f0,rel_f1,rel_f2,rel_f3,rel_f4,lost,standing\n";
  feet << std::setprecision(12) << "id,anchor,fbx,fby,fbz,qx_c,qy_c,qz_c,nx_c,ny_c,nz_c,sigma,rmse,radius\n";
  auto iso = [](std::ostream& o, const Eigen::Isometry3d& T) {
    const Eigen::Quaterniond q(T.linear());
    o << ',' << T.translation().x() << ',' << T.translation().y() << ',' << T.translation().z() << ',' << q.x() << ','
      << q.y() << ',' << q.z() << ',' << q.w();
  };
  for (const auto& node : nodes_) {
    nodes << node.id << ',' << std::fixed << std::setprecision(9) << node.time << std::defaultfloat << std::setprecision(12) << ',' << node.covisible << ',' << node.spread << ',' << node.has_leg;
    iso(nodes, node.vo); iso(nodes, node.leg); iso(nodes, node.T_wc); iso(nodes, node.T_c_b);
    nodes << ',' << node.superframe << ',' << node.slip_mean;
    for (int k = 0; k < 5; ++k) nodes << ',' << (node.rel_feat.size() == 5 ? node.rel_feat[k] : std::nan(""));
    nodes << ',' << node.lost << ',' << node.standing << '\n';
    for (const auto& c : node.feet)
      feet << node.id << ',' << c.id_anchor << ',' << c.foot_b.x() << ',' << c.foot_b.y() << ',' << c.foot_b.z() << ','
           << c.q_c.x() << ',' << c.q_c.y() << ',' << c.q_c.z() << ',' << c.n_c.x() << ',' << c.n_c.y() << ','
           << c.n_c.z() << ',' << c.sigma << ',' << c.rmse << ',' << c.radius << '\n';
  }
  if (cfg_.export_extras) SaveBackendExtras(dir, nodes_);
}

// Learned leg reliability: nodes that have stayed leg_rel_lag nodes in the window give the log of their whitened
// leg translation residual against the back-end poses; ridge regression on the centred features. Only the
// relative weighting is learned (centred features, exp(0) = 1 at the mean); the absolute level stays with VCE.
void FusionBackend::LearnReliability(size_t end) {
  const size_t last = end > static_cast<size_t>(cfg_.leg_rel_lag) ? end - cfg_.leg_rel_lag : 0;
  const Eigen::Vector3d sigma(cfg_.leg_sigma[1], cfg_.leg_sigma[2], cfg_.leg_sigma[3]);
  for (; rel_done_ < last; ++rel_done_) {
    const BackendNode& node = nodes_[rel_done_];
    if (!node.has_leg || node.rel_feat.empty()) continue;
    const Eigen::Isometry3d rel = est_[rel_done_ - 1].inverse() * est_[rel_done_];
    const Eigen::Vector3d e = (node.leg.translation() - rel.translation()).cwiseQuotient(sigma);  // camera frame
    rel_X_.push_back(node.rel_feat);
    rel_y_.push_back(std::log(e.squaredNorm() / 3.0 + 1e-4));
    while (static_cast<int>(rel_y_.size()) > cfg_.leg_rel_history) { rel_X_.pop_front(); rel_y_.pop_front(); }
  }
  if (static_cast<int>(rel_y_.size()) < cfg_.leg_rel_min) return;
  const int n = rel_y_.size(), d = rel_X_.front().size();
  Eigen::MatrixXd X(n, d);
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; ++i) { X.row(i) = Eigen::Map<const Eigen::RowVectorXd>(rel_X_[i].data(), d); y[i] = rel_y_[i]; }
  rel_mean_ = X.colwise().mean().transpose();
  X.rowwise() -= rel_mean_.transpose();
  y.array() -= y.mean();
  const Eigen::MatrixXd A = X.transpose() * X + cfg_.leg_rel_ridge * Eigen::MatrixXd::Identity(d, d);
  rel_theta_ = A.ldlt().solve(X.transpose() * y);
}

// export_extras: sensor meta, IMU samples and map point observations, each in its own file (absent in older dumps):
//   backend_meta.csv   fx,fy,cx,cy,bf,has_imu,T_i_c (t xyz, q xyzw),gyr_noise,acc_noise,gyr_walk,acc_walk (one row)
//   backend_imu.csv    id,t,gx,gy,gz,ax,ay,az
//   backend_points.csv id,point,u,v,ur,px,py,pz,n_obs,ba_obs,ba_outliers
void SaveBackendExtras(const std::string& dir, const std::vector<BackendNode>& nodes) {
  const std::filesystem::path root(dir);
  const auto meta_it = std::find_if(nodes.begin(), nodes.end(), [](const BackendNode& n) { return n.meta.valid; });
  if (meta_it != nodes.end()) {
    const SensorMeta& m = meta_it->meta;
    const Eigen::Quaterniond q(m.T_i_c.linear());
    const Eigen::Vector3d& t = m.T_i_c.translation();
    std::ofstream f((root / "backend_meta.csv").string());
    f << std::setprecision(12) << "fx,fy,cx,cy,bf,has_imu,ti_x,ti_y,ti_z,qi_x,qi_y,qi_z,qi_w,gyr_noise,acc_noise,gyr_walk,acc_walk\n"
      << m.fx << ',' << m.fy << ',' << m.cx << ',' << m.cy << ',' << m.bf << ',' << m.has_imu << ',' << t.x() << ','
      << t.y() << ',' << t.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ',' << m.gyr_noise
      << ',' << m.acc_noise << ',' << m.gyr_walk << ',' << m.acc_walk << '\n';
  }
  std::ofstream imu((root / "backend_imu.csv").string()), pts((root / "backend_points.csv").string());
  imu << "id,t,gx,gy,gz,ax,ay,az\n";
  pts << std::setprecision(12) << "id,point,u,v,ur,px,py,pz,n_obs,ba_obs,ba_outliers\n";
  for (const auto& n : nodes) {
    for (const ImuData& d : n.imu)
      imu << n.id << ',' << std::fixed << std::setprecision(9) << d.timestamp << std::defaultfloat << std::setprecision(12)
          << ',' << d.gyr.x() << ',' << d.gyr.y() << ',' << d.gyr.z() << ',' << d.acc.x() << ',' << d.acc.y() << ',' << d.acc.z() << '\n';
    for (const LandmarkObs& o : n.obs)
      pts << n.id << ',' << o.point << ',' << o.kp.x() << ',' << o.kp.y() << ',' << o.kp.z() << ',' << o.pw.x() << ','
          << o.pw.y() << ',' << o.pw.z() << ',' << o.n_obs << ',' << o.ba_obs << ',' << o.ba_outliers << '\n';
  }
}

void LoadBackendExtras(const std::string& dir, std::vector<BackendNode>* nodes) {
  const std::filesystem::path root(dir);
  auto rows = [](const std::filesystem::path& path) {
    std::vector<std::vector<double>> out;
    std::ifstream f(path.string());
    if (!f) return out;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string cell;
      std::vector<double> row;
      while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
      if (!row.empty()) out.push_back(std::move(row));
    }
    return out;
  };
  std::map<int, BackendNode*> by_id;
  for (auto& n : *nodes) by_id[n.id] = &n;
  const auto meta_rows = rows(root / "backend_meta.csv");
  if (!meta_rows.empty() && meta_rows[0].size() >= 17) {
    const auto& r = meta_rows[0];
    SensorMeta meta;
    meta.valid = true;
    meta.fx = r[0]; meta.fy = r[1]; meta.cx = r[2]; meta.cy = r[3]; meta.bf = r[4];
    meta.has_imu = r[5] > 0.5;
    meta.T_i_c.translation() = Eigen::Vector3d(r[6], r[7], r[8]);
    meta.T_i_c.linear() = Eigen::Quaterniond(r[12], r[9], r[10], r[11]).normalized().toRotationMatrix();
    meta.gyr_noise = r[13]; meta.acc_noise = r[14]; meta.gyr_walk = r[15]; meta.acc_walk = r[16];
    for (auto& n : *nodes) n.meta = meta;
  }
  for (const auto& r : rows(root / "backend_imu.csv")) {
    auto it = by_id.find(static_cast<int>(r[0]));
    if (it == by_id.end() || r.size() < 8) continue;
    ImuData d;
    d.timestamp = r[1];
    d.gyr = Eigen::Vector3d(r[2], r[3], r[4]);
    d.acc = Eigen::Vector3d(r[5], r[6], r[7]);
    it->second->imu.push_back(d);
  }
  for (const auto& r : rows(root / "backend_points.csv")) {
    auto it = by_id.find(static_cast<int>(r[0]));
    if (it == by_id.end() || r.size() < 11) continue;
    LandmarkObs o;
    o.point = static_cast<long>(r[1]);
    o.kp = Eigen::Vector3d(r[2], r[3], r[4]);
    o.pw = Eigen::Vector3d(r[5], r[6], r[7]);
    o.n_obs = static_cast<int>(r[8]); o.ba_obs = static_cast<int>(r[9]); o.ba_outliers = static_cast<int>(r[10]);
    it->second->obs.push_back(o);
  }
}

bool LoadBackendDump(const std::string& dir, std::vector<BackendNode>* nodes, std::string* error) {
  auto read = [](const std::string& path, std::vector<std::vector<double>>* rows) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string cell;
      std::vector<double> row;
      while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
      rows->push_back(row);
    }
    return true;
  };
  auto iso = [](const std::vector<double>& r, int k) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = Eigen::Vector3d(r[k], r[k + 1], r[k + 2]);
    T.linear() = Eigen::Quaterniond(r[k + 6], r[k + 3], r[k + 4], r[k + 5]).normalized().toRotationMatrix();
    return T;
  };
  std::vector<std::vector<double>> node_rows, feet_rows;
  const std::filesystem::path root(dir);
  if (!read((root / "backend_nodes.csv").string(), &node_rows) || !read((root / "backend_feet.csv").string(), &feet_rows)) {
    if (error) *error = "missing backend_nodes.csv / backend_feet.csv in " + dir;
    return false;
  }
  std::map<int, VectorOfFootConstraints> feet;
  for (const auto& r : feet_rows) {
    FootSurfaceConstraint c;
    c.id_pose = static_cast<int>(r[0]);
    c.id_anchor = static_cast<int>(r[1]);
    c.foot_b = Eigen::Vector3d(r[2], r[3], r[4]);
    c.q_c = Eigen::Vector3d(r[5], r[6], r[7]);
    c.n_c = Eigen::Vector3d(r[8], r[9], r[10]);
    c.sigma = r[11];
    c.rmse = r[12];
    c.radius = r[13];
    feet[c.id_pose].push_back(c);
  }
  nodes->clear();
  for (const auto& r : node_rows) {
    BackendNode n;
    n.id = static_cast<int>(r[0]);
    n.time = r[1];
    n.covisible = static_cast<int>(r[2]);
    n.spread = r[3];
    n.has_leg = r[4] > 0.5;
    n.vo = iso(r, 5);
    n.leg = iso(r, 12);
    n.T_wc = iso(r, 19);
    n.T_c_b = iso(r, 26);
    n.superframe = r.size() > 33 && r[33] > 0.5;
    n.slip_mean = r.size() > 34 ? r[34] : -1.0;
    if (r.size() > 39 && std::isfinite(r[35])) n.rel_feat.assign(r.begin() + 35, r.begin() + 40);
    n.lost = r.size() > 40 && r[40] > 0.5;  // older dumps: no column
    n.standing = r.size() > 41 && r[41] > 0.5;  // passive (carried) motion test of the global back-end; older dumps: no column
    if (feet.count(n.id)) n.feet = feet[n.id];
    nodes->push_back(std::move(n));
  }
  LoadBackendExtras(dir, nodes);
  return true;
}
