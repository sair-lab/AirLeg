// S3 online fusion unit tests: foot-surface factor geometry/Jacobian/observability, fusion back-end,
// leg relative-pose factor, sink modes through LocalmapOptimization, stream slicing,
// and terrain plane fitting / step rejection on synthetic depth.
// Build: colcon ... --cmake-args -DAIRSLAM_BUILD_TESTS=ON; run: online_fusion_tests <camera.yaml>
#include <atomic>
#include <cmath>
#include <functional>
#include <random>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <g2o/core/block_solver.h>
#include <g2o/core/jacobian_workspace.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include "g2o_optimization/edge_leg_odom.h"
#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/g2o_optimization.h"
#include "online_fusion.h"
#include "fusion_backend.h"
#include "vce/statistics.h"

namespace {
int failures = 0;

struct ProbeEdge : EdgeFootSurface {
  using EdgeFootSurface::EdgeFootSurface;
  Eigen::MatrixXd Jacobian(int i) const { return _jacobianOplus[i]; }
};
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "[ OK ] " : "[FAIL] ") << what << std::endl;
  failures += !ok;
}

Eigen::Isometry3d Pose(double roll, double pitch, double yaw, const Eigen::Vector3d& t) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  T.translation() = t;
  return T;
}

// Camera looking forward (optical z along world x) and pitched down, at height h.
Eigen::Isometry3d CameraPose(double x, double h, double pitch_down) {
  Eigen::Matrix3d R_wc0;
  R_wc0 << 0, 0, 1, -1, 0, 0, 0, -1, 0;  // C z -> W x, C x -> W -y, C y -> W -z
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = R_wc0 * Eigen::AngleAxisd(-pitch_down, Eigen::Vector3d::UnitX()).toRotationMatrix();
  T.translation() << x, 0, h;
  return T;
}

VertexVIPose* AddPose(g2o::SparseOptimizer& opt, int id, const Eigen::Isometry3d& T_WC, bool fixed) {
  auto* v = new VertexVIPose();
  const Eigen::Matrix3d Rcw = T_WC.linear().transpose();
  v->setEstimate(VIPose(Rcw, -Rcw * T_WC.translation(), Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
  v->setId(id);
  v->setFixed(fixed);
  opt.addVertex(v);
  return v;
}

g2o::SparseOptimizer* MakeOptimizer() {
  auto* opt = new g2o::SparseOptimizer();
  auto linear = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
  linear->setWriteDebug(false);  // g2o would dump debug.txt into the cwd on a Cholesky failure
  opt->setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linear))));
  return opt;
}

Eigen::Isometry3d Estimate(const VertexVIPose* v) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = v->estimate().Rcw.transpose();
  T.translation() = -T.linear() * v->estimate().tcw;
  return T;
}

// Ground plane z=0 (and an optional step to z=step_h for world x >= step_x) seen by camera T_WC.
cv::Mat SyntheticDepth(const CameraPtr& camera, const Eigen::Isometry3d& T_WC, double step_x, double step_h) {
  cv::Mat depth(camera->ImageHeight(), camera->ImageWidth(), CV_32F, cv::Scalar(0));
  for (int v = 0; v < depth.rows; ++v) {
    for (int u = 0; u < depth.cols; ++u) {
      const Eigen::Vector3d ray_c((u - camera->Cx()) / camera->Fx(), (v - camera->Cy()) / camera->Fy(), 1.0);
      const Eigen::Vector3d ray_w = T_WC.linear() * ray_c, o = T_WC.translation();
      if (ray_w.z() >= -1e-6) continue;
      double s = -o.z() / ray_w.z();
      if ((o + s * ray_w).x() >= step_x) {  // hits the riser (x = step_x) or the upper level
        s = (step_x - o.x()) / ray_w.x();
        if ((o + s * ray_w).z() > step_h) s = (step_h - o.z()) / ray_w.z();
      }
      if (s > 0 && s < 4.0) depth.at<float>(v, u) = static_cast<float>(s);  // ray_c.z = 1 -> s is z-depth
    }
  }
  return depth;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: online_fusion_tests <camera.yaml>" << std::endl;
    return 2;
  }
  CameraPtr camera = std::make_shared<Camera>(argv[1]);
  const Eigen::Isometry3d T_C_B = Pose(-M_PI / 2, 0.0, -M_PI / 2, Eigen::Vector3d(0.03, 0.2, -0.15));

  // ---- 1. Foot-surface residual signs and rigid invariance --------------------------------
  {
    const Eigen::Isometry3d T_WA = CameraPose(0.0, 0.4, 0.5), T_WI = CameraPose(0.8, 0.4, 0.5);
    const Eigen::Vector3d n_w = Eigen::Vector3d::UnitZ(), q_w(1.0, 0.1, 0.0);
    const Eigen::Vector3d q_c = T_WA.inverse() * q_w, n_c = T_WA.linear().transpose() * n_w;
    const double a = 0.022;
    // Choose f_B so the sphere centre sits a - delta above the plane.
    auto foot_for = [&](double height) {
      const Eigen::Vector3d p_w(0.9, 0.05, height);
      return Eigen::Vector3d((T_WI * T_C_B).inverse() * p_w);
    };
    auto residual = [&](const Eigen::Isometry3d& world, const Eigen::Vector3d& foot, double delta) {
      std::unique_ptr<g2o::SparseOptimizer> opt(MakeOptimizer());
      auto* vi = AddPose(*opt, 0, world * T_WI, false);
      auto* va = AddPose(*opt, 1, world * T_WA, true);
      auto* vd = new VertexLegTimeDelay(-0.02, 0.05);
      vd->setId(2); vd->setEstimate(delta); opt->addVertex(vd);
      EdgeFootSurface e(T_C_B, foot, q_c, n_c, a);
      e.setVertex(0, vi); e.setVertex(1, va); e.setVertex(2, vd);
      e.computeError();
      const double r = e.error()[0];
      e.setVertex(0, nullptr); e.setVertex(1, nullptr); e.setVertex(2, nullptr);
      return r;
    };
    const Eigen::Isometry3d I = Eigen::Isometry3d::Identity();
    Check(std::abs(residual(I, foot_for(a), 0.0)) < 1e-9, "rigid sphere contact: r = 0");
    Check(std::abs(residual(I, foot_for(a - 0.015), 0.015)) < 1e-9, "sinkage delta = 1.5 cm: r = 0");
    Check(std::abs(residual(I, foot_for(a + 0.01), -0.01)) < 1e-9, "resting 1 cm above mean plane, delta = -1 cm: r = 0");
    Check(std::abs(residual(I, foot_for(a + 0.03), 0.0) - 0.03) < 1e-9, "3 cm above surface: r = +3 cm");
    const Eigen::Isometry3d G = Pose(0.3, -0.2, 1.1, Eigen::Vector3d(5, -2, 1));
    Check(std::abs(residual(G, foot_for(a + 0.02), 0.0) - residual(I, foot_for(a + 0.02), 0.0)) < 1e-9,
          "same world rigid motion of foot and anchor leaves r unchanged");
  }

  // ---- 2. Jacobian (g2o linearisation vs central differences through VIPose::Update) -------
  {
    std::unique_ptr<g2o::SparseOptimizer> opt(MakeOptimizer());
    const Eigen::Isometry3d T_WA = CameraPose(0.0, 0.4, 0.5), T_WI = CameraPose(0.7, 0.42, 0.45);
    auto* vi = AddPose(*opt, 0, T_WI, false);
    auto* va = AddPose(*opt, 1, T_WA, true);
    auto* vd = new VertexLegTimeDelay(-0.02, 0.05);
    vd->setId(2); vd->setEstimate(0.01); opt->addVertex(vd);
    auto* e = new ProbeEdge(T_C_B, Eigen::Vector3d(0.2, 0.12, -0.33), T_WA.inverse() * Eigen::Vector3d(1, 0, 0),
                                  T_WA.linear().transpose() * Eigen::Vector3d::UnitZ(), 0.022);
    e->setVertex(0, vi); e->setVertex(1, va); e->setVertex(2, vd);
    e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    opt->addEdge(e);
    g2o::JacobianWorkspace workspace;
    workspace.updateSize(e);
    workspace.allocate();
    e->linearizeOplus(workspace);
    const Eigen::MatrixXd J = e->Jacobian(0);
    double max_diff = 0.0;
    for (int c = 0; c < 6; ++c) {
      double plus[6] = {0}, minus[6] = {0};
      plus[c] = 1e-6; minus[c] = -1e-6;
      vi->push(); vi->oplus(plus); e->computeError(); const double rp = e->error()[0]; vi->pop();
      vi->push(); vi->oplus(minus); e->computeError(); const double rm = e->error()[0]; vi->pop();
      max_diff = std::max(max_diff, std::abs((rp - rm) / 2e-6 - J(0, c)));
    }
    Check(max_diff < 1e-5, "pose Jacobian matches central differences (max diff " + std::to_string(max_diff) + ")");
    Check(std::abs(e->Jacobian(2)(0, 0) - 1.0) < 1e-6, "d r / d delta = 1");
  }

  // ---- 2b. Analytic Jacobians of all edges vs central differences (VIPose with a body offset) ----
  {
    std::unique_ptr<g2o::SparseOptimizer> opt(MakeOptimizer());
    const Eigen::Isometry3d T_CB = Pose(0.1, -0.2, 0.3, Eigen::Vector3d(0.05, -0.02, 0.1));
    int next_id = 0;
    auto pose = [&](const Eigen::Isometry3d& T_WC) {
      auto* v = new VertexVIPose();
      const Eigen::Matrix3d Rcw = T_WC.linear().transpose();
      v->setEstimate(VIPose(Rcw, -Rcw * T_WC.translation(), T_CB.linear(), T_CB.translation()));
      v->setId(next_id++);
      opt->addVertex(v);
      return v;
    };
    // compares every vertex block of linearizeOplus with central differences through oplus
    auto compare = [&](g2o::OptimizableGraph::Edge* e, const std::string& name) {
      g2o::JacobianWorkspace ws;
      ws.updateSize(e);
      ws.allocate();
      e->linearizeOplus(ws);
      const int d = e->dimension();
      double worst = 0.0;
      for (size_t i = 0; i < e->vertices().size(); ++i) {
        auto* v = static_cast<g2o::OptimizableGraph::Vertex*>(e->vertices()[i]);
        const Eigen::Map<Eigen::MatrixXd> J(ws.workspaceForVertex(i), d, v->dimension());
        for (int c = 0; c < v->dimension(); ++c) {
          std::vector<double> dx(v->dimension(), 0.0);
          dx[c] = 1e-6;
          v->push(); v->oplus(dx.data()); e->computeError();
          const Eigen::VectorXd plus = Eigen::Map<const Eigen::VectorXd>(e->errorData(), d);
          v->pop();
          dx[c] = -1e-6;
          v->push(); v->oplus(dx.data()); e->computeError();
          const Eigen::VectorXd numeric = (plus - Eigen::Map<const Eigen::VectorXd>(e->errorData(), d)) / 2e-6;
          v->pop();
          worst = std::max(worst, (numeric - J.col(c)).cwiseAbs().maxCoeff() / std::max(1.0, numeric.cwiseAbs().maxCoeff()));
        }
      }
      e->computeError();
      Check(worst < 1e-6, name + ": analytic Jacobian matches central differences (" + std::to_string(worst) + ")");
    };
    const Eigen::Isometry3d T_W1 = CameraPose(0.0, 0.4, 0.5), T_W2 = Pose(0.05, -0.03, 0.2, Eigen::Vector3d(0.3, 0.1, 0.02)) * T_W1;
    auto* v1 = pose(T_W1);
    auto* v2 = pose(T_W2);
    auto* point = new g2o::VertexPointXYZ();
    point->setId(next_id++);
    point->setEstimate(T_W1 * Eigen::Vector3d(0.3, -0.2, 2.5));
    opt->addVertex(point);
    auto* mono = new EdgeSE3ProjectPoint();
    mono->fx = 380; mono->fy = 381; mono->cx = 320; mono->cy = 240;
    mono->setVertex(0, point); mono->setVertex(1, v2); mono->setMeasurement(Eigen::Vector2d(300, 250));
    compare(mono, "EdgeSE3ProjectPoint");
    auto* stereo = new EdgeSE3ProjectStereoPoint();
    stereo->fx = 380; stereo->fy = 381; stereo->cx = 320; stereo->cy = 240; stereo->bf = 19.0;
    stereo->setVertex(0, point); stereo->setVertex(1, v2); stereo->setMeasurement(Eigen::Vector3d(300, 250, 292));
    compare(stereo, "EdgeSE3ProjectStereoPoint");
    for (double noise : {0.002, 0.2}) {  // both branches of SO3Log (small-angle and exact)
      auto* rel = new EdgeLegRelativePose();
      rel->setVertex(0, v1); rel->setVertex(1, v2);
      rel->setMeasurement(Pose(noise, -noise, 0.5 * noise, Eigen::Vector3d(0.01, 0.02, -0.03)) * T_W1.inverse() * T_W2);
      compare(rel, "EdgeLegRelativePose (residual " + std::to_string(noise) + " rad)");
      delete rel;
    }
    auto* bias = new VertexVec<3>();
    bias->setId(next_id++);
    bias->setEstimate(Eigen::Vector3d(0.01, -0.02, 0.015));
    opt->addVertex(bias);
    auto* gyro = new EdgeLegGyroBias(T_C_B, 0.5);
    gyro->setVertex(0, v1); gyro->setVertex(1, v2); gyro->setVertex(2, bias);
    gyro->setMeasurement(T_C_B.inverse() * Pose(0.01, 0.0, -0.01, Eigen::Vector3d::Zero()) * T_W1.inverse() * T_W2 * T_C_B);
    compare(gyro, "EdgeLegGyroBias");
    auto* delay = new VertexLegTimeDelay(-0.02, 0.05);
    delay->setId(next_id++);
    delay->setEstimate(0.01);
    opt->addVertex(delay);
    auto* foot = new EdgeFootSurface(T_C_B, Eigen::Vector3d(0.2, 0.12, -0.33), T_W1.inverse() * Eigen::Vector3d(1, 0, 0),
                                     T_W1.linear().transpose() * Eigen::Vector3d::UnitZ(), 0.022);
    foot->setVertex(0, v2); foot->setVertex(1, v1); foot->setVertex(2, delay);
    compare(foot, "EdgeFootSurface");
    delete mono; delete stereo; delete gyro; delete foot;
  }

  // ---- 3. Leg relative pose edge: zero error at truth ---------------------------------------
  {
    std::unique_ptr<g2o::SparseOptimizer> opt(MakeOptimizer());
    const Eigen::Isometry3d T1 = CameraPose(0.0, 0.4, 0.5), T2 = Pose(0.01, 0.02, 0.1, Eigen::Vector3d(0.3, 0.05, 0)) * T1;
    auto* v1 = AddPose(*opt, 0, T1, true);
    auto* v2 = AddPose(*opt, 1, T2, false);
    auto* e = new EdgeLegRelativePose();
    e->setVertex(0, v1); e->setVertex(1, v2);
    e->setMeasurement(T1.inverse() * T2);
    e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    opt->addEdge(e);
    e->computeError();
    Check(e->error().norm() < 1e-9, "EdgeLegRelativePose: zero error for the true relative pose");
  }

  // ---- 4. Leg edges through LocalmapOptimization (foot-surface edges are back-end only) -------
  {
    const Eigen::Isometry3d T_WA = CameraPose(0.0, 0.4, 0.5), T_WI = CameraPose(0.8, 0.4, 0.5);
    std::vector<CameraPtr> cameras{camera};
    MapOfPoses poses;
    for (auto [id, T, fixed] : {std::make_tuple(1, T_WA, true), std::make_tuple(2, T_WI, false)}) {
      Pose3d p; p.fixed = fixed; p.id_camera = 0; p.R = T.linear(); p.p = T.translation();
      poses[id] = p;
    }
    poses[2].p += Eigen::Vector3d(0.05, -0.03, 0.02);  // perturbed initial guess
    MapOfPoints3d points; MapOfVelocity velocities; MapOfBias biases;
    VectorOfMonoPointConstraints mono; VectorOfStereoPointConstraints stereo; VectorOfIMUConstraints imu;
    LocalBAExtras extras;
    extras.T_C_B = T_C_B;
    LegRelativeConstraint leg;
    leg.id_pose1 = 1; leg.id_pose2 = 2; leg.T_c1_c2 = T_WA.inverse() * T_WI;
    leg.information = Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
    extras.legs.push_back(leg);
    OptimizationConfig cfg; cfg.mono_point = 50; cfg.stereo_point = 75;
    LocalmapOptimization(poses, points, velocities, biases, cameras, mono, stereo, imu, Eigen::Matrix3d::Identity(), cfg, &extras);
    Check((poses.at(2).p - T_WI.translation()).norm() < 1e-6, "leg-only local BA recovers the free pose");
  }

  // ---- 5. Stream slice integrates like the full buffer --------------------------------------
  {
    LegOdomStream stream(10.0);
    LegOdomBuffer::Samples all;
    for (int k = 0; k <= 500; ++k) {
      LegVelocitySample s;
      s.time = 100.0 + k * 0.008;
      s.linear << 0.5 + 0.1 * std::sin(k * 0.05), 0.05, 0.0;
      s.angular << 0.0, 0.01, 0.2 * std::cos(k * 0.03);
      s.variance.setConstant(1e-4);
      all.push_back(s);
      stream.Push(s, FootState());
    }
    LegOdomBuffer full(all), slice;
    Eigen::Isometry3d a, b;
    const bool ok = stream.Slice(100.5013, 103.2007, &slice) &&
        full.Integrate(100.5013, 103.2007, Vector6d::Zero(), &a) && slice.Integrate(100.5013, 103.2007, Vector6d::Zero(), &b);
    Check(ok && (a.matrix() - b.matrix()).norm() < 1e-12, "LegOdomStream::Slice + Integrate == full buffer");
    Check(!stream.Slice(99.0, 101.0, &slice), "uncovered interval rejected");
  }

  // ---- 6. Terrain map: flat plane fit and step rejection --------------------------------------
  {
    SurfaceConfig cfg;
    cfg.stride = 1;
    for (double step_h : {0.0, 0.10}) {
      SuperframeMap map(cfg, camera);
      FramePtr frame = std::make_shared<Frame>(7, false, camera, 1.0);
      const Eigen::Isometry3d T_WC = CameraPose(0.0, 0.45, 0.6);
      frame->SetPose(T_WC.matrix());
      map.InsertDepth(frame, [&] {  // gravity-aligned frame as in OnKeyframe
        const Eigen::Matrix3d R_CW = T_WC.linear().transpose();
        const Eigen::Vector3d z = R_CW * Eigen::Vector3d::UnitZ();
        Eigen::Vector3d x = (Eigen::Vector3d::UnitZ() - z * z.z()).normalized();
        Eigen::Matrix3d R; R << x, z.cross(x), z; return R;
      }(), SyntheticDepth(camera, T_WC, 1.0, step_h));
      auto anchors = map.Anchors([](int) { return true; });
      if (anchors.size() != 1) { Check(false, "superframe inserted"); continue; }
      const auto& record = *anchors[0];
      auto plane_at = [&](double x) {
        const Eigen::Vector3d p_w(x, 0.0, 0.022);
        return map.FitPlane(record, record.R_CG.transpose() * (T_WC.inverse() * p_w));
      };
      const auto flat = plane_at(0.8);
      const Eigen::Vector3d n_w = T_WC.linear() * (record.R_CG * flat.n);
      Check(flat.reject == SurfaceReject::COUNT && n_w.z() > 0.999 && flat.rmse < 1e-3,
            "plane inside one level accepted, normal up (step " + std::to_string(step_h) + ")");
      if (step_h > 0.0) {
        Check(plane_at(0.95).reject == SurfaceReject::STEP_REGION, "foot 5 cm from a 10 cm step: STEP_REGION");
      }
    }
  }


  // ---- Light-weight back-end on synthetic chains -------------------------------------------
  {
    EdgeScalarRandomWalk walk;
    VertexLegTimeDelay a(-1, 1), b(-1, 1);
    a.setEstimate(0.01); b.setEstimate(0.035);
    walk.setVertex(0, &a); walk.setVertex(1, &b); walk.setMeasurement(0.0);
    walk.computeError();
    Check(std::abs(walk.error()[0] - 0.025) < 1e-12, "scalar random walk residual = x_j - x_i");
    walk.setVertex(0, nullptr); walk.setVertex(1, nullptr);

    // Camera looks along the walking direction (optical z forward, y down); world z up.
    Eigen::Matrix3d R0;
    R0 << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    const int N = 400;
    std::vector<Eigen::Isometry3d> truth(N);
    for (int k = 0; k < N; ++k) {
      truth[k] = Eigen::Isometry3d::Identity();
      const double yaw = 0.01 * k + 0.3 * std::sin(0.02 * k);
      truth[k].linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0;
      truth[k].translation() = Eigen::Vector3d::Zero();
      if (k > 0) truth[k].translation() = truth[k - 1].translation() + 0.3 * Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0);
    }
    auto noisy = [](const Eigen::Isometry3d& rel, double sr, const Eigen::Vector3d& st, std::mt19937& g,
                    const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
      std::normal_distribution<double> n01;
      const Eigen::Vector3d w(sr * n01(g), sr * n01(g), sr * n01(g));
      Eigen::Isometry3d d = Eigen::Isometry3d::Identity();
      if (w.norm() > 0) d.linear() = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix();
      d.translation() = Eigen::Vector3d(st.x() * n01(g), st.y() * n01(g), st.z() * n01(g)) + bias;
      return rel * d;
    };
    const Eigen::Vector3d vo_t(0.05, 0.02, 0.05), leg_t(0.017, 0.035, 0.017);
    // customize(k, node) may edit the node (visual/leg measurements, support, feet).
    auto run = [&](BackendConfig cfg, int n, double leg_scale, const std::function<void(int, BackendNode&)>& customize,
                   Eigen::Vector4d* alpha = nullptr) {
      std::mt19937 g(7);
      FusionBackend backend(cfg);
      for (int k = 0; k < n; ++k) {
        BackendNode node;
        node.id = k + 1;
        node.time = 0.5 * k;
        node.T_wc = truth[k];
        node.covisible = 200;
        node.spread = 1.0;
        if (k > 0) {
          const Eigen::Isometry3d rel = truth[k - 1].inverse() * truth[k];
          node.vo = noisy(rel, 0.002, vo_t, g);
          node.leg = noisy(rel, 0.0094 * leg_scale, leg_t * leg_scale, g);
          node.has_leg = true;
        }
        customize(k, node);
        backend.Push(node);
      }
      backend.Stop();
      if (alpha) *alpha = backend.Alpha();
      std::vector<Eigen::Isometry3d> out;
      for (const auto& p : backend.Trajectory()) out.push_back(p.second);
      return out;
    };
    auto chain = [&](int n, bool leg, double leg_scale) {
      std::mt19937 g(7);
      std::vector<Eigen::Isometry3d> out{truth[0]};
      for (int k = 1; k < n; ++k) {
        const Eigen::Isometry3d rel = truth[k - 1].inverse() * truth[k];
        const Eigen::Isometry3d vo = noisy(rel, 0.002, vo_t, g), lg = noisy(rel, 0.0094 * leg_scale, leg_t * leg_scale, g);
        out.push_back(out.back() * (leg ? lg : vo));
      }
      return out;
    };
    auto error = [&](const std::vector<Eigen::Isometry3d>& est, int axis = -1) {
      double sum = 0.0;
      for (size_t k = 0; k < est.size(); ++k) {
        const Eigen::Vector3d d = est[k].translation() - truth[k].translation();
        sum += axis < 0 ? d.squaredNorm() : d[axis] * d[axis];
      }
      return std::sqrt(sum / est.size());
    };
    auto none = [](int, BackendNode&) {};
    BackendConfig cfg;
    cfg.enabled = true; cfg.vce = false; cfg.foot = false;
    const int M = 200;
    const double e_fused = error(run(cfg, M, 1.0, none)), e_vo = error(chain(M, false, 1.0)), e_leg = error(chain(M, true, 1.0));
    Check(e_fused < e_vo && e_fused < e_leg, "back-end: VO rotation + leg translation beat both chains (" +
          std::to_string(e_fused) + " vs VO " + std::to_string(e_vo) + ", leg " + std::to_string(e_leg) + ")");

    // VCE: leg translation 4x noisier than configured -> its horizontal/vertical scales grow towards 16.
    BackendConfig vce_cfg = cfg;
    vce_cfg.vce = true; vce_cfg.vce_upper = 100.0;
    Eigen::Vector4d alpha;
    run(vce_cfg, N, 1.0, [&](int k, BackendNode& node) {
      if (k == 0) return;
      std::mt19937 g(1000 + k);
      const Eigen::Isometry3d rel = truth[k - 1].inverse() * truth[k];
      node.leg = noisy(rel, 0.0094, leg_t * 4.0, g);
    }, &alpha);
    Check(alpha[1] > 6.0 && alpha[2] > 6.0 && alpha[0] > 0.4 && alpha[0] < 2.5,
          "VCE: leg translation scales grow, rotation stays ~1 (alpha " + std::to_string(alpha[0]) + ", " +
          std::to_string(alpha[1]) + ", " + std::to_string(alpha[2]) + ")");

    // Degeneracy: a broken visual edge with no covisible support is dropped; the leg edge carries the pair.
    auto broken = [&](int covisible) {
      return [&, covisible](int k, BackendNode& node) {
        if (k != 30) return;
        node.vo.translation() += Eigen::Vector3d(1.0, 0.0, 0.0);
        node.covisible = covisible;
      };
    };
    const double e_clean = error(run(cfg, 80, 1.0, none)), e_lost = error(run(cfg, 80, 1.0, broken(5)));
    const double e_trusted = error(run(cfg, 80, 1.0, broken(200)));
    Check(std::abs(e_lost - e_clean) < 0.01 && e_trusted > e_lost, "lost visual edge dropped (" + std::to_string(e_lost) +
          " m, clean " + std::to_string(e_clean) + " m, trusted " + std::to_string(e_trusted) + " m)");

    // Foot edges: VO and leg both drift upwards; contacts on flat ground hold the height. Anchors are the
    // superframes every 10 nodes (a FIXED lag would let the contact offset absorb a constant drift rate).
    BackendConfig foot_cfg = cfg;
    foot_cfg.foot = true;
    auto feet = [&](double tilt_deg, double plane_offset) {
      return [&, tilt_deg, plane_offset](int k, BackendNode& node) {
        if (k > 0) {
          Eigen::Isometry3d up = Eigen::Isometry3d::Identity();
          up.translation() = Eigen::Vector3d(0.0, -0.006, 0.0);  // camera y is down
          node.vo = node.vo * up;
          node.leg = node.leg * up;
        }
        const int anchor = 10 * ((k - 1) / 10);
        if (k < 1) return;
        const Eigen::Vector3d p_w = truth[k] * Eigen::Vector3d(0.0, 0.4, 0.3) + Eigen::Vector3d(0, 0, 0);
        const Eigen::Vector3d foot_w(p_w.x(), p_w.y(), 0.022);
        const Eigen::Vector3d n_w = Eigen::AngleAxisd(tilt_deg * M_PI / 180.0, Eigen::Vector3d::UnitX()) * Eigen::Vector3d::UnitZ();
        FootSurfaceConstraint c;
        c.id_pose = node.id;
        c.id_anchor = anchor + 1;
        c.foot_b = truth[k].inverse() * foot_w;
        c.q_c = truth[anchor].inverse() * Eigen::Vector3d(p_w.x(), p_w.y(), plane_offset);
        c.n_c = truth[anchor].linear().transpose() * n_w;
        c.radius = 0.022; c.sigma = 0.01; c.rmse = 0.002;
        node.feet.push_back(c);
      };
    };
    const double z_none = error(run(cfg, M, 1.0, feet(0.0, 0.0)), 2);
    const double z_foot = error(run(foot_cfg, M, 1.0, feet(0.0, 0.0)), 2);
    const double z_gated = error(run(foot_cfg, M, 1.0, feet(15.0, 0.1)), 2);
    Check(z_foot < 0.5 * z_none, "foot edges hold the height (" + std::to_string(z_foot) + " vs " + std::to_string(z_none) + " m)");
    Check(std::abs(z_gated - z_none) < 1e-6, "foot edges on a 15 deg plane are slope-gated");
  }

  // ---- FFS disparity post-processing: GPU path bit-identical to the CPU reference ------------------
  {
    SurfaceConfig sc;
    sc.enabled = true;
    SuperframeMap map(sc, camera);
    const int w = camera->ImageWidth(), h = camera->ImageHeight();
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    int identical = 0, total_points = 0;
    const int trials = 6;
    for (int t = 0; t < trials; ++t) {
      // ground plane disparity with a step, noise, holes, NaN / inf / negative / tiny values
      std::vector<float> disparity(static_cast<size_t>(w) * h);
      for (int v = 0; v < h; ++v)
        for (int u = 0; u < w; ++u) {
          float d = 2.0f + 40.0f * std::max(0.0f, (v - 0.4f * h) / h) + (u > w / 2 + 30 * t ? 6.0f : 0.0f);
          d += 0.3f * (uni(rng) - 0.5f);
          const float r = uni(rng);
          if (r < 0.01f) d = std::nanf("");
          else if (r < 0.015f) d = INFINITY;
          else if (r < 0.02f) d = -1.0f;
          else if (r < 0.04f) d = 0.0f;
          disparity[static_cast<size_t>(v) * w + u] = d;
        }
      const Eigen::Matrix3d R_CG = (Eigen::AngleAxisd(0.3 * t - 0.7, Eigen::Vector3d(0.2, 1.0, 0.1).normalized()) *
                                    Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX())).toRotationMatrix();
      std::vector<float> xyz_cpu, xyz_gpu;
      std::vector<int> keys_cpu, keys_gpu;
      map.DepthPoints(R_CG, map.DepthFromDisparity(disparity, w, h), &xyz_cpu, &keys_cpu);
      const bool ok = map.DisparityPointsGpu(disparity, w, h, R_CG, &xyz_gpu, &keys_gpu);
      identical += ok && xyz_cpu == xyz_gpu && keys_cpu == keys_gpu;
      total_points += xyz_cpu.size() / 3;
    }
    Check(identical == trials && total_points > 1000, "GPU depth post-processing bit-identical to the CPU reference (" +
          std::to_string(identical) + "/" + std::to_string(trials) + " images, " + std::to_string(total_points) + " points)");
  }

  // ---- Sparse VCE statistics against the dense reference ----------------------------------------
  {
    // chain of 40 3-dof states: absolute observations (group 0, fixed), differences (groups 1-2) and
    // a few long-range differences (group 3, fill-in); exact moments from an explicit inverse
    std::mt19937 rng(7);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const int nodes = 40, n = 3 * nodes;
    std::vector<std::vector<Eigen::Triplet<double>>> trip(4);
    std::vector<int> m(4, 0);
    auto add = [&](int g, int a, int b) {
      for (int r = 0; r < 3; ++r, ++m[g])
        for (int c = 0; c < 3; ++c) {
          trip[g].emplace_back(m[g], 3 * a + c, gauss(rng) + (r == c ? 2.0 : 0.0));
          if (b >= 0) trip[g].emplace_back(m[g], 3 * b + c, gauss(rng) - (r == c ? 2.0 : 0.0));
        }
    };
    for (int i = 0; i < nodes; i += 4) add(0, i, -1);
    for (int i = 0; i + 1 < nodes; ++i) { add(1, i, i + 1); add(2, i + 1, i); }
    for (int i = 0; i + 7 < nodes; i += 5) add(3, i, i + 7);
    std::vector<vce::Group> sparse(4), dense(4);
    for (int g = 0; g < 4; ++g) {
      sparse[g].sparse_jacobian.resize(m[g], n);
      sparse[g].sparse_jacobian.setFromTriplets(trip[g].begin(), trip[g].end());
      sparse[g].residual = Eigen::VectorXd::NullaryExpr(m[g], [&] { return (1.0 + g) * gauss(rng); });
      sparse[g].fixed = g == 0;
      dense[g] = sparse[g];
      dense[g].jacobian = Eigen::MatrixXd(sparse[g].sparse_jacobian);
      dense[g].sparse_jacobian = Eigen::SparseMatrix<double>();
    }
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n, n);
    for (const auto& g : dense) h += g.jacobian.transpose() * g.jacobian;
    const Eigen::MatrixXd inv = h.inverse();
    Eigen::VectorXd nu(4);
    Eigen::MatrixXd S(4, 4);
    for (int s = 0; s < 4; ++s) {
      const Eigen::MatrixXd xs = inv * dense[s].jacobian.transpose() * dense[s].jacobian;
      nu[s] = m[s] - xs.trace();
      for (int t = 0; t < 4; ++t) {
        const Eigen::MatrixXd xt = inv * dense[t].jacobian.transpose() * dense[t].jacobian;
        S(s, t) = (xs * xt).trace() + (s == t ? m[s] - 2 * xs.trace() : 0.0);
      }
    }
    vce::Limits complete;
    complete.complete_moments = true;
    const vce::Statistics a = vce::Estimate(sparse, complete), b = vce::Estimate(dense, complete);
    const vce::Statistics c = vce::Estimate(sparse);
    Check(a.valid && b.valid && c.valid, "sparse VCE valid (" + a.reason + b.reason + c.reason + ")");
    Check((a.nu - nu).cwiseAbs().maxCoeff() < 1e-9 && (a.S - S).cwiseAbs().maxCoeff() < 1e-9,
          "sparse VCE redundancies and moments match the explicit inverse (" +
          std::to_string((a.S - S).cwiseAbs().maxCoeff()) + ")");
    Check((a.beta - b.beta).cwiseAbs().maxCoeff() < 1e-12 && (a.beta - c.beta).cwiseAbs().maxCoeff() < 1e-12,
          "dense / sparse Jacobian and partial moments give the same beta");
    std::vector<vce::Group> singular = sparse;  // a state no group observes
    for (auto& g : singular) g.sparse_jacobian.col(n - 1) *= 0.0;
    Check(!vce::Estimate(singular).valid, "unobserved state rejected");
  }

  // ---- Back-end residual gyro bias ------------------------------------------------------------
  {
    const Eigen::Isometry3d Tcb = Pose(-M_PI / 2, 0.0, -M_PI / 2, Eigen::Vector3d(0.03, 0.2, -0.15));
    const Eigen::Isometry3d body = Pose(0.01, -0.02, 0.05, Eigen::Vector3d(0.3, 0.02, 0.01));
    std::unique_ptr<g2o::SparseOptimizer> opt(MakeOptimizer());
    auto* vi = AddPose(*opt, 0, Pose(0, 0, 0, Eigen::Vector3d::Zero()), false);
    auto* vj = AddPose(*opt, 1, Pose(0.1, 0.0, 0.2, Eigen::Vector3d(0.1, 0.3, 0.0)), false);
    VertexVec<3> vb;
    vb.setEstimate(Eigen::Vector3d::Zero());
    EdgeLegGyroBias eb(Tcb, 0.4);
    eb.setVertex(0, vi); eb.setVertex(1, vj); eb.setVertex(2, &vb); eb.setMeasurement(body);
    EdgeLegRelativePose er;
    er.setVertex(0, vi); er.setVertex(1, vj); er.setMeasurement(Tcb * body * Tcb.inverse());
    eb.computeError(); er.computeError();
    Check((eb.error() - er.error()).norm() < 1e-9, "gyro-bias leg edge with zero bias == relative-pose edge");
    eb.setVertex(2, nullptr);

    // synthetic chain (camera = body): leg rotation drifts with a 0.003 rad/s bias about the camera y axis
    auto run = [&](bool on) {
      BackendConfig c;
      c.enabled = true; c.foot = false; c.vce = false; c.leg_slip = false; c.gyro_bias = on;
      FusionBackend be(c);
      Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
      step.translation() = Eigen::Vector3d(0, 0, 0.3);
      step.linear() = Eigen::AngleAxisd(0.02, Eigen::Vector3d::UnitY()).toRotationMatrix();
      for (int k = 0; k < 200; ++k) {
        BackendNode n;
        n.id = k + 1; n.time = 0.5 * k; n.covisible = 200; n.spread = 1.0;
        if (k > 0) {
          n.vo = step;
          n.leg = step;
          n.leg.linear() = step.linear() * Eigen::AngleAxisd(0.003 * 0.5, Eigen::Vector3d::UnitY()).toRotationMatrix();
          n.has_leg = true;
        }
        be.Push(n);
      }
      be.Stop();
      return be.LatestGyroBias();
    };
    const Eigen::Vector3d b_on = run(true), b_off = run(false);
    Check(std::abs(b_on.y() - 0.003) < 0.0005 && b_off.norm() == 0.0,
          "back-end recovers a 0.003 rad/s residual gyro bias (" + std::to_string(b_on.y()) + ")");
  }

  // ---- Slip modulation of the leg translation ------------------------------------------------
  {
    auto run = [](bool slip) {
      BackendConfig c;
      c.enabled = true; c.foot = false; c.vce = false; c.leg_slip = slip;
      c.vo_sigma = {0.002, 0.05, 0.05, 0.05};  // visual weaker than the leg in translation
      FusionBackend b(c);
      Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
      step.translation() = Eigen::Vector3d(0.0, 0.0, 0.3);
      for (int k = 0; k < 60; ++k) {
        BackendNode n;
        n.id = k + 1; n.time = 0.5 * k; n.covisible = 200; n.spread = 1.0;
        if (k > 0) {
          n.vo = step; n.leg = step; n.has_leg = true; n.slip_mean = 0.01;
          if (k == 30) { n.leg.translation() += Eigen::Vector3d(0.2, 0.0, 0.0); n.slip_mean = 0.2; }  // slipping
        }
        b.Push(n);
      }
      b.Stop();
      double worst = 0.0;
      const auto traj = b.Trajectory();
      for (size_t k = 0; k < traj.size(); ++k) worst = std::max(worst, std::abs(traj[k].second.translation().x()));
      return worst;
    };
    const double off = run(false), on = run(true);
    Check(on < 0.6 * off, "slipping stance feet down-weight the leg translation (" + std::to_string(on) + " vs " +
          std::to_string(off) + " m lateral error)");
  }

  // ---- Back-end solve trigger ---------------------------------------------------------------
  {
    auto solves = [](const std::string& trigger, int every_superframe, double* error) {
      BackendConfig c;
      c.enabled = true; c.foot = false; c.vce = false; c.trigger = trigger; c.stride = 10;
      FusionBackend b(c);
      std::mt19937 g(3);
      std::normal_distribution<double> n01;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      std::vector<Eigen::Isometry3d> truth;
      for (int k = 0; k < 60; ++k) {
        BackendNode n;
        n.id = k + 1; n.time = 0.5 * k; n.covisible = 200; n.spread = 1.0;
        n.superframe = every_superframe > 0 && k % every_superframe == every_superframe - 1;
        Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
        step.translation() = Eigen::Vector3d(0.0, 0.0, 0.3);
        if (k > 0) {
          T = T * step;
          n.vo = step; n.vo.translation() += Eigen::Vector3d(0.03 * n01(g), 0.01 * n01(g), 0.03 * n01(g));
          n.leg = step; n.leg.translation() += Eigen::Vector3d(0.01 * n01(g), 0.02 * n01(g), 0.01 * n01(g));
          n.has_leg = true;
        }
        truth.push_back(T);
        b.Push(n);
      }
      b.Stop();
      double sum = 0.0;
      const auto traj = b.Trajectory();
      for (size_t k = 0; k < traj.size(); ++k) sum += (traj[k].second.translation() - truth[k].translation()).squaredNorm();
      *error = std::sqrt(sum / traj.size());
      return b.Solves();
    };
    double e_super = 0.0, e_stride = 0.0, e_fallback = 0.0;
    const size_t n_super = solves("superframe", 3, &e_super), n_stride = solves("stride", 0, &e_stride);
    const size_t n_fallback = solves("superframe", 0, &e_fallback);
    Check(n_super == 20, "superframe trigger: one solve per superframe (" + std::to_string(n_super) + " solves for 20)");
    Check(n_fallback == n_stride && n_stride == 6, "no superframe: fallback every stride nodes (" + std::to_string(n_fallback) + ")");
    Check(e_super < 1.2 * e_stride + 1e-3, "superframe-triggered trajectory as accurate as the stride one (" +
          std::to_string(e_super) + " vs " + std::to_string(e_stride) + " m)");
  }

  // ---- Learned leg reliability and the back-end path callback ---------------------------------
  {
    // leg translation noise x5 when feature 0 is high (30% of the nodes); feature 1 is irrelevant
    auto run = [](bool reliability, double* error, size_t* paths) {
      BackendConfig c;
      c.enabled = true; c.foot = false; c.vce = false; c.trigger = "stride"; c.stride = 10;
      c.leg_reliability = reliability; c.leg_slip = false; c.gyro_bias = false;
      FusionBackend b(c);
      std::atomic<size_t> calls{0};
      b.SetPathCallback([&calls](const std::vector<std::pair<double, Eigen::Isometry3d>>& p) { if (!p.empty()) ++calls; });
      std::mt19937 g(5);
      std::normal_distribution<double> n01;
      std::uniform_real_distribution<double> u01;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      std::vector<Eigen::Isometry3d> truth;
      for (int k = 0; k < 600; ++k) {
        BackendNode n;
        n.id = k + 1; n.time = 0.5 * k; n.covisible = 200; n.spread = 1.0;
        const bool bad = u01(g) < 0.3;
        n.rel_feat = {bad ? 1.0 : 0.0, u01(g)};
        Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
        step.translation() = Eigen::Vector3d(0.0, 0.0, 0.3);
        if (k > 0) {
          T = T * step;
          n.vo = step; n.vo.translation() += Eigen::Vector3d(0.05 * n01(g), 0.02 * n01(g), 0.05 * n01(g));
          const double s = bad ? 5.0 : 1.0;
          n.leg = step; n.leg.translation() += s * Eigen::Vector3d(0.017 * n01(g), 0.035 * n01(g), 0.017 * n01(g));
          n.has_leg = true;
        }
        truth.push_back(T);
        b.Push(n);
      }
      b.Stop();
      double sum = 0.0;
      const auto traj = b.Trajectory();
      for (size_t k = 1; k < traj.size(); ++k) {
        const Eigen::Vector3d d = (traj[k - 1].second.inverse() * traj[k].second).translation() -
                                  (truth[k - 1].inverse() * truth[k]).translation();
        sum += d.squaredNorm();
      }
      *error = std::sqrt(sum / (traj.size() - 1));
      *paths = calls.load();
      return b.Solves();
    };
    double off = 0.0, on = 0.0;
    size_t paths = 0, unused = 0;
    run(false, &off, &unused);
    const size_t solves = run(true, &on, &paths);
    Check(on < 0.9 * off, "learned leg reliability down-weights the noisy leg edges (" + std::to_string(on) + " vs " +
          std::to_string(off) + " m step error)");
    Check(paths == solves && solves > 0, "back-end path callback after every solve (" + std::to_string(paths) + "/" +
          std::to_string(solves) + ")");
  }

  // ---- Global back-end (S4): one window over the whole trajectory, loop edge, initial poses --------------
  {
    // square loop (4 x 25 steps of 0.4 m, 90 deg turns); VO and leg both carry a small yaw drift
    auto run = [](bool loop, bool init, double* end_error) {
      BackendConfig c;
      c.enabled = true; c.foot = false; c.vce = false; c.leg_slip = false; c.gyro_bias = false;
      c.window = 100000; c.stride = 100000; c.trigger = "stride"; c.iterations = 30;
      FusionBackend b(c);
      std::vector<Eigen::Isometry3d> truth;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      const int N = 100;
      for (int k = 0; k < N; ++k) {
        Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
        step.translation() = Eigen::Vector3d(0.0, 0.0, 0.4);
        if (k % 25 == 24) step.linear() = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitY()).toRotationMatrix();
        if (k > 0) T = T * step;
        truth.push_back(T);
      }
      if (loop) b.AddLoop(1, N, truth[0].inverse() * truth[N - 1]);
      for (int k = 0; k < N; ++k) {
        BackendNode n;
        n.id = k + 1; n.time = 0.5 * k; n.covisible = 200; n.spread = 1.0;
        if (k > 0) {
          Eigen::Isometry3d drift = Eigen::Isometry3d::Identity();
          drift.linear() = Eigen::AngleAxisd(0.004, Eigen::Vector3d::UnitY()).toRotationMatrix();
          n.vo = truth[k - 1].inverse() * truth[k] * drift;
          n.leg = n.vo;
          n.has_leg = true;
        }
        if (init) { n.has_init = true; n.T_init = truth[k]; }
        b.Push(n);
      }
      b.Stop();
      const auto traj = b.Trajectory();
      *end_error = (traj.back().second.translation() - truth.back().translation()).norm();
      return traj.size();
    };
    double open = 0.0, closed = 0.0, closed_init = 0.0;
    run(false, false, &open);
    run(true, false, &closed);
    const size_t n = run(true, true, &closed_init);
    Check(closed < 0.3 * open, "global back-end: a loop edge removes the drift (" + std::to_string(closed) + " vs " +
          std::to_string(open) + " m end error)");
    Check(n == 100 && closed_init < 0.3 * open, "global back-end: initial poses accepted (" + std::to_string(closed_init) + " m)");
  }

  // ---- Leg data are not used before T_C_B is known ---------------------------------------
  {
    auto constraint = [&](bool latched, bool in_config) {
      OnlineFusionConfig cfg;
      cfg.mode = "fusion_only";
      cfg.surface.enabled = false;
      cfg.backend.enabled = false;
      cfg.leg.has_extrinsic = in_config;
      OnlineFusion online(cfg, camera);
      for (int k = 0; k <= 125; ++k) {
        LegVelocitySample sample;
        sample.time = k * 0.008;
        sample.linear = Eigen::Vector3d(0.5, 0.0, 0.0);
        online.Stream().Push(sample, FootState());
      }
      if (latched) online.Stream().SetExtrinsic(T_C_B);
      FramePtr f1 = std::make_shared<Frame>(1, false, camera, 0.2), f2 = std::make_shared<Frame>(2, false, camera, 0.6);
      LegRelativeConstraint c;
      return online.LegConstraint(f1, f2, &c);
    };
    Check(!constraint(false, false), "no leg constraint before the leg node publishes T_C_B");
    Check(constraint(true, false), "leg constraint once T_C_B is latched");
    Check(constraint(false, true), "leg constraint with T_C_B given in the config");
  }

  // ---- Late leg data: tracking queries wait; cached constraints are identical ---------------
  {
    OnlineFusionConfig cfg;
    cfg.mode = "fusion_only";
    cfg.surface.enabled = false;
    cfg.backend.enabled = false;
    OnlineFusion online(cfg, camera);
    online.Stream().SetExtrinsic(T_C_B);
    auto push = [&](int k0, int k1) {
      for (int k = k0; k <= k1; ++k) {
        LegVelocitySample sample;
        sample.time = k * 0.008;
        sample.linear = Eigen::Vector3d(0.5, 0.01 * k, 0.0);
        sample.angular = Eigen::Vector3d(0.0, 0.0, 0.1);
        online.Stream().Push(sample, FootState());
      }
    };
    push(0, 60);  // covers up to 0.48 s
    FramePtr f1 = std::make_shared<Frame>(1, false, camera, 0.2), f2 = std::make_shared<Frame>(2, false, camera, 0.6);
    LegRelativeConstraint c0, c1, c2;
    Check(!online.LegConstraint(f1, f2, &c0), "late leg data without waiting: no constraint");
    std::thread producer([&] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); push(61, 125); });
    const bool waited = online.LegConstraint(f1, f2, &c1, 0.5);
    producer.join();
    Check(waited, "late leg data: the tracking query waits for the stream");
    Check(online.LegConstraint(f1, f2, &c2) && c1.T_c1_c2.matrix() == c2.T_c1_c2.matrix() &&
          c1.information == c2.information, "cached leg constraint identical to the computed one");
    const std::string stats = online.LegStats();
    Check(stats.find("late=1") != std::string::npos && stats.find("ok_after_wait=1") != std::string::npos &&
          stats.find("cached=1") != std::string::npos, "leg query counters (" + stats + ")");
  }

  std::cout << (failures ? "FAILED: " : "ALL PASSED") << (failures ? std::to_string(failures) : "") << std::endl;
  return failures ? 1 : 0;
}
