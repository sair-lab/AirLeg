#include "g2o_optimization/g2o_optimization.h"

#include <fstream>
#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "read_configs.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/edge_imu.h"
#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/edge_relative_pose.h"
#include "g2o_optimization/edge_leg_odom.h"

bool AddPointConstraint(int id_pose, int id_point, const Eigen::Vector3d& keypoint,
                        VectorOfMonoPointConstraints& mono, VectorOfStereoPointConstraints& stereo){
  if(keypoint(2) > 0){
    StereoPointConstraintPtr c = std::make_shared<StereoPointConstraint>();
    c->id_pose = id_pose;
    c->id_point = id_point;
    c->id_camera = 0;
    c->inlier = true;
    c->keypoint = keypoint;
    c->pixel_sigma = 0.8;
    stereo.push_back(c);
    return true;
  }
  MonoPointConstraintPtr c = std::make_shared<MonoPointConstraint>();
  c->id_pose = id_pose;
  c->id_point = id_point;
  c->id_camera = 0;
  c->inlier = true;
  c->keypoint = keypoint.head(2);
  c->pixel_sigma = 0.8;
  mono.push_back(c);
  return false;
}

void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, bool fix_this_frame){
  int frame_id = frame->GetFrameId();
  Eigen::Matrix4d& frame_pose = frame->GetPose();
  Pose3d pose;
  pose.R = frame_pose.block<3, 3>(0, 0);
  pose.p = frame_pose.block<3, 1>(0, 3);
  pose.id_camera = id_camera;
  pose.fixed = fix_this_frame;
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));
}

void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, MapOfVelocity& velocities, MapOfBias& biases,
    VectorOfIMUConstraints& imu_constraints, bool fix_this_frame, bool add_imu_constraint, bool use_updated_bias){
  int frame_id = frame->GetFrameId();
  FramePtr last_frame = frame->PreviousFrame();
  PreinterationPtr preinteration = frame->GetIMUPreinteration();

  Eigen::Matrix4d& frame_pose = frame->GetPose();
  Pose3d pose;
  pose.R = frame_pose.block<3, 3>(0, 0);
  pose.p = frame_pose.block<3, 1>(0, 3);
  pose.id_camera = id_camera;
  pose.fixed = fix_this_frame;
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));

  Bias bias;
  if(use_updated_bias){
    bias.gyr_bias = preinteration->bg + preinteration->dbg;
    bias.acc_bias = preinteration->ba + preinteration->dba;
  }else{
    bias.gyr_bias = preinteration->bg;
    bias.acc_bias = preinteration->ba;
  }
  bias.fixed = fix_this_frame;
  biases.insert(std::pair<int, Bias>(frame_id, bias));

  Velocity velocity;
  velocity.velocity = frame->GetVelocity();
  velocity.fixed = fix_this_frame;
  velocities.insert(std::pair<int, Velocity>(frame_id, velocity));

  if(last_frame != nullptr && add_imu_constraint){
    IMUConstraintPtr imu_constraint = std::shared_ptr<ImuConstraint>(new ImuConstraint());
    imu_constraint->id_pose1 = last_frame->GetFrameId();
    imu_constraint->id_pose2 = frame_id;
    imu_constraint->id_camera1 = 0;
    imu_constraint->id_camera2 = 0;
    imu_constraint->preinteration = preinteration;
    imu_constraints.emplace_back(imu_constraint);
  }
}

void AddLegEdges(g2o::SparseOptimizer& optimizer, const VectorOfLegConstraints& legs, double huber){
  for(const LegRelativeConstraint& c : legs){
    auto* v1 = dynamic_cast<VertexVIPose*>(optimizer.vertex(c.id_pose1));
    auto* v2 = dynamic_cast<VertexVIPose*>(optimizer.vertex(c.id_pose2));
    if(!v1 || !v2 || (v1->fixed() && v2->fixed())) continue;
    auto* e = new EdgeLegRelativePose();
    e->setVertex(0, v1);
    e->setVertex(1, v2);
    e->setMeasurement(c.T_c1_c2);
    e->setInformation(c.information);
    auto* rk = new g2o::RobustKernelHuber;
    rk->setDelta(huber);
    e->setRobustKernel(rk);
    optimizer.addEdge(e);
  }
}

void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg,
    LocalBAExtras* extras){

  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
  solver->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    VIPose vi_pose = ToVIPose(kv.second.R, kv.second.p, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  }
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : points){
    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first+max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 5. velocity vertex
  int max_velocity_id = max_point_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_point_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
  }
  max_velocity_id++;

  // 6. bias vertex
  int max_bias_id = max_velocity_id;
  for(auto& kv : biases){
    VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(kv.second.gyr_bias);
    int gyr_bias_id = max_velocity_id + kv.first * 2;
    gyr_bias_vertex->setId(gyr_bias_id);
    gyr_bias_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(gyr_bias_vertex);

    VertexAccBias* acc_bias_vertex = new VertexAccBias(kv.second.acc_bias);
    int acc_bias_id = max_velocity_id + kv.first * 2 + 1;
    max_bias_id = std::max(max_bias_id, acc_bias_id);
    acc_bias_vertex->setId(acc_bias_id);
    acc_bias_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(acc_bias_vertex);
  }
  max_bias_id++;

  // 7. gravity direction vertex, will be fixed in this function.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(true);
  optimizer.addVertex(gravity_direction_vertex);

  // 8. point edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges;
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  // 8.1 mono point edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mpc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // 8.2 stereo point edges
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((spc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(spc->id_pose)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
  }

  // 10 imu edges
  std::vector<EdgeIMU*> imu_edges;
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    // 10.1 pose and velocity edges
    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);

    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_point_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2);
    g2o::HyperGraph::Vertex *va1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2 + 1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_point_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2);
    g2o::HyperGraph::Vertex *va2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2 + 1);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);
    if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    if(poses[ipc->id_pose1].fixed || poses[ipc->id_pose2].fixed || false){
      g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
      e_imu->setRobustKernel(rki);
      e_imu->setInformation(e_imu->information() * 1e-2);
      rki->setDelta(sqrt(16.92));
    }
    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);

    // bias edges
    EdgeGyr* e_gyr = new EdgeGyr();
    EdgeAcc* e_acc = new EdgeAcc();

    e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
    e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
    e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

    Eigen::Matrix3d info_g = ipc->preinteration->Cov.block<3,3>(9,9).inverse();
    Eigen::Matrix3d info_a = ipc->preinteration->Cov.block<3,3>(12,12).inverse();
    e_gyr->setInformation(info_g);
    e_acc->setInformation(info_a);

    optimizer.addEdge(e_gyr);
    optimizer.addEdge(e_acc);
    gyr_edges.push_back(e_gyr);
    acc_edges.push_back(e_acc);
  }

  // S3 leg relative-pose edges (foot-surface edges are back-end only)
  if(extras) AddLegEdges(optimizer, extras->legs, extras->leg_huber);

  // solve
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  // check inlier observations
  for(size_t i=0; i < mono_edges.size(); i++){
    EdgeSE3ProjectPoint* e = mono_edges[i];
    if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < stereo_edges.size(); i++){
    EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
    if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
        e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  // optimize again without the outliers
  optimizer.initializeOptimization(0);
  optimizer.optimize(15);

  // check inlier observations
  for(size_t i = 0; i < mono_edges.size(); i++){
    EdgeSE3ProjectPoint* e = mono_edges[i];
    mono_point_constraints[i]->inlier = (e->chi2() <= cfg.mono_point && e->isDepthPositive());
  }

  for(size_t i = 0; i < stereo_edges.size(); i++){
    EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
    stereo_point_constraints[i]->inlier = (e->chi2() <= cfg.stereo_point && e->isDepthPositive());
  }

  // recover optimized data
  // keyframes
  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }

  // points
  for(MapOfPoints3d::iterator it = points.begin(); it != points.end(); ++it){
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(it->first+max_frame_id));
    it->second.p = point_vertex->estimate();
  }

  // velocities
  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_point_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases
  for(MapOfBias::iterator it = biases.begin(); it != biases.end(); it++){
    VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(it->first*2+max_velocity_id));
    VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(it->first*2+max_velocity_id+1));
    it->second.gyr_bias = gyr_bias_vertex->estimate();
    it->second.acc_bias = acc_bias_vertex->estimate();
  }

}

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg,
    const VectorOfLegConstraints* leg_constraints, double leg_huber){

  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
  solver->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  VIPose current_pose;
  VertexVIPose* current_frame;
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    VIPose vi_pose = ToVIPose(kv.second.R, kv.second.p, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->updateCache();
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    frame_vertex->setId(kv.first);
    if(!kv.second.fixed){
      current_frame = frame_vertex;
      current_pose = vi_pose;
    }
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  }
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : points){
    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first+max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 5. velocity vertex
  VertexVelocity* velocity_vertex_debug;
  int max_velocity_id = max_point_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_point_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
    if(!kv.second.fixed) velocity_vertex_debug = velocity_vertex;
  }
  max_velocity_id++;

  // 6. bias vertex
  int max_bias_id = max_velocity_id;
  for(auto& kv : biases){
    VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(kv.second.gyr_bias);
    int gyr_bias_id = max_velocity_id + kv.first * 2;
    gyr_bias_vertex->setId(gyr_bias_id);
    gyr_bias_vertex->setFixed(kv.second.fixed);
    // gyr_bias_vertex->setFixed(true);
    optimizer.addVertex(gyr_bias_vertex);
    // std::cout << "kv.second.gyr_bias = " << kv.second.gyr_bias.transpose() << std::endl;

    VertexAccBias* acc_bias_vertex = new VertexAccBias(kv.second.acc_bias);
    int acc_bias_id = max_velocity_id + kv.first * 2 + 1;
    max_bias_id = std::max(max_bias_id, acc_bias_id);
    acc_bias_vertex->setId(acc_bias_id);
    acc_bias_vertex->setFixed(kv.second.fixed);
    // acc_bias_vertex->setFixed(true);
    optimizer.addVertex(acc_bias_vertex);
  }
  max_bias_id++;

  // 7. gravity direction vertex, will be fixed in this function.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(true);
  optimizer.addVertex(gravity_direction_vertex);

  // 8. point edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges;
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  // 8.1 mono point edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mpc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // 8.2 stereo point edges
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((spc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(spc->id_pose)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
  }

  // 10 imu edges
  std::vector<EdgeIMU*> imu_edges;
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    // 10.1 pose and velocity edges
    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);

    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_point_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2);
    g2o::HyperGraph::Vertex *va1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2 + 1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_point_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2);
    g2o::HyperGraph::Vertex *va2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2 + 1);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);
    if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    if(poses[ipc->id_pose1].fixed || poses[ipc->id_pose2].fixed || false){
      g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
      e_imu->setRobustKernel(rki);
      e_imu->setInformation(e_imu->information() * 1e-2);
      rki->setDelta(sqrt(16.92));
    }
    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);

    // bias edges
    EdgeGyr* e_gyr = new EdgeGyr();
    EdgeAcc* e_acc = new EdgeAcc();

    e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
    e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
    e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

    Eigen::Matrix3d info_g = ipc->preinteration->Cov.block<3,3>(9,9).inverse();
    Eigen::Matrix3d info_a = ipc->preinteration->Cov.block<3,3>(12,12).inverse();
    e_gyr->setInformation(info_g);
    e_acc->setInformation(info_a);

    optimizer.addEdge(e_gyr);
    optimizer.addEdge(e_acc);
    gyr_edges.push_back(e_gyr);
    acc_edges.push_back(e_acc);
  }

  if(leg_constraints) AddLegEdges(optimizer, *leg_constraints, leg_huber);

  // solve
  const int its[4]={10, 10, 10, 10};

  int num_outlier = 0;
  for(size_t iter = 0; iter < 3; iter++){
    current_frame->setEstimate(current_pose);
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[iter]);

    num_outlier=0;
    for(size_t i = 0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      if(!mono_point_constraints[i]->inlier){
        e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.mono_point){
        mono_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        mono_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }

      if(iter == 2) e->setRobustKernel(0);
    }

    for(size_t i = 0; i < stereo_edges.size(); i++){
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
      if(!stereo_point_constraints[i]->inlier){
         e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.stereo_point){
        stereo_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        stereo_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if(iter == 2) e->setRobustKernel(0);
    }

    if(optimizer.edges().size()<10) break;
  }

  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }

  // velocities
  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_point_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases
  for(MapOfBias::iterator it = biases.begin(); it != biases.end(); it++){
    VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(it->first*2+max_velocity_id));
    VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(it->first*2+max_velocity_id+1));
    it->second.gyr_bias = gyr_bias_vertex->estimate();
    it->second.acc_bias = acc_bias_vertex->estimate();
  }

  return (mono_point_constraints.size() + stereo_point_constraints.size() - num_outlier);
}

bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg){
  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
  solver->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // frame vertex
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    VIPose vi_pose = ToVIPose(kv.second.R, kv.second.p, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  }
  max_frame_id++;

  // velocity vertex
  int max_velocity_id = max_frame_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_frame_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
  }
  max_velocity_id++;

  // bias vertex
  Eigen::Vector3d prior_gyr_bias = bias.gyr_bias;
  Eigen::Vector3d prior_acc_bias = bias.acc_bias;
  int max_bias_id = max_velocity_id;
  VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(prior_gyr_bias);
  gyr_bias_vertex->setId(max_bias_id);
  gyr_bias_vertex->setFixed(false);
  optimizer.addVertex(gyr_bias_vertex);
  max_bias_id++;

  VertexAccBias* acc_bias_vertex = new VertexAccBias(prior_acc_bias);
  acc_bias_vertex->setId(max_bias_id);
  acc_bias_vertex->setFixed(false);
  optimizer.addVertex(acc_bias_vertex);
  max_bias_id++;

  VertexGyrBias* prior_gyr_bias_vertex = new VertexGyrBias(prior_gyr_bias);
  prior_gyr_bias_vertex->setId(max_bias_id);
  prior_gyr_bias_vertex->setFixed(true);
  optimizer.addVertex(prior_gyr_bias_vertex);
  max_bias_id++;

  VertexAccBias* prior_acc_bias_vertex = new VertexAccBias(prior_acc_bias);
  prior_acc_bias_vertex->setId(max_bias_id);
  prior_acc_bias_vertex->setFixed(true);
  optimizer.addVertex(prior_acc_bias_vertex);
  max_bias_id++;

  // gravity direction vertex.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(false);
  optimizer.addVertex(gravity_direction_vertex);

  // prior bias edges
  g2o::HyperGraph::Vertex *vg = optimizer.vertex(max_velocity_id);
  g2o::HyperGraph::Vertex *va = optimizer.vertex(max_velocity_id + 1);
  g2o::HyperGraph::Vertex *pvg = optimizer.vertex(max_velocity_id + 2);
  g2o::HyperGraph::Vertex *pva = optimizer.vertex(max_velocity_id + 3);

  EdgeGyr* e_gyr_prior = new EdgeGyr();
  EdgeAcc* e_acc_prior = new EdgeAcc();
  e_gyr_prior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pvg));
  e_gyr_prior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg));
  e_gyr_prior->setInformation(1e2 * Eigen::Matrix3d::Identity());
  e_acc_prior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va));
  e_acc_prior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pva));
  e_acc_prior->setInformation(1e5 * Eigen::Matrix3d::Identity());
  optimizer.addEdge(e_gyr_prior);
  optimizer.addEdge(e_acc_prior);

  // imu edges
  std::vector<EdgeIMU*> imu_edges;
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;
  std::vector<EdgeGyr*> prior_gyr_edges;
  std::vector<EdgeAcc*> prior_acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_frame_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_frame_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);

    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);
    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);
  }

  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(200);

  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_frame_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases
  bias.gyr_bias = gyr_bias_vertex->estimate();
  bias.acc_bias = acc_bias_vertex->estimate();
  std::cout << "gyr_bias = " << bias.gyr_bias.transpose() << ", acc_bias = " << bias.acc_bias.transpose() << std::endl;

  Rwg = NormalizeRotation(gravity_direction_vertex->estimate().Rwg);
  return true;
}

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints,
    Eigen::Matrix4d& pose, std::vector<int>& inliers, double reprojection_error){
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<int> point_indexes;
  cv::Mat camera_matrix, dist_coeffs;
  CameraPtr camera = frame->GetCamera();
  camera->GetCamerMatrix(camera_matrix);
  camera->GetDistCoeffs(dist_coeffs);
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat cv_inliers;

  for(size_t i = 0; i < mappoints.size(); i++){
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint;
    if(!frame->GetKeypointPosition(i, keypoint)) continue;
    const Eigen::Vector3d& point_position = mpt->GetPosition();
    object_points.emplace_back(point_position(0), point_position(1), point_position(2));
    image_points.emplace_back(keypoint(0), keypoint(1));
    point_indexes.emplace_back(i);
  }
  if(object_points.size() < 8) return 0;

  try{
    cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs,
        rotation_vector, translation_vector, false, 200, reprojection_error, 0.999, cv_inliers);
  }catch(...){
    return 0;
  }

  if(cv_inliers.rows < 4 || rotation_vector.empty() || translation_vector.empty()) return 0;

  cv::Mat cv_Rcw;
  cv::Rodrigues(rotation_vector, cv_Rcw);
  Eigen::Matrix3d eigen_Rcw;
  Eigen::Vector3d eigen_tcw;
  cv::cv2eigen(cv_Rcw, eigen_Rcw);
  cv::cv2eigen(translation_vector, eigen_tcw);
  Eigen::Matrix3d eigen_Rwc = eigen_Rcw.transpose();
  pose.block<3, 3>(0, 0) = eigen_Rwc;
  pose.block<3, 1>(0, 3) = eigen_Rwc * (-eigen_tcw);

  inliers = std::vector<int>(mappoints.size(), -1);
  for(int i = 0; i < cv_inliers.rows; i++){
    int inlier_idx = cv_inliers.at<int>(i, 0);
    int point_idx = point_indexes[inlier_idx];
    inliers[point_idx] = mappoints[point_idx]->GetId();
  }
  return cv_inliers.rows;
}

bool ComputeGyrBias(std::vector<FramePtr>& frames, Eigen::Vector3d& dbg){
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();

  for(size_t i = 0; i < frames.size() - 1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];
    Eigen::Matrix3d Ai = frame->GetIMUPreinteration()->JRg;
    Eigen::Matrix3d delta_R = last_frame->IMUPose().block<3, 3>(0, 0).transpose() * frame->IMUPose().block<3, 3>(0, 0);
    Eigen::Vector3d delta_r;
    SO3Log(frame->GetIMUPreinteration()->dR.transpose() * delta_R, delta_r);
    Eigen::Vector3d bi = delta_r;

    A += Ai.transpose() * Ai;
    b += Ai.transpose() * bi;
  }

  dbg = A.ldlt().solve(b);

  return true;
}

bool ComputeVelocity(std::vector<FramePtr>& frames, Eigen::Vector3d& gw){
  int N = frames.size();
  int D = 3 * (N+1);
  Eigen::MatrixXd A{D, D};
  Eigen::VectorXd b{D};

  A.setZero();
  b.setZero();

  for(int i = 0; i < N-1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];

    double dt = frame->GetIMUPreinteration()->dT;
    Eigen::Matrix4d pose_i = last_frame->IMUPose();
    Eigen::Matrix4d pose_j = frame->IMUPose();

    Eigen::MatrixXd tmp_A(6, 9);
    Eigen::VectorXd tmp_b(6);

    tmp_A.setZero();
    tmp_b.setZero();

    tmp_A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = -dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 6) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();

    tmp_b.block<3, 1>(0, 0) = pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dV;
    tmp_b.block<3, 1>(3, 0) = pose_j.block<3, 1>(0, 3) - pose_i.block<3, 1>(0, 3) - pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dP;

    Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
    Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

    A.block<6, 6>(i*3, i*3) += r_A.topLeftCorner<6, 6>();
    A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
    A.block<6, 3>(i*3, D-3) += r_A.topRightCorner<6, 3>();
    A.block<3, 6>(D-3, i*3) += r_A.bottomLeftCorner<3, 6>();

    b.segment<6>(i*3) += r_b.head<6>();
    b.tail<3>() += r_b.tail<3>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  Eigen::VectorXd x = A.ldlt().solve(b);
  gw = x.tail<3>();

  std::cout << "x = " << x.transpose() << std::endl;

  for(int i = 0; i < N; i++){
    frames[i]->SetVelocaity(x.segment<3>(i*3));
  }

  // ValidateVelocity(frames, x);

  return true;
}

void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
    VectorOfRelativePoseConstraints& relative_pose_constraints){
  if(poses.empty() || camera_list.empty() || relative_pose_constraints.empty()){
    return;
  }

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
  solver->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    VIPose vi_pose = ToVIPose(kv.second.R, kv.second.p, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
  }

  // 3. add relative pose constraints
  std::vector<EdgeRelativePose*> relative_pose_edges;
  relative_pose_edges.reserve(relative_pose_constraints.size());
  std::vector<bool> relative_pose_is_loop;
  relative_pose_is_loop.reserve(relative_pose_constraints.size());
  for(RelativePoseConstraintPtr& rpc : relative_pose_constraints){
    EdgeRelativePose* e = new EdgeRelativePose();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(rpc->id_pose1)));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(rpc->id_pose2)));
    e->Rc1c2 = rpc->Rc1c2;
    e->tc1c2 = rpc->tc1c2;
    e->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * rpc->information_scale);
    if(rpc->robust){
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(std::sqrt(12.592));
      e->setRobustKernel(rk);
    }
    optimizer.addEdge(e);
    relative_pose_edges.emplace_back(e);
    relative_pose_is_loop.emplace_back(rpc->robust);
  }

  // First solve with robust loop edges, then reject inconsistent loops and
  // refine without robust kernels.
  optimizer.initializeOptimization();
  if(optimizer.optimize(20) <= 0){
    for(RelativePoseConstraintPtr& rpc : relative_pose_constraints){
      if(rpc->robust) rpc->active = false;
    }
    std::cout << "Reject pose graph result: optimizer failed." << std::endl;
    return;
  }

  int accepted_loops = 0;
  int rejected_loops = 0;
  for(size_t i = 0; i < relative_pose_edges.size(); i++){
    if(!relative_pose_is_loop[i]) continue;
    EdgeRelativePose* e = relative_pose_edges[i];
    e->computeError();
    const double chi2 = e->chi2();
    if(!std::isfinite(chi2) || chi2 > 25.0){
      e->setLevel(1);
      relative_pose_constraints[i]->active = false;
      rejected_loops++;
      std::cout << "Reject pose graph loop: "
                << relative_pose_constraints[i]->id_pose1 << " -> "
                << relative_pose_constraints[i]->id_pose2
                << ", chi2=" << chi2 << std::endl;
    }else{
      accepted_loops++;
    }
    e->setRobustKernel(nullptr);
  }

  optimizer.initializeOptimization(0);
  if(optimizer.optimize(30) <= 0){
    for(RelativePoseConstraintPtr& rpc : relative_pose_constraints){
      if(rpc->robust) rpc->active = false;
    }
    std::cout << "Reject pose graph result: refinement failed." << std::endl;
    return;
  }
  std::cout << "Pose graph loops: accepted=" << accepted_loops
            << ", rejected=" << rejected_loops << std::endl;

  // recover optimized data
  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }
  return;
}

bool GlobalBA(MapPtr _map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    int first_iterations, int second_iterations, LegOdomConfig* leg_cfg,
    const std::string& leg_velocity_path, bool structure_only){
  std::map<int, MappointPtr>& mappoints = _map->GetAllMappoints();
  std::map<int, FramePtr>& keyframes = _map->GetAllKeyframes();
  CameraPtr camera = _map->GetCameraPtr();
  if(keyframes.empty() || mappoints.empty() || !camera) return false;

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
  solver->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag; no debug.txt
  solver->setUserLambdaInit(1e-3);
  solver->setMaxTrialsAfterFailure(10);

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  int max_frame_id = 0;
  Eigen::Matrix4d Tcb = camera->BodyToCamera();
  Eigen::Matrix3d Rcb = Tcb.block<3, 3>(0, 0);
  Eigen::Vector3d tcb = Tcb.block<3, 1>(0, 3);
  Eigen::Matrix3d Rwg = _map->GetRwg();
  for(auto& kv : keyframes){
    bool fix_this_frame = (kv.first == keyframes.begin()->first);

    Eigen::Matrix4d Twc = kv.second->GetPose();
    Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
    Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);

    VIPose vi_pose = ToVIPose(Rwc, twc, Rcb, tcb);
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(fix_this_frame);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  }
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(mpt->GetPosition());
    int point_id = kv.first + max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(false);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  int max_bias_id = -1;
  int gravity_direction_id = -1;
  if(_map->IMUInit()){
    max_bias_id = max_point_id;
    for(auto& kv : keyframes){
      bool fix_this_frame = (kv.first == keyframes.begin()->first);
      FramePtr frame = kv.second;

      // 5. velocity vertex
      assert(frame->VelocityIsInitialized());
      VertexVelocity* velocity_vertex = new VertexVelocity(frame->GetVelocity());
      int velocity_id = kv.first * 3 + max_point_id;
      velocity_vertex->setId(velocity_id);
      velocity_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(velocity_vertex);

      // 6. bias vertex
      Eigen::Vector3d gyr_bias, acc_bias;
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(fix_this_frame){
        gyr_bias = preinteration->bg + preinteration->dbg;
        acc_bias = preinteration->ba + preinteration->dba;
      }else{
        gyr_bias = preinteration->bg;
        acc_bias = preinteration->ba;
      }
      VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(gyr_bias);
      int gyr_bias_id = kv.first * 3 + 1 + max_point_id;
      gyr_bias_vertex->setId(gyr_bias_id);
      gyr_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(gyr_bias_vertex);

      VertexAccBias* acc_bias_vertex = new VertexAccBias(acc_bias);
      int acc_bias_id = kv.first * 3 + 2 + max_point_id;
      max_bias_id = std::max(max_bias_id, acc_bias_id);
      acc_bias_vertex->setId(acc_bias_id);
      acc_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(acc_bias_vertex);
    }

    // 7. gravity direction vertex, will be fixed in this function.
    gravity_direction_id = max_bias_id + 1;
    VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
    gravity_direction_vertex->setId(gravity_direction_id);
    gravity_direction_vertex->setFixed(true);
    optimizer.addVertex(gravity_direction_vertex);
  }

  std::shared_ptr<LegOdomBuffer> leg_buffer;
  VertexLegExtrinsic* leg_extrinsic_vertex = nullptr;
  VertexLegTimeDelay* leg_time_vertex = nullptr;
  std::map<int, VertexLegBias*> leg_bias_vertices;
  if(leg_cfg && leg_cfg->enabled){
    leg_buffer = std::make_shared<LegOdomBuffer>();
    std::string load_error;
    if(!leg_buffer->LoadCsv(leg_velocity_path, &load_error)){
      std::cout << "Reject leg-enabled global BA: " << load_error << std::endl;
      return false;
    }
  }
  if(leg_buffer){
    int next_leg_id = std::max(max_point_id, std::max(max_bias_id, gravity_direction_id)) + 1;
    leg_extrinsic_vertex = new VertexLegExtrinsic();
    Eigen::Quaterniond q_cb(leg_cfg->extrinsic_quat_xyzw[3],
        leg_cfg->extrinsic_quat_xyzw[0], leg_cfg->extrinsic_quat_xyzw[1],
        leg_cfg->extrinsic_quat_xyzw[2]);
    q_cb.normalize();
    Eigen::Isometry3d T_cb = Eigen::Isometry3d::Identity();
    T_cb.linear() = q_cb.toRotationMatrix();
    T_cb.translation() = leg_cfg->extrinsic_xyz;
    leg_extrinsic_vertex->setEstimate(T_cb);
    leg_extrinsic_vertex->setId(next_leg_id++);
    leg_extrinsic_vertex->setFixed(!leg_cfg->optimize_extrinsic);
    optimizer.addVertex(leg_extrinsic_vertex);

    leg_time_vertex = new VertexLegTimeDelay(
        leg_cfg->time_delay_min, leg_cfg->time_delay_max);
    leg_time_vertex->setEstimate(leg_cfg->time_delay);
    leg_time_vertex->setId(next_leg_id++);
    leg_time_vertex->setFixed(!leg_cfg->optimize_time_delay);
    optimizer.addVertex(leg_time_vertex);

    for(const auto& kv : keyframes){
      auto* bias_vertex = new VertexLegBias();
      bias_vertex->setEstimate(Vector6d::Zero());
      bias_vertex->setId(next_leg_id++);
      bias_vertex->setFixed(!leg_cfg->use_bias);
      optimizer.addVertex(bias_vertex);
      leg_bias_vertices[kv.first] = bias_vertex;
    }

    if(leg_cfg->optimize_extrinsic && leg_cfg->use_extrinsic_prior){
      auto* prior = new EdgeLegExtrinsicPrior();
      prior->setVertex(0, leg_extrinsic_vertex);
      prior->setMeasurement(T_cb);
      Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
      information.diagonal().head<3>().setConstant(
          1.0 / std::pow(leg_cfg->extrinsic_rotation_prior_sigma, 2));
      information.diagonal().tail<3>().setConstant(
          1.0 / std::pow(leg_cfg->extrinsic_translation_prior_sigma, 2));
      prior->setInformation(information);
      optimizer.addEdge(prior);
    }
    if(leg_cfg->optimize_time_delay){
      auto* prior = new EdgeLegTimeDelayPrior();
      prior->setVertex(0, leg_time_vertex);
      prior->setMeasurement(0.0);
      prior->setInformation(Eigen::Matrix<double, 1, 1>::Constant(
          1.0 / std::pow(leg_cfg->time_delay_prior_sigma, 2)));
      optimizer.addEdge(prior);
    }
    if(leg_cfg->use_bias){
      for(const auto& kv : leg_bias_vertices){
        auto* prior = new EdgeLegBiasPrior();
        prior->setVertex(0, kv.second);
        prior->setMeasurement(Vector6d::Zero());
        prior->setInformation(Eigen::Matrix<double, 6, 6>::Identity() /
            std::pow(leg_cfg->bias_prior_sigma, 2));
        optimizer.addEdge(prior);
      }
    }
  }

  // edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges;
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  std::vector<EdgeIMU*> imu_edges;
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  double fx = camera->Fx();
  double fy = camera->Fy();
  double cx = camera->Cx();
  double cy = camera->Cy();
  double bf = camera->BF();

  FramePtr last_frame = std::shared_ptr<Frame>(nullptr);
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;

    int frame_vertex_id = frame_id;

    // 8. point edges
    std::vector<MappointPtr>& frame_mpts = frame->GetAllMappoints();
    for(size_t i = 0; i < frame_mpts.size(); i++){
      MappointPtr mpt = frame_mpts[i];
      if(!mpt || !mpt->IsValid()) continue;

      Eigen::Vector3d keypoint;
      if(!frame->GetKeypointPosition(i, keypoint)) continue;

      int mpt_vertex_id = max_frame_id + mpt->GetId();
      if(keypoint(2) < 0){
        EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint.head<2>());
        e->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMonoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        optimizer.addEdge(e);
        mono_edges.push_back(e);
      }else{
        EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint);
        e->setInformation(Eigen::Matrix3d::Identity());
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->bf = bf;
        optimizer.addEdge(e);
        stereo_edges.push_back(e);
      }
    }

    // 10. imu edges
    if(_map->IMUInit() && last_frame){
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(!preinteration->Valid()){  // the next IMU edge must start at this frame, not at an older one
        last_frame = frame;
        continue;
      }

      EdgeIMU* e_imu = new EdgeIMU(preinteration);

      int last_frame_vertex_id = last_frame->GetFrameId();
      g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(last_frame_vertex_id);
      g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(last_frame_vertex_id * 3 + max_point_id);
      g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(last_frame_vertex_id * 3 + 1 + max_point_id);
      g2o::HyperGraph::Vertex *va1 = optimizer.vertex(last_frame_vertex_id * 3 + 2 + max_point_id);
      g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(frame_vertex_id);
      g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(frame_vertex_id * 3 + max_point_id);
      g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(frame_vertex_id * 3 + 1 + max_point_id);
      g2o::HyperGraph::Vertex *va2 = optimizer.vertex(frame_vertex_id * 3 + 2 + max_point_id);
      g2o::HyperGraph::Vertex *vG = optimizer.vertex(gravity_direction_id);
      if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

      e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
      e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
      e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
      e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
      e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
      e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

      if(last_frame_vertex_id == keyframes.begin()->first){
        g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
        e_imu->setRobustKernel(rki);
        e_imu->setInformation(e_imu->information() * 1e-2);
        rki->setDelta(sqrt(16.92));
      }
      optimizer.addEdge(e_imu);
      imu_edges.push_back(e_imu);

      // bias edges
      EdgeGyr* e_gyr = new EdgeGyr();
      EdgeAcc* e_acc = new EdgeAcc();

      e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
      e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
      e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

      Eigen::Matrix3d info_g = preinteration->Cov.block<3,3>(9,9).inverse();
      Eigen::Matrix3d info_a = preinteration->Cov.block<3,3>(12,12).inverse();
      e_gyr->setInformation(info_g);
      e_acc->setInformation(info_a);

      optimizer.addEdge(e_gyr);
      optimizer.addEdge(e_acc);
      gyr_edges.push_back(e_gyr);
      acc_edges.push_back(e_acc);
    }

    last_frame = frame;
  }

  size_t leg_edge_count = 0;
  if(leg_buffer){
    auto first = keyframes.begin();
    auto second = first;
    if(second != keyframes.end()) ++second;
    for(; second != keyframes.end(); ++first, ++second){
      const double t0 = first->second->GetTimestamp();
      const double t1 = second->second->GetTimestamp();
      if(t1 - t0 < leg_cfg->min_keyframe_dt) continue;
      Eigen::Isometry3d initial_delta;
      Vector6d integrated_variance;
      if(!leg_buffer->Integrate(t0 + leg_cfg->time_delay,
              t1 + leg_cfg->time_delay, Vector6d::Zero(), &initial_delta,
              &integrated_variance)) continue;

      auto* edge = new EdgeLegOdom(leg_buffer, t0, t1);
      if(!leg_cfg->use_bias && !leg_cfg->optimize_time_delay){
        edge->SetFixedBodyDelta(initial_delta);
      }
      edge->setVertex(0, optimizer.vertex(first->first));
      edge->setVertex(1, optimizer.vertex(second->first));
      edge->setVertex(2, leg_extrinsic_vertex);
      edge->setVertex(3, leg_time_vertex);
      edge->setVertex(4, leg_bias_vertices[first->first]);
      edge->setMeasurement(Vector6d::Zero());
      Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
      for(int axis = 0; axis < 3; ++axis){
        information(axis, axis) = 1.0 /
            (std::pow(leg_cfg->rotation_sigma, 2) + integrated_variance[axis + 3]);
        information(axis + 3, axis + 3) = 1.0 /
            (std::pow(leg_cfg->translation_sigma, 2) + integrated_variance[axis]);
      }
      edge->setInformation(information);
      auto* robust = new g2o::RobustKernelHuber();
      robust->setDelta(leg_cfg->huber_delta);
      edge->setRobustKernel(robust);
      optimizer.addEdge(edge);
      ++leg_edge_count;

      if(leg_cfg->use_bias){
        auto* random_walk = new EdgeLegBiasRandomWalk();
        random_walk->setVertex(0, leg_bias_vertices[first->first]);
        random_walk->setVertex(1, leg_bias_vertices[second->first]);
        random_walk->setMeasurement(Vector6d::Zero());
        const double covariance = std::pow(leg_cfg->bias_random_walk_sigma, 2) *
            std::max(t1 - t0, 1e-6);
        random_walk->setInformation(Eigen::Matrix<double, 6, 6>::Identity() / covariance);
        optimizer.addEdge(random_walk);
      }
    }
    std::cout << "Leg odometry samples = " << leg_buffer->size()
              << ", factors = " << leg_edge_count << std::endl;
    if(leg_edge_count == 0){
      std::cout << "Reject leg-enabled global BA: no leg factors overlap keyframe timestamps."
                << std::endl;
      return false;
    }
  }

  // Structure only (S4 global back-end): every keyframe pose stays fixed. The points are re-solved on these poses
  // first: points moved rigidly with their first observer would otherwise look like outliers to the other observers.
  if(structure_only){
    for(auto& kv : keyframes) optimizer.vertex(kv.first)->setFixed(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
  }

  // Quarantine only grossly inconsistent observations before the first solve.
  // They remain in the map and can be reconsidered by a later refinement run.
  size_t quarantined_observations = 0;
  std::map<int, int> active_point_constraint_dimensions;
  for(EdgeSE3ProjectPoint* e : mono_edges){
    e->computeError();
    if(!e->isDepthPositive() || e->chi2() > cfg.mono_point){
      e->setLevel(1);
      quarantined_observations++;
    }else{
      active_point_constraint_dimensions[e->vertexXn<0>()->id()] += 2;
    }
  }
  for(EdgeSE3ProjectStereoPoint* e : stereo_edges){
    e->computeError();
    if(!e->isDepthPositive() || e->chi2() > cfg.stereo_point){
      e->setLevel(1);
      quarantined_observations++;
    }else{
      active_point_constraint_dimensions[e->vertexXn<0>()->id()] += 3;
    }
  }
  size_t fixed_underconstrained_points = 0;
  for(const auto& kv : mappoints){
    const int point_vertex_id = max_frame_id + kv.first;
    g2o::VertexPointXYZ* point_vertex =
        static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_vertex_id));
    if(point_vertex && active_point_constraint_dimensions[point_vertex_id] < 3){
      point_vertex->setFixed(true);
      fixed_underconstrained_points++;
    }
  }
  std::cout << "Global BA quarantined observations = "
            << quarantined_observations
            << ", fixed underconstrained points = "
            << fixed_underconstrained_points << std::endl;

  // solve
  optimizer.initializeOptimization(0);
  if(optimizer.optimize(first_iterations) <= 0){
    std::cout << "Reject global BA result: first optimization failed."
              << std::endl;
    return false;
  }

  auto optimized_trajectory_is_healthy = [&](){
    if(keyframes.size() < 2) return true;
    auto frame_it = keyframes.begin();
    auto next_frame_it = frame_it;
    next_frame_it++;
    for(; next_frame_it != keyframes.end(); frame_it++, next_frame_it++){
      const Eigen::Matrix4d& old_pose1 = frame_it->second->GetPose();
      const Eigen::Matrix4d& old_pose2 = next_frame_it->second->GetPose();
      const double old_step = (old_pose2.block<3, 1>(0, 3) -
          old_pose1.block<3, 1>(0, 3)).norm();

      VertexVIPose* vertex1 = static_cast<VertexVIPose*>(optimizer.vertex(frame_it->first));
      VertexVIPose* vertex2 = static_cast<VertexVIPose*>(optimizer.vertex(next_frame_it->first));
      if(!vertex1 || !vertex2) return false;
      const g2o::SE3Quat pose1 = vertex1->estimate().Tcw.inverse();
      const g2o::SE3Quat pose2 = vertex2->estimate().Tcw.inverse();
      const double new_step = (pose2.translation() - pose1.translation()).norm();
      const double step_limit = std::max(2.0, old_step * 3.0);
      if(!std::isfinite(new_step) || new_step > step_limit){
        std::cout << "Reject global BA result: adjacent frames "
                  << frame_it->first << " -> " << next_frame_it->first
                  << ", old_step=" << old_step
                  << ", new_step=" << new_step << std::endl;
        return false;
      }
    }
    return true;
  };

  if(point_outlier_rejection)
  {
    active_point_constraint_dimensions.clear();
    // check inlier observations
    for(size_t i=0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
        e->setLevel(1);
      }else{
        e->setLevel(0);
        active_point_constraint_dimensions[e->vertexXn<0>()->id()] += 2;
      }
      e->setRobustKernel(0);
    }

    for(size_t i=0; i < stereo_edges.size(); i++){
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
      if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
          e->setLevel(1);
      }else{
          e->setLevel(0);
          active_point_constraint_dimensions[e->vertexXn<0>()->id()] += 3;
      }
      e->setRobustKernel(0);
    }

    for(const auto& kv : mappoints){
      const int point_vertex_id = max_frame_id + kv.first;
      g2o::VertexPointXYZ* point_vertex =
          static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_vertex_id));
      if(point_vertex && active_point_constraint_dimensions[point_vertex_id] < 3){
        point_vertex->setFixed(true);
      }
    }
  }

  if(second_iterations > 0){
    // optimize again without the outliers
    optimizer.initializeOptimization(0);
    if(optimizer.optimize(second_iterations) <= 0){
      std::cout << "Reject global BA result: second optimization failed."
                << std::endl;
      return false;
    }

    if(!optimized_trajectory_is_healthy()) return false;

    // check outlier observations
    std::vector<std::pair<FramePtr, MappointPtr>> mappoint_outliers;
    for(size_t i = 0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      e->computeError();
      if(e->chi2() <= cfg.mono_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    for(size_t i = 0; i < stereo_edges.size(); i++){
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];

      e->computeError();
      if(e->chi2() <= cfg.stereo_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    std::cout << "mappoint_outliers = " << mappoint_outliers.size() << std::endl;
    const size_t observation_num = mono_edges.size() + stereo_edges.size();
    const double outlier_ratio = observation_num > 0
        ? static_cast<double>(mappoint_outliers.size()) / observation_num : 0.0;
    if(outlier_ratio > 0.10){
      std::cout << "Reject global BA result: outlier ratio " << outlier_ratio
                << " exceeds safety limit 0.10; observations are preserved."
                << std::endl;
      return false;
    }
    _map->RemoveOutliers(mappoint_outliers);
  }else if(!optimized_trajectory_is_healthy()){
    return false;
  }

  // recover optimized data
  if(leg_buffer && leg_extrinsic_vertex && leg_time_vertex){
    const Eigen::Isometry3d& T_cb = leg_extrinsic_vertex->estimate();
    const Eigen::Quaterniond q_cb(T_cb.linear());
    leg_cfg->extrinsic_xyz = T_cb.translation();
    leg_cfg->extrinsic_quat_xyzw =
        Eigen::Vector4d(q_cb.x(), q_cb.y(), q_cb.z(), q_cb.w());
    leg_cfg->time_delay = leg_time_vertex->estimate();
    std::cout << "Optimized leg calibration: xyz="
              << leg_cfg->extrinsic_xyz.transpose() << ", q_xyzw="
              << leg_cfg->extrinsic_quat_xyzw.transpose()
              << ", time_delay=" << leg_cfg->time_delay << std::endl;
  }

  // keyframes
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;

    // pose
    int frame_vertex_id = frame_id;
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(frame_vertex_id));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.block<3, 3>(0, 0) = SE3quat.rotation().toRotationMatrix();
    Twc.block<3, 1>(0, 3) = SE3quat.translation();
    frame->SetPose(Twc);

    if(_map->IMUInit()){
      // velocity
      int velocity_vertex_id = frame_vertex_id * 3 + max_point_id;
      VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(velocity_vertex_id));
      Eigen::Vector3d velocity = velocity_vertex->estimate();
      frame->SetVelocaity(velocity);

      // biases
      int gyr_vertex_id = frame_vertex_id * 3 + 1 + max_point_id;
      int acc_vertex_id = frame_vertex_id * 3 + 2 + max_point_id;
      VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(gyr_vertex_id));
      VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(acc_vertex_id));
      Eigen::Vector3d gyr_bias = gyr_bias_vertex->estimate();
      Eigen::Vector3d acc_bias = acc_bias_vertex->estimate();
      frame->SetBias(gyr_bias, acc_bias);
    }
  }

  // 3. points
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    int point_id = kv.first + max_frame_id;
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_id));
    if(point_vertex){
      Eigen::Vector3d pw = point_vertex->estimate();
      mpt->SetPosition(pw);
    }
  }

  return true;
}
