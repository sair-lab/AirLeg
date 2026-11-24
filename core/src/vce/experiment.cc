#include "vce/experiment.h"
#include "vce/statistics.h"
#include "vce/covariance.h"
#include "vce/transaction.h"
#include "vce/adjoint.h"
#include "map.h"
#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/edge_leg_odom.h"
#include "g2o_optimization/edge_imu.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <Eigen/Cholesky>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <sstream>

namespace vce {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
template<class T> T Read(const YAML::Node& n,const char* key,T fallback) { return n[key]?n[key].as<T>():fallback; }
template<class Derived> YAML::Node Array(const Eigen::MatrixBase<Derived>& a) {
  YAML::Node out(YAML::NodeType::Sequence);
  for (int i=0;i<a.rows();++i) {
    if (a.cols()==1) out.push_back(a(i,0));
    else { YAML::Node row; for (int j=0;j<a.cols();++j) row.push_back(a(i,j)); out.push_back(row); }
  }
  return out;
}
void WriteYaml(const fs::path& path,const YAML::Node& node) {
  std::ofstream stream(path); if (!stream) throw std::runtime_error("cannot write "+path.string());
  stream << std::setprecision(17) << node << '\n';
  if (!stream) throw std::runtime_error("write failed: "+path.string());
}
void Keys(const YAML::Node& n,const std::set<std::string>& allowed) {
  for (auto item:n) if (!allowed.count(item.first.as<std::string>()))
    throw std::runtime_error("unsupported VCE option: "+item.first.as<std::string>()+
      " (IMU, 6D leg bias, extrinsic/time-delay optimization and marginalization are not supported)");
}
struct Config {
  int window=8,points=128,outer=5,iterations=100;
  double damping=.5,lower=.01,upper=100,step=2,tolerance=.05;
  double rotation_sigma=.08,translation_sigma=.08,delay=0;
  double mono_chi2=5.991,stereo_chi2=7.815;
  double global_tolerance=1e-8;   // relative LM plateau of the whole-map optimizations (vce-global)
  // vce-global initial guess (InitGlobal): "none" (the visual map) or "leg_reanchor" with its gates.
  std::string global_init="none";
  double init_gate_t=.5,init_gate_deg=15;  // visual/leg conflict of a consecutive keyframe pair (m, deg)
  int init_min_shared=20;                  // ... replaced only below this many shared landmarks
  bool inlier_cache=false;        // vce-global: cache the robust visual inlier selection next to map_file
  fs::path inlier_cache_file;     // resolved from map_file when inlier_cache is set
  bool export_training=false;
  Limits limits;
  Eigen::Vector2d alpha=Eigen::Vector2d::Ones();
  Eigen::Isometry3d ext=Eigen::Isometry3d::Identity();
  // Global mode only: per-keyframe gyro bias with random walk, initialized from CSV.
  bool gyro_bias=false;
  double bias_random_walk_sigma=1e-3,bias_prior_sigma=.02;
  std::vector<std::pair<double,Eigen::Vector3d>> initial_bias;  // keyframe time, bias
};
// Linear interpolation, clamped beyond the ends; zero when no initial bias is given.
Eigen::Vector3d BiasAt(const Config& cfg,double t) {
  const auto& b=cfg.initial_bias;
  if (b.empty()) return Eigen::Vector3d::Zero();
  if (t<=b.front().first) return b.front().second;
  if (t>=b.back().first) return b.back().second;
  auto it=std::upper_bound(b.begin(),b.end(),t,[](double t,const auto& s){return t<s.first;});
  const double u=(t-(it-1)->first)/(it->first-(it-1)->first);
  return (1-u)*(it-1)->second+u*it->second;
}
std::vector<std::pair<double,Eigen::Vector3d>> LoadBias(const fs::path& path) {
  std::ifstream input(path); if (!input) throw std::runtime_error("cannot open gyro bias CSV: "+path.string());
  std::string line; std::getline(input,line);  // time,bgx,bgy,bgz
  std::vector<std::pair<double,Eigen::Vector3d>> out;
  while (std::getline(input,line)) {
    if (line.empty()) continue;
    std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
    double t; Eigen::Vector3d b;
    if (!(row>>t>>b.x()>>b.y()>>b.z()) || !b.allFinite() || (!out.empty() && t<=out.back().first))
      throw std::runtime_error("invalid gyro bias CSV row: "+path.string());
    out.emplace_back(t,b);
  }
  if (out.empty()) throw std::runtime_error("empty gyro bias CSV: "+path.string());
  return out;
}
struct Observation { int frame, point, index; Eigen::Vector3d uv; };
struct Window {
  std::vector<int> frame_ids,point_ids;
  std::vector<FramePtr> frames;
  // Optional pose snapshot (online VCE runs off the tracking thread); empty -> frame poses.
  std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
  std::vector<Eigen::Vector3d> points;
  std::vector<Observation> observations;
  std::vector<Motion> motion;
  std::vector<bool> motion_valid;  // Global mode may have uncovered boundary intervals.
};
struct Factor {
  g2o::OptimizableGraph::Edge* edge;
  int group;
  Eigen::MatrixXd base_information;
};
// A graph owns its vertices/edges. No map object is modified by this experiment.
struct Graph {
  g2o::SparseOptimizer optimizer;
  std::vector<VertexVIPose*> poses;
  std::vector<g2o::VertexPointXYZ*> points;
  std::vector<VertexGyrBias*> biases;  // Per keyframe; empty unless cfg.gyro_bias.
  std::vector<Factor> factors;
  std::vector<g2o::OptimizableGraph::Vertex*> free;
  std::map<g2o::OptimizableGraph::Vertex*,int> offsets;
  int states=0;

  Graph(const Window& w,CameraPtr camera,const Config& cfg,bool visual_only=false) {
    auto linear=g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto algorithm=new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(std::move(linear)));
    algorithm->setWriteDebug(false);  // the algorithm property (default true) overrides the linear solver flag
    algorithm->setUserLambdaInit(1e-3); algorithm->setMaxTrialsAfterFailure(10);
    optimizer.setAlgorithm(algorithm); optimizer.setVerbose(false);
    int id=0;
    auto add=[&](g2o::OptimizableGraph::Vertex* v,bool fixed) {
      v->setId(id++); v->setFixed(fixed); optimizer.addVertex(v);
      if (!fixed) { offsets[v]=states; states+=v->dimension(); free.push_back(v); }
    };
    const auto tcb=camera->BodyToCamera();
    for (size_t i=0;i<w.frames.size();++i) {
      const Eigen::Matrix4d twc=w.poses.size()==w.frames.size()?w.poses[i]:w.frames[i]->GetPose();
      auto v=new VertexVIPose;
      v->setEstimate(ToVIPose(twc.topLeftCorner<3,3>(),twc.topRightCorner<3,1>(),tcb.topLeftCorner<3,3>(),tcb.topRightCorner<3,1>()));
      add(v,i==0); poses.push_back(v);
    }
    for (const auto& position:w.points) {
      auto p=new g2o::VertexPointXYZ; p->setEstimate(position); p->setMarginalized(true);
      add(p,false); points.push_back(p);
    }
    for (const auto& obs:w.observations) {
      g2o::OptimizableGraph::Edge* edge;
      if (obs.uv.z()<0) {
        auto e=new EdgeSE3ProjectPoint;
        e->fx=camera->Fx(); e->fy=camera->Fy(); e->cx=camera->Cx(); e->cy=camera->Cy();
        e->setMeasurement(obs.uv.head<2>()); e->setInformation(Eigen::Matrix2d::Identity()); edge=e;
      } else {
        auto e=new EdgeSE3ProjectStereoPoint;
        e->fx=camera->Fx(); e->fy=camera->Fy(); e->cx=camera->Cx(); e->cy=camera->Cy(); e->bf=camera->BF();
        e->setMeasurement(obs.uv); e->setInformation(Eigen::Matrix3d::Identity()); edge=e;
      }
      edge->setVertex(0,points[obs.point]); edge->setVertex(1,poses[obs.frame]);
      if (visual_only) {
        auto robust=new g2o::RobustKernelHuber;
        robust->setDelta(std::sqrt(obs.uv.z()<0?cfg.mono_chi2:cfg.stereo_chi2)); edge->setRobustKernel(robust);
      }
      optimizer.addEdge(edge);
      factors.push_back({edge,0,Eigen::MatrixXd::Identity(edge->dimension(),edge->dimension())});
    }
    if (!visual_only) {
      auto ext=new VertexLegExtrinsic; ext->setEstimate(cfg.ext); add(ext,true);
      auto delay=new VertexLegTimeDelay; delay->setEstimate(cfg.delay); add(delay,true);
      auto bias=new VertexLegBias; bias->setEstimate(Vector6::Zero()); add(bias,true);
      if (cfg.gyro_bias) AddGyroBias(w,cfg,add);
      for (size_t i=0;i+1<poses.size();++i) {
        if (!w.motion_valid.empty() && !w.motion_valid[i]) continue;
        g2o::BaseMultiEdge<6,Vector6d>* e;
        if (cfg.gyro_bias) {
          // The CSV angular velocity already subtracts the initial bias; linearize there.
          e=new EdgeLegOdomGyrBias(cfg.ext,w.motion[i].body,w.motion[i].bias_jacobian,
                                   .5*(biases[i]->estimate()+biases[i+1]->estimate()));
          e->setVertex(2,biases[i]); e->setVertex(3,biases[i+1]);
        } else {
          auto leg=new EdgeLegOdom(nullptr,w.frames[i]->GetTimestamp(),w.frames[i+1]->GetTimestamp());
          leg->setVertex(2,ext); leg->setVertex(3,delay); leg->setVertex(4,bias);
          leg->SetFixedBodyDelta(w.motion[i].body); e=leg;
        }
        e->setVertex(0,poses[i]); e->setVertex(1,poses[i+1]); e->setMeasurement(Vector6::Zero());
        Eigen::LLT<Matrix6> chol(w.motion[i].residual_covariance);
        if (chol.info()!=Eigen::Success) { delete e; throw std::runtime_error("nonpositive leg covariance"); }
        const Matrix6 info=chol.solve(Matrix6::Identity());
        e->setInformation(info); optimizer.addEdge(e); factors.push_back({e,1,info});
      }
    }
  }
  // Known-covariance bias constraints stay outside `factors`: they are never VCE-scaled.
  template<class Add> void AddGyroBias(const Window& w,const Config& cfg,Add& add) {
    for (const auto& frame:w.frames) {
      auto v=new VertexGyrBias(BiasAt(cfg,frame->GetTimestamp())); add(v,false); biases.push_back(v);
    }
    auto prior=new EdgeGyrBiasPrior; prior->setVertex(0,biases.front());
    prior->setMeasurement(biases.front()->estimate());
    prior->setInformation(Eigen::Matrix3d::Identity()/(cfg.bias_prior_sigma*cfg.bias_prior_sigma));
    optimizer.addEdge(prior);
    for (size_t i=0;i+1<biases.size();++i) {
      const double dt=std::max(1e-3,w.frames[i+1]->GetTimestamp()-w.frames[i]->GetTimestamp());
      auto e=new EdgeGyr; e->setVertex(0,biases[i]); e->setVertex(1,biases[i+1]);
      e->setInformation(Eigen::Matrix3d::Identity()/(cfg.bias_random_walk_sigma*cfg.bias_random_walk_sigma*dt));
      optimizer.addEdge(e);
    }
  }
  void Weights(const Eigen::Vector2d& alpha) {
    for (auto& f:factors) {
      const int d=f.edge->dimension();
      Eigen::Map<Eigen::MatrixXd>(f.edge->informationData(),d,d)=f.base_information/alpha[f.group];
    }
  }
  void Push() { for (auto* v:free) v->push(); }
  void Pop() { for (auto* v:free) v->pop(); }
  void Commit() { for (auto* v:free) v->discardTop(); }
  // Numeric local blocks follow the exact g2o retraction/residual conventions.
  // Assemble all free columns; optimizer's damped/Schur Hessian is never read.
  std::vector<Group> Linearize() {
    std::vector<Group> groups(2);
    int rows[2]={0,0}; for (const auto& f:factors) rows[f.group]+=f.edge->dimension();
    for (int s=0;s<2;++s) { groups[s].jacobian=Eigen::MatrixXd::Zero(rows[s],states); groups[s].residual.resize(rows[s]); }
    int start[2]={0,0};
    for (auto& f:factors) {
      auto* edge=f.edge; const int d=edge->dimension(),s=f.group,row=start[s]; start[s]+=d;
      const Eigen::MatrixXd info=Eigen::Map<const Eigen::MatrixXd>(edge->informationData(),d,d);
      Eigen::LLT<Eigen::MatrixXd> llt(info);
      if (llt.info()!=Eigen::Success) throw std::runtime_error("invalid factor information");
      const Eigen::MatrixXd W=llt.matrixU();
      edge->computeError();
      groups[s].residual.segment(row,d)=W*Eigen::Map<const Eigen::VectorXd>(edge->errorData(),d);
      for (auto* raw:edge->vertices()) {
        auto* v=static_cast<g2o::OptimizableGraph::Vertex*>(raw);
        if (v->fixed()) continue;
        Eigen::MatrixXd j(d,v->dimension());
        for (int c=0;c<v->dimension();++c) {
          constexpr double eps=1e-6;
          Eigen::VectorXd dx=Eigen::VectorXd::Zero(v->dimension()); dx[c]=eps;
          v->push(); v->oplus(dx.data()); edge->computeError();
          const Eigen::VectorXd plus=Eigen::Map<const Eigen::VectorXd>(edge->errorData(),d); v->pop();
          dx[c]=-eps; v->push(); v->oplus(dx.data()); edge->computeError();
          j.col(c)=(plus-Eigen::Map<const Eigen::VectorXd>(edge->errorData(),d))/(2*eps); v->pop();
        }
        groups[s].jacobian.block(row,offsets.at(v),d,v->dimension())=W*j;
      }
      edge->computeError();
    }
    return groups;
  }
  // tolerance: relative cost change over 5 LM steps that counts as a plateau (whole-map runs pass
  // vce.global_tolerance; the default 1e-8 keeps LM running to max_iterations on full sequences).
  bool Optimize(int max_iterations,bool robust=false,bool window_statistics=true,double tolerance=1e-8) {
    optimizer.initializeOptimization(); optimizer.computeActiveErrors();
    double previous=optimizer.activeRobustChi2();
    if (!std::isfinite(previous)) return false;
    for (int i=0;i<max_iterations;i+=5) {
      const int steps=optimizer.optimize(std::min(5,max_iterations-i));
      optimizer.computeActiveErrors(); const double cost=optimizer.activeRobustChi2();
      if (!std::isfinite(cost) || cost>previous+1e-6*std::max(1.0,previous)) return false;
      const bool plateau=std::abs(previous-cost)<tolerance*std::max(1.0,previous);
      if (!window_statistics && (i%25==0 || plateau || i+5>=max_iterations))
        std::cout<<"Global step "<<i+5<<": cost="<<std::setprecision(12)<<cost
                 <<", relative_change="<<std::abs(previous-cost)/std::max(1.0,previous)<<std::endl;
      previous=cost;
      if (robust && (plateau || i+5>=max_iterations)) return true;
      if (!window_statistics) {
        // Whole maps use sparse g2o BA only; NEVER allocate the dense window Jacobian.
        if (plateau) return true;
        if (steps<=0) return false;
        continue;
      }
      if (plateau || steps==0 || i+5>=max_iterations) {
        const auto groups=Linearize();
        Eigen::VectorXd gradient=Eigen::VectorXd::Zero(states),norm=gradient;
        double q=0;
        for (const auto& g:groups) { gradient+=g.jacobian.transpose()*g.residual; norm+=g.jacobian.colwise().squaredNorm().transpose(); q+=g.residual.squaredNorm(); }
        const double scaled=(gradient.array().abs()/(norm.array().sqrt()+1e-30)).maxCoeff()/std::max(1.0,std::sqrt(q));
        if (scaled<1e-5) return true;
        if (steps<=0) return false;
      }
    }
    return false;
  }
  bool PositiveDepth() const {
    for (const auto& f:factors) if (f.group==0) {
      if (f.edge->dimension()==2) { if (!static_cast<EdgeSE3ProjectPoint*>(f.edge)->isDepthPositive()) return false; }
      else if (!static_cast<EdgeSE3ProjectStereoPoint*>(f.edge)->isDepthPositive()) return false;
    }
    return true;
  }
};

Window Select(const std::vector<std::pair<int,FramePtr>>& frames,size_t begin,const Config& cfg) {
  Window w;
  std::map<int,std::vector<Observation>> obs;
  std::map<int,Eigen::Vector3d> positions;
  for (int f=0;f<cfg.window;++f) {
    const auto& item=frames.at(begin+f); w.frame_ids.push_back(item.first); w.frames.push_back(item.second);
    const auto& points=item.second->GetAllMappoints();
    const Eigen::Matrix4d tcw=item.second->GetPose().inverse();
    std::set<int> seen;
    for (size_t i=0;i<points.size();++i) {
      const auto& p=points[i];
      if (!p || !p->IsValid() || !p->GetPosition().allFinite() || !seen.insert(p->GetId()).second) continue;
      const Eigen::Vector3d pc=tcw.topLeftCorner<3,3>()*p->GetPosition()+tcw.topRightCorner<3,1>();
      Eigen::Vector3d uv;
      if (pc.z()<=0 || !item.second->GetKeypointPosition(i,uv) || !uv.allFinite()) continue;
      obs[p->GetId()].push_back({f,0,static_cast<int>(i),uv}); positions[p->GetId()]=p->GetPosition();
    }
  }
  std::vector<int> ids;
  for (const auto& item:obs) if (item.second.size()>=2 && std::any_of(item.second.begin(),item.second.end(),[](const Observation& o){return o.uv.z()>=0;})) ids.push_back(item.first);
  std::sort(ids.begin(),ids.end(),[&](int a,int b){return obs[a].size()!=obs[b].size()?obs[a].size()>obs[b].size():a<b;});
  if (ids.size()>static_cast<size_t>(cfg.points)) ids.resize(cfg.points);
  for (int id:ids) {
    for (auto o:obs[id]) { o.point=w.points.size(); w.observations.push_back(o); }
    w.point_ids.push_back(id); w.points.push_back(positions.at(id));
  }
  return w;
}
// Robust visual-only fit: inlier flag per observation of w (positive depth, chi2 below the gate).
std::vector<char> RobustInliers(const Window& w,CameraPtr camera,const Config& cfg,double tolerance) {
  if (w.points.empty()) throw std::runtime_error("no eligible landmarks");
  Graph visual(w,camera,cfg,true);
  if (!visual.Optimize(cfg.iterations,true,true,tolerance)) throw std::runtime_error("robust visual fit failed");
  std::vector<char> inlier(w.observations.size(),0);
  for (size_t i=0;i<w.observations.size();++i) {
    const auto& o=w.observations[i]; auto* e=visual.factors[i].edge; e->computeError();
    const bool positive=e->dimension()==2?static_cast<EdgeSE3ProjectPoint*>(e)->isDepthPositive():static_cast<EdgeSE3ProjectStereoPoint*>(e)->isDepthPositive();
    inlier[i]=positive && e->chi2()<=(o.uv.z()<0?cfg.mono_chi2:cfg.stereo_chi2);
  }
  return inlier;
}
// Keeps the flagged observations of landmarks with >= 2 of them, at least one stereo.
void KeepInliers(Window& w,const std::vector<char>& inlier) {
  std::vector<std::vector<Observation>> kept(w.points.size());
  for (size_t i=0;i<w.observations.size();++i) if (inlier[i]) kept[w.observations[i].point].push_back(w.observations[i]);
  Window selected; selected.frames=w.frames; selected.frame_ids=w.frame_ids; selected.poses=w.poses;
  for (size_t i=0;i<kept.size();++i) {
    if (kept[i].size()<2 || !std::any_of(kept[i].begin(),kept[i].end(),[](const Observation& o){return o.uv.z()>=0;})) continue;
    for (auto o:kept[i]) { o.point=selected.points.size(); selected.observations.push_back(o); }
    selected.points.push_back(w.points[i]); selected.point_ids.push_back(w.point_ids[i]);
  }
  if (selected.points.empty()) throw std::runtime_error("no landmarks after visual inlier selection");
  w=std::move(selected); // Both methods restart from the SAME ORIGINAL map state.
}
void FreezeInliers(Window& w,CameraPtr camera,const Config& cfg,double tolerance=1e-8) {
  KeepInliers(w,RobustInliers(w,camera,cfg,tolerance));
}
// Whole-map selection, optionally cached in `cache`: it depends only on the fixed visual map and the gate settings,
// so every S2 round with the same map_file reuses it. The key covers the settings and the exact observation list.
// Returns "hit", "miss" or "off".
std::string FreezeInliersCached(Window& w,CameraPtr camera,const Config& cfg,const fs::path& cache) {
  if (cache.empty()) { FreezeInliers(w,camera,cfg,cfg.global_tolerance); return "off"; }
  std::uint64_t hash=1469598103934665603ull;
  auto mix=[&](std::uint64_t v){ hash^=v; hash*=1099511628211ull; };
  for (const auto& o:w.observations) {
    mix(static_cast<std::uint64_t>(w.frame_ids[o.frame])); mix(static_cast<std::uint64_t>(w.point_ids[o.point]));
    mix(static_cast<std::uint64_t>(o.index)); mix(o.uv.z()<0);
  }
  std::ostringstream key; key<<std::setprecision(17)<<"vce_inliers v1 observations="<<w.observations.size()
    <<" points="<<w.points.size()<<" hash="<<hash<<" mono_chi2="<<cfg.mono_chi2<<" stereo_chi2="<<cfg.stereo_chi2
    <<" iterations="<<cfg.iterations<<" tolerance="<<cfg.global_tolerance;
  if (fs::exists(cache)) {
    std::ifstream in(cache); std::string header,flags;
    if (std::getline(in,header) && header==key.str() && std::getline(in,flags) && flags.size()==w.observations.size()) {
      std::vector<char> inlier(flags.size());
      for (size_t i=0;i<flags.size();++i) inlier[i]=flags[i]=='1';
      KeepInliers(w,inlier); return "hit";
    }
  }
  const auto inlier=RobustInliers(w,camera,cfg,cfg.global_tolerance);
  const fs::path temporary=cache.string()+".tmp";
  { std::ofstream out(temporary); out<<key.str()<<'\n'; for (char f:inlier) out<<(f?'1':'0'); out<<'\n';
    if (!out) throw std::runtime_error("cannot write inlier cache: "+temporary.string()); }
  fs::rename(temporary,cache);
  KeepInliers(w,inlier); return "miss";
}
void SaveState(const fs::path& dir,const std::string& prefix,const Window& w,const Graph& graph) {
  std::ofstream poses(dir/(prefix+"_poses.tum")),points(dir/(prefix+"_points.csv"));
  poses<<std::setprecision(17); points<<std::setprecision(17)<<"point_id,x,y,z\n";
  for (size_t i=0;i<graph.poses.size();++i) {
    const auto& pose=graph.poses[i]->estimate();
    const Eigen::Matrix3d rwc=pose.Rcw.transpose(); const Eigen::Vector3d t=-rwc*pose.tcw;
    const Eigen::Quaterniond q(rwc);
    poses<<w.frames[i]->GetTimestamp()<<' '<<t.transpose()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';
  }
  for (size_t i=0;i<graph.points.size();++i) { const auto& p=graph.points[i]->estimate(); points<<w.point_ids[i]<<','<<p.x()<<','<<p.y()<<','<<p.z()<<'\n'; }
  if (!poses || !points) throw std::runtime_error("cannot save window state");
}
// Export only observations participating in the accepted Gaussian graph.
void ExportTraining(MapPtr map,const Window& w,const Graph& graph,const Config& cfg,
                    const fs::path& output) {
  std::set<std::pair<int,int>> observations;
  for (const auto& o:w.observations) observations.emplace(w.frame_ids[o.frame],o.index);
  for (size_t i=0;i<w.frames.size();++i) {
    const auto& pose=graph.poses[i]->estimate();
    Eigen::Matrix4d twc=Eigen::Matrix4d::Identity();
    twc.block<3,3>(0,0)=pose.Rcw.transpose();
    twc.block<3,1>(0,3)=-pose.Rcw.transpose()*pose.tcw;
    w.frames[i]->SetPose(twc);
    for (size_t j=0;j<w.frames[i]->FeatureNum();++j)
      if (!observations.count({w.frame_ids[i],int(j)})) {
        w.frames[i]->RemoveMappoint(int(j));
        w.frames[i]->SetTrackId(j,-1);
      }
  }
  std::set<int> retained(w.point_ids.begin(),w.point_ids.end());
  auto& points=map->GetAllMappoints();
  for (auto it=points.begin();it!=points.end();)
    if (!retained.count(it->first)) it=points.erase(it); else ++it;
  for (size_t i=0;i<w.point_ids.size();++i) {
    auto point=points.at(w.point_ids[i]);
    point->SetPosition(graph.points[i]->estimate());
    auto& observers=point->GetAllObversers();
    for (auto it=observers.begin();it!=observers.end();)
      if (!observations.count({it->first,it->second})) it=observers.erase(it); else ++it;
  }
  std::ofstream stream(output/"AirSLAM_mapv1.bin",std::ios::binary);
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  { boost::archive::binary_oarchive archive(stream); archive<<map; }
  map->SaveKeyframeTrajectory((output/"trajectory_v1.txt").string());
  map->ExportTrainingData(output.string(),false);
  YAML::Node calibration;
  Eigen::Quaterniond q(cfg.ext.linear());
  calibration["extrinsic"]=std::vector<double>{cfg.ext.translation().x(),cfg.ext.translation().y(),
    cfg.ext.translation().z(),q.x(),q.y(),q.z(),q.w()};
  calibration["time_delay"]=cfg.delay;
  WriteYaml(output/"calibration.yaml",calibration);
  if (graph.biases.empty()) return;
  std::ofstream bias(output/"leg_odom"/"gyro_bias.csv"); bias<<std::setprecision(17)<<"time,bgx,bgy,bgz\n";
  for (size_t i=0;i<w.frames.size();++i) {
    const Eigen::Vector3d b=graph.biases[i]->estimate();
    bias<<w.frames[i]->GetTimestamp()<<','<<b.x()<<','<<b.y()<<','<<b.z()<<'\n';
  }
  if (!bias) throw std::runtime_error("cannot write gyro bias CSV");
}
// Window-centre scale profile of the global solve (log interpolation in ScaleAt).
struct Scale { double time; Eigen::Vector2d alpha; std::string status; };
std::vector<Scale> ScaleProfile(const std::vector<std::pair<int,FramePtr>>& frames,const Config& cfg,const fs::path& report_path) {
  const auto local=YAML::LoadFile(report_path.string());
  std::vector<Scale> scales;
  for (auto w:local["windows"]) {
    Eigen::Vector2d alpha=Eigen::Vector2d::Ones();
    const std::string status=w["status"].as<std::string>();
    if (status=="converged") {
      auto values=w["alpha"].as<std::vector<double>>();
      if (values.size()!=2 || !std::isfinite(values[0]) || !std::isfinite(values[1]) || values[0]<=0 || values[1]<=0)
        throw std::runtime_error("invalid converged window scale");
      alpha<<values[0],values[1];
    }
    scales.push_back({.5*(w["begin"].as<double>()+w["end"].as<double>()),alpha,status});
  }
  if (scales.empty() || frames.size()<2) throw std::runtime_error("empty global scale profile/map");
  std::sort(scales.begin(),scales.end(),[](const Scale& a,const Scale& b){return a.time<b.time;});
  for(size_t i=1;i<scales.size();++i) if(scales[i].time<=scales[i-1].time) throw std::runtime_error("unordered scale centers");
  return scales;
}
Eigen::Vector2d ScaleAt(const std::vector<Scale>& scales,double t) {
  if(t<=scales.front().time) return scales.front().alpha;
  if(t>=scales.back().time) return scales.back().alpha;
  auto it=std::upper_bound(scales.begin(),scales.end(),t,[](double t,const Scale& s){return t<s.time;});
  const auto& b=*it; const auto& a=*(it-1); const double u=(t-a.time)/(b.time-a.time);
  return ((1-u)*a.alpha.array().log()+u*b.alpha.array().log()).exp().matrix();
}
// Leg intervals of the global window (uncovered boundary intervals get no factor); returns the covered count.
size_t GlobalMotion(Window& w,const LegOdomBuffer& buffer,const Config& cfg) {
  size_t leg_count=0;
  for(size_t i=0;i+1<w.frames.size();++i) {
    const double t0=w.frames[i]->GetTimestamp()+cfg.delay,t1=w.frames[i+1]->GetTimestamp()+cfg.delay;
    const bool covered=t0>=buffer.start_time() && t1<=buffer.end_time();
    w.motion_valid.push_back(covered);
    w.motion.push_back(covered?Integrate(buffer,t0,t1,cfg.ext,cfg.rotation_sigma,cfg.translation_sigma):Motion{});
    if(covered)++leg_count;
  }
  return leg_count;
}
// Factor information of the global solve: base / alpha, alpha = 1 (fixed) or the scale profile (vce).
void GlobalWeights(Graph& graph,const Window& w,const std::string& mode,const std::vector<Scale>& scales) {
  for(auto& factor:graph.factors) {
    double time;
    if(factor.group==0) {
      auto* pose=static_cast<VertexVIPose*>(factor.edge->vertex(1));
      time=w.frames.at(pose->id())->GetTimestamp();
    }else{
      auto* a=static_cast<VertexVIPose*>(factor.edge->vertex(0));auto* b=static_cast<VertexVIPose*>(factor.edge->vertex(1));
      time=.5*(w.frames.at(a->id())->GetTimestamp()+w.frames.at(b->id())->GetTimestamp());
    }
    const double alpha=mode=="fixed"?1.:ScaleAt(scales,time)[factor.group];
    const int d=factor.edge->dimension();
    Eigen::Map<Eigen::MatrixXd>(factor.edge->informationData(),d,d)=factor.base_information/alpha;
  }
}
// Initial guess of the global fused solve (Config::global_init). A loop-closed visual map can contain wrong relinks
// after tracking losses: consecutive keyframes metres / tens of degrees apart that share no landmarks. The sparse BA
// corrects such a segment only slowly (many LM steps along a weakly conditioned direction) and can stop in
// a worse minimum. leg_reanchor chains the visual relative poses and replaces a consecutive pair by the leg odometry
// where the two disagree by more than init_gate_t / init_gate_deg AND the pair shares fewer than init_min_shared
// landmarks (a visually constrained pair is kept: there the leg odometry is the likelier outlier). The later keyframes
// and the landmarks (mean of the corrections of their observing keyframes) move with it. No replaced pair: no change.
void InitGlobal(Window& w,const Config& cfg,YAML::Node& report) {
  if (cfg.global_init=="none") return;
  const auto tick=Clock::now();
  const size_t n=w.frames.size();
  std::vector<Eigen::Isometry3d> T(n),Tn(n);
  for (size_t i=0;i<n;++i) T[i].matrix()=w.poses.size()==n?w.poses[i]:w.frames[i]->GetPose();
  auto angle_deg=[](const Eigen::Matrix3d& R){ return Eigen::AngleAxisd(R).angle()*180/M_PI; };
  std::vector<int> shared(n,0);   // landmarks observed by keyframes i and i+1
  { std::vector<std::vector<int>> observers(w.points.size());
    for (const auto& o:w.observations) observers[o.point].push_back(o.frame);
    for (auto& f:observers) {
      std::sort(f.begin(),f.end()); f.erase(std::unique(f.begin(),f.end()),f.end());
      for (size_t a=0;a+1<f.size();++a) if (f[a+1]==f[a]+1) ++shared[f[a]];
    } }
  YAML::Node r,pairs(YAML::NodeType::Sequence); r["method"]=cfg.global_init;
  Tn[0]=T[0]; int replaced=0;
  for (size_t i=0;i+1<n;++i) {
    Eigen::Isometry3d step=T[i].inverse()*T[i+1];
    if (i<w.motion_valid.size() && w.motion_valid[i]) {
      const Eigen::Isometry3d leg=cfg.ext*w.motion[i].body*cfg.ext.inverse(),diff=step.inverse()*leg;
      const double dt=diff.translation().norm(),dr=angle_deg(diff.linear());
      if (dt>cfg.init_gate_t || dr>cfg.init_gate_deg) {
        const bool weak=shared[i]<cfg.init_min_shared;
        YAML::Node pr; pr["frame"]=w.frame_ids[i]; pr["translation"]=dt; pr["rotation_deg"]=dr; pr["shared"]=shared[i];
        pr["replaced"]=weak; pairs.push_back(pr);
        if (weak) { step=leg; ++replaced; }
      }
    }
    Tn[i+1]=Tn[i]*step;
  }
  r["conflicts"]=pairs; r["replaced_pairs"]=replaced;
  if (replaced) {
    std::vector<Eigen::Isometry3d> C(n);
    double max_t=0;
    for (size_t i=0;i<n;++i) { C[i]=Tn[i]*T[i].inverse(); max_t=std::max(max_t,(Tn[i].translation()-T[i].translation()).norm()); }
    std::vector<Eigen::Vector3d> sum(w.points.size(),Eigen::Vector3d::Zero()); std::vector<int> count(w.points.size(),0);
    for (const auto& o:w.observations) { sum[o.point]+=C[o.frame]*w.points[o.point]; ++count[o.point]; }
    for (size_t p=0;p<w.points.size();++p) if (count[p]) w.points[p]=sum[p]/count[p];
    w.poses.assign(n,Eigen::Matrix4d::Identity());
    for (size_t i=0;i<n;++i) w.poses[i]=Tn[i].matrix();
    r["max_keyframe_shift"]=max_t;
  }
  report["global_init"]=r;
  std::cout<<"Global: init "<<cfg.global_init<<", "<<replaced<<" of "<<pairs.size()<<" conflicting consecutive pairs replaced ("
           <<Seconds(tick)<<" s)"<<std::endl;
}
// The adjoint view of a global graph (vertex ids = frame indices for poses).
AdjointGraph MakeAdjointGraph(Graph& graph,const Window& w,const Config& cfg) {
  if (cfg.gyro_bias) throw std::runtime_error("implicit gradient: only EdgeLegOdom (gyro_bias disabled) is supported");
  AdjointGraph ag; ag.poses=graph.poses; ag.points=graph.points;
  for (auto& f:graph.factors) {
    if (f.group==0) { ag.visual.push_back(f.edge); continue; }
    auto* edge=dynamic_cast<EdgeLegOdom*>(f.edge);
    if (!edge) throw std::runtime_error("implicit gradient: unexpected leg factor type");
    const int i=static_cast<VertexVIPose*>(edge->vertex(0))->id();
    ag.legs.push_back({edge,w.motion.at(i).body,cfg.ext});
  }
  return ag;
}
// Transfer local group scales to global factors; this is not a global VCE fit.
int RunGlobal(const std::vector<std::pair<int,FramePtr>>& frames,MapPtr map,CameraPtr camera,
              const LegOdomBuffer& buffer,Config cfg,const fs::path& report_path,const fs::path& output) {
  const std::vector<Scale> scales=ScaleProfile(frames,cfg,report_path);
  auto scale_at=[&](double t)->Eigen::Vector2d { return ScaleAt(scales,t); };
  std::ofstream profile(output/"scale_profile.csv");profile<<std::setprecision(17)<<"time,alpha_visual,alpha_leg,status\n";
  for(const auto& s:scales) profile<<s.time<<','<<s.alpha[0]<<','<<s.alpha[1]<<','<<s.status<<'\n';
  cfg.window=frames.size();cfg.points=std::max<size_t>(1,map->GetAllMappoints().size());
  std::cout<<"Global: selecting all eligible map points, frames="<<frames.size()<<std::endl;
  Window w=Select(frames,0,cfg);
  const auto started=Clock::now();
  std::cout<<"Global: visual-only robust inlier selection, points="<<w.points.size()<<std::endl;
  const std::string cache_status=FreezeInliersCached(w,camera,cfg,cfg.inlier_cache_file);
  std::cout<<"Global: inlier selection "<<cache_status<<", points="<<w.points.size()<<std::endl;
  YAML::Node result;
  result["scope"]="full sequence sparse BA; one fixed first pose, all eligible landmarks free; local VCE scales transferred, not re-estimated globally";
  result["scale_mapping"]="log interpolation at window centers; failed/nonconverged windows alpha=1; nearest endpoint beyond coverage; visual at frame time, leg at interval midpoint";
  result["frames"]=w.frames.size();result["points"]=w.points.size();result["visual_observations"]=w.observations.size();
  result["preselection_seconds"]=Seconds(started);
  result["inlier_cache"]=cache_status;
  result["global_tolerance"]=cfg.global_tolerance;
  std::ofstream observations(output/"observations.csv");observations<<std::setprecision(17)<<"frame_id,point_id,keypoint_index,u,v,u_right,alpha_visual\n";
  for(const auto& o:w.observations) observations<<w.frame_ids[o.frame]<<','<<w.point_ids[o.point]<<','<<o.index<<','<<o.uv.x()<<','<<o.uv.y()<<','<<o.uv.z()<<','<<scale_at(w.frames[o.frame]->GetTimestamp())[0]<<'\n';
  const size_t leg_count=GlobalMotion(w,buffer,cfg);
  if(!leg_count) throw std::runtime_error("global map has no covered leg intervals");
  result["leg_factors"]=leg_count;result["uncovered_leg_intervals"]=w.frames.size()-1-leg_count;
  std::ofstream legs(output/"leg_covariances.csv");legs<<std::setprecision(17)<<"frame_i,frame_j,alpha_leg";
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)legs<<",cov_"<<i<<j;legs<<'\n';
  for(size_t i=0;i<w.motion.size();++i)if(w.motion_valid[i]) {
    legs<<w.frame_ids[i]<<','<<w.frame_ids[i+1]<<','<<scale_at(.5*(w.frames[i]->GetTimestamp()+w.frames[i+1]->GetTimestamp()))[1];
    for(int a=0;a<6;++a)for(int b=0;b<6;++b)legs<<','<<w.motion[i].residual_covariance(a,b);legs<<'\n';
  }
  InitGlobal(w,cfg,result);
  std::unique_ptr<Graph> accepted;
  for(const std::string mode:{"fixed","vce"}) {
    std::cout<<"Global: "<<mode<<" BA begins"<<std::endl;
    auto candidate=std::make_unique<Graph>(w,camera,cfg);
    auto& graph=*candidate;
    if(mode=="fixed")SaveState(output,"visual",w,graph); // Original loop-closed visual BA.
    GlobalWeights(graph,w,mode,scales);
    const auto tick=Clock::now();
    bool converged=false;
    try { converged=graph.Optimize(cfg.iterations,false,false,cfg.global_tolerance); }
    catch (const std::exception& error) {
      if (!cfg.export_training || !accepted) throw;
      result["fallback_reason"]=std::string("VCE solver failed: ")+error.what();
      break;
    }
    const bool positive=graph.PositiveDepth();
    graph.optimizer.computeActiveErrors();
    result[mode]["converged_cost"]=converged;result[mode]["positive_depth"]=positive;
    result[mode]["cost"]=graph.optimizer.activeChi2();result[mode]["seconds"]=Seconds(tick);
    SaveState(output,mode,w,graph);
    std::cout<<"Global: "<<mode<<" complete, converged="<<converged<<", positive_depth="<<positive<<std::endl;
    WriteYaml(output/"global_report.yaml",result);
    if(!positive || !converged) {
      if (cfg.export_training && accepted) {
        result["fallback_reason"]="VCE solve not converged/positive; using fixed weights";
        break;
      }
      throw std::runtime_error("global "+mode+" solve not converged/positive; provisional output only");
    }
    if (!graph.biases.empty()) {
      Eigen::Vector3d mean=Eigen::Vector3d::Zero(); double step=0;
      for (size_t i=0;i<graph.biases.size();++i) {
        mean+=graph.biases[i]->estimate()/graph.biases.size();
        if (i) step=std::max(step,(graph.biases[i]->estimate()-graph.biases[i-1]->estimate()).norm());
      }
      result[mode]["gyro_bias_mean"]=Array(mean); result[mode]["gyro_bias_max_step"]=step;
    }
    accepted=std::move(candidate);
    result["selected"]=mode;
  }
  if (cfg.export_training) {
    ExportTraining(map,w,*accepted,cfg,output);
    result["training_export_complete"]=true;
  }
  WriteYaml(output/"global_report.yaml",result);
  return 0;
}

// Implicit hypergradient (S2 implicit_gradient) through the final global solve of a fusion run.
// Rebuilds exactly the vce-global graph (same Select / FreezeInliers / Graph / weights) at the saved solution of the
// selected mode, checks stationarity (Gauss-Newton refinement if needed), maps the upper gradient dU/dX^k (per
// observation, camera-frame point of the observing keyframe) to dU/dmu, solves H w = dU/dmu and writes
// g_e = -J_z^T Omega J_mu w_e per leg edge (vce/adjoint.h).
int RunAdjoint(const std::vector<std::pair<int,FramePtr>>& frames,MapPtr map,CameraPtr camera,
               const LegOdomBuffer& buffer,Config cfg,const fs::path& report_path,const YAML::Node& a,
               const fs::path& base,const fs::path& output) {
  const auto started=Clock::now();
  auto rel=[&](const char* key){ fs::path p=a[key].as<std::string>(); return fs::weakly_canonical(p.is_absolute()?p:base/p); };
  const fs::path fusion=rel("fusion_dir");
  const YAML::Node global=YAML::LoadFile((fusion/"global_report.yaml").string());
  const std::string selected=global["selected"].as<std::string>();
  const double tol=Read(a,"stationarity_tol",1e-6);
  const int max_refine=Read(a,"max_reconverge_iterations",20);
  YAML::Node result; result["fusion_dir"]=fusion.string(); result["selected"]=selected;
  if (global["fallback_reason"]) result["fusion_fallback_reason"]=global["fallback_reason"];
  const auto scales=ScaleProfile(frames,cfg,report_path);
  cfg.window=frames.size(); cfg.points=std::max<size_t>(1,map->GetAllMappoints().size());
  Window w=Select(frames,0,cfg);
  // Same inlier selection as RunGlobal: its cache (written by the fusion run) and global_tolerance. With the plain
  // FreezeInliers(w,camera,cfg) (default tolerance) the robust fit would stop elsewhere, the inlier set would differ
  // from the fusion run and the implicit gradient would be skipped.
  result["inlier_cache"]=FreezeInliersCached(w,camera,cfg,cfg.inlier_cache_file);
  if (global["frames"].as<size_t>()!=w.frames.size() || global["points"].as<size_t>()!=w.points.size() ||
      global["visual_observations"].as<size_t>()!=w.observations.size())
    throw std::runtime_error("adjoint: rebuilt inliers differ from the fusion run");
  GlobalMotion(w,buffer,cfg);
  {  // the selected solution as the linearization point
    std::ifstream poses(fusion/(selected+"_poses.tum"));
    if (!poses) throw std::runtime_error("adjoint: missing "+selected+"_poses.tum");
    w.poses.clear();
    double t; Eigen::Vector3d p; Eigen::Quaterniond q;
    for (size_t i=0;i<w.frames.size();++i) {
      if (!(poses>>t>>p.x()>>p.y()>>p.z()>>q.x()>>q.y()>>q.z()>>q.w())) throw std::runtime_error("adjoint: short pose file");
      if (std::abs(t-w.frames[i]->GetTimestamp())>1e-6) throw std::runtime_error("adjoint: pose timestamps differ");
      Eigen::Matrix4d T=Eigen::Matrix4d::Identity(); T.topLeftCorner<3,3>()=q.normalized().toRotationMatrix(); T.topRightCorner<3,1>()=p;
      w.poses.push_back(T);
    }
    std::ifstream points(fusion/(selected+"_points.csv")); std::string line; std::getline(points,line);
    for (size_t i=0;i<w.points.size();++i) {
      if (!std::getline(points,line)) throw std::runtime_error("adjoint: short point file");
      std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line); long id; Eigen::Vector3d X;
      if (!(row>>id>>X.x()>>X.y()>>X.z()) || id!=w.point_ids[i]) throw std::runtime_error("adjoint: point ids differ");
      w.points[i]=X;
    }
  }
  Graph graph(w,camera,cfg);
  GlobalWeights(graph,w,selected,scales);
  graph.optimizer.initializeOptimization(); graph.optimizer.computeActiveErrors();
  result["cost_rebuilt"]=graph.optimizer.activeChi2();
  if (global[selected]["cost"]) result["cost_fusion"]=global[selected]["cost"].as<double>();
  const AdjointGraph ag=MakeAdjointGraph(graph,w,cfg);
  const bool exact=Read(a,"exact",true);   // exact second derivatives (Gauss-Newton is inexact with inconsistent data)
  result["exact_second_derivatives"]=exact;
  AdjointSystem system(ag,exact);
  system.Linearize();
  Stationarity st=system.Check();
  result["stationarity_before"]["scaled_max"]=st.scaled_max; result["stationarity_before"]["gradient_norm"]=st.gradient_norm;
  // Re-converge only near a stationary point: far from it (> 100 x tol) the backtracking Gauss-Newton steps take very
  // long and still do not converge; such a sequence is skipped.
  if (st.scaled_max>=tol && st.scaled_max<100*tol) {  // re-converge with Gauss-Newton steps (PSD Hessian), then linearize the exact system again
    AdjointSystem gn(ag,false);
    gn.Refine(max_refine,tol);
    system.Linearize(); st=system.Check();
  }
  result["stationarity"]["scaled_max"]=st.scaled_max; result["stationarity"]["gradient_norm"]=st.gradient_norm;
  result["stationarity"]["cost"]=st.cost; result["stationarity"]["tolerance"]=tol;
  result["converged"]=st.scaled_max<tol;
  if (st.scaled_max>=tol) {
    WriteYaml(output/"adjoint_report.yaml",result);
    std::cerr<<"adjoint: BA not stationary (scaled gradient "<<st.scaled_max<<"); no hypergradient written"<<std::endl;
    return 3;
  }
  // observation lookup: (frame, u, v) rounded to 1e-3 px
  std::map<std::tuple<int,long long,long long>,int> lookup;
  for (size_t k=0;k<w.observations.size();++k) {
    const auto& o=w.observations[k];
    lookup[{o.frame,std::llround(o.uv.x()*1e3),std::llround(o.uv.y()*1e3)}]=static_cast<int>(k);
  }
  auto frame_of=[&](double t)->int {
    auto it=std::lower_bound(w.frames.begin(),w.frames.end(),t-1e-6,[](const FramePtr& f,double x){return f->GetTimestamp()<x;});
    if (it==w.frames.end() || std::abs((*it)->GetTimestamp()-t)>1e-6) return -1;
    return static_cast<int>(it-w.frames.begin());
  };
  AdjointVector b=system.Zero();
  size_t rows=0,matched=0;
  {
    std::ifstream input(rel("upper_grad_csv")); if (!input) throw std::runtime_error("adjoint: missing upper_grad_csv");
    std::string line; std::getline(input,line);  // kf_time,uL,vL,gx,gy,gz
    while (std::getline(input,line)) {
      if (line.empty() || line[0]=='#') continue;
      std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
      double t,u,v; Eigen::Vector3d g;
      if (!(row>>t>>u>>v>>g.x()>>g.y()>>g.z())) throw std::runtime_error("adjoint: bad upper gradient row");
      ++rows;
      const int f=frame_of(t); if (f<0) continue;
      auto it=lookup.find({f,std::llround(u*1e3),std::llround(v*1e3)}); if (it==lookup.end()) continue;
      ++matched;
      const auto& o=w.observations[it->second];
      auto* pose=graph.poses[o.frame];
      const Eigen::Vector3d X=graph.points[o.point]->estimate();
      const int pi=system.PoseIndex(pose);
      if (pi>=0) b.pose[pi]+=CameraPointPoseJacobian(pose,X).transpose()*g;
      b.point[o.point]+=pose->estimate().Rcw.transpose()*g;
    }
  }
  result["upper_rows"]=rows; result["upper_matched"]=matched; result["b_norm"]=b.Norm();
  if (rows && matched<rows) std::cerr<<"adjoint: "<<rows-matched<<" of "<<rows<<" upper-gradient rows unmatched"<<std::endl;
  const auto solve_started=Clock::now();
  const AdjointVector wv=system.Solve(b);
  result["solve_seconds"]=Seconds(solve_started); result["w_norm"]=wv.Norm();
  result["clipped_point_blocks"]=system.ClippedPoints();
  const std::vector<Vec6> g=system.LegHypergradients(wv);
  double g2=0; for (const auto& x:g) g2+=x.squaredNorm(); result["g_norm"]=std::sqrt(g2);
  std::ofstream out(output/"implicit_grad.csv"), legs(output/"leg_edges.csv"), frame_table(output/"frames.csv");
  out<<std::setprecision(17)<<"# g = dU/d delta, delta = [phi; rho]: M <- [R_M Exp(phi), t_M + R_M rho] (camera relative motion T_Ci_Cj)\n"
     <<"edge_id,frame_i,frame_j,t_i,t_j,g_rx,g_ry,g_rz,g_tx,g_ty,g_tz\n";
  legs<<std::setprecision(17)<<"edge_id,frame_i,frame_j,t_i,t_j,alpha_leg,tx,ty,tz,qx,qy,qz,qw\n";
  for (size_t k=0;k<ag.legs.size();++k) {
    const int i=static_cast<VertexVIPose*>(ag.legs[k].edge->vertex(0))->id(), j=i+1;
    const double ti=w.frames[i]->GetTimestamp(), tj=w.frames[j]->GetTimestamp();
    out<<i<<','<<w.frame_ids[i]<<','<<w.frame_ids[j]<<','<<ti<<','<<tj;
    for (int c=0;c<6;++c) out<<','<<g[k][c];
    out<<'\n';
    const Eigen::Isometry3d M=cfg.ext*ag.legs[k].body*cfg.ext.inverse();
    const Eigen::Quaterniond q(M.linear());
    legs<<i<<','<<w.frame_ids[i]<<','<<w.frame_ids[j]<<','<<ti<<','<<tj<<','<<(selected=="fixed"?1.:ScaleAt(scales,.5*(ti+tj))[1])
        <<','<<M.translation().x()<<','<<M.translation().y()<<','<<M.translation().z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<'\n';
  }
  frame_table<<std::setprecision(17)<<"graph_index,frame_id,timestamp,fixed\n";
  for (size_t i=0;i<w.frames.size();++i) frame_table<<i<<','<<w.frame_ids[i]<<','<<w.frames[i]->GetTimestamp()<<','<<graph.poses[i]->fixed()<<'\n';
  result["legs"]=ag.legs.size(); result["seconds"]=Seconds(started);
  WriteYaml(output/"adjoint_report.yaml",result);
  std::cout<<"adjoint: legs="<<ag.legs.size()<<" matched="<<matched<<"/"<<rows<<" |b|="<<b.Norm()<<" |w|="<<wv.Norm()
           <<" |g|="<<std::sqrt(g2)<<" scaled_grad="<<st.scaled_max<<std::endl;
  return 0;
}

YAML::Node Diagnostics(const Statistics& s) {
  YAML::Node n; n["valid"]=s.valid; n["reason"]=s.reason; n["rank"]=s.rank; n["states"]=s.states;
  n["observations"]=s.observations; n["q"]=Array(s.q); n["nu"]=Array(s.nu); n["S"]=Array(s.S); n["beta"]=Array(s.beta);
  n["condition"]=s.condition; n["fit_residual"]=s.fit_residual; n["redundancy_error"]=s.redundancy_error; return n;
}
}  // namespace

int RunExperiment(const std::string& config_path,bool global,bool adjoint) {
  const YAML::Node root=YAML::LoadFile(config_path);
  if (adjoint) global=true;  // vce-adjoint reads the fusion run's vce-global config
  if (adjoint) Keys(root,{"vce","adjoint"}); else Keys(root,{"vce"});
  const auto n=root["vce"];
  if (!n || !Read(n,"enabled",false)) { std::cout<<"VCE disabled; no input loaded or output written\n"; return 0; }
  Keys(n,{"enabled","map_file","velocity_csv","camera_config","output_dir","T_C_B","time_delay",
    "rotation_sigma","translation_sigma","window_frames","max_points","alpha","max_outer","log_tolerance",
    "damping","alpha_min","alpha_max","max_change_factor","min_redundancy","max_condition","rank_tolerance",
    "max_iterations","mono_chi2","stereo_chi2","window_report","export_training","gyro_bias",
    "global_tolerance","inlier_cache","global_init","init_gate_t","init_gate_deg","init_min_shared"});
  if (n["window_report"] && !global) throw std::runtime_error("window_report requires vce-global mode");
  Config cfg;
  cfg.export_training=Read(n,"export_training",false);
  if (!global && cfg.export_training) throw std::runtime_error("export_training requires global mode");
  cfg.window=Read(n,"window_frames",8); cfg.points=Read(n,"max_points",128); cfg.outer=Read(n,"max_outer",5);
  cfg.iterations=Read(n,"max_iterations",100); cfg.tolerance=Read(n,"log_tolerance",.05);
  cfg.damping=Read(n,"damping",.5); cfg.lower=Read(n,"alpha_min",.01); cfg.upper=Read(n,"alpha_max",100.);
  cfg.step=Read(n,"max_change_factor",2.); cfg.delay=Read(n,"time_delay",0.);
  cfg.rotation_sigma=Read(n,"rotation_sigma",.08); cfg.translation_sigma=Read(n,"translation_sigma",.08);
  cfg.mono_chi2=Read(n,"mono_chi2",5.991); cfg.stereo_chi2=Read(n,"stereo_chi2",7.815);
  cfg.global_tolerance=Read(n,"global_tolerance",1e-8); cfg.inlier_cache=Read(n,"inlier_cache",false);
  cfg.global_init=Read<std::string>(n,"global_init","none");
  if (cfg.global_init!="none" && cfg.global_init!="leg_reanchor")
    throw std::runtime_error("global_init must be none or leg_reanchor");
  cfg.init_gate_t=Read(n,"init_gate_t",.5); cfg.init_gate_deg=Read(n,"init_gate_deg",15.);
  cfg.init_min_shared=Read(n,"init_min_shared",20);
  if (!(cfg.init_gate_t>0) || !(cfg.init_gate_deg>0) || cfg.init_min_shared<1) throw std::runtime_error("invalid global_init gates");
  if (!(cfg.global_tolerance>0 && cfg.global_tolerance<1)) throw std::runtime_error("global_tolerance must be in (0, 1)");
  cfg.limits.rank_tolerance=Read(n,"rank_tolerance",1e-10); cfg.limits.min_redundancy=Read(n,"min_redundancy",5.);
  cfg.limits.complete_moments=true;
  cfg.limits.max_condition=Read(n,"max_condition",1e8);
  if (n["alpha"]) { auto v=n["alpha"].as<std::vector<double>>(); if (v.size()!=2) throw std::runtime_error("alpha requires [visual, leg]"); cfg.alpha<<v[0],v[1]; }
  auto ext=n["T_C_B"].as<std::vector<double>>();
  if (ext.size()!=7) throw std::runtime_error("T_C_B requires [tx,ty,tz,qx,qy,qz,qw]");
  Eigen::Quaterniond q(ext[6],ext[3],ext[4],ext[5]);
  if (!q.coeffs().allFinite() || std::abs(q.norm()-1)>1e-3) throw std::runtime_error("invalid extrinsic quaternion");
  cfg.ext.linear()=q.normalized().toRotationMatrix(); cfg.ext.translation()<<ext[0],ext[1],ext[2];
  if (!cfg.ext.matrix().allFinite() || !std::isfinite(cfg.delay) || cfg.window<2 || cfg.points<1 || cfg.outer<1 || cfg.iterations<5 ||
      !(cfg.tolerance>0) || !(cfg.rotation_sigma>0) || !(cfg.translation_sigma>0) || !(cfg.mono_chi2>0) || !(cfg.stereo_chi2>0) ||
      !(cfg.limits.rank_tolerance>0 && cfg.limits.rank_tolerance<1) || !(cfg.limits.min_redundancy>=0) || !(cfg.limits.max_condition>=1))
    throw std::runtime_error("invalid VCE settings");
  for (double value:{cfg.tolerance,cfg.rotation_sigma,cfg.translation_sigma,cfg.mono_chi2,cfg.stereo_chi2,
                     cfg.limits.rank_tolerance,cfg.limits.min_redundancy,cfg.limits.max_condition})
    if (!std::isfinite(value)) throw std::runtime_error("VCE settings must be finite");
  UpdateScales(cfg.alpha,Eigen::Vector2d::Ones(),cfg.damping,cfg.lower,cfg.upper,cfg.step);
  if ((cfg.alpha.array()<cfg.lower).any() || (cfg.alpha.array()>cfg.upper).any()) throw std::runtime_error("initial alpha outside bounds");
  const auto base=fs::absolute(config_path).parent_path();
  auto path=[&](const char* key) { fs::path p=n[key].as<std::string>(); return fs::weakly_canonical(p.is_absolute()?p:base/p); };
  if (const auto b=n["gyro_bias"]) {
    Keys(b,{"enabled","initial_csv","random_walk_sigma","prior_sigma"});
    cfg.gyro_bias=Read(b,"enabled",false);
    cfg.bias_random_walk_sigma=Read(b,"random_walk_sigma",cfg.bias_random_walk_sigma);
    cfg.bias_prior_sigma=Read(b,"prior_sigma",cfg.bias_prior_sigma);
    if (cfg.gyro_bias && !global) throw std::runtime_error("gyro_bias requires vce-global mode");
    if (!(cfg.bias_random_walk_sigma>0) || !(cfg.bias_prior_sigma>0)) throw std::runtime_error("invalid gyro_bias sigmas");
    if (cfg.gyro_bias && b["initial_csv"]) {
      const fs::path p=b["initial_csv"].as<std::string>();
      cfg.initial_bias=LoadBias(p.is_absolute()?p:base/p);
    }
  }
  const auto map_path=path("map_file"),csv_path=path("velocity_csv"),camera_path=path("camera_config"),output=path("output_dir");
  if (global && cfg.inlier_cache) cfg.inlier_cache_file=map_path.parent_path()/"vce_global_inliers.txt";
  if (global) {
    const auto profile_path=path("window_report");
    const auto profile=YAML::LoadFile(profile_path.string());
    for(const char* key:{"map_file","velocity_csv","camera_config"})
      if(!profile[key] || fs::weakly_canonical(profile[key].as<std::string>())!=path(key))
        throw std::runtime_error(std::string("window/global input mismatch: ")+key);
    const auto previous=YAML::LoadFile((profile_path.parent_path()/"config.yaml").string())["vce"];
    for(const char* key:{"rotation_sigma","translation_sigma","time_delay"})
      if(std::abs(Read(previous,key,std::string(key)=="time_delay"?0.:.08)-Read(n,key,std::string(key)=="time_delay"?0.:.08))>1e-12)
        throw std::runtime_error(std::string("window/global calibration mismatch: ")+key);
    const auto prior_ext=previous["T_C_B"].as<std::vector<double>>();
    if(prior_ext.size()!=ext.size()) throw std::runtime_error("window/global extrinsic size mismatch");
    for(size_t i=0;i<ext.size();++i) if(std::abs(prior_ext[i]-ext[i])>1e-12)
      throw std::runtime_error("window/global extrinsic mismatch");
  }
  fs::path adjoint_output;
  if (adjoint) {
    const auto a=root["adjoint"];
    Keys(a,{"fusion_dir","upper_grad_csv","output_dir","stationarity_tol","max_reconverge_iterations","exact"});
    fs::path p=a["output_dir"].as<std::string>(); adjoint_output=fs::weakly_canonical(p.is_absolute()?p:base/p);
    if (fs::exists(adjoint_output) && !fs::is_empty(adjoint_output)) throw std::runtime_error("adjoint output_dir must be new or empty");
  } else if (fs::exists(output) && !fs::is_empty(output)) throw std::runtime_error("output_dir must be new or empty: "+output.string());
  if (!fs::is_regular_file(map_path) || !fs::is_regular_file(camera_path)) throw std::runtime_error("map/camera input missing");
  auto camera=std::make_shared<Camera>(camera_path.string());
  if (camera->UseIMU()) throw std::runtime_error("camera_config requests IMU; VCE supports visual and leg factors only");
  const LegOdomBuffer buffer=LoadVelocity(csv_path.string());
  MapPtr map(new Map); std::ifstream input(map_path,std::ios::binary);
  { boost::archive::binary_iarchive archive(input); archive>>map; }
  if (map->IMUInit()) throw std::runtime_error("IMU-initialized maps unsupported: supply a pure visual map");
  const auto stored=map->GetCameraPtr();
  if (!stored || std::abs(stored->Fx()-camera->Fx())>1e-6 || std::abs(stored->Fy()-camera->Fy())>1e-6 ||
      std::abs(stored->Cx()-camera->Cx())>1e-6 || std::abs(stored->Cy()-camera->Cy())>1e-6 || std::abs(stored->BF()-camera->BF())>1e-6)
    throw std::runtime_error("camera calibration differs from serialized rectified map camera");
  std::vector<std::pair<int,FramePtr>> frames(map->GetAllKeyframes().begin(),map->GetAllKeyframes().end());
  for (const auto& frame:frames) if (!std::isfinite(frame.second->GetTimestamp()))
    throw std::runtime_error("invalid keyframe timestamp");
  std::sort(frames.begin(),frames.end(),[](const auto& a,const auto& b){return a.second->GetTimestamp()!=b.second->GetTimestamp()?a.second->GetTimestamp()<b.second->GetTimestamp():a.first<b.first;});
  for (size_t i=0;i<frames.size();++i) if (!std::isfinite(frames[i].second->GetTimestamp()) ||
      (i && frames[i].second->GetTimestamp()<=frames[i-1].second->GetTimestamp())) throw std::runtime_error("invalid keyframe timestamps");
  if (adjoint) {
    fs::create_directories(adjoint_output); WriteYaml(adjoint_output/"config.yaml",root);
    return RunAdjoint(frames,map,camera,buffer,cfg,path("window_report"),root["adjoint"],base,adjoint_output);
  }
  fs::create_directories(output); WriteYaml(output/"config.yaml",root);
  if (global) return RunGlobal(frames,map,camera,buffer,cfg,path("window_report"),output);
  YAML::Node report;
  report["map_file"]=map_path.string(); report["velocity_csv"]=csv_path.string(); report["camera_config"]=camera_path.string();
  report["conventions"]="poses T_W_C; T_C_B conjugates T_Bi_Bj; residual [rotation(rad),translation(m)] in measured Cj axes; visual residual pixels";
  report["covariance_model"]="independent left-held velocity samples, variances already include checkpoint linear scale squared; no temporal/cross-edge correlation model; floors added in residual coordinates";
  report["scope"]="window factor-group effective covariance scales, not instantaneous velocity labels; no accuracy claim without GT";
  report["total_keyframes"]=frames.size();
  report["window_frames"]=cfg.window;
  report["windows"]=YAML::Node(YAML::NodeType::Sequence);
  report["trailing_frames_skipped"]=frames.size()%cfg.window;
  for (size_t begin=0;begin+cfg.window<=frames.size();begin+=cfg.window) {
    const auto started=Clock::now(); YAML::Node record;
    std::ostringstream name; name<<"window_"<<std::setw(5)<<std::setfill('0')<<begin/cfg.window;
    const auto dir=output/name.str(); fs::create_directory(dir);
    record["name"]=name.str(); record["begin"]=frames[begin].second->GetTimestamp(); record["end"]=frames[begin+cfg.window-1].second->GetTimestamp();
    try {
      Window w=Select(frames,begin,cfg); const size_t candidate=w.points.size();
      FreezeInliers(w,camera,cfg); record["candidate_points"]=candidate; record["points"]=w.points.size(); record["frame_ids"]=w.frame_ids;
      for (size_t i=0;i+1<w.frames.size();++i) w.motion.push_back(Integrate(buffer,w.frames[i]->GetTimestamp()+cfg.delay,w.frames[i+1]->GetTimestamp()+cfg.delay,cfg.ext,cfg.rotation_sigma,cfg.translation_sigma));
      std::ofstream obs(dir/"observations.csv"); obs<<std::setprecision(17)<<"frame_id,point_id,keypoint_index,u,v,u_right\n";
      for (const auto& o:w.observations) obs<<w.frame_ids[o.frame]<<','<<w.point_ids[o.point]<<','<<o.index<<','<<o.uv.x()<<','<<o.uv.y()<<','<<o.uv.z()<<'\n';
      Graph graph(w,camera,cfg); Eigen::Vector2d alpha=cfg.alpha; graph.Weights(alpha);
      SaveState(dir,"initial",w,graph);
      auto clock=Clock::now();
      if (!graph.Optimize(cfg.iterations) || !graph.PositiveDepth()) throw std::runtime_error("fixed-weight baseline did not converge or has negative depth");
      record["baseline_seconds"]=Seconds(clock); SaveState(dir,"baseline",w,graph);
      record["baseline_alpha"]=Array(alpha); record["states"]=graph.states;
      record["status"]="max_outer"; clock=Clock::now();
      for (int iteration=0;iteration<cfg.outer;++iteration) {
        const auto tick=Clock::now(); const Statistics stats=Estimate(graph.Linearize(),cfg.limits);
        YAML::Node step=Diagnostics(stats); step["iteration"]=iteration; step["alpha"]=Array(alpha);
        if (!stats.valid) { record["status"]="fallback"; record["reason"]=stats.reason; step["seconds"]=Seconds(tick); record["iterations"].push_back(step); break; }
        const auto update=UpdateScales(alpha,stats.beta,cfg.damping,cfg.lower,cfg.upper,cfg.step);
        step["proposed_alpha"]=Array(update.alpha); step["limited"]=update.limited; step["log_change"]=update.log_change;
        std::string failure;
        const StateUpdate transaction{[&]{graph.Push();},[&]{graph.Pop();},[&]{graph.Commit();},
          [&](const Eigen::VectorXd& a){graph.Weights(a);},
          [&]{return graph.Optimize(cfg.iterations) && graph.PositiveDepth();}};
        if (!ApplyCandidate(alpha,update.alpha,transaction,failure)) {
          record["status"]="fallback"; record["reason"]=failure;
          step["accepted"]=false; step["seconds"]=Seconds(tick); record["iterations"].push_back(step); break;
        }
        alpha=update.alpha; step["accepted"]=true; step["seconds"]=Seconds(tick);
        record["iterations"].push_back(step);
        // A bound-limited fixed point is not evidence of moment convergence.
        if (stats.beta.array().log().abs().maxCoeff()<cfg.tolerance) { record["status"]="converged"; break; }
      }
      record["vce_extra_seconds"]=Seconds(clock); record["alpha"]=Array(alpha); SaveState(dir,"vce",w,graph);
      record["final_statistics"]=Diagnostics(Estimate(graph.Linearize(),cfg.limits));
      YAML::Node covariance;
      covariance["visual_base_covariance"]="identity (2D mono or 3D stereo), pixel squared";
      covariance["visual_baseline_scale"]=cfg.alpha[0]; covariance["visual_final_scale"]=alpha[0];
      covariance["T_C_B"]=Array(cfg.ext.matrix()); covariance["time_delay"]=cfg.delay;
      covariance["residual_order"]="rx,ry,rz,tx,ty,tz";
      for (size_t i=0;i<w.motion.size();++i) {
        YAML::Node item; item["frame_i"]=w.frame_ids[i]; item["frame_j"]=w.frame_ids[i+1];
        item["body_delta"]=Array(w.motion[i].body.matrix()); item["base"]=Array(w.motion[i].residual_covariance);
        item["baseline"]=Array(cfg.alpha[1]*w.motion[i].residual_covariance); item["final"]=Array(alpha[1]*w.motion[i].residual_covariance);
        covariance["leg"].push_back(item);
      }
      WriteYaml(dir/"covariances.yaml",covariance);
    } catch (const std::exception& e) { record["status"]="rejected"; record["reason"]=e.what(); }
    record["seconds"]=Seconds(started); WriteYaml(dir/"diagnostics.yaml",record); report["windows"].push_back(record);
    std::cout<<name.str()<<": "<<record["status"].as<std::string>(); if(record["reason"]) std::cout<<" ("<<record["reason"].as<std::string>()<<')'; std::cout<<std::endl;
    WriteYaml(output/"report.yaml",report);
  }
  WriteYaml(output/"report.yaml",report);
  return 0;
}
}  // namespace vce
