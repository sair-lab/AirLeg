#include "vce/covariance.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <cmath>

namespace vce {
Eigen::Matrix3d Exp(const Eigen::Vector3d& v) {
  const double a = v.norm();
  if (a < 1e-10) return Eigen::Matrix3d::Identity() + Skew(v) + 0.5*Skew(v)*Skew(v);
  return Eigen::AngleAxisd(a, v/a).toRotationMatrix();
}
void StepJacobians(const Eigen::Matrix3d& r, const Eigen::Vector3d& v,
                   const Eigen::Vector3d& w, double dt, Matrix6& F, Matrix6& G) {
  const Eigen::Vector3d phi = w*dt;
  const double a = phi.norm();
  const Eigen::Matrix3d k = Skew(phi);
  const Eigen::Matrix3d jr = Eigen::Matrix3d::Identity() -
      (a < 1e-5 ? 0.5-a*a/24 : (1-std::cos(a))/(a*a))*k +
      (a < 1e-5 ? 1.0/6-a*a/120 : (a-std::sin(a))/(a*a*a))*k*k;
  F.setIdentity();
  F.topLeftCorner<3,3>() = Exp(phi).transpose();
  F.bottomLeftCorner<3,3>() = -r*Skew(v*dt);
  G.setZero();
  G.topRightCorner<3,3>() = jr*dt;
  G.bottomLeftCorner<3,3>() = r*dt;
}
Matrix6 ResidualJacobian(const Eigen::Isometry3d& body, const Eigen::Isometry3d& ext) {
  // Residual [Log(R_meas^T R_pred), R_meas^T(t_pred-t_meas)] at pred=meas.
  // Conjugating T_B0_B1 by T_C_B also moves the camera's lever arm.
  const Eigen::Matrix3d rc = ext.linear()*body.linear()*ext.linear().transpose();
  Matrix6 j = Matrix6::Zero();
  j.topLeftCorner<3,3>() = -ext.linear();
  j.bottomRightCorner<3,3>() = -rc.transpose()*ext.linear();
  j.bottomLeftCorner<3,3>() = -rc.transpose()*ext.linear()*body.linear()*
      Skew(ext.linear().transpose()*ext.translation());
  return j;
}
LegOdomBuffer LoadVelocity(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open velocity CSV: " + path);
  auto split = [](const std::string& line) {
    std::stringstream stream(line); std::string field; std::vector<std::string> fields;
    while (std::getline(stream,field,',')) { if (!field.empty() && field.back()=='\r') field.pop_back(); fields.push_back(field); }
    return fields;
  };
  std::string line; std::getline(input,line);
  const auto header = split(line);
  std::map<std::string,int> columns;
  for (int i=0;i<static_cast<int>(header.size());++i) columns[header[i]]=i;
  const std::vector<std::string> required={"time","vx","vy","vz","wx","wy","wz",
      "var_vx","var_vy","var_vz","var_wx","var_wy","var_wz"};
  for (const auto& name:required) if (!columns.count(name))
    throw std::runtime_error("VCE requires CSV column: " + name);
  double previous=-INFINITY;
  LegOdomBuffer::Samples samples;
  while (std::getline(input,line)) {
    if (line.empty()) continue;
    const auto fields=split(line);
    LegVelocitySample sample;
    for (int i=0;i<13;++i) {
      const auto& field=fields.at(columns.at(required[i]));
      size_t consumed=0;
      const double x=std::stod(field,&consumed);
      if (field.find_first_not_of(" \t",consumed)!=std::string::npos) throw std::runtime_error("invalid CSV numeric field");
      if (!std::isfinite(x) || (i>=7 && x<0)) throw std::runtime_error("invalid CSV value");
      if (i==0) {
        if (x<=previous) throw std::runtime_error("VCE requires strictly increasing CSV timestamps");
        previous=x; sample.time=x;
      } else if (i<=3) sample.linear[i-1]=x;
      else if (i<=6) sample.angular[i-4]=x;
      else sample.variance[i-7]=x;
    }
    samples.push_back(sample);
  }
  if (samples.size()<2) throw std::runtime_error("velocity CSV needs at least two samples");
  return LegOdomBuffer(std::move(samples));
}
Motion Integrate(const LegOdomBuffer& buffer, double begin, double end,
                 const Eigen::Isometry3d& ext, double rotation_sigma, double translation_sigma) {
  Motion out;
  if (!buffer.Integrate(begin,end,Vector6::Zero(),&out.body))
    throw std::runtime_error("velocity CSV does not cover complete keyframe interval");
  const auto& samples=buffer.samples();
  auto it=std::upper_bound(samples.begin(),samples.end(),begin,
      [](double t,const LegVelocitySample& s){return t<s.time;});
  size_t i=static_cast<size_t>(it-samples.begin()-1);
  Eigen::Matrix3d rotation=Eigen::Matrix3d::Identity();
  for (;i+1<samples.size() && samples[i].time<end;++i) {
    const double dt=std::min(end,samples[i+1].time)-std::max(begin,samples[i].time);
    if (dt<=0) continue;
    Matrix6 f,g;
    StepJacobians(rotation,samples[i].linear,samples[i].angular,dt,f,g);
    out.body_covariance=f*out.body_covariance*f.transpose()+g*samples[i].variance.asDiagonal()*g.transpose();
    out.bias_jacobian=f*out.bias_jacobian-g.rightCols<3>();
    rotation=rotation*Exp(samples[i].angular*dt);
  }
  const Matrix6 j=ResidualJacobian(out.body,ext);
  out.residual_covariance=j*out.body_covariance*j.transpose();
  out.residual_covariance.diagonal().head<3>().array()+=rotation_sigma*rotation_sigma;
  out.residual_covariance.diagonal().tail<3>().array()+=translation_sigma*translation_sigma;
  out.residual_covariance=(0.5*(out.residual_covariance+out.residual_covariance.transpose())).eval();
  return out;
}
}  // namespace vce
