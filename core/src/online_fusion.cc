#include "online_fusion.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>
#include <random>

#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

#include "g2o_optimization/g2o_optimization.h"
#include "mappoint.h"
#include "vce/covariance.h"

#ifdef AIRSLAM_WITH_FFS
#include "ffs_infer.hpp"
#endif

namespace {

const char* kRejectNames[] = {"NO_CONTACT", "CONTACT_TRANSIENT", "NO_LEG_DATA", "NO_ANCHOR", "NO_COVERAGE",
    "STEP_REGION", "EDGE_OR_HOLE", "NON_PLANAR", "ASSOCIATION_TOO_FAR", "SEGMENT_INCONSISTENT", "BUDGET"};

// Gravity-aligned frame G at the camera: z up, x along the horizontal camera heading.
Eigen::Matrix3d GravityFrame(const Eigen::Matrix4d& T_WC) {
  const Eigen::Matrix3d R_CW = T_WC.topLeftCorner<3, 3>().transpose();
  const Eigen::Vector3d z = R_CW * Eigen::Vector3d::UnitZ();
  Eigen::Vector3d x = Eigen::Vector3d::UnitZ() - z * z.z();  // camera optical axis, made horizontal
  if (x.norm() < 1e-3) x = Eigen::Vector3d::UnitX() - z * z.x();
  x.normalize();
  Eigen::Matrix3d R_CG;
  R_CG << x, z.cross(x), z;
  return R_CG;
}

Eigen::Vector3d CameraFromWorld(const Eigen::Matrix4d& T_WC, const Eigen::Vector3d& p_w) {
  return T_WC.topLeftCorner<3, 3>().transpose() * (p_w - T_WC.topRightCorner<3, 1>());
}

Eigen::Vector3d WorldFoot(const FramePtr& frame, const Eigen::Isometry3d& T_C_B, const Eigen::Vector3d& foot_b) {
  const Eigen::Matrix4d& T_WC = frame->GetPose();
  return T_WC.topLeftCorner<3, 3>() * (T_C_B * foot_b) + T_WC.topRightCorner<3, 1>();
}

// Conservative metric depth from left-referenced disparity (after ffs_infer main_test.cpp
// disparityToDepth): border crop, x-d>=0 (visible in the right image), min disparity,
// depth range, and rejection of pixels next to any 3x3 depth discontinuity or hole.
cv::Mat DisparityToDepth(const std::vector<float>& disparity, int width, int height, double bf,
                         const SurfaceConfig& cfg) {
  cv::Mat depth(height, width, CV_32F, cv::Scalar(0));
  for (int v = cfg.border; v < height - cfg.border; ++v) {
    for (int u = cfg.border; u < width - cfg.border; ++u) {
      const float d = disparity[static_cast<size_t>(v) * width + u];
      if (!std::isfinite(d) || d < cfg.min_disp || u - d < 0.0f) continue;
      const float z = static_cast<float>(bf / d);
      if (z >= cfg.min_depth && z <= cfg.max_depth) depth.at<float>(v, u) = z;
    }
  }
  cv::Mat max_depth, min_depth;
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
  cv::dilate(depth, max_depth, kernel);
  cv::erode(depth, min_depth, kernel);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      float& z = depth.at<float>(v, u);
      if (z > 0.0f && (max_depth.at<float>(v, u) - min_depth.at<float>(v, u)) > cfg.grad_rel * z) z = 0.0f;
    }
  }
  return depth;
}

}  // namespace

// ------------------------------------------------------------------ SuperframeMap

SuperframeMap::SuperframeMap(const SurfaceConfig& cfg, CameraPtr camera) : cfg_(cfg), camera_(camera) {
#ifdef AIRSLAM_WITH_FFS
  if (cfg_.enabled && !cfg_.ffs_yaml.empty()) {
    // Loaded here, before the node subscribes to any input: a missing engine is built first (minutes) instead of
    // during the run (otherwise a run without the engine has no depth and the foot-surface factors are silently
    // off). A failure stops the start-up: the foot-surface factors are part of the configured method.
    try {
      ffs_ = std::make_unique<ffs_infer::FFSInfer>(cfg_.ffs_yaml);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("[S3] cannot load or build the FFS engine (online.surface.ffs_yaml ") +
                               cfg_.ffs_yaml + "): " + e.what());
    }
    depth_available_ = true;
    depth_thread_ = std::thread(&SuperframeMap::DepthLoop, this);
  }
#endif
  if (!depth_available_) {
    std::cerr << "[S3] superframe dense depth unavailable (build with AIRSLAM_WITH_FFS=ON and set "
                 "online.surface.ffs_yaml); foot-surface factors disabled" << std::endl;
  }
}

SuperframeMap::~SuperframeMap() {
  {
    std::lock_guard<std::mutex> lock(job_mutex_);
    stop_ = true;
  }
  job_cv_.notify_all();
  if (depth_thread_.joinable()) depth_thread_.join();
}

void SuperframeMap::DepthLoop() {
#ifdef AIRSLAM_WITH_FFS
  ffs_infer::FFSInfer* ffs = ffs_.get();
  std::vector<float> disparity;
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(job_mutex_);
      job_cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
      if (stop_) return;
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    cv::Mat left = job.left, right = job.right;
    if (left.channels() == 1) cv::cvtColor(job.left, left, cv::COLOR_GRAY2BGR);
    if (right.channels() == 1) cv::cvtColor(job.right, right, cv::COLOR_GRAY2BGR);
    auto lap = std::chrono::steady_clock::now();
    try {
      ffs->infer(left.data, right.data, left.cols, left.rows, disparity, static_cast<int>(left.step));
    } catch (const std::exception& e) {
      std::cerr << "[S3] FFS inference failed: " << e.what() << std::endl;
      continue;
    }
    lap = times.Lap("ffs_infer", lap);
    std::vector<float> xyz;
    std::vector<int> keys;
    if (depth_on_cpu_ || !DisparityPointsGpu(disparity, left.cols, left.rows, job.R_CG, &xyz, &keys)) {
      DepthPoints(job.R_CG, DisparityToDepth(disparity, left.cols, left.rows, camera_->BF(), cfg_), &xyz, &keys);
      lap = times.Lap("depth_points_cpu", lap);
    } else {
      lap = times.Lap("depth_points_gpu", lap);
      if (depth_verify_) {
        std::vector<float> xyz_cpu;
        std::vector<int> keys_cpu;
        DepthPoints(job.R_CG, DisparityToDepth(disparity, left.cols, left.rows, camera_->BF(), cfg_), &xyz_cpu, &keys_cpu);
        ++depth_checked_;
        depth_mismatch_ += xyz_cpu != xyz || keys_cpu != keys;
        std::cout << "[S3] depth GPU/CPU check: " << depth_mismatch_ << " mismatching of " << depth_checked_
                  << " (points " << xyz.size() / 3 << ")" << std::endl;
        lap = std::chrono::steady_clock::now();
      }
    }
    InsertPoints(job.frame, job.R_CG, xyz, keys);
    times.Lap("insert_points", lap);
  }
#endif
}

void SuperframeMap::InsertDepth(const FramePtr& frame, const Eigen::Matrix3d& R_CG, const cv::Mat& depth) {
  std::vector<float> xyz;
  std::vector<int> keys;
  DepthPoints(R_CG, depth, &xyz, &keys);
  InsertPoints(frame, R_CG, xyz, keys);
}

cv::Mat SuperframeMap::DepthFromDisparity(const std::vector<float>& disparity, int width, int height) const {
  return DisparityToDepth(disparity, width, height, camera_->BF(), cfg_);
}

void SuperframeMap::DepthPoints(const Eigen::Matrix3d& R_CG, const cv::Mat& depth, std::vector<float>* xyz,
                                std::vector<int>* keys) const {
  const Eigen::Matrix3f R_GC = R_CG.transpose().cast<float>();
  const float fx = camera_->Fx(), fy = camera_->Fy(), cx = camera_->Cx(), cy = camera_->Cy();
  std::unordered_set<VoxelKey, VoxelKeyHash> voxels;
  xyz->clear();
  keys->clear();
  for (int v = 0; v < depth.rows; v += cfg_.stride) {
    for (int u = 0; u < depth.cols; u += cfg_.stride) {
      const float z = depth.at<float>(v, u);
      if (!(z > 0.0f)) continue;
      const Eigen::Vector3f p = R_GC * Eigen::Vector3f((u - cx) * z / fx, (v - cy) * z / fy, z);
      const int ix = static_cast<int>(std::floor(p.x() / cfg_.voxel_xy));
      const int iy = static_cast<int>(std::floor(p.y() / cfg_.voxel_xy));
      const int iz = static_cast<int>(std::floor(p.z() / cfg_.voxel_z));
      if (!voxels.insert(VoxelKey{ix, iy, iz}).second) continue;
      xyz->insert(xyz->end(), {p.x(), p.y(), p.z()});
      keys->insert(keys->end(), {ix, iy, iz});
    }
  }
}

bool SuperframeMap::DisparityPointsGpu(const std::vector<float>& disparity, int width, int height,
                                       const Eigen::Matrix3d& R_CG, std::vector<float>* xyz, std::vector<int>* keys) {
  try {
    if (!voxelizer_) voxelizer_ = std::make_unique<depth_cuda::Voxelizer>();
    depth_cuda::Params p;
    p.width = width; p.height = height; p.border = cfg_.border; p.stride = cfg_.stride;
    p.bf = camera_->BF(); p.min_disp = cfg_.min_disp; p.min_depth = cfg_.min_depth; p.max_depth = cfg_.max_depth;
    p.grad_rel = cfg_.grad_rel; p.voxel_xy = cfg_.voxel_xy; p.voxel_z = cfg_.voxel_z;
    p.fx = camera_->Fx(); p.fy = camera_->Fy(); p.cx = camera_->Cx(); p.cy = camera_->Cy();
    const Eigen::Matrix3f R_GC = R_CG.transpose().cast<float>();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) p.R_GC[3 * r + c] = R_GC(r, c);
    return static_cast<int>(disparity.size()) == width * height && voxelizer_->Run(disparity.data(), p, xyz, keys);
  } catch (const std::exception& e) {
    std::cerr << "[S3] GPU depth post-processing failed (" << e.what() << "); CPU path" << std::endl;
    depth_on_cpu_ = true;
    return false;
  }
}

void SuperframeMap::InsertPoints(const FramePtr& frame, const Eigen::Matrix3d& R_CG, const std::vector<float>& xyz,
                                 const std::vector<int>& keys) {
  auto record = std::make_shared<SuperframeRecord>();
  record->frame_id = frame->GetFrameId();
  record->frame = frame;
  record->R_CG = R_CG;
  record->points.reserve(xyz.size() / 3);
  for (size_t i = 0; 3 * i + 2 < xyz.size(); ++i) {
    record->columns[VoxelKey{keys[3 * i], keys[3 * i + 1], 0}].push_back(static_cast<int>(record->points.size()));
    record->points.emplace_back(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
  }
  if (cfg_.old_plane) DetectPlanes(*record);
  std::unique_lock<std::shared_mutex> lock(records_mutex_);
  records_.push_back(record);
  while (static_cast<int>(records_.size()) > cfg_.max_superframes) records_.pop_front();
}

// Elevation-map anchors: up to plane_max RANSAC planes of the superframe cloud (G, z up), kept when the fit rmse, tilt
// and extent pass; inliers stored in the superframe camera frame so the plane moves with its keyframe pose.
void SuperframeMap::DetectPlanes(const SuperframeRecord& record) {
  // 5 x 5 x 2 cm downsampling in G (z up)
  std::unordered_set<long long> seen;
  std::vector<Eigen::Vector3d> pts;
  for (const auto& p : record.points) {
    const long long kx = static_cast<long long>(std::floor(p.x() / 0.05f)) + (1 << 20);
    const long long ky = static_cast<long long>(std::floor(p.y() / 0.05f)) + (1 << 20);
    const long long kz = static_cast<long long>(std::floor(p.z() / 0.02f)) + (1 << 20);
    if (seen.insert((kx << 42) | (ky << 21) | kz).second) pts.push_back(p.cast<double>());
  }
  std::mt19937 rng(static_cast<unsigned>(record.frame_id));
  std::vector<int> rest(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) rest[i] = static_cast<int>(i);
  auto fit = [&](const std::vector<int>& ids, Eigen::Vector3d* n, Eigen::Vector3d* c) {
    c->setZero();
    for (int i : ids) *c += pts[i];
    *c /= static_cast<double>(ids.size());
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int i : ids) cov += (pts[i] - *c) * (pts[i] - *c).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    *n = es.eigenvectors().col(0);
    if (n->z() < 0) *n = -*n;
    return std::sqrt(std::max(0.0, es.eigenvalues()(0)) / static_cast<double>(ids.size()));
  };
  std::vector<OldPlane> found;
  for (int k = 0; k < cfg_.plane_max && static_cast<int>(rest.size()) >= cfg_.plane_min_inliers; ++k) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(rest.size()) - 1);
    std::vector<int> best;
    for (int it = 0; it < cfg_.plane_ransac_iter; ++it) {
      const Eigen::Vector3d& a = pts[rest[pick(rng)]];
      const Eigen::Vector3d& b = pts[rest[pick(rng)]];
      const Eigen::Vector3d& d = pts[rest[pick(rng)]];
      Eigen::Vector3d n = (b - a).cross(d - a);
      if (n.norm() < 1e-9) continue;
      n.normalize();
      std::vector<int> inl;
      for (int i : rest) if (std::abs((pts[i] - a).dot(n)) < cfg_.plane_ransac_thr) inl.push_back(i);
      if (inl.size() > best.size()) best.swap(inl);
    }
    if (static_cast<int>(best.size()) < cfg_.plane_min_inliers) break;
    Eigen::Vector3d n, c;
    fit(best, &n, &c);
    std::vector<int> inl, out;
    for (int i : rest) (std::abs((pts[i] - c).dot(n)) < cfg_.plane_ransac_thr ? inl : out).push_back(i);
    rest.swap(out);
    if (static_cast<int>(inl.size()) < cfg_.plane_min_inliers) continue;
    const double rmse = fit(inl, &n, &c);
    Eigen::Vector2d lo(1e9, 1e9), hi(-1e9, -1e9);
    for (int i : inl) { lo = lo.cwiseMin(pts[i].head<2>()); hi = hi.cwiseMax(pts[i].head<2>()); }
    const double tilt = std::acos(std::min(1.0, std::abs(n.z()))) * 180.0 / M_PI;
    if (rmse >= cfg_.plane_rmse || tilt > cfg_.plane_tilt_deg || (hi - lo).maxCoeff() <= cfg_.plane_min_extent) continue;
    OldPlane p;
    p.frame_id = record.frame_id; p.frame = record.frame; p.time = record.frame->GetTimestamp(); p.rmse = rmse;
    p.c_c = record.R_CG * c;
    p.q_c = p.c_c;
    p.n_c = record.R_CG * n;
    if (p.n_c.dot(-p.q_c) < 0) p.n_c = -p.n_c;  // towards the camera, as the online planes
    const size_t step = std::max<size_t>(1, inl.size() / 1500);
    for (size_t j = 0; j < inl.size(); j += step) {
      const Eigen::Vector3d q = record.R_CG * pts[inl[j]];
      p.pts_c.push_back(q.cast<float>());
      p.radius = std::max(p.radius, (q - p.c_c).norm());
    }
    found.push_back(std::move(p));
  }
  std::lock_guard<std::mutex> lock(old_mutex_);
  ++old_superframes_;
  if (found.empty()) ++old_superframes_empty_;
  old_planes_kept_ += found.size();
  for (auto& p : found) old_planes_.push_back(std::move(p));
}

bool SuperframeMap::OldPlaneConstraint(const FramePtr& frame, int leg, const Eigen::Vector3d& foot_b,
                                       const Eigen::Vector3d& p_w, FootSurfaceConstraint* c) {
  std::lock_guard<std::mutex> lock(old_mutex_);
  const auto key = std::make_pair(frame->GetFrameId(), leg);
  auto cached = old_cache_.find(key);
  if (cached != old_cache_.end()) {
    if (cached->second.first) *c = cached->second.second;
    return cached->second.first;
  }
  bool found = false;
  const double t = frame->GetTimestamp();
  int last_frame = -1;
  bool frame_hit = false;
  for (const OldPlane& p : old_planes_) {           // in insertion (time) order: oldest first
    if (p.time > t - cfg_.old_plane_dt) break;
    if (p.frame_id != last_frame) { last_frame = p.frame_id; frame_hit = false; }
    if (frame_hit) continue;                        // the first plane under the foot decides for this superframe
    const Eigen::Vector3d pc = CameraFromWorld(p.frame->GetPose(), p_w);
    if ((pc - p.c_c).norm() > p.radius + cfg_.old_plane_near) continue;
    const double d = p.n_c.dot(pc - p.q_c);
    const Eigen::Vector3f proj = (pc - d * p.n_c).cast<float>();
    float best = 1e9f;
    for (const auto& q : p.pts_c) best = std::min(best, (q - proj).squaredNorm());
    if (best > cfg_.old_plane_near * cfg_.old_plane_near) continue;
    frame_hit = true;
    if (std::abs(d) > cfg_.foot_radius + cfg_.old_plane_gate) continue;   // the foot is not on this plane
    c->id_pose = frame->GetFrameId();
    c->id_anchor = p.frame_id;
    c->foot_b = foot_b;
    c->q_c = pc - d * p.n_c;
    c->n_c = p.n_c;
    c->radius = cfg_.foot_radius;
    c->sigma = cfg_.sigma;
    c->rmse = p.rmse;
    found = true;
    ++old_edges_;
    break;
  }
  old_cache_[key] = std::make_pair(found, *c);
  while (old_cache_.size() > 4096) old_cache_.erase(old_cache_.begin());
  return found;
}

Eigen::Vector3d SuperframeMap::CorrectedFoot(const Eigen::Vector3d& foot_b) const {
  if (cfg_.foot_corr_roll_deg == 0.0 && cfg_.foot_corr_pitch_deg == 0.0 && cfg_.foot_corr_dz == 0.0) return foot_b;
  // R = Ry(pitch) Rx(roll), the convention of the fit (scipy from_euler('xy'))
  const Eigen::Matrix3d R = (Eigen::AngleAxisd(cfg_.foot_corr_pitch_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(cfg_.foot_corr_roll_deg * M_PI / 180.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return R * foot_b + Eigen::Vector3d(0.0, 0.0, cfg_.foot_corr_dz);
}

std::vector<int> SuperframeMap::Near(const SuperframeRecord& record, const Eigen::Vector3d& p_g,
                                     double r, double band) const {
  std::vector<int> ids;
  const int x0 = static_cast<int>(std::floor((p_g.x() - r) / cfg_.voxel_xy));
  const int x1 = static_cast<int>(std::floor((p_g.x() + r) / cfg_.voxel_xy));
  const int y0 = static_cast<int>(std::floor((p_g.y() - r) / cfg_.voxel_xy));
  const int y1 = static_cast<int>(std::floor((p_g.y() + r) / cfg_.voxel_xy));
  const Eigen::Vector3f c = p_g.cast<float>();
  for (int ix = x0; ix <= x1; ++ix) {
    for (int iy = y0; iy <= y1; ++iy) {
      const auto it = record.columns.find(VoxelKey{ix, iy, 0});
      if (it == record.columns.end()) continue;
      for (int id : it->second) {
        const Eigen::Vector3f& p = record.points[id];
        if (std::abs(p.z() - c.z()) <= band && (p - c).head<2>().squaredNorm() <= r * r) ids.push_back(id);
      }
    }
  }
  return ids;
}

SuperframeMap::Plane SuperframeMap::FitPlane(const SuperframeRecord& record, const Eigen::Vector3d& foot_g) const {
  Plane plane;
  // 1. Height-discontinuity check over the margin-extended neighbourhood.
  const double margin = cfg_.edge_margin + cfg_.edge_sigma_k * cfg_.pose_sigma_xy;
  const std::vector<int> wide = Near(record, foot_g, cfg_.radius + margin, cfg_.step_band);
  if (static_cast<int>(wide.size()) < cfg_.min_pts) { plane.reject = SurfaceReject::NO_COVERAGE; return plane; }
  const int bins = static_cast<int>(std::ceil(2.0 * cfg_.step_band / 0.01)) + 1;
  std::vector<int> histogram(bins, 0);
  std::vector<double> heights;
  heights.reserve(wide.size());
  for (int id : wide) {
    const double dz = record.points[id].z() - foot_g.z();
    heights.push_back(dz);
    histogram[std::clamp(static_cast<int>((dz + cfg_.step_band) / 0.01), 0, bins - 1)]++;
  }
  // Layers: runs of occupied bins (1-bin gaps tolerated) holding >= 5% of the points.
  const int min_layer = std::max(3, static_cast<int>(0.05 * wide.size()));
  std::vector<std::pair<double, int>> layers;  // (centre height, count)
  for (int b = 0; b < bins;) {
    if (histogram[b] == 0) { ++b; continue; }
    int count = 0, gap = 0, e = b;
    double sum = 0.0;
    for (; e < bins && gap <= 1; ++e) {
      if (histogram[e] == 0) { ++gap; continue; }
      gap = 0;
      count += histogram[e];
      sum += histogram[e] * (e * 0.01 - cfg_.step_band + 0.005);
    }
    if (count >= min_layer) layers.emplace_back(sum / count, count);
    b = e;
  }
  std::sort(heights.begin(), heights.end());
  const double span = heights[heights.size() * 95 / 100] - heights[heights.size() * 5 / 100];
  bool step = span > cfg_.max_height_span;
  for (size_t i = 0; i < layers.size() && !step; ++i)
    for (size_t j = i + 1; j < layers.size(); ++j)
      if (std::abs(layers[i].first - layers[j].first) > cfg_.step_min) step = true;
  if (step || layers.empty()) { plane.reject = SurfaceReject::STEP_REGION; return plane; }
  const double layer_z = foot_g.z() + std::max_element(layers.begin(), layers.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; })->first;

  // 2. Main-layer points within the plane radius; all four quadrants must be supported.
  std::vector<Eigen::Vector3d> pts;
  std::array<int, 4> quadrant{};
  for (int id : Near(record, Eigen::Vector3d(foot_g.x(), foot_g.y(), layer_z), cfg_.radius, cfg_.layer_band)) {
    const Eigen::Vector3d p = record.points[id].cast<double>();
    pts.push_back(p);
    quadrant[(p.x() >= foot_g.x() ? 0 : 1) + (p.y() >= foot_g.y() ? 0 : 2)]++;
  }
  if (static_cast<int>(pts.size()) < cfg_.min_pts) { plane.reject = SurfaceReject::NO_COVERAGE; return plane; }
  if (*std::min_element(quadrant.begin(), quadrant.end()) < cfg_.min_quadrant_pts) {
    plane.reject = SurfaceReject::EDGE_OR_HOLE;
    return plane;
  }

  // 3. Huber-IRLS weighted PCA.
  std::vector<double> weight(pts.size(), 1.0);
  Eigen::Vector3d centroid, normal;
  Eigen::Vector3d lambda;
  for (int iteration = 0; iteration < 3; ++iteration) {
    double wsum = 0.0;
    centroid.setZero();
    for (size_t i = 0; i < pts.size(); ++i) { centroid += weight[i] * pts[i]; wsum += weight[i]; }
    centroid /= wsum;
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < pts.size(); ++i) cov += weight[i] * (pts[i] - centroid) * (pts[i] - centroid).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov / wsum);
    lambda = eig.eigenvalues();  // ascending
    normal = eig.eigenvectors().col(0);
    if (normal.z() < 0.0) normal = -normal;
    for (size_t i = 0; i < pts.size(); ++i) {
      const double d = std::abs(normal.dot(pts[i] - centroid));
      weight[i] = d <= cfg_.max_rmse ? 1.0 : cfg_.max_rmse / d;
    }
  }
  double sq = 0.0;
  int inliers = 0;
  for (const auto& p : pts) {
    const double d = normal.dot(p - centroid);
    sq += d * d;
    inliers += std::abs(d) <= 2.0 * cfg_.max_rmse;
  }
  plane.rmse = std::sqrt(sq / pts.size());
  const double zc = (record.R_CG * foot_g).z();
  const double depth_sigma = cfg_.depth_sigma_coeff * zc * zc;
  const double max_rmse = std::sqrt(cfg_.max_rmse * cfg_.max_rmse + depth_sigma * depth_sigma);
  const double total = lambda.sum();
  const bool planar = plane.rmse <= max_rmse &&
      inliers >= cfg_.min_inlier_ratio * pts.size() &&
      total > 0.0 && lambda[0] / total <= cfg_.planarity &&
      lambda[1] >= cfg_.min_extent_ratio * lambda[2] &&
      std::acos(std::min(1.0, normal.z())) <= cfg_.max_slope_deg * M_PI / 180.0;
  if (!planar) { plane.reject = SurfaceReject::NON_PLANAR; return plane; }
  plane.q = centroid;
  plane.n = normal;
  return plane;
}

bool SuperframeMap::Covered(const Eigen::Vector3d& p_w) const {
  std::shared_lock<std::shared_mutex> lock(records_mutex_);
  for (const auto& record : records_) {
    const Eigen::Vector3d p_g = record->R_CG.transpose() * CameraFromWorld(record->frame->GetPose(), p_w);
    if (static_cast<int>(Near(*record, p_g, cfg_.radius, cfg_.step_band).size()) >= cfg_.min_pts) return true;
  }
  return false;
}

void SuperframeMap::OnKeyframe(const FramePtr& frame, const cv::Mat& left, const cv::Mat& right,
                               const LegOdomStream& legs, const Eigen::Isometry3d& T_C_B, double time_delay) {
  if (!depth_available_) return;
  {
    std::lock_guard<std::mutex> lock(job_mutex_);
    if (static_cast<int>(jobs_.size()) >= cfg_.max_depth_queue) return;
  }
  const Eigen::Matrix4d& T_WC = frame->GetPose();
  bool select = !last_superframe_;
  if (!select) {
    const double dt = frame->GetTimestamp() - last_superframe_->GetTimestamp();
    if (dt < cfg_.min_dt) return;
    const Eigen::Matrix4d rel = last_superframe_->GetPose().inverse() * T_WC;
    const double angle = Eigen::AngleAxisd(Eigen::Matrix3d(rel.topLeftCorner<3, 3>())).angle();
    select = dt > cfg_.max_dt || rel.topRightCorner<3, 1>().norm() > cfg_.trans ||
             angle > cfg_.rot_deg * M_PI / 180.0;
    FootState foot;
    std::array<bool, 4> stable;
    if (!select && legs.FootAt(frame->GetTimestamp() + time_delay, &foot, &stable)) {
      int contacts = 0, covered = 0;
      for (int leg = 0; leg < 4; ++leg) {
        if (!foot.contact[leg]) continue;
        ++contacts;
        covered += Covered(WorldFoot(frame, T_C_B, CorrectedFoot(foot.foot_b[leg])));
      }
      select = contacts > 0 && covered < cfg_.min_foot_coverage * contacts;
    }
  }
  if (!select) return;
  last_superframe_ = frame;
  {
    std::lock_guard<std::mutex> lock(selected_mutex_);
    selected_.insert(frame->GetFrameId());
    while (selected_.size() > 1024) selected_.erase(selected_.begin());
  }
  {
    std::lock_guard<std::mutex> lock(job_mutex_);
    jobs_.push_back({frame, GravityFrame(T_WC), left.clone(), right.clone()});
  }
  job_cv_.notify_one();
}

bool SuperframeMap::IsSuperframe(int frame_id) const {
  std::lock_guard<std::mutex> lock(selected_mutex_);
  return selected_.count(frame_id) > 0;
}

std::vector<SuperframeRecordPtr> SuperframeMap::Anchors(const std::function<bool(int)>& usable) const {
  std::vector<SuperframeRecordPtr> anchors;
  std::shared_lock<std::shared_mutex> lock(records_mutex_);
  for (auto it = records_.rbegin(); it != records_.rend() &&
       static_cast<int>(anchors.size()) < cfg_.num_anchor_superframes; ++it) {
    if (usable((*it)->frame_id)) anchors.push_back(*it);
  }
  return anchors;
}

bool SuperframeMap::Verify(const SegmentPlane& plane, const SuperframeRecord& anchor, const Eigen::Vector3d& p_w) const {
  const Eigen::Vector3d d = CameraFromWorld(anchor.frame->GetPose(), p_w) - plane.q_c;
  const double normal = plane.n_c.dot(d);
  const double gate = cfg_.foot_radius + cfg_.contact_offset_margin + 3.0 * plane.sigma;
  return std::abs(normal) <= gate && (d - normal * plane.n_c).norm() <= cfg_.radius;
}

void SuperframeMap::Associate(const std::vector<FramePtr>& free_frames, const std::vector<SuperframeRecordPtr>& anchors,
                              const LegOdomStream& legs, const Eigen::Isometry3d& T_C_B, double time_delay,
                              VectorOfFootConstraints* out) {
  std::map<std::pair<uint32_t, int>, int> used;
  for (const FramePtr& frame : free_frames) {
    FootState foot;
    std::array<bool, 4> stable;
    if (!legs.FootAt(frame->GetTimestamp() + time_delay, &foot, &stable)) { Reject(SurfaceReject::NO_LEG_DATA); continue; }
    for (int leg = 0; leg < 4; ++leg) {
      if (!foot.contact[leg]) { Reject(SurfaceReject::NO_CONTACT); continue; }
      if (!stable[leg]) { Reject(SurfaceReject::CONTACT_TRANSIENT); continue; }
      if (anchors.empty()) { Reject(SurfaceReject::NO_ANCHOR); continue; }
      const auto key = std::make_pair(foot.segment[leg], leg);
      if (used[key] >= cfg_.max_obs_per_segment) { Reject(SurfaceReject::BUDGET); continue; }
      const Eigen::Vector3d foot_b = CorrectedFoot(foot.foot_b[leg]);
      const Eigen::Vector3d p_w = WorldFoot(frame, T_C_B, foot_b);
      auto emit = [&](const SegmentPlane& plane) {
        FootSurfaceConstraint c;
        c.id_pose = frame->GetFrameId();
        c.id_anchor = plane.anchor_id;
        c.foot_b = foot_b;
        c.q_c = plane.q_c;
        c.n_c = plane.n_c;
        c.radius = cfg_.foot_radius;
        c.sigma = plane.sigma;
        c.rmse = plane.rmse;
        out->push_back(c);
        used[key]++;
      };

      auto try_old = [&]() {
        if (!cfg_.old_plane) return;
        FootSurfaceConstraint oc;
        if (OldPlaneConstraint(frame, leg, foot_b, p_w, &oc)) out->push_back(oc);
      };
      // Contact-segment consistency: one physical surface per stance; later frames re-verify it.
      auto cached = segments_.find(key);
      if (cached != segments_.end()) {
        if (cached->second.bad) { Reject(SurfaceReject::SEGMENT_INCONSISTENT); continue; }
        const auto anchor = std::find_if(anchors.begin(), anchors.end(),
            [&](const SuperframeRecordPtr& a) { return a->frame_id == cached->second.anchor_id; });
        if (anchor != anchors.end()) {
          if (Verify(cached->second, **anchor, p_w)) { emit(cached->second); try_old(); }
          else { cached->second.bad = true; Reject(SurfaceReject::SEGMENT_INCONSISTENT); }
          continue;
        }
        segments_.erase(cached);  // anchor left the anchor set: associate again
      }

      // Closest observation first: depth noise grows with z^2.
      std::vector<std::pair<double, SuperframeRecordPtr>> candidates;
      for (const auto& anchor : anchors) {
        const double z = CameraFromWorld(anchor->frame->GetPose(), p_w).z();
        if (z > 0.0) candidates.emplace_back(z, anchor);
      }
      std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      SurfaceReject reason = SurfaceReject::NO_COVERAGE;
      bool done = false;
      for (size_t k = 0; k < candidates.size() && static_cast<int>(k) < cfg_.max_anchor_trials && !done; ++k) {
        const SuperframeRecord& anchor = *candidates[k].second;
        const Eigen::Vector3d foot_g = anchor.R_CG.transpose() * CameraFromWorld(anchor.frame->GetPose(), p_w);
        const Plane plane = FitPlane(anchor, foot_g);
        if (plane.reject != SurfaceReject::COUNT) {
          reason = plane.reject;
          if (reason == SurfaceReject::STEP_REGION) break;  // a step is a property of the terrain, not the view
          continue;
        }
        const double depth_sigma = cfg_.depth_sigma_coeff * candidates[k].first * candidates[k].first;
        SegmentPlane segment{anchor.frame_id, anchor.R_CG * plane.q, anchor.R_CG * plane.n,
            std::sqrt(cfg_.sigma * cfg_.sigma + depth_sigma * depth_sigma + plane.rmse * plane.rmse), plane.rmse};
        if (!Verify(segment, anchor, p_w)) { reason = SurfaceReject::ASSOCIATION_TOO_FAR; continue; }
        segments_[key] = segment;
        emit(segment);
        try_old();
        done = true;
      }
      if (!done) Reject(reason);
    }
  }
  while (segments_.size() > 512) segments_.erase(segments_.begin());
}

std::string SuperframeMap::RejectSummary() {
  std::ostringstream text;
  for (size_t i = 0; i < rejects_.size(); ++i)
    if (rejects_[i]) text << ' ' << kRejectNames[i] << '=' << rejects_[i];
  rejects_.fill(0);
  if (cfg_.old_plane) {
    std::lock_guard<std::mutex> lock(old_mutex_);
    text << " OLD_PLANE_EDGES=" << old_edges_ << " PLANES=" << old_planes_kept_ << "/" << old_superframes_
         << " EMPTY_SF=" << old_superframes_empty_;
  }
  return text.str();
}

// ------------------------------------------------------------------ OnlineFusion

OnlineFusion::OnlineFusion(const OnlineFusionConfig& cfg, CameraPtr camera)
    : cfg_(cfg), camera_(camera), stream_(cfg.buffer_s) {
  if (cfg_.surface.enabled) superframes_ = std::make_unique<SuperframeMap>(cfg_.surface, camera_);
  if (cfg_.backend.enabled) backend_ = std::make_unique<FusionBackend>(cfg_.backend);
  std::cout << "[S3] online fusion mode=" << cfg_.mode << ", leg topic=" << cfg_.leg_topic << std::endl;
}

OnlineFusion::~OnlineFusion() = default;

Eigen::Isometry3d OnlineFusion::Extrinsic() const {
  Eigen::Isometry3d T_C_B;
  if (stream_.Extrinsic(&T_C_B)) return T_C_B;
  T_C_B.setIdentity();
  const Eigen::Vector4d& q = cfg_.leg.extrinsic_quat_xyzw;
  T_C_B.linear() = Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
  T_C_B.translation() = cfg_.leg.extrinsic_xyz;
  return T_C_B;
}

bool OnlineFusion::LegReady() const {
  Eigen::Isometry3d T;
  if (stream_.Extrinsic(&T) || cfg_.leg.has_extrinsic) return true;
  if (!warned_extrinsic_.exchange(true))
    std::cerr << "[S3] leg data ignored until the leg node publishes T_C_B (<leg_topic>/extrinsic)" << std::endl;
  return false;
}

bool OnlineFusion::LegSlip(const FramePtr& f1, const FramePtr& f2, const LegRelativeConstraint& c, double* slip) const {
  FootState s1, s2;
  std::array<bool, 4> k1, k2;
  const double delay = cfg_.leg.time_delay;
  if (!stream_.FootAt(f1->GetTimestamp() + delay, &s1, &k1) || !stream_.FootAt(f2->GetTimestamp() + delay, &s2, &k2)) return false;
  const Eigen::Isometry3d T_C_B = Extrinsic();
  double sum = 0.0;
  int n = 0;
  for (int leg = 0; leg < 4; ++leg) {
    if (!s1.contact[leg] || !s2.contact[leg] || !k1[leg] || !k2[leg] || s1.segment[leg] != s2.segment[leg]) continue;
    sum += (T_C_B * s1.foot_b[leg] - c.T_c1_c2 * (T_C_B * s2.foot_b[leg])).norm();
    ++n;
  }
  if (!n) return false;
  *slip = sum / n;
  return true;
}

bool OnlineFusion::Standing(double t1, double t2, double max_foot_m) const {
  FootState s1, s2;
  std::array<bool, 4> k1, k2;
  const double delay = cfg_.leg.time_delay;
  if (!stream_.FootAt(t1 + delay, &s1, &k1)) return false;
  const int n = std::min(100, std::max(5, static_cast<int>(std::ceil((t2 - t1) / 0.1))));
  for (int i = 1; i <= n; ++i) {
    if (!stream_.FootAt(t1 + (t2 - t1) * i / n + delay, &s2, &k2)) return false;
    for (int leg = 0; leg < 4; ++leg)
      if ((s1.foot_b[leg] - s2.foot_b[leg]).norm() >= max_foot_m) return false;
  }
  return true;
}

bool OnlineFusion::LegConstraint(const FramePtr& f1, const FramePtr& f2, LegRelativeConstraint* c, double wait_s) const {
  if (!LegReady()) return false;
  const double t0 = f1->GetTimestamp() + cfg_.leg.time_delay, t1 = f2->GetTimestamp() + cfg_.leg.time_delay;
  if (t1 - t0 < cfg_.leg.min_keyframe_dt) return false;
  const Eigen::Isometry3d T_C_B = Extrinsic();
  const std::pair<int, int> key(f1->GetFrameId(), f2->GetFrameId());
  Eigen::Isometry3d body;
  Eigen::Matrix<double, 6, 6> covariance;
  bool cached = false;
  {
    std::lock_guard<std::mutex> lock(leg_cache_mutex_);
    auto it = leg_cache_.find(key);
    if (it != leg_cache_.end() && it->second.T_C_B.matrix() == T_C_B.matrix()) {
      body = it->second.body;
      covariance = it->second.covariance;
      cached = true;
    }
  }
  if (cached) {
    ++leg_cached_;
  } else {
    LegOdomBuffer slice;
    bool waited = false;
    if (!stream_.Slice(t0, t1, &slice)) {
      // Late data (the leg network lags the camera): wait for the stream to reach t1, then retry once.
      waited = wait_s > 0.0 && stream_.EndTime() < t1 && stream_.WaitUntil(t1, wait_s);
      if (!waited || !stream_.Slice(t0, t1, &slice)) {
        ++(stream_.EndTime() < t1 ? leg_late_ : leg_uncovered_);
        return false;
      }
    }
    try {
      const vce::Motion motion = vce::Integrate(slice, t0, t1, T_C_B, cfg_.leg.rotation_sigma, cfg_.leg.translation_sigma);
      body = motion.body;
      covariance = motion.residual_covariance;
    } catch (const std::exception&) {
      ++leg_uncovered_;
      return false;
    }
    ++(waited ? leg_waited_ : leg_ok_);
    std::lock_guard<std::mutex> lock(leg_cache_mutex_);
    leg_cache_[key] = {T_C_B, body, covariance};
    while (leg_cache_.size() > 4096) leg_cache_.erase(leg_cache_.begin());  // oldest pairs (ids increase)
  }
  c->id_pose1 = f1->GetFrameId();
  c->id_pose2 = f2->GetFrameId();
  c->T_c1_c2 = T_C_B * body * T_C_B.inverse();
  c->information = covariance.inverse();
  return c->information.allFinite() && c->T_c1_c2.matrix().allFinite();
}

std::string OnlineFusion::LegStats() const {
  std::ostringstream out;
  out << "leg_queries ok=" << leg_ok_ << " ok_after_wait=" << leg_waited_ << " late=" << leg_late_
      << " not_covered=" << leg_uncovered_ << " cached=" << leg_cached_;
  return out.str();
}

void OnlineFusion::PrepareLocalBA(const std::vector<FramePtr>& window, MapOfPoses& poses,
                                  const std::function<FramePtr(int)>& keyframe, LocalBAExtras* extras) {
  const Eigen::Isometry3d T_C_B = Extrinsic();
  extras->T_C_B = T_C_B;
  extras->leg_huber = cfg_.leg.huber_delta;
  for (size_t i = 0; i + 1 < window.size(); ++i) {
    LegRelativeConstraint c;
    if (LegConstraint(window[i + 1], window[i], &c)) extras->legs.push_back(c);
  }
  if (!LegReady() || !superframes_ || !superframes_->DepthAvailable()) return;

  std::vector<FramePtr> free_frames;
  for (const FramePtr& frame : window) {
    auto it = poses.find(frame->GetFrameId());
    if (it != poses.end() && !it->second.fixed) free_frames.push_back(frame);
  }
  // Anchors: alive keyframes that are not optimised in this BA.
  const auto anchors = superframes_->Anchors([&](int id) {
    if (!keyframe(id)) return false;
    auto it = poses.find(id);
    return it == poses.end() || it->second.fixed;
  });
  VectorOfFootConstraints surfaces;
  superframes_->Associate(free_frames, anchors, stream_, T_C_B, cfg_.leg.time_delay, &surfaces);
  if (backend_) {
    std::map<int, VectorOfFootConstraints> fresh;
    for (const auto& c : surfaces) fresh[c.id_pose].push_back(c);
    for (auto& kv : fresh) feet_[kv.first] = std::move(kv.second);
  }
}

void OnlineFusion::FinishLocalBA(const std::vector<FramePtr>& window, const LocalBAExtras& extras) {
  std::cout << "[S3] BA legs=" << extras.legs.size();
  if (superframes_) std::cout << superframes_->RejectSummary();
  std::cout << std::endl;
}

namespace {
// Map points of `frame` also observed by `prev`, and the spread of their keypoints (smallest
// eigenvalue of the normalized pixel covariance, 1 for a uniform image coverage).
void Covisibility(const FramePtr& prev, const FramePtr& frame, double width, double height, int* count, double* spread) {
  const std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  const std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  Eigen::Vector2d sum = Eigen::Vector2d::Zero();
  Eigen::Matrix2d sq = Eigen::Matrix2d::Zero();
  int n = 0;
  for (size_t i = 0; i < mappoints.size() && i < keypoints.size(); ++i) {
    const MappointPtr& mpt = mappoints[i];
    if (!mpt || !mpt->IsValid() || !mpt->GetAllObversers().count(prev->GetFrameId())) continue;
    const Eigen::Vector2d u(keypoints[i].pt.x / width, keypoints[i].pt.y / height);
    sum += u;
    sq += u * u.transpose();
    ++n;
  }
  *count = n;
  *spread = 0.0;
  if (n < 3) return;
  const Eigen::Vector2d mean = sum / n;
  const Eigen::Matrix2d cov = sq / n - mean * mean.transpose();
  *spread = 12.0 * Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(cov).eigenvalues()[0];
}
Eigen::Isometry3d Iso(const Eigen::Matrix4d& T) { return Eigen::Isometry3d(T); }
}  // namespace

void OnlineFusion::Finalize(const std::vector<FramePtr>& window, const std::vector<FramePtr>& finished) {
  if (!backend_ || window.empty()) return;
  for (const FramePtr& frame : finished) {
    if (last_final_ && frame->GetFrameId() <= last_final_->GetFrameId()) continue;
    BackendNode node;
    node.id = frame->GetFrameId();
    node.time = frame->GetTimestamp();
    node.T_wc = Iso(frame->GetPose());
    node.T_c_b = Extrinsic();
    node.superframe = superframes_ && superframes_->IsSuperframe(node.id);
    node.lost = frame->track_state == TrackState::Lost && frame->pose_source != PoseSource::Visual;  // motion-model pose
    if (last_final_) {
      node.vo = Iso(last_final_->GetPose()).inverse() * node.T_wc;
      Covisibility(last_final_, frame, camera_->ImageWidth(), camera_->ImageHeight(), &node.covisible, &node.spread);
      LegRelativeConstraint c;
      if (LegConstraint(last_final_, frame, &c)) { node.has_leg = true; node.leg = c.T_c1_c2; }
      if (cfg_.backend.passive_vo_t > 0.0)
        node.standing = Standing(last_final_->GetTimestamp(), node.time, cfg_.backend.passive_foot_m);
      // Stance-foot slip: a foot in the same contact segment at both keyframes should not move in the world,
      // so its drift under the leg relative motion measures how inconsistent that motion is.
      FootState f0, f1;
      std::array<bool, 4> s0, s1;
      const double delay = cfg_.leg.time_delay;
      if (node.has_leg && stream_.FootAt(last_final_->GetTimestamp() + delay, &f0, &s0) &&
          stream_.FootAt(node.time + delay, &f1, &s1)) {
        double sum = 0.0;
        int n = 0;
        for (int leg = 0; leg < 4; ++leg) {
          if (!f0.contact[leg] || !f1.contact[leg] || !s0[leg] || !s1[leg] || f0.segment[leg] != f1.segment[leg]) continue;
          sum += (node.T_c_b * f0.foot_b[leg] - node.leg * (node.T_c_b * f1.foot_b[leg])).norm();
          ++n;
        }
        if (n) node.slip_mean = sum / n;
      }
      if (node.has_leg && cfg_.backend.leg_reliability)
        ReliabilityFeatures(last_final_->GetTimestamp() + cfg_.leg.time_delay, node.time + cfg_.leg.time_delay,
                            node.slip_mean, &node.rel_feat);
    }
    auto it = feet_.find(node.id);
    if (it != feet_.end()) node.feet = std::move(it->second);
    feet_.erase(feet_.begin(), feet_.upper_bound(node.id));
    node.gyro_bias_applied = stream_.LatestBias();
    if (cfg_.backend.export_extras) ExportExtras(frame, &node);
    backend_->Push(node);
    last_final_ = frame;
  }
  backend_->RecordRealtime(window.front()->GetTimestamp(), Iso(window.front()->GetPose()));
}

int OnlineFusion::NetGyroCorrect(ImuDataList* samples) {
  std::lock_guard<std::mutex> lock(imu_mutex_);
  return NetGyroCorrectLocked(samples);
}

int OnlineFusion::NetGyroCorrectLocked(ImuDataList* samples) {
  const Eigen::Matrix3d R_BI = NetGyroRBI(cfg_.backend), R_IB = R_BI.transpose();
  int corrected = 0;
  for (ImuData& d : *samples) {
    double tk = 0.0;
    Eigen::Vector3d w;
    // nearest network sample (100 Hz: within half a period plus margin), then the raw gyro at its time (the network
    // samples are IMU samples, same stamps)
    if (!stream_.NearestAngular(d.timestamp, 0.006, &tk, &w)) continue;
    auto it = std::lower_bound(imu_buf_.begin(), imu_buf_.end(), tk - 5e-4,
                               [](const ImuData& a, double t) { return a.timestamp < t; });
    if (it == imu_buf_.end() || std::abs(it->timestamp - tk) > 5e-4) continue;
    d.gyr += R_IB * (w - R_BI * it->gyr);
    ++corrected;
  }
  return corrected;
}

void OnlineFusion::PushImu(const ImuDataList& samples) {
  std::lock_guard<std::mutex> lock(imu_mutex_);
  for (const ImuData& d : samples)
    if (imu_buf_.empty() || d.timestamp > imu_buf_.back().timestamp) imu_buf_.push_back(d);   // batches overlap
  while (!imu_buf_.empty() && imu_buf_.front().timestamp < imu_buf_.back().timestamp - cfg_.buffer_s) imu_buf_.pop_front();
}

// Inputs of the back-end IMU factors and window marginalisation, copied from the front-end state at finalization (the
// front-end estimate is unchanged): sensor meta, raw IMU of (previous node, node] with one sample of margin on each
// side, and the node's map point observations (the map points change later).
void OnlineFusion::ExportExtras(const FramePtr& frame, BackendNode* node) {
  SensorMeta& meta = node->meta;
  meta.valid = true;
  meta.fx = camera_->Fx(); meta.fy = camera_->Fy(); meta.cx = camera_->Cx(); meta.cy = camera_->Cy(); meta.bf = camera_->BF();
  meta.has_imu = camera_->ImuMode() > 0;
  meta.T_i_c = Eigen::Isometry3d(camera_->TImuCam());
  meta.gyr_noise = camera_->GyrNoise(); meta.acc_noise = camera_->AccNoise();
  meta.gyr_walk = camera_->GyrWalk(); meta.acc_walk = camera_->AccWalk();
  if (meta.has_imu && last_final_) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    const double t0 = last_final_->GetTimestamp(), t1 = frame->GetTimestamp();
    for (size_t k = 0; k < imu_buf_.size(); ++k) {
      const double t = imu_buf_[k].timestamp;
      const bool before = k + 1 < imu_buf_.size() && imu_buf_[k + 1].timestamp > t0 && t <= t0;
      const bool after = k > 0 && imu_buf_[k - 1].timestamp < t1 && t >= t1;
      if ((t > t0 && t < t1) || before || after) node->imu.push_back(imu_buf_[k]);
    }
    if (cfg_.backend.net_gyro) {   // raw 400 Hz gyro + the network correction (preintegration, ZUPT)
      node->imu_net = NetGyroCorrectLocked(&node->imu);
      node->imu_raw = static_cast<int>(node->imu.size()) - node->imu_net;
    }
  }
  std::vector<MappointPtr>& mpts = frame->GetAllMappoints();
  for (size_t i = 0; i < mpts.size(); ++i) {
    const MappointPtr& m = mpts[i];
    Eigen::Vector3d kp;
    if (!m || !m->IsValid() || !frame->GetKeypointPosition(i, kp)) continue;
    LandmarkObs o;
    o.point = m->GetId();
    o.kp = kp;
    o.pw = m->GetPosition();
    o.n_obs = m->ObverserNum();
    node->obs.push_back(o);
  }
}

void OnlineFusion::PushAccel(double t, const Eigen::Vector3d& f_b) {
  std::lock_guard<std::mutex> lock(accel_mutex_);
  if (!accel_norm_.empty() && t <= accel_norm_.back().first) return;
  accel_norm_.emplace_back(t, f_b.norm());
  while (!accel_norm_.empty() && accel_norm_.front().first < t - cfg_.buffer_s) accel_norm_.pop_front();
}

bool OnlineFusion::ReliabilityFeatures(double t0, double t1, double slip, std::vector<double>* f) const {
  f->clear();
  LegOdomBuffer slice;
  if (!stream_.Slice(t0, t1, &slice)) return false;
  double speed = 0.0, rate = 0.0, var = 0.0;
  int n = 0;
  for (const auto& s : slice.samples()) {
    if (s.time < t0 || s.time >= t1) continue;
    speed += s.linear.norm();
    rate += s.angular.norm();
    var += s.variance.head<3>().sum();
    ++n;
  }
  double sum = 0.0, sq = 0.0;
  int m = 0;
  {
    std::lock_guard<std::mutex> lock(accel_mutex_);
    auto it = std::lower_bound(accel_norm_.begin(), accel_norm_.end(), t0,
                               [](const std::pair<double, double>& a, double t) { return a.first < t; });
    for (; it != accel_norm_.end() && it->first < t1; ++it, ++m) { sum += it->second; sq += it->second * it->second; }
  }
  if (n < 3 || m < 3) return false;
  const double mean = sum / m;
  const std::vector<double> raw = {std::log(std::max(slip >= 0.0 ? slip : 0.01, 1e-3)), speed / n, rate / n,
                                   std::log(var / n + 1e-9), std::sqrt(std::max(0.0, sq / m - mean * mean))};
  const auto& scale = cfg_.backend.leg_rel_feature_scale;
  for (size_t k = 0; k < raw.size(); ++k) f->push_back(raw[k] / scale[k]);
  return true;
}

void OnlineFusion::OnKeyframe(const FramePtr& frame, const cv::Mat& left, const cv::Mat& right) {
  if (superframes_ && LegReady()) superframes_->OnKeyframe(frame, left, right, stream_, Extrinsic(), cfg_.leg.time_delay);
}
