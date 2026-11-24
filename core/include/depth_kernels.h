#pragma once

#include <memory>
#include <vector>

// GPU post-processing of the FFS disparity of a superframe, bit-identical to the CPU reference
// (DisparityToDepth + SuperframeMap::InsertDepth in online_fusion.cc): disparity -> metric depth with border /
// visibility / range checks, rejection next to 3x3 depth discontinuities, strided back-projection into the
// gravity-aligned frame G and anisotropic voxel deduplication (first point in scan order per voxel).
namespace depth_cuda {

struct Params {
  int width = 0, height = 0, border = 16, stride = 2;
  double bf = 0.0, min_disp = 2.0, min_depth = 0.3, max_depth = 3.5, grad_rel = 0.05;
  double voxel_xy = 0.02, voxel_z = 0.01;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  float R_GC[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // row-major
};

class Voxelizer {
 public:
  Voxelizer();
  ~Voxelizer();
  // Points (x y z, in G) and voxel keys (ix iy iz) of the kept points, in scan order. False if a voxel index
  // is outside the packable range (|i| >= 2^20): the caller then uses the CPU path.
  bool Run(const float* disparity, const Params& params, std::vector<float>* points, std::vector<int>* keys);
  // Depth image after the discontinuity rejection (row-major), from the last Run (tests).
  std::vector<float> LastDepth() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace depth_cuda
