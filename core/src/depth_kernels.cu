#include "depth_kernels.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <algorithm>
#include <cstdint>

#include "cuda_utils.h"

namespace depth_cuda {
namespace {

constexpr int64_t kBias = 1 << 20;            // voxel indices packed as 3 x 21 bits
constexpr uint64_t kInvalid = ~uint64_t(0);   // sorts last

// Same operations and conversions as DisparityToDepth's first loop (double bf / float d, float range test
// promoted to double).
__global__ void DepthKernel(const float* disparity, Params p, float* depth) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= p.width || v >= p.height) return;
  float z_out = 0.0f;
  if (u >= p.border && u < p.width - p.border && v >= p.border && v < p.height - p.border) {
    const float d = disparity[static_cast<size_t>(v) * p.width + u];
    if (isfinite(d) && !(static_cast<double>(d) < p.min_disp) && !(__fsub_rn(static_cast<float>(u), d) < 0.0f)) {
      const float z = __double2float_rn(__ddiv_rn(p.bf, static_cast<double>(d)));
      if (static_cast<double>(z) >= p.min_depth && static_cast<double>(z) <= p.max_depth) z_out = z;
    }
  }
  depth[static_cast<size_t>(v) * p.width + u] = z_out;
}

// cv::dilate / cv::erode with a 3x3 kernel (out-of-image pixels ignored), then the relative-gradient test.
__global__ void EdgeKernel(const float* depth, Params p, float* out) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x, v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= p.width || v >= p.height) return;
  const size_t i = static_cast<size_t>(v) * p.width + u;
  float z = depth[i];
  if (z > 0.0f) {
    float hi = -INFINITY, lo = INFINITY;
    for (int y = max(0, v - 1); y <= min(p.height - 1, v + 1); ++y)
      for (int x = max(0, u - 1); x <= min(p.width - 1, u + 1); ++x) {
        const float w = depth[static_cast<size_t>(y) * p.width + x];
        hi = fmaxf(hi, w);
        lo = fminf(lo, w);
      }
    if (static_cast<double>(__fsub_rn(hi, lo)) > __dmul_rn(p.grad_rel, static_cast<double>(z))) z = 0.0f;
  }
  out[i] = z;
}

// InsertDepth's strided loop: back-projection, rotation into G (Eigen's x0 + (x1 + x2) order, no FMA), voxel key.
__global__ void SampleKernel(const float* depth, Params p, int cols, int n, float* xyz, int* key3, uint64_t* key,
                             int* index, int* overflow) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  const int v = (s / cols) * p.stride, u = (s % cols) * p.stride;
  index[s] = s;
  key[s] = kInvalid;
  const float z = depth[static_cast<size_t>(v) * p.width + u];
  if (!(z > 0.0f)) return;
  const float x = __fdiv_rn(__fmul_rn(__fsub_rn(static_cast<float>(u), p.cx), z), p.fx);
  const float y = __fdiv_rn(__fmul_rn(__fsub_rn(static_cast<float>(v), p.cy), z), p.fy);
  const float* R = p.R_GC;
  float q[3];
  for (int r = 0; r < 3; ++r)
    q[r] = __fadd_rn(__fmul_rn(R[3 * r], x), __fadd_rn(__fmul_rn(R[3 * r + 1], y), __fmul_rn(R[3 * r + 2], z)));
  const int ix = static_cast<int>(floor(__ddiv_rn(static_cast<double>(q[0]), p.voxel_xy)));
  const int iy = static_cast<int>(floor(__ddiv_rn(static_cast<double>(q[1]), p.voxel_xy)));
  const int iz = static_cast<int>(floor(__ddiv_rn(static_cast<double>(q[2]), p.voxel_z)));
  if (abs(ix) >= kBias || abs(iy) >= kBias || abs(iz) >= kBias) { atomicExch(overflow, 1); return; }
  xyz[3 * s] = q[0]; xyz[3 * s + 1] = q[1]; xyz[3 * s + 2] = q[2];
  key3[3 * s] = ix; key3[3 * s + 1] = iy; key3[3 * s + 2] = iz;
  key[s] = (static_cast<uint64_t>(ix + kBias) << 42) | (static_cast<uint64_t>(iy + kBias) << 21) |
           static_cast<uint64_t>(iz + kBias);
}

// After the (stable) radix sort by key, the first entry of every run is the first sample, in scan order, of
// that voxel.
__global__ void FirstKernel(const uint64_t* key, const int* index, int n, unsigned char* keep) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  keep[index[i]] = key[i] != kInvalid && (i == 0 || key[i] != key[i - 1]);
}

__global__ void GatherKernel(const int* kept, const int* count, const float* xyz, const int* key3, float* out_xyz,
                             int* out_key3) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= *count) return;
  const int s = kept[j];
  for (int c = 0; c < 3; ++c) {
    out_xyz[3 * j + c] = xyz[3 * s + c];
    out_key3[3 * j + c] = key3[3 * s + c];
  }
}

}  // namespace

// Persistent stream, device buffers, CUB temporary storage and pinned host memory: no allocation (and no
// device-wide synchronising cudaFree) per superframe.
struct Voxelizer::Impl {
  lightglue::CudaStream stream;
  lightglue::CudaBuffer<float> disparity, depth, depth2, xyz, out_xyz;
  lightglue::CudaBuffer<int> key3, index, index_sorted, kept, out_key3, counters;  // counters: overflow, count
  lightglue::CudaBuffer<uint64_t> key, key_sorted;
  lightglue::CudaBuffer<unsigned char> keep, temp;
  lightglue::PinnedBuffer<float> host_disparity, host_xyz;
  lightglue::PinnedBuffer<int> host_key3, host_counters;
};

Voxelizer::Voxelizer() : impl_(std::make_unique<Impl>()) {}
Voxelizer::~Voxelizer() = default;

bool Voxelizer::Run(const float* disparity, const Params& p, std::vector<float>* points, std::vector<int>* keys) {
  Impl& m = *impl_;
  const cudaStream_t stream = m.stream.get();
  const size_t pixels = static_cast<size_t>(p.width) * p.height;
  m.host_disparity.reserve(pixels);
  std::copy_n(disparity, pixels, m.host_disparity.get());
  m.disparity.copyFromHost(m.host_disparity.get(), pixels, stream);
  m.depth.resize(pixels);
  m.depth2.resize(pixels);
  const dim3 block(32, 8), grid((p.width + 31) / 32, (p.height + 7) / 8);
  DepthKernel<<<grid, block, 0, stream>>>(m.disparity.get(), p, m.depth.get());
  EdgeKernel<<<grid, block, 0, stream>>>(m.depth.get(), p, m.depth2.get());

  const int rows = (p.height + p.stride - 1) / p.stride, cols = (p.width + p.stride - 1) / p.stride;
  const int n = rows * cols;
  m.xyz.resize(3 * static_cast<size_t>(n));
  m.key3.resize(3 * static_cast<size_t>(n));
  m.key.resize(n);
  m.key_sorted.resize(n);
  m.index.resize(n);
  m.index_sorted.resize(n);
  m.keep.resize(n);
  m.kept.resize(n);
  m.out_xyz.resize(3 * static_cast<size_t>(n));
  m.out_key3.resize(3 * static_cast<size_t>(n));
  m.counters.resize(2);
  LG_CUDA_CHECK(cudaMemsetAsync(m.counters.get(), 0, 2 * sizeof(int), stream));
  SampleKernel<<<(n + 255) / 256, 256, 0, stream>>>(m.depth2.get(), p, cols, n, m.xyz.get(), m.key3.get(),
                                                    m.key.get(), m.index.get(), m.counters.get());
  LG_CUDA_CHECK(cudaGetLastError());

  // temporary storage for both CUB calls, grown once
  size_t sort_bytes = 0, select_bytes = 0;
  cub::CountingInputIterator<int> samples(0);
  LG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, m.key.get(), m.key_sorted.get(), m.index.get(),
                                                m.index_sorted.get(), n, 0, 64, stream));
  LG_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, select_bytes, samples, m.keep.get(), m.kept.get(),
                                           m.counters.get() + 1, n, stream));
  m.temp.resize(std::max(sort_bytes, select_bytes));
  LG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(m.temp.get(), sort_bytes, m.key.get(), m.key_sorted.get(), m.index.get(),
                                                m.index_sorted.get(), n, 0, 64, stream));
  FirstKernel<<<(n + 255) / 256, 256, 0, stream>>>(m.key_sorted.get(), m.index_sorted.get(), n, m.keep.get());
  LG_CUDA_CHECK(cub::DeviceSelect::Flagged(m.temp.get(), select_bytes, samples, m.keep.get(), m.kept.get(),
                                           m.counters.get() + 1, n, stream));
  GatherKernel<<<(n + 255) / 256, 256, 0, stream>>>(m.kept.get(), m.counters.get() + 1, m.xyz.get(), m.key3.get(),
                                                    m.out_xyz.get(), m.out_key3.get());
  LG_CUDA_CHECK(cudaGetLastError());
  m.host_counters.reserve(2);
  m.host_xyz.reserve(3 * static_cast<size_t>(n));
  m.host_key3.reserve(3 * static_cast<size_t>(n));
  m.counters.copyToHost(m.host_counters.get(), 2, stream);
  // copy the worst case (all samples) asynchronously, so a single synchronisation suffices
  m.out_xyz.copyToHost(m.host_xyz.get(), 3 * static_cast<size_t>(n), stream);
  m.out_key3.copyToHost(m.host_key3.get(), 3 * static_cast<size_t>(n), stream);
  m.stream.synchronize();
  const int count = m.host_counters[1];
  points->assign(m.host_xyz.get(), m.host_xyz.get() + 3 * static_cast<size_t>(count));
  keys->assign(m.host_key3.get(), m.host_key3.get() + 3 * static_cast<size_t>(count));
  return m.host_counters[0] == 0;
}

std::vector<float> Voxelizer::LastDepth() const {
  std::vector<float> out(impl_->depth2.size());
  impl_->depth2.copyToHost(out.data(), out.size(), impl_->stream.get());
  impl_->stream.synchronize();
  return out;
}

}  // namespace depth_cuda
