#include "super_point_kernels.h"

#include "cuda_utils.h"

namespace sp_cuda {
namespace {

__global__ void NmsKernel(const float* heat, int height, int width, int border, float threshold,
                          int* count, int* index, float* score, int capacity) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < border || x >= width - border || y < border || y >= height - border) return;
  const float s = heat[y * width + x];
  if (s < threshold) return;
  for (int yy = max(0, y - 4); yy <= min(height - 1, y + 4); ++yy)
    for (int xx = max(0, x - 4); xx <= min(width - 1, x + 4); ++xx)
      if (heat[yy * width + xx] > s) return;
  const int slot = atomicAdd(count, 1);
  if (slot < capacity) {
    index[slot] = y * width + x;
    score[slot] = s;
  }
}

__device__ __forceinline__ int Clip(int value, int max_value) { return max(0, min(value, max_value - 1)); }

// One thread per keypoint: the channel sums run in the CPU order, with explicit IEEE round-to-nearest
// operations so that the compiler cannot contract them into FMAs.
__global__ void DescriptorKernel(const float* desc, int dh, int dw, float stride, const float* xy, int n, float* out,
                                 int official) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  float x, y;
  if (official) {   // official SuperPoint cell centres (grid_sample, align_corners): (p - s/2 + 0.5) / (w s - s/2 - 0.5) (w - 1)
    const float h = 0.5f * stride - 0.5f;
    x = (xy[2 * k] - h) / (dw * stride - h - 1.0f) * (dw - 1);
    y = (xy[2 * k + 1] - h) / (dh * stride - h - 1.0f) * (dh - 1);
  } else {
    x = __fdiv_rn(xy[2 * k], stride);
    y = __fdiv_rn(xy[2 * k + 1], stride);
  }
  const int x0 = Clip(static_cast<int>(floorf(x)), dw);
  const int y0 = Clip(static_cast<int>(floorf(y)), dh);
  const int x1 = Clip(x0 + 1, dw);
  const int y1 = Clip(y0 + 1, dh);
  const float dx = __fsub_rn(x, static_cast<float>(x0));
  const float dy = __fsub_rn(y, static_cast<float>(y0));
  const float w00 = __fmul_rn(__fsub_rn(1.0f, dx), __fsub_rn(1.0f, dy));
  const float w01 = __fmul_rn(dx, __fsub_rn(1.0f, dy));
  const float w10 = __fmul_rn(__fsub_rn(1.0f, dx), dy);
  const float w11 = __fmul_rn(dx, dy);
  const int plane = dh * dw;
  float* o = out + static_cast<size_t>(k) * 256;
  float norm = 0.0f;
  for (int c = 0; c < 256; ++c) {
    const float* d = desc + static_cast<size_t>(c) * plane;
    float v = __fmul_rn(w00, d[y0 * dw + x0]);
    v = __fadd_rn(v, __fmul_rn(w01, d[y0 * dw + x1]));
    v = __fadd_rn(v, __fmul_rn(w10, d[y1 * dw + x0]));
    v = __fadd_rn(v, __fmul_rn(w11, d[y1 * dw + x1]));
    o[c] = v;
    norm = __fadd_rn(norm, __fmul_rn(v, v));
  }
  norm = __fsqrt_rn(norm);
  if (norm > 1e-6f)
    for (int c = 0; c < 256; ++c) o[c] = __fdiv_rn(o[c], norm);
}

}  // namespace

void LaunchNms(const float* heat_map, int height, int width, int border, float threshold,
               int* count, int* index, float* score, int capacity, cudaStream_t stream) {
  LG_CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  NmsKernel<<<grid, block, 0, stream>>>(heat_map, height, width, border, threshold, count, index, score, capacity);
  LG_CUDA_CHECK(cudaGetLastError());
}

void LaunchDescriptors(const float* descriptors, int height, int width, int stride,
                       const float* xy, int n, float* out, cudaStream_t stream, bool official) {
  if (n <= 0) return;
  DescriptorKernel<<<(n + 127) / 128, 128, 0, stream>>>(descriptors, height / stride, width / stride,
                                                       static_cast<float>(stride), xy, n, out, official ? 1 : 0);
  LG_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sp_cuda
