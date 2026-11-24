#include "dynamic_segmenter_kernels.h"

#include "cuda_utils.h"

namespace yolo_cuda {
namespace {

constexpr int kCoefficients = 32;

__global__ void LetterboxKernel(const uint8_t* gray, int rw, int rh, float* input, int iw, int ih) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= iw || y >= ih) return;
  const float v = (x < rw && y < rh) ? gray[y * rw + x] * (1.0f / 255.0f) : 114.0f / 255.0f;
  const int plane = iw * ih, i = y * iw + x;
  input[i] = v;                  // grey -> the same value in the 3 (RGB) channels
  input[plane + i] = v;
  input[2 * plane + i] = v;
}

__global__ void DecodeKernel(const float* out, int anchors, int classes, uint64_t lo, uint64_t hi, float confidence,
                             float* candidates, int* count, int capacity) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= anchors) return;
  const float* scores = out + 4 * anchors + a;   // row-major [4 + classes + 32, anchors]: coalesced over anchors
  float best = scores[0];
  int cls = 0;
  for (int c = 1; c < classes; ++c) {
    const float s = scores[c * anchors];
    if (s > best) { best = s; cls = c; }
  }
  if (best < confidence) return;
  const bool selected = cls < 64 ? ((lo >> cls) & 1ULL) : ((hi >> (cls - 64)) & 1ULL);
  if (!selected) return;
  const int slot = atomicAdd(count, 1);
  if (slot >= capacity) return;
  const float cx = out[a], cy = out[anchors + a], w = out[2 * anchors + a], h = out[3 * anchors + a];
  float* c = candidates + 7 * slot;
  c[0] = cx - 0.5f * w; c[1] = cy - 0.5f * h; c[2] = cx + 0.5f * w; c[3] = cy + 0.5f * h;
  c[4] = best; c[5] = static_cast<float>(cls); c[6] = static_cast<float>(a);
}

__global__ void MaskRoisKernel(const float* out, int anchors, int classes, const float* protos, int ph, int pw,
                               const int* rois, float* result) {
  const int* r = rois + 6 * blockIdx.x;   // one block per detection
  const int x0 = r[0], y0 = r[1], w = r[2], h = r[3], offset = r[4], a = r[5];
  __shared__ float coefficient[kCoefficients];
  if (threadIdx.x < kCoefficients) coefficient[threadIdx.x] = out[(4 + classes + threadIdx.x) * anchors + a];
  __syncthreads();
  const int plane = ph * pw;
  for (int i = threadIdx.x; i < w * h; i += blockDim.x) {
    const int p = (y0 + i / w) * pw + x0 + i % w;
    float logit = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < kCoefficients; ++c) logit += coefficient[c] * protos[c * plane + p];
    result[offset + i] = 1.0f / (1.0f + __expf(-logit));
  }
}

}  // namespace

void LaunchLetterbox(const uint8_t* gray, int rw, int rh, float* input, int iw, int ih, cudaStream_t stream) {
  const dim3 block(32, 8), grid((iw + 31) / 32, (ih + 7) / 8);
  LetterboxKernel<<<grid, block, 0, stream>>>(gray, rw, rh, input, iw, ih);
  LG_CUDA_CHECK(cudaGetLastError());
}

void LaunchDecode(const float* output0, int anchors, int classes, uint64_t lo, uint64_t hi, float confidence,
                  float* candidates, int* count, int capacity, cudaStream_t stream) {
  LG_CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));
  DecodeKernel<<<(anchors + 255) / 256, 256, 0, stream>>>(output0, anchors, classes, lo, hi, confidence,
                                                          candidates, count, capacity);
  LG_CUDA_CHECK(cudaGetLastError());
}

void LaunchMaskRois(const float* output0, int anchors, int classes, const float* protos, int ph, int pw,
                    const int* rois, int n, float* result, cudaStream_t stream) {
  if (n <= 0) return;
  MaskRoisKernel<<<n, 256, 0, stream>>>(output0, anchors, classes, protos, ph, pw, rois, result);
  LG_CUDA_CHECK(cudaGetLastError());
}

}  // namespace yolo_cuda
