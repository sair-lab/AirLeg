#pragma once

#include <cuda_runtime.h>

// GPU post-processing of the dense SuperPoint outputs, bit-identical to the CPU reference
// (DetectPoint / ExtractDescriptors in super_point.cpp).
namespace sp_cuda {
// Candidates: score >= threshold, inside the border, no pixel of the 9x9 window strictly greater.
// Writes up to `capacity` (row-major index, score) pairs in arbitrary order; *count is the total found.
void LaunchNms(const float* heat_map, int height, int width, int border, float threshold,
               int* count, int* index, float* score, int capacity, cudaStream_t stream);
// Bilinear descriptor sampling at the (x, y) pixel positions and L2 normalisation (norm > 1e-6),
// output n x 256 (keypoint-major). Same operation order as the CPU (no fused multiply-add).
void LaunchDescriptors(const float* descriptors, int height, int width, int stride,
                       const float* xy, int n, float* out, cudaStream_t stream, bool official = false);
}  // namespace sp_cuda
