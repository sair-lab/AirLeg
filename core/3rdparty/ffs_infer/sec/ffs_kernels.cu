#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename T>
__device__ __forceinline__ float asFloat(T value) { return static_cast<float>(value); }
template <>
__device__ __forceinline__ float asFloat(__half value) { return __half2float(value); }

template <typename T>
__device__ __forceinline__ T fromFloat(float value) { return static_cast<T>(value); }
template <>
__device__ __forceinline__ __half fromFloat(float value) { return __float2half(value); }

template <typename Input, typename Output>
__global__ void gwcKernel(const Input* left, const Input* right, Output* output,
                          int batch, int channels, int height, int width,
                          int disparities, int groups, bool normalize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = blockIdx.z;
    if (x >= width || y >= height) return;

    const int disparity = index % disparities;
    const int group = (index / disparities) % groups;
    const int n = index / (disparities * groups);
    if (n >= batch) return;

    const int xr = x - disparity;
    const int out = ((n * groups + group) * disparities + disparity) *
                    height * width + y * width + x;
    if (xr < 0) {
        output[out] = fromFloat<Output>(0.0f);
        return;
    }

    const int group_channels = channels / groups;
    const int plane = height * width;
    int li = (n * channels + group * group_channels) * plane + y * width + x;
    int ri = (n * channels + group * group_channels) * plane + y * width + xr;
    float dot = 0.0f, left_norm = 0.0f, right_norm = 0.0f;
    for (int c = 0; c < group_channels; ++c, li += plane, ri += plane) {
        const float l = asFloat(left[li]);
        const float r = asFloat(right[ri]);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if (normalize) dot /= sqrtf(left_norm * right_norm) + 1e-5f;
    output[out] = fromFloat<Output>(dot);
}

__global__ void preprocessKernel(const uint8_t* bgr, float* chw, int src_h,
                                 int src_w, int dst_h, int dst_w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const int dst = y * dst_w + x;
    const int plane = dst_h * dst_w;
    if (x < src_w && y < src_h) {
        const int src = (y * src_w + x) * 3;
        chw[dst] = static_cast<float>(bgr[src + 2]);
        chw[plane + dst] = static_cast<float>(bgr[src + 1]);
        chw[2 * plane + dst] = static_cast<float>(bgr[src]);
    } else {
        chw[dst] = chw[plane + dst] = chw[2 * plane + dst] = 0.0f;
    }
}

__global__ void resizePadKernel(const uint8_t* bgr, float* chw, int src_h,
                                int src_w, int scaled_h, int scaled_w,
                                int dst_h, int dst_w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int px = min(x, scaled_w - 1);
    const int py = min(y, scaled_h - 1);
    const float source_x = (px + 0.5f) * src_w / scaled_w - 0.5f;
    const float source_y = (py + 0.5f) * src_h / scaled_h - 0.5f;
    const int raw_x0 = __float2int_rd(source_x);
    const int raw_y0 = __float2int_rd(source_y);
    const float wx = source_x - raw_x0;
    const float wy = source_y - raw_y0;
    const int x0 = max(0, min(raw_x0, src_w - 1));
    const int y0 = max(0, min(raw_y0, src_h - 1));
    const int x1 = max(0, min(raw_x0 + 1, src_w - 1));
    const int y1 = max(0, min(raw_y0 + 1, src_h - 1));
    const int dst = y * dst_w + x;
    const int plane = dst_h * dst_w;

    for (int c = 0; c < 3; ++c) {
        const int source_channel = 2 - c;
        const float v00 = bgr[(y0 * src_w + x0) * 3 + source_channel];
        const float v01 = bgr[(y0 * src_w + x1) * 3 + source_channel];
        const float v10 = bgr[(y1 * src_w + x0) * 3 + source_channel];
        const float v11 = bgr[(y1 * src_w + x1) * 3 + source_channel];
        chw[c * plane + dst] =
            (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) +
            wy * ((1.0f - wx) * v10 + wx * v11);
    }
}

__global__ void cropKernel(const float* source, float* output, int src_width,
                           int dst_height, int dst_width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dst_width && y < dst_height) {
        output[y * dst_width + x] = source[y * src_width + x];
    }
}

__global__ void upsampleKernel(const float* source, float* output, int src_w,
                               int src_h, int dst_w, int dst_h, float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    const int sx = min(static_cast<int>(static_cast<float>(x) * src_w / dst_w), src_w - 1);
    const int sy = min(static_cast<int>(static_cast<float>(y) * src_h / dst_h), src_h - 1);
    output[y * dst_w + x] = source[sy * src_w + sx] * scale;
}

__global__ void clampKernel(float* disparity, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) disparity[i] = fmaxf(disparity[i], 0.0f);
}

dim3 grid2d(int width, int height) {
    return dim3((width + 31) / 32, (height + 15) / 16);
}

}  // namespace

extern "C" void ffsGwcFloat(const float* left, const float* right, float* output,
                             int n, int c, int h, int w, int d, int g,
                             bool normalize, cudaStream_t stream) {
    gwcKernel<float, float><<<dim3((w + 15) / 16, (h + 15) / 16, n * g * d),
                              dim3(16, 16), 0, stream>>>(
        left, right, output, n, c, h, w, d, g, normalize);
}

extern "C" void ffsGwcHalf(const __half* left, const __half* right, __half* output,
                            int n, int c, int h, int w, int d, int g,
                            bool normalize, cudaStream_t stream) {
    gwcKernel<__half, __half><<<dim3((w + 15) / 16, (h + 15) / 16, n * g * d),
                               dim3(16, 16), 0, stream>>>(
        left, right, output, n, c, h, w, d, g, normalize);
}

extern "C" void ffsGwcFloatHalf(const float* left, const float* right,
                                 __half* output, int n, int c, int h, int w,
                                 int d, int g, bool normalize,
                                 cudaStream_t stream) {
    gwcKernel<float, __half><<<dim3((w + 15) / 16, (h + 15) / 16, n * g * d),
                               dim3(16, 16), 0, stream>>>(
        left, right, output, n, c, h, w, d, g, normalize);
}

extern "C" void ffsGwcHalfFloat(const __half* left, const __half* right,
                                 float* output, int n, int c, int h, int w,
                                 int d, int g, bool normalize,
                                 cudaStream_t stream) {
    gwcKernel<__half, float><<<dim3((w + 15) / 16, (h + 15) / 16, n * g * d),
                              dim3(16, 16), 0, stream>>>(
        left, right, output, n, c, h, w, d, g, normalize);
}

extern "C" void ffsPreprocessBgr(const uint8_t* input, float* output, int src_h,
                                  int src_w, int dst_h, int dst_w,
                                  cudaStream_t stream) {
    preprocessKernel<<<grid2d(dst_w, dst_h), dim3(32, 16), 0, stream>>>(
        input, output, src_h, src_w, dst_h, dst_w);
}

extern "C" void ffsResizePadBgr(const uint8_t* input, float* output, int src_h,
                                 int src_w, int scaled_h, int scaled_w,
                                 int dst_h, int dst_w, cudaStream_t stream) {
    resizePadKernel<<<grid2d(dst_w, dst_h), dim3(32, 16), 0, stream>>>(
        input, output, src_h, src_w, scaled_h, scaled_w, dst_h, dst_w);
}

extern "C" void ffsCropDisp(const float* input, float* output, int, int src_w,
                             int dst_h, int dst_w, cudaStream_t stream) {
    cropKernel<<<grid2d(dst_w, dst_h), dim3(32, 16), 0, stream>>>(
        input, output, src_w, dst_h, dst_w);
}

extern "C" void ffsUpsampleDisp(const float* input, float* output, int src_w,
                                 int src_h, int dst_w, int dst_h, float scale,
                                 cudaStream_t stream) {
    upsampleKernel<<<grid2d(dst_w, dst_h), dim3(32, 16), 0, stream>>>(
        input, output, src_w, src_h, dst_w, dst_h, scale);
}

extern "C" void ffsClampDisp(float* disparity, int count, cudaStream_t stream) {
    clampKernel<<<(count + 255) / 256, 256, 0, stream>>>(disparity, count);
}
