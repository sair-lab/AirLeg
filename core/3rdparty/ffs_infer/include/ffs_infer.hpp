#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ffs_infer {

// Lightweight Fast-FoundationStereo TensorRT inference interface.
// The disparity is left-referenced and expressed in input-image pixels.
class FFSInfer {
public:
    explicit FFSInfer(const std::string& yaml_path);
    ~FFSInfer();

    FFSInfer(const FFSInfer&) = delete;
    FFSInfer& operator=(const FFSInfer&) = delete;
    FFSInfer(FFSInfer&&) noexcept;
    FFSInfer& operator=(FFSInfer&&) noexcept;

    // Simple synchronous CPU API. Images are interleaved uint8 BGR.
    // row_stride_bytes may be 0 for tightly packed images.
    void infer(const uint8_t* left_bgr,
               const uint8_t* right_bgr,
               int width,
               int height,
               std::vector<float>& disparity,
               int row_stride_bytes = 0);

    // Fast asynchronous GPU API. Inputs are tightly packed HWC uint8 BGR and
    // output is HxW float32, all in device memory. Work is queued on stream().
    void inferDevice(const uint8_t* d_left_bgr,
                     const uint8_t* d_right_bgr,
                     int width,
                     int height,
                     float* d_disparity);

    void synchronize();
    cudaStream_t stream() const noexcept;
    int modelWidth() const noexcept;
    int modelHeight() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ffs_infer
