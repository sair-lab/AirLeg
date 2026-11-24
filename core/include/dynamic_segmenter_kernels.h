#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// GPU stages of DynamicSegmenter (YOLO11n-seg: output0 [4 + classes + 32, anchors], output1 [32, ph, pw]).
namespace yolo_cuda {
// Grey image (rw x rh, already resized) -> network input [3, ih, iw] in [0, 1], padded right / bottom with 114 / 255.
void LaunchLetterbox(const uint8_t* gray, int rw, int rh, float* input, int iw, int ih, cudaStream_t stream);
// One thread per anchor: best class over all classes; kept when its score >= confidence and the class is selected
// (class_mask bit). Writes (x1, y1, x2, y2, score, class, anchor) in input pixels; *count = total found (may exceed
// capacity: only the first `capacity` are written).
void LaunchDecode(const float* output0, int anchors, int classes, uint64_t class_mask_lo, uint64_t class_mask_hi,
                  float confidence, float* candidates, int* count, int capacity, cudaStream_t stream);
// Mask probability of each detection at the prototype resolution inside its region of interest (its box grown by a
// margin, not cropped: the crop happens at the image resolution after the upsampling, as Ultralytics
// process_mask_native): one block per detection; rois: n x (x0, y0, w, h, offset, anchor) in prototype cells; out[offset
// + y * w + x] = sigmoid(coefficients . prototypes(x0 + x, y0 + y)); the 32 coefficients of a detection are output0 rows
// 4 + classes .. at its anchor.
void LaunchMaskRois(const float* output0, int anchors, int classes, const float* protos, int ph, int pw,
                    const int* rois, int n, float* out, cudaStream_t stream);
}  // namespace yolo_cuda
