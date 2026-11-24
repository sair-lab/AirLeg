#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "read_configs.h"

// Online dynamic-object masks: YOLO11n-seg instance segmentation (TensorRT FP16) of the rectified left image at a
// reduced input size (DynamicMaskConfig::input_size long side, letterboxed to multiples of 32). All per-pixel work runs
// on the GPU (letterbox + normalisation, candidate decoding, mask-coefficient x prototype products at 1/4 input
// resolution); the CPU only sorts the few candidates (NMS) and upsamples / dilates the mask inside the detection boxes.
// Launch() is asynchronous (own CUDA stream) so that the network runs while SuperPoint extracts features; Collect()
// waits for it and returns the mask (255 = dynamic, image size), or an empty Mat when nothing dynamic is detected.
class DynamicSegmenter {
public:
  DynamicSegmenter(const DynamicMaskConfig& config, int image_width, int image_height);
  ~DynamicSegmenter();
  DynamicSegmenter(const DynamicSegmenter&) = delete;
  DynamicSegmenter& operator=(const DynamicSegmenter&) = delete;

  void Launch(const cv::Mat& gray);   // CV_8UC1, image size
  cv::Mat Collect();                  // after Launch(); empty Mat = no dynamic object
  int InputWidth() const;
  int InputHeight() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
using DynamicSegmenterPtr = std::shared_ptr<DynamicSegmenter>;
