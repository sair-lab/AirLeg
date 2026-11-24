#include "feature_detector.h"

#include <future>
#include <iostream>
#include <stdexcept>

// Two instances (own execution context, stream and buffers each) so that the left and right images of a
// stereo pair are detected concurrently.
FeatureDetector::FeatureDetector(const SuperPointConfig& config) {
  _superpoint = std::make_shared<SuperPoint>(config);
  _superpoint_right = std::make_shared<SuperPoint>(config);
  if (!_superpoint->build() || !_superpoint_right->build()) {
    throw std::runtime_error("failed to build SuperPoint feature detector");
  }
}

bool FeatureDetector::Detect(
    cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic>& features) {
  const bool ok = _superpoint->infer(image, features);
  if (!ok) std::cerr << "Failed when extracting point features!" << std::endl;
  return ok;
}

bool FeatureDetector::Detect(
    cv::Mat& image_left, cv::Mat& image_right,
    Eigen::Matrix<float, 259, Eigen::Dynamic>& left_features,
    Eigen::Matrix<float, 259, Eigen::Dynamic>& right_features) {
  auto right = std::async(std::launch::async, [&] { return _superpoint_right->infer(image_right, right_features); });
  const bool left_ok = _superpoint->infer(image_left, left_features);
  const bool right_ok = right.get();
  if (!left_ok || !right_ok) std::cerr << "Failed when extracting point features!" << std::endl;
  return left_ok && right_ok;
}
