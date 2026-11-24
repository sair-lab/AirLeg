#ifndef FEATURE_DETECTOR_H_
#define FEATURE_DETECTOR_H_

#include "super_point.h"
#include "read_configs.h"

class FeatureDetector{
public:
  FeatureDetector(const SuperPointConfig& superpoint_config);

  bool Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
  bool Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features);

private:
  SuperPointPtr _superpoint, _superpoint_right;
};

typedef std::shared_ptr<FeatureDetector> FeatureDetectorPtr;

#endif  // FEATURE_DETECTOR_H_ 
