#ifndef POINT_MATCHING_H_
#define POINT_MATCHING_H_

#include <mutex>

#include "light_glue.h"
#include "read_configs.h"

class PointMatcher{
public:
  PointMatcher(const PointMatcherConfig& _config);

  void NormalizeKeypoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic>& normalized_features, 
      int width, int height, float scale);

  int MatchingPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features0, 
      const Eigen::Matrix<float, 259, Eigen::Dynamic>& features1, 
      std::vector<cv::DMatch>& matches,  bool outlier_rejection=false);

private:
  PointMatcherConfig _config;
  SuperPointLightGluePtr _lightglue;
  // one TensorRT execution context: the feature thread and the tracking thread (TryRelink after a tracking loss) both match
  std::mutex _infer_mutex;
};


typedef std::shared_ptr<PointMatcher> PointMatcherPtr;

#endif  // POINT_MATCHING_H_ 
