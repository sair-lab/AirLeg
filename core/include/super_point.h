//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "read_configs.h"

class SuperPoint {
public:
    explicit SuperPoint(const SuperPointConfig &super_point_config);
    ~SuperPoint();

    bool build();

    bool infer(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif //SUPER_POINT_H_
