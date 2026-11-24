//
// Created by haoyuefan on 2021/9/22.
//

#ifndef LIGHT_GLUE_H_
#define LIGHT_GLUE_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "read_configs.h"

class SuperPointLightGlue {
public:
    SuperPointLightGlue();
    explicit SuperPointLightGlue(const PointMatcherConfig &lightglue_config);
    ~SuperPointLightGlue();

    bool build();

    bool infer(const Eigen::Matrix<float, 258, Eigen::Dynamic> &features0,
               const Eigen::Matrix<float, 258, Eigen::Dynamic> &features1,
               Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index,
               Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

typedef std::shared_ptr<SuperPointLightGlue> SuperPointLightGluePtr;

#endif //LIGHT_GLUE_H_
