#include "ffs_infer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <ffs.yaml> <left.png> <right.png> [disp_vis.png]\n";
        return 1;
    }

    try {
        cv::Mat left = cv::imread(argv[2], cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(argv[3], cv::IMREAD_COLOR);
        if (left.empty() || right.empty()) {
            throw std::runtime_error("failed to load input images");
        }
        if (left.size() != right.size() || left.type() != CV_8UC3 ||
            right.type() != CV_8UC3) {
            throw std::runtime_error("inputs must be same-sized 8-bit BGR images");
        }

        ffs_infer::FFSInfer model(argv[1]);
        std::vector<float> disparity;

        // Warm-up before timing.
        model.infer(left.data, right.data, left.cols, left.rows, disparity,
                    static_cast<int>(left.step));

        const auto begin = std::chrono::steady_clock::now();
        model.infer(left.data, right.data, left.cols, left.rows, disparity,
                    static_cast<int>(left.step));
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "disparity=" << left.cols << "x" << left.rows
                  << ", latency=" << ms << " ms\n";

        cv::Mat disp(left.rows, left.cols, CV_32FC1, disparity.data());
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(disp, &min_value, &max_value);
        cv::Mat disp8;
        const double range = std::max(1e-6, max_value - min_value);
        disp.convertTo(disp8, CV_8UC1, 255.0 / range, -255.0 * min_value / range);
        cv::Mat color;
        cv::applyColorMap(disp8, color, cv::COLORMAP_TURBO);
        const std::string output = argc == 5 ? argv[4] : "disp_vis.png";
        if (!cv::imwrite(output, color)) {
            throw std::runtime_error("failed to write " + output);
        }
        std::cout << "saved " << output << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
