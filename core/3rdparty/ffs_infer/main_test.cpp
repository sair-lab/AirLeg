#include "ffs_infer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultSequence =
    "/path/to/go2_data_20260810/seq/"
    "20260810/grass_seq1_slope_flat_20260810";

struct StereoCalibration {
    float fx = 0.0f;
    float baseline_m = 0.0f;
    int width = 0;
    int height = 0;
};

StereoCalibration loadCalibration(const fs::path& path) {
    try {
        const YAML::Node storage = YAML::LoadFile(path.string());
        StereoCalibration result;
        const YAML::Node cam0 = storage["cam0"];
        const YAML::Node intrinsics = cam0["k"];
        if (!intrinsics.IsSequence() || intrinsics.size() != 9) {
            throw std::runtime_error("camera.yaml does not contain cam0.k[9]");
        }
        result.fx = intrinsics[0].as<float>();
        result.width = cam0["width"].as<int>();
        result.height = cam0["height"].as<int>();
        result.baseline_m = storage["stereo"]["baseline_m"].as<float>();
        if (!std::isfinite(result.fx) || !std::isfinite(result.baseline_m) ||
            !(result.fx > 0.0f) || !(result.baseline_m > 0.0f) ||
            result.width <= 0 || result.height <= 0) {
            throw std::runtime_error("invalid fx, baseline or image size in camera.yaml");
        }
        return result;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("cannot read camera calibration " + path.string() +
                                 ": " + e.what());
    }
}

std::vector<fs::path> findStereoPairs(const fs::path& left_dir,
                                      const fs::path& right_dir) {
    if (!fs::is_directory(left_dir) || !fs::is_directory(right_dir)) {
        throw std::runtime_error("cam0/data or cam1/data is missing");
    }
    std::vector<fs::path> pairs;
    for (const auto& entry : fs::directory_iterator(left_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if ((extension == ".png" || extension == ".jpg" || extension == ".jpeg") &&
            fs::is_regular_file(right_dir / entry.path().filename())) {
            pairs.push_back(entry.path());
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });
    if (pairs.empty()) throw std::runtime_error("no matching stereo image pairs found");
    return pairs;
}

cv::Mat loadBgr(const fs::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("cannot read image: " + path.string());
    if (image.depth() != CV_8U) {
        throw std::runtime_error("only uint8 input images are supported: " + path.string());
    }
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    } else if (image.channels() != 3) {
        throw std::runtime_error("unsupported image channels: " + path.string());
    }
    return image;
}

// uint16 centimeters: 0 is invalid, 1..65535 represent 0.01..655.35 m.
constexpr double kDepthUnitsPerMeter = 100.0;
constexpr double kMaxPngDepthMeters = 65535.0 / kDepthUnitsPerMeter;

void writeDepthPng(const fs::path& path, const std::vector<float>& depth,
                   int height, int width) {
    cv::Mat image(height, width, CV_16UC1);
    for (int y = 0; y < height; ++y) {
        auto* row = image.ptr<uint16_t>(y);
        for (int x = 0; x < width; ++x) {
            const float z = depth[static_cast<size_t>(y) * width + x];
            row[x] = std::isfinite(z) && z > 0.0f
                         ? static_cast<uint16_t>(std::clamp(
                               std::lround(z * kDepthUnitsPerMeter), 1L, 65535L))
                         : 0;
        }
    }
    if (!cv::imwrite(path.string(), image, {cv::IMWRITE_PNG_COMPRESSION, 1})) {
        throw std::runtime_error("failed writing depth PNG: " + path.string());
    }
}

void writeDepthColorPng(const fs::path& path, const std::vector<float>& depth,
                        int height, int width, float max_depth_m) {
    cv::Mat indices(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat invalid(height, width, CV_8UC1, cv::Scalar(255));
    // Fixed log-depth scale improves near-field contrast without changing
    // the meaning of a color between frames. Near = blue, far = red.
    const double scale = std::log1p(static_cast<double>(max_depth_m));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float z = depth[static_cast<size_t>(y) * width + x];
            if (std::isfinite(z) && z > 0.0f && z <= max_depth_m) {
                const double t = std::clamp(std::log1p(static_cast<double>(z)) / scale,
                                            0.0, 1.0);
                indices.at<uint8_t>(y, x) = cv::saturate_cast<uint8_t>(255.0 * t);
                invalid.at<uint8_t>(y, x) = 0;
            }
        }
    }
    cv::Mat color;
    cv::applyColorMap(indices, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), invalid);
    if (!cv::imwrite(path.string(), color, {cv::IMWRITE_PNG_COMPRESSION, 1})) {
        throw std::runtime_error("failed writing color depth PNG: " + path.string());
    }
}

void disparityToDepth(const std::vector<float>& disparity, int width, int height,
                      float fx, float baseline_m, float max_depth_m,
                      std::vector<float>& depth) {
    depth.resize(disparity.size());
    const float numerator = fx * baseline_m;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * width + x;
            const float d = disparity[index];
            float z = 0.0f;
            // x-d < 0 means the left-reference point is outside the right image.
            if (std::isfinite(d) && d > 1e-3f && x - d >= 0.0f) {
                z = numerator / d;
                if (!std::isfinite(z) || z <= 0.0f || z > max_depth_m) z = 0.0f;
            }
            depth[index] = z;
        }
    }
}

void writeMetadata(const fs::path& output_dir, const StereoCalibration& calibration,
                   float max_depth_m, size_t pair_count) {
    std::ofstream output(output_dir / "metadata.txt");
    if (!output) throw std::runtime_error("cannot create depth metadata");
    output << std::setprecision(10)
           << "format=png_uint16\n"
           << "unit=centimeter\n"
           << "depth_meters_per_unit=0.01\n"
           << "visualization_directory=../depth_ffs_color\n"
           << "visualization_colormap=TURBO\n"
           << "visualization_mapping=255*log(1+depth_m)/log(1+max_depth_m)\n"
           << "visualization_invalid_color=black\n"
           << "processing=one_stereo_pair_at_a_time\n"
           << "reference_camera=cam0_left\n"
           << "invalid_depth=0\n"
           << "fx=" << calibration.fx << '\n'
           << "baseline_m=" << calibration.baseline_m << '\n'
           << "width=" << calibration.width << '\n'
           << "height=" << calibration.height << '\n'
           << "max_depth_m=" << max_depth_m << '\n'
           << "pair_count=" << pair_count << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [ffs.yaml] [sequence_dir] [max_depth_m]\n";
        return 1;
    }

    const fs::path ffs_config = argc >= 2 ? argv[1] : "ffs.yaml";
    const fs::path sequence_dir = argc >= 3 ? argv[2] : kDefaultSequence;

    try {
        const auto total_begin = std::chrono::steady_clock::now();
        const float max_depth_m = argc >= 4 ? std::stof(argv[3]) : 100.0f;
        if (!std::isfinite(max_depth_m) || !(max_depth_m > 0.0f) ||
            max_depth_m > static_cast<float>(kMaxPngDepthMeters)) {
            throw std::invalid_argument("max_depth_m must be in (0, 655.35] for uint16 centimeter PNG");
        }
        const auto calibration = loadCalibration(sequence_dir / "camera.yaml");
        const fs::path left_dir = sequence_dir / "cam0" / "data";
        const fs::path right_dir = sequence_dir / "cam1" / "data";
        const fs::path output_dir = sequence_dir / "depth_ffs";
        const fs::path color_dir = sequence_dir / "depth_ffs_color";
        const auto pairs = findStereoPairs(left_dir, right_dir);
        fs::create_directories(output_dir);
        fs::create_directories(color_dir);
        writeMetadata(output_dir, calibration, max_depth_m, pairs.size());

        std::cout << "Loading Fast-FoundationStereo engine...\n";
        ffs_infer::FFSInfer model(ffs_config.string());
        std::cout << "Pairs: " << pairs.size() << "\n"
                  << "Calibration: fx=" << calibration.fx
                  << ", baseline=" << calibration.baseline_m << " m\n"
                  << "Output: " << output_dir << "\n"
                  << "Color visualization: " << color_dir << " (near=blue, far=red, invalid=black)\n";

        std::vector<float> disparity;
        std::vector<float> depth;
        double image_read_ms_sum = 0.0;
        double inference_ms_sum = 0.0;
        double depth_convert_ms_sum = 0.0;
        double depth_write_ms_sum = 0.0;
        size_t completed = 0;
        const auto processing_begin = std::chrono::steady_clock::now();

        // Keep one stereo pair in memory. infer() is synchronous; finish saving
        // this frame before loading the next pair. The pairs list contains paths only.
        for (const fs::path& left_path : pairs) {
            const fs::path right_path = right_dir / left_path.filename();

            const auto read_begin = std::chrono::steady_clock::now();
            cv::Mat left = loadBgr(left_path);
            cv::Mat right = loadBgr(right_path);
            const auto read_end = std::chrono::steady_clock::now();
            image_read_ms_sum +=
                std::chrono::duration<double, std::milli>(read_end - read_begin).count();

            if (left.size() != right.size()) {
                throw std::runtime_error("stereo image size mismatch: " +
                                         left_path.filename().string());
            }
            if (left.cols != calibration.width || left.rows != calibration.height) {
                throw std::runtime_error("image size does not match camera.yaml: " +
                                         left_path.filename().string());
            }

            const auto inference_begin = std::chrono::steady_clock::now();
            model.infer(left.data, right.data, left.cols, left.rows, disparity,
                        static_cast<int>(left.step));
            const auto inference_end = std::chrono::steady_clock::now();
            inference_ms_sum +=
                std::chrono::duration<double, std::milli>(inference_end - inference_begin).count();

            const auto convert_begin = std::chrono::steady_clock::now();
            disparityToDepth(disparity, left.cols, left.rows, calibration.fx,
                             calibration.baseline_m, max_depth_m, depth);
            const auto convert_end = std::chrono::steady_clock::now();
            depth_convert_ms_sum +=
                std::chrono::duration<double, std::milli>(convert_end - convert_begin).count();

            const auto write_begin = std::chrono::steady_clock::now();
            const fs::path depth_path = output_dir / (left_path.stem().string() + ".png");
            writeDepthPng(depth_path, depth, left.rows, left.cols);
            writeDepthColorPng(color_dir / depth_path.filename(), depth,
                               left.rows, left.cols, max_depth_m);
            const auto write_end = std::chrono::steady_clock::now();
            depth_write_ms_sum +=
                std::chrono::duration<double, std::milli>(write_end - write_begin).count();

            ++completed;
            std::cout << "Processed " << completed << '/' << pairs.size()
                      << ", saved depth + color " << depth_path.filename().string()
                      << ", inference " << std::fixed << std::setprecision(2)
                      << std::chrono::duration<double, std::milli>(
                             inference_end - inference_begin).count()
                      << " ms" << std::endl;
        }
        const auto processing_end = std::chrono::steady_clock::now();
        const auto total_end = processing_end;
        const double processing_ms =
            std::chrono::duration<double, std::milli>(processing_end - processing_begin).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(total_end - total_begin).count();
        const double count = static_cast<double>(completed);

        std::cout << "\n\nTiming summary\n"
                  << "  Frames             : " << completed << '\n'
                  << "  Image read         : " << image_read_ms_sum / count << " ms/frame\n"
                  << "  FFS inference      : " << inference_ms_sum / count << " ms/frame"
                  << " (" << 1000.0 * count / inference_ms_sum << " FPS)\n"
                  << "  Disparity -> depth : " << depth_convert_ms_sum / count << " ms/frame\n"
                  << "  Depth write        : " << depth_write_ms_sum / count << " ms/frame\n"
                  << "  Processing total   : " << processing_ms / 1000.0 << " s, "
                  << processing_ms / count << " ms/frame"
                  << " (" << 1000.0 * count / processing_ms << " FPS)\n"
                  << "  Program total      : " << total_ms / 1000.0 << " s\n"
                  << "Done. Depth PNG unit is centimeters (meters = pixel / 100); invalid pixels are 0.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
