#include "super_point.h"

#include "cuda_utils.h"
#include "super_point_kernels.h"
#include "trt_engine.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool Contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

size_t Volume(const nvinfer1::Dims& dims) {
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return 0;
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
}

template <typename Coordinate>
std::vector<int> SelectKeypointIndexes(const std::vector<float>& scores,
                                       const std::vector<Coordinate>& xs,
                                       const std::vector<Coordinate>& ys,
                                       int height, int width,
                                       const SuperPointConfig& config) {
    if (!config.grid_filter) {
        std::vector<int> selected;
        selected.reserve(scores.size());
        for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
            if (scores[i] >= config.keypoint_threshold) selected.push_back(i);
        }
        std::sort(selected.begin(), selected.end(),
                  [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
        if (selected.size() > static_cast<size_t>(config.max_keypoints)) {
            selected.resize(config.max_keypoints);
        }
        return selected;
    }

    const int patch_count = config.grid_cols * config.grid_rows;
    const int patch_max = config.max_keypoints / patch_count;
    std::vector<std::vector<int>> patches(patch_count);
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        if (scores[i] < config.l_thr) continue;
        const int patch_x = std::clamp(
            static_cast<int>(xs[i] * config.grid_cols / width), 0, config.grid_cols - 1);
        const int patch_y = std::clamp(
            static_cast<int>(ys[i] * config.grid_rows / height), 0, config.grid_rows - 1);
        patches[patch_y * config.grid_cols + patch_x].push_back(i);
    }

    size_t selected_count = 0;
    for (auto& patch : patches) {
        std::sort(patch.begin(), patch.end(),
                  [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
        const auto first_low = std::find_if(
            patch.begin(), patch.end(),
            [&scores, &config](int index) { return scores[index] < config.h_thr; });
        const size_t high_count = std::distance(patch.begin(), first_low);
        const size_t keep_count = high_count < static_cast<size_t>(patch_max)
            ? std::min(patch.size(), static_cast<size_t>(patch_max))
            : high_count;
        patch.resize(keep_count);
        selected_count += keep_count;
    }

    if (selected_count > static_cast<size_t>(config.max_keypoints)) {
        std::vector<int> patch_order(patch_count);
        std::iota(patch_order.begin(), patch_order.end(), 0);
        std::sort(patch_order.begin(), patch_order.end(),
                  [&patches](int lhs, int rhs) {
                      return patches[lhs].size() > patches[rhs].size();
                  });
        for (int patch_index : patch_order) {
            auto& patch = patches[patch_index];
            if (selected_count <= static_cast<size_t>(config.max_keypoints)) break;
            if (patch.size() <= static_cast<size_t>(patch_max)) continue;
            selected_count -= patch.size() - patch_max;
            patch.resize(patch_max);
        }
    }

    std::vector<int> selected;
    selected.reserve(selected_count);
    for (const auto& patch : patches) {
        selected.insert(selected.end(), patch.begin(), patch.end());
    }
    std::sort(selected.begin(), selected.end(),
              [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
    return selected;
}

int Clip(int value, int max_value) {
    return std::max(0, std::min(value, max_value - 1));
}

void DetectPoint(const float* heat_map, Eigen::Matrix<float, 259, Eigen::Dynamic>& features,
                 int height, int width, int border, const SuperPointConfig& config) {
    std::vector<float> scores;
    std::vector<int> ys;
    std::vector<int> xs;
    scores.reserve(height * width / 16);
    ys.reserve(height * width / 16);
    xs.reserve(height * width / 16);

    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            const float score = heat_map[y * width + x];
            const float candidate_threshold = config.grid_filter
                ? config.l_thr : config.keypoint_threshold;
            if (score < candidate_threshold) continue;

            bool is_max = true;
            for (int yy = y - 4; yy <= y + 4 && is_max; ++yy) {
                for (int xx = x - 4; xx <= x + 4; ++xx) {
                    if (yy == y && xx == x) continue;
                    if (heat_map[yy * width + xx] > score) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (!is_max) continue;

            scores.push_back(score);
            ys.push_back(y);
            xs.push_back(x);
        }
    }

    const std::vector<int> selected = SelectKeypointIndexes(
        scores, xs, ys, height, width, config);
    const int count = selected.size();
    features.resize(259, count);
    for (int i = 0; i < count; ++i) {
        const int idx = selected[i];
        features(0, i) = scores[idx];
        features(1, i) = static_cast<float>(xs[idx]);
        features(2, i) = static_cast<float>(ys[idx]);
    }
}

void ExtractDescriptors(const float* descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic>& features,
                        int height, int width, int stride, bool official = false) {
    const int descriptor_height = height / stride;
    const int descriptor_width = width / stride;
    const float s = static_cast<float>(stride), h = 0.5f * s - 0.5f;

    for (int col = 0; col < features.cols(); ++col) {
        // official SuperPoint cell centres: (p - s/2 + 0.5) / (w s - s/2 - 0.5) (w - 1), ~(p - 3.5) / 8
        const float x = official ? (features(1, col) - h) / (descriptor_width * s - h - 1.0f) * (descriptor_width - 1)
                                 : features(1, col) / s;
        const float y = official ? (features(2, col) - h) / (descriptor_height * s - h - 1.0f) * (descriptor_height - 1)
                                 : features(2, col) / s;
        const int x0 = Clip(static_cast<int>(std::floor(x)), descriptor_width);
        const int y0 = Clip(static_cast<int>(std::floor(y)), descriptor_height);
        const int x1 = Clip(x0 + 1, descriptor_width);
        const int y1 = Clip(y0 + 1, descriptor_height);
        const float dx = x - static_cast<float>(x0);
        const float dy = y - static_cast<float>(y0);

        float norm = 0.0f;
        for (int channel = 0; channel < 256; ++channel) {
            const float d00 = descriptors[channel * descriptor_height * descriptor_width + y0 * descriptor_width + x0];
            const float d01 = descriptors[channel * descriptor_height * descriptor_width + y0 * descriptor_width + x1];
            const float d10 = descriptors[channel * descriptor_height * descriptor_width + y1 * descriptor_width + x0];
            const float d11 = descriptors[channel * descriptor_height * descriptor_width + y1 * descriptor_width + x1];
            const float value = (1.0f - dx) * (1.0f - dy) * d00 +
                                dx * (1.0f - dy) * d01 +
                                (1.0f - dx) * dy * d10 +
                                dx * dy * d11;
            features(channel + 3, col) = value;
            norm += value * value;
        }

        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (int channel = 0; channel < 256; ++channel) {
                features(channel + 3, col) /= norm;
            }
        }
    }
}

}  // namespace

struct SuperPoint::Impl {
    explicit Impl(const SuperPointConfig& cfg) : config(cfg) {}

    SuperPointConfig config;
    lightglue::TRTEngine engine;
    lightglue::TrtUniquePtr<nvinfer1::IExecutionContext> context;

    int resized_width = 512;
    int resized_height = 512;
    std::string input_name;
    std::string keypoints_name;
    std::string scores_name;
    std::string descriptors_name;
    bool topk_output = false;
    // persistent stream, device buffers and pinned host memory (no per-frame allocation)
    lightglue::CudaStream stream;
    lightglue::CudaBuffer<float> d_input, d_keypoints, d_scores, d_descriptors, d_candidate_scores, d_xy, d_features;
    lightglue::CudaBuffer<int> d_count, d_candidate_index;
    lightglue::PinnedBuffer<float> h_input, h_scores, h_descriptors, h_candidate_scores, h_xy, h_features;
    lightglue::PinnedBuffer<int> h_count, h_candidate_index;
    // CPU reference post-processing (copies the dense maps back), for verification
    bool cpu_post = std::getenv("AIRSLAM_SP_CPU_POST") != nullptr;
};

SuperPoint::SuperPoint(const SuperPointConfig& super_point_config)
    : impl_(std::make_unique<Impl>(super_point_config)) {}

SuperPoint::~SuperPoint() = default;

bool SuperPoint::build() {
    if (impl_->config.grid_filter) {
        std::cout << "SuperPoint grid filter: " << impl_->config.grid_cols << 'x'
                  << impl_->config.grid_rows << ", h_thr=" << impl_->config.h_thr
                  << ", l_thr=" << impl_->config.l_thr << ", patch_max="
                  << impl_->config.max_keypoints /
                         (impl_->config.grid_cols * impl_->config.grid_rows)
                  << std::endl;
    }
    lightglue::EngineConfig engine_config;
    engine_config.onnxPath = impl_->config.onnx_file;
    engine_config.enginePath = impl_->config.engine_file;
    engine_config.fp16 = true;

    lightglue::ProfileDims image_dims;
    image_dims.min = nvinfer1::Dims4{1, 1, 100, 100};
    image_dims.opt = nvinfer1::Dims4{1, 1, 500, 500};
    image_dims.max = nvinfer1::Dims4{1, 1, 1500, 1500};
    engine_config.profiles["input"] = image_dims;
    engine_config.profiles["images"] = image_dims;
    if (!impl_->config.input_tensor_names.empty()) {
        engine_config.profiles[impl_->config.input_tensor_names[0]] = image_dims;
    }

    if (!impl_->engine.build(engine_config)) return false;
    if (impl_->engine.inputNames().empty()) return false;
    impl_->input_name = impl_->engine.inputNames().front();

    for (const auto& name : impl_->engine.outputNames()) {
        if (Contains(name, "keypoints") || Contains(name, "kpts")) {
            impl_->keypoints_name = name;
        } else if (Contains(name, "scores")) {
            impl_->scores_name = name;
        } else if (Contains(name, "descriptors") || Contains(name, "desc")) {
            impl_->descriptors_name = name;
        }
    }

    impl_->topk_output = !impl_->keypoints_name.empty();
    if (impl_->scores_name.empty() || impl_->descriptors_name.empty()) {
        std::cerr << "SuperPoint: unsupported ONNX outputs. Need scores/descriptors, optionally keypoints." << std::endl;
        return false;
    }

    impl_->context.reset(impl_->engine.createContext());
    return impl_->context != nullptr;
}

bool SuperPoint::infer(const cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic>& features) {
    if (!impl_->context) return false;
    if (image.empty()) {
        features.resize(259, 0);
        return true;
    }

    try {
        impl_->resized_width = impl_->config.input_width > 0 ? impl_->config.input_width : image.cols;
        impl_->resized_height = impl_->config.input_height > 0 ? impl_->config.input_height : image.rows;
        const float width_scale = static_cast<float>(image.cols) / static_cast<float>(impl_->resized_width);
        const float height_scale = static_cast<float>(image.rows) / static_cast<float>(impl_->resized_height);

        cv::Mat resized = image;
        if (resized.cols != impl_->resized_width || resized.rows != impl_->resized_height)
            cv::resize(image, resized, cv::Size(impl_->resized_width, impl_->resized_height));

        cv::Mat gray;
        if (resized.channels() == 3) {
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = resized;
        }

        cv::Mat input_float;
        gray.convertTo(input_float, CV_32FC1, 1.0 / 255.0);
        if (!input_float.isContinuous()) input_float = input_float.clone();

        Impl& m = *impl_;
        const cudaStream_t stream = m.stream.get();
        m.h_input.reserve(input_float.total());
        std::copy_n(reinterpret_cast<const float*>(input_float.data), input_float.total(), m.h_input.get());
        m.d_input.copyFromHost(m.h_input.get(), input_float.total(), stream);
        lightglue::CudaBuffer<float>& input = m.d_input;

        const auto input_dims = nvinfer1::Dims4{1, 1, input_float.rows, input_float.cols};
        if (!impl_->context->setInputShape(impl_->input_name.c_str(), input_dims)) return false;
        impl_->context->setTensorAddress(impl_->input_name.c_str(), input.get());

        if (impl_->topk_output) {
            const auto keypoint_dims = impl_->context->getTensorShape(impl_->keypoints_name.c_str());
            const auto score_dims = impl_->context->getTensorShape(impl_->scores_name.c_str());
            const auto descriptor_dims = impl_->context->getTensorShape(impl_->descriptors_name.c_str());
            const size_t keypoint_values = Volume(keypoint_dims);
            const size_t score_values = Volume(score_dims);
            const size_t descriptor_values = Volume(descriptor_dims);
            if (keypoint_values == 0 || score_values == 0 || descriptor_values == 0) return false;
            const int candidate_count = std::min<int>(
                static_cast<int>(keypoint_values / 2),
                std::min<int>(static_cast<int>(score_values), static_cast<int>(descriptor_values / 256)));

            std::vector<float> h_keypoints(keypoint_values);
            std::vector<float> h_scores(score_values);
            std::vector<float> h_descriptors(descriptor_values);
            lightglue::CudaBuffer<float>& d_keypoints = m.d_keypoints;
            lightglue::CudaBuffer<float>& d_scores = m.d_scores;
            lightglue::CudaBuffer<float>& d_descriptors = m.d_descriptors;
            d_keypoints.resize(h_keypoints.size());
            d_scores.resize(h_scores.size());
            d_descriptors.resize(h_descriptors.size());

            impl_->context->setTensorAddress(impl_->keypoints_name.c_str(), d_keypoints.get());
            impl_->context->setTensorAddress(impl_->scores_name.c_str(), d_scores.get());
            impl_->context->setTensorAddress(impl_->descriptors_name.c_str(), d_descriptors.get());

            if (!impl_->context->enqueueV3(stream)) return false;
            d_keypoints.copyToHost(h_keypoints.data(), h_keypoints.size(), stream);
            d_scores.copyToHost(h_scores.data(), h_scores.size(), stream);
            d_descriptors.copyToHost(h_descriptors.data(), h_descriptors.size(), stream);
            m.stream.synchronize();

            std::vector<float> candidate_scores(candidate_count);
            std::vector<float> candidate_xs(candidate_count);
            std::vector<float> candidate_ys(candidate_count);
            for (int i = 0; i < candidate_count; ++i) {
                candidate_scores[i] = h_scores[i];
                candidate_xs[i] = h_keypoints[i * 2];
                candidate_ys[i] = h_keypoints[i * 2 + 1];
            }
            const std::vector<int> selected_indices = SelectKeypointIndexes(
                candidate_scores, candidate_xs, candidate_ys,
                input_float.rows, input_float.cols, impl_->config);
            const int feature_count = selected_indices.size();

            features.resize(259, feature_count);
            for (int col = 0; col < feature_count; ++col) {
                const int source_col = selected_indices[col];
                features(0, col) = h_scores[source_col];
                features(1, col) = h_keypoints[source_col * 2] * width_scale;
                features(2, col) = h_keypoints[source_col * 2 + 1] * height_scale;
                for (int dim = 0; dim < 256; ++dim) {
                    features(dim + 3, col) = h_descriptors[source_col * 256 + dim];
                }
            }
        } else {
            const int score_count = input_float.rows * input_float.cols;
            const int descriptor_height = input_float.rows / 8;
            const int descriptor_width = input_float.cols / 8;
            const int descriptor_count = 256 * descriptor_height * descriptor_width;

            m.d_scores.resize(score_count);
            m.d_descriptors.resize(descriptor_count);
            impl_->context->setTensorAddress(impl_->scores_name.c_str(), m.d_scores.get());
            impl_->context->setTensorAddress(impl_->descriptors_name.c_str(), m.d_descriptors.get());
            if (!impl_->context->enqueueV3(stream)) return false;

            const int height = input_float.rows, width = input_float.cols;
            const float candidate_threshold = impl_->config.grid_filter
                ? impl_->config.l_thr : impl_->config.keypoint_threshold;
            if (m.cpu_post) {
                m.h_scores.reserve(score_count);
                m.h_descriptors.reserve(descriptor_count);
                m.d_scores.copyToHost(m.h_scores.get(), score_count, stream);
                m.d_descriptors.copyToHost(m.h_descriptors.get(), descriptor_count, stream);
                m.stream.synchronize();
                DetectPoint(m.h_scores.get(), features, height, width, impl_->config.remove_borders, impl_->config);
                ExtractDescriptors(m.h_descriptors.get(), features, height, width, 8, impl_->config.official_desc_sampling);
            } else {
                // GPU: NMS candidates -> (CPU) selection in the original row-major order -> GPU descriptors.
                m.d_count.resize(1);
                m.d_candidate_index.resize(score_count);
                m.d_candidate_scores.resize(score_count);
                m.h_count.reserve(1);
                sp_cuda::LaunchNms(m.d_scores.get(), height, width, impl_->config.remove_borders, candidate_threshold,
                                   m.d_count.get(), m.d_candidate_index.get(), m.d_candidate_scores.get(), score_count, stream);
                m.d_count.copyToHost(m.h_count.get(), 1, stream);
                m.stream.synchronize();
                const int candidates = std::min(m.h_count[0], score_count);
                m.h_candidate_index.reserve(candidates);
                m.h_candidate_scores.reserve(candidates);
                m.d_candidate_index.copyToHost(m.h_candidate_index.get(), candidates, stream);
                m.d_candidate_scores.copyToHost(m.h_candidate_scores.get(), candidates, stream);
                m.stream.synchronize();
                std::vector<int> order(candidates);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&m](int a, int b) {
                    return m.h_candidate_index[a] < m.h_candidate_index[b];
                });
                std::vector<float> scores(candidates);
                std::vector<int> xs(candidates), ys(candidates);
                for (int i = 0; i < candidates; ++i) {
                    scores[i] = m.h_candidate_scores[order[i]];
                    ys[i] = m.h_candidate_index[order[i]] / width;
                    xs[i] = m.h_candidate_index[order[i]] % width;
                }
                const std::vector<int> selected = SelectKeypointIndexes(scores, xs, ys, height, width, impl_->config);
                const int count = selected.size();
                features.resize(259, count);
                m.h_xy.reserve(2 * std::max(count, 1));
                for (int i = 0; i < count; ++i) {
                    features(0, i) = scores[selected[i]];
                    features(1, i) = static_cast<float>(xs[selected[i]]);
                    features(2, i) = static_cast<float>(ys[selected[i]]);
                    m.h_xy[2 * i] = features(1, i);
                    m.h_xy[2 * i + 1] = features(2, i);
                }
                if (count > 0) {
                    m.d_xy.copyFromHost(m.h_xy.get(), 2 * count, stream);
                    m.d_features.resize(static_cast<size_t>(count) * 256);
                    m.h_features.reserve(static_cast<size_t>(count) * 256);
                    sp_cuda::LaunchDescriptors(m.d_descriptors.get(), height, width, 8, m.d_xy.get(), count,
                                               m.d_features.get(), stream, impl_->config.official_desc_sampling);
                    m.d_features.copyToHost(m.h_features.get(), static_cast<size_t>(count) * 256, stream);
                    m.stream.synchronize();
                    for (int i = 0; i < count; ++i)
                        for (int c = 0; c < 256; ++c) features(c + 3, i) = m.h_features[static_cast<size_t>(i) * 256 + c];
                }
            }
            for (int col = 0; col < features.cols(); ++col) {
                features(1, col) *= width_scale;
                features(2, col) *= height_scale;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "SuperPoint::infer failed: " << e.what() << std::endl;
        return false;
    }
}

