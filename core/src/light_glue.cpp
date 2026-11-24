#include "light_glue.h"

#include "cuda_utils.h"
#include "trt_engine.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool Contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

nvinfer1::Dims MakeDims1(int a) {
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = a;
  return dims;
}

void FilterScoreMatrix(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& scores,
                       Eigen::Matrix<int, Eigen::Dynamic, 2>& matches_index,
                       Eigen::Matrix<float, Eigen::Dynamic, 1>& matches_score,
                       float threshold = 0.1f) {
  std::vector<std::pair<int, float>> row_max(scores.rows());
  for (int row = 0; row < scores.rows(); ++row) {
    float max_value = -FLT_MAX;
    for (int col = 0; col < scores.cols(); ++col) {
      if (scores(row, col) > max_value) {
        row_max[row] = {col, scores(row, col)};
        max_value = scores(row, col);
      }
    }
  }

  std::vector<std::pair<int, float>> col_max(scores.cols());
  for (int col = 0; col < scores.cols(); ++col) {
    float max_value = -FLT_MAX;
    for (int row = 0; row < scores.rows(); ++row) {
      if (scores(row, col) > max_value) {
        col_max[col] = {row, scores(row, col)};
        max_value = scores(row, col);
      }
    }
  }

  std::vector<int> index0;
  std::vector<int> index1;
  std::vector<float> score_values;
  for (int row = 0; row < static_cast<int>(row_max.size()); ++row) {
    const int col = row_max[row].first;
    if (col < 0 || col >= static_cast<int>(col_max.size())) continue;
    if (row == col_max[col].first) {
      const float score_exp = std::exp(row_max[row].second);
      if (score_exp > threshold) {
        index0.push_back(row);
        index1.push_back(col);
        score_values.push_back(score_exp);
      }
    }
  }

  matches_index.resize(index0.size(), 2);
  matches_score.resize(score_values.size(), 1);
  for (int i = 0; i < static_cast<int>(index0.size()); ++i) {
    matches_index(i, 0) = index0[i];
    matches_index(i, 1) = index1[i];
    matches_score(i) = score_values[i];
  }
}

class DynOutputAllocator : public nvinfer1::IOutputAllocator {
public:
  void* reallocateOutputAsync(const char*, void*, uint64_t size, uint64_t, cudaStream_t) noexcept override {
    if (size > capacity_) {
      if (ptr_) cudaFree(ptr_);
      if (cudaMalloc(&ptr_, size) != cudaSuccess) {
        ptr_ = nullptr;
        capacity_ = 0;
        reported_size_ = 0;
        return nullptr;
      }
      capacity_ = size;
    }
    reported_size_ = size;
    return ptr_;
  }

  void notifyShape(const char*, nvinfer1::Dims const& dims) noexcept override {
    std::lock_guard<std::mutex> lock(mu_);
    dims_ = dims;
    has_shape_ = true;
  }

  template <typename T>
  std::vector<T> toHost(cudaStream_t stream) const {
    size_t count = 1;
    if (has_shape_) {
      for (int i = 0; i < dims_.nbDims; ++i) count *= static_cast<size_t>(std::max<int64_t>(0, dims_.d[i]));
    } else {
      count = reported_size_ / sizeof(T);
    }
    std::vector<T> out(count);
    if (count > 0) {
      LG_CUDA_CHECK(cudaMemcpyAsync(out.data(), ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    return out;
  }

  ~DynOutputAllocator() override {
    if (ptr_) cudaFree(ptr_);
  }

private:
  void* ptr_ = nullptr;
  uint64_t capacity_ = 0;
  uint64_t reported_size_ = 0;
  nvinfer1::Dims dims_{};
  bool has_shape_ = false;
  mutable std::mutex mu_;
};

}  // namespace

struct SuperPointLightGlue::Impl {
  explicit Impl(const PointMatcherConfig& cfg) : config(cfg) {}

  PointMatcherConfig config;
  lightglue::TRTEngine engine;
  lightglue::TrtUniquePtr<nvinfer1::IExecutionContext> context;

  bool dynamic_matches_output = false;
  std::string keypoints0_name;
  std::string keypoints1_name;
  std::string descriptors0_name;
  std::string descriptors1_name;
  std::string keypoints_name;
  std::string descriptors_name;
  std::string image_size_name;
  std::string scores_name;
  std::string matches_name;
  std::string mscores_name;
  // persistent stream, device buffers, output allocators and pinned host memory (no per-call allocation)
  lightglue::CudaStream stream;
  lightglue::CudaBuffer<float> d_keypoints, d_descriptors;
  lightglue::CudaBuffer<int64_t> d_image_size;
  lightglue::PinnedBuffer<float> h_keypoints, h_descriptors;
  DynOutputAllocator matches_allocator, mscores_allocator;
};

SuperPointLightGlue::SuperPointLightGlue() = default;

SuperPointLightGlue::SuperPointLightGlue(const PointMatcherConfig& lightglue_config)
    : impl_(std::make_unique<Impl>(lightglue_config)) {}

SuperPointLightGlue::~SuperPointLightGlue() = default;

bool SuperPointLightGlue::build() {
  if (!impl_) return false;

  lightglue::EngineConfig engine_config;
  engine_config.onnxPath = impl_->config.onnx_file;
  engine_config.enginePath = impl_->config.engine_file;
  engine_config.fp16 = true;

  lightglue::ProfileDims kpt_old;
  kpt_old.min = nvinfer1::Dims3{1, 1, 2};
  kpt_old.opt = nvinfer1::Dims3{1, 512, 2};
  kpt_old.max = nvinfer1::Dims3{1, 1024, 2};
  lightglue::ProfileDims desc_old;
  desc_old.min = nvinfer1::Dims3{1, 1, 256};
  desc_old.opt = nvinfer1::Dims3{1, 512, 256};
  desc_old.max = nvinfer1::Dims3{1, 1024, 256};

  engine_config.profiles["keypoints_0"] = kpt_old;
  engine_config.profiles["keypoints_1"] = kpt_old;
  engine_config.profiles["descriptors_0"] = desc_old;
  engine_config.profiles["descriptors_1"] = desc_old;
  for (size_t i = 0; i < impl_->config.input_tensor_names.size(); ++i) {
    engine_config.profiles[impl_->config.input_tensor_names[i]] = (i < 2) ? kpt_old : desc_old;
  }

  lightglue::ProfileDims kpt_new;
  kpt_new.min = nvinfer1::Dims3{2, 1, 2};
  kpt_new.opt = nvinfer1::Dims3{2, 512, 2};
  kpt_new.max = nvinfer1::Dims3{2, 1024, 2};
  lightglue::ProfileDims desc_new;
  desc_new.min = nvinfer1::Dims3{2, 1, 256};
  desc_new.opt = nvinfer1::Dims3{2, 512, 256};
  desc_new.max = nvinfer1::Dims3{2, 1024, 256};
  lightglue::ProfileDims size_new;
  size_new.min = MakeDims1(2);
  size_new.opt = MakeDims1(2);
  size_new.max = MakeDims1(2);
  engine_config.profiles["keypoints"] = kpt_new;
  engine_config.profiles["descriptors"] = desc_new;
  engine_config.profiles["image_size"] = size_new;

  if (!impl_->engine.build(engine_config)) return false;

  for (const auto& name : impl_->engine.inputNames()) {
    if (name == "keypoints_0") impl_->keypoints0_name = name;
    else if (name == "keypoints_1") impl_->keypoints1_name = name;
    else if (name == "descriptors_0") impl_->descriptors0_name = name;
    else if (name == "descriptors_1") impl_->descriptors1_name = name;
    else if (name == "keypoints" || (Contains(name, "keypoints") && impl_->keypoints_name.empty())) impl_->keypoints_name = name;
    else if (name == "descriptors" || (Contains(name, "descriptors") && impl_->descriptors_name.empty())) impl_->descriptors_name = name;
    else if (name == "image_size" || Contains(name, "image_size")) impl_->image_size_name = name;
  }

  for (const auto& name : impl_->engine.outputNames()) {
    if (name == "scores" || Contains(name, "scores")) impl_->scores_name = name;
    if (name == "matches" || Contains(name, "matches")) impl_->matches_name = name;
    if (name == "mscores" || Contains(name, "mscores")) impl_->mscores_name = name;
  }

  impl_->dynamic_matches_output = !impl_->matches_name.empty() && !impl_->mscores_name.empty();
  if (!impl_->dynamic_matches_output && impl_->scores_name.empty()) {
    std::cerr << "LightGlue: unsupported ONNX outputs. Need scores or matches/mscores." << std::endl;
    return false;
  }

  impl_->context.reset(impl_->engine.createContext());
  return impl_->context != nullptr;
}

bool SuperPointLightGlue::infer(const Eigen::Matrix<float, 258, Eigen::Dynamic>& features0,
                                const Eigen::Matrix<float, 258, Eigen::Dynamic>& features1,
                                Eigen::Matrix<int, Eigen::Dynamic, 2>& matches_index,
                                Eigen::Matrix<float, Eigen::Dynamic, 1>& matches_score) {
  if (!impl_ || !impl_->context) return false;
  const int n0 = features0.cols();
  const int n1 = features1.cols();
  if (n0 <= 0 || n1 <= 0) {
    matches_index.resize(0, 2);
    matches_score.resize(0, 1);
    return true;
  }

  try {
    if (impl_->dynamic_matches_output) {
      Impl& m = *impl_;
      const cudaStream_t stream = m.stream.get();
      const int k = std::max(n0, n1);
      const size_t keypoint_count = static_cast<size_t>(2) * k * 2;
      const size_t descriptor_count = static_cast<size_t>(2) * k * 256;
      m.h_keypoints.reserve(keypoint_count);
      m.h_descriptors.reserve(descriptor_count);
      float* h_keypoints = m.h_keypoints.get();
      float* h_descriptors = m.h_descriptors.get();
      std::fill_n(h_keypoints, keypoint_count, 0.0f);
      std::fill_n(h_descriptors, descriptor_count, 0.0f);
      const float scale = 0.5f;
      const float max_dim = static_cast<float>(std::max(impl_->config.image_width, impl_->config.image_height));

      for (int col = 0; col < n0; ++col) {
        h_keypoints[col * 2] = features0(0, col) * max_dim / scale + impl_->config.image_width * 0.5f;
        h_keypoints[col * 2 + 1] = features0(1, col) * max_dim / scale + impl_->config.image_height * 0.5f;
        for (int dim = 0; dim < 256; ++dim) h_descriptors[col * 256 + dim] = features0(dim + 2, col);
      }
      const size_t kpt1_offset = static_cast<size_t>(k) * 2;
      const size_t desc1_offset = static_cast<size_t>(k) * 256;
      for (int col = 0; col < n1; ++col) {
        h_keypoints[kpt1_offset + col * 2] = features1(0, col) * max_dim / scale + impl_->config.image_width * 0.5f;
        h_keypoints[kpt1_offset + col * 2 + 1] = features1(1, col) * max_dim / scale + impl_->config.image_height * 0.5f;
        for (int dim = 0; dim < 256; ++dim) h_descriptors[desc1_offset + col * 256 + dim] = features1(dim + 2, col);
      }

      int64_t image_size[2] = {impl_->config.image_width, impl_->config.image_height};
      m.d_keypoints.copyFromHost(h_keypoints, keypoint_count, stream);
      m.d_descriptors.copyFromHost(h_descriptors, descriptor_count, stream);
      m.d_image_size.copyFromHost(image_size, 2, stream);

      if (!impl_->context->setInputShape(impl_->keypoints_name.c_str(), nvinfer1::Dims3{2, k, 2}) ||
          !impl_->context->setInputShape(impl_->descriptors_name.c_str(), nvinfer1::Dims3{2, k, 256}) ||
          !impl_->context->setInputShape(impl_->image_size_name.c_str(), MakeDims1(2))) {
        return false;
      }

      impl_->context->setTensorAddress(impl_->keypoints_name.c_str(), m.d_keypoints.get());
      impl_->context->setTensorAddress(impl_->descriptors_name.c_str(), m.d_descriptors.get());
      impl_->context->setTensorAddress(impl_->image_size_name.c_str(), m.d_image_size.get());
      impl_->context->setOutputAllocator(impl_->matches_name.c_str(), &m.matches_allocator);
      impl_->context->setOutputAllocator(impl_->mscores_name.c_str(), &m.mscores_allocator);

      if (!impl_->context->enqueueV3(stream)) return false;
      std::vector<int64_t> raw_matches = m.matches_allocator.toHost<int64_t>(stream);
      std::vector<float> raw_scores = m.mscores_allocator.toHost<float>(stream);
      m.stream.synchronize();

      const int raw_count = std::min<int>(raw_scores.size(), raw_matches.size() / 3);
      std::vector<int> index0;
      std::vector<int> index1;
      std::vector<float> score_values;
      for (int i = 0; i < raw_count; ++i) {
        const int64_t pair = raw_matches[i * 3];
        const int64_t k0 = raw_matches[i * 3 + 1];
        const int64_t k1 = raw_matches[i * 3 + 2];
        if (pair == 0 && k0 >= 0 && k0 < n0 && k1 >= 0 && k1 < n1) {
          index0.push_back(static_cast<int>(k0));
          index1.push_back(static_cast<int>(k1));
          score_values.push_back(raw_scores[i]);
        }
      }

      matches_index.resize(index0.size(), 2);
      matches_score.resize(score_values.size(), 1);
      for (int i = 0; i < static_cast<int>(index0.size()); ++i) {
        matches_index(i, 0) = index0[i];
        matches_index(i, 1) = index1[i];
        matches_score(i) = score_values[i];
      }
      return true;
    }

    lightglue::CudaStream stream;
    std::vector<float> h_keypoints0(static_cast<size_t>(n0) * 2);
    std::vector<float> h_keypoints1(static_cast<size_t>(n1) * 2);
    std::vector<float> h_descriptors0(static_cast<size_t>(n0) * 256);
    std::vector<float> h_descriptors1(static_cast<size_t>(n1) * 256);

    for (int col = 0; col < n0; ++col) {
      h_keypoints0[col * 2] = features0(0, col);
      h_keypoints0[col * 2 + 1] = features0(1, col);
      for (int dim = 0; dim < 256; ++dim) h_descriptors0[col * 256 + dim] = features0(dim + 2, col);
    }
    for (int col = 0; col < n1; ++col) {
      h_keypoints1[col * 2] = features1(0, col);
      h_keypoints1[col * 2 + 1] = features1(1, col);
      for (int dim = 0; dim < 256; ++dim) h_descriptors1[col * 256 + dim] = features1(dim + 2, col);
    }

    lightglue::CudaBuffer<float> d_keypoints0(h_keypoints0.size());
    lightglue::CudaBuffer<float> d_keypoints1(h_keypoints1.size());
    lightglue::CudaBuffer<float> d_descriptors0(h_descriptors0.size());
    lightglue::CudaBuffer<float> d_descriptors1(h_descriptors1.size());
    lightglue::CudaBuffer<float> d_scores(static_cast<size_t>(n0) * n1);
    d_keypoints0.copyFromHost(h_keypoints0.data(), h_keypoints0.size(), stream.get());
    d_keypoints1.copyFromHost(h_keypoints1.data(), h_keypoints1.size(), stream.get());
    d_descriptors0.copyFromHost(h_descriptors0.data(), h_descriptors0.size(), stream.get());
    d_descriptors1.copyFromHost(h_descriptors1.data(), h_descriptors1.size(), stream.get());

    if (!impl_->context->setInputShape(impl_->keypoints0_name.c_str(), nvinfer1::Dims3{1, n0, 2}) ||
        !impl_->context->setInputShape(impl_->keypoints1_name.c_str(), nvinfer1::Dims3{1, n1, 2}) ||
        !impl_->context->setInputShape(impl_->descriptors0_name.c_str(), nvinfer1::Dims3{1, n0, 256}) ||
        !impl_->context->setInputShape(impl_->descriptors1_name.c_str(), nvinfer1::Dims3{1, n1, 256})) {
      return false;
    }

    impl_->context->setTensorAddress(impl_->keypoints0_name.c_str(), d_keypoints0.get());
    impl_->context->setTensorAddress(impl_->keypoints1_name.c_str(), d_keypoints1.get());
    impl_->context->setTensorAddress(impl_->descriptors0_name.c_str(), d_descriptors0.get());
    impl_->context->setTensorAddress(impl_->descriptors1_name.c_str(), d_descriptors1.get());
    impl_->context->setTensorAddress(impl_->scores_name.c_str(), d_scores.get());

    if (!impl_->context->enqueueV3(stream.get())) return false;
    std::vector<float> h_scores(static_cast<size_t>(n0) * n1);
    d_scores.copyToHost(h_scores.data(), h_scores.size(), stream.get());
    stream.synchronize();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> scores_matrix(n0, n1);
    for (int row = 0; row < n0; ++row) {
      for (int col = 0; col < n1; ++col) {
        scores_matrix(row, col) = h_scores[row * n1 + col];
      }
    }
    FilterScoreMatrix(scores_matrix, matches_index, matches_score);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "SuperPointLightGlue::infer failed: " << e.what() << std::endl;
    return false;
  }
}

