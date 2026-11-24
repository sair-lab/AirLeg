#pragma once

#include <NvInfer.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lightglue {

struct ProfileDims {
  nvinfer1::Dims min{};
  nvinfer1::Dims opt{};
  nvinfer1::Dims max{};
};

struct EngineConfig {
  std::string onnxPath;
  std::string enginePath;
  std::string tag;
  bool fp16 = true;
  size_t workspaceSize = 1ULL << 30;
  std::map<std::string, ProfileDims> profiles;
};

struct TrtDeleter {
  template <typename T>
  void operator()(T* ptr) const { delete ptr; }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

class TRTEngine {
public:
  bool build(const EngineConfig& config);

  nvinfer1::ICudaEngine* engine() { return engine_.get(); }
  nvinfer1::IExecutionContext* createContext();

  const std::vector<std::string>& inputNames() const { return input_names_; }
  const std::vector<std::string>& outputNames() const { return output_names_; }

private:
  bool buildFromOnnx(const EngineConfig& config);
  bool loadSerialized(const std::string& path);
  bool saveEngine(const std::string& path);
  std::string deriveEnginePath(const EngineConfig& config) const;
  void enumerateBindings();

  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace lightglue
