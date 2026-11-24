#include "trt_engine.h"

#include "logger.h"

#include <NvOnnxParser.h>

#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

namespace lightglue {

namespace {

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open file: " + path);
  const size_t size = file.tellg();
  file.seekg(0);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

std::string DimsToString(const nvinfer1::Dims& dims) {
  std::ostringstream os;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i) os << "x";
    os << dims.d[i];
  }
  return os.str();
}

}  // namespace

std::string TRTEngine::deriveEnginePath(const EngineConfig& config) const {
  if (!config.enginePath.empty()) return config.enginePath;
  fs::path onnx(config.onnxPath);
  std::string stem = onnx.stem().string();
  if (!config.tag.empty()) stem += "_" + config.tag;
  stem += config.fp16 ? "_fp16.engine" : "_fp32.engine";
  return (onnx.parent_path() / stem).string();
}

bool TRTEngine::build(const EngineConfig& config) {
  const std::string engine_path = deriveEnginePath(config);
  if (fs::exists(engine_path)) {
    std::cout << "[TRTEngine] Loading cached engine: " << engine_path << std::endl;
    if (loadSerialized(engine_path)) {
      enumerateBindings();
      return true;
    }
    std::cerr << "[TRTEngine] Failed to load cached engine, rebuilding..." << std::endl;
  }

  std::cout << "[TRTEngine] Building engine from ONNX: " << config.onnxPath << std::endl;
  if (!buildFromOnnx(config)) return false;
  enumerateBindings();
  if (!saveEngine(engine_path)) {
    std::cerr << "[TRTEngine] Warning: could not cache engine to " << engine_path << std::endl;
  }
  return true;
}

bool TRTEngine::buildFromOnnx(const EngineConfig& config) {
  auto& logger = getLogger();
  TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) return false;

  const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
  if (!network) return false;

  TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser) return false;

  if (!parser->parseFromFile(config.onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "[TRTEngine] ONNX parse errors:" << std::endl;
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "  " << parser->getError(i)->desc() << std::endl;
    }
    return false;
  }

  TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config(builder->createBuilderConfig());
  if (!builder_config) return false;
  builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspaceSize);

  if (config.fp16) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "[TRTEngine] FP16 enabled" << std::endl;
  }

  if (!config.profiles.empty()) {
    std::unordered_set<std::string> input_names;
    for (int i = 0; i < network->getNbInputs(); ++i) {
      input_names.insert(network->getInput(i)->getName());
    }

    auto* profile = builder->createOptimizationProfile();
    bool has_profile_dim = false;
    for (const auto& [name, dims] : config.profiles) {
      if (!input_names.count(name)) continue;
      std::cout << "[TRTEngine]   profile " << name
                << "  min=" << DimsToString(dims.min)
                << "  opt=" << DimsToString(dims.opt)
                << "  max=" << DimsToString(dims.max) << std::endl;
      profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims.min);
      profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims.opt);
      profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims.max);
      has_profile_dim = true;
    }
    if (has_profile_dim) builder_config->addOptimizationProfile(profile);
  }

  TrtUniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *builder_config));
  if (!serialized || serialized->size() == 0) return false;

  runtime_.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
  return engine_ != nullptr;
}

bool TRTEngine::loadSerialized(const std::string& path) {
  auto data = ReadFile(path);
  runtime_.reset(nvinfer1::createInferRuntime(getLogger()));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  return engine_ != nullptr;
}

bool TRTEngine::saveEngine(const std::string& path) {
  if (!engine_) return false;
  TrtUniquePtr<nvinfer1::IHostMemory> serialized(engine_->serialize());
  if (!serialized) return false;
  fs::path engine_path(path);
  if (engine_path.has_parent_path()) fs::create_directories(engine_path.parent_path());
  // atomic: an interrupted save never leaves a truncated engine, and concurrent nodes building the same engine
  // replace each other's complete file
  const std::string temporary = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream file(temporary, std::ios::binary);
    if (!file) return false;
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    if (!file) { file.close(); fs::remove(temporary); return false; }
  }
  std::error_code error;
  fs::rename(temporary, path, error);
  if (error) { fs::remove(temporary); return false; }
  std::cout << "[TRTEngine] Engine saved to: " << path << std::endl;
  return true;
}

nvinfer1::IExecutionContext* TRTEngine::createContext() {
  if (!engine_) return nullptr;
  return engine_->createExecutionContext();
}

void TRTEngine::enumerateBindings() {
  input_names_.clear();
  output_names_.clear();
  if (!engine_) return;
  const int count = engine_->getNbIOTensors();
  for (int i = 0; i < count; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.emplace_back(name);
    } else {
      output_names_.emplace_back(name);
    }
  }
  std::cout << "[TRTEngine] Inputs (" << input_names_.size() << "):";
  for (const auto& name : input_names_) std::cout << " " << name;
  std::cout << std::endl;
  std::cout << "[TRTEngine] Outputs (" << output_names_.size() << "):";
  for (const auto& name : output_names_) std::cout << " " << name;
  std::cout << std::endl;
}

}  // namespace lightglue
