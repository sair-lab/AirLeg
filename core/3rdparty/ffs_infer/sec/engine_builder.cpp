#include "engine_builder.hpp"

#include <iostream>
#include <stdexcept>

#ifdef FFS_HAS_ONNX_PARSER
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cstdlib>
#include <sys/stat.h>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>
#include <unistd.h>

extern "C" bool ffsRegisterGwcPlugin();

namespace {
class BuildLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kINFO) std::cerr << "[TensorRT] " << message << '\n';
    }
};
}
#endif

namespace ffs_infer {
void buildEngine(const std::filesystem::path& onnx_path,
                 const std::filesystem::path& engine_path, size_t workspace_mb) {
    if (!std::filesystem::is_regular_file(onnx_path)) {
        throw std::runtime_error("ONNX model not found: " + onnx_path.string() +
                                 "; export the model first or set 'onnx' in ffs.yaml");
    }
#ifdef FFS_HAS_ONNX_PARSER
    if (workspace_mb == 0 || workspace_mb > std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL)) {
        throw std::invalid_argument("workspace_mb must be positive and fit in bytes");
    }
    if (!ffsRegisterGwcPlugin()) throw std::runtime_error("plugin registration failed");
    BuildLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("createInferBuilder failed");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) throw std::runtime_error("network creation failed");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("parser creation failed");
    if (!parser->parseFromFile(onnx_path.string().c_str(),
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("ONNX parsing failed: " + onnx_path.string());
    }
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("builder config creation failed");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                              workspace_mb * 1024ULL * 1024ULL);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT engine build failed");

    if (!engine_path.parent_path().empty()) {
        std::filesystem::create_directories(engine_path.parent_path());
    }
    // Publish only a completely written engine; failures must not leave a cache hit.
    const std::string pattern = engine_path.string() + ".tmp.XXXXXX";
    std::vector<char> temp_name(pattern.begin(), pattern.end());
    temp_name.push_back('\0');
    const int fd = mkstemp(temp_name.data());
    if (fd < 0) throw std::runtime_error("cannot create temporary engine: " + engine_path.string());
    // mkstemp creates the file 0600: give the published engine the usual permissions (0666 & ~umask) so that other
    // users of the workspace can load it instead of failing / rebuilding
    const mode_t mask = umask(0);
    umask(mask);
    fchmod(fd, 0666 & ~mask);
    close(fd);
    const std::filesystem::path temp_path(temp_name.data());
    try {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char*>(plan->data()), plan->size());
        output.close();
        if (!output) throw std::runtime_error("failed to write engine: " + engine_path.string());
        std::filesystem::rename(temp_path, engine_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_path, ignored);
        throw;
    }
    std::cout << "Built FP16 engine: " << engine_path << '\n';
#else
    (void)engine_path;
    (void)workspace_mb;
    throw std::runtime_error("automatic engine build requires TensorRT ONNX parser; install it and rebuild ffs_infer");
#endif
}
}
