#include "ffs_infer.hpp"
#include "engine_builder.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

extern "C" bool ffsRegisterGwcPlugin();
extern "C" void ffsPreprocessBgr(const uint8_t*, float*, int, int, int, int,
                                  cudaStream_t);
extern "C" void ffsResizePadBgr(const uint8_t*, float*, int, int, int, int,
                                int, int, cudaStream_t);
extern "C" void ffsCropDisp(const float*, float*, int, int, int, int,
                            cudaStream_t);
extern "C" void ffsUpsampleDisp(const float*, float*, int, int, int, int,
                                float, cudaStream_t);
extern "C" void ffsClampDisp(float*, int, cudaStream_t);

namespace ffs_infer {
namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

Logger g_logger;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string unquote(std::string value) {
    value = trim(std::move(value));
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

struct RuntimeConfig {
    std::filesystem::path engine;
    std::filesystem::path onnx;
    int height = 0;
    int width = 0;
};

RuntimeConfig loadConfig(const std::filesystem::path& yaml_path) {
    std::ifstream input(yaml_path);
    if (!input) throw std::runtime_error("cannot open config: " + yaml_path.string());

    RuntimeConfig config;
    std::string line;
    while (std::getline(input, line)) {
        if (const auto hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        const std::string value = unquote(line.substr(colon + 1));
        if (key == "engine") config.engine = value;
        else if (key == "onnx") config.onnx = value;
        else if (key == "image_height") config.height = std::stoi(value);
        else if (key == "image_width") config.width = std::stoi(value);
    }

    if (config.engine.empty() || config.height <= 0 || config.width <= 0) {
        throw std::runtime_error("ffs.yaml requires engine, image_height and image_width");
    }
    if (config.engine.is_relative()) config.engine = yaml_path.parent_path() / config.engine;
    config.engine = std::filesystem::weakly_canonical(config.engine);
    if (config.onnx.empty()) {
        config.onnx = config.engine.parent_path() / "fast_foundationstereo_plugin.onnx";
    } else if (config.onnx.is_relative()) {
        config.onnx = yaml_path.parent_path() / config.onnx;
    }
    return config;
}

std::vector<char> readBinary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open engine: " + path.string());
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("empty engine: " + path.string());
    std::vector<char> data(static_cast<size_t>(size));
    input.seekg(0);
    input.read(data.data(), size);
    if (!input) throw std::runtime_error("cannot read engine: " + path.string());
    return data;
}

template <typename T>
void deviceFree(T*& pointer) noexcept {
    if (pointer) cudaFree(pointer);
    pointer = nullptr;
}

template <typename T>
void deviceAlloc(T*& pointer, size_t count, const char* name) {
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), name);
}

bool hasTensor(nvinfer1::ICudaEngine& engine, const char* name) {
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i) {
        if (std::string(engine.getIOTensorName(i)) == name) return true;
    }
    return false;
}

}  // namespace

struct FFSInfer::Impl {
    explicit Impl(const std::string& yaml_path) : config(loadConfig(yaml_path)) {
        if (!std::filesystem::exists(config.engine)) {
            std::cout << "Engine not found; building from " << config.onnx << std::endl;
            buildEngine(config.onnx, config.engine);
        } else {
            std::cout << "Using existing engine: " << config.engine << std::endl;
        }
        checkCuda(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
        try {
            if (!ffsRegisterGwcPlugin()) {
                throw std::runtime_error("failed to register FFSGWCVolume plugin");
            }
            runtime.reset(nvinfer1::createInferRuntime(g_logger));
            if (!runtime) throw std::runtime_error("createInferRuntime failed");
            const auto bytes = readBinary(config.engine);
            engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
            if (!engine) throw std::runtime_error("failed to deserialize TensorRT engine");
            context.reset(engine->createExecutionContext());
            if (!context) throw std::runtime_error("createExecutionContext failed");
            validateEngine();
            allocateModelBuffers();
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void validateEngine() {
        for (const char* name : {"left", "right", "disp"}) {
            if (!hasTensor(*engine, name)) {
                throw std::runtime_error(std::string("engine is missing tensor: ") + name);
            }
        }
        if (engine->getTensorDataType("left") != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType("right") != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType("disp") != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("engine left/right/disp tensors must be float32");
        }
        const auto dims = engine->getTensorShape("left");
        if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3 ||
            dims.d[2] != config.height || dims.d[3] != config.width) {
            throw std::runtime_error("engine input shape does not match ffs.yaml");
        }
    }

    void allocateModelBuffers() {
        const size_t pixels = static_cast<size_t>(config.height) * config.width;
        deviceAlloc(d_left_chw, 3 * pixels, "cudaMalloc left CHW");
        deviceAlloc(d_right_chw, 3 * pixels, "cudaMalloc right CHW");
        deviceAlloc(d_model_disp, pixels, "cudaMalloc model disparity");
        deviceAlloc(d_crop_disp, pixels, "cudaMalloc cropped disparity");
        if (!context->setTensorAddress("left", d_left_chw) ||
            !context->setTensorAddress("right", d_right_chw) ||
            !context->setTensorAddress("disp", d_model_disp)) {
            throw std::runtime_error("setTensorAddress failed");
        }
    }

    void ensureIoBuffers(int width, int height) {
        const size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= io_capacity) return;
        deviceFree(d_upload_left);
        deviceFree(d_upload_right);
        deviceFree(d_download_disp);
        deviceAlloc(d_upload_left, pixels * 3, "cudaMalloc upload left");
        deviceAlloc(d_upload_right, pixels * 3, "cudaMalloc upload right");
        deviceAlloc(d_download_disp, pixels, "cudaMalloc download disparity");
        io_capacity = pixels;
    }

    void preprocess(const uint8_t* input, int width, int height, float* output,
                    int& scaled_width, int& scaled_height) {
        if (width == config.width && height == config.height) {
            scaled_width = width;
            scaled_height = height;
            ffsPreprocessBgr(input, output, height, width, config.height,
                             config.width, cuda_stream);
            return;
        }
        const float scale = std::min(static_cast<float>(config.width) / width,
                                     static_cast<float>(config.height) / height);
        scaled_width = std::max(1, static_cast<int>(std::lround(width * scale)));
        scaled_height = std::max(1, static_cast<int>(std::lround(height * scale)));
        ffsResizePadBgr(input, output, height, width, scaled_height, scaled_width,
                        config.height, config.width, cuda_stream);
    }

    void inferDevice(const uint8_t* left, const uint8_t* right, int width,
                     int height, float* disparity) {
        if (!left || !right || !disparity || width <= 0 || height <= 0) {
            throw std::invalid_argument("invalid input to inferDevice");
        }

        int left_w = 0, left_h = 0, right_w = 0, right_h = 0;
        preprocess(left, width, height, d_left_chw, left_w, left_h);
        preprocess(right, width, height, d_right_chw, right_w, right_h);
        if (left_w != right_w || left_h != right_h) {
            throw std::runtime_error("internal stereo preprocessing mismatch");
        }

        const bool exact = width == config.width && height == config.height;
        if (exact && !context->setTensorAddress("disp", disparity)) {
            throw std::runtime_error("set output tensor address failed");
        }
        if (!context->enqueueV3(cuda_stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }

        if (exact) {
            ffsClampDisp(disparity, width * height, cuda_stream);
            context->setTensorAddress("disp", d_model_disp);
        } else {
            ffsClampDisp(d_model_disp, config.width * config.height, cuda_stream);
            ffsCropDisp(d_model_disp, d_crop_disp, config.height, config.width,
                        left_h, left_w, cuda_stream);
            const float disparity_scale = static_cast<float>(width) / left_w;
            ffsUpsampleDisp(d_crop_disp, disparity, left_w, left_h, width, height,
                            disparity_scale, cuda_stream);
        }
    }

    void release() noexcept {
        if (cuda_stream) cudaStreamSynchronize(cuda_stream);
        deviceFree(d_left_chw);
        deviceFree(d_right_chw);
        deviceFree(d_model_disp);
        deviceFree(d_crop_disp);
        deviceFree(d_upload_left);
        deviceFree(d_upload_right);
        deviceFree(d_download_disp);
        context.reset();
        engine.reset();
        runtime.reset();
        if (cuda_stream) cudaStreamDestroy(cuda_stream);
        cuda_stream = nullptr;
    }

    RuntimeConfig config;
    cudaStream_t cuda_stream = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    float* d_left_chw = nullptr;
    float* d_right_chw = nullptr;
    float* d_model_disp = nullptr;
    float* d_crop_disp = nullptr;
    uint8_t* d_upload_left = nullptr;
    uint8_t* d_upload_right = nullptr;
    float* d_download_disp = nullptr;
    size_t io_capacity = 0;
};

FFSInfer::FFSInfer(const std::string& yaml_path)
    : impl_(std::make_unique<Impl>(yaml_path)) {}

FFSInfer::~FFSInfer() = default;
FFSInfer::FFSInfer(FFSInfer&&) noexcept = default;
FFSInfer& FFSInfer::operator=(FFSInfer&&) noexcept = default;

void FFSInfer::infer(const uint8_t* left, const uint8_t* right, int width,
                     int height, std::vector<float>& disparity,
                     int row_stride_bytes) {
    if (!left || !right || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid CPU stereo input");
    }
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    const size_t stride = row_stride_bytes > 0 ? static_cast<size_t>(row_stride_bytes)
                                               : row_bytes;
    if (stride < row_bytes) throw std::invalid_argument("row stride is too small");

    impl_->ensureIoBuffers(width, height);
    checkCuda(cudaMemcpy2DAsync(impl_->d_upload_left, row_bytes, left, stride,
                                row_bytes, height, cudaMemcpyHostToDevice,
                                impl_->cuda_stream), "copy left image");
    checkCuda(cudaMemcpy2DAsync(impl_->d_upload_right, row_bytes, right, stride,
                                row_bytes, height, cudaMemcpyHostToDevice,
                                impl_->cuda_stream), "copy right image");
    impl_->inferDevice(impl_->d_upload_left, impl_->d_upload_right, width, height,
                       impl_->d_download_disp);
    disparity.resize(static_cast<size_t>(width) * height);
    checkCuda(cudaMemcpyAsync(disparity.data(), impl_->d_download_disp,
                              disparity.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, impl_->cuda_stream),
              "copy disparity");
    synchronize();
}

void FFSInfer::inferDevice(const uint8_t* left, const uint8_t* right, int width,
                           int height, float* disparity) {
    impl_->inferDevice(left, right, width, height, disparity);
}

void FFSInfer::synchronize() {
    checkCuda(cudaStreamSynchronize(impl_->cuda_stream), "inference synchronize");
}

cudaStream_t FFSInfer::stream() const noexcept { return impl_->cuda_stream; }
int FFSInfer::modelWidth() const noexcept { return impl_->config.width; }
int FFSInfer::modelHeight() const noexcept { return impl_->config.height; }

}  // namespace ffs_infer
