#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern "C" void ffsGwcFloat(const float*, const float*, float*, int, int, int,
                             int, int, int, bool, cudaStream_t);
extern "C" void ffsGwcHalf(const __half*, const __half*, __half*, int, int, int,
                            int, int, int, bool, cudaStream_t);
extern "C" void ffsGwcFloatHalf(const float*, const float*, __half*, int, int,
                                 int, int, int, int, bool, cudaStream_t);
extern "C" void ffsGwcHalfFloat(const __half*, const __half*, float*, int, int,
                                 int, int, int, int, bool, cudaStream_t);

namespace {

constexpr char kName[] = "FFSGWCVolume";
constexpr char kVersion[] = "1";

struct Parameters {
    int32_t max_disp = 0;
    int32_t cv_group = 0;
    int32_t normalize = 1;
};

int32_t fieldInt(const nvinfer1::PluginField& field, int32_t fallback) {
    if (!field.data || field.length <= 0) return fallback;
    if (field.type == nvinfer1::PluginFieldType::kINT32)
        return *static_cast<const int32_t*>(field.data);
    if (field.type == nvinfer1::PluginFieldType::kINT64)
        return static_cast<int32_t>(*static_cast<const int64_t*>(field.data));
    return fallback;
}

class GwcPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit GwcPlugin(Parameters params) : params_(params) {}
    GwcPlugin(const void* data, size_t size) {
        if (data && size == sizeof(params_)) std::memcpy(&params_, data, size);
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* result = new GwcPlugin(params_);
        result->setPluginNamespace(namespace_.c_str());
        return result;
    }
    const char* getPluginType() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t index, const nvinfer1::DimsExprs* inputs, int32_t input_count,
        nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output{};
        if (index != 0 || input_count != 2 || inputs[0].nbDims != 4) {
            output.nbDims = -1;
            return output;
        }
        output.nbDims = 5;
        output.d[0] = inputs[0].d[0];
        output.d[1] = builder.constant(params_.cv_group);
        output.d[2] = builder.constant(params_.max_disp);
        output.d[3] = inputs[0].d[2];
        output.d[4] = inputs[0].d[3];
        return output;
    }

    bool supportsFormatCombination(int32_t pos,
        const nvinfer1::PluginTensorDesc* io, int32_t inputs,
        int32_t outputs) noexcept override {
        if (inputs != 2 || outputs != 1 || pos < 0 || pos >= 3 ||
            io[pos].format != nvinfer1::TensorFormat::kLINEAR) return false;
        if (pos == 0) return io[0].type == nvinfer1::DataType::kFLOAT ||
                             io[0].type == nvinfer1::DataType::kHALF;
        if (pos == 1) return io[1].type == io[0].type;
        return io[2].type == nvinfer1::DataType::kFLOAT ||
               io[2].type == nvinfer1::DataType::kHALF;
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                         const nvinfer1::DynamicPluginTensorDesc*,
                         int32_t) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int32_t,
                            const nvinfer1::PluginTensorDesc*,
                            int32_t) const noexcept override { return 0; }

    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputs,
                    const nvinfer1::PluginTensorDesc* outputs,
                    const void* const* input_ptrs, void* const* output_ptrs,
                    void*, cudaStream_t stream) noexcept override {
        if (!inputs || !outputs || !input_ptrs || !output_ptrs) return 1;
        const auto dims = inputs[0].dims;
        if (dims.nbDims != 4) return 1;
        const int n = dims.d[0], c = dims.d[1], h = dims.d[2], w = dims.d[3];
        const bool normalize = params_.normalize != 0;

        if (inputs[0].type == nvinfer1::DataType::kFLOAT &&
            outputs[0].type == nvinfer1::DataType::kFLOAT) {
            ffsGwcFloat(static_cast<const float*>(input_ptrs[0]),
                        static_cast<const float*>(input_ptrs[1]),
                        static_cast<float*>(output_ptrs[0]), n, c, h, w,
                        params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputs[0].type == nvinfer1::DataType::kFLOAT &&
            outputs[0].type == nvinfer1::DataType::kHALF) {
            ffsGwcFloatHalf(static_cast<const float*>(input_ptrs[0]),
                            static_cast<const float*>(input_ptrs[1]),
                            static_cast<__half*>(output_ptrs[0]), n, c, h, w,
                            params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputs[0].type == nvinfer1::DataType::kHALF &&
            outputs[0].type == nvinfer1::DataType::kHALF) {
            ffsGwcHalf(static_cast<const __half*>(input_ptrs[0]),
                       static_cast<const __half*>(input_ptrs[1]),
                       static_cast<__half*>(output_ptrs[0]), n, c, h, w,
                       params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputs[0].type == nvinfer1::DataType::kHALF &&
            outputs[0].type == nvinfer1::DataType::kFLOAT) {
            ffsGwcHalfFloat(static_cast<const __half*>(input_ptrs[0]),
                            static_cast<const __half*>(input_ptrs[1]),
                            static_cast<float*>(output_ptrs[0]), n, c, h, w,
                            params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        return 1;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t, const nvinfer1::DataType* input_types,
        int32_t input_count) const noexcept override {
        return input_count > 0 && input_types[0] == nvinfer1::DataType::kHALF
                   ? nvinfer1::DataType::kHALF
                   : nvinfer1::DataType::kFLOAT;
    }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return sizeof(params_); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &params_, sizeof(params_));
    }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* value) noexcept override {
        namespace_ = value ? value : "";
    }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    void attachToContext(cudnnContext*, cublasContext*,
                         nvinfer1::IGpuAllocator*) noexcept override {}
    void detachFromContext() noexcept override {}

private:
    Parameters params_;
    std::string namespace_;
};

class GwcCreator final : public nvinfer1::IPluginCreator {
public:
    GwcCreator() {
        fields_.emplace_back("max_disp", nullptr,
                             nvinfer1::PluginFieldType::kINT32, 1);
        fields_.emplace_back("cv_group", nullptr,
                             nvinfer1::PluginFieldType::kINT32, 1);
        fields_.emplace_back("normalize", nullptr,
                             nvinfer1::PluginFieldType::kINT32, 1);
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
    }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &collection_;
    }
    nvinfer1::IPluginV2* createPlugin(
        const char*, const nvinfer1::PluginFieldCollection* fields) noexcept override {
        Parameters params;
        if (fields) {
            for (int i = 0; i < fields->nbFields; ++i) {
                const auto& field = fields->fields[i];
                if (!std::strcmp(field.name, "max_disp"))
                    params.max_disp = fieldInt(field, params.max_disp);
                else if (!std::strcmp(field.name, "cv_group"))
                    params.cv_group = fieldInt(field, params.cv_group);
                else if (!std::strcmp(field.name, "normalize"))
                    params.normalize = fieldInt(field, params.normalize);
            }
        }
        if (params.max_disp <= 0 || params.cv_group <= 0) return nullptr;
        auto* result = new GwcPlugin(params);
        result->setPluginNamespace(namespace_.c_str());
        return result;
    }
    nvinfer1::IPluginV2* deserializePlugin(
        const char*, const void* data, size_t size) noexcept override {
        auto* result = new GwcPlugin(data, size);
        result->setPluginNamespace(namespace_.c_str());
        return result;
    }
    void setPluginNamespace(const char* value) noexcept override {
        namespace_ = value ? value : "";
    }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection collection_{};
    std::string namespace_;
};

GwcCreator g_creator;

}  // namespace

extern "C" bool ffsRegisterGwcPlugin() {
    static bool result = false;
    static std::once_flag once;
    std::call_once(once, [] {
        auto register_with = [](nvinfer1::IPluginRegistry* registry) {
            if (!registry) return false;
            if (registry->getPluginCreator(kName, kVersion, "")) return true;
            return registry->registerCreator(g_creator, "");
        };
        const bool runtime_ok = register_with(getPluginRegistry());
        const bool builder_ok = register_with(nvinfer1::getBuilderPluginRegistry(
            nvinfer1::EngineCapability::kSTANDARD));
        result = runtime_ok || builder_ok;
    });
    return result;
}

REGISTER_TENSORRT_PLUGIN(GwcCreator);
