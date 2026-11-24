#pragma once

#include <cstddef>
#include <filesystem>

namespace ffs_infer {
void buildEngine(const std::filesystem::path& onnx_path,
                 const std::filesystem::path& engine_path,
                 size_t workspace_mb = 4096);
}
