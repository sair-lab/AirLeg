#include "engine_builder.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <model.engine> [workspace_mb]\n";
        return 1;
    }
    try {
        const size_t workspace_mb = argc > 3 ? std::stoull(argv[3]) : 4096;
        ffs_infer::buildEngine(argv[1], argv[2], workspace_mb);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
