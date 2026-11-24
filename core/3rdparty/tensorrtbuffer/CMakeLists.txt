cmake_minimum_required(VERSION 3.17)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# AirSLAM only uses the shared TensorRT logger (trt_engine.cpp).
# The heavier NVIDIA sample helper sources pull in deprecated TRT10-removed APIs
# (legacy parsers, fully-connected layer helpers, etc.), so we keep this target
# limited to the pieces the project actually uses.
add_library(tensorrtbuffer SHARED src/logger.cpp)


