cmake_minimum_required(VERSION 3.18)
project(ffs_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(FFS_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures, e.g. 87 for Orin, 86 for RTX30, 89 for RTX40")
set(CMAKE_CUDA_ARCHITECTURES ${FFS_CUDA_ARCHITECTURES})
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT installation prefix")

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} $ENV{CONDA_PREFIX} /usr /usr/local/tensorrt
  PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(NVINFER_LIBRARY NAMES nvinfer nvinfer.so.10
  HINTS ${TENSORRT_ROOT} $ENV{CONDA_PREFIX} /usr /usr/local/tensorrt
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser.so.10
  HINTS ${TENSORRT_ROOT} $ENV{CONDA_PREFIX} /usr /usr/local/tensorrt
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT runtime not found; set -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

add_library(ffs_infer STATIC
  sec/engine_builder.cpp
  sec/ffs_infer.cpp
  sec/ffs_gwc_plugin.cpp
  sec/ffs_kernels.cu)
set_target_properties(ffs_infer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(ffs_infer
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(ffs_infer PUBLIC CUDA::cudart ${NVINFER_LIBRARY})
target_compile_options(ffs_infer PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
  add_executable(ffs_demo main.cpp)
  target_link_libraries(ffs_demo PRIVATE ffs_infer ${OpenCV_LIBS})

  find_package(yaml-cpp REQUIRED)
  add_executable(ffs_main_test main_test.cpp)
  target_link_libraries(ffs_main_test PRIVATE ffs_infer ${OpenCV_LIBS} yaml-cpp)
else()
  message(STATUS "OpenCV not found: ffs_demo and ffs_main_test will not be built")
endif()

if(NVONNXPARSER_LIBRARY)
  target_compile_definitions(ffs_infer PRIVATE FFS_HAS_ONNX_PARSER=1)
  target_link_libraries(ffs_infer PUBLIC ${NVONNXPARSER_LIBRARY})
  add_executable(ffs_build_engine tools/build_engine.cpp)
  target_include_directories(ffs_build_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sec)
  target_link_libraries(ffs_build_engine PRIVATE ffs_infer ${NVONNXPARSER_LIBRARY})
else()
  message(STATUS "TensorRT ONNX parser not found: automatic engine building and ffs_build_engine are unavailable")
endif()

install(TARGETS ffs_infer ARCHIVE DESTINATION lib)
install(FILES include/ffs_infer.hpp DESTINATION include)
