cmake_minimum_required(VERSION 3.18)
project(air_slam_ros2)
# CUDA kernels of the core (SuperPoint post-processing); PTX for 86 is JIT-compiled on newer GPUs
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
# -march=native lets the compiler contract floating-point operations (FMA) and vectorise differently:
# faster, but results are no longer bit-identical to a generic build. Off by default.
option(AIRSLAM_NATIVE_ARCH "Compile C++ with -march=native" OFF)
if(AIRSLAM_NATIVE_ARCH)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
if(NOT CMAKE_BUILD_TYPE)  # do not override -DCMAKE_BUILD_TYPE=...
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_definitions(BOOST_BIND_GLOBAL_PLACEHOLDERS)

get_filename_component(AIR_SLAM_ROS2_DIR "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(AIR_SLAM_REPO_DIR "${AIR_SLAM_ROS2_DIR}/../.." REALPATH)
set(AIR_SLAM_CORE_DIR "${AIR_SLAM_REPO_DIR}/core")

list(APPEND CMAKE_MODULE_PATH ${AIR_SLAM_CORE_DIR}/cmake)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(G2O REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(lowstate_republisher REQUIRED)

# S3 superframe dense depth (Fast-FoundationStereo TensorRT, vendored in core/3rdparty/ffs_infer; model in robots/common/models,
# see its ffs.yaml). OFF builds without it: the foot-surface factors are then disabled at run time.
option(AIRSLAM_WITH_FFS "Link ffs_infer for S3 superframe stereo depth" ON)
set(AIRSLAM_FFS_DIR "${AIR_SLAM_CORE_DIR}/3rdparty/ffs_infer" CACHE PATH "ffs_infer source directory")
if(AIRSLAM_WITH_FFS)
  add_subdirectory(${AIRSLAM_FFS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ffs_infer)
endif()

add_subdirectory(${AIR_SLAM_CORE_DIR}/3rdparty/tensorrtbuffer ${CMAKE_CURRENT_BINARY_DIR}/tensorrtbuffer)
add_subdirectory(${AIR_SLAM_CORE_DIR}/3rdparty/DBoW2 ${CMAKE_CURRENT_BINARY_DIR}/DBoW2)

include(${AIR_SLAM_CORE_DIR}/cmake/air_slam_core.cmake)

include_directories(
  ${AIR_SLAM_CORE_DIR}
  ${AIR_SLAM_CORE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${gflags_INCLUDE_DIRS}
  ${glog_INCLUDE_DIRS}
)

add_library(air_slam_core SHARED ${AIR_SLAM_CORE_SOURCES})
target_compile_definitions(air_slam_core PUBLIC G2O_USE_VENDORED_CERES)
target_compile_options(air_slam_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>  # a missing return is undefined behaviour
)
target_link_libraries(air_slam_core
  nvinfer
  nvonnxparser
  ${OpenCV_LIBRARIES}
  CUDA::cudart
  ${Boost_LIBRARIES}
  ${G2O_LIBRARIES}
  ${gflags_LIBRARIES}
  ${glog_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
  DBoW2
  -lboost_serialization
)
if(AIRSLAM_WITH_FFS)
  target_link_libraries(air_slam_core ffs_infer)
  target_compile_definitions(air_slam_core PUBLIC AIRSLAM_WITH_FFS)
endif()

add_library(air_slam_ros2_support SHARED
  src/ros2_publisher.cpp
)
ament_target_dependencies(air_slam_ros2_support
  cv_bridge
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)
target_link_libraries(air_slam_ros2_support air_slam_core)

add_executable(visual_odometry_node src/visual_odometry_node.cpp)
ament_target_dependencies(visual_odometry_node rclcpp geometry_msgs nav_msgs lowstate_republisher)
target_link_libraries(visual_odometry_node air_slam_core air_slam_ros2_support)

add_executable(map_refinement_node src/map_refinement_node.cpp)
ament_target_dependencies(map_refinement_node rclcpp)
target_link_libraries(map_refinement_node air_slam_core air_slam_ros2_support)

add_executable(relocalization_node src/relocalization_node.cpp)
ament_target_dependencies(relocalization_node rclcpp)
target_link_libraries(relocalization_node air_slam_core air_slam_ros2_support)

add_executable(map_optimization_backend ${AIR_SLAM_CORE_DIR}/src/map_optimization_backend.cc)
target_link_libraries(map_optimization_backend air_slam_core)

option(AIRSLAM_BUILD_TESTS "Build S3 online fusion unit tests" OFF)
if(AIRSLAM_BUILD_TESTS)
  add_executable(online_fusion_tests ${AIR_SLAM_CORE_DIR}/test/test_online_fusion.cc)
  target_link_libraries(online_fusion_tests air_slam_core)
  # replays dumped back-end inputs (backend_nodes.csv / backend_feet.csv) with other back-end settings
  add_executable(backend_replay ${AIR_SLAM_CORE_DIR}/test/backend_replay.cc)
  target_link_libraries(backend_replay air_slam_core)
endif()

install(TARGETS
  tensorrtbuffer
  air_slam_core
  air_slam_ros2_support
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  visual_odometry_node
  map_refinement_node
  map_optimization_backend
  relocalization_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# Robot-agnostic RViz view (robot_viz.launch.py): visualisation node, the robot description shared with the launch
# file, and the MJCF -> URDF converter it uses
install(PROGRAMS
  scripts/robot_viz.py
  scripts/robot_viz_model.py
  scripts/robot_viz_geometry.py
  scripts/mjcf_to_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
