cmake_minimum_required(VERSION 3.0.2)
project(loc_bev)

# ==========================================
# 0. 编译标准与输出路径
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ==========================================
# 1. 查找 ROS（必须最先）
# ==========================================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  novatel_msgs
  self_state
  world_state
  behavior
)

# ==========================================
# 2. OpenCV
# ==========================================
find_package(OpenCV REQUIRED)

message(STATUS "----------------------------------------")
message(STATUS "Using OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "----------------------------------------")

# ==========================================
# 3. PCL
# ==========================================
find_package(PCL REQUIRED)

# ==========================================
# 4. LibTorch
# ==========================================
set(Torch_DIR
  /home/hsw/Library/libtorch-shared-with-deps-2.9.0+cu126/libtorch/share/cmake/Torch
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ==========================================
# 5. GTSAM
# ==========================================
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

# ==========================================
# 6. catkin_package
# ==========================================
catkin_package(
  INCLUDE_DIRS src/include
  CATKIN_DEPENDS roscpp
)

# ==========================================
# 7. 头文件
# ==========================================
include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_definitions(${PCL_DEFINITIONS})

# ==========================================
# 8. 源文件组织
# ==========================================
set(LOCALIZATION_COMMON_SOURCES
  src/cpp/CInterface.cpp
  src/cpp/CommonStruct.cpp
  src/cpp/CoordConverter.cpp
  src/cpp/Tool.cpp
)

set(LOCALIZATION_LINK_LIBS
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
  gtsam
  cnpy
)

function(add_localization_target target_name main_source particle_filter_source)
  add_executable(${target_name}
    ${main_source}
    ${LOCALIZATION_COMMON_SOURCES}
    ${particle_filter_source}
  )

  target_link_libraries(${target_name}
    ${LOCALIZATION_LINK_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  message(STATUS "Target [${target_name}] output path: ${CMAKE_BINARY_DIR}")
endfunction()

# ==========================================
# 9. 可执行文件
# ==========================================

# --- Executable 1: bev ---
add_executable(bev src/semantic_bev.cpp)

target_link_libraries(bev
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)


add_executable(bev_uncertain src/semantic_bev_uncertain.cpp)

target_link_libraries(bev_uncertain
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

set_target_properties(bev PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
message(STATUS "Target [bev] output path: ${CMAKE_BINARY_DIR}")
set_target_properties(bev_uncertain PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
message(STATUS "Target [bev_uncertain] output path: ${CMAKE_BINARY_DIR}")


add_localization_target(localization_baseline
  src/localization_baseline.cpp
  src/cpp/ParticleFilter_baseline.cpp
)

add_localization_target(localization_ourmethod
  src/localization_ourmethod.cpp
  src/cpp/ParticleFilter_our.cpp
)

add_localization_target(localization_forloop
  src/localization_forloop.cpp
  src/cpp/ParticleFilter_forloop.cpp
)


# [注意]: cnpy 如果没有 find_package，系统库里必须有，或者你需要手动 link 它的绝对路径
