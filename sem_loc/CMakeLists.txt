cmake_minimum_required(VERSION 3.0.2)
project(loc_bev)

# 未指定类型时启用优化；显式 Debug / RelWithDebInfo 设置仍由调用者决定。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

# ROS 1.17.4 XML-RPC retry fix, scoped to these executables via RPATH.
# Keep the original SONAME so libroscpp uses the same patched implementation.
find_package(rostime REQUIRED)
find_package(cpp_common REQUIRED)
add_library(sem_loc_xmlrpcpp SHARED
  third_party/xmlrpcpp/src/XmlRpcClient.cpp
  third_party/xmlrpcpp/src/XmlRpcDispatch.cpp
  third_party/xmlrpcpp/src/XmlRpcServer.cpp
  third_party/xmlrpcpp/src/XmlRpcServerConnection.cpp
  third_party/xmlrpcpp/src/XmlRpcServerMethod.cpp
  third_party/xmlrpcpp/src/XmlRpcSocket.cpp
  third_party/xmlrpcpp/src/XmlRpcSource.cpp
  third_party/xmlrpcpp/src/XmlRpcUtil.cpp
  third_party/xmlrpcpp/src/XmlRpcValue.cpp
  third_party/xmlrpcpp/libb64/src/cdecode.c
  third_party/xmlrpcpp/libb64/src/cencode.c
)
target_include_directories(sem_loc_xmlrpcpp BEFORE PRIVATE
  third_party/xmlrpcpp/include third_party/xmlrpcpp/libb64/include)
target_link_libraries(sem_loc_xmlrpcpp ${rostime_LIBRARIES} ${cpp_common_LIBRARIES})
set_target_properties(sem_loc_xmlrpcpp PROPERTIES
  OUTPUT_NAME xmlrpcpp
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sem_loc_ros_lib"
)

# Source ROS setup normally puts /opt/ros in LD_LIBRARY_PATH. Use DT_RPATH,
# with this private directory first, so these nodes consistently load the fix.
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/sem_loc_ros_lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

set(LOCALIZATION_LINK_LIBS
  sem_loc_xmlrpcpp
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
  gtsam
  cnpy
)

# ==========================================
# 9. 统一运行入口：方法均由 config/sem_loc.yaml 选择
# ==========================================
add_executable(bev src/semantic_bev.cpp)
target_compile_definitions(bev PRIVATE SEM_LOC_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(bev
  sem_loc_xmlrpcpp
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(sem_localization
  src/sem_localization.cpp
  src/cpp/ParticleFilter.cpp
  ${LOCALIZATION_COMMON_SOURCES}
)
target_compile_definitions(sem_localization PRIVATE SEM_LOC_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(sem_localization ${LOCALIZATION_LINK_LIBS})

# 两个统一运行入口：./build/bev 和 ./build/sem_localization。
set_target_properties(bev sem_localization PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Motion regressions are opt-in build targets, so normal node builds stay unchanged.
if(CATKIN_ENABLE_TESTING)
  add_executable(test_xmlrpc_retry EXCLUDE_FROM_ALL tests/test_xmlrpc_retry.cpp)
  target_link_libraries(test_xmlrpc_retry sem_loc_xmlrpcpp)
  add_test(NAME xmlrpc_retry COMMAND test_xmlrpc_retry)
  add_executable(test_motion2d EXCLUDE_FROM_ALL tests/test_motion2d.cpp)
  target_link_libraries(test_motion2d gtsam)
  add_test(NAME motion2d COMMAND test_motion2d)
  add_executable(test_particle_motion EXCLUDE_FROM_ALL
    tests/test_particle_motion.cpp src/cpp/ParticleFilter.cpp src/cpp/CoordConverter.cpp)
  target_link_libraries(test_particle_motion ${LOCALIZATION_LINK_LIBS})
  add_test(NAME particle_motion COMMAND test_particle_motion)

  add_executable(test_scoring_routes EXCLUDE_FROM_ALL tests/test_scoring_routes.cpp
    src/cpp/ParticleFilter.cpp src/cpp/CoordConverter.cpp)
  target_link_libraries(test_scoring_routes ${LOCALIZATION_LINK_LIBS})
  add_test(NAME scoring_routes COMMAND test_scoring_routes)
  add_test(NAME bev_config COMMAND ${PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/bev_config_test.py $<TARGET_FILE:bev>)
endif()
