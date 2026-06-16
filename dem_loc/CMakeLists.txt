cmake_minimum_required(VERSION 3.0.2)
project(loc_bev)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 1. 查找 ROS 依赖 ---
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  grid_map_msgs
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

catkin_python_setup()


# --- 2. 查找第三方库 ---
find_package(GDAL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# Torch 设置 (保持不变)
set(Torch_DIR /home/hsw/Library/libtorch-shared-with-deps-2.9.0+cu126/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- 3. Catkin 包配置 ---
# 关键修改：删除了 'INCLUDE_DIRS include'，解决了报错
# 因为你的头文件在 src/include，且只供本节点内部使用，不需要导出
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs nav_msgs grid_map_msgs cv_bridge image_transport pcl_ros pcl_conversions tf2 tf2_ros tf2_geometry_msgs tf2_eigen
  DEPENDS OpenCV PCL GDAL
)

# --- 4. 头文件包含路径 ---
include_directories(
  ${PROJECT_SOURCE_DIR}/src/include   # <--- 核心修改：指向 src/include (你的头文件所在地)
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${GDAL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# --- 5. 编译可执行文件 ---
# 关键修改：根据 tree 结构，添加 src/ 前缀
add_executable(localization
  src/localization.cpp
  src/cpp/CInterface.cpp
  src/cpp/CommonStruct.cpp
  src/cpp/CoordConverter.cpp
  src/cpp/Tool.cpp
)

# --- 6. 链接库 ---
target_link_libraries(localization
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${GDAL_LIBRARIES}
  cnpy     # 确保已安装系统库 (如 libcnpy.so)
)

# 打印信息用于确认路径
set_target_properties(localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
message(STATUS "Target [localization] output path: ${CMAKE_BINARY_DIR}")

message(STATUS "Project Source Dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "Included Headers from: ${PROJECT_SOURCE_DIR}/src/include")

catkin_install_python(PROGRAMS
  scripts/localization_python.py
  scripts/lidar_height_pf_localization.py
  scripts/test_cinterface.py
  scripts/visualize_terrain_roughness.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
