cmake_minimum_required(VERSION 3.5)
project(lio_bev)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(LIO_BEV_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin" CACHE PATH
  "Directory for the lio_bev runtime executable")

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  grid_map_core
  grid_map_ros
  grid_map_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    tf2_ros
    grid_map_core
    grid_map_ros
    grid_map_msgs
  DEPENDS
    EIGEN3
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(lio_bev src/lio_bev_node.cpp)
set_target_properties(lio_bev PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${LIO_BEV_RUNTIME_OUTPUT_DIRECTORY}")

target_link_libraries(lio_bev
  ${catkin_LIBRARIES}
  Threads::Threads
)

# Useful warnings without imposing flags on downstream catkin packages.
target_compile_options(lio_bev PRIVATE -Wall -Wextra -Wpedantic)

catkin_install_python(PROGRAMS scripts/surface_diagnostics.py scripts/height_viewer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
