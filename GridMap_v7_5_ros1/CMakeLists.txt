cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(GridMap_v7_5)

# Build type configuration
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Compiler flags
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-return-type -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-return-type -Wno-deprecated-declarations")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_DEBUG_POSTFIX _d)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# Find ROS packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    message_filters
    cv_bridge
    geometry_msgs
    grid_map_core
    grid_map_ros
    grid_map_cv
    grid_map_msgs
    self_state
    world_state
)

# OpenMP configuration
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    /opt/ros/noetic/include
    ../../../devel/include
    ./include
    ./
)

# Source files
set(SOURCES
    src/main.cpp
    src/sensor_map.cpp
    src/lidar_map.cpp
    src/image_proj.cpp
    src/my_config.cpp
    src/colors.cpp
    src/gridmap_handle.cpp
    src/gridmap_handle_v2.cpp
    src/gridmap_handle_v3.cpp
    src/gridmap_handle_v4.cpp
    src/lidar_bev_builder.cpp
    src/odom_stabilizer.cpp
)

# Header files
set(HEADERS
    include/map/gridmap_handle.h
    include/map/gridmap_handle_v2.h
    include/map/gridmap_handle_v3.h
    include/map/gridmap_handle_v4.h
    include/map/lidar_bev_builder.h
    include/map/odom_stabilizer.h
    include/map/lidar_map.h
    include/map/image_proj.h
    include/map/my_config.h
    include/map/colors.hpp
    include/map/sensor_map.h
    include/map/SensorDataCollection.h
    include/map/ars548_map.h
    include/map/threadqueue.h
)

# Libraries to link
set(LIBRARIES
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    pthread
    gomp
)



# Create executable
add_executable(GridMap ${SOURCES} ${HEADERS})



# Set target properties
set_target_properties(GridMap PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# Link libraries
target_link_libraries(GridMap ${LIBRARIES})

# OpenMP compile options
if(OpenMP_CXX_FOUND)
    target_link_libraries(GridMap OpenMP::OpenMP_CXX)
endif()

# Install rules (optional)
install(TARGETS GridMap
    RUNTIME DESTINATION bin
)
