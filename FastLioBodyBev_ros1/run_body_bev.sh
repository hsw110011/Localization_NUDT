#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
if [ -f /home/hsw/catkin_ws/devel/setup.bash ]; then
  source /home/hsw/catkin_ws/devel/setup.bash
fi

export ROS_PACKAGE_PATH="${PROGRAM_DIR}:${ROS_PACKAGE_PATH:-}"

if [ ! -x "${PROJECT_DIR}/bin/fast_lio_body_bev" ]; then
  echo "Missing executable: ${PROJECT_DIR}/bin/fast_lio_body_bev" >&2
  echo "Build it first: cmake -S ${PROJECT_DIR} -B ${PROJECT_DIR}/build && cmake --build ${PROJECT_DIR}/build -j\$(nproc)" >&2
  exit 1
fi

exec roslaunch fast_lio_body_bev_ros1 body_bev.launch "$@"
