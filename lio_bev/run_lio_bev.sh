#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash
if [ -f /home/hsw/catkin_ws/devel/setup.bash ]; then
  source /home/hsw/catkin_ws/devel/setup.bash
fi

export ROS_PACKAGE_PATH="${PROGRAM_DIR}:${ROS_PACKAGE_PATH:-}"

RESOLVED_PACKAGE="$(rospack find lio_bev)"
if [ "$(readlink -f "${RESOLVED_PACKAGE}")" != "$(readlink -f "${PROJECT_DIR}")" ]; then
  echo "Package mismatch: expected ${PROJECT_DIR}, resolved ${RESOLVED_PACKAGE}" >&2
  exit 1
fi

if [ ! -x "${PROJECT_DIR}/bin/lio_bev" ]; then
  echo "Missing executable: ${PROJECT_DIR}/bin/lio_bev" >&2
  echo "Build it first: cmake -S ${PROJECT_DIR} -B ${PROJECT_DIR}/build && cmake --build ${PROJECT_DIR}/build -j\$(nproc)" >&2
  exit 1
fi

echo "lio_bev project: ${PROJECT_DIR}"
echo "lio_bev executable: ${PROJECT_DIR}/bin/lio_bev"
exec roslaunch "${PROJECT_DIR}/launch/lio_bev.launch" "$@"
