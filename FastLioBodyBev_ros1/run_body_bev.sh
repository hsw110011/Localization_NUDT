#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/noetic/setup.bash
if [ -f /home/hsw/catkin_ws/devel/setup.bash ]; then
  source /home/hsw/catkin_ws/devel/setup.bash
fi

rosparam load "${PROJECT_DIR}/config/body_bev.yaml" /fast_lio_body_bev
exec "${PROJECT_DIR}/bin/fast_lio_body_bev" __name:=fast_lio_body_bev
