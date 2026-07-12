# FastLioBodyBev_ros1

该节点读取 FAST-LIO 的车体系点云，生成局部 BEV `grid_map`。

## 话题

- 点云输入: `/cloud_registered_body`
- 里程计输入: `/Odometry`
- BEV 输出: `/lidar_bev/grid_map`
- 节点名: `/fast_lio_body_bev`

`dem_loc` 会消费 `/lidar_bev/grid_map`，因此输出图层名需要保持不变:

- `H_rel_surf`
- `M_L`
- `G_long_L`
- `G_lat_L`

## 编译

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

编译产物:

```bash
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/bin/fast_lio_body_bev
```

## 运行

最简单方式:

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
./run_body_bev.sh
```

等价的 ROS launch 方式:

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
export ROS_PACKAGE_PATH=/home/hsw/catkin_ws/program:${ROS_PACKAGE_PATH}
roslaunch fast_lio_body_bev_ros1 body_bev.launch
```

指定其他参数文件:

```bash
roslaunch fast_lio_body_bev_ros1 body_bev.launch config:=/path/to/body_bev.yaml
```

## 参数

参数文件:

```bash
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/config/body_bev.yaml
```

当前关键配置:

```yaml
input_topic: "/cloud_registered_body"
odometry_topic: "/Odometry"
output_topic: "/lidar_bev/grid_map"
resolution: 0.2
map_size_x: 64.0
map_size_y: 64.0
accumulation_frame_count: 0
cell_max_points: 0
show_windows: false
```

说明:

- `resolution`: BEV 栅格分辨率，单位 m。
- `map_size_x`, `map_size_y`: BEV 覆盖范围，单位 m。
- `accumulation_frame_count: 0`: 不限制历史帧数，窗口内历史点全部保留。
- `cell_max_points: 0`: 每个栅格不限制样本数。
- `show_windows: false`: 默认不弹 OpenCV 调试窗口，适合 SSH 或后台运行。

## 检查

运行前确认 FAST-LIO 输入存在:

```bash
rostopic hz /cloud_registered_body
rostopic hz /Odometry
```

运行后检查输出:

```bash
rosnode list | grep fast_lio_body_bev
rostopic hz /lidar_bev/grid_map
rostopic info /lidar_bev/grid_map
```

查看当前加载参数:

```bash
rosparam get /fast_lio_body_bev/input_topic
rosparam get /fast_lio_body_bev/output_topic
rosparam get /fast_lio_body_bev/resolution
rosparam get /fast_lio_body_bev/map_size_x
rosparam get /fast_lio_body_bev/map_size_y
```

## 后台运行

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
nohup ./run_body_bev.sh > /tmp/fast_lio_body_bev.log 2>&1 &
```

查看日志:

```bash
tail -f /tmp/fast_lio_body_bev.log
```

停止:

```bash
pkill -f fast_lio_body_bev
```
