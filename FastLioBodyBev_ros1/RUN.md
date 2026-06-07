# FastLioBodyBev_ros1 运行说明

## 1. 程序作用

该节点从 FAST-LIO 输出中读取车体系点云和里程计，生成局部 BEV `grid_map`。

当前默认话题：

```bash
点云输入: /cloud_registered_body
里程计输入: /Odometry
BEV 输出: /lidar_bev/grid_map
节点名: /fast_lio_body_bev
```

`/cloud_registered_body` 已经是 FAST-LIO 输出的 body frame 点云，节点内部不会再按原始雷达话题重复做雷达到车体的转换。

## 2. 前置条件

先保证 ROS master 和 FAST-LIO 已经运行，并且有下面两个话题：

```bash
rostopic hz /cloud_registered_body
rostopic hz /Odometry
```

如果两个话题没有数据，BEV 节点可以启动，但不会生成有效地图。

## 3. 编译

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
cmake --build build -j$(nproc)
```

编译成功后，可执行文件在：

```bash
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/bin/fast_lio_body_bev
```

## 4. 推荐运行方式

使用项目自带脚本：

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
./run_body_bev.sh
```

脚本会自动执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
rosparam load config/body_bev.yaml /fast_lio_body_bev
bin/fast_lio_body_bev __name:=fast_lio_body_bev
```

## 5. 直接运行可执行文件

如果不使用脚本，也可以手动加载参数后直接运行二进制：

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash

rosparam load /home/hsw/catkin_ws/program/FastLioBodyBev_ros1/config/body_bev.yaml /fast_lio_body_bev
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/bin/fast_lio_body_bev __name:=fast_lio_body_bev
```

注意：修改 `config/body_bev.yaml` 后，需要重新执行 `rosparam load`，否则 ROS 参数服务器里仍然可能是旧参数。

## 6. 后台运行

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
nohup ./run_body_bev.sh > /tmp/fast_lio_body_bev.log 2>&1 &
```

查看日志：

```bash
tail -f /tmp/fast_lio_body_bev.log
```

停止后台节点：

```bash
pkill -f fast_lio_body_bev
```

## 7. 运行状态检查

查看节点：

```bash
rosnode list | grep fast_lio_body_bev
```

查看输入输出连接：

```bash
rostopic info /cloud_registered_body
rostopic info /Odometry
rostopic info /lidar_bev/grid_map
```

查看输出频率：

```bash
rostopic hz /lidar_bev/grid_map
```

查看当前参数：

```bash
rosparam get /fast_lio_body_bev/input_topic
rosparam get /fast_lio_body_bev/odometry_topic
rosparam get /fast_lio_body_bev/output_topic
rosparam get /fast_lio_body_bev/resolution
rosparam get /fast_lio_body_bev/map_size_x
rosparam get /fast_lio_body_bev/map_size_y
rosparam get /fast_lio_body_bev/height_quantile
```

## 8. 当前关键参数

参数文件：

```bash
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/config/body_bev.yaml
```

当前 BEV 设置：

```yaml
resolution: 0.15
map_size_x: 60.0
map_size_y: 60.0
height_quantile: 1.0
accumulation_frame_count: 8
temporal_height_alpha: 0.35
temporal_height_max_jump: 0.35
height_smoothing_radius: 2
height_smoothing_max_delta: 0.35
height_smoothing_min_neighbors: 3
```

含义：

- `resolution: 0.15`：每个 BEV 栅格 0.15 m。
- `map_size_x/y: 60.0`：总范围 60 m x 60 m，对应约 400 x 400 栅格。
- `height_quantile: 1.0`：每个栅格只取最高点。
- `accumulation_frame_count: 8`：多帧累计最近 8 帧。
- `temporal_height_alpha` 和 `temporal_height_max_jump`：限制前后帧高度突变。
- `height_smoothing_*`：对相对高度图做空间平滑，降低道路粗糙感。

## 9. 可视化窗口

`show_windows: true` 时会弹出 OpenCV 窗口：

```bash
H_rel_surf  相对表面高度图
H_range     栅格内高度差
M_L         有效观测 mask
W_L         观测置信度
G_long_L    车体纵向高度梯度
G_lat_L     车体横向高度梯度
```

如果在无显示器或 SSH 环境运行，可以把参数改为：

```yaml
show_windows: false
```

然后重新加载参数并重启节点。

## 10. 常见问题

### 没有输出 `/lidar_bev/grid_map`

检查 FAST-LIO 是否正在发布输入：

```bash
rostopic hz /cloud_registered_body
rostopic hz /Odometry
```

再检查 BEV 节点是否订阅成功：

```bash
rostopic info /cloud_registered_body
rostopic info /Odometry
```

### 修改参数后没有生效

重新加载参数并重启节点：

```bash
pkill -f fast_lio_body_bev
rosparam load /home/hsw/catkin_ws/program/FastLioBodyBev_ros1/config/body_bev.yaml /fast_lio_body_bev
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/bin/fast_lio_body_bev __name:=fast_lio_body_bev
```

### 高度图仍然跳变

可以优先调小：

```yaml
temporal_height_alpha: 0.25
temporal_height_max_jump: 0.25
```

### 道路梯度仍然粗糙

可以优先调大：

```yaml
height_smoothing_radius: 3
edge_min_valid_neighbors: 6
```

但平滑太强会削弱路沿和小障碍物边缘。
