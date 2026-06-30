# AI Code Index for ROS1 Mapping and DEM Localization

生成时间: 2026-06-30

本文档用于后续 AI 编程时快速理解 `/home/hsw/catkin_ws/program` 下三个重点程序:

- `FastLioBodyBev_ros1`
- `GridMap_v7_5_ros1`
- `dem_loc`

目标不是替代源码阅读, 而是提供可检索的工程索引: 入口、核心类、关键话题、参数文件、跨包接口、常见修改位置和注意事项。

## 1. 总体数据流

### 1.1 两条 BEV 生成路径

当前工程里有两套 LiDAR BEV 生成逻辑:

1. `FastLioBodyBev_ros1`
   - 输入 FAST-LIO 的车体系点云 `/cloud_registered_body`
   - 输入里程计 `/Odometry`
   - 输出 `grid_map_msgs/GridMap` 到 `/lidar_bev/grid_map`
   - 主要面向 FAST-LIO body frame 点云快速生成 BEV

2. `GridMap_v7_5_ros1`
   - 输入原始/预处理 LiDAR 点云, 根据车型配置转换到车体系
   - 输入 `/self_state/LocalPose`, `/self_state/LidarLocalPose`, 可选 `/sensor/Odometry`
   - 输出传统地图结果和 LiDAR BEV
   - 同样输出 `grid_map_msgs/GridMap` 到 `/lidar_bev/grid_map`

### 1.2 DEM 定位消费路径

`dem_loc` 订阅 `/lidar_bev/grid_map`, 从消息中解析以下固定图层:

- `H_rel_surf`
- `M_L`
- `G_long_L`
- `G_lat_L`

然后结合 DSM/DEM 文件、GlobalPose、LocalPose 和粒子滤波配置, 做 DSM-BEV 匹配评分、轨迹可视化和粒子滤波定位。

重要接口约束:

- 如果修改 BEV 图层名、图层含义、分辨率、地图尺寸或 `grid_map_msgs/GridMap` 的层数据排布, 必须同步检查 `dem_loc/loc_tool/cinterface.py` 和 `dem_loc/loc_tool/dsm_patch.py`。
- `dem_loc` 中 `parse_lidar_bev_grid_map()` 将 `G_long_L` 映射为 `Gy_L`, 将 `G_lat_L` 映射为 `Gx_L`。不要只按变量名直觉修改梯度方向。

## 2. FastLioBodyBev_ros1

### 2.1 作用

独立 ROS1 C++ 节点, 从 FAST-LIO 输出的车体系点云生成局部 BEV `grid_map`。

默认话题:

- 输入点云: `/cloud_registered_body`
- 输入里程计: `/Odometry`
- 输出 BEV: `/lidar_bev/grid_map`
- 节点名: `fast_lio_body_bev`

### 2.2 目录和文件

- `FastLioBodyBev_ros1/src/fast_lio_body_bev_node.cpp`
  - 单文件实现, 同时包含 ROS 节点包装和 BEV 构建算法。
- `FastLioBodyBev_ros1/config/body_bev.yaml`
  - 运行参数, 包括输入输出话题、BEV 尺寸、分辨率、高度过滤、自车过滤、地面估计、梯度/平滑参数。
- `FastLioBodyBev_ros1/CMakeLists.txt`
  - C++17, 依赖 Eigen3、PCL、OpenCV、ROS、grid_map、pcl_conversions。
  - 可执行文件输出到 `FastLioBodyBev_ros1/bin/fast_lio_body_bev`。
- `FastLioBodyBev_ros1/RUN.md`
  - 运行说明。
- `FastLioBodyBev_ros1/run_body_bev.sh`
  - 加载 ROS 环境, `rosparam load config/body_bev.yaml /fast_lio_body_bev`, 然后启动二进制。

未发现 `.launch` 文件。

### 2.3 核心类和函数

- `BodyBevNode`
  - 文件: `FastLioBodyBev_ros1/src/fast_lio_body_bev_node.cpp`
  - 负责读取私有参数、订阅点云和里程计、发布 BEV。
  - 关键回调:
    - `odometryCallback(const nav_msgs::Odometry::ConstPtr&)`
    - `cloudCallback(const sensor_msgs::PointCloud2::ConstPtr&)`

- `BodyBevBuilder`
  - 文件: `FastLioBodyBev_ros1/src/fast_lio_body_bev_node.cpp`
  - 负责点云过滤、多帧累计、地面参考估计、相对高度层、梯度层、mask 层和调试窗口。
  - 关键接口:
    - `configure(const Config&)`
    - `build(const std::vector<PointXYZ>&, const ros::Time&, const OdometryState*)`
    - `publish(const ros::Publisher&) const`
    - `printBevSummary() const`

### 2.4 输出 GridMap 图层

`BodyBevBuilder` 初始化并发布以下图层:

- `H_rel_surf`: 相对地面参考的表面高度
- `M_L`: LiDAR 观测有效 mask
- `G_long_L`: 车体纵向/前后方向梯度
- `G_lat_L`: 车体横向/左右方向梯度

这些图层是 `dem_loc` 的直接输入。后续 AI 修改算法时, 不要随意改名或改变含义。

### 2.5 常见修改入口

- 改输入/输出话题:
  - `FastLioBodyBev_ros1/config/body_bev.yaml`
  - `input_topic`, `odometry_topic`, `output_topic`
- 改 BEV 范围和精度:
  - `resolution`, `map_size_x`, `map_size_y`
- 改点云过滤:
  - `point_min_z`, `point_max_z`
  - `ego_filter_enabled`, `ego_box_*`
- 改地面估计:
  - `use_ransac_ground`
  - `near_inner_radius`, `near_outer_radius`
  - `ground_candidate_min_z`, `ground_candidate_max_z`
  - `ground_front_half_angle_deg`, `ground_require_forward`
  - `ground_min_points`, `ground_failure_fallback_z`
- 改高度/梯度平滑:
  - `height_quantile`
  - `h_rel_deadzone_half_width`
  - `grad_deadzone_half_width`
  - `enable_temporal_height_filter`
  - `enable_spatial_height_smoothing`
- 改 ROS 节点行为:
  - `BodyBevNode` 构造函数和两个回调
- 改 BEV 算法:
  - `BodyBevBuilder::build()`
  - `fillMapFromFrameHistory()`
  - `computeDirectionalGradients()`
  - `showDebugWindows()`

注意: `BodyBevNode` 源码支持读取的参数比当前 `body_bev.yaml` 显式写出的更多。若要调 `height_quantile`, `count_saturation`, `distance_decay_alpha`, `temporal_height_*`, `height_smoothing_*` 等参数, 可以在 YAML 中新增同名字段; 默认值以 `fast_lio_body_bev_node.cpp` 中 `pnh_.param(...)` 为准。

### 2.6 构建和运行

推荐运行:

```bash
cd /home/hsw/catkin_ws/program/FastLioBodyBev_ros1
./run_body_bev.sh
```

手动运行:

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
rosparam load /home/hsw/catkin_ws/program/FastLioBodyBev_ros1/config/body_bev.yaml /fast_lio_body_bev
/home/hsw/catkin_ws/program/FastLioBodyBev_ros1/bin/fast_lio_body_bev __name:=fast_lio_body_bev
```

## 3. GridMap_v7_5_ros1

### 3.1 作用

较完整的 LiDAR/GridMap 处理节点, 支持不同车型 LiDAR 话题和标定文件, 输出传统 terrain/color/obstacle 地图, 同时可输出和 `dem_loc` 对接的 LiDAR BEV `grid_map`。

主节点:

- 源码入口: `GridMap_v7_5_ros1/src/main.cpp`
- ROS 节点名: `GradMap`
- 主类: `SensorMap`
- 可执行文件: `GridMap_v7_5_ros1/bin/GridMap`

### 3.2 目录和文件

- `GridMap_v7_5_ros1/src/main.cpp`
  - 初始化 ROS 节点 `GradMap`, 创建 `SensorMap`, 调用 `sensor_map.output()`。
- `GridMap_v7_5_ros1/src/sensor_map.cpp`
  - 主控逻辑: 参数读取、订阅发布、点云处理线程、BEV worker、传统 gridmap 处理。
- `GridMap_v7_5_ros1/include/map/sensor_map.h`
  - `SensorMap` 类定义。
- `GridMap_v7_5_ros1/src/my_config.cpp`
  - 读取 `config/GridMapInit.ini`, 加载标定和运行参数。
- `GridMap_v7_5_ros1/include/map/my_config.h`
  - 参数字段、默认话题、坐标变换、显示开关。
- `GridMap_v7_5_ros1/src/lidar_bev_builder.cpp`
  - 该包内 LiDAR BEV 构建器实现。
- `GridMap_v7_5_ros1/include/map/lidar_bev_builder.h`
  - `LidarBevBuilder::Config`, `OdometryState`, `build()`, `publish()`。
- `GridMap_v7_5_ros1/src/gridmap_handle_v2.cpp`
  - 当前 `SensorMap::handle_points()` 中实际调用的传统栅格处理方案。
- `GridMap_v7_5_ros1/src/odom_stabilizer.cpp`
  - LocalPose/Odometry 平滑和限幅。
- `GridMap_v7_5_ros1/src/image_proj.cpp`
  - 相机去畸变和点云投影逻辑, 当前主流程中多处被注释关闭。
- `GridMap_v7_5_ros1/config/GridMapInit.ini`
  - 车型、显示、GridMap、LiDAR BEV、标定文件参数。
- `GridMap_v7_5_ros1/config/Calib/`
  - LM/HM/AX7 等标定文件。

未发现 `package.xml` 和 `.launch` 文件。当前是 CMake 工程风格, 但依赖 ROS catkin 包。

### 3.3 关键 ROS 话题

输入话题由 `choose_car` 和配置决定:

- `choose_car = LM`
  - LiDAR: `/sensor/RS128Points` 或 `/sensor/RS128Points_tztek`
- `choose_car = HM`
  - LiDAR: `/sensor/RS128Points_tztek`
- `choose_car = new_LM`
  - LiDAR: `/sensor/RSM1Points`
  - LiDAR: `/sensor/RSBPPoints`

通用输入:

- `/self_state/LocalPose`
- `/self_state/LidarLocalPose`
- `/imu/data`
- `/sensor/Odometry` 仅当 `lidar_bev_pose_source = odometry` 时订阅

输出:

- `/car_points`
- `/accumulated_points`
- `grid_map`
- `/world_state/TerrainMap`
- `/world_state/ColorMap`
- `/world_state/StaticObjArray`
- `/lidar_bev/grid_map`

### 3.4 核心流程

1. `main.cpp`
   - `ros::init(argc, argv, "GradMap")`
   - `SensorMap sensor_map`
   - `sensor_map.output()`

2. `SensorMap::output()`
   - 调用 `Init()`
   - 启动处理线程, 周期调用 `handle_points()`
   - 主线程 `ros::spinOnce()`

3. `SensorMap::Init()`
   - `params->readPara()` 读取 `GridMapInit.ini`
   - 根据 `choose_car` 订阅对应 LiDAR
   - 订阅 LocalPose/LidarLocalPose/IMU/Odometry
   - 初始化发布器
   - 用 `params` 填充 `LidarBevBuilder::Config`
   - 启动 LiDAR BEV worker

4. `SensorMap::handle_points()`
   - 从队列取 LiDAR 点云
   - 当前相机投影/着色大多关闭, 默认生成灰色 `PointXYZRGBValid`
   - 调用 `params->trans_color_lidar2car()` 转到车体系
   - 如果启用 `b_enable_lidar_bev`, 取得当前 BEV pose, 提交给 BEV worker
   - 调用 `grid_map_handler_v2.gridmap_process(colored_car_points, body_pose)`

5. `SensorMap::lidarBevWorkerLoop()`
   - 异步调用 `lidar_bev_builder.build(work.points, &work.pose)`
   - 发布到 `params->lidar_bev_topic`, 默认 `/lidar_bev/grid_map`

### 3.5 LiDAR BEV 配置项

文件: `GridMap_v7_5_ros1/config/GridMapInit.ini`

关键字段:

- `b_enable_lidar_bev`
- `b_show_lidar_bev_layers`
- `lidar_bev_pose_source`
  - 可选含义: `localpose`, `odometry`, `body`
- `lidar_odometry_topic`
- `lidar_bev_topic`
- `lidar_bev_frame_id`
- `lidar_bev_body_frame_id`
- `lidar_bev_resolution`
- `lidar_bev_map_size_x`
- `lidar_bev_map_size_y`
- `lidar_bev_point_min_z`
- `lidar_bev_point_max_z`
- `lidar_bev_update_rate_hz`
- `b_lidar_bev_enable_ego_filter`
- `lidar_bev_ego_box_*`
- `lidar_bev_ground_*`
- `b_lidar_bev_use_ransac_ground`
- `lidar_bev_ground_ransac_distance`
- `lidar_bev_ground_max_plane_tilt_deg`
- `lidar_bev_ground_fallback_quantile`
- `lidar_bev_height_quantile`
- `lidar_bev_h_rel_deadzone_half`
- `lidar_bev_grad_deadzone_half`
- `lidar_bev_cell_max_points`
- `lidar_bev_edge_min_valid_neighbors`

注意: `my_config.cpp` 支持读取的部分字段可能没有在当前 `GridMapInit.ini` 中显式写出。新增字段时使用同名 key, 默认值参考 `include/map/my_config.h`。

### 3.6 常见修改入口

- 改车型/输入 LiDAR:
  - `GridMap_v7_5_ros1/config/GridMapInit.ini`
  - `choose_car`, `LM_Lidar_Type`, 标定文件字段
  - `GridMap_v7_5_ros1/include/map/my_config.h`
  - `GridMap_v7_5_ros1/src/my_config.cpp`
- 改主流程:
  - `GridMap_v7_5_ros1/src/sensor_map.cpp`
  - `SensorMap::Init()`, `SensorMap::handle_points()`, `SensorMap::output()`
- 改 BEV 输出:
  - `GridMap_v7_5_ros1/src/lidar_bev_builder.cpp`
  - `GridMap_v7_5_ros1/include/map/lidar_bev_builder.h`
- 改传统栅格/terrain/obstacle:
  - `GridMap_v7_5_ros1/src/gridmap_handle_v2.cpp`
  - `GridMap_v7_5_ros1/include/map/gridmap_handle_v2.h`
- 改姿态平滑:
  - `GridMap_v7_5_ros1/src/odom_stabilizer.cpp`
  - `GridMap_v7_5_ros1/include/map/odom_stabilizer.h`
- 恢复相机投影:
  - `GridMap_v7_5_ros1/src/image_proj.cpp`
  - `SensorMap::Init()` 和 `SensorMap::handle_points()` 中目前注释掉的相机订阅/投影代码

### 3.7 与 FastLioBodyBev 的关系

`GridMap_v7_5_ros1/src/lidar_bev_builder.cpp` 和 `FastLioBodyBev_ros1/src/fast_lio_body_bev_node.cpp` 中的 BEV 构建逻辑高度相似, 但不是同一个源码文件。

后续如果修改 BEV 图层语义、梯度计算、mask 规则、地面参考规则:

- 如果系统可能同时使用两套 BEV 生成路径, 需要同步检查两边。
- `dem_loc` 只认最终 `/lidar_bev/grid_map` 的消息契约, 不关心来自哪个包。

## 4. dem_loc

### 4.1 作用

ROS1 Python 定位程序, 读取 DSM/DEM, 订阅车辆位姿和 LiDAR BEV GridMap, 做:

- GlobalPose / LocalPose / Odometry 轨迹绘制
- DSM-BEV 匹配评分
- 可选粒子滤波定位
- 输出粒子滤波后的 GlobalPose

包名:

- `loc_bev`

实际主要入口:

- `dem_loc/scripts/localization_python.py`

### 4.2 目录和文件

- `dem_loc/scripts/localization_python.py`
  - 主运行脚本, 节点名 `localization_python_node`。
- `dem_loc/scripts/debug_dsm_bev_score.py`
  - 等待 `/lidar_bev/grid_map`, 直接调试 DSM-BEV score。
- `dem_loc/scripts/test_cinterface.py`
  - 话题连通性调试工具, 打印 CInterface 订阅的消息摘要。
- `dem_loc/loc_tool/cinterface.py`
  - Python 版 ROS 订阅接口, 与 C++ `CInterface` 风格对应。
  - 解析 `/lidar_bev/grid_map`。
- `dem_loc/loc_tool/common_struct.py`
  - `InputData` 等数据容器。
- `dem_loc/loc_tool/dem_tool.py`
  - DEM/DSM GeoTIFF 读取和基础数据结构。
- `dem_loc/loc_tool/coord_converter.py`
  - WGS84/Gauss/pixel/local pose 坐标转换。
- `dem_loc/loc_tool/dsm_patch.py`
  - 从 DSM 裁剪和生成 BEV 对齐 patch。
- `dem_loc/loc_tool/dsm_bev_score.py`
  - DSM-BEV 单帧评分。
- `dem_loc/loc_tool/dsm_bev_score_batch.py`
  - Torch batch 评分。
- `dem_loc/loc_tool/dsm_bev_score_runner.py`
  - GlobalPose/LocalPose/Pose perturbation 评分封装。
- `dem_loc/loc_tool/dsm_bev_score_vis.py`
  - 评分可视化图像拼接。
- `dem_loc/loc_tool/particle_filter.py`
  - 粒子滤波核心。
- `dem_loc/loc_tool/particle_filter_runner.py`
  - 将粒子滤波和 ROS 位姿/BEV 消息连接起来。
- `dem_loc/config/particle_filter.ini`
  - 粒子滤波默认配置。
- `dem_loc/doc/`
  - DSM-BEV score 相关设计文档。

### 4.3 构建状态注意

`dem_loc/CMakeLists.txt` 定义了 C++ 可执行目标 `localization`, 并引用:

- `src/localization.cpp`
- `src/cpp/CInterface.cpp`
- `src/cpp/CommonStruct.cpp`
- `src/cpp/CoordConverter.cpp`
- `src/cpp/Tool.cpp`

但当前 `dem_loc` 目录下没有 `src/` 目录。后续 AI 修改或构建时需要注意:

- 当前实际可读、可运行的主链路是 Python 版本。
- 如果要恢复 C++ 目标, 需要先找回或重建 `dem_loc/src/`。
- 如果只改 Python 定位逻辑, 不要被 CMake 中的历史 C++ 目标误导。

### 4.4 主节点流程

`dem_loc/scripts/localization_python.py`

1. `main()`
   - `rospy.init_node("localization_python_node")`
   - `_ensure_track_draw_defaults()`
   - `_load_pf_ini_defaults()`
   - 创建 `PythonLocalizationNode`
   - 调用 `node.spin()`

2. `PythonLocalizationNode.__init__()`
   - 设置 DSM 路径:
     - `DEM_PATH = "/home/hsw/catkin_ws/doc/miluo_dsm.tif"`
   - 默认 BEV:
     - `DEFAULT_BEV_MAP_SIZE_X = 64.0`
     - `DEFAULT_BEV_MAP_SIZE_Y = 64.0`
     - `DEFAULT_BEV_RESOLUTION = 0.2`
   - 读取 ROS 私有参数和粒子滤波 INI 默认值
   - 加载 DSM
   - 创建 `CoordConverter`
   - 创建 `CInterface`
   - 创建 `DsmPatchCropper`
   - 如果启用粒子滤波, 创建 `ParticleFilterRunner` 并发布 `/pf_global_pose`

3. `PythonLocalizationNode.spin()`
   - 周期调用 `self.interface.ConvertToLocalData(self.input)`
   - 处理 `GlobalPose`, `LocalPose`
   - 同步最新 `LidarBevGridMap`
   - 调用 `_try_update_bev_frame()`

4. `_try_update_bev_frame()`
   - 解析 BEV 图层
   - 粒子滤波可用时调用 `ParticleFilterRunner.on_bev_message()`
   - 需要 score 可视化时调用 `score_global_and_local()` 或 `score_at_pose()`
   - 可选保存 debug 图像

### 4.5 CInterface 订阅话题

文件: `dem_loc/loc_tool/cinterface.py`

`TOPIC_DEFINITIONS` 包含:

- `/self_state/GlobalPose`
- `/self_state/LocalPose`
- `/self_state/LidarLocalPose`
- `/world_state/ColorMap`
- `/world_state/EntityMap_false`
- `/world_state/TerrainMap`
- `/lidar_bev/grid_map`
- `/world_state/SemanticMap`
- `/world_state/SimilarityMap`
- `/behavior/ReferencePath`
- `/Odometry`
- `/kitti/oxts/gps/inspvax`
- `/navsat/odom`
- `/kitti/oxts/gps/fix`

`parse_lidar_bev_grid_map(msg)` 输出字典键:

- `H_L`
- `Gx_L`
- `Gy_L`
- `M_obs`
- `H_rel_surf`
- `G_long_L`
- `G_lat_L`
- `M_L`

### 4.6 粒子滤波配置

文件: `dem_loc/config/particle_filter.ini`

关键配置:

- `enable`
- `num_particles`
- `random_seed`
- `device`
- `init_std_x`, `init_std_y`, `init_std_yaw`
- `motion_std_x`, `motion_std_y`, `motion_std_yaw`
- `score_is_cost`
- `score_temperature`
- `estimate_mode`
- `elite_top_fraction`
- `resample_threshold`
- `enable_roughening`
- `save_debug`
- `debug_dir`
- `pf_update_every_n`
- `pf_log_every_n`
- `pf_pose_topic`, 默认 `/pf_global_pose`

### 4.7 常见修改入口

- 改 DSM 路径或默认 BEV 尺寸:
  - `dem_loc/scripts/localization_python.py`
  - `DEM_PATH`, `DEFAULT_BEV_*`
- 改订阅话题:
  - `dem_loc/loc_tool/cinterface.py`
  - `TOPIC_DEFINITIONS`
- 改 BEV 图层解析:
  - `dem_loc/loc_tool/cinterface.py`
  - `extract_grid_map_layer()`
  - `parse_lidar_bev_grid_map()`
- 改 DSM crop 几何和 BEV 对齐:
  - `dem_loc/loc_tool/dsm_patch.py`
  - `DsmPatchCropper`
- 改 score 函数:
  - `dem_loc/loc_tool/dsm_bev_score.py`
  - `dem_loc/loc_tool/dsm_bev_score_batch.py`
  - `dem_loc/loc_tool/dsm_bev_score_runner.py`
- 改粒子滤波:
  - `dem_loc/loc_tool/particle_filter.py`
  - `dem_loc/loc_tool/particle_filter_runner.py`
  - `dem_loc/config/particle_filter.ini`
- 改可视化:
  - `dem_loc/scripts/localization_python.py`
  - `dem_loc/loc_tool/dsm_bev_score_vis.py`
- 调试话题连通:
  - `dem_loc/scripts/test_cinterface.py`
- 调试 DSM-BEV 单帧评分:
  - `dem_loc/scripts/debug_dsm_bev_score.py`

## 5. 跨包接口契约

### 5.1 `/lidar_bev/grid_map`

生产者:

- `FastLioBodyBev_ros1`
- `GridMap_v7_5_ros1`

消费者:

- `dem_loc`

消息类型:

- `grid_map_msgs/GridMap`

必须保留的图层:

- `H_rel_surf`
- `M_L`
- `G_long_L`
- `G_lat_L`

几何字段:

- `msg.info.length_x`
- `msg.info.length_y`
- `msg.info.resolution`

`dem_loc` 会从消息同步 BEV 几何。如果生成端改了分辨率或尺寸, 通常不需要手改 `dem_loc` 默认值, 但要确保消息字段正确。

### 5.2 姿态和坐标系

常见 frame:

- `body`
- `map`
- `camera_init`

`FastLioBodyBev_ros1`:

- 默认 `body_frame_id = body`
- 默认 `odometry_frame_id = camera_init`
- 实际发布 GridMap frame 为 `body_frame_id`

`GridMap_v7_5_ros1`:

- `lidar_bev_pose_source = localpose` 时使用 `/self_state/LocalPose`
- `lidar_bev_pose_source = odometry` 时使用 `/sensor/Odometry`
- `lidar_bev_pose_source = body` 时使用 body frame identity

`dem_loc`:

- GlobalPose 默认按 math 约定: 东向 0 度, 逆时针
- LocalPose heading convention 可由 ROS 参数控制

后续改 heading/yaw 时, 必须同时检查:

- `dem_loc/scripts/localization_python.py`
- `dem_loc/loc_tool/coord_converter.py`
- `dem_loc/loc_tool/dsm_bev_score_runner.py`
- `dem_loc/loc_tool/particle_filter_runner.py`
- `GridMap_v7_5_ros1/src/odom_stabilizer.cpp`

## 6. 后续 AI 编程建议

### 6.1 首先确认要改哪条链路

修改前先判断当前任务是:

- 改 FAST-LIO body 点云到 BEV: 优先看 `FastLioBodyBev_ros1`
- 改原始 LiDAR/GridMap 多车型链路: 优先看 `GridMap_v7_5_ros1`
- 改 DEM 匹配、score、粒子滤波、可视化: 优先看 `dem_loc`
- 改 `/lidar_bev/grid_map` 的图层语义: 必须同时看生产者和 `dem_loc` 消费者

### 6.2 不要优先改构建产物

忽略这些目录和文件:

- `build/`
- `bin/` 中的二进制
- `__pycache__/`
- `scripts/output/`

只改源码、配置和文档。

### 6.3 高风险修改点

- 改 `H_rel_surf`, `M_L`, `G_long_L`, `G_lat_L` 名称或含义
- 改 grid_map 数据 reshape 方式
- 改车体坐标轴方向或 yaw convention
- 改地面参考估计逻辑
- 改 `lidar_bev_pose_source`
- 改 `dem_loc` 粒子滤波 `score_is_cost` 或 `score_temperature`
- 恢复 `dem_loc` C++ 目标但未补齐 `src/`

### 6.4 建议验证命令

检查输入:

```bash
rostopic hz /cloud_registered_body
rostopic hz /Odometry
rostopic hz /sensor/RS128Points_tztek
rostopic hz /self_state/LocalPose
```

检查 BEV 输出:

```bash
rostopic hz /lidar_bev/grid_map
rostopic echo -n 1 /lidar_bev/grid_map/info
```

检查 dem_loc 粒子滤波输出:

```bash
rostopic hz /pf_global_pose
```

无显示器或 SSH 环境:

- `FastLioBodyBev_ros1/config/body_bev.yaml`: 设置 `show_windows: false`
- `GridMap_v7_5_ros1/config/GridMapInit.ini`: 关闭显示开关
- `dem_loc`: 设置私有参数 `show_window:=false`

## 7. 快速搜索关键词

后续 AI 可直接搜索这些关键词定位代码:

- `BodyBevNode`
- `BodyBevBuilder`
- `cloudCallback`
- `odometryCallback`
- `SensorMap::Init`
- `SensorMap::handle_points`
- `SensorMap::lidarBevWorkerLoop`
- `LidarBevBuilder`
- `gridmap_process`
- `parse_lidar_bev_grid_map`
- `DsmPatchCropper`
- `compute_dsm_bev_score`
- `score_global_and_local`
- `ParticleFilterRunner`
- `ParticleFilter`
- `TOPIC_DEFINITIONS`
- `H_rel_surf`
- `M_L`
- `G_long_L`
- `G_lat_L`
