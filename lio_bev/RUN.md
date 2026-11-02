# lio_bev

接收 LIO 点云与里程计，发布以车辆为中心的局部高程 BEV。节点没有内部窗口、绘图或坡度计算，查看结果使用 RViz 的 GridMap 显示插件。

## 输入输出

当前 `config/lio_bev.yaml`：

- 点云：`/lio/cloud_world`，`sensor_msgs/PointCloud2`，世界坐标输入。
- 里程计：`/lio/odom`，`nav_msgs/Odometry`。
- 输出：`/lidar_bev/grid_map`，`grid_map_msgs/GridMap`，当前配置 frame 为 `bev_local`，时间戳为输入点云时间。
- 范围 64 × 64 m，分辨率 0.2 m。

保留 `dem_loc` 使用的四层：`H_stable_surf`、`M_L`、`G_long_L`、`G_lat_L`。独立的 `S_local_deg` 坡度层已移除，不影响高度层和纵横梯度的构造。

世界点云的 header.frame_id 必须与里程计的父 frame 一致；里程计描述的刚体必须与配置的 body frame 相符。节点不自动查询输入 TF 或应用传感器外参。使用 `input_cloud_in_world: false` 时，输入应已经在此刚体坐标系中。改变输入源之前请核对 frame 与外参。

## 编译与运行

在本目录执行，或将路径替换为当前 checkout。需要 ROS Noetic、Eigen3 和 grid_map 依赖，节点不再直接依赖 PCL/OpenCV。grid_map_ros 本身仍可能间接依赖图像相关库。

```bash
source /opt/ros/noetic/setup.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./run_lio_bev.sh
```

产物为 `bin/lio_bev`。脚本按自身位置发现包，加载 ROS 环境，并在存在时加载 `/home/hsw/catkin_ws/devel/setup.bash`。也可用 `roslaunch lio_bev lio_bev.launch`，或给脚本传入 `config:=/absolute/path/to/config.yaml`。

同名包可能同时存在于主目录和 Codex worktree；两份目录的源码、配置及编译产物彼此独立。脚本现在校验 ROS 解析到的包与自身目录一致，并打印项目和可执行文件路径。日常从 `/home/hsw/catkin_ws/program/lio_bev/run_lio_bev.sh` 启动时，应在该目录编译。修改 worktree 不会自动更新主目录；脚本也不会自动重新编译。

需要把编译产物放在工作区之外时，配置 `-DLIO_BEV_RUNTIME_OUTPUT_DIRECTORY=/absolute/output/path`；运行脚本仍期望默认的 bin 路径，外置产物可直接启动。

## 实时处理和同步

- 两个 ROS 回调线程只接收数据；单独的工作线程独占 BEV 构建器。共享点云槽位、位姿缓存由同一互斥锁保护，重计算时不持锁。
- ROS 点云订阅队列与待处理槽位容量均为 1。过载时覆盖未处理旧帧，不堆积历史输入；不能承诺算力不足时仍处理每帧。
- `update_rate_hz: 0.0` 默认处理每个可用输入。正值对工作线程按单调时钟限频，不再根据相邻消息是否不足 100 ms 丢帧。
- 点云优先使用同时间戳位姿；有前后位姿时平移线性插值、旋转 slerp。最多等待 `odometry_wait_sec: 0.05` 秒；插值两端最大跨度 `odometry_max_gap_sec: 0.20` 秒。
- 超时后允许使用时间差不超过 `odometry_nearest_tolerance_sec: 0.03` 秒的最近位姿；设为 0 可要求精确匹配或插值。拒绝过期、无效或 frame 不一致的数据。
- 位姿缓存同时限制为 3 秒、400 条。时间倒退/坐标系切换清理相关历史；关闭节点时唤醒并 join 工作线程。
- 参数在启动时读取，修改后需重启。

## 地图算法

世界坐标滚动栅格保存校正后的表面高度，每格最多保留 `corrected_sample_capacity: 32` 个蓄水池代表样本，取最高 `corrected_surface_top_fraction: 0.05` 比例的均值。

高度/倾斜稳定化保留每个历史车体栅格的最低代表点，最多 `stabilization_history_frames: 10` 帧；采用鲁棒平面残差拟合，并检查求解退化。相比重复变换所有历史原始点，这是对历史观测的栅格摘要，在边缘和遮挡变化处仍需实测评估。

达到任一条件时写入新观测：移动 `accumulation_update_distance_m: 0.5` m、转动 `accumulation_update_yaw_deg: 2.0` 度、经过 `accumulation_max_interval_sec: 0.5` 秒。避免静止或原地旋转时长期不更新。任一阈值设为 0，都会使该条件每帧触发。

实际三维表面格投影到车体平面，栅格化其有限面积覆盖，避免零高度逆查在 roll/pitch 下错位，以及只投中心点导致的旋转空洞。多个表面覆盖同一格时取上表面。以发布图中最近中心的有效格为相对高度零点，再计算原有中值滤波与 Sobel 纵横梯度；可选的空间高度平滑仍保留。

中心基准使用最近中心有效格的高度作为候选值。候选值与上一帧实际使用的基准相差超过 1 m 时，保持上一帧基准；不超过 1 m（含恰好 1 m）时直接更新。连续超限也一直保持，不逐步靠近候选值。首个有观测的帧直接初始化，空图保留上次有效基准，时间倒退、里程计父坐标系切换或地图重新初始化时重新建立基准。TF 同步使用最终基准，仍可还原世界表面位置。此限制只约束整图减去的标量，不限幅单格高度或压平真实地形；拒绝更新时中心格不一定为零。

旧的 `show_windows`、`debug_window_stride`、`slope_*`、`use_ransac_ground`、旧地面 ROI 参数、未实现的 `enable_temporal_height_filter` 及相关参数已移除；旧 YAML 中残留这些键不会生效。

## 性能日志与 RViz

`performance_log_interval_sec: 5.0` 控制 `BEV perf` 日志，设为 0 关闭。包括发布频率、累计收帧数、槽位覆盖数、同步失败数、无效帧数，以及平均/最大处理耗时、排队和同步等待、解码、构建、发布和构建子阶段耗时，单位 ms。`overwritten` 仅统计节点槽位覆盖，不包括 ROS 传输或订阅队列之前的丢帧。

```bash
rostopic hz /lio/cloud_world
rostopic hz /lio/odom
rostopic hz /lidar_bev/grid_map
rostopic info /lidar_bev/grid_map
```

RViz 添加 GridMap 显示并选择输出话题，可切换高度和梯度层。当前模式 Fixed Frame 设为 `bev_local` 或输入里程计的父 frame（Super-LIO 为 `world`）。节点会发布父 frame 到 bev_local 的同时间戳 TF，不依赖 LIO 是否提供 body TF。

## 路面连续性诊断（独立可选节点）

`surface_diagnostics.py` 仅订阅主 BEV，以默认 1 Hz 对车辆前后 15 m、左右 1.5 m 的路面 ROI 拟合鲁棒参考平面。它不弹窗口，不改变主 BEV、累计算法或定位特征。ROI 只是几何筛选，不是语义地面分割；弯道/路边障碍较多时需要调整范围。

主节点已经运行时，直接另起诊断，无需重启主节点：

```bash
python3 scripts/surface_diagnostics.py
```

后续也可以随主节点启动：`./run_lio_bev.sh diagnostics:=true`。独立 launch 支持 ROI 与速率参数：

```bash
roslaunch lio_bev surface_diagnostics.launch x_min:=-15 x_max:=15 y_min:=-1.5 y_max:=1.5
```

诊断输出：

- `/lidar_bev/debug/road_points`：实际发布高度的路面 ROI 点云。RViz 使用 XYZ / FlatColor（白色），排除彩虹配色干扰。
- `/lidar_bev/debug/plane_points`：同坐标与时间戳下的拟合平面，建议 FlatColor（橙色）对照。
- `/lidar_bev/debug/residual_points`：点的 z 为相对参考平面的偏差，单位 m；连续斜面的整体倾斜已消除，真实台阶/凸起仍保留。RViz 可用 Intensity 着色，固定范围 -0.15～0.15 m。
- `/lidar_bev/debug/grid_map`：独立诊断图层 `height`、`plane`、`residual`、`neighbor_jump`；仅 ROI 有效。可以 Height Layer=height、Color Layer=residual，或 Height Layer=residual 查看残差几何。
- `/lidar_bev/debug/stats`：JSON，含平面系数、坡度、残差 p50/p95/max、内点率、相邻格残差跳变，以及每 1 m 的纵向高度/残差中位数剖面。

配套 RViz 配置只显示诊断话题，默认采用白色实际几何；通过勾选切换橙色参考平面、残差显示，建议逐项单独看：

```bash
rviz -d rviz/surface_diagnostics.rviz
rostopic echo /lidar_bev/debug/stats
```

纯斜面即使颜色变化，平面残差仍接近零；若残差出现分块或邻接跳变，说明实际高度数据不共面。残差大可能来自真实路面、障碍或累计/稳定化误差，单靠此诊断不能判定来源，也不会自动修平地图。所有显示沿用输入 GridMap 的 frame；默认 RViz 配置使用 bev_local，不修改主话题的颜色或高度数据。

没有新 BEV 时诊断节点等待输入；输出有 latch，新订阅者可收到诊断节点存活期间的最后一帧。停止诊断使用 `rosnode kill /lio_bev_surface_diagnostics`。

## OpenCV 高度热力图（独立可选窗口）

需要直观看完整高度分布时，可另开窗口，不重启主节点：

```bash
python3 scripts/height_viewer.py
```

默认显示完整 `H_stable_surf`：固定色标 -2～+2 m（右侧色条，超范围饱和并显示比例），未观测格为黑色，使用最近邻放大保留原始格子。图像上方为车体前方 +X，左侧为 +Y；鼠标查看实际 x/y/高度数值。底部曲线为 |y|≤1 m 中心道路带的逐行高度中位数，横轴是车体 x（m）、纵轴为高度（m），缺失数据不连线。整图、剖面均不修改高度值。

按 A 切换 2%～98% 自动色标，R 恢复启动时的固定色标，[ / ] 缩小/扩大色标范围，S 保存当前图片到 `/tmp/lio_bev_height_viewer`，Q/Esc 退出。对比相邻帧建议保持固定色标。画面时间戳及接收超时提示用于区分实时数据与暂停后的最后一帧。

启动参数可直接指定：

```bash
python3 scripts/height_viewer.py _height_min:=-1.0 _height_max:=1.0 _rate_hz:=5.0
# 或与主节点一起启动窗口：
./run_lio_bev.sh height_viewer:=true
# 单独启动窗口的 launch：
roslaunch lio_bev height_viewer.launch height_min:=-1.0 height_max:=1.0
```

窗口在独立进程中默认 5 Hz 刷新，主 C++ 节点仍不包含绘图。需要 Python OpenCV 的桌面 GUI 支持和 DISPLAY/Wayland。也会发布 `/lidar_bev/debug/height_image`（bgr8，带色条与剖面），可用 `_show_window:=false` 只发布图像话题。没有新 BEV 时窗口显示等待，或保留有时间标识的最后一帧。

## 消除传感器姿态引入的表观坡度

当前 YAML 启用 `gravity_aligned_output: true`，`bev_frame_id: bev_local`、`publish_bev_tf: true`。假定输入 LIO 的 world Z 已完成重力对齐（当前 Super-LIO 初始化如此）。

点云配准、车体过滤、历史稳定化、世界滚动累计、32 个代表样本的最高 5% 均值全部继续使用原有完整位姿与算法。仅将最终 BEV 的参考方向改为当前 yaw，Z 与 world Z 一致，因此 IMU 的 roll/pitch 不再把世界水平路面变成前后/左右斜面；真实道路坡度和障碍物仍保留。中心参考高度减法不变，不做道路平面拟合压平。

自动发布的 TF 使用实际里程计父 frame（通常 world），XY 在车辆位置，Z 在本帧选中的表面参考高度，旋转仅含 yaw。这样相对高度与 TF 的组合可还原世界表面位置，不会用倾斜的 body TF 再次引入坡度。输入数据中真实存在的高度漂移、错误外参或最高样本偏差不会被此坐标修复自动消除。

四层名称、话题、分辨率和数据排列保持不变。dem_loc 的 DSM patch 本来按 yaw 采样，使用重力方向高程可与它对齐；本次不修改其评分或粒子滤波。

需要复现旧输出时，在 YAML 中设置 `gravity_aligned_output: false` 后重启，消息恢复 `body_frame_id`，且节点不发布 BEV TF。C++/裸节点的参数回退默认保留旧模式，正式运行请加载随项目提供的 YAML。切换模式必须重启，不能只修改运行中 rosparam。

