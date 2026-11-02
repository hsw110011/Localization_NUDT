我用的kitti raw的bag是以东向为0度的，逆时针为正。
amzith是以degree(度°)为单位的。


## 仅使用里程计的运动传播

sem_loc 的运动传播只使用 `/lio/odom`，不依赖 `/self_state/LocalPose`。
初始化等待里程计和 `/kitti/oxts/gps/inspvax`，用 GNSS 设置初始位置和地图航向。
KITTI 当前 INSPVAX 的 azimuth 按度、东向为零、逆时针为正解释。
旧 bag 使用 `/Odometry` 时显式传 `_odom_topic:=/Odometry`。

```text
dp_body = R(-yaw_odom_prev) * (p_odom_now - p_odom_prev)
dyaw = wrap(yaw_odom_now - yaw_odom_prev)
p_particle_now = p_particle_prev + R(yaw_particle_prev) * (dp_body + noise_body)
yaw_particle_now = wrap(yaw_particle_prev + dyaw + noise_yaw)
```

当前统一入口的所有 method/backend 都搜索 `x/y/yaw`。`Motion2D.h` 提取相对里程计增量，
每个粒子用自己的旧航向旋转位移，再累计相对转角；不会用里程计绝对航向覆盖所有粒子。
GNSS 只设置初始位置/航向，之后不依赖 LocalPose 或后续 GNSS 来传播 PF。
沿用旧版初始分布：x/y 标准差各 4m，yaw 标准差 8 度；旋转噪声为
`0.15*abs(delta_yaw_deg) + distance_m` 度，前向/侧向噪声系数保持不变。
纯旋转保留角度不确定性但不增加 XY 抖动；静止不增加噪声。航向估计使用加权圆均值。
UTM 状态保留 double 精度，地图评分和重采样仍共用同一实现。
baseline、无熵和有熵评分，以及 GPU/forloop 后端均使用同一个 ParticleFilter 实现。

初始化使用最近收到的 GNSS 和里程计，未增加时间戳同步或安装外参补偿。
此实现属于 sem_loc；dem_loc 的运动模型没有修改。

参考实现：
- [Nav2 AMCL 相对运动采样](https://github.com/ros-navigation/navigation2/blob/main/nav2_amcl/src/motion_model/differential_motion_model.cpp)
- [GTSAM Pose2 离散位姿组合](https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Pose2.h)

编译与回归测试：

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
cd /home/hsw/catkin_ws/program/sem_loc
cmake --build build --target bev sem_localization -j2
cmake --build build --target test_motion2d test_particle_motion test_scoring_routes test_xmlrpc_retry -j1
cd build
ctest --output-on-failure
```

纯运动测试与 GTSAM 的 between/compose 对照，覆盖坐标原点/朝向变化不变性、转弯、跨 ±180°、上一帧航向使用顺序。统一入口的 PF 测试覆盖三自由度传播、每粒子航向、UTM 小位移、角度跨界、重采样，以及有熵/无熵对真实后验和航向估计的影响。旧 XY 独立实现及其专属测试已删除。

## 在同一 CPP 内通过 config 选择方法

直接编辑项目的 `config/sem_loc.yaml`。`bev` 与 `sem_localization` 在启动时直接读取此文件，
不需要 launch，也不需要 `rosparam load`；修改后重启程序即可。

- baseline：`bev.mode: certain`，`particle_filter.method: baseline`。
- 本文方法无熵：`bev.mode: uncertain`，`particle_filter.method: ours_no_entropy`。
- 本文方法有熵：`bev.mode: uncertain`，`particle_filter.method: ours_entropy`（默认）。

三种评分函数及 switch 路由全部在 **`src/cpp/ParticleFilter.cpp`**：
`BaselineScore` 使用三类 one-hot × 有效观测 mask，`NoEntropyScore` 使用概率 × mask，`EntropyScore` 使用
`p * (1-H) * mask`。三种方式共用地图采样和贝叶斯权重更新。
`src/sem_localization.cpp` 根据同一配置选择原硬标签或概率话题，不需要改注释或更换程序。
当前统一入口已恢复三自由度粒子搜索，里程计仅提供相对增量；UTM 双精度与运动测试保留。
旧独立定位入口和重复滤波源码已删除；三种评分统一运行 `sem_localization`。

两个 BEV 模式均在 **`src/semantic_bev.cpp`**，前三通道固定为 road / building / plant，mask 放在最后：

- `/Semantic_Bev/ClassMask`：`8UC4`，`[road, building, plant, mask]`，值为 0/1。
- `/Semantic_Bev/Probs`：`32FC5`，`[p_road, p_building, p_plant, entropy, mask]`。

mask=1 表示有仍保留在空间窗口内的 LiDAR 栅格观测；mask=0 时所有通道为零。
label=3（others）或三类概率全零仍可有几何观测，此时 mask=1、前三通道为零，评分贡献为零。
上游未归入这三类的概率质量保留，不强行把三个概率归一化为 1；entropy 按三类条件概率计算并归一化到 [0,1]。
输出沿用原来的平面 yaw 图像约定和 `base_link` 标记，header.stamp 使用匹配后的点云采集时间。
未移植 lio_bev 的高度/倾斜稳定器；这次修正的是语义投影、时间匹配和累计，不是地形高度建图。

2DPASS 的 Python、C++ 和概率混淆中继已统一为 road / building / plant，
`probability_channels: [0,1,2]` 不再交换通道。旧版已记录 bag 若仍是 road / plant / building，
读取旧 bag 时才使用 `[0,2,1]`；它不会自动随新代码改变。
`visualize: 0` 可关闭 BEV 窗口，`1` 开启。
配置用 OpenCV FileStorage 读取，保留文件开头 `%YAML:1.0`；带行尾注释的字符串保留引号。
定位消费者已同步为上述新编码，并在所有评分中使用 mask，概率图缩放时 mask 使用最近邻插值。
定位统一使用 `sem_localization`。

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
cd /home/hsw/catkin_ws/program/sem_loc
cmake --build build --target bev sem_localization -j2
# 两个终端分别运行（ROS master 应已启动）
./build/bev
./build/sem_localization
# 可选：改用另一份配置，两个进程指定同一个路径
./build/bev _config:=/absolute/path/sem_loc.yaml
./build/sem_localization _config:=/absolute/path/sem_loc.yaml
```

里程计话题参数继续沿用上面的运动传播说明；本次不改变地图 NPZ 输入方式。
除原运动测试外，`test_scoring_routes` 覆盖三种评分的 CPU/CUDA 数值与真实 Update 路由；
`bev_config` 测试在独立 ROS master 下验证直接读取配置、通道与 mask、时间插值、空间窗口累计、长时间保留、窗口移出清理、时间回退和非法配置。
所有测试可用 `/tmp` 下的独立构建目录执行，避免覆盖已有 build 产物。

BEV 当前订阅 `/lio/odom` 与 `/lio/cloud_world`。`input_cloud_in_world: 1` 时要求点云 frame 与里程计父 frame 一致，
直接使用 world 点，避免重复变换；使用已经变换到 body 的点云时设置为 0，并按采集时刻的完整里程计姿态转到 world。
里程计缓冲按点云时间插值（最大间隔 0.20 秒），最多等待 0.05 秒，最近邻容差 0.03 秒；不再使用任意最新姿态。

累计参考 lio_bev 的 world 固定栅格、有限历史和关键帧更新原则：

- 每帧先按格分组，只在该格最高表面下 `surface_band_m` 的范围内聚合语义。
- 每帧每格仅贡献一次；certain 对帧内类别投票，uncertain 对帧内概率求均值，再对最近帧等权融合。
- 地图按空间窗口持续累计，不按秒数或总帧数过期；窗口内未再次观测到的格子仍保留，移出滚动存储范围才清理。
- 每格最多保留 `history_frames` 个融合样本，它只限制每格的内存和融合权重，不会删除未观测到的格子。
- 位移、转角、时间任意一个达到阈值才累计新关键帧；参数沿用 lio_bev 的 0.5m / 2° / 0.5s 默认值。
- 内部 world 存储范围覆盖旋转视野的对角线，移动时清除超出范围的格子；时间回退或 world frame 改变时清空地图。
- 输出尺寸始终为 `view_range / resolution`，没有“攒满窗口才发布”的门槛，未观测像素的 mask=0。
- 每个输出像素使用连续车位反查 world 栅格，不反复旋转历史图、不用整数截断车位；mask 始终二值。

语义 BEV 当前为 `60 / 0.15 = 400×400`；lio_bev 当前为 `64 / 0.2 = 320×320`。
两者的像素数量可以不同，空间保留原则相同；本次保持语义 BEV 原有尺寸不变。
`accumulation_max_interval_sec` 是关键帧更新触发时间，不是历史删除时间。
只有收到新的、可匹配的点云才发布，不再用当前时间重复发布旧图。启动会打印模式和输入话题，失败会直接打印原因。

CMake 默认只编译 `bev` 和 `sem_localization` 两个运行入口。
`test_motion2d`、`test_particle_motion`、`test_scoring_routes` 和 `test_xmlrpc_retry` 是按需构建目标，
另有 `bev_config` ROS 集成测试。测试仅针对统一实现。

## GPU / for-loop 计算方式

`particle_filter.backend` 独立于 `method`，三种评分方法都支持以下两种计算方式：

```yaml
particle_filter:
  backend: "gpu"          # "gpu" 或 "forloop"
  method: "baseline"      # 或 "ours_no_entropy" / "ours_entropy"
```

- `gpu`：保持现有 LibTorch 批量 grid_sample；CUDA 不可用时会提示并使用 CPU 批量计算。
- `forloop`：CPU 上逐粒子进行原 OpenCV warpAffine 采样和代价计算，TDF 保存在 CPU 内存。

两条分支与三种评分均在 `ParticleFilter.cpp`，共用 x/y/yaw 运动模型、相对里程计增量、
评分权重、贝叶斯更新和重采样。有熵模式在两个 backend 均采用 `p*(1-H)`。
OpenCV 与 LibTorch 的双线性插值实现存在精度差异，不保证逐位一致。
CPU 路径的计时不再同步 CUDA。

本次代码更新需要首次重新编译 `bev` 和 `sem_localization`；之后修改 backend/method 只需保存
config 并重启相应程序，不需要再次编译。启动定位时会打印 backend 和实际 device。

## BEV 帧率与编译优化

未指定 `CMAKE_BUILD_TYPE` 时，CMake 默认使用 Release（启用优化）；已有显式 Debug 等设置不会被覆盖。
默认 `history_frames: 5`，仍仅限制每格融合样本数，空间窗口内的地图不会因时间或帧数被清除。
类别均值与 entropy 仅在该格获得新观测时更新，输出时读取缓存，避免反复扫描所有格子的历史或重复计算 entropy。

BEV 继续对新点云发布，不做固定频率的旧图重发；真实输出达到 10 Hz 需要输入点云本身至少 10 Hz。
如果开启监视窗口后帧率下降，可用 `visualize: 0` 对照定位 GUI 的开销。


## 定位打印与窗口

定位启动时打印 `method=baseline`、`ours_entropy (有熵)` 或 `ours_no_entropy (无熵)`，并标明 backend。
运行时每次里程计更新仅打印 `weight_max` 和 `weight_min`，它们是当前归一化粒子权重的极值，
取值时机在观测更新之后、重采样之前；没有有效 BEV 时仍打印当前预测粒子的权重。
正常的地图加载、计时和误差调试输出不再刷屏；真实错误、CUDA 不可用等警告仍保留。
定位显示 `Satellite Map`（黑色 Odom、白色 GPS、黄色 PF）与 `PF Particle Weights` 两个窗口。
粒子窗口显示重采样前的位置分布、权重颜色、Neff 和权重统计；只显示，不逐帧保存 PNG。
不弹出 TDF 或语义局部图。
同一时间戳的 BEV 只更新一次权重，后续只有里程计到来时照常传播，不重复使用旧观测计权。
CSV 轨迹记录与原有内部融合计算保留。

BEV 启动时打印实际发布的 `output`（含话题重映射结果）与 `encoding`。
`XmlRpcDispatch::work: error in poll (-1)` 是 ROS XML-RPC 的系统调用失败，`-1` 不是 errno。
当前安装的 xmlrpcpp 1.17.4 存在 accept 重试后重复注册监听项的问题：监听项持续增加可导致
`poll()` 的数量超过软 FD 限制，报 `EINVAL`。项目内 `third_party/xmlrpcpp` 使用同版本源码，
仅修复一次性重试注册，以及 EAGAIN/EINTR 的非阻塞 accept 处理；不替换 `/opt/ros` 系统库。
运行时请保留 `build/sem_loc_ros_lib`，两个程序通过 RPATH 使用其中的修补库。
BEV 已在原错误后补充 `errno=数字 (原因)`；其他 XML-RPC 错误仍由原处理器报告。
仅凭旧日志无法判断具体原因；仍需复现时可用以下命令记录系统调用的真实错误（不修改 ROS 库）：

```bash
cd /home/hsw/catkin_ws/program/sem_loc/build
strace -f -tt -e trace=poll,ppoll,accept,accept4 -o /tmp/sem_loc_bev_xmlrpc.log ./bev
```

在日志中查找失败行末尾的 `EINVAL`、`ENOMEM` 等实际 errno，再针对原因处理。


## 统一观测模型：线性 TDF 直接求和

`D = clamp(raw_TDF / tdf_value_scale, 0, 1)`。当前 NPZ 保存截断在 100 像素的原始距离，
因此 `tdf_value_scale: 100.0`；若输入已经归一化，明确设置为 1，不根据地图观测到的最大值猜测尺度。
原 NPZ 不会被程序改写。类别通道固定为 road / building / plant。

三种观测权重：baseline 为 `A*M`，无熵为 `A*P`，有熵为 `A*P*(1-H)`。
每个粒子的分数均为 `S = sum(weight * D)`，对应位置/类别通道逐元素相乘后求和，
不对 TDF 平方，不除以观测数量或概率/置信度总和。A 为最后通道的二值有效观测 mask。
地图外的采样点仍保留，三个 TDF 通道均取 1；GPU 与 forloop 包含严格的近边界越界检查。

H 是三类条件概率 q=P/sum(P) 的归一化熵 `-sum(q*log(q))/log(3)`；0 概率按 0log0=0 处理。
原始 P 保留三类概率质量，sum(P) 不足 1 的部分不强行补入；H=1 的点贡献为零。
无观测或所有贡献为零时，跳过观测更新并保持先验。

`log_w_post = log(clamp(w_prior, 1e-12)) - score_lambda * S`，减去最大 log 权重后
指数归一化。默认 `score_lambda: 0.001` 为直接求和评分的调参起点，非已验证最优值。
旧 `sigma_obs`、`entropy_scale` 配置项已删除；自定义配置请同步迁移到新参数。
三种评分和两个 backend 使用同一套公式；x/y/yaw 搜索、里程计输入和粒子窗口保持现有行为。


## 统一程序命名

定位入口源文件为 `src/sem_localization.cpp`，可执行文件和 ROS 节点名均为 `sem_localization`。
统一粒子滤波实现在 `src/cpp/ParticleFilter.cpp` 和 `src/include/ParticleFilter.h`。
`bev` 继续通过配置选择 certain/uncertain；定位继续通过同一配置选择三种评分和两种后端。
项目目录 `sem_loc`、ROS 包名 `loc_bev`、配置 `config/sem_loc.yaml` 与传感器话题保持现有名称。
新定位结果写入 `output/pose_compare_sem_localization.csv`；原有历史结果文件保留。
