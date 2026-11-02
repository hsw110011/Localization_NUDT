# Python 定位程序的性能与优化思路

## 当前实现

默认使用 128 个粒子。每个粒子仍使用完整的 320×320 BEV 网格，评分公式、运动模型和最终融合方式保持不变。减少粒子会减少对位姿空间的采样数量，因此性能收益伴随着采样覆盖减少，定位精度需要结合实际回放评估。

全局轨迹位于主窗口右侧，粒子分布仍为第二个窗口。底图在启动时按完整 DSM 像素着色并缓存，GPS、LocalPose、OdomDR 和 PF 轨迹保留完整地图坐标及亚像素位置。右侧固定宽 520 像素的是可缩放视口，不是底图分辨率：滚轮缩放、左键拖动、右键以光标处为中心切换 1:1、双击返回全图。匹配和评分所用的 DSM 分辨率保持不变。主窗口复用合成画布，悬停读数仍对应左侧 12 个原始图层。

## 为什么用了 GPU 仍有延迟

GPU 的并行资源和显存带宽有限。256×320×320 是 26,214,400 个网格点，一个 Float32 中间图层约 100 MiB；128 个粒子则约 50 MiB。中值滤波、Sobel、归一化、Huber 损失和有效性掩膜会多次读写这些数据。GPU 能同时处理很多点，但不能让任意数量的点都在同一时刻完成。只有在工作量较小、GPU 尚未饱和时，增加粒子数才可能对耗时影响较小。

定位输出还依赖 ROS 消息解析、输入等待、CPU/GPU 数据交换、权重统计、PNG/CSV 写入及 OpenCV 绘图。Python 对 CUDA 张量调用 `.item()`、`.cpu()`，或在 `if` 中判断 CUDA 张量时，可能需要等待之前排队的 GPU 工作完成。性能分析中这段等待会显示在 Python 调用上，不能全部解释为标量复制本身的开销。

`/pf_global_pose` 在 PF 计算完成后、主图绘制之前发布，但同一线程的绘图仍会影响下一帧开始处理的时间。程序不生成虚假的重复消息来提高话题 Hz；静止时也会按现有逻辑跳过 PF 观测更新。

## 本轮采用的优化

- 先测量计算、显存和绘图开销，再决定改动；区分计算吞吐、发布频率和端到端延迟。
- 只对观测掩膜内的高度像素，以及腐蚀后的有效梯度像素评分，避免未观测区域的大量损失张量。保留全部有效观测，计算公式不变；稀疏度越高收益越大，满掩膜不保证更快。
- 按每组 32 个粒子计算裁剪和评分，限制 GPU 中间张量峰值。所有组完成后统一更新后验权重，不是只使用 32 个粒子，也不是跳帧。
- 合并最优粒子和位姿均值的部分标量回传，减少 Python 等待 GPU 的次数。
- 死区处理使用张量 `where`，避免布尔索引赋值。
- 完整分辨率静态底图、固定大小视口、增量轨迹和复用显示画布。底图仅在缩放/拖动/视口尺寸变化时重采样当前视口；新轨迹点只画新增线段。缩放后由完整坐标重新绘制 1 像素细轨迹，避免放大缩略图和粗线。

当前运行环境检查到 PyTorch 1.10.1+cu111。额外测试的自动算子融合路径出现 NVRTC 架构编译错误，因此没有采用该路径，也没有自动升级 ROS/PyTorch 环境。

## 本轮固定输入测量

128 粒子在当前设备上，10 个成功的离屏计算与绘图帧平均约 115 ms/帧。输入为固定 DSM 派生图层，包含主图/粒子图的图像生成，未开启第二套 Qt 显示上下文，不包含 ROS 消息等待或实际窗口显示耗时，因此不能将其直接写成实际话题频率。

在 32 粒子的独立评分对照中，连续有效区域占 20% 时，评分阶段约从 6.58 ms 降到 0.95 ms；占 80% 时约从 6.31 ms 降到 4.69 ms；满掩膜时约从 6.55 ms 变为 6.95 ms。这里仅比较评分阶段，说明稀疏计算的收益依赖实际观测覆盖率。

## 验证方法

加载 ROS 环境后运行回归测试：

```bash
source /opt/ros/noetic/setup.bash
source /home/hsw/catkin_ws/devel/setup.bash
cd /home/hsw/catkin_ws/program
PYTHONPATH="$PWD/dem_loc:$PYTHONPATH" python3 -m unittest discover -s dem_loc/tests -v
```

检查实际消息流：

```bash
rostopic hz /lidar_bev/grid_map
rostopic hz /pf_global_pose
rosparam get /localization_python_node/pf_num_particles
```

比较速度时应使用相同数据段、相同 BEV 尺寸、相同掩膜及相同可视化设置，并预热 GPU。GPU 分段计时应使用 CUDA Event；CPU 墙钟计时需要在测量边界同步 GPU。固定输入的计算与绘图耗时不等于实车或回放的端到端延迟。

## 大坐标精度与图像台阶

UTM Northing 约 3,000,000 米时，Float32 的相邻可表示数间隔为 0.25 米。不能把 0.2 米的局部网格直接加到 Float32 绝对坐标，再减地图原点，否则会形成不均匀或重复的采样位置。

粒子绝对位姿及初始化保持 Float64；裁剪时先用高精度减去地图原点，再把局部坐标转换为 Float32。稠密图层、采样网格和评分继续使用 Float32，避免整幅 patch 改为双精度造成额外开销。GPS/PF 参考位姿也保留高精度至减原点之后。

Viridis 伪彩色仍会让不同高度和梯度呈现不同颜色；梯度死区以及原始 DSM 自身的台阶也会形成色带。精度修复只消除数值采样引入的台阶，不会为了视觉平滑而改变原始地形或评分公式。

DSM 高度差直接使用 patch 高程减去各自中心格高程，不再执行 ±0.20 米置零。GPS/PF 显示与粒子匹配采用相同处理；旧的 `dsm_h_rel_deadzone_half` ROS 参数不再读取。梯度仍使用中值滤波、原有梯度死区和限幅。

## 只滤波 x/y，LocalPose 提供航向

PF 与 sem_loc 统一使用上一帧车体坐标系中的离散位移：

```text
dp_body = R(-yaw_lio_prev) * (p_lio_now - p_lio_prev)
dp_map  = R(yaw_local_map_prev) * (dp_body + noise_body)
p_particle_now = p_particle_prev + dp_map
```

前向/侧向噪声在车体坐标系中生成，再一起旋转到地图系。上一帧地图航向单独缓存在消费 LIO 消息的时刻；同一运动间隔内多次 LocalPose 更新不会覆盖它。当前 LocalPose 航向用于当前 DSM 裁剪。LIO yaw 只用于提取相对平移，不用于传播 PF yaw，也不额外加入半个转角。

LocalPose 航向复用 `local_pose_gauss_theta` / `LocalPoseToGlobal` 的单位、方向约定和首帧对齐转换。所有粒子共用该航向；初始化、运动噪声和重采样扩散只作用于 x/y。第三列 yaw 保留用于 DSM 裁剪和显示，但不参与随机搜索或加权估计。旧的三个 yaw 噪声配置项已移除。PF 输出 `/pf_global_pose` 的位置来自粒子滤波，azimuth 来自转换后的 LocalPose，单位仍为 math 角度。

启动需要同时收到 GlobalPose、LocalPose、`/lio/odom`。静止判定仍使用 LocalPose.vehicle_speed；静止期间更新公共航向，跳过位置传播和观测更新。全局红色 OdomDR 轨迹保留为首帧对齐后的纯 LIO 参考，不混入 LocalPose 后续航向修正；绿色 PF 轨迹使用上述混合传播。

两套程序统一的是运动合成方式；各自已有的 x/y 噪声参数和观测评分保持各自配置。LocalPose 和 LIO 当前使用最近收到的消息配对，没有新增时间戳插值同步或安装外参标定；真实数据上仍需核对延迟、时钟和 IMU/车辆参考点。

参考：Nav2 AMCL `differential_motion_model.cpp` 的相对运动采样，以及 GTSAM `Pose2::between` / `compose` 的离散 SE(2) 组合。这里采用已知外部航向的 x/y 条件模型，不直接复制 AMCL 的 yaw 噪声。
- https://github.com/ros-navigation/navigation2/blob/main/nav2_amcl/src/motion_model/differential_motion_model.cpp
- https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Pose2.h

## 全局轨迹显示分辨率

`GlobalTrackView` 在 CPU 缓存完整分辨率 BGR DSM，新增轨迹不重绘底图、不重建 GPU DSM。缩放和拖动只更新右侧视口；完整轨迹坐标不会随显示缩放量化。固定两窗口及左侧三行四列保持不变。启用轨迹图片保存时导出完整分辨率地图及轨迹。
