# DSM-BEV Score 与梯度生成说明（区分度增强版）

本文档描述当前代码实现中 **DSM-BEV 匹配分数** 的计算公式、参数建议，以及 **LiDAR BEV / DSM Patch** 两侧 `G_long_L / G_lat_L` 的生成流程。

本版本的目标不是马上进入完整粒子滤波，而是先在 **GlobalPose 单候选 / GlobalPose 周围扰动候选** 阶段验证 score 是否具备足够区分度：

```text
正确位姿或 GlobalPose 附近候选应获得更低 J_total；
明显平移 / 旋转偏移候选应获得更高 J_total；
score 曲面应在正确位姿附近形成清晰 basin。
```

对应源码：

| 模块 | 文件 |
|------|------|
| Score 计算 | `program/dem_loc/loc_tool/dsm_bev_score.py` |
| Score 调度 | `program/dem_loc/loc_tool/dsm_bev_score_runner.py` |
| ROS 入口 | `program/dem_loc/scripts/localization_python.py` |
| LiDAR BEV 生成 | `program/GridMap_v7_5_ros1/src/lidar_bev_builder.cpp` |
| DSM Patch 生成 | `program/dem_loc/loc_tool/dsm_patch.py` |
| LiDAR BEV 参数 | `program/GridMap_v7_5_ros1/config/GridMapInit.ini` |

---

## 1. 图层命名对照

| 设计文档 / Score | LiDAR BEV (GridMap) | DSM Patch | 说明 |
|------------------|---------------------|-----------|------|
| `H_L` / `H_D` | `H_rel_surf` | `H_rel_surf` | 相对高度，单位 m |
| `Gx_L` / `Gx_D` | `G_lat_L` | `G_lat_L` | 横向梯度，单位 m/m |
| `Gy_L` / `Gy_D` | `G_long_L` | `G_long_L` | 纵向梯度，单位 m/m |
| `M_obs` | `M_L` | 打分使用 LiDAR 的 `M_L` | LiDAR 有效观测 mask |
| `M_grad` | 由 `M_L` 和梯度有效性生成 | 打分使用 LiDAR 的 `M_grad` | 梯度 score mask |

注意：

```text
Gx = G_lat  横向梯度
Gy = G_long 纵向梯度
```

后续配置 `alpha_gx / alpha_gy` 时要注意这一点。

---

## 2. 当前阶段验证策略

### 2.1 不直接上完整粒子滤波

当前阶段建议先使用两种模式验证 score：

```text
模式 A：GlobalPose 单候选
    N = 1
    只输出 J_h / J_gx / J_gy / J_total
    不关注 weight

模式 B：GlobalPose 周围扰动候选
    N > 1
    在 GlobalPose 周围生成 dx / dy / dtheta 扰动
    观察正确位姿附近是否 J_total 更低
```

### 2.2 推荐扰动候选

```text
dx     ∈ [-4, -2, 0, 2, 4] m
dy     ∈ [-4, -2, 0, 2, 4] m
dtheta ∈ [-4, 0, 4] deg
```

得到：

```text
N = 5 × 5 × 3 = 75
```

需要检查：

```text
1. GlobalPose 附近候选的 J_total 是否更低；
2. 平移 2~4 m 后 J_total 是否明显增大；
3. 旋转 ±4° 后梯度项是否有变化；
4. 正确位姿是否进入 top-k，建议 top-5；
5. J_gap = J_second_best - J_best 是否足够大。
```

---

## 3. DSM-BEV Score 总体流程

```mermaid
flowchart TD
    A[LiDAR BEV: H_L, Gx_L, Gy_L, M_obs] --> B[build M_obs and M_grad]
    C[GlobalPose / LocalPose / 扰动候选位姿] --> D[grid_sample 裁剪 DSM patch]
    D --> E[DSM: H_D, Gx_D, Gy_D]
    B --> F[统计 H_max: 仅使用 LiDAR 有效非地面高度]
    E --> G[LiDAR 与 DSM 共用 H_max 归一化高度]
    F --> G
    G --> H[高度结构权重 W_h]
    H --> I[高度 Huber 代价 J_h]
    B --> J[梯度 clip + M_grad]
    E --> J
    J --> K[梯度 Huber 代价 J_gx / J_gy]
    I --> L[加权总代价 J_total]
    K --> L
    L --> M[候选组内归一化 score_i = exp(-(J_i - J_min)/tau)]
```

---

## 4. Mask 设计

### 4.1 高度项 mask

高度项使用 LiDAR 有效观测区域：

\[
M_{obs}(i)=\mathbb{1}[M_L(i)>0.5 \land \text{finite}(H_L(i))]
\]

高度项只在 `M_obs=1` 的位置计算。

---

### 4.2 梯度项 mask

梯度项使用更严格的 `M_grad`。

推荐定义：

\[
M_{grad}=\text{erode}(M_{obs}, n_{erode})
\land \text{finite}(G_{x,L},G_{y,L},G_{x,D},G_{y,D})
\]

同时建议在 LiDAR 梯度生成阶段输出 `M_grad`，其条件包括：

```text
1. 当前 cell 的 M_obs > 0.5；
2. 当前 H_rel 有限；
3. 3×3 邻域内有效点数量 >= edge_min_valid_neighbors；
4. mask-aware median 有效；
5. G_long_L / G_lat_L 有限；
6. 不在明显无观测边界区域。
```

建议参数：

```ini
grad_mask_erode_px = 1
```

不建议第一版使用 `grad_mask_erode_px = 2`，因为 LiDAR 梯度本身已经比较稀疏，腐蚀过强会导致参与梯度打分的像素过少。

---

## 5. 高度归一化

### 5.1 统计统一高度上限 `H_max`

只从 LiDAR 当前帧有效非地面高度统计：

\[
\mathcal{V}_h=\{i \mid M_{obs}(i)=1 \land H_L(i)>h_{ground}\}
\]

如果：

\[
|\mathcal{V}_h| \ge N_{min}
\]

则：

\[
H_{max}=\mathrm{clip}
\left(
\mathrm{percentile}_{p}(H_L[\mathcal{V}_h]),
 h_{min},
 h_{max}
\right)
\]

否则：

\[
H_{max}=h_{min}
\]

推荐参数：

```ini
ground_threshold = 0.20
hmax_min = 5.0
hmax_max = 40.0
height_percentile = 98.0
min_valid_height_points = 10
```

### 5.2 高度归一化

LiDAR 和 DSM 必须使用同一个 `H_max`：

\[
H_{L,n}(i)=\frac{\mathrm{clip}(H_L(i),0,H_{max})}{H_{max}}
\]

\[
H_{D,n}(i)=\frac{\mathrm{clip}(H_D(i),0,H_{max})}{H_{max}}
\]

禁止 LiDAR 和 DSM 各自单独归一化。

---

## 6. 高度代价 `J_h`（增强区分度版）

### 6.1 高度残差

\[
e_h(i)=H_{L,n}(i)-H_{D,n}(i)
\]

### 6.2 非对称高度惩罚

如果 LiDAR 看到高结构，但 DSM patch 没有对应高结构，应强惩罚：

\[
\lambda_h(i)=
\begin{cases}
\lambda_{lidar\uparrow}, & H_{L,n}(i)>H_{D,n}(i) \\
\lambda_{dsm\uparrow}, & \text{otherwise}
\end{cases}
\]

推荐：

```ini
lambda_lidar_higher = 1.5
lambda_dsm_higher = 0.3
```

如果希望进一步增强偏移惩罚，可以临时测试：

```ini
lambda_lidar_higher = 2.0
lambda_dsm_higher = 0.3
```

---

### 6.3 高度结构权重 `W_h`

为了避免大面积道路平坦区域主导 score，增加基于 LiDAR 高度结构的权重：

\[
W_h(i)=M_{obs}(i)\cdot\left(w_{h,base}+w_{h,height}\cdot H_{L,n}(i)\right)
\]

推荐：

```ini
w_h_base = 0.2
w_h_height = 0.8
```

含义：

```text
平地仍然参与，但权重较低；
LiDAR 实际观测到的高结构区域权重大；
位姿偏移后，高结构错位会更明显增加 J_h。
```

可选增强版本：

\[
G_{abs,L}=\frac{\mathrm{clip}(|G_{x,L}|+|G_{y,L}|,0,g_{cap})}{g_{cap}}
\]

\[
W_h(i)=M_{obs}(i)\cdot(0.20+0.60H_{L,n}(i)+0.20G_{abs,L}(i))
\]

第一版建议先使用简单版本：

\[
W_h=M_{obs}\cdot(0.2+0.8H_{L,n})
\]

---

### 6.4 Huber 高度损失

\[
\mathrm{Huber}(e,\delta)=
\begin{cases}
\frac{1}{2}e^2, & |e|\le\delta \\
\delta(|e|-\frac{1}{2}\delta), & |e|>\delta
\end{cases}
\]

推荐：

```ini
delta_h = 0.30
```

说明：

```text
delta_h 不要过小。
过小会让残差过早进入线性区，反而削弱中大偏差的区分度。
0.30 比 0.20 更适合当前阶段增强偏移惩罚。
```

---

### 6.5 高度代价

\[
J_h=\frac{\sum_i W_h(i)\lambda_h(i)\mathrm{Huber}(e_h(i),\delta_h)}{\sum_i W_h(i)+\varepsilon}
\]

其中仅对 `M_obs=1` 且有限值的位置求和。

---

## 7. 梯度代价 `J_gx / J_gy`（区分度增强但防误伤）

### 7.1 梯度截断与归一化

打分前对 LiDAR 与 DSM 梯度统一截断：

\[
G_c=\frac{\mathrm{clip}(G,-g_{cap},g_{cap})}{g_{cap}}
\]

推荐：

```ini
grad_cap = 3.0
```

如果梯度图中存在大量强边缘导致残差过饱和，可以测试：

```ini
grad_cap = 5.0
```

但当前为了提高区分度，优先使用：

```ini
grad_cap = 3.0
```

---

### 7.2 梯度残差

横向梯度：

\[
e_{gx}(i)=G_{x,L,c}(i)-G_{x,D,c}(i)
\]

纵向梯度：

\[
e_{gy}(i)=G_{y,L,c}(i)-G_{y,D,c}(i)
\]

其中：

```text
Gx = G_lat  横向梯度
Gy = G_long 纵向梯度
```

---

### 7.3 梯度权重

只在 `M_grad` 内比较梯度，并用 LiDAR 梯度强度加权：

\[
w_{gx}(i)=M_{grad}(i)\cdot\left(w_{g,base}+(1-w_{g,base})|G_{x,L,c}(i)|\right)
\]

\[
w_{gy}(i)=M_{grad}(i)\cdot\left(w_{g,base}+(1-w_{g,base})|G_{y,L,c}(i)|\right)
\]

推荐：

```ini
w_g_base = 0.05
```

说明：

```text
LiDAR 梯度强的位置权重大；
弱梯度区域保留少量基础权重；
w_g_base 不建议再用 0.2，当前 LiDAR 梯度较碎，0.2 容易让弱梯度区域引入 DSM 纹理噪声。
```

---

### 7.4 梯度 Huber 损失

推荐：

```ini
delta_g = 0.30
```

梯度代价：

\[
J_{gx}=\frac{\sum_i w_{gx}(i)\mathrm{Huber}(e_{gx}(i),\delta_g)}{\sum_i w_{gx}(i)+\varepsilon}
\]

\[
J_{gy}=\frac{\sum_i w_{gy}(i)\mathrm{Huber}(e_{gy}(i),\delta_g)}{\sum_i w_{gy}(i)+\varepsilon}
\]

---

## 8. 总匹配代价 `J_total`

当前可视化中，高度结构最稳定；横向梯度比纵向梯度更有区分度；纵向梯度更容易受 LiDAR 稀疏和 mask 影响。

因此推荐第一版：

\[
J_{total}=0.65J_h+0.25J_{gx}+0.10J_{gy}
\]

对应配置：

```ini
alpha_h = 0.65
alpha_gx = 0.25
alpha_gy = 0.10
```

注意：这里假设：

```text
Gx = G_lat  横向梯度
Gy = G_long 纵向梯度
```

如果代码里 `gx/gy` 映射相反，则需要交换 `alpha_gx` 和 `alpha_gy`。

可选保守版本：

```ini
alpha_h = 0.70
alpha_gx = 0.20
alpha_gy = 0.10
```

可选梯度增强版本：

```ini
alpha_h = 0.60
alpha_gx = 0.30
alpha_gy = 0.10
```

不建议当前直接使用：

```ini
alpha_h = 0.50
alpha_gx = 0.25
alpha_gy = 0.25
```

因为当前 LiDAR 纵向梯度较碎，纵向梯度权重过高容易误伤正确位姿。

---

## 9. Score 计算

### 9.1 单候选模式

如果当前只有一个 GlobalPose 候选：

```text
N = 1
```

则无需关注权重，直接输出：

```text
J_h
J_gx
J_gy
J_total
```

`score` 可以作为参考：

\[
score=\exp(-J_{total}/\tau)
\]

但单候选模式下，`weight=1`，没有排序意义。

---

### 9.2 多候选 / 粒子模式

对于 GlobalPose 周围扰动候选或正式粒子滤波，应使用候选组内归一化 score：

\[
J_{min}=\min_i J_{total,i}
\]

\[
score_i=\exp\left(-\frac{J_{total,i}-J_{min}}{\tau}\right)
\]

推荐：

```ini
tau = 0.10
```

如果权重过度塌缩，可以调大：

```ini
tau = 0.15
```

如果候选差异仍然不明显，可以调小：

```ini
tau = 0.07
```

权重：

\[
w_i=\frac{w_i^{old}score_i}{\sum_j w_j^{old}score_j+\varepsilon}
\]

如果没有历史权重：

\[
w_i=\frac{score_i}{\sum_j score_j+\varepsilon}
\]

---

## 10. LiDAR BEV 梯度生成建议

### 10.1 不要先乘 mask 再 Sobel

禁止：

```text
H_masked = H_rel * M_obs
G = Sobel(H_masked)
```

原因：

```text
M_obs=0 的区域会被置 0；
mask 边界会产生人工高度跳变；
Sobel 会把无观测边界当成真实结构边缘。
```

---

### 10.2 推荐 mask-aware median + center-fill Sobel

流程：

```text
1. 对 M_obs=1 且 H_rel 有限的 cell，取 3×3 有效邻居；
2. 有效邻居数 >= edge_min_valid_neighbors，计算 median；
3. Sobel 时，有效邻居用自己的 smoothed H；
4. 无效邻居用中心格 smoothed H 替代；
5. 这样无效邻居不制造人工梯度，也不会要求 3×3 全有效。
```

建议不要强制要求 3×3 全有效，因为 LiDAR BEV 稀疏时会导致 `M_grad` 过少。

---

### 10.3 LiDAR 梯度后处理参数

推荐：

```ini
lidar_bev_h_rel_deadzone_half = 0.20
lidar_bev_grad_deadzone_half = 0.15
lidar_bev_edge_min_valid_neighbors = 5
```

不推荐当前使用：

```ini
lidar_bev_grad_deadzone_half = 2.0
```

原因：

```text
2.0 m/m 的梯度死区过大，会把大量可用边缘清零；
DSM 侧默认梯度死区约 0.15 m/m，LiDAR / DSM 不一致会导致梯度项天然不匹配。
```

---

## 11. DSM Patch 梯度生成建议

### 11.1 DSM 与 LiDAR 高度 deadzone 对齐

建议：

```ini
dsm_h_rel_deadzone_half = 0.20
```

并与 LiDAR 保持一致：

```ini
lidar_bev_h_rel_deadzone_half = 0.20
```

不要一边是 `0.5`，另一边是 `0`，否则高度项和梯度项都会出现系统差异。

---

### 11.2 DSM 梯度公式

如果 DSM patch 满足：

```text
patch 图像上方 = 车头方向
patch 图像右方 = 车体右侧
```

并且定义：

```text
G_long > 0：沿车头方向高度升高
G_lat  > 0：沿车体右侧高度升高
```

则建议：

\[
G_{long,D}=-\frac{\mathrm{conv}(K_y,\tilde{H}_D)}{8r}
\]

\[
G_{lat,D}=+\frac{\mathrm{conv}(K_x,\tilde{H}_D)}{8r}
\]

注意：原代码中如果使用：

\[
G_{lat,D}=-\frac{\mathrm{conv}(K_x,\tilde{H}_D)}{8r}
\]

则横向符号可能与 LiDAR 侧相反。

建议增加参数开关：

```ini
dsm_long_sign = 1
dsm_lat_sign = 1
```

实际打分前：

```python
G_long_D = dsm_long_sign * G_long_D_raw
G_lat_D  = dsm_lat_sign  * G_lat_D_raw
```

用 GlobalPose 周围扰动候选测试四组符号组合：

```text
1.  G_long_D,  G_lat_D
2. -G_long_D,  G_lat_D
3.  G_long_D, -G_lat_D
4. -G_long_D, -G_lat_D
```

选择使 GlobalPose 附近 `J_total` 最低、扰动偏移后 `J_total` 增大的符号组合。

---

## 12. 推荐参数汇总

### 12.1 Score 参数

```ini
[score]
eps = 1e-6
tau = 0.10

[height_score]
ground_threshold = 0.20
hmax_min = 5.0
hmax_max = 40.0
height_percentile = 98.0
min_valid_height_points = 10
lambda_lidar_higher = 1.5
lambda_dsm_higher = 0.3
delta_h = 0.30
w_h_base = 0.2
w_h_height = 0.8

[gradient_score]
grad_cap = 3.0
delta_g = 0.30
w_g_base = 0.05
grad_mask_erode_px = 1

[score_weight]
alpha_h = 0.65
alpha_gx = 0.25
alpha_gy = 0.10
```

### 12.2 LiDAR BEV 参数

```ini
lidar_bev_h_rel_deadzone_half = 0.20
lidar_bev_grad_deadzone_half = 0.15
lidar_bev_edge_min_valid_neighbors = 5
```

### 12.3 DSM Patch 参数

```ini
dsm_h_rel_deadzone_half = 0.20
dsm_grad_deadzone_half = 0.15
dsm_long_sign = 1
dsm_lat_sign = 1
```

`dsm_lat_sign` 需要通过实验确认。

---

## 13. Debug 输出建议

每帧 / 每组候选输出：

```text
timestamp
candidate_id
x_gauss
y_gauss
theta
J_h
J_gx
J_gy
J_total
score
weight
H_max
valid_h_num
valid_grad_num
```

多候选模式额外输出：

```text
best_candidate_id
best_dx
best_dy
best_dtheta
J_best
J_second_best
J_gap = J_second_best - J_best
rank_of_globalpose_candidate
score_best
score_globalpose
score_ratio = score_best / (score_globalpose + eps)
```

建议保存 debug 图：

```text
LiDAR H_rel / DSM H_rel
Height residual map
Height weighted penalty map
LiDAR G_lat / DSM G_lat
G_lat residual map
LiDAR G_long / DSM G_long
G_long residual map
M_obs
M_grad
J_total heatmap over dx-dy
```

---

## 14. 进入粒子滤波前的验收标准

建议满足以下条件后再进入完整粒子滤波：

```text
1. GlobalPose 单候选时，DSM patch 与 LiDAR BEV 在 H_rel 上宏观对齐；
2. GlobalPose 周围扰动候选中，中心或中心附近候选 J_total 排名前 5；
3. dx/dy 偏移 2~4 m 时，J_total 有明显增大趋势；
4. dtheta 偏移 ±4° 时，梯度项 J_gx / J_gy 有可解释变化；
5. M_grad 有足够有效像素，valid_grad_num 不应过低；
6. 更换多帧测试后，score 排序稳定，不只在单帧偶然成立；
7. DSM 横向梯度符号已通过扰动候选验证。
```

---

## 15. 当前版本核心结论

```text
1. 高度项作为主约束，权重提高到 0.65；
2. 高度项加入 W_h，使高结构区域对偏移更敏感；
3. LiDAR 高于 DSM 的非对称惩罚增强到 1.5；
4. 梯度项使用 M_grad，不使用 M_obs 全图比较；
5. 梯度项先 clip / normalize，grad_cap 建议 3.0；
6. 横向梯度 G_lat 比纵向梯度 G_long 更可靠，因此 alpha_gx > alpha_gy；
7. score 在多候选 / 粒子模式下使用 exp(-(J_i - J_min)/tau)，tau 建议 0.10；
8. LiDAR / DSM 的高度 deadzone 和梯度 deadzone 需要统一；
9. DSM G_lat 符号必须通过扰动候选实验确认；
10. 满足扰动候选区分度验收标准后，再进入完整粒子滤波。
```
