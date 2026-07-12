#include "ParticleFilter_our.h"
#include "CudaScopedTimer.h"
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace torch::indexing;

ParticleFilter::ParticleFilter(int num_particles) : num_particles_(num_particles), device_(torch::kCUDA)
{
    if (!torch::cuda::is_available())
    {
        std::cerr << "[ParticleFilter] Warning: CUDA not available. Running on CPU!" << std::endl;
        device_ = torch::kCPU;
    }
}

void ParticleFilter::Init(const SatelliteData& map_data,
                          double init_x, double init_y, double init_theta,
                          double std_x, double std_y, double std_theta)
{
    converter_ = std::make_shared<CoordConverter>(map_data);
    map_rows_ = map_data.tdf_map.rows;
    map_cols_ = map_data.tdf_map.cols;

    // 地图数据转 Tensor
    cv::Mat tdf_float;
    map_data.tdf_map.convertTo(tdf_float, CV_32F);
    int channels = tdf_float.channels();

    torch::Tensor tdf_cpu = torch::from_blob(tdf_float.data, {map_rows_, map_cols_, channels}, torch::kFloat);
    global_tdf_ = tdf_cpu.permute({2, 0, 1}).unsqueeze(0).to(device_).clone();  //(1, C, H, W)
    // 初始化粒子 (在 ROS Gauss 坐标系下)

    torch::Tensor mean = torch::tensor({init_x, init_y, init_theta}, device_);
    torch::Tensor std = torch::tensor({std_x, std_y, std_theta}, device_);
    particles_ = torch::randn({num_particles_, 3}, device_) * std + mean;
    weights_ = torch::ones({num_particles_}, device_) / num_particles_;
    std::cout << "粒子初始化完成，初始位置 gauss_degree (x,y,theta)" << init_x << ", " << init_y << ", " << init_theta << std::endl;
}

void ParticleFilter::Predict(double local_dx, double local_dy, double dtheta)
{
    CudaScopedTimer timer("PF::Predict", false);

    // =====================================================================
    // 标准粒子滤波运动模型 (Standard PF Motion Model)
    //
    // 输入: 车体局部坐标系增量 (local_dx, local_dy, dtheta)
    //       local_dx — 纵向 (前进为正), m
    //       local_dy — 侧向 (左正右负), m
    //       dtheta   — 航向增量, deg
    //
    // 三项核心改进:
    //   1. 各向异性噪声 (Anisotropic Noise):
    //      纵向 σ_fwd >> 侧向 σ_lat (阿克曼非完整约束)
    //   2. 精确 SE(2) 右乘合成 (Exact SE(2) Composition):
    //      p_new = p_old * T_action, 离散位姿增量的精确群运算
    //   3. 全 Tensor 并行, 无 for 循环
    // =====================================================================

    float distance = std::sqrt(local_dx * local_dx + local_dy * local_dy);
    float abs_dtheta = std::abs(dtheta);

    if (distance < 1e-6f && abs_dtheta < 1e-6f)
    {
        return;
    }

    // -----------------------------------------------------------------
    // 1) 各向异性运动噪声 (Anisotropic Motion Noise)
    //
    //    纵向 σ_fwd: 与行驶距离成正比, 累积里程计漂移
    //    侧向 σ_lat: 极小, 车辆不会凭空侧移 (Ackermann 约束)
    //    旋转 σ_yaw: 与角度变化和距离均相关 (弯道时累积更多)
    //
    //    经验模型: KITTI ~10Hz, Fast-LIO 作为里程计
    //      σ_fwd = α₁ * |d| + α₂         (默认 0.05*d + 0.02)
    //      σ_lat = α₃ * |d| + α₄         (默认 0.01*d + 0.005)
    //      σ_yaw = α₅ * |dθ| + α₆ * |d|  (默认 0.15*dθ + 1.0*d, deg)
    // -----------------------------------------------------------------
    const float alpha1 = 0.05f, alpha2 = 0.02f;   // 纵向 (较大)
    const float alpha3 = 0.01f, alpha4 = 0.005f;   // 侧向 (极小, 非完整约束)
    const float alpha5 = 0.15f, alpha6 = 1.0f;     // 旋转

    float sigma_fwd   = alpha1 * distance   + alpha2;            // m
    float sigma_lat   = alpha3 * distance   + alpha4;            // m
    float sigma_yaw   = alpha5 * abs_dtheta + alpha6 * distance; // deg

    auto noise = torch::randn({num_particles_, 3}, device_);

    // 在车体局部坐标系下注入各向异性噪声
    auto dx_noisy = local_dx + noise.select(1, 0) * sigma_fwd;   // (N) 纵向
    auto dy_noisy = local_dy + noise.select(1, 1) * sigma_lat;   // (N) 侧向
    auto dth_noisy = dtheta  + noise.select(1, 2) * sigma_yaw;   // (N) 旋转, deg

    // -----------------------------------------------------------------
    // 2) 精确 SE(2) 右乘合成 (Exact SE(2) Right-Multiplication)
    //
    //    参照 manifpy SE(2) 实现:
    //      p_new = p_old * T_action  (body-frame composition)
    //
    //    坐标展开:
    //      Δx_global = cos(θ_old) * dx_local - sin(θ_old) * dy_local
    //      Δy_global = sin(θ_old) * dx_local + cos(θ_old) * dy_local
    //      θ_new     = θ_old + Δθ
    //
    //    注: 里程计提供离散 SE(2) 增量 (非连续速度积分),
    //        中点欧拉近似在此场景下反而引入额外误差.
    // -----------------------------------------------------------------
    auto theta_rad = particles_.select(1, 2) * DEG_TO_RAD;  // (N), rad

    auto cos_th = torch::cos(theta_rad);
    auto sin_th = torch::sin(theta_rad);

    auto dx_global = cos_th * dx_noisy - sin_th * dy_noisy;
    auto dy_global = sin_th * dx_noisy + cos_th * dy_noisy;

    // -----------------------------------------------------------------
    // 3) 更新粒子状态 + 角度归一化
    // -----------------------------------------------------------------
    particles_.select(1, 0) += dx_global;
    particles_.select(1, 1) += dy_global;
    particles_.select(1, 2) += dth_noisy;

    // 角度归一化到 (-180, 180]
    auto theta_new = particles_.select(1, 2);
    particles_.select(1, 2) = theta_new - 360.0 * torch::floor((theta_new + 180.0) / 360.0);
}

torch::Tensor ParticleFilter::ComputeAffineMatrices()
{
    int view_size = obs_size_;
    int half_size = view_size / 2;

    // 1. 生成局部网格 (Image Coords, y axis down)
    auto y_range = torch::arange(-half_size, half_size, device_);
    auto x_range = torch::arange(-half_size, half_size, device_);
    auto grid_pair = torch::meshgrid({y_range, x_range}, "ij");
    auto grid_y = grid_pair[0];
    auto grid_x = grid_pair[1];

    // 2. 获取粒子状态 (ROS Coords)
    auto px = particles_.index({Slice(), 0});
    auto py = particles_.index({Slice(), 1});
    auto theta = particles_.index({Slice(), 2});

    auto blh_pair = converter_->gauss_to_wgs84_gpu(px, py);
    auto pixel_pair = converter_->wgs84_to_pixel_gpu(blh_pair.first, blh_pair.second);

    torch::Tensor cx = pixel_pair.first;  // 直接使用，不要重新 from_blob
    torch::Tensor cy = pixel_pair.second;

    auto theta_rot = (90-theta)*DEG_TO_RAD;     //90 - 朝向角
    auto cos_t = torch::cos(theta_rot).view({-1, 1, 1});
    auto sin_t = torch::sin(theta_rot).view({-1, 1, 1});

    auto grid_x_batch = grid_x.unsqueeze(0);
    auto grid_y_batch = grid_y.unsqueeze(0);
    auto cx_batch = cx.view({-1, 1, 1});
    auto cy_batch = cy.view({-1, 1, 1});

    // X_src = X_grid * cos - Y_grid * sin + cx
    // Y_src = X_grid * sin + Y_grid * cos + cy
    auto grid_x_rot = grid_x_batch * cos_t - grid_y_batch * sin_t + cx_batch;
    auto grid_y_rot = grid_x_batch * sin_t + grid_y_batch * cos_t + cy_batch;

    // 归一化
    float w_minus_1 = (float)map_cols_ - 1.0;
    float h_minus_1 = (float)map_rows_ - 1.0;
    auto norm_grid_x = 2.0 * (grid_x_rot / w_minus_1) - 1.0;
    auto norm_grid_y = 2.0 * (grid_y_rot / h_minus_1) - 1.0;

    return torch::stack({norm_grid_x, norm_grid_y}, -1);
}

                /*我的方法，不确定性无熵进行权重计算*/
// void ParticleFilter::Update(const cv::Mat& bev_data, float sigma_obs)
// {
//     // ==================================================================================
//     // 0. 基础参数检查 (保持不变)
//     // ==================================================================================
//     if (sigma_obs < 0.1f) sigma_obs = 0.1f;
//     if (bev_data.empty() || bev_data.type() != CV_32FC4) return;

//     // ==================================================================================
//     // 1. 数据准备 (保持不变)
//     // ==================================================================================
//     // (1, 4, H, W)
//     auto obs_tensor = torch::from_blob((void*)bev_data.data,
//                                        {obs_size_, obs_size_, 4},
//                                        torch::kFloat)
//                         .clone().permute({2, 0, 1}).unsqueeze(0).to(device_);

//     // probs: (1, 3, H, W)
//     auto probs = obs_tensor.slice(1, 0, 3);

//     // 【消融实验修改点 1】: 虽然输入里有熵 (slice(1,3,4))，但我们直接忽略它
//     // auto entropy = obs_tensor.slice(1, 3, 4); // 不再需要

//     // ==================================================================================
//     // 2. 构建"纯概率权重" (Probability-Only Weights) - 【消融实验核心】
//     // ==================================================================================

//     // 逻辑：Weight_c = Prob_c
//     // 我们假设 reliability = 1.0 (即完全信任概率值)

//     // 直接使用概率作为权重
//     // (1, 3, H, W)
//     auto w_conf = probs;

//     // [可选] 为了防止极低概率的背景噪声干扰计算，通常可以加一个极小的阈值
//     // 但为了保持数学上的纯粹性(Pure Probabilistic)，直接用 raw probs 也是对的。
//     // w_conf = torch::where(w_conf < 0.01, torch::zeros_like(w_conf), w_conf);

//     // ==================================================================================
//     // 3. 全局地图采样 (保持不变)
//     // ==================================================================================
//     auto affine_grids = ComputeAffineMatrices();
//     auto map_batch = global_tdf_.expand({num_particles_, -1, -1, -1});

//     // sampled_dist: (N, 3, H, W)
//     auto sampled_dist = torch::nn::functional::grid_sample(
//         map_batch, affine_grids,
//         torch::nn::functional::GridSampleFuncOptions()
//             .mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true));

//     // ==================================================================================
//     // 4. 计算加权均方误差 (Weighted MSE) - 【逻辑保持一致】
//     // ==================================================================================

//     // 公式: J = Sum(Prob * Dist^2) / Sum(Prob)

//     // A. 分子
//     auto dist_sq = sampled_dist.pow(2);
//     auto weighted_sse = (dist_sq * w_conf).sum({1, 2, 3}); // (N)

//     // B. 分母
//     // 这里计算的是图像中所有类别的"总概率质量" (Total Probability Mass)
//     auto total_weight = w_conf.sum(); // Scalar

//     float weight_sum_val = total_weight.item<float>();
//     if (weight_sum_val < 1e-6) {
//         return;
//     }

//     // C. Cost
//     auto cost_mse = weighted_sse / total_weight; // (N)

//     // ==================================================================================
//     // 5. 似然更新 (保持不变)
//     // ==================================================================================
//     auto log_weights = - cost_mse / (2.0f * std::pow(sigma_obs, 2));

//     auto max_log = log_weights.max();
//     weights_ = torch::exp(log_weights - max_log);

//     // ==================================================================================
//     // 6. 归一化 (保持不变)
//     // ==================================================================================
//     auto w_sum = weights_.sum();

//     if (w_sum.item<double>() < 1e-9 || torch::isnan(w_sum).item<bool>()) {
//         std::cerr << "[PF Critical] Weights collapsed." << std::endl;
//         weights_ = torch::ones({num_particles_}, device_) / num_particles_;
//     } else {
//         weights_ /= w_sum;
//     }
// }




                /*我的方法，不确定性带入语义熵进行权重计算*/
void ParticleFilter::Update(const cv::Mat& bev_data, float sigma_obs)
{
    CudaScopedTimer timer_total("PF::Update", false);
    // =========================
    // 0) Checks
    // =========================
    if (bev_data.empty()) return;
    if (bev_data.type() != CV_32FC4) return;
    if (!bev_data.isContinuous()) return;

    const int H = bev_data.rows;
    const int W = bev_data.cols;
    if (H != obs_size_ || W != obs_size_) return;

    sigma_obs = std::max(sigma_obs, 0.1f);

    // Ensure weights exist
    if (!weights_.defined() || weights_.numel() != num_particles_) {
        weights_ = torch::ones({num_particles_},
                    torch::TensorOptions().device(device_).dtype(torch::kFloat32))
                   / (float)num_particles_;
    }

    // =========================
    // 1) Obs tensor (1,4,H,W)
    // =========================
    auto obs_tensor = torch::from_blob(
            (void*)bev_data.data, {H, W, 4},
            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .permute({2, 0, 1})
        .unsqueeze(0)
        .to(device_);

    auto probs   = obs_tensor.slice(1, 0, 3); // (1,3,H,W)
    auto entropy = obs_tensor.slice(1, 3, 4); // (1,1,H,W)

    // =========================
    // 2) w_conf = probs*(1-entropy)
    // =========================
    //auto reliability = (1.0f - entropy).clamp(0.0f, 1.0f); // (1,1,H,W)
    // 1) linear
    //auto r1 = (1.0f - entropy).clamp(0.0f, 1.0f);

    // 2) exponential
    auto reliability = torch::exp(-5.0f * entropy);

    // 3) power
    //auto reliability = torch::pow((1.0f - entropy).clamp(0.0f, 1.0f), 2.0);

    // 4) inverse
    //auto reliability = 1.0f / (1.0f + 2.0f * entropy);



    auto w_conf = probs * reliability;                     // (1,3,H,W)

    auto total_weight = w_conf.sum().clamp_min(1e-6);      // scalar
    if (total_weight.item<float>() < 1.0f)
    {
        return;
    }

    // =========================
    // 3) Sample global TDF
    // =========================
    torch::Tensor affine_grids;
    {
        CudaScopedTimer t("PF::Update/AffineGrid", false);
        affine_grids = ComputeAffineMatrices();
    }

    torch::Tensor sampled_dist;
    {
        CudaScopedTimer t("PF::Update/GridSample", false);
        auto map_batch = global_tdf_.expand({num_particles_, -1, -1, -1});
        sampled_dist = torch::nn::functional::grid_sample(
            map_batch, affine_grids,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true));
    }

    // =========================
    // 4) Weighted MSE cost
    // =========================
    auto dist_sq = sampled_dist.pow(2);                    // (N,3,H,W)
    auto weighted_sse = (dist_sq * w_conf).sum({1,2,3});   // (N)
    auto cost_mse = weighted_sse / total_weight;           // (N)

    // =========================
    // 5) 标准高斯似然 (Standard Gaussian Likelihood)
    // =========================
    // 标准 SIR 粒子滤波: log p(z|x) = -cost / (2σ²)
    auto log_like = -cost_mse / (2.0f * sigma_obs * sigma_obs); // (N)

    // =========================
    // 6) 贝叶斯权重更新: log w_post = log w_prior + log_like
    // =========================
    // 自适应重采样: 未重采样时保留先验权重; 重采样后 w_prior=1/N (常数项, 归一化后消除)
    auto log_w_prior = torch::log(weights_.clamp_min(1e-12));
    auto log_w_post  = log_w_prior + log_like;

    // Log-Sum-Exp 数值稳定归一化
    auto max_log = log_w_post.max();
    auto w_unnorm = torch::exp(log_w_post - max_log);

    auto w_sum = w_unnorm.sum();
    if (!torch::isfinite(w_sum).item<bool>() || w_sum.item<float>() < 1e-12f) {
        weights_.fill_(1.0f / (float)num_particles_);
        return;
    }
    weights_ = w_unnorm / w_sum;

    // GPU Profiling: 每 100 帧打印一次汇总
    static int pf_profile_counter = 0;
    if (++pf_profile_counter % 100 == 0) {
        CudaScopedTimer::PrintSummary();
    }

    // Debug 输出
    float neff_val = GetNeff();
    printf("PF: cost[min,mean,max]=[%.3f, %.3f, %.3f], Neff=%.1f, w_max=%.3e\n",
           cost_mse.min().item<float>(),
           cost_mse.mean().item<float>(),
           cost_mse.max().item<float>(),
           neff_val,
           weights_.max().item<float>());
}



/*低方差采样*/
void ParticleFilter::Resample()
{
    CudaScopedTimer timer("PF::Resample", false);
    std::cout << "[ParticleFilter] 进入系统重采样 (Systematic Resampling)" << std::endl;
    // 1) Guard
    if (!weights_.defined() || weights_.numel() != num_particles_) {
        weights_ = torch::ones({num_particles_},
                    torch::TensorOptions().device(device_).dtype(torch::kFloat32))
                   / (float)num_particles_;
        return;
    }

    auto w_sum = weights_.sum();
    if (!torch::isfinite(w_sum).item<bool>() ||
        w_sum.item<double>() <= 1e-9 ||
        torch::isnan(weights_).any().item<bool>())
    {
        weights_.fill_(1.0f / (float)num_particles_);
        return;
    }

    // 2) Normalize once (CPU)
    auto w_cpu = (weights_ / w_sum).to(torch::kCPU).contiguous(); // (N) CPU float32

    // 4) CDF
    auto cdf = torch::cumsum(w_cpu, 0);
    cdf[-1] = 1.0f; // 必须 = 1.0，别 >1

    // 5) Systematic points
    const int N = num_particles_;
    float step = 1.0f / (float)N;
    float u0 = ((float)rand() / (float)RAND_MAX) * step;

    std::vector<int64_t> indices(N);
    int i = 0;
    for (int j = 0; j < N; ++j) {
        float uj = u0 + j * step;
        while (uj > cdf[i].item<float>() && i < N - 1) i++;
        indices[j] = i;
    }

    // 6) Apply indices
    auto idx = torch::from_blob(indices.data(), {N},
                torch::TensorOptions().dtype(torch::kLong)).clone().to(device_);

    particles_ = particles_.index_select(0, idx);

    // 7) Reset weights uniform
    weights_.fill_(1.0f / (float)N);
}


std::vector<double> ParticleFilter::EstimatePose()
{
    CudaScopedTimer timer("PF::EstimatePose", false);
    // 标准粒子滤波: 全部粒子加权平均
    auto w = weights_ / (weights_.sum() + 1e-12);

    auto mean_x = (particles_.index({Slice(), 0}) * w).sum().item<double>();
    auto mean_y = (particles_.index({Slice(), 1}) * w).sum().item<double>();

    // 圆形均值 (circular mean) 处理角度环绕
    auto theta = particles_.index({Slice(), 2}); // deg
    auto theta_rad = theta * DEG_TO_RAD;

    auto mean_theta_rad = std::atan2((torch::sin(theta_rad) * w).sum().item<double>(),
                                     (torch::cos(theta_rad) * w).sum().item<double>());


    return {mean_x, mean_y, mean_theta_rad * RAD_TO_DEG};
}

double ParticleFilter::GetNeff() const
{
    if (!weights_.defined() || weights_.numel() == 0) return 0.0;
    double sum_sq = weights_.pow(2).sum().item<double>();
    return 1.0 / (sum_sq + 1e-12);
}

// 辅助函数：将 0-1 的归一化权重转换为 BGR 颜色 (Jet Colormap)
cv::Scalar GetColorFromWeight(float val)
{
    // 限制范围在 [0, 1]
    val = std::max(0.0f, std::min(1.0f, val));

    float b, g, r;

    if (val < 0.5f) {
        // Blue (1, 0, 0) -> Green (0, 1, 0)
        float f = val / 0.5f;
        b = 1.0f - f;
        g = f;
        r = 0.0f;
    } else {
        // Green (0, 1, 0) -> Red (0, 0, 1)
        float f = (val - 0.5f) / 0.5f;
        b = 0.0f;
        g = 1.0f - f;
        r = f;
    }

    // OpenCV 使用 BGR 顺序, 范围 0-255
    return cv::Scalar((int)(b * 255), (int)(g * 255), (int)(r * 255));
}

std::vector<ParticleFilter::ParticleState> ParticleFilter::GetRawParticles()
{
    std::vector<ParticleState> res;
    res.reserve(num_particles_);

    // 确保不需要梯度
    torch::NoGradGuard no_grad;

    // 移动到 CPU 并转为 Double
    auto p_cpu = particles_.to(torch::kFloat64).cpu();
    auto w_cpu = weights_.to(torch::kFloat64).cpu();

    // 使用 accessor 访问效率更高
    auto p_data = p_cpu.accessor<double, 2>();
    auto w_data = w_cpu.accessor<double, 1>();

    for(int i = 0; i < num_particles_; ++i)
    {
        ParticleState p;
        p.x = p_data[i][0];
        p.y = p_data[i][1];
        p.theta = p_data[i][2]; // 保持内部单位 (Degrees) by default
        p.weight = w_data[i];
        res.push_back(p);
    }
    return res;
}
