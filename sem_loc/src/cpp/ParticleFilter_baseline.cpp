#include "ParticleFilter_baseline.h"
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


        /*baseline 硬标签，直接匹配*/
void ParticleFilter::Update(const cv::Mat& bev_mask, float sigma_obs)
{
    CudaScopedTimer timer_total("PF::Update", false);
    // =========================
    // 0) Checks
    // =========================
    if (bev_mask.empty()) return;
    sigma_obs = std::max(sigma_obs, 0.1f);

    if (!weights_.defined() || weights_.numel() != num_particles_) 
    {
        weights_ = torch::ones({num_particles_},
                    torch::TensorOptions().device(device_).dtype(torch::kFloat32))
                   / (float)num_particles_;
    }

    // =========================
    // 1) Obs tensor (1,C,H,W)
    // =========================
    cv::Mat mask_float;
    if (bev_mask.type() != CV_32F && bev_mask.type() != CV_32FC3) 
    {
        bev_mask.convertTo(mask_float, CV_32F);
    } 
    else 
    {
        mask_float = bev_mask;
    }

    if (!mask_float.isContinuous()) 
    {
        mask_float = mask_float.clone();
    }

    const int H = mask_float.rows;
    const int W = mask_float.cols;
    if (H != obs_size_ || W != obs_size_) return;

    auto mask_tensor = torch::from_blob(
            (void*)mask_float.data,
            {H, W, mask_float.channels()},
            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .permute({2, 0, 1})
        .unsqueeze(0)
        .to(device_);

    // 对硬标签 mask，w_conf 就是 mask 本身
    auto w_conf = mask_tensor;
    auto total_weight = w_conf.sum().clamp_min(1e-6); // scalar

    if (total_weight.item<float>() < 1.0f) 
    {
        return;
    }

    // =========================
    // 2) Sample global TDF
    // =========================
    torch::Tensor grid;
    {
        CudaScopedTimer t("PF::Update/AffineGrid", false);
        grid = ComputeAffineMatrices();
    }

    torch::Tensor sampled_tdf;
    {
        CudaScopedTimer t("PF::Update/GridSample", false);
        auto map_batch = global_tdf_.expand({num_particles_, -1, -1, -1});
        sampled_tdf = torch::nn::functional::grid_sample(
            map_batch, grid,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true));
    }

    // =========================
    // 3) Weighted MSE cost
    // =========================
    auto dist_sq = sampled_tdf.pow(2);
    auto weighted_sse = (dist_sq * w_conf).sum({1, 2, 3}); // (N)
    auto cost_mse = weighted_sse / total_weight;            // (N)

    // =========================
    // 4) 标准高斯似然 (Standard Gaussian Likelihood)
    // =========================
    auto log_like = -cost_mse / (2.0f * sigma_obs * sigma_obs);

    // =========================
    // 5) 贝叶斯权重更新
    // =========================
    auto log_w_prior = torch::log(weights_.clamp_min(1e-12));
    auto log_w_post = log_w_prior + log_like;

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

std::vector<ParticleFilter::ParticleState> ParticleFilter::GetRawParticles()
{
    std::vector<ParticleState> res;
    res.reserve(num_particles_);

    torch::NoGradGuard no_grad;
    auto p_cpu = particles_.to(torch::kFloat64).cpu();
    auto w_cpu = weights_.to(torch::kFloat64).cpu();
    auto p_data = p_cpu.accessor<double, 2>();
    auto w_data = w_cpu.accessor<double, 1>();

    for (int i = 0; i < num_particles_; ++i)
    {
        ParticleState p;
        p.x = p_data[i][0];
        p.y = p_data[i][1];
        p.theta = p_data[i][2];
        p.weight = w_data[i];
        res.push_back(p);
    }
    return res;
}
