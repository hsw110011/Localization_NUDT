#include "ParticleFilter_forloop.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>

// ============================================================================
// For-loop baseline 粒子滤波器实现
//
// 数学逻辑严格等价于 ParticleFilter_our.cpp (batch/libtorch 版本):
//   - 运动模型: SE(2) 右乘 + 各向异性噪声  (同参数)
//   - 观测模型: TDF 加权 MSE + 语义熵可靠性  (同公式)
//   - 权重更新: 贝叶斯 log w_post = log w_prior + log_like  (同逻辑)
//   - 重采样:   低方差系统重采样  (同算法)
//
// 唯一区别: batch tensor 操作 → 逐粒子 for-loop (CPU)
// ============================================================================

ParticleFilterForloop::ParticleFilterForloop(int num_particles)
    : num_particles_(num_particles),
      rng_(std::chrono::steady_clock::now().time_since_epoch().count())
{
}

// ============================================================================
// Init: 高斯初始化 (等价于 torch::randn * std + mean)
// ============================================================================
void ParticleFilterForloop::Init(const SatelliteData& map_data,
                                  double init_x, double init_y, double init_theta,
                                  double std_x, double std_y, double std_theta)
{
    converter_ = std::make_shared<CoordConverter>(map_data);
    map_rows_ = map_data.tdf_map.rows;
    map_cols_ = map_data.tdf_map.cols;
    map_channels_ = map_data.tdf_map.channels();

    // TDF 地图存为 CPU float (HxWxC)
    map_data.tdf_map.convertTo(global_tdf_, CV_32F);

    // 粒子高斯初始化
    std::normal_distribution<double> dist_x(init_x, std_x);
    std::normal_distribution<double> dist_y(init_y, std_y);
    std::normal_distribution<double> dist_theta(init_theta, std_theta);

    particles_.resize(num_particles_);
    weights_.assign(num_particles_, 1.0 / num_particles_);

    for (int i = 0; i < num_particles_; ++i) {
        particles_[i].x     = dist_x(rng_);
        particles_[i].y     = dist_y(rng_);
        particles_[i].theta = dist_theta(rng_);
    }

    cost_mse_.resize(num_particles_, 0.0);
    log_like_.resize(num_particles_, 0.0);

    std::cout << "[ForLoopPF] Init done, N=" << num_particles_
              << " at gauss_deg (" << init_x << ", " << init_y << ", " << init_theta << ")" << std::endl;
}

// ============================================================================
// Predict: SE(2) 右乘 + 各向异性噪声 (逐粒子 for-loop)
//
// 与 batch 版本完全等价:
//   dx_noisy  = local_dx + N(0, sigma_fwd)
//   dy_noisy  = local_dy + N(0, sigma_lat)
//   dth_noisy = dtheta   + N(0, sigma_yaw)
//
//   dx_global = cos(θ) * dx_noisy - sin(θ) * dy_noisy
//   dy_global = sin(θ) * dx_noisy + cos(θ) * dy_noisy
//   θ_new     = θ + dth_noisy,  归一化到 (-180, 180]
// ============================================================================
void ParticleFilterForloop::Predict(double local_dx, double local_dy, double dtheta)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    float distance = std::sqrt(
        static_cast<float>(local_dx * local_dx + local_dy * local_dy));
    float abs_dtheta = std::abs(static_cast<float>(dtheta));

    if (distance < 1e-6f && abs_dtheta < 1e-6f) {
        last_timing_.predict_ms = 0.0;
        return;
    }

    // === 各向异性噪声参数 (与 batch 版本完全相同) ===
    const float alpha1 = 0.05f, alpha2 = 0.02f;   // 纵向
    const float alpha3 = 0.01f, alpha4 = 0.005f;   // 侧向
    const float alpha5 = 0.15f, alpha6 = 1.0f;     // 旋转

    float sigma_fwd = alpha1 * distance   + alpha2;
    float sigma_lat = alpha3 * distance   + alpha4;
    float sigma_yaw = alpha5 * abs_dtheta + alpha6 * distance;

    std::normal_distribution<double> n_fwd(0.0, sigma_fwd);
    std::normal_distribution<double> n_lat(0.0, sigma_lat);
    std::normal_distribution<double> n_yaw(0.0, sigma_yaw);

    for (int i = 0; i < num_particles_; ++i) {
        double dx_noisy  = local_dx + n_fwd(rng_);
        double dy_noisy  = local_dy + n_lat(rng_);
        double dth_noisy = dtheta   + n_yaw(rng_);

        // SE(2) 右乘: 车体局部增量 → 全局 Gauss 坐标系
        double theta_rad = particles_[i].theta * DEG_TO_RAD;
        double cos_th = std::cos(theta_rad);
        double sin_th = std::sin(theta_rad);

        double dx_global = cos_th * dx_noisy - sin_th * dy_noisy;
        double dy_global = sin_th * dx_noisy + cos_th * dy_noisy;

        particles_[i].x     += dx_global;
        particles_[i].y     += dy_global;
        particles_[i].theta += dth_noisy;

        // 角度归一化到 (-180, 180]  (与 batch 版 floor 公式一致)
        particles_[i].theta -= 360.0 * std::floor(
            (particles_[i].theta + 180.0) / 360.0);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_timing_.predict_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// Update: TDF 加权 MSE 观测模型 (逐粒子 for-loop)
//
// 与 batch 版本等价流程:
//   1) w_conf = probs * clamp(1 - entropy, 0, 1)          —— 置信度权重
//   2) total_weight = Σ w_conf                            —— 全局归一化因子
//   3) 对每个粒子 i:
//      a) gauss→wgs84→pixel, 构建仿射矩阵
//      b) warpAffine 采样 TDF 地图 → 400x400 局部 patch
//      c) cost_mse[i] = Σ_c Σ_hw (tdf_c² × w_conf_c) / total_weight
//   4) log_like = -cost_mse / (2σ²)
//   5) log w_post = log w_prior + log_like
//   6) Log-Sum-Exp 归一化
//
// 采样等价说明:
//   batch 版使用 grid_sample(bilinear, border, align_corners=true);
//   for-loop 版使用 warpAffine(INTER_LINEAR, BORDER_REPLICATE),
//   仿射矩阵从 ComputeAffineMatrices 的数学公式直接推导, 保证坐标映射一致.
// ============================================================================
void ParticleFilterForloop::Update(const cv::Mat& bev_data, float sigma_obs)
{
    auto t_total = std::chrono::high_resolution_clock::now();

    // ---------- 0) 检查 (与 batch 版一致) ----------
    if (bev_data.empty()) return;
    if (bev_data.type() != CV_32FC4) return;
    if (!bev_data.isContinuous()) return;

    const int H = bev_data.rows, W = bev_data.cols;
    if (H != obs_size_ || W != obs_size_) return;

    sigma_obs = std::max(sigma_obs, 0.1f);

    // 确保权重已初始化
    if (weights_.empty() || (int)weights_.size() != num_particles_) {
        weights_.assign(num_particles_, 1.0 / num_particles_);
    }

    // ---------- 1) 观测数据准备 ----------
    auto t_prep = std::chrono::high_resolution_clock::now();

    // 分离 4 通道: [0..2] = probs (Road, Plant, Building), [3] = entropy
    std::vector<cv::Mat> ch4;
    cv::split(bev_data, ch4);

    // reliability = clamp(1 - entropy, 0, 1)
    cv::Mat reliability;
    cv::subtract(cv::Scalar(1.0f), ch4[3], reliability);
    reliability = cv::max(reliability, 0.0f);
    reliability = cv::min(reliability, 1.0f);

    // w_conf[c] = probs[c] * reliability    (1, 3, H, W) 在 batch 版是广播乘)
    std::vector<cv::Mat> w_conf(3);
    double total_weight = 0.0;
    for (int c = 0; c < 3; ++c) {
        cv::multiply(ch4[c], reliability, w_conf[c]);
        total_weight += cv::sum(w_conf[c])[0];
    }
    total_weight = std::max(total_weight, 1e-6);

    auto t_prep_end = std::chrono::high_resolution_clock::now();
    last_timing_.update_prep_ms =
        std::chrono::duration<double, std::milli>(t_prep_end - t_prep).count();

    // ---------- 2) 逐粒子: 仿射采样 + 代价计算 ----------
    auto t_sample_start = std::chrono::high_resolution_clock::now();

    cost_mse_.resize(num_particles_);
    log_like_.resize(num_particles_);

    double sample_ms_accum = 0.0;
    double cost_ms_accum   = 0.0;

    for (int i = 0; i < num_particles_; ++i) {
        // (a) 坐标转换: gauss → wgs84 → pixel
        //     与 batch 版 ComputeAffineMatrices 中
        //     converter_->gauss_to_wgs84_gpu / wgs84_to_pixel_gpu 等价
        auto t_s0 = std::chrono::high_resolution_clock::now();

        BLH_Point blh = converter_->gauss_to_wgs84(particles_[i].x, particles_[i].y);
        cv::Point2f pix = converter_->wgs84_to_pixel(blh.Lon, blh.Lat);

        // (b) 构建仿射矩阵 —— 严格等价于 batch 版 ComputeAffineMatrices
        //
        //   batch 版公式:
        //     theta_rot = (90 - theta) * DEG_TO_RAD
        //     src_col = grid_x * cos_t - grid_y * sin_t + cx
        //     src_row = grid_x * sin_t + grid_y * cos_t + cy
        //   其中 grid_x = j - 200, grid_y = i - 200
        //
        //   等价 2×3 仿射矩阵:
        //     M = [[cos_t, -sin_t, -200*cos_t + 200*sin_t + cx],
        //          [sin_t,  cos_t, -200*sin_t - 200*cos_t + cy]]
        double theta_rot = (90.0 - particles_[i].theta) * DEG_TO_RAD;
        double cos_t = std::cos(theta_rot);
        double sin_t = std::sin(theta_rot);
        double cx = static_cast<double>(pix.x);
        double cy = static_cast<double>(pix.y);

        cv::Mat M = (cv::Mat_<double>(2, 3) <<
            cos_t, -sin_t, -200.0 * cos_t + 200.0 * sin_t + cx,
            sin_t,  cos_t, -200.0 * sin_t - 200.0 * cos_t + cy);

        // (c) warpAffine 采样 TDF 地图
        //     注意: M 是按 "dst(x,y)->src(x,y)" 推导的, 必须开启 WARP_INVERSE_MAP。
        //     否则 OpenCV 会把 M 当作 src->dst 再求逆, 导致旋转方向/平移与 batch 不一致。
        cv::Mat patch;
        cv::warpAffine(global_tdf_, patch, M,
                        cv::Size(obs_size_, obs_size_),
                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);

        auto t_s1 = std::chrono::high_resolution_clock::now();
        sample_ms_accum += std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();

        // (d) 加权 MSE 代价
        //     cost_mse[i] = Σ_c Σ_{h,w} [tdf_c(h,w)² × w_conf_c(h,w)] / total_weight
        auto t_c0 = std::chrono::high_resolution_clock::now();

        std::vector<cv::Mat> tdf_ch;
        cv::split(patch, tdf_ch);

        double weighted_sse = 0.0;
        int nc = std::min(static_cast<int>(tdf_ch.size()), 3);
        for (int c = 0; c < nc; ++c) {
            cv::Mat dsq;
            cv::multiply(tdf_ch[c], tdf_ch[c], dsq);   // tdf²
            cv::Mat wtd;
            cv::multiply(dsq, w_conf[c], wtd);          // tdf² × w_conf
            weighted_sse += cv::sum(wtd)[0];
        }
        cost_mse_[i] = weighted_sse / total_weight;

        auto t_c1 = std::chrono::high_resolution_clock::now();
        cost_ms_accum += std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
    }

    last_timing_.update_sample_ms = sample_ms_accum;
    last_timing_.update_cost_ms   = cost_ms_accum;

    // ---------- 3) 贝叶斯权重更新 (与 batch 版完全一致) ----------
    //
    //   log_like = -cost_mse / (2 * sigma_obs²)
    //   log w_post = log w_prior + log_like
    //   w = exp(log w_post - max) / Σ exp(...)
    auto t_w = std::chrono::high_resolution_clock::now();

    double sigma_sq_2 = 2.0 * sigma_obs * sigma_obs;
    for (int i = 0; i < num_particles_; ++i) {
        log_like_[i] = -cost_mse_[i] / sigma_sq_2;
    }

    std::vector<double> log_w_post(num_particles_);
    for (int i = 0; i < num_particles_; ++i) {
        log_w_post[i] = std::log(std::max(weights_[i], 1e-12)) + log_like_[i];
    }

    // Log-Sum-Exp 数值稳定归一化
    double max_log = *std::max_element(log_w_post.begin(), log_w_post.end());
    double w_sum = 0.0;
    for (int i = 0; i < num_particles_; ++i) {
        weights_[i] = std::exp(log_w_post[i] - max_log);
        w_sum += weights_[i];
    }

    if (!std::isfinite(w_sum) || w_sum < 1e-12) {
        std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
    } else {
        for (int i = 0; i < num_particles_; ++i) {
            weights_[i] /= w_sum;
        }
    }

    auto t_w_end = std::chrono::high_resolution_clock::now();
    last_timing_.update_weight_ms =
        std::chrono::duration<double, std::milli>(t_w_end - t_w).count();
    last_timing_.update_total_ms =
        std::chrono::duration<double, std::milli>(t_w_end - t_total).count();
    last_timing_.frame_total_ms =
        last_timing_.predict_ms + last_timing_.update_total_ms;

    // 累计统计
    cum_predict_ms_ += last_timing_.predict_ms;
    cum_update_ms_  += last_timing_.update_total_ms;
    cum_total_ms_   += last_timing_.frame_total_ms;
    ++frame_count_;

    // Debug 输出 (格式与 batch 版 printf 对齐)
    double min_c = *std::min_element(cost_mse_.begin(), cost_mse_.end());
    double max_c = *std::max_element(cost_mse_.begin(), cost_mse_.end());
    double mean_c = std::accumulate(cost_mse_.begin(), cost_mse_.end(), 0.0) / num_particles_;
    double neff = GetNeff();
    double w_max = *std::max_element(weights_.begin(), weights_.end());

    printf("[ForLoopPF] cost[min,mean,max]=[%.3f, %.3f, %.3f], Neff=%.1f, w_max=%.3e\n",
           min_c, mean_c, max_c, neff, w_max);
    PrintTiming();
}

// ============================================================================
// Resample: 低方差系统重采样 (与 batch 版算法一致)
// ============================================================================
void ParticleFilterForloop::Resample()
{
    std::cout << "[ForLoopPF] Systematic Resampling" << std::endl;

    // 归一化
    double w_sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (!std::isfinite(w_sum) || w_sum < 1e-9) {
        std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
        return;
    }
    for (auto& w : weights_) w /= w_sum;

    // CDF
    std::vector<double> cdf(num_particles_);
    cdf[0] = weights_[0];
    for (int i = 1; i < num_particles_; ++i) {
        cdf[i] = cdf[i - 1] + weights_[i];
    }
    cdf.back() = 1.0; // 与 batch 版 cdf[-1]=1.0 一致

    // 系统采样: u0 ~ Uniform(0, 1/N), 步长 1/N
    const int N = num_particles_;
    double step = 1.0 / N;
    std::uniform_real_distribution<double> unif(0.0, step);
    double u0 = unif(rng_);

    std::vector<Particle> new_particles(N);
    int idx = 0;
    for (int j = 0; j < N; ++j) {
        double uj = u0 + j * step;
        while (uj > cdf[idx] && idx < N - 1) ++idx;
        new_particles[j] = particles_[idx];
    }

    particles_ = std::move(new_particles);
    std::fill(weights_.begin(), weights_.end(), 1.0 / N);
}

// ============================================================================
// EstimatePose: 加权均值 + 圆形均值处理角度 (与 batch 版一致)
// ============================================================================
std::vector<double> ParticleFilterForloop::EstimatePose()
{
    double w_sum = std::accumulate(weights_.begin(), weights_.end(), 0.0) + 1e-12;

    double mean_x = 0.0, mean_y = 0.0;
    double sum_sin = 0.0, sum_cos = 0.0;

    for (int i = 0; i < num_particles_; ++i) {
        double wi = weights_[i] / w_sum;
        mean_x += particles_[i].x * wi;
        mean_y += particles_[i].y * wi;

        double thr = particles_[i].theta * DEG_TO_RAD;
        sum_sin += std::sin(thr) * wi;
        sum_cos += std::cos(thr) * wi;
    }

    double mean_theta = std::atan2(sum_sin, sum_cos) * RAD_TO_DEG;
    return {mean_x, mean_y, mean_theta};
}

// ============================================================================
// GetNeff
// ============================================================================
double ParticleFilterForloop::GetNeff() const
{
    double sum_sq = 0.0;
    for (auto w : weights_) sum_sq += w * w;
    return 1.0 / (sum_sq + 1e-12);
}

// ============================================================================
// GetRawParticles (字段与 ParticleFilter::ParticleState 一致)
// ============================================================================
std::vector<ParticleFilterForloop::ParticleState> ParticleFilterForloop::GetRawParticles()
{
    std::vector<ParticleState> res(num_particles_);
    for (int i = 0; i < num_particles_; ++i) {
        res[i].x      = particles_[i].x;
        res[i].y      = particles_[i].y;
        res[i].theta  = particles_[i].theta;  // degrees
        res[i].weight = weights_[i];
    }
    return res;
}

// ============================================================================
// PrintTiming: 单帧耗时
// ============================================================================
void ParticleFilterForloop::PrintTiming() const
{
    printf("[ForLoopPF] Predict: %.2f ms | Update: %.2f ms "
           "(prep=%.2f, sample=%.2f, cost=%.2f, weight=%.2f) | Total: %.2f ms\n",
           last_timing_.predict_ms,
           last_timing_.update_total_ms,
           last_timing_.update_prep_ms,
           last_timing_.update_sample_ms,
           last_timing_.update_cost_ms,
           last_timing_.update_weight_ms,
           last_timing_.frame_total_ms);
}

// ============================================================================
// PrintCumulativeTiming: 累计耗时 (程序退出前调用)
// ============================================================================
void ParticleFilterForloop::PrintCumulativeTiming() const
{
    std::cout << "\n========== ForLoop PF Cumulative Time ==========" << std::endl;
    std::cout << "  Total frames : " << frame_count_ << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Predict total: " << cum_predict_ms_ << " ms" << std::endl;
    std::cout << "  Update total : " << cum_update_ms_  << " ms" << std::endl;
    std::cout << "  Frame total  : " << cum_total_ms_   << " ms" << std::endl;
    if (frame_count_ > 0) {
        std::cout << "  Avg per frame: " << cum_total_ms_ / frame_count_ << " ms" << std::endl;
    }
    std::cout << "================================================\n" << std::endl;
}
