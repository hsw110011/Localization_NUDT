#include "ParticleFilter.h"
#include "ParticleScoring.h"
#include "CudaScopedTimer.h"
#include "Motion2D.h"
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace torch::indexing;

namespace sem_loc {
namespace {
torch::Tensor BaselineScore(const torch::Tensor& observation)
{
    return observation.slice(1, 0, 3) * observation.slice(1, 3, 4); // 三类 one-hot × observed mask。
}

torch::Tensor NoEntropyScore(const torch::Tensor& observation)
{
    return observation.slice(1, 0, 3) * observation.slice(1, 4, 5); // 类别概率 × observed mask。
}

torch::Tensor EntropyScore(const torch::Tensor& observation)
{
    return NoEntropyScore(observation) * (1.0 - observation.slice(1, 3, 4).clamp(0.0, 1.0));
}
}

torch::Tensor ObservationWeights(const torch::Tensor& observation, ScoreMode mode)
{
    switch (mode) {
    case ScoreMode::Baseline:  return BaselineScore(observation);
    case ScoreMode::NoEntropy: return NoEntropyScore(observation);
    case ScoreMode::Entropy:   return EntropyScore(observation);
    }
    throw std::invalid_argument("unknown particle filter method");
}

torch::Tensor ComputeParticleCosts(const torch::Tensor& tdf, const torch::Tensor& grids,
                                   const torch::Tensor& weights)
{
    auto sampled = torch::nn::functional::grid_sample(
        tdf.expand({grids.size(0), -1, -1, -1}), grids,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
    // 变换后越界的观测仍参与比较，所有类别通道均取最大归一化距离 1。
    auto inside = grids.abs().le(1.0).all(-1).unsqueeze(1);
    sampled.masked_fill_(inside.logical_not(), 1.0);
    return (sampled * weights).sum({1, 2, 3});
}
// CPU 逐粒子采样，与 GPU 使用相同的线性求和和严格越界惩罚。
torch::Tensor ComputeForloopParticleCosts(const cv::Mat& tdf, const torch::Tensor& pixel_poses,
                                          const torch::Tensor& weights)
{
    const int height = weights.size(2), width = weights.size(3);
    auto poses = pixel_poses.contiguous();
    auto pose = poses.accessor<double, 2>();
    auto hwc_weights = weights.squeeze(0).permute({1, 2, 0}).contiguous();
    cv::Mat weight_image(height, width, CV_32FC3, hwc_weights.data_ptr<float>());
    std::vector<cv::Mat> weight_channels;
    cv::split(weight_image, weight_channels);
    auto costs = torch::empty({pixel_poses.size(0)}, weights.options());
    auto cost = costs.accessor<float, 1>();
    for (int64_t i = 0; i < pixel_poses.size(0); ++i) {
        const double angle = (90.0 - pose[i][2]) * DEG_TO_RAD;
        const double c = std::cos(angle), s = std::sin(angle);
        const double hx = width / 2, hy = height / 2;
        cv::Mat affine = (cv::Mat_<double>(2, 3) <<
            c, -s, -hx*c + hy*s + pose[i][0],
            s,  c, -hx*s - hy*c + pose[i][1]);
        cv::Mat patch;
        cv::warpAffine(tdf, patch, affine, cv::Size(width, height),
                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                       cv::BORDER_CONSTANT, cv::Scalar(1, 1, 1));
        // 仅常量填充仍会在边缘插值；明确将每个越界坐标整体置为 1。
        for (int row = 0; row < height; ++row) {
            auto* values = patch.ptr<cv::Vec3f>(row);
            for (int col = 0; col < width; ++col) {
                const double sx = c*(col-hx) - s*(row-hy) + pose[i][0];
                const double sy = s*(col-hx) + c*(row-hy) + pose[i][1];
                if (sx < 0 || sx > tdf.cols-1 || sy < 0 || sy > tdf.rows-1)
                    values[col] = cv::Vec3f(1, 1, 1);
            }
        }
        std::vector<cv::Mat> channels;
        cv::split(patch, channels);
        double weighted_sum = 0.0;
        for (int channel = 0; channel < 3; ++channel) {
            cv::Mat weighted;
            cv::multiply(channels[channel], weight_channels[channel], weighted);
            weighted_sum += cv::sum(weighted)[0];
        }
        cost[i] = weighted_sum;
    }
    return costs;
}
} // namespace sem_loc


ParticleFilter::ParticleFilter(int num_particles, const sem_loc::Config& config)
    : config_(config), num_particles_(num_particles), device_(config.backend == sem_loc::ComputeBackend::Forloop ? torch::kCPU : torch::kCUDA)
{
    if (device_.is_cuda() && !torch::cuda::is_available())
    {
        std::cerr << "[ParticleFilter] Warning: CUDA not available. Running on CPU!" << std::endl;
        device_ = torch::kCPU;
    }
    ROS_DEBUG_STREAM("[PF] backend=" << (config_.backend == sem_loc::ComputeBackend::Gpu ? "gpu" : "forloop")
              << ", device=" << device_);
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
    map_data.tdf_map.convertTo(tdf_float, CV_32F, 1.0 / config_.tdf_value_scale);
    cv::max(tdf_float, 0.0, tdf_float);
    cv::min(tdf_float, 1.0, tdf_float);
    int channels = tdf_float.channels();

    if (config_.backend == sem_loc::ComputeBackend::Forloop) {
        cpu_tdf_ = tdf_float;
    } else {
        auto tdf_cpu = torch::from_blob(tdf_float.data, {map_rows_, map_cols_, channels}, torch::kFloat);
        global_tdf_ = tdf_cpu.permute({2, 0, 1}).unsqueeze(0).to(device_).clone();
    }
    // 初始化粒子 (在 ROS Gauss 坐标系下)

    auto options = torch::TensorOptions().device(device_).dtype(torch::kFloat64);
    const double heading = motion2d::Normalize(init_theta * DEG_TO_RAD) * RAD_TO_DEG;
    auto mean = torch::tensor({init_x, init_y, heading}, options);
    auto std = torch::tensor({std_x, std_y, std_theta}, options);
    particles_ = torch::randn({num_particles_, 3}, options) * std + mean;
    auto yaw = particles_.select(1, 2);
    yaw.copy_(yaw - 360.0 * torch::floor((yaw + 180.0) / 360.0));
    weights_ = torch::ones({num_particles_}, device_) / num_particles_;
    ROS_DEBUG_STREAM("PF initialized at " << init_x << ", " << init_y << ", " << init_theta);
}

void ParticleFilter::Predict(double local_dx, double local_dy, double dtheta)
{
    CudaScopedTimer timer("PF::Predict", false, device_.is_cuda());
    const double distance = std::hypot(local_dx, local_dy);
    const double delta_deg = motion2d::Normalize(dtheta * DEG_TO_RAD) * RAD_TO_DEG;
    if (distance < 1e-12 && std::abs(delta_deg) < 1e-12) return;

    // 沿用旧版三自由度运动噪声；UTM 状态保持 double 精度。
    const double sigma_fwd = 0.05 * distance + 0.02;
    const double sigma_lat = 0.01 * distance + 0.005;
    const double sigma_yaw = 0.15 * std::abs(delta_deg) + distance; // deg
    auto noise = torch::randn({num_particles_, 3}, particles_.options());
    auto theta = particles_.select(1, 2);
    if (distance >= 1e-12) {
        auto radians = theta * DEG_TO_RAD;
        auto c = torch::cos(radians), s = torch::sin(radians);
        auto dx = local_dx + noise.select(1, 0) * sigma_fwd;
        auto dy = local_dy + noise.select(1, 1) * sigma_lat;
        particles_.select(1, 0) += c * dx - s * dy;
        particles_.select(1, 1) += s * dx + c * dy;
    }
    // 先用旧航向旋转位移，再叠加里程计相对旋转及噪声。
    auto next_yaw = theta + delta_deg + noise.select(1, 2) * sigma_yaw;
    theta.copy_(next_yaw - 360.0 * torch::floor((next_yaw + 180.0) / 360.0));
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

    torch::Tensor cx = pixel_pair.first.to(torch::kFloat32);  // 直接使用，不要重新 from_blob
    torch::Tensor cy = pixel_pair.second.to(torch::kFloat32);

    auto theta_rot = ((90-theta)*DEG_TO_RAD).to(torch::kFloat32);     //90 - 朝向角
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

// 三种观测模型只负责生成权重；地图采样和贝叶斯更新共用。
void ParticleFilter::Update(const cv::Mat& bev_data, float score_lambda)
{
    CudaScopedTimer timer_total("PF::Update", false, device_.is_cuda());
    torch::NoGradGuard no_grad;
    if (bev_data.empty()) return;
    const bool certain = config_.score_mode == sem_loc::ScoreMode::Baseline;
    if (bev_data.type() != (certain ? CV_8UC4 : CV_32FC(5)) ||
        bev_data.rows != obs_size_ || bev_data.cols != obs_size_)
        throw std::invalid_argument("BEV type or dimensions do not match the configured score mode");

    cv::Mat observation;
    bev_data.convertTo(observation, CV_32F);
    if (!observation.isContinuous()) observation = observation.clone();
    auto obs_tensor = torch::from_blob(observation.data,
        {observation.rows, observation.cols, observation.channels()}, torch::kFloat32)
        .clone().permute({2, 0, 1}).unsqueeze(0).to(device_);

    auto w_conf = sem_loc::ObservationWeights(obs_tensor, config_.score_mode);
    if (w_conf.sum().item<float>() <= 0.0f) return; // 无观测或全部为最大熵时保留先验。

    torch::Tensor score;
    if (config_.backend == sem_loc::ComputeBackend::Forloop) {
        // 粒子与 TDF 均在 CPU，沿用原逐粒子坐标转换。
        auto pixel_poses = torch::empty({num_particles_, 3}, particles_.options());
        auto pose = pixel_poses.accessor<double, 2>();
        auto particles = particles_.accessor<double, 2>();
        for (int i = 0; i < num_particles_; ++i) {
            const auto blh = converter_->gauss_to_wgs84(particles[i][0], particles[i][1]);
            const auto pixel = converter_->wgs84_to_pixel(blh.Lon, blh.Lat);
            pose[i][0] = pixel.x;
            pose[i][1] = pixel.y;
            pose[i][2] = particles[i][2];
        }
        score = sem_loc::ComputeForloopParticleCosts(cpu_tdf_, pixel_poses, w_conf);
    } else {
        auto grids = ComputeAffineMatrices();
        score = sem_loc::ComputeParticleCosts(global_tdf_, grids, w_conf);
    }
    auto log_like = -score_lambda * score;
    auto log_w_post = torch::log(weights_.clamp_min(1e-12)) + log_like;
    auto w_unnorm = torch::exp(log_w_post - log_w_post.max());
    auto w_sum = w_unnorm.sum();
    if (!torch::isfinite(w_sum).item<bool>() || w_sum.item<float>() < 1e-12f) {
        weights_.fill_(1.0f / num_particles_);
        return;
    }
    weights_ = w_unnorm / w_sum;


}

/*低方差采样*/
void ParticleFilter::Resample()
{
    CudaScopedTimer timer("PF::Resample", false, device_.is_cuda());
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
    CudaScopedTimer timer("PF::EstimatePose", false, device_.is_cuda());
    // 标准粒子滤波: 全部粒子加权平均
    auto w = weights_.to(torch::kFloat64);
    w = w / w.sum();

    auto mean_x = (particles_.index({Slice(), 0}) * w).sum().item<double>();
    auto mean_y = (particles_.index({Slice(), 1}) * w).sum().item<double>();

    // 加权圆均值，避免 +/-180 度两侧的粒子被平均到 0 度。
    auto radians = particles_.select(1, 2) * DEG_TO_RAD;
    const double sin_mean = (torch::sin(radians) * w).sum().item<double>();
    const double cos_mean = (torch::cos(radians) * w).sum().item<double>();
    return {mean_x, mean_y, std::atan2(sin_mean, cos_mean) * RAD_TO_DEG};
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
