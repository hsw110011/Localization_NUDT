#ifndef PARTICLE_FILTER_FORLOOP_H
#define PARTICLE_FILTER_FORLOOP_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

#include "Tool.h"

// ============================================================================
// For-loop baseline 粒子滤波器 (CPU-only, 无 libtorch 运行时依赖)
//
// 与 ParticleFilter (batch/libtorch 版本) 数学上等价, 仅把 batch tensor 操作
// 改写为逐粒子 for-loop, 用于:
//   1. 正确性对比 (cost / weight / pose 是否一致)
//   2. 耗时对比 (GPU batch vs CPU for-loop)
//
// 接口签名与 ParticleFilter 完全一致, 可并排调用.
// ============================================================================
class ParticleFilterForloop {
public:
    ParticleFilterForloop(int num_particles = 168);
    ~ParticleFilterForloop() = default;

    // ---- 与 ParticleFilter 对齐的公共接口 ----

    void Init(const SatelliteData& map_data,
              double init_x, double init_y, double init_theta,
              double std_x = 2.0, double std_y = 2.0, double std_theta = 0.5);

    // 运动模型: local_dx/dy (m), dtheta (deg)
    void Predict(double local_dx, double local_dy, double dtheta);

    // 观测更新: bev_data = 400x400 CV_32FC4 (Road,Plant,Building,Entropy)
    void Update(const cv::Mat& bev_data, float sigma_obs = 1.0);

    // 加权均值位姿估计, 返回 [x, y, theta(deg)]
    std::vector<double> EstimatePose();

    // 系统重采样 (低方差)
    void Resample();

    // 有效粒子数 N_eff = 1 / Σ(w_i²)
    double GetNeff() const;

    // 粒子状态 (与 ParticleFilter::ParticleState 字段一致)
    struct ParticleState {
        double x;
        double y;
        double theta;   // degrees
        double weight;
    };
    std::vector<ParticleState> GetRawParticles();

    // ---- 可访问的中间量 (用于和 batch 版对比) ----
    const std::vector<double>& GetCostMse() const { return cost_mse_; }
    const std::vector<double>& GetLogLike() const { return log_like_; }

    // ---- 耗时统计 ----
    struct TimingStats {
        double predict_ms     = 0;
        double update_prep_ms = 0;   // 观测数据准备
        double update_sample_ms = 0; // warpAffine 采样
        double update_cost_ms = 0;   // 代价计算
        double update_weight_ms = 0; // 权重更新
        double update_total_ms = 0;
        double frame_total_ms  = 0;
    };

    TimingStats GetLastTiming() const { return last_timing_; }
    void PrintTiming() const;
    void PrintCumulativeTiming() const;

private:
    int num_particles_;

    struct Particle {
        double x, y, theta; // gauss 坐标, theta 单位 degrees
    };

    std::vector<Particle> particles_;
    std::vector<double> weights_;

    // 地图 (CPU float, HxWxC)
    cv::Mat global_tdf_;
    int map_rows_, map_cols_, map_channels_;

    std::shared_ptr<CoordConverter> converter_;
    int obs_size_ = 400;

    // 中间量
    std::vector<double> cost_mse_;
    std::vector<double> log_like_;

    // 单帧计时
    TimingStats last_timing_;

    // 累计计时
    double cum_predict_ms_ = 0;
    double cum_update_ms_  = 0;
    double cum_total_ms_   = 0;
    int    frame_count_    = 0;

    // 随机引擎
    std::mt19937 rng_;
};

#endif // PARTICLE_FILTER_FORLOOP_H
