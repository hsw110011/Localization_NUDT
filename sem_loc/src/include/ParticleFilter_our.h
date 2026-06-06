#ifndef PARTICLE_FILTER_OUR_H
#define PARTICLE_FILTER_OUR_H

#include <ros/ros.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "Tool.h"

// 粒子滤泼器类 - 使用 LibTorch 进行 GPU 加速
class ParticleFilter {
public:
    // 构造函数
    // num_particles: 粒子数量 (128)
    // device: 运行设备 (CUDA)
    ParticleFilter(int num_particles = 168);
    ~ParticleFilter() = default;

    // 初始化滤波器
    // map_data: 包含全局 TDF 地图的数据
    // init_x, init_y, init_theta: 初始位姿
    // std_x, std_y, std_theta: 初始分布的标准差
    void Init(const SatelliteData& map_data, double init_x, double init_y, double init_theta, 
             double std_x = 2.0, double std_y = 2.0, double std_theta = 0.5);

    // 预测步骤 (Motion Model — Exact SE(2) Composition + 各向异性噪声)
    // local_dx: 车体纵向位移增量 (m, 前进为正)
    // local_dy: 车体侧向位移增量 (m, 左正右负)
    // dtheta:   航向角增量 (deg)
    void Predict(double local_dx, double local_dy, double dtheta);

    // 更新步骤 (Observation Model)
    // bev_mask: 局部 BEV 观测 (400x400)
    // sigma_obs: 观测噪声标准差
    void Update(const cv::Mat& bev_mask, float sigma_obs = 1.0);

    // 状态估计 (加权平均, Weighted Mean)
    // Returns: [x, y, theta]
    std::vector<double> EstimatePose();

    // 重采样 (Low Variance Systematic Resampling)
    void Resample();

    // 有效粒子数 N_eff = 1/Σ(w_i²)
    double GetNeff() const;

    // --- 新增: 获取粒子原始数据 (用于外部可视化) ---
    // Returns: vector of {x, y, theta, weight}
    struct ParticleState {
        double x;
        double y;
        double theta;
        double weight;
    };

    // 获取所有粒子的原始数据 (用于计算协方差/融合)
    std::vector<ParticleState> GetRawParticles();

    // 可视化粒子，返回绘制好的图像（Jet 颜色映射）
    cv::Mat VisualizeParticles(const torch::Tensor& particles,
                               const torch::Tensor& weights,
                               cv::Size map_size,
                               double resolution,
                               double origin_x,
                               double origin_y);

    // 便捷重载：直接使用内部粒子与权重
    cv::Mat VisualizeParticles(cv::Size map_size,
                               double resolution,
                               double origin_x,
                               double origin_y);

private:
    // 计算 Affine Grid 的变换矩阵
    torch::Tensor ComputeAffineMatrices();

    // 成员变量
    int num_particles_;
    torch::Device device_;

    // 粒子状态 [N, 3] -> (x, y, theta)
    torch::Tensor particles_;
    // 粒子权重 [N]
    torch::Tensor weights_;

    // 地图数据 (GPU Tensor)
    // Shape: [1, C, H, W]
    torch::Tensor global_tdf_;
    
    // 地图元数据
    double map_origin_lon_, map_origin_lat_;
    double map_origin_x_, map_origin_y_;
    double res_lon_, res_lat_;
    double map_resolution_;
    double map_width_m_, map_height_m_;
    int map_cols_, map_rows_;
    
    // 观测参数
    double obs_resolution_;
    int obs_size_ = 400; // 观测窗口大小 (Pixels)

    // 坐标转换器智能指针
    std::shared_ptr<CoordConverter> converter_;
};

#endif // PARTICLE_FILTER_H
