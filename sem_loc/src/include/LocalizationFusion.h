// #ifndef LOCALIZATION_FUSION_H
// #define LOCALIZATION_FUSION_H

// #include <vector>
// #include <cmath>
// #include <iostream>
// #include <memory>

// // Eigen & GTSAM Includes
// #include <Eigen/Dense>
// #include <gtsam/geometry/Pose2.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/Values.h>
// #include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/inference/Symbol.h>

// /**
//  * @brief 3DOF 粒子滤波与 Fast-LIO 里程计融合的因子图优化器
//  */
// class LocalizationFusion {
// public:
//     // --- 数据结构定义 ---
//     struct FusionParticle {
//         double x;       // [m]
//         double y;       // [m]
//         double theta;   // [rad] (-pi ~ pi)
//         double weight;  // 权重
//     };

//     struct FusionResult {
//         double x;
//         double y;
//         double theta;
//         Eigen::Matrix3d cov; // 融合后的协方差
//         bool is_pure_odom;   // 标记本次是否降级为纯里程计推算 (True=Fallback, False=Fusion)
//     };

//     // --- 构造函数 ---
//     LocalizationFusion() {
//         gtsam::ISAM2Params params;
//         params.relinearizeThreshold = 0.1;
//         params.relinearizeSkip = 1;
//         isam_ = std::make_unique<gtsam::ISAM2>(params);

//         Reset();
        
//         // 默认参数
//         // 协方差膨胀系数：用于解决 "PF 内部使用了 Odom 导致的双重计算" 问题
//         // 建议值：1.5 ~ 5.0。越大表示越不完全信任 PF 的协方差。
//         pf_covariance_inflation_ratio_ = 2.0; 
        
//         // PF 信任阈值：如果 PF 协方差迹(trace)超过此值，认为 PF 失效，不加 Prior
//         pf_trust_threshold_ = 5.0; 

//         // 里程计噪声 (x, y, theta)
//         odom_noise_sigma_ << 0.002, 0.002, 0.001; // [m], [m], [rad]
//     }

//     // --- 重置系统 ---
//     void Reset() {
//         isam_ = std::make_unique<gtsam::ISAM2>(); // 重新生成求解器
//         initialized_ = false;
//         key_ = 0;
//     }

//     // --- 核心处理函数 ---
//     FusionResult Process(const std::vector<FusionParticle>& particles, const gtsam::Pose2& current_odom_pose) 
//     {
//         gtsam::NonlinearFactorGraph graph;
//         gtsam::Values initial_estimate;
        
//         // 1. 计算粒子群统计量 (均值 & 协方差)
//         gtsam::Pose2 pf_mean;
//         Eigen::Matrix3d pf_cov;
//         ComputeStatistics(particles, pf_mean, pf_cov);
//         double pf_cov_trace = pf_cov.trace();

//         // 2. 初始化逻辑 (第一帧)
//         if (!initialized_) 
//         {
//             last_odom_pose_ = current_odom_pose;
            
//             // 如果 PF 还没有收敛，就用 Odom 初始位姿，否则用 PF
//             gtsam::Pose2 init_pose = (pf_cov_trace < 100.0) ? pf_mean : current_odom_pose;

//             // 添加初始 Prior (给一个相对较小的噪声以固定原点)
//             auto noise = gtsam::noiseModel::Gaussian::Covariance(pf_cov + Eigen::Matrix3d::Identity() * 1e-3);
//             graph.add(gtsam::PriorFactor<gtsam::Pose2>(gtsam::Symbol('x', key_), init_pose, noise));
//             initial_estimate.insert(gtsam::Symbol('x', key_), init_pose);

//             isam_->update(graph, initial_estimate);
//             last_estimated_pose_ = isam_->calculateEstimate<gtsam::Pose2>(gtsam::Symbol('x', key_));
//             initialized_ = true;

//             // 初始化通常认为使用了观测，因此 is_pure_odom = false
//             return {last_estimated_pose_.x(), last_estimated_pose_.y(), last_estimated_pose_.theta(), pf_cov, false};
//         }

//         // 3. 计算相对运动 (Odometry Delta)
//         // Delta = T_{k-1}^{-1} * T_{k}
//         gtsam::Pose2 odom_delta = last_odom_pose_.between(current_odom_pose);
//         last_odom_pose_ = current_odom_pose;

//         // 4. 推进时间步
//         key_++;
//         gtsam::Symbol prev_key('x', key_ - 1);
//         gtsam::Symbol curr_key('x', key_);

//         // 5. [关键] 始终添加里程计因子 (保证轨迹连续性)
//         auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(odom_noise_sigma_);
//         graph.add(gtsam::BetweenFactor<gtsam::Pose2>(prev_key, curr_key, odom_delta, odom_noise));

//         // 预测当前位姿 (作为优化初值)
//         gtsam::Pose2 pred_pose = last_estimated_pose_.compose(odom_delta);
//         initial_estimate.insert(curr_key, pred_pose);

//         // 6. [关键] 条件添加 PF 观测 (PriorFactor)
//         bool use_pf = (pf_cov_trace < pf_trust_threshold_);
        
//         if (use_pf) {
//             // 膨胀协方差：因为 PF 包含了 Odom 信息，为了防止 Over-confident，手动放大不确定度
//             Eigen::Matrix3d inflated_cov = pf_cov * pf_covariance_inflation_ratio_;
            
//             // 确保协方差正定 (数值稳定性)
//             inflated_cov += Eigen::Matrix3d::Identity() * 1e-6;

//             auto pf_noise = gtsam::noiseModel::Gaussian::Covariance(inflated_cov);
//             graph.add(gtsam::PriorFactor<gtsam::Pose2>(curr_key, pf_mean, pf_noise));
//         }

//         // 7. 执行优化 (iSAM2)
//         isam_->update(graph, initial_estimate);
        
//         // 8. 获取最新结果
//         last_estimated_pose_ = isam_->calculateEstimate<gtsam::Pose2>(curr_key);

//         // 返回结果 (为了性能，直接返回 PF 的 Cov 或 膨胀后的 Cov，不进行复杂的 Marginal 计算)
//         // use_pf 为 true 表示使用了 PF 修正，因此 is_pure_odom 为 (!use_pf)
//         return {last_estimated_pose_.x(), last_estimated_pose_.y(), last_estimated_pose_.theta(), pf_cov, !use_pf};
//     }

//     // --- 静态工具：粒子群统计 (含角度正确处理) ---
//     static void ComputeStatistics(const std::vector<FusionParticle>& particles, 
//                                   gtsam::Pose2& out_mean, 
//                                   Eigen::Matrix3d& out_cov) 
//     {
//         if (particles.empty()) 
//         {
//             out_mean = gtsam::Pose2();
//             out_cov = Eigen::Matrix3d::Identity() * 100.0;
//             return;
//         }

//         double sum_w = 0.0;
//         double w_sine = 0.0;
//         double w_cosine = 0.0;
//         Eigen::Vector2d w_pos(0, 0);

//         for (const auto& p : particles) 
//         {
//             sum_w += p.weight;
//             w_pos.x() += p.weight * p.x;
//             w_pos.y() += p.weight * p.y;
//             w_sine += p.weight * std::sin(p.theta);
//             w_cosine += p.weight * std::cos(p.theta);
//         }

//         // 归一化处理
//         if (sum_w < 1e-9) sum_w = 1e-9;
//         double mean_x = w_pos.x() / sum_w;
//         double mean_y = w_pos.y() / sum_w;
//         double mean_theta = std::atan2(w_sine, w_cosine); // 核心：使用 vector sum 求角度均值

//         out_mean = gtsam::Pose2(mean_x, mean_y, mean_theta);

//         // 计算协方差
//         out_cov.setZero();
//         for (const auto& p : particles) 
//         {
//             double w = p.weight / sum_w;
//             double dx = p.x - mean_x;
//             double dy = p.y - mean_y;
//             double dtheta = p.theta - mean_theta;

//             // [重要] 角度差归一化到 [-pi, pi]
//             while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
//             while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

//             Eigen::Vector3d err;
//             err << dx, dy, dtheta;
//             out_cov += w * (err * err.transpose());
//         }
//     }

//     // 参数设置接口
//     void SetCovarianceInflation(double ratio) { pf_covariance_inflation_ratio_ = ratio; }
//     void SetOdomNoise(double x, double y, double t) { odom_noise_sigma_ << x, y, t; }

// private:
//     std::unique_ptr<gtsam::ISAM2> isam_;
//     gtsam::Pose2 last_estimated_pose_;
//     gtsam::Pose2 last_odom_pose_;
//     uint64_t key_;
//     bool initialized_;

//     // Params
//     double pf_covariance_inflation_ratio_;
//     double pf_trust_threshold_;
//     Eigen::Vector3d odom_noise_sigma_;
// };

// #endif // LOCALIZATION_FUSION_H




#ifndef LOCALIZATION_FUSION_H
#define LOCALIZATION_FUSION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <memory>

// Eigen & GTSAM Includes
#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

/**
 * @brief 3DOF 粒子滤波 + LIO 里程计 SE(2) 因子图融合
 *        - PriorFactor: PF 地图匹配绝对位姿 (动态协方差 + Huber 鲁棒核)
 *        - BetweenFactor: LIO 里程计相对增量
 *        - 滑动窗口防止因子图无限增长
 */
class LocalizationFusion {
public:
    struct FusionParticle {
        double x;       // [m]
        double y;       // [m]
        double theta;   // [rad]
        double weight;
    };

    struct FusionResult {
        double x;
        double y;
        double theta;
        Eigen::Matrix3d cov;
        bool is_pure_odom;
    };

    LocalizationFusion() {
        isam_params_.relinearizeThreshold = 0.1;
        isam_params_.relinearizeSkip = 1;
        
        Reset();
        
        // Odom per-step noise (@10Hz): 应反映 LIO 实际漂移量级
        // 旧值 0.01,0.01,0.0003 导致 GTSAM 几乎 100% 信任里程计
        odom_noise_sigma_ << 0.05, 0.05, 0.003;
        
        // PF 协方差温和膨胀，旧值 5.0 让 PriorFactor 信息量趋近于零
        pf_covariance_inflation_ratio_ = 1.5;
        
        // 粒子群 trace(Cov) 超限 → PF 发散，不添加 PriorFactor
        pf_trust_threshold_ = 50.0;
        
        // Huber 鲁棒核参数 k (Mahalanobis 距离尺度)
        huber_k_ = 1.345;
        
        // 马氏距离平方阈值: 预测 vs PF 偏差过大时拒绝
        mahalanobis_threshold_ = 25.0;
        
        // 滑动窗口大小
        max_graph_keys_ = 200;
    }

    void Reset() {
        isam_ = std::make_unique<gtsam::ISAM2>(isam_params_);
        initialized_ = false;
        key_ = 0;
    }

    // --- 核心处理函数 (接口完全不变) ---
    FusionResult Process(const std::vector<FusionParticle>& particles, const gtsam::Pose2& current_odom_pose) 
    {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial_estimate;
        
        // 1. 计算统计量
        gtsam::Pose2 pf_mean;
        Eigen::Matrix3d pf_cov;
        ComputeStatistics(particles, pf_mean, pf_cov);
        double pf_cov_trace = pf_cov.trace();

        // 2. 初始化逻辑
        if (!initialized_) {
            last_odom_pose_ = current_odom_pose;
            gtsam::Pose2 init_pose = (pf_cov_trace < pf_trust_threshold_) ? pf_mean : current_odom_pose;

            Eigen::Matrix3d init_cov;
            if (pf_cov_trace < pf_trust_threshold_) {
                init_cov = pf_cov + Eigen::Matrix3d::Identity() * 1e-3;
            } else {
                init_cov = Eigen::Matrix3d::Identity() * 10.0;
            }

            auto init_noise = gtsam::noiseModel::Gaussian::Covariance(init_cov);
            graph.add(gtsam::PriorFactor<gtsam::Pose2>(gtsam::Symbol('x', key_), init_pose, init_noise));
            initial_estimate.insert(gtsam::Symbol('x', key_), init_pose);

            isam_->update(graph, initial_estimate);
            last_estimated_pose_ = isam_->calculateEstimate<gtsam::Pose2>(gtsam::Symbol('x', key_));
            initialized_ = true;
            return {last_estimated_pose_.x(), last_estimated_pose_.y(),
                    NormalizeAngle(last_estimated_pose_.theta()), pf_cov, false};
        }

        // 3. 里程计增量
        gtsam::Pose2 odom_delta = last_odom_pose_.between(current_odom_pose);
        last_odom_pose_ = current_odom_pose;

        key_++;
        gtsam::Symbol prev_key('x', key_ - 1);
        gtsam::Symbol curr_key('x', key_);

        // 4. 添加里程计因子 (强约束)
        auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(odom_noise_sigma_);
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(prev_key, curr_key, odom_delta, odom_noise));

        // 预测
        gtsam::Pose2 pred_pose = last_estimated_pose_.compose(odom_delta);
        initial_estimate.insert(curr_key, pred_pose);

        // 5. PF 绝对观测因子 (PriorFactor + 动态协方差 + Huber 鲁棒核)
        bool use_pf = false;
        if (pf_cov_trace < pf_trust_threshold_)
        {
            // Mahalanobis 距离野值检测
            Eigen::Matrix3d cov_check = pf_cov + Eigen::Matrix3d::Identity() * 1e-6;
            gtsam::Vector3 err = pred_pose.localCoordinates(pf_mean);
            double maha2 = err.transpose() * cov_check.inverse() * err;

            if (maha2 < mahalanobis_threshold_)
            {
                use_pf = true;

                // 粒子群协方差 → GTSAM 噪声模型 (温和膨胀)
                Eigen::Matrix3d inflated_cov = pf_cov * pf_covariance_inflation_ratio_;
                inflated_cov += Eigen::Matrix3d::Identity() * 1e-6;

                auto base_noise = gtsam::noiseModel::Gaussian::Covariance(inflated_cov);

                // Huber 鲁棒核: 偶发极端误匹配时自动降权
                auto robust_noise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Huber::Create(huber_k_),
                    base_noise);

                graph.add(gtsam::PriorFactor<gtsam::Pose2>(curr_key, pf_mean, robust_noise));
            }
        }

        // 6. iSAM2 求解
        isam_->update(graph, initial_estimate);
        last_estimated_pose_ = isam_->calculateEstimate<gtsam::Pose2>(curr_key);

        // 滑动窗口: 防止因子图无限增长
        if (key_ >= max_graph_keys_)
        {
            SlidingWindowReset(curr_key, pf_cov);
        }

        return {last_estimated_pose_.x(), last_estimated_pose_.y(),
                NormalizeAngle(last_estimated_pose_.theta()), pf_cov, !use_pf};
    }

    // --- 静态工具 (接口不变) ---
    static void ComputeStatistics(const std::vector<FusionParticle>& particles, 
                                  gtsam::Pose2& out_mean, 
                                  Eigen::Matrix3d& out_cov) 
    {
        if (particles.empty()) {
            out_mean = gtsam::Pose2();
            out_cov = Eigen::Matrix3d::Identity() * 100.0;
            return;
        }

        double sum_w = 0.0;
        double w_sine = 0.0, w_cosine = 0.0;
        Eigen::Vector2d w_pos(0, 0);

        for (const auto& p : particles) {
            sum_w += p.weight;
            w_pos.x() += p.weight * p.x;
            w_pos.y() += p.weight * p.y;
            w_sine += p.weight * std::sin(p.theta);
            w_cosine += p.weight * std::cos(p.theta);
        }

        if (sum_w < 1e-9) sum_w = 1e-9;
        double mean_x = w_pos.x() / sum_w;
        double mean_y = w_pos.y() / sum_w;
        double mean_theta = std::atan2(w_sine, w_cosine); 

        out_mean = gtsam::Pose2(mean_x, mean_y, mean_theta);

        out_cov.setZero();
        for (const auto& p : particles) 
        {
            double w = p.weight / sum_w;
            double dx = p.x - mean_x;
            double dy = p.y - mean_y;
            double dtheta = p.theta - mean_theta;

            dtheta = NormalizeAngle(dtheta);

            Eigen::Vector3d err;
            err << dx, dy, dtheta;
            out_cov += w * (err * err.transpose());
        }
    }

    void SetCovarianceInflation(double ratio) { pf_covariance_inflation_ratio_ = ratio; }
    void SetOdomNoise(double x, double y, double t) { odom_noise_sigma_ << x, y, t; }
    void SetHuberK(double k) { huber_k_ = k; }
    void SetMaxGraphKeys(uint64_t n) { max_graph_keys_ = n; }

private:
    // 角度归一化到 [-π, π]
    static double NormalizeAngle(double a) {
        while (a > M_PI)  a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    // 滑动窗口重置: 保存当前状态，重建因子图
    void SlidingWindowReset(gtsam::Symbol curr_key, const Eigen::Matrix3d& fallback_cov) {
        gtsam::Pose2 saved_pose = last_estimated_pose_;
        gtsam::Pose2 saved_odom = last_odom_pose_;  // 关键: 保留里程计状态

        Eigen::Matrix3d curr_cov;
        try {
            curr_cov = isam_->marginalCovariance(curr_key);
        } catch (...) {
            curr_cov = fallback_cov + Eigen::Matrix3d::Identity() * 1e-2;
        }

        Reset();

        gtsam::NonlinearFactorGraph new_graph;
        gtsam::Values new_initial;
        auto noise = gtsam::noiseModel::Gaussian::Covariance(curr_cov);
        new_graph.add(gtsam::PriorFactor<gtsam::Pose2>(gtsam::Symbol('x', 0), saved_pose, noise));
        new_initial.insert(gtsam::Symbol('x', 0), saved_pose);
        isam_->update(new_graph, new_initial);

        last_estimated_pose_ = saved_pose;
        last_odom_pose_ = saved_odom;  // 关键: 恢复里程计状态
        initialized_ = true;
    }

    std::unique_ptr<gtsam::ISAM2> isam_;
    gtsam::ISAM2Params isam_params_;
    gtsam::Pose2 last_estimated_pose_;
    gtsam::Pose2 last_odom_pose_;
    uint64_t key_;
    bool initialized_;

    double pf_covariance_inflation_ratio_;
    double pf_trust_threshold_;
    double huber_k_;
    double mahalanobis_threshold_;
    uint64_t max_graph_keys_;
    Eigen::Vector3d odom_noise_sigma_;
};

#endif // LOCALIZATION_FUSION_H