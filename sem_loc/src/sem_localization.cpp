#include "CInterface.h"
#include <cv_bridge/cv_bridge.h>
#include <CoordConverter.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <cmath>
#include <algorithm>
#include "cnpy.h"
#include "Tool.h"
#include "Motion2D.h"
#include "ParticleFilter.h"
#include "SemLocConfig.h"
#include "LocalizationFusion.h"
#include "ParticleWeightVisualization.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char **argv)
{
    // =========================
    // 0) ROS / 基础对象初始化
    // =========================
    ros::init(argc, argv, "sem_localization");

    //类对象创建
    ros::NodeHandle nh;
    sem_loc::Config config;
    try {
        config = sem_loc::LoadConfig(std::string(SEM_LOC_SOURCE_DIR) + "/config/sem_loc.yaml");
    } catch (const std::exception& error) {
        ROS_FATAL("Invalid sem_loc config: %s", error.what());
        return 1;
    }
    const bool certain = config.score_mode == sem_loc::ScoreMode::Baseline;
    CInterface interface(nh, certain);
    InputData input;
    Tool tool_func;

    //结构体创建
    SatelliteData sat_data;

    //变量创建
    double3D Base_point;
    WORLD_POINT Odom_World;    //定义的朝向角都是角度输出
    WORLD_POINT GNSS_World;
    WORLD_POINT Loc_World;
    WORLD_POINT GTSAM_World;
    bool Init_localization = false;

    // 两帧时序平滑：当前帧与上一帧均衡融合
    const double kTemporalSmoothAlpha = 1.0; // 当前帧权重
    bool has_prev_loc_world = false;
    WORLD_POINT prev_loc_world;

    // Gauss 坐标系下的辅助变量（用于 Delta 计算）
    WORLD_POINT last_Odom_World;
    double last_map_heading = 0.0; // 预留变量：Robot heading in Gauss Map Frame

    // 原始里程计状态缓存（直接提取局部增量，避免 global→local 绕行）
    motion2d::OdomMotion motion;
    bool have_odom = false, have_gnss = false;
    ros::Time last_bev_stamp;
    ros::Time previous_gnss_stamp, minimum_bev_stamp;
    constexpr double kGpsGapSeconds = 0.2;
    bool recovering_from_gps_gap = false;
    int particle_view_frame = 0;
    const char* method = config.score_mode == sem_loc::ScoreMode::Baseline ? "baseline"
        : config.score_mode == sem_loc::ScoreMode::Entropy ? "ours_entropy (有熵)"
        : "ours_no_entropy (无熵)";
    std::cout << "[Localization] method=" << method
              << ", backend=" << (config.backend == sem_loc::ComputeBackend::Gpu ? "gpu" : "forloop")
              << std::endl;

    // 粒子滤波器
    ParticleFilter pf(config.num_particles, config);
    // 因子图优化器 (3DOF PF + Odom Fusion)
    LocalizationFusion fusion_optimizer;

    ros::Rate rate(10);


    // =========================
    // 1) 加载卫星地图数据 (NPZ)
    // =========================
    bool npz_loaded = tool_func.LoadSatelliteNpz(config.map_path, sat_data);
    if (!npz_loaded || !sat_data.is_valid || sat_data.tdf_map.empty()) {
        ROS_FATAL("Failed to load semantic/TDF map: %s", config.map_path.c_str());
        return 1;
    }
    CoordConverter converter(sat_data);


    // =========================
    // 2) 日志输出文件
    // =========================
    // 记录 Odom / GNSS / PF 结果（可扩展 GTSAM）到 CSV
    const std::string csv_dir = std::string(SEM_LOC_SOURCE_DIR) + "/output";
    particle_weight_vis::MakeDirRecursive(csv_dir);
    std::ofstream csv_file(csv_dir + "/pose_compare_sem_localization.csv", std::ios::out | std::ios::trunc);
    if (!csv_file.is_open())
    {
        std::cerr << "Failed to open CSV for logging." << std::endl;
        return -1;
    }
    // CSV Header
    csv_file << "timestamp,"
             << "odom_x,odom_y,odom_theta,"
             << "pf_x,pf_y,pf_theta,"
             //<< "gtsam_x,gtsam_y,gtsam_theta,"
             << "gnss_x,gnss_y,gnss_theta"
             << std::endl;

    // =========================
    // 3) 可视化对象
    // =========================
    cv::Mat satellite_map = sat_data.satellite_map.clone();

    // 用于存储最新因子图融合结果的变量 (循环外定义)
    LocalizationFusion::FusionResult gtsam_result = {0.0, 0.0, 0.0, {}, true};



    cv::namedWindow("Satellite Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Satellite Map", 800, 800);
    cv::namedWindow("PF Particle Weights", cv::WINDOW_NORMAL);
    cv::resizeWindow("PF Particle Weights", 512, 512);

    // =========================
    // 5) 主循环
    // =========================
    while (ros::ok())
    {
        ros::spinOnce();
        interface.ConvertToLocalData(&input);
        have_odom |= input.Odom_refreshflag;
        have_gnss |= input.Inspvax_refreshflag;

        // INSPVAX 没有标准 ROS 时间戳，用 GPS 回调接收的 ROS 时间检测缺口。
        // bag 使用 --clock 时，暂停回放不会使 GPS 超时。
        if (input.Inspvax_refreshflag) {
            const auto stamp = interface.LastGnssReception();
            if (!previous_gnss_stamp.isZero() &&
                (stamp - previous_gnss_stamp).toSec() > kGpsGapSeconds && Init_localization) {
                Init_localization = false;
                recovering_from_gps_gap = true;
                ROS_WARN("GPS data gap detected; localization will reinitialize.");
            }
            previous_gnss_stamp = stamp;
        }
        const bool gps_available = have_odom && have_gnss &&
            !input.Odom->header.stamp.isZero() && !interface.LastGnssReception().isZero() &&
            std::abs((ros::Time::now() - interface.LastGnssReception()).toSec()) <= kGpsGapSeconds;
        if (!gps_available) {
            if (Init_localization) {
                ROS_WARN("GPS missing for more than 0.2 s; localization paused.");
                recovering_from_gps_gap = true;
                Init_localization = false;
            }
            cv::waitKey(1);
            rate.sleep();
            continue;
        }

        if(input.Odom_refreshflag && !Init_localization)   // 首次初始化或 GPS 恢复后的重初始化
        {

            // (5.1) 初始化 GPS：经度、纬度、高度、航向
            GNSS_World.BLH.Lon = input.Inspvax->longitude;
            GNSS_World.BLH.Lat = input.Inspvax->latitude;
            GNSS_World.BLH.Height = input.Inspvax->altitude;
            GNSS_World.gauss = converter.wgs84_to_gauss(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);
            GNSS_World.heading = input.Inspvax->azimuth;    //角度为单位degree,坐标系(东向为0度)
            GNSS_World.pixel = converter.wgs84_to_pixel(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);

            Odom_World = GNSS_World;    //第一帧时，里程计位置等于GNSS位置
            last_Odom_World = GNSS_World;
            Loc_World = GNSS_World;
            prev_loc_world = Loc_World;
            has_prev_loc_world = true;

            const auto& q0 = input.Odom->pose.pose.orientation;
            motion.Initialize(input.Odom->pose.pose.position.x,
                              input.Odom->pose.pose.position.y,
                              motion2d::OdomYaw(q0.x, q0.y, q0.z, q0.w),
                              GNSS_World.heading * DEG_TO_RAD);

            Base_point = tool_func.GetBase(input.Odom, GNSS_World);   // 得到坐标系转换点（弧度单位）
            pf.Init(sat_data, GNSS_World.gauss.x, GNSS_World.gauss.y, GNSS_World.heading, 4.0, 4.0, 8.0);   // 初始化粒子滤波器
            // 初始化状态（用于下一帧计算增量）
            Init_localization = true;
            fusion_optimizer.Reset();
            // 恢复前缓存的 BEV 不用于新粒子，等待当前时刻之后的观测。
            minimum_bev_stamp = input.Odom->header.stamp;
            last_bev_stamp = ros::Time();
            if (recovering_from_gps_gap) {
                ROS_INFO("GPS restored; particles and odometry reference reinitialized.");
                recovering_from_gps_gap = false;
            }

            ROS_DEBUG("Localization initialized from GNSS and odometry.");
        }
        else if(input.Odom_refreshflag && Init_localization)
        {
            // (5.2.1) 更新 GPS：用于可视化与误差对比
            GNSS_World.BLH.Lon = input.Inspvax->longitude;
            GNSS_World.BLH.Lat = input.Inspvax->latitude;
            GNSS_World.BLH.Height = input.Inspvax->altitude;
            GNSS_World.gauss = converter.wgs84_to_gauss(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);
            GNSS_World.heading = input.Inspvax->azimuth;    //角度为单位degree,坐标系(东向为0度)
            GNSS_World.pixel = converter.wgs84_to_pixel(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);

            // (5.2.2) 更新里程计在各坐标系中的位姿
            Odom_World = tool_func.LocalToGlobal(input.Odom, Base_point);
            Odom_World.BLH = converter.gauss_to_wgs84(Odom_World.gauss.x, Odom_World.gauss.y);
            Odom_World.pixel = converter.wgs84_to_pixel(Odom_World.BLH.Lon, Odom_World.BLH.Lat);

            // T_prev^-1 * T_now gives previous-body translation. Rotate by
            // the map heading propagated from the previous odometry sample.
            const auto& q = input.Odom->pose.pose.orientation;
            const auto step = motion.Advance(input.Odom->pose.pose.position.x,
                                              input.Odom->pose.pose.position.y,
                                              motion2d::OdomYaw(q.x, q.y, q.z, q.w));
            const double delta_yaw = motion2d::Normalize(step.heading - step.previous_heading) * RAD_TO_DEG;
            pf.Predict(step.body.x, step.body.y, delta_yaw);

            last_Odom_World = Odom_World;

            bool should_resample = false;


            // (5.2.5) 按配置选择硬标签或概率观测；话题和编码保持原契约。
            const auto* bev_message = certain ? input.BevMasks : input.BevProbs;
            // 首张 BEV 到达后才更新；同一帧观测不能因里程计更快而重复计权。
            if (!bev_message->data.empty() && bev_message->header.stamp >= minimum_bev_stamp &&
                bev_message->header.stamp != last_bev_stamp)
            {
                cv_bridge::CvImagePtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(*bev_message, certain
                        ? sensor_msgs::image_encodings::TYPE_8UC4
                        : "32FC5");
                } catch (const cv_bridge::Exception& error) {
                    ROS_ERROR("BEV conversion failed: %s", error.what());
                }
                if (cv_ptr && !cv_ptr->image.empty()) {
                    cv::Mat bev_resized;
                    cv::resize(cv_ptr->image, bev_resized, cv::Size(400, 400), 0, 0,
                               certain ? cv::INTER_NEAREST : cv::INTER_LINEAR);
                    if (!certain) {
                        // mask 保持二值，不能随概率一起双线性插值。
                        cv::Mat mask, resized_mask;
                        cv::extractChannel(cv_ptr->image, mask, 4);
                        cv::resize(mask, resized_mask, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
                        cv::insertChannel(resized_mask, bev_resized, 4);
                    }
                    pf.Update(bev_resized, config.score_lambda);
                    last_bev_stamp = bev_message->header.stamp;
                    should_resample = pf.GetNeff() < config.num_particles * (200.0 / 256.0);
                }
            }

            // (5.2.6) 状态估计
            std::vector<double> pf_pose = pf.EstimatePose(); // [x, y, theta]

            // 更新 Loc_World,得到粒子滤波结果
            Loc_World.gauss.x = pf_pose[0];
            Loc_World.gauss.y = pf_pose[1];
            Loc_World.heading = pf_pose[2];
            Loc_World.BLH = converter.gauss_to_wgs84(Loc_World.gauss.x, Loc_World.gauss.y);
            Loc_World.pixel = converter.wgs84_to_pixel(Loc_World.BLH.Lon, Loc_World.BLH.Lat);

            // (5.2.6.1) 前后两帧平滑：当前帧与上一帧各占 50%
            if (has_prev_loc_world)
            {
                WORLD_POINT smoothed_loc = Loc_World;
                smoothed_loc.gauss.x = kTemporalSmoothAlpha * Loc_World.gauss.x +
                                      (1.0 - kTemporalSmoothAlpha) * prev_loc_world.gauss.x;
                smoothed_loc.gauss.y = kTemporalSmoothAlpha * Loc_World.gauss.y +
                                      (1.0 - kTemporalSmoothAlpha) * prev_loc_world.gauss.y;

                // 航向角使用单位圆插值，避免 0/360 附近直接平均导致跳变
                smoothed_loc.heading = Loc_World.heading;

                smoothed_loc.BLH = converter.gauss_to_wgs84(smoothed_loc.gauss.x, smoothed_loc.gauss.y);
                smoothed_loc.pixel = converter.wgs84_to_pixel(smoothed_loc.BLH.Lon, smoothed_loc.BLH.Lat);
                Loc_World = smoothed_loc;
            }
            prev_loc_world = Loc_World;
            has_prev_loc_world = true;

            // ================== [新增] 因子图融合优化 ==================
            // 1) 获取粒子原始数据（Degrees）
            auto raw_particles = pf.GetRawParticles();
            // 打印后验归一化权重；重采样后权重会重新变为均匀分布。
            const auto extrema = std::minmax_element(raw_particles.begin(), raw_particles.end(),
                [](const auto& a, const auto& b) { return a.weight < b.weight; });
            std::cout << "[PF] weight_max=" << std::scientific << std::setprecision(6)
                      << extrema.second->weight << " weight_min=" << extrema.first->weight << std::endl;

            cv::Mat particle_view = particle_weight_vis::BuildParticleWeightView(
                sat_data.satellite_map, converter, raw_particles, Loc_World.pixel,
                particle_view_frame++, pf.GetNeff());
            if (!particle_view.empty()) cv::imshow("PF Particle Weights", particle_view);

            // 2) 准备融合所需数据结构
            std::vector<LocalizationFusion::FusionParticle> fusion_particles;
            fusion_particles.reserve(raw_particles.size());

            // 常量转换

            for(const auto& p : raw_particles)
            {
                fusion_particles.push_back({
                    p.x,
                    p.y,
                    p.theta * DEG_TO_RAD, // Degree -> Radian
                    p.weight
                });
            }

            if (should_resample)
            {
                pf.Resample();
            }

            // 3) 准备当前 LIO 里程计位姿（GTSAM Pose2，Radian）
            // 注意: Odom_World.heading 是 Degree
            gtsam::Pose2 current_odom_pose(
                Odom_World.gauss.x,
                Odom_World.gauss.y,
                Odom_World.heading * DEG_TO_RAD
            );

            // 4) 执行融合 (Process)
            // 自动处理: 计算粒子协方差 -> 判断可信度 -> 决定是 Prior 修正还是 Odom 推算
            gtsam_result = fusion_optimizer.Process(fusion_particles, current_odom_pose);

            // 5) 输出因子图融合结果
            GTSAM_World.heading = gtsam_result.theta * RAD_TO_DEG;
            while(GTSAM_World.heading > 180) GTSAM_World.heading -= 360;
            while(GTSAM_World.heading < -180) GTSAM_World.heading += 360;
            GTSAM_World.gauss.x = gtsam_result.x;
            GTSAM_World.gauss.y = gtsam_result.y;
            GTSAM_World.BLH = converter.gauss_to_wgs84(GTSAM_World.gauss.x, GTSAM_World.gauss.y);
            GTSAM_World.pixel = converter.wgs84_to_pixel(GTSAM_World.BLH.Lon, GTSAM_World.BLH.Lat);


            // CSV 保持原有 0..360 度输出约定。
            if (Loc_World.heading < 0) Loc_World.heading += 360.0;
            if (Loc_World.heading > 360) Loc_World.heading -= 360.0;
            if (Odom_World.heading < 0) Odom_World.heading += 360.0;
            if (Odom_World.heading > 360) Odom_World.heading -= 360.0;

            // 写入 CSV：Odom / 粒子滤波定位 / GTSAM / GNSS 的 Gauss 坐标与朝向
            if (csv_file.is_open())
            {
                double stamp = ros::Time::now().toSec();

                csv_file << std::fixed << std::setprecision(6)
                         << stamp << ","
                         << Odom_World.gauss.x << "," << Odom_World.gauss.y << "," << Odom_World.heading << ","
                         << Loc_World.gauss.x << "," << Loc_World.gauss.y << "," << Loc_World.heading << ","
                        // << GTSAM_World.gauss.x << "," << GTSAM_World.gauss.y << "," << GTSAM_World.heading << ","
                         << GNSS_World.gauss.x << "," << GNSS_World.gauss.y << "," << GNSS_World.heading
                         << std::endl;
            }
        }

        if (Init_localization) {
            cv::circle(satellite_map, Odom_World.pixel, 5, cv::Scalar(0, 0, 0), -1);
            cv::circle(satellite_map, GNSS_World.pixel, 5, cv::Scalar(255, 255, 255), -1);
            cv::circle(satellite_map, Loc_World.pixel, 5, cv::Scalar(0, 255, 255), -1);
        }
        // 图例绘制到显示副本，避免反复覆盖累计轨迹。
        cv::Mat display = satellite_map.clone();
        cv::Rect legend(15, 15, 520, 185);
        legend &= cv::Rect(0, 0, display.cols, display.rows);
        if (legend.area() > 0) {
            cv::Mat roi = display(legend);
            cv::Mat white(roi.size(), roi.type(), cv::Scalar(255, 255, 255));
            cv::addWeighted(white, 0.75, roi, 0.25, 0.0, roi);
        }
        const cv::Scalar colors[] = {{0, 0, 0}, {255, 255, 255}, {0, 255, 255}};
        const char* labels[] = {"Odom (Fast-LIO)", "GPS", "PF"};
        for (int i = 0; i < 3; ++i) {
            const cv::Point center(40, 50 + 55 * i);
            cv::circle(display, center, 12, cv::Scalar(60, 60, 60), -1);
            cv::circle(display, center, 9, colors[i], -1);
            cv::putText(display, labels[i], cv::Point(70, center.y + 9),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(20, 20, 20), 2);
        }
        cv::imshow("Satellite Map", display);
        cv::waitKey(1);
        rate.sleep();
    }
    return 0;
}
