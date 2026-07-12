#include "CInterface.h"
#include <cv_bridge/cv_bridge.h>
#include <CoordConverter.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <cmath>
#include <chrono>
#include "cnpy.h"
#include "Tool.h"
#include "ParticleFilter_our.h"
#include "LocalizationFusion.h"
#include "ParticleWeightVisualization.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// =============================================
// 粒子滤波整体耗时统计类 (含 CUDA 同步)
// =============================================
class PFTimeStat {
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    void Stop() {
        // LibTorch/CUDA 异步执行，必须同步后计时才准确
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        total_ms_ += ms;
        ++frame_count_;
    }
    void Print() const {
        std::cout << "\n========== PF Time Statistics =========="  << std::endl;
        std::cout << "  Total frames : " << frame_count_            << std::endl;
        std::cout << "  Total time   : " << std::fixed << std::setprecision(2) << total_ms_ << " ms" << std::endl;
        if (frame_count_ > 0) {
            std::cout << "  Avg per frame: " << total_ms_ / frame_count_ << " ms" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
    double total_ms_  = 0.0;
    int    frame_count_ = 0;
};

int main(int argc, char **argv)
{
    // =========================
    // 0) ROS / 基础对象初始化
    // =========================
    ros::init(argc, argv, "localization_node");

    //类对象创建
    ros::NodeHandle nh;
    CInterface interface(nh);
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
    WORLD_POINT Test_Point;
    bool Init_localization = false;

    // 两帧时序平滑：当前帧与上一帧均衡融合
    const double kTemporalSmoothAlpha = 1.0; // 当前帧权重
    bool has_prev_loc_world = false;
    WORLD_POINT prev_loc_world;
    int pf_weight_vis_frame = 0;

    // Gauss 坐标系下的辅助变量（用于 Delta 计算）
    WORLD_POINT last_Odom_World;
    double last_map_heading = 0.0; // 预留变量：Robot heading in Gauss Map Frame

    // 原始里程计状态缓存（直接提取局部增量，避免 global→local 绕行）
    double last_odom_x = 0.0, last_odom_y = 0.0, last_odom_yaw = 0.0;

    // 粒子滤波器
    ParticleFilter pf(256); // 粒子数量 (可调)
    // 粒子滤波耗时统计
    PFTimeStat pf_timer;
    // 因子图优化器 (3DOF PF + Odom Fusion)
    LocalizationFusion fusion_optimizer;
    // 可视化图像
    cv::Mat map_vis;

    ros::Rate rate(10);


    // =========================
    // 1) 加载卫星地图数据 (NPZ)
    // =========================
    //std::string path ="/home/hsw/catkin_ws/doc/kitti_raw_0930_0027.npz"; // 替换你的实际路径
    //std::string path ="/home/hsw/catkin_ws/doc/kitti_raw_1003_0034.npz";
    std::string path ="/home/hsw/catkin_ws/doc/kitti_raw_1003_0027.npz";
    //std::string path ="/home/hsw/catkin_ws/doc/kitti_raw_0930_0033.npz";
    bool npz_loaded = tool_func.LoadSatelliteNpz(path, sat_data);
    CoordConverter converter(sat_data);


    // =========================
    // 2) 日志输出文件
    // =========================
    // 记录 Odom / GNSS / PF 结果（可扩展 GTSAM）到 CSV
    const std::string csv_dir = std::string(SEM_LOC_SOURCE_DIR) + "/output";
    particle_weight_vis::MakeDirRecursive(csv_dir);
    std::ofstream csv_file(csv_dir + "/pose_compare_ourmethod.csv", std::ios::out | std::ios::trunc);
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
    cv::Mat semantic_map;
    cv::Mat satellite_map = sat_data.satellite_map.clone();

    // 用于存储最新因子图融合结果的变量 (循环外定义)
    LocalizationFusion::FusionResult gtsam_result = {0.0, 0.0, 0.0, {}, true};
    bool has_gtsam_result = false;



    // =========================
    // 4) NPZ 读取后检查与可视化
    // =========================
    if (npz_loaded)
    {
        std::cout << "----- 地图加载成功! ------" << std::endl;
        printf("地图的gauss分辨率 : %.10f\n", sat_data.resolution);
        printf("地理经纬度(左上--右下): [%.10f, %.10f, %.10f, %.10f]\n", sat_data.geo_bounds[0], sat_data.geo_bounds[1],
                                                                      sat_data.geo_bounds[2], sat_data.geo_bounds[3]);
        std::cout << "==========================" << std::endl;

        //--- 语义地图可视化 ---
        if (!sat_data.semantic_map.empty())
        {
            std::vector<cv::Mat> splited;
            cv::split(sat_data.semantic_map, splited);

            // 显示前三个通道 (Color Img)
            std::vector<cv::Mat> bgr_ch = {splited[0], splited[1], splited[2]};
            cv::Mat bgr;
            cv::merge(bgr_ch, bgr);
            semantic_map = bgr.clone(); // 保存用于后续绘制
            cv::namedWindow("Check Read - BGR", cv::WINDOW_NORMAL);
            cv::resizeWindow("Check Read - BGR", 600, 600);
            cv::imshow("Check Read - BGR", bgr);
        }

        //---  卫星 RGB 可视化 ---
        if (!sat_data.satellite_map.empty())
        {
            std::cout << "[Info] 卫星地图可视化..." << std::endl;
            cv::namedWindow("Check Read - Satellite", cv::WINDOW_NORMAL);
            cv::resizeWindow("Check Read - Satellite", 600, 600);
            cv::imshow("Check Read - Satellite", sat_data.satellite_map);
        }

        //---  TDF Map 可视化 ---
        if (!sat_data.tdf_map.empty())
        {
            std::cout << "[Info] 可视化 TDF 地图..." << std::endl;
            int num_channels = sat_data.tdf_map.channels();

            // 如果是多通道的，分离每个通道分别显示
            std::vector<cv::Mat> tdf_channels;
            cv::split(sat_data.tdf_map, tdf_channels);

            for (size_t i = 0; i < tdf_channels.size(); ++i)
            {
                cv::Mat chan = tdf_channels[i];
                cv::Mat norm_chan, vis_gray, heatmap;

                // 1. 归一化到 0-255
                cv::normalize(chan, norm_chan, 0, 255, cv::NORM_MINMAX, CV_8U);

                // 2. 反转亮度 (0/Near=White, Max/Far=Black) - 类似你的Python逻辑
                // 或者按照你的热力图逻辑: 255 - norm
                cv::subtract(cv::Scalar(255), norm_chan, vis_gray);

                // 3. 应用热力图
                cv::applyColorMap(vis_gray, heatmap, cv::COLORMAP_JET);
                std::string win_name = "Check Read - TDF Class " + std::to_string(i);
                cv::namedWindow(win_name, cv::WINDOW_NORMAL);   // 自适应窗口
                cv::resizeWindow(win_name, 600, 600);           // 设一个合理的初始大小
                cv::imshow(win_name, heatmap);
            }
        }

        cv::waitKey(1);
    }
    else
    {
        std::cerr << "Failed to read data." << std::endl;
    }

    // =========================
    // 5) 主循环
    // =========================
    while (ros::ok())
    {
        ros::spinOnce();
        interface.ConvertToLocalData(&input);

        if(input.Odom_refreshflag == true && input.Inspvax_refreshflag == true && Init_localization == false)   // 第一帧定位初始化
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
            Test_Point = GNSS_World; // 用于后续测试坐标转换的一致性

            const auto& q0 = input.Odom->pose.pose.orientation;
            last_odom_x = input.Odom->pose.pose.position.x;
            last_odom_y = input.Odom->pose.pose.position.y;
            last_odom_yaw = std::atan2(
                2.0 * (q0.w * q0.z + q0.x * q0.y),
                1.0 - 2.0 * (q0.y * q0.y + q0.z * q0.z));   // rad
            //Test_Point.heading = last_odom_yaw ; // 弧度


            Base_point = tool_func.GetBase(input.Odom, GNSS_World);   // 得到坐标系转换点（弧度单位）
            pf.Init(sat_data, GNSS_World.gauss.x, GNSS_World.gauss.y, GNSS_World.heading, 4.0, 4.0, 8.0);   // 初始化粒子滤波器
            // 初始化状态（用于下一帧计算增量）
            Init_localization = true;

            std::cout << "定位初始化完成" << std::endl;
        }
        else if(input.Odom_refreshflag == true && input.Inspvax_refreshflag == true && Init_localization == true)
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

            // (5.2.3) 从原始里程计直接提取车体局部增量
            //
            // 里程计话题 /Odometry 给出 odom 坐标系下的累积绝对位姿:
            //   T_odom = (odom_x, odom_y, odom_yaw)
            //
            // 局部增量 = T_{t-1}^{-1} * T_t  (SE(2) 逆变换)
            //   local = R(-yaw_{t-1}) * (pos_t - pos_{t-1})
            //   dtheta = yaw_t - yaw_{t-1}
            //
            // 相比经由全局坐标 (LocalToGlobal → global diff → R^T):
            //   - 无 GetBase 刚体变换带来的累积数值误差
            //   - 物理含义更清晰: 直接是 body-frame SE(2) 增量
            {
                const auto& q = input.Odom->pose.pose.orientation;
                double odom_x   = input.Odom->pose.pose.position.x;
                double odom_y   = input.Odom->pose.pose.position.y;
                double odom_yaw = std::atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));  // rad

                double dx_odom = odom_x - last_odom_x;
                double dy_odom = odom_y - last_odom_y;

                double c = std::cos(last_odom_yaw);
                double s = std::sin(last_odom_yaw);

                double local_dx =  c * dx_odom + s * dy_odom;   // 纵向 (forward)
                double local_dy = -s * dx_odom + c * dy_odom;   // 侧向 (leftward)

                double dTheta = odom_yaw - last_odom_yaw;

                while (dTheta > M_PI) dTheta -= 2.0 * M_PI;
                while (dTheta < -M_PI) dTheta += 2.0 * M_PI;
                dTheta = dTheta * RAD_TO_DEG;

                pf_timer.Start();  // >>> PF 计时开始
                pf.Predict(local_dx, local_dy, dTheta);   //输入角度增量（Degree）

                // 将局部增量旋转到全局 Gauss 坐标系并累加到 Test_Point
                {
                    double heading_rad = Test_Point.heading * DEG_TO_RAD;
                    double dtheta_rad_mid = dTheta * DEG_TO_RAD * 0.5;
                    double cw = std::cos(heading_rad + dtheta_rad_mid);
                    double sw = std::sin(heading_rad + dtheta_rad_mid);
                    Test_Point.gauss.x += cw * local_dx - sw * local_dy;
                    Test_Point.gauss.y += sw * local_dx + cw * local_dy;
                    Test_Point.heading += dTheta;
                    // 归一化到 [0, 360)
                    Test_Point.heading -= 360.0 * std::floor(Test_Point.heading / 360.0);
                    Test_Point.BLH = converter.gauss_to_wgs84(Test_Point.gauss.x, Test_Point.gauss.y);
                    Test_Point.pixel = converter.wgs84_to_pixel(Test_Point.BLH.Lon, Test_Point.BLH.Lat);
                }

                last_odom_x   = odom_x;
                last_odom_y   = odom_y;
                last_odom_yaw = odom_yaw;
            }


            // Odom_World 供可视化/GTSAM/日志使用 (5.2.2 中已计算)
            last_Odom_World = Odom_World;

            bool should_resample = false;


            // (5.2.5) 观测更新（本文方法）
            // 使用 4 通道概率图（Road/Plant/Building/Entropy）进行 PF 更新：
            // 相比硬标签，减少量化误差并利用语义不确定性信息。
            if (input.BevProbs != nullptr)
            {
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    // 接收 4 通道 32F 数据 (Road, Plant, Building, Entropy)
                    cv_ptr = cv_bridge::toCvCopy(*input.BevProbs, sensor_msgs::image_encodings::TYPE_32FC4);
                }
                catch (cv_bridge::Exception& e)
                {
                   ROS_ERROR("cv_bridge exception: %s", e.what());
                }

                if (cv_ptr && !cv_ptr->image.empty())
                {
                    int cn = cv_ptr->image.channels();
                    if (cn != 4)
                    {
                         ROS_WARN_THROTTLE(2.0, "[Data Check] Probs Channels=%d (Expect 4)", cn);
                    }

                    cv::Mat bev_data_resized;
                    // 确保是 400x400
                    if (cv_ptr->image.rows != 400 || cv_ptr->image.cols != 400)
                    {
                        // 浮点数概率建议使用双线性插值
                        cv::resize(cv_ptr->image, bev_data_resized, cv::Size(400, 400), 0, 0, cv::INTER_LINEAR);
                        std::cout << "[Info] Resize Probs to 400x400." << std::endl;
                    }
                    else
                    {
                        bev_data_resized = cv_ptr->image;
                    }

                    // 使用 4 通道的概率图更新粒子
                    std::cout << "====================*************************===================" <<std::endl;
                    pf.Update(bev_data_resized,20); // 观测噪声参数 (可调)

                    // 标准 SIR: 根据有效粒子数自适应重采样；估计/可视化后再执行。
                    should_resample = (pf.GetNeff() < 200.0);
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
                double curr_heading_rad = Loc_World.heading * DEG_TO_RAD;
                double prev_heading_rad = prev_loc_world.heading * DEG_TO_RAD;
                double mix_sin = kTemporalSmoothAlpha * std::sin(curr_heading_rad) +
                                (1.0 - kTemporalSmoothAlpha) * std::sin(prev_heading_rad);
                double mix_cos = kTemporalSmoothAlpha * std::cos(curr_heading_rad) +
                                (1.0 - kTemporalSmoothAlpha) * std::cos(prev_heading_rad);
                smoothed_loc.heading = std::atan2(mix_sin, mix_cos) * RAD_TO_DEG;
                if (smoothed_loc.heading < 0.0)
                {
                    smoothed_loc.heading += 360.0;
                }

                smoothed_loc.BLH = converter.gauss_to_wgs84(smoothed_loc.gauss.x, smoothed_loc.gauss.y);
                smoothed_loc.pixel = converter.wgs84_to_pixel(smoothed_loc.BLH.Lon, smoothed_loc.BLH.Lat);
                Loc_World = smoothed_loc;
            }
            prev_loc_world = Loc_World;
            has_prev_loc_world = true;
            pf_timer.Stop();   // >>> PF 计时结束 (含 CUDA sync)

            // ================== [新增] 因子图融合优化 ==================
            // 1) 获取粒子原始数据（Degrees）
            double pf_neff_before_resample = pf.GetNeff();
            auto raw_particles = pf.GetRawParticles();
            particle_weight_vis::SaveAndShowParticleWeightView(
                sat_data.satellite_map,
                converter,
                raw_particles,
                Loc_World.pixel,
                pf_weight_vis_frame++,
                pf_neff_before_resample);

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
            has_gtsam_result = true;

            // 5) 输出因子图融合结果
            GTSAM_World.heading = gtsam_result.theta * RAD_TO_DEG;
            while(GTSAM_World.heading > 180) GTSAM_World.heading -= 360;
            while(GTSAM_World.heading < -180) GTSAM_World.heading += 360;
            GTSAM_World.gauss.x = gtsam_result.x;
            GTSAM_World.gauss.y = gtsam_result.y;
            GTSAM_World.BLH = converter.gauss_to_wgs84(GTSAM_World.gauss.x, GTSAM_World.gauss.y);
            GTSAM_World.pixel = converter.wgs84_to_pixel(GTSAM_World.BLH.Lon, GTSAM_World.BLH.Lat);


            // 计算误差
            double err_x = GTSAM_World.gauss.x - GNSS_World.gauss.x;
            double err_y = GTSAM_World.gauss.y - GNSS_World.gauss.y;
            double err_dist = std::sqrt(err_x*err_x + err_y*err_y);

            double err_theta = GTSAM_World.heading - GNSS_World.heading;
            while(err_theta > 180) err_theta -= 360;
            while(err_theta < -180) err_theta += 360;

            std::string status_str = gtsam_result.is_pure_odom ? "[Fallback LIO]" : "[Fusion (PF)]";
            std::cout << "[GTSAM] " << status_str
            << " PosErr: " << std::fixed << std::setprecision(3) << err_dist
            << " m | AngErr: " << std::setprecision(2) << err_theta << " deg"
            << " | Est: (" << GTSAM_World.gauss.x << ", " << GTSAM_World.gauss.y << ", " << GTSAM_World.heading << ")"
            << std::endl;


            // --- [Debug] 计算粒子滤波定位误差 ---
            double err_x_result = Loc_World.gauss.x - GNSS_World.gauss.x;
            double err_y_result = Loc_World.gauss.y - GNSS_World.gauss.y;
            if(Loc_World.heading<0) Loc_World.heading += 360.0; // 归一化到0-360度
            if(Loc_World.heading>360) Loc_World.heading -= 360.0;
            double err_dist_result = std::sqrt(err_x_result*err_x_result + err_y_result*err_y_result);
            double err_theta_result = Loc_World.heading - GNSS_World.heading;
            while(err_theta_result > 180) err_theta_result -= 360;
            while(err_theta_result < -180) err_theta_result += 360;
            std::cout << "[Running_result] PosErr: " << std::fixed << std::setprecision(3) << err_dist_result
                      << " m | AngErr: " << std::setprecision(2) << err_theta_result << " deg"
                      << " | Est: (" << Loc_World.gauss.x << ", " << Loc_World.gauss.y << ", " << Loc_World.heading << ")"
                      << std::endl;

            // --- [Debug] 计算里程计定位误差 ---
            double err_x_odometry = Odom_World.gauss.x - GNSS_World.gauss.x;
            double err_y_odometry = Odom_World.gauss.y - GNSS_World.gauss.y;
            if(Odom_World.heading<0) Odom_World.heading += 360.0; // 归一化到0-360度
            if(Odom_World.heading>360) Odom_World.heading -= 360.0;
            double err_dist_odometry = std::sqrt(err_x_odometry*err_x_odometry + err_y_odometry*err_y_odometry);
            double err_theta_odometry = Odom_World.heading - GNSS_World.heading;
            while(err_theta_odometry > 180) err_theta_odometry -= 360;
            while(err_theta_odometry < -180) err_theta_odometry += 360;
            std::cout << "[Running_Odometry] PosErr: " << std::fixed << std::setprecision(3) << err_dist_odometry
                      << " m | AngErr: " << std::setprecision(2) << err_theta_odometry << " deg"
                      << " | Est: (" << Odom_World.gauss.x << ", " << Odom_World.gauss.y << ", " << Odom_World.heading << ")"
                      << std::endl;
            std::cout << "====================*************************===================" <<std::endl;

            // (8) 可视化准备
            // 绘制 PF 估计结果


            // 写入 CSV：Odom / 粒子滤波定位 / GTSAM / GNSS 的 Gauss 坐标与朝向
            if (csv_file.is_open())
            {
                double stamp = ros::Time::now().toSec();
                double gtsam_val_x = has_gtsam_result ? GTSAM_World.gauss.x : 0.0;
                double gtsam_val_y = has_gtsam_result ? GTSAM_World.gauss.y : 0.0;
                double gtsam_val_theta = has_gtsam_result ? GTSAM_World.heading : 0.0;

                csv_file << std::fixed << std::setprecision(6)
                         << stamp << ","
                         << Odom_World.gauss.x << "," << Odom_World.gauss.y << "," << Odom_World.heading << ","
                         << Loc_World.gauss.x << "," << Loc_World.gauss.y << "," << Loc_World.heading << ","
                        // << GTSAM_World.gauss.x << "," << GTSAM_World.gauss.y << "," << GTSAM_World.heading << ","
                         << GNSS_World.gauss.x << "," << GNSS_World.gauss.y << "," << GNSS_World.heading
                         << std::endl;
            }
            if (!semantic_map.empty())
            {
                // =====================================================
                // 1. 旋转与平移矩阵 (World -> Ego Center)
                // =====================================================
                // heading: 0 = East, CCW positive. 转为车头朝上需补偿 90 度
                double rotate_deg = 90.0 - Loc_World.heading;

                cv::Mat rot_mat = cv::getRotationMatrix2D(Loc_World.pixel, rotate_deg, 1.0);

                // 将车辆在全局图的位置平移到 400x400 画布中心 (200, 200)
                rot_mat.at<double>(0, 2) += 200.0 - Loc_World.pixel.x;
                rot_mat.at<double>(1, 2) += 200.0 - Loc_World.pixel.y;

                // =====================================================
                // 2. 仿射变换生成局部图
                // =====================================================
                cv::Mat local_roi;
                cv::warpAffine(
                    semantic_map,
                    local_roi,
                    rot_mat,
                    cv::Size(400, 400),
                    cv::INTER_LINEAR,
                    cv::BORDER_CONSTANT,
                    cv::Scalar(0, 0, 0)
                );

                // =====================================================
                // 3. 绘制白色车模 (Ego View)
                // =====================================================
                cv::Point center(200, 200);

                // 设置精致尺寸 (宽 14, 长 28)
                int w = 14;
                int h = 28;
                int r = 4; // 圆角弧度

                // 定义 8 个顶点模拟圆角矩形
                std::vector<cv::Point> car_poly = {
                    {center.x - w/2 + r, center.y - h/2}, {center.x + w/2 - r, center.y - h/2},
                    {center.x + w/2, center.y - h/2 + r}, {center.x + w/2, center.y + h/2 - r},
                    {center.x + w/2 - r, center.y + h/2}, {center.x - w/2 + r, center.y + h/2},
                    {center.x - w/2, center.y + h/2 - r}, {center.x - w/2, center.y - h/2 + r}
                };

                // --- 绘制车身填充 (纯白) ---
                cv::fillConvexPoly(local_roi, car_poly, cv::Scalar(255, 255, 255), cv::LINE_AA);

                // --- 绘制深灰色边框 (让白车在浅色背景下也清晰) ---
                cv::polylines(local_roi, car_poly, true, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

                // --- 绘制前挡风玻璃 (区分车头) ---
                // 在车头 1/4 处画一条细横线
                int glass_y = center.y - h/4;
                cv::line(
                    local_roi,
                    cv::Point(center.x - w/2 + 2, glass_y),
                    cv::Point(center.x + w/2 - 2, glass_y),
                    cv::Scalar(100, 100, 100),
                    2,
                    cv::LINE_AA
                );

                // --- 绘制中心点 (装饰用红色小点) ---
                cv::circle(local_roi, center, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

                // =====================================================
                // 4. 显示
                // =====================================================
                cv::imshow("PF Local View", local_roi);
                cv::waitKey(1);
            }
        }

        cv::circle(satellite_map, Odom_World.pixel, 5, cv::Scalar(0, 0, 0), -1);
        cv::circle(satellite_map, GNSS_World.pixel, 5, cv::Scalar(255, 255, 255), -1);
        cv::circle(satellite_map, Loc_World.pixel, 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(satellite_map, GTSAM_World.pixel, 5, cv::Scalar(0, 0, 255), -1);
        //cv::circle(satellite_map, Test_Point.pixel, 5, cv::Scalar(255, 0, 0), -1);  // 局部增量累计轨迹 (蓝色)

        // -------------------------------------------------
        // 添加图例 (Legend) 开始
        // -------------------------------------------------
        int legend_x = 30;           // 起始 X 稍微往右挪一点
        int legend_y = 60;           // 起始 Y 稍微往下挪一点
        int line_gap = 70;           // 每行间隔 (拉大，防止拥挤)
        int circle_radius = 14;      // 【关键】圆点半径大幅增加 (之前是8)
        double font_scale =1.5;     // 【关键】字体变大 (之前是0.7)
        int font_thickness = 4;      // 字体加粗
        int border_thick = 4;        // 圆圈边框加粗
        cv::Scalar text_color(20, 20, 20); // 接近纯黑的文字

        // 2. 绘制半透明背景框 (尺寸相应调大)
        // 宽度改为 380, 高度改为 180 以容纳大号字体和图标
        cv::Rect legend_rect(legend_x - 15, legend_y - 25, 700, 370);
        legend_rect &= cv::Rect(0, 0, satellite_map.cols, satellite_map.rows);

        if (legend_rect.area() > 0)
        {
            cv::Mat roi = satellite_map(legend_rect);
            cv::Mat color_box(roi.size(), CV_8UC3, cv::Scalar(255, 255, 255)); // 纯白背景
            double alpha = 0.65; //稍微不那么透明，保证对比度
            cv::addWeighted(color_box, alpha, roi, 1.0 - alpha, 0.0, roi);
        }

        // 辅助函数：画一个带粗边框的醒目大圆点
        auto draw_legend_item = [&](cv::Point center, cv::Scalar fill_color, cv::Scalar border_color, const std::string& text)
        {
            // 1. 画实心圆
            cv::circle(satellite_map, center, circle_radius, fill_color, -1);
            // 2. 画粗边框 (高对比度)
            cv::circle(satellite_map, center, circle_radius, border_color, border_thick);
            // 3. 写字 (垂直居中调整)
            int text_offset_y = circle_radius / 2 + 2;
            cv::putText(satellite_map, text, cv::Point(center.x + 35, center.y + text_offset_y),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness);
        };

        // --- 第一项：里程计 (黑色 Odom) ---
        // 黑点，给它加一个【白色粗边框】，保证在任何背景都能看见
        draw_legend_item(cv::Point(legend_x, legend_y),
                        cv::Scalar(0, 0, 0),         // 填充黑
                        cv::Scalar(255, 255, 255),   // 边框白
                        "Odom (Fast-LIO)");

        // --- 第二项：真值 (白色 GNSS) ---
        // 白点，给它加一个【黑色粗边框】
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap),
                        cv::Scalar(255, 255, 255),   // 填充白
                        cv::Scalar(0, 0, 0),         // 边框黑
                        "GNSS (Ground Truth)");

        // --- 第三项：PF 结果 (黄色) ---
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 2),
                        cv::Scalar(0, 255, 255),
                        cv::Scalar(0, 0, 0),
                        "PF Est (Yellow)");

        // --- 第四项：GTSAM 结果 (红色) ---
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 3),
                        cv::Scalar(0, 0, 255),
                        cv::Scalar(0, 0, 0),
                        "GTSAM Fused (Red)");

        // --- 第五项：局部增量累计 (蓝色 Test_Point) ---
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 4),
                        cv::Scalar(255, 0, 0),
                        cv::Scalar(0, 0, 0),
                        "DeltaAccum (Blue)");

        // -------------------------------------------------
        // 添加图例 结束
        // -------------------------------------------------

        cv::namedWindow("Satellite Map", cv::WINDOW_NORMAL);
        cv::resizeWindow("Satellite Map", 800, 800);
        cv::imshow("Satellite Map", satellite_map);
        cv::waitKey(1);
        rate.sleep();
    }
    // 程序退出前打印粒子滤波平均耗时
    pf_timer.Print();
    return 0;
}
