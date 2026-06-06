#include "CInterface.h"
#include <cv_bridge/cv_bridge.h>
#include <CoordConverter.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "cnpy.h"
#include "Tool.h"
#include "ParticleFilter_forloop.h"
#include "LocalizationFusion.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// =============================================
// ForLoop PF 整体耗时统计类 (纯 CPU, 无需 CUDA sync)
// =============================================
class PFTimeStat {
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    void Stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        total_ms_ += ms;
        ++frame_count_;
    }
    void Print() const {
        std::cout << "\n========== ForLoop PF Time Statistics =========="  << std::endl;
        std::cout << "  Total frames : " << frame_count_            << std::endl;
        std::cout << "  Total time   : " << std::fixed << std::setprecision(2) << total_ms_ << " ms" << std::endl;
        if (frame_count_ > 0) {
            std::cout << "  Avg per frame: " << total_ms_ / frame_count_ << " ms" << std::endl;
        }
        std::cout << "================================================\n" << std::endl;
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

    ros::NodeHandle nh;
    CInterface interface(nh);
    InputData input; 
    Tool tool_func;

    SatelliteData sat_data;

    double3D Base_point;
    WORLD_POINT Odom_World;
    WORLD_POINT GNSS_World;
    WORLD_POINT Loc_World;
    WORLD_POINT GTSAM_World;
    WORLD_POINT Test_Point;
    bool Init_localization = false;

    const double kTemporalSmoothAlpha = 0.8;
    bool has_prev_loc_world = false;
    WORLD_POINT prev_loc_world;

    WORLD_POINT last_Odom_World;
    double last_map_heading = 0.0;

    double last_odom_x = 0.0, last_odom_y = 0.0, last_odom_yaw = 0.0;

    // *** 使用 ForLoop 版本粒子滤波器 ***
    ParticleFilterForloop pf(256);
    PFTimeStat pf_timer;
    LocalizationFusion fusion_optimizer;
    cv::Mat map_vis;

    ros::Rate rate(10);

    // =========================
    // 1) 加载卫星地图数据 (NPZ)
    // =========================
    std::string path ="/home/hsw/catkin_ws/doc/kitti_raw_1003_0027.npz";
    bool npz_loaded = tool_func.LoadSatelliteNpz(path, sat_data);
    CoordConverter converter(sat_data);

    // =========================
    // 2) 日志输出文件
    // =========================
    std::ofstream csv_file("/home/hsw/catkin_ws/doc/log_1003_0027_forloop_512.csv", std::ios::out | std::ios::trunc);
    if (!csv_file.is_open())
    {
        std::cerr << "Failed to open CSV for logging." << std::endl;
        return -1;
    }
    csv_file << "timestamp,"
             << "odom_x,odom_y,odom_theta,"
             << "pf_x,pf_y,pf_theta,"
             << "gnss_x,gnss_y,gnss_theta" 
             << std::endl;

    // =========================
    // 3) 可视化对象
    // =========================
    cv::Mat semantic_map;
    cv::Mat satellite_map = sat_data.satellite_map.clone();

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
        
        if (!sat_data.semantic_map.empty()) 
        {
            std::vector<cv::Mat> splited;
            cv::split(sat_data.semantic_map, splited);
            std::vector<cv::Mat> bgr_ch = {splited[0], splited[1], splited[2]};
            cv::Mat bgr;
            cv::merge(bgr_ch, bgr);
            semantic_map = bgr.clone();
            cv::namedWindow("Check Read - BGR", cv::WINDOW_NORMAL); 
            cv::resizeWindow("Check Read - BGR", 600, 600); 
            cv::imshow("Check Read - BGR", bgr);
        }

        if (!sat_data.satellite_map.empty())
        {
            std::cout << "[Info] 卫星地图可视化..." << std::endl;
            cv::namedWindow("Check Read - Satellite", cv::WINDOW_NORMAL);
            cv::resizeWindow("Check Read - Satellite", 600, 600);
            cv::imshow("Check Read - Satellite", sat_data.satellite_map);
        }
        
        if (!sat_data.tdf_map.empty()) 
        {
            std::cout << "[Info] 可视化 TDF 地图..." << std::endl;
            int num_channels = sat_data.tdf_map.channels();
            std::vector<cv::Mat> tdf_channels;
            cv::split(sat_data.tdf_map, tdf_channels);
            
            for (size_t i = 0; i < tdf_channels.size(); ++i) 
            {
                cv::Mat chan = tdf_channels[i];
                cv::Mat norm_chan, vis_gray, heatmap;
                cv::normalize(chan, norm_chan, 0, 255, cv::NORM_MINMAX, CV_8U);
                cv::subtract(cv::Scalar(255), norm_chan, vis_gray);
                cv::applyColorMap(vis_gray, heatmap, cv::COLORMAP_JET);
                std::string win_name = "Check Read - TDF Class " + std::to_string(i);
                cv::namedWindow(win_name, cv::WINDOW_NORMAL);
                cv::resizeWindow(win_name, 600, 600);
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

        if(input.Odom_refreshflag == true && input.Inspvax_refreshflag == true && Init_localization == false)
        {
            // (5.1) 初始化 GPS
            GNSS_World.BLH.Lon = input.Inspvax->longitude;
            GNSS_World.BLH.Lat = input.Inspvax->latitude;
            GNSS_World.BLH.Height = input.Inspvax->altitude;
            GNSS_World.gauss = converter.wgs84_to_gauss(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);
            GNSS_World.heading = input.Inspvax->azimuth;
            GNSS_World.pixel = converter.wgs84_to_pixel(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);

            Odom_World = GNSS_World;
            last_Odom_World = GNSS_World; 
            Loc_World = GNSS_World;
            prev_loc_world = Loc_World;
            has_prev_loc_world = true;
            Test_Point = GNSS_World;

            const auto& q0 = input.Odom->pose.pose.orientation;
            last_odom_x = input.Odom->pose.pose.position.x;
            last_odom_y = input.Odom->pose.pose.position.y;
            last_odom_yaw = std::atan2(
                2.0 * (q0.w * q0.z + q0.x * q0.y),
                1.0 - 2.0 * (q0.y * q0.y + q0.z * q0.z));
            
            Base_point = tool_func.GetBase(input.Odom, GNSS_World);
            pf.Init(sat_data, GNSS_World.gauss.x, GNSS_World.gauss.y, GNSS_World.heading, 4.0, 4.0, 8.0);
            Init_localization = true;
            
            std::cout << "[ForLoopPF] 定位初始化完成" << std::endl;
        }
        else if(input.Odom_refreshflag == true && input.Inspvax_refreshflag == true && Init_localization == true)
        {
            // (5.2.1) 更新 GPS
            GNSS_World.BLH.Lon = input.Inspvax->longitude;
            GNSS_World.BLH.Lat = input.Inspvax->latitude;
            GNSS_World.BLH.Height = input.Inspvax->altitude;
            GNSS_World.gauss = converter.wgs84_to_gauss(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);
            GNSS_World.heading = input.Inspvax->azimuth;
            GNSS_World.pixel = converter.wgs84_to_pixel(GNSS_World.BLH.Lon, GNSS_World.BLH.Lat);

            // (5.2.2) 更新里程计
            Odom_World = tool_func.LocalToGlobal(input.Odom, Base_point);
            Odom_World.BLH = converter.gauss_to_wgs84(Odom_World.gauss.x, Odom_World.gauss.y);
            Odom_World.pixel = converter.wgs84_to_pixel(Odom_World.BLH.Lon, Odom_World.BLH.Lat);
            
            // (5.2.3) 从原始里程计直接提取车体局部增量
            {
                const auto& q = input.Odom->pose.pose.orientation;
                double odom_x   = input.Odom->pose.pose.position.x;
                double odom_y   = input.Odom->pose.pose.position.y;
                double odom_yaw = std::atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z));

                double dx_odom = odom_x - last_odom_x;
                double dy_odom = odom_y - last_odom_y;

                double c = std::cos(last_odom_yaw);
                double s = std::sin(last_odom_yaw);

                double local_dx =  c * dx_odom + s * dy_odom;
                double local_dy = -s * dx_odom + c * dy_odom;

                double dTheta = odom_yaw - last_odom_yaw;

                while (dTheta > M_PI) dTheta -= 2.0 * M_PI;
                while (dTheta < -M_PI) dTheta += 2.0 * M_PI;
                dTheta = dTheta * RAD_TO_DEG;

                pf_timer.Start();
                pf.Predict(local_dx, local_dy, dTheta);

                // 将局部增量旋转到全局 Gauss 坐标系并累加到 Test_Point
                {
                    double heading_rad = Test_Point.heading * DEG_TO_RAD;
                    double dtheta_rad_mid = dTheta * DEG_TO_RAD * 0.5;
                    double cw = std::cos(heading_rad + dtheta_rad_mid);
                    double sw = std::sin(heading_rad + dtheta_rad_mid);
                    Test_Point.gauss.x += cw * local_dx - sw * local_dy;
                    Test_Point.gauss.y += sw * local_dx + cw * local_dy;
                    Test_Point.heading += dTheta;
                    Test_Point.heading -= 360.0 * std::floor(Test_Point.heading / 360.0);
                    Test_Point.BLH = converter.gauss_to_wgs84(Test_Point.gauss.x, Test_Point.gauss.y);
                    Test_Point.pixel = converter.wgs84_to_pixel(Test_Point.BLH.Lon, Test_Point.BLH.Lat);
                }

                last_odom_x   = odom_x;
                last_odom_y   = odom_y;
                last_odom_yaw = odom_yaw;
            }

            last_Odom_World = Odom_World;

            // (5.2.5) 观测更新
            if (input.BevProbs != nullptr) 
            {
                cv_bridge::CvImagePtr cv_ptr;
                try 
                {
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
                    if (cv_ptr->image.rows != 400 || cv_ptr->image.cols != 400) 
                    {
                        cv::resize(cv_ptr->image, bev_data_resized, cv::Size(400, 400), 0, 0, cv::INTER_LINEAR);
                        std::cout << "[Info] Resize Probs to 400x400." << std::endl;
                    } 
                    else
                    {
                        bev_data_resized = cv_ptr->image;
                    }
                    
                    std::cout << "====================*************************===================" <<std::endl;
                    pf.Update(bev_data_resized, 20);

                    if (pf.GetNeff() < 60.0) 
                    {
                        pf.Resample();
                    }
                }
            }
            
            // (5.2.6) 状态估计
            std::vector<double> pf_pose = pf.EstimatePose();

            Loc_World.gauss.x = pf_pose[0];
            Loc_World.gauss.y = pf_pose[1];
            Loc_World.heading = pf_pose[2];
            Loc_World.BLH = converter.gauss_to_wgs84(Loc_World.gauss.x, Loc_World.gauss.y);
            Loc_World.pixel = converter.wgs84_to_pixel(Loc_World.BLH.Lon, Loc_World.BLH.Lat);

            // (5.2.6.1) 前后两帧平滑
            if (has_prev_loc_world)
            {
                WORLD_POINT smoothed_loc = Loc_World;
                smoothed_loc.gauss.x = kTemporalSmoothAlpha * Loc_World.gauss.x +
                                      (1.0 - kTemporalSmoothAlpha) * prev_loc_world.gauss.x;
                smoothed_loc.gauss.y = kTemporalSmoothAlpha * Loc_World.gauss.y +
                                      (1.0 - kTemporalSmoothAlpha) * prev_loc_world.gauss.y;

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
            pf_timer.Stop();

            // ================== 因子图融合优化 ==================
            auto raw_particles = pf.GetRawParticles();
            
            std::vector<LocalizationFusion::FusionParticle> fusion_particles;
            fusion_particles.reserve(raw_particles.size());
            
            for(const auto& p : raw_particles) 
            {
                fusion_particles.push_back({
                    p.x, 
                    p.y, 
                    p.theta * DEG_TO_RAD,
                    p.weight
                });
            }

            gtsam::Pose2 current_odom_pose(
                Odom_World.gauss.x, 
                Odom_World.gauss.y, 
                Odom_World.heading * DEG_TO_RAD
            );

            gtsam_result = fusion_optimizer.Process(fusion_particles, current_odom_pose);
            has_gtsam_result = true;
            
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

            // --- [Debug] 粒子滤波定位误差 ---
            double err_x_result = Loc_World.gauss.x - GNSS_World.gauss.x;
            double err_y_result = Loc_World.gauss.y - GNSS_World.gauss.y;
            if(Loc_World.heading<0) Loc_World.heading += 360.0;
            if(Loc_World.heading>360) Loc_World.heading -= 360.0;
            double err_dist_result = std::sqrt(err_x_result*err_x_result + err_y_result*err_y_result);
            double err_theta_result = Loc_World.heading - GNSS_World.heading;
            while(err_theta_result > 180) err_theta_result -= 360;
            while(err_theta_result < -180) err_theta_result += 360;
            std::cout << "[Running_result] PosErr: " << std::fixed << std::setprecision(3) << err_dist_result 
                      << " m | AngErr: " << std::setprecision(2) << err_theta_result << " deg"
                      << " | Est: (" << Loc_World.gauss.x << ", " << Loc_World.gauss.y << ", " << Loc_World.heading << ")" 
                      << std::endl;

            // --- [Debug] 里程计定位误差 ---
            double err_x_odometry = Odom_World.gauss.x - GNSS_World.gauss.x;
            double err_y_odometry = Odom_World.gauss.y - GNSS_World.gauss.y;
            if(Odom_World.heading<0) Odom_World.heading += 360.0;
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

            // 写入 CSV
            if (csv_file.is_open())
            {
                double stamp = ros::Time::now().toSec();

                csv_file << std::fixed << std::setprecision(6)
                         << stamp << ","
                         << Odom_World.gauss.x << "," << Odom_World.gauss.y << "," << Odom_World.heading << ","
                         << Loc_World.gauss.x << "," << Loc_World.gauss.y << "," << Loc_World.heading << ","
                         << GNSS_World.gauss.x << "," << GNSS_World.gauss.y << "," << GNSS_World.heading
                         << std::endl;
            }
            if (!semantic_map.empty())
            {
                double rotate_deg = 90.0 - Loc_World.heading;
                cv::Mat rot_mat = cv::getRotationMatrix2D(Loc_World.pixel, rotate_deg, 1.0);
                rot_mat.at<double>(0, 2) += 200.0 - Loc_World.pixel.x;
                rot_mat.at<double>(1, 2) += 200.0 - Loc_World.pixel.y;

                cv::Mat local_roi;
                cv::warpAffine(
                    semantic_map, local_roi, rot_mat,
                    cv::Size(400, 400), cv::INTER_LINEAR,
                    cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

                cv::Point center(200, 200);
                int w = 14, h = 28, r = 4;
                std::vector<cv::Point> car_poly = {
                    {center.x - w/2 + r, center.y - h/2}, {center.x + w/2 - r, center.y - h/2},
                    {center.x + w/2, center.y - h/2 + r}, {center.x + w/2, center.y + h/2 - r},
                    {center.x + w/2 - r, center.y + h/2}, {center.x - w/2 + r, center.y + h/2},
                    {center.x - w/2, center.y + h/2 - r}, {center.x - w/2, center.y - h/2 + r}
                };
                cv::fillConvexPoly(local_roi, car_poly, cv::Scalar(255, 255, 255), cv::LINE_AA);
                cv::polylines(local_roi, car_poly, true, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
                int glass_y = center.y - h/4;
                cv::line(local_roi, 
                    cv::Point(center.x - w/2 + 2, glass_y), 
                    cv::Point(center.x + w/2 - 2, glass_y), 
                    cv::Scalar(100, 100, 100), 2, cv::LINE_AA);
                cv::circle(local_roi, center, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

                cv::imshow("PF Local View", local_roi);
                cv::waitKey(1);
            }
        }
        
        cv::circle(satellite_map, Odom_World.pixel, 5, cv::Scalar(0, 0, 0), -1);
        cv::circle(satellite_map, GNSS_World.pixel, 5, cv::Scalar(255, 255, 255), -1);
        cv::circle(satellite_map, Loc_World.pixel, 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(satellite_map, Test_Point.pixel, 5, cv::Scalar(255, 0, 0), -1);

        // 图例
        int legend_x = 30, legend_y = 60, line_gap = 70, circle_radius = 14;
        double font_scale = 1.5;
        int font_thickness = 4, border_thick = 4;
        cv::Scalar text_color(20, 20, 20);

        cv::Rect legend_rect(legend_x - 15, legend_y - 25, 700, 370);
        legend_rect &= cv::Rect(0, 0, satellite_map.cols, satellite_map.rows);
        if (legend_rect.area() > 0) 
        {
            cv::Mat roi = satellite_map(legend_rect);
            cv::Mat color_box(roi.size(), CV_8UC3, cv::Scalar(255, 255, 255));
            double alpha = 0.65;
            cv::addWeighted(color_box, alpha, roi, 1.0 - alpha, 0.0, roi);
        }

        auto draw_legend_item = [&](cv::Point center, cv::Scalar fill_color, cv::Scalar border_color, const std::string& text) 
        {
            cv::circle(satellite_map, center, circle_radius, fill_color, -1);
            cv::circle(satellite_map, center, circle_radius, border_color, border_thick);
            int text_offset_y = circle_radius / 2 + 2; 
            cv::putText(satellite_map, text, cv::Point(center.x + 35, center.y + text_offset_y),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness);
        };

        draw_legend_item(cv::Point(legend_x, legend_y), 
                        cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), "Odom (Fast-LIO)");
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap), 
                        cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0), "GNSS (Ground Truth)");
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 2), 
                        cv::Scalar(0, 255, 255), cv::Scalar(0, 0, 0), "PF Est (Yellow)");
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 3), 
                        cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 0), "GTSAM Fused (Red)"); 
        draw_legend_item(cv::Point(legend_x, legend_y + line_gap * 4), 
                        cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 0), "DeltaAccum (Blue)"); 

        cv::namedWindow("Satellite Map", cv::WINDOW_NORMAL);
        cv::resizeWindow("Satellite Map", 800, 800);
        cv::imshow("Satellite Map", satellite_map);
        cv::waitKey(1);
        rate.sleep();
    }
    // 程序退出前打印耗时
    pf_timer.Print();
    pf.PrintCumulativeTiming();
    return 0;
}
