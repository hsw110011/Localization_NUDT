#include "map/my_config.h"
#include <sstream>
#include <iomanip>
std::shared_ptr<my_config> params = std::make_shared<my_config>();

void my_config::readPara()
{
    std::string filename = "config/GridMapInit.ini";
    std::string params_filename;
    std::ifstream fin(filename.c_str());
    if(!fin.is_open()) {
        std::cerr<<"Fail to open params files :"<< filename <<std::endl;
        abort();
    }

    std::string line;
//    std::regex key_value_regex(R"(^(\w+)\s*(\S+.*)?)");
//    std::regex key_value_regex(R"(^(\w+)\s+(.+?)\s*(#.*)?$)");
    std::regex key_value_regex(R"(^(\w+)\s+(.+?)\s*(#.*)?$)");

    std::smatch match;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // 忽略空行和注释
        }

        if (std::regex_match(line, match, key_value_regex)) {
            std::string key = match[1];
            std::string value = match[2];

            if (key == "choose_car")
                choose_car = value;
            else if (key == "LM_Lidar_Type")
                b_LM_Lidar_Type = std::stoi(value);
            else if (key == "b_show_proj")
                b_show_proj = std::stoi(value);
            else if (key == "b_show_camera_front")
                b_show_camera_front = std::stoi(value);
            else if (key == "b_show_undistort")
                b_show_undistort = std::stoi(value);
            else if (key == "f_show_bev_color")
                f_show_bev_color = std::stoi(value);
            else if (key == "b_show_rgb_map")
                b_show_rgb_map = std::stoi(value);
            else if (key == "b_gridmap_use_rgb")
                b_gridmap_use_rgb = std::stoi(value);
            // GridMap相关显示开关
            else if (key == "b_show_height_diff")
                b_show_height_diff = std::stoi(value);
            else if (key == "b_show_slope_map")
                b_show_slope_map = std::stoi(value);
            else if (key == "b_show_terrain_roughness")
                b_show_terrain_roughness = std::stoi(value);
            else if (key == "b_show_terrain_slope")
                b_show_terrain_slope = std::stoi(value);
            else if (key == "b_show_terrain_labels")
                b_show_terrain_labels = std::stoi(value);
            else if (key == "b_show_colormap_vehicle")
                b_show_colormap_vehicle = std::stoi(value);
            else if (key == "b_show_obstacle_detection")
                b_show_obstacle_detection = std::stoi(value);
            else if (key == "b_show_lidar_points")
                b_show_lidar_points = std::stoi(value);
            else if (key == "b_enable_point_z_filter")
                b_enable_point_z_filter = std::stoi(value);
            else if (key == "point_z_max")
                point_z_max = std::stod(value);
            else if (key == "terrain_roughness_scale")
                terrain_roughness_scale = std::stod(value);
            else if (key == "b_enable_lidar_bev")
                b_enable_lidar_bev = std::stoi(value);
            else if (key == "b_lidar_bev_use_odometry")
                b_lidar_bev_use_odometry = std::stoi(value);
            else if (key == "b_lidar_bev_use_ransac_ground")
                b_lidar_bev_use_ransac_ground = std::stoi(value);
            else if (key == "b_show_lidar_bev_layers")
                b_show_lidar_bev_layers = std::stoi(value);
            else if (key == "lidar_bev_pose_source")
                lidar_bev_pose_source = value;
            else if (key == "lidar_odometry_topic")
                lidar_odometry_topic = value;
            else if (key == "lidar_bev_topic")
                lidar_bev_topic = value;
            else if (key == "lidar_bev_frame_id")
                lidar_bev_frame_id = value;
            else if (key == "lidar_bev_body_frame_id")
                lidar_bev_body_frame_id = value;
            else if (key == "lidar_bev_resolution")
                lidar_bev_resolution = std::stod(value);
            else if (key == "lidar_bev_map_size_x")
                lidar_bev_map_size_x = std::stod(value);
            else if (key == "lidar_bev_map_size_y")
                lidar_bev_map_size_y = std::stod(value);
            else if (key == "lidar_bev_point_min_z")
                lidar_bev_point_min_z = std::stod(value);
            else if (key == "lidar_bev_point_max_z")
                lidar_bev_point_max_z = std::stod(value);
            else if (key == "lidar_bev_update_rate_hz")
                lidar_bev_update_rate_hz = std::stod(value);
            else if (key == "b_lidar_bev_enable_ego_filter")
                b_lidar_bev_enable_ego_filter = std::stoi(value);
            else if (key == "lidar_bev_ego_box_min_x")
                lidar_bev_ego_box_min_x = std::stod(value);
            else if (key == "lidar_bev_ego_box_max_x")
                lidar_bev_ego_box_max_x = std::stod(value);
            else if (key == "lidar_bev_ego_box_min_y")
                lidar_bev_ego_box_min_y = std::stod(value);
            else if (key == "lidar_bev_ego_box_max_y")
                lidar_bev_ego_box_max_y = std::stod(value);
            else if (key == "lidar_bev_near_inner_radius")
                lidar_bev_near_inner_radius = std::stod(value);
            else if (key == "lidar_bev_near_outer_radius")
                lidar_bev_near_outer_radius = std::stod(value);
            else if (key == "lidar_bev_ground_candidate_min_z")
                lidar_bev_ground_candidate_min_z = std::stod(value);
            else if (key == "lidar_bev_ground_candidate_max_z")
                lidar_bev_ground_candidate_max_z = std::stod(value);
            else if (key == "lidar_bev_ground_ransac_distance")
                lidar_bev_ground_ransac_distance = std::stod(value);
            else if (key == "lidar_bev_ground_max_plane_tilt_deg")
                lidar_bev_ground_max_plane_tilt_deg = std::stod(value);
            else if (key == "lidar_bev_ground_fallback_quantile")
                lidar_bev_ground_fallback_quantile = std::stod(value);
            else if (key == "lidar_bev_ground_front_half_angle_deg")
                lidar_bev_ground_front_half_angle_deg = std::stod(value);
            else if (key == "b_lidar_bev_ground_require_forward")
                b_lidar_bev_ground_require_forward = std::stoi(value);
            else if (key == "lidar_bev_ground_failure_fallback_z")
                lidar_bev_ground_failure_fallback_z = std::stod(value);
            else if (key == "lidar_bev_ground_min_points")
                lidar_bev_ground_min_points = std::stoi(value);
            else if (key == "lidar_bev_height_quantile")
                lidar_bev_height_quantile = std::stod(value);
            else if (key == "lidar_bev_h_rel_deadzone_half")
                lidar_bev_h_rel_deadzone_half = std::stod(value);
            else if (key == "lidar_bev_grad_deadzone_half")
                lidar_bev_grad_deadzone_half = std::stod(value);
            else if (key == "lidar_bev_cell_max_points")
                lidar_bev_cell_max_points = std::stoi(value);
            else if (key == "lidar_bev_debug_window_stride")
                lidar_bev_debug_window_stride = std::stoi(value);
            else if (key == "lidar_bev_edge_min_valid_neighbors")
                lidar_bev_edge_min_valid_neighbors = std::stoi(value);


            else if (key == "b_enable_imu_undistortion")
                b_enable_imu_undistortion = std::stoi(value);
            else if (key == "imu_topic")
                imu_topic = value;
            else if (key == "imu_deskew_time_ratio")
                imu_deskew_time_ratio = std::stod(value);
            else if (key == "b_enable_imu_visualization")
                b_enable_imu_visualization = std::stoi(value);
            else if (key == "imu_lidar_total_rows")
                imu_lidar_total_rows = std::stoi(value);
            else if (key == "imu_lidar_total_cols")
                imu_lidar_total_cols = std::stoi(value);
            else if (key.find(choose_car+"_"+"camera_calib_file") != std::string::npos) {
                camera_calib_file.push_back(value);
            }
            else if (key.find(choose_car+"_"+"Lidar2Car_calib_file") != std::string::npos) {
                Lidar2Car_calib_file.push_back(value);
            }
        }
    }

    cout<<choose_car+" Grid Map Start ~"<<endl;
    //对config的参数进行变更
    if(choose_car == "LM"){
        T_LM_lidar2car = read_ini(Lidar2Car_calib_file.at(0));
        camera_front = read_CameraParaV2(camera_calib_file.at(0));
        //根据txt选择激光雷达话题(0:Points 1:Points_tztek)
        if(b_LM_Lidar_Type == 0)
            LM_Lidar_Topic = LM_Lidar_Topic1;
        else if(params->b_LM_Lidar_Type == 1)
            LM_Lidar_Topic = LM_Lidar_Topic2;
//        cout<<LM_Lidar_Topic<<endl;
    }
    else if(choose_car == "HM"){
        camera_front = read_CameraParaV2(camera_calib_file.at(0));
        T_HM_rotate = read_ini(Lidar2Car_calib_file.at(0));
        T_HM_l2c = read_ini(Lidar2Car_calib_file.at(1));
    }
}

Eigen::Matrix4f my_config::read_ini(std::string& radar_type){
    Eigen::Matrix4f Rotation_Translation = Eigen::Matrix4f::Identity();
    char filename[300];
    // sprintf(filename, "%s", radar_type);
    std::snprintf(filename, sizeof(filename), "%s", radar_type.c_str());

    std::ifstream fin2(filename);
    if(fin2.is_open()!=1) {
        std::cout << "Fail to open params file: " << filename << std::endl;
        abort();
        // return;
    }
    std::string t_s2;
    while(fin2 >> t_s2) {
        if(t_s2[0]=='#'||t_s2[0]=='/')
            getline(fin2,t_s2);
        else if( t_s2 == "Rotate") {
            fin2>>Rotation_Translation(0,0);
            fin2>>Rotation_Translation(0,1);
            fin2>>Rotation_Translation(0,2);
            fin2>>Rotation_Translation(1,0);
            fin2>>Rotation_Translation(1,1);
            fin2>>Rotation_Translation(1,2);
            fin2>>Rotation_Translation(2,0);
            fin2>>Rotation_Translation(2,1);
            fin2>>Rotation_Translation(2,2);
        }
        else if( t_s2 == "Translate") {
            fin2>>Rotation_Translation(0,3);
            fin2>>Rotation_Translation(1,3);
            fin2>>Rotation_Translation(2,3);
        }
    }
    fin2.close();
//    cout<<radar_type<<endl;
//    std::cout<<" RT parameters are: "<<std::endl<<Rotation_Translation<<std::endl<<std::endl;
    return Rotation_Translation;
}

Radar2Camera my_config::read_CameraParaV2(std::string& camera_type)
{
    bool b_rt_or_p = 1;
    Eigen::Matrix3f            K_ = Eigen::Matrix3f::Identity();  // K
    Eigen::Matrix<float, 1, 5> KP_; // K1K2P1P2K3
    KP_.setZero();
    Eigen::Matrix4f            RT_ = Eigen::Matrix4f::Identity(); // RT
    Eigen::Matrix4f            P_ = Eigen::Matrix4f::Identity();  // P
    char filename[300];
    std::snprintf(filename, sizeof(filename), "%s", camera_type.c_str());
    std::ifstream fin2(filename);
    if(fin2.is_open()!=1) {
        std::cout << "Fail to open params file: " << filename << std::endl;
        abort();
        // return;
    }
    std::string t_s2;
    while(fin2 >> t_s2) {
        if(t_s2[0]=='#'||t_s2[0]=='/')
            getline(fin2,t_s2);
        else if( t_s2 == "K") {
            fin2>>K_(0,0);
            fin2>>K_(0,1);
            fin2>>K_(0,2);
            fin2>>K_(1,0);
            fin2>>K_(1,1);
            fin2>>K_(1,2);
            fin2>>K_(2,0);
            fin2>>K_(2,1);
            fin2>>K_(2,2);
        }
        else if( t_s2 == "Rotate") {
            fin2>>RT_(0,0);
            fin2>>RT_(0,1);
            fin2>>RT_(0,2);
            fin2>>RT_(1,0);
            fin2>>RT_(1,1);
            fin2>>RT_(1,2);
            fin2>>RT_(2,0);
            fin2>>RT_(2,1);
            fin2>>RT_(2,2);
        }
        else if( t_s2 == "Translate") {
            fin2>>RT_(0,3);
            fin2>>RT_(1,3);
            fin2>>RT_(2,3);
        }
        else if( t_s2 == "k1k2k3") {
            fin2>>KP_(0,0);
            fin2>>KP_(0,1);
            fin2>>KP_(0,4);
        }
        else if( t_s2 == "p1p2") {
            fin2>>KP_(0,2);
            fin2>>KP_(0,3);
        }
        else if( t_s2 == "P") {
            fin2>>P_(0,0);
            fin2>>P_(0,1);
            fin2>>P_(0,2);
            fin2>>P_(0,3);
            fin2>>P_(1,0);
            fin2>>P_(1,1);
            fin2>>P_(1,2);
            fin2>>P_(1,3);
            fin2>>P_(2,0);
            fin2>>P_(2,1);
            fin2>>P_(2,2);
            fin2>>P_(2,3);
        }
        else if( t_s2 == "b_rt_or_p") {
            fin2>>b_rt_or_p;
        }
    }
    fin2.close();
    RT_(3,0) = 0; RT_(3,1) = 0; RT_(3,2) = 0; RT_(3,3) = 1;
    P_(3,0) = 0; P_(3,1) = 0; P_(3,2) = 0; P_(3,3) = 1;

    if(b_rt_or_p==0)
        P_.block<3, 4>(0, 0) = K_ * RT_.block<3, 4>(0, 0);

    // mm -> mi LM 20240509
    RT_ = solveRT(K_, P_);
    // P_.block<3, 4>(0, 0) = K_ * RT_.block<3, 4>(0, 0);
    Radar2Camera radar2camera_i;
    radar2camera_i.P = P_;
    radar2camera_i.camera_K = K_;
    radar2camera_i.camera_KP = KP_;
    std::cout<<camera_type<<endl;
    std::cout<<" K parameters are: "<<std::endl<<K_<<std::endl<<std::endl;
    std::cout<<" kp parameters are: "<<std::endl<<KP_<<std::endl<<std::endl;
    std::cout<<" RT parameters are: "<<std::endl<<RT_<<std::endl<<std::endl;
    std::cout<<" P parameters are: "<<std::endl<<P_<<std::endl<<std::endl;
    return radar2camera_i;
}
Eigen::Matrix4f my_config::solveRT(Eigen::Matrix3f &K, Eigen::Matrix4f &P)
{   // 假设你已经知道了相机内参矩阵 K 和投影矩阵 P 的值
    // 从投影矩阵 P 中提取旋转矩阵 R 和平移向量 t
    Eigen::Matrix3f R = P.block<3, 3>(0, 0);
    Eigen::Vector3f T = P.block<3, 1>(0, 3);

    // 使用相机内参矩阵 K 进行分解
    // Eigen::Matrix3d Kd = K.cast<double>();
    Eigen::Matrix3f K_inv = K.inverse();

    // 计算外参矩阵 RT
    Eigen::Matrix4f RT = Eigen::Matrix4f::Identity();;
    RT.block<3, 3>(0, 0) = K_inv * R;
    // mm => mi
    // RT.block<3, 1>(0, 3) = K_inv * T / 1000.f;
    RT.block<3, 1>(0, 3) = K_inv * T;

    // 第四行这个数字是求不出来的，本来就是手动补上去的，重新计算P阵时，第四行也没有被用来计算
    // P_r = K_ * RT_.block<3, 4>(0, 0);
    // 手动补上
    // RT.row(3) << 0, 0, 0, 1;

    return RT;
}

void my_config::Draw_Points_Whole(std::vector<pcl::PointXYZ> &lidar_points)
{
    int rows = IMAGE_HEIGHT;
    int cols = IMAGE_WIDTH;
    // 设置图像中心作为原点
    int center_x = cols / 2;
    int center_y = rows / 2;
    // 缩放因子（根据需要调整）
    double scale_x = 5;
    double scale_y = 5;

    cv::Mat RadarImg = cv::Mat::zeros(rows, cols, CV_8UC3);
//    cv::Mat RadarImg(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
    int delta = rows / 2 -5;  // 向下移动 x 个像素
    for(const auto& point : lidar_points) {
        int x = static_cast<int>(-point.y * scale_x + cols / 2);
        int y = static_cast<int>(rows / 2 - point.x * scale_y + delta);
        if(x > 0 && y > 0 && x < cols && y < rows)
            cv::circle(RadarImg, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
    }

    // 画出坐标原点
    cv::circle(RadarImg, cv::Point(center_x, center_y + delta),3, cv::Scalar(255, 255, 255), -1);

    // 创建一个与原始图像相同大小的透明遮罩
    cv::Mat mask = cv::Mat::zeros(RadarImg.size(), RadarImg.type());
    // 绘制X轴箭头，并标记"y"
    cv::arrowedLine(mask, cv::Point(cols - 1, center_y + delta), cv::Point(0, center_y + delta),  cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "y", cv::Point(10, center_y + delta - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    int x_delta = 50; //偏50像素
    // 绘制Y轴箭头，并标记"x"
//    cv::arrowedLine(mask, cv::Point(center_x, rows - 1), cv::Point(center_x, 0), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::arrowedLine(mask, cv::Point(cols-x_delta, rows - 1), cv::Point(cols-x_delta, 0), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "x", cv::Point(cols-x_delta + 10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // 添加坐标轴刻度和标签
    int tick_length = 3;  // 刻度线长度
    double tick_step_x = 5.0;   // 每个刻度代表的y值
    double tick_step_y = 5.0;   // 每个刻度代表的x值
    // 自适应计算刻度数量
    int num_ticks_x = cols / (2 * scale_x);  // 根据图像宽度和缩放因子调整X轴刻度数量
    int num_ticks_y = rows / (2 * scale_y);  // 根据图像高度和缩放因子调整Y轴刻度数量

    // X轴的刻度和标签
    for (int i = -num_ticks_x; i <= num_ticks_x; ++i) {
        int x = static_cast<int>(center_x + i * scale_x * tick_step_x);
        if (x >= x_delta*1.2 && x < cols-x_delta*1.2) {
            // 画刻度线
            cv::line(mask, cv::Point(x, center_y + delta - tick_length), cv::Point(x, center_y + delta + tick_length), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(-i * static_cast<int>(tick_step_x));
                cv::putText(mask, label, cv::Point(x - 10, rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // Y轴的刻度和标签
    for (int i = -num_ticks_y; i <= num_ticks_y; ++i) {
        int y = static_cast<int>(center_y + delta - i * scale_y * tick_step_y);
        if (y >= 0 && y < rows) {
            // 画刻度线
            cv::line(mask, cv::Point(cols-x_delta - tick_length, y), cv::Point(cols-x_delta + tick_length, y), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(i * static_cast<int>(tick_step_y));
                cv::putText(mask, label, cv::Point(cols-x_delta + 10, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }
    // 将透明线条的透明度调整为50%
    double alpha = 0.5; // 透明度系数
    cv::addWeighted(mask, alpha, RadarImg, 1.0, 0, RadarImg); // 合并透明遮罩和原始图像
    if(b_show_lidar_points) {
        cv::namedWindow("lidar points", cv::WINDOW_NORMAL);
        cv::imshow("lidar points", RadarImg);
        cv::waitKey(1);
    }

    return;
}

// 鼠标回调函数
void my_config::onMouse(int event, int x, int y, int flags, void* userdata)
{
    MouseClickInfo* info = static_cast<MouseClickInfo*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        // 左键点击：显示高度差值
        if (x >= 0 && x < info->has_data.cols && y >= 0 && y < info->has_data.rows) {
            if (info->has_data.at<uchar>(y, x) > 0) {
                info->click_point = cv::Point(x, y);
                info->height_value = info->height_map.at<float>(y, x);
                info->is_valid = true;

                std::cout << "点击位置: (" << x << ", " << y << "), 高度差: " << info->height_value << "m" << std::endl;
            } else {
                std::cout << "点击位置无有效数据" << std::endl;
            }
        }
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        // 右键点击：清除显示信息
        info->is_valid = false;
        std::cout << "清除显示信息" << std::endl;
    }
}

void my_config::generateColorMap(const cv::Mat &height_diff, const cv::Mat& has_data, std::string window_name)
{
    int rows = height_diff.rows;
    int cols = height_diff.cols;
    int center_x = cols / 2;
    int center_y = rows / 2;
    // 缩放因子（根据需要调整）
    double scale_x = 5;
    double scale_y = 5;
    cv::Mat color_map = cv::Mat::zeros(height_diff.size(), CV_8UC3);
    for (int r = 0; r < height_diff.rows; ++r) {
        for (int c = 0; c < height_diff.cols; ++c) {
            if (has_data.at<uchar>(r, c) == 0) {
                continue; // 保持黑色背景
            }
            float z_value = height_diff.at<float>(r, c);
            uint8_t b = 0, g = 0, r_val = 0;
            setColorByHeight(z_value, r_val, g, b);
            // uint8_t b = 255, g = 255, r_val = 255;
            color_map.at<cv::Vec3b>(r, c) = cv::Vec3b(b, g, r_val);
        }
    }

    // 画出坐标原点
    cv::circle(color_map, cv::Point(center_x, center_y),3, cv::Scalar(255, 255, 255), -1);
    // 创建一个与原始图像相同大小的透明遮罩
    cv::Mat mask = cv::Mat::zeros(color_map.size(), color_map.type());
    // 绘制X轴箭头，并标记"y"
    cv::arrowedLine(mask, cv::Point(cols - 1, center_y), cv::Point(0, center_y),  cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "y", cv::Point(10, center_y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    int x_delta = 40; //偏50像素
    // 绘制Y轴箭头，并标记"x"
//    cv::arrowedLine(mask, cv::Point(center_x, rows - 1), cv::Point(center_x, 0), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::arrowedLine(mask, cv::Point(cols-x_delta, rows - 1), cv::Point(cols-x_delta, 0), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "x", cv::Point(cols-x_delta + 10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // 添加坐标轴刻度和标签
    int tick_length = 3;  // 刻度线长度
    double tick_step_x = 5.0;   // 每个刻度代表的y值
    double tick_step_y = 5.0;   // 每个刻度代表的x值
    // 自适应计算刻度数量
    int num_ticks_x = cols / (2 * scale_x);  // 根据图像宽度和缩放因子调整X轴刻度数量
    int num_ticks_y = rows / (2 * scale_y);  // 根据图像高度和缩放因子调整Y轴刻度数量

    // X轴（v_doppler）的刻度和标签
    for (int i = -num_ticks_x; i <= num_ticks_x; ++i) {
        int x = static_cast<int>(center_x + i * scale_x * tick_step_x);
        if (x >= x_delta*1.2 && x < cols-x_delta*1.2) {
            // 画刻度线
            cv::line(mask, cv::Point(x, center_y  - tick_length), cv::Point(x, center_y  + tick_length), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(-i * static_cast<int>(tick_step_x));
                cv::putText(mask, label, cv::Point(x - 10, center_y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // Y轴（x值）的刻度和标签
    for (int i = -num_ticks_y; i <= num_ticks_y; ++i) {
        int y = static_cast<int>(center_y  - i * scale_y * tick_step_y);
        if (y >= 0 && y < rows) {
            // 画刻度线
            cv::line(mask, cv::Point(cols-x_delta - tick_length, y), cv::Point(cols-x_delta + tick_length, y), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(i * static_cast<int>(tick_step_y));
                cv::putText(mask, label, cv::Point(cols-x_delta + 10, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }
    // 将透明线条的透明度调整为50%
    double alpha = 0.5; // 透明度系数
    cv::addWeighted(mask, alpha, color_map, 1.0, 0, color_map); // 合并透明遮罩和原始图像

    double grid_size = 0.2;
    // 在中心位置绘制3x2米的绿色长方形
    int center_row = rows / 2;
    int center_col = cols / 2;

    // 计算3x2米在栅格中对应的尺寸
    int rect_width_cells = static_cast<int>(2.0 / grid_size);
    int rect_height_cells = static_cast<int>(3.0 / grid_size);

    // 计算矩形的左上角和右下角
    int top = center_row - rect_height_cells / 2;
    int left = center_col - rect_width_cells / 2;
    int bottom = center_row + rect_height_cells / 2;
    int right = center_col + rect_width_cells / 2;

    // 绘制绿色矩形
    for (int r = top; r <= bottom; r++) {
        for (int c = left; c <= right; c++) {
            // 绿色: BGR=(0,255,0)
            color_map.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
        }
    }

    if(b_show_height_diff) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);

        // 设置鼠标回调 - 使用静态变量保持状态
        static MouseClickInfo mouse_info;
        static bool callback_set = false;

        // 更新数据但保持点击状态
        mouse_info.height_map = height_diff.clone();
        mouse_info.has_data = has_data.clone();
        mouse_info.display_image = color_map.clone();
        mouse_info.window_name = window_name;

        // 如果有有效的点击信息，重新绘制到新图像上
        cv::Mat display_image = color_map.clone();
        if (mouse_info.is_valid) {
            // 更新该位置的最新高度值
            int x = mouse_info.click_point.x;
            int y = mouse_info.click_point.y;
            if (x >= 0 && x < has_data.cols && y >= 0 && y < has_data.rows &&
                has_data.at<uchar>(y, x) > 0) {
                mouse_info.height_value = height_diff.at<float>(y, x);
            }

            // 重新绘制点击标记和信息
            cv::circle(display_image, mouse_info.click_point, 5, cv::Scalar(0, 0, 255), 2);

            // 显示高度差值（保留3位小数）
            std::ostringstream height_stream;
            height_stream << std::fixed << std::setprecision(3) << mouse_info.height_value;
            std::string height_text = "Height: " + height_stream.str() + "m";
            cv::putText(display_image, height_text,
                       cv::Point(mouse_info.click_point.x + 10, mouse_info.click_point.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

            // 显示坐标信息
            std::string coord_text = "(" + std::to_string(mouse_info.click_point.x) + "," +
                                    std::to_string(mouse_info.click_point.y) + ")";
            cv::putText(display_image, coord_text,
                       cv::Point(mouse_info.click_point.x + 10, mouse_info.click_point.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        cv::imshow(window_name, display_image);

        // 只设置一次鼠标回调
        if (!callback_set) {
            cv::setMouseCallback(window_name, onMouse, &mouse_info);
            callback_set = true;
        }

        cv::waitKey(1);
    }
}

void my_config::generateRGBMap(const cv::Mat &rgb_data, const cv::Mat& has_data, std::string window_name)
{
    int rows = rgb_data.rows;
    int cols = rgb_data.cols;
    int center_x = cols / 2;
    int center_y = rows / 2;

    // 创建显示图像，复制RGB数据
    cv::Mat display_map = cv::Mat::zeros(rgb_data.size(), CV_8UC3);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (has_data.at<uchar>(r, c) == 0) {
                continue; // 保持黑色背景
            }
            // 直接使用RGB数据
            display_map.at<cv::Vec3b>(r, c) = rgb_data.at<cv::Vec3b>(r, c);
        }
    }

    // 画出坐标原点
    cv::circle(display_map, cv::Point(center_x, center_y), 3, cv::Scalar(255, 255, 255), -1);

    // 创建一个与原始图像相同大小的透明遮罩
    cv::Mat mask = cv::Mat::zeros(display_map.size(), display_map.type());

    // 绘制X轴箭头，并标记"y"
    cv::arrowedLine(mask, cv::Point(cols - 1, center_y), cv::Point(0, center_y), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "y", cv::Point(10, center_y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    int x_delta = 40; //偏50像素
    // 绘制Y轴箭头，并标记"x"
    cv::arrowedLine(mask, cv::Point(cols-x_delta, rows - 1), cv::Point(cols-x_delta, 0), cv::Scalar(255, 255, 255), 1, cv::LINE_8, 0, 0.01);
    cv::putText(mask, "x", cv::Point(cols-x_delta + 10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // 添加坐标轴刻度和标签
    int tick_length = 3;  // 刻度线长度
    double tick_step_x = 5.0;   // 每个刻度代表的y值
    double tick_step_y = 5.0;   // 每个刻度代表的x值
    double scale_x = 5;
    double scale_y = 5;

    // 自适应计算刻度数量
    int num_ticks_x = cols / (2 * scale_x);  // 根据图像宽度和缩放因子调整X轴刻度数量
    int num_ticks_y = rows / (2 * scale_y);  // 根据图像高度和缩放因子调整Y轴刻度数量

    // X轴（y方向）的刻度和标签
    for (int i = -num_ticks_x; i <= num_ticks_x; ++i) {
        int x = static_cast<int>(center_x + i * scale_x * tick_step_x);
        if (x >= x_delta*1.2 && x < cols-x_delta*1.2) {
            // 画刻度线
            cv::line(mask, cv::Point(x, center_y - tick_length), cv::Point(x, center_y + tick_length), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(-i * static_cast<int>(tick_step_x));
                cv::putText(mask, label, cv::Point(x - 10, center_y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // Y轴（x方向）的刻度和标签
    for (int i = -num_ticks_y; i <= num_ticks_y; ++i) {
        int y = static_cast<int>(center_y - i * scale_y * tick_step_y);
        if (y >= 0 && y < rows) {
            // 画刻度线
            cv::line(mask, cv::Point(cols-x_delta - tick_length, y), cv::Point(cols-x_delta + tick_length, y), cv::Scalar(255, 255, 255), 1);

            // 跳过原点
            if (i != 0) {
                // 添加标签
                std::string label = std::to_string(i * static_cast<int>(tick_step_y));
                cv::putText(mask, label, cv::Point(cols-x_delta + 10, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // 将透明线条的透明度调整为50%
    double alpha = 0.5; // 透明度系数
    cv::addWeighted(mask, alpha, display_map, 1.0, 0, display_map); // 合并透明遮罩和原始图像

    double grid_size = 0.2;
    // 在中心位置绘制3x2米的绿色长方形
    int center_row = rows / 2;
    int center_col = cols / 2;

    // 计算3x2米在栅格中对应的尺寸
    int rect_width_cells = static_cast<int>(2.0 / grid_size);
    int rect_height_cells = static_cast<int>(3.0 / grid_size);

    // 计算矩形的左上角和右下角
    int top = center_row - rect_height_cells / 2;
    int left = center_col - rect_width_cells / 2;
    int bottom = center_row + rect_height_cells / 2;
    int right = center_col + rect_width_cells / 2;

    // 绘制绿色矩形
    for (int r = top; r <= bottom; r++) {
        for (int c = left; c <= right; c++) {
            // 绿色: BGR=(0,255,0)
            display_map.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
        }
    }

    if(b_show_rgb_map) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, display_map);
        cv::waitKey(1);
    }
}

// 定义一个函数来根据高度差设置颜色
void my_config::setColorByHeight(float z_value, uint8_t &r, uint8_t &g, uint8_t &b)
{
    // 初始化为黑色
    // 新增负值处理（紫色渐变）
    if(z_value <= 0.0f) {
        float t = std::min(1.0f, -z_value); // t ∈ [0,1]
        r = static_cast<uint8_t>(200 * t);  // 红色分量随负值增大
        b = static_cast<uint8_t>(255 * t);  // 蓝色分量主导
        g = 0;
        return;
    }
    if (z_value > 0.0 && z_value < 0.1) {
        b = 255; // 蓝
    } else if (z_value < 0.2) {
        b = 230; g = 30; // 深蓝略带绿
    } else if (z_value < 0.3) {
        b = 200; g = 60; // 蓝绿过渡
    } else if (z_value < 0.4) {
        b = 170; g = 90; // 蓝绿
    } else if (z_value < 0.5) {
        b = 140; g = 120; // 青色
    } else if (z_value < 0.6) {
        b = 110; g = 150; // 青绿
    } else if (z_value < 0.7) {
        b = 80; g = 180; // 绿青
    } else if (z_value < 0.8) {
        b = 50; g = 210; // 绿色
    } else if (z_value < 0.9) {
        r = 50; g = 240; b = 30; // 黄绿
    } else if (z_value < 1.0) {
        r = 100; g = 255; b = 0; // 黄色
    } else if (z_value < 1.1) {
        r = 150; g = 220; b = 0; // 黄橙
    } else if (z_value < 1.2) {
        r = 200; g = 180; b = 0; // 橙色
    } else if (z_value < 1.3) {
        r = 230; g = 130; b = 0; // 橙红
    } else if (z_value < 1.4) {
        r = 255; g = 80; b = 0; // 红橙
    } else if (z_value < 1.5) {
        r = 255; g = 40; b = 0; // 深红色
    } else {
        r = 255; g = 0; b = 0; // 纯红色（超过1.5米的高度差）
    }
}


void my_config::LM_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points)
{
    for(const auto& point : lidar_points) {
        Eigen::Vector4f point_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_car = T_LM_lidar2car * point_lidar;
        pcl::PointXYZ car_point(point_car(0), point_car(1), point_car(2));
        car_points.push_back(car_point);

    }
}
void my_config::HM_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points)
{
    for(const auto& point : lidar_points) {
        Eigen::Vector4f point_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_car = T_HM_l2c * T_HM_rotate * point_lidar;
        pcl::PointXYZ car_point(point_car(0), point_car(1), point_car(2));
        car_points.push_back(car_point);

    }
}
void my_config::new_LM_M1_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points)
{
    for(const auto& point : lidar_points) {
        Eigen::Vector4f point_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_car = T_New_LM_lidar2car * point_lidar;
        pcl::PointXYZ car_point(point_car(0), point_car(1), point_car(2));
        car_points.push_back(car_point);
    }
}

void my_config::new_LM_BP_2_M1_lidar(std::vector<pcl::PointXYZ> &BP_lidar_points, std::vector<pcl::PointXYZ> &M1_lidar_points)
{
    for(const auto& point : BP_lidar_points) {
        Eigen::Vector4f point_lidar(point.x, point.y, point.z, 1.0);
        //去除打在自车上的点        
        if(point_lidar(0) > -0.3 && point_lidar(0) < 0.3 &&
           point_lidar(1) > -1.0 && point_lidar(1) < 1.0)
        {
            continue;
        }
        Eigen::Vector4f M1_point = T_BP_2_M1 * point_lidar;
        pcl::PointXYZ M1_pcl_point(M1_point(0), M1_point(1), M1_point(2));
        M1_lidar_points.push_back(M1_pcl_point);
    }
}

void my_config::trans_color_lidar2car(const std::vector<PointXYZRGBValid> &colored_lidar_points, std::vector<PointXYZRGBValid> &colored_car_points)
{
    Eigen::Matrix4f T_lidar2car;
    if (choose_car == "LM")
    {
        T_lidar2car = T_LM_lidar2car;
    }
    else if (choose_car == "HM")
    {
        // T_lidar2car = T_HM_l2c * T_HM_rotate;
        T_lidar2car = T_HM_l2c;
    }
    else if (choose_car == "new_LM")
    {
        T_lidar2car = T_New_LM_lidar2car;
    }
    for (const auto &point : colored_lidar_points)
    {
        Eigen::Vector4f point_lidar(point.x, point.y, point.z, 1.0);
        Eigen::Vector4f point_car = T_lidar2car * point_lidar;
        PointXYZRGBValid car_point(point_car(0), point_car(1), point_car(2), point.r, point.g, point.b, point.has_rgb);
        colored_car_points.push_back(car_point);
    }
}

Eigen::Vector2d my_config::trans_car_to_global(
    Eigen::Vector2d point_car,
    const self_state::LocalPose &body_pose) const {
    Eigen::Vector2d point_global;

    double dr_theta = body_pose.dr_heading;
    double sindt = sin(dr_theta);
    double cosdt = cos(dr_theta);

    double dx = body_pose.dr_x;
    double dy = body_pose.dr_y;

    point_global(0) = cosdt * point_car(0) - sindt * point_car(1) + dx;
    point_global(1) = sindt * point_car(0) + cosdt * point_car(1) + dy;

    return point_global;
}

Eigen::Vector2d my_config::trans_global_to_car(
    Eigen::Vector2d point_global,
    const self_state::LocalPose &body_pose) const {
    Eigen::Vector2d point_car;

    double dr_theta = body_pose.dr_heading; //弧度
    double sindt = sin(dr_theta);
    double cosdt = cos(dr_theta);

    // 反转平移：减去车体位置
    double dx = point_global(0) - body_pose.dr_x;
    double dy = point_global(1) - body_pose.dr_y;

    // 反转旋转
    point_car(0) = cosdt * dx + sindt * dy;
    point_car(1) = -sindt * dx + cosdt * dy;
    return point_car;
}

Eigen::Matrix3d my_config::eulerAnglesToRotationMatrix(double roll, double pitch, double yaw)
{
    // 计算 sin / cos
    double sr = std::sin(roll);
    double cr = std::cos(roll);
    double sp = std::sin(pitch);
    double cp = std::cos(pitch);
    double sy = std::sin(yaw);
    double cy = std::cos(yaw);

    // 绕 x 轴的旋转矩阵 Rx(roll)
    Eigen::Matrix3d Rx;
    Rx << 1,   0,    0,
          0,   cr,  -sr,
          0,   sr,   cr;

    // 绕 y 轴的旋转矩阵 Ry(pitch)
    Eigen::Matrix3d Ry;
    Ry <<  cp,  0,  sp,
           0,   1,   0,
          -sp,  0,  cp;

    // 绕 z 轴的旋转矩阵 Rz(yaw)
    Eigen::Matrix3d Rz;
    Rz <<  cy,  -sy,  0,
           sy,   cy,  0,
           0,    0,   1;

    // Z-Y-X 顺序: Rz(yaw) * Ry(pitch) * Rx(roll)
    Eigen::Matrix3d R = Rz * Ry * Rx;
    return R;
}

Eigen::Vector3d my_config::trans_car_to_global_3d(
    const Eigen::Vector3d &point_car,
    const self_state::LidarLocalPose &lidar_localpose)
{
    // 1) 构造旋转矩阵 R
    Eigen::Matrix3d R = eulerAnglesToRotationMatrix(
        lidar_localpose.roll,
        lidar_localpose.pitch,
        lidar_localpose.azimuth
    );

    // 2) 得到平移向量 T
    Eigen::Vector3d T(lidar_localpose.x, lidar_localpose.y, lidar_localpose.z);

    // 3) 应用变换
    Eigen::Vector3d point_global = R * point_car + T;
    return point_global;
}

Eigen::Vector3d my_config::trans_global_to_car_3d(
    const Eigen::Vector3d &point_global,
    const self_state::LidarLocalPose &lidar_localpose)
{
    // 1) 构造旋转矩阵 R
    Eigen::Matrix3d R = eulerAnglesToRotationMatrix(
        lidar_localpose.roll,
        lidar_localpose.pitch,
        lidar_localpose.azimuth
    );

    // 2) 得到平移向量 T
    Eigen::Vector3d T(lidar_localpose.x, lidar_localpose.y, lidar_localpose.z);

    // 3) 先做平移的逆操作
    Eigen::Vector3d temp = point_global - T;

    // 4) 再乘以 R^T （旋转的逆）
    Eigen::Vector3d point_car = R.transpose() * temp;

    return point_car;
}
