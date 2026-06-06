#include "map/image_proj.h"
#include "map/sensor_map.h"



void IMAGE_PROJ::ComputeUVMap()
{
    Eigen::Matrix<float, 3, 3> camera_K_ = params->camera_front.camera_K;
    Eigen::Matrix<float, 5, 1> camera_distcoeff_ = params->camera_front.camera_KP;
    double fx, fy, cx, cy;
    double k1, k2, k3, p1, p2;
    fx = camera_K_(0, 0);
    cx = camera_K_(0, 2);
    fy = camera_K_(1, 1);
    cy = camera_K_(1, 2);
    k1 = camera_distcoeff_(0);
    k2 = camera_distcoeff_(1);
    p1 = camera_distcoeff_(2);
    p2 = camera_distcoeff_(3);
    k3 = camera_distcoeff_(4);
    // 计算去畸变后图像的内容
    const int numel = COLORHEIGHT * COLORWIDTH;
    for (int i = 0; i < numel; i++) {
        int row = i / COLORWIDTH;
        int col = i % COLORWIDTH;

        // 标准归一化像平面坐标
        double x = (col - cx) / fx;
        double y = (row - cy) / fy;

        double xy = x * y;
        double x2 = x * x;
        double y2 = y * y;

        double r2 = x2 + y2;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        // 畸变后归一化像平面坐标
        double x_distorted = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * xy +
                p2 * (r2 + 2 * x2);
        double y_distorted = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) +
                p1 * (r2 + 2 * y2) + 2 * p2 * xy;

        // 像素坐标
        int u_d = fx * x_distorted + cx;
        int v_d = fy * y_distorted + cy;

        // record valid pair
        if (u_d >= 0 && u_d < COLORWIDTH && v_d >= 0 && v_d < COLORHEIGHT) {
            this->uv_map.push_back(
                        std::make_pair(Eigen::Vector2i(row, col), Eigen::Vector2i(v_d, u_d)));
        }
    }
}


void IMAGE_PROJ::Undistort()
{

    
    if(params->img_map.find("Camera Front") == params->img_map.end())
        return;

    // 获取原始图像

    SensorMap::img_map_mutex.lock();
    cv::Mat original_image = params->img_map["Camera Front"].clone();
    cv::Mat input_image;
    SensorMap::img_map_mutex.unlock();    
    // 如果原始图像尺寸不是目标尺寸，则进行缩放
    if(original_image.cols != COLORWIDTH || original_image.rows != COLORHEIGHT) {
        cv::resize(original_image, input_image, cv::Size(COLORWIDTH, COLORHEIGHT));
        // std::cout << "图像从 " << original_image.cols << "x" << original_image.rows
        //           << " 缩放到 " << COLORWIDTH << "x" << COLORHEIGHT << std::endl;
        // LCPP_INFO(this->get_logger(), "图像从 %dx%d 缩放到 %dx%d", 
        //     original_image.cols, original_image.rows, COLORWIDTH, COLORHEIGHT);
    } else {
        input_image = original_image;
    }

    // 图像去畸变
    cv::Mat UndistortImage = cv::Mat::zeros(COLORHEIGHT,COLORWIDTH,CV_8UC3);
    UndistortImage.setTo(0);
    auto clock_start = clock();
    for (const std::pair<Eigen::Vector2i, Eigen::Vector2i> &vec : this->uv_map) {
        // 检查索引边界
        if(vec.first(0) >= 0 && vec.first(0) < COLORHEIGHT &&
           vec.first(1) >= 0 && vec.first(1) < COLORWIDTH &&
           vec.second(0) >= 0 && vec.second(0) < input_image.rows &&
           vec.second(1) >= 0 && vec.second(1) < input_image.cols) {
            UndistortImage.at<cv::Vec3b>(vec.first(0), vec.first(1)) =
                    input_image.at<cv::Vec3b>(vec.second(0), vec.second(1));
        }
    }
    auto clock_end = clock();
    // std::cout << "去畸变用时: " << (clock_end - clock_start) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
    // auto duration_ms = (clock_end - clock_start) * 1000 / CLOCKS_PER_SEC;
    // RCLCPP_INFO(this->get_logger(), "去畸变用时: %ld ms", duration_ms);
    params->img_map["Camera Front Undistort"] = UndistortImage;
}

std::vector<PointXYZRGBValid> IMAGE_PROJ::proj_points(std::vector<pcl::PointXYZ> &lidar_points)
{
    std::vector<PointXYZRGBValid> colored_points;

    // 获取去畸变后的图像
    cv::Mat undistort_image;
    cv::Mat visualization_image;
    bool has_image = false;

    if(params->img_map.find("Camera Front Undistort") != params->img_map.end()) {
        undistort_image = params->img_map["Camera Front Undistort"];
        visualization_image = undistort_image.clone(); // 用于可视化的图像副本
        has_image = true;
    }

    // 获取原始投影矩阵并缩放以适应新分辨率
    Eigen::Matrix<float, 3, 4> Pmatrix = params->camera_front.P.block<3, 4>(0, 0);

    // 缩放投影矩阵以适应新的分辨率 (960x540 vs 原始分辨率)
    const double ORIGINAL_WIDTH = 1920.0;
    const double ORIGINAL_HEIGHT = 1080.0;
    double scale_x = static_cast<double>(COLORWIDTH) / ORIGINAL_WIDTH;
    double scale_y = static_cast<double>(COLORHEIGHT) / ORIGINAL_HEIGHT;

    // 缩放投影矩阵的前两行（对应u和v坐标）
    Pmatrix.row(0) *= scale_x;  // u坐标缩放
    Pmatrix.row(1) *= scale_y;  // v坐标缩放

    
    int valid_projections = 0;
    int total_points = lidar_points.size();

    // 处理所有激光雷达点
    for (const auto &point : lidar_points) {
        // 创建扩展点云点，先设置3D坐标
        PointXYZRGBValid enhanced_point(point.x, point.y, point.z);

        // 尝试投影到图像（只对前方的点进行投影）
        bool projection_success = false;

        if(has_image && point.x > 0) {  // 有图像且点在前方
            Eigen::Vector4f p(point.x, point.y, point.z, 1.0);
            Eigen::Vector3f p1 = Pmatrix * p;

            // 检查深度是否为正
            if(p1(2) > 0) {
                int u = static_cast<int>(p1(0) / p1(2));
                int v = static_cast<int>(p1(1) / p1(2));

                if (u >= 0 && u < COLORWIDTH && v >= 0 && v < COLORHEIGHT)
                {
                    // 投影成功，从图像中获取RGB颜色
                    cv::Vec3b pixel_color = undistort_image.at<cv::Vec3b>(v, u);
                    enhanced_point.r = pixel_color[2];  // OpenCV BGR -> RGB
                    enhanced_point.g = pixel_color[1];
                    enhanced_point.b = pixel_color[0];
                    enhanced_point.has_rgb = true;

                    projection_success = true;
                    valid_projections++;

                    if (params->b_show_proj)
                    {
                        // 在可视化图像上绘制投影点（保持原来的jet color逻辑）
                        double dist = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
                        int mapped_color_index = std::min(static_cast<int>((dist / 75) * 640), 639);
                        // 动态半径计算 - 根据当前分辨率自动调整
                        // 基于图像分辨率动态计算半径，避免硬编码
                        const double ORIGINAL_WIDTH = 1920.0;
                        const double scale_factor = static_cast<double>(COLORWIDTH) / ORIGINAL_WIDTH;
                        constexpr int original_base_r = 6, original_min_r = 2;
                        int base_r = static_cast<int>(original_base_r * scale_factor);
                        int min_r = static_cast<int>(original_min_r * scale_factor);
                        // 确保最小半径至少为1
                        base_r = std::max(base_r, 1);
                        min_r = std::max(min_r, 1);

                        constexpr double max_dist = 100;
                        int radius = static_cast<int>(base_r - (dist / max_dist) * (base_r - min_r));
                        radius = std::clamp(radius, min_r, base_r);

                        // 使用jet colormap着色
                        visualization_image.at<cv::Vec3b>(v, u)[0] = jet_color_map[mapped_color_index][2];
                        visualization_image.at<cv::Vec3b>(v, u)[1] = jet_color_map[mapped_color_index][1];
                        visualization_image.at<cv::Vec3b>(v, u)[2] = jet_color_map[mapped_color_index][0];
                        cv::circle(visualization_image, cv::Point(u, v), radius,
                                   cv::Scalar(jet_color_map[mapped_color_index][2],
                                              jet_color_map[mapped_color_index][1],
                                              jet_color_map[mapped_color_index][0]),
                                   -1, 8);
                    }
                }
            }
        }

        // 如果投影失败，保持默认灰色和has_rgb=false（构造函数已设置）
        // 所有点都添加到结果中，无论是否投影成功
        colored_points.push_back(enhanced_point);
    }

    // 保存可视化结果（如果有图像）
    if(has_image) {
        params->img_map["Lidar Proj"] = visualization_image;
    }

    std::cout << "点云处理完成 - 总点数: " << total_points
              << ", 成功投影: " << valid_projections
              << ", 灰色点: " << (total_points - valid_projections) << std::endl;
    // RCLCPP_INFO(this->get_logger(), "点云处理完成 - 总点数: %zu, 成功投影: %zu, 灰色点: %zu",
    //         total_points, valid_projections, total_points - valid_projections);

    return colored_points;
}




// 新增：点云投影到图像功能
void IMAGE_PROJ::projectPointCloudToImage(const std::vector<pcl::PointXYZ>& cloud,
                                          cv::Mat& image,
                                          bool show_depth_color)
{
    if (cloud.empty()) return;

    // 深度范围设置
    double z_min = 10.0;
    double z_max = 50.0;
    int projected_points = 0;

    // 获取缩放后的投影矩阵
    Eigen::Matrix<float, 3, 4> Pmatrix = params->camera_front.P.block<3, 4>(0, 0);

    // 缩放投影矩阵以适应当前图像分辨率
    const double ORIGINAL_WIDTH = 1920.0;
    const double ORIGINAL_HEIGHT = 1080.0;
    double scale_x = static_cast<double>(image.cols) / ORIGINAL_WIDTH;
    double scale_y = static_cast<double>(image.rows) / ORIGINAL_HEIGHT;

    Pmatrix.row(0) *= scale_x;  // u坐标缩放
    Pmatrix.row(1) *= scale_y;  // v坐标缩放

    for (const auto& pt : cloud)
    {
        // 只处理前方的点
        if (pt.x <= 0) continue;

        // 投影到图像坐标
        Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector3f p_img = Pmatrix * p;

        if (p_img(2) <= 0) continue; // 忽略相机后方的点

        double u = p_img(0) / p_img(2);
        double v = p_img(1) / p_img(2);

        // 检查投影点是否在图像范围内
        if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
        {
            cv::Scalar color;

            if (show_depth_color)
            {
                // 计算深度并进行彩色编码
                double depth = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                double z_norm = (depth - z_min) / (z_max - z_min); // 归一化到[0,1]
                z_norm = std::min(std::max(z_norm, 0.0), 1.0);

                // 彩色映射：近红远蓝
                if (z_norm < 0.5)
                {
                    // 近距离 -> 红到绿
                    int r = 255;
                    int g = static_cast<int>(z_norm / 0.5 * 255);
                    int b = 0;
                    color = cv::Scalar(b, g, r);
                }
                else
                {
                    // 远距离 -> 绿到蓝
                    int r = static_cast<int>((1.0 - (z_norm - 0.5) / 0.5) * 255);
                    int g = 255;
                    int b = static_cast<int>((z_norm - 0.5) / 0.5 * 255);
                    color = cv::Scalar(b, g, r);
                }
            }
            else
            {
                // 使用固定颜色
                color = cv::Scalar(0, 255, 0); // 绿色
            }

            cv::circle(image, cv::Point(static_cast<int>(u), static_cast<int>(v)), 2, color, -1);
            projected_points++;
        }
    }

    // 显示投影点数量
    std::string text = "Projected Points: " + std::to_string(projected_points);
    cv::putText(image, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    std::cout << "[投影] 成功投影 " << projected_points << " 个点到图像" << std::endl;
}

// 新增：点云着色功能（从图像中提取颜色）
void IMAGE_PROJ::getColoredPointCloud(const std::vector<pcl::PointXYZ>& cloud_in,
                                      const cv::Mat& image,
                                      std::vector<PointXYZRGBValid>& cloud_out)
{
    cloud_out.clear();
    if (cloud_in.empty() || image.empty()) return;

    // 获取缩放后的投影矩阵
    Eigen::Matrix<float, 3, 4> Pmatrix = params->camera_front.P.block<3, 4>(0, 0);

    // 缩放投影矩阵以适应当前图像分辨率
    const double ORIGINAL_WIDTH = 1920.0;
    const double ORIGINAL_HEIGHT = 1080.0;
    double scale_x = static_cast<double>(image.cols) / ORIGINAL_WIDTH;
    double scale_y = static_cast<double>(image.rows) / ORIGINAL_HEIGHT;

    Pmatrix.row(0) *= scale_x;  // u坐标缩放
    Pmatrix.row(1) *= scale_y;  // v坐标缩放

    int colored_points = 0;

    for (const auto& pt : cloud_in)
    {
        // 创建输出点，先设置3D坐标
        PointXYZRGBValid colored_pt(pt.x, pt.y, pt.z);

        // 只对前方的点进行投影着色
        if (pt.x > 0) {
            // 投影到图像坐标
            Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector3f p_img = Pmatrix * p;

            if (p_img(2) > 0) { // 检查深度
                double u = p_img(0) / p_img(2);
                double v = p_img(1) / p_img(2);

                // 检查投影点是否在图像范围内
                if (u >= 0 && u < image.cols && v >= 0 && v < image.rows)
                {
                    // 从图像中提取颜色
                    cv::Vec3b color_uv = image.at<cv::Vec3b>(static_cast<int>(v), static_cast<int>(u));
                    colored_pt.r = color_uv[2];  // OpenCV BGR -> RGB
                    colored_pt.g = color_uv[1];
                    colored_pt.b = color_uv[0];
                    colored_pt.has_rgb = true;
                    colored_points++;
                }
            }
        }

        // 添加到输出点云（无论是否成功着色）
        cloud_out.push_back(colored_pt);
    }

    std::cout << "[着色] 成功为 " << colored_points << " / " << cloud_in.size()
              << " 个点添加颜色" << std::endl;
}

