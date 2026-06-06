#ifndef IMAGE_PROJ_H
#define IMAGE_PROJ_H
#pragma once
#include "include/map/my_config.h"
#include "include/map/colors.hpp"

// #define COLORWIDTH 1920
// #define COLORHEIGHT 1080
#define COLORWIDTH 960
#define COLORHEIGHT 540


class IMAGE_PROJ
{
public:
    IMAGE_PROJ()
    {
        //计算相机去畸变参数
        ComputeUVMap();
    }
    ~IMAGE_PROJ() {}

    void ComputeUVMap();
    void Undistort();
    std::vector<PointXYZRGBValid> proj_points(std::vector<pcl::PointXYZ> &car_points);


    

    // 新增：点云投影到图像功能
    void projectPointCloudToImage(const std::vector<pcl::PointXYZ>& cloud,
                                  cv::Mat& image,
                                  bool show_depth_color = true);

    // 新增：点云着色功能（从图像中提取颜色）
    void getColoredPointCloud(const std::vector<pcl::PointXYZ>& cloud_in,
                              const cv::Mat& image,
                              std::vector<PointXYZRGBValid>& cloud_out);

    void run();
private:
    std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>> uv_map; //去畸变
};
using IMAGE_PROJPtr = std::shared_ptr<IMAGE_PROJ>;

#endif // IMAGE_PROJ_H