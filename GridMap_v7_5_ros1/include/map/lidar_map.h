#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "include/map/my_config.h"
#include <unordered_map>
#include <cmath>
#include <chrono>

template<typename T>
struct lidarhash;

template<> 
struct lidarhash<std::pair<int, int>> {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return static_cast<size_t>(p.first) << 32 | (static_cast<size_t>(p.second) & 0xFFFFFFFF);
    }
};

static float weights_kernel[5][5] = {
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
};

struct LidarCell{
    bool visited = false;
    double max_height;
    double temp_min_height;
    double min_height;
    double height_diff = 0.0;
    std::deque<double> history_height_diffs;  // 存储历史高度差
    cv::Vec3b rgb = cv::Vec3b(128, 128, 128);  // 默认灰色
    bool has_valid_color = false;  // 标志是否有有效的颜色信息


    // 带参数的构造函数
    LidarCell(bool vis, double max, double temp_min, double min, double hd)
        : visited(vis), max_height(max), temp_min_height(temp_min), min_height(min), height_diff(hd),
          rgb(cv::Vec3b(128, 128, 128)), has_valid_color(false) {}

    // 带RGB的构造函数
    LidarCell(bool vis, double max, double temp_min, double min, double hd, const cv::Vec3b& color)
        : visited(vis), max_height(max), temp_min_height(temp_min), min_height(min), height_diff(hd),
          rgb(color), has_valid_color(true) {}

    // 默认构造函数
    LidarCell() = default;

    // 添加新的高度差到历史记录
    void addData(double hd) {
        history_height_diffs.push_back(hd);
        if (history_height_diffs.size() > 5) {
            history_height_diffs.pop_front();  // 删除最早的记录
        }
    }

    // RGB相关操作函数
    void setRGB(uint8_t r, uint8_t g, uint8_t b) {
        rgb = cv::Vec3b(b, g, r);  // 注意OpenCV是BGR顺序
    }

    void setRGB(const cv::Vec3b& color) {
        rgb = color;
    }

    // RGB平均融合,默认
    void fuseRGBWith(const LidarCell& other) {
        rgb = (rgb + other.rgb) / 2;
    }

    // 加权RGB融合
    void fuseRGBWith(const LidarCell& other, double weight) {
        rgb = rgb * (1.0 - weight) + other.rgb * weight;
    }

    // 计算并返回历史高度差的平均值
    double getAverageHeightDiff() const {
        if (history_height_diffs.empty()) {
            return 0.0;  // 如果容器为空，返回0
        }
        
        double sum = 0.0;
        for (const auto& hd : history_height_diffs) {
            sum += hd;

        }

        return sum / history_height_diffs.size();
    }
};

class LidarMap
{
public:
    LidarMap();
    ~LidarMap(){};
    cv::Mat height_map;
    cv::Mat has_data;
    cv::Mat rgb_map;

    void process(const std::vector<PointXYZRGBValid> &colored_car_points, const self_state::LocalPose &body_pose);
private:
    std::unordered_map<std::pair<int, int>, LidarCell, lidarhash<std::pair<int, int>>> lidar_cells;
    // 栅格参数
    double grid_size = 0.2;
    double inv_grid_size = 1.0 / grid_size;
    double x_max = 50.0;  // x前向范围 [-50m,50m]
    double y_max = 50.0;  // y左向范围 [-50m,50m]
    double z_max = 2.5;
//    int cols = static_cast<int>((x_max - x_min) / grid_size); // 500列
//    int rows = static_cast<int>((y_max - y_min) / grid_size); // 500行
    int cols=500;
    int rows=500;
    int center_col = cols / 2;  // 中心列索引250
    int center_row = rows / 2;  // 中心行索引250
    

};
