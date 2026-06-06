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
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <ros/ros.h>
#include <algorithm>
#include <limits>

/**
 * @brief GridMap处理类，用于替代原有的稀疏哈希表实现
 * 提供高性能的栅格地图累积和处理功能
 */
class GridMapHandler_v4 {
public:
    /**
     * @brief 构造函数
     */
    GridMapHandler_v4();

    /**
     * @brief 析构函数
     */
    ~GridMapHandler_v4() = default;

    /**
     * @brief 初始化GridMap
     * @param map_size_x X方向地图大小(米)
     * @param map_size_y Y方向地图大小(米)
     * @param resolution 栅格分辨率(米)
     * @param frame_id 坐标系ID
     */
    void initialize(double map_size_x = 50.0, double map_size_y = 50.0,
                   double resolution = 0.2, const std::string& frame_id = "map");

    //计算残差
    void calculate_Residual();

    /**
     * @brief 主处理函数，处理彩色点云数据
     * @param colored_car_points 彩色点云数据
     * @param body_pose 车辆位姿信息
     */
    void gridmap_process(const std::vector<PointXYZRGBValid>& colored_car_points,
                const self_state::LocalPose& body_pose);

    /**
     * @brief 生成可视化图像，与原有接口兼容
     * @param height_map 输出高度图
     * @param rgb_map 输出RGB图
     * @param has_data 输出数据掩码图
     * @param current_heading 当前车辆朝向（弧度）
     */
    void generateVisualization(cv::Mat& height_map, cv::Mat& rgb_map, cv::Mat& has_data, double current_heading = 0.0);

    /**
     * @brief 生成坡度可视化图像
     * @param slope_map 输出坡度图
     * @param has_data 输出数据掩码图
     * @param current_heading 当前车辆朝向（弧度）
     */
    void generateSlopeVisualization(cv::Mat& slope_map, cv::Mat& has_data, double current_heading = 0.0);

    /**
     * @brief 生成可视化图像（优化版本 - 使用预计算的车体坐标索引）
     * @param height_map 输出高度差图像
     * @param rgb_map 输出RGB图像
     * @param has_data 输出数据掩码图
     * @param current_heading 当前车辆朝向（弧度）
     */
    void generateVisualization_v2(cv::Mat& height_map, cv::Mat& rgb_map, cv::Mat& has_data, double current_heading = 0.0);

    void generateVisualization_v3(cv::Mat &height_map, cv::Mat &rgb_map, cv::Mat &has_data, double current_heading);

    /**
     * @brief 生成坡度可视化颜色图
     * @param slope_map 坡度图像（度数）
     * @param has_data 数据有效性图像
     * @param window_name 窗口名称
     */
    void generateSlopeColorMap(cv::Mat &slope_map, cv::Mat &has_data, const std::string& window_name);

    /**
     * @brief 获取GridMap引用
     * @return GridMap引用
     */
    const grid_map::GridMap& getGridMap() const { return map_; }

    /**
     * @brief 发布GridMap消息
     * @param publisher ROS发布者
     */
    void publishGridMap(ros::Publisher& publisher);

    /**
     * @brief 显示高差图和RGB图（与原有接口兼容）
     * @param current_heading 当前车辆朝向（弧度），用于以自车为坐标系的显示
     */
    void showMaps(double current_heading = 0.0);

    /**
     * @brief 使用OpenCV显示GridMap数据
     * @param current_heading 当前车辆朝向（弧度）
     */
    void opencv_showMaps(double current_heading = 0.0);

private:
    // GridMap核心对象
    grid_map::GridMap map_;

    // 地图参数
    double map_size_x_;
    double map_size_y_;
    double resolution_;
    std::string frame_id_;

    // 可视化参数
    int img_rows_;
    int img_cols_;
    double scale_factor_;
    int img_size_;

    //残差
    double residual_x = 0;
    double residual_y = 0;

    // 性能统计
    std::chrono::high_resolution_clock::time_point last_process_time_;

    /**
     * @brief 更新地图位置，跟随机器人移动
     * @param body_pose 车辆位姿
     */
    void updateMapPosition(const self_state::LocalPose& body_pose);

    /**
     * @brief 批量处理点云数据
     * @param points 点云数据
     * @param body_pose 车辆位姿
     */
    void processPointCloud(const std::vector<PointXYZRGBValid>& points,
                          const self_state::LocalPose& body_pose);

    /**
     * @brief 寻找地面基准（3x3邻域最小值）
     */
    void find_ground();

    /**
     * @brief 计算高度差（最高点减去地面基准）
     */
    void get_HeightDiff();

    void get_HeightDiff_v2();

    void find_ground_v2();

    /**
     * @brief 填补孔洞（使用中值滤波）
     */
    void fillHoles();

    /**
     * @brief 计算坡度（基于邻域梯度）
     */
    void calculateSlope();

    /**
     * @brief 处理动态物体邻域
     */
    void processDynamicObjects();

    /**
     * @brief 重置绝对高度相关的层（每帧处理完后调用）
     * 由于车辆Z轴位置未补偿，绝对高度值只能在单帧内使用
     */
    void resetAbsoluteHeightLayers();

    // 固定栅格更新中新数据权重
    float new_weight = 0.7f;

    // 动态物体检测阈值
    static constexpr float dynamic_threshold = 0.2f;  // 高度差变化阈值（米）
    static constexpr float ground_height_value = 0.01f;  // 地面高度差值（米）
    static constexpr int min_observations = 5;  // 最小观测次数

    // 临时变量
    Eigen::MatrixXf temp_intrested_flag;
    Eigen::MatrixXf history_intrested_flag;

};

