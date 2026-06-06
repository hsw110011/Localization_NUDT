#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <memory>
#include "include/map/my_config.h"
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <deque>
#include <algorithm>
#include <limits>
#include <string>

#include "self_state/LocalPose.h"

// 前向声明
template<typename CellType>
class CRollingGridMap_v3;

/**
 * @brief 高度差栅格单元，结合ColorPoint的滚动栅格技术和lidar_map的高度差计算
 * 专门用于高度差地图和RGB颜色地图的生成
 */
class HeightDiffCell {
public:
    // 构造函数
    HeightDiffCell() {
        reset();
    }

    ~HeightDiffCell() = default;

    // 核心数据成员
    float max_height = -std::numeric_limits<float>::max();     // 最大高度
    float min_height = std::numeric_limits<float>::max();      // 单帧栅格落入点云的最低点
    float ground_height = std::numeric_limits<float>::max();   // 3×3邻域的基准地面高度
    float height_diff = std::numeric_limits<float>::quiet_NaN(); // 高度差(障碍物高度)

    // RGB颜色信息
    uint8_t r = 128, g = 128, b = 128;                        // RGB颜色值
    bool has_valid_color = false;                             // 颜色有效性标志
    bool has_data = false;                                    // 数据有效性标志(当前)

    // 历史数据管理(用于时间滤波)
    std::deque<float> history_height_diffs;                   // 历史高度差队列
    int observation_count = 0;                                // 观测次数

    /**
     * @brief 重置栅格单元
     */
    void reset() {
        max_height = -std::numeric_limits<float>::max();
        min_height = std::numeric_limits<float>::max();
        ground_height = std::numeric_limits<float>::max();
        height_diff = 0.0f;
        r = 128; g = 128; b = 128;
        has_valid_color = false;
        has_data = false;
        history_height_diffs.clear();
        observation_count = 0;
    }

    /**
     * @brief 使用PointXYZRGBValid更新栅格单元
     * @param point 带颜色的点云数据
     */
    void updateWithPoint(const PointXYZRGBValid& point) {
        has_data = true;
        observation_count++;

        // 更新高度信息
        if (point.z > max_height) {
            max_height = point.z;
            // 使用最高点的颜色(如果有效)
            if (point.has_rgb) {
                r = point.r;
                g = point.g;
                b = point.b;
                has_valid_color = true;
            }
        }

        // 更新单帧最小高度
        if (point.z < min_height) {
            min_height = point.z;
        }
    }

    /**
     * @brief 添加历史高度差数据
     * @param hd 高度差值
     */
    void addHistoryData(float hd) {
        history_height_diffs.push_back(hd);
        if (history_height_diffs.size() > 5) {
            history_height_diffs.pop_front();  // 保持最近5帧
        }
    }

    /**
     * @brief 获取平均高度差
     * @return 历史高度差的平均值
     */
    float getAverageHeightDiff() const {
        if (history_height_diffs.empty()) {
            return height_diff;  // 可能是NaN，调用者需要检查
        }

        float sum = 0.0f;
        int valid_count = 0;
        for (const auto& hd : history_height_diffs) {
            if (!std::isnan(hd)) {  // 只计算有效值
                sum += hd;
                valid_count++;
            }
        }

        if (valid_count == 0) {
            return std::numeric_limits<float>::quiet_NaN();  // 所有历史值都是NaN
        }

        return sum / valid_count;
    }

    /**
     * @brief 线性空间RGB颜色融合
     * @param other 另一个栅格单元
     * @param weight_new 新观测的权重(0.0-1.0)
     */
    void fuseRGBLinear(const HeightDiffCell& other, float weight_new = 0.6f) {
        if (!other.has_valid_color) return;

        if (!has_valid_color) {
            // 当前无颜色，直接使用新颜色
            r = other.r;
            g = other.g;
            b = other.b;
            has_valid_color = true;
        } else {
            // 线性空间融合
            auto toLinear = [](uint8_t val) -> float {
                float normalized = val / 255.0f;
                return normalized * normalized;  // 简化的伽马校正
            };

            auto toSRGB = [](float linear) -> uint8_t {
                float val = sqrt(linear) * 255.0f;
                val = std::max(0.0f, std::min(255.0f, val));  // 手动实现clamp
                return static_cast<uint8_t>(val);
            };

            float weight_old = 1.0f - weight_new;

            float r_old_linear = toLinear(r);
            float g_old_linear = toLinear(g);
            float b_old_linear = toLinear(b);

            float r_new_linear = toLinear(other.r);
            float g_new_linear = toLinear(other.g);
            float b_new_linear = toLinear(other.b);

            float r_fused = r_old_linear * weight_old + r_new_linear * weight_new;
            float g_fused = g_old_linear * weight_old + g_new_linear * weight_new;
            float b_fused = b_old_linear * weight_old + b_new_linear * weight_new;

            r = toSRGB(r_fused);
            g = toSRGB(g_fused);
            b = toSRGB(b_fused);
        }
    }

    /**
     * @brief 指数衰减更新(用于未观测区域)
     * @param decay_factor 衰减因子(0.0-1.0)
     */
    void applyDecay(float decay_factor = 0.5f) {
        if (has_data) {
            height_diff *= decay_factor;
            addHistoryData(height_diff);
        }
    }
};

/**
 * @brief 滚动栅格地图模板类，基于ColorPoint项目的CRollingGridMap技术
 * 提供高效的内存管理和实时地图更新能力
 */
template<typename CellType>
class CRollingGridMap_v3 {
public:
    /**
     * @brief 构造函数
     * @param resolution 栅格分辨率(米)
     * @param rows 栅格行数
     * @param cols 栅格列数
     */
    CRollingGridMap_v3(double resolution, int rows, int cols);

    /**
     * @brief 析构函数
     */
    ~CRollingGridMap_v3();

    /**
     * @brief 重新定位地图中心，实现滚动更新
     * @param x 新的中心X坐标(米)
     * @param y 新的中心Y坐标(米)
     * @return 是否发生了地图滚动
     */
    bool reCenter(double x, double y);

    /**
     * @brief 重置整个地图
     */
    void reset();

    /**
     * @brief 根据局部坐标获取栅格单元
     * @param local_x 局部X坐标(米)
     * @param local_y 局部Y坐标(米)
     * @return 栅格单元指针，如果超出范围则返回nullptr
     */
    CellType* getCell(float local_x, float local_y);

    /**
     * @brief 根据局部坐标获取栅格单元，同时返回行列索引
     * @param local_x 局部X坐标(米)
     * @param local_y 局部Y坐标(米)
     * @param r 输出行索引
     * @param c 输出列索引
     * @return 栅格单元指针，如果超出范围则返回nullptr
     */
    CellType* getCell(float local_x, float local_y, int& r, int& c);

    /**
     * @brief 根据行列索引获取栅格单元
     * @param r 行索引
     * @param c 列索引
     * @return 栅格单元指针，如果超出范围则返回nullptr
     */
    CellType* getRCLocal(int r, int c) const;

    /**
     * @brief 根据行列索引获取栅格单元(不进行边界检查)
     * @param r 行索引
     * @param c 列索引
     * @return 栅格单元指针
     */
    CellType* getRCLocalUnsafe(int r, int c) const;

    /**
     * @brief 设置残差偏移量
     * @param lidar_pose_x LiDAR位置X
     * @param lidar_pose_y LiDAR位置Y
     */
    void setResidualFromLidarPose(double lidar_pose_x, double lidar_pose_y);

    /**
     * @brief 直接设置残差偏移量
     * @param residual_x X方向残差
     * @param residual_y Y方向残差
     */
    void setResidual(double residual_x, double residual_y);

    /**
     * @brief 获取地图参数
     */
    double getResolution() const { return resolution_; }
    int getRows() const { return rows_; }
    int getCols() const { return cols_; }
    double getCenterX() const { return center_x_; }
    double getCenterY() const { return center_y_; }
    double getResidualX() const { return residual_x_; }
    double getResidualY() const { return residual_y_; }

private:
    // 地图参数
    double resolution_;              // 栅格分辨率(米)
    int rows_, cols_;               // 栅格行列数
    int map_r0_, map_c0_;          // 地图左下角的栅格坐标
    int array_r0_, array_c0_;      // 数组左上角在内存中的位置
    double center_x_, center_y_;    // 地图中心坐标
    double residual_x_, residual_y_; // 残差偏移量

    // 数据存储
    CellType* cells_;              // 栅格数据数组
    CellType default_value_;       // 默认栅格值

    // 内部方法
    void addColumnEast();          // 向东添加列
    void addColumnWest();          // 向西添加列
    void addRowNorth();            // 向北添加行
    void addRowSouth();            // 向南添加行

    /**
     * @brief 环形索引包装函数
     * @param x 索引值
     * @param max 最大值
     * @return 包装后的索引值[0, max)
     */
    static int wrap(int x, int max) {
        if (x >= max) {
            while (x >= max) x -= max;
        } else if (x < 0) {
            while (x < 0) x += max;
        }
        return x;
    }
};

/**
 * @brief 高级栅格地图处理类v3，结合ColorPoint滚动栅格技术和lidar_map高度差计算
 * 输入：带颜色的LiDAR点云 + LocalPose
 * 输出：栅格高度差图 + 栅格RGB图
 */
class GridMapHandler_v3 {
public:
    /**
     * @brief 构造函数
     */
    GridMapHandler_v3();

    /**
     * @brief 析构函数
     */
    ~GridMapHandler_v3();

    /**
     * @brief 初始化栅格地图
     * @param resolution 栅格分辨率(米)，默认0.2m
     * @param map_size 地图大小(米)，默认100m×100m
     * @param rows 栅格行数，默认500
     * @param cols 栅格列数，默认500
     */
    void initialize(double resolution = 0.2, double map_size = 100.0,
                   int rows = 500, int cols = 500);

    /**
     * @brief 主处理函数：处理带颜色的LiDAR点云数据
     * @param colored_car_points 车体坐标系下的彩色点云
     * @param body_pose 车辆位姿信息(LocalPose)
     */
    void process(const std::vector<PointXYZRGBValid>& colored_car_points,
                const self_state::LocalPose& body_pose);

    /**
     * @brief 获取高度差地图(OpenCV格式)
     * @return 高度差地图，CV_32FC1格式
     */
    cv::Mat getHeightDiffMap() const { return height_diff_map_; }

    /**
     * @brief 获取RGB颜色地图(OpenCV格式)
     * @return RGB颜色地图，CV_8UC3格式
     */
    cv::Mat getRGBMap() const { return rgb_map_; }

    /**
     * @brief 获取数据有效性地图
     * @return 数据有效性地图，CV_8UC1格式
     */
    cv::Mat getValidDataMap() const { return valid_data_map_; }


    /**
     * @brief 重置地图
     */
    void reset();

    /**
     * @brief 获取地图参数
     */
    double getResolution() const { return resolution_; }
    int getRows() const { return rows_; }
    int getCols() const { return cols_; }

private:
    // 核心滚动栅格地图
    CRollingGridMap_v3<HeightDiffCell>* rolling_map_;

    // 地图参数
    double resolution_;              // 栅格分辨率(米)
    double map_size_;               // 地图大小(米)
    int rows_, cols_;               // 栅格行列数

    // 输出地图
    cv::Mat height_diff_map_;       // 高度差地图
    cv::Mat rgb_map_;               // RGB颜色地图
    cv::Mat valid_data_map_;        // 数据有效性地图

    // 处理参数
    double x_max_, y_max_, z_max_;  // 点云过滤范围

    // my_config参数（用于自动显示）
    my_config* my_config_params_;   // my_config参数指针

    // 性能统计
    std::chrono::high_resolution_clock::time_point last_process_time_;

    /**
     * @brief 第零阶段：设置滚动地图中心并重置绝对高度信息
     * 由于LocalPose的z不准确，每帧都需要重置所有包含绝对高度的信息
     * @param body_pose 车辆位姿
     */
    void stage0_Remove_and_Reset(const self_state::LocalPose& body_pose);

    /**
     * @brief 第一阶段：创建临时栅格并进行初始处理
     * @param colored_car_points 彩色点云
     * @param body_pose 车辆位姿
     */
    void stage1_createTempGrids(const std::vector<PointXYZRGBValid>& colored_car_points,
                               const self_state::LocalPose& body_pose);

    /**
     * @brief 第二阶段：邻域地面基准检测
     */
    void stage2_groundDetection();

    /**
     * @brief 第三阶段：障碍物高度差计算
     * @param colored_car_points 彩色点云
     * @param body_pose 车辆位姿
     */
    void stage3_heightDiffCalculation(const std::vector<PointXYZRGBValid>& colored_car_points,
                                     const self_state::LocalPose& body_pose);

    /**
     * @brief 第四阶段：历史地图更新与融合
     */
    void stage4_historyMapUpdate();

    /**
     * @brief 第五阶段：生成输出地图
     * @param body_pose 车辆位姿
     */
    void stage5_generateOutputMaps(const self_state::LocalPose& body_pose);

    /**
     * @brief 坐标变换：车体坐标系到全局坐标系
     * @param point_car 车体坐标系下的点
     * @param body_pose 车辆位姿
     * @param global_x 输出全局X坐标
     * @param global_y 输出全局Y坐标
     */
    void transformCarToGlobal(const PointXYZRGBValid& point_car,
                             const self_state::LocalPose& body_pose,
                             double& global_x, double& global_y);

    /**
     * @brief 全局坐标到栅格索引
     * @param global_x 全局X坐标
     * @param global_y 全局Y坐标
     * @param grid_col 输出栅格列索引
     * @param grid_row 输出栅格行索引
     */
    void globalToGrid(double global_x, double global_y, int& grid_col, int& grid_row);

    /**
     * @brief 点云范围过滤
     * @param point 点云数据
     * @return 是否在有效范围内
     */
    bool isPointInRange(const PointXYZRGBValid& point);
};

// ============================================================================
// CRollingGridMap_v3 模板类实现
// ============================================================================

template<typename CellType>
CRollingGridMap_v3<CellType>::CRollingGridMap_v3(double resolution, int rows, int cols)
    : resolution_(resolution), rows_(rows), cols_(cols),
      map_r0_(0), map_c0_(0), array_r0_(0), array_c0_(0),
      center_x_(0.0), center_y_(0.0), residual_x_(0.0), residual_y_(0.0)
{
    cells_ = new CellType[rows_ * cols_];
    reset();
}

template<typename CellType>
CRollingGridMap_v3<CellType>::~CRollingGridMap_v3()
{
    delete[] cells_;
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::reset()
{
    for (int i = 0; i < rows_ * cols_; i++) {
        cells_[i] = default_value_;
    }
}

template<typename CellType>
bool CRollingGridMap_v3<CellType>::reCenter(double x, double y)
{
    center_x_ = x;
    center_y_ = y;

    int corner_r = static_cast<int>(std::floor(y / resolution_)) - rows_ / 2;
    int corner_c = static_cast<int>(std::floor(x / resolution_)) - cols_ / 2;

    int dr = corner_r - map_r0_;
    int dc = corner_c - map_c0_;

    if (dr == 0 && dc == 0) {
        return false;  // 无需滚动
    }

    if (std::abs(dr) >= rows_ || std::abs(dc) >= cols_) {
        // 移动距离过大，直接重置
        reset();
        map_r0_ = corner_r;
        map_c0_ = corner_c;
        array_r0_ = 0;
        array_c0_ = 0;
    } else {
        // 逐步滚动
        if (dr > 0) {
            for (int i = 0; i < dr; i++) {
                addRowNorth();
            }
        } else if (dr < 0) {
            for (int i = 0; i < std::abs(dr); i++) {
                addRowSouth();
            }
        }

        if (dc > 0) {
            for (int i = 0; i < dc; i++) {
                addColumnEast();
            }
        } else if (dc < 0) {
            for (int i = 0; i < std::abs(dc); i++) {
                addColumnWest();
            }
        }
    }

    return true;
}

template<typename CellType>
CellType* CRollingGridMap_v3<CellType>::getCell(float local_x, float local_y)
{
    int c = cols_ / 2 + static_cast<int>(std::floor((local_x - residual_x_) / resolution_));
    int r = rows_ / 2 - 1 - static_cast<int>(std::floor((local_y - residual_y_) / resolution_));

    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        return nullptr;
    }

    return getRCLocalUnsafe(r, c);
}

template<typename CellType>
CellType* CRollingGridMap_v3<CellType>::getCell(float local_x, float local_y, int& r, int& c)
{
    c = cols_ / 2 + static_cast<int>(std::floor((local_x - residual_x_) / resolution_));
    r = rows_ / 2 - 1 - static_cast<int>(std::floor((local_y - residual_y_) / resolution_));

    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        return nullptr;
    }

    return getRCLocalUnsafe(r, c);
}

template<typename CellType>
CellType* CRollingGridMap_v3<CellType>::getRCLocal(int r, int c) const
{
    if (r < 0 || c < 0 || r >= rows_ || c >= cols_) {
        return nullptr;
    }

    r = wrap(r + array_r0_, rows_);
    c = wrap(c + array_c0_, cols_);
    return &cells_[r * cols_ + c];
}

template<typename CellType>
CellType* CRollingGridMap_v3<CellType>::getRCLocalUnsafe(int r, int c) const
{
    r = wrap(r + array_r0_, rows_);
    c = wrap(c + array_c0_, cols_);
    return &cells_[r * cols_ + c];
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::setResidualFromLidarPose(double lidar_pose_x, double lidar_pose_y)
{
    residual_x_ = std::floor(lidar_pose_x / resolution_) * resolution_ - lidar_pose_x;
    residual_y_ = std::floor(lidar_pose_y / resolution_) * resolution_ - lidar_pose_y;
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::setResidual(double residual_x, double residual_y)
{
    residual_x_ = residual_x;
    residual_y_ = residual_y;
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::addColumnEast()
{
    for (int r = 0; r < rows_; r++) {
        CellType* cell = &cells_[r * cols_ + array_c0_];
        *cell = default_value_;
    }

    map_c0_++;
    array_c0_++;
    if (array_c0_ == cols_) {
        array_c0_ = 0;
    }
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::addColumnWest()
{
    int new_array_c0 = array_c0_ - 1;
    if (new_array_c0 < 0) {
        new_array_c0 = cols_ - 1;
    }

    for (int r = 0; r < rows_; r++) {
        CellType* cell = &cells_[r * cols_ + new_array_c0];
        *cell = default_value_;
    }

    map_c0_--;
    array_c0_ = new_array_c0;
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::addRowSouth()
{
    for (int c = 0; c < cols_; c++) {
        CellType* cell = &cells_[array_r0_ * cols_ + c];
        *cell = default_value_;
    }

    map_r0_--;
    array_r0_++;
    if (array_r0_ == rows_) {
        array_r0_ = 0;
    }
}

template<typename CellType>
void CRollingGridMap_v3<CellType>::addRowNorth()
{
    int new_array_r0 = array_r0_ - 1;
    if (new_array_r0 < 0) {
        new_array_r0 = rows_ - 1;
    }

    for (int c = 0; c < cols_; c++) {
        CellType* cell = &cells_[new_array_r0 * cols_ + c];
        *cell = default_value_;
    }

    map_r0_++;
    array_r0_ = new_array_r0;
}

