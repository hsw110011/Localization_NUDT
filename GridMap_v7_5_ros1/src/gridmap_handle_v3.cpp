#include "map/gridmap_handle_v3.h"

// ============================================================================
// GridMapHandler_v3 类实现
// ============================================================================

GridMapHandler_v3::GridMapHandler_v3()
    : rolling_map_(nullptr), resolution_(0.2), map_size_(100.0),
      rows_(500), cols_(500), x_max_(50.0), y_max_(50.0), z_max_(2.5),
      my_config_params_(nullptr)
{
    // 构造函数中不初始化rolling_map_，等待initialize()调用
}

GridMapHandler_v3::~GridMapHandler_v3()
{
    if (rolling_map_) {
        delete rolling_map_;
        rolling_map_ = nullptr;
    }
}

void GridMapHandler_v3::initialize(double resolution, double map_size, int rows, int cols)
{
    resolution_ = resolution;
    map_size_ = map_size;
    rows_ = rows;
    cols_ = cols;

    // 计算点云过滤范围
    x_max_ = map_size / 2.0;
    y_max_ = map_size / 2.0;
    z_max_ = 2.5;  // 最大高度限制

    // 创建滚动栅格地图
    if (rolling_map_) {
        delete rolling_map_;
    }
    rolling_map_ = new CRollingGridMap_v3<HeightDiffCell>(resolution_, rows_, cols_);

    // 初始化输出地图
    height_diff_map_ = cv::Mat::zeros(rows_, cols_, CV_32FC1);
    rgb_map_ = cv::Mat::zeros(rows_, cols_, CV_8UC3);
    valid_data_map_ = cv::Mat::zeros(rows_, cols_, CV_8UC1);

    // 重置状态完成

    std::cout << "[GridMapHandler_v3] 初始化完成:" << std::endl;
    std::cout << "  - 分辨率: " << resolution_ << "m" << std::endl;
    std::cout << "  - 地图大小: " << map_size_ << "m x " << map_size_ << "m" << std::endl;
    std::cout << "  - 栅格数量: " << rows_ << " x " << cols_ << std::endl;
    std::cout << "  - 点云过滤范围: ±" << x_max_ << "m (x), ±" << y_max_ << "m (y), " << z_max_ << "m (z)" << std::endl;
}

void GridMapHandler_v3::reset()
{
    if (rolling_map_) {
        rolling_map_->reset();
    }

    height_diff_map_.setTo(0);
    rgb_map_.setTo(cv::Scalar(128, 128, 128));  // 默认灰色
    valid_data_map_.setTo(0);

    std::cout << "[GridMapHandler_v3] 地图已重置" << std::endl;
}

void GridMapHandler_v3::process(const std::vector<PointXYZRGBValid>& colored_car_points,
                               const self_state::LocalPose& body_pose)
{
    if (!rolling_map_) {
        std::cerr << "[GridMapHandler_v3] 错误：地图未初始化，请先调用initialize()" << std::endl;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 六阶段处理流程
    stage0_Remove_and_Reset(body_pose);                 // 阶段0：设置地图中心并重置绝对高度信息
    stage1_createTempGrids(colored_car_points, body_pose);  // 阶段1：处理点云数据
    stage2_groundDetection();                               // 阶段2：地面基准检测
    stage3_heightDiffCalculation(colored_car_points, body_pose); // 阶段3：高度差计算
    stage4_historyMapUpdate();                              // 阶段4：历史数据更新
    stage5_generateOutputMaps(body_pose);                   // 阶段5：生成输出地图

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    //     std::cout << "[GridMapHandler_v3] 处理时间: " << duration.count() << " ms" << std::endl;
}

bool GridMapHandler_v3::isPointInRange(const PointXYZRGBValid& point)
{
    return (std::abs(point.x) <= x_max_ &&
            std::abs(point.y) <= y_max_ &&
            point.z <= z_max_ &&
            point.z >= -2.0);  // 最低高度限制
}


void GridMapHandler_v3::globalToGrid(double global_x, double global_y, int& grid_col, int& grid_row)
{
    // 全局坐标到栅格索引的转换
    grid_col = static_cast<int>(std::floor(global_x / resolution_)) + cols_ / 2;
    grid_row = static_cast<int>(std::floor(global_y / resolution_)) + rows_ / 2;
}

void GridMapHandler_v3::stage0_Remove_and_Reset(const self_state::LocalPose& body_pose)
{
    // 第零阶段：设置滚动地图中心并重置绝对高度信息
    // 由于LocalPose的z不准确，每帧都需要重置包含绝对高度的信息

    // 1. 设置滚动地图的中心和残差
    double global_center_x = body_pose.dr_x;
    double global_center_y = body_pose.dr_y;

    rolling_map_->setResidualFromLidarPose(global_center_x, global_center_y);
    rolling_map_->reCenter(global_center_x, global_center_y);

    // 2. 重置所有栅格的绝对高度信息
    #pragma omp parallel for
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            HeightDiffCell* cell = rolling_map_->getRCLocal(r, c);
            if (cell) {
                // 重置绝对高度信息
                cell->min_height = std::numeric_limits<float>::max();
                cell->max_height = -std::numeric_limits<float>::max();
                cell->ground_height = std::numeric_limits<float>::max();
                cell->has_data = false;

                // 保留相对信息：height_diff, RGB颜色, 历史数据等不重置
                // 这些信息可以跨帧使用，因为它们不依赖绝对高度基准
            }
        }
    }
}

void GridMapHandler_v3::stage1_createTempGrids(const std::vector<PointXYZRGBValid>& colored_car_points,
                                              const self_state::LocalPose& body_pose)
{
    // 第一阶段：处理点云数据并填充到栅格中

    // 获取当前车辆在全局坐标系中的位置
    double global_center_x = body_pose.dr_x;
    double global_center_y = body_pose.dr_y;

    for (const auto& point_car : colored_car_points) {
        // 点云范围过滤
        if (!isPointInRange(point_car)) {
            continue;
        }

        // 坐标变换：车体坐标系到旋转坐标系
        double cos_theta = std::cos(body_pose.dr_heading);
        double sin_theta = std::sin(body_pose.dr_heading);
        double rotated_x = point_car.x * cos_theta - point_car.y * sin_theta;
        double rotated_y = point_car.x * sin_theta + point_car.y * cos_theta;

        // 获取对应的栅格单元
        HeightDiffCell* cell = rolling_map_->getCell(rotated_x, rotated_y);
        if (cell) {
            // 直接使用原始点云数据，不进行高度偏移校正
            cell->updateWithPoint(point_car);
        }
    }
}

void GridMapHandler_v3::stage2_groundDetection()
{
    // 第二阶段：邻域地面基准检测
    // 使用3x3邻域的最小高度作为地面基准，无需临时数组

    const int kernel_size = 3;
    const int half_kernel = kernel_size / 2;

    // 直接计算每个栅格的邻域最小高度并设置
    int ground_cells_updated = 0;
    #pragma omp parallel for reduction(+:ground_cells_updated)
    for (int r = half_kernel; r < rows_ - half_kernel; r++) {
        for (int c = half_kernel; c < cols_ - half_kernel; c++) {
            HeightDiffCell* center_cell = rolling_map_->getRCLocal(r, c);
            if (!center_cell || !center_cell->has_data) {
                continue;
            }

            float neighbor_min_height = std::numeric_limits<float>::max();
            bool has_valid_neighbor = false;

            // 遍历3x3邻域，寻找最小高度
            for (int dr = -half_kernel; dr <= half_kernel; dr++) {
                for (int dc = -half_kernel; dc <= half_kernel; dc++) {
                    HeightDiffCell* neighbor_cell = rolling_map_->getRCLocal(r + dr, c + dc);
                    if (neighbor_cell && neighbor_cell->has_data) {
                        neighbor_min_height = std::min(neighbor_min_height, neighbor_cell->min_height);
                        has_valid_neighbor = true;
                    }
                }
            }

            // 直接设置邻域基准地面高度
            if (has_valid_neighbor) {
                center_cell->ground_height = neighbor_min_height;
            }
        }
    }
}

void GridMapHandler_v3::stage3_heightDiffCalculation(const std::vector<PointXYZRGBValid>& colored_car_points,
                                                    const self_state::LocalPose& body_pose)
{
    // 第三阶段：障碍物高度差计算
    // 直接使用栅格数据计算高度差，无需重新遍历点云
    int height_diff_updates = 0;
    #pragma omp parallel for reduction(+:height_diff_updates)
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            HeightDiffCell* cell = rolling_map_->getRCLocal(r, c);
            if (cell && cell->has_data) {

                // 直接用最大高度减去基准地面高度
                cell->height_diff = std::max(0.0f, cell->max_height - cell->ground_height);
            }
        }
    }

}

void GridMapHandler_v3::stage4_historyMapUpdate()
{
    // 第四阶段：历史地图更新与融合
    // 对所有栅格应用时间衰减和历史数据管理

    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            HeightDiffCell* cell = rolling_map_->getRCLocal(r, c);
            if (cell) {
                if (cell->has_data) {
                    // 当前帧有数据的栅格：添加历史数据
                    if (!std::isnan(cell->height_diff)) {
                        cell->addHistoryData(cell->height_diff);
                    }
                } else {
                    // 当前帧无数据的栅格：如果有历史高度差，应用衰减
                    if (!std::isnan(cell->height_diff)) {
                        cell->applyDecay(0.95f);  // 5%衰减
                    }
                }
            }
        }
    }
}

void GridMapHandler_v3::stage5_generateOutputMaps(const self_state::LocalPose& body_pose)
{
    // 第五阶段：生成输出地图
    // 将滚动栅格地图转换为OpenCV格式的输出地图

    // 重置输出地图
    height_diff_map_.setTo(0);
    rgb_map_.setTo(cv::Scalar(0, 0, 0));  // 默认黑色
    valid_data_map_.setTo(0);


    // 遍历所有栅格，生成输出地图
    for (int r = 0; r < rows_; r++) {
        for (int c = 0; c < cols_; c++) {
            HeightDiffCell* cell = rolling_map_->getRCLocal(r, c);
            if (cell) {
                // 使用历史平均高度差
                float avg_height_diff = cell->getAverageHeightDiff();

                // 高度差地图 - 检查height_diff是否有效（非NaN）
                // 参考my_config::setColorByHeight，使用1cm阈值以保持精度
                if (!std::isnan(avg_height_diff)) { 
                    height_diff_map_.at<float>(r, c) = avg_height_diff;
                }

                // RGB颜色地图
                if (cell->has_valid_color) {
                    cv::Vec3b& pixel = rgb_map_.at<cv::Vec3b>(r, c);
                    pixel[0] = cell->b;  // OpenCV是BGR顺序
                    pixel[1] = cell->g;
                    pixel[2] = cell->r;
                }

                // 数据有效性地图
                valid_data_map_.at<uint8_t>(r, c) = 255;
            }
        }
    }


    // 使用my_config的高度差-颜色映射关系显示高度差地图
    params->generateColorMap(height_diff_map_, valid_data_map_, "GridMap_v3_Height");

    // 使用my_config的RGB地图显示
    params->generateRGBMap(rgb_map_, valid_data_map_, "GridMap_v3_RGB");

}
