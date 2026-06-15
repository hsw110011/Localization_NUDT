#include "map/gridmap_handle_v2.h"
#include <grid_map_cv/grid_map_cv.hpp>
#include <geometry_msgs/Pose2D.h>
#include "world_state/TerrainMap.h"
#include "world_state/ColorMap.h"
#include <world_state/StaticObj.h>
#include <world_state/StaticObjArray.h>
#include <tf/tf.h>
#ifdef _OPENMP
#include <omp.h>
#include "map/gridmap_handle_v2.h"
#endif


GridMapHandler_v2::GridMapHandler_v2()
    : map_size_x_(100.0)
    , map_size_y_(100.0)
    , resolution_(0.2)
    , frame_id_("map")
{
    // 自动初始化GridMap，使用构造函数中设置的100×100m
    initialize(map_size_x_, map_size_y_, resolution_, frame_id_);
}

void GridMapHandler_v2::initialize(double map_size_x, double map_size_y,
                               double resolution, const std::string& frame_id) {
    // 保存参数
    map_size_x_ = map_size_x;
    map_size_y_ = map_size_y;
    resolution_ = resolution;
    frame_id_ = frame_id;

    // 计算可视化参数
    img_rows_ = std::max(1, static_cast<int>(std::round(map_size_y_ / resolution_)));
    img_cols_ = std::max(1, static_cast<int>(std::round(map_size_x_ / resolution_)));
    scale_factor_ = img_rows_ / map_size_y_;  // 像素/米
    img_size_ = std::max(img_rows_, img_cols_);

    // 创建GridMap数据层
    std::vector<std::string> layers = {
        "elevation",      // 高度差（障碍物检测用）
        "min_height",     // 最小高度
        "max_height",     // 最大高度
        "ground_height",  // 地面基准高度（3x3邻域最小值）
        "slope",          // 坡度（度数）
        "dynamic_flag",   // 动态物体标志（0=静态，1=动态）
        "rgb_r",          // 红色分量
        "rgb_g",          // 绿色分量
        "rgb_b",          // 蓝色分量
        "visit_count",    // 访问计数
        "has_valid_color", // 颜色有效性标志
        "intrested_flag",    //自车前的区域
        "car_col",        // 车体坐标系列索引
        "car_row",         // 车体坐标系行索引
        "points_num"        // 每个栅格的点云数量
    };

    // 初始化GridMap
    try {
        map_ = grid_map::GridMap(layers);
        map_.setFrameId(frame_id_);
        map_.setGeometry(grid_map::Length(map_size_x_, map_size_y_), resolution_);
    } catch(const std::exception& e) {
        std::cerr << "Error initializing GridMap: " << e.what() << std::endl;
        throw;
    }

    // 初始化各层数据
    map_["elevation"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["min_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["max_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["ground_height"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["slope"].setConstant(std::numeric_limits<float>::quiet_NaN());
    map_["dynamic_flag"].setConstant(0.0f);  // 默认为静态
    map_["rgb_r"].setConstant(0.0f);// 默认灰色
    map_["rgb_g"].setConstant(0.0f);
    map_["rgb_b"].setConstant(0.0f);
    map_["visit_count"].setConstant(0);
    map_["has_valid_color"].setConstant(0.0f);  // 0表示无有效颜色
    map_["intrested_flag"].setConstant(0);  // 默认为无
    map_["car_col"].setConstant(-1);  // 车体坐标系列索引，-1表示无效
    map_["car_row"].setConstant(-1);  // 车体坐标系行索引，-1表示无效
    map_["points_num"].setConstant(0);  // 默认为0个点
    

    // std::cout << "GridMapHandler_v2 initialized: "
    //           << map_size_x_ << "x" << map_size_y_ << "m, "
    //           << "resolution=" << resolution_ << "m, "
    //           << "size=" << map_.getSize().transpose() << " cells, "
    //           << "img_size=" << img_rows_ << "x" << img_cols_ << " pixels" << std::endl;
}

void GridMapHandler_v2::gridmap_process(const std::vector<PointXYZRGBValid>& colored_car_points,
                            const self_state::LocalPose& body_pose) {
    auto start_time0 = std::chrono::high_resolution_clock::now();


    // 安全检查：确保GridMap已初始化
    if(map_.getLayers().empty()) {
        std::cerr << "Error: GridMap not initialized! Call initialize() first." << std::endl;
        return;
    }    

    // 步骤1：更新地图位置（每次都移动，无阈值）
    auto start_time1 = std::chrono::high_resolution_clock::now();
    updateMapPosition(body_pose);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1);
    // std::cout << "updateMapPosition duration: " << duration1.count() << " ms" << std::endl;

    // 步骤2：批量处理点云数据
    auto start_time2 = std::chrono::high_resolution_clock::now();
    processPointCloud(colored_car_points, body_pose);
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    // std::cout << "processPointCloud duration: " << duration2.count() << " ms" << std::endl;

    // 步骤3：寻找地面基准（3x3邻域）
    auto start_time3 = std::chrono::high_resolution_clock::now();
    find_ground();
    // find_ground_v2();
    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
    // std::cout << "find_ground duration: " << duration3.count() << " ms" << std::endl;

    // 步骤4：计算高度差
    auto start_time4 = std::chrono::high_resolution_clock::now();
    get_HeightDiff();
    auto end_time4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4 - start_time4);
    // std::cout << "get_HeightDiff duration: " << duration4.count() << " ms" << std::endl;

    // 步骤5：计算高度差v2
    auto start_time5 = std::chrono::high_resolution_clock::now();
    get_HeightDiff_v2();
    auto end_time5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5);
    // std::cout << "get_HeightDiff_v2 duration: " << duration5.count() << " ms" << std::endl;

    // 步骤5：处理动态物体邻域
    // auto start_time5 = std::chrono::high_resolution_clock::now();
    // processDynamicObjects();
    // auto end_time5 = std::chrono::high_resolution_clock::now();
    // auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5);
    // std::cout << "processDynamicObjects duration: " << duration5.count() << " ms" << std::endl;

    // 步骤7：填补孔洞
    // auto start_time7 = std::chrono::high_resolution_clock::now();
    // fillHoles();
    // auto end_time7 = std::chrono::high_resolution_clock::now();
    // auto duration7 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time7 - start_time7);
    // std::cout << "fillHoles duration: " << duration7.count() << " ms" << std::endl;

    // 步骤8：计算坡度
    auto start_time8 = std::chrono::high_resolution_clock::now();
    calculateSlope();
    auto end_time8 = std::chrono::high_resolution_clock::now();
    auto duration8 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time8 - start_time8);
    // std::cout << "calculateSlope duration: " << duration8.count() << " ms" << std::endl;

    // 步骤9：处理远距离区域（50-100m）
    auto start_time9 = std::chrono::high_resolution_clock::now();
    cv::Mat far_range_height_diff = processFarRangePoints(colored_car_points);
    auto end_time9 = std::chrono::high_resolution_clock::now();
    auto duration9 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time9 - start_time9);
    // std::cout << "processFarRangePoints duration: " << duration9.count() << " ms" << std::endl;

    // 步骤10：远距离障碍物检测已移至obstacle_detection统一处理

    // 最后，显示高差图和RGB图（以自车为坐标系）
    // auto start_time9 = std::chrono::high_resolution_clock::now();
    // showMaps(body_pose.dr_heading);  // 传入车辆朝向
    // auto end_time9 = std::chrono::high_resolution_clock::now();
    // auto duration9 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time9 - start_time9);
    // std::cout << "showMaps duration: " << duration9.count() << " ms" << std::endl;

    // 使用opencv可视化
    auto start_time10 = std::chrono::high_resolution_clock::now();
    opencv_showMaps(body_pose.dr_heading);
    auto end_time10 = std::chrono::high_resolution_clock::now();
    auto duration10 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time10 - start_time10);
    // std::cout << "opencv_showMaps duration: " << duration10.count() << " ms" << std::endl;

    // 统计障碍并发布
    obstacle_detection(vehicle_height_mat, far_range_height_diff);

    // 重置绝对高度相关的层（为下一帧准备）
    resetAbsoluteHeightLayers();

    auto end_time0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time0 - start_time0);
    std::cout << "gridmap_process duration: " << duration0.count() << " ms" << std::endl;
}

void GridMapHandler_v2::updateMapPosition(const self_state::LocalPose& body_pose) {
    // 每次都将地图中心移动到当前机器人位置
    grid_map::Position robot_position(body_pose.dr_x, body_pose.dr_y);

    // 使用move()方法，自动处理数据保留和清理
    map_.move(robot_position);

}


void GridMapHandler_v2::processPointCloud(const std::vector<PointXYZRGBValid>& points,
                                      const self_state::LocalPose& body_pose) {
    if(points.empty()) return;

    // 预计算变换参数
    const double cos_theta = cos(body_pose.dr_heading);
    const double sin_theta = sin(body_pose.dr_heading);
    const double dx = body_pose.dr_x;
    const double dy = body_pose.dr_y;
    double tan_60=tan(PI/3);

    // 获取数据层引用，避免重复查找
    auto& min_height = map_["min_height"];      // 最小高度层
    auto& max_height = map_["max_height"];      // 最大高度层
    auto& rgb_r = map_["rgb_r"];
    auto& rgb_g = map_["rgb_g"];
    auto& rgb_b = map_["rgb_b"];
    auto& has_valid_color = map_["has_valid_color"];
    auto& intrested_flag = map_["intrested_flag"];
    auto& car_col = map_["car_col"];
    auto& car_row = map_["car_row"];
    const grid_map::Size map_size = map_.getSize();
    auto& points_num = map_["points_num"];

    uint8_t kernal_size = 5;
    uint8_t half_kernal = kernal_size / 2;

    // 直接处理每个点，更新栅格的最大最小值等信息
    for(const auto& point : points) {
        // 车体坐标范围过滤（前后左右50m，总共100×100m）
        const double half_size_x = map_size_x_ / 2.0;  // 50m
        const double half_size_y = map_size_y_ / 2.0;  // 50m
        // 注意开区间
        if(point.x <= -half_size_x || point.x >= half_size_x ||
           point.y <= -half_size_y || point.y >= half_size_y ||
           (params->b_enable_point_z_filter && point.z > params->point_z_max)) {
            continue;
        }

        int point_car_col = static_cast<int>(std::floor((half_size_y-point.y) / resolution_));
        int point_car_row = static_cast<int>(std::floor((half_size_x-point.x) / resolution_));

        // 车体坐标 → 全局坐标
        double world_x = cos_theta * point.x - sin_theta * point.y + dx;
        double world_y = sin_theta * point.x + cos_theta * point.y + dy;

        // 全局坐标 → GridMap索引
        grid_map::Position world_pos(world_x, world_y);
        grid_map::Index index;

        if(map_.getIndex(world_pos, index)) {
            const int row = index(0);
            const int col = index(1);

            //计算车体栅格索引
            car_col(row, col) = point_car_col;
            car_row(row, col) = point_car_row;

            // 计算点云数量
            points_num(row, col) += 1;

            // if(point.x > 4){
            //     if(abs(point.y) < (point.x - 4) * tan_60 && abs(point.y) < 15){
                    intrested_flag(row, col) = 1; //标记为感兴趣区域
                    for(int dr = -half_kernal; dr <= half_kernal; ++dr) {
                        for(int dc = -half_kernal; dc <= half_kernal; ++dc) {
                            int neighbor_row = row + dr;
                            int neighbor_col = col + dc;

                            // 检查邻域栅格是否在地图范围内
                            if(neighbor_row >= 0 && neighbor_row < map_size(0) &&
                                neighbor_col >= 0 && neighbor_col < map_size(1)) {
                                intrested_flag(neighbor_row, neighbor_col) = 1;
                            }
                        }
                    }
            //     }
            // }
            
            // 直接更新栅格的最大最小高度
            if(std::isnan(min_height(row, col))) {
                // 第一次访问该栅格，初始化
                min_height(row, col) = point.z;
                max_height(row, col) = point.z;

                // 使用第一个点的RGB信息
                if(point.has_rgb) {
                    rgb_r(row, col) = static_cast<float>(point.r);
                    rgb_g(row, col) = static_cast<float>(point.g);
                    rgb_b(row, col) = static_cast<float>(point.b);
                    has_valid_color(row, col) = 1.0f;
                }
            } else {
                // 更新最高点和最低点
                if(point.z > max_height(row, col)) {
                    max_height(row, col) = point.z;

                    // 使用最高点的RGB信息（最高点策略）
                    if(point.has_rgb) {
                        rgb_r(row, col) = static_cast<float>(point.r);
                        rgb_g(row, col) = static_cast<float>(point.g);
                        rgb_b(row, col) = static_cast<float>(point.b);
                        has_valid_color(row, col) = 1.0f;
                    }
                }

                if(point.z < min_height(row, col)) {
                    min_height(row, col) = point.z;
                }
            }
        }
    }

    // 融合历史信息的intrested_flag
    temp_intrested_flag = intrested_flag;
    if(history_intrested_flag.size() != 0){
        intrested_flag = intrested_flag + history_intrested_flag;
    }
    history_intrested_flag = temp_intrested_flag;
}

void GridMapHandler_v2::find_ground() {
    // 参考原始lidar_map.cpp的逻辑：遍历3x3邻域寻找地面基准
    auto& min_height = map_["min_height"];
    auto& ground_height = map_["ground_height"];

    // 重新初始化地面基准层（清除上一帧的数据）
    ground_height.setConstant(std::numeric_limits<float>::quiet_NaN());
    const uint8_t kernel_size = 3;
    const uint8_t half_kernel = kernel_size / 2;  // 3x3邻域，半径为1
    const grid_map::Size map_size = map_.getSize();

    // 遍历所有栅格
    for(int row = 0; row < map_size(0); ++row) {
        for(int col = 0; col < map_size(1); ++col) {
            // 只处理有数据的栅格
            if(std::isnan(min_height(row, col))) {
                continue;
            }

            float neighbor_min = min_height(row, col);  // 初始化为当前栅格的最小高度

            // 在3x3邻域中查找最小高度作为地面基准
            for(int dr = -half_kernel; dr <= half_kernel; ++dr) {
                for(int dc = -half_kernel; dc <= half_kernel; ++dc) {
                    int neighbor_row = row + dr;
                    int neighbor_col = col + dc;

                    // 检查邻域栅格是否在地图范围内
                    if(neighbor_row >= 0 && neighbor_row < map_size(0) &&
                       neighbor_col >= 0 && neighbor_col < map_size(1)) {

                        if(!std::isnan(min_height(neighbor_row, neighbor_col))) {
                            neighbor_min = std::min(neighbor_min, static_cast<float>(min_height(neighbor_row, neighbor_col)));
                        }
                    }
                }
            }

            // 设置地面基准高度
            ground_height(row, col) = neighbor_min;
        }
    }
}

void GridMapHandler_v2::find_ground_v2() {
    // 参考原始lidar_map.cpp的逻辑：遍历3x3邻域寻找地面基准
    auto& max_height = map_["max_height"];
    auto& min_height = map_["min_height"];
    auto& ground_height = map_["ground_height"];

    // 重新初始化地面基准层（清除上一帧的数据）
    ground_height.setConstant(std::numeric_limits<float>::quiet_NaN());
    const uint8_t kernel_size = 5;
    const uint8_t half_kernel = kernel_size / 2;  // 3x3邻域，半径为1
    const uint8_t min_valid_neighbors = kernel_size * (kernel_size - 1);  // 3×2=6
    const grid_map::Size map_size = map_.getSize();

    // 遍历所有栅格
    for(int row = 0; row < map_size(0); ++row) {
        for(int col = 0; col < map_size(1); ++col) {
            uint8_t valid_count = 0;

            float neighbor_min = min_height(row, col);  // 初始化为当前栅格的最小高度

            // 在3x3邻域中查找最小高度作为地面基准
            for(int dr = -half_kernel; dr <= half_kernel; ++dr) {
                for(int dc = -half_kernel; dc <= half_kernel; ++dc) {
                    int neighbor_row = row + dr;
                    int neighbor_col = col + dc;

                    // 检查邻域栅格是否在地图范围内
                    if(neighbor_row >= 0 && neighbor_row < map_size(0) &&
                       neighbor_col >= 0 && neighbor_col < map_size(1)) {

                        if(!std::isnan(min_height(neighbor_row, neighbor_col))) {
                            valid_count++;
                            neighbor_min = std::min(neighbor_min, static_cast<float>(min_height(neighbor_row, neighbor_col)));
                        }
                    }
                }
            }

            // 设置地面基准高度
            if(!std::isnan(min_height(row, col))) {
                // 该栅格有数据
                ground_height(row, col) = neighbor_min;
            } else {
                // 该栅格无数据
                // if(valid_count >= min_valid_neighbors) {
                    // 该栅格邻域有数据
                    ground_height(row, col) = neighbor_min;
                    max_height(row, col) = neighbor_min + ground_height_value;
                // }
            }
        }
    }
}

void GridMapHandler_v2::get_HeightDiff() {
    // 计算高度差：最高点减去地面基准，并与历史数据进行权重融合
    // 同时处理RGB数据的权重融合
    auto& elevation = map_["elevation"];
    auto& max_height = map_["max_height"];
    auto& ground_height = map_["ground_height"];
    auto& visit_count = map_["visit_count"];
    auto& dynamic_flag = map_["dynamic_flag"];
    auto& rgb_r = map_["rgb_r"];
    auto& rgb_g = map_["rgb_g"];
    auto& rgb_b = map_["rgb_b"];
    auto& has_valid_color = map_["has_valid_color"];
    auto& intrested_flag = map_["intrested_flag"];

    const grid_map::Size map_size = map_.getSize();
    const uint8_t kernel_size = 3;
    const uint8_t half_kernel = kernel_size / 2;

    // 遍历所有栅格
    for(int row = 0; row < map_size(0); ++row) {
        for(int col = 0; col < map_size(1); ++col) {

            // 当前栅格有值
            if(!std::isnan(ground_height(row, col))) {
                // 计算当前帧的高度差：最高点减去地面基准
                float current_height_diff = max_height(row, col) - ground_height(row, col);
                // 历史栅格没值
                if(std::isnan(elevation(row, col))) {
                    // 第一次访问该栅格，直接赋值
                    elevation(row, col) = current_height_diff;
                    visit_count(row, col) = 1;
                    // RGB数据已经在processPointCloud中设置，无需额外处理
                } else {
                    // 历史栅格有值，进行权重融合
                    float historical_height_diff = elevation(row, col);
                    uint16_t current_visit_count = visit_count(row, col);

                    // 检测动态物体：只有观测次数大于5次才执行判断
                    float height_change = historical_height_diff - current_height_diff;
                    if(current_visit_count > min_observations && height_change > dynamic_threshold) {
                        // 检测到动态物体，设置标志和高度
                        // elevation(row, col) = ground_height_value;
                        if(intrested_flag(row, col) == 1){
                            // float fused_height_diff = current_height_diff;
                            elevation(row, col) = current_height_diff;
                            dynamic_flag(row, col) = 1.0f;  // 标记为动态物体
                        }
                        
                        // test，把5×5邻域都标记为动态物体，使用当前观测值
                        // for(int dr = -2; dr <= 2; ++dr) {
                        //     for(int dc = -2; dc <= 2; ++dc) {
                        //         int neighbor_row = row + dr;
                        //         int neighbor_col = col + dc;

                        //         // 检查邻域栅格是否在地图范围内
                        //         if(neighbor_row >= 0 && neighbor_row < map_size(0) &&
                        //            neighbor_col >= 0 && neighbor_col < map_size(1)) {
                        //             elevation(neighbor_row, neighbor_col) = current_height_diff;
                        //             dynamic_flag(neighbor_row, neighbor_col) = 1.0f;
                        //         }
                        //     }
                        // }
                    }
                    else {
                        // 正常的静态环境，进行权重融合
                        float fused_height_diff = (1.0f - new_weight) * historical_height_diff + new_weight * current_height_diff;
                        elevation(row, col) = fused_height_diff;
                    }
                    // 更新访问计数
                    visit_count(row, col) += 1;
                }
            } else{
                // 方法一 当前栅格没值，历史栅格有值，进行衰减
                // if(!std::isnan(elevation(row, col))) {
                //     if(intrested_flag(row, col) == 1){
                //         elevation(row, col) *= 0.5f;  
                //     }
                // }
            }
        }
    }
}

void GridMapHandler_v2::get_HeightDiff_v2()
{
    auto& elevation = map_["elevation"];
    auto& max_height = map_["max_height"];
    auto& ground_height = map_["ground_height"];
    auto& visit_count = map_["visit_count"];
    auto& dynamic_flag = map_["dynamic_flag"];
    auto& rgb_r = map_["rgb_r"];
    auto& rgb_g = map_["rgb_g"];
    auto& rgb_b = map_["rgb_b"];
    auto& has_valid_color = map_["has_valid_color"];
    auto& intrested_flag = map_["intrested_flag"];

    const grid_map::Size map_size = map_.getSize();
    const uint8_t kernel_size = 5;
    const uint8_t half_kernel = kernel_size / 2;

    // 遍历所有栅格
    for(int row = 0; row < map_size(0); ++row) {
        for(int col = 0; col < map_size(1); ++col) {

            // 当前栅格没值
            if(std::isnan(ground_height(row, col))) {
                // 方法二 当前栅格没值，历史栅格有值，根据附近当前有值栅格进行映射
                if (!std::isnan(elevation(row, col)))
                {
                    // if (intrested_flag(row, col) == 1)
                    // {
                        float neighbor_elevation_min = std::numeric_limits<float>::max();  // 修复：初始化为最大值
                        uint8_t valid_count = 0;
                        for (int dr = -half_kernel; dr <= half_kernel; ++dr)
                        {
                            for (int dc = -half_kernel; dc <= half_kernel; ++dc)
                            {
                                int neighbor_row = row + dr;
                                int neighbor_col = col + dc;

                                // 检查邻域栅格是否在地图范围内
                                if (neighbor_row >= 0 && neighbor_row < map_size(0) &&
                                    neighbor_col >= 0 && neighbor_col < map_size(1))
                                {
                                    // 修复：检查邻域栅格是否存在当前值
                                    if (!std::isnan(ground_height(neighbor_row, neighbor_col)))
                                    {
                                        if(!std::isnan(elevation(neighbor_row, neighbor_col))){
                                            //使用邻域内的最小高度差
                                            neighbor_elevation_min = std::min(neighbor_elevation_min, elevation(neighbor_row, neighbor_col));
                                            valid_count++;
                                        }
                                    }
                                }
                            }
                        }
                        if (valid_count > 0)
                        {
                            // 方案1：保持累积效果 - 融合历史值和邻域最小值
                            elevation(row, col) = elevation(row, col) * 0.5f + 0.5f * neighbor_elevation_min;

                            // 方案2：直接使用最小值， 有黑洞
                            // elevation(row, col) = neighbor_elevation_min;
                        } else {
                            // elevation(row, col) *= 0.8f;
                        }
                        
                    // }
                }
            }
        }
    }
}

void GridMapHandler_v2::processDynamicObjects() {
    // 基于邻域动态栅格密度的传播机制：
    // 如果某个栅格不是动态的，但邻域内动态栅格数大于阈值，就认为这个栅格也是动态的
    const int kernel_size = 3;  // 邻域大小（5x5）
    const int half_kernel = kernel_size / 2;  // 邻域半径
    const int min_dynamic_neighbors = 5;  // 最小动态邻居数阈值

    auto &elevation = map_["elevation"];
    auto &dynamic_flag = map_["dynamic_flag"];
    const grid_map::Size map_size = map_.getSize();

    // 创建临时标记数组，避免在遍历过程中修改原数据影响判断
    grid_map::Matrix temp_dynamic_flag = dynamic_flag;
    int newly_marked_count = 0;

    // 遍历所有栅格，检查非动态栅格的邻域
    for (grid_map::GridMapIterator iterator(map_); !iterator.isPastEnd(); ++iterator)
    {
        const grid_map::Index index(*iterator);
        const int row = index(0);
        const int col = index(1);

        // 只处理当前不是动态物体的栅格
        if (dynamic_flag(row, col) <= 0.5f)
        {
            uint8_t dynamic_neighbor_count = 0;

            // 遍历n×n邻域
            for (int dr = -half_kernel; dr <= half_kernel; ++dr)
            {
                for (int dc = -half_kernel; dc <= half_kernel; ++dc)
                {
                    int neighbor_row = row + dr;
                    int neighbor_col = col + dc;

                    // 边界检查
                    if (neighbor_row >= 0 && neighbor_row < map_size(0) &&
                        neighbor_col >= 0 && neighbor_col < map_size(1))
                    {
                        // 检查邻居是否为动态物体
                        if (dynamic_flag(neighbor_row, neighbor_col) > 0.5f)
                        {
                            dynamic_neighbor_count++;
                        }
                    }
                }
            }

            // 如果动态邻居数量超过阈值，将当前栅格也标记为动态
            if (dynamic_neighbor_count >= min_dynamic_neighbors)
            {
                temp_dynamic_flag(row, col) = 1.0f;
                newly_marked_count++;
            }
        }
    }

    // 将临时结果应用到原数据，并设置对应的高度值
    for (grid_map::GridMapIterator iterator(map_); !iterator.isPastEnd(); ++iterator)
    {
        const grid_map::Index index(*iterator);
        const int row = index(0);
        const int col = index(1);

        // 如果栅格被新标记为动态物体或原本就是动态物体
        if (temp_dynamic_flag(row, col) > 0.5f)
        {
            dynamic_flag(row, col) = 1.0f;
            elevation(row, col) = ground_height_value;  // 设为地面最低高度
        }
    }

    if (newly_marked_count > 0)
    {
        std::cout << "Dynamic propagation: marked " << newly_marked_count
                  << " additional cells as dynamic based on neighborhood density" << std::endl;
    }
}

void GridMapHandler_v2::fillHoles() {
    // 使用3×3邻域手动实现的高效孔洞填充
    // 防止边缘扩张：对于3×3邻域，需要大于3×(3-1)=6个有效邻居
    const int kernel_size = 3;  // 邻域大小
    const int half_kernel = kernel_size / 2;  // 邻域半径
    const int min_valid_neighbors = kernel_size * (kernel_size - 1);  // 3×2=6
    const int max_neighbors = kernel_size * kernel_size; // 最大邻居数

    auto &elevation = map_["elevation"];
    const grid_map::Size map_size = map_.getSize();

    // 遍历所有栅格，寻找需要填充的孔洞
    for (grid_map::GridMapIterator iterator(map_); !iterator.isPastEnd(); ++iterator)
    {
        const grid_map::Index index(*iterator);
        const int row = index(0);
        const int col = index(1);

        // 只处理NaN值（孔洞）
        if (std::isnan(elevation(row, col)))
        {
            float sum_elevation = 0.0f;
            uint8_t valid_count = 0;

            // n×n邻域循环
            for (int dr = -half_kernel; dr <= half_kernel; ++dr)
            {
                for (int dc = -half_kernel; dc <= half_kernel; ++dc)
                {
                    int neighbor_row = row + dr;
                    int neighbor_col = col + dc;

                    // 边界检查
                    if (neighbor_row >= 0 && neighbor_row < map_size(0) &&
                        neighbor_col >= 0 && neighbor_col < map_size(1))
                    {
                        float neighbor_value = elevation(neighbor_row, neighbor_col);
                        if (!std::isnan(neighbor_value))
                        {
                            sum_elevation += neighbor_value;
                            valid_count++;
                        }
                    }
                }
            }

            // 只有被足够多有效数据包围的孔洞才填充
            if (valid_count > min_valid_neighbors)
            {
                // 计算平均值
                elevation(row, col) = sum_elevation / valid_count;
            }
        }
    }
}

void GridMapHandler_v2::calculateSlope()
{
    // 基于邻域梯度计算坡度

    auto &ground_height = map_["ground_height"];
    auto &slope = map_["slope"];
    const grid_map::Size map_size = map_.getSize();

    // 遍历所有栅格计算坡度
    for (grid_map::GridMapIterator iterator(map_); !iterator.isPastEnd(); ++iterator)
    {
        const grid_map::Index index(*iterator);
        const int row = index(0);
        const int col = index(1);

        // 只处理有有效高度差的栅格
        if (std::isnan(ground_height(row, col)))
        {
            continue;
        }

        // 计算X方向梯度（左右相邻栅格）
        float grad_x = 0.0f;
        bool has_grad_x = false;
        if (col > 0 && col < map_size(1) - 1)
        {
            if (!std::isnan(ground_height(row, col - 1)) && !std::isnan(ground_height(row, col + 1)))
            {
                grad_x = (ground_height(row, col + 1) - ground_height(row, col - 1)) / (2.0f * resolution_);
                has_grad_x = true;
            }
        }

        // 计算Y方向梯度（上下相邻栅格）
        float grad_y = 0.0f;
        bool has_grad_y = false;
        if (row > 0 && row < map_size(0) - 1)
        {
            if (!std::isnan(ground_height(row - 1, col)) && !std::isnan(ground_height(row + 1, col)))
            {
                grad_y = (ground_height(row + 1, col) - ground_height(row - 1, col)) / (2.0f * resolution_);
                has_grad_y = true;
            }
        }

        float slope_angle = 0.0f;

        // 计算坡度角（度数）
        if (has_grad_x && has_grad_y)
        {
            float gradient_magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
            slope_angle = atan(gradient_magnitude) * 180.0f / M_PI; // 弧度转换为度数
        }
        else if (has_grad_x)
        {
            // 只有X方向梯度
            slope_angle = atan(std::abs(grad_x)) * 180.0f / M_PI; // 弧度转换为度数
        }
        else if (has_grad_y)
        {
            // 只有Y方向梯度
            slope_angle = atan(std::abs(grad_y)) * 180.0f / M_PI; // 弧度转换为度数
        }

        // 更新坡度值
        if(std::isnan(slope(row, col))){
            slope(row, col) = slope_angle;
        }
        else{
            slope(row, col) = slope(row, col) * (1 - new_weight) + new_weight * slope_angle;
        }
    }
}

void GridMapHandler_v2::resetAbsoluteHeightLayers() {
    // 重置所有与绝对高度相关的层
    // 这些层的数据只能在单帧内使用，因为车辆Z轴位置未补偿

    auto& min_height = map_["min_height"];
    auto& max_height = map_["max_height"];
    auto& ground_height = map_["ground_height"];
    // auto& slope = map_["slope"];
    auto& dynamic_flag = map_["dynamic_flag"];
    auto& intrested_flag = map_["intrested_flag"];
    auto& points_num = map_["points_num"];

    // 将这些层重置为NaN，为下一帧准备
    min_height.setConstant(std::numeric_limits<float>::quiet_NaN());
    max_height.setConstant(std::numeric_limits<float>::quiet_NaN());
    ground_height.setConstant(std::numeric_limits<float>::quiet_NaN());
    // slope.setConstant(std::numeric_limits<float>::quiet_NaN());
    dynamic_flag.setConstant(0.0f);  // 重置动态物体标志
    intrested_flag.setConstant(0);  // 重置感兴趣区域标志
    points_num.setConstant(0);  // 重置点云数量
    

    // 注意：elevation层保留，因为它存储的是高度差（相对值），可以跨帧使用
    // 注意：RGB相关层保留，因为颜色信息不受Z轴颠簸影响
}

void GridMapHandler_v2::publishGridMap(ros::Publisher& publisher) {
    if(publisher.getNumSubscribers() > 0) {
        grid_map_msgs::GridMap message;
        grid_map::GridMapRosConverter::toMessage(map_, message);
        publisher.publish(message);
    }
}


void GridMapHandler_v2::publishTerrainMap(
    const ros::Publisher &publisher,
    const geometry_msgs::Pose &local_pose)
{
    if (!publisher) return;

    world_state::TerrainMap msg;

    auto now = ros::Time::now();
    msg.local_time  = now.toNSec() / 1e6;  // ms
    msg.UTC_time    = now.toNSec() / 1e6;
    msg.message_num = 0;

    msg.localPoseStamped.x = local_pose.position.x;
    msg.localPoseStamped.y = local_pose.position.y;
    msg.localPoseStamped.theta = tf::getYaw(local_pose.orientation) * 180.0 / M_PI;

    // 优化：实时计算车体坐标系变换，避免图像旋转操作
    auto& elev_layer = map_["elevation"];
    auto& slope_layer = map_["slope"];
    const grid_map::Size map_size = map_.getSize();

    int rows = map_size(0);
    int cols = map_size(1);

    // 初始化车体坐标系的数据数组（500x500）
    cv::Mat vehicle_roughness = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);
    cv::Mat vehicle_slope = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);
    cv::Mat vehicle_label = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);

    // 获取当前发布时的车辆位姿
    double current_yaw = tf::getYaw(local_pose.orientation);
    double cos_theta = cos(current_yaw);
    double sin_theta = sin(current_yaw);

    // 获取地图中心位置
    grid_map::Position map_center = map_.getPosition();

    // 预计算常量，避免重复计算
    const double half_rows = img_rows_ / 2.0;
    const double half_cols = img_cols_ / 2.0;
    const double map_center_x = map_center.x();
    const double map_center_y = map_center.y();
    float roughness_scale = static_cast<float>(params->terrain_roughness_scale);
    if (roughness_scale <= 1e-6f) {
        roughness_scale = 1e-6f;
    }

    // 遍历车体坐标系图像的每个像素
    for (int img_row = 0; img_row < img_rows_; ++img_row) {
        // 预计算行相关的值
        const double car_x = -(img_row - half_rows) * resolution_;
        const double cos_car_x = cos_theta * car_x;
        const double sin_car_x = sin_theta * car_x;

        for (int img_col = 0; img_col < img_cols_; ++img_col) {

            // 图像坐标 → 车体坐标
            const double car_y = -(img_col - half_cols) * resolution_;

            // 车体坐标 → 世界坐标（优化：减少重复计算）
            const double world_x = cos_car_x - sin_theta * car_y + map_center_x;
            const double world_y = sin_car_x + cos_theta * car_y + map_center_y;

            // 世界坐标 → GridMap索引
            grid_map::Position world_pos(world_x, world_y);
            grid_map::Index grid_index;

            if(map_.getIndex(world_pos, grid_index)) {
                const int row = grid_index(0);
                const int col = grid_index(1);

                // 检查是否有有效数据
                if (!std::isnan(elev_layer(row, col))) {
                    float elev = elev_layer(row, col);
                    float slope_val = slope_layer(row, col);

                    // 计算roughness值（归一化到0-255）
                    float diff = elev;
                    float normalized_diff = std::clamp(diff / roughness_scale, 0.0f, 1.0f);
                    uint8_t roughness_val = static_cast<uint8_t>(std::round(normalized_diff * 255.0f));

                    // 计算slope值
                    float slope_degree = std::clamp(slope_val, 0.0f, 25.5f);
                    uint8_t slope_uint8 = static_cast<uint8_t>(std::round(slope_degree * 10.0f));

                    // 根据高程差进行标签分类（移除负障碍和悬挂障碍物）
                    // uint8_t label_class = 0;  // 默认为unknown

                    // if (std::isnan(diff)) {
                    //     label_class = 0;  // unknown - 无数据
                    // } else if (diff <= 0.05f) {
                    //     label_class = 1;  // terrain - 可通行地形（包含轻微负高程差）
                    // } else if (diff <= 0.3f) {
                    //     label_class = 2;  // gray - 中等高程差
                    // } else if (diff <= 1.0f) {
                    //     label_class = 3;  // obstacle - 障碍物
                    // } else {
                    //     label_class = 6;  // semantic_edge - 语义边缘/极高障碍
                    // }

                    // // 创建组合标签：高4位是roughness（降低分辨率），低4位是label
                    // uint8_t roughness_4bit = roughness_val >> 4;  // 将8位roughness压缩到4位
                    // uint8_t label_val = (roughness_4bit << 4) | (label_class & 0x0F);

                    // 直接填充到车体坐标系位置
                    vehicle_roughness.at<uint8_t>(img_row, img_col) = roughness_val;
                    vehicle_slope.at<uint8_t>(img_row, img_col) = slope_uint8;
                    // vehicle_label.at<uint8_t>(img_row, img_col) = label_val;
                }
            }
        }
    }

    // 直接使用车体坐标系数据填充消息，无需旋转
    for (int x = 0; x < img_rows_; ++x) {
        for (int y = 0; y < img_cols_; ++y) {
            int idx = x * img_cols_ + y;
            if (idx >= 250000) continue;
            msg.roughness[idx] = vehicle_roughness.at<uint8_t>(x, y);
            msg.slope[idx] = vehicle_slope.at<uint8_t>(x, y);
            msg.label[idx] = vehicle_label.at<uint8_t>(x, y);
        }
    }

    publisher.publish(msg);

    // 可视化显示（仅在需要显示时才进行计算）
    bool need_any_display = params->b_show_terrain_roughness ||
                           params->b_show_terrain_slope ||
                           params->b_show_terrain_labels;

    if (need_any_display) {
        // 只有在需要显示时才创建可视化图像
        if(params->b_show_terrain_roughness) {
            cv::Mat rough_color;
            cv::applyColorMap(vehicle_roughness, rough_color, cv::COLORMAP_JET);
            cv::imshow("Published Terrain Roughness", rough_color);
        }

        if(params->b_show_terrain_slope) {
            cv::Mat slope_color;
            cv::applyColorMap(vehicle_slope, slope_color, cv::COLORMAP_JET);
            cv::imshow("Published Terrain Slope", slope_color);
        }

        // if(params->b_show_terrain_labels) {
        //     // 创建标签的彩色可视化
        //     cv::Mat label_visual(img_rows_, img_cols_, CV_8UC3);
        //     for (int x = 0; x < img_rows_; ++x) {
        //         for (int y = 0; y < img_cols_; ++y) {
        //             uint8_t label_val = vehicle_label.at<uint8_t>(x, y);
        //             uint8_t label_class = label_val & 0x0F;  // 提取低4位标签

        //             cv::Vec3b color;
        //             switch (label_class) {
        //                 case 0: color = cv::Vec3b(0, 0, 0);       break;  // unknown - 黑色
        //                 case 1: color = cv::Vec3b(0, 255, 0);     break;  // terrain - 绿色
        //                 case 2: color = cv::Vec3b(128, 128, 128); break;  // gray - 灰色
        //                 case 3: color = cv::Vec3b(0, 0, 255);     break;  // obstacle - 红色
        //                 case 6: color = cv::Vec3b(255, 0, 255);   break;  // semantic_edge - 紫色
        //                 default: color = cv::Vec3b(0, 0, 0);      break;  // 默认黑色
        //             }
        //             label_visual.at<cv::Vec3b>(x, y) = color;
        //         }
        //     }
        //     cv::imshow("Published Terrain Labels", label_visual);
        // }

        cv::waitKey(1);
    }
}


void GridMapHandler_v2::publishColorMap(
    const ros::Publisher &publisher,
    const geometry_msgs::Pose &local_pose)
{
    if (!publisher) return;

    world_state::ColorMap msg;

    auto now = ros::Time::now();
    msg.local_time  = now.toNSec() / 1e6;
    msg.UTC_time    = now.toNSec() / 1e6;
    msg.message_num = 0;

    msg.localPoseStamped.x = local_pose.position.x;
    msg.localPoseStamped.y = local_pose.position.y;
    msg.localPoseStamped.theta = tf::getYaw(local_pose.orientation) * 180.0 / M_PI;

    // 优化：实时计算车体坐标系变换，避免图像旋转操作
    auto& rgb_r_layer = map_["rgb_r"];
    auto& rgb_g_layer = map_["rgb_g"];
    auto& rgb_b_layer = map_["rgb_b"];
    const auto map_size = map_.getSize();

    int rows = map_size(0);
    int cols = map_size(1);

    // 初始化车体坐标系的RGB数据数组（500x500）
    cv::Mat vehicle_rgb_r = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);
    cv::Mat vehicle_rgb_g = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);
    cv::Mat vehicle_rgb_b = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);

    // 获取当前发布时的车辆位姿
    double current_yaw = tf::getYaw(local_pose.orientation);
    double cos_theta = cos(current_yaw);
    double sin_theta = sin(current_yaw);

    // 获取地图中心位置
    grid_map::Position map_center = map_.getPosition();

    // 预计算常量，避免重复计算
    const double half_rows = img_rows_ / 2.0;
    const double half_cols = img_cols_ / 2.0;
    const double map_center_x = map_center.x();
    const double map_center_y = map_center.y();

    // 遍历车体坐标系图像的每个像素
    for (int img_row = 0; img_row < img_rows_; ++img_row) {
        // 预计算行相关的值
        const double car_x = -(img_row - half_rows) * resolution_;
        const double cos_car_x = cos_theta * car_x;
        const double sin_car_x = sin_theta * car_x;

        for (int img_col = 0; img_col < img_cols_; ++img_col) {

            // 图像坐标 → 车体坐标
            const double car_y = -(img_col - half_cols) * resolution_;

            // 车体坐标 → 世界坐标（优化：减少重复计算）
            const double world_x = cos_car_x - sin_theta * car_y + map_center_x;
            const double world_y = sin_car_x + cos_theta * car_y + map_center_y;

            // 世界坐标 → GridMap索引
            grid_map::Position world_pos(world_x, world_y);
            grid_map::Index grid_index;

            if(map_.getIndex(world_pos, grid_index)) {
                const int row = grid_index(0);
                const int col = grid_index(1);

            
                // 检查是否有有效数据
                if (!std::isnan(rgb_r_layer(row, col))&&
                    !std::isnan(rgb_g_layer(row, col)) &&
                    !std::isnan(rgb_b_layer(row, col))) {
                    // 直接填充到车体坐标系位置
                    vehicle_rgb_r.at<uint8_t>(img_row, img_col) = static_cast<uint8_t>(rgb_r_layer(row, col));
                    vehicle_rgb_g.at<uint8_t>(img_row, img_col) = static_cast<uint8_t>(rgb_g_layer(row, col));
                    vehicle_rgb_b.at<uint8_t>(img_row, img_col) = static_cast<uint8_t>(rgb_b_layer(row, col));
                }
            }  
        }
    }

    // 直接使用车体坐标系数据填充消息，无需旋转
    for (int i = 0; i < 250000; ++i) {
        int x = i / img_cols_;
        int y = i % img_cols_;
        if(x >= img_rows_) break;
        msg.r[i] = vehicle_rgb_r.at<uint8_t>(x, y);
        msg.g[i] = vehicle_rgb_g.at<uint8_t>(x, y);
        msg.b[i] = vehicle_rgb_b.at<uint8_t>(x, y);
    }

    publisher.publish(msg);

    // ColorMap BEV 显示暂时关闭。
    // if(params->b_show_colormap_vehicle) {
    //     cv::Mat display_mat;
    //     std::vector<cv::Mat> channels = {vehicle_rgb_b, vehicle_rgb_g, vehicle_rgb_r}; // BGR顺序
    //     cv::merge(channels, display_mat);
    //     cv::imshow("ColorMap (Vehicle Frame)", display_mat);
    //     cv::waitKey(1);
    // }
}





void GridMapHandler_v2::publishObstacleMap(
    const ros::Publisher &publisher,
    const std::vector<ObstacleRect> &near_obstacles,
    const std::vector<ObstacleRect> &far_obstacles,
    const geometry_msgs::Pose &local_pose)
{
    if (!publisher) return;

    // -----------------------------
    // 构建障碍物消息
    // -----------------------------
    world_state::StaticObjArray obstacle_list_msg;
    obstacle_list_msg.header.stamp = ros::Time::now();
    obstacle_list_msg.header.frame_id = "map"; // 发布到世界坐标系

    const float map_resolution = 0.1f; // 每像素米数
    int id_count = 0;

    // 获取车辆朝向
    float cos_yaw = cos(tf::getYaw(local_pose.orientation));
    float sin_yaw = sin(tf::getYaw(local_pose.orientation));

    // 假设栅格地图大小
    const int near_rows = 500;
    const int near_cols = 500;
    const int far_rows = 250;
    const int far_cols = 500;

    const int total_rows = near_rows + far_rows;
    const int total_cols = near_cols; // 和近处一致

    const int origin_x = total_cols / 2; // 栅格中心列
    const int origin_y_far = far_rows;   // 远处区域在上半部分
    const int origin_y_near = far_rows + near_rows; // 近处区域在下半部分

    // -----------------------------
    // 处理远处障碍物（上半部分）
    // -----------------------------
    for (const auto &obs : far_obstacles)
    {
        // 栅格坐标 -> 世界坐标
        float dx = (obs.center.x - origin_x) * map_resolution;
        float dy = (origin_y_far - obs.center.y) * map_resolution;

        float world_x = cos_yaw * dx - sin_yaw * dy + local_pose.position.x;
        float world_y = sin_yaw * dx + cos_yaw * dy + local_pose.position.y;

        float radius = 0.5f * std::sqrt(obs.rect.width*obs.rect.width + obs.rect.height*obs.rect.height) * map_resolution;

        world_state::StaticObj obs_msg;
        obs_msg.id = id_count++;
        obs_msg.type = 1;
        obs_msg.confidence = 1.0;
        obs_msg.center_x = world_x;
        obs_msg.center_y = world_y;
        obs_msg.radius = radius;
        obs_msg.height = 1.2;
        obs_msg.convexhull_vertex_number = 4;

        // 简单矩形凸包
        obs_msg.convexhull_x[0] = world_x - radius; obs_msg.convexhull_y[0] = world_y - radius;
        obs_msg.convexhull_x[1] = world_x + radius; obs_msg.convexhull_y[1] = world_y - radius;
        obs_msg.convexhull_x[2] = world_x + radius; obs_msg.convexhull_y[2] = world_y + radius;
        obs_msg.convexhull_x[3] = world_x - radius; obs_msg.convexhull_y[3] = world_y + radius;

        obstacle_list_msg.obstacles.push_back(obs_msg);
    }

    // -----------------------------
    // 处理近处障碍物（下半部分）
    // -----------------------------
    for (const auto &obs : near_obstacles)
    {
        // 栅格坐标 -> 世界坐标
        float dx = (obs.center.x - origin_x) * map_resolution;
        float dy = (origin_y_near - obs.center.y) * map_resolution;

        float world_x = cos_yaw * dx - sin_yaw * dy + local_pose.position.x;
        float world_y = sin_yaw * dx + cos_yaw * dy + local_pose.position.y;

        float radius = 0.5f * std::sqrt(obs.rect.width*obs.rect.width + obs.rect.height*obs.rect.height) * map_resolution;

        world_state::StaticObj obs_msg;
        obs_msg.id = id_count++;
        obs_msg.type = 1;
        obs_msg.confidence = 1.0;
        obs_msg.center_x = world_x;
        obs_msg.center_y = world_y;
        obs_msg.radius = radius;
        obs_msg.height = 0.0;
        obs_msg.convexhull_vertex_number = 4;

        obs_msg.convexhull_x[0] = world_x - radius; obs_msg.convexhull_y[0] = world_y - radius;
        obs_msg.convexhull_x[1] = world_x + radius; obs_msg.convexhull_y[1] = world_y - radius;
        obs_msg.convexhull_x[2] = world_x + radius; obs_msg.convexhull_y[2] = world_y + radius;
        obs_msg.convexhull_x[3] = world_x - radius; obs_msg.convexhull_y[3] = world_y + radius;

        obstacle_list_msg.obstacles.push_back(obs_msg);
    }

    // -----------------------------
    // 发布消息
    // -----------------------------
    publisher.publish(obstacle_list_msg);
    // ROS_INFO("Published obstacle map with %lu obstacles", obstacle_list_msg.obstacles.size());

    // -----------------------------
    // 可视化
    // -----------------------------
    cv::Mat vis_map(total_rows, total_cols, CV_8UC3, cv::Scalar(0,0,0));

    // 绘制远处障碍物
    for (const auto &obs : far_obstacles) 
    {
        cv::rectangle(vis_map, obs.rect, cv::Scalar(255,0,0), 1);
        cv::circle(vis_map, obs.center, 1, cv::Scalar(0,0,255), cv::FILLED);
    }

    // 绘制近处障碍物
    for (const auto &obs : near_obstacles)
    {
        cv::Rect shifted_rect = obs.rect + cv::Point(0, far_rows); // 下移
        cv::rectangle(vis_map, shifted_rect, cv::Scalar(0,255,0), 1);
        cv::Point shifted_center = obs.center + cv::Point(0, far_rows);
        cv::circle(vis_map, shifted_center, 1, cv::Scalar(0,0,255), cv::FILLED);
    }

    if(params->b_show_obstacle_detection) {
        cv::namedWindow("Published Obstacle Map", cv::WINDOW_NORMAL);
        cv::imshow("Published Obstacle Map", vis_map);
        cv::waitKey(1);
    }
}






void GridMapHandler_v2::showMaps(double current_heading) {
    // 安全检查
    if(map_.getLayers().empty()) {
        std::cerr << "Error: GridMap not initialized!" << std::endl;
        return;
    }

    // 生成可视化图像（以自车为坐标系）
    cv::Mat height_map, rgb_map, has_data;

    // 测试原版本性能
    auto start_time = std::chrono::high_resolution_clock::now();
    generateVisualization(height_map, rgb_map, has_data, current_heading);
    // generateVisualization_v2(height_map, rgb_map, has_data, current_heading);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "generateVisualization duration: " << duration.count() << " ms" << std::endl;

    // 生成坡度图
    cv::Mat slope_map;
    generateSlopeVisualization(slope_map, has_data, current_heading);

    // 使用您现有的可视化方法（保持一致性）
    // height_map 已经包含了高度差信息，高度差BEV可视化暂时关闭。
    auto start_time1 = std::chrono::high_resolution_clock::now();
    // params->generateColorMap(height_map, has_data, "GridMap Height Diff");
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1);
    std::cout << "generateColorMap duration: " << duration1.count() << " ms" << std::endl;

    auto start_time2 = std::chrono::high_resolution_clock::now();
    params->generateRGBMap(rgb_map, has_data, "GridMap RGB");
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    std::cout << "generateRGBMap duration: " << duration2.count() << " ms" << std::endl;

    auto start_time3 = std::chrono::high_resolution_clock::now();
    generateSlopeColorMap(slope_map, has_data, "GridMap Slope");
    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
    std::cout << "generateSlopeVisualization duration: " << duration3.count() << " ms" << std::endl;
}

void GridMapHandler_v2::generateVisualization(cv::Mat& height_map, cv::Mat& rgb_map, cv::Mat& has_data, double current_heading) {
    // 初始化输出图像
    height_map = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    rgb_map = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC3);
    has_data = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);

    // 预计算常量值（避免重复计算）
    const double cos_heading = cos(current_heading);
    const double sin_heading = sin(current_heading);
    const double half_rows = img_rows_ * 0.5;
    const double half_cols = img_cols_ * 0.5;
    const double neg_resolution = -resolution_;

    // 获取地图中心位置
    const grid_map::Position map_center = map_.getPosition();
    const double center_x = map_center.x();
    const double center_y = map_center.y();

    // 获取数据层引用
    const auto& elevation = map_["elevation"];
    const auto& rgb_r = map_["rgb_r"];
    const auto& rgb_g = map_["rgb_g"];
    const auto& rgb_b = map_["rgb_b"];
    const auto& has_valid_color = map_["has_valid_color"];
    const auto& intrested_flag = map_["intrested_flag"];
    const auto& dynamic_flag = map_["dynamic_flag"];

    // 获取直接指针访问，提高内存访问效率
    uint8_t* has_data_ptr = has_data.ptr<uint8_t>();
    float* height_ptr = height_map.ptr<float>();
    cv::Vec3b* rgb_ptr = rgb_map.ptr<cv::Vec3b>();

    // 遍历图像像素（优化版本 + 并行化）
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int img_row = 0; img_row < img_rows_; ++img_row) {
        // 预计算行相关的坐标分量
        const double car_x = (img_row - half_rows) * neg_resolution;
        const double cos_car_x = cos_heading * car_x;
        const double sin_car_x = sin_heading * car_x;

        for(int img_col = 0; img_col < img_cols_; ++img_col) {
            const int pixel_idx = img_row * img_cols_ + img_col;

            // 优化的坐标变换计算
            const double car_y = (img_col - half_cols) * neg_resolution;

            // 最终世界坐标（减少重复计算）
            const double world_x = cos_car_x - sin_heading * car_y + center_x;
            const double world_y = sin_car_x + cos_heading * car_y + center_y;

            // 世界坐标 → GridMap索引
            grid_map::Position world_pos(world_x, world_y);
            grid_map::Index grid_index;

            if(map_.getIndex(world_pos, grid_index)) {
                const int row = grid_index(0);
                const int col = grid_index(1);

                // 一次性获取所有需要的数据
                const float elev_val = elevation(row, col);

                if(!std::isnan(elev_val)) {
                    // 有有效数据（使用指针直接访问）
                    has_data_ptr[pixel_idx] = 255;
                    height_ptr[pixel_idx] = elev_val;

                    uint8_t r, g, b;
                    if(has_valid_color(row, col) > 0.5f) {
                        r = rgb_r(row, col);
                        g = rgb_g(row, col);
                        b = rgb_b(row, col);
                    } else {
                        // 默认灰色
                        r = g = b = 128;
                    }

                    // test ,看一下动态区域
                    if(dynamic_flag(row, col) > 0.5f) {
                        r = 255;
                        g = 0;
                        b = 0;
                    }

                    rgb_ptr[pixel_idx] = cv::Vec3b(b, g, r);  // OpenCV是BGR
                } 
            }
        }
    }
}


void GridMapHandler_v2::generateSlopeVisualization(cv::Mat& slope_map, cv::Mat& has_data, double current_heading) {
    // 初始化输出图像
    slope_map = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);

    // 获取数据层
    const auto& slope = map_["slope"];      // 坡度层
    const auto& elevation = map_["elevation"];  // 用于判断数据有效性

    grid_map::Position map_center = map_.getPosition();

    // 遍历图像像素
    for(int img_row = 0; img_row < img_rows_; ++img_row) {
        for(int img_col = 0; img_col < img_cols_; ++img_col) {

            // 图像坐标 → 车体坐标
            double car_x = -(img_row - img_rows_/2) * resolution_;
            double car_y = -(img_col - img_cols_/2) * resolution_;

            // 车体坐标 → 世界坐标（考虑车辆朝向）
            double cos_heading = cos(current_heading);
            double sin_heading = sin(current_heading);

            double dx = cos_heading * car_x - sin_heading * car_y;
            double dy = sin_heading * car_x + cos_heading * car_y;

            // 最终世界坐标
            double world_x = map_center.x() + dx;
            double world_y = map_center.y() + dy;

            // 世界坐标 → GridMap索引
            grid_map::Position world_pos(world_x, world_y);
            grid_map::Index grid_index;

            if(map_.getIndex(world_pos, grid_index)) {
                const int row = grid_index(0);
                const int col = grid_index(1);

                // 检查是否有有效数据（使用elevation作为数据有效性标志）
                if(!std::isnan(elevation(row, col))) {
                    // 有有效数据
                    has_data.at<uint8_t>(img_row, img_col) = 255;

                    // 获取坡度值
                    if(!std::isnan(slope(row, col))) {
                        slope_map.at<float>(img_row, img_col) = slope(row, col);  // 坡度值（度数）
                    } else {
                        slope_map.at<float>(img_row, img_col) = 0.0f;  // 没有坡度数据，设为0度
                    }
                } else {
                    // 没有数据的栅格
                    has_data.at<uint8_t>(img_row, img_col) = 0;
                    slope_map.at<float>(img_row, img_col) = 0.0f;
                }
            } else {
                // 超出地图范围的区域
                has_data.at<uint8_t>(img_row, img_col) = 0;
                slope_map.at<float>(img_row, img_col) = 0.0f;
            }
        }
    }
}

void GridMapHandler_v2::generateVisualization_v2(cv::Mat& height_map, cv::Mat& rgb_map, cv::Mat& has_data, double current_heading) {
    // 初始化输出图像
    height_map = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    rgb_map = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC3);
    has_data = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);

    // 获取数据层
    const auto& elevation = map_["elevation"];    // 高度差层
    const auto& rgb_r = map_["rgb_r"];           // RGB红色分量
    const auto& rgb_g = map_["rgb_g"];           // RGB绿色分量
    const auto& rgb_b = map_["rgb_b"];           // RGB蓝色分量
    const auto& car_col_layer = map_["car_col"]; // 车体坐标系列索引
    const auto& car_row_layer = map_["car_row"]; // 车体坐标系行索引

    // 获取直接指针访问，提高内存访问效率
    float* height_ptr = height_map.ptr<float>();
    cv::Vec3b* rgb_ptr = rgb_map.ptr<cv::Vec3b>();
    uint8_t* has_data_ptr = has_data.ptr<uint8_t>();

    // 获取地图参数
    const int map_rows = map_.getSize()(0);
    const int map_cols = map_.getSize()(1);

    // 使用预计算的车体坐标索引进行直接映射
    for (int global_row = 0; global_row < map_rows; ++global_row) {
        for (int global_col = 0; global_col < map_cols; ++global_col) {

            // 检查是否有有效数据
            if (std::isnan(elevation(global_row, global_col))) {
                continue;  // 跳过无效数据
            }

            // 直接从预计算的车体坐标索引层获取车体栅格坐标
            const float car_col_f = car_col_layer(global_row, global_col);
            const float car_row_f = car_row_layer(global_row, global_col);

            // 检查车体坐标索引是否有效（-1表示无效）
            if (car_col_f < 0 || car_row_f < 0) {
                continue;  // 跳过无效的车体坐标
            }

            // 转换为整数索引
            const int vehicle_col = static_cast<int>(car_col_f);
            const int vehicle_row = static_cast<int>(car_row_f);

            // 检查车体栅格坐标是否在有效范围内
            if (vehicle_row >= 0 && vehicle_row < img_rows_ &&
                vehicle_col >= 0 && vehicle_col < img_cols_) {

                const int pixel_idx = vehicle_row * img_cols_ + vehicle_col;

                // 标记有有效数据
                has_data_ptr[pixel_idx] = 255;

                // 获取高度差值
                if (!std::isnan(elevation(global_row, global_col))) {
                    height_ptr[pixel_idx] = elevation(global_row, global_col);  // 高度差值
                } else {
                    height_ptr[pixel_idx] = 0.0f;  // 没有高度数据，设为0
                }

                // 获取RGB颜色值
                const float r_float = rgb_r(global_row, global_col);
                const float g_float = rgb_g(global_row, global_col);
                const float b_float = rgb_b(global_row, global_col);

                const uint8_t r_val = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, r_float)));
                const uint8_t g_val = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, g_float)));
                const uint8_t b_val = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, b_float)));

                rgb_ptr[pixel_idx] = cv::Vec3b(b_val, g_val, r_val);  // OpenCV使用BGR顺序
            }
        }
    }
}

void GridMapHandler_v2::generateVisualization_v3(cv::Mat& height_map, cv::Mat& rgb_map, cv::Mat& has_data, double current_heading) {
    // 初始化输出图像
    height_map = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    rgb_map = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC3);
    has_data = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC1);

    // 预计算常量值（避免重复计算）
    const double cos_heading = cos(current_heading);
    const double sin_heading = sin(current_heading);
    const double half_rows = img_rows_ * 0.5;
    const double half_cols = img_cols_ * 0.5;
    const double neg_resolution = -resolution_;

    // 获取地图中心位置
    const grid_map::Position map_center = map_.getPosition();
    const double center_x = map_center.x();
    const double center_y = map_center.y();

    // 获取数据层引用
    const auto& elevation = map_["elevation"];
    const auto& rgb_r = map_["rgb_r"];
    const auto& rgb_g = map_["rgb_g"];
    const auto& rgb_b = map_["rgb_b"];

    // 获取直接指针访问，提高内存访问效率
    uint8_t* has_data_ptr = has_data.ptr<uint8_t>();
    float* height_ptr = height_map.ptr<float>();
    cv::Vec3b* rgb_ptr = rgb_map.ptr<cv::Vec3b>();

    // 遍历图像像素（优化版本 + 并行化）
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int img_row = 0; img_row < img_rows_; ++img_row) {
        // 预计算行相关的坐标分量
        const double car_x = (img_row - half_rows) * neg_resolution;
        const double cos_car_x = cos_heading * car_x;
        const double sin_car_x = sin_heading * car_x;

        for(int img_col = 0; img_col < img_cols_; ++img_col) {
            const int pixel_idx = img_row * img_cols_ + img_col;

            // 优化的坐标变换计算
            const double car_y = (img_col - half_cols) * neg_resolution;

            // 最终世界坐标（减少重复计算）
            const double world_x = cos_car_x - sin_heading * car_y + center_x;
            const double world_y = sin_car_x + cos_heading * car_y + center_y;

            // 世界坐标 → GridMap索引
            grid_map::Position world_pos(world_x, world_y);
            grid_map::Index grid_index;

            if(map_.getIndex(world_pos, grid_index)) {
                const int row = grid_index(0);
                const int col = grid_index(1);

                // 一次性获取所有需要的数据
                const float elev_val = elevation(row, col);

                if(!std::isnan(elev_val)) {
                    // 有有效数据（使用指针直接访问）
                    has_data_ptr[pixel_idx] = 255;
                    height_ptr[pixel_idx] = elev_val;

                    // 优化的RGB处理（减少重复计算）
                    const float r_val = rgb_r(row, col);
                    const float g_val = rgb_g(row, col);
                    const float b_val = rgb_b(row, col);

                    uint8_t r, g, b;
                    if(!std::isnan(r_val)) {
                        // 快速RGB转换（避免重复的min/max调用）
                        // r = (r_val < 0.0f) ? 0 : (r_val > 255.0f) ? 255 : static_cast<uint8_t>(r_val);
                        // g = (g_val < 0.0f) ? 0 : (g_val > 255.0f) ? 255 : static_cast<uint8_t>(g_val);
                        // b = (b_val < 0.0f) ? 0 : (b_val > 255.0f) ? 255 : static_cast<uint8_t>(b_val);
                        r = r_val;
                        g = g_val;
                        b = b_val;
                    } else {
                        // 默认灰色
                        r = g = b = 128;
                    }

                    rgb_ptr[pixel_idx] = cv::Vec3b(b, g, r);  // OpenCV是BGR
                }
            }
        }
    }
}

void GridMapHandler_v2::generateSlopeColorMap(cv::Mat &slope_map, cv::Mat &has_data, const std::string& window_name) {
    // 创建彩色坡度图像
    cv::Mat slope_color_map = cv::Mat::zeros(slope_map.rows, slope_map.cols, CV_8UC3);

    // 坡度颜色映射：绿色(平地) -> 黄色(缓坡) -> 橙色(中坡) -> 红色(陡坡) -> 深红色(极陡)
    for(int row = 0; row < slope_map.rows; ++row) {
        for(int col = 0; col < slope_map.cols; ++col) {
            // 检查是否有有效数据
            if(has_data.at<uint8_t>(row, col) == 0) {
                // 无数据区域显示为黑色
                slope_color_map.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
                continue;
            }

            float slope_degree = slope_map.at<float>(row, col);

            // 处理NaN值
            if(std::isnan(slope_degree)) {
                slope_color_map.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);  // 黑色
                continue;
            }

            // 限制坡度范围到[0, 90]度
            slope_degree = std::max(0.0f, std::min(90.0f, slope_degree));

            uint8_t r, g, b;

            if(slope_degree <= 5.0f) {
                // 0-5度：蓝色 (平地)
                float ratio = slope_degree / 5.0f;
                r = 0;
                g = static_cast<uint8_t>(ratio * 100);        // 0 -> 100
                b = static_cast<uint8_t>(255 - ratio * 55);   // 255 -> 200
            }
            else if(slope_degree <= 15.0f) {
                // 5-15度：蓝色到绿色 (缓坡)
                float ratio = (slope_degree - 5.0f) / 10.0f;
                r = 0;
                g = static_cast<uint8_t>(100 + ratio * 155);  // 100 -> 255
                b = static_cast<uint8_t>(200 - ratio * 200);  // 200 -> 0
            }
            else if(slope_degree <= 30.0f) {
                // 15-30度：绿色到橙色 (中坡)
                float ratio = (slope_degree - 15.0f) / 15.0f;
                r = static_cast<uint8_t>(ratio * 255);        // 0 -> 255
                g = 255;                                       // 保持最大
                b = 0;
            }
            else if(slope_degree <= 45.0f) {
                // 30-45度：橙色到红色 (陡坡)
                float ratio = (slope_degree - 30.0f) / 15.0f;
                r = 255;                                       // 保持最大
                g = static_cast<uint8_t>(255 - ratio * 165);  // 255 -> 90
                b = 0;
            }
            else {
                // 45-90度：红色到深红色 (极陡)
                float ratio = (slope_degree - 45.0f) / 45.0f;
                r = static_cast<uint8_t>(255 - ratio * 100);  // 255 -> 155
                g = static_cast<uint8_t>(90 - ratio * 90);    // 90 -> 0
                b = 0;
            }

            slope_color_map.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);  // OpenCV是BGR格式
        }
    }

    // 显示图像
    if(params->b_show_slope_map) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, slope_color_map);
        cv::waitKey(1);
    }
}

void GridMapHandler_v2::opencv_showMaps(double current_heading) {
    if(map_.getLayers().empty()) {
        std::cerr << "Error: GridMap not initialized!" << std::endl;
        return;
    }
    if (!map_.exists("elevation")) {
        std::cerr << "Error: elevation layer not found!" << std::endl;
        return;
    }

    auto start_time0 = std::chrono::high_resolution_clock::now();
    map_.convertToDefaultStartIndex();
    auto end_time0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time0 - start_time0);
    // std::cout << "opencv_showMaps convertToDefaultStartIndex: " << duration0.count() << " ms" << std::endl;

    auto start_time1 = std::chrono::high_resolution_clock::now();
    cv::Mat vehicle_rgb_mat;
    cv::Mat vehicle_has_data;
    generateVisualization(vehicle_height_mat, vehicle_rgb_mat, vehicle_has_data, current_heading);
    auto mid_time1 = std::chrono::high_resolution_clock::now();
    auto duration_mid1 = std::chrono::duration_cast<std::chrono::milliseconds>(mid_time1 - start_time1);
    // std::cout << "opencv_showMaps target-grid height: " << duration_mid1.count() << " ms" << std::endl;

    // 高度差BEV可视化暂时关闭。
    // params->generateColorMap(vehicle_height_mat, vehicle_has_data, "GridMap Height Diff");
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - mid_time1);
    // std::cout << "opencv_showMaps height draw: " << duration1.count() << " ms" << std::endl;

    auto start_time3 = std::chrono::high_resolution_clock::now();
    if (!vehicle_rgb_mat.empty()) {
        auto mid_time3 = std::chrono::high_resolution_clock::now();
        auto duration_mid3 = std::chrono::duration_cast<std::chrono::milliseconds>(mid_time3 - start_time3);
        // std::cout << "opencv_showMaps target-grid RGB: " << duration_mid3.count() << " ms" << std::endl;
        params->generateRGBMap(vehicle_rgb_mat, vehicle_has_data, "GridMap RGB (Vehicle Frame)");
        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - mid_time3);
        // std::cout << "opencv_showMaps RGB draw: " << duration3.count() << " ms" << std::endl;
    }

    auto start_time4 = std::chrono::high_resolution_clock::now();
    if (map_.exists("slope")) {
        cv::Mat vehicle_slope_mat;
        cv::Mat vehicle_slope_has_data = vehicle_has_data.clone();
        generateSlopeVisualization(vehicle_slope_mat, vehicle_slope_has_data, current_heading);

        auto mid_time4 = std::chrono::high_resolution_clock::now();
        auto duration_mid4 = std::chrono::duration_cast<std::chrono::milliseconds>(mid_time4 - start_time4);
        // std::cout << "opencv_showMaps target-grid slope: " << duration_mid4.count() << " ms" << std::endl;
        generateSlopeColorMap(vehicle_slope_mat, vehicle_slope_has_data, "GridMap Slope (Vehicle Frame)");
        auto end_time4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4 - mid_time4);
        // std::cout << "opencv_showMaps slope draw: " << duration4.count() << " ms" << std::endl;
    }
}


// 远距离区域处理函数 - 简单实时处理，不考虑历史融合
cv::Mat GridMapHandler_v2::processFarRangePoints(const std::vector<PointXYZRGBValid>& colored_points) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 远距离区域参数
    const float x_min = 50.0f;   // X方向50m开始
    const float x_max = 100.0f;  // X方向100m结束
    const float y_min = -50.0f;  // Y方向左侧50m（与近处区域对齐）
    const float y_max = 50.0f;   // Y方向右侧50m（与近处区域对齐）
    const float point_y_min = -20.0f;  // Y方向左侧20m,点云实际范围
    const float point_y_max = 20.0f;   // Y方向右侧20m
    const float grid_size = 0.2f; // 栅格大小0.2m

    // 计算栅格地图尺寸
    // 注意：在您的坐标系统中，row对应X轴，col对应Y轴
    const int rows = static_cast<int>((x_max - x_min) / grid_size); // 250行 (X方向50m/0.2m)
    const int cols = static_cast<int>((y_max - y_min) / grid_size); // 500列 (Y方向100m/0.2m)

    // std::cout << "远距离区域栅格地图尺寸: " << rows << "x" << cols << " (250×500)" << std::endl;

    // 创建高度统计矩阵
    cv::Mat max_height = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat min_height = cv::Mat::ones(rows, cols, CV_32F) * 1000.0f; // 初始化为很大的值
    cv::Mat point_count = cv::Mat::zeros(rows, cols, CV_32S);

    // 投影点云到栅格
    int valid_points = 0;
    for(const auto& point : colored_points) {
        // 检查点是否在远距离区域内
        if(point.x >= x_min && point.x <= x_max &&
           point.y >= point_y_min && point.y <= point_y_max) {

            // 计算栅格索引
            int col = static_cast<int>((-point.y + y_max) / grid_size);
            int row = static_cast<int>((-point.x + x_max) / grid_size);

            // 边界检查
            if(row >= 0 && row < rows && col >= 0 && col < cols) {
                // 更新高度统计
                float& max_h = max_height.at<float>(row, col);
                float& min_h = min_height.at<float>(row, col);
                int& count = point_count.at<int>(row, col);

                max_h = std::max(max_h, point.z);
                min_h = std::min(min_h, point.z);
                count++;
                valid_points++;
            }
        }
    }

    // 计算高度差矩阵
    cv::Mat height_diff = cv::Mat::zeros(rows, cols, CV_32F);
    int valid_cells = 0;

    for(int row = 0; row < rows; ++row) {
        for(int col = 0; col < cols; ++col) {
            int count = point_count.at<int>(row, col);
            if(count > 0) {
                float max_h = max_height.at<float>(row, col);
                float min_h = min_height.at<float>(row, col);

                // 如果最小高度还是初始值，说明没有有效点
                if(min_h < 999.0f) {
                    height_diff.at<float>(row, col) = max_h - min_h;
                    valid_cells++;
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // std::cout << "远距离处理完成: " << valid_points << " 个有效点, "
    //           << valid_cells << " 个有效栅格, 耗时: " << duration.count() << " ms" << std::endl;

    return height_diff;
}

// 障碍物检测 - 分区域处理：近处3×3，远处5×5
void GridMapHandler_v2::obstacle_detection(cv::Mat& vehicle_height_mat, cv::Mat& far_range_height_diff) {
    
    auto start_time1 = std::chrono::high_resolution_clock::now();
    near_obstacles_.clear();
    far_obstacles_.clear();
    
    // std::cout << "=== 开始分区域障碍物检测 ===" << std::endl;
    // std::cout << "近处区域: " << vehicle_height_mat.rows << "x" << vehicle_height_mat.cols << " (50m内，3×3检测)" << std::endl;
    // std::cout << "远处区域: " << far_range_height_diff.rows << "x" << far_range_height_diff.cols << " (50-100m，5×5检测)" << std::endl;

    // ========== 第一部分：近处区域处理（50m内，3×3检测）==========
    auto near_start_time = std::chrono::high_resolution_clock::now();

    // 近处区域配置
    const int near_size = 3;
    const int near_threshold = 6; //判断为障碍物的阈值
    const int near_half_size = near_size / 2;

    // 创建近处区域障碍物二值图
    cv::Mat near_obstacle_mat = cv::Mat::zeros(vehicle_height_mat.rows, vehicle_height_mat.cols, CV_8UC1); // 障碍物栅格0-1图
    cv::Mat near_detection_mat = cv::Mat::zeros(vehicle_height_mat.rows, vehicle_height_mat.cols, CV_8UC1); // 障碍物检测中心图



    int near_obstacle_count = 0;
    for(int row = 0; row < vehicle_height_mat.rows; ++row) {
        const float* height_ptr = vehicle_height_mat.ptr<float>(row);
        uint8_t* obstacle_ptr = near_obstacle_mat.ptr<uint8_t>(row);
        for(int col = 0; col < vehicle_height_mat.cols; ++col) {
            if(!std::isnan(height_ptr[col]) && height_ptr[col] > 0.2f) {
                obstacle_ptr[col] = 1;
                near_obstacle_count++;
            } 
        }
    }

    // 近处区域3×3检测
    int near_detection_count = 0;
    cv::Mat near_used_mat = cv::Mat::zeros(vehicle_height_mat.rows, vehicle_height_mat.cols, CV_8UC1); // 已使用区域标记

    for(int row = near_half_size; row < near_obstacle_mat.rows - near_half_size; ++row) {
        for(int col = near_half_size; col < near_obstacle_mat.cols - near_half_size; ++col) {
            // 检查中心是否已被使用
            if(near_used_mat.at<uint8_t>(row, col) > 0) continue;

            // 统计3×3邻域内障碍物数量
            int neighbor_count = 0;
            for(int dr = -near_half_size; dr <= near_half_size; ++dr) {
                const uint8_t* neighbor_ptr = near_obstacle_mat.ptr<uint8_t>(row + dr);
                for(int dc = -near_half_size; dc <= near_half_size; ++dc) {
                    if(neighbor_ptr[col + dc] == 1 && near_used_mat.at<uint8_t>(row + dr, col + dc) == 0) {
                        neighbor_count++;
                    }
                }
            }

            // 如果满足阈值，标记为3×3障碍物
            if(neighbor_count >= near_threshold) {
                near_detection_mat.at<uint8_t>(row, col) = 1;
                near_detection_count++;

                ObstacleRect obs;
                obs.center = cv::Point(col, row);
                obs.rect = cv::Rect(col - near_half_size, row - near_half_size, near_size, near_size);
                near_obstacles_.push_back(obs);

                // 标记使用区域，避免重叠
                for(int dr = -near_half_size; dr <= near_half_size; ++dr) {
                    for(int dc = -near_half_size; dc <= near_half_size; ++dc) {
                        int r = row + dr;
                        int c = col + dc;
                        if(r >= 0 && r < near_used_mat.rows && c >= 0 && c < near_used_mat.cols) {
                            near_used_mat.at<uint8_t>(r, c) = 1;
                        }
                    }
                }
            }
        }
    }

    auto near_end_time = std::chrono::high_resolution_clock::now();
    auto near_duration = std::chrono::duration_cast<std::chrono::milliseconds>(near_end_time - near_start_time);
    // std::cout << "近处区域检测完成: " << near_detection_count << "个3×3障碍物, 耗时: " << near_duration.count() << "ms" << std::endl;

    // ========== 第二部分：远处区域处理（50-100m，5×5检测）==========
    auto far_start_time = std::chrono::high_resolution_clock::now();

    // 远处区域配置
    const int far_size = 5;
    const int far_threshold = 10;  // 障碍物阈值
    const int far_half_size = far_size / 2;

    // 创建远处区域障碍物二值图和检测结果
    cv::Mat far_obstacle_mat = cv::Mat::zeros(far_range_height_diff.rows, far_range_height_diff.cols, CV_8UC1);
    cv::Mat far_detection_mat = cv::Mat::zeros(far_range_height_diff.rows, far_range_height_diff.cols, CV_8UC1);





    int far_obstacle_count = 0;
    for(int row = 0; row < far_range_height_diff.rows; ++row) {
        const float* height_ptr = far_range_height_diff.ptr<float>(row);
        uint8_t* obstacle_ptr = far_obstacle_mat.ptr<uint8_t>(row);
        for(int col = 0; col < far_range_height_diff.cols; ++col) {
            if(!std::isnan(height_ptr[col]) && height_ptr[col] > 0.1f) {
                obstacle_ptr[col] = 1;
                far_obstacle_count++;
            }
        }
    }

    // 远处区域5×5检测
    int far_detection_count = 0;
    cv::Mat far_used_mat = cv::Mat::zeros(far_range_height_diff.rows, far_range_height_diff.cols, CV_8UC1);

    for(int row = far_half_size; row < far_obstacle_mat.rows - far_half_size; ++row) {
        for(int col = far_half_size; col < far_obstacle_mat.cols - far_half_size; ++col) {
            // 检查中心是否已被使用
            if(far_used_mat.at<uint8_t>(row, col) > 0) continue;

            // 统计5×5邻域内障碍物数量
            int neighbor_count = 0;
            for(int dr = -far_half_size; dr <= far_half_size; ++dr) {
                const uint8_t* neighbor_ptr = far_obstacle_mat.ptr<uint8_t>(row + dr);
                for(int dc = -far_half_size; dc <= far_half_size; ++dc) {
                    if(neighbor_ptr[col + dc] == 1 && far_used_mat.at<uint8_t>(row + dr, col + dc) == 0) {
                        neighbor_count++;
                    }
                }
            }

            // 如果满足阈值，标记为5×5障碍物
            if(neighbor_count >= far_threshold) {
                far_detection_mat.at<uint8_t>(row, col) = 1;
                far_detection_count++;


                ObstacleRect obs;
                obs.center = cv::Point(col, row);
                obs.rect = cv::Rect(col - far_half_size, row - far_half_size, near_size, near_size);
                far_obstacles_.push_back(obs);

                
                // 标记使用区域，避免重叠
                for(int dr = -far_half_size; dr <= far_half_size; ++dr) {
                    for(int dc = -far_half_size; dc <= far_half_size; ++dc) {
                        int r = row + dr;
                        int c = col + dc;
                        if(r >= 0 && r < far_used_mat.rows && c >= 0 && c < far_used_mat.cols) {
                            far_used_mat.at<uint8_t>(r, c) = 1;
                        }
                    }
                }
            }
        }
    }

    auto far_end_time = std::chrono::high_resolution_clock::now();
    auto far_duration = std::chrono::duration_cast<std::chrono::milliseconds>(far_end_time - far_start_time);
    // std::cout << "远处区域检测完成: " << far_detection_count << "个5×5障碍物, 耗时: " << far_duration.count() << "ms" << std::endl;

    // ========== 统计总结果 ==========
    // std::cout << "=== 分区域障碍物检测完成 ===" << std::endl;
    // std::cout << "近处区域(50m内): " << near_obstacle_count << "个障碍物栅格 -> " << near_detection_count << "个3×3障碍物(0.6m)" << std::endl;
    // std::cout << "远处区域(50-100m): " << far_obstacle_count << "个障碍物栅格 -> " << far_detection_count << "个5×5障碍物(1.0m)" << std::endl;
    // std::cout << "总检测结果: " << (near_detection_count + far_detection_count) << "个障碍物" << std::endl;

    // ========== 统一可视化：显示-50到100m的障碍物图 ==========
    auto vis_start_time = std::chrono::high_resolution_clock::now();
    visualizeCombinedObstacleDetection(vehicle_height_mat, far_range_height_diff,
                                     near_detection_mat, far_detection_mat,
                                     near_obstacle_mat, far_obstacle_mat);
    auto vis_end_time = std::chrono::high_resolution_clock::now();
    auto vis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vis_end_time - vis_start_time);
    // std::cout << "统一可视化耗时: " << vis_duration.count() << "ms" << std::endl;

    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vis_end_time - start_time1);
    std::cout << "obstacle_detection 总耗时: " << total_duration.count() << "ms" << std::endl;
}

// 统一可视化函数：显示从-50到100m的障碍物检测结果
void GridMapHandler_v2::visualizeCombinedObstacleDetection(const cv::Mat& near_height_mat, const cv::Mat& far_height_mat,
                                                          const cv::Mat& near_detection_mat, const cv::Mat& far_detection_mat,
                                                          const cv::Mat& near_obstacle_mat, const cv::Mat& far_obstacle_mat) {
    // std::cout << "开始创建统一可视化图像..." << std::endl;

    // 计算统一图像的尺寸
    // 近处区域：500×500 (对应-50到50m，100m×100m)
    // 远处区域：250×250 (对应50到100m，50m×50m，左右各25m)
    const int near_rows = near_height_mat.rows;
    const int near_cols = near_height_mat.cols;
    const int far_rows = far_height_mat.rows;
    const int far_cols = far_height_mat.cols;

    // 修正拼图逻辑：根据原始坐标系统
    // 在图像中：行(row)对应X轴(前后)，列(col)对应Y轴(左右)
    // 近处：0到50m (X方向) -> 近处区域的下半部分
    // 远处：50到100m (X方向) -> 远处区域的上半部分
    // 两个区域Y方向范围完全相同：都是-50到50m (100m宽)
    // 近处：500×500，远处：250×500，完美对齐
    const int total_rows = near_rows + far_rows;  // 行方向总像素数(对应X轴0-100m)
    const int total_cols = near_cols;  // 列方向相同(都是500列，对应Y轴-50到50m)

    // std::cout << "统一图像尺寸: " << total_rows << "x" << total_cols << std::endl;
    // std::cout << "近处区域: " << near_rows << "x" << near_cols << ", 远处区域: " << far_rows << "x" << far_cols << std::endl;

    // 创建统一的可视化图像
    cv::Mat combined_visualization = cv::Mat::zeros(total_rows, total_cols, CV_8UC3);

    // ========== 绘制远处区域（上半部分，50到100m）==========
    for(int row = 0; row < far_rows; ++row) {
        for(int col = 0; col < far_cols && col < total_cols; ++col) {
            const int combined_row = row; // 远处区域在图像的上半部分

            // 1. 先绘制原始障碍物栅格（灰色背景）- 使用检测时相同的障碍物矩阵
            if(far_obstacle_mat.at<uint8_t>(row, col) > 0) {
                combined_visualization.at<cv::Vec3b>(combined_row, col) = cv::Vec3b(128, 128, 128); // 灰色
            }
        }
    }

    // 2. 绘制5×5检测结果 - 显示实际参与检测的障碍物栅格
    for(int row = 0; row < far_rows; ++row) {
        for(int col = 0; col < far_cols && col < total_cols; ++col) {
            const int combined_row = row;

            // 如果这里是检测中心
            if(far_detection_mat.at<uint8_t>(row, col) > 0) {
                // 显示这个检测中心周围5×5区域内所有的障碍物栅格为蓝色
                for(int dr = -2; dr <= 2; ++dr) {
                    for(int dc = -2; dc <= 2; ++dc) {
                        int check_row = row + dr;
                        int check_col = col + dc;

                        // 边界检查
                        if(check_row >= 0 && check_row < far_rows &&
                           check_col >= 0 && check_col < far_cols) {
                            // 检查这个位置是否是障碍物栅格
                            if(far_obstacle_mat.at<uint8_t>(check_row, check_col) > 0) {
                                // 显示为蓝色（参与了检测的障碍物栅格）
                                int vis_row = check_row;
                                int vis_col = check_col;
                                if(vis_col < total_cols) {
                                    combined_visualization.at<cv::Vec3b>(vis_row, vis_col) = cv::Vec3b(255, 0, 0); // 蓝色
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. 最后绘制红色中心点（确保可见）
    for(int row = 0; row < far_rows; ++row) {
        for(int col = 0; col < far_cols && col < total_cols; ++col) {
            const int combined_row = row;

            if(far_detection_mat.at<uint8_t>(row, col) > 0) {
                combined_visualization.at<cv::Vec3b>(combined_row, col) = cv::Vec3b(0, 0, 255); // 红色中心点
            }
        }
    }

    // ========== 绘制近处区域（下半部分，0到50m）==========
    const int near_start_row = far_rows; // 近处区域在图像中的起始行
    for(int row = 0; row < near_rows; ++row) {
        for(int col = 0; col < near_cols && col < total_cols; ++col) {
            const int combined_row = near_start_row + row; // 在统一图像中的行位置

            // 1. 先绘制原始障碍物栅格（灰色背景）- 使用检测时相同的障碍物矩阵
            if(near_obstacle_mat.at<uint8_t>(row, col) > 0) {
                combined_visualization.at<cv::Vec3b>(combined_row, col) = cv::Vec3b(128, 128, 128); // 灰色
            }
        }
    }

    // 2. 绘制3×3检测结果 - 显示实际参与检测的障碍物栅格
    for(int row = 0; row < near_rows; ++row) {
        for(int col = 0; col < near_cols && col < total_cols; ++col) {
            const int combined_row = near_start_row + row;

            // 如果这里是检测中心
            if(near_detection_mat.at<uint8_t>(row, col) > 0) {
                // 显示这个检测中心周围3×3区域内所有的障碍物栅格为绿色
                for(int dr = -1; dr <= 1; ++dr) {
                    for(int dc = -1; dc <= 1; ++dc) {
                        int check_row = row + dr;
                        int check_col = col + dc;

                        // 边界检查
                        if(check_row >= 0 && check_row < near_rows &&
                           check_col >= 0 && check_col < near_cols) {
                            // 检查这个位置是否是障碍物栅格
                            if(near_obstacle_mat.at<uint8_t>(check_row, check_col) > 0) {
                                // 显示为绿色（参与了检测的障碍物栅格）
                                int vis_row = near_start_row + check_row;
                                int vis_col = check_col;
                                if(vis_col < total_cols) {
                                    combined_visualization.at<cv::Vec3b>(vis_row, vis_col) = cv::Vec3b(0, 255, 0); // 绿色
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. 最后绘制红色中心点（确保可见）
    for(int row = 0; row < near_rows; ++row) {
        for(int col = 0; col < near_cols && col < total_cols; ++col) {
            const int combined_row = near_start_row + row;

            if(near_detection_mat.at<uint8_t>(row, col) > 0) {
                combined_visualization.at<cv::Vec3b>(combined_row, col) = cv::Vec3b(0, 0, 255); // 红色中心点
            }
        }
    }


    // ========== 在已有像素基础上绘制OpenCV矩形框 ==========
    // for(const auto& obs : far_obstacles_) {
    //     cv::rectangle(combined_visualization, obs.rect, cv::Scalar(255, 0, 0), 1); // 蓝色框
    //     cv::circle(combined_visualization, obs.center, 1, cv::Scalar(0, 0, 255), cv::FILLED); // 红色中心点
    // }

    // for(const auto& obs : near_obstacles_) {
    //     cv::Rect shifted_rect = obs.rect + cv::Point(0, far_rows); // 行偏移
    //     cv::rectangle(combined_visualization, shifted_rect, cv::Scalar(0, 255, 0), 1); // 绿色框
    //     cv::Point shifted_center = obs.center + cv::Point(0, far_rows);
    //     cv::circle(combined_visualization, shifted_center, 1, cv::Scalar(0, 0, 255), cv::FILLED);
    // }

    // ========== 添加图例（移除分界线实现无缝拼接）==========
    // 注释掉分界线，实现无缝拼接
    // cv::line(combined_visualization,
    //          cv::Point(0, far_rows),
    //          cv::Point(total_cols-1, far_rows),
    //          cv::Scalar(255, 255, 255), 2);

    // 添加图例
    const int legend_x = 10;
    int legend_y = 30;
    const int legend_spacing = 25;

    cv::putText(combined_visualization, "Combined Obstacle Detection (0-100m forward)",
                cv::Point(legend_x, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    legend_y += legend_spacing;

    cv::putText(combined_visualization, "Gray: Original obstacles (>0.2m)",
                cv::Point(legend_x, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 128, 128), 1);
    legend_y += legend_spacing;

    cv::putText(combined_visualization, "Green: 3x3 obstacles (0-50m, 0.6m)",
                cv::Point(legend_x, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    legend_y += legend_spacing;

    cv::putText(combined_visualization, "Blue: 5x5 obstacles (50-100m, 1.0m)",
                cv::Point(legend_x, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    // 显示结果
    if(params->b_show_obstacle_detection) {
        cv::namedWindow("Combined Obstacle Detection (0-100m forward)", cv::WINDOW_NORMAL);
        cv::imshow("Combined Obstacle Detection (0-100m forward)", combined_visualization);
        cv::waitKey(1);
    }

    // std::cout << "统一可视化完成，显示范围：前方0-100m, 左右±25m" << std::endl;
}
