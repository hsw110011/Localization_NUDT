#include "map/lidar_map.h"
// LocalPose方法
LidarMap::LidarMap()
{
}

void LidarMap::process(const std::vector<PointXYZRGBValid> &colored_car_points, const self_state::LocalPose &body_pose)
{
    // body_pose信息初始化
    double dr_theta = body_pose.dr_heading;
    double sindt = sin(dr_theta);
    double cosdt = cos(dr_theta);
    double dx = body_pose.dr_x;
    double dy = body_pose.dr_y;

    // double ego_speed=sqrt(body_pose.speed_x*body_pose.speed_x+body_pose.speed_y*body_pose.speed_y+body_pose.speed_y*body_pose.speed_y);
    // ego_speed=std::max(0.0,ego_speed-5.0);
    // y_max=std::max(25.0,50-ego_speed*5.0);
    // x_max=std::min(50.0,40.0+ego_speed*5.0);
    // 遍历点云，创建栅格
    std::unordered_map<std::pair<int, int>, LidarCell, lidarhash<std::pair<int, int>>> temp_lidar_cells;

    auto start_time1 = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < colored_car_points.size(); ++i){
        const auto& point = colored_car_points[i];
        if(point.x < -x_max || point.x > x_max ||
            point.y < -y_max || point.y > y_max ||
            // point.x*point.x+point.y*point.y>x_max*x_max ||
            point.z > z_max )
        {
            continue;
        }

        // 计算全局坐标
        double global_x = cosdt * point.x - sindt * point.y + dx;
        double global_y = sindt * point.x + cosdt * point.y + dy;
        // 全局栅格索引
        int global_col = static_cast<int>(std::floor(global_x * inv_grid_size));
        int global_row = static_cast<int>(std::floor(global_y * inv_grid_size));

        auto it = temp_lidar_cells.find({global_col, global_row});
        if(it == temp_lidar_cells.end()){
            // 新栅格
            LidarCell new_cell{true, point.z, point.z, 0.0, 0.0, cv::Vec3b(point.b, point.g, point.r)};
            new_cell.has_valid_color = point.has_rgb;  // 设置颜色有效性
            temp_lidar_cells[{global_col, global_row}] = new_cell;
        }
        else{
            // 已有栅格
            
            it->second.temp_min_height = std::min(it->second.temp_min_height, static_cast<double>(point.z));

            // 如果当前点比已有的最高点更高，则使用当前点的RGB颜色
            if(point.z > it->second.max_height) {
                cv::Vec3b point_rgb(point.b, point.g, point.r);
                it->second.rgb = point_rgb;
                it->second.max_height = point.z;  // 更新最高点
                it->second.has_valid_color = point.has_rgb;  // 设置颜色有效性
            }
        }
    }
    // 在邻域中查找地面基准高度
    int neighbor_size = 1;
    for(auto& [current_key, current_cell] : temp_lidar_cells){
//        double neighbor_max = current_cell.max_height;
        double neighbor_min = current_cell.temp_min_height;
        
        // 遍历3x3邻域
        for(int cell_dx = -neighbor_size; cell_dx <= neighbor_size; ++cell_dx) {
            for(int cell_dy = -neighbor_size; cell_dy <= neighbor_size; ++cell_dy) {
                auto neighbor_key = std::make_pair(
                    current_key.first + cell_dx,
                    current_key.second + cell_dy
                );
                
                // 查找相邻栅格
                auto neighbor_it = temp_lidar_cells.find(neighbor_key);
                if(neighbor_it != temp_lidar_cells.end()) {
//                    neighbor_max = std::max(neighbor_max, neighbor_it->second.max_height);
                    neighbor_min = std::min(neighbor_min, neighbor_it->second.temp_min_height);
                }
            }
        }
        current_cell.min_height=neighbor_min;
        current_cell.max_height=current_cell.min_height;
        current_cell.height_diff=0;
    }

    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1);
    //     std::cout << "time 1: " << duration1.count() << " ms" << std::endl;

    // 计算栅格高度差
    auto start_time2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < colored_car_points.size(); ++i){
        const auto& point = colored_car_points[i];
        if(point.x < -x_max || point.x > x_max ||
        point.y < -y_max || point.y > y_max ||
        // point.x*point.x+point.y*point.y>x_max*x_max ||
        point.z > z_max )                continue;
        // 计算全局坐标
        double global_x = cosdt * point.x - sindt * point.y + dx;
        double global_y = sindt * point.x + cosdt * point.y + dy;
        // 全局栅格索引
        int global_col = static_cast<int>(std::floor(global_x * inv_grid_size));
        int global_row = static_cast<int>(std::floor(global_y * inv_grid_size));

        auto it = temp_lidar_cells.find({global_col, global_row});
        double temp_height_diff=point.z-it->second.min_height;
        //高度差小于车辆高度的点作为障碍
        if(temp_height_diff<2.5)
        {
            it->second.max_height = std::max(it->second.max_height, static_cast<double>(point.z));
            it->second.height_diff=it->second.max_height-it->second.min_height;
        }
    }
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    //     std::cout << "time2: " << duration2.count() << " ms" << std::endl;

    //更新历史地图中当前传感器范围内的地图信息
    auto start_time3 = std::chrono::high_resolution_clock::now();    
    double tan_60=tan(PI/3);
    //遍历传感器感知范围内所有栅格
    for(double car_x = 4.5 + grid_size/2; car_x < 50; car_x += grid_size){
        double temp_y=(car_x-4.0)*tan_60;
        temp_y=std::max(temp_y,15.0);
        for(double car_y = -temp_y + grid_size/2; car_y < temp_y; car_y += grid_size){
            double cell_global_x = car_x * cosdt - car_y * sindt + dx;
            double cell_global_y = car_x * sindt + car_y * cosdt + dy;
            int global_col = static_cast<int>(std::floor(cell_global_x * inv_grid_size));
            int global_row = static_cast<int>(std::floor(cell_global_y * inv_grid_size));
            auto temp_it = temp_lidar_cells.find({global_col, global_row});
            if(temp_it != temp_lidar_cells.end()){
                //在当前观测中该栅格有值
                auto it = lidar_cells.find({global_col, global_row});
                if (it != lidar_cells.end()) {
                    //当前栅格在历史中已经存在，进行累积，并均值滤波
                    auto& current_cell = it->second;
                    current_cell.addData(temp_it->second.height_diff);
                    current_cell.height_diff = current_cell.getAverageHeightDiff();

                    // RGB融合：根据颜色有效性进行智能融合
                    if(temp_it->second.has_valid_color && current_cell.has_valid_color) {
                        // 两个栅格都有有效颜色，进行线性空间融合
                        auto toLinear = [](uint8_t val) -> double {
                            double normalized = val / 255.0;
                            return normalized * normalized;  // 简化的平方函数近似伽马校正
                        };

                        auto toSRGB = [](double linear) -> uint8_t {
                            return static_cast<uint8_t>(std::clamp(sqrt(linear) * 255.0, 0.0, 255.0));
                        };

                        // 在线性空间进行RGB融合（给新观测更高权重）
                        double weight_new = 0.6;  // 新观测权重
                        double weight_old = 1.0 - weight_new;

                        double r_old_linear = toLinear(current_cell.rgb[2]);
                        double g_old_linear = toLinear(current_cell.rgb[1]);
                        double b_old_linear = toLinear(current_cell.rgb[0]);

                        double r_new_linear = toLinear(temp_it->second.rgb[2]);
                        double g_new_linear = toLinear(temp_it->second.rgb[1]);
                        double b_new_linear = toLinear(temp_it->second.rgb[0]);

                        double r_fused = r_old_linear * weight_old + r_new_linear * weight_new;
                        double g_fused = g_old_linear * weight_old + g_new_linear * weight_new;
                        double b_fused = b_old_linear * weight_old + b_new_linear * weight_new;

                        current_cell.rgb = cv::Vec3b(toSRGB(b_fused), toSRGB(g_fused), toSRGB(r_fused));
                    }
                    else if(temp_it->second.has_valid_color && !current_cell.has_valid_color) {
                        // 新栅格有颜色，历史栅格无颜色，使用新颜色
                        current_cell.rgb = temp_it->second.rgb;
                        current_cell.has_valid_color = true;
                    }
                    // 如果新栅格无颜色，保持历史栅格的颜色不变

                    // 更新最高点信息
                    if(temp_it->second.max_height > current_cell.max_height) {
                        current_cell.max_height = temp_it->second.max_height;
                    }
                }
                else{
                    //当前栅格存在，但历史中没有
                    lidar_cells[{global_col, global_row}] = temp_it->second;
                }
            }
            else{
                //在当前观测中该栅格没有值
                auto it = lidar_cells.find({global_col, global_row});
                if (it != lidar_cells.end() && car_x > 8) {
                    //在传感器感知范围内没有看到的障碍按指数衰减
                    auto& current_cell = it->second;
                    current_cell.addData(current_cell.height_diff*0.5); 
                    current_cell.height_diff = current_cell.getAverageHeightDiff();
                }
                else{
                    //该栅格在到目前为止一直没有观测，直接跳过
                    continue;
                }
            }
        }
    }
    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
    //     std::cout << "time3: " << duration3.count() << " ms" << std::endl;

    // 绘制累计地图
    auto start_time4 = std::chrono::high_resolution_clock::now();
    const int IMG_SIZE = 500; // 500x500像素
    const double MAP_RANGE = 50.0; // 50米范围（可根据需求调整）
    const double SCALE_FACTOR = IMG_SIZE / (2 * MAP_RANGE); // 缩放因子
    // cv::Mat color_map = cv::Mat::zeros(rows, cols, CV_8UC3);
    //初始化地图
    has_data = cv::Mat::zeros(rows, cols, CV_8UC1);
    height_map = cv::Mat::zeros(rows, cols, CV_32FC1);
    rgb_map = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::Mat min_height_map = cv::Mat::zeros(rows, cols, CV_32FC1);
    std::vector<std::pair<int, int>> cells_to_remove;
    // 自车全局位置
    const double self_global_x = body_pose.dr_x;
    const double self_global_y = body_pose.dr_y;
    const double cos_theta = cos(body_pose.dr_heading);
    const double sin_theta = sin(body_pose.dr_heading);
    for(const auto& [key, cell] : lidar_cells){
        int global_col = key.first;
        int global_row = key.second;
        double cell_global_x = (global_col + 0.5) * grid_size;
        double cell_global_y = (global_row + 0.5) * grid_size;
        double dx = cell_global_x - self_global_x;
        double dy = cell_global_y - self_global_y;

    //原先写法，不抖动
     double car_x = cos_theta * dx + sin_theta * dy;
     double car_y = -sin_theta * dx + cos_theta * dy;
      if(std::abs(car_x) > MAP_RANGE || std::abs(car_y) > MAP_RANGE)
      {
          cells_to_remove.push_back(key);
          continue;
      }
      // 在车体坐标系下绘制地图
      int neighbor_cell = 1;
      int car_col = static_cast<int>((-car_y * SCALE_FACTOR) + IMG_SIZE/2);
      int car_row = static_cast<int>((-car_x * SCALE_FACTOR) + IMG_SIZE/2);
        
        //准备用opencv图旋转写法，还没有写好，会抖动
        // if(std::abs(dx) > MAP_RANGE || std::abs(dy) > MAP_RANGE)
        // {
        //     cells_to_remove.push_back(key);
        //     continue;
        // }
        // // 在车体坐标系下绘制地图
        // int neighbor_cell = 0;
        // int car_col = static_cast<int>((-dy * SCALE_FACTOR) + IMG_SIZE/2);
        // int car_row = static_cast<int>((-dx * SCALE_FACTOR) + IMG_SIZE/2);


        for(int i= -neighbor_cell; i<=neighbor_cell; i++){
            for(int j= -neighbor_cell; j<=neighbor_cell; j++){
                int col = car_col + i;   
                int row = car_row + j;
                if(col >= 0 && col < IMG_SIZE && row >= 0 && row < IMG_SIZE)
                {
                    // 如果当前栅格没有数据
                    if(!has_data.at<uchar>(row, col))
                    {
                        // uint8_t r = 0, g = 0, b = 0;
                        // params->setColorByHeight(cell.height_diff, r, g, b);
                        // color_map.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);
                        // cv::circle(color_map, cv::Point(col, row), 1, cv::Scalar(b, g, r), -1);
                        has_data.at<uchar>(row, col) = 1;
                        height_map.at<float>(row, col) = cell.height_diff;
                        min_height_map.at<float>(row, col) = cell.min_height + 0.5;
                        // 填充RGB数据
                        rgb_map.at<cv::Vec3b>(row, col) = cell.rgb;
                    }
                    else if(has_data.at<uchar>(row, col))
                    {
                        // 如果当前栅格有数据，就赋予更大高度差
                        if(cell.height_diff > height_map.at<float>(row, col))
                        {
                            height_map.at<float>(row, col) = cell.height_diff;
                        }
                        // 如果当前栅格有数据，就赋予最小高度
                        if(cell.min_height < min_height_map.at<float>(row, col))
                        {
                            min_height_map.at<float>(row, col) = cell.min_height + 0.5;
                        }
                        // RGB策略：使用最高点的RGB颜色
                        // 如果当前栅格的最高点比已记录的更高，则使用当前栅格的RGB
                        if(cell.height_diff > height_map.at<float>(row, col)) {
                            rgb_map.at<cv::Vec3b>(row, col) = cell.rgb;
                        }
                    }
                }
            }
        }
    }
    for(const auto& key : cells_to_remove){
        lidar_cells.erase(key);
    }
    auto end_time4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4 - start_time4);
    //     std::cout << "time4: " << duration4.count() << " ms" << std::endl;

    auto start_time5 = std::chrono::high_resolution_clock::now(); 
    params->generateColorMap(height_map, has_data, "Lidar Height Map");
    params->generateRGBMap(rgb_map, has_data, "Lidar RGB Map");
    auto end_time5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5);
    //     std::cout << "time5: " << duration5.count() << " ms" << std::endl;
    // params->generateColorMap(min_height_map, has_data, "Lidar Min Height Map");

}


