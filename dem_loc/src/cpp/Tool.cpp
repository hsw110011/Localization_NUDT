#include "Tool.h"
#include "cnpy.h"
Tool::Tool()
{
}
// C++ 原生读取特定的 NPZ 数据
DemTiffData Tool::LoadDemTiff(const std::string& tiff_path)
{
    DemTiffData out_data;
    out_data.is_valid = false; // 默认无效

    GDALDataset* poDataset = nullptr;

    try {
        // 0. 注册 GDAL 驱动 (只需调用一次，放在程序入口更好，但放这里为了独立性)
        GDALAllRegister();

        // 1. 打开文件
        poDataset = (GDALDataset*)GDALOpen(tiff_path.c_str(), GA_ReadOnly);
        if (poDataset == nullptr) {
            std::cerr << "[Error] Failed to open TIFF file: " << tiff_path << std::endl;
            return out_data;
        }

        
        int width = poDataset->GetRasterXSize();
        int height = poDataset->GetRasterYSize();
        std::cout << "[Info] TIFF Size: " << width << "x" << height << std::endl;

        // 2. 读取地理坐标转换信息 (GeoTransform)
        double adfGeoTransform[6];
        if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
            // adfGeoTransform[0] // top left x
            // adfGeoTransform[1] // w-e pixel resolution
            // adfGeoTransform[2] // 0
            // adfGeoTransform[3] // top left y
            // adfGeoTransform[4] // 0
            // adfGeoTransform[5] // n-s pixel resolution (negative value)

            out_data.resolution_x = adfGeoTransform[1];
            out_data.resolution_y = std::abs(adfGeoTransform[5]); //以此为正值方便理解
            out_data.resolution = out_data.resolution_x;

            double minLon = adfGeoTransform[0];
            double maxLat = adfGeoTransform[3];
            double maxLon = minLon + width * out_data.resolution_x;
            double minLat = maxLat + height * adfGeoTransform[5]; // 注意 [5] 通常是负数

            out_data.geo_bounds = {minLon, maxLat, maxLon, minLat};
            
            // 计算中心经度以确定 UTM 带号
            double centerLon = (minLon + maxLon) / 2.0;
            out_data.zone_num = static_cast<int>((centerLon + 180.0) / 6.0) + 1;

            std::cout << "[Info] Loaded GeoBounds: [L:" << minLon << ", T:" << maxLat 
                      << ", R:" << maxLon << ", B:" << minLat << "]" << std::endl;
            std::cout << "[Info] Loaded Resolution: " << out_data.resolution_x << std::endl;
            std::cout << "[Info] UTM Zone: " << out_data.zone_num << std::endl;
        } else {
            std::cerr << "[Warning] GeoTransform not found in TIFF." << std::endl;
        }

        // 3. 读取高程数据 & 极值
        GDALRasterBand* poBand = poDataset->GetRasterBand(1); // 读取第一个波段
        
        // 3.1 获取最大最小值
        double minMax[2];
        int bGotMin, bGotMax;
        // 先尝试获取元数据里的极值，如果没有则计算
        double min_val = poBand->GetMinimum(&bGotMin);
        double max_val = poBand->GetMaximum(&bGotMax);
        
        if( !(bGotMin && bGotMax) ){
            // 如果元数据里没有，强制扫描计算 (FALSE表示精确计算)
            GDALComputeRasterMinMax((GDALRasterBandH)poBand, FALSE, minMax);
            out_data.min_height = minMax[0];
            out_data.max_height = minMax[1];
        } else {
            out_data.min_height = min_val;
            out_data.max_height = max_val;
        }
        std::cout << "[Info] Elevation Range: " << out_data.min_height << " ~ " << out_data.max_height << std::endl;

        // 4. 读取原始数据到 cv::Mat (Raw Elevation)
        // 假设 DEM 是 float32，这是最常见的格式
        out_data.raw_elevation_map = cv::Mat(height, width, CV_32FC1);
        
        // 直接读取到 Mat 的 data 指针中，避免 memcpy
        CPLErr err = poBand->RasterIO(GF_Read, 0, 0, width, height,
                                      out_data.raw_elevation_map.data, width, height,
                                      GDT_Float32, 0, 0);

        if (err != CE_None) {
             throw std::runtime_error("RasterIO read failed.");
        }

        // 5. 生成可视化图 (Visual Mat)
        // 将 float 数据归一化到 0-255 并应用 colormap
        if (!out_data.raw_elevation_map.empty()) {
            double minVal = out_data.min_height;
            double maxVal = out_data.max_height;
            double scale = 255.0 / (maxVal - minVal);
            
            cv::Mat gray_map;
            // 方法A: 使用 OpenCV convertTo (线性变换: pixel * alpha + beta)
            // pixel -> (pixel - min) * (255 / (max-min))
            out_data.raw_elevation_map.convertTo(gray_map, CV_8UC1, scale, -minVal * scale);
            
            // 应用 Jet Colormap
            cv::applyColorMap(gray_map, out_data.visual_map, cv::COLORMAP_JET);

            std::cout << "[Info] Generated visual map: " << out_data.visual_map.size() << std::endl;
        }

        out_data.is_valid = true;
        std::cout << "[Info] Successfully loaded TIFF data." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Error] Load TIFF failed: " << e.what() << std::endl;
        out_data.is_valid = false;
    }

    // 清理 GDAL 资源
    if (poDataset) {
        GDALClose((GDALDatasetH)poDataset);
    }

    return out_data;
}

double3D Tool::GetBase(const nav_msgs::Odometry *odom , WORLD_POINT globalpoint)
{
    double3D base;
    
    double siny_cosp = 2.0 * (odom->pose.pose.orientation.w * odom->pose.pose.orientation.z + odom->pose.pose.orientation.x * odom->pose.pose.orientation.y);
    double cosy_cosp = 1.0 - 2.0 * (odom->pose.pose.orientation.y * odom->pose.pose.orientation.y + odom->pose.pose.orientation.z * odom->pose.pose.orientation.z);
    double odom_heading = atan2(siny_cosp, cosy_cosp) * RAD_TO_DEG; // CCW, 0=East

    // 1. Odom 使用 ROS 标准数学坐标系：X轴(东)为0，逆时针(CCW)为正
    // 2. Global (GPS/罗盘) 使用航向角(Azimuth)：北为0，顺时针(CW)为正
    double global_math_heading = globalpoint.heading; 

    double odom_x = odom->pose.pose.position.x;
    double odom_y = odom->pose.pose.position.y;

    base.theta = (global_math_heading - odom_heading)*DEG_TO_RAD;
    base.x = globalpoint.gauss.x-(odom_x*cos(base.theta)-odom_y*sin(base.theta));
    base.y = globalpoint.gauss.y-(odom_x*sin(base.theta)+odom_y*cos(base.theta));
    return base;                                                                //得到的是gauss坐标系下的基点
}

WORLD_POINT Tool::LocalToGlobal(const nav_msgs::Odometry *odom,double3D BasePoint)
{
    WORLD_POINT global_point;
    global_point.timeflag = odom->header.stamp.toSec() * 1000.0;
    
    double siny_cosp = 2.0 * (odom->pose.pose.orientation.w * odom->pose.pose.orientation.z + odom->pose.pose.orientation.x * odom->pose.pose.orientation.y);
    double cosy_cosp = 1.0 - 2.0 * (odom->pose.pose.orientation.y * odom->pose.pose.orientation.y + odom->pose.pose.orientation.z * odom->pose.pose.orientation.z);
    double odom_heading = atan2(siny_cosp, cosy_cosp) * RAD_TO_DEG;

    double odom_x = odom->pose.pose.position.x;
    double odom_y = odom->pose.pose.position.y;

    // 1. Transform Position
    global_point.gauss.x = odom_x* cos(BasePoint.theta)-odom_y*sin(BasePoint.theta)+ BasePoint.x;
    global_point.gauss.y = odom_x* sin(BasePoint.theta)+odom_y*cos(BasePoint.theta)+ BasePoint.y;

    // 2. Transform Heading
    // Global_Math (CCW) = Local_Math (CCW) + Theta (CCW)
    double global_math_heading = odom_heading + (BasePoint.theta)*RAD_TO_DEG;
    
    // Convert back to Azimuth (CW from North)
    // Azimuth = 90 - Math_Angle
    global_point.heading = global_math_heading;
    
    // Normalize to 0-360
    while(global_point.heading < 0) global_point.heading += 360.0;
    while(global_point.heading >= 360.0) global_point.heading -= 360.0;

    return global_point;
}

