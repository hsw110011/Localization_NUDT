#ifndef TOOL_H
#define TOOL_H

#include "CommonStruct.h"
#include "CInterface.h"
#include <gdal_priv.h>
#include <iostream>
#include <vector>

// 定义 GeoTIFF 数据结构体
struct DemTiffData 
{
    // 1. 可视化的图像 (RGB/BGR if colormapped, or 0-255 grayscale)
    cv::Mat visual_map;

    // 1.1 原始高程数据 (Float32)，用于计算
    cv::Mat raw_elevation_map; 

    // 2. 左上右下的经纬度范围 (Left, Top, Right, Bottom)
    // 顺序: [MinLon(西), MaxLat(北), MaxLon(东), MinLat(南)]
    std::vector<double> geo_bounds; 

    // 3. 分辨率 (X, Y)
    double resolution_x = 0.0;
    double resolution_y = 0.0;
    double resolution = 0.0; // 统一分辨率 (通常取 resolution_x)

    // 4. 高度最大最小值
    double min_height = 0.0;
    double max_height = 0.0;

    // 5. UTM 带号
    int zone_num = 0;

    // 标志位
    bool is_valid = false;
};

class Tool
{ 
public:
    Tool();
    static DemTiffData LoadDemTiff(const std::string& tiff_path);  //获取卫星数据

    double3D GetBase(const nav_msgs::Odometry *odom ,WORLD_POINT Globalpoint);
    WORLD_POINT LocalToGlobal(const nav_msgs::Odometry *odom,double3D BasePoint);
   
};




#endif // TOOL_H