#ifndef TOOL_H
#define TOOL_H

#include "CoordConverter.h"
#include "CommonStruct.h"
#include "CInterface.h"

struct SatelliteData 
{
    cv::Mat semantic_map;      // HxWx4 (BGR + ClassID)
    cv::Mat tdf_map;           // HxWxN (Distance fields)
    cv::Mat satellite_map;     // HxWx3 (RGB/BGR Satellite Image) // New Field
    cv::Mat geo_map;           // HxWx3 (Geometric fields)
    std::vector<double> geo_bounds; // [Lon1, Lat1, Lon2, Lat2]
    double origin_x = 0.0;
    double origin_y = 0.0;
    double resolution = 0.0;
    int zone_num = 0;
    bool is_valid = false;
};

class Tool
{ 
public:
    Tool();
    bool LoadSatelliteNpz(const std::string& npz_path, SatelliteData& out_data);  //获取卫星数据

    double3D GetBase(const nav_msgs::Odometry *odom ,WORLD_POINT Globalpoint);
    WORLD_POINT LocalToGlobal(const nav_msgs::Odometry *odom,double3D BasePoint);
   
};




#endif // TOOL_H