#include "CInterface.h"
#include <cv_bridge/cv_bridge.h>
#include <CoordConverter.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> // For std::setprecision
#include "cnpy.h"
#include "Tool.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
    /*输入都是角度，在函数里做角度->弧度*/
int main(int argc, char **argv)
{
    // 如果你只想测试读取，可以先注释掉 ROS 初始化，或者留着不使用
    ros::init(argc, argv, "localization_node");

    //类对象创建
    ros::NodeHandle nh;
    CInterface interface(nh);
    InputData input; 
    Tool tool;
    std::string dem_path = "/home/hsw/catkin_ws/doc/miluo_dsm.tif"; // 替换你的实际路径
    DemTiffData dem_data = tool.LoadDemTiff(dem_path);
    CoordConverter coord_converter(dem_data);


    cv::namedWindow("Visual Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Visual Map", 800, 800);   
    cv::Mat color_map;
    // 从原始高度数据重新生成灰度图，以便应用热度图 (COLORMAP_HOT)
    cv::Mat gray_map;
    if(!dem_data.raw_elevation_map.empty()) 
    {
         double scale = 255.0 / (dem_data.max_height - dem_data.min_height);
         dem_data.raw_elevation_map.convertTo(gray_map, CV_8UC1, scale, -dem_data.min_height * scale);
         cv::applyColorMap(gray_map, color_map, cv::COLORMAP_HOT);
    } 
    else 
    {
        color_map = dem_data.visual_map.clone();
    }
    
    cv::imshow("Visual Map", color_map);
    cv::waitKey(1); // 等待按键关闭窗口
    cv::Mat vis_track = color_map.clone();

    WORLD_POINT global_point;


    ros::Rate loop_rate(10); // 10 Hz
    while (ros::ok())
    {
        ros::spinOnce(); // 处理回调函数
        interface.ConvertToLocalData(&input);
        
        if(input.GlobalPose_refreshflag)
        {
            global_point.BLH.Lon = input.GlobalPose->longitude;
            global_point.BLH.Lat = input.GlobalPose->latitude;
            global_point.gauss = coord_converter.wgs84_to_gauss(global_point.BLH.Lon, global_point.BLH.Lat);
            global_point.pixel = coord_converter.wgs84_to_pixel(global_point.BLH.Lon, global_point.BLH.Lat);
            global_point.heading = input.GlobalPose->azimuth;
            cv::circle(vis_track, cv::Point(global_point.pixel.x, global_point.pixel.y), 5, cv::Scalar(255, 255, 255), -1);
            cv::namedWindow("Track", cv::WINDOW_NORMAL);
            cv::resizeWindow("Track", 800, 800);
            cv::imshow("Track", vis_track);
            cv::waitKey(1);
        }



        loop_rate.sleep();
    }


    return 0;
}