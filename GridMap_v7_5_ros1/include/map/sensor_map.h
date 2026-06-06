#pragma once
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "include/map/my_config.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <sensor_msgs/CompressedImage.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>
#include <thread>
#include "include/map/lidar_map.h"
#include "include/map/gridmap_handle.h"
#include "include/map/gridmap_handle_v2.h"
#include "include/map/gridmap_handle_v3.h"
#include "include/map/gridmap_handle_v4.h"
#include "include/map/lidar_bev_builder.h"
#include "include/map/odom_stabilizer.h"
#include "include/map/SensorDataCollection.h"
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <cmath>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include "include/map/image_proj.h"
//#include <grid_map_cv/grid_map_cv.hpp>

class SensorMap
{
public:
    SensorMap() {};
    ~SensorMap() {};
    void Init();
    void Odemetry_callback(const nav_msgs::Odometry::ConstPtr& msg);
    void get_rostopic_state();
    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void camera_Front_callback(const sensor_msgs::CompressedImageConstPtr &msg);
    void body_pose_callback(const self_state::LocalPose &msg);
    void lidar_localpose_callback(const self_state::LidarLocalPose &msg);

    void imu_callback(const sensor_msgs::Imu &msg);  // 新增IMU回调
    void M1_lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void BP_lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void handle_points();
    void output();

    // 运动补偿相关方法
    std::deque<self_state::LocalPose> getRecentPoses(double time_window = 1.0);


    void Pub_Points(std::vector<pcl::PointXYZ> &car_points);

    void accumulateAndPublishPoints(const std::vector<pcl::PointXYZ> &car_points, const Eigen::Isometry3d &T_odmetry);

    void fillGridMapWithHeight();
    void fillGridMapWithRGB();

    void publishGridMap();
    void Draw_img_map();
    void Draw_lidar_bev(const std::vector<PointXYZRGBValid> &colored_points);
    LidarBevBuilder::OdometryState getLatestLidarOdometry();
    LidarBevBuilder::OdometryState getCurrentBevPose();
    Pose6D poseFromLocalPose(const self_state::LocalPose& pose) const;
    Eigen::Isometry3d pose6DToIsometry(const Pose6D& pose) const;
    cv::Mat accum_map;
    cv::Mat accum_visited;
    std::vector<std::vector<pcl::PointXYZ>> lidar_vec_points;
    SensorDataQueue<std::vector<pcl::PointXYZ>> lidar_msg;
    SensorDataQueue<std::vector<pcl::PointXYZ>> M1_lidar_msg;
    SensorDataQueue<std::vector<pcl::PointXYZ>> BP_lidar_msg;
    SensorDataQueue<self_state::LocalPose> local_pose_msg;
    SensorDataQueue<Eigen::Isometry3d> lidar_odemetry_msg;

    std::deque<IMUData> imu_data_queue;  // IMU数据队列

    Eigen::Isometry3d T_odmetry; //激光雷达里程计

    // 累积点云相关
    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;  // 累积的全局点云
    int frame_counter = 0;  // 帧计数器

    static std::mutex img_map_mutex;
    std::mutex lidar_msg_mutex;
    std::mutex local_pose_msg_mutex;
    std::mutex odometry_mutex_;
    std::mutex odom_stabilizer_mutex_;
    

private:
    LidarMap lidar_map;
    GridMapHandler grid_map_handler;
    GridMapHandler_v2 grid_map_handler_v2;
    GridMapHandler_v3 grid_map_handler_v3;
    GridMapHandler_v4 grid_map_handler_v4;
    LidarBevBuilder lidar_bev_builder;
    IMAGE_PROJPtr image_proj;

    //订阅消息
    ros::NodeHandle node;
    ros::Subscriber LM_Lidar_sub;
    ros::Subscriber HM_Lidar_sub;
    ros::Subscriber camera_front_sub;
    ros::Subscriber local_pose_sub;
    ros::Subscriber lidar_localpose_sub;

    ros::Subscriber imu_sub;  // 新增IMU订阅者
    ros::Subscriber M1_Lidar_sub;
    ros::Subscriber BP_Lidar_sub;
    ros::Subscriber Odemetry_sub;

    //发布消息
    ros::Publisher car_points_pub;
    ros::Publisher accumulated_points_pub;  // 发布累积点云
    ros::Publisher grid_map_pub_;  // 用来发布消息
    ros::Publisher terrain_map_pub_;
    ros::Publisher color_map_pub_;
    ros::Publisher obstacle_pub_;
    ros::Publisher lidar_bev_pub_;

    //变量
    grid_map::GridMap grid_map_;  // 用来存储栅格地图
    int64_t current_time;

    //选择车的类型，LM,HM,new_LM
    std::string choose_car;
    self_state::LocalPose body_pose;    // 当前姿态数据
    self_state::LocalPose last_pose;    // 上一个姿态数据
    self_state::LocalPose history_body_pose;

    // 激光雷达LocalPose
    self_state::LidarLocalPose lidar_localpose;
    LidarBevBuilder::OdometryState latest_lidar_odometry_;
    OdomStabilizer local_pose_stabilizer_;
    OdomStabilizer lidar_odom_stabilizer_;
    std::optional<Pose6D> latest_smoothed_local_pose_;
    std::atomic<bool> local_pose_valid_{false};



    // 栅格参数
    double grid_size = 0.2;
    double x_min = -50.0, x_max = 50.0;  // x前向范围 [-50m,50m]
    double y_min = -50.0, y_max = 50.0;  // y左向范围 [-50m,50m]
    int cols = static_cast<int>((x_max - x_min) / grid_size); // 500列
    int rows = static_cast<int>((y_max - y_min) / grid_size); // 500行
    int center_col = cols / 2;  // 中心列索引250
    int center_row = rows / 2;  // 中心行索引250



    // 冻结坐标成员变量
    geometry_msgs::Pose getCurrentLocalPose() const;
    double frozen_x_ = 0.0;
    double frozen_y_ = 0.0;
    double frozen_theta_ = 0.0;

    std::atomic<bool> stop_flag{false};
    
};
