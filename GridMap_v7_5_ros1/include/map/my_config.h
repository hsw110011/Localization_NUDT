#pragma once
#include "ros/ros.h"
#include "self_state/LocalPose.h"
#include "self_state/LidarLocalPose.h"
#include "ars548_msg/DetectionList.h"
#include "ars548_msg/detections.h"
#include <sensor_msgs/Imu.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/viz.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <regex>
#define PI 3.14159265
using namespace std;

namespace LM {
  struct EIGEN_ALIGN16 LidarPoint {
    PCL_ADD_POINT4D;
    float intensity;       // 对应 msg 中的 intensity
    uint16_t ring;         // 激光线号
    float time;            // 点的采样时间（对应ROS消息中的"time"字段）
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

POINT_CLOUD_REGISTER_POINT_STRUCT(LM::LidarPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (float, time, time)  // 字段名必须与ROS消息中的"time"完全一致！
)

// IMU数据结构
struct IMUData {
    double timestamp;           // 时间戳（秒）
    double angular_vel[3];      // 角速度 [x, y, z] (rad/s)
    double linear_acc[3];       // 线性加速度 [x, y, z] (m/s²)

    IMUData() : timestamp(0.0) {
        for(int i = 0; i < 3; i++) {
            angular_vel[i] = 0.0;
            linear_acc[i] = 0.0;
        }
    }

    IMUData(double t, const sensor_msgs::Imu& imu_msg) : timestamp(t) {
        angular_vel[0] = imu_msg.angular_velocity.x;
        angular_vel[1] = imu_msg.angular_velocity.y;
        angular_vel[2] = imu_msg.angular_velocity.z;
        linear_acc[0] = imu_msg.linear_acceleration.x;
        linear_acc[1] = imu_msg.linear_acceleration.y;
        linear_acc[2] = imu_msg.linear_acceleration.z;
    }
};

class Radar2Camera
{
public:
    Radar2Camera() { setZero(); }
    ~Radar2Camera() {}

    Eigen::Matrix3f            camera_K;     // K
    Eigen::Matrix<float, 1, 5> camera_KP;    // K1K2P1P2K3
    Eigen::Matrix4f            RT;           // K1K2P1P2K3
    Eigen::Matrix4f            P;

    void setZero(){
        RT.setZero();
        P.setZero();
        camera_K.setZero();
        camera_KP.setZero();
    }
};

// 扩展的点云结构，包含RGB有效性标志
struct PointXYZRGBValid {
    float x, y, z;
    uint8_t r, g, b;
    bool has_rgb;  // 标志位：是否有有效的RGB信息

    // 构造函数
    PointXYZRGBValid() : x(0), y(0), z(0), r(128), g(128), b(128), has_rgb(false) {}
    PointXYZRGBValid(float x_, float y_, float z_) : x(x_), y(y_), z(z_), r(128), g(128), b(128), has_rgb(false) {}
    PointXYZRGBValid(float x_, float y_, float z_, uint8_t r_, uint8_t g_, uint8_t b_, bool has_rgb_)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), has_rgb(has_rgb_) {}
};

class my_config
{
public:
    std::string choose_car;
    std::map<std::string,cv::Mat> img_map; //储存图像

    //LM
    std::string LM_Camera_Topic = "/sensor/image/ar0231_front/compressed"; //LM前视相机
    std::string LM_Lidar_Topic = "";
    std::string LM_Lidar_Topic1 = "/sensor/RS128Points"; //LM激光雷达1
    std::string LM_Lidar_Topic2 = "/sensor/RS128Points_tztek"; //LM激光雷达2
    std::string topicRadar548FrontData = "/sensor/radar/ars548_front/detection_list"; //前向雷达
    std::string topicRadar548LeftData = "/sensor/radar/ars548_leftfront/detection_list"; //左向雷达
    std::string topicRadar548RightData = "/sensor/radar/ars548_rightfront/detection_list"; //右向雷达
    //HM
    std::string HM_Camera_Topic = "/sensor/image/ar0231_front/compressed"; //HM前视相机
    std::string HM_Lidar_Topic = "/sensor/RS128Points_tztek"; //HM激光雷达

    //std::string HM_Camera_Topic = "/video/camera_front_mid_image_raw/compressed"; //HM前视相机
    //std::string HM_Lidar_Topic = "/ls_front/lslidar_point_cloud_front"; //HM激光雷达
    // std::string HM_Lidar_Topic = "/perception/preprocessing/lidar/downsample"; //HM激光雷达
    
    //std::string HM_Camera_Topic = "/video/camera_front_mid_image_raw/compressed"; //HM前视相机
    //std::string HM_Lidar_Topic = "/velodyne_points"; //HM激光雷达
    
    //new_LM
    std::string new_LM_Camera_Topic = "/GreenMengShi_HNHT/sensor/camera/normal/front/compressed"; //new_LM前视相机
    std::string LM_M1_Lidar_Topic = "/sensor/RSM1Points"; //new_LM M1 激光雷达
    std::string LM_BP_Lidar_Topic = "/sensor/RSBPPoints"; //new_LM BP 激光雷达

    std::string topicSelfLocalPose = "/self_state/LocalPose"; //自车位姿
    std::string topicLidarLocalPose = "/self_state/LidarLocalPose"; //激光雷达位姿
    std::string imu_topic = "/imu/data"; // IMU话题名称
    std::string lidar_odometry_topic = "/sensor/Odometry"; // 激光雷达里程计话题

//    Eigen::Matrix4f T_LM_lidar2car = (Eigen::Matrix4f() <<
//                                  0.999677, 0.011081, -0.0228663, 2.75,
//                                  -0.0109223, 0.999915, 0.00705384, 0,
//                                  0.0229425, -0.00680182, 0.999714, 1.70,
//                                  0, 0, 0, 1).finished();
//    Eigen::Matrix4f T_HM_rotate = (Eigen::Matrix4f() << 0, 1, 0, 0,
//                                             -1, 0, 0, 0,
//                                             0, 0, 1, 0,
//                                             0, 0, 0, 1).finished();
//    Eigen::Matrix4f T_HM_lidar2car = (Eigen::Matrix4f() << 0.988771, 0.000747188, 0.149436, 2.25,
//                                             0, 0.999988, -0.00499998, 0,
//                                             -0.149438, 0.00494383, 0.988759, 1.65,
//                                                 0, 0, 0, 1).finished();
    Eigen::Matrix4f T_BP_2_M1 = (Eigen::Matrix4f() << 0.999951,0.00865909,0.00491691,0.000590029,
                                                    -0.0086504,0.999961,-0.00178514,0.00127456,
                                                    -0.0049322,0.00174252,0.999986,0.0252275,
                                                 0, 0, 0, 1).finished();
    Eigen::Matrix4f T_New_LM_lidar2car = (Eigen::Matrix4f() <<
                                          1,0,0,4.32,
                                          0,1,0,0,
                                          0,0,1,1.4,
                                          0, 0, 0, 1).finished();
    // image calib
    Radar2Camera camera_front;
    // LM
    Eigen::Matrix4f T_LM_lidar2car = Eigen::Matrix4f::Identity();
    // HM
    Eigen::Matrix4f T_HM_rotate = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_HM_l2c = Eigen::Matrix4f::Identity();
    std::vector<std::string> Lidar2Car_calib_file;
    std::vector<std::string> camera_calib_file;





public:
    int IMAGE_HEIGHT = 800;
    int IMAGE_WIDTH = 800;
    bool b_LM_Lidar_Type;

    // 地图旋转方法选择：true=OpenCV旋转，false=数学变换
    bool use_opencv_rotation = false;

    // 显示控制开关
    bool b_show_proj = false;                // 是否显示投影图像
    bool b_show_camera_front = false;       // 是否显示原始相机图像
    bool b_show_undistort = true;           // 是否显示相机去畸变后的图像
    uint8_t f_show_bev_color = 1;           // 是否显示lidar bev，0:不显示 1:only color points 2:all points
    bool b_show_rgb_map = true;             // 是否显示RGB Map
    bool b_gridmap_use_rgb = false;         // GridMap颜色模式：false=高度着色，true=RGB着色

    // GridMap相关显示开关
    bool b_show_height_diff = true;         // 是否显示高度差图
    bool b_show_slope_map = true;           // 是否显示坡度图
    bool b_show_terrain_roughness = true;   // 是否显示地形粗糙度
    bool b_show_terrain_slope = true;       // 是否显示地形坡度
    bool b_show_terrain_labels = true;      // 是否显示地形标签
    bool b_show_colormap_vehicle = true;    // 是否显示车体坐标系颜色地图
    bool b_show_obstacle_detection = true;  // 是否显示障碍物检测可视化
    bool b_show_lidar_points = true;        // 是否显示激光雷达点云可视化

    // GridMap处理参数
    bool b_enable_point_z_filter = false;   // 是否过滤超过point_z_max的点，默认不过滤高度
    double point_z_max = 2.5;               // 点云高度过滤上限，仅b_enable_point_z_filter为true时生效
    double terrain_roughness_scale = 2.5;   // TerrainMap roughness归一化尺度，单位m

    // LiDAR BEV surface map parameters.
    bool b_enable_lidar_bev = true;
    bool b_lidar_bev_use_odometry = true;
    bool b_lidar_bev_use_ransac_ground = true;
    bool b_show_lidar_bev_layers = true;
    std::string lidar_bev_pose_source = "localpose"; // localpose, odometry, body
    std::string lidar_bev_topic = "/lidar_bev/grid_map";
    std::string lidar_bev_frame_id = "map";
    std::string lidar_bev_body_frame_id = "body";
    double lidar_bev_resolution = 0.2;
    double lidar_bev_map_size_x = 100.0;
    double lidar_bev_map_size_y = 100.0;
    double lidar_bev_point_min_z = -5.0;
    double lidar_bev_point_max_z = 80.0;
    double lidar_bev_update_rate_hz = 10.0;
    bool b_lidar_bev_enable_ego_filter = true;
    double lidar_bev_ego_box_min_x = -2.0;
    double lidar_bev_ego_box_max_x = 2.0;
    double lidar_bev_ego_box_min_y = -2.0;
    double lidar_bev_ego_box_max_y = 2.0;
    double lidar_bev_near_inner_radius = 2.0;
    double lidar_bev_near_outer_radius = 15.0;
    double lidar_bev_ground_candidate_min_z = -3.0;
    double lidar_bev_ground_candidate_max_z = 1.0;
    double lidar_bev_ground_ransac_distance = 0.18;
    double lidar_bev_ground_max_plane_tilt_deg = 25.0;
    double lidar_bev_ground_fallback_quantile = 0.35;
    double lidar_bev_ground_front_half_angle_deg = 7.5;
    bool b_lidar_bev_ground_require_forward = true;
    double lidar_bev_ground_failure_fallback_z = 0.0;
    int lidar_bev_ground_min_points = 30;
    double lidar_bev_height_quantile = 0.90;
    int lidar_bev_cell_max_points = 500;
    int lidar_bev_debug_window_stride = 2;
    int lidar_bev_edge_min_valid_neighbors = 4;

    // IMU去畸变相关配置
    bool b_enable_imu_undistortion = true; // 是否启用IMU去畸变功能
    double imu_deskew_time_ratio = 0.5; // 去畸变参考时间比例（0.0=开始，0.5=中间，1.0=结束）
    bool b_enable_imu_visualization = false; // 是否启用IMU去畸变可视化
    size_t imu_lidar_total_rows = 400; // 激光雷达总行数
    size_t imu_lidar_total_cols = 500; // 激光雷达总列数
    void Draw_Points_Whole(std::vector<pcl::PointXYZ> &lidar_points);
    void generateColorMap(const cv::Mat &height_diff, const cv::Mat &has_data, std::string window_name);
    void generateRGBMap(const cv::Mat &rgb_data, const cv::Mat &has_data, std::string window_name);
    void setColorByHeight(float z_value, uint8_t &r, uint8_t &g, uint8_t &b);

    // 鼠标回调相关
    static void onMouse(int event, int x, int y, int flags, void* userdata);

    // 存储鼠标点击信息的结构体
    struct MouseClickInfo {
        cv::Point click_point;
        float height_value;
        bool is_valid;
        cv::Mat height_map;
        cv::Mat has_data;
        cv::Mat display_image;
        std::string window_name;

        MouseClickInfo() : is_valid(false) {}
    };
    void LM_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points);
    void HM_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points);
    void new_LM_M1_trans_lidar2car(std::vector<pcl::PointXYZ> &lidar_points, std::vector<pcl::PointXYZ> &car_points);
    void new_LM_BP_2_M1_lidar(std::vector<pcl::PointXYZ> &BP_lidar_points, std::vector<pcl::PointXYZ> &M1_lidar_points);
    void trans_color_lidar2car(const std::vector<PointXYZRGBValid> &colored_lidar_points, std::vector<PointXYZRGBValid> &colored_car_points);
    Eigen::Vector2d trans_car_to_global(Eigen::Vector2d point_car, const self_state::LocalPose &body_pose) const;
    Eigen::Vector2d trans_global_to_car(Eigen::Vector2d point_global, const self_state::LocalPose& body_pose) const;
    Eigen::Matrix3d eulerAnglesToRotationMatrix(double roll, double pitch, double yaw);
    Eigen::Vector3d trans_car_to_global_3d(const Eigen::Vector3d &point_car, const self_state::LidarLocalPose &lidar_localpose);
    Eigen::Vector3d trans_global_to_car_3d(const Eigen::Vector3d &point_global, const self_state::LidarLocalPose &lidar_localpose);
    Eigen::Matrix4f read_ini(std::string &radar_type);
    Radar2Camera read_CameraParaV2(std::string &camera_type);
    Eigen::Matrix4f solveRT(Eigen::Matrix3f &K, Eigen::Matrix4f &P);
    void readPara();
};
extern std::shared_ptr<my_config> params;
