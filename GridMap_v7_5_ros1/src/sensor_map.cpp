#include "map/sensor_map.h"
#include <geometry_msgs/Pose.h>
#include <world_state/TerrainMap.h>
#include <world_state/ColorMap.h>
#include <world_state/StaticObj.h>
#include <world_state/StaticObjArray.h>
#include <tf/transform_datatypes.h> // ROS1 的 tf 库
#include <iomanip>
#include <chrono> // 用于时间测量

using namespace std;
std::mutex SensorMap::img_map_mutex; 

namespace {
Pose6D poseFromIsometry(double timestamp, const Eigen::Isometry3d& pose)
{
    Pose6D pose6d;
    pose6d.timestamp = timestamp;
    pose6d.x = pose.translation().x();
    pose6d.y = pose.translation().y();
    pose6d.z = pose.translation().z();

    const Eigen::Vector3d ypr = pose.rotation().eulerAngles(2, 1, 0);
    pose6d.yaw = ypr(0);
    pose6d.pitch = ypr(1);
    pose6d.roll = ypr(2);
    return OdomStabilizer::NormalizePoseAngles(pose6d);
}
}  // namespace

void SensorMap::Init()
{
    params->readPara();
    choose_car = params->choose_car;
    cout<<"Init "<< "choose_car: "<< choose_car <<endl;

    // 图像-雷达投影暂时关闭，保留 image_proj 代码但不初始化/订阅相机。
    // image_proj = std::make_shared<IMAGE_PROJ>();
    if(choose_car == "LM")
    {
        //LM
        // camera_front_sub = node.subscribe(params->LM_Camera_Topic, 1, &SensorMap::camera_Front_callback, this);
        LM_Lidar_sub = node.subscribe(params->LM_Lidar_Topic, 1, &SensorMap::lidarCallback, this);
    } else if(choose_car == "HM") {
        //HM
        // camera_front_sub = node.subscribe(params->HM_Camera_Topic, 1, &SensorMap::camera_Front_callback, this);
        HM_Lidar_sub = node.subscribe(params->HM_Lidar_Topic, 1, &SensorMap::lidarCallback, this);
    } else if(choose_car == "new_LM"){
        //new_LM
        // camera_front_sub = node.subscribe(params->new_LM_Camera_Topic, 1, &SensorMap::camera_Front_callback, this);
        M1_Lidar_sub = node.subscribe(params->LM_M1_Lidar_Topic, 1, &SensorMap::M1_lidarCallback, this);
        BP_Lidar_sub = node.subscribe(params->LM_BP_Lidar_Topic, 1, &SensorMap::BP_lidarCallback, this);
    }
    local_pose_sub   =  node.subscribe(params->topicSelfLocalPose, 1, &SensorMap::body_pose_callback, this); //自车位姿
    lidar_localpose_sub = node.subscribe(params->topicLidarLocalPose, 1, &SensorMap::lidar_localpose_callback, this); //激光雷达位姿

    imu_sub = node.subscribe(params->imu_topic, 100, &SensorMap::imu_callback, this); //IMU数据
    if (params->b_enable_lidar_bev && params->lidar_bev_pose_source == "odometry") {
        Odemetry_sub = node.subscribe(params->lidar_odometry_topic, 20, &SensorMap::Odemetry_callback, this);
    }
    history_body_pose.dr_x = 0;

    // 添加发布者初始化
    car_points_pub = node.advertise<sensor_msgs::PointCloud2>("/car_points", 1);
    accumulated_points_pub = node.advertise<sensor_msgs::PointCloud2>("/accumulated_points", 1);

    grid_map_pub_ = node.advertise<grid_map_msgs::GridMap>("grid_map", 1);
    terrain_map_pub_ = node.advertise<world_state::TerrainMap>("/world_state/TerrainMap", 1);
    color_map_pub_   = node.advertise<world_state::ColorMap>("/world_state/ColorMap", 1);
    obstacle_pub_ = node.advertise<world_state::StaticObjArray>("/world_state/StaticObjArray", 1);
    lidar_bev_pub_ = node.advertise<grid_map_msgs::GridMap>(params->lidar_bev_topic, 1);

    LidarBevBuilder::Config bev_config;
    bev_config.enabled = params->b_enable_lidar_bev;
    bev_config.use_odometry_frame = (params->lidar_bev_pose_source != "body");
    bev_config.use_ransac_ground = params->b_lidar_bev_use_ransac_ground;
    bev_config.show_windows = params->b_show_lidar_bev_layers;
    bev_config.topic = params->lidar_bev_topic;
    bev_config.odometry_frame_id = params->lidar_bev_frame_id;
    bev_config.body_frame_id = params->lidar_bev_body_frame_id;
    bev_config.resolution = params->lidar_bev_resolution;
    bev_config.map_size_x = params->lidar_bev_map_size_x;
    bev_config.map_size_y = params->lidar_bev_map_size_y;
    bev_config.point_min_z = params->lidar_bev_point_min_z;
    bev_config.point_max_z = params->lidar_bev_point_max_z;
    bev_config.ego_filter_enabled = params->b_lidar_bev_enable_ego_filter;
    bev_config.ego_box_min_x = params->lidar_bev_ego_box_min_x;
    bev_config.ego_box_max_x = params->lidar_bev_ego_box_max_x;
    bev_config.ego_box_min_y = params->lidar_bev_ego_box_min_y;
    bev_config.ego_box_max_y = params->lidar_bev_ego_box_max_y;
    bev_config.near_inner_radius = params->lidar_bev_near_inner_radius;
    bev_config.near_outer_radius = params->lidar_bev_near_outer_radius;
    bev_config.ground_candidate_min_z = params->lidar_bev_ground_candidate_min_z;
    bev_config.ground_candidate_max_z = params->lidar_bev_ground_candidate_max_z;
    bev_config.ground_ransac_distance = params->lidar_bev_ground_ransac_distance;
    bev_config.ground_max_plane_tilt_deg = params->lidar_bev_ground_max_plane_tilt_deg;
    bev_config.ground_fallback_quantile = params->lidar_bev_ground_fallback_quantile;
    bev_config.ground_min_points = params->lidar_bev_ground_min_points;
    bev_config.height_quantile = params->lidar_bev_height_quantile;
    bev_config.accumulation_frame_count = params->lidar_bev_accumulation_frame_count;
    bev_config.count_saturation = params->lidar_bev_count_saturation;
    bev_config.distance_decay_alpha = params->lidar_bev_distance_decay_alpha;
    bev_config.edge_gradient_threshold = params->lidar_bev_edge_gradient_threshold;
    bev_config.edge_min_height = params->lidar_bev_edge_min_height;
    bev_config.edge_min_jump = params->lidar_bev_edge_min_jump;
    bev_config.edge_min_valid_neighbors = params->lidar_bev_edge_min_valid_neighbors;
    lidar_bev_builder.configure(bev_config);

    grid_map_.setFrameId("ars548");
    grid_map_.setGeometry(grid_map::Length(x_max - x_min, y_max - y_min), grid_size);
    grid_map_.add("elevation", 0.0);
    grid_map_.add("color_map", grid_map::Matrix(rows, cols));

    // 初始化GridMapHandler_v3
    grid_map_handler_v3.initialize(0.2, 100.0, 500, 500);  // 20cm分辨率，100m×100m地图，500×500栅格
    std::cout << "[SensorMap] GridMapHandler_v3 初始化完成" << std::endl;


}

void SensorMap::Odemetry_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    double odom_time = msg->header.stamp.toNSec()/1e6;
    // 1. 提取平移向量 t
    Eigen::Vector3d t(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z);

    // 2. 构造四元数，注意 Eigen 的顺序是 w, x, y, z
    Eigen::Quaterniond q(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);

    // // 打印处理后的 Eigen 数据
    // std::cout << "=== 里程计 Eigen 数据 ===" << std::endl;
    // // 设置高精度输出（保留8位小数）
    // std::cout << std::fixed << std::setprecision(8);
    // // 打印 Eigen 平移向量 t
    // std::cout << "平移向量 t: [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;
    // // 打印 Eigen 四元数 q (w, x, y, z)
    // std::cout << "四元数 q (w, x, y, z): [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]" << std::endl;
    // // 恢复默认输出格式
    // std::cout << std::resetiosflags(std::ios::fixed | std::ios::showpoint);
    // std::cout << "========================" << std::endl;

    // 3. 构造变换矩阵 T（车体 → 全局）
    Eigen::Isometry3d odometry_pose = Eigen::Isometry3d::Identity();
    odometry_pose.linear() = q.toRotationMatrix(); // 旋转
    odometry_pose.translation() = t;               // 平移

    const Pose6D raw_pose = poseFromIsometry(msg->header.stamp.toSec(), odometry_pose);
    Eigen::Isometry3d smoothed_odometry_pose = odometry_pose;
    {
        std::lock_guard<std::mutex> lock(odom_stabilizer_mutex_);
        const Pose6D smoothed_pose = lidar_odom_stabilizer_.AddRawPose(raw_pose);
        smoothed_odometry_pose = pose6DToIsometry(smoothed_pose);
    }

    // 坐标转换，Eigen::Vector3d global_point = T * car_point;
    {
        std::lock_guard<std::mutex> lock(odometry_mutex_);
        T_odmetry = odometry_pose;
        latest_lidar_odometry_.pose = smoothed_odometry_pose;
        latest_lidar_odometry_.stamp = msg->header.stamp;
        latest_lidar_odometry_.frame_id =
            msg->header.frame_id.empty() ? params->lidar_bev_frame_id : msg->header.frame_id;
        latest_lidar_odometry_.valid = true;
    }
    lidar_odemetry_msg.addData(odometry_pose, odom_time, 5);
}

LidarBevBuilder::OdometryState SensorMap::getLatestLidarOdometry()
{
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    return latest_lidar_odometry_;
}

Pose6D SensorMap::poseFromLocalPose(const self_state::LocalPose& pose) const
{
    Pose6D pose6d;
    pose6d.timestamp = pose.local_time > 0 ? pose.local_time * 0.001 : ros::Time::now().toSec();
    pose6d.x = pose.dr_x;
    pose6d.y = pose.dr_y;
    pose6d.z = pose.dr_z;
    pose6d.roll = pose.dr_roll;
    pose6d.pitch = pose.dr_pitch;
    pose6d.yaw = pose.dr_heading;
    return OdomStabilizer::NormalizePoseAngles(pose6d);
}

Eigen::Isometry3d SensorMap::pose6DToIsometry(const Pose6D& pose) const
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.matrix() = OdomStabilizer::PoseToMatrix(pose);
    return transform;
}

LidarBevBuilder::OdometryState SensorMap::getCurrentBevPose()
{
    if (params->lidar_bev_pose_source == "body") {
        LidarBevBuilder::OdometryState body_state;
        body_state.valid = true;
        body_state.stamp = ros::Time::now();
        body_state.frame_id = params->lidar_bev_body_frame_id;
        body_state.pose = Eigen::Isometry3d::Identity();
        return body_state;
    }

    if (params->lidar_bev_pose_source == "odometry") {
        return getLatestLidarOdometry();
    }

    LidarBevBuilder::OdometryState localpose_state;
    localpose_state.valid = local_pose_valid_.load();
    if (!localpose_state.valid) {
        return localpose_state;
    }

    localpose_state.frame_id = params->lidar_bev_frame_id;
    std::optional<Pose6D> smoothed_local_pose;
    {
        std::lock_guard<std::mutex> lock(odom_stabilizer_mutex_);
        smoothed_local_pose = latest_smoothed_local_pose_;
    }

    if (smoothed_local_pose) {
        localpose_state.stamp.fromSec(smoothed_local_pose->timestamp);
        localpose_state.pose = pose6DToIsometry(*smoothed_local_pose);
    } else {
        const Pose6D raw_local_pose = poseFromLocalPose(body_pose);
        localpose_state.stamp.fromSec(raw_local_pose.timestamp);
        localpose_state.pose = pose6DToIsometry(raw_local_pose);
    }

    return localpose_state;
}

void SensorMap::get_rostopic_state()
{
    //根据当前话题自动判断是LM还是HM
    std::vector<ros::master::TopicInfo> topics;
    ros::master::getTopics(topics);

    // 检查HM激光雷达话题是否存在
    auto hm_topic_exists = std::any_of(topics.begin(), topics.end(),
        [&](const ros::master::TopicInfo& ti) { return ti.name == params->HM_Lidar_Topic; });

    // 检查LM激光雷达话题1/2是否存在
    auto lm_topic1_exists = std::any_of(topics.begin(), topics.end(),
        [&](const ros::master::TopicInfo& ti) { return ti.name == params->LM_Lidar_Topic1; });

    auto lm_topic2_exists = std::any_of(topics.begin(), topics.end(),
        [&](const ros::master::TopicInfo& ti) { return ti.name == params->LM_Lidar_Topic2; });

    if (hm_topic_exists) {
        choose_car = "HM";
        ROS_INFO("Detected HM lidar topic, choose_car set to HM");
    } else if (lm_topic1_exists) {
        choose_car = "LM";
        params->LM_Lidar_Topic = params->LM_Lidar_Topic1;
        ROS_INFO("Detected LM lidar topic1, choose_car set to LM");
    } else if (lm_topic2_exists) {
        choose_car = "LM";
        params->LM_Lidar_Topic = params->LM_Lidar_Topic2;
        ROS_INFO("Detected LM lidar topic2, choose_car set to LM");
    } else {
        ROS_WARN("No valid lidar topic found!");
    }

}

void SensorMap::body_pose_callback(const self_state::LocalPose &msg)
{
    auto callback_start_time = std::chrono::high_resolution_clock::now();

    current_time = body_pose.local_time;
    this->last_pose = this->body_pose;
    this->body_pose = msg;
    // 将角度值转换为弧度制
    body_pose.dr_roll = body_pose.dr_roll * PI / 180.0;
    body_pose.dr_pitch = body_pose.dr_pitch * PI / 180.0;
    body_pose.dr_heading = body_pose.dr_heading * PI / 180.0;
    {
        std::lock_guard<std::mutex> lock(odom_stabilizer_mutex_);
        latest_smoothed_local_pose_ =
            local_pose_stabilizer_.AddRawPose(poseFromLocalPose(body_pose));
    }
    //重新计算速度，（HM速度不准）
    body_pose.speed_x = (body_pose.dr_x - last_pose.dr_x) * 1000.0 / (body_pose.local_time - last_pose.local_time);
    body_pose.speed_y = (body_pose.dr_y - last_pose.dr_y) * 1000.0 / (body_pose.local_time - last_pose.local_time);
    body_pose.speed_z = (body_pose.dr_z - last_pose.dr_z) * 1000.0 / (body_pose.local_time - last_pose.local_time);
    //加入到消息缓存队列，并标记时间和传感器类型

    local_pose_msg_mutex.lock();
    local_pose_msg.addData(body_pose, current_time, 1);
    local_pose_msg_mutex.unlock();
    local_pose_valid_.store(true);


    auto callback_end_time = std::chrono::high_resolution_clock::now();
    auto callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end_time - callback_start_time);
    if(callback_duration.count() > 0) {
        std::cout << "[回调] body_pose_callback: " << callback_duration.count() << "ms" << std::endl;
    }
}

void SensorMap::lidar_localpose_callback(const self_state::LidarLocalPose &msg)
{
    // auto callback_start_time = std::chrono::high_resolution_clock::now();

    this->lidar_localpose = msg;
    // 将角度值转换为弧度制
    lidar_localpose.azimuth = lidar_localpose.azimuth * PI / 180.0;
    lidar_localpose.pitch = lidar_localpose.pitch * PI / 180.0;
    lidar_localpose.roll = lidar_localpose.roll * PI / 180.0;

    // auto callback_end_time = std::chrono::high_resolution_clock::now();
    // auto callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end_time - callback_start_time);
    // if(callback_duration.count() > 0) {
    //     std::cout << "[回调] lidar_localpose_callback: " << callback_duration.count() << "ms" << std::endl;
    // }
}



void SensorMap::imu_callback(const sensor_msgs::Imu &msg)
{
    auto callback_start_time = std::chrono::high_resolution_clock::now();

    // 创建IMU数据 - 使用消息时间戳（更准确的时间同步）
    double timestamp = msg.header.stamp.toSec();
    IMUData imu_data(timestamp, msg);

    // 添加到IMU数据队列
    imu_data_queue.push_back(imu_data);

    // 保持队列大小，只保留最近1秒的数据
    double time_window = 1.0;  // 1秒
    while (!imu_data_queue.empty() &&
           (timestamp - imu_data_queue.front().timestamp) > time_window) {
        imu_data_queue.pop_front();
    }

    auto callback_end_time = std::chrono::high_resolution_clock::now();
    auto callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end_time - callback_start_time);
    if(callback_duration.count() > 0) {
        std::cout << "[回调] imu_callback: " << callback_duration.count() << "ms, 队列大小: "
                  << imu_data_queue.size() << std::endl;
    }
}

void SensorMap::M1_lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    double M1_time = msg->header.stamp.toNSec()/1e6;
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    // std::cout<<"pcl_cloud points size: "<<pcl_cloud.points.size()<<std::endl;
    if(true)
    {
        //体素化
        // auto start_time = std::chrono::high_resolution_clock::now();
        pcl::PassThrough<pcl::PointXYZ> pass_filter;

        pass_filter.setInputCloud(pcl_cloud.makeShared());
        pass_filter.setFilterFieldName("x");
        pass_filter.setFilterLimits(5,45);
        pass_filter.filter(pcl_cloud);

//        pass_filter.setInputCloud(pcl_cloud.makeShared());
//        pass_filter.setFilterFieldName("y");
//        pass_filter.setFilterLimits(-30,30);
//        pass_filter.filter(pcl_cloud);

//        pass_filter.setInputCloud(pcl_cloud.makeShared());
//        pass_filter.setFilterFieldName("z");
//        pass_filter.setFilterLimits(-10,10);
//        pass_filter.filter(pcl_cloud);


        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(pcl_cloud.makeShared());
        voxel_filter.setLeafSize(0.4f, 0.1f, 0.1f); // 设置体素大小
//        voxel_filter.setMinimumPointsNumberPerVoxel(2);


        voxel_filter.filter(pcl_cloud); // 进行体素化处理

    }

    // 将点云转换为std::vector
    std::vector<pcl::PointXYZ> lidar_points(pcl_cloud.points.begin(), pcl_cloud.points.end());

    M1_lidar_msg.addData(lidar_points, M1_time, 3); //3代表M1_lidar

}

void SensorMap::BP_lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    double BP_time = msg->header.stamp.toNSec()/1e6;
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);
    if(true)
    {
        //体素化
        // auto start_time = std::chrono::high_resolution_clock::now();
        pcl::PassThrough<pcl::PointXYZ> pass_filter;

        pass_filter.setInputCloud(pcl_cloud.makeShared());
        pass_filter.setFilterFieldName("x");
        pass_filter.setFilterLimits(0.5,8.0);
        pass_filter.filter(pcl_cloud);



        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(pcl_cloud.makeShared());
        voxel_filter.setLeafSize(0.4f, 0.1f, 0.1f); // 设置体素大小
        voxel_filter.filter(pcl_cloud); // 进行体素化处理

        pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusfilter;
        radiusfilter.setInputCloud(pcl_cloud.makeShared());
        radiusfilter.setRadiusSearch(0.4);
        radiusfilter.setMinNeighborsInRadius(2);
        radiusfilter.filter(pcl_cloud);

        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "[回调] BP voxel_filter duration: " << duration.count() << " ms" << std::endl;
    }

    // 将点云转换为std::vector
    std::vector<pcl::PointXYZ> lidar_points(pcl_cloud.points.begin(), pcl_cloud.points.end());
    // BP->M1
    std::vector<pcl::PointXYZ> M1_lidar_points;
    params->new_LM_BP_2_M1_lidar(lidar_points, M1_lidar_points);

    BP_lidar_msg.addData(M1_lidar_points, BP_time, 4);  //4代表BP_lidar
}

void SensorMap::lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    auto callback_start_time = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(*msg, pcl_cloud);

    if(true)
    {
        //体素化
        auto start_time = std::chrono::high_resolution_clock::now();
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(pcl_cloud.makeShared());

        float leaf_size = 0.2f;
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size); // 设置体素大小
        voxel_filter.filter(pcl_cloud); // 进行体素化处理
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "[回调] voxel_filter duration: " << duration.count() << " ms" << std::endl;


        // ROI滤波 - 移除车体附近的点云
        auto roi_start_time = std::chrono::high_resolution_clock::now();

        // ROI参数定义（车体坐标系）
        const float RONI_min_x = -2.0f;   // 车体后方2米
        const float RONI_max_x = 2.0f;    // 车体前方2米
        const float RONI_min_y = -4.0f;   // 车体右侧4米
        const float RONI_max_y = 2.5f;    // 车体左侧2.5米
        const float RONI_min_z = -1.3f;   // 车体下方1.3米
        const float RONI_max_z = 1.0f;    // 车体上方1米

        // 手动滤除ROI内的点（避免检测到车体本身）
        pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
        filtered_cloud.reserve(pcl_cloud.size());

        int removed_points = 0;
        for(const auto& point : pcl_cloud.points) {
            // 检查点是否在ROI范围内
            bool in_roi = (point.x >= RONI_min_x && point.x <= RONI_max_x &&
                          point.y >= RONI_min_y && point.y <= RONI_max_y &&
                          point.z >= RONI_min_z && point.z <= RONI_max_z);

            // 保留ROI外的点
            if(!in_roi) {
                filtered_cloud.points.push_back(point);
            } else {
                removed_points++;
            }
        }

        // 更新点云
        pcl_cloud = filtered_cloud;
        pcl_cloud.width = pcl_cloud.points.size();
        pcl_cloud.height = 1;
        pcl_cloud.is_dense = false;

        auto roi_end_time = std::chrono::high_resolution_clock::now();
        auto roi_duration = std::chrono::duration_cast<std::chrono::milliseconds>(roi_end_time - roi_start_time);
        // std::cout << "[回调] ROI filter duration: " << roi_duration.count() << " ms, removed "
        //           << removed_points << " points" << std::endl;
    }

    // 将点云转换为std::vector
    std::vector<pcl::PointXYZ> lidar_points(pcl_cloud.points.begin(), pcl_cloud.points.end());
 
    lidar_msg_mutex.lock();
    lidar_msg.addData(lidar_points, current_time, 2);  //2代表lidar
    lidar_msg_mutex.unlock(); 

    auto callback_end_time = std::chrono::high_resolution_clock::now();
    auto callback_total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end_time - callback_start_time);
    std::cout << "[回调] LiDAR总时间: " << callback_total_duration.count() << "ms" << std::endl;

}



void SensorMap::camera_Front_callback(const sensor_msgs::CompressedImageConstPtr &msg)
{
    auto callback_start_time = std::chrono::high_resolution_clock::now();

    // 图像解压缩
    cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

    // 检查图像是否成功解码
    if (img.empty())
    {
        cout << "Failed to decode image from topic:" << msg->format.c_str();
        return;
    }

    // 存储图像
    img_map_mutex.lock();
    params->img_map["Camera Front"] = img; //存储到img_map中
    img_map_mutex.unlock();

}

void SensorMap::handle_points()
{
    // 开始计时整个handle_points处理
    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::vector<pcl::PointXYZ> lidar_points;
    if(choose_car == "LM" || choose_car == "HM") {
        lidar_msg_mutex.lock();
        if(!lidar_msg.isQueueEmpty()){
            std::shared_ptr<const TimestampedData<std::vector<pcl::PointXYZ>>> lidar_TimestampedData_ptr = lidar_msg.getbackdata();
            if(abs(current_time - lidar_TimestampedData_ptr->timestamp) > 300){ //最大延迟
                lidar_TimestampedData_ptr = nullptr;
            }
            else {
                lidar_points = *(lidar_TimestampedData_ptr->data);
            }
        }
        lidar_msg_mutex.unlock();
    }
//     else if(choose_car == "new_LM") {
//         double main_time = 0;
//         if(!M1_lidar_msg.isQueueEmpty()){
//             std::shared_ptr<const TimestampedData<std::vector<pcl::PointXYZ>>> M1_TimestampedData_ptr = M1_lidar_msg.getbackdata();
//             if(abs(current_time - M1_TimestampedData_ptr->timestamp) > 300){ //最大延迟
//                 M1_TimestampedData_ptr = nullptr;
//             }
//             else {
//                 lidar_points = *(M1_TimestampedData_ptr->data);
//                 main_time = M1_TimestampedData_ptr->timestamp;
// //                std::cout<<std::fixed << std::setprecision(0)<<"main_time: "<<main_time<<std::endl;
//             }
//         }

//         if(!BP_lidar_msg.isQueueEmpty()){
//             std::shared_ptr<const TimestampedData<std::vector<pcl::PointXYZ>>> BP_TimestampedData_ptr;
//             if(main_time < 1e-6){
//                 //说明没有获取到M1的点云
//                 BP_TimestampedData_ptr = nullptr;
//             }
//             else {
//                 BP_TimestampedData_ptr = BP_lidar_msg.getDatabyTimestampNearest(main_time);
//                 lidar_points.insert(lidar_points.end(), BP_TimestampedData_ptr->data->begin(), BP_TimestampedData_ptr->data->end());
// //                std::cout<<std::fixed << std::setprecision(0)<<"BP_time: "<<BP_TimestampedData_ptr->timestamp<<std::endl;
//             }
//         }
//     }



    else if(choose_car == "new_LM") {
            double main_time = 0;

            //  获取最新的 M1 点云
            if(!M1_lidar_msg.isQueueEmpty()){
                std::shared_ptr<const TimestampedData<std::vector<pcl::PointXYZ>>> M1_TimestampedData_ptr = M1_lidar_msg.getbackdata();
                if(abs(current_time - M1_TimestampedData_ptr->timestamp) > 300){ //最大延迟
                    M1_TimestampedData_ptr = nullptr;
                } else {
                    lidar_points = *(M1_TimestampedData_ptr->data);
                    main_time = M1_TimestampedData_ptr->timestamp;
//                std::cout<<std::fixed << std::setprecision(0)<<"main_time: "<<main_time<<std::endl;
                }
            }

            // 获取与 M1 时间最接近的 BP 点云并融合
            if(!BP_lidar_msg.isQueueEmpty()){
                std::shared_ptr<const TimestampedData<std::vector<pcl::PointXYZ>>> BP_TimestampedData_ptr;

                if(main_time < 1e-6){
                    //说明没有获取到M1的点云
                    BP_TimestampedData_ptr = nullptr;
                } else {
                    // 用 M1 队列时间戳查找最接近的 BP 点云
                    BP_TimestampedData_ptr = BP_lidar_msg.getDatabyTimestampNearest(main_time);
                    if(BP_TimestampedData_ptr){

                        //融合
                        lidar_points.insert(
                            lidar_points.end(),
                            BP_TimestampedData_ptr->data->begin(),
                            BP_TimestampedData_ptr->data->end()
                        );
                        // std::cout<<std::fixed << std::setprecision(0)<<"BP_time: "<<BP_TimestampedData_ptr->timestamp<<std::endl;
                    }
                }
            }
        }

        // std::cout << "Step1: lidar_points.size() = " << lidar_points.size() << std::endl;


    if(lidar_points.empty()) {
        std::cout << "[时间统计] 无点云数据，跳过处理" << std::endl;
        return;
    }

    // 1. 图像去畸变暂时关闭。
    // auto undistort_start_time = std::chrono::high_resolution_clock::now();
    // image_proj->Undistort();
    // auto undistort_end_time = std::chrono::high_resolution_clock::now();
    // auto undistort_duration = std::chrono::duration_cast<std::chrono::milliseconds>(undistort_end_time - undistort_start_time);
    // std::cout << "Undistort duration: " << undistort_duration.count() << " ms" << std::endl;

    // 2. 点云投影到图像暂时关闭，直接使用未着色的雷达点云。
    // auto projection_start_time = std::chrono::high_resolution_clock::now();
    std::vector<PointXYZRGBValid> colored_lidar_points;
    colored_lidar_points.reserve(lidar_points.size());
    for (const auto& point : lidar_points) {
        colored_lidar_points.emplace_back(point.x, point.y, point.z);
    }
    // colored_lidar_points = image_proj->proj_points(lidar_points);
    // auto projection_end_time = std::chrono::high_resolution_clock::now();
    // auto projection_duration = std::chrono::duration_cast<std::chrono::milliseconds>(projection_end_time - projection_start_time);
    // std::cout << "投影: " << projection_duration.count() << " ms" << std::endl;

    // 3. BEV绘制 (根据配置决定是否显示BEV，优化：降低频率)
    auto bev_start_time = std::chrono::high_resolution_clock::now();
    if(params->f_show_bev_color > 0 ) { 
        Draw_lidar_bev(colored_lidar_points);
    }
    auto bev_end_time = std::chrono::high_resolution_clock::now();
    auto bev_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bev_end_time - bev_start_time);
    // std::cout << "Draw_lidar_bev: " << bev_duration.count() << " ms" << std::endl;

    // 4. 坐标转换：激光雷达坐标系 -> 车体坐标系
    auto coord_transform_start_time = std::chrono::high_resolution_clock::now();
    std::vector<PointXYZRGBValid> colored_car_points;
    params->trans_color_lidar2car(colored_lidar_points, colored_car_points);
    auto coord_transform_end_time = std::chrono::high_resolution_clock::now();
    auto coord_transform_duration = std::chrono::duration_cast<std::chrono::milliseconds>(coord_transform_end_time - coord_transform_start_time);
    // std::cout << "激光雷达坐标系 -> 车体坐标系: " << coord_transform_duration.count() << " ms" << std::endl;

    if (params->b_enable_lidar_bev) {
        static bool has_last_lidar_bev_update = false;
        static std::chrono::steady_clock::time_point last_lidar_bev_update_time;

        const double update_rate_hz = params->lidar_bev_update_rate_hz;
        const bool limit_update_rate = update_rate_hz > 0.0;
        const auto now = std::chrono::steady_clock::now();
        const double min_update_interval_sec =
            limit_update_rate ? 1.0 / update_rate_hz : 0.0;
        const bool should_update_lidar_bev =
            !limit_update_rate ||
            !has_last_lidar_bev_update ||
            std::chrono::duration<double>(now - last_lidar_bev_update_time).count() >=
                min_update_interval_sec;

        if (should_update_lidar_bev) {
            auto bev_map_start_time = std::chrono::high_resolution_clock::now();
            LidarBevBuilder::OdometryState bev_pose = getCurrentBevPose();
            if (params->lidar_bev_pose_source != "body" && !bev_pose.valid) {
                ROS_WARN_THROTTLE(1.0, "Waiting for %s before publishing LiDAR BEV.",
                                  params->lidar_bev_pose_source.c_str());
            } else {
                lidar_bev_builder.build(colored_car_points, &bev_pose);
                lidar_bev_builder.publish(lidar_bev_pub_);
                last_lidar_bev_update_time = now;
                has_last_lidar_bev_update = true;
            }
            auto bev_map_end_time = std::chrono::high_resolution_clock::now();
            auto bev_map_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bev_map_end_time - bev_map_start_time);
            std::cout << "lidar_bev time: " << std::setw(4) << bev_map_duration.count() << " ms" << std::endl;
        }
    }


    // 计算高度差,方案一
    //  auto start_time2 = std::chrono::high_resolution_clock::now();
    //  lidar_map.process(colored_car_points, body_pose);
    //  auto end_time2 = std::chrono::high_resolution_clock::now();
    //  auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    //  std::cout<<"colored_car_points.size(): "<<colored_car_points.size()<<std::endl;
    //  std::cout << "lidar_map.output duration: " << duration2.count() << " ms\n" << std::endl;

    // 方案二,gridmap
    // auto start_time3 = std::chrono::high_resolution_clock::now();
    // grid_map_handler.gridmap_process(colored_car_points, body_pose);
    // auto end_time3 = std::chrono::high_resolution_clock::now();
    // auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
    // std::cout << "grid_map_handler.duration: " << duration3.count() << " ms\n" << std::endl;

    // 5. 栅格地图处理 (方案三,gridmap_v2)
   auto gridmap_start_time = std::chrono::high_resolution_clock::now();
   grid_map_handler_v2.gridmap_process(colored_car_points, body_pose);
   auto gridmap_end_time = std::chrono::high_resolution_clock::now();
   auto gridmap_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gridmap_end_time - gridmap_start_time);
    // std::cout << "gridmap_duration: " << gridmap_duration.count() << " ms" << std::endl;

    // 方案四，gridmap_v3 (基于ColorPoint滚动栅格技术)
    // auto start_time5 = std::chrono::high_resolution_clock::now();
    // grid_map_handler_v3.process(colored_car_points, body_pose);
    // auto end_time5 = std::chrono::high_resolution_clock::now();
    // auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time5 - start_time5);
    // std::cout << "grid_map_handler_v3.duration: " << duration5.count() << " ms, points: " << colored_car_points.size() << std::endl;

    // 方案五，gridmap_v4 (基于gridmap_v2的改进)
    // auto start_time6 = std::chrono::high_resolution_clock::now();
    // grid_map_handler_v4.gridmap_process(colored_car_points, body_pose);
    // auto end_time6 = std::chrono::high_resolution_clock::now();
    // auto duration6 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time6 - start_time6);
    // std::cout << "grid_map_handler_v4.duration: " << duration6.count() << " ms, points: " << colored_car_points.size() << std::endl;



    //发布点云
//    Pub_Points(car_points);

    //累积并发布全局点云到RViz
    // accumulateAndPublishPoints(car_points, T_odmetry); //点太多了

    // 更新frozen变量为当前body_pose的实时值
    frozen_x_ = body_pose.dr_x;
    frozen_y_ = body_pose.dr_y;
    frozen_theta_ = body_pose.dr_heading;  // body_pose.dr_heading已经在body_pose_callback中转换为弧度

    // 6. 发布ROS话题（条件优化）
    auto publish_start_time = std::chrono::high_resolution_clock::now();
    auto local_pose = getCurrentLocalPose(); // geometry_msgs::msg::Pose2D

    // 优化：检查是否有订阅者，避免无用的计算
    grid_map_handler_v2.publishTerrainMap(terrain_map_pub_, local_pose);
    grid_map_handler_v2.publishColorMap(color_map_pub_, local_pose);
    grid_map_handler_v2.publishObstacleMap(obstacle_pub_, grid_map_handler_v2.near_obstacles_, grid_map_handler_v2.far_obstacles_, local_pose);


    
    auto publish_end_time = std::chrono::high_resolution_clock::now();
    auto publish_duration = std::chrono::duration_cast<std::chrono::milliseconds>(publish_end_time - publish_start_time);
    // 计算总处理时间
    std::cout << "publish time:   " << std::setw(4) << publish_duration.count() << " ms" << std::endl;
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);






    //发布到grid_map
    // if(params->b_gridmap_use_rgb) {
    //     fillGridMapWithRGB();     // 使用RGB颜色
    // } else {
    //     fillGridMapWithHeight();  // 使用高度着色
    // }
}


geometry_msgs::Pose SensorMap::getCurrentLocalPose() const
{
    geometry_msgs::Pose local_pose;

    // 位姿位置
    local_pose.position.x = frozen_x_;
    local_pose.position.y = frozen_y_;
    local_pose.position.z = 0.0;

    // 将 yaw 转换为四元数
    local_pose.orientation = tf::createQuaternionMsgFromYaw(frozen_theta_);

    return local_pose;
}






// void SensorMap::output()
// {
//     ros::NodeHandle nh;
//     ros::Rate loop_rate(20);  // 设置循环帧率
//     Init();
// //    不是New_LM时,根据当前话题自动选择
// //    if(choose_car != "new_LM")
// //        get_rostopic_state();
//     while (ros::ok())
//     {
//         // 开始计时整个循环周期
//         auto loop_start_time = std::chrono::high_resolution_clock::now();

//         // 1. ROS消息处理时间统计
//         auto spinonce_start_time = std::chrono::high_resolution_clock::now();
//         ros::spinOnce();
//         auto spinonce_end_time = std::chrono::high_resolution_clock::now();
//         auto spinonce_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spinonce_end_time - spinonce_start_time);

//         // 2. 主要处理时间统计
//         auto handle_start_time = std::chrono::high_resolution_clock::now();
//         handle_points();
//         auto handle_end_time = std::chrono::high_resolution_clock::now();
//         auto handle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(handle_end_time - handle_start_time);

//         // 3. 图像显示时间统计
//         auto draw_start_time = std::chrono::high_resolution_clock::now();
//         Draw_img_map();
//         auto draw_end_time = std::chrono::high_resolution_clock::now();
//         auto draw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(draw_end_time - draw_start_time);

//         // 计算整个循环周期时间
//         auto loop_end_time = std::chrono::high_resolution_clock::now();
//         auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - loop_start_time);

//         // 详细时间统计输出
//         std::cout << "\n========== 主循环时间分解 ==========" << std::endl;
//         std::cout << "1. ROS回调函数:    " << std::setw(4) << spinonce_duration.count() << " ms" << std::endl;
//         std::cout << "2. handle_points:  " << std::setw(4) << handle_duration.count() << " ms" << std::endl;
//         std::cout << "3. Draw_img_map:   " << std::setw(4) << draw_duration.count() << " ms" << std::endl;
//         std::cout << "-----------------------------------" << std::endl;
//         std::cout << "主循环总时间:      " << std::setw(4) << loop_duration.count() << " ms" << std::endl;
//         std::cout << "未统计时间:        " << std::setw(4) << (loop_duration.count() - spinonce_duration.count() - handle_duration.count() - draw_duration.count()) << " ms" << std::endl;
//         std::cout << "======================================\n" << std::endl;

//         loop_rate.sleep();
//     }
// }

void SensorMap::output()
{
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);
    Init();

    // === 启动子线程，执行 handle_points + Draw_img_map ===
    std::thread processing_thread([this]() {
        ros::Rate rate(10);  // 子线程执行频率
        while (ros::ok()&& !stop_flag) {

            auto handle_start_time = std::chrono::high_resolution_clock::now();

            handle_points();     // 顺序执行
            // 图像和雷达投影显示暂时关闭。
            // Draw_img_map();

            auto handle_end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                handle_end_time - handle_start_time);
            std::cout << "[Thread] 处理线程耗时: " << duration.count() << " ms" << std::endl;
            rate.sleep();
        }
    });

    // === 主线程继续spinOnce与打印时间 ===
    while (ros::ok())
    {

        auto loop_start_time = std::chrono::high_resolution_clock::now();

        ros::spinOnce();  // 主线程处理ROS回调

        auto loop_end_time = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 loop_end_time - loop_start_time);
        std::cout << "[Main] spinOnce耗时: " << loop_duration.count() << " ms" << std::endl;

        loop_rate.sleep();

    }
    stop_flag = true;
    // === 程序退出前等待线程结束 ===
    if (processing_thread.joinable())
        processing_thread.join();


}





void SensorMap::Pub_Points(std::vector<pcl::PointXYZ> &car_points)
{
    pcl::PointCloud<pcl::PointXYZ> output_cloud;
    output_cloud.points.assign(car_points.begin(), car_points.end());
    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(output_cloud, output_msg);
    output_msg.header.stamp = ros::Time::now();
    output_msg.header.frame_id = "body";
    car_points_pub.publish(output_msg);
}

void SensorMap::accumulateAndPublishPoints(const std::vector<pcl::PointXYZ> &car_points, const Eigen::Isometry3d &T_odmetry)
{
    frame_counter++;

    // 将车体坐标系的点转换到全局坐标系并累积
    for(const auto& point : car_points) {
        // 坐标变换：车体坐标 -> 全局坐标
        Eigen::Vector3d point_global = T_odmetry * Eigen::Vector3d(point.x, point.y, point.z);

        // 添加到累积点云
        pcl::PointXYZ global_point;
        global_point.x = point_global(0);
        global_point.y = point_global(1);
        global_point.z = point_global(2);
        accumulated_cloud.points.push_back(global_point);
    }

    // 每帧都进行体素化和发布
//    std::cout << "累积点云总数: " << accumulated_cloud.points.size() << " 个点" << std::endl;

    // 体素化处理，减少点云密度
    pcl::PointCloud<pcl::PointXYZ> voxelized_cloud;
    if(accumulated_cloud.points.size() > 0) {
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(accumulated_cloud.makeShared());
        float leaf_size = 0.1f;  // 10cm体素大小
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_filter.filter(voxelized_cloud);

        std::cout << "体素化后点云数: " << voxelized_cloud.points.size() << " 个点" << std::endl;

        // 发布到RViz
        sensor_msgs::PointCloud2 accumulated_msg;
        pcl::toROSMsg(voxelized_cloud, accumulated_msg);
        accumulated_msg.header.stamp = ros::Time::now();
        accumulated_msg.header.frame_id = "body";  // 全局坐标系
        accumulated_points_pub.publish(accumulated_msg);
    }

}

void SensorMap::fillGridMapWithHeight()
{
    // 获取层的引用
    grid_map::Matrix& elevation_layer = grid_map_["elevation"];
    grid_map::Matrix& color_map_layer = grid_map_["color_map"];

    // 清空所有层
    elevation_layer.setConstant(0.0);
    color_map_layer.setConstant(0.0);

    // 设置过滤阈值，忽略高度低于该值的栅格
    const float height_threshold = 0.1f;

    // 遍历整个 grid_map，填充高度信息和颜色
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // 获取对应的高度值
            float height = lidar_map.height_map.at<float>(r, c);
            // 获取是否有数据
            uchar has_data_value = lidar_map.has_data.at<uchar>(r, c);

            // 只有有效数据并且高度大于阈值时才填充
            if (has_data_value > 0 && height >= height_threshold) {
                elevation_layer(r, c) = height;  // 填充高度

                // 根据高度值设置 RGB 颜色
                uint8_t r_color = 0, g_color = 0, b_color = 0;
                params->setColorByHeight(height, r_color, g_color, b_color);

                // 使用GridMap的颜色转换方式
                Eigen::Vector3i colorVector;
                colorVector[0] = r_color;  // R
                colorVector[1] = g_color;  // G
                colorVector[2] = b_color;  // B

                float color_value = 0.0f;
                grid_map::colorVectorToValue(colorVector, color_value);
                color_map_layer(r, c) = color_value;
            } else {
                // 高度低于阈值的栅格可以保持为 0 或 NaN
                elevation_layer(r, c) = std::numeric_limits<float>::quiet_NaN();  // 设置为 NaN
            }
        }
    }

    // 发布最新的 GridMap
    publishGridMap();
}

void SensorMap::fillGridMapWithRGB()
{
    // 获取层的引用
    grid_map::Matrix& elevation_layer = grid_map_["elevation"];
    grid_map::Matrix& color_map_layer = grid_map_["color_map"];

    // 清空所有层
    elevation_layer.setConstant(0.0);
    color_map_layer.setConstant(0.0);

    // 设置过滤阈值，忽略高度低于该值的栅格
    const float height_threshold = 0.0f;

    // 遍历整个 grid_map，填充高度信息和RGB颜色
    int rgb_cells_count = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // 获取对应的高度值
            float height = lidar_map.height_map.at<float>(r, c);
            // 获取是否有数据
            uchar has_data_value = lidar_map.has_data.at<uchar>(r, c);

            // 只有有效数据并且高度大于阈值时才填充
            if (has_data_value > 0 && height >= height_threshold) {
                elevation_layer(r, c) = height;  // 填充高度

                // 从RGB地图获取真实RGB颜色
                cv::Vec3b rgb_color = lidar_map.rgb_map.at<cv::Vec3b>(r, c);
                uint8_t r_color = rgb_color[2];  // OpenCV BGR -> RGB
                uint8_t g_color = rgb_color[1];
                uint8_t b_color = rgb_color[0];

                // 使用GridMap的颜色转换方式
                Eigen::Vector3i colorVector;
                colorVector[0] = r_color;  // R
                colorVector[1] = g_color;  // G
                colorVector[2] = b_color;  // B

                float color_value = 0.0f;
                grid_map::colorVectorToValue(colorVector, color_value);
                color_map_layer(r, c) = color_value;
                rgb_cells_count++;
            } else {
                // 高度低于阈值的栅格可以保持为 0 或 NaN
                elevation_layer(r, c) = std::numeric_limits<float>::quiet_NaN();  // 设置为 NaN
            }
        }
    }

    std::cout << "GridMap RGB模式 - 填充了 " << rgb_cells_count << " 个RGB栅格" << std::endl;

    // 发布最新的 GridMap
    publishGridMap();
}

void SensorMap::publishGridMap()
{
    grid_map_msgs::GridMap grid_map_msg;
    grid_map::GridMapRosConverter::toMessage(grid_map_, grid_map_msg);
    grid_map_pub_.publish(grid_map_msg);
    ROS_INFO("Published GridMap message");
}

void SensorMap::Draw_img_map()
{
    // 开始计时图像显示处理
    auto draw_start_time = std::chrono::high_resolution_clock::now();

    // 设定显示窗口的最大尺寸
    int max_width = 600;
    int max_height = 600;
    //不画出来的图像
    std::vector<std::string> windows_not_show;
    windows_not_show =  {};

    // 根据配置控制相机图像显示
    if(!params->b_show_camera_front) {
        windows_not_show.push_back("Camera Front");
    }
    if(!params->b_show_undistort) {
        windows_not_show.push_back("Camera Front Undistort");
    }

    // 根据配置控制投影图像显示
    if(!params->b_show_proj) {
        windows_not_show.push_back("Lidar Proj");
    }

    // 根据配置控制BEV图像显示
    if(params->f_show_bev_color == 0) {
        windows_not_show.push_back("LiDAR BEV (RGB Colored)");
    }

    if(params->img_map.size() > 0){
        for(auto it=params->img_map.begin(); it!=params->img_map.end(); it++){
            cv::Mat img = it->second;
            if(std::find(windows_not_show.begin(), windows_not_show.end(), it->first) != windows_not_show.end())
                continue;
            int img_width = img.cols;
            int img_height = img.rows;

            // 如果图像尺寸超过 600×600，则调整大小
            if (img_width > max_width || img_height > max_height)
            {
                // 计算缩放比例，确保图像在600×600内
                double scale_x = static_cast<double>(max_width) / img_width;
                double scale_y = static_cast<double>(max_height) / img_height;
                double scale = std::min(scale_x, scale_y);  // 选择较小的比例，确保不超出

                // 调整图像大小
                cv::resize(img, img, cv::Size(), scale, scale);
            }
            cv::namedWindow(it->first, cv::WINDOW_NORMAL);
            cv::imshow(it->first, img);
        }
        cv::waitKey(1);
    }

    //清理自己的画图队列
    // params->img_map.clear();

    // 计算图像显示时间
    auto draw_end_time = std::chrono::high_resolution_clock::now();
    auto draw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(draw_end_time - draw_start_time);
    // std::cout << "图像显示处理时间: " << draw_duration.count() << " ms" << std::endl;

    return;
}

void SensorMap::Draw_lidar_bev(const std::vector<PointXYZRGBValid> &colored_points)
{
    // BEV图像参数
    const int BEV_WIDTH = 800;   // BEV图像宽度
    const int BEV_HEIGHT = 800;  // BEV图像高度
    const double BEV_RANGE = 50.0; // BEV范围：±50米
    const double SCALE = BEV_WIDTH / (2.0 * BEV_RANGE); // 像素/米比例

    // 创建BEV图像（黑色背景）
    cv::Mat bev_image = cv::Mat::zeros(BEV_HEIGHT, BEV_WIDTH, CV_8UC3);

    // 统计信息
    int valid_points = 0;
    int out_of_range_points = 0;

    for(const auto& point : colored_points)
    {
        // 将3D点投影到BEV平面（忽略Z坐标）
        double x = point.x;  // 前后方向
        double y = point.y;  // 左右方向

        // 检查是否在BEV范围内
        if(std::abs(x) <= BEV_RANGE && std::abs(y) <= BEV_RANGE)
        {
            // 转换为图像坐标
            // X轴（前后）对应图像的行（上下），车辆前方在图像上方
            // Y轴（左右）对应图像的列（左右），激光雷达坐标系：前左上
            int col = static_cast<int>((-y + BEV_RANGE) * SCALE);  // Y -> 列（左为正，所以取负）
            int row = static_cast<int>((BEV_RANGE - x) * SCALE);   // X -> 行（翻转，前方在上）

            // 确保坐标在图像范围内
            if(row >= 0 && row < BEV_HEIGHT && col >= 0 && col < BEV_WIDTH)
            {
                cv::Vec3b& pixel = bev_image.at<cv::Vec3b>(row, col);

                // 根据配置决定显示哪些点
                if(params->f_show_bev_color == 1 && !point.has_rgb) {
                    // 模式1：只显示有颜色的点，跳过无颜色点
                    continue;
                }

                if(point.has_rgb) {
                    // 使用真实的RGB颜色信息
                    pixel[0] = point.b;  // Blue
                    pixel[1] = point.g;  // Green
                    pixel[2] = point.r;  // Red
                } else {
                    // 使用灰色（投影失败的点）
                    pixel[0] = 128;  // Blue
                    pixel[1] = 128;  // Green
                    pixel[2] = 128;  // Red
                }

                valid_points++;
            }
        }
        else
        {
            out_of_range_points++;
        }
    }

    // 绘制车辆位置（中心点，白色十字）
    int center_x = BEV_WIDTH / 2;
    int center_y = BEV_HEIGHT / 2;
    cv::line(bev_image, cv::Point(center_x - 10, center_y), cv::Point(center_x + 10, center_y),
             cv::Scalar(255, 255, 255), 2);
    cv::line(bev_image, cv::Point(center_x, center_y - 10), cv::Point(center_x, center_y + 10),
             cv::Scalar(255, 255, 255), 2);

    // 绘制车辆轮廓（假设车辆尺寸为4m×2m）
    double car_length = 4.0;  // 车长
    double car_width = 2.0;   // 车宽
    int car_length_pixels = static_cast<int>(car_length * SCALE);
    int car_width_pixels = static_cast<int>(car_width * SCALE);

    cv::Rect car_rect(center_x - car_width_pixels/2,
                      center_y - car_length_pixels/2,
                      car_width_pixels,
                      car_length_pixels);
    cv::rectangle(bev_image, car_rect, cv::Scalar(0, 255, 0), 2);  // 绿色车辆轮廓

    // 添加文本信息
    std::string info_text = "Points: " + std::to_string(valid_points) +
                           " | Range: " + std::to_string(static_cast<int>(BEV_RANGE)) + "m";
    cv::putText(bev_image, info_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // 显示BEV图像
    // cv::imshow("LiDAR BEV (RGB Colored)", bev_image);
    params->img_map["LiDAR BEV (RGB Colored)"] = bev_image;

    // 统计RGB点数
    int rgb_points = 0;
    int gray_points = 0;
    for(const auto& point : colored_points) {
        if(point.has_rgb) rgb_points++;
        else gray_points++;
    }

    std::string mode_str;
    switch(params->f_show_bev_color) {
        case 0: mode_str = "不显示"; break;
        case 1: mode_str = "仅彩色点"; break;
        case 2: mode_str = "所有点"; break;
        default: mode_str = "未知模式"; break;
    }

    std::cout << "BEV绘制完成 [" << mode_str << "] - 总点数: " << colored_points.size()
              << ", BEV内点数: " << valid_points
              << ", RGB点数: " << rgb_points
              << ", 灰色点数: " << gray_points
              << ", 超出范围: " << out_of_range_points << std::endl;
}
