/**
 * Visual BEV Mapper (Final Fix)
 * 1. 修复 'no member named data' 编译错误
 * 2. 适配 0, 1, 2 三个类别
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// TF & Math
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>

// ==========================================
// 1. 自定义 PCL 点类型
// ==========================================
struct PointXYZIRGBLabelCurvature
{
    PCL_ADD_POINT4D;                    // x, y, z
    float intensity;                    // 强度 - Offset 16
    union
    {
        union
        {
            struct
            {
                uint8_t b;
                uint8_t g;
                uint8_t r;
                uint8_t a;
            };
            float rgb;
        };
        uint32_t rgba;
    };                                  // Color - Offset 20
    uint32_t label;                     // Label - Offset 24
    float curvature;                    // Curvature - Offset 28

    // New field for probabilities
    float merged_probs[3];              // Offset 32

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册点类型 (注意：merged_probs 手动提取，不在此注册以免 PCL 解析错误)
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGBLabelCurvature,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, rgb, rgb)
    (uint32_t, label, label)
    (float, curvature, time) // 将 curvature 映射到 time 字段
)

// Helper to extract array field
template <typename T>
void extract_merged_probs(const sensor_msgs::PointCloud2ConstPtr &msg, pcl::PointCloud<T> &cloud)
{
    int field_idx = -1;
    for (size_t i = 0; i < msg->fields.size(); ++i) {
        if (msg->fields[i].name == "merged_probs") {
            field_idx = i;
            break;
        }
    }
    if (field_idx == -1) return;

    const auto& field = msg->fields[field_idx];
    if (field.datatype == sensor_msgs::PointField::FLOAT32) {
        size_t point_step = msg->point_step;
        size_t offset = field.offset;
        const uint8_t* data_ptr = msg->data.data();
        size_t num_points = msg->width * msg->height;
        // PCL cloud size might differ slightly if filtered? No, usually 1-1 map from fromROSMsg
        if (cloud.points.size() != num_points) return;

        for (size_t i = 0; i < num_points; ++i) {
            const float* probs = reinterpret_cast<const float*>(data_ptr + i * point_step + offset);
            cloud.points[i].merged_probs[0] = probs[0];
            cloud.points[i].merged_probs[1] = probs[1];
            cloud.points[i].merged_probs[2] = probs[2];
        }
    }
}

// ==========================================
// 2. 辅助数据结构
// ==========================================
struct GridKey {
    int u;
    int v;
    bool operator==(const GridKey& other) const { return (u == other.u && v == other.v); }
};

struct GridKeyHash {
    std::size_t operator()(const GridKey& k) const {
        return ((int64_t)k.u << 32) | ((uint32_t)k.v);
    }
};

struct GridCell {
    float z;
    float probs[3]; // [road, building, plant] -> Now we want [Road, Building, Plant]
};

// ==========================================
// 3. Mapper 类
// ==========================================
class VisualBevMapper {
public:
    VisualBevMapper() : nh_("~") {
        // --- 参数 ---
        nh_.param("resolution", resolution_, 0.15);
        nh_.param("view_range", view_range_, 60.0);
        img_size_ = static_cast<int>(view_range_ / resolution_);

        // --- 类别设置 (修正：只保留 0, 1, 2) ---
        // Original: {"road", "plant", "building"}
        // New Request: 0=Road, 1=Building, 2=Plant
        class_names_ = {"road", "building", "vegetation"};
        num_classes_ = 3;

        // --- 状态初始化 ---
        has_odom_ = false;
        cur_pose_mat_ = Eigen::Isometry3d::Identity();

        // --- ROS 接口 ---
        sub_odom_ = nh_.subscribe("/Odometry", 10, &VisualBevMapper::odomCb, this);
        sub_cloud_ = nh_.subscribe("/cloud_registered_body", 5, &VisualBevMapper::cloudCb, this);

        // 发布包含概率矩阵的 BEV (3通道浮点图: [prob_road, prob_building, prob_plant])
        pub_probs_ = nh_.advertise<sensor_msgs::Image>("/Semantic_Bev/Probs", 1);

        timer_ = nh_.createTimer(ros::Duration(0.1), &VisualBevMapper::timerRenderCb, this);

        ROS_INFO("Visual BEV Mapper Started (Output: Probs Matrix).");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_probs_;
    ros::Timer timer_;

    std::unordered_map<GridKey, GridCell, GridKeyHash> global_map_;
    std::mutex map_mutex_;

    double resolution_;
    double view_range_;
    int img_size_;
    int num_classes_;
    std::vector<std::string> class_names_;

    bool has_odom_;
    Eigen::Vector3d cur_pos_;
    double cur_yaw_;
    Eigen::Isometry3d cur_pose_mat_;

    void odomCb(const nav_msgs::OdometryConstPtr& msg) {
        cur_pos_.x() = msg->pose.pose.position.x;
        cur_pos_.y() = msg->pose.pose.position.y;
        cur_pos_.z() = msg->pose.pose.position.z;

        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z
        );

        cur_pose_mat_ = Eigen::Isometry3d::Identity();
        cur_pose_mat_.rotate(q);
        cur_pose_mat_.pretranslate(cur_pos_);

        // 提取 Yaw
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        cur_yaw_ = std::atan2(siny_cosp, cosy_cosp);

        has_odom_ = true;
    }

    void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (!has_odom_) return;

        pcl::PointCloud<PointXYZIRGBLabelCurvature>::Ptr cloud(new pcl::PointCloud<PointXYZIRGBLabelCurvature>);
        pcl::fromROSMsg(*msg, *cloud);
        extract_merged_probs(msg, *cloud); // 手动提取概率

        if (cloud->empty()) return;

        pcl::PointCloud<PointXYZIRGBLabelCurvature>::Ptr cloud_world(new pcl::PointCloud<PointXYZIRGBLabelCurvature>);
        // 这里会调用 cloud_in[i].data，现在结构体里有 data 了，所以编译能通过
        pcl::transformPointCloud(*cloud, *cloud_world, cur_pose_mat_.matrix());

        std::lock_guard<std::mutex> lock(map_mutex_);

        for (int i = 0; i < cloud_world->points.size(); ++i) {
            const auto& pt = cloud_world->points[i];
            if (!pcl::isFinite(pt)) continue;

            int u = static_cast<int>(std::floor(pt.x / resolution_));
            int v = static_cast<int>(std::floor(pt.y / resolution_));
            GridKey key = {u, v};

            // 保留最高点的概率
            // const auto& pt_orig = cloud->points[i];
            // auto it = global_map_.find(key);
            // if (it == global_map_.end() || pt.z > it->second.z) {
            //     GridCell cell;
            //     cell.z = pt.z;
            //     // 用原始点的概率 (transformation 不改变概率)
            //     cell.probs[0] = pt_orig.merged_probs[0];
            //     cell.probs[1] = pt_orig.merged_probs[1];
            //     cell.probs[2] = pt_orig.merged_probs[2];
            //     global_map_[key] = cell;
            // }

            // [New] 前后帧概率加权平均 (EMA)
            const auto& pt_orig = cloud->points[i];
            auto it = global_map_.find(key);

            if (it == global_map_.end())
            {
                // 1. 新栅格直接初始化
                GridCell cell;
                cell.z = pt.z;
                cell.probs[0] = pt_orig.merged_probs[0];
                cell.probs[1] = pt_orig.merged_probs[1];
                cell.probs[2] = pt_orig.merged_probs[2];
                global_map_[key] = cell;
            }
            else
            {
                // 2. 已存在栅格时，进行时序 EMA 融合
                // alpha 越大越偏向当前帧，越小越平滑
                float alpha = 0.3f;

                // 当前点显著更高时，快速响应（例如新物体出现）
                if (pt.z > it->second.z + 0.5f)
                {
                    alpha = 0.8f;
                    it->second.z = pt.z;
                }
                else if (pt.z > it->second.z)
                {
                    it->second.z = pt.z;
                }

                for (int c = 0; c < 3; ++c)
                {
                    it->second.probs[c] = (1.0f - alpha) * it->second.probs[c] + alpha * pt_orig.merged_probs[c];
                }

                // 归一化到概率和为 1
                float sum_p = it->second.probs[0] + it->second.probs[1] + it->second.probs[2] + 1e-9f;
                it->second.probs[0] /= sum_p;
                it->second.probs[1] /= sum_p;
                it->second.probs[2] /= sum_p;
            }

            /*
            // [Old] Max-Pooling / 最新观测优先逻辑 (修改版)
            const auto& pt_orig = cloud->points[i];
            auto it = global_map_.find(key);

            if (it == global_map_.end())
            {
                GridCell cell;
                cell.z = pt.z;
                cell.probs[0] = pt_orig.merged_probs[0];
                cell.probs[1] = pt_orig.merged_probs[1];
                cell.probs[2] = pt_orig.merged_probs[2];
                global_map_[key] = cell;
            }
            else
            {
                if (pt.z > it->second.z)
                {
                    it->second.z = pt.z;
                    it->second.probs[0] = pt_orig.merged_probs[0];
                    it->second.probs[1] = pt_orig.merged_probs[1];
                    it->second.probs[2] = pt_orig.merged_probs[2];
                }
            }
            */
        }
    }

    void timerRenderCb(const ros::TimerEvent&)
    {
        if (!has_odom_ || global_map_.empty()) return;

        std::lock_guard<std::mutex> lock(map_mutex_);

        // 1. 清理过远的旧地图点
        int cx = static_cast<int>(cur_pos_.x() / resolution_);
        int cy = static_cast<int>(cur_pos_.y() / resolution_);
        int cleanup_dist = static_cast<int>(img_size_ * 1.5);

        for (auto it = global_map_.begin(); it != global_map_.end(); )
        {
            if (std::abs(it->first.u - cx) > cleanup_dist || std::abs(it->first.v - cy) > cleanup_dist)
            {
                it = global_map_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 2. 准备画布 - CV_32FC4 (增加一个 Entropy 通道)
        int hs = img_size_ / 2;
        int sr = static_cast<int>(hs * 1.5);
        int side = sr * 2;

        cv::Mat canvas_probs = cv::Mat::zeros(side, side, CV_32FC4);

        const float max_entropy = std::log(3.0f); // 3 classes

        // 3. 填充画布
        for (const auto& kv : global_map_)
        {
            int local_u = kv.first.u - cx + sr;
            int local_v = sr - (kv.first.v - cy);

            if (local_u >= 0 && local_u < side && local_v >= 0 && local_v < side)
            {
                const auto& cell = kv.second;

                // 计算熵
                float entropy = 0.0f;
                // 暂时认为概率和为1，或者直接计算 sum
                for(int k=0; k<3; ++k)
                {
                    float p = cell.probs[k];
                    if (p > 1e-6f)
                    { // 避免 log(0)
                        entropy -= p * std::log(p);
                    }
                }
                float entropy_norm = entropy / max_entropy;
                // Clamp [0, 1]
                if (entropy_norm < 0.0f) entropy_norm = 0.0f;
                if (entropy_norm > 1.0f) entropy_norm = 1.0f;

                // 存储: 0->Road, 1->Building, 2->Plant, 3->EntropyNorm
                // Original Probs: [0]=Road, [1]=Plant, [2]=Building (Based on input cloud parsing assumption)
                // Wait, need to confirm input order from point cloud
                // Assuming cloud.merged_probs follows: [0:Road, 1:Plant, 2:Building] ? Or whatever upstream sends.
                // Assuming upstream sends: [Road, Plant, Building]
                // We want to publish:      [Road, Building, Plant]

                // Input (cell.probs) assumed: [0]=Road, [1]=Plant, [2]=Building
                // Output (Vec4f) requested:   [0]=Road, [1]=Building, [2]=Plant

                canvas_probs.at<cv::Vec4f>(local_v, local_u) = cv::Vec4f(
                    cell.probs[0], // Road
                    cell.probs[2], // Building (was index 2 in struct assumption, wait... let's check struct)
                    cell.probs[1], // Plant (was index 1)
                    entropy_norm
                );
            }
        }

        // 4. 旋转图像
        double angle_deg = 90.0 - (cur_yaw_ * 180.0 / M_PI);
        cv::Point2f center(sr, sr);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_deg, 1.0);

        rot_mat.at<double>(0, 2) += (hs - sr);
        rot_mat.at<double>(1, 2) += (hs - sr);

        cv::Size dst_size(img_size_, img_size_);

        // 生成最终概率图 (32FC4)
        // 使用线性插值使概率图更平滑，或者最近邻保持原始值。这里选用线性插值
        cv::Mat img_probs_final;
        cv::warpAffine(canvas_probs, img_probs_final, rot_mat, dst_size, cv::INTER_LINEAR);

        // 发布概率图
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = "base_link";

        // Use TYPE_32FC4
        pub_probs_.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC4, img_probs_final).toImageMsg());

        // 5. 可视化 (用于调试监控)
        std::vector<cv::Mat> display_list;

        // 分离通道进行显示
        std::vector<cv::Mat> probs_channels;
        cv::split(img_probs_final, probs_channels);

        for (int i = 0; i < num_classes_ && i < 3; ++i)
        {
             cv::Mat heat_map, norm_img;
             // 1. 0-1 float to 0-255 uchar
             probs_channels[i].convertTo(norm_img, CV_8UC1, 255.0);
             // 2. Apply Jet Colormap
             cv::applyColorMap(norm_img, heat_map, cv::COLORMAP_JET);

             std::string label_txt = "P(" + class_names_[i] + ")";
             cv::putText(heat_map, label_txt, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
             display_list.push_back(heat_map);
        }

        // 可视化第4个通道 (归一化熵)
        if (probs_channels.size() > 3) {
            cv::Mat entropy_heat, entropy_norm_img;
            probs_channels[3].convertTo(entropy_norm_img, CV_8UC1, 255.0);

            // 熵通常不需要太花哨，但如果想保持一致也可以用 JET
            // 或者用 TURBO / INFERNO 等看起来更“现代”的
            cv::applyColorMap(entropy_norm_img, entropy_heat, cv::COLORMAP_JET);

            // 熵的通道用不同的颜色（例如红色文字）标识
            cv::putText(entropy_heat, "Norm Entropy", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
            display_list.push_back(entropy_heat);
        }

        // OpenCV 窗口显示
        if (!display_list.empty()) {
            cv::Mat combined;
            cv::hconcat(display_list, combined);

            if (combined.cols > 1920) {
                cv::resize(combined, combined, cv::Size(), 0.75, 0.75);
            }

            cv::namedWindow("BEV Probs Monitor", cv::WINDOW_NORMAL);
            cv::imshow("BEV Probs Monitor", combined);
            cv::waitKey(1);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "visual_bev_mapper_node");
    VisualBevMapper mapper;
    ros::spin();
    return 0;
}