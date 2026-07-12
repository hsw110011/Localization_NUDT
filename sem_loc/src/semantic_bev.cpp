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
    // --------------------------------------------------------
    // [Offset 0-15] 坐标 + 强度
    // --------------------------------------------------------
    union
    {
        // PCL 变换函数强制要求这个数组存在 (float data[4])
        float data[4];

        // 这里的 x, y, z, intensity 与 data 共享内存
        // data[0]=x, data[1]=y, data[2]=z, data[3]=intensity
        struct
        {
            float x;
            float y;
            float z;
            float intensity; // Offset 12
        };
    };

    // --------------------------------------------------------
    // [Offset 16-19] 颜色
    // --------------------------------------------------------
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
    };

    // --------------------------------------------------------
    // [Offset 20-23] 标签 (uint32)
    // --------------------------------------------------------
    uint32_t label;

    // --------------------------------------------------------
    // [Offset 24-27] 曲率
    // --------------------------------------------------------
    float curvature;

    // --------------------------------------------------------
    // [Offset 28-31] 填充 (凑齐 32 字节)
    // --------------------------------------------------------
    uint32_t _padding;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGBLabelCurvature,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, rgb, rgb)
    (uint32_t, label, label)
    (float, curvature, curvature)
)

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
    uint32_t label;
    uint8_t r, g, b;
    // float intensity;
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
        class_names_ = {"road", "building", "plant"};
        num_classes_ = 3;

        // --- 状态初始化 ---
        has_odom_ = false;
        cur_pose_mat_ = Eigen::Isometry3d::Identity();

        // --- ROS 接口 ---
        sub_odom_ = nh_.subscribe("/Odometry", 10, &VisualBevMapper::odomCb, this);
        sub_cloud_ = nh_.subscribe("/cloud_registered_body", 5, &VisualBevMapper::cloudCb, this);

        pub_rgb_ = nh_.advertise<sensor_msgs::Image>("/Semantic_Bev/Rgb", 1);

        // 发布一个包含所有 mask 的话题 (多通道)
        // 通道 0: Mask 0 (Road)
        // 通道 1: Mask 1 (Building)
        // 通道 2: Mask 2 (Plant)
        pub_all_masks_ = nh_.advertise<sensor_msgs::Image>("/Semantic_Bev/ClassMask", 1);

        timer_ = nh_.createTimer(ros::Duration(0.1), &VisualBevMapper::timerRenderCb, this);

        ROS_INFO("Visual BEV Mapper Started (Classes: 0, 1, 2).");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_cloud_;
    ros::Publisher pub_rgb_;
    ros::Publisher pub_all_masks_; // 单一话题发布所有 mask
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

        if (cloud->empty()) return;

        pcl::PointCloud<PointXYZIRGBLabelCurvature>::Ptr cloud_world(new pcl::PointCloud<PointXYZIRGBLabelCurvature>);
        // 这里会调用 cloud_in[i].data，现在结构体里有 data 了，所以编译能通过
        pcl::transformPointCloud(*cloud, *cloud_world, cur_pose_mat_.matrix());

        std::lock_guard<std::mutex> lock(map_mutex_);

        for (const auto& pt : cloud_world->points) {
            if (!pcl::isFinite(pt)) continue;

            int u = static_cast<int>(std::floor(pt.x / resolution_));
            int v = static_cast<int>(std::floor(pt.y / resolution_));
            GridKey key = {u, v};

            auto it = global_map_.find(key);
            if (it == global_map_.end() || pt.z > it->second.z) {
                GridCell cell;
                cell.z = pt.z;
                cell.label = pt.label;
                cell.r = pt.r;
                cell.g = pt.g;
                cell.b = pt.b;
                global_map_[key] = cell;
            }
        }
    }

    void timerRenderCb(const ros::TimerEvent&) {
        if (!has_odom_ || global_map_.empty()) return;

        std::lock_guard<std::mutex> lock(map_mutex_);

        // 1. 清理过远的旧地图点
        int cx = static_cast<int>(cur_pos_.x() / resolution_);
        int cy = static_cast<int>(cur_pos_.y() / resolution_);
        int cleanup_dist = static_cast<int>(img_size_ * 1.5);

        for (auto it = global_map_.begin(); it != global_map_.end(); ) {
            if (std::abs(it->first.u - cx) > cleanup_dist || std::abs(it->first.v - cy) > cleanup_dist) {
                it = global_map_.erase(it);
            } else {
                ++it;
            }
        }

        // 2. 准备画布
        int hs = img_size_ / 2;
        int sr = static_cast<int>(hs * 1.5);
        int side = sr * 2;

        cv::Mat canvas_rgb = cv::Mat::zeros(side, side, CV_8UC3);
        // CV_8UC1 默认初始化为0
        std::vector<cv::Mat> canvas_masks(num_classes_);
        for(int i=0; i<num_classes_; ++i) canvas_masks[i] = cv::Mat::zeros(side, side, CV_8UC1);

        // 3. 填充画布
        for (const auto& kv : global_map_) {
            int local_u = kv.first.u - cx + sr;
            int local_v = sr - (kv.first.v - cy);

            if (local_u >= 0 && local_u < side && local_v >= 0 && local_v < side) {
                const auto& cell = kv.second;

                // RGB 用于显示，还是保持 0-255
                canvas_rgb.at<cv::Vec3b>(local_v, local_u) = cv::Vec3b(cell.b, cell.g, cell.r);

                // 生成 Mask 矩阵里只有 0 和 1
                if (cell.label < static_cast<uint32_t>(num_classes_)) {
                    canvas_masks[cell.label].at<uint8_t>(local_v, local_u) = 1;
                }
            }
        }

        // 4. 旋转图像
        double angle_deg = 90.0 - (cur_yaw_ * 180.0 / M_PI);
        cv::Point2f center(sr, sr);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle_deg, 1.0);

        rot_mat.at<double>(0, 2) += (hs - sr);
        rot_mat.at<double>(1, 2) += (hs - sr);

        cv::Size dst_size(img_size_, img_size_);

        // 生成最终 RGB 图 (用于显示)
        cv::Mat img_rgb_final;
        cv::warpAffine(canvas_rgb, img_rgb_final, rot_mat, dst_size, cv::INTER_LINEAR);

        // 发布 RGB (只是给人看的，无所谓)
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = "base_link";
        pub_rgb_.publish(cv_bridge::CvImage(header, "bgr8", img_rgb_final).toImageMsg());

        // 5. 生成并发布各类 Mask (合并为一个包含所有 mask 的话题)

        // 合并所有 mask 到一个多通道图像
        cv::Mat combined_mask_src;
        cv::merge(canvas_masks, combined_mask_src);

        cv::Mat combined_mask_final;
        // 使用最近邻插值，保证结果仍然是纯粹的 0 和 1
        cv::warpAffine(combined_mask_src, combined_mask_final, rot_mat, dst_size, cv::INTER_NEAREST);

        // 发布合并后的 mask (3通道, bgr8)
        // 注意：数据本身是 0/1，在 ROS 图像查看器中看起来可能是全黑的。
        // 但这是作为数据传输最准确的方式。
        pub_all_masks_.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_8UC3, combined_mask_final).toImageMsg());

        // 6. 可视化 (用于调试监控)
        std::vector<cv::Mat> display_list;

        // 可视化列表加入 RGB
        cv::Mat vis_rgb = img_rgb_final.clone();
        cv::putText(vis_rgb, "RGB View", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
        display_list.push_back(vis_rgb);

        // 将合并后的 mask 拆分出来进行可视化
        std::vector<cv::Mat> final_masks_split;
        cv::split(combined_mask_final, final_masks_split);

        for (int i = 0; i < num_classes_; ++i) {
            // 可视化处理：转彩色并增强亮度
            cv::Mat vis_mask;

            // 转为彩色 (此时值还是0和1)
            cv::cvtColor(final_masks_split[i], vis_mask, cv::COLOR_GRAY2BGR);

            // 乘以 255，让 1 变成 255 (纯亮)，这样人眼才能看见
            vis_mask = vis_mask * 255;

            // 打上标签文字
            std::string label_txt = "Class " + std::to_string(i);
            if (i < class_names_.size()) {
                label_txt += ": " + class_names_[i];
            }
            // 文字用绿色画
            cv::putText(vis_mask, label_txt, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 1);

            display_list.push_back(vis_mask);
        }

        // OpenCV 窗口显示 (只要有数据就显示)
        if (!display_list.empty()) {
            cv::Mat combined;
            cv::hconcat(display_list, combined);

            // 如果图太宽，缩放一下，不然屏幕放不下
            // 调整为 0.75 (原来0.5)，并允许手动调整窗口
            if (combined.cols > 1920) {
                cv::resize(combined, combined, cv::Size(), 0.75, 0.75);
            }

            cv::namedWindow("BEV Mapper Monitor", cv::WINDOW_NORMAL);
            cv::imshow("BEV Mapper Monitor", combined);
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