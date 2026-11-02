#include <ros/ros.h>
#include <xmlrpcpp/XmlRpcUtil.h>
#include <cerrno>
#include <cstdio>
#include "SemLocConfig.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <unordered_map>

using SemanticPoint = pcl::PointXYZL;
struct GridKey {
    int x, y;
    bool operator==(const GridKey& other) const { return x == other.x && y == other.y; }
};
struct GridKeyHash {
    std::size_t operator()(const GridKey& key) const {
        return (static_cast<uint64_t>(static_cast<uint32_t>(key.x)) << 32) |
               static_cast<uint32_t>(key.y);
    }
};

// 一帧对一个格子仅贡献一次，避免点密度和遍历顺序改变时序融合权重。
struct CellObservation {
    std::array<float, 4> votes{}; // road, building, plant, others
    std::array<float, 3> probs{};
};
struct GridCell {
    std::deque<CellObservation> history;
    std::array<float, 3> probs{};
    int label = 3;
    float entropy = 0;

    void refresh() {
        std::array<float, 4> votes{};
        probs.fill(0);
        for (const auto& observation : history) {
            for (int c = 0; c < 4; ++c) votes[c] += observation.votes[c];
            for (int c = 0; c < 3; ++c) probs[c] += observation.probs[c];
        }
        for (float& p : probs) p /= history.size();
        label = std::max_element(votes.begin(), votes.end()) - votes.begin();
        const float sum=probs[0]+probs[1]+probs[2];
        entropy=0;
        for (float value : probs) {
            const float p=sum>0 ? value/sum : 0;
            if (p>0.f) entropy-=p*std::log(p);
        }
        entropy=std::clamp(entropy/std::log(3.f),0.f,1.f);
    }
};
struct PoseSample {
    ros::Time stamp;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    std::string frame;
};

class VisualBevMapper {
public:
    explicit VisualBevMapper(const sem_loc::Config& config) : nh_("~"), config_(config) {
        nh_.param("resolution", resolution_, config.resolution);
        nh_.param("view_range", view_range_, config.view_range);
        if (!std::isfinite(resolution_) || resolution_ <= 0 ||
            !std::isfinite(view_range_) || view_range_ < 2 * resolution_)
            throw std::invalid_argument("invalid BEV geometry");
        size_ = static_cast<int>(view_range_ / resolution_);
        radius_ = static_cast<int>(std::ceil(view_range_ / std::sqrt(2.0) / resolution_)) + 1;
        sub_odom_ = nh_.subscribe("/lio/odom", 100, &VisualBevMapper::odomCb, this);
        sub_cloud_ = nh_.subscribe("/lio/cloud_world", 5, &VisualBevMapper::cloudCb, this);
        if (config_.bev_mode == sem_loc::BevMode::Certain) {
            pub_semantic_ = nh_.advertise<sensor_msgs::Image>("/Semantic_Bev/ClassMask", 1);
        } else {
            pub_semantic_ = nh_.advertise<sensor_msgs::Image>("/Semantic_Bev/Probs", 1);
        }
        timer_ = nh_.createWallTimer(ros::WallDuration(0.01), &VisualBevMapper::timerCb, this);
        std::cout << "[BEV] Started: " << (certain() ? "certain [road,building,plant,mask]"
            : "uncertain [road,building,plant,entropy,mask]")
            << ", output=" << pub_semantic_.getTopic()
            << ", encoding=" << (certain() ? "8UC4" : "32FC5")
            << ", odom=" << sub_odom_.getTopic() << ", cloud=" << sub_cloud_.getTopic() << std::endl;
    }

private:
    ros::NodeHandle nh_;
    sem_loc::Config config_;
    ros::Subscriber sub_odom_, sub_cloud_;
    ros::Publisher pub_semantic_;
    ros::WallTimer timer_;
    double resolution_, view_range_;
    int size_, radius_;
    std::deque<PoseSample> poses_;
    std::unordered_map<GridKey, GridCell, GridKeyHash> grid_;
    sensor_msgs::PointCloud2ConstPtr pending_;
    ros::WallTime pending_received_;
    ros::Time last_received_, last_processed_, last_fused_;
    Eigen::Vector2d last_position_ = Eigen::Vector2d::Zero();
    double last_yaw_ = 0;
    bool have_fused_ = false;

    bool certain() const { return config_.bev_mode == sem_loc::BevMode::Certain; }
    GridKey key(double x, double y) const {
        return {static_cast<int>(std::floor(x / resolution_ + 1e-9)), static_cast<int>(std::floor(y / resolution_ + 1e-9))};
    }
    void resetAccumulation() {
        grid_.clear();
        have_fused_ = false;
        last_processed_ = ros::Time();
    }
    void odomCb(const nav_msgs::OdometryConstPtr& msg) {
        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        if (msg->header.stamp.isZero() || msg->header.frame_id.empty() || !position.allFinite() ||
            !q.coeffs().allFinite() || q.norm() < 1e-6) {
            ROS_WARN_THROTTLE(2.0, "BEV: invalid odometry timestamp/frame/pose");
            return;
        }
        if (!poses_.empty() && (poses_.back().frame != msg->header.frame_id ||
            (poses_.back().stamp - msg->header.stamp).toSec() > 0.5)) {
            poses_.clear();
            resetAccumulation();
        }
        PoseSample sample;
        sample.stamp = msg->header.stamp;
        sample.frame = msg->header.frame_id;
        sample.pose.linear() = q.normalized().toRotationMatrix();
        sample.pose.translation() = position;
        auto it = std::lower_bound(poses_.begin(), poses_.end(), sample.stamp,
            [](const PoseSample& p, const ros::Time& t) { return p.stamp < t; });
        if (it != poses_.end() && it->stamp == sample.stamp) *it = sample;
        else poses_.insert(it, sample);
        while (poses_.size() > 400 || (poses_.size() > 1 &&
               (poses_.back().stamp - poses_.front().stamp).toSec() > 3.0)) poses_.pop_front();
        processPending();
    }
    bool poseAt(const ros::Time& stamp, bool allow_nearest, PoseSample& out) const {
        if (poses_.empty()) return false;
        auto upper = std::lower_bound(poses_.begin(), poses_.end(), stamp,
            [](const PoseSample& p, const ros::Time& t) { return p.stamp < t; });
        if (upper != poses_.end() && upper->stamp == stamp) { out = *upper; return true; }
        if (upper != poses_.begin() && upper != poses_.end()) {
            const auto& a = *std::prev(upper);
            const auto& b = *upper;
            const double gap = (b.stamp - a.stamp).toSec();
            if (gap <= 0.20 && a.frame == b.frame) {
                const double t = (stamp - a.stamp).toSec() / gap;
                out = a;
                out.stamp = stamp;
                out.pose.translation() = (1-t)*a.pose.translation() + t*b.pose.translation();
                out.pose.linear() = Eigen::Quaterniond(a.pose.linear()).slerp(t,
                    Eigen::Quaterniond(b.pose.linear())).normalized().toRotationMatrix();
                return true;
            }
        }
        if (!allow_nearest) return false;
        auto nearest = upper == poses_.end() ? std::prev(poses_.end()) : upper;
        if (upper != poses_.begin() && std::abs((std::prev(upper)->stamp-stamp).toSec()) <
            std::abs((nearest->stamp-stamp).toSec())) nearest = std::prev(upper);
        if (std::abs((nearest->stamp-stamp).toSec()) > 0.03) return false;
        out = *nearest;
        out.stamp = stamp;
        return true;
    }
    void cloudCb(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (msg->header.stamp.isZero()) {
            ROS_WARN_THROTTLE(2.0, "BEV: cloud requires a nonzero timestamp");
            return;
        }
        if (msg->header.stamp < last_received_) resetAccumulation();
        last_received_ = msg->header.stamp;
        if (msg->header.stamp == last_processed_) return;
        pending_ = msg;
        pending_received_ = ros::WallTime::now();
        processPending();
    }
    void timerCb(const ros::WallTimerEvent&) { processPending(); }
    void processPending() {
        if (!pending_) return;
        const bool expired = (ros::WallTime::now() - pending_received_).toSec() >= 0.05;
        PoseSample pose;
        if (!poseAt(pending_->header.stamp, expired, pose)) {
            if (expired) {
                ROS_WARN_THROTTLE(2.0, "BEV: no time-matched odometry; dropping cloud");
                pending_.reset();
            }
            return;
        }
        auto cloud = pending_;
        pending_.reset();
        if (config_.input_cloud_in_world && cloud->header.frame_id != pose.frame) {
            ROS_ERROR_THROTTLE(2.0, "BEV: world cloud frame must equal odometry parent frame");
            return;
        }
        integrate(*cloud, pose);
    }
    bool readProbabilities(const sensor_msgs::PointCloud2& msg, std::vector<std::array<float,3>>& result) const {
        auto f = std::find_if(msg.fields.begin(), msg.fields.end(),
            [](const sensor_msgs::PointField& field) { return field.name == "merged_probs"; });
        if (f == msg.fields.end() || f->datatype != sensor_msgs::PointField::FLOAT32 || f->count != 3 ||
            f->offset + 12 > msg.point_step || msg.is_bigendian ||
            uint64_t(msg.width)*msg.point_step > msg.row_step ||
            uint64_t(msg.row_step)*msg.height > msg.data.size()) {
            ROS_ERROR_THROTTLE(2.0, "BEV: requires valid little-endian FLOAT32 merged_probs[3]");
            return false;
        }
        result.resize(static_cast<size_t>(msg.width)*msg.height);
        for (size_t r=0; r<msg.height; ++r) for (size_t c=0; c<msg.width; ++c) {
            std::array<float,3> input;
            std::memcpy(input.data(), msg.data.data()+r*msg.row_step+c*msg.point_step+f->offset, 12);
            if (!std::all_of(input.begin(), input.end(), [](float p) { return std::isfinite(p) && p>=-1e-6f && p<=1.000001f; })) {
                ROS_ERROR_THROTTLE(2.0, "BEV: invalid semantic probabilities");
                return false;
            }
            for (int k=0;k<3;++k) result[r*msg.width+c][k] = std::clamp(input[config_.probability_channels[k]],0.f,1.f);
        }
        return true;
    }
    void integrate(const sensor_msgs::PointCloud2& msg, const PoseSample& sample) {
        if (certain()) {
            const auto label = std::find_if(msg.fields.begin(),msg.fields.end(),
                [](const sensor_msgs::PointField& f) { return f.name=="label"; });
            if (label==msg.fields.end() || label->datatype!=sensor_msgs::PointField::UINT32 ||
                label->count!=1 || label->offset+4>msg.point_step ||
                uint64_t(msg.width)*msg.point_step>msg.row_step || uint64_t(msg.row_step)*msg.height>msg.data.size()) {
                ROS_ERROR_THROTTLE(2.0,"BEV: certain mode requires a valid UINT32 label field");
                return;
            }
        }
        std::vector<std::array<float,3>> probs;
        if (!certain() && !readProbabilities(msg, probs)) return;
        pcl::PointCloud<SemanticPoint> input, world;
        pcl::fromROSMsg(msg, input);
        if (config_.input_cloud_in_world) world = std::move(input);
        else pcl::transformPointCloud(input, world, sample.pose.matrix());
        const Eigen::Vector2d position = sample.pose.translation().head<2>();
        const double yaw = std::atan2(sample.pose.linear()(1,0),sample.pose.linear()(0,0));
        const double dyaw = std::abs(std::atan2(std::sin(yaw-last_yaw_),std::cos(yaw-last_yaw_)));
        const bool fuse = !have_fused_ || (position-last_position_).norm() >= config_.accumulation_distance ||
            dyaw >= config_.accumulation_yaw * M_PI/180 ||
            (msg.header.stamp-last_fused_).toSec() >= config_.accumulation_interval;
        const GridKey center = key(position.x(),position.y());
        if (fuse) {
            struct FrameCell { float top = -std::numeric_limits<float>::infinity(); std::vector<size_t> indices; };
            std::unordered_map<GridKey,FrameCell,GridKeyHash> frame;
            for (size_t i=0;i<world.size();++i) {
                const auto& point = world[i];
                if (!pcl::isFinite(point)) continue;
                const auto k = key(point.x,point.y);
                if (std::abs(k.x-center.x)>radius_ || std::abs(k.y-center.y)>radius_) continue;
                auto& cell=frame[k];
                cell.top=std::max(cell.top,point.z);
                cell.indices.push_back(i);
            }
            for (const auto& entry : frame) {
                CellObservation observation;
                int count=0;
                for (size_t i : entry.second.indices) {
                    const auto& point=world[i];
                    if (point.z < entry.second.top-config_.surface_band) continue;
                    observation.votes[point.label<3 ? point.label : 3] += 1;
                    if (!certain()) for (int c=0;c<3;++c) observation.probs[c] += probs[i][c];
                    ++count;
                }
                for (float& v : observation.votes) v/=count;
                for (float& v : observation.probs) v/=count;
                auto& cell = grid_[entry.first];
                auto& history = cell.history;
                history.push_back(observation);
                while (history.size()>static_cast<size_t>(config_.history_frames)) history.pop_front();
                cell.refresh(); // 只更新本帧观测到的格子，其他格子复用统计缓存。
            }
            last_fused_=msg.header.stamp;
            last_position_=position;
            last_yaw_=yaw;
            have_fused_=true;
        }
        // 地图按空间窗口保留，与观测年龄无关；每格的样本数只约束融合内存。
        for (auto it=grid_.begin();it!=grid_.end();) {
            if (std::abs(it->first.x-center.x)>radius_ || std::abs(it->first.y-center.y)>radius_)
                it=grid_.erase(it);
            else ++it;
        }
        last_processed_=msg.header.stamp;
        render(sample,msg.header.stamp,yaw);
    }
    void render(const PoseSample& pose, const ros::Time& stamp, double yaw) {
        cv::Mat output=cv::Mat::zeros(size_,size_,certain()?CV_8UC4:CV_32FC(5));
        const double c=std::cos(yaw), s=std::sin(yaw);
        // 从每个输出像素反查 world 栅格：用连续车位，不再整格截断中心或重复旋转旧图。
        for (int row=0;row<size_;++row) for (int col=0;col<size_;++col) {
            const double forward=(size_/2-row)*resolution_, left=(size_/2-col)*resolution_;
            const auto k=key(pose.pose.translation().x()+c*forward-s*left,
                             pose.pose.translation().y()+s*forward+c*left);
            auto it=grid_.find(k);
            if (it==grid_.end()) continue;
            const auto& cell=it->second;
            if (certain()) {
                auto& pixel=output.at<cv::Vec4b>(row,col);
                if (cell.label<3) pixel[cell.label]=1;
                pixel[3]=1;
            } else {
                auto& pixel=output.at<cv::Vec<float,5>>(row,col);
                for (int i=0;i<3;++i) pixel[i]=cell.probs[i];
                pixel[3]=cell.entropy;
                pixel[4]=1;
            }
        }
        std_msgs::Header header;
        header.stamp=stamp;
        header.frame_id="base_link";
        pub_semantic_.publish(cv_bridge::CvImage(header,certain()?"8UC4":"32FC5",output).toImageMsg());
        if (!config_.visualize) return;
        std::vector<cv::Mat> channels,views;
        cv::split(output,channels);
        const std::vector<std::string> names=certain()?std::vector<std::string>{"road","building","plant","mask"}
            :std::vector<std::string>{"road","building","plant","entropy","mask"};
        for (size_t i=0;i<channels.size();++i) {
            cv::Mat image,heat;
            channels[i].convertTo(image,CV_8U,255);
            cv::applyColorMap(image,heat,cv::COLORMAP_JET);
            cv::putText(heat,names[i],cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(255,255,255));
            views.push_back(heat);
        }
        cv::Mat display;
        cv::hconcat(views,display);
        if (display.cols>1920) cv::resize(display,display,cv::Size(),0.75,0.75);
        cv::imshow("BEV Mapper Monitor",display);
        cv::waitKey(1);
    }
};

// 上游日志只给 poll 返回值 -1；补充当前 errno，便于定位持续通信错误。
class XmlRpcPollDiagnostics : public XmlRpc::XmlRpcErrorHandler {
public:
    XmlRpcPollDiagnostics() : previous_(getErrorHandler()) { setErrorHandler(this); }
    ~XmlRpcPollDiagnostics() override { setErrorHandler(previous_); }
    void error(const char* message) override {
        const int saved_errno = errno;
        if (std::strstr(message, "XmlRpcDispatch::work: error in poll")) {
            // XML-RPC 启动早于 ROS 日志/时间初始化，直接报告以免漏掉首次错误。
            std::fprintf(stderr, "[BEV XML-RPC] %s errno=%d (%s)\n",
                         message, saved_errno, std::strerror(saved_errno));
        } else {
            previous_->error(message);
        }
        errno = saved_errno;
    }
private:
    XmlRpc::XmlRpcErrorHandler* previous_;
};

int main(int argc,char** argv) {
    ros::init(argc,argv,"visual_bev_mapper_node");
    XmlRpcPollDiagnostics xmlrpc_diagnostics;
    try {
        VisualBevMapper mapper(sem_loc::LoadConfig(std::string(SEM_LOC_SOURCE_DIR)+"/config/sem_loc.yaml"));
        ros::spin();
        ros::shutdown();
    } catch (const std::exception& error) {
        std::cerr << "[BEV] Startup failed: " << error.what() << std::endl;
        ros::shutdown();
        return 1;
    }
}
