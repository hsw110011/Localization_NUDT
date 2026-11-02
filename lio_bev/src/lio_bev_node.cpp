#include "lio_bev/body_bev_builder.hpp"

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

using BodyBevBuilder = lio_bev::BodyBevBuilder;
using PointXYZ = lio_bev::PointXYZ;
using SteadyClock = lio_bev::SteadyClock;
using lio_bev::elapsedMs;

class BodyBevNode {
public:
  BodyBevNode() : pnh_("~")
  {
    pnh_.param<std::string>("input_topic", input_topic_, "/cloud_registered_body");
    pnh_.param<std::string>("odometry_topic", odometry_topic_, "/Odometry");
    pnh_.param<std::string>("output_topic", output_topic_, "/lidar_bev/grid_map");
    // Zero means process each available cloud. A positive value schedules the
    // worker by steady-clock deadlines; it never rejects 99 ms header intervals.
    pnh_.param<double>("update_rate_hz", update_rate_hz_, 0.0);
    pnh_.param<double>("odometry_wait_sec", odometry_wait_sec_, 0.05);
    pnh_.param<double>("odometry_max_gap_sec", odometry_max_gap_sec_, 0.20);
    pnh_.param<double>("odometry_nearest_tolerance_sec", odometry_nearest_tolerance_sec_, 0.03);
    pnh_.param<double>("performance_log_interval_sec", log_interval_sec_, 5.0);
    for (double value : {update_rate_hz_, odometry_wait_sec_, odometry_max_gap_sec_,
                         odometry_nearest_tolerance_sec_, log_interval_sec_}) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument("invalid scheduling or synchronization parameter");
      }
    }
    if ((update_rate_hz_ > 0.0 && update_rate_hz_ < 0.01) || update_rate_hz_ > 1000.0 ||
        odometry_wait_sec_ > 1.0 || odometry_max_gap_sec_ <= 0.0 ||
        odometry_max_gap_sec_ > 3.0 || odometry_nearest_tolerance_sec_ > odometry_max_gap_sec_) {
      throw std::invalid_argument("out-of-range scheduling or synchronization parameter");
    }
    pnh_.param<bool>("publish_bev_tf", publish_bev_tf_, true);
    BodyBevBuilder::Config config;
    pnh_.param<bool>("gravity_aligned_output", config.gravity_aligned_output, false);
    pnh_.param<std::string>("bev_frame_id", config.bev_frame_id, "bev_local");
    pnh_.param<bool>("use_odometry_frame", config.use_odometry_frame, true);
    pnh_.param<bool>("input_cloud_in_world", config.input_cloud_in_world, false);
    pnh_.param<std::string>("odometry_frame_id", config.odometry_frame_id, "camera_init");
    pnh_.param<std::string>("body_frame_id", config.body_frame_id, "body");
    pnh_.param<double>("resolution", config.resolution, 0.2);
    pnh_.param<double>("map_size_x", config.map_size_x, 100.0);
    pnh_.param<double>("map_size_y", config.map_size_y, 100.0);
    pnh_.param<double>("point_min_z", config.point_min_z, -5.0);
    pnh_.param<double>("point_max_z", config.point_max_z, 80.0);
    pnh_.param<bool>("ego_filter_enabled", config.ego_filter_enabled, true);
    pnh_.param<double>("ego_box_min_x", config.ego_box_min_x, -2.0);
    pnh_.param<double>("ego_box_max_x", config.ego_box_max_x, 2.0);
    pnh_.param<double>("ego_box_min_y", config.ego_box_min_y, -2.0);
    pnh_.param<double>("ego_box_max_y", config.ego_box_max_y, 2.0);
    pnh_.param<double>("h_rel_deadzone_half_width", config.h_rel_deadzone_half_width, 0.20);
    pnh_.param<double>("grad_deadzone_half_width", config.grad_deadzone_half_width, 0.15);
    pnh_.param<double>("grad_cap", config.grad_cap, 0.0);
    pnh_.param<int>("stabilization_history_frames",
                    config.stabilization_history_frames, 10);
    pnh_.param<bool>("enable_spatial_height_smoothing", config.enable_spatial_height_smoothing, false);
    pnh_.param<int>("height_smoothing_radius", config.height_smoothing_radius, 2);
    pnh_.param<double>("height_smoothing_max_delta", config.height_smoothing_max_delta, 0.35);
    pnh_.param<int>("height_smoothing_min_neighbors", config.height_smoothing_min_neighbors, 3);
    pnh_.param<int>("edge_min_valid_neighbors", config.edge_min_valid_neighbors, 5);
    pnh_.param<int>("stabilization_min_overlap_cells", config.stabilization_min_overlap_cells, 80);
    pnh_.param<double>("stabilization_inlier_threshold", config.stabilization_inlier_threshold, 0.15);
    pnh_.param<double>("stabilization_max_z", config.stabilization_max_z, 0.30);
    pnh_.param<double>("stabilization_max_tilt_deg", config.stabilization_max_tilt_deg, 1.5);
    pnh_.param<double>("stabilization_alpha", config.stabilization_alpha, 0.25);
    pnh_.param<int>("corrected_sample_capacity", config.corrected_sample_capacity, 32);
    pnh_.param<double>("corrected_surface_top_fraction",
                       config.corrected_surface_top_fraction, 0.05);
    pnh_.param<double>("accumulation_update_distance_m",
                       config.accumulation_update_distance_m, 0.5);

    pnh_.param<double>("accumulation_update_yaw_deg", config.accumulation_update_yaw_deg, 2.0);
    pnh_.param<double>("accumulation_max_interval_sec", config.accumulation_max_interval_sec, 0.5);
    gravity_aligned_output_ = config.gravity_aligned_output;
    use_odometry_frame_ = config.use_odometry_frame;
    input_cloud_in_world_ = config.input_cloud_in_world;
    if (input_cloud_in_world_ && !use_odometry_frame_) {
      throw std::invalid_argument("world cloud input requires odometry");
    }
    odometry_frame_id_ = config.odometry_frame_id;
    builder_.configure(config);
    pub_ = nh_.advertise<grid_map_msgs::GridMap>(output_topic_, 1);
    cloud_sub_ = nh_.subscribe(input_topic_, 1, &BodyBevNode::cloudCallback, this,
                               ros::TransportHints().tcpNoDelay());
    if (use_odometry_frame_) {
      odom_sub_ = nh_.subscribe(odometry_topic_, 200, &BodyBevNode::odometryCallback,
                                this, ros::TransportHints().tcpNoDelay());
    }
    worker_ = std::thread(&BodyBevNode::workerLoop, this);
    ROS_INFO("lio_bev: output frame=%s, gravity-aligned=%s, publish BEV TF=%s",
             (gravity_aligned_output_ ? config.bev_frame_id : config.body_frame_id).c_str(),
             gravity_aligned_output_ ? "true" : "false",
             gravity_aligned_output_ && publish_bev_tf_ ? "true" : "false");
    ROS_INFO("lio_bev: cloud=%s (%s), odometry=%s, output=%s; latest-only worker, no GUI",
             input_topic_.c_str(), input_cloud_in_world_ ? "world" : "body",
             use_odometry_frame_ ? odometry_topic_.c_str() : "disabled", output_topic_.c_str());
  }

  ~BodyBevNode()
  {
    cloud_sub_.shutdown();
    odom_sub_.shutdown();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }
  BodyBevNode(const BodyBevNode&) = delete;
  BodyBevNode& operator=(const BodyBevNode&) = delete;

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber cloud_sub_, odom_sub_;
  ros::Publisher pub_;
  BodyBevBuilder builder_;  // Accessed only by the worker.
  tf2_ros::TransformBroadcaster bev_broadcaster_;
  bool gravity_aligned_output_ = false;
  bool publish_bev_tf_ = true;
  std::string input_topic_, odometry_topic_, output_topic_, odometry_frame_id_;
  bool use_odometry_frame_ = true;
  bool input_cloud_in_world_ = false;
  double update_rate_hz_ = 0.0;
  double odometry_wait_sec_ = 0.05;
  double odometry_max_gap_sec_ = 0.20;
  double odometry_nearest_tolerance_sec_ = 0.03;
  double log_interval_sec_ = 5.0;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool stopping_ = false;
  sensor_msgs::PointCloud2ConstPtr pending_cloud_;
  SteadyClock::time_point pending_received_;
  ros::Time last_received_stamp_;
  std::uint64_t epoch_ = 0;
  lio_bev::PoseBuffer poses_;  // Protected by mutex_.
  std::thread worker_;
  std::atomic<std::uint64_t> received_{0}, overwritten_{0}, sync_dropped_{0}, invalid_{0};

  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y,
                                msg->pose.pose.position.z);
    const double quaternion_norm = q.coeffs().stableNorm();
    if (msg->header.stamp.isZero() || !q.coeffs().allFinite() ||
        !translation.allFinite() || translation.cwiseAbs().maxCoeff() > 1e12 ||
        !std::isfinite(quaternion_norm) || quaternion_norm < 1e-6) {
      ++invalid_;
      ROS_WARN_THROTTLE(2.0, "lio_bev: rejected invalid odometry");
      return;
    }
    q.coeffs() /= quaternion_norm;
    lio_bev::PoseSample sample;
    sample.pose.linear() = q.toRotationMatrix();
    sample.pose.translation() = translation;
    sample.stamp = msg->header.stamp;
    sample.frame_id = msg->header.frame_id.empty() ? odometry_frame_id_ : msg->header.frame_id;
    sample.valid = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      poses_.insert(sample);
    }
    condition_.notify_one();
  }

  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
  {
    ++received_;
    if (msg->header.stamp.isZero()) {
      ++invalid_;
      ROS_WARN_THROTTLE(2.0, "lio_bev: point cloud must have a nonzero timestamp");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (msg->header.stamp < last_received_stamp_) ++epoch_;
      last_received_stamp_ = msg->header.stamp;
      if (pending_cloud_) ++overwritten_;
      pending_cloud_ = msg;
      pending_received_ = SteadyClock::now();
    }
    condition_.notify_one();
  }

  void workerLoop()
  {
    std::vector<PointXYZ> points;
    auto next_due = SteadyClock::now();
    auto log_start = next_due;
    double decode_sum = 0.0, build_sum = 0.0, publish_sum = 0.0, max_ms = 0.0;
    double queue_sum = 0.0;
    BodyBevBuilder::Timing stage_sum;
    std::uint64_t published = 0, total_published = 0;
    while (true) {
      sensor_msgs::PointCloud2ConstPtr cloud;
      lio_bev::PoseSample pose;
      SteadyClock::time_point received;
      std::uint64_t epoch;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return stopping_ || pending_cloud_; });
        if (stopping_) return;
        if (update_rate_hz_ > 0.0 && SteadyClock::now() < next_due) {
          condition_.wait_until(lock, next_due, [&] { return stopping_; });
          if (stopping_) return;
        }
        cloud.swap(pending_cloud_);
        received = pending_received_;
        epoch = epoch_;
        if (use_odometry_frame_) {
          const auto deadline = SteadyClock::now() + std::chrono::duration<double>(odometry_wait_sec_);
          bool matched = poses_.sample(cloud->header.stamp, odometry_max_gap_sec_,
                                       odometry_nearest_tolerance_sec_, false, pose);
          while (!matched && !stopping_ && epoch == epoch_ && SteadyClock::now() < deadline) {
            condition_.wait_until(lock, deadline);
            matched = poses_.sample(cloud->header.stamp, odometry_max_gap_sec_,
                                    odometry_nearest_tolerance_sec_, false, pose);
          }
          if (stopping_) return;
          if (epoch != epoch_) continue;
          if (!matched) matched = poses_.sample(cloud->header.stamp, odometry_max_gap_sec_,
                                                odometry_nearest_tolerance_sec_, true, pose);
          if (!matched) {
            ++sync_dropped_;
            ROS_WARN_THROTTLE(2.0, "lio_bev: no time-matched odometry for cloud; dropping frame");
            continue;
          }
          if (input_cloud_in_world_ && cloud->header.frame_id != pose.frame_id) {
            ++invalid_;
            ROS_WARN_THROTTLE(2.0, "lio_bev: world cloud frame and odometry parent frame differ");
            continue;
          }
        }
      }
      const auto start = SteadyClock::now();
      if (update_rate_hz_ > 0.0) {
        const auto period = std::chrono::duration_cast<SteadyClock::duration>(
          std::chrono::duration<double>(1.0 / update_rate_hz_));
        next_due = start + period;
      }
      try {
        lio_bev::readXYZ(*cloud, points);
        const double decode_ms = elapsedMs(start);
        auto stage_start = SteadyClock::now();
        builder_.build(points, cloud->header.stamp, use_odometry_frame_ ? &pose : nullptr);
        const double build_ms = elapsedMs(stage_start);
        stage_start = SteadyClock::now();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (epoch != epoch_) continue;  // Do not publish an old rosbag epoch.
        }
        if (gravity_aligned_output_ && publish_bev_tf_) {
          geometry_msgs::TransformStamped transform;
          transform.header.stamp = cloud->header.stamp;
          transform.header.frame_id = builder_.outputParentFrame();
          transform.child_frame_id = builder_.map().getFrameId();
          const auto pose_out = builder_.outputPose();
          const Eigen::Quaterniond q(pose_out.linear());
          transform.transform.translation.x = pose_out.translation().x();
          transform.transform.translation.y = pose_out.translation().y();
          transform.transform.translation.z = pose_out.translation().z();
          transform.transform.rotation.x = q.x();
          transform.transform.rotation.y = q.y();
          transform.transform.rotation.z = q.z();
          transform.transform.rotation.w = q.w();
          bev_broadcaster_.sendTransform(transform);
        }
        builder_.publish(pub_);
        const double publish_ms = elapsedMs(stage_start);
        if (total_published == 0) log_start = start;
        ++published;
        ++total_published;
        decode_sum += decode_ms;
        build_sum += build_ms;
        publish_sum += publish_ms;
        max_ms = std::max(max_ms, elapsedMs(start));
        queue_sum += std::chrono::duration<double, std::milli>(start - received).count();
        const auto& t = builder_.timing();
        stage_sum.prepare_ms += t.prepare_ms;
        stage_sum.stabilize_ms += t.stabilize_ms;
        stage_sum.accumulate_ms += t.accumulate_ms;
        stage_sum.rasterize_ms += t.rasterize_ms;
        stage_sum.gradient_ms += t.gradient_ms;
        const double interval = std::chrono::duration<double>(SteadyClock::now() - log_start).count();
        if (log_interval_sec_ > 0.0 && interval >= log_interval_sec_) {
          const double n = static_cast<double>(published);
          ROS_INFO("BEV perf: %.2f Hz total=%llu received=%llu overwritten=%llu sync_drop=%llu invalid=%llu "
                   "ms avg=%.2f max=%.2f queue_wait=%.2f decode=%.2f build=%.2f publish=%.2f "
                   "[prepare=%.2f stabilize=%.2f accumulate=%.2f raster=%.2f gradient=%.2f]",
                   n / interval, static_cast<unsigned long long>(total_published),
                   static_cast<unsigned long long>(received_.load()),
                   static_cast<unsigned long long>(overwritten_.load()),
                   static_cast<unsigned long long>(sync_dropped_.load()),
                   static_cast<unsigned long long>(invalid_.load()),
                   (decode_sum + build_sum + publish_sum) / n, max_ms, queue_sum / n,
                   decode_sum / n, build_sum / n, publish_sum / n,
                   stage_sum.prepare_ms / n, stage_sum.stabilize_ms / n,
                   stage_sum.accumulate_ms / n, stage_sum.rasterize_ms / n, stage_sum.gradient_ms / n);
          published = 0;
          decode_sum = build_sum = publish_sum = queue_sum = max_ms = 0.0;
          stage_sum = {};
          log_start = SteadyClock::now();
        }
      } catch (const std::exception& error) {
        ++invalid_;
        ROS_ERROR_THROTTLE(2.0, "lio_bev: frame failed: %s", error.what());
      }
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lio_bev");
  try {
    BodyBevNode node;
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();
    spinner.stop();
  } catch (const std::exception& error) {
    ROS_FATAL("lio_bev: %s", error.what());
    return 1;
  }
  return 0;
}
