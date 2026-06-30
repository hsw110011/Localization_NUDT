#include <Eigen/Geometry>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kSurfaceTopHeightFraction = 0.15;
constexpr std::size_t kMaxGroundHistoryFrames = 10;

struct BevFrameSample {
  bool valid = false;
  std::size_t linear_index = 0;
  float z = 0.0f;
};

float applyScalarDeadzone(float value, double half_width)
{
  if (half_width <= 0.0 || !std::isfinite(value)) {
    return value;
  }
  if (value >= -static_cast<float>(half_width) &&
      value <= static_cast<float>(half_width)) {
    return 0.0f;
  }
  return value;
}

float medianOfSmallArray(float* values, int count)
{
  if (count <= 0) {
    return 0.0f;
  }
  const int median_index = count / 2;
  std::nth_element(values, values + median_index, values + count);
  return values[median_index];
}

}  // namespace

struct PointXYZ {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

namespace {

std::size_t gridIndex(int row, int col, int cols)
{
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
         static_cast<std::size_t>(col);
}

bool isFinite(double value)
{
  return std::isfinite(value);
}

}  // namespace

class BodyBevBuilder {
public:
  struct Config {
    bool use_odometry_frame = true;
    bool use_ransac_ground = true;
    bool show_windows = true;

    std::string odometry_frame_id = "camera_init";
    std::string body_frame_id = "body";

    double resolution = 0.2;
    double map_size_x = 100.0;
    double map_size_y = 100.0;
    double point_min_z = -5.0;
    double point_max_z = 80.0;
    bool ego_filter_enabled = true;
    double ego_box_min_x = -2.0;
    double ego_box_max_x = 2.0;
    double ego_box_min_y = -2.0;
    double ego_box_max_y = 2.0;

    double near_inner_radius = 2.0;
    double near_outer_radius = 15.0;
    double ground_candidate_min_z = -3.0;
    double ground_candidate_max_z = 1.0;
    double ground_ransac_distance = 0.18;
    double ground_max_plane_tilt_deg = 25.0;
    double ground_front_half_angle_deg = 15.0;
    bool ground_require_forward = true;
    double ground_failure_fallback_z = 0.0;
    int ground_min_points = 30;

    double surface_top_height_fraction = kSurfaceTopHeightFraction;
    double h_rel_deadzone_half_width = 0.20;
    double grad_deadzone_half_width = 0.15;
    double grad_cap = 0.0;

    double height_quantile = 0.90;
    int accumulation_frame_count = 5;
    int cell_max_points = 0;
    int count_saturation = 8;
    double distance_decay_alpha = 0.02;
    bool enable_temporal_height_filter = false;
    double temporal_height_alpha = 0.35;
    double temporal_height_max_jump = 0.35;
    bool enable_spatial_height_smoothing = false;
    int height_smoothing_radius = 2;
    double height_smoothing_max_delta = 0.35;
    int height_smoothing_min_neighbors = 3;
    int edge_min_valid_neighbors = 5;
    int debug_window_stride = 2;
  };

  struct OdometryState {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    ros::Time stamp;
    std::string frame_id;
    bool valid = false;
  };

  void configure(const Config& config)
  {
    config_ = config;
    config_.resolution = std::max(0.01, config_.resolution);
    config_.map_size_x = std::max(config_.resolution, config_.map_size_x);
    config_.map_size_y = std::max(config_.resolution, config_.map_size_y);
    if (config_.ego_box_min_x > config_.ego_box_max_x) {
      std::swap(config_.ego_box_min_x, config_.ego_box_max_x);
    }
    if (config_.ego_box_min_y > config_.ego_box_max_y) {
      std::swap(config_.ego_box_min_y, config_.ego_box_max_y);
    }
    config_.height_quantile = std::max(0.0, std::min(1.0, config_.height_quantile));
    config_.surface_top_height_fraction =
      std::max(0.01, std::min(1.0, config_.surface_top_height_fraction));
    config_.h_rel_deadzone_half_width = std::max(0.0, config_.h_rel_deadzone_half_width);
    config_.grad_deadzone_half_width = std::max(0.0, config_.grad_deadzone_half_width);
    config_.grad_cap = std::max(0.0, config_.grad_cap);
    config_.ground_front_half_angle_deg =
      std::max(0.1, std::min(89.9, config_.ground_front_half_angle_deg));
    if (config_.ground_candidate_min_z > config_.ground_candidate_max_z) {
      std::swap(config_.ground_candidate_min_z, config_.ground_candidate_max_z);
    }
    config_.near_inner_radius = std::max(0.0, config_.near_inner_radius);
    config_.near_outer_radius = std::max(config_.near_inner_radius, config_.near_outer_radius);
    config_.ground_ransac_distance = std::max(0.0, config_.ground_ransac_distance);
    config_.ground_max_plane_tilt_deg =
      std::max(0.0, std::min(89.0, config_.ground_max_plane_tilt_deg));
    config_.ground_min_points = std::max(3, config_.ground_min_points);
    config_.accumulation_frame_count = std::max(0, config_.accumulation_frame_count);
    config_.cell_max_points = std::max(0, config_.cell_max_points);
    config_.count_saturation = std::max(1, config_.count_saturation);
    config_.temporal_height_alpha =
      std::max(0.0, std::min(1.0, config_.temporal_height_alpha));
    config_.temporal_height_max_jump = std::max(0.0, config_.temporal_height_max_jump);
    config_.height_smoothing_radius =
      std::max(0, std::min(5, config_.height_smoothing_radius));
    config_.height_smoothing_max_delta = std::max(0.0, config_.height_smoothing_max_delta);
    config_.height_smoothing_min_neighbors =
      std::max(1, config_.height_smoothing_min_neighbors);
    config_.edge_min_valid_neighbors = std::max(1, std::min(9, config_.edge_min_valid_neighbors));
    config_.debug_window_stride = std::max(1, config_.debug_window_stride);
    if (!config_.enable_spatial_height_smoothing) {
      config_.height_smoothing_radius = 0;
    }
  }

  void build(const std::vector<PointXYZ>& body_points,
             const ros::Time& cloud_stamp,
             const OdometryState* odometry)
  {
    const bool use_odom =
      config_.use_odometry_frame && odometry != nullptr && odometry->valid;

    const std::string frame_id = config_.body_frame_id;
    const grid_map::Position center(0.0, 0.0);
    const Eigen::Isometry3d capture_pose =
      use_odom ? odometry->pose : Eigen::Isometry3d::Identity();
    const bool capture_pose_valid = use_odom;
    stamp_ = capture_pose_valid && !odometry->stamp.isZero()
      ? odometry->stamp
      : (cloud_stamp.isZero() ? ros::Time::now() : cloud_stamp);
    display_pose_ = Eigen::Isometry3d::Identity();
    display_pose_valid_ = true;

    ensureMap(frame_id, center);

    std::vector<PreparedPoint> prepared_points;
    prepared_points.reserve(body_points.size());

    for (const auto& point : body_points) {
      if (!isFinitePoint(point)) {
        continue;
      }
      if (point.z < config_.point_min_z || point.z > config_.point_max_z) {
        continue;
      }
      if (shouldRejectEgoPoint(point)) {
        continue;
      }

      PreparedPoint prepared;
      prepared.body_x = point.x;
      prepared.body_y = point.y;
      prepared.body_z = point.z;

      if (use_odom) {
        const Eigen::Vector3d point_odom =
          odometry->pose * Eigen::Vector3d(point.x, point.y, point.z);
        prepared.x = point_odom.x();
        prepared.y = point_odom.y();
        prepared.z = point_odom.z();
      } else {
        prepared.x = point.x;
        prepared.y = point.y;
        prepared.z = point.z;
      }

      prepared_points.push_back(prepared);
    }

    current_capture_pose_ = capture_pose;
    current_capture_pose_valid_ = capture_pose_valid;

    addFrameToHistory(capture_pose, capture_pose_valid, stamp_, prepared_points);
    if (config_.accumulation_frame_count > 0) {
      while (frame_history_.size() >
             static_cast<std::size_t>(config_.accumulation_frame_count)) {
        frame_history_.pop_front();
      }
    }
    const Eigen::Isometry3d current_pose =
      capture_pose_valid ? capture_pose : Eigen::Isometry3d::Identity();
    pruneFrameHistoryCellPointLimit(current_pose, capture_pose_valid);
    const double ground_reference =
      estimateGroundReferenceFromHistory(current_pose, capture_pose_valid);
    last_ground_reference_ = ground_reference;
    fillMapFromFrameHistory(current_pose, capture_pose_valid, ground_reference);
    if (config_.enable_spatial_height_smoothing &&
        config_.height_smoothing_radius > 0) {
      smoothRelativeHeightLayer();
    }
    computeDirectionalGradients();

    if (config_.show_windows) {
      showDebugWindows();
    }
  }

  void publish(const ros::Publisher& publisher) const
  {
    if (map_.getLayers().empty()) {
      return;
    }

    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(map_, message);
    message.info.header.stamp = stamp_;
    message.info.header.frame_id = map_.getFrameId();
    publisher.publish(message);
  }

  double getGroundReference() const { return last_ground_reference_; }

  int getGroundRoiCandidateCount() const
  {
    return last_ground_roi_candidate_count_;
  }

  int getGroundRoiPlanarCount() const { return last_ground_roi_planar_count_; }

  double getCenterRelativeHeight() const
  {
    if (!map_.exists("H_rel_surf") || !map_.exists("M_L")) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    grid_map::Index center_index;
    if (!map_.getIndex(grid_map::Position(0.0, 0.0), center_index)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const auto& mask = map_["M_L"];
    const auto& h_rel = map_["H_rel_surf"];
    if (mask(center_index(0), center_index(1)) <= 0.5f) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const float value = h_rel(center_index(0), center_index(1));
    return std::isfinite(value) ? static_cast<double>(value)
                                : std::numeric_limits<double>::quiet_NaN();
  }

  double getMaxAbsGradientValue(const std::string& layer_name) const
  {
    if (!map_.exists(layer_name) || !map_.exists("M_L")) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const auto& layer = map_[layer_name];
    const int cols = map_cols_;
    double max_abs = 0.0;
    bool found = false;

    for (const std::size_t linear_index : active_cell_indices_) {
      const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
      const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
      const grid_map::Index index(row, col);
      if (!isDebugPixelValid(index)) {
        continue;
      }

      const float value = layer(index(0), index(1));
      if (!std::isfinite(value)) {
        continue;
      }

      max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
      found = true;
    }

    return found ? max_abs : std::numeric_limits<double>::quiet_NaN();
  }

  double getMaxAbsLongitudinalGradient() const
  {
    return getMaxAbsGradientValue("G_long_L");
  }

  double getMaxAbsLateralGradient() const
  {
    return getMaxAbsGradientValue("G_lat_L");
  }

  void printBevSummary() const
  {
    const double ground_ref = getGroundReference();
    const int roi_pts = getGroundRoiCandidateCount();
    const int planar_pts = getGroundRoiPlanarCount();
    const double center_h_rel = getCenterRelativeHeight();
    const double max_g_long = getMaxAbsLongitudinalGradient();
    const double max_g_lat = getMaxAbsLateralGradient();

    std::cout << "[BEV] ground_roi_pts: " << roi_pts << "/"
              << config_.ground_min_points << " (planar " << planar_pts
              << ") ground_ref: ";
    if (std::isfinite(ground_ref)) {
      std::cout << std::fixed << std::setprecision(3) << ground_ref << " m";
    } else {
      std::cout << "N/A";
    }
    std::cout << " center H_rel_surf: ";
    if (std::isfinite(center_h_rel)) {
      std::cout << std::fixed << std::setprecision(3) << center_h_rel << " m";
      if (std::isfinite(last_center_h_rel_raw_) &&
          std::abs(last_center_h_rel_raw_ - center_h_rel) > 1e-4) {
        std::cout << " (raw " << std::fixed << std::setprecision(3)
                  << last_center_h_rel_raw_ << " m)";
      }
    } else {
      std::cout << "N/A";
    }
    std::cout << " max|G_long_L|: ";
    if (std::isfinite(max_g_long)) {
      std::cout << std::fixed << std::setprecision(4) << max_g_long;
    } else {
      std::cout << "N/A";
    }
    std::cout << " max|G_lat_L|: ";
    if (std::isfinite(max_g_lat)) {
      std::cout << std::fixed << std::setprecision(4) << max_g_lat;
    } else {
      std::cout << "N/A";
    }
    std::cout << std::endl;
  }

private:
  struct PreparedPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double body_x = 0.0;
    double body_y = 0.0;
    double body_z = 0.0;
  };

  struct StoredPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
  };

  struct CellAccumulator {
    int count = 0;
    std::vector<float> top_heights;
  };

  struct FrameObservation {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    ros::Time stamp;
    bool pose_valid = false;
    std::vector<StoredPoint> points;
  };

  Config config_;
  grid_map::GridMap map_;
  bool map_initialized_ = false;
  int map_rows_ = 0;
  int map_cols_ = 0;
  double map_half_x_ = 0.0;
  double map_half_y_ = 0.0;
  double map_inv_resolution_ = 0.0;
  ros::Time stamp_;
  Eigen::Isometry3d display_pose_ = Eigen::Isometry3d::Identity();
  bool display_pose_valid_ = false;
  std::deque<FrameObservation> frame_history_;
  std::vector<CellAccumulator> cell_accumulators_;
  std::vector<std::size_t> active_cell_indices_;
  std::vector<int> cell_keep_counts_;
  std::vector<float> smoothed_height_;
  std::vector<uint8_t> smoothed_valid_;
  std::vector<StoredPoint> prune_kept_buffer_;
  mutable std::vector<float> height_sort_buffer_;
  mutable int debug_window_counter_ = 0;
  Eigen::Isometry3d current_capture_pose_ = Eigen::Isometry3d::Identity();
  bool current_capture_pose_valid_ = false;
  double last_ground_reference_ = std::numeric_limits<double>::quiet_NaN();
  mutable int last_ground_roi_candidate_count_ = 0;
  mutable int last_ground_roi_planar_count_ = 0;
  mutable double last_center_h_rel_raw_ = std::numeric_limits<double>::quiet_NaN();

  void updateMapIndexLookup()
  {
    map_rows_ = map_.getSize()(0);
    map_cols_ = map_.getSize()(1);
    map_half_x_ = config_.map_size_x * 0.5;
    map_half_y_ = config_.map_size_y * 0.5;
    map_inv_resolution_ = 1.0 / config_.resolution;

    const std::size_t cell_count =
      static_cast<std::size_t>(map_rows_) * static_cast<std::size_t>(map_cols_);
    if (cell_accumulators_.size() != cell_count) {
      cell_accumulators_.assign(cell_count, CellAccumulator{});
      active_cell_indices_.clear();
      cell_keep_counts_.assign(cell_count, 0);
    }
    if (smoothed_height_.size() != cell_count) {
      smoothed_height_.assign(cell_count, kNaN);
      smoothed_valid_.assign(cell_count, 0);
    }
  }

  bool positionToCellIndex(double x, double y, int& row, int& col) const
  {
    row = static_cast<int>(std::floor((map_half_x_ - x) * map_inv_resolution_));
    col = static_cast<int>(std::floor((map_half_y_ - y) * map_inv_resolution_));
    return row >= 0 && row < map_rows_ && col >= 0 && col < map_cols_;
  }

  void initializeMap(const std::string& frame_id, const grid_map::Position& center)
  {
    const std::vector<std::string> layers = {
      "H_rel_surf",
      "M_L",
      "G_long_L",
      "G_lat_L",
    };

    map_ = grid_map::GridMap(layers);
    map_.setFrameId(frame_id);
    map_.setGeometry(grid_map::Length(config_.map_size_x, config_.map_size_y),
                     config_.resolution,
                     center);
    map_initialized_ = true;
    updateMapIndexLookup();
    resetMapLayers();
  }

  void resetMapLayers()
  {
    map_["H_rel_surf"].setConstant(kNaN);
    map_["G_long_L"].setConstant(kNaN);
    map_["G_lat_L"].setConstant(kNaN);
    map_["M_L"].setConstant(0.0f);
  }

  void ensureMap(const std::string& frame_id, const grid_map::Position& center)
  {
    const bool same_geometry =
      map_initialized_ &&
      std::abs(map_.getResolution() - config_.resolution) < 1e-9 &&
      std::abs(map_.getLength().x() - config_.map_size_x) < 1e-6 &&
      std::abs(map_.getLength().y() - config_.map_size_y) < 1e-6 &&
      map_.getFrameId() == frame_id;

    if (!same_geometry) {
      initializeMap(frame_id, center);
      return;
    }

    map_.setPosition(center);
    resetMapLayers();
  }

  void addFrameToHistory(const Eigen::Isometry3d& capture_pose,
                         bool pose_valid,
                         const ros::Time& frame_stamp,
                         const std::vector<PreparedPoint>& points)
  {
    if (!pose_valid) {
      frame_history_.clear();
    } else if (!frame_history_.empty() && !frame_history_.back().pose_valid) {
      frame_history_.clear();
    }

    FrameObservation frame;
    frame.pose = capture_pose;
    frame.stamp = frame_stamp;
    frame.pose_valid = pose_valid;
    frame.points.reserve(points.size());
    for (const auto& point : points) {
      frame.points.push_back(
        StoredPoint{point.body_x, point.body_y, point.body_z});
    }

    frame_history_.push_back(std::move(frame));
  }

  void pruneFrameHistoryCellPointLimit(const Eigen::Isometry3d& current_pose,
                                       bool current_pose_valid)
  {
    const int cols = map_cols_;
    const std::size_t cell_count =
      static_cast<std::size_t>(map_rows_) * static_cast<std::size_t>(cols);
    if (cell_keep_counts_.size() != cell_count) {
      cell_keep_counts_.assign(cell_count, 0);
    } else {
      std::fill(cell_keep_counts_.begin(), cell_keep_counts_.end(), 0);
    }

    const bool limit_cell_points = config_.cell_max_points > 0;
    const Eigen::Matrix3d current_rotation = current_pose.linear();
    const Eigen::Vector3d current_translation = current_pose.translation();
    const Eigen::Matrix3d current_rotation_inverse = current_rotation.transpose();
    const Eigen::Vector3d current_translation_inverse =
      -current_rotation_inverse * current_translation;

    for (auto frame_it = frame_history_.rbegin(); frame_it != frame_history_.rend();
         ++frame_it) {
      FrameObservation& frame = *frame_it;
      if (current_pose_valid != frame.pose_valid) {
        frame.points.clear();
        continue;
      }

      Eigen::Matrix3d frame_to_current_rotation = Eigen::Matrix3d::Identity();
      Eigen::Vector3d frame_to_current_translation = Eigen::Vector3d::Zero();
      if (current_pose_valid && frame.pose_valid) {
        const Eigen::Matrix3d frame_rotation = frame.pose.linear();
        const Eigen::Vector3d frame_translation = frame.pose.translation();
        frame_to_current_rotation = current_rotation_inverse * frame_rotation;
        frame_to_current_translation =
          current_rotation_inverse * frame_translation + current_translation_inverse;
      }

      prune_kept_buffer_.clear();
      prune_kept_buffer_.reserve(frame.points.size());

      for (auto point_it = frame.points.rbegin(); point_it != frame.points.rend();
           ++point_it) {
        const Eigen::Vector3d local_point =
          frame_to_current_rotation *
            Eigen::Vector3d(point_it->x, point_it->y, point_it->z) +
          frame_to_current_translation;

        const double local_x = local_point.x();
        const double local_y = local_point.y();
        if (!std::isfinite(local_x) || !std::isfinite(local_y)) {
          continue;
        }

        int row = 0;
        int col = 0;
        if (!positionToCellIndex(local_x, local_y, row, col)) {
          continue;
        }

        if (limit_cell_points) {
          const std::size_t linear_index = gridIndex(row, col, cols);
          int& kept_count = cell_keep_counts_[linear_index];
          if (kept_count >= config_.cell_max_points) {
            continue;
          }
          ++kept_count;
        }

        prune_kept_buffer_.push_back(*point_it);
      }

      frame.points.assign(prune_kept_buffer_.rbegin(), prune_kept_buffer_.rend());
    }

    frame_history_.erase(
      std::remove_if(frame_history_.begin(),
                     frame_history_.end(),
                     [](const FrameObservation& frame) {
                       return frame.points.empty();
                     }),
      frame_history_.end());
  }

  void addTopHeight(CellAccumulator& cell, float height, int max_top_count) const
  {
    if (!std::isfinite(height)) {
      return;
    }

    if (max_top_count <= 0 ||
        cell.top_heights.size() < static_cast<std::size_t>(max_top_count)) {
      cell.top_heights.push_back(height);
      return;
    }

    auto min_it = std::min_element(cell.top_heights.begin(), cell.top_heights.end());
    if (min_it != cell.top_heights.end() && height > *min_it) {
      *min_it = height;
    }
  }

  void fillMapFromFrameHistory(const Eigen::Isometry3d& current_pose,
                               bool current_pose_valid,
                               double ground_reference)
  {
    auto& h_rel_layer = map_["H_rel_surf"];
    auto& mask_layer = map_["M_L"];
    const int cols = map_cols_;

    for (const std::size_t index : active_cell_indices_) {
      CellAccumulator& cell = cell_accumulators_[index];
      cell.count = 0;
      cell.top_heights.clear();
    }
    active_cell_indices_.clear();
    last_center_h_rel_raw_ = std::numeric_limits<double>::quiet_NaN();

    const int max_top_height_count = config_.cell_max_points > 0
      ? std::max(1,
                 static_cast<int>(
                   std::ceil(kSurfaceTopHeightFraction *
                             static_cast<double>(config_.cell_max_points))))
      : 0;
    const Eigen::Matrix3d current_rotation = current_pose.linear();
    const Eigen::Vector3d current_translation = current_pose.translation();
    const Eigen::Matrix3d current_rotation_inverse = current_rotation.transpose();
    const Eigen::Vector3d current_translation_inverse =
      -current_rotation_inverse * current_translation;

    for (const auto& frame : frame_history_) {
      if (current_pose_valid != frame.pose_valid) {
        continue;
      }

      Eigen::Matrix3d frame_to_current_rotation = Eigen::Matrix3d::Identity();
      Eigen::Vector3d frame_to_current_translation = Eigen::Vector3d::Zero();
      if (current_pose_valid && frame.pose_valid) {
        const Eigen::Matrix3d frame_rotation = frame.pose.linear();
        const Eigen::Vector3d frame_translation = frame.pose.translation();
        frame_to_current_rotation = current_rotation_inverse * frame_rotation;
        frame_to_current_translation =
          current_rotation_inverse * frame_translation + current_translation_inverse;
      }

      std::vector<BevFrameSample> frame_samples(frame.points.size());
      for (std::size_t point_index = 0; point_index < frame.points.size(); ++point_index) {
        const auto& point = frame.points[point_index];
        BevFrameSample& sample = frame_samples[point_index];
        sample.valid = false;

        const Eigen::Vector3d local_point =
          frame_to_current_rotation *
            Eigen::Vector3d(point.x, point.y, point.z) +
          frame_to_current_translation;

        int row = 0;
        int col = 0;
        if (!positionToCellIndex(local_point.x(), local_point.y(), row, col)) {
          continue;
        }

        sample.valid = true;
        sample.linear_index = gridIndex(row, col, cols);
        sample.z = static_cast<float>(local_point.z());
      }

      for (const BevFrameSample& sample : frame_samples) {
        if (!sample.valid) {
          continue;
        }

        CellAccumulator& cell = cell_accumulators_[sample.linear_index];
        if (cell.count == 0) {
          active_cell_indices_.push_back(sample.linear_index);
        }
        ++cell.count;
        addTopHeight(cell, sample.z, max_top_height_count);
      }
    }

    for (const std::size_t linear_index : active_cell_indices_) {
      CellAccumulator& cell = cell_accumulators_[linear_index];
      if (cell.count <= 0) {
        continue;
      }

      const int row =
        static_cast<int>(linear_index / static_cast<std::size_t>(cols));
      const int col =
        static_cast<int>(linear_index % static_cast<std::size_t>(cols));
      const float h_q = static_cast<float>(
        meanTopHeightFraction(cell.top_heights,
                              config_.surface_top_height_fraction,
                              cell.count));
      const float h_rel_raw =
        static_cast<float>(static_cast<double>(h_q) - ground_reference);
      if (!std::isfinite(h_rel_raw)) {
        continue;
      }

      h_rel_layer(row, col) =
        applyScalarDeadzone(h_rel_raw, config_.h_rel_deadzone_half_width);
      mask_layer(row, col) = 1.0f;

      int center_row = 0;
      int center_col = 0;
      if (positionToCellIndex(0.0, 0.0, center_row, center_col) &&
          row == center_row && col == center_col) {
        last_center_h_rel_raw_ = static_cast<double>(h_rel_raw);
      }
    }
  }

  void smoothRelativeHeightLayer()
  {
    if (config_.height_smoothing_radius <= 0 || !map_.exists("H_rel_surf") ||
        !map_.exists("M_L")) {
      return;
    }

    auto& h_rel_layer = map_["H_rel_surf"];
    const auto& mask_layer = map_["M_L"];
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);
    std::vector<float> smoothed(cell_count, kNaN);
    std::vector<uint8_t> valid(cell_count, 0);

    const int radius = config_.height_smoothing_radius;
    const double max_delta = config_.height_smoothing_max_delta;

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const float center_height = h_rel_layer(row, col);
        if (mask_layer(row, col) <= 0.5f || !std::isfinite(center_height)) {
          continue;
        }

        double weighted_sum = static_cast<double>(center_height) * 2.0;
        double weight_sum = 2.0;
        int accepted_neighbors = 1;

        for (int dr = -radius; dr <= radius; ++dr) {
          for (int dc = -radius; dc <= radius; ++dc) {
            if (dr == 0 && dc == 0) {
              continue;
            }

            const int neighbor_row = row + dr;
            const int neighbor_col = col + dc;
            if (neighbor_row < 0 || neighbor_row >= rows ||
                neighbor_col < 0 || neighbor_col >= cols) {
              continue;
            }
            if (mask_layer(neighbor_row, neighbor_col) <= 0.5f) {
              continue;
            }

            const float neighbor_height = h_rel_layer(neighbor_row, neighbor_col);
            if (!std::isfinite(neighbor_height)) {
              continue;
            }
            if (max_delta > 0.0 &&
                std::abs(static_cast<double>(neighbor_height - center_height)) > max_delta) {
              continue;
            }

            const double distance_sq = static_cast<double>(dr * dr + dc * dc);
            const double spatial_weight = 1.0 / (1.0 + distance_sq);
            weighted_sum += static_cast<double>(neighbor_height) * spatial_weight;
            weight_sum += spatial_weight;
            ++accepted_neighbors;
          }
        }

        const std::size_t index = gridIndex(row, col, cols);
        if (accepted_neighbors >= config_.height_smoothing_min_neighbors && weight_sum > 1e-6) {
          smoothed[index] = static_cast<float>(weighted_sum / weight_sum);
        } else {
          smoothed[index] = center_height;
        }
        valid[index] = 1;
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const std::size_t index = gridIndex(row, col, cols);
        if (valid[index]) {
          h_rel_layer(row, col) = smoothed[index];
        }
      }
    }
  }

  bool shouldRejectEgoPoint(const PointXYZ& point) const
  {
    if (!config_.ego_filter_enabled) {
      return false;
    }

    return point.x >= config_.ego_box_min_x && point.x <= config_.ego_box_max_x &&
           point.y >= config_.ego_box_min_y && point.y <= config_.ego_box_max_y;
  }

  void computeDirectionalGradients()
  {
    auto& h_rel_layer = map_["H_rel_surf"];
    auto& mask_layer = map_["M_L"];
    auto& longitudinal_gradient_layer = map_["G_long_L"];
    auto& lateral_gradient_layer = map_["G_lat_L"];

    longitudinal_gradient_layer.setConstant(kNaN);
    lateral_gradient_layer.setConstant(kNaN);

    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);

    std::vector<float> smoothed_height(cell_count, kNaN);
    std::vector<uint8_t> smoothed_valid(cell_count, 0);

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        if (mask_layer(row, col) <= 0.5f || !std::isfinite(h_rel_layer(row, col))) {
          continue;
        }

        float values[9];
        int value_count = 0;
        for (int dr = -1; dr <= 1; ++dr) {
          for (int dc = -1; dc <= 1; ++dc) {
            const int neighbor_row = row + dr;
            const int neighbor_col = col + dc;
            if (neighbor_row < 0 || neighbor_row >= rows ||
                neighbor_col < 0 || neighbor_col >= cols) {
              continue;
            }
            if (mask_layer(neighbor_row, neighbor_col) <= 0.5f ||
                !std::isfinite(h_rel_layer(neighbor_row, neighbor_col))) {
              continue;
            }
            values[value_count] = h_rel_layer(neighbor_row, neighbor_col);
            ++value_count;
          }
        }

        if (value_count < config_.edge_min_valid_neighbors) {
          continue;
        }

        const std::size_t index = gridIndex(row, col, cols);
        smoothed_height[index] = medianOfSmallArray(values, value_count);
        smoothed_valid[index] = 1;
      }
    }

    static const int sobel_x[3][3] = {
      {-1, 0, 1},
      {-2, 0, 2},
      {-1, 0, 1}
    };
    static const int sobel_y[3][3] = {
      {-1, -2, -1},
      {0, 0, 0},
      {1, 2, 1}
    };

    Eigen::Vector2d index0_unit(1.0, 0.0);
    Eigen::Vector2d index1_unit(0.0, 1.0);
    const int reference_row = std::min(rows - 1, std::max(0, rows / 2));
    const int reference_col = std::min(cols - 1, std::max(0, cols / 2));
    grid_map::Position reference_position;
    grid_map::Position index0_position;
    grid_map::Position index1_position;
    const bool has_reference_position =
      map_.getPosition(grid_map::Index(reference_row, reference_col), reference_position);
    if (has_reference_position && reference_row + 1 < rows &&
        map_.getPosition(grid_map::Index(reference_row + 1, reference_col), index0_position)) {
      index0_unit = Eigen::Vector2d(index0_position.x() - reference_position.x(),
                                    index0_position.y() - reference_position.y());
      if (index0_unit.norm() > 1e-6) {
        index0_unit.normalize();
      }
    }
    if (has_reference_position && reference_col + 1 < cols &&
        map_.getPosition(grid_map::Index(reference_row, reference_col + 1), index1_position)) {
      index1_unit = Eigen::Vector2d(index1_position.x() - reference_position.x(),
                                    index1_position.y() - reference_position.y());
      if (index1_unit.norm() > 1e-6) {
        index1_unit.normalize();
      }
    }

    Eigen::Vector2d longitudinal_unit(1.0, 0.0);
    Eigen::Vector2d lateral_unit(0.0, 1.0);
    if (display_pose_valid_) {
      longitudinal_unit = Eigen::Vector2d(display_pose_.linear()(0, 0),
                                          display_pose_.linear()(1, 0));
      lateral_unit = Eigen::Vector2d(display_pose_.linear()(0, 1),
                                   display_pose_.linear()(1, 1));
      if (longitudinal_unit.norm() > 1e-6) {
        longitudinal_unit.normalize();
      }
      if (lateral_unit.norm() > 1e-6) {
        lateral_unit.normalize();
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const std::size_t index = gridIndex(row, col, cols);
        if (!smoothed_valid[index]) {
          continue;
        }

        const float center_height = smoothed_height[index];
        if (!std::isfinite(center_height)) {
          continue;
        }

        int valid_neighbor_count = 0;
        double grad_index1 = 0.0;
        double grad_index0 = 0.0;

        for (int dr = -1; dr <= 1; ++dr) {
          for (int dc = -1; dc <= 1; ++dc) {
            const int neighbor_row = row + dr;
            const int neighbor_col = col + dc;
            if (neighbor_row < 0 || neighbor_row >= rows ||
                neighbor_col < 0 || neighbor_col >= cols) {
              continue;
            }

            const std::size_t neighbor_index = gridIndex(neighbor_row, neighbor_col, cols);
            float value = center_height;
            if (smoothed_valid[neighbor_index]) {
              const float neighbor_value = smoothed_height[neighbor_index];
              if (!std::isfinite(neighbor_value)) {
                continue;
              }
              value = neighbor_value;
              if (dr != 0 || dc != 0) {
                ++valid_neighbor_count;
              }
            }

            grad_index0 += static_cast<double>(sobel_y[dr + 1][dc + 1]) *
                           static_cast<double>(value);
            grad_index1 += static_cast<double>(sobel_x[dr + 1][dc + 1]) *
                           static_cast<double>(value);
          }
        }

        if (valid_neighbor_count < config_.edge_min_valid_neighbors) {
          continue;
        }

        const double index0_gradient = grad_index0 / (8.0 * config_.resolution);
        const double index1_gradient = grad_index1 / (8.0 * config_.resolution);
        const Eigen::Vector2d map_gradient =
          index0_gradient * index0_unit + index1_gradient * index1_unit;

        const float longitudinal_gradient = applyScalarDeadzone(
          static_cast<float>(map_gradient.dot(longitudinal_unit)),
          config_.grad_deadzone_half_width);
        const float lateral_gradient = applyScalarDeadzone(
          static_cast<float>(map_gradient.dot(lateral_unit)),
          config_.grad_deadzone_half_width);

        if (config_.grad_cap > 0.0) {
          const float cap = static_cast<float>(config_.grad_cap);
          longitudinal_gradient_layer(row, col) = std::max(
            -cap, std::min(cap, longitudinal_gradient));
          lateral_gradient_layer(row, col) = std::max(
            -cap, std::min(cap, lateral_gradient));
        } else {
          longitudinal_gradient_layer(row, col) = longitudinal_gradient;
          lateral_gradient_layer(row, col) = lateral_gradient;
        }
      }
    }
  }

  bool isGroundRoiPlanarPoint(double x, double y) const
  {
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return false;
    }

    const double distance_sq = x * x + y * y;
    const double inner_sq =
      config_.near_inner_radius * config_.near_inner_radius;
    const double outer_sq =
      config_.near_outer_radius * config_.near_outer_radius;
    if (distance_sq < inner_sq || distance_sq > outer_sq) {
      return false;
    }

    if (config_.ground_require_forward && x <= 0.0) {
      return false;
    }

    const double front_half_angle_rad =
      config_.ground_front_half_angle_deg * M_PI / 180.0;
    const double angle = std::atan2(y, x);
    return std::abs(angle) <= front_half_angle_rad;
  }

  bool isGroundRoiBodyPoint(double x, double y, double z) const
  {
    if (!isGroundRoiPlanarPoint(x, y) || !std::isfinite(z)) {
      return false;
    }
    return z >= config_.ground_candidate_min_z && z <= config_.ground_candidate_max_z;
  }

  double estimateGroundReferenceFromHistory(const Eigen::Isometry3d& current_pose,
                                            bool current_pose_valid) const
  {
    const std::size_t reference_point_count =
      static_cast<std::size_t>(config_.ground_min_points);
    std::vector<std::pair<double, double>> ground_candidates;
    ground_candidates.reserve(4096);
    int planar_count = 0;

    const Eigen::Matrix3d current_rotation = current_pose.linear();
    const Eigen::Vector3d current_translation = current_pose.translation();
    const Eigen::Matrix3d current_rotation_inverse = current_rotation.transpose();
    const Eigen::Vector3d current_translation_inverse =
      -current_rotation_inverse * current_translation;

    std::size_t frames_used = 0;
    for (auto frame_it = frame_history_.rbegin();
         frame_it != frame_history_.rend() &&
         frames_used < kMaxGroundHistoryFrames;
         ++frame_it, ++frames_used) {
      const FrameObservation& frame = *frame_it;
      if (current_pose_valid != frame.pose_valid) {
        continue;
      }

      Eigen::Matrix3d frame_to_current_rotation = Eigen::Matrix3d::Identity();
      Eigen::Vector3d frame_to_current_translation = Eigen::Vector3d::Zero();
      if (current_pose_valid && frame.pose_valid) {
        const Eigen::Matrix3d frame_rotation = frame.pose.linear();
        const Eigen::Vector3d frame_translation = frame.pose.translation();
        frame_to_current_rotation = current_rotation_inverse * frame_rotation;
        frame_to_current_translation =
          current_rotation_inverse * frame_translation + current_translation_inverse;
      }

      for (const auto& point : frame.points) {
        const Eigen::Vector3d local_point =
          frame_to_current_rotation *
            Eigen::Vector3d(point.x, point.y, point.z) +
          frame_to_current_translation;
        const double local_x = local_point.x();
        const double local_y = local_point.y();
        const double local_z = local_point.z();
        if (isGroundRoiPlanarPoint(local_x, local_y)) {
          ++planar_count;
        }
        if (!isGroundRoiBodyPoint(local_x, local_y, local_z)) {
          continue;
        }
        const double distance_sq = local_x * local_x + local_y * local_y;
        ground_candidates.emplace_back(distance_sq, local_z);
      }
    }

    last_ground_roi_planar_count_ = planar_count;
    last_ground_roi_candidate_count_ =
      static_cast<int>(ground_candidates.size());

    if (ground_candidates.size() >= reference_point_count) {
      const std::size_t used_count =
        std::min(reference_point_count, ground_candidates.size());
      std::nth_element(ground_candidates.begin(),
                       ground_candidates.begin() +
                         static_cast<std::ptrdiff_t>(used_count - 1),
                       ground_candidates.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                       });

      double z_sum = 0.0;
      for (std::size_t i = 0; i < used_count; ++i) {
        z_sum += ground_candidates[i].second;
      }
      return z_sum / static_cast<double>(used_count);
    }

    return config_.ground_failure_fallback_z;
  }

  double meanTopHeightFraction(const std::vector<float>& values,
                                 double top_fraction,
                                 int total_count) const
  {
    height_sort_buffer_.clear();
    height_sort_buffer_.reserve(values.size());
    for (float value : values) {
      if (std::isfinite(value)) {
        height_sort_buffer_.push_back(value);
      }
    }
    if (height_sort_buffer_.empty()) {
      return 0.0;
    }

    if (!std::isfinite(top_fraction) || top_fraction <= 0.0) {
      top_fraction = kSurfaceTopHeightFraction;
    }
    top_fraction = std::min(1.0, top_fraction);

    const int bounded_count = std::max(1, total_count);
    const std::size_t requested_count = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(
        std::ceil(top_fraction * static_cast<double>(bounded_count))));
    const std::size_t used_count =
      std::min(requested_count, height_sort_buffer_.size());

    if (used_count < height_sort_buffer_.size()) {
      std::nth_element(height_sort_buffer_.begin(),
                       height_sort_buffer_.begin() +
                         static_cast<std::ptrdiff_t>(used_count - 1),
                       height_sort_buffer_.end(),
                       std::greater<float>());
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < used_count; ++i) {
      sum += static_cast<double>(height_sort_buffer_[i]);
    }
    return sum / static_cast<double>(used_count);
  }

  bool isFinitePoint(const PointXYZ& point) const
  {
    return isFinite(point.x) && isFinite(point.y) && isFinite(point.z);
  }

  double percentile(std::vector<float> values, double quantile) const
  {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](float value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty()) {
      return 0.0;
    }

    quantile = std::max(0.0, std::min(1.0, quantile));
    const std::size_t index = static_cast<std::size_t>(
      std::floor(quantile * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
  }

  std::pair<double, double> getRobustLayerRange(const std::string& layer_name,
                                                double fallback_min,
                                                double fallback_max) const
  {
    if (!map_.exists(layer_name) || !map_.exists("M_L")) {
      return {fallback_min, fallback_max};
    }

    const auto& layer = map_[layer_name];
    const int cols = map_cols_;
    std::vector<float> values;
    values.reserve(active_cell_indices_.size());

    for (const std::size_t linear_index : active_cell_indices_) {
      const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
      const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
      const grid_map::Index index(row, col);
      if (!isDebugPixelValid(index)) {
        continue;
      }

      const float value = layer(index(0), index(1));
      if (!std::isfinite(value)) {
        continue;
      }
      values.push_back(value);
    }

    if (values.size() < 2) {
      return {fallback_min, fallback_max};
    }

    const double low = percentile(values, 0.02);
    const double high = percentile(values, 0.98);
    if (!std::isfinite(low) || !std::isfinite(high) || high - low < 1e-3) {
      return {fallback_min, fallback_max};
    }

    return {low, high};
  }

  bool isDebugPixelValid(const grid_map::Index& index) const
  {
    if (!map_.exists("M_L") || !map_.exists("H_rel_surf")) {
      return false;
    }

    const auto& mask = map_["M_L"];
    const auto& height = map_["H_rel_surf"];
    return mask(index(0), index(1)) > 0.5f &&
           std::isfinite(height(index(0), index(1)));
  }

  cv::Vec3b viridisColor(double normalized_value) const
  {
    static const std::array<cv::Vec3b, 256> lut = []() {
      cv::Mat gray(1, 256, CV_8UC1);
      for (int i = 0; i < gray.cols; ++i) {
        gray.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
      }
      cv::Mat color;
      cv::applyColorMap(gray, color, cv::COLORMAP_VIRIDIS);
      std::array<cv::Vec3b, 256> table;
      for (int i = 0; i < color.cols; ++i) {
        table[static_cast<std::size_t>(i)] = color.at<cv::Vec3b>(0, i);
      }
      return table;
    }();

    const double t = std::max(0.0, std::min(1.0, normalized_value));
    const int lut_index = static_cast<int>(std::round(t * 255.0));
    return lut[static_cast<std::size_t>(std::max(0, std::min(255, lut_index)))];
  }

  void vehicleBodyToImagePixel(double x, double y, double& row, double& col) const
  {
    row = (map_half_x_ - x) * map_inv_resolution_ - 0.5;
    col = (map_half_y_ - y) * map_inv_resolution_ - 0.5;
  }

  void drawCenterMark(cv::Mat& image) const
  {
    if (image.empty()) {
      return;
    }

    double origin_row = 0.0;
    double origin_col = 0.0;
    vehicleBodyToImagePixel(0.0, 0.0, origin_row, origin_col);
    const int center_col = static_cast<int>(std::round(origin_col));
    const int center_row = static_cast<int>(std::round(origin_row));
    const int radius =
      std::max(3, static_cast<int>(std::round(0.5 / config_.resolution)));
    cv::circle(image,
               cv::Point(center_col, center_row),
               radius,
               cv::Scalar(0, 0, 255),
               -1,
               cv::LINE_AA);
  }

  void drawGroundReferenceRegion(cv::Mat& image) const
  {
    if (image.empty() || map_rows_ <= 0 || map_cols_ <= 0) {
      return;
    }
    if (image.rows != map_rows_ || image.cols != map_cols_ || image.type() != CV_8UC3) {
      return;
    }

    double origin_row = 0.0;
    double origin_col = 0.0;
    vehicleBodyToImagePixel(0.0, 0.0, origin_row, origin_col);
    const cv::Point origin(static_cast<int>(std::round(origin_col)),
                           static_cast<int>(std::round(origin_row)));

    const int inner_px = std::max(
      1, static_cast<int>(std::round(config_.near_inner_radius * map_inv_resolution_)));
    const int outer_px = std::max(
      inner_px + 1,
      static_cast<int>(std::round(config_.near_outer_radius * map_inv_resolution_)));

    const double half_angle_rad = config_.ground_front_half_angle_deg * M_PI / 180.0;
    const double inner_r = config_.near_inner_radius;
    const double outer_r = config_.near_outer_radius;

    cv::Mat overlay = image.clone();

    constexpr int kArcSteps = 36;
    std::vector<cv::Point> sector_annulus;
    sector_annulus.reserve(static_cast<std::size_t>(kArcSteps) * 2 + 2);
    for (int i = 0; i <= kArcSteps; ++i) {
      const double angle =
        -half_angle_rad +
        (2.0 * half_angle_rad) * static_cast<double>(i) / static_cast<double>(kArcSteps);
      double row = 0.0;
      double col = 0.0;
      vehicleBodyToImagePixel(inner_r * std::cos(angle), inner_r * std::sin(angle), row, col);
      sector_annulus.emplace_back(static_cast<int>(std::round(col)),
                                  static_cast<int>(std::round(row)));
    }
    for (int i = kArcSteps; i >= 0; --i) {
      const double angle =
        -half_angle_rad +
        (2.0 * half_angle_rad) * static_cast<double>(i) / static_cast<double>(kArcSteps);
      double row = 0.0;
      double col = 0.0;
      vehicleBodyToImagePixel(outer_r * std::cos(angle), outer_r * std::sin(angle), row, col);
      sector_annulus.emplace_back(static_cast<int>(std::round(col)),
                                  static_cast<int>(std::round(row)));
    }
    if (sector_annulus.size() >= 3) {
      const std::vector<std::vector<cv::Point>> contours{sector_annulus};
      cv::fillPoly(overlay, contours, cv::Scalar(255, 200, 0));
    }

    cv::Mat blended;
    cv::addWeighted(overlay, 0.30, image, 0.70, 0.0, blended);
    blended.copyTo(image);

    const cv::Scalar ring_color(0, 255, 255);
    const cv::Scalar sector_color(255, 0, 255);
    cv::circle(image, origin, inner_px, ring_color, 1, cv::LINE_AA);
    cv::circle(image, origin, outer_px, ring_color, 1, cv::LINE_AA);

    for (const double angle : {-half_angle_rad, half_angle_rad}) {
      double row = 0.0;
      double col = 0.0;
      vehicleBodyToImagePixel(outer_r * std::cos(angle), outer_r * std::sin(angle), row, col);
      cv::line(image,
               origin,
               cv::Point(static_cast<int>(std::round(col)),
                         static_cast<int>(std::round(row))),
               sector_color,
               1,
               cv::LINE_AA);
    }

    if (config_.ground_require_forward) {
      double row_neg = 0.0;
      double col_neg = 0.0;
      double row_pos = 0.0;
      double col_pos = 0.0;
      vehicleBodyToImagePixel(0.0, -outer_r, row_neg, col_neg);
      vehicleBodyToImagePixel(0.0, outer_r, row_pos, col_pos);
      cv::line(image,
               cv::Point(static_cast<int>(std::round(col_neg)),
                         static_cast<int>(std::round(row_neg))),
               cv::Point(static_cast<int>(std::round(col_pos)),
                         static_cast<int>(std::round(row_pos))),
               cv::Scalar(180, 180, 180),
               1,
               cv::LINE_AA);
    }
  }

  cv::Mat renderColorizedLayer(const std::string& layer_name,
                               double min_value,
                               double max_value) const
  {
    const int rows = map_rows_;
    const int cols = map_cols_;
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name) || !map_.exists("M_L") || !map_.exists("H_rel_surf")) {
      return image;
    }

    const auto& layer = map_[layer_name];
    const double inv_range =
      max_value > min_value ? 1.0 / (max_value - min_value) : 1.0;

    for (const std::size_t linear_index : active_cell_indices_) {
      const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
      const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
      const grid_map::Index index(row, col);
      if (!isDebugPixelValid(index)) {
        continue;
      }

      const float value = layer(index(0), index(1));
      if (!std::isfinite(value)) {
        continue;
      }

      const double normalized =
        (static_cast<double>(value) - min_value) * inv_range;
      image.at<cv::Vec3b>(row, col) = viridisColor(normalized);
    }

    return image;
  }

  cv::Mat renderMaskLayer() const
  {
    const int rows = map_rows_;
    const int cols = map_cols_;
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists("M_L")) {
      return image;
    }

    const auto& layer = map_["M_L"];
    for (const std::size_t linear_index : active_cell_indices_) {
      const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
      const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
      if (layer(row, col) > 0.5f) {
        image.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
      }
    }

    drawCenterMark(image);
    return image;
  }

  cv::Mat renderStackedDebugLayers() const
  {
    const std::pair<double, double> height_range =
      getRobustLayerRange("H_rel_surf", 0.0, 1.0);
    const std::pair<double, double> longitudinal_range =
      getRobustLayerRange("G_long_L", 0.0, 1.0);
    const std::pair<double, double> lateral_range =
      getRobustLayerRange("G_lat_L", 0.0, 1.0);

    cv::Mat height_image =
      renderColorizedLayer("H_rel_surf", height_range.first, height_range.second);
    cv::Mat longitudinal_image =
      renderColorizedLayer("G_long_L", longitudinal_range.first, longitudinal_range.second);
    cv::Mat lateral_image =
      renderColorizedLayer("G_lat_L", lateral_range.first, lateral_range.second);
    cv::Mat mask_image = renderMaskLayer();

    if (!height_image.empty()) {
      drawGroundReferenceRegion(height_image);
      drawCenterMark(height_image);
    }
    if (!longitudinal_image.empty()) {
      drawCenterMark(longitudinal_image);
    }
    if (!lateral_image.empty()) {
      drawCenterMark(lateral_image);
    }

    cv::Mat combined;
    if (height_image.empty() || longitudinal_image.empty() || lateral_image.empty() ||
        mask_image.empty() ||
        height_image.rows != longitudinal_image.rows ||
        height_image.rows != lateral_image.rows ||
        height_image.rows != mask_image.rows) {
      return height_image;
    }
    cv::hconcat(height_image, longitudinal_image, combined);
    cv::hconcat(combined, lateral_image, combined);
    cv::hconcat(combined, mask_image, combined);
    return combined;
  }

  void showDebugWindows() const
  {
    if (map_.getLayers().empty()) {
      return;
    }

    ++debug_window_counter_;
    if (config_.debug_window_stride > 1 &&
        debug_window_counter_ % config_.debug_window_stride != 0) {
      cv::waitKey(1);
      return;
    }

    static const std::string window_name = "BEV H_rel | G_long | G_lat | M_L";
    static bool windows_created = false;
    if (!windows_created) {
      cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
      cv::resizeWindow(window_name, 2000, 500);
      windows_created = true;
    }

    cv::imshow(window_name, renderStackedDebugLayers());
    cv::waitKey(1);
  }
};

class BodyBevNode {
public:
  BodyBevNode()
    : nh_(),
      pnh_("~")
  {
    pnh_.param<std::string>("input_topic", input_topic_, "/cloud_registered_body");
    pnh_.param<std::string>("odometry_topic", odometry_topic_, "/Odometry");
    pnh_.param<std::string>("output_topic", output_topic_, "/lidar_bev/grid_map");
    pnh_.param<double>("update_rate_hz", update_rate_hz_, 10.0);

    BodyBevBuilder::Config config;
    pnh_.param<bool>("use_odometry_frame", config.use_odometry_frame, true);
    pnh_.param<bool>("use_ransac_ground", config.use_ransac_ground, true);
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
    pnh_.param<double>("near_inner_radius", config.near_inner_radius, 2.0);
    pnh_.param<double>("near_outer_radius", config.near_outer_radius, 15.0);
    pnh_.param<double>("ground_candidate_min_z", config.ground_candidate_min_z, -3.0);
    pnh_.param<double>("ground_candidate_max_z", config.ground_candidate_max_z, 1.0);
    pnh_.param<double>("ground_ransac_distance", config.ground_ransac_distance, 0.18);
    pnh_.param<double>("ground_max_plane_tilt_deg", config.ground_max_plane_tilt_deg, 25.0);
    pnh_.param<double>("ground_front_half_angle_deg", config.ground_front_half_angle_deg, 15.0);
    pnh_.param<bool>("ground_require_forward", config.ground_require_forward, true);
    pnh_.param<double>("ground_failure_fallback_z", config.ground_failure_fallback_z, 0.0);
    pnh_.param<int>("ground_min_points", config.ground_min_points, 30);
    pnh_.param<double>("surface_top_height_fraction", config.surface_top_height_fraction, 0.15);
    pnh_.param<double>("h_rel_deadzone_half_width", config.h_rel_deadzone_half_width, 0.20);
    pnh_.param<double>("grad_deadzone_half_width", config.grad_deadzone_half_width, 0.15);
    pnh_.param<double>("grad_cap", config.grad_cap, 0.0);
    pnh_.param<double>("height_quantile", config.height_quantile, 0.90);
    pnh_.param<int>("accumulation_frame_count", config.accumulation_frame_count, 5);
    pnh_.param<int>("cell_max_points", config.cell_max_points, 0);
    pnh_.param<int>("count_saturation", config.count_saturation, 8);
    pnh_.param<double>("distance_decay_alpha", config.distance_decay_alpha, 0.02);
    pnh_.param<bool>("enable_temporal_height_filter", config.enable_temporal_height_filter, false);
    pnh_.param<double>("temporal_height_alpha", config.temporal_height_alpha, 0.35);
    pnh_.param<double>("temporal_height_max_jump", config.temporal_height_max_jump, 0.35);
    pnh_.param<bool>("enable_spatial_height_smoothing", config.enable_spatial_height_smoothing, false);
    pnh_.param<int>("height_smoothing_radius", config.height_smoothing_radius, 2);
    pnh_.param<double>("height_smoothing_max_delta", config.height_smoothing_max_delta, 0.35);
    pnh_.param<int>("height_smoothing_min_neighbors", config.height_smoothing_min_neighbors, 3);
    pnh_.param<int>("edge_min_valid_neighbors", config.edge_min_valid_neighbors, 5);
    pnh_.param<int>("debug_window_stride", config.debug_window_stride, 2);
    pnh_.param<bool>("show_windows", config.show_windows, true);

    use_odometry_frame_ = config.use_odometry_frame;
    odometry_frame_id_ = config.odometry_frame_id;
    builder_.configure(config);
    pub_ = nh_.advertise<grid_map_msgs::GridMap>(output_topic_, 5);
    cloud_sub_ = nh_.subscribe(input_topic_, 5, &BodyBevNode::cloudCallback, this);
    if (use_odometry_frame_) {
      odom_sub_ = nh_.subscribe(odometry_topic_, 20, &BodyBevNode::odometryCallback, this);
    }

    ROS_INFO("FastLioBodyBev: cloud=%s, odometry=%s, publish=%s, frame=%s",
             input_topic_.c_str(),
             use_odometry_frame_ ? odometry_topic_.c_str() : "(disabled)",
             output_topic_.c_str(),
             (config.body_frame_id).c_str());
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher pub_;
  BodyBevBuilder builder_;
  std::string input_topic_;
  std::string odometry_topic_;
  std::string output_topic_;
  std::string odometry_frame_id_;
  bool use_odometry_frame_ = true;
  double update_rate_hz_ = 10.0;
  ros::Time last_update_;
  mutable std::mutex odometry_mutex_;
  BodyBevBuilder::OdometryState latest_odometry_;

  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
  {
    Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                         msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y,
                         msg->pose.pose.orientation.z);
    if (q.norm() < 1e-6) {
      q = Eigen::Quaterniond::Identity();
    } else {
      q.normalize();
    }

    BodyBevBuilder::OdometryState odometry;
    odometry.pose = Eigen::Isometry3d::Identity();
    odometry.pose.linear() = q.toRotationMatrix();
    odometry.pose.translation() = Eigen::Vector3d(msg->pose.pose.position.x,
                                                  msg->pose.pose.position.y,
                                                  msg->pose.pose.position.z);
    odometry.stamp = msg->header.stamp;
    odometry.frame_id =
      msg->header.frame_id.empty() ? odometry_frame_id_ : msg->header.frame_id;
    odometry.valid = true;

    std::lock_guard<std::mutex> lock(odometry_mutex_);
    latest_odometry_ = odometry;
  }

  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
  {
    if (update_rate_hz_ > 0.0 && !last_update_.isZero()) {
      const double dt = (msg->header.stamp - last_update_).toSec();
      if (dt >= 0.0 && dt < 1.0 / update_rate_hz_) {
        return;
      }
    }

    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::vector<PointXYZ> points;
    points.reserve(cloud.points.size());
    for (const auto& pt : cloud.points) {
      PointXYZ point;
      point.x = pt.x;
      point.y = pt.y;
      point.z = pt.z;
      points.push_back(point);
    }

    BodyBevBuilder::OdometryState odometry;
    const BodyBevBuilder::OdometryState* odometry_ptr = nullptr;
    if (use_odometry_frame_) {
      std::lock_guard<std::mutex> lock(odometry_mutex_);
      odometry = latest_odometry_;
      if (odometry.valid) {
        odometry_ptr = &odometry;
      }
    }

    if (use_odometry_frame_ && odometry_ptr == nullptr) {
      ROS_WARN_THROTTLE(1.0, "Waiting for %s before publishing LiDAR BEV.",
                        odometry_topic_.c_str());
      return;
    }

    builder_.build(points, msg->header.stamp, odometry_ptr);
    builder_.publish(pub_);
    builder_.printBevSummary();
    last_update_ = msg->header.stamp;
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "fast_lio_body_bev");
  BodyBevNode node;
  ros::spin();
  return 0;
}
