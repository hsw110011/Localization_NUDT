#pragma once

#include <Eigen/Geometry>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "lio_bev/rolling_grid.hpp"
#include "lio_bev/input_utils.hpp"

#include <chrono>
#include <stdexcept>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lio_bev {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
using SteadyClock = std::chrono::steady_clock;

inline double elapsedMs(SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

inline float applyScalarDeadzone(float value, double half_width)
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

inline float medianOfSmallArray(float* values, int count)
{
  if (count <= 0) {
    return 0.0f;
  }
  const int median_index = count / 2;
  std::nth_element(values, values + median_index, values + count);
  return values[median_index];
}

inline std::size_t gridIndex(int row, int col, int cols)
{
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
         static_cast<std::size_t>(col);
}

class BodyBevBuilder {
public:
  struct Config {
    bool use_odometry_frame = true;
    bool input_cloud_in_world = false;
    // Only the published grid reference changes; capture/stabilization retain SE(3).
    bool gravity_aligned_output = false;

    std::string odometry_frame_id = "camera_init";
    std::string body_frame_id = "body";
    std::string bev_frame_id = "bev_local";

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

    double h_rel_deadzone_half_width = 0.20;
    double grad_deadzone_half_width = 0.15;
    double grad_cap = 0.0;

    int stabilization_history_frames = 10;
    bool enable_spatial_height_smoothing = false;
    int height_smoothing_radius = 2;
    double height_smoothing_max_delta = 0.35;
    int height_smoothing_min_neighbors = 3;
    int edge_min_valid_neighbors = 5;

    int stabilization_min_overlap_cells = 80;
    double stabilization_inlier_threshold = 0.15;
    double stabilization_max_z = 0.30;
    double stabilization_max_tilt_deg = 1.5;
    double stabilization_alpha = 0.25;
    int corrected_sample_capacity = 32;
    double corrected_surface_top_fraction = 0.05;
    double accumulation_update_distance_m = 0.5;
    double accumulation_update_yaw_deg = 2.0;
    double accumulation_max_interval_sec = 0.5;
  };

  using OdometryState = lio_bev::PoseSample;

  struct Timing {
    double prepare_ms = 0.0;
    double stabilize_ms = 0.0;
    double accumulate_ms = 0.0;
    double rasterize_ms = 0.0;
    double gradient_ms = 0.0;
  };
  const Timing& timing() const { return timing_; }
  const grid_map::GridMap& map() const { return map_; }
  const std::string& outputParentFrame() const { return output_parent_frame_; }
  Eigen::Isometry3d outputPose() const
  {
    // Relative heights subtract a shared reference. Include that offset in TF
    // so that displayed surfaces still coincide with the world surface.
    Eigen::Isometry3d pose = output_pose_;
    pose.translation() += output_pose_.linear() * Eigen::Vector3d(0.0, 0.0, height_reference_);
    return pose;
  }

  void configure(const Config& config)
  {
    config_ = config;
    for (double value : {config_.resolution, config_.map_size_x, config_.map_size_y,
         config_.point_min_z, config_.point_max_z, config_.ego_box_min_x,
         config_.ego_box_max_x, config_.ego_box_min_y, config_.ego_box_max_y,
         config_.h_rel_deadzone_half_width, config_.grad_deadzone_half_width,
         config_.grad_cap, config_.height_smoothing_max_delta,
         config_.stabilization_inlier_threshold, config_.stabilization_max_z,
         config_.stabilization_max_tilt_deg, config_.stabilization_alpha,
         config_.corrected_surface_top_fraction, config_.accumulation_update_distance_m,
         config_.accumulation_update_yaw_deg, config_.accumulation_max_interval_sec}) {
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite BEV parameter");
    }
    if (config_.resolution <= 0.0 || config_.map_size_x <= 0.0 ||
        config_.map_size_y <= 0.0 ||
        std::hypot(config_.map_size_x, config_.map_size_y) / config_.resolution > 4096) {
      throw std::invalid_argument("invalid or excessive BEV grid geometry");
    }
    if (config_.point_min_z > config_.point_max_z) {
      std::swap(config_.point_min_z, config_.point_max_z);
    }
    config_.resolution = std::max(0.01, config_.resolution);
    config_.map_size_x = std::max(config_.resolution, config_.map_size_x);
    config_.map_size_y = std::max(config_.resolution, config_.map_size_y);
    config_.map_size_x = std::round(config_.map_size_x / config_.resolution) * config_.resolution;
    config_.map_size_y = std::round(config_.map_size_y / config_.resolution) * config_.resolution;
    if (config_.body_frame_id.empty()) throw std::invalid_argument("empty BEV body frame");
    if (config_.gravity_aligned_output &&
        (!config_.use_odometry_frame || config_.bev_frame_id.empty() ||
         config_.bev_frame_id == config_.body_frame_id)) {
      throw std::invalid_argument("gravity-aligned BEV requires odometry and a distinct bev_frame_id");
    }
    if (config_.ego_box_min_x > config_.ego_box_max_x) {
      std::swap(config_.ego_box_min_x, config_.ego_box_max_x);
    }
    if (config_.ego_box_min_y > config_.ego_box_max_y) {
      std::swap(config_.ego_box_min_y, config_.ego_box_max_y);
    }
    config_.h_rel_deadzone_half_width = std::max(0.0, config_.h_rel_deadzone_half_width);
    config_.grad_deadzone_half_width = std::max(0.0, config_.grad_deadzone_half_width);
    config_.grad_cap = std::max(0.0, config_.grad_cap);
    config_.stabilization_history_frames =
      std::max(2, std::min(100, config_.stabilization_history_frames));
    config_.height_smoothing_radius =
      std::max(0, std::min(5, config_.height_smoothing_radius));
    config_.height_smoothing_max_delta = std::max(0.0, config_.height_smoothing_max_delta);
    config_.height_smoothing_min_neighbors =
      std::max(1, config_.height_smoothing_min_neighbors);
    config_.edge_min_valid_neighbors = std::max(1, std::min(9, config_.edge_min_valid_neighbors));
    config_.stabilization_min_overlap_cells = std::max(12, config_.stabilization_min_overlap_cells);
    config_.stabilization_inlier_threshold = std::max(0.01, config_.stabilization_inlier_threshold);
    config_.stabilization_max_z = std::max(0.01, config_.stabilization_max_z);
    config_.stabilization_max_tilt_deg = std::max(0.05, std::min(5.0, config_.stabilization_max_tilt_deg));
    config_.stabilization_alpha = std::max(0.0, std::min(1.0, config_.stabilization_alpha));
    config_.corrected_sample_capacity = std::max(4, std::min(256, config_.corrected_sample_capacity));
    config_.corrected_surface_top_fraction = std::max(
      0.01, std::min(1.0, config_.corrected_surface_top_fraction));
    config_.accumulation_update_distance_m =
      std::max(0.0, config_.accumulation_update_distance_m);
    config_.accumulation_update_yaw_deg = std::max(0.0, config_.accumulation_update_yaw_deg);
    config_.accumulation_max_interval_sec = std::max(0.0, config_.accumulation_max_interval_sec);
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

    const std::string frame_id = config_.gravity_aligned_output
      ? config_.bev_frame_id : config_.body_frame_id;
    const grid_map::Position center(0.0, 0.0);
    const Eigen::Isometry3d capture_pose =
      use_odom ? odometry->pose : Eigen::Isometry3d::Identity();
    const bool capture_pose_valid = use_odom;
    output_pose_ = capture_pose;
    output_parent_frame_ = use_odom && !odometry->frame_id.empty()
      ? odometry->frame_id : config_.odometry_frame_id;
    if (config_.gravity_aligned_output) {
      if (!use_odom || output_parent_frame_ == config_.bev_frame_id) {
        throw std::invalid_argument("missing odometry or BEV frame equals its parent");
      }
      const double yaw = std::atan2(capture_pose.linear()(1, 0), capture_pose.linear()(0, 0));
      output_pose_.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    stamp_ = cloud_stamp;
    auto stage_start = SteadyClock::now();

    ensureMap(frame_id, center);
    if ((!last_input_stamp_.isZero() && !cloud_stamp.isZero() && cloud_stamp < last_input_stamp_) ||
        (use_odom && !last_odometry_frame_.empty() && last_odometry_frame_ != odometry->frame_id)) {
      rolling_grid_->reset();
      frame_history_.clear();
      have_last_accumulation_position_ = false;
      height_reference_ = 0.0f;
      have_height_reference_ = false;
      ROS_WARN("lio_bev: input time/frame changed; rolling accumulation reset");
    }
    if (!cloud_stamp.isZero()) last_input_stamp_ = cloud_stamp;
    if (use_odom) last_odometry_frame_ = odometry->frame_id;

    prepared_points_.clear();
    prepared_points_.reserve(body_points.size());
    const Eigen::Isometry3d body_from_world = capture_pose.inverse();

    for (const auto& point : body_points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }

      Eigen::Vector3d body_point(point.x, point.y, point.z);
      if (use_odom && config_.input_cloud_in_world) body_point = body_from_world * body_point;

      if (!body_point.allFinite() || body_point.z() < config_.point_min_z ||
          body_point.z() > config_.point_max_z ||
          shouldRejectEgoPoint(body_point.x(), body_point.y())) {
        continue;
      }

      int row = 0;
      int col = 0;
      if (!positionToCellIndex(body_point.x(), body_point.y(), row, col)) continue;
      prepared_points_.push_back({body_point.x(), body_point.y(), body_point.z()});
    }

    addFrameToHistory(capture_pose, capture_pose_valid, stamp_);
    timing_.prepare_ms = elapsedMs(stage_start);
    stage_start = SteadyClock::now();
    estimateLatestFrameStabilization(capture_pose, capture_pose_valid);
    timing_.stabilize_ms = elapsedMs(stage_start);
    stage_start = SteadyClock::now();
    const Eigen::Isometry3d current_pose =
      capture_pose_valid ? capture_pose : Eigen::Isometry3d::Identity();
    updateRollingAccumulator(current_pose, capture_pose_valid);
    // Retain only per-cell minima for old frames. Reuse raw-point storage.
    frame_history_.back().points.swap(prepared_points_);
    prepared_points_.clear();
    timing_.accumulate_ms = elapsedMs(stage_start);
    stage_start = SteadyClock::now();
    // Capture and accumulation above use the unmodified full pose. Only here
    // choose a yaw-aligned grid, with Z following the gravity-aligned world Z.
    fillMapFromRollingAccumulator(output_pose_);
    if (config_.enable_spatial_height_smoothing &&
        config_.height_smoothing_radius > 0) {
      smoothRelativeHeightLayer();
    }
    timing_.rasterize_ms = elapsedMs(stage_start);
    stage_start = SteadyClock::now();
    computeDirectionalGradients();
    timing_.gradient_ms = elapsedMs(stage_start);
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

private:
  struct StoredPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
  };

  struct FrameObservation {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    ros::Time stamp;
    bool pose_valid = false;
    std::vector<StoredPoint> points;
    std::vector<StoredPoint> stabilization_points;
    double correction_z = 0.0;
    double correction_x_slope = 0.0;
    double correction_y_slope = 0.0;
  };

  struct RollingCell {
    std::uint64_t sample_count = 0;
    std::vector<float> corrected_world_z_samples;
    float surface_world_z = kNaN;
    std::uint64_t last_update_generation = 0;
  };

  struct ResidualSample { double x, y, residual, weight; };
  Timing timing_;
  std::vector<StoredPoint> prepared_points_;
  std::vector<int> minimum_indices_;
  std::vector<float> current_min_, history_min_, absolute_residuals_, sample_scratch_;
  std::vector<float> spatial_height_;
  std::vector<ResidualSample> stabilization_samples_;
  std::vector<RollingCell*> touched_cells_;
  double last_accumulation_yaw_ = 0.0;
  ros::Time last_accumulation_stamp_;
  Eigen::Isometry3d output_pose_ = Eigen::Isometry3d::Identity();
  std::string output_parent_frame_;
  float height_reference_ = 0.0f;
  bool have_height_reference_ = false;
  Config config_;
  grid_map::GridMap map_;
  bool map_initialized_ = false;
  int map_rows_ = 0;
  int map_cols_ = 0;
  double map_half_x_ = 0.0;
  double map_half_y_ = 0.0;
  double map_inv_resolution_ = 0.0;
  ros::Time stamp_;
  ros::Time last_input_stamp_;
  std::string last_odometry_frame_;
  std::deque<FrameObservation> frame_history_;
  std::unique_ptr<lio_bev::RollingGrid<RollingCell>> rolling_grid_;
  std::vector<std::size_t> active_cell_indices_;
  std::vector<float> smoothed_height_;
  std::vector<uint8_t> smoothed_valid_;
  Eigen::Vector2d last_accumulation_position_ = Eigen::Vector2d::Zero();
  bool have_last_accumulation_position_ = false;
  std::uint64_t accumulation_generation_ = 0;

  void updateMapIndexLookup()
  {
    map_rows_ = map_.getSize()(0);
    map_cols_ = map_.getSize()(1);
    map_half_x_ = map_.getLength().x() * 0.5;
    map_half_y_ = map_.getLength().y() * 0.5;
    map_inv_resolution_ = 1.0 / config_.resolution;

    const std::size_t cell_count =
      static_cast<std::size_t>(map_rows_) * static_cast<std::size_t>(map_cols_);
    active_cell_indices_.clear();
    if (smoothed_height_.size() != cell_count) {
      smoothed_height_.assign(cell_count, kNaN);
      smoothed_valid_.assign(cell_count, 0);
    }
  }

  bool positionToCellIndex(double x, double y, int& row, int& col) const
  {
    if (!std::isfinite(x) || !std::isfinite(y) ||
        x > map_half_x_ || x <= -map_half_x_ ||
        y > map_half_y_ || y <= -map_half_y_) return false;
    row = static_cast<int>(std::floor((map_half_x_ - x) * map_inv_resolution_));
    col = static_cast<int>(std::floor((map_half_y_ - y) * map_inv_resolution_));
    return row >= 0 && row < map_rows_ && col >= 0 && col < map_cols_;
  }

  void initializeMap(const std::string& frame_id, const grid_map::Position& center)
  {
    const std::vector<std::string> layers = {
      "H_stable_surf",
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
    const double backing_size = std::hypot(config_.map_size_x, config_.map_size_y) +
      4.0 * config_.resolution;
    const int backing_cells = std::max(
      3, static_cast<int>(std::ceil(backing_size / config_.resolution)));
    rolling_grid_.reset(new lio_bev::RollingGrid<RollingCell>(
      config_.resolution, backing_cells, backing_cells));
    frame_history_.clear();
    height_reference_ = 0.0f;
    have_height_reference_ = false;
    resetMapLayers();
  }

  void resetMapLayers()
  {
    map_["H_stable_surf"].setConstant(kNaN);
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
                         bool pose_valid, const ros::Time& frame_stamp)
  {
    if (!pose_valid) frame_history_.clear();
    while (frame_history_.size() >= static_cast<std::size_t>(config_.stabilization_history_frames)) {
      frame_history_.pop_front();
    }
    FrameObservation frame;
    frame.pose = capture_pose;
    frame.stamp = frame_stamp;
    frame.pose_valid = pose_valid;
    frame.points.swap(prepared_points_);
    const std::size_t cell_count = std::size_t(map_rows_) * map_cols_;
    minimum_indices_.assign(cell_count, -1);
    for (std::size_t i = 0; i < frame.points.size(); ++i) {
      const auto& p = frame.points[i];
      int row, col;
      if (!positionToCellIndex(p.x, p.y, row, col)) continue;
      int& index = minimum_indices_[gridIndex(row, col, map_cols_)];
      if (index < 0 || p.z < frame.points[static_cast<std::size_t>(index)].z) {
        index = static_cast<int>(i);
      }
    }
    frame.stabilization_points.reserve(std::min(cell_count, frame.points.size()));
    for (int index : minimum_indices_) {
      if (index >= 0) frame.stabilization_points.push_back(frame.points[index]);
    }
    frame_history_.push_back(std::move(frame));
  }

  void updateRollingAccumulator(const Eigen::Isometry3d& current_pose,
                                bool current_pose_valid)
  {
    if (!rolling_grid_ || frame_history_.empty()) {
      return;
    }
    if (!current_pose_valid) {
      rolling_grid_->reset();
      have_last_accumulation_position_ = false;
    }
    rolling_grid_->recenter(current_pose.translation().x(),
                            current_pose.translation().y());

    const Eigen::Vector2d position = current_pose.translation().head<2>();
    const double yaw = std::atan2(current_pose.linear()(1, 0), current_pose.linear()(0, 0));
    const double delta_yaw = std::abs(std::atan2(std::sin(yaw - last_accumulation_yaw_),
                                                std::cos(yaw - last_accumulation_yaw_)));
    const double age = (stamp_ - last_accumulation_stamp_).toSec();
    if (have_last_accumulation_position_ &&
        (position - last_accumulation_position_).norm() < config_.accumulation_update_distance_m &&
        delta_yaw < config_.accumulation_update_yaw_deg * M_PI / 180.0 &&
        age < config_.accumulation_max_interval_sec) return;
    last_accumulation_yaw_ = yaw;
    last_accumulation_stamp_ = stamp_;
    last_accumulation_position_ = position;
    have_last_accumulation_position_ = true;
    ++accumulation_generation_;

    FrameObservation& frame = frame_history_.back();
    auto& touched_cells = touched_cells_;
    touched_cells.clear();
    touched_cells.reserve(frame.points.size() / 4 + 1);

    for (const StoredPoint& point : frame.points) {
      const Eigen::Vector3d corrected_world = frame.pose * correctedFramePoint(frame, point);
      RollingCell* cell = rolling_grid_->cellByWorld(
        corrected_world.x(), corrected_world.y());
      if (cell == nullptr) {
        continue;
      }
      if (cell->last_update_generation != accumulation_generation_) {
        cell->last_update_generation = accumulation_generation_;
        touched_cells.push_back(cell);
      }

      if (cell->sample_count < std::numeric_limits<std::uint64_t>::max()) {
        ++cell->sample_count;
      }
      const std::size_t capacity =
        static_cast<std::size_t>(config_.corrected_sample_capacity);
      if (cell->corrected_world_z_samples.size() < capacity) {
        cell->corrected_world_z_samples.push_back(
          static_cast<float>(corrected_world.z()));
      } else {
        std::uint64_t hash = cell->sample_count + 0x9e3779b97f4a7c15ULL;
        hash = (hash ^ (hash >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        hash = (hash ^ (hash >> 27U)) * 0x94d049bb133111ebULL;
        hash ^= hash >> 31U;
        const std::uint64_t slot = hash % cell->sample_count;
        if (slot < capacity) {
          cell->corrected_world_z_samples[static_cast<std::size_t>(slot)] =
            static_cast<float>(corrected_world.z());
        }
      }

    }

    for (RollingCell* cell : touched_cells) {
      auto& sorted = sample_scratch_;
      sorted.assign(cell->corrected_world_z_samples.begin(), cell->corrected_world_z_samples.end());
      if (sorted.empty()) {
        continue;
      }
      const std::size_t top_count = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(
          config_.corrected_surface_top_fraction * sorted.size())));
      if (top_count < sorted.size()) {
        std::nth_element(sorted.begin(), sorted.begin() + top_count, sorted.end(), std::greater<float>());
      }
      double sum = 0.0;
      for (std::size_t i = 0; i < top_count; ++i) {
        sum += sorted[i];
      }
      cell->surface_world_z = static_cast<float>(sum / top_count);
    }
  }

  void fillMapFromRollingAccumulator(const Eigen::Isometry3d& current_pose)
  {
    if (!rolling_grid_) return;
    auto& height = map_["H_stable_surf"];
    auto& mask = map_["M_L"];
    active_cell_indices_.clear();
    const Eigen::Isometry3d body_from_world = current_pose.inverse();
    // Rasterize the finite world-cell footprint, not just its center. This
    // avoids rotation holes and accounts for roll/pitch at the actual surface Z.
    const Eigen::Matrix2d footprint = body_from_world.linear().topLeftCorner<2, 2>();
    if (std::abs(footprint.determinant()) < 1e-6) {
      throw std::runtime_error("BEV projection is singular: vehicle is nearly vertical");
    }
    const Eigen::Matrix2d inverse_footprint = footprint.inverse();
    const double half = config_.resolution * 0.5;
    const double extent_x = half * footprint.row(0).cwiseAbs().sum();
    const double extent_y = half * footprint.row(1).cwiseAbs().sum();
    const auto& rotation = current_pose.linear();
    for (int r = 0; r < rolling_grid_->rows(); ++r) {
      for (int c = 0; c < rolling_grid_->cols(); ++c) {
        const RollingCell* cell = rolling_grid_->cell(r, c);
        if (!std::isfinite(cell->surface_world_z)) continue;
        double wx, wy;
        if (!rolling_grid_->localToWorldCenter(r, c, wx, wy)) continue;
        const Eigen::Vector3d p = body_from_world * Eigen::Vector3d(wx, wy, cell->surface_world_z);
        if (!p.allFinite() || p.x() + extent_x <= -map_half_x_ ||
            p.x() - extent_x >= map_half_x_ || p.y() + extent_y <= -map_half_y_ ||
            p.y() - extent_y >= map_half_y_) continue;
        const int r0 = std::max(0, static_cast<int>(std::ceil(
          (map_half_x_ - std::min(map_half_x_, p.x() + extent_x)) * map_inv_resolution_ - 0.5)));
        const int r1 = std::min(map_rows_ - 1, static_cast<int>(std::floor(
          (map_half_x_ - std::max(-map_half_x_, p.x() - extent_x)) * map_inv_resolution_ - 0.5)));
        const int c0 = std::max(0, static_cast<int>(std::ceil(
          (map_half_y_ - std::min(map_half_y_, p.y() + extent_y)) * map_inv_resolution_ - 0.5)));
        const int c1 = std::min(map_cols_ - 1, static_cast<int>(std::floor(
          (map_half_y_ - std::max(-map_half_y_, p.y() - extent_y)) * map_inv_resolution_ - 0.5)));
        for (int row = r0; row <= r1; ++row) {
          const double bx = map_half_x_ - (row + 0.5) * config_.resolution;
          for (int col = c0; col <= c1; ++col) {
            const double by = map_half_y_ - (col + 0.5) * config_.resolution;
            const Eigen::Vector2d offset = inverse_footprint * Eigen::Vector2d(bx - p.x(), by - p.y());
            if (offset.cwiseAbs().maxCoeff() > half + 1e-9) continue;
            const float z = static_cast<float>((cell->surface_world_z - current_pose.translation().z() -
              rotation(2, 0) * bx - rotation(2, 1) * by) / rotation(2, 2));
            if (mask(row, col) == 0.0f) {
              mask(row, col) = 1.0f;
              height(row, col) = z;
              active_cell_indices_.push_back(gridIndex(row, col, map_cols_));
            } else {
              height(row, col) = std::max(height(row, col), z);
            }
          }
        }
      }
    }
    // An empty grid supplies no reference measurement; keep the last valid one.
    if (active_cell_indices_.empty()) return;
    double nearest = std::numeric_limits<double>::infinity();
    float reference = 0.0f;
    for (std::size_t index : active_cell_indices_) {
      const int row = static_cast<int>(index / map_cols_);
      const int col = static_cast<int>(index % map_cols_);
      const double x = map_half_x_ - (row + 0.5) * config_.resolution;
      const double y = map_half_y_ - (col + 0.5) * config_.resolution;
      if (x*x + y*y < nearest) {
        nearest = x*x + y*y;
        reference = height(row, col);
      }
    }
    // Reject jumps over 1 m; compare against the last reference actually used.
    if (have_height_reference_ && std::abs(reference - height_reference_) > 1.0f) {
      reference = height_reference_;
    }
    height_reference_ = reference;
    have_height_reference_ = true;
    for (std::size_t index : active_cell_indices_) {
      const int row = static_cast<int>(index / map_cols_);
      const int col = static_cast<int>(index % map_cols_);
      height(row, col) = applyScalarDeadzone(height(row, col) - reference,
                                            config_.h_rel_deadzone_half_width);
    }
  }

  Eigen::Vector3d correctedFramePoint(const FrameObservation& frame,
                                      const StoredPoint& point) const
  {
    const double corrected_z = point.z - frame.correction_z -
      frame.correction_x_slope * point.x - frame.correction_y_slope * point.y;
    return Eigen::Vector3d(point.x, point.y, corrected_z);
  }

  void estimateLatestFrameStabilization(const Eigen::Isometry3d& current_pose,
                                        bool current_pose_valid)
  {
    if (!current_pose_valid || frame_history_.size() < 2) {
      return;
    }

    FrameObservation& current_frame = frame_history_.back();
    const std::size_t cell_count =
      static_cast<std::size_t>(map_rows_) * static_cast<std::size_t>(map_cols_);
    auto& current_min = current_min_;
    auto& history_min = history_min_;
    current_min.assign(cell_count, kNaN);
    history_min.assign(cell_count, kNaN);

    for (const auto& point : current_frame.stabilization_points) {
      const Eigen::Vector3d corrected_point = correctedFramePoint(current_frame, point);
      int row = 0;
      int col = 0;
      if (!positionToCellIndex(corrected_point.x(), corrected_point.y(), row, col)) {
        continue;
      }
      const std::size_t index = gridIndex(row, col, map_cols_);
      if (!std::isfinite(current_min[index]) || corrected_point.z() < current_min[index]) {
        current_min[index] = static_cast<float>(corrected_point.z());
      }
    }

    const Eigen::Matrix3d current_rotation_inverse = current_pose.linear().transpose();
    const Eigen::Vector3d current_translation_inverse =
      -current_rotation_inverse * current_pose.translation();
    for (auto frame_it = frame_history_.begin();
         frame_it != std::prev(frame_history_.end()); ++frame_it) {
      const FrameObservation& frame = *frame_it;
      if (!frame.pose_valid) {
        continue;
      }
      const Eigen::Matrix3d frame_to_current_rotation =
        current_rotation_inverse * frame.pose.linear();
      const Eigen::Vector3d frame_to_current_translation =
        current_rotation_inverse * frame.pose.translation() + current_translation_inverse;
      for (const auto& point : frame.stabilization_points) {
        const Eigen::Vector3d local_point = frame_to_current_rotation *
          correctedFramePoint(frame, point) + frame_to_current_translation;
        int row = 0;
        int col = 0;
        if (!positionToCellIndex(local_point.x(), local_point.y(), row, col)) {
          continue;
        }
        const std::size_t index = gridIndex(row, col, map_cols_);
        if (!std::isfinite(history_min[index]) || local_point.z() < history_min[index]) {
          history_min[index] = static_cast<float>(local_point.z());
        }
      }
    }

    auto& samples = stabilization_samples_;
    samples.clear();
    samples.reserve(cell_count / 4);
    for (std::size_t index = 0; index < cell_count; ++index) {
      if (!std::isfinite(current_min[index]) || !std::isfinite(history_min[index])) {
        continue;
      }
      const int row = static_cast<int>(index / static_cast<std::size_t>(map_cols_));
      const int col = static_cast<int>(index % static_cast<std::size_t>(map_cols_));
      grid_map::Position position;
      if (!map_.getPosition(grid_map::Index(row, col), position)) {
        continue;
      }
      const double residual = static_cast<double>(current_min[index] - history_min[index]);
      if (std::abs(residual) > 1.0) {
        continue;
      }
      samples.push_back({position.x(), position.y(), residual, 1.0});
    }
    if (static_cast<int>(samples.size()) < config_.stabilization_min_overlap_cells) {
      return;
    }

    Eigen::Vector3d correction = Eigen::Vector3d::Zero();
    for (int iteration = 0; iteration < 5; ++iteration) {
      Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
      Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
      for (const auto& sample : samples) {
        const Eigen::Vector3d basis(sample.x, sample.y, 1.0);
        normal.noalias() += sample.weight * basis * basis.transpose();
        rhs.noalias() += sample.weight * basis * sample.residual;
      }
      Eigen::LDLT<Eigen::Matrix3d> solver(normal);
      if (solver.info() != Eigen::Success ||
          solver.vectorD().cwiseAbs().minCoeff() < 1e-9) {
        return;
      }
      correction = solver.solve(rhs);
      if (!correction.allFinite()) {
        return;
      }

      if (iteration == 4) break;
      auto& absolute_residuals = absolute_residuals_;
      absolute_residuals.clear();
      absolute_residuals.reserve(samples.size());
      for (const auto& sample : samples) {
        const double predicted = correction.x() * sample.x +
          correction.y() * sample.y + correction.z();
        absolute_residuals.push_back(
          static_cast<float>(std::abs(sample.residual - predicted)));
      }
      const double robust_scale = std::max(
        0.02, 1.4826 * static_cast<double>(medianOfSmallArray(
          absolute_residuals.data(), static_cast<int>(absolute_residuals.size()))));
      const double huber_threshold = 1.5 * robust_scale;
      for (auto& sample : samples) {
        const double predicted = correction.x() * sample.x +
          correction.y() * sample.y + correction.z();
        const double error = std::abs(sample.residual - predicted);
        sample.weight = error <= huber_threshold
          ? 1.0 : huber_threshold / std::max(error, 1e-9);
      }
    }

    int inlier_count = 0;
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& sample : samples) {
      const double predicted = correction.x() * sample.x +
        correction.y() * sample.y + correction.z();
      if (std::abs(sample.residual - predicted) >
          config_.stabilization_inlier_threshold) {
        continue;
      }
      ++inlier_count;
      min_x = std::min(min_x, sample.x);
      max_x = std::max(max_x, sample.x);
      min_y = std::min(min_y, sample.y);
      max_y = std::max(max_y, sample.y);
    }
    if (inlier_count < config_.stabilization_min_overlap_cells ||
        max_x - min_x < 4.0 || max_y - min_y < 4.0) {
      return;
    }

    const double max_slope = std::tan(config_.stabilization_max_tilt_deg * M_PI / 180.0);
    correction.x() = std::max(-max_slope, std::min(max_slope, correction.x()));
    correction.y() = std::max(-max_slope, std::min(max_slope, correction.y()));
    correction.z() = std::max(-config_.stabilization_max_z,
                              std::min(config_.stabilization_max_z, correction.z()));
    current_frame.correction_x_slope = config_.stabilization_alpha * correction.x();
    current_frame.correction_y_slope = config_.stabilization_alpha * correction.y();
    current_frame.correction_z = config_.stabilization_alpha * correction.z();

  }

  void smoothRelativeHeightLayer()
  {
    if (config_.height_smoothing_radius <= 0 || !map_.exists("H_stable_surf") ||
        !map_.exists("M_L")) {
      return;
    }

    auto& h_rel_layer = map_["H_stable_surf"];
    const auto& mask_layer = map_["M_L"];
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);
    auto& smoothed = spatial_height_;
    smoothed.assign(cell_count, kNaN);

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
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const std::size_t index = gridIndex(row, col, cols);
        if (std::isfinite(smoothed[index])) {
          h_rel_layer(row, col) = smoothed[index];
        }
      }
    }
  }

  bool shouldRejectEgoPoint(double x, double y) const
  {
    if (!config_.ego_filter_enabled) {
      return false;
    }

    return x >= config_.ego_box_min_x && x <= config_.ego_box_max_x &&
           y >= config_.ego_box_min_y && y <= config_.ego_box_max_y;
  }

  void computeDirectionalGradients()
  {
    auto& h_rel_layer = map_["H_stable_surf"];
    auto& mask_layer = map_["M_L"];
    auto& longitudinal_gradient_layer = map_["G_long_L"];
    auto& lateral_gradient_layer = map_["G_lat_L"];

    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);

    auto& smoothed_height = smoothed_height_;
    auto& smoothed_valid = smoothed_valid_;
    smoothed_height.assign(cell_count, kNaN);
    smoothed_valid.assign(cell_count, 0);

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

};

}  // namespace lio_bev
