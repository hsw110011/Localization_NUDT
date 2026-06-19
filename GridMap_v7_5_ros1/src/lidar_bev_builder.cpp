#include "map/lidar_bev_builder.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kSurfaceTopHeightFraction = 0.15;

struct BevFrameSample {
    bool valid = false;
    std::size_t linear_index = 0;
    float z = 0.0f;
};  

bool isFinite(double value)
{
    return std::isfinite(value);
}

std::size_t gridIndex(int row, int col, int cols)
{
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
           static_cast<std::size_t>(col);
}
}  // namespace

void LidarBevBuilder::configure(const Config& config)
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
    config_.ground_fallback_quantile =
        std::max(0.0, std::min(1.0, config_.ground_fallback_quantile));
    config_.near_inner_radius = std::max(0.0, config_.near_inner_radius);
    config_.near_outer_radius = std::max(config_.near_inner_radius, config_.near_outer_radius);
    if (config_.ground_candidate_min_z > config_.ground_candidate_max_z) {
        std::swap(config_.ground_candidate_min_z, config_.ground_candidate_max_z);
    }
    config_.ground_min_points = std::max(1, config_.ground_min_points);
    config_.ground_front_half_angle_deg =
        std::max(0.1, std::min(89.9, config_.ground_front_half_angle_deg));
    config_.cell_max_points = std::max(0, config_.cell_max_points);
    config_.debug_window_stride = std::max(1, config_.debug_window_stride);
    config_.edge_min_valid_neighbors =
        std::max(1, std::min(9, config_.edge_min_valid_neighbors));
}

void LidarBevBuilder::initializeMap(const std::string& frame_id, const grid_map::Position& center)
{
    const std::vector<std::string> layers = {
        "H_rel_surf",
        "M_L",
        "G_long_L",
        "G_lat_L"
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

void LidarBevBuilder::ensureMap(const std::string& frame_id, const grid_map::Position& center)
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

void LidarBevBuilder::resetMapLayers()
{
    map_["H_rel_surf"].setConstant(kNaN);
    map_["G_long_L"].setConstant(kNaN);
    map_["G_lat_L"].setConstant(kNaN);
    map_["M_L"].setConstant(0.0f);
}

void LidarBevBuilder::updateMapIndexLookup()
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

bool LidarBevBuilder::positionToCellIndex(double x, double y, int& row, int& col) const
{
    row = static_cast<int>(std::floor((map_half_x_ - x) * map_inv_resolution_));
    col = static_cast<int>(std::floor((map_half_y_ - y) * map_inv_resolution_));
    return row >= 0 && row < map_rows_ && col >= 0 && col < map_cols_;
}

float LidarBevBuilder::medianOfSmallArray(float* values, int count)
{
    if (count <= 0) {
        return 0.0f;
    }
    const int median_index = count / 2;
    std::nth_element(values, values + median_index, values + count);
    return values[median_index];
}

void LidarBevBuilder::build(const std::vector<PointXYZRGBValid>& car_points,
                            const OdometryState* odometry)
{
    const bool use_odom =
        config_.use_odometry_frame && odometry != nullptr && odometry->valid;

    const std::string frame_id = config_.body_frame_id;
    const grid_map::Position center(0.0, 0.0);
    const Eigen::Isometry3d capture_pose =
        use_odom ? odometry->pose : Eigen::Isometry3d::Identity();
    const bool capture_pose_valid = use_odom;
    const Eigen::Isometry3d current_pose =
        capture_pose_valid ? capture_pose : Eigen::Isometry3d::Identity();
    stamp_ = capture_pose_valid ? odometry->stamp : ros::Time::now();
    display_pose_ = Eigen::Isometry3d::Identity();
    display_pose_valid_ = true;

    ensureMap(frame_id, center);

    prepared_points_buffer_.clear();
    prepared_points_buffer_.reserve(car_points.size());

    struct PreparedPointSlot {
        bool valid = false;
        PreparedPoint point;
    };
    std::vector<PreparedPointSlot> prepared_slots(car_points.size());

    Eigen::Matrix3d odom_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d odom_translation = Eigen::Vector3d::Zero();
    if (use_odom) {
        odom_rotation = odometry->pose.linear();
        odom_translation = odometry->pose.translation();
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < static_cast<int>(car_points.size()); ++i) {
        const auto& point = car_points[static_cast<std::size_t>(i)];
        PreparedPointSlot& slot = prepared_slots[static_cast<std::size_t>(i)];
        slot.valid = false;

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
        prepared.car_x = point.x;
        prepared.car_y = point.y;
        prepared.car_z = point.z;

        if (use_odom) {
            const Eigen::Vector3d point_odom =
                odom_rotation * Eigen::Vector3d(point.x, point.y, point.z) + odom_translation;
            prepared.x = point_odom.x();
            prepared.y = point_odom.y();
            prepared.z = point_odom.z();
        } else {
            prepared.x = point.x;
            prepared.y = point.y;
            prepared.z = point.z;
        }

        slot.valid = true;
        slot.point = prepared;
    }

    for (const PreparedPointSlot& slot : prepared_slots) {
        if (slot.valid) {
            prepared_points_buffer_.push_back(slot.point);
        }
    }
    std::vector<PreparedPoint>& prepared_points = prepared_points_buffer_;

    addFrameToHistory(capture_pose, capture_pose_valid, stamp_, prepared_points);
    pruneFrameHistoryCellPointLimit(current_pose, capture_pose_valid);
    const double ground_reference =
        estimateGroundReferenceFromHistory(current_pose, capture_pose_valid);
    last_ground_reference_ = ground_reference;
    fillMapFromFrameHistory(current_pose, capture_pose_valid, ground_reference);
    computeDirectionalGradients();

    if (config_.show_windows) {
        showDebugWindows();
    }
}

void LidarBevBuilder::addFrameToHistory(
    const Eigen::Isometry3d& capture_pose,
    bool pose_valid,
    const ros::Time& stamp,
    const std::vector<PreparedPoint>& points)
{
    if (!pose_valid) {
        frame_history_.clear();
    } else if (!frame_history_.empty() && !frame_history_.back().pose_valid) {
        frame_history_.clear();
    }

    FrameObservation frame;
    frame.pose = capture_pose;
    frame.stamp = stamp;
    frame.pose_valid = pose_valid;

    frame.points.reserve(points.size());
    for (const auto& point : points) {
        frame.points.push_back(StoredPoint{point.car_x, point.car_y, point.car_z});
    }

    frame_history_.push_back(std::move(frame));
}

void LidarBevBuilder::pruneFrameHistoryCellPointLimit(
    const Eigen::Isometry3d& current_pose,
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

    for (auto frame_it = frame_history_.rbegin(); frame_it != frame_history_.rend(); ++frame_it) {
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

        for (auto point_it = frame.points.rbegin(); point_it != frame.points.rend(); ++point_it) {
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

void LidarBevBuilder::fillMapFromFrameHistory(
    const Eigen::Isometry3d& current_pose,
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
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
        for (int point_index = 0; point_index < static_cast<int>(frame.points.size()); ++point_index) {
            const auto& point = frame.points[static_cast<std::size_t>(point_index)];
            BevFrameSample& sample = frame_samples[static_cast<std::size_t>(point_index)];
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

        const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
        const float h_q =
            static_cast<float>(
                meanTopHeightFraction(cell.top_heights,
                                      kSurfaceTopHeightFraction,
                                      cell.count));
        const float h_rel =
            static_cast<float>(static_cast<double>(h_q) - ground_reference);
        if (!std::isfinite(h_rel)) {
            continue;
        }

        h_rel_layer(row, col) = h_rel;
        mask_layer(row, col) = 1.0f;
    }
}

void LidarBevBuilder::addTopHeight(CellAccumulator& cell,
                                   float height,
                                   int max_top_count) const
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

bool LidarBevBuilder::shouldRejectEgoPoint(const PointXYZRGBValid& point) const
{
    if (!config_.ego_filter_enabled) {
        return false;
    }

    return point.x >= config_.ego_box_min_x && point.x <= config_.ego_box_max_x &&
           point.y >= config_.ego_box_min_y && point.y <= config_.ego_box_max_y;
}

void LidarBevBuilder::computeDirectionalGradients()
{
    auto& h_rel_layer = map_["H_rel_surf"];
    auto& mask_layer = map_["M_L"];
    auto& longitudinal_gradient_layer = map_["G_long_L"];
    auto& lateral_gradient_layer = map_["G_lat_L"];

    longitudinal_gradient_layer.setConstant(kNaN);
    lateral_gradient_layer.setConstant(kNaN);

    const int rows = map_rows_;
    const int cols = map_cols_;

    std::fill(smoothed_height_.begin(), smoothed_height_.end(), kNaN);
    std::memset(smoothed_valid_.data(), 0, smoothed_valid_.size());

    // Smooth H_rel_surf with a 3x3 median before Sobel gradients.
    const int active_count = static_cast<int>(active_cell_indices_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int active_i = 0; active_i < active_count; ++active_i) {
        const std::size_t index = active_cell_indices_[static_cast<std::size_t>(active_i)];
        const int row = static_cast<int>(index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(index % static_cast<std::size_t>(cols));
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

        smoothed_height_[index] = medianOfSmallArray(values, value_count);
        smoothed_valid_[index] = 1;
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
        longitudinal_unit =
            Eigen::Vector2d(display_pose_.linear()(0, 0),
                            display_pose_.linear()(1, 0));
        lateral_unit =
            Eigen::Vector2d(display_pose_.linear()(0, 1),
                            display_pose_.linear()(1, 1));
        if (longitudinal_unit.norm() > 1e-6) {
            longitudinal_unit.normalize();
        }
        if (lateral_unit.norm() > 1e-6) {
            lateral_unit.normalize();
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int active_i = 0; active_i < active_count; ++active_i) {
        const std::size_t index = active_cell_indices_[static_cast<std::size_t>(active_i)];
        if (!smoothed_valid_[index]) {
            continue;
        }

        const int row = static_cast<int>(index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(index % static_cast<std::size_t>(cols));
        const float center_height = smoothed_height_[index];
        int valid_neighbor_count = 0;
        double grad_index1 = 0.0;
        double grad_index0 = 0.0;

        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                const int neighbor_row = row + dr;
                const int neighbor_col = col + dc;
                float value = center_height;

                if (neighbor_row >= 0 && neighbor_row < rows &&
                    neighbor_col >= 0 && neighbor_col < cols) {
                    const std::size_t neighbor_index =
                        gridIndex(neighbor_row, neighbor_col, cols);
                    if (smoothed_valid_[neighbor_index]) {
                        value = smoothed_height_[neighbor_index];
                        ++valid_neighbor_count;
                    }
                }

                grad_index1 += static_cast<double>(sobel_x[dr + 1][dc + 1]) *
                               static_cast<double>(value);
                grad_index0 += static_cast<double>(sobel_y[dr + 1][dc + 1]) *
                               static_cast<double>(value);
            }
        }

        if (valid_neighbor_count < config_.edge_min_valid_neighbors) {
            continue;
        }

        const double index0_gradient =
            grad_index0 / (8.0 * config_.resolution);
        const double index1_gradient =
            grad_index1 / (8.0 * config_.resolution);
        const Eigen::Vector2d map_gradient =
            index0_gradient * index0_unit + index1_gradient * index1_unit;

        const float longitudinal_gradient =
            static_cast<float>(map_gradient.dot(longitudinal_unit));
        const float lateral_gradient =
            static_cast<float>(map_gradient.dot(lateral_unit));

        longitudinal_gradient_layer(row, col) = longitudinal_gradient;
        lateral_gradient_layer(row, col) = lateral_gradient;
    }
}

void LidarBevBuilder::showDebugWindows() const
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

    static const std::string layers_window_name = "BEV H_rel | BEV G_long | BEV G_lat";
    static const std::string mask_window_name = "BEV M_L";
    static bool windows_created = false;
    if (!windows_created) {
        int flags = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO;
        cv::namedWindow(layers_window_name, flags);
        cv::namedWindow(mask_window_name, flags);
        cv::resizeWindow(layers_window_name, 1500, 500);
        cv::resizeWindow(mask_window_name, 500, 500);
        windows_created = true;
    }

    cv::imshow(layers_window_name, renderStackedDebugLayers());
    cv::imshow(mask_window_name, renderBinaryLayer("M_L", false));
    cv::waitKey(1);
}

cv::Mat LidarBevBuilder::renderStackedDebugLayers() const
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

    if (!height_image.empty()) {
        drawGroundReferenceRegion(height_image);
    }

    cv::Mat combined;
    if (height_image.empty() || longitudinal_image.empty() || lateral_image.empty() ||
        height_image.rows != longitudinal_image.rows ||
        height_image.rows != lateral_image.rows) {
        return height_image;
    }
    cv::hconcat(height_image, longitudinal_image, combined);
    cv::hconcat(combined, lateral_image, combined);
    return combined;
}

cv::Mat LidarBevBuilder::renderColorizedLayer(const std::string& layer_name,
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

cv::Mat LidarBevBuilder::renderHeightLayer(const std::string& layer_name) const
{
    const std::pair<double, double> range = getRobustLayerRange(layer_name, 0.0, 1.0);
    return renderColorizedLayer(layer_name, range.first, range.second);
}

cv::Mat LidarBevBuilder::renderBinaryLayer(const std::string& layer_name,
                                            bool draw_center_mark) const
{
    const int rows = map_rows_;
    const int cols = map_cols_;
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name)) {
        return image;
    }

    const auto& layer = map_[layer_name];
    for (const std::size_t linear_index : active_cell_indices_) {
        const int row = static_cast<int>(linear_index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(linear_index % static_cast<std::size_t>(cols));
        if (layer(row, col) > 0.5f) {
            image.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
        }
    }

    if (draw_center_mark) {
        drawCenterMark(image);
    }
    drawGroundReferenceRegion(image);
    return image;
}

bool LidarBevBuilder::isDebugPixelValid(const grid_map::Index& index) const
{
    if (!map_.exists("M_L") || !map_.exists("H_rel_surf")) {
        return false;
    }

    const auto& mask = map_["M_L"];
    const auto& height = map_["H_rel_surf"];
    return mask(index(0), index(1)) > 0.5f &&
           std::isfinite(height(index(0), index(1)));
}

cv::Vec3b LidarBevBuilder::viridisColor(double normalized_value) const
{
    static const std::array<cv::Vec3b, 256> viridis_lut = []() {
        cv::Mat gray(1, 256, CV_8UC1);
        for (int i = 0; i < gray.cols; ++i) {
            gray.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
        }

        cv::Mat color;
        cv::applyColorMap(gray, color, cv::COLORMAP_VIRIDIS);

        std::array<cv::Vec3b, 256> lut;
        for (int i = 0; i < color.cols; ++i) {
            lut[static_cast<std::size_t>(i)] = color.at<cv::Vec3b>(0, i);
        }
        return lut;
    }();

    const double t = std::max(0.0, std::min(1.0, normalized_value));
    const int lut_index = static_cast<int>(std::round(t * 255.0));
    return viridis_lut[static_cast<std::size_t>(std::max(0, std::min(255, lut_index)))];
}

std::pair<double, double> LidarBevBuilder::getRobustLayerRange(
    const std::string& layer_name,
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

bool LidarBevBuilder::getVehicleFrameIndex(int image_row,
                                           int image_col,
                                           grid_map::Index& index) const
{
    if (!display_pose_valid_) {
        return false;
    }

    const int image_rows = map_.getSize()(0);
    const int image_cols = map_.getSize()(1);
    if (image_row < 0 || image_row >= image_rows ||
        image_col < 0 || image_col >= image_cols) {
        return false;
    }

    const bool identity_display =
        display_pose_.translation().head<2>().norm() < 1e-9 &&
        (display_pose_.linear() - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 1e-9;
    if (identity_display) {
        index = grid_map::Index(image_row, image_col);
        return true;
    }

    const double x_vehicle =
        0.5 * static_cast<double>(image_rows) * config_.resolution -
        (static_cast<double>(image_row) + 0.5) * config_.resolution;
    const double y_vehicle =
        0.5 * static_cast<double>(image_cols) * config_.resolution -
        (static_cast<double>(image_col) + 0.5) * config_.resolution;

    const Eigen::Vector3d map_point =
        display_pose_ * Eigen::Vector3d(x_vehicle, y_vehicle, 0.0);
    return map_.getIndex(grid_map::Position(map_point.x(), map_point.y()), index);
}

void LidarBevBuilder::vehicleBodyToImagePixel(double x, double y, double& row, double& col) const
{
    row = (map_half_x_ - x) * map_inv_resolution_ - 0.5;
    col = (map_half_y_ - y) * map_inv_resolution_ - 0.5;
}

void LidarBevBuilder::drawGroundReferenceRegion(cv::Mat& image) const
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

    const int inner_px =
        std::max(1, static_cast<int>(std::round(config_.near_inner_radius * map_inv_resolution_)));
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
        sector_annulus.emplace_back(static_cast<int>(std::round(col)), static_cast<int>(std::round(row)));
    }
    for (int i = kArcSteps; i >= 0; --i) {
        const double angle =
            -half_angle_rad +
            (2.0 * half_angle_rad) * static_cast<double>(i) / static_cast<double>(kArcSteps);
        double row = 0.0;
        double col = 0.0;
        vehicleBodyToImagePixel(outer_r * std::cos(angle), outer_r * std::sin(angle), row, col);
        sector_annulus.emplace_back(static_cast<int>(std::round(col)), static_cast<int>(std::round(row)));
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
                 cv::Point(static_cast<int>(std::round(col)), static_cast<int>(std::round(row))),
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

    cv::putText(image,
                "ground ROI",
                cv::Point(8, 18),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(255, 200, 0),
                1,
                cv::LINE_AA);
    cv::putText(image,
                "ring+sector+z",
                cv::Point(8, 36),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                ring_color,
                1,
                cv::LINE_AA);
}

void LidarBevBuilder::drawCenterMark(cv::Mat& image) const
{
    if (image.empty()) {
        return;
    }

    double origin_row = 0.0;
    double origin_col = 0.0;
    vehicleBodyToImagePixel(0.0, 0.0, origin_row, origin_col);
    const int center_col = static_cast<int>(std::round(origin_col));
    const int center_row = static_cast<int>(std::round(origin_row));
    const int marker = std::max(4, static_cast<int>(1.0 / config_.resolution));
    cv::line(image,
             cv::Point(center_col - marker, center_row),
             cv::Point(center_col + marker, center_row),
             cv::Scalar(255, 255, 255),
             1);
    cv::line(image,
             cv::Point(center_col, center_row - marker),
             cv::Point(center_col, center_row + marker),
             cv::Scalar(255, 255, 255),
             1);
}

bool LidarBevBuilder::isGroundRoiPlanarPoint(double x, double y) const
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

bool LidarBevBuilder::isGroundRoiBodyPoint(double x, double y, double z) const
{
    if (!isGroundRoiPlanarPoint(x, y)) {
        return false;
    }

    if (!std::isfinite(z)) {
        return false;
    }

    return z >= config_.ground_candidate_min_z && z <= config_.ground_candidate_max_z;
}

double LidarBevBuilder::estimateGroundReferenceFromHistory(
    const Eigen::Isometry3d& current_pose,
    bool current_pose_valid) const
{
    const std::size_t reference_point_count =
        static_cast<std::size_t>(config_.ground_min_points);
    constexpr std::size_t kMaxGroundHistoryFrames = 10;

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
         frame_it != frame_history_.rend() && frames_used < kMaxGroundHistoryFrames;
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
    last_ground_roi_candidate_count_ = static_cast<int>(ground_candidates.size());

    if (ground_candidates.size() >= static_cast<std::size_t>(config_.ground_min_points)) {
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

double LidarBevBuilder::meanTopHeightFraction(const std::vector<float>& values,
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

double LidarBevBuilder::percentile(const std::vector<float>& values, double quantile) const
{
    if (values.empty()) {
        return 0.0;
    }

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

    quantile = std::max(0.0, std::min(1.0, quantile));
    const std::size_t index = static_cast<std::size_t>(
        std::floor(quantile * static_cast<double>(height_sort_buffer_.size() - 1)));
    std::nth_element(height_sort_buffer_.begin(),
                     height_sort_buffer_.begin() + static_cast<std::ptrdiff_t>(index),
                     height_sort_buffer_.end());
    return height_sort_buffer_[index];
}

bool LidarBevBuilder::isFinitePoint(const PointXYZRGBValid& point) const
{
    return isFinite(point.x) && isFinite(point.y) && isFinite(point.z);
}

double LidarBevBuilder::getGroundReference() const
{
    return last_ground_reference_;
}

int LidarBevBuilder::getGroundRoiCandidateCount() const
{
    return last_ground_roi_candidate_count_;
}

int LidarBevBuilder::getGroundRoiPlanarCount() const
{
    return last_ground_roi_planar_count_;
}

double LidarBevBuilder::getCenterRelativeHeight() const
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

void LidarBevBuilder::publish(const ros::Publisher& publisher) const
{
    if (!publisher || map_.getLayers().empty()) {
        return;
    }

    grid_map_msgs::GridMap message;
    grid_map::GridMapRosConverter::toMessage(map_, message);
    message.info.header.stamp = stamp_;
    message.info.header.frame_id = map_.getFrameId();
    publisher.publish(message);
}
