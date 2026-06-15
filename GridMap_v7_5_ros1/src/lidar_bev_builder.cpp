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
#include <functional>
#include <limits>
#include <utility>

namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kSurfaceTopHeightFraction = 0.15;  

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

    map_["H_rel_surf"].setConstant(kNaN);
    map_["G_long_L"].setConstant(kNaN);
    map_["G_lat_L"].setConstant(kNaN);
    map_["M_L"].setConstant(0.0f);
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

    initializeMap(frame_id, center);

    std::vector<PreparedPoint> prepared_points;
    prepared_points.reserve(car_points.size());

    for (const auto& point : car_points) {
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

    const double ground_reference = estimateGroundReference(prepared_points);

    addFrameToHistory(capture_pose, capture_pose_valid, stamp_, prepared_points);
    pruneFrameHistoryCellPointLimit(current_pose, capture_pose_valid);
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
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);
    if (cell_keep_counts_.size() != cell_count) {
        cell_keep_counts_.assign(cell_count, 0);
    } else {
        std::fill(cell_keep_counts_.begin(), cell_keep_counts_.end(), 0);
    }

    const bool limit_cell_points = config_.cell_max_points > 0;
    const Eigen::Isometry3d current_pose_inverse = current_pose.inverse();

    for (auto frame_it = frame_history_.rbegin(); frame_it != frame_history_.rend(); ++frame_it) {
        FrameObservation& frame = *frame_it;
        if (current_pose_valid != frame.pose_valid) {
            frame.points.clear();
            continue;
        }

        Eigen::Isometry3d frame_to_current = Eigen::Isometry3d::Identity();
        if (current_pose_valid && frame.pose_valid) {
            frame_to_current = current_pose_inverse * frame.pose;
        }

        std::vector<StoredPoint> kept_reversed;
        kept_reversed.reserve(frame.points.size());

        for (auto point_it = frame.points.rbegin(); point_it != frame.points.rend(); ++point_it) {
            Eigen::Vector3d local_point(point_it->x, point_it->y, point_it->z);
            local_point = frame_to_current * local_point;

            if (!std::isfinite(local_point.x()) || !std::isfinite(local_point.y())) {
                continue;
            }

            grid_map::Index index;
            if (!map_.getIndex(grid_map::Position(local_point.x(), local_point.y()), index)) {
                continue;
            }

            if (limit_cell_points) {
                const std::size_t linear_index = gridIndex(index(0), index(1), cols);
                int& kept_count = cell_keep_counts_[linear_index];
                if (kept_count >= config_.cell_max_points) {
                    continue;
                }
                ++kept_count;
            }

            kept_reversed.push_back(*point_it);
        }

        frame.points.assign(kept_reversed.rbegin(), kept_reversed.rend());
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

    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);
    if (cell_accumulators_.size() != cell_count) {
        cell_accumulators_.clear();
        cell_accumulators_.resize(cell_count);
        active_cell_indices_.clear();
    } else {
        for (const std::size_t index : active_cell_indices_) {
            CellAccumulator& cell = cell_accumulators_[index];
            cell.count = 0;
            cell.top_heights.clear();
        }
        active_cell_indices_.clear();
    }

    const int max_top_height_count = config_.cell_max_points > 0
        ? std::max(1,
                   static_cast<int>(
                       std::ceil(kSurfaceTopHeightFraction *
                                 static_cast<double>(config_.cell_max_points))))
        : 0;
    const Eigen::Isometry3d current_pose_inverse = current_pose.inverse();

    for (const auto& frame : frame_history_) {
        if (current_pose_valid != frame.pose_valid) {
            continue;
        }

        Eigen::Isometry3d frame_to_current = Eigen::Isometry3d::Identity();
        if (current_pose_valid && frame.pose_valid) {
            frame_to_current = current_pose_inverse * frame.pose;
        }

        for (const auto& point : frame.points) {
            Eigen::Vector3d local_point(point.x, point.y, point.z);
            local_point = frame_to_current * local_point;

            grid_map::Index index;
            if (!map_.getIndex(grid_map::Position(local_point.x(), local_point.y()), index)) {
                continue;
            }

            const std::size_t linear_index = gridIndex(index(0), index(1), cols);
            CellAccumulator& cell = cell_accumulators_[linear_index];
            if (cell.count == 0) {
                active_cell_indices_.push_back(linear_index);
            }
            const float z = static_cast<float>(local_point.z());
            ++cell.count;
            addTopHeight(cell, z, max_top_height_count);
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

    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    const std::size_t cell_count = static_cast<std::size_t>(rows * cols);

    std::vector<float> smoothed_height(cell_count, kNaN);
    std::vector<uint8_t> smoothed_valid(cell_count, 0);

    // Smooth H_rel_surf with a 3x3 median before Sobel gradients.
    for (const std::size_t index : active_cell_indices_) {
        const int row = static_cast<int>(index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(index % static_cast<std::size_t>(cols));
        if (mask_layer(row, col) <= 0.5f || !std::isfinite(h_rel_layer(row, col))) {
            continue;
        }

        std::array<float, 9> values;
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
                values[static_cast<std::size_t>(value_count)] =
                    h_rel_layer(neighbor_row, neighbor_col);
                ++value_count;
            }
        }

        if (value_count < config_.edge_min_valid_neighbors) {
            continue;
        }

        std::sort(values.begin(), values.begin() + value_count);
        const float median_height = values[static_cast<std::size_t>(value_count / 2)];
        smoothed_height[index] = median_height;
        smoothed_valid[index] = 1;
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

    for (const std::size_t index : active_cell_indices_) {
        if (!smoothed_valid[index]) {
            continue;
        }

        const int row = static_cast<int>(index / static_cast<std::size_t>(cols));
        const int col = static_cast<int>(index % static_cast<std::size_t>(cols));
        const float center_height = smoothed_height[index];
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
                    if (smoothed_valid[neighbor_index]) {
                        value = smoothed_height[neighbor_index];
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

    static const std::array<std::string, 4> window_names = {
        "BEV Relative Surface Height Map (H_rel_surf)",
        "BEV Valid Observation Mask (M_L)",
        "BEV Longitudinal Height Gradient Map (G_long_L)",
        "BEV Lateral Height Gradient Map (G_lat_L)"
    };
    static bool windows_created = false;
    if (!windows_created) {
        for (const auto& window_name : window_names) {
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        }
        windows_created = true;
    }

    cv::imshow(window_names[0], renderHeightLayer("H_rel_surf"));
    cv::imshow(window_names[1], renderBinaryLayer("M_L"));
    cv::imshow(window_names[2], renderHeightLayer("G_long_L"));
    cv::imshow(window_names[3], renderHeightLayer("G_lat_L"));
    cv::waitKey(1);
}

cv::Mat LidarBevBuilder::renderHeightLayer(const std::string& layer_name) const
{
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name) || !map_.exists("M_L")) {
        return image;
    }

    const auto& layer = map_[layer_name];
    const auto& mask = map_["M_L"];
    const std::pair<double, double> range = getRobustLayerRange(layer_name, 0.0, 1.0);
    const double min_value = range.first;
    const double max_value = range.second;
    const double inv_range = max_value > min_value ? 1.0 / (max_value - min_value) : 1.0;

    for (int image_row = 0; image_row < rows; ++image_row) {
        for (int image_col = 0; image_col < cols; ++image_col) {
            grid_map::Index index;
            if (!getVehicleFrameIndex(image_row, image_col, index)) {
                continue;
            }

            const float value = layer(index(0), index(1));
            if (mask(index(0), index(1)) <= 0.5f || !std::isfinite(value)) {
                continue;
            }

            const double normalized =
                (static_cast<double>(value) - min_value) * inv_range;
            image.at<cv::Vec3b>(image_row, image_col) = viridisColor(normalized);
        }
    }

    drawCenterMark(image);
    return image;
}

cv::Mat LidarBevBuilder::renderBinaryLayer(const std::string& layer_name) const
{
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name)) {
        return image;
    }

    const auto& layer = map_[layer_name];
    for (int image_row = 0; image_row < rows; ++image_row) {
        for (int image_col = 0; image_col < cols; ++image_col) {
            grid_map::Index index;
            if (!getVehicleFrameIndex(image_row, image_col, index)) {
                continue;
            }

            if (layer(index(0), index(1)) > 0.5f) {
                image.at<cv::Vec3b>(image_row, image_col) = cv::Vec3b(255, 255, 255);
            }
        }
    }

    drawCenterMark(image);
    return image;
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
    const auto& mask = map_["M_L"];
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(rows * cols));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float value = layer(row, col);
            if (mask(row, col) <= 0.5f || !std::isfinite(value)) {
                continue;
            }
            values.push_back(value);
        }
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

void LidarBevBuilder::drawCenterMark(cv::Mat& image) const
{
    if (image.empty()) {
        return;
    }

    const int center_col = image.cols / 2;
    const int center_row = image.rows / 2;
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

double LidarBevBuilder::estimateGroundReference(
    const std::vector<PreparedPoint>& points) const
{
    constexpr double kFrontHalfAngleRad = M_PI / 12.0;
    constexpr std::size_t kReferencePointCount = 20;

    std::vector<std::pair<double, double>> front_candidates;
    std::vector<std::pair<double, double>> fallback_candidates;
    front_candidates.reserve(points.size());
    fallback_candidates.reserve(points.size());

    for (const auto& point : points) {
        if (!std::isfinite(point.z)) {
            continue;
        }

        const double distance_sq = point.car_x * point.car_x + point.car_y * point.car_y;
        const double distance = std::sqrt(distance_sq);
        if (distance < config_.near_inner_radius || distance > config_.near_outer_radius) {
            continue;
        }
        if (point.car_z < config_.ground_candidate_min_z ||
            point.car_z > config_.ground_candidate_max_z) {
            continue;
        }

        fallback_candidates.emplace_back(distance_sq, point.car_z);

        if (point.car_x <= 0.0) {
            continue;
        }

        const double angle = std::atan2(point.car_y, point.car_x);
        if (std::abs(angle) > kFrontHalfAngleRad) {
            continue;
        }

        front_candidates.emplace_back(distance_sq, point.car_z);
    }

    auto averageNearestHeights =
        [kReferencePointCount](std::vector<std::pair<double, double>>& candidates) -> double {
            if (candidates.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            const std::size_t used_count =
                std::min(kReferencePointCount, candidates.size());
            std::nth_element(candidates.begin(),
                             candidates.begin() + static_cast<std::ptrdiff_t>(used_count - 1),
                             candidates.end(),
                             [](const auto& lhs, const auto& rhs) {
                                 return lhs.first < rhs.first;
                             });

            double z_sum = 0.0;
            for (std::size_t i = 0; i < used_count; ++i) {
                z_sum += candidates[i].second;
            }
            return z_sum / static_cast<double>(used_count);
        };

    const double front_reference =
        front_candidates.size() >= static_cast<std::size_t>(config_.ground_min_points)
            ? averageNearestHeights(front_candidates)
            : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(front_reference)) {
        return front_reference;
    }

    const double fallback_reference =
        fallback_candidates.size() >= static_cast<std::size_t>(config_.ground_min_points)
            ? averageNearestHeights(fallback_candidates)
            : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(fallback_reference)) {
        return fallback_reference;
    }

    return 0.0;
}

double LidarBevBuilder::meanTopHeightFraction(std::vector<float> values,
                                              double top_fraction,
                                              int total_count) const
{
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](float value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty()) {
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
    const std::size_t used_count = std::min(requested_count, values.size());

    if (used_count < values.size()) {
        std::nth_element(values.begin(),
                         values.begin() + static_cast<std::ptrdiff_t>(used_count - 1),
                         values.end(),
                         std::greater<float>());
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < used_count; ++i) {
        sum += static_cast<double>(values[i]);
    }
    return sum / static_cast<double>(used_count);
}

double LidarBevBuilder::percentile(std::vector<float> values, double quantile) const
{
    if (values.empty()) {
        return 0.0;
    }

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

bool LidarBevBuilder::isFinitePoint(const PointXYZRGBValid& point) const
{
    return isFinite(point.x) && isFinite(point.y) && isFinite(point.z);
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
