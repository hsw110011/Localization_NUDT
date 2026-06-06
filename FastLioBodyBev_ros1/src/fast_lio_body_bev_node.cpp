#include <Eigen/Geometry>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
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
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

struct PointXYZ {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

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
    std::string frame_id = "body";
    double resolution = 0.2;
    double map_size_x = 100.0;
    double map_size_y = 100.0;
    double point_min_z = -10.0;
    double point_max_z = 200.0;
    bool ego_filter_enabled = true;
    double ego_box_min_x = -3.0;
    double ego_box_max_x = 1.0;
    double ego_box_min_y = -1.0;
    double ego_box_max_y = 1.0;
    double height_quantile = 0.90;
    int accumulation_frame_count = 5;
    int count_saturation = 20;
    double distance_decay_alpha = 0.03;
    double edge_gradient_threshold = 2.0;
    double edge_min_height = 0.5;
    double edge_min_jump = 0.5;
    int edge_min_valid_neighbors = 3;
    bool show_windows = false;
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
    config_.accumulation_frame_count = std::max(1, config_.accumulation_frame_count);
    config_.count_saturation = std::max(1, config_.count_saturation);
    config_.edge_gradient_threshold = std::max(0.0, config_.edge_gradient_threshold);
    config_.edge_min_height = std::max(0.0, config_.edge_min_height);
    config_.edge_min_jump = std::max(0.0, config_.edge_min_jump);
    config_.edge_min_valid_neighbors = std::max(1, std::min(9, config_.edge_min_valid_neighbors));
  }

  void build(const std::vector<PointXYZ>& body_points, const ros::Time& stamp)
  {
    stamp_ = stamp.isZero() ? ros::Time::now() : stamp;
    initializeMap(config_.frame_id, grid_map::Position(0.0, 0.0));

    std::vector<PointXYZ> prepared_points;
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
      prepared_points.push_back(point);
    }

    const double ground_reference = estimateGroundReference(prepared_points);
    std::unordered_map<GlobalCellKey, CellAccumulator, GlobalCellKeyHash> current_cells;

    for (const auto& point : prepared_points) {
      if (!map_.isInside(grid_map::Position(point.x, point.y))) {
        continue;
      }

      const GlobalCellKey key = makeGlobalCellKey(point.x, point.y);
      CellAccumulator& cell = current_cells[key];
      cell.max_z = std::max(cell.max_z, point.z);
      ++cell.count;
    }

    for (const auto& entry : current_cells) {
      const CellAccumulator& cell = entry.second;
      if (cell.count <= 0) {
        continue;
      }

      const grid_map::Position cell_center = getGlobalCellCenter(entry.first);
      const double distance = std::hypot(cell_center.x(), cell_center.y());
      const double count_weight =
        std::min(static_cast<double>(cell.count) /
                 static_cast<double>(config_.count_saturation),
                 1.0);
      const double distance_weight = std::exp(-config_.distance_decay_alpha * distance);

      CellObservation observation;
      observation.stamp = stamp_;
      observation.h_rel = static_cast<float>(cell.max_z - ground_reference);
      observation.h_q = cell.max_z;
      observation.count = static_cast<float>(cell.count);
      observation.confidence = static_cast<float>(count_weight * distance_weight);
      addObservationToHistory(entry.first, observation);
    }

    fillMapFromHistory();
    computeRobustEdges();

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

private:
  struct CellAccumulator {
    float max_z = -std::numeric_limits<float>::max();
    int count = 0;
  };

  struct GlobalCellKey {
    int col = 0;
    int row = 0;

    bool operator==(const GlobalCellKey& other) const
    {
      return col == other.col && row == other.row;
    }
  };

  struct GlobalCellKeyHash {
    std::size_t operator()(const GlobalCellKey& key) const noexcept
    {
      const std::size_t col_hash = std::hash<int>{}(key.col);
      const std::size_t row_hash = std::hash<int>{}(key.row);
      return col_hash ^ (row_hash + 0x9e3779b9u + (col_hash << 6) + (col_hash >> 2));
    }
  };

  struct CellObservation {
    ros::Time stamp;
    float h_rel = 0.0f;
    float h_q = 0.0f;
    float count = 0.0f;
    float confidence = 0.0f;
  };

  struct CellHistory {
    std::deque<CellObservation> observations;
  };

  Config config_;
  grid_map::GridMap map_;
  ros::Time stamp_;
  std::unordered_map<GlobalCellKey, CellHistory, GlobalCellKeyHash> cell_history_;

  void initializeMap(const std::string& frame_id, const grid_map::Position& center)
  {
    const std::vector<std::string> layers = {
      "H_rel_surf", "M_L", "W_L", "G_L", "G_long_L",
      "G_lat_L", "E_L", "N_L", "H_q"
    };

    map_ = grid_map::GridMap(layers);
    map_.setFrameId(frame_id);
    map_.setGeometry(grid_map::Length(config_.map_size_x, config_.map_size_y),
                     config_.resolution,
                     center);

    map_["H_rel_surf"].setConstant(kNaN);
    map_["G_L"].setConstant(kNaN);
    map_["G_long_L"].setConstant(kNaN);
    map_["G_lat_L"].setConstant(kNaN);
    map_["H_q"].setConstant(kNaN);
    map_["M_L"].setConstant(0.0f);
    map_["W_L"].setConstant(0.0f);
    map_["E_L"].setConstant(0.0f);
    map_["N_L"].setConstant(0.0f);
  }

  GlobalCellKey makeGlobalCellKey(double x, double y) const
  {
    GlobalCellKey key;
    key.col = static_cast<int>(std::floor(x / config_.resolution));
    key.row = static_cast<int>(std::floor(y / config_.resolution));
    return key;
  }

  grid_map::Position getGlobalCellCenter(const GlobalCellKey& key) const
  {
    return grid_map::Position((static_cast<double>(key.col) + 0.5) * config_.resolution,
                              (static_cast<double>(key.row) + 0.5) * config_.resolution);
  }

  void addObservationToHistory(const GlobalCellKey& key, const CellObservation& observation)
  {
    CellHistory& history = cell_history_[key];
    history.observations.push_back(observation);

    std::stable_sort(history.observations.begin(),
                     history.observations.end(),
                     [](const CellObservation& lhs, const CellObservation& rhs) {
                       return lhs.stamp < rhs.stamp;
                     });

    while (history.observations.size() >
           static_cast<std::size_t>(config_.accumulation_frame_count)) {
      history.observations.pop_front();
    }
  }

  float averageHistoryValue(const CellHistory& history, float CellObservation::*member) const
  {
    if (history.observations.empty()) {
      return kNaN;
    }

    double sum = 0.0;
    std::size_t count = 0;
    for (const auto& observation : history.observations) {
      const float value = observation.*member;
      if (!std::isfinite(value)) {
        continue;
      }
      sum += value;
      ++count;
    }

    if (count == 0) {
      return kNaN;
    }
    return static_cast<float>(sum / static_cast<double>(count));
  }

  void fillMapFromHistory()
  {
    auto& h_rel_layer = map_["H_rel_surf"];
    auto& mask_layer = map_["M_L"];
    auto& confidence_layer = map_["W_L"];
    auto& count_layer = map_["N_L"];
    auto& h_q_layer = map_["H_q"];

    std::vector<GlobalCellKey> cells_to_remove;

    for (const auto& entry : cell_history_) {
      const CellHistory& history = entry.second;
      const grid_map::Position cell_center = getGlobalCellCenter(entry.first);
      if (history.observations.empty() || !map_.isInside(cell_center)) {
        cells_to_remove.push_back(entry.first);
        continue;
      }

      grid_map::Index index;
      if (!map_.getIndex(cell_center, index)) {
        cells_to_remove.push_back(entry.first);
        continue;
      }

      const float h_rel = averageHistoryValue(history, &CellObservation::h_rel);
      if (!std::isfinite(h_rel)) {
        cells_to_remove.push_back(entry.first);
        continue;
      }

      const int row = index(0);
      const int col = index(1);
      if (mask_layer(row, col) > 0.5f &&
          std::isfinite(h_rel_layer(row, col)) &&
          h_rel <= h_rel_layer(row, col)) {
        continue;
      }

      h_rel_layer(row, col) = h_rel;
      h_q_layer(row, col) = averageHistoryValue(history, &CellObservation::h_q);
      count_layer(row, col) = averageHistoryValue(history, &CellObservation::count);
      confidence_layer(row, col) = averageHistoryValue(history, &CellObservation::confidence);
      mask_layer(row, col) = 1.0f;
    }

    for (const auto& key : cells_to_remove) {
      cell_history_.erase(key);
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

  void computeRobustEdges()
  {
    auto& h_rel_layer = map_["H_rel_surf"];
    auto& mask_layer = map_["M_L"];
    auto& gradient_layer = map_["G_L"];
    auto& longitudinal_gradient_layer = map_["G_long_L"];
    auto& lateral_gradient_layer = map_["G_lat_L"];
    auto& edge_layer = map_["E_L"];

    gradient_layer.setConstant(kNaN);
    longitudinal_gradient_layer.setConstant(kNaN);
    lateral_gradient_layer.setConstant(kNaN);
    edge_layer.setConstant(0.0f);

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
        const std::size_t index = gridIndex(row, col, cols);
        smoothed_height[index] = median_height;
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

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const std::size_t index = gridIndex(row, col, cols);
        if (!smoothed_valid[index]) {
          continue;
        }

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
              const std::size_t neighbor_index = gridIndex(neighbor_row, neighbor_col, cols);
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

        grid_map::Position center_position;
        grid_map::Position index0_position;
        grid_map::Position index1_position;
        const bool has_center_position =
          map_.getPosition(grid_map::Index(row, col), center_position);
        const bool has_index0_direction =
          row + 1 < rows && map_.getPosition(grid_map::Index(row + 1, col), index0_position);
        const bool has_index1_direction =
          col + 1 < cols && map_.getPosition(grid_map::Index(row, col + 1), index1_position);

        Eigen::Vector2d index0_unit(1.0, 0.0);
        Eigen::Vector2d index1_unit(0.0, 1.0);
        if (has_center_position && has_index0_direction) {
          index0_unit = Eigen::Vector2d(index0_position.x() - center_position.x(),
                                        index0_position.y() - center_position.y());
          if (index0_unit.norm() > 1e-6) {
            index0_unit.normalize();
          }
        }
        if (has_center_position && has_index1_direction) {
          index1_unit = Eigen::Vector2d(index1_position.x() - center_position.x(),
                                        index1_position.y() - center_position.y());
          if (index1_unit.norm() > 1e-6) {
            index1_unit.normalize();
          }
        }

        const double index0_gradient = grad_index0 / (8.0 * config_.resolution);
        const double index1_gradient = grad_index1 / (8.0 * config_.resolution);
        const Eigen::Vector2d map_gradient =
          index0_gradient * index0_unit + index1_gradient * index1_unit;

        longitudinal_gradient_layer(row, col) = static_cast<float>(map_gradient.x());
        lateral_gradient_layer(row, col) = static_cast<float>(map_gradient.y());
        gradient_layer(row, col) = static_cast<float>(map_gradient.norm());
      }
    }
  }

  double estimateGroundReference(const std::vector<PointXYZ>& points) const
  {
    constexpr double kFrontHalfAngleRad = M_PI / 6.0;
    constexpr std::size_t kReferencePointCount = 20;

    std::vector<std::pair<double, double>> front_candidates;
    std::vector<std::pair<double, double>> fallback_candidates;
    front_candidates.reserve(points.size());
    fallback_candidates.reserve(points.size());

    for (const auto& point : points) {
      if (!std::isfinite(point.z)) {
        continue;
      }

      const double distance_sq = point.x * point.x + point.y * point.y;
      fallback_candidates.emplace_back(distance_sq, point.z);

      if (point.x <= 0.0) {
        continue;
      }

      const double angle = std::atan2(point.y, point.x);
      if (std::abs(angle) > kFrontHalfAngleRad) {
        continue;
      }

      front_candidates.emplace_back(distance_sq, point.z);
    }

    const double front_reference = averageNearestHeights(front_candidates);
    if (std::isfinite(front_reference)) {
      return front_reference;
    }

    const double fallback_reference = averageNearestHeights(fallback_candidates);
    if (std::isfinite(fallback_reference)) {
      return fallback_reference;
    }

    return 0.0;
  }

  double averageNearestHeights(std::vector<std::pair<double, double>>& candidates) const
  {
    constexpr std::size_t kReferencePointCount = 20;
    if (candidates.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t used_count = std::min(kReferencePointCount, candidates.size());
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

  bool getVehicleFrameIndex(int image_row, int image_col, grid_map::Index& index) const
  {
    const int image_rows = map_.getSize()(0);
    const int image_cols = map_.getSize()(1);
    const double x_vehicle =
      0.5 * static_cast<double>(image_rows) * config_.resolution -
      (static_cast<double>(image_row) + 0.5) * config_.resolution;
    const double y_vehicle =
      0.5 * static_cast<double>(image_cols) * config_.resolution -
      (static_cast<double>(image_col) + 0.5) * config_.resolution;

    return map_.getIndex(grid_map::Position(x_vehicle, y_vehicle), index);
  }

  void drawCenterMark(cv::Mat& image) const
  {
    if (image.empty()) {
      return;
    }

    const int center_col = image.cols / 2;
    const int center_row = image.rows / 2;
    const int marker = std::max(4, static_cast<int>(1.0 / config_.resolution));
    cv::line(image, cv::Point(center_col - marker, center_row),
             cv::Point(center_col + marker, center_row), cv::Scalar(255, 255, 255), 1);
    cv::line(image, cv::Point(center_col, center_row - marker),
             cv::Point(center_col, center_row + marker), cv::Scalar(255, 255, 255), 1);
  }

  cv::Mat renderHeightLayer(const std::string& layer_name) const
  {
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name) || !map_.exists("M_L")) {
      return image;
    }

    const auto& layer = map_[layer_name];
    const auto& mask = map_["M_L"];
    const auto range = getRobustLayerRange(layer_name, 0.0, 1.0);
    const double inv_range = range.second > range.first
      ? 1.0 / (range.second - range.first)
      : 1.0;

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

        image.at<cv::Vec3b>(image_row, image_col) =
          viridisColor((static_cast<double>(value) - range.first) * inv_range);
      }
    }

    drawCenterMark(image);
    return image;
  }

  cv::Mat renderNormalizedLayer(const std::string& layer_name,
                                double min_value,
                                double max_value) const
  {
    const int rows = map_.getSize()(0);
    const int cols = map_.getSize()(1);
    cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    if (!map_.exists(layer_name) || !map_.exists("M_L") || max_value <= min_value) {
      return image;
    }

    const auto& layer = map_[layer_name];
    const auto& mask = map_["M_L"];
    const double inv_range = 1.0 / (max_value - min_value);

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

        const double clipped =
          std::max(min_value, std::min(max_value, static_cast<double>(value)));
        image.at<cv::Vec3b>(image_row, image_col) =
          viridisColor((clipped - min_value) * inv_range);
      }
    }

    drawCenterMark(image);
    return image;
  }

  cv::Mat renderBinaryLayer(const std::string& layer_name) const
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

  void showDebugWindows() const
  {
    if (map_.getLayers().empty()) {
      return;
    }

    static const std::array<std::string, 5> window_names = {
      "BEV Relative Surface Height Map (H_rel_surf)",
      "BEV Valid Observation Mask (M_L)",
      "BEV Observation Confidence Map (W_L)",
      "BEV Longitudinal Height Gradient Map (G_long_L)",
      "BEV Lateral Height Gradient Map (G_lat_L)"
    };

    cv::imshow(window_names[0], renderHeightLayer("H_rel_surf"));
    cv::imshow(window_names[1], renderBinaryLayer("M_L"));
    cv::imshow(window_names[2], renderNormalizedLayer("W_L", 0.0, 1.0));
    cv::imshow(window_names[3], renderHeightLayer("G_long_L"));
    cv::imshow(window_names[4], renderHeightLayer("G_lat_L"));
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
    pnh_.param<std::string>("output_topic", output_topic_, "/lidar_bev/grid_map");
    pnh_.param<double>("update_rate_hz", update_rate_hz_, 10.0);

    BodyBevBuilder::Config config;
    pnh_.param<std::string>("frame_id", config.frame_id, "body");
    pnh_.param<double>("resolution", config.resolution, 0.2);
    pnh_.param<double>("map_size_x", config.map_size_x, 100.0);
    pnh_.param<double>("map_size_y", config.map_size_y, 100.0);
    pnh_.param<double>("point_min_z", config.point_min_z, -10.0);
    pnh_.param<double>("point_max_z", config.point_max_z, 200.0);
    pnh_.param<bool>("ego_filter_enabled", config.ego_filter_enabled, true);
    pnh_.param<double>("ego_box_min_x", config.ego_box_min_x, -3.0);
    pnh_.param<double>("ego_box_max_x", config.ego_box_max_x, 1.0);
    pnh_.param<double>("ego_box_min_y", config.ego_box_min_y, -1.0);
    pnh_.param<double>("ego_box_max_y", config.ego_box_max_y, 1.0);
    pnh_.param<double>("height_quantile", config.height_quantile, 0.90);
    pnh_.param<int>("accumulation_frame_count", config.accumulation_frame_count, 5);
    pnh_.param<int>("count_saturation", config.count_saturation, 20);
    pnh_.param<double>("distance_decay_alpha", config.distance_decay_alpha, 0.03);
    pnh_.param<double>("edge_gradient_threshold", config.edge_gradient_threshold, 2.0);
    pnh_.param<double>("edge_min_height", config.edge_min_height, 0.5);
    pnh_.param<double>("edge_min_jump", config.edge_min_jump, 0.5);
    pnh_.param<int>("edge_min_valid_neighbors", config.edge_min_valid_neighbors, 3);
    pnh_.param<bool>("show_windows", config.show_windows, false);

    builder_.configure(config);
    pub_ = nh_.advertise<grid_map_msgs::GridMap>(output_topic_, 5);
    sub_ = nh_.subscribe(input_topic_, 5, &BodyBevNode::cloudCallback, this);

    ROS_INFO("FastLioBodyBev: subscribe %s, publish %s, frame_id=%s",
             input_topic_.c_str(), output_topic_.c_str(), config.frame_id.c_str());
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  BodyBevBuilder builder_;
  std::string input_topic_;
  std::string output_topic_;
  double update_rate_hz_ = 10.0;
  ros::Time last_update_;

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

    builder_.build(points, msg->header.stamp);
    builder_.publish(pub_);
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
