#pragma once

#include "include/map/my_config.h"

#include <Eigen/Geometry>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <deque>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class LidarBevBuilder {
public:
    struct Config {
        bool enabled = true;
        bool use_odometry_frame = true;
        bool use_ransac_ground = true;
        bool show_windows = true;

        std::string topic = "/lidar_bev/grid_map";
        std::string odometry_frame_id = "map";
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
        double ground_fallback_quantile = 0.35;
        int ground_min_points = 30;

        double height_quantile = 0.90;
        int accumulation_frame_count = 5;
        int count_saturation = 8;
        double distance_decay_alpha = 0.02;
        double edge_gradient_threshold = 1.0;
        double edge_min_height = 0.45;
        double edge_min_jump = 0.50;
        int edge_min_valid_neighbors = 4;
    };

    struct OdometryState {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        ros::Time stamp;
        std::string frame_id;
        bool valid = false;
    };

    LidarBevBuilder() = default;

    void configure(const Config& config);
    void build(const std::vector<PointXYZRGBValid>& car_points,
               const OdometryState* odometry);
    void publish(const ros::Publisher& publisher) const;

    const grid_map::GridMap& getGridMap() const { return map_; }
    const Config& getConfig() const { return config_; }

private:
    struct PreparedPoint {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double car_x = 0.0;
        double car_y = 0.0;
        double car_z = 0.0;
    };

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
        std::size_t operator()(const GlobalCellKey& key) const noexcept;
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
    Eigen::Isometry3d display_pose_ = Eigen::Isometry3d::Identity();
    bool display_pose_valid_ = false;
    std::unordered_map<GlobalCellKey, CellHistory, GlobalCellKeyHash> cell_history_;

    void initializeMap(const std::string& frame_id, const grid_map::Position& center);
    GlobalCellKey makeGlobalCellKey(double x, double y) const;
    grid_map::Position getGlobalCellCenter(const GlobalCellKey& key) const;
    void addObservationToHistory(const GlobalCellKey& key, const CellObservation& observation);
    void fillMapFromHistory(const grid_map::Position& center);
    float averageHistoryValue(const CellHistory& history, float CellObservation::*member) const;
    bool shouldRejectEgoPoint(const PointXYZRGBValid& point) const;
    void computeRobustEdges();
    void showDebugWindows() const;
    cv::Mat renderHeightLayer(const std::string& layer_name) const;
    cv::Mat renderNormalizedLayer(const std::string& layer_name,
                                  double min_value,
                                  double max_value) const;
    cv::Mat renderBinaryLayer(const std::string& layer_name) const;
    cv::Vec3b viridisColor(double normalized_value) const;
    std::pair<double, double> getRobustLayerRange(const std::string& layer_name,
                                                  double fallback_min,
                                                  double fallback_max) const;
    bool getVehicleFrameIndex(int image_row, int image_col, grid_map::Index& index) const;
    void drawCenterMark(cv::Mat& image) const;
    double estimateGroundReference(const std::vector<PreparedPoint>& points) const;
    double percentile(std::vector<float> values, double quantile) const;
    bool isFinitePoint(const PointXYZRGBValid& point) const;
};
