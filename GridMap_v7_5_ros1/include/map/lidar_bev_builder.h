#pragma once

#include "include/map/my_config.h"

#include <Eigen/Geometry>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <cstddef>
#include <deque>
#include <limits>
#include <string>
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
        double ground_front_half_angle_deg = 7.5;
        bool ground_require_forward = true;
        double ground_failure_fallback_z = 0.0;
        int ground_min_points = 30;

        double height_quantile = 0.90;
        int cell_max_points = 500;
        int debug_window_stride = 2;
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
    double getCenterRelativeHeight() const;
    double getGroundReference() const;
    int getGroundRoiCandidateCount() const;
    int getGroundRoiPlanarCount() const;

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
        int count = 0;
        std::vector<float> top_heights;
    };

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
    };

    Config config_;
    grid_map::GridMap map_;
    ros::Time stamp_;
    Eigen::Isometry3d display_pose_ = Eigen::Isometry3d::Identity();
    bool display_pose_valid_ = false;
    std::deque<FrameObservation> frame_history_;
    std::vector<CellAccumulator> cell_accumulators_;
    std::vector<std::size_t> active_cell_indices_;
    std::vector<int> cell_keep_counts_;
    mutable int debug_window_counter_ = 0;
    bool map_initialized_ = false;
    int map_rows_ = 0;
    int map_cols_ = 0;
    double map_half_x_ = 0.0;
    double map_half_y_ = 0.0;
    double map_inv_resolution_ = 0.0;
    std::vector<float> smoothed_height_;
    std::vector<uint8_t> smoothed_valid_;
    mutable std::vector<float> height_sort_buffer_;
    std::vector<StoredPoint> prune_kept_buffer_;
    std::vector<PreparedPoint> prepared_points_buffer_;
    mutable double last_ground_reference_ = std::numeric_limits<double>::quiet_NaN();
    mutable int last_ground_roi_candidate_count_ = 0;
    mutable int last_ground_roi_planar_count_ = 0;

    void initializeMap(const std::string& frame_id, const grid_map::Position& center);
    void ensureMap(const std::string& frame_id, const grid_map::Position& center);
    void resetMapLayers();
    void updateMapIndexLookup();
    bool positionToCellIndex(double x, double y, int& row, int& col) const;
    static float medianOfSmallArray(float* values, int count);
    void addFrameToHistory(const Eigen::Isometry3d& capture_pose,
                           bool pose_valid,
                           const ros::Time& stamp,
                           const std::vector<PreparedPoint>& points);
    void pruneFrameHistoryCellPointLimit(const Eigen::Isometry3d& current_pose,
                                         bool current_pose_valid);
    void fillMapFromFrameHistory(const Eigen::Isometry3d& current_pose,
                                 bool current_pose_valid,
                                 double ground_reference);
    void addTopHeight(CellAccumulator& cell, float height, int max_top_count) const;
    bool shouldRejectEgoPoint(const PointXYZRGBValid& point) const;
    void computeDirectionalGradients();
    void showDebugWindows() const;
    cv::Mat renderStackedDebugLayers() const;
    cv::Mat renderColorizedLayer(const std::string& layer_name,
                                 double min_value,
                                 double max_value) const;
    cv::Mat renderHeightLayer(const std::string& layer_name) const;
    cv::Mat renderBinaryLayer(const std::string& layer_name, bool draw_center_mark) const;
    cv::Vec3b viridisColor(double normalized_value) const;
    std::pair<double, double> getRobustLayerRange(const std::string& layer_name,
                                                  double fallback_min,
                                                  double fallback_max) const;
    bool getVehicleFrameIndex(int image_row, int image_col, grid_map::Index& index) const;
    bool isDebugPixelValid(const grid_map::Index& index) const;
    void drawCenterMark(cv::Mat& image) const;
    void drawGroundReferenceRegion(cv::Mat& image) const;
    void vehicleBodyToImagePixel(double x, double y, double& row, double& col) const;
    double estimateGroundReferenceFromHistory(const Eigen::Isometry3d& current_pose,
                                              bool current_pose_valid) const;
    bool isGroundRoiBodyPoint(double x, double y, double z) const;
    bool isGroundRoiPlanarPoint(double x, double y) const;
    double meanTopHeightFraction(const std::vector<float>& values,
                                 double top_fraction,
                                 int total_count) const;
    double percentile(const std::vector<float>& values, double quantile) const;
    bool isFinitePoint(const PointXYZRGBValid& point) const;
};
