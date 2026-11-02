#pragma once

#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sem_loc {
enum class BevMode { Certain, Uncertain };
enum class ScoreMode { Baseline, NoEntropy, Entropy };
enum class ComputeBackend { Gpu, Forloop };

struct Config {
    BevMode bev_mode = BevMode::Uncertain;
    ScoreMode score_mode = ScoreMode::Entropy;
    ComputeBackend backend = ComputeBackend::Gpu;
    double resolution = 0.15;
    double view_range = 60.0;
    bool visualize = true;
    std::array<int, 3> probability_channels = {0, 1, 2};
    bool input_cloud_in_world = true;
    int history_frames = 5;
    double surface_band = 0.5;
    double accumulation_distance = 0.5;
    double accumulation_yaw = 2.0;
    double accumulation_interval = 0.5;
    int num_particles = 256;
    double tdf_value_scale = 100.0; // 当前 NPZ 的原始截断距离值；已归一化地图填 1。
    double score_lambda = 0.001;
    std::string map_path = "/home/hsw/catkin_ws/doc/kitti_raw_1003_0027.npz";
};

// 两个程序都直接读取本地 YAML；_config:=... 可覆盖路径，不需要 launch/rosparam load。
inline Config LoadConfig(const std::string& default_path)
{
    std::string path;
    ros::NodeHandle("~").param<std::string>("config", path, default_path);
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("cannot open config: " + path);
    const auto bev = file["bev"];
    const auto pf = file["particle_filter"];
    std::string mode, method, backend;
    bev["mode"] >> mode;
    pf["method"] >> method;
    cv::read(pf["backend"], backend, std::string("gpu"));
    Config config;
    if (backend == "forloop") config.backend = ComputeBackend::Forloop;
    else if (backend != "gpu") throw std::invalid_argument("particle_filter.backend must be gpu or forloop");
    if (mode == "certain") config.bev_mode = BevMode::Certain;
    else if (mode != "uncertain") throw std::invalid_argument("bev.mode must be certain or uncertain");
    if (method == "baseline") config.score_mode = ScoreMode::Baseline;
    else if (method == "ours_no_entropy") config.score_mode = ScoreMode::NoEntropy;
    else if (method != "ours_entropy")
        throw std::invalid_argument("particle_filter.method must be baseline, ours_no_entropy or ours_entropy");
    if ((config.bev_mode == BevMode::Certain) != (config.score_mode == ScoreMode::Baseline))
        throw std::invalid_argument("baseline requires certain BEV; ours_* requires uncertain BEV");

    cv::read(file["map_path"], config.map_path, config.map_path);
    cv::read(bev["resolution"], config.resolution, config.resolution);
    cv::read(bev["view_range"], config.view_range, config.view_range);
    int visualize = 1;
    cv::read(bev["visualize"], visualize, visualize);
    config.visualize = visualize != 0;
    int world = 1;
    cv::read(bev["input_cloud_in_world"], world, world);
    config.input_cloud_in_world = world != 0;
    cv::read(bev["history_frames"], config.history_frames, config.history_frames);
    cv::read(bev["surface_band_m"], config.surface_band, config.surface_band);
    cv::read(bev["accumulation_update_distance_m"], config.accumulation_distance, config.accumulation_distance);
    cv::read(bev["accumulation_update_yaw_deg"], config.accumulation_yaw, config.accumulation_yaw);
    cv::read(bev["accumulation_max_interval_sec"], config.accumulation_interval, config.accumulation_interval);
    if (config.history_frames < 1 || config.history_frames > 100 ||
        !std::isfinite(config.surface_band) || config.surface_band < 0 ||
        !std::isfinite(config.accumulation_distance) || config.accumulation_distance < 0 ||
        !std::isfinite(config.accumulation_yaw) || config.accumulation_yaw < 0 ||
        !std::isfinite(config.accumulation_interval) || config.accumulation_interval < 0)
        throw std::invalid_argument("invalid BEV accumulation parameters");
    cv::read(pf["num_particles"], config.num_particles, config.num_particles);
    cv::read(pf["tdf_value_scale"], config.tdf_value_scale, config.tdf_value_scale);
    cv::read(pf["score_lambda"], config.score_lambda, config.score_lambda);
    if (!std::isfinite(config.resolution) || config.resolution <= 0 ||
        !std::isfinite(config.view_range) || config.view_range < 2 * config.resolution ||
        config.num_particles <= 0 ||
        !std::isfinite(config.tdf_value_scale) || config.tdf_value_scale <= 0 ||
        !std::isfinite(config.score_lambda) || config.score_lambda <= 0)
        throw std::invalid_argument("invalid BEV dimensions or particle filter parameters");
    if (!bev["probability_channels"].empty()) {
        std::vector<int> channels;
        bev["probability_channels"] >> channels;
        auto sorted = channels;
        std::sort(sorted.begin(), sorted.end());
        if (sorted != std::vector<int>({0, 1, 2}))
            throw std::invalid_argument("bev.probability_channels must be a permutation of [0, 1, 2]");
        std::copy(channels.begin(), channels.end(), config.probability_channels.begin());
    }
    ROS_DEBUG("Config: %s, BEV=%s, PF=%s, backend=%s",
             path.c_str(), mode.c_str(), method.c_str(), backend.c_str());
    return config;
}
}
