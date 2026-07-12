#ifndef PARTICLE_WEIGHT_VISUALIZATION_H
#define PARTICLE_WEIGHT_VISUALIZATION_H

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>

#include "CoordConverter.h"

#ifndef SEM_LOC_SOURCE_DIR
#define SEM_LOC_SOURCE_DIR "."
#endif

namespace particle_weight_vis {

inline std::string DefaultOutputDir()
{
    return std::string(SEM_LOC_SOURCE_DIR) + "/output/particle_filter_debug/pf_weight_vis";
}

inline bool MakeDirRecursive(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    std::string current;
    current.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        current.push_back(path[i]);
        if (path[i] != '/' && i + 1 != path.size()) {
            continue;
        }
        if (current.empty() || current == "/") {
            continue;
        }
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

inline cv::Scalar WeightToBgr(double weight_norm)
{
    double t = std::max(0.0, std::min(1.0, weight_norm));
    return cv::Scalar(255.0 * (1.0 - t), 0.0, 255.0 * t);
}

inline cv::Mat ToBgr8(const cv::Mat& image)
{
    if (image.empty()) {
        return cv::Mat();
    }
    cv::Mat u8;
    if (image.depth() == CV_8U) {
        u8 = image;
    } else {
        image.convertTo(u8, CV_8U);
    }

    cv::Mat bgr;
    if (u8.channels() == 1) {
        cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
    } else if (u8.channels() == 3) {
        bgr = u8.clone();
    } else if (u8.channels() == 4) {
        cv::cvtColor(u8, bgr, cv::COLOR_BGRA2BGR);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(u8, channels);
        std::vector<cv::Mat> first_three = {channels[0], channels[1], channels[2]};
        cv::merge(first_three, bgr);
    }
    return bgr;
}

inline double Clamp01(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

inline void PutTextBounded(cv::Mat& image,
                           const std::string& text,
                           cv::Point origin,
                           double font_scale = 0.42,
                           int thickness = 1,
                           cv::Scalar color = cv::Scalar(255, 255, 255))
{
    if (origin.y < 0 || origin.y >= image.rows) {
        return;
    }

    std::string clipped = text;
    int baseline = 0;
    cv::Size size = cv::getTextSize(
        clipped, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    int max_width = image.cols - origin.x - 8;
    while (!clipped.empty() && size.width > max_width) {
        clipped.pop_back();
        size = cv::getTextSize(
            clipped + "...", cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    }
    if (clipped != text) {
        clipped += "...";
    }

    cv::putText(image, clipped, origin, cv::FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv::LINE_AA);
}

inline void DrawInfoPanel(cv::Mat& image,
                          int frame_id,
                          double neff,
                          int total_particles,
                          double weight_min,
                          double weight_mean,
                          double weight_max,
                          double weight_std)
{
    const int panel_h = std::min(96, std::max(74, image.rows / 5));
    cv::Mat roi = image(cv::Rect(0, 0, image.cols, panel_h));
    cv::Mat panel(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(panel, 0.72, roi, 0.28, 0.0, roi);

    char line[192];
    std::snprintf(line, sizeof(line), "PF weights | frame %d", frame_id);
    PutTextBounded(image, line, cv::Point(10, 20), 0.48, 1);

    std::snprintf(line, sizeof(line), "Neff %.1f / N %d", neff, total_particles);
    PutTextBounded(image, line, cv::Point(10, 43), 0.44, 1);

    std::snprintf(line, sizeof(line), "Score(weight) min %.6f  mean %.6f  max %.6f",
                  weight_min, weight_mean, weight_max);
    PutTextBounded(image, line, cv::Point(10, 66), 0.39, 1);

    std::snprintf(line, sizeof(line), "Score(weight) std %.6f  range %.6f",
                  weight_std, weight_max - weight_min);
    PutTextBounded(image, line, cv::Point(10, 88), 0.39, 1);
}

inline void DrawLegend(cv::Mat& image, double weight_min, double weight_max)
{
    int h = image.rows;
    int w = image.cols;
    int bar_h = std::min(140, std::max(72, h / 4));
    int bar_w = 14;
    int x0 = std::max(8, w - 46);
    int y0 = std::min(h - bar_h - 34, 110);
    y0 = std::max(104, y0);

    cv::Rect bg_rect(std::max(0, x0 - 78), std::max(0, y0 - 28),
                     std::min(w - std::max(0, x0 - 78), 112),
                     std::min(h - std::max(0, y0 - 28), bar_h + 58));
    if (bg_rect.area() > 0) {
        cv::Mat roi = image(bg_rect);
        cv::Mat panel(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
        cv::addWeighted(panel, 0.60, roi, 0.40, 0.0, roi);
    }

    for (int i = 0; i < bar_h; ++i) {
        double t = 1.0 - static_cast<double>(i) / std::max(1, bar_h - 1);
        cv::line(image, cv::Point(x0, y0 + i), cv::Point(x0 + bar_w, y0 + i), WeightToBgr(t), 1);
    }

    cv::rectangle(image, cv::Rect(x0, y0, bar_w, bar_h), cv::Scalar(255, 255, 255), 1);
    char text[128];
    std::snprintf(text, sizeof(text), "high %.6f", weight_max);
    PutTextBounded(image, text, cv::Point(std::max(2, x0 - 74), std::max(14, y0 - 8)), 0.34, 1);
    std::snprintf(text, sizeof(text), "low %.6f", weight_min);
    PutTextBounded(image, text, cv::Point(std::max(2, x0 - 70), y0 + bar_h + 18), 0.34, 1);
}

template <typename ParticleT>
cv::Mat BuildParticleWeightView(const cv::Mat& satellite_map,
                                CoordConverter& converter,
                                const std::vector<ParticleT>& particles,
                                const cv::Point2f& center_pixel,
                                int frame_id,
                                double neff,
                                int crop_size = 512)
{
    cv::Mat background = ToBgr8(satellite_map);
    if (background.empty()) {
        return cv::Mat();
    }

    crop_size = std::max(128, crop_size);
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 0.5 * crop_size - center_pixel.x,
                         0.0, 1.0, 0.5 * crop_size - center_pixel.y);
    cv::Mat image;
    cv::warpAffine(background, image, transform, cv::Size(crop_size, crop_size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    double weight_min = 0.0;
    double weight_max = 0.0;
    double weight_mean = 0.0;
    double weight_std = 0.0;
    if (!particles.empty()) {
        weight_min = particles.front().weight;
        weight_max = particles.front().weight;
        for (const auto& p : particles) {
            weight_min = std::min(weight_min, p.weight);
            weight_max = std::max(weight_max, p.weight);
            weight_mean += p.weight;
        }
        weight_mean /= static_cast<double>(particles.size());
        for (const auto& p : particles) {
            double diff = p.weight - weight_mean;
            weight_std += diff * diff;
        }
        weight_std = std::sqrt(weight_std / static_cast<double>(particles.size()));
    }
    double span = weight_max - weight_min;
    if (std::abs(span) < 1e-12) {
        span = 1.0;
    }

    cv::Mat overlay = image.clone();
    int best_idx = -1;
    double best_weight = -1.0;
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        const auto& p = particles[i];
        cv::Point2f full_px = converter.gauss_to_pixel(p.x, p.y);
        cv::Point2f local_px(full_px.x + 0.5f * crop_size - center_pixel.x,
                             full_px.y + 0.5f * crop_size - center_pixel.y);
        if (local_px.x < 0.0f || local_px.y < 0.0f ||
            local_px.x >= crop_size || local_px.y >= crop_size) {
            continue;
        }

        double t = Clamp01((p.weight - weight_min) / span);
        int radius = static_cast<int>(std::round(2.0 + 4.0 * t));
        cv::circle(overlay, local_px, radius, WeightToBgr(t), -1, cv::LINE_AA);
        if (p.weight > best_weight) {
            best_weight = p.weight;
            best_idx = i;
        }
    }
    cv::addWeighted(overlay, 0.78, image, 0.22, 0.0, image);

    if (best_idx >= 0) {
        cv::Point2f best_full = converter.gauss_to_pixel(particles[best_idx].x, particles[best_idx].y);
        cv::Point2f best_local(best_full.x + 0.5f * crop_size - center_pixel.x,
                               best_full.y + 0.5f * crop_size - center_pixel.y);
        if (best_local.x >= 0.0f && best_local.y >= 0.0f &&
            best_local.x < crop_size && best_local.y < crop_size) {
            cv::circle(image, best_local, 8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

    cv::Point center(static_cast<int>(std::round(0.5 * crop_size)),
                     static_cast<int>(std::round(0.5 * crop_size)));
    cv::drawMarker(image, center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 16, 2, cv::LINE_AA);

    DrawInfoPanel(image, frame_id, neff, static_cast<int>(particles.size()),
                  weight_min, weight_mean, weight_max, weight_std);
    DrawLegend(image, weight_min, weight_max);
    return image;
}

template <typename ParticleT>
void SaveAndShowParticleWeightView(const cv::Mat& satellite_map,
                                   CoordConverter& converter,
                                   const std::vector<ParticleT>& particles,
                                   const cv::Point2f& center_pixel,
                                   int frame_id,
                                   double neff,
                                   const std::string& out_dir = DefaultOutputDir(),
                                   int crop_size = 512,
                                   bool show_window = true)
{
    cv::Mat view = BuildParticleWeightView(
        satellite_map, converter, particles, center_pixel, frame_id, neff, crop_size);
    if (view.empty()) {
        return;
    }

    if (MakeDirRecursive(out_dir)) {
        char filename[64];
        std::snprintf(filename, sizeof(filename), "pf_weight_%06d.png", frame_id);
        cv::imwrite(out_dir + "/" + filename, view);
    }

    if (show_window) {
        cv::namedWindow("PF Particle Weights", cv::WINDOW_NORMAL);
        cv::imshow("PF Particle Weights", view);
    }
}

} // namespace particle_weight_vis

#endif // PARTICLE_WEIGHT_VISUALIZATION_H
