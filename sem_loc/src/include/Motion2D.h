#pragma once

#include <cmath>

// Planar relative odometry, aligned to the map once at initialization.
// Relative odometry translation belongs to the PREVIOUS body frame.
namespace motion2d {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

inline double Normalize(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

struct Translation {
    double x, y;
};

inline Translation Rotate(double x, double y, double heading) {
    const double c = std::cos(heading), s = std::sin(heading);
    return {c * x - s * y, s * x + c * y};
}

inline Translation BodyDelta(double prev_x, double prev_y, double prev_yaw,
                             double x, double y) {
    return Rotate(x - prev_x, y - prev_y, -prev_yaw);
}

inline double OdomYaw(double x, double y, double z, double w) {
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

struct Step {
    Translation body;
    Translation global;
    double previous_heading;
    double heading;
};

// GNSS supplies the initial map heading; subsequent rotation comes only from odometry.
class OdomMotion {
public:
    void Initialize(double x, double y, double yaw,
                    double map_heading) {
        x_ = x; y_ = y; yaw_ = yaw;
        heading_ = Normalize(map_heading);
    }

    Step Advance(double x, double y, double yaw) {
        const Translation body = BodyDelta(x_, y_, yaw_, x, y);
        const double previous_heading = heading_;
        heading_ = Normalize(previous_heading + Normalize(yaw - yaw_));
        const Translation global = Rotate(body.x, body.y, previous_heading);
        x_ = x; y_ = y; yaw_ = yaw;
        return {body, global, previous_heading, heading_};
    }

private:
    double x_ = 0.0, y_ = 0.0, yaw_ = 0.0;
    double heading_ = 0.0;
};
}  // namespace motion2d
