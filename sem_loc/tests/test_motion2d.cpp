#include "Motion2D.h"
#include <gtsam/geometry/Pose2.h>
#include <iostream>
#include <random>
#include <stdexcept>

void close(double actual, double expected, double tolerance = 1e-8) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("motion composition differs from GTSAM");
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> a(-3.14, 3.14), d(-10., 10.);
    for (int i = 0; i < 1000; ++i) {
        const double x = 700000. + d(rng), y = 3173000. + d(rng);
        const double yaw = a(rng), nx = x + d(rng), ny = y + d(rng), nyaw = a(rng);
        const double map_heading = a(rng);
        motion2d::OdomMotion motion;
        motion.Initialize(x, y, yaw, map_heading);
        const auto step = motion.Advance(nx, ny, nyaw);
        const auto relative = gtsam::Pose2(x, y, yaw).between(gtsam::Pose2(nx, ny, nyaw));
        close(step.body.x, relative.x());
        close(step.body.y, relative.y());
        const auto expected = gtsam::Pose2(0., 0., map_heading).compose(relative);
        close(step.global.x, expected.x());
        close(step.global.y, expected.y());
        close(motion2d::Normalize(step.heading - expected.theta()), 0.);
        close(motion2d::Normalize(step.heading - (map_heading + nyaw - yaw)), 0.);

        // Changing the arbitrary LIO world origin/orientation cannot alter motion.
        const double beta = a(rng);
        const auto p = motion2d::Rotate(x, y, beta), np = motion2d::Rotate(nx, ny, beta);
        motion2d::OdomMotion transformed;
        transformed.Initialize(p.x + 100., p.y - 200., yaw + beta, map_heading);
        const auto same = transformed.Advance(np.x + 100., np.y - 200., nyaw + beta);
        close(same.global.x, step.global.x);
        close(same.global.y, step.global.y);
    }
    // A 90-degree odometry turn changes the next interval's translation direction.
    motion2d::OdomMotion motion;
    motion.Initialize(0., 0., 0., motion2d::kPi / 2.);
    const auto first = motion.Advance(1., 0., motion2d::kPi / 2.);
    const auto second = motion.Advance(1., 1., motion2d::kPi);
    close(first.global.x, 0.); close(first.global.y, 1.);
    close(second.global.x, -1.); close(second.global.y, 0.);
    // Quaternion yaw wraps at +/-pi; only the small physical turn is propagated.
    motion.Initialize(0., 0., 179. * motion2d::kDegToRad, 20. * motion2d::kDegToRad);
    const auto wrapped = motion.Advance(0., 0., -179. * motion2d::kDegToRad);
    close(wrapped.heading, 22. * motion2d::kDegToRad);
    close(wrapped.global.x, 0.); close(wrapped.global.y, 0.);
    std::cout << "1000 GTSAM relative/compose cases, frame invariance and heading timing: OK\n";
}
