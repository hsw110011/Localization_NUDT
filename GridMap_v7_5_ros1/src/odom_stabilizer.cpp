#include "map/odom_stabilizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
}

OdomStabilizer::OdomStabilizer()
    : OdomStabilizer(Params())
{
}

OdomStabilizer::OdomStabilizer(const Params& params)
{
    setParams(params);
}

void OdomStabilizer::setParams(const Params& params)
{
    params_ = params;
    params_.median_window_size = std::max<std::size_t>(1, params_.median_window_size);
    params_.max_pose_buffer_size = std::max<std::size_t>(1, params_.max_pose_buffer_size);
    params_.max_v = std::max(0.0, params_.max_v);
    params_.max_z_rate = std::max(0.0, params_.max_z_rate);
    params_.max_roll_rate = std::max(0.0, params_.max_roll_rate);
    params_.max_pitch_rate = std::max(0.0, params_.max_pitch_rate);
    params_.max_yaw_rate = std::max(0.0, params_.max_yaw_rate);
    params_.low_pass_alpha = std::clamp(params_.low_pass_alpha, 0.0, 1.0);
    params_.max_update_dt = std::max(1e-3, params_.max_update_dt);
}

void OdomStabilizer::reset()
{
    raw_pose_buffer_.clear();
    smoothed_pose_by_time_.clear();
    prev_smoothed_pose_.reset();
}

Pose6D OdomStabilizer::AddRawPose(const Pose6D& raw_pose)
{
    return UpdateSmoothedPose(raw_pose);
}

std::optional<Pose6D> OdomStabilizer::GetSmoothedPose(double timestamp) const
{
    return lookupPose(smoothed_pose_by_time_, timestamp);
}

std::optional<Eigen::Matrix4d> OdomStabilizer::GetRelativeTransform(
    double history_timestamp,
    double current_timestamp) const
{
    const auto history_pose = GetSmoothedPose(history_timestamp);
    const auto current_pose = GetSmoothedPose(current_timestamp);
    if (!history_pose || !current_pose) {
        return std::nullopt;
    }
    return GetRelativeTransform(*history_pose, *current_pose);
}

bool OdomStabilizer::IsHistoryFrameValid(double history_timestamp,
                                         double current_timestamp) const
{
    const auto history_pose = GetSmoothedPose(history_timestamp);
    const auto current_pose = GetSmoothedPose(current_timestamp);
    if (!history_pose || !current_pose) {
        return false;
    }

    return isHistoryFrameValid(*history_pose,
                               *current_pose,
                               params_.history_max_dt,
                               params_.history_max_ds,
                               params_.history_max_dz,
                               params_.history_max_roll,
                               params_.history_max_pitch,
                               params_.history_max_yaw);
}

bool OdomStabilizer::IsHistoryFrameValidForGradient(double history_timestamp,
                                                    double current_timestamp) const
{
    const auto history_pose = GetSmoothedPose(history_timestamp);
    const auto current_pose = GetSmoothedPose(current_timestamp);
    if (!history_pose || !current_pose) {
        return false;
    }

    return isHistoryFrameValid(*history_pose,
                               *current_pose,
                               params_.gradient_max_dt,
                               params_.gradient_max_ds,
                               params_.gradient_max_dz,
                               params_.gradient_max_roll,
                               params_.gradient_max_pitch,
                               params_.gradient_max_yaw);
}

double OdomStabilizer::degToRad(double degrees)
{
    return degrees * kPi / 180.0;
}

double OdomStabilizer::wrapAngle(double angle)
{
    if (!std::isfinite(angle)) {
        return 0.0;
    }

    double wrapped = std::fmod(angle + kPi, kTwoPi);
    if (wrapped < 0.0) {
        wrapped += kTwoPi;
    }
    return wrapped - kPi;
}

Pose6D OdomStabilizer::NormalizePoseAngles(const Pose6D& pose)
{
    Pose6D normalized = pose;
    normalized.roll = wrapAngle(normalized.roll);
    normalized.pitch = wrapAngle(normalized.pitch);
    normalized.yaw = wrapAngle(normalized.yaw);
    return normalized;
}

DeltaPose6D OdomStabilizer::ComputeDeltaPose(const Pose6D& prev_pose,
                                             const Pose6D& curr_pose)
{
    DeltaPose6D delta;
    delta.dx = curr_pose.x - prev_pose.x;
    delta.dy = curr_pose.y - prev_pose.y;
    delta.dz = curr_pose.z - prev_pose.z;
    delta.droll = wrapAngle(curr_pose.roll - prev_pose.roll);
    delta.dpitch = wrapAngle(curr_pose.pitch - prev_pose.pitch);
    delta.dyaw = wrapAngle(curr_pose.yaw - prev_pose.yaw);
    return delta;
}

DeltaPose6D OdomStabilizer::LimitDeltaPose(const DeltaPose6D& delta,
                                           double dt,
                                           const Params& params)
{
    if (!std::isfinite(dt) || dt <= 0.0) {
        DeltaPose6D zero;
        return zero;
    }

    DeltaPose6D limited = delta;
    const double max_trans = std::max(0.0, params.max_v) * dt;
    const double trans = std::hypot(limited.dx, limited.dy);
    if (trans > max_trans && trans > 1e-9) {
        const double scale = max_trans / trans;
        limited.dx *= scale;
        limited.dy *= scale;
    }

    limited.dz = std::clamp(limited.dz,
                            -std::max(0.0, params.max_z_rate) * dt,
                            std::max(0.0, params.max_z_rate) * dt);
    limited.droll = std::clamp(wrapAngle(limited.droll),
                               -std::max(0.0, params.max_roll_rate) * dt,
                               std::max(0.0, params.max_roll_rate) * dt);
    limited.dpitch = std::clamp(wrapAngle(limited.dpitch),
                                -std::max(0.0, params.max_pitch_rate) * dt,
                                std::max(0.0, params.max_pitch_rate) * dt);
    limited.dyaw = std::clamp(wrapAngle(limited.dyaw),
                              -std::max(0.0, params.max_yaw_rate) * dt,
                              std::max(0.0, params.max_yaw_rate) * dt);
    return limited;
}

Pose6D OdomStabilizer::MedianFilterPoseBuffer(const std::deque<Pose6D>& buffer,
                                              std::size_t window_size)
{
    if (buffer.empty()) {
        return Pose6D();
    }

    window_size = std::max<std::size_t>(1, window_size);
    const std::size_t count = std::min(window_size, buffer.size());
    const auto first = buffer.end() - static_cast<std::ptrdiff_t>(count);
    const Pose6D& latest = buffer.back();

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<double> rolls;
    std::vector<double> pitches;
    std::vector<double> yaws;
    xs.reserve(count);
    ys.reserve(count);
    zs.reserve(count);
    rolls.reserve(count);
    pitches.reserve(count);
    yaws.reserve(count);

    for (auto it = first; it != buffer.end(); ++it) {
        xs.push_back(it->x);
        ys.push_back(it->y);
        zs.push_back(it->z);
        rolls.push_back(unwrapAngleNearReference(it->roll, latest.roll));
        pitches.push_back(unwrapAngleNearReference(it->pitch, latest.pitch));
        yaws.push_back(unwrapAngleNearReference(it->yaw, latest.yaw));
    }

    Pose6D filtered;
    filtered.timestamp = latest.timestamp;
    filtered.x = median(xs);
    filtered.y = median(ys);
    filtered.z = median(zs);
    filtered.roll = wrapAngle(median(rolls));
    filtered.pitch = wrapAngle(median(pitches));
    filtered.yaw = wrapAngle(median(yaws));
    return filtered;
}

Pose6D OdomStabilizer::LowPassPose(const Pose6D& raw_pose,
                                   const Pose6D& prev_filtered_pose,
                                   double alpha)
{
    alpha = std::clamp(alpha, 0.0, 1.0);

    Pose6D filtered;
    filtered.timestamp = raw_pose.timestamp;
    filtered.x = prev_filtered_pose.x + alpha * (raw_pose.x - prev_filtered_pose.x);
    filtered.y = prev_filtered_pose.y + alpha * (raw_pose.y - prev_filtered_pose.y);
    filtered.z = prev_filtered_pose.z + alpha * (raw_pose.z - prev_filtered_pose.z);
    filtered.roll = wrapAngle(prev_filtered_pose.roll +
                              alpha * wrapAngle(raw_pose.roll - prev_filtered_pose.roll));
    filtered.pitch = wrapAngle(prev_filtered_pose.pitch +
                               alpha * wrapAngle(raw_pose.pitch - prev_filtered_pose.pitch));
    filtered.yaw = wrapAngle(prev_filtered_pose.yaw +
                             alpha * wrapAngle(raw_pose.yaw - prev_filtered_pose.yaw));
    return filtered;
}

Eigen::Matrix4d OdomStabilizer::PoseToMatrix(const Pose6D& pose)
{
    const Eigen::AngleAxisd roll_angle(pose.roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch_angle(pose.pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw_angle(pose.yaw, Eigen::Vector3d::UnitZ());

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) =
        (yaw_angle * pitch_angle * roll_angle).toRotationMatrix();
    transform.block<3, 1>(0, 3) = Eigen::Vector3d(pose.x, pose.y, pose.z);
    return transform;
}

Eigen::Matrix4d OdomStabilizer::GetRelativeTransform(const Pose6D& history_smoothed_pose,
                                                     const Pose6D& current_smoothed_pose)
{
    const Eigen::Matrix4d t_world_history = PoseToMatrix(history_smoothed_pose);
    const Eigen::Matrix4d t_world_current = PoseToMatrix(current_smoothed_pose);
    return t_world_current.inverse() * t_world_history;
}

double OdomStabilizer::ComputeZBias(double history_ground_h0, double current_ground_h0)
{
    if (!std::isfinite(history_ground_h0) || !std::isfinite(current_ground_h0)) {
        return 0.0;
    }
    return std::clamp(current_ground_h0 - history_ground_h0, -0.3, 0.3);
}

Pose6D OdomStabilizer::UpdateSmoothedPose(const Pose6D& raw_pose)
{
    const Pose6D normalized_raw = NormalizePoseAngles(raw_pose);
    raw_pose_buffer_.push_back(normalized_raw);
    while (raw_pose_buffer_.size() > params_.median_window_size) {
        raw_pose_buffer_.pop_front();
    }

    const Pose6D median_pose =
        MedianFilterPoseBuffer(raw_pose_buffer_, params_.median_window_size);

    Pose6D curr_smoothed;
    if (!prev_smoothed_pose_) {
        curr_smoothed = median_pose;
    } else {
        double dt = median_pose.timestamp - prev_smoothed_pose_->timestamp;
        if (!std::isfinite(dt) || dt <= 0.0) {
            dt = 0.0;
        } else {
            dt = std::min(dt, params_.max_update_dt);
        }

        const DeltaPose6D delta = ComputeDeltaPose(*prev_smoothed_pose_, median_pose);
        const DeltaPose6D limited_delta = LimitDeltaPose(delta, dt, params_);
        curr_smoothed =
            ApplyDeltaPose(*prev_smoothed_pose_, limited_delta, median_pose.timestamp);

        if (params_.enable_low_pass) {
            curr_smoothed =
                LowPassPose(curr_smoothed, *prev_smoothed_pose_, params_.low_pass_alpha);
        }
    }

    curr_smoothed = NormalizePoseAngles(curr_smoothed);
    prev_smoothed_pose_ = curr_smoothed;
    smoothed_pose_by_time_[curr_smoothed.timestamp] = curr_smoothed;
    while (smoothed_pose_by_time_.size() > params_.max_pose_buffer_size) {
        smoothed_pose_by_time_.erase(smoothed_pose_by_time_.begin());
    }

    return curr_smoothed;
}

Pose6D OdomStabilizer::ApplyDeltaPose(const Pose6D& pose,
                                      const DeltaPose6D& delta,
                                      double timestamp)
{
    Pose6D result;
    result.timestamp = timestamp;
    result.x = pose.x + delta.dx;
    result.y = pose.y + delta.dy;
    result.z = pose.z + delta.dz;
    result.roll = wrapAngle(pose.roll + delta.droll);
    result.pitch = wrapAngle(pose.pitch + delta.dpitch);
    result.yaw = wrapAngle(pose.yaw + delta.dyaw);
    return result;
}

double OdomStabilizer::median(std::vector<double> values)
{
    values.erase(std::remove_if(values.begin(),
                                values.end(),
                                [](double value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double result = values[mid];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(),
                         values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                         values.end());
        result = 0.5 * (result + values[mid - 1]);
    }
    return result;
}

double OdomStabilizer::unwrapAngleNearReference(double angle, double reference)
{
    return reference + wrapAngle(angle - reference);
}

std::optional<Pose6D> OdomStabilizer::lookupPose(const std::map<double, Pose6D>& poses,
                                                 double timestamp)
{
    if (poses.empty()) {
        return std::nullopt;
    }

    auto lower = poses.lower_bound(timestamp);
    if (lower == poses.begin()) {
        return lower->second;
    }
    if (lower == poses.end()) {
        return poses.rbegin()->second;
    }

    auto prev = std::prev(lower);
    const double lower_error = std::abs(lower->first - timestamp);
    const double prev_error = std::abs(prev->first - timestamp);
    return lower_error < prev_error ? lower->second : prev->second;
}

bool OdomStabilizer::isHistoryFrameValid(const Pose6D& history_pose,
                                         const Pose6D& current_pose,
                                         double max_dt,
                                         double max_ds,
                                         double max_dz,
                                         double max_roll,
                                         double max_pitch,
                                         double max_yaw) const
{
    const double dt = current_pose.timestamp - history_pose.timestamp;
    if (!std::isfinite(dt) || dt < 0.0 || dt > max_dt) {
        return false;
    }

    const DeltaPose6D delta = ComputeDeltaPose(history_pose, current_pose);
    const double ds = std::hypot(delta.dx, delta.dy);

    return ds <= max_ds &&
           std::abs(delta.dz) <= max_dz &&
           std::abs(delta.droll) <= max_roll &&
           std::abs(delta.dpitch) <= max_pitch &&
           std::abs(delta.dyaw) <= max_yaw;
}
