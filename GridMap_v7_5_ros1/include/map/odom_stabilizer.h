#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <deque>
#include <map>
#include <optional>
#include <vector>

struct Pose6D {
    double timestamp = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
};

struct DeltaPose6D {
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double droll = 0.0;
    double dpitch = 0.0;
    double dyaw = 0.0;
};

class OdomStabilizer {
public:
    struct Params {
        std::size_t median_window_size = 5;
        std::size_t max_pose_buffer_size = 2000;

        double max_v = 15.0;
        double max_z_rate = 2.0;
        double max_roll_rate = 1.3962634015954636;   // 80 deg/s.
        double max_pitch_rate = 1.3962634015954636;  // 80 deg/s.
        double max_yaw_rate = 2.0943951023931953;    // 120 deg/s.

        bool enable_low_pass = false;
        double low_pass_alpha = 0.4;
        double max_update_dt = 1.0;

        double history_max_dt = 0.5;
        double history_max_ds = 3.0;
        double history_max_dz = 0.5;
        double history_max_roll = 0.03490658503988659;   // 2 deg.
        double history_max_pitch = 0.03490658503988659;  // 2 deg.
        double history_max_yaw = 0.08726646259971647;    // 5 deg.

        double gradient_max_dt = 0.2;
        double gradient_max_ds = 1.0;
        double gradient_max_dz = 0.3;
        double gradient_max_roll = 0.017453292519943295;   // 1 deg.
        double gradient_max_pitch = 0.017453292519943295;  // 1 deg.
        double gradient_max_yaw = 0.03490658503988659;     // 2 deg.
    };

    OdomStabilizer();
    explicit OdomStabilizer(const Params& params);

    void setParams(const Params& params);
    const Params& getParams() const { return params_; }
    void reset();

    Pose6D AddRawPose(const Pose6D& raw_pose);
    std::optional<Pose6D> GetSmoothedPose(double timestamp) const;
    std::optional<Eigen::Matrix4d> GetRelativeTransform(double history_timestamp,
                                                        double current_timestamp) const;
    bool IsHistoryFrameValid(double history_timestamp, double current_timestamp) const;
    bool IsHistoryFrameValidForGradient(double history_timestamp,
                                        double current_timestamp) const;

    static double degToRad(double degrees);
    static double wrapAngle(double angle);
    static Pose6D NormalizePoseAngles(const Pose6D& pose);
    static DeltaPose6D ComputeDeltaPose(const Pose6D& prev_pose, const Pose6D& curr_pose);
    static DeltaPose6D LimitDeltaPose(const DeltaPose6D& delta,
                                      double dt,
                                      const Params& params);
    static Pose6D MedianFilterPoseBuffer(const std::deque<Pose6D>& buffer,
                                         std::size_t window_size = 5);
    static Pose6D LowPassPose(const Pose6D& raw_pose,
                              const Pose6D& prev_filtered_pose,
                              double alpha = 0.4);
    static Eigen::Matrix4d PoseToMatrix(const Pose6D& pose);
    static Eigen::Matrix4d GetRelativeTransform(const Pose6D& history_smoothed_pose,
                                                const Pose6D& current_smoothed_pose);
    static double ComputeZBias(double history_ground_h0, double current_ground_h0);

private:
    Params params_;
    std::deque<Pose6D> raw_pose_buffer_;
    std::map<double, Pose6D> smoothed_pose_by_time_;
    std::optional<Pose6D> prev_smoothed_pose_;

    Pose6D UpdateSmoothedPose(const Pose6D& raw_pose);
    static Pose6D ApplyDeltaPose(const Pose6D& pose, const DeltaPose6D& delta, double timestamp);
    static double median(std::vector<double> values);
    static double unwrapAngleNearReference(double angle, double reference);
    static std::optional<Pose6D> lookupPose(const std::map<double, Pose6D>& poses,
                                            double timestamp);
    bool isHistoryFrameValid(const Pose6D& history_pose,
                             const Pose6D& current_pose,
                             double max_dt,
                             double max_ds,
                             double max_dz,
                             double max_roll,
                             double max_pitch,
                             double max_yaw) const;
};
