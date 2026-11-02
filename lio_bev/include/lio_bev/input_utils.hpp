#pragma once

#include <Eigen/Geometry>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace lio_bev {

struct PointXYZ {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// No aligned pointer casts: input field offsets and row padding are arbitrary.
inline void readXYZ(const sensor_msgs::PointCloud2& msg, std::vector<PointXYZ>& out)
{
  out.clear();
  if (msg.width == 0 || msg.height == 0) return;
  if (msg.point_step == 0 ||
      std::uint64_t(msg.width) * msg.point_step > msg.row_step ||
      std::uint64_t(msg.row_step) * msg.height > msg.data.size()) {
    throw std::invalid_argument("invalid PointCloud2 row/point layout");
  }
  struct Field { std::size_t offset; std::size_t size; } fields[3];
  const char* names[] = {"x", "y", "z"};
  for (int i = 0; i < 3; ++i) {
    const auto it = std::find_if(msg.fields.begin(), msg.fields.end(),
      [&](const sensor_msgs::PointField& f) { return f.name == names[i]; });
    if (it == msg.fields.end() || it->count != 1 ||
        (it->datatype != sensor_msgs::PointField::FLOAT32 &&
         it->datatype != sensor_msgs::PointField::FLOAT64)) {
      throw std::invalid_argument("PointCloud2 requires scalar float x/y/z fields");
    }
    const std::size_t size = it->datatype == sensor_msgs::PointField::FLOAT32 ? 4 : 8;
    if (std::uint64_t(it->offset) + size > msg.point_step) {
      throw std::invalid_argument("PointCloud2 field exceeds point_step");
    }
    fields[i] = {it->offset, size};
  }
  const std::uint16_t one = 1;
  const bool host_big_endian = *reinterpret_cast<const std::uint8_t*>(&one) == 0;
  const bool swap = host_big_endian != msg.is_bigendian;
  out.reserve(std::size_t(msg.width) * msg.height);
  for (std::size_t row = 0; row < msg.height; ++row) {
    const auto* base = msg.data.data() + row * msg.row_step;
    for (std::size_t col = 0; col < msg.width; ++col) {
      float values[3];
      for (int axis = 0; axis < 3; ++axis) {
        const auto& f = fields[axis];
        std::uint8_t bytes[8];
        std::memcpy(bytes, base + col * msg.point_step + f.offset, f.size);
        if (swap) std::reverse(bytes, bytes + f.size);
        if (f.size == 4) {
          std::memcpy(&values[axis], bytes, 4);
        } else {
          double value;
          std::memcpy(&value, bytes, 8);
          values[axis] = std::isfinite(value) &&
              std::abs(value) <= std::numeric_limits<float>::max()
            ? static_cast<float>(value) : std::numeric_limits<float>::quiet_NaN();
        }
      }
      out.push_back({values[0], values[1], values[2]});
    }
  }
}

struct PoseSample {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  ros::Time stamp;
  std::string frame_id;
  bool valid = false;
};

// Caller owns synchronization. Bounded by both sample count and elapsed time.
class PoseBuffer {
public:
  void clear() { samples_.clear(); }
  void insert(const PoseSample& sample)
  {
    if (!sample.valid || sample.stamp.isZero() || !sample.pose.matrix().allFinite()) return;
    if (!samples_.empty() &&
        (sample.frame_id != samples_.back().frame_id ||
         (samples_.back().stamp - sample.stamp).toSec() > 0.5)) {
      clear();  // New frame or rosbag rewind.
    }
    auto it = std::lower_bound(samples_.begin(), samples_.end(), sample.stamp,
      [](const PoseSample& p, const ros::Time& t) { return p.stamp < t; });
    if (it != samples_.end() && it->stamp == sample.stamp) *it = sample;
    else samples_.insert(it, sample);
    while (samples_.size() > 400 ||
           (samples_.size() > 1 &&
            (samples_.back().stamp - samples_.front().stamp).toSec() > 3.0)) {
      samples_.pop_front();
    }
  }

  bool sample(const ros::Time& stamp, double max_gap, double nearest_tolerance,
              bool allow_nearest, PoseSample& result) const
  {
    if (samples_.empty() || stamp.isZero()) return false;
    auto upper = std::lower_bound(samples_.begin(), samples_.end(), stamp,
      [](const PoseSample& p, const ros::Time& t) { return p.stamp < t; });
    if (upper != samples_.end() && upper->stamp == stamp) {
      result = *upper;
      return true;
    }
    if (upper != samples_.begin() && upper != samples_.end()) {
      const auto& a = *std::prev(upper);
      const auto& b = *upper;
      const double gap = (b.stamp - a.stamp).toSec();
      if (gap <= max_gap && a.frame_id == b.frame_id) {
        const double alpha = (stamp - a.stamp).toSec() / gap;
        result = a;
        result.stamp = stamp;
        result.pose.translation() =
          (1.0 - alpha) * a.pose.translation() + alpha * b.pose.translation();
        result.pose.linear() = Eigen::Quaterniond(a.pose.linear()).slerp(
          alpha, Eigen::Quaterniond(b.pose.linear())).normalized().toRotationMatrix();
        return true;
      }
    }
    if (!allow_nearest) return false;
    const PoseSample* nearest = upper == samples_.end() ? &samples_.back() : &*upper;
    if (upper != samples_.begin()) {
      const auto& before = *std::prev(upper);
      if (std::abs((before.stamp - stamp).toSec()) <
          std::abs((nearest->stamp - stamp).toSec())) nearest = &before;
    }
    if (std::abs((nearest->stamp - stamp).toSec()) > nearest_tolerance) return false;
    result = *nearest;
    result.stamp = stamp;
    return true;
  }

private:
  std::deque<PoseSample> samples_;
};

}  // namespace lio_bev
