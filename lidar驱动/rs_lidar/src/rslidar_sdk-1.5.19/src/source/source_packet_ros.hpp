/*********************************************************************************************************************
Copyright (c) 2020 RoboSense
All rights reserved

By downloading, copying, installing or using the software you agree to this license. If you do not agree to this
license, do not download, install, copy or use the software.

License Agreement
For RoboSense LiDAR SDK Library
(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the names of the RoboSense, nor Suteng Innovation Technology, nor the names of other contributors may be used
to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********************************************************************************************************************/

#pragma once

#include "source/source_driver.hpp"

#ifdef ROS_FOUND

#ifdef ENABLE_SOURCE_PACKET_LEGACY
#include "msg/ros_msg/rslidar_packet_legacy.hpp"
#include "msg/ros_msg/rslidar_scan_legacy.hpp"
#endif

#include <rslidar_msg/RslidarPacket.h>
#include <rslidar_msg/RslidarPacketArray.h>
#include <ros/ros.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace robosense
{
namespace lidar
{

inline Packet toRsMsg(const rslidar_msg::RslidarPacket& ros_msg)
{
  Packet rs_msg;
  rs_msg.timestamp = ros_msg.header.stamp.toSec();
  rs_msg.seq = ros_msg.header.seq;
  rs_msg.is_difop = ros_msg.is_difop;
  rs_msg.is_frame_begin = ros_msg.is_frame_begin; 

  for (size_t i = 0; i < ros_msg.data.size(); i++)
  {
    rs_msg.buf_.emplace_back(ros_msg.data[i]);
  }

  return rs_msg;
}

class SourcePacketRos : public SourceDriver
{ 
public: 

  virtual void init(const YAML::Node& config);

  SourcePacketRos();

private:

#ifdef ENABLE_SOURCE_PACKET_LEGACY
  void putPacketLegacy(const rslidar_msgs::rslidarScan& msg);
  void putPacketDifopLegacy(const rslidar_msgs::rslidarPacket& msg);
  ros::Subscriber pkt_sub_legacy_;
  ros::Subscriber pkt_sub_difop_legacy_;
  LidarType lidar_type_;
#endif

  void putPacket(const rslidar_msg::RslidarPacket& msg);
  void putPacketArray(const rslidar_msg::RslidarPacketArray& msg);
  ros::Subscriber pkt_sub_;

  std::unique_ptr<ros::NodeHandle> nh_;
};

SourcePacketRos::SourcePacketRos()
  : SourceDriver(SourceType::MSG_FROM_ROS_PACKET)
{
}

void SourcePacketRos::init(const YAML::Node& config)
{
  SourceDriver::init(config);

  nh_ = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());

#ifdef ENABLE_SOURCE_PACKET_LEGACY
  std::string lidar_type;
  yamlReadAbort<std::string>(config["driver"], "lidar_type", lidar_type);
  lidar_type_ = strToLidarType(lidar_type);

  std::string ros_recv_topic_legacy;
  yamlRead<std::string>(config["ros"], "ros_recv_packet_legacy_topic", 
      ros_recv_topic_legacy, "rslidar_packets");

  pkt_sub_legacy_ = nh_->subscribe(ros_recv_topic_legacy, 1, &SourcePacketRos::putPacketLegacy, this);
  pkt_sub_difop_legacy_ = nh_->subscribe(ros_recv_topic_legacy + "_difop", 1, &SourcePacketRos::putPacketDifopLegacy, this);
#endif

  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_recv_packet_topic", ros_recv_topic, "rslidar_packets");

  bool packet_batching_enabled = false;
  yamlRead<bool>(config["ros"], "packet_batching_enabled", packet_batching_enabled, false);
  if (packet_batching_enabled)
  {
    pkt_sub_ = nh_->subscribe(ros_recv_topic, 10, &SourcePacketRos::putPacketArray, this);
    RS_INFO << "Receive packet arrays from ROS topic: " << ros_recv_topic << RS_REND;
  }
  else
  {
    pkt_sub_ = nh_->subscribe(ros_recv_topic, 100, &SourcePacketRos::putPacket, this);
  }

}

#ifdef ENABLE_SOURCE_PACKET_LEGACY
inline Packet toRsMsg(const rslidar_msgs::rslidarPacket& ros_msg, LidarType lidar_type, bool isDifop)
{
  size_t pkt_len = 1248;
  if (lidar_type == LidarType::RSM1)
  {
    pkt_len = isDifop ? 256 : 1210;
  }

  Packet rs_msg;
  for (size_t i = 0; i < pkt_len; i++)
  {
    rs_msg.buf_.emplace_back(ros_msg.data[i]);
  }

  return rs_msg;
}

void SourcePacketRos::putPacketLegacy(const rslidar_msgs::rslidarScan& msg)
{
  for (uint32_t i = 0; i < msg.packets.size(); i++)
  {
    const rslidar_msgs::rslidarPacket& packet = msg.packets[i];
    driver_ptr_->decodePacket(toRsMsg(packet, lidar_type_, false));
  }
}

void SourcePacketRos::putPacketDifopLegacy(const rslidar_msgs::rslidarPacket& msg)
{
  driver_ptr_->decodePacket(toRsMsg(msg, lidar_type_, true));
}
#endif

void SourcePacketRos::putPacket(const rslidar_msg::RslidarPacket& msg)
{
  driver_ptr_->decodePacket(toRsMsg(msg));
}

// 批量消息直接重新输入驱动进行点云解析
void SourcePacketRos::putPacketArray(const rslidar_msg::RslidarPacketArray& msg)
{
  for (const auto& packet : msg.packets)
  {
    driver_ptr_->decodePacket(toRsMsg(packet));
  }
}

inline rslidar_msg::RslidarPacket toRosMsg(const Packet& rs_msg, const std::string& frame_id)
{
  rslidar_msg::RslidarPacket ros_msg;
  ros_msg.header.stamp = ros_msg.header.stamp.fromSec(rs_msg.timestamp);
  ros_msg.header.seq = rs_msg.seq;
  ros_msg.header.frame_id = frame_id;
  ros_msg.is_difop = rs_msg.is_difop;
  ros_msg.is_frame_begin = rs_msg.is_frame_begin;

  for (size_t i = 0; i < rs_msg.buf_.size(); i++)
  {
    ros_msg.data.emplace_back(rs_msg.buf_[i]);
  }

  return ros_msg;
}

class DestinationPacketRos : public DestinationPacket
{
public:

  virtual void init(const YAML::Node& config);
  virtual void start() override;
  virtual void stop() override;
  virtual void sendPacket(const Packet& msg);
  virtual ~DestinationPacketRos();

  inline uint64_t getRosPacketSentCount() const;
  virtual bool getPacketSentCount(uint64_t& count) const override;
  virtual std::string getPacketDestinationName() const override;
  virtual size_t getPacketSubscriberCount() const override;
  virtual bool isPacketBatchingEnabled() const override;
  virtual size_t getPacketPendingCount() const override;
  virtual uint64_t getPublishedMessageCount() const override;

private:

  void processPacketBatch();
  void publishPacketBatch(std::vector<rslidar_msg::RslidarPacket>& packets);

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher pkt_pub_;
  ros::Publisher pkt_array_pub_;
  std::string frame_id_;
  std::string ros_send_topic_;
  bool packet_batching_enabled_{false};
  double packet_batch_publish_frequency_{10.0};
  size_t packet_batch_max_packets_{10000};
  mutable std::mutex packet_batch_mutex_;
  std::condition_variable packet_batch_cv_;
  std::vector<rslidar_msg::RslidarPacket> packet_batch_;
  std::thread packet_batch_thread_;
  bool packet_batch_exit_{false};
  bool packet_batch_started_{false};
  std::atomic<uint64_t> ros_packet_sent_count_{0};
  std::atomic<uint64_t> ros_message_sent_count_{0};
  std::atomic<uint32_t> ros_batch_seq_{0};
};

inline void DestinationPacketRos::init(const YAML::Node& config)
{
  yamlRead<std::string>(config["ros"],
      "ros_frame_id", frame_id_, "rslidar");

  yamlRead<std::string>(config["ros"], "ros_send_packet_topic",
      ros_send_topic_, "rslidar_packets");
  yamlRead<bool>(config["ros"], "packet_batching_enabled", packet_batching_enabled_, false);
  yamlRead<double>(config["ros"], "packet_batch_publish_frequency", packet_batch_publish_frequency_, 10.0);
  uint32_t batch_max_packets = 10000;
  yamlRead<uint32_t>(config["ros"], "packet_batch_max_packets", batch_max_packets, 10000);
  packet_batch_max_packets_ = static_cast<size_t>(batch_max_packets);

  if (packet_batch_publish_frequency_ <= 0.0)
  {
    RS_WARNING << "packet_batch_publish_frequency must be > 0; using 10 Hz" << RS_REND;
    packet_batch_publish_frequency_ = 10.0;
  }
  if (packet_batch_max_packets_ == 0)
  {
    packet_batch_max_packets_ = 10000;
  }

  nh_ = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
  if (packet_batching_enabled_)
  {
    pkt_array_pub_ = nh_->advertise<rslidar_msg::RslidarPacketArray>(ros_send_topic_, 10);
    packet_batch_.reserve(packet_batch_max_packets_);
    RS_INFO << "Packet batching enabled: topic=" << ros_send_topic_
            << ", frequency=" << packet_batch_publish_frequency_ << " Hz"
            << ", max_packets=" << packet_batch_max_packets_ << RS_REND;
  }
  else
  {
    pkt_pub_ = nh_->advertise<rslidar_msg::RslidarPacket>(ros_send_topic_, 100);
  }
}

inline void DestinationPacketRos::start()
{
  if (!packet_batching_enabled_)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(packet_batch_mutex_);
  if (!packet_batch_started_)
  {
    packet_batch_exit_ = false;
    packet_batch_started_ = true;
    packet_batch_thread_ = std::thread(&DestinationPacketRos::processPacketBatch, this);
  }
}

inline void DestinationPacketRos::stop()
{
  if (!packet_batching_enabled_)
  {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(packet_batch_mutex_);
    if (!packet_batch_started_)
    {
      return;
    }
    packet_batch_exit_ = true;
  }
  packet_batch_cv_.notify_all();
  if (packet_batch_thread_.joinable())
  {
    packet_batch_thread_.join();
  }

  std::vector<rslidar_msg::RslidarPacket> remaining;
  {
    std::lock_guard<std::mutex> lock(packet_batch_mutex_);
    remaining.swap(packet_batch_);
    packet_batch_started_ = false;
  }
  publishPacketBatch(remaining);
}

inline void DestinationPacketRos::sendPacket(const Packet& msg)
{
  rslidar_msg::RslidarPacket ros_msg = toRosMsg(msg, frame_id_);
  if (!packet_batching_enabled_)
  {
    pkt_pub_.publish(ros_msg);
    ros_packet_sent_count_.fetch_add(1, std::memory_order_relaxed);
    ros_message_sent_count_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  std::vector<rslidar_msg::RslidarPacket> full_batch;
  {
    std::lock_guard<std::mutex> lock(packet_batch_mutex_);
    packet_batch_.emplace_back(std::move(ros_msg));
    if (packet_batch_.size() >= packet_batch_max_packets_)
    {
      full_batch.swap(packet_batch_);
      packet_batch_.reserve(packet_batch_max_packets_);
    }
  }

  // The size limit is a memory-safety fallback. 
  // Under normal load the timer, publishes at packet_batch_publish_frequency instead.
  publishPacketBatch(full_batch);
}

inline void DestinationPacketRos::processPacketBatch()
{
  const std::chrono::duration<double> period(1.0 / packet_batch_publish_frequency_);
  while (true)
  {
    std::vector<rslidar_msg::RslidarPacket> packets;
    {
      std::unique_lock<std::mutex> lock(packet_batch_mutex_);
      packet_batch_cv_.wait_for(lock, period, [this]() { return packet_batch_exit_; });
      if (packet_batch_exit_)
      {
        break;
      }
      packets.swap(packet_batch_);
      packet_batch_.reserve(packet_batch_max_packets_);
    }
    publishPacketBatch(packets);
  }
}

inline void DestinationPacketRos::publishPacketBatch(
    std::vector<rslidar_msg::RslidarPacket>& packets)
{
  if (packets.empty())
  {
    return;
  }

  rslidar_msg::RslidarPacketArray array_msg;
  array_msg.header.seq = ros_batch_seq_.fetch_add(1, std::memory_order_relaxed);
  array_msg.header.stamp = ros::Time::now();
  array_msg.header.frame_id = frame_id_;
  array_msg.packets.swap(packets);
  const uint64_t packet_count = static_cast<uint64_t>(array_msg.packets.size());

  pkt_array_pub_.publish(array_msg);
  ros_packet_sent_count_.fetch_add(packet_count, std::memory_order_relaxed);
  ros_message_sent_count_.fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t DestinationPacketRos::getRosPacketSentCount() const
{
  return ros_packet_sent_count_.load(std::memory_order_relaxed);
}

inline bool DestinationPacketRos::getPacketSentCount(uint64_t& count) const
{
  count = getRosPacketSentCount();
  return true;
}

inline std::string DestinationPacketRos::getPacketDestinationName() const
{
  return ros_send_topic_;
}

inline size_t DestinationPacketRos::getPacketSubscriberCount() const
{
  return packet_batching_enabled_ ? pkt_array_pub_.getNumSubscribers() : pkt_pub_.getNumSubscribers();
}

inline bool DestinationPacketRos::isPacketBatchingEnabled() const
{
  return packet_batching_enabled_;
}

inline size_t DestinationPacketRos::getPacketPendingCount() const
{
  if (!packet_batching_enabled_)
  {
    return 0;
  }
  std::lock_guard<std::mutex> lock(packet_batch_mutex_);
  return packet_batch_.size();
}

inline uint64_t DestinationPacketRos::getPublishedMessageCount() const
{
  return ros_message_sent_count_.load(std::memory_order_relaxed);
}

inline DestinationPacketRos::~DestinationPacketRos()
{
  stop();
  RS_INFO << "[PacketCounter][Final] UDP packets converted to ROS1 packets and published on "
          << ros_send_topic_ << ": packets=" << getRosPacketSentCount()
          << ", ros_messages=" << getPublishedMessageCount() << RS_REND;
}

}  // namespace lidar
}  // namespace robosense

#endif  // ROS_FOUND

#ifdef ROS2_FOUND
#include "rslidar_msg/msg/rslidar_packet.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sstream>

namespace robosense
{
namespace lidar
{

inline Packet toRsMsg(const rslidar_msg::msg::RslidarPacket& ros_msg)
{
  Packet rs_msg;
  rs_msg.timestamp = ros_msg.header.stamp.sec + double(ros_msg.header.stamp.nanosec) / 1e9;
  //rs_msg.seq = ros_msg.header.seq;
  rs_msg.is_difop = ros_msg.is_difop;
  rs_msg.is_frame_begin = ros_msg.is_frame_begin; 

  for (size_t i = 0; i < ros_msg.data.size(); i++)
  {
    rs_msg.buf_.emplace_back(ros_msg.data[i]);
  }

  return rs_msg;
}

class SourcePacketRos : public SourceDriver
{ 
public: 

  virtual void init(const YAML::Node& config);

  SourcePacketRos();

private:
  void spin();
  void putPacket(const rslidar_msg::msg::RslidarPacket::SharedPtr msg) const;
  std::thread subscription_spin_thread_;
  std::shared_ptr<rclcpp::Node> node_ptr_;
  rclcpp::Subscription<rslidar_msg::msg::RslidarPacket>::SharedPtr pkt_sub_;
};

SourcePacketRos::SourcePacketRos()
  : SourceDriver(SourceType::MSG_FROM_ROS_PACKET)
{
}
void SourcePacketRos::spin(){rclcpp::spin(node_ptr_);}  
void SourcePacketRos::init(const YAML::Node& config)
{
  SourceDriver::init(config);

  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_recv_packet_topic", 
      ros_recv_topic, "rslidar_packets");

  static int node_index = 0;
  std::stringstream node_name;
  node_name << "rslidar_packets_source_" << node_index++;

  node_ptr_.reset(new rclcpp::Node(node_name.str()));
  pkt_sub_ = node_ptr_->create_subscription<rslidar_msg::msg::RslidarPacket>(ros_recv_topic, 100, 
      std::bind(&SourcePacketRos::putPacket, this, std::placeholders::_1));
  subscription_spin_thread_ = std::thread(std::bind(&SourcePacketRos::spin,this));
} 

void SourcePacketRos::putPacket(const rslidar_msg::msg::RslidarPacket::SharedPtr msg) const
{
  driver_ptr_->decodePacket(toRsMsg(*msg));
}

inline rslidar_msg::msg::RslidarPacket toRosMsg(const Packet& rs_msg, const std::string& frame_id)
{
  rslidar_msg::msg::RslidarPacket ros_msg;
  ros_msg.header.stamp.sec = (uint32_t)floor(rs_msg.timestamp);
  ros_msg.header.stamp.nanosec = (uint32_t)round((rs_msg.timestamp - ros_msg.header.stamp.sec) * 1e9);
  //ros_msg.header.seq = rs_msg.seq;
  ros_msg.header.frame_id = frame_id;
  ros_msg.is_difop = rs_msg.is_difop;
  ros_msg.is_frame_begin = rs_msg.is_frame_begin;

  for (size_t i = 0; i < rs_msg.buf_.size(); i++)
  {
    ros_msg.data.emplace_back(rs_msg.buf_[i]);
  }

  return ros_msg;
}

class DestinationPacketRos : public DestinationPacket
{
public:

  virtual void init(const YAML::Node& config);
  virtual void sendPacket(const Packet& msg);
  virtual ~DestinationPacketRos() = default;

private:

  std::shared_ptr<rclcpp::Node> node_ptr_;
  rclcpp::Publisher<rslidar_msg::msg::RslidarPacket>::SharedPtr pkt_pub_;
  std::string frame_id_;
};

inline void DestinationPacketRos::init(const YAML::Node& config)
{
  yamlRead<std::string>(config["ros"], 
      "ros_frame_id", frame_id_, "rslidar");

  std::string ros_send_topic;
  yamlRead<std::string>(config["ros"], "ros_send_packet_topic", 
      ros_send_topic, "rslidar_packets");

  size_t ros_queue_length;
  yamlRead<size_t>(config["ros"], "ros_queue_length", ros_queue_length, 100);

  static int node_index = 0;
  std::stringstream node_name;
  node_name << "rslidar_packets_destination_" << node_index++;

  node_ptr_.reset(new rclcpp::Node(node_name.str()));
  pkt_pub_ = node_ptr_->create_publisher<rslidar_msg::msg::RslidarPacket>(ros_send_topic, ros_queue_length);
}

inline void DestinationPacketRos::sendPacket(const Packet& msg)
{
  pkt_pub_->publish(toRosMsg(msg, frame_id_));
}

}  // namespace lidar
}  // namespace robosense

#endif  // ROS2_FOUND


