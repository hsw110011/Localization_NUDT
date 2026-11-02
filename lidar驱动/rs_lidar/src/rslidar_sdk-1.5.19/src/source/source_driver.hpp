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

#include "source/source.hpp"

#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/utility/sync_queue.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>

namespace robosense
{
namespace lidar
{

class SourceDriver : public Source
{
public:

  virtual void init(const YAML::Node& config);
  virtual void start();
  virtual void stop();
  virtual void regPacketCallback(DestinationPacket::Ptr dst);
  virtual ~SourceDriver();

  SourceDriver(SourceType src_type);

protected:

  std::shared_ptr<LidarPointCloudMsg> getPointCloud(void);
  void putPointCloud(std::shared_ptr<LidarPointCloudMsg> msg);
  void putPacket(const Packet& msg);
  void putException(const lidar::Error& msg);
  void processPointCloud();
  void processPacketStats();

  std::shared_ptr<lidar::LidarDriver<LidarPointCloudMsg>> driver_ptr_;
  SyncQueue<std::shared_ptr<LidarPointCloudMsg>> free_point_cloud_queue_;
  SyncQueue<std::shared_ptr<LidarPointCloudMsg>> point_cloud_queue_;
#ifdef ENABLE_IMU_DATA_PARSE
  std::shared_ptr<ImuData> getImuData(void);
  void putImuData(const std::shared_ptr<ImuData>& msg);
  void processImuData();
  SyncQueue<std::shared_ptr<ImuData>> free_imu_data_queue_;
  SyncQueue<std::shared_ptr<ImuData>> imu_data_queue_;
  std::thread imu_data_process_thread_;
#endif
  std::thread point_cloud_process_thread_;
  std::thread packet_stats_thread_;
  std::atomic<bool> to_exit_process_;
  std::atomic<uint64_t> valid_packet_count_;
  double packet_stats_interval_sec_;
};

SourceDriver::SourceDriver(SourceType src_type)
  : Source(src_type), to_exit_process_(false), valid_packet_count_(0), packet_stats_interval_sec_(1.0)
{
}

inline void SourceDriver::init(const YAML::Node& config)
{
  YAML::Node driver_config = yamlSubNodeAbort(config, "driver");
  lidar::RSDriverParam driver_param;

  yamlRead<double>(config["ros"], "packet_counter_interval", packet_stats_interval_sec_, 1.0);

  // input related
  yamlRead<uint16_t>(driver_config, "msop_port", driver_param.input_param.msop_port, 6699);
  yamlRead<uint16_t>(driver_config, "difop_port", driver_param.input_param.difop_port, 7788);
#ifdef ENABLE_IMU_DATA_PARSE
  yamlRead<uint16_t>(driver_config, "imu_port", driver_param.input_param.imu_port, 6688);
#endif
  yamlRead<std::string>(driver_config, "host_address", driver_param.input_param.host_address, "0.0.0.0");
  yamlRead<std::string>(driver_config, "group_address", driver_param.input_param.group_address, "0.0.0.0");
  yamlRead<bool>(driver_config, "use_vlan", driver_param.input_param.use_vlan, false);
  yamlRead<std::string>(driver_config, "pcap_path", driver_param.input_param.pcap_path, "");
  yamlRead<float>(driver_config, "pcap_rate", driver_param.input_param.pcap_rate, 1);
  yamlRead<bool>(driver_config, "pcap_repeat", driver_param.input_param.pcap_repeat, true);
  yamlRead<uint16_t>(driver_config, "user_layer_bytes", driver_param.input_param.user_layer_bytes, 0);
  yamlRead<uint16_t>(driver_config, "tail_layer_bytes", driver_param.input_param.tail_layer_bytes, 0);
  yamlRead<uint32_t>(driver_config, "socket_recv_buf", driver_param.input_param.socket_recv_buf, 106496);
  // decoder related
  std::string lidar_type;
  yamlReadAbort<std::string>(driver_config, "lidar_type", lidar_type);
  driver_param.lidar_type = strToLidarType(lidar_type);

  // decoder
  yamlRead<bool>(driver_config, "wait_for_difop", driver_param.decoder_param.wait_for_difop, true);
  yamlRead<bool>(driver_config, "use_lidar_clock", driver_param.decoder_param.use_lidar_clock, false);
  yamlRead<float>(driver_config, "min_distance", driver_param.decoder_param.min_distance, 0.2);
  yamlRead<float>(driver_config, "max_distance", driver_param.decoder_param.max_distance, 200);
  yamlRead<float>(driver_config, "start_angle", driver_param.decoder_param.start_angle, 0);
  yamlRead<float>(driver_config, "end_angle", driver_param.decoder_param.end_angle, 360);
  yamlRead<bool>(driver_config, "dense_points", driver_param.decoder_param.dense_points, false);
  yamlRead<bool>(driver_config, "ts_first_point", driver_param.decoder_param.ts_first_point, false);

  // mechanical decoder
  yamlRead<bool>(driver_config, "config_from_file", driver_param.decoder_param.config_from_file, false);
  yamlRead<std::string>(driver_config, "angle_path", driver_param.decoder_param.angle_path, "");

  uint16_t split_frame_mode;
  yamlRead<uint16_t>(driver_config, "split_frame_mode", split_frame_mode, 1);
  driver_param.decoder_param.split_frame_mode = SplitFrameMode(split_frame_mode);

  yamlRead<float>(driver_config, "split_angle", driver_param.decoder_param.split_angle, 0);
  yamlRead<uint16_t>(driver_config, "num_blks_split", driver_param.decoder_param.num_blks_split, 0);

  // transform
  yamlRead<float>(driver_config, "x", driver_param.decoder_param.transform_param.x, 0);
  yamlRead<float>(driver_config, "y", driver_param.decoder_param.transform_param.y, 0);
  yamlRead<float>(driver_config, "z", driver_param.decoder_param.transform_param.z, 0);
  yamlRead<float>(driver_config, "roll", driver_param.decoder_param.transform_param.roll, 0);
  yamlRead<float>(driver_config, "pitch", driver_param.decoder_param.transform_param.pitch, 0);
  yamlRead<float>(driver_config, "yaw", driver_param.decoder_param.transform_param.yaw, 0);

  switch (src_type_)
  {
    case SourceType::MSG_FROM_LIDAR:
      driver_param.input_type = InputType::ONLINE_LIDAR;
      break;
    case SourceType::MSG_FROM_PCAP:
      driver_param.input_type = InputType::PCAP_FILE;
      break;
    default:
      driver_param.input_type = InputType::RAW_PACKET;
      break;
  }

  driver_param.print();

  driver_ptr_.reset(new lidar::LidarDriver<LidarPointCloudMsg>());
  driver_ptr_->regPointCloudCallback(std::bind(&SourceDriver::getPointCloud, this), 
      std::bind(&SourceDriver::putPointCloud, this, std::placeholders::_1));
  driver_ptr_->regExceptionCallback(
      std::bind(&SourceDriver::putException, this, std::placeholders::_1));
  point_cloud_process_thread_ = std::thread(std::bind(&SourceDriver::processPointCloud, this));

#ifdef ENABLE_IMU_DATA_PARSE
  driver_ptr_->regImuDataCallback(std::bind(&SourceDriver::getImuData, this),std::bind(&SourceDriver::putImuData, this, std::placeholders::_1));
  imu_data_process_thread_ = std::thread(std::bind(&SourceDriver::processImuData, this));
#endif

  if (!driver_ptr_->init(driver_param))
  {
    RS_ERROR << "Driver Initialize Error...." << RS_REND;
    exit(-1);
  }
}

inline void SourceDriver::start()
{
  for (const auto& dst : pkt_cb_vec_)
  {
    if (dst)
    {
      dst->start();
    }
  }
  driver_ptr_->start();
  if ((src_type_ == SourceType::MSG_FROM_LIDAR) && (packet_stats_interval_sec_ > 0.0) &&
      !packet_stats_thread_.joinable())
  {
    RS_INFO << "[PacketStats] realtime counter started, interval="
            << packet_stats_interval_sec_ << " second(s)" << RS_REND;
    packet_stats_thread_ = std::thread(std::bind(&SourceDriver::processPacketStats, this));
  }
}

inline SourceDriver::~SourceDriver()
{
  stop();
}

inline void SourceDriver::stop()
{
  driver_ptr_->stop();
  for (const auto& dst : pkt_cb_vec_)
  {
    if (dst)
    {
      dst->stop();
    }
  }

  to_exit_process_.store(true, std::memory_order_relaxed);
  if (packet_stats_thread_.joinable())
  {
    packet_stats_thread_.join();
  }
  if (point_cloud_process_thread_.joinable())
  {
    point_cloud_process_thread_.join();
  }
#ifdef ENABLE_IMU_DATA_PARSE
  if (imu_data_process_thread_.joinable())
  {
    imu_data_process_thread_.join();
  }
#endif
}

inline std::shared_ptr<LidarPointCloudMsg> SourceDriver::getPointCloud(void)
{
  std::shared_ptr<LidarPointCloudMsg> point_cloud = free_point_cloud_queue_.pop();
  if (point_cloud.get() != NULL)
  {
    return point_cloud;
  }

  return std::make_shared<LidarPointCloudMsg>();
}

inline void SourceDriver::regPacketCallback(DestinationPacket::Ptr dst)
{
  Source::regPacketCallback(dst);
  if (pkt_cb_vec_.size() == 1)
  {
    driver_ptr_->regPacketCallback(
        std::bind(&SourceDriver::putPacket, this, std::placeholders::_1));
  }
}

inline void SourceDriver::putPacket(const Packet& msg)
{
  valid_packet_count_.fetch_add(1, std::memory_order_relaxed);
  sendPacket(msg);
}

void SourceDriver::putPointCloud(std::shared_ptr<LidarPointCloudMsg> msg)
{
  point_cloud_queue_.push(msg);
}
#ifdef ENABLE_IMU_DATA_PARSE
inline std::shared_ptr<ImuData> SourceDriver::getImuData(void)
{
  std::shared_ptr<ImuData> imuDataPtr = free_imu_data_queue_.pop();
  if (imuDataPtr.get() != NULL)
  {
    return imuDataPtr;
  }
  return std::make_shared<ImuData>();
}
void SourceDriver::putImuData(const std::shared_ptr<ImuData>& msg)
{
  imu_data_queue_.push(msg);
}

void SourceDriver::processImuData()
{
  while (!to_exit_process_.load(std::memory_order_relaxed))
  {
    std::shared_ptr<ImuData> msg = imu_data_queue_.popWait(100);
    if (msg.get() == NULL)
    {
      continue;
    }
    sendImuData(msg);

    free_imu_data_queue_.push(msg);
  }
}
#endif
void SourceDriver::processPointCloud()
{
  while (!to_exit_process_.load(std::memory_order_relaxed))
  {
    std::shared_ptr<LidarPointCloudMsg> msg = point_cloud_queue_.popWait(1000);
    if (msg.get() == NULL)
    {
      continue;
    }
    sendPointCloud(msg);
    
    free_point_cloud_queue_.push(msg);
  }
}

inline void SourceDriver::processPacketStats()
{
  uint64_t previous_udp = 0;
  uint64_t previous_valid = 0;
  uint64_t previous_ros = 0;
  uint64_t previous_ros_messages = 0;

  while (!to_exit_process_.load(std::memory_order_relaxed))
  {
    std::this_thread::sleep_for(std::chrono::duration<double>(packet_stats_interval_sec_));
    if (to_exit_process_.load(std::memory_order_relaxed))
    {
      break;
    }

    uint64_t ros_sent = 0;
    size_t subscriber_count = 0;
    size_t batch_pending = 0;
    uint64_t ros_message_count = 0;
    bool batching_enabled = false;
    std::string destination;
    bool has_ros_counter = false;
    for (const auto& dst : pkt_cb_vec_)
    {
      if (dst && dst->getPacketSentCount(ros_sent))
      {
        destination = dst->getPacketDestinationName();
        subscriber_count = dst->getPacketSubscriberCount();
        batching_enabled = dst->isPacketBatchingEnabled();
        batch_pending = dst->getPacketPendingCount();
        ros_message_count = dst->getPublishedMessageCount();
        has_ros_counter = true;
        break;
      }
    }
    const uint64_t udp_received = driver_ptr_->getUdpPacketCount();
    const uint64_t valid_packets = valid_packet_count_.load(std::memory_order_relaxed);

    if (!has_ros_counter)
    {
      // Keep reporting UDP activity instead of silently printing nothing. 
      // This normally means common.send_packet_ros is false or no ROS destination was registered for this source.
      RS_WARNING << "[PacketStats] udp_rx=" << udp_received
                 << " (+" << (udp_received - previous_udp) << ")"
                 << " valid_msop_difop=" << valid_packets
                 << " (+" << (valid_packets - previous_valid) << ")"
                 << " ros1_pub=0 (+0) ros1_enabled=false"
                 << " status=ROS_PUBLISH_DISABLED" << RS_REND;

      previous_udp = udp_received;
      previous_valid = valid_packets;
      previous_ros = 0;
      previous_ros_messages = 0;
      continue;
    }

    const uint64_t input_pending = (udp_received > valid_packets) ? (udp_received - valid_packets) : 0;
    const uint64_t publish_pending = (valid_packets > ros_sent) ? (valid_packets - ros_sent) : 0;

    RS_INFO << "[PacketStats] topic=" << destination
            << " udp_rx=" << udp_received << " (+" << (udp_received - previous_udp) << ")"
            << " valid_msop_difop=" << valid_packets << " (+" << (valid_packets - previous_valid) << ")"
            << " ros1_pub_packets=" << ros_sent << " (+" << (ros_sent - previous_ros) << ")"
            << " ros1_messages=" << ros_message_count
            << " (+" << (ros_message_count - previous_ros_messages) << ")"
            << " batching=" << (batching_enabled ? "true" : "false")
            << " batch_queue=" << batch_pending
            << " input_pending_or_filtered=" << input_pending
            << " publish_pending=" << publish_pending
            << " subscribers=" << subscriber_count
            << " status=" << ((subscriber_count == 0) ? "NO_SUBSCRIBER" :
                              (batching_enabled ? "BATCHING_OK" :
                               ((publish_pending == 0) ? "OK" : "CHECK"))) << RS_REND;

    previous_udp = udp_received;
    previous_valid = valid_packets;
    previous_ros = ros_sent;
    previous_ros_messages = ros_message_count;
  }
}

inline void SourceDriver::putException(const lidar::Error& msg)
{
  switch (msg.error_code_type)
  {
    case lidar::ErrCodeType::INFO_CODE:
      RS_INFO << msg.toString() << RS_REND;
      break;
    case lidar::ErrCodeType::WARNING_CODE:
      RS_WARNING << msg.toString() << RS_REND;
      break;
    case lidar::ErrCodeType::ERROR_CODE:
      RS_ERROR << msg.toString() << RS_REND;
      break;
  }
}

}  // namespace lidar
}  // namespace robosense
