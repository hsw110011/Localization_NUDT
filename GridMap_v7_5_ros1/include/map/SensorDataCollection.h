#ifndef SENSORDATACOLLECTION_H
#define SENSORDATACOLLECTION_H

#include <deque>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "threadqueue.h"
#include "ars548_msg/DetectionList.h"
#include "ars548_msg/detections.h"

// 时间对数据结构
template<typename dataType>
struct TimestampedData {
    std::shared_ptr<dataType> data;
    int64_t timestamp = 0;  // 时间戳，单位为毫秒
    size_t sensor_type = 0; //传感器类别，1指localpose, 2指lidar, 3指M1, 4指BP, 5指lidar_odemetry
};

// 传感器数据结构
template<typename dataType>
class SensorDataQueue{
public:
    SensorDataQueue() {}
    ~SensorDataQueue() {}
    //添加数据
    void addData(const dataType& data, const int64_t &timestamp,const int64_t &sensor_type) {
        TimestampedData<dataType> timestampedData;
        timestampedData.data = std::make_shared<dataType>(data);
        timestampedData.timestamp = timestamp;
        timestampedData.sensor_type = sensor_type;
        auto timestampedDataPtr = std::make_shared<TimestampedData<dataType>>(std::move(timestampedData));
        {
            std::unique_lock<std::mutex> lock(this->mutex_);
            dqueue_.push_back(timestampedDataPtr);

            // 如果队列长度超过限制，移除最旧的数据
            size_t my_queue_length_ = data_queue_length_;
            if(sensor_type == 1){
                my_queue_length_ = local_pose_queue_length_;
            }
            while (dqueue_.size() > my_queue_length_) {
                dqueue_.pop_front();
            }
        }
        this->cv_.notify_one();
    }
    //根据最近时间戳返回数据
    std::shared_ptr<const TimestampedData<dataType>> getDatabyTimestampNearest(const int64_t &timestamp) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        auto nearestIterator = std::begin(dqueue_);
        auto nearestData = *nearestIterator;
        int nearestTimeDiff = std::abs(nearestData->timestamp - timestamp);
        //从第二个元素开始遍历队列
        for (auto iter = std::next(std::begin(dqueue_)); iter != std::end(dqueue_); ++iter) {
            const auto& currentData = *iter;
            int currentDiff = std::abs(currentData->timestamp - timestamp);
            if (currentDiff < nearestTimeDiff) {
                nearestTimeDiff = currentDiff;
                nearestData = currentData;
            }
        }
        return nearestData;
    }

    bool isQueueEmpty(){
        std::lock_guard<std::mutex> lock(this->mutex_);
        return dqueue_.empty();
    }

    size_t Size() {
        std::lock_guard<std::mutex> lock(this->mutex_);
        return dqueue_.size();
    }

    std::shared_ptr<const TimestampedData<dataType>> getbackdata() {
        std::lock_guard<std::mutex> lock(this->mutex_);
//        if(dqueue_.size() > 1) {
//            dqueue_.pop_front();
//        }
        return dqueue_.back();
    }
    std::shared_ptr<const TimestampedData<dataType>> getfrontdata() {
        std::lock_guard<std::mutex> lock(this->mutex_);
//        if(dqueue_.size() > 1) {
//            dqueue_.pop_front();
//        }
        return dqueue_.front();
    }
    void popbackdata(){
        std::lock_guard<std::mutex> lock(this->mutex_);
        if(!dqueue_.empty())
            dqueue_.pop_back();
    }
    void popfrontdata(){
        std::lock_guard<std::mutex> lock(this->mutex_);
        if(!dqueue_.empty())
            dqueue_.pop_front();
    }
    void cleardata(){
        std::lock_guard<std::mutex> lock(this->mutex_);
        if(!dqueue_.empty())
            dqueue_.clear();
    }
    std::shared_ptr<TimestampedData<dataType>> waitForData(int timeout_ms) {
        std::unique_lock<std::mutex> lock(this->mutex_);
        if (dqueue_.empty()) {
            if (timeout_ms == 0) {
                throw std::runtime_error("Sensor Data queue is empty.");
            } else {
                if (this->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this](){ return !dqueue_.empty(); })) {
                    auto ptr = dqueue_.front();
                    dqueue_.pop_front();
                    return ptr;
                } else {
                    throw std::runtime_error("Sensor Data queue is empty.");
                }
            }
        } else {
            auto ptr = dqueue_.front();
            dqueue_.pop_front();
            return ptr;
        }
    }

private:
    std::deque<std::shared_ptr<TimestampedData<dataType>>> dqueue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t data_queue_length_ = 1;  //队列长度
    size_t local_pose_queue_length_ = 20;  //localpose队列长度
};



#endif // SENSORDATACOLLECTION_H
