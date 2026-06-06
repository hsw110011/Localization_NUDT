//#ifndef THREAD_QUEUE_H
//#define THREAD_QUEUE_H
//#include <chrono>
//#include <ctime>
//#include <queue>
//#include <sstream>
//#include <iostream>
//#include <iomanip>
//#include <thread>
//#include <mutex>
//#include <condition_variable>

//template <typename T>
//struct tData{
//public:
//    std::shared_ptr<T> data;
//    int readCount=0;

//    // 构造函数，用于初始化 data 成员
//    // tData(const T& d) : data(d) {}
//};

//// 线程安全的队列类
//// 如果你的模板参数 T 已经是一个共享指针 std::shared_ptr<SensorDataFrozen>，直接使用 T 作为参数类型是合适的。
//// 你可以将 Push 方法的参数类型更改为 T，而不需要额外的共享指针。
//template <typename T>  //
//class SafeQueue{  // 先进先出
//public:
//    SafeQueue() {}
//    ~SafeQueue() {}

//    void setLength(size_t data_queue_length) { data_queue_length_ = data_queue_length;}
//    void setReadNum(int readNumSet) { readNumSet_ = readNumSet;}

//    // 将数据压入队列
//    void addData(const T& item) {
//    // void addData(std::shared_ptr<T> item) {

//        tData<T> t_data;
//        t_data.data = std::make_shared<T>(item);

//        auto t_data_ptr = std::make_shared<tData<T>>(std::move(t_data));

//        {
//            std::unique_lock<std::mutex> lock(mutex_);
//            // emplace 允许你在队列中直接构造元素，而不是创建一个临时对象并将其复制到队列中。
//            // tData titem(item);
//            // queue_.emplace(item);
//            // queue_.push(std::make_shared<tData>(item));
//            // dqueue_.push_back(std::make_shared<tData>(item));
//            dqueue_.push_back(t_data_ptr);

//            // 如果队列长度超过限制，移除最旧的数据
//            while (dqueue_.size() > data_queue_length_) {
//                dqueue_.pop_front();
//            }
//        }  // 在临界区外释放锁  ==> lock.unlock

//        // 唤醒等待的线程
//        // 它会唤醒一个等待中的线程（如果有的话），并允许其继续执行。
//        // 如果没有等待中的线程，则该函数不会做任何操作。
//        cv_.notify_one();
//        // cv_.notify_all(); // 通知消费者线程有新数据
//    }

//    std::shared_ptr<const T> readFrontData() {
//        std::unique_lock<std::mutex> lock(mutex_);
//        cv_.wait(lock, [this] { return !dqueue_.empty(); });
//        auto item = dqueue_.front();
//        while (item->readCount == readNumSet_) {
//            dqueue_.pop_front(); // 弹出已经被多次读取的元素
//            // 因为用了智能指针才能这么写
//            if (dqueue_.empty()) {
//                // 如果队列为空，则等待数据压入
//                // std::cout<<"start 03----> : " <<std::endl;
//                cv_.wait(lock, [this] { return !dqueue_.empty(); });
//                item = dqueue_.front();
//                // std::cout<<"start 04----> : " <<std::endl;
//            } else {
//                // 如果队列不为空，则继续检查下一个元素
//                item = dqueue_.front();
//            }
//        }
//        item->readCount++;
//        // 将修改后的元素放回队列头部
//        // dqueue_.push_front(item);
//        return item->data;
//    }

//    // T readFrontDataNoWait() {
//    //     std::unique_lock<std::mutex> lock(mutex_);
//    //     auto item = dqueue_.front();
//    //     while (item->readCount == readNumSet_) {
//    //         dqueue_.pop_front(); // 弹出已经被多次读取的元素
//    //         // 因为用了智能指针才能这么写
//    //         if (dqueue_.empty()) {
//    //             // 如果队列为空，则等待数据压入
//    //             // std::cout<<"start 03----> : " <<std::endl;
//    //             cv_.wait(lock, [this] { return !dqueue_.empty(); });
//    //             item = dqueue_.front();
//    //             // std::cout<<"start 04----> : " <<std::endl;
//    //         } else {
//    //             // 如果队列不为空，则继续检查下一个元素
//    //             item = dqueue_.front();
//    //         }
//    //     }
//    //     item->readCount++;
//    //     // 将修改后的元素放回队列头部
//    //     // dqueue_.push_front(item);
//    //     return item->data;
//    // }

//    // 弹出队头数据
//    void popData() {
//        std::unique_lock<std::mutex> lock(mutex_);
//        cv_.wait(lock, [this] { return !dqueue_.empty(); });
//        dqueue_.pop();
//    }

//    // 检查队列是否为空
//    bool isQueueEmpty() const {
//        std::lock_guard<std::mutex> lock(mutex_);
//        return dqueue_.empty();
//    }
//    size_t Size() {
//        std::lock_guard<std::mutex> lock(mutex_);
//        return dqueue_.size();
//    }

//    void CheckQueueSize(){
//        std::lock_guard<std::mutex> lock(mutex_);
//        if(dqueue_.size() > data_queue_length_){
//            Wangjie::Helpers::Log("DataQueue deal slowly : " +std::to_string(dqueue_.size())+ "!");
//        }
//        while(dqueue_.size() > data_queue_length_) {
//            dqueue_.pop_front();
//        }
//    }

//    std::mutex& getDataMutex() {
//        return mutex_;
//    }

//protected: // private:
//    size_t data_queue_length_=1;
//    int readNumSet_=1;
//    // std::queue<T> queue_;
//    // std::queue<std::shared_ptr<tData>> queue_;
//    std::deque<std::shared_ptr<tData<T>>> dqueue_;
//    // std::mutex mutex_;
//    mutable std::mutex mutex_;
//    mutable std::condition_variable cv_;
//};


//class ConditionVariableFlag{
//public:
//    ConditionVariableFlag(){ setZero(); };
//    ~ConditionVariableFlag(){};

//    bool isSynchronized_;

//    void setZero(){
//        isSynchronized_=false;
//    }

//    void waitSyncDone();
//    void waitSyncStart();

//    void setSync(bool isSynchronized);
//    bool getSync();

//private:
//    std::mutex mutex_;
//    std::condition_variable cv_;
//};

//#endif
