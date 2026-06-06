#ifndef CUDA_SCOPED_TIMER_H
#define CUDA_SCOPED_TIMER_H

#include <torch/torch.h>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <mutex>

/**
 * @brief 基于 RAII 的 CUDA 精确作用域计时器
 *
 * 构造和析构时均强制调用 torch::cuda::synchronize()，
 * 确保测量的是真实 GPU 计算耗时而非异步下发延迟。
 *
 * 用法 1 - 函数级测量:
 *   void Foo() {
 *       CudaScopedTimer t("Foo");
 *       // ... GPU 计算 ...
 *   }
 *
 * 用法 2 - 子阶段测量:
 *   torch::Tensor result;
 *   {
 *       CudaScopedTimer t("Foo/GridSample", false);
 *       result = torch::nn::functional::grid_sample(...);
 *   }
 *
 * 用法 3 - 周期性汇总:
 *   if (frame % 100 == 0) CudaScopedTimer::PrintSummary();
 */
class CudaScopedTimer {
public:
    explicit CudaScopedTimer(const std::string& label, bool print_on_destroy = true)
        : label_(label), print_(print_on_destroy)
    {
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~CudaScopedTimer() {
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();

        {
            std::lock_guard<std::mutex> lock(GetMutex());
            auto& s = GetStatsMap()[label_];
            s.total_ms += ms;
            s.count++;
            if (ms < s.min_ms) s.min_ms = ms;
            if (ms > s.max_ms) s.max_ms = ms;
        }

        if (print_) {
            std::cout << "[GPU Timer] " << label_ << ": "
                      << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        }
    }

    CudaScopedTimer(const CudaScopedTimer&) = delete;
    CudaScopedTimer& operator=(const CudaScopedTimer&) = delete;

    struct Stats {
        double total_ms = 0.0;
        double min_ms   = 1e9;
        double max_ms   = 0.0;
        uint64_t count  = 0;

        double avg_ms() const { return count > 0 ? total_ms / count : 0.0; }
    };

    static void PrintSummary() {
        std::lock_guard<std::mutex> lock(GetMutex());
        const auto& m = GetStatsMap();
        if (m.empty()) return;

        std::cout << "\n========== GPU Profiling Summary ==========\n";
        std::cout << std::left  << std::setw(32) << "Label"
                  << std::right << std::setw(8)  << "Count"
                  << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Min(ms)"
                  << std::setw(10) << "Max(ms)"
                  << std::setw(12) << "Total(ms)" << "\n";
        std::cout << std::string(82, '-') << "\n";

        for (const auto& kv : m) {
            const auto& s = kv.second;
            std::cout << std::left  << std::setw(32) << kv.first
                      << std::right << std::setw(8)  << s.count
                      << std::fixed << std::setprecision(2)
                      << std::setw(10) << s.avg_ms()
                      << std::setw(10) << (s.min_ms < 1e8 ? s.min_ms : 0.0)
                      << std::setw(10) << s.max_ms
                      << std::setw(12) << s.total_ms << "\n";
        }
        std::cout << "============================================\n" << std::endl;
    }

    static void ResetStats() {
        std::lock_guard<std::mutex> lock(GetMutex());
        GetStatsMap().clear();
    }

private:
    std::string label_;
    bool print_;
    std::chrono::high_resolution_clock::time_point start_;

    static std::unordered_map<std::string, Stats>& GetStatsMap() {
        static std::unordered_map<std::string, Stats> s;
        return s;
    }

    static std::mutex& GetMutex() {
        static std::mutex m;
        return m;
    }
};

#endif // CUDA_SCOPED_TIMER_H
