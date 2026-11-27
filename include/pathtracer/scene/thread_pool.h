#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pathtracer::scene {

// Persistent worker-thread pool: spawning and joining hardware_concurrency() threads per pass is a measurable share of the frame.
class ThreadPool {
public:
    // One worker per hardware thread, floored at 1. Named rather than a default argument so the spec block reports the number used.
    [[nodiscard]] static unsigned int defaultThreadCount() {
        return std::max(1U, std::thread::hardware_concurrency());
    }

    // threadCount is floored at 1: a pool of zero workers would let parallelFor return having run nothing, silently.
    explicit ThreadPool(unsigned int threadCount = defaultThreadCount());
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Runs fn(i) over [0, count), unspecified order; not reentrant, fn must be concurrency-safe, an exception escaping fn is rethrown here.
    void parallelFor(int count, const std::function<void(int)>& fn);

    // Worker count, for callers partitioning into per-worker buckets rather than one index per output element.
    [[nodiscard]] unsigned int threadCount() const { return static_cast<unsigned int>(workers_.size()); }

    // The last parallelFor's summed worker CPU time over threadCount x its wall time: the share of the pool's cores it kept busy.
    [[nodiscard]] double lastCoreUse() const { return lastCoreUse_; }

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable dispatchCv_;
    std::condition_variable doneCv_;

    // Bumped by parallelFor to hand off a dispatch; each worker remembers the last epoch it acted on, so a wait cannot miss it.
    std::uint64_t epoch_ = 0;
    bool shuttingDown_ = false;

    const std::function<void(int)>* fn_ = nullptr;
    std::atomic<int> nextIndex_{0};
    int count_ = 0;
    unsigned int workersRemaining_ = 0;
    // First exception escaping fn this dispatch, held under mutex_ and rethrown by parallelFor once every worker has decremented.
    std::exception_ptr firstException_;
    // Worker CPU time this dispatch, summed under mutex_ at each worker's decrement; read by parallelFor once all have decremented.
    std::int64_t dispatchCpuNs_ = 0;
    double lastCoreUse_ = 0.0;
};

}  // namespace pathtracer::scene
