#include "pathtracer/scene/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <utility>

namespace pathtracer::scene {

namespace {

// CPU time of the calling thread: time on a core, so a worker preempted mid-dispatch is not counted as using one.
std::int64_t threadCpuNs() {
    timespec now{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
    return (static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000) + now.tv_nsec;
}

}  // namespace

ThreadPool::ThreadPool(unsigned int threadCount) {
    const unsigned int count = std::max(1U, threadCount);
    workers_.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        shuttingDown_ = true;
        ++epoch_;
    }
    dispatchCv_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::parallelFor(int count, const std::function<void(int)>& fn) {
    if (count <= 0) {
        return;
    }
    const auto dispatchStart = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    fn_ = &fn;
    count_ = count;
    nextIndex_.store(0, std::memory_order_relaxed);
    workersRemaining_ = static_cast<unsigned int>(workers_.size());
    dispatchCpuNs_ = 0;
    ++epoch_;
    lock.unlock();
    dispatchCv_.notify_all();

    lock.lock();
    doneCv_.wait(lock, [this] { return workersRemaining_ == 0; });
    fn_ = nullptr;
    const std::chrono::duration<double, std::nano> wall = std::chrono::steady_clock::now() - dispatchStart;
    lastCoreUse_ = static_cast<double>(dispatchCpuNs_) / (static_cast<double>(workers_.size()) * wall.count());
    // Rethrown on the caller's thread, after every worker is accounted for: the pass is invalid, and swallowing it would publish it.
    if (firstException_) {
        std::rethrow_exception(std::exchange(firstException_, nullptr));
    }
}

void ThreadPool::workerLoop() {
    std::uint64_t lastEpoch = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        dispatchCv_.wait(lock, [this, lastEpoch] { return epoch_ != lastEpoch; });
        lastEpoch = epoch_;
        if (shuttingDown_) {
            return;
        }
        const std::function<void(int)>* fn = fn_;
        const int count = count_;
        lock.unlock();
        const std::int64_t cpuStart = threadCpuNs();

        // A throwing task must still reach the decrement below, which parallelFor waits on -- an escape would hang the caller forever.
        try {
            int index = 0;
            while ((index = nextIndex_.fetch_add(1, std::memory_order_relaxed)) < count) {
                (*fn)(index);
            }
        } catch (...) {
            const std::lock_guard<std::mutex> guard(mutex_);
            if (!firstException_) {
                firstException_ = std::current_exception();
            }
        }

        const std::int64_t cpuNs = threadCpuNs() - cpuStart;
        lock.lock();
        dispatchCpuNs_ += cpuNs;
        --workersRemaining_;
        const bool isLast = workersRemaining_ == 0;
        lock.unlock();
        if (isLast) {
            doneCv_.notify_one();
        }
    }
}

}  // namespace pathtracer::scene
