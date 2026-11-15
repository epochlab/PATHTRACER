#pragma once

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace tools {

// body(i) for every i < count on min(threads, count) workers pulling from one counter: uneven rows balance, and each writes its own slot.
template <typename Body>
void parallelFor(int count, int threads, Body body) {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < std::min(threads, count); ++w) {
        workers.emplace_back([&] {
            for (int i = next++; i < count; i = next++) {
                body(i);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

// Every hardware thread, for the offline generators ctest never runs and so never budgets.
inline int hardwareThreads() { return static_cast<int>(std::max(1U, std::thread::hardware_concurrency())); }

}  // namespace tools
