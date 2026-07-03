#pragma once

// Minimal persistent thread pool for the CPU backend's parallel loops.
// One process-wide pool of (hardware_concurrency - 1) workers; the calling
// thread participates too, so a P-way split uses P threads with no idle
// caller. parallel_for statically partitions [0, n) and blocks until done.
//
// Deliberately simple (static partitioning, no work stealing): GEMM's M-loop
// blocks are near-equal cost, so a balanced static split is close to optimal
// and avoids per-task queue overhead. A lazy singleton so tensor-free
// binaries never spawn threads.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tl {
namespace cpu {

class thread_pool {
 public:
  static thread_pool& instance() {
    static thread_pool* p = new thread_pool();  // leaked; outlives all callers
    return *p;
  }

  int size() const { return nthreads_; }

  // Run fn(begin, end) over a static partition of [0, n) across `size()`
  // threads (caller runs one share). Blocks until all shares complete.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    int p = static_cast<int>(std::min<int64_t>(nthreads_, n));
    if (p <= 1) {
      fn(0, n);
      return;
    }
    auto chunk = [&](int t) -> std::pair<int64_t, int64_t> {
      int64_t base = n / p, rem = n % p;
      int64_t b = t * base + std::min<int64_t>(t, rem);
      int64_t e = b + base + (t < rem ? 1 : 0);
      return {b, e};
    };
    {
      std::lock_guard<std::mutex> lk(m_);
      task_ = &fn;
      task_chunk_ = chunk;
      remaining_ = p - 1;  // workers 1..p-1; caller does chunk 0
      active_workers_ = p - 1;
      generation_++;
    }
    cv_start_.notify_all();
    // Caller runs chunk 0.
    auto [b0, e0] = chunk(0);
    fn(b0, e0);
    // Wait for workers to finish their chunks.
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return remaining_ == 0; });
    task_ = nullptr;
  }

  ~thread_pool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
      generation_++;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
  }

 private:
  thread_pool() {
    unsigned hc = std::thread::hardware_concurrency();
    nthreads_ = hc > 1 ? static_cast<int>(hc) : 1;
    for (int t = 1; t < nthreads_; t++) {
      workers_.emplace_back([this, t] { worker_(t); });
    }
  }

  void worker_(int t) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(m_);
      cv_start_.wait(lk, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      if (t >= active_workers_ + 1) continue;  // not needed this round
      auto chunk = task_chunk_;
      const auto* fn = task_;
      lk.unlock();
      auto [b, e] = chunk(t);
      (*fn)(b, e);
      lk.lock();
      if (--remaining_ == 0) cv_done_.notify_one();
    }
  }

  int nthreads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_start_, cv_done_;
  const std::function<void(int64_t, int64_t)>* task_ = nullptr;
  std::function<std::pair<int64_t, int64_t>(int)> task_chunk_;
  int remaining_ = 0;
  int active_workers_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace cpu
}  // namespace tl
