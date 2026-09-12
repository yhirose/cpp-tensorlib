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
//
// Each worker sleeps on its own condition variable, and a P-way split wakes
// exactly P-1 of them. With one shared variable every call woke the whole
// pool, and the workers a small split did not need still took the mutex
// before going back to sleep — ~130 µs per call on a 20-thread box, more
// than a 30x784x10 gemm computes in.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
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

  // Run fn(begin, end) over a static partition of [0, n) across up to
  // min(size(), max_threads) threads (caller runs one share). Blocks until
  // all shares complete. A caller that knows its work is small caps the
  // split with max_threads, so a tiny job never pays for threads it cannot
  // feed; max_threads <= 1 runs inline.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn,
                    int max_threads = 1 << 30) {
    if (n <= 0) return;
    int p = static_cast<int>(std::min<int64_t>(std::min(nthreads_, max_threads), n));
    if (p <= 1) {
      fn(0, n);
      return;
    }
    task_ = &fn;
    task_n_ = n;
    task_p_ = p;
    remaining_.store(p - 1, std::memory_order_relaxed);
    // Publish the task through each worker's own mutex: the lock/unlock pair
    // orders the writes above before that worker's reads of them.
    for (int t = 1; t < p; t++) {
      auto& s = *slots_[t - 1];
      {
        std::lock_guard<std::mutex> lk(s.m);
        s.generation++;
      }
      s.cv.notify_one();
    }
    // Caller runs chunk 0.
    auto [b0, e0] = chunk_(0, n, p);
    fn(b0, e0);
    // Wait for workers to finish their chunks.
    std::unique_lock<std::mutex> lk(done_m_);
    done_cv_.wait(lk, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
    task_ = nullptr;
  }

  ~thread_pool() {
    stop_.store(true, std::memory_order_relaxed);
    for (auto& s : slots_) {
      {
        std::lock_guard<std::mutex> lk(s->m);
        s->generation++;
      }
      s->cv.notify_one();
    }
    for (auto& w : workers_) w.join();
  }

 private:
  // One per worker: its wake-up signal. Heap-allocated so the vector can
  // grow without moving a mutex.
  struct slot {
    std::mutex m;
    std::condition_variable cv;
    uint64_t generation = 0;
  };

  thread_pool() {
    unsigned hc = std::thread::hardware_concurrency();
    nthreads_ = hc > 1 ? static_cast<int>(hc) : 1;
    for (int t = 1; t < nthreads_; t++) slots_.push_back(std::make_unique<slot>());
    for (int t = 1; t < nthreads_; t++) {
      workers_.emplace_back([this, t] { worker_(t); });
    }
  }

  static std::pair<int64_t, int64_t> chunk_(int t, int64_t n, int p) {
    int64_t base = n / p, rem = n % p;
    int64_t b = t * base + std::min<int64_t>(t, rem);
    int64_t e = b + base + (t < rem ? 1 : 0);
    return {b, e};
  }

  void worker_(int t) {
    auto& s = *slots_[t - 1];
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(s.m);
        s.cv.wait(lk, [&] { return s.generation != seen; });
        seen = s.generation;
      }
      if (stop_.load(std::memory_order_relaxed)) return;
      auto [b, e] = chunk_(t, task_n_, task_p_);
      (*task_)(b, e);
      // The decrement happens under done_m_ so the caller's predicate cannot
      // miss it between its check and its wait.
      std::lock_guard<std::mutex> lk(done_m_);
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        done_cv_.notify_one();
      }
    }
  }

  int nthreads_ = 1;
  std::vector<std::unique_ptr<slot>> slots_;  // slots_[t - 1] belongs to worker t
  std::vector<std::thread> workers_;
  std::mutex done_m_;
  std::condition_variable done_cv_;
  std::atomic<int> remaining_{0};
  const std::function<void(int64_t, int64_t)>* task_ = nullptr;
  int64_t task_n_ = 0;
  int task_p_ = 0;
  std::atomic<bool> stop_{false};
};

}  // namespace cpu
}  // namespace tl
