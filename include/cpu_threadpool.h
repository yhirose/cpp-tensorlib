#pragma once

// Minimal persistent thread pool for the CPU backend's parallel loops.
// One process-wide pool of (hardware_concurrency - 1) workers; the calling
// thread participates too, so a P-way split uses P threads with no idle
// caller. parallel_for hands [0, n) out in chunks from one shared counter
// and blocks until done.
//
// Deliberately simple (one atomic counter, no queues or stealing): the chunks
// are near-equal cost, so a counter is all the balancing a hybrid or
// SMT-shared set of cores needs — the fast threads simply take more of them.
// A lazy singleton so tensor-free binaries never spawn threads.
//
// Each worker sleeps on its own condition variable, and a P-way split wakes
// exactly P-1 of them. With one shared variable every call woke the whole
// pool, and the workers a small split did not need still took the mutex
// before going back to sleep — ~130 µs per call on a 20-thread box, more
// than a 30x784x10 gemm computes in.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <type_traits>
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

  // Run fn(begin, end) over [0, n) across up to min(size(), max_threads)
  // threads (the caller works too). Blocks until the whole range is done. A
  // caller that knows its work is small caps the split with max_threads, so a
  // tiny job never pays for threads it cannot feed; max_threads <= 1 runs
  // inline. The range is handed out from a shared counter in chunks of about
  // n/(4p), not as p equal slices: the cores are not equal — a hybrid part's
  // efficiency cores run a gemm chunk at a fraction of a performance core's
  // pace, and two SMT threads share one core's units — so an equal split
  // waits for its slowest thread (measured 2.7x on 8 threads for a gemm that
  // reaches 7x when the fast threads take the slow ones' leftovers; a coarser
  // grain that kept gemm's MC row groups whole measured slower — balance
  // beats the extra B traffic). fn may run more than once per thread, each
  // time on a disjoint range. Any callable: it is held by pointer for the
  // call's duration, never copied, so no std::function and no allocation.
  template <class F>
  void parallel_for(int64_t n, F&& fn, int max_threads = 1 << 30) {
    if (n <= 0) return;
    int p = static_cast<int>(std::min<int64_t>(std::min(nthreads_, max_threads), n));
    if (p <= 1) {
      fn(0, n);
      return;
    }
    task_ctx_ = const_cast<void*>(static_cast<const void*>(&fn));
    task_fn_ = [](void* ctx, int64_t b, int64_t e) {
      (*static_cast<std::remove_reference_t<F>*>(ctx))(b, e);
    };
    task_n_ = n;
    task_grain_ = (n + 4 * p - 1) / (4 * p);
    task_next_.store(0, std::memory_order_relaxed);
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
    run_chunks_();  // the caller is a worker too
    // Wait for the workers to drain the counter.
    std::unique_lock<std::mutex> lk(done_m_);
    done_cv_.wait(lk, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
    task_ctx_ = nullptr;
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

  // Take chunks off the shared counter until the range is exhausted.
  void run_chunks_() {
    for (;;) {
      int64_t b = task_next_.fetch_add(task_grain_, std::memory_order_relaxed);
      if (b >= task_n_) return;
      task_fn_(task_ctx_, b, std::min(task_n_, b + task_grain_));
    }
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
      run_chunks_();
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
  void* task_ctx_ = nullptr;  // the caller's callable, borrowed for the call
  void (*task_fn_)(void*, int64_t, int64_t) = nullptr;
  int64_t task_n_ = 0;
  int64_t task_grain_ = 1;
  std::atomic<int64_t> task_next_{0};
  std::atomic<bool> stop_{false};
};

}  // namespace cpu
}  // namespace tl
