#pragma once

// Minimal persistent thread pool for the CPU backend's parallel loops.
// One process-wide pool of (hardware_concurrency - 1) workers; the calling
// thread participates too, so a P-way split uses P threads with no idle
// caller. parallel_for hands [0, n) out in chunks from one shared counter
// and blocks until every chunk is done.
//
// Deliberately simple (one atomic counter, no queues or stealing): the chunks
// are near-equal cost, so a counter is all the balancing a hybrid or
// SMT-shared set of cores needs — the fast threads simply take more of them.
// A lazy singleton so tensor-free binaries never spawn threads.
//
// Wake-up is the cost that matters for the small and medium jobs a model's
// forward pass is made of, so it has three parts:
//  - A worker that has just finished spins on its slot's generation for a
//    short window (TL_CPU_SPIN_US, default 500 µs) before it sleeps on its
//    condition variable. Within the window the next parallel_for reaches it
//    with one atomic store — no syscall, no scheduler. This is how OpenMP
//    runtimes keep a chain of small ops fast: torch's elementwise ops on a
//    20-thread box go from 7 µs to 150 µs under OMP_WAIT_POLICY=passive, and
//    this pool's 256x512x512 gemm from 460 µs to 160 µs with the window.
//  - Sleeping workers are woken through a shared counter of "next worker to
//    wake": the caller wakes the first, every woken worker helps wake the
//    rest before it takes chunks, so a cold P-way start costs about log2(P)
//    wake latencies instead of P-1 sequential futex calls (140 → 85 µs for
//    20 threads on a WSL2 box).
//  - The join counts finished elements, not finished workers: a worker the
//    scheduler holds back (a vCPU the hypervisor took, a core a host process
//    borrowed) delays the round only if it holds a chunk, never by merely
//    having been signalled. Waiting for every signalled worker to report in
//    measured 250 µs rounds whenever 6+ threads spun on a 20-vCPU WSL2 guest.
// Each worker has its own condition variable, so a P-way split touches
// exactly P-1 workers; with one shared variable every call woke the whole
// pool, and the workers a small split did not need still took the mutex
// before going back to sleep — ~130 µs per call on a 20-thread box.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

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
    // Close the descriptor and let any worker still looking at the previous
    // task leave (it finds no chunk and departs within nanoseconds), so the
    // fields below are never rewritten under a reader.
    task_.open.store(false, std::memory_order_seq_cst);
    while (task_.readers.load(std::memory_order_seq_cst) != 0) pause_();
    task_.ctx = const_cast<void*>(static_cast<const void*>(&fn));
    task_.fn = [](void* ctx, int64_t b, int64_t e) {
      (*static_cast<std::remove_reference_t<F>*>(ctx))(b, e);
    };
    task_.n = n;
    task_.grain = (n + 4 * p - 1) / (4 * p);
    task_.p = p;
    task_.next.store(0, std::memory_order_relaxed);
    task_.done.store(0, std::memory_order_relaxed);
    task_.wake_next.store(1, std::memory_order_relaxed);
    task_.open.store(true, std::memory_order_seq_cst);
    help_wake_();
    run_chunks_();  // the caller is a worker too
    wait_done_();
    task_.ctx = nullptr;
  }

  ~thread_pool() {
    stop_.store(true, std::memory_order_relaxed);
    for (int t = 1; t < nthreads_; t++) wake_(t);
    for (auto& w : workers_) w.join();
  }

 private:
  using clk = std::chrono::steady_clock;
  enum : int { kAwake = 0, kSleeping = 1 };

  // One per worker: its wake-up signal. Heap-allocated so the vector can
  // grow without moving a mutex.
  struct slot {
    std::mutex m;
    std::condition_variable cv;
    std::atomic<uint64_t> generation{0};
    std::atomic<int> state{kAwake};  // kSleeping only around cv.wait
  };

  // The current parallel_for. Workers enter through readers/open so the
  // caller can tell when nobody is reading before it rewrites the fields.
  struct task {
    void* ctx = nullptr;  // the caller's callable, borrowed for the call
    void (*fn)(void*, int64_t, int64_t) = nullptr;
    int64_t n = 0;
    int64_t grain = 1;
    int p = 1;
    std::atomic<int64_t> next{0};  // first element not yet handed out
    std::atomic<int64_t> done{0};  // elements finished
    std::atomic<int> wake_next{1};  // next worker owed a wake-up
    std::atomic<int> readers{0};
    std::atomic<bool> open{false};
  };

  thread_pool() {
    unsigned hc = std::thread::hardware_concurrency();
    nthreads_ = hc > 1 ? static_cast<int>(hc) : 1;
    if (const char* e = std::getenv("TL_CPU_SPIN_US")) {
      char* end = nullptr;
      long long us = std::strtoll(e, &end, 10);
      if (end != e && us >= 0) spin_ = std::chrono::microseconds(us);
    }
    for (int t = 1; t < nthreads_; t++) slots_.push_back(std::make_unique<slot>());
    for (int t = 1; t < nthreads_; t++) {
      workers_.emplace_back([this, t] { worker_(t); });
    }
  }

  static void pause_() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("isb" ::: "memory");
#endif
  }

  // Spin on `ready` for the spin window; true when it held before the
  // window ran out.
  template <class Pred>
  bool spin_until_(Pred&& ready) const {
    if (spin_.count() == 0) return ready();
    auto t0 = clk::now();
    for (;;) {
      for (int i = 0; i < 64; i++) {
        if (ready()) return true;
        pause_();
      }
      if (clk::now() - t0 > spin_) return ready();
    }
  }

  // Signal worker t: bump its generation, and only if it is asleep take the
  // futex path. Bump (seq_cst) then read state, against the worker's write
  // state then read generation: one side always sees the other, so a worker
  // going to sleep either sees the bump in its wait predicate or is found
  // sleeping and notified. Taking the mutex before notify orders the notify
  // after the worker is inside cv.wait.
  void wake_(int t) {
    auto& s = *slots_[t - 1];
    s.generation.fetch_add(1, std::memory_order_seq_cst);
    if (s.state.load(std::memory_order_seq_cst) == kSleeping) {
      { std::lock_guard<std::mutex> lk(s.m); }
      s.cv.notify_one();
    }
  }

  // Wake the workers this task still owes a signal to, sharing the list with
  // every worker already awake (each calls this before taking chunks).
  void help_wake_() {
    for (;;) {
      int t = task_.wake_next.fetch_add(1, std::memory_order_relaxed);
      if (t >= task_.p) return;
      wake_(t);
    }
  }

  // Take chunks off the shared counter until the range is exhausted; count
  // each finished chunk, and wake the caller if the last one finds it asleep
  // (same write-flag-then-read protocol as wake_).
  void run_chunks_() {
    for (;;) {
      int64_t b = task_.next.fetch_add(task_.grain, std::memory_order_relaxed);
      if (b >= task_.n) return;
      int64_t e = std::min(task_.n, b + task_.grain);
      task_.fn(task_.ctx, b, e);
      if (task_.done.fetch_add(e - b, std::memory_order_seq_cst) + (e - b) == task_.n &&
          caller_sleeping_.load(std::memory_order_seq_cst)) {
        { std::lock_guard<std::mutex> lk(done_m_); }
        done_cv_.notify_one();
      }
    }
  }

  // Caller side of the join: spin for the tail, then sleep.
  void wait_done_() {
    auto done = [&] { return task_.done.load(std::memory_order_acquire) == task_.n; };
    if (spin_until_(done)) return;
    std::unique_lock<std::mutex> lk(done_m_);
    caller_sleeping_.store(true, std::memory_order_seq_cst);
    done_cv_.wait(lk, [&] { return task_.done.load(std::memory_order_seq_cst) == task_.n; });
    caller_sleeping_.store(false, std::memory_order_relaxed);
  }

  void worker_(int t) {
    auto& s = *slots_[t - 1];
    uint64_t seen = 0;
    for (;;) {
      auto signalled = [&] {
        return s.generation.load(std::memory_order_acquire) != seen;
      };
      if (!spin_until_(signalled)) {
        std::unique_lock<std::mutex> lk(s.m);
        s.state.store(kSleeping, std::memory_order_seq_cst);
        s.cv.wait(lk, [&] { return s.generation.load(std::memory_order_seq_cst) != seen; });
        s.state.store(kAwake, std::memory_order_relaxed);
      }
      seen = s.generation.load(std::memory_order_acquire);
      if (stop_.load(std::memory_order_relaxed)) return;
      // Enter the descriptor: register as a reader, then check it is open.
      // Against the caller's close-then-wait-for-readers, one side always
      // sees the other, so a late worker either finds the task closed or is
      // waited for — and a worker signalled for a round that already ended
      // simply finds no chunk in whatever round is current.
      task_.readers.fetch_add(1, std::memory_order_seq_cst);
      if (task_.open.load(std::memory_order_seq_cst)) {
        help_wake_();
        run_chunks_();
      }
      task_.readers.fetch_sub(1, std::memory_order_seq_cst);
    }
  }

  int nthreads_ = 1;
  std::chrono::microseconds spin_{500};
  std::vector<std::unique_ptr<slot>> slots_;  // slots_[t - 1] belongs to worker t
  std::vector<std::thread> workers_;
  task task_;
  std::mutex done_m_;
  std::condition_variable done_cv_;
  std::atomic<bool> caller_sleeping_{false};
  std::atomic<bool> stop_{false};
};

}  // namespace cpu
}  // namespace tl
