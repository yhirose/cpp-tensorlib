#pragma once

// tl::profile — where evaluation goes, by op and by kernel, on every backend.
//
// A scope is a label held open for the extent of a C++ block ("backward",
// "Dot"); nested scopes join into a path ("backward/Dot"). Every kernel launch,
// transfer and blocking wait a backend performs is attributed to the innermost
// open scope, and the evaluator opens one per graph node under its op name, so
// a consumer that labels only its own phases still sees which op, and which
// kernel under it, each phase spent its time in.
//
// What a backend can stamp differs: CUDA brackets every launch with events and
// hands the elapsed time back when it drains; Metal, which knows GPU time only
// per command buffer, commits each launch as its own buffer while a profile
// runs and reads them back at the flush; WebGPU counts. A row therefore
// says how many of its launches its device time covers (`device_timed`) — an
// untimed launch is not a free one, and a zero must not read as fast. An
// event pair spans from where the stream reached the launch to where the
// kernel finished, so on a busy stream the gap before the kernel starts is
// in its time too: summed, the rows are the device's timeline, a few percent
// over the kernels alone.
//
// Off, the cost is one predicate per scope and per launch. The state is per
// thread and never freed: the report at exit (TL_PROFILE=1) runs after the
// thread-local destructors would have.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tl {
namespace profile {

struct row {
  enum class kind_t { scope, launch, transfer, wait };
  kind_t kind = kind_t::scope;
  std::string path;    // "/"-joined scope labels
  std::string kernel;  // launch: the kernel's name; transfer: "h2d" / "d2h"
  uint64_t count = 0;         // scope entries, or launches / transfers / waits
  double host_us = 0;         // scope: inclusive wall; transfer and wait: how
                              // long the host was blocked
  double device_us = 0;       // launch: summed kernel time, where timed
  uint64_t device_timed = 0;  // launches device_us covers
  uint64_t bytes = 0;         // transfer: bytes moved
};

// Totals over the rows, plus what only the batch knows.
struct summary {
  uint64_t scopes = 0, launches = 0, launches_timed = 0;
  double device_us = 0, wait_us = 0;
  uint64_t batches = 0;       // Metal: command buffers with a GPU time
  double batch_device_us = 0;  // their summed GPU time
};

namespace detail {

using clock = std::chrono::steady_clock;

inline double us_since(clock::time_point t) {
  return std::chrono::duration<double, std::micro>(clock::now() - t).count();
}

struct state {
  bool active = false;
  std::string path;
  struct frame {
    size_t mark;  // path length before this scope's label
    clock::time_point start;
  };
  std::vector<frame> open;
  std::unordered_map<std::string, row> rows;  // key: path, kind, kernel
  uint64_t batches = 0;
  double batch_device_us = 0;
};

inline state& st() {
  static thread_local state* s = new state();  // leaked: see the header note
  return *s;
}

// The row for (current path, kind, kernel), made on first use. Pointers into
// the map stay valid across rehash, which is what lets a backend hold one
// until its device time comes back.
inline row& row_(state& s, row::kind_t kind, std::string_view kernel) {
  thread_local std::string key;
  key.assign(s.path);
  key.push_back('\0');
  key.push_back(static_cast<char>(kind));
  key.append(kernel);
  auto it = s.rows.find(key);
  if (it == s.rows.end()) {
    it = s.rows.emplace(key, row{}).first;
    it->second.kind = kind;
    it->second.path = s.path;
    it->second.kernel = std::string(kernel);
  }
  return it->second;
}

// ---- the backend side ----

// A backend that stamps launches asynchronously resolves them here: called
// before rows are read or cleared, and by the backend itself whenever it
// knows the device is idle.
inline void (*drain_hook)() = nullptr;

// One kernel launch under the current scope. Null when profiling is off;
// otherwise the row to add the launch's device time to once known.
inline row* launch(std::string_view kernel) {
  auto& s = st();
  if (!s.active) return nullptr;
  row& r = row_(s, row::kind_t::launch, kernel);
  r.count++;
  return &r;
}

inline void device_time(row* r, double us) {
  r->device_us += us;
  r->device_timed++;
}

// A host<->device copy ("h2d" / "d2h") under the current scope; `host_us` is
// how long the host was blocked for it (0 for an async one).
inline void transfer(const char* kind, uint64_t bytes, double host_us) {
  auto& s = st();
  if (!s.active) return;
  row& r = row_(s, row::kind_t::transfer, kind);
  r.count++;
  r.bytes += bytes;
  r.host_us += host_us;
}

// The host blocked waiting for the device (a flush, a read-back's sync).
inline void wait(double us) {
  auto& s = st();
  if (!s.active) return;
  row& r = row_(s, row::kind_t::wait, "");
  r.count++;
  r.host_us += us;
}

// The device time of one finished command buffer (Metal's): with a row per
// launch under a profile, or a batch of launches the row cannot name.
inline void batch_device(double us) {
  auto& s = st();
  if (!s.active) return;
  s.batches++;
  s.batch_device_us += us;
}

// Times a blocking call for the backends: a wait, or a transfer of `bytes`
// when `kind` names one.
struct blocked {
  const char* kind = nullptr;
  uint64_t bytes = 0;
  clock::time_point t0 = clock::now();
  ~blocked() {
    const double us = us_since(t0);
    if (kind) transfer(kind, bytes, us);
    else wait(us);
  }
};

}  // namespace detail

inline bool active() { return detail::st().active; }

// Begin a session on this thread: drops what the last one recorded. Scopes
// already open stay open (their rows land when they close).
inline void start() {
  auto& s = detail::st();
  if (detail::drain_hook) detail::drain_hook();
  s.rows.clear();
  s.batches = 0;
  s.batch_device_us = 0;
  s.active = true;
}

// End the session: what was launched is resolved and the rows stay readable.
inline void stop() {
  auto& s = detail::st();
  if (!s.active) return;
  if (detail::drain_hook) detail::drain_hook();
  s.active = false;
}

class scope {
 public:
  explicit scope(std::string_view label) {
    auto& s = detail::st();
    if (!s.active) return;
    open_ = true;
    s.open.push_back({s.path.size(), detail::clock::now()});
    if (!s.path.empty()) s.path.push_back('/');
    s.path.append(label);
  }
  ~scope() {
    if (!open_) return;
    auto& s = detail::st();
    const auto f = s.open.back();
    s.open.pop_back();
    if (s.active) {
      row& r = detail::row_(s, row::kind_t::scope, "");
      r.count++;
      r.host_us += detail::us_since(f.start);
    }
    s.path.resize(f.mark);
  }
  scope(const scope&) = delete;
  scope& operator=(const scope&) = delete;

 private:
  bool open_ = false;
};

// The session's rows, grouped by path — paths in descending order of their
// scope's inclusive time, each scope row first, its kernels by device time —
// so the table reads top-down from where the time went.
inline std::vector<row> rows() {
  auto& s = detail::st();
  if (s.active && detail::drain_hook) detail::drain_hook();
  std::unordered_map<std::string_view, double> weight;
  for (const auto& kv : s.rows) {
    if (kv.second.kind == row::kind_t::scope) {
      weight[kv.second.path] = kv.second.host_us;
    }
  }
  std::vector<row> out;
  out.reserve(s.rows.size());
  for (const auto& kv : s.rows) out.push_back(kv.second);
  auto key = [&](const row& r) {
    auto w = weight.find(r.path);
    return std::make_tuple(w == weight.end() ? 1.0 : -w->second,
                           std::string_view(r.path),
                           r.kind != row::kind_t::scope, -r.device_us,
                           -r.host_us, std::string_view(r.kernel));
  };
  std::sort(out.begin(), out.end(),
            [&](const row& a, const row& b) { return key(a) < key(b); });
  return out;
}

inline summary summarize() {
  auto& s = detail::st();
  summary t;
  for (const auto& kv : s.rows) {
    const row& r = kv.second;
    switch (r.kind) {
      case row::kind_t::scope: t.scopes += r.count; break;
      case row::kind_t::launch:
        t.launches += r.count;
        t.launches_timed += r.device_timed;
        t.device_us += r.device_us;
        break;
      case row::kind_t::wait: t.wait_us += r.host_us; break;
      case row::kind_t::transfer: break;
    }
  }
  t.batches = s.batches;
  t.batch_device_us = s.batch_device_us;
  return t;
}

// The text table: one line per row, indented one level for a launch, transfer
// or wait under its scope. Times in ms; a device column left blank was not
// timed.
inline void report(FILE* out) {
  const summary t = summarize();
  std::fprintf(out,
               "tl profile: %llu scope entries, %llu launches (%llu timed), "
               "device %.3f ms, waits %.3f ms",
               (unsigned long long)t.scopes, (unsigned long long)t.launches,
               (unsigned long long)t.launches_timed, t.device_us / 1000.0,
               t.wait_us / 1000.0);
  if (t.batches) {
    std::fprintf(out, ", %llu batches %.3f ms on the device",
                 (unsigned long long)t.batches, t.batch_device_us / 1000.0);
  }
  std::fprintf(out, "\n%10s %11s %11s  %s\n", "count", "host ms", "device ms",
               "path / kernel");
  const std::string* group = nullptr;  // the path whose rows are printing
  for (const row& r : rows()) {
    const char* label = r.path.empty() ? "(top)" : r.path.c_str();
    char host[24] = "", dev[24] = "";
    if (r.kind == row::kind_t::scope || r.host_us > 0) {
      std::snprintf(host, sizeof host, "%.3f", r.host_us / 1000.0);
    }
    if (r.device_timed) {
      std::snprintf(dev, sizeof dev, "%.3f", r.device_us / 1000.0);
    }
    if (r.kind == row::kind_t::scope) {
      std::fprintf(out, "%10llu %11s %11s  %s\n", (unsigned long long)r.count,
                   host, dev, label);
      group = &r.path;
      continue;
    }
    if (!group || *group != r.path) {
      // work under a path that opened no scope of its own (eager work at
      // the top level): name the path before its rows
      std::fprintf(out, "%10s %11s %11s  %s\n", "", "", "", label);
      group = &r.path;
    }
    const char* name =
        r.kind == row::kind_t::wait ? "wait" : r.kernel.c_str();
    std::fprintf(out, "%10llu %11s %11s    %s", (unsigned long long)r.count,
                 host, dev, name);
    if (r.bytes) std::fprintf(out, "  %.1f MB", r.bytes / 1048576.0);
    std::fputc('\n', out);
  }
}

namespace detail {
// TL_PROFILE=1: profile the whole process from the first evaluation or kernel
// launch (gpu::launched — a decoder on the model path never reaches the
// evaluator) and print the table to stderr at exit. The first call does the
// work (the exit handler reads the calling thread's state, which is the
// evaluating one in every consumer).
inline void env_autostart() {
  static bool once = false;
  if (once) return;
  once = true;
  if (!std::getenv("TL_PROFILE")) return;
  start();
  std::atexit([] {
    if (!active()) return;
    stop();
    report(stderr);
  });
}
}  // namespace detail

}  // namespace profile
}  // namespace tl
