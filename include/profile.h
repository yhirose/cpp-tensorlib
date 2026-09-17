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
// hands the elapsed time back when it drains; Metal knows the GPU time of each
// command buffer (one batch, not one launch); WebGPU counts. A row therefore
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
#include <unordered_map>
#include <vector>

namespace tl {
namespace profile {

struct row {
  std::string path;    // "/"-joined scope labels
  std::string kernel;  // empty for the scope's own row; "h2d"/"d2h"/"wait"
                       // for a transfer or a blocking wait under it
  uint64_t count = 0;         // scope entries, or launches / transfers / waits
  double host_us = 0;         // scope rows: inclusive wall; transfer and wait
                              // rows: the time the host was blocked
  double device_us = 0;       // kernel rows: summed launch time, where timed
  uint64_t device_timed = 0;  // launches device_us covers
  uint64_t bytes = 0;         // transfer rows: bytes moved
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
  std::vector<size_t> marks;              // path length before each push
  std::vector<clock::time_point> starts;  // each open scope's entry time
  std::unordered_map<std::string, row> rows;  // key: path + '\0' + kernel
  uint64_t batches = 0;
  double batch_device_us = 0;
};

inline state& st() {
  static thread_local state* s = new state();  // leaked: see the header note
  return *s;
}

// The row for (current path, kernel), made on first use. Pointers into the
// map stay valid across rehash, which is what lets a backend hold one until
// its device time comes back.
inline row& row_(state& s, std::string_view kernel) {
  thread_local std::string key;
  key.assign(s.path);
  key.push_back('\0');
  key.append(kernel);
  auto it = s.rows.find(key);
  if (it == s.rows.end()) {
    it = s.rows.emplace(key, row{}).first;
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
  row& r = row_(s, kernel);
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
  row& r = row_(s, kind);
  r.count++;
  r.bytes += bytes;
  r.host_us += host_us;
}

// The host blocked waiting for the device (a flush, a read-back's sync).
inline void wait(double us) {
  auto& s = st();
  if (!s.active) return;
  row& r = row_(s, "wait");
  r.count++;
  r.host_us += us;
}

// The device time of one finished batch, where the backend only knows it per
// batch (Metal's command buffer).
inline void batch_device(double us) {
  auto& s = st();
  if (!s.active) return;
  s.batches++;
  s.batch_device_us += us;
}

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

namespace detail {
// End the session: what was launched is resolved and the rows stay readable.
inline void stop_(state& s) {
  if (!s.active) return;
  if (drain_hook) drain_hook();
  s.active = false;
}
}  // namespace detail

inline void stop() { detail::stop_(detail::st()); }

class scope {
 public:
  explicit scope(std::string_view label) {
    auto& s = detail::st();
    if (!s.active) return;
    s_ = &s;
    s.marks.push_back(s.path.size());
    if (!s.path.empty()) s.path.push_back('/');
    s.path.append(label);
    s.starts.push_back(detail::clock::now());
  }
  ~scope() {
    if (!s_ || s_->marks.empty()) return;
    auto& s = *s_;
    const double us = detail::us_since(s.starts.back());
    s.starts.pop_back();
    if (s.active) {
      row& r = detail::row_(s, "");
      r.count++;
      r.host_us += us;
    }
    s.path.resize(s.marks.back());
    s.marks.pop_back();
  }
  scope(const scope&) = delete;
  scope& operator=(const scope&) = delete;

 private:
  detail::state* s_ = nullptr;
};

namespace detail {

// The session's rows, grouped by path — paths in descending order of their
// scope's inclusive time, each scope row first, its kernels by device time —
// so the table reads top-down from where the time went.
inline std::vector<row> rows_(state& s) {
  if (s.active && drain_hook) drain_hook();
  std::vector<row> out;
  out.reserve(s.rows.size());
  for (const auto& kv : s.rows) out.push_back(kv.second);
  std::unordered_map<std::string, double> weight;
  for (const auto& r : out) {
    if (r.kernel.empty()) weight[r.path] = r.host_us;
  }
  std::sort(out.begin(), out.end(), [&](const row& a, const row& b) {
    if (a.path != b.path) {
      const double wa = weight.count(a.path) ? weight[a.path] : -1.0;
      const double wb = weight.count(b.path) ? weight[b.path] : -1.0;
      if (wa != wb) return wa > wb;
      return a.path < b.path;
    }
    if (a.kernel.empty() != b.kernel.empty()) return a.kernel.empty();
    if (a.device_us != b.device_us) return a.device_us > b.device_us;
    if (a.host_us != b.host_us) return a.host_us > b.host_us;
    return a.kernel < b.kernel;
  });
  return out;
}

inline summary summarize_(state& s) {
  summary t;
  for (const auto& kv : s.rows) {
    const row& r = kv.second;
    if (r.kernel.empty()) {
      t.scopes += r.count;
    } else if (r.kernel == "wait") {
      t.wait_us += r.host_us;
    } else if (r.kernel != "h2d" && r.kernel != "d2h") {
      t.launches += r.count;
      t.launches_timed += r.device_timed;
      t.device_us += r.device_us;
    }
  }
  t.batches = s.batches;
  t.batch_device_us = s.batch_device_us;
  return t;
}

// The text table: one line per row, indented one level for a kernel under
// its scope. Times in ms; a device column left blank was not timed.
inline void report_(state& s, FILE* out) {
  const summary t = summarize_(s);
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
  for (const row& r : rows_(s)) {
    char host[24] = "", dev[24] = "";
    if (r.kernel.empty() || r.host_us > 0) {
      std::snprintf(host, sizeof host, "%.3f", r.host_us / 1000.0);
    }
    if (r.device_timed) {
      std::snprintf(dev, sizeof dev, "%.3f", r.device_us / 1000.0);
    }
    if (r.kernel.empty()) {
      std::fprintf(out, "%10llu %11s %11s  %s\n", (unsigned long long)r.count,
                   host, dev, r.path.empty() ? "(top)" : r.path.c_str());
    } else if (!group || *group != r.path) {
      // launches under a path that opened no scope of its own (eager work
      // at the top level): name the path before its kernels
      std::fprintf(out, "%10s %11s %11s  %s\n", "", "", "",
                   r.path.empty() ? "(top)" : r.path.c_str());
    }
    group = &r.path;
    if (r.kernel.empty()) {
      continue;
    } else if (r.bytes) {
      std::fprintf(out, "%10llu %11s %11s    %s  %.1f MB\n",
                   (unsigned long long)r.count, host, dev, r.kernel.c_str(),
                   r.bytes / 1048576.0);
    } else {
      std::fprintf(out, "%10llu %11s %11s    %s\n",
                   (unsigned long long)r.count, host, dev, r.kernel.c_str());
    }
  }
}

// TL_PROFILE=1: profile the whole process from the first evaluation and
// print the table to stderr at exit. Called once from the evaluator; the
// handler names the state it started rather than the thread it runs on.
inline void env_autostart() {
  if (!std::getenv("TL_PROFILE")) return;
  start();
  static state* main_state = &st();
  std::atexit([] {
    if (!main_state->active) return;
    stop_(*main_state);
    report_(*main_state, stderr);
  });
}

}  // namespace detail

inline std::vector<row> rows() { return detail::rows_(detail::st()); }
inline summary summarize() { return detail::summarize_(detail::st()); }
inline void report(FILE* out) { detail::report_(detail::st(), out); }

}  // namespace profile
}  // namespace tl
