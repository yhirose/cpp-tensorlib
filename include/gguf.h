#pragma once

// GGUF v3 reader (llama.cpp model container). Memory-maps a .gguf file
// read-only and exposes its metadata KVs + tensor directory with zero-copy
// pointers into the mapping — tensor_info::data aims straight at the mmap'd
// weight bytes, so loading a model is O(header), not O(file). Little-endian
// wire format throughout (matches every target we run on). Quantized tensor
// codes (ggml_type) are stored verbatim; dequantization is the consumer's job.
//
// Header-only, no third-party deps, C++17. Memory mapping is POSIX mmap on
// Unix and CreateFileMapping/MapViewOfFile on Windows (see unmap_()/the ctor).

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tl::gguf {

// Metadata value types on the wire (u32).
enum class val_type : uint32_t {
  u8 = 0, i8, u16, i16, u32, i32, f32,
  boolean,  // 1 byte
  string,   // u64 len + bytes (not null-terminated)
  array,    // u32 elem_type + u64 count + packed elements
  u64, i64, f64,
};

// GGML tensor storage codes, verbatim from the file (subset we recognize).
enum class ggml_type : uint32_t {
  f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8,
  q8_1 = 9, q2_k = 10, q3_k = 11, q4_k = 12, q5_k = 13, q6_k = 14, q8_k = 15,
};

// Per-type block geometry: how many elements one block packs and its byte
// size. nbytes = n_elements / block_elems * block_bytes (f32/f16 are the
// degenerate 1-element "block").
inline void ggml_block(ggml_type t, uint64_t& elems, uint64_t& bytes) {
  switch (t) {
    case ggml_type::f32:  elems = 1;   bytes = 4;   return;
    case ggml_type::f16:  elems = 1;   bytes = 2;   return;
    case ggml_type::q4_0: elems = 32;  bytes = 18;  return;
    case ggml_type::q4_1: elems = 32;  bytes = 20;  return;
    case ggml_type::q5_0: elems = 32;  bytes = 22;  return;
    case ggml_type::q5_1: elems = 32;  bytes = 24;  return;
    case ggml_type::q8_0: elems = 32;  bytes = 34;  return;
    case ggml_type::q8_1: elems = 32;  bytes = 36;  return;
    case ggml_type::q2_k: elems = 256; bytes = 84;  return;
    case ggml_type::q3_k: elems = 256; bytes = 110; return;
    case ggml_type::q4_k: elems = 256; bytes = 144; return;
    case ggml_type::q5_k: elems = 256; bytes = 176; return;
    case ggml_type::q6_k: elems = 256; bytes = 210; return;
    case ggml_type::q8_k: elems = 256; bytes = 292; return;
  }
  throw std::runtime_error("gguf: unknown ggml type " +
                           std::to_string(static_cast<uint32_t>(t)));
}

// One decoded metadata value. Scalars land in u/i/f by signedness (bool in u),
// strings in s, arrays in arr (elements are metadata_values of arr_elem_type).
struct metadata_value {
  val_type type{};
  uint64_t u = 0;  // u8/u16/u32/u64/bool
  int64_t i = 0;   // i8/i16/i32/i64
  double f = 0;    // f32/f64
  std::string s;   // string
  val_type arr_elem_type{};        // valid when type==array
  std::vector<metadata_value> arr; // array elements

  // Typed accessors — throw on mismatch so a wrong key/type fails loudly
  // instead of reading a garbage hyperparameter.
  uint32_t as_u32() const { need(val_type::u32); return static_cast<uint32_t>(u); }
  int32_t as_i32() const { need(val_type::i32); return static_cast<int32_t>(i); }
  float as_f32() const { need(val_type::f32); return static_cast<float>(f); }
  uint64_t as_u64() const { need(val_type::u64); return u; }
  const std::string& as_str() const { need(val_type::string); return s; }

  // String-array helper (tokenizer.ggml.tokens / .merges).
  std::vector<std::string> as_str_array() const {
    if (type != val_type::array || arr_elem_type != val_type::string)
      throw std::runtime_error("gguf: value is not a string array");
    std::vector<std::string> v;
    v.reserve(arr.size());
    for (auto& e : arr) v.push_back(e.s);
    return v;
  }

 private:
  void need(val_type t) const {
    if (type != t)
      throw std::runtime_error("gguf: value type mismatch (have " +
                               std::to_string(static_cast<uint32_t>(type)) +
                               ", want " +
                               std::to_string(static_cast<uint32_t>(t)) + ")");
  }
};

// One tensor-directory entry. dims are as stored — "ne" order, dims[0] is the
// fastest-varying/contiguous axis. data points into the mapping (zero-copy).
struct tensor_info {
  std::string name;
  ggml_type type{};
  std::vector<uint64_t> dims;
  uint64_t rel_offset = 0;  // relative to the data section start
  const void* data = nullptr;
  uint64_t nbytes = 0;  // computed from dims + block geometry
};

class model {
 public:
  // mmap + parse the whole directory up front; throws std::runtime_error on
  // IO errors, bad magic, or an unsupported version.
  explicit model(const std::string& path) {
#ifdef _WIN32
    HANDLE fh = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (fh == INVALID_HANDLE_VALUE)
      throw std::runtime_error("gguf: cannot open " + path);
    LARGE_INTEGER li;
    if (!::GetFileSizeEx(fh, &li) || li.QuadPart <= 0) {
      ::CloseHandle(fh);
      throw std::runtime_error("gguf: cannot stat " + path);
    }
    size_ = static_cast<size_t>(li.QuadPart);
    HANDLE mh = ::CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    ::CloseHandle(fh);  // the mapping object keeps the file alive
    if (!mh) throw std::runtime_error("gguf: mmap failed for " + path);
    map_ = ::MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    ::CloseHandle(mh);  // the view keeps the mapping alive
    if (!map_) throw std::runtime_error("gguf: mmap failed for " + path);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      throw std::runtime_error("gguf: cannot stat " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // the mapping keeps the file alive
    if (map_ == MAP_FAILED) {
      map_ = nullptr;
      throw std::runtime_error("gguf: mmap failed for " + path);
    }
#endif
    try {
      parse_();
    } catch (...) {
      unmap_();
      throw;
    }
  }

  ~model() { unmap_(); }

  model(const model&) = delete;
  model& operator=(const model&) = delete;
  model(model&& o) noexcept { *this = std::move(o); }
  model& operator=(model&& o) noexcept {
    if (this != &o) {
      unmap_();
      map_ = o.map_; size_ = o.size_; o.map_ = nullptr; o.size_ = 0;
      version_ = o.version_; alignment_ = o.alignment_;
      kv_ = std::move(o.kv_);
      tensors_ = std::move(o.tensors_);
      index_ = std::move(o.index_);
    }
    return *this;
  }

  uint32_t version() const { return version_; }
  uint64_t tensor_count() const { return tensors_.size(); }
  uint64_t alignment() const { return alignment_; }

  bool has(const std::string& key) const { return kv_.count(key) != 0; }
  const metadata_value& kv(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second;
  }
  const std::unordered_map<std::string, metadata_value>& metadata() const {
    return kv_;
  }

  const tensor_info* tensor(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
  }
  const std::vector<tensor_info>& tensors() const { return tensors_; }

 private:
  // Platform-appropriate teardown of the mapping; null-safe and idempotent so
  // the ctor's catch, the dtor, and move-assign can all share it.
  void unmap_() noexcept {
    if (!map_) return;
#ifdef _WIN32
    ::UnmapViewOfFile(map_);
#else
    ::munmap(map_, size_);
#endif
    map_ = nullptr;
  }

  // ---- bounds-checked little-endian cursor over the mapping ----------------
  const uint8_t* base_() const { return static_cast<const uint8_t*>(map_); }

  void want_(size_t pos, size_t n) const {
    if (n > size_ || pos > size_ - n)
      throw std::runtime_error("gguf: truncated file (read past end)");
  }
  template <typename T>
  T rd_(size_t& pos) const {  // POD scalar, LE (host is LE on all targets)
    want_(pos, sizeof(T));
    T v;
    std::memcpy(&v, base_() + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::string rd_str_(size_t& pos) const {  // gguf_string: u64 len + bytes
    uint64_t len = rd_<uint64_t>(pos);
    want_(pos, len);
    std::string s(reinterpret_cast<const char*>(base_() + pos),
                  static_cast<size_t>(len));
    pos += len;
    return s;
  }

  metadata_value rd_value_(size_t& pos, val_type t) const {
    metadata_value v;
    v.type = t;
    switch (t) {
      case val_type::u8:  v.u = rd_<uint8_t>(pos); break;
      case val_type::i8:  v.i = rd_<int8_t>(pos); break;
      case val_type::u16: v.u = rd_<uint16_t>(pos); break;
      case val_type::i16: v.i = rd_<int16_t>(pos); break;
      case val_type::u32: v.u = rd_<uint32_t>(pos); break;
      case val_type::i32: v.i = rd_<int32_t>(pos); break;
      case val_type::f32: v.f = rd_<float>(pos); break;
      case val_type::boolean: v.u = rd_<uint8_t>(pos) != 0; break;
      case val_type::string: v.s = rd_str_(pos); break;
      case val_type::u64: v.u = rd_<uint64_t>(pos); break;
      case val_type::i64: v.i = rd_<int64_t>(pos); break;
      case val_type::f64: v.f = rd_<double>(pos); break;
      case val_type::array: {
        v.arr_elem_type = static_cast<val_type>(rd_<uint32_t>(pos));
        if (v.arr_elem_type == val_type::array)
          throw std::runtime_error("gguf: nested arrays unsupported");
        uint64_t count = rd_<uint64_t>(pos);
        v.arr.reserve(static_cast<size_t>(count));
        for (uint64_t k = 0; k < count; k++)
          v.arr.push_back(rd_value_(pos, v.arr_elem_type));
        break;
      }
      default:
        throw std::runtime_error("gguf: unknown value type " +
                                 std::to_string(static_cast<uint32_t>(t)));
    }
    return v;
  }

  void parse_() {
    size_t pos = 0;
    if (rd_<uint32_t>(pos) != 0x46554747u)  // 'GGUF' LE
      throw std::runtime_error("gguf: bad magic (not a GGUF file)");
    version_ = rd_<uint32_t>(pos);
    if (version_ != 3)
      throw std::runtime_error("gguf: unsupported version " +
                               std::to_string(version_));
    uint64_t n_tensors = rd_<uint64_t>(pos);
    uint64_t n_kv = rd_<uint64_t>(pos);

    for (uint64_t k = 0; k < n_kv; k++) {
      std::string key = rd_str_(pos);
      auto t = static_cast<val_type>(rd_<uint32_t>(pos));
      kv_.emplace(std::move(key), rd_value_(pos, t));
    }
    // Alignment may be overridden by metadata (any int width in the wild).
    if (auto it = kv_.find("general.alignment"); it != kv_.end())
      alignment_ = it->second.type == val_type::u32 ? it->second.as_u32()
                                                    : it->second.u;

    tensors_.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t k = 0; k < n_tensors; k++) {
      tensor_info t;
      t.name = rd_str_(pos);
      uint32_t n_dims = rd_<uint32_t>(pos);
      t.dims.reserve(n_dims);
      uint64_t n_elems = 1;
      for (uint32_t d = 0; d < n_dims; d++) {
        t.dims.push_back(rd_<uint64_t>(pos));
        n_elems *= t.dims.back();
      }
      t.type = static_cast<ggml_type>(rd_<uint32_t>(pos));
      t.rel_offset = rd_<uint64_t>(pos);
      uint64_t be, bb;
      ggml_block(t.type, be, bb);
      t.nbytes = n_elems / be * bb;
      index_.emplace(t.name, tensors_.size());
      tensors_.push_back(std::move(t));
    }

    // Data section starts at the directory end padded up to the alignment.
    size_t data_start = (pos + alignment_ - 1) / alignment_ * alignment_;
    for (auto& t : tensors_) {
      want_(data_start + static_cast<size_t>(t.rel_offset),
            static_cast<size_t>(t.nbytes));
      t.data = base_() + data_start + t.rel_offset;
    }
  }

  void* map_ = nullptr;
  size_t size_ = 0;
  uint32_t version_ = 0;
  uint64_t alignment_ = 32;  // GGUF default; general.alignment overrides
  std::unordered_map<std::string, metadata_value> kv_;
  std::vector<tensor_info> tensors_;
  std::unordered_map<std::string, size_t> index_;
};

}  // namespace tl::gguf
