#pragma once

// Device selection. v1 computes in F32 only (BF16 lands later as a storage
// type); dtype is therefore not part of the public type system.

namespace tl {

enum class device_type { cpu, gpu, auto_ };

inline device_type device_ = device_type::cpu;

inline void use_cpu() { device_ = device_type::cpu; }
inline void use_gpu() { device_ = device_type::gpu; }
inline void use_auto() { device_ = device_type::auto_; }

// True when a GPU backend is compiled in AND a device is reachable at
// runtime (Metal on macOS; CUDA driver via dlopen on Linux/Windows — on
// WSL2 the driver lives at /usr/lib/wsl/lib/libcuda.so). No GPU backend
// exists yet, so gpu/auto modes fall back to the reference CPU path; CI
// runs all three modes so the fallback stays permanently tested.
inline bool gpu_available() { return false; }

}  // namespace tl
