#pragma once

// Device selection. v1 computes in F32 only (BF16 lands later as a storage
// type); dtype is therefore not part of the public type system.

namespace tl {

enum class device_type { cpu, gpu, auto_ };

inline device_type device_ = device_type::cpu;

inline void use_cpu() { device_ = device_type::cpu; }
inline void use_gpu() { device_ = device_type::gpu; }
inline void use_auto() { device_ = device_type::auto_; }

// tl::gpu_available() — true when a GPU backend is compiled in AND a device
// is reachable at runtime — is defined in metal.h (Metal on macOS; the CUDA
// dlopen probe joins it in M6). Where no device exists, gpu/auto modes fall
// back to the CPU path; CI runs all three modes so the fallback stays
// permanently tested.

// Testing/benchmark toggle: when false, evaluation ignores accelerated CPU
// backends and runs everything through the ref:: oracle. Lets tests compare
// backend results against the oracle in-process.
inline bool use_accelerate_ = true;

}  // namespace tl
