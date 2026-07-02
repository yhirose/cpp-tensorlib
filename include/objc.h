#pragma once

// Minimal Objective-C runtime bridge so the Metal backend stays plain C++
// (no .mm files, header-only). Casting objc_msgSend to the exact function
// type per call site is the documented-correct way to use it from C.

#ifdef __APPLE__

#include <objc/message.h>
#include <objc/runtime.h>

#include <string>

namespace tl {
namespace objc {

using id = void*;

inline id cls(const char* name) {
  return reinterpret_cast<id>(objc_getClass(name));
}

template <typename Ret = id, typename... Args>
Ret send(id obj, const char* selector, Args... args) {
  using fn = Ret (*)(id, SEL, Args...);
  return reinterpret_cast<fn>(objc_msgSend)(obj, sel_registerName(selector),
                                            args...);
}

inline std::string error_str(id err) {
  if (!err) return "unknown error";
  auto desc = send(err, "localizedDescription");
  auto* cstr = send<const char*>(desc, "UTF8String");
  return cstr ? cstr : "unknown error";
}

}  // namespace objc
}  // namespace tl

#endif  // __APPLE__
