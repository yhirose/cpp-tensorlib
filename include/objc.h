#pragma once

// Minimal Objective-C runtime bridge so the Metal backend stays plain C++
// (no .mm files, header-only). Casting objc_msgSend to the exact function
// type per call site is the documented-correct way to use it from C.
//
// Deliberately does NOT include <objc/message.h> / <objc/runtime.h>: those
// define nil/Nil/YES/NO as macros, which stomps on embedders' identifiers
// (culebra's `Value::Nil` collided). Instead the three functions we use are
// declared with prototypes exactly matching the system headers (Class and
// SEL are pointers to these forward-declared structs), so a TU that also
// includes the real headers sees compatible redeclarations.

#ifdef __APPLE__

#include <string>

extern "C" {
struct objc_class;
struct objc_selector;
struct objc_class* objc_getClass(const char* name);
struct objc_selector* sel_registerName(const char* str);
void objc_msgSend(void);
}

namespace tl {
namespace objc {

using id = void*;
using sel_t = objc_selector*;

inline id cls(const char* name) {
  return reinterpret_cast<id>(objc_getClass(name));
}

template <typename Ret = id, typename... Args>
Ret send(id obj, const char* selector, Args... args) {
  using fn = Ret (*)(id, sel_t, Args...);
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
