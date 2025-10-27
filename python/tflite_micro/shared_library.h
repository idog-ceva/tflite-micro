/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file is forked from TFLite's implementation in
// //depot/google3/third_party/tensorflow/lite/shared_library.h and contains a
// subset of it that's required by the TFLM interpreter. The Windows' ifdef is
// removed because TFLM doesn't support Windows.

#ifndef TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
#define TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
#pragma once

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>

  // Track the most recent dlopen() so that dlsym(RTLD_DEFAULT, ...)
  // can resolve from "the" loaded library, which mirrors how TFLM uses it.
  inline HMODULE& _last_dlopen_module() {
    static HMODULE g_last = nullptr;
    return g_last;
  }

  // POSIX flag placeholders to keep code compiling.
  #ifndef RTLD_NOW
  #define RTLD_NOW 0
  #endif
  #ifndef RTLD_LOCAL
  #define RTLD_LOCAL 0
  #endif
  #ifndef RTLD_GLOBAL
  #define RTLD_GLOBAL 0
  #endif
  // GNU/Linux: RTLD_DEFAULT is (void*)0 or (void*)-2 depending on libc.
  // We just pick a sentinel and check for it.
  #ifndef RTLD_DEFAULT
  #define RTLD_DEFAULT ((void*)-2)
  #endif

  inline void* dlopen(const char* path, int /*flags*/) {
    HMODULE h = LoadLibraryA(path);
    if (h) _last_dlopen_module() = h;
    return reinterpret_cast<void*>(h);
  }

  inline void* dlsym(void* handle, const char* symbol) {
    HMODULE mod = nullptr;
    if (handle == RTLD_DEFAULT) {
      // Best-effort: consult the last module we loaded.
      mod = _last_dlopen_module();
      if (!mod) {
        // As a fallback, check the main module.
        mod = GetModuleHandleA(nullptr);
      }
    } else {
      mod = reinterpret_cast<HMODULE>(handle);
    }
    if (!mod) return nullptr;
    FARPROC p = GetProcAddress(mod, symbol);
    return reinterpret_cast<void*>(p);
  }

  inline int dlclose(void* handle) {
    HMODULE mod = reinterpret_cast<HMODULE>(handle);
    if (mod == _last_dlopen_module()) _last_dlopen_module() = nullptr;
    return FreeLibrary(mod) ? 0 : 1;
  }

  inline const char* dlerror() {
    // Minimal stub; expand with FormatMessageA if you need details.
    return "dynamic loader error";
  }

#else

	#include <dlfcn.h>
#endif

namespace tflite {

// SharedLibrary provides a uniform set of APIs across different platforms to
// handle dynamic library operations
class SharedLibrary {
 public:
  static inline void* GetSymbol(const char* symbol) {
    return dlsym(RTLD_DEFAULT, symbol);
  }
  static inline const char* GetError() { return dlerror(); }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
