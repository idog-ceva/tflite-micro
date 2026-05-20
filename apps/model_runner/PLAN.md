# model_runner — Design & Workplan

Standalone CLI app that loads any tflite-micro–compatible `.tflite` model at
runtime, feeds inputs (from file or generated), runs inference, and writes
outputs. Built with CMake + Ninja so it imports cleanly into VSCode on Linux
and Windows using open-source toolchains only (gcc/clang on Linux; MinGW-w64
gcc or clang on Windows).

This document is the source-of-truth design + workplan for the implementation.
Pick up from here in a fresh session — the todo list at the bottom is the
exact list to feed into `TodoWrite`.

---

## 1. Goal & Scope

**MVP scope** (locked in with the user):

- Load any `.tflite` file from disk at runtime.
- Use the all-ops resolver — every op the build supports is available; no
  per-model resolver tailoring.
- Inputs: raw binary per input tensor, or generated (zeros / random).
- Outputs: written as raw binary, one file per output tensor.
- Stdout: only status + a small shape/dtype summary. No JSON, no .npy, no
  per-op timing, no `--describe`, no `--expected` comparison.
- Cross-platform: Linux (gcc, clang) and Windows (MinGW-w64 gcc, clang).
- VSCode importable: open the folder, CMake Tools picks up presets, no extra
  config needed to build/debug.

**Explicit non-goals for MVP:**

- No per-op profiling output (`tflm_benchmark` already does this).
- No model description / dry-run mode.
- No output-vs-expected comparison.
- No .npy or JSON I/O — raw binary only.
- No Bazel target. Make and Bazel users can already use the existing tools.

---

## 2. Why a new app vs. reusing existing tools

The repo already contains:

- [tensorflow/lite/micro/tools/benchmarking/generic_model_benchmark.cc](../../tensorflow/lite/micro/tools/benchmarking/generic_model_benchmark.cc)
  — loads a `.tflite` at runtime, but focused on profiling and wired to the
  Make build only.
- [tensorflow/lite/micro/tools/benchmarking/op_resolver.h](../../tensorflow/lite/micro/tools/benchmarking/op_resolver.h)
  — a ready "all-ops" `MicroMutableOpResolver<117>` that registers every
  builtin and signal op.

We **reuse the all-ops resolver verbatim** (include it directly from
`tools/benchmarking/op_resolver.h`) and build a thinner, user-facing runner
around it with a CMake build so it's VSCode-importable on Linux and Windows.

---

## 3. Directory Layout

All new code is contained under `apps/model_runner/` (repo root level):

```
apps/model_runner/
├── CMakeLists.txt               # top-level: TFLM static lib + app exe
├── CMakePresets.json            # linux-gcc, linux-clang, windows-mingw, windows-clang
├── PLAN.md                      # this file
├── README.md                    # user-facing build & run instructions
├── .vscode/
│   ├── settings.json            # CMake Tools defaults
│   ├── launch.json              # cppdbg debug configs (linux, windows-mingw)
│   └── extensions.json          # recommend cmake-tools, clangd
├── src/
│   ├── main.cc                  # arg parsing, entry point
│   ├── model_runner.{h,cc}      # MicroInterpreter wrapper + arena auto-grow
│   ├── io_utils.{h,cc}          # file read/write, zero/random fill
│   └── all_ops_resolver.h       # thin include of tools/benchmarking/op_resolver.h
└── cmake/
    ├── tflm_sources.cmake       # collects TFLM core + kernels + signal sources
    └── third_party.cmake        # FetchContent for flatbuffers, gemmlowp, ruy, kissfft
```

---

## 4. CLI Surface

```
model_runner [OPTIONS] <model.tflite>

  --input=FILE              raw binary, one per input tensor (repeatable)
  --input-mode=zero|random  if no --input given (default: zero)
  --seed=N                  RNG seed for --input-mode=random (default: 0)
  --arena=BYTES             arena size (default 1 MiB; auto-grow on failure)
  --arena-max=BYTES         auto-grow cap (default 64 MiB)
  --iterations=N            run inference N times (default 1)
  --output=DIR              write output_<i>.bin into DIR (default: cwd)
  -h, --help
```

**Stdout format** (one line per fact, easy to grep):

```
model: path/to/model.tflite (N bytes)
arena: 1048576 bytes used
inputs: 1
  input[0]: name="input" dtype=INT8 shape=[1,96,96,1] bytes=9216
outputs: 1
  output[0]: name="MobilenetV1/Predictions/Reshape_1" dtype=INT8 shape=[1,2] bytes=2
inference: 1 iteration(s), avg=X.XX ms
wrote: out/output_0.bin
```

---

## 5. Build System

- **CMake ≥ 3.20**, Ninja generator (single-config).
- **Defines** (PUBLIC on the TFLM lib): `TF_LITE_STATIC_MEMORY`,
  `TF_LITE_DISABLE_X86_NEON`, `FLATBUFFERS_LOCALE_INDEPENDENT=0` (replaces the
  flatbuffers.patch effect for the base.h tweak — verify rest of patch is
  inert for host builds; if not, apply patch via FetchContent PATCH_COMMAND).
- **C++17, C11**. No `-Werror` (upstream code has warnings under recent gcc).
- **TFLM static library target** (`tflm_micro`) built from a source set that
  mirrors [sources.inc](../../tensorflow/lite/micro/tools/make/sources.inc):
  - `tensorflow/lite/micro/*.cc`
  - `tensorflow/lite/micro/arena_allocator/*.cc`
  - `tensorflow/lite/micro/memory_planner/*.cc`
  - `tensorflow/lite/micro/tflite_bridge/*.cc`
  - `tensorflow/lite/micro/kernels/*.cc` (excluding `*_test.cc`, `*test_util*.cc`)
  - `signal/micro/kernels/*.cc` + `signal/src/*.cc`
  - Plus `find tensorflow -path tensorflow/lite/experimental -prune
    -o -path tensorflow/lite/micro -prune -o -name '*.cc'` — i.e. shared
    `tensorflow/...` sources outside `lite/experimental` and `lite/micro`.
  - **Excluded**: `MICROLITE_TEST_SRCS` from sources.inc, plus
    `MICROLITE_BENCHMARK_SRCS`, plus `tensorflow/lite/array.cc` (when
    `TF_LITE_STATIC_MEMORY` is set, per sources.inc lines 11-19).
- **Third-party** via `FetchContent` (versions pinned from
  [third_party_downloads.inc](../../tensorflow/lite/micro/tools/make/third_party_downloads.inc)):
  - flatbuffers `v25.9.23` (apply
    [tensorflow/lite/micro/tools/make/flatbuffers.patch](../../tensorflow/lite/micro/tools/make/flatbuffers.patch)
    via `PATCH_COMMAND`).
  - gemmlowp `719139ce755a0f31cbf1c37f7f98adcc7fc9f425`
  - ruy `d37128311b445e758136b8602d1bbd2a755e115d` (only need
    `profiler/instrumentation.h`)
  - kissfft `v130` (apply [third_party/kissfft/kissfft.patch](../../third_party/kissfft/kissfft.patch))

### CMakePresets.json

| preset name              | OS      | compiler                      | build type |
|--------------------------|---------|-------------------------------|------------|
| `linux-gcc-debug`        | Linux   | gcc                           | Debug      |
| `linux-gcc-release`      | Linux   | gcc                           | Release    |
| `linux-clang-release`    | Linux   | clang                         | Release    |
| `windows-mingw-release`  | Windows | MinGW-w64 gcc (MSYS2)         | Release    |
| `windows-clang-release`  | Windows | clang (MSYS2 or LLVM Windows) | Release    |

All presets use Ninja.

### Windows note

Open-source toolchain on Windows = MinGW-w64 from MSYS2:

```
pacman -S mingw-w64-x86_64-gcc \
          mingw-w64-x86_64-clang \
          mingw-w64-x86_64-ninja \
          mingw-w64-x86_64-cmake
```

No MSVC, no clang-cl. Code uses only `<fstream>` / `<filesystem>` with binary
mode, no POSIX-only APIs, no `LoadLibrary`/`dlopen`.

---

## 6. Key Implementation Notes

### Arena sizing
Arbitrary models can't have a fixed arena size. Strategy: start at
`--arena` (default 1 MiB), allocate, call `AllocateTensors()`. If it fails,
double and retry up to `--arena-max`. On success, log the size that worked.

### Op resolver
Direct `#include
"tensorflow/lite/micro/tools/benchmarking/op_resolver.h"` from
`src/all_ops_resolver.h` and call `tflite::CreateOpResolver(resolver)`. No
duplication.

### Inputs
- One `--input=FILE` per input tensor, in subgraph order. Mismatched count =
  error.
- Each file's byte length must equal `tensor->bytes`. Mismatch = error.
- If no `--input` given, fill all input tensors per `--input-mode`:
  - `zero`: memset 0
  - `random`: `std::mt19937` seeded by `--seed`, then byte-fill (works for
    any dtype; users who care about value range supply real files).

### Outputs
- For each output tensor, write `output_<i>.bin` into `--output` dir (create
  dir if missing). Raw byte copy.
- Also print a one-line summary per tensor to stdout.

### Source globbing vs explicit lists
Use `file(GLOB ...)` per directory at configure time. The hazard is that
`file(GLOB)` doesn't trigger reconfigure on new files — acceptable here
because the user reruns CMake after pulling. Globs exclude `*_test.cc` and
`*test_util*.cc`.

### Risks to verify during build
1. **flatbuffers.patch** — only the `FLATBUFFERS_LOCALE_INDEPENDENT` chunk is
   covered by our `-D`. Check remaining hunks; if they affect host builds,
   wire `PATCH_COMMAND` (use `git apply` if available, else `patch -p1`).
2. **kissfft.patch** — verify it's needed for the signal kernels we compile.
   If only relevant to specific kernels not used by typical models, may be
   deferrable.
3. **MinGW vs gcc flag drift** — keep toolchain-specific flags in presets,
   not in `CMakeLists.txt`.
4. **`tensorflow/lite/array.cc` must be excluded** when
   `TF_LITE_STATIC_MEMORY` is defined (per sources.inc:11-19).
5. **`kiss_fft.c` is included as a header in some places** — see sources.inc
   comment around line 42. May need `OBJECT` library trick if it shows up as
   a TU.

---

## 7. Workplan / Todo List

Feed this list verbatim into `TodoWrite` at the start of the next session.

1. **Scaffold `apps/model_runner/`** with empty `CMakeLists.txt`, `src/`,
   `cmake/`, `.vscode/`, `README.md`. (already partially done — the dirs
   exist; this PLAN.md is in place.)
2. **`cmake/tflm_sources.cmake`** — globs TFLM core + kernels + signal
   sources mirroring sources.inc; excludes `*_test.cc`, `*test_util*.cc`,
   `tensorflow/lite/array.cc`, and the explicit
   `MICROLITE_TEST_SRCS`/`MICROLITE_BENCHMARK_SRCS` sets.
3. **`cmake/third_party.cmake`** — FetchContent for flatbuffers (with patch),
   gemmlowp, ruy, kissfft (with patch), at the pinned versions/commits from
   `third_party_downloads.inc`.
4. **Top-level `CMakeLists.txt`** — `tflm_micro` static lib + `model_runner`
   exe; defines, include paths, C++17.
5. **`CMakePresets.json`** — linux-gcc-debug/release, linux-clang-release,
   windows-mingw-release, windows-clang-release. All Ninja.
6. **`src/io_utils.{h,cc}`** — file read/write, zero/random fill, simple
   tensor metadata pretty-print.
7. **`src/model_runner.{h,cc}`** — `MicroInterpreter` wrapper: load model,
   resolve ops, allocate tensors with arena auto-grow, set inputs, invoke,
   expose outputs.
8. **`src/main.cc`** — CLI parsing (handwritten, no external dep), wires
   everything together, prints status.
9. **`src/all_ops_resolver.h`** — single-line include of
   `tensorflow/lite/micro/tools/benchmarking/op_resolver.h`.
10. **`.vscode/{settings.json, launch.json, extensions.json}`** — CMake Tools
    defaults; `cppdbg` configurations for "Debug on Linux (gdb)" and "Debug
    on Windows (gdb from MinGW)"; recommended extensions: `ms-vscode.cmake-tools`,
    `llvm-vs-code-extensions.vscode-clangd`.
11. **`README.md`** — prerequisites per OS, build with each preset, VSCode
    open/import steps, CLI usage examples.
12. **Build smoke test on Linux** — `cmake --preset linux-gcc-release && cmake
    --build --preset linux-gcc-release`. Run against a sample `.tflite`
    (e.g. one of the embedded examples re-exported, or any model the user
    has handy). Verify it produces correct output bytes.
13. **Capture Windows-specific notes** discovered during the Linux build —
    add to README under "Windows gotchas" if anything needs a different flag
    or include path. (No Windows host to test directly; rely on the user's
    Windows test pass.)

---

## 8. Decisions log

(Captured from the design Q&A — don't relitigate without reason.)

| Question | Decision |
|---|---|
| App location | `apps/model_runner/` at repo root |
| Third-party deps | `FetchContent` (no reuse of `tools/make/downloads/`) |
| I/O format | Raw binary only — no JSON, no .npy |
| CLI surface | MVP minimal — no per-op timing, no `--describe`, no `--expected` |
| Build system | CMake + Ninja; presets for both OSes; no Bazel target |
| Op resolver | Include & reuse `tools/benchmarking/op_resolver.h` verbatim |

---

## 9. How to resume in a new session

Paste this into the new chat:

> Continue the `model_runner` implementation. The full design is in
> `apps/model_runner/PLAN.md`. Read it, then feed the workplan in section 7
> into TodoWrite and proceed from task 2 (`cmake/tflm_sources.cmake`) — task 1
> (scaffolding) is done.
