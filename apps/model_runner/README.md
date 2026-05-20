# model_runner

Standalone CLI that loads any `.tflite` model at runtime, runs inference with
the TFLM all-ops resolver, and writes outputs to disk.

This README is the operational guide — how to install prerequisites, build,
and run. For the design rationale and the source-of-truth workplan, see
[PLAN.md](PLAN.md).

---

## 1. Prerequisites

### Linux (Ubuntu/Debian)

```sh
sudo apt install cmake ninja-build git patch build-essential clang
```

- `cmake` ≥ 3.21 (for `CMakePresets.json` v3).
- `patch` is needed at build time — third-party sources are patched via
  GNU `patch -p1`, not `git apply` (the latter silently skips hunks when the
  populated source directory sits inside the outer TFLM git working tree).
- `clang` is only needed if you plan to use the `linux-clang-release` preset.

### Windows (MSYS2 / MinGW-w64)

Open the **MSYS2 MinGW64** shell (not the plain MSYS2 or UCRT64 shells) and:

```sh
pacman -S --needed \
  mingw-w64-x86_64-gcc \
  mingw-w64-x86_64-clang \
  mingw-w64-x86_64-ninja \
  mingw-w64-x86_64-cmake \
  patch
```

No MSVC, no clang-cl. Run all `cmake` commands from the MinGW64 shell so
that `gcc`, `g++`, `ninja`, `cmake`, and `patch` resolve to the MinGW
versions on `PATH`.

### Network

The first configure of each preset downloads four third-party archives
(flatbuffers, gemmlowp, ruy, kissfft) into the build directory via
`FetchContent`. After that, the build is fully offline.

---

## 2. Build

All commands are run from `apps/model_runner/`:

```sh
cd apps/model_runner
```

Available presets:

| Preset                    | Host    | Toolchain                | Build type |
|---------------------------|---------|--------------------------|------------|
| `linux-gcc-debug`         | Linux   | gcc                      | Debug      |
| `linux-gcc-release`       | Linux   | gcc                      | Release    |
| `linux-clang-release`     | Linux   | clang                    | Release    |
| `windows-mingw-release`   | Windows | MinGW-w64 gcc (MSYS2)    | Release    |
| `windows-clang-release`   | Windows | clang (MSYS2 / LLVM)     | Release    |

Configure and build any preset with two commands:

```sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release
```

The binary lands in `build/<preset>/model_runner` (`.exe` on Windows).
For example:

```sh
./build/linux-gcc-release/model_runner --help
```

To rebuild after pulling new code, just rerun the `cmake --build` step.
To re-run configure (e.g. after adding files to the globbed source dirs),
rerun the `cmake --preset` step. To start completely fresh, delete the
preset's build directory:

```sh
rm -rf build/linux-gcc-release
```

### What gets built

- `libtflm_micro.a` — TFLM static library: ~216 .cc files, with the
  all-ops resolver from `tensorflow/lite/micro/tools/benchmarking/op_resolver.h`
  (117 builtin + signal ops).
- `model_runner` — the CLI executable.

The flatbuffers and ruy archives are downloaded for their headers only;
their own CMakeLists.txt files are intentionally skipped via `SOURCE_SUBDIR`
so we don't compile unused tools (flatc, ruy's GEMM kernels, etc.).

---

## 3. Usage

```
model_runner [OPTIONS] <model.tflite>

  --input=FILE              raw binary, one per input tensor (repeatable, in
                            subgraph order); byte count must equal tensor->bytes
  --input-mode=zero|random  fallback when no --input is given (default: zero)
  --seed=N                  RNG seed for random fill (default: 0)
  --arena=BYTES             initial tensor-arena size  (default: 1048576 = 1 MiB,
                            floored at 4096)
  --arena-max=BYTES         auto-grow cap              (default: 67108864 = 64 MiB)
  --iterations=N            run inference N times      (default: 1)
  --output=DIR              write output_<i>.bin into DIR (default: cwd)
  -h, --help                show this help and exit
```

### Arena sizing

If `AllocateTensors()` fails at the initial arena size, the runner doubles
the arena and retries up to `--arena-max`. The size that actually worked is
printed under `arena:`. If you know what your model needs, pass
`--arena=<bytes>` directly to skip the retries (the value is reported in
the failure messages from the previous attempt).

### Inputs

Either provide one `--input=FILE` per input tensor (in subgraph order), or
let the runner fill inputs with zeros (default) or random bytes
(`--input-mode=random`). Mismatched count or per-file byte length are
errors. Random fill is a byte-level fill across all dtypes — supply real
files if you care about value ranges (e.g. for INT8 with non-default
zero-point).

### Outputs

Each output tensor is written as raw bytes to `<output-dir>/output_<i>.bin`
in subgraph order. The output directory is created if it doesn't exist.

### Stdout format

One line per fact, easy to grep:

```
model: path/to/model.tflite (300568 bytes)
arena: 1048576 bytes used
inputs: 1
  input[0]: name="input" dtype=INT8 shape=[1,96,96,1] bytes=9216
inference: 3 iteration(s), avg=5.256 ms
outputs: 1
  output[0]: name="MobilenetV1/Predictions/Reshape_1" dtype=INT8 shape=[1,2] bytes=2
wrote: out/output_0.bin
```

Exit code is `0` on success, `1` on runtime error (file I/O, allocation,
invoke, shape mismatch), `2` on bad CLI arguments.

---

## 4. Examples

Run a tiny scalar model with default zero-filled input:

```sh
./build/linux-gcc-release/model_runner \
  ../../tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  --output=out
```

Run person_detect three times against a random-filled 96×96 INT8 image and
write the 2-byte logits to `out/output_0.bin`:

```sh
./build/linux-gcc-release/model_runner \
  ../../tensorflow/lite/micro/models/person_detect.tflite \
  --input-mode=random --seed=42 --iterations=3 --output=out
```

Feed your own raw bytes (one file per input tensor, in subgraph order):

```sh
./build/linux-gcc-release/model_runner model.tflite \
  --input=input0.bin --input=input1.bin --output=out
```

Force the arena auto-grow path (start small, watch it ratchet up):

```sh
./build/linux-gcc-release/model_runner model.tflite --arena=4096 --output=out
```

---

## 5. VSCode

Open the `apps/model_runner` folder in VSCode. With the recommended
extensions installed (`ms-vscode.cmake-tools`,
`llvm-vs-code-extensions.vscode-clangd`), CMake Tools picks up
`CMakePresets.json` automatically:

1. **Ctrl+Shift+P → CMake: Select Configure Preset** → pick e.g.
   `linux-gcc-debug`.
2. **CMake: Build** (or F7).
3. **Run and Debug** → pick *Debug model_runner (Linux gdb)*; you'll be
   prompted for the model path.

Two debug launch configurations are pre-wired in `.vscode/launch.json` —
Linux (gdb) against `build/linux-gcc-debug/`, and Windows MinGW (gdb)
against `build/windows-mingw-release/`. Both prompt for the model path on
launch.

The repo's parent `.gitignore` excludes `.vscode/`, so these files are not
committed. They live in the local working tree as a convenience for opening
the app in VSCode.

---

## 6. Windows gotchas

- Use the **MSYS2 MinGW64** shell for the `windows-mingw-release` preset.
  The plain MSYS2 or UCRT64 shells use a different gcc identity than the
  preset detects.
- `patch` is required at build time. On MSYS2 it's the `patch` package.
- The MinGW build statically links `libstdc++` and `libgcc` so the
  resulting `.exe` runs without copying MSYS2 DLLs.
- For `windows-clang-release` using a standalone LLVM Windows install
  (not MSYS2), `git`, `patch`, and `ninja` must all be on `PATH` in the
  same shell — install them via MSYS2 or your package manager of choice.

---

## 7. Layout

```
apps/model_runner/
├── CMakeLists.txt           # tflm_micro static lib + model_runner exe
├── CMakePresets.json        # the 5 presets above
├── PLAN.md                  # design + workplan
├── README.md                # this file
├── .gitignore               # ignores build/ and compile_commands.json
├── cmake/
│   ├── tflm_sources.cmake   # globs the TFLM source set (mirrors sources.inc)
│   └── third_party.cmake    # FetchContent for flatbuffers, gemmlowp, ruy, kissfft
├── src/
│   ├── main.cc              # CLI parsing, entry point
│   ├── model_runner.{h,cc}  # MicroInterpreter wrapper + arena auto-grow
│   ├── io_utils.{h,cc}      # file I/O, fill helpers, tensor pretty-print
│   └── all_ops_resolver.h   # include of tools/benchmarking/op_resolver.h
└── .vscode/                 # CMake Tools defaults + cppdbg launch configs
```

---

## 8. Troubleshooting

**"patch: command not found"** — install GNU `patch`
(`sudo apt install patch` on Debian/Ubuntu, `pacman -S patch` on MSYS2).

**"Failed to allocate tail memory. Requested: N, available M, missing: K"** —
the arena is too small for your model. Either:

- Let the runner grow it: don't set `--arena` (default 1 MiB), and
  optionally raise `--arena-max` if 64 MiB isn't enough.
- Or pass `--arena=<bytes>` directly with a generous size — the error line
  tells you the deficit, so you can size it precisely on the next run.

**Build downloads keep retrying / time out** — `FetchContent` needs network
access on the first configure. Once `build/<preset>/_deps/` is populated,
no further downloads happen. If a download is interrupted, delete the
build directory and reconfigure.

**Configure says "TFLM sources = N .cc files" with a wrong-looking N** —
`cmake/tflm_sources.cmake` globs at configure time. If you add new source
files under `tensorflow/lite/micro/`, `signal/`, or the shared TF paths,
rerun `cmake --preset <name>` to repopulate the glob.

**Output bytes look wrong** — they're raw tensor bytes in subgraph order,
in the tensor's native dtype. To interpret them, check the
`output[i]: dtype=... shape=...` line in stdout. For an INT8 [1,2] output,
that's two signed bytes; for FLOAT32 [1,10], that's 40 little-endian
bytes representing 10 floats.
