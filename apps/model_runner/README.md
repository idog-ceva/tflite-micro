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

Install [MSYS2](https://www.msys2.org/), then open the **MSYS2 MinGW64** shell
(not the plain MSYS2 or UCRT64 shells) and run:

```sh
pacman -S --needed \
  mingw-w64-x86_64-gcc \
  mingw-w64-x86_64-gdb \
  mingw-w64-x86_64-clang \
  mingw-w64-x86_64-ninja \
  mingw-w64-x86_64-cmake \
  patch
```

| Package | Purpose |
|---|---|
| `mingw-w64-x86_64-gcc` | C/C++ compiler (`gcc`, `g++`) |
| `mingw-w64-x86_64-gdb` | Debugger (`gdb`) — required for F5 in VS Code |
| `mingw-w64-x86_64-clang` | Only needed for the `windows-clang-release` preset |
| `mingw-w64-x86_64-ninja` | Build system (`ninja`) |
| `mingw-w64-x86_64-cmake` | Build configurator — or install via `pip install cmake` |
| `patch` | GNU patch — applies source patches during configure |

No MSVC, no clang-cl. Run all `cmake` commands from the MinGW64 shell so
that `gcc`, `g++`, `ninja`, `cmake`, and `patch` resolve to the MinGW
versions on `PATH`.

> **VS Code users:** the `.vscode/settings.json` in this folder injects
> `C:\msys64\mingw64\bin` and `C:\msys64\usr\bin` into the CMake Tools
> process environment so configure and debug work without modifying your
> system PATH. If MSYS2 is installed somewhere other than `C:\msys64`,
> update the `cmake.environment.PATH` value in `.vscode/settings.json`.

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
| `windows-mingw-debug`     | Windows | MinGW-w64 gcc (MSYS2)    | Debug      |
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

### 5.1 Open

The fastest path is to open the multi-root workspace file, which gives you
both the app folder and the full repo in the file explorer:

```
File → Open Workspace from File... → apps/model_runner/model_runner.code-workspace
```

If you'd rather just see the app, `File → Open Folder...` →
`apps/model_runner/` works too — the same launch/task configs apply.

On first open, VSCode will offer the recommended extensions:

- `ms-vscode.cmake-tools` — preset-aware CMake integration
- `llvm-vs-code-extensions.vscode-clangd` — code intelligence

Install both.

### 5.2 Build (Ctrl+Shift+B)

The default `build` task picks the debug preset for the current OS:

- Linux → `linux-gcc-debug`
- Windows → `windows-mingw-debug`

It auto-configures if the build directory doesn't exist yet (`dependsOn:
configure`), so a fresh clone works on the first keystroke. Other build
tasks (`build: linux-gcc-release`, `build: windows-mingw-release`, …) are
available via `Tasks: Run Task`.

### 5.3 Debug (F5)

Run-and-Debug → **Debug model_runner (Linux gdb)** (or the MinGW one on
Windows). VSCode will:

1. Prompt for a `.tflite` model path.
2. Run the `build` task as `preLaunchTask` — same cross-OS task as
   Ctrl+Shift+B, so you always debug the freshly built binary against the
   current OS's debug preset.
3. Launch under `gdb`, stopping at any breakpoints you've set.

Set breakpoints anywhere in `src/*.cc` or in the TFLM sources — clangd
exposes the full TFLM compilation database, so `Go to Definition` works
into the framework as well as the app.

### 5.4 What ships in `.vscode/` and `.clangd`

- `.vscode/tasks.json` — cross-OS `build` + `configure` tasks (with
  `windows` overrides selecting the MinGW preset), plus explicit
  per-preset variants
- `.vscode/launch.json` — Linux gdb + Windows MinGW gdb launches, both
  wired to auto-build via `preLaunchTask: build`
- `.vscode/settings.json` — CMake-Tools preset mode, disables MS C/C++
  IntelliSense in favor of clangd, and sets the Windows integrated
  terminal to **MSYS2 MinGW64 bash** so tasks see the right `PATH`
  (`cmake`/`gcc`/`ninja`/`patch`). Adjust the `path` entry if MSYS2
  isn't at `C:\msys64`
- `.vscode/extensions.json` — recommended extensions
- `.clangd` — `CompilationDatabase: .` ; the build's
  `copy_compile_commands` target mirrors the active preset's
  `compile_commands.json` into the source root, so the same `.clangd`
  works for any OS / preset

The repo-root `.gitignore` excludes `.vscode/` and `.clangd` everywhere;
this app overrides that locally via `apps/model_runner/.gitignore` (with
`!.vscode/` and `!.clangd`) so the setup is committed and travels with
clones.

---

## 6. Windows gotchas

- Use the **MSYS2 MinGW64** shell for the `windows-mingw-*` presets from
  the CLI. The plain MSYS2 or UCRT64 shells use a different gcc identity
  than the preset detects.
- In VSCode the equivalent is the bundled
  `terminal.integrated.defaultProfile.windows = "MSYS2 MinGW64"` setting
  (see section 5.4) — without it, tasks running in PowerShell or cmd.exe
  won't find `cmake`/`gcc`/`ninja`/`patch`. Edit the profile's `path` if
  MSYS2 isn't at `C:\msys64`.
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

**"Unable to start debugging … program exited with code 0xc0000135"** —
a required MinGW runtime DLL wasn't found when GDB launched the executable.
Make sure `C:\msys64\mingw64\bin` is in `PATH` for the debugger process. In
VS Code this is handled by the `environment` entry in `.vscode/launch.json`;
if you moved MSYS2, update that path too.

**"MIDebuggerPath is not found"** — `gdb.exe` is missing. Install it with
`pacman -S mingw-w64-x86_64-gdb` in the MSYS2 MinGW64 shell.

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
