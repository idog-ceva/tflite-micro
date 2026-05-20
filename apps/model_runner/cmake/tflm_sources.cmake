# tflm_sources.cmake
#
# Collects the TFLM core, kernel, signal, and shared TFLite source files needed
# to build a self-contained host build of TFLM. Mirrors the lists computed by
# tensorflow/lite/micro/tools/make/sources.inc.
#
# Globs run at configure time. If new sources are added under the globbed
# directories, rerun CMake.

# Repo root: apps/model_runner/../..
get_filename_component(_TFLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(TFLM_ROOT "${_TFLM_ROOT}" CACHE INTERNAL "TFLM repo root")

# --- TFLM core ---------------------------------------------------------------
file(GLOB _tflm_core
  "${TFLM_ROOT}/tensorflow/lite/micro/*.cc"
  "${TFLM_ROOT}/tensorflow/lite/micro/arena_allocator/*.cc"
  "${TFLM_ROOT}/tensorflow/lite/micro/memory_planner/*.cc"
  "${TFLM_ROOT}/tensorflow/lite/micro/tflite_bridge/*.cc"
)

# --- TFLM kernels (top-level only; subdirs are optimized variants) -----------
file(GLOB _tflm_kernels
  "${TFLM_ROOT}/tensorflow/lite/micro/kernels/*.cc"
)

# --- Signal library ----------------------------------------------------------
file(GLOB _tflm_signal
  "${TFLM_ROOT}/signal/micro/kernels/*.cc"
  "${TFLM_ROOT}/signal/src/*.cc"
  "${TFLM_ROOT}/signal/src/kiss_fft_wrappers/*.cc"
)

# --- Shared tensorflow sources (outside lite/experimental and lite/micro) ----
# Mirrors:
#   find tensorflow -type d \( -path tensorflow/lite/experimental \
#                              -o -path tensorflow/lite/micro \) -prune -false \
#                    -o -name '*.cc' -o -name '*.c'
set(_tflm_shared
  "${TFLM_ROOT}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc"
  "${TFLM_ROOT}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc"
  "${TFLM_ROOT}/tensorflow/lite/core/api/flatbuffer_conversions.cc"
  "${TFLM_ROOT}/tensorflow/lite/core/api/tensor_utils.cc"
  "${TFLM_ROOT}/tensorflow/lite/core/c/common.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/common.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/quantization_util.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/reference/comparisons.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/runtime_shape.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_ctypes.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_utils.cc"
  "${TFLM_ROOT}/tensorflow/lite/kernels/kernel_util.cc"
)
# tensorflow/lite/array.cc is excluded when TF_LITE_STATIC_MEMORY is set
# (matches sources.inc:11-19).

set(TFLM_SOURCES
  ${_tflm_core}
  ${_tflm_kernels}
  ${_tflm_signal}
  ${_tflm_shared}
)

# --- Exclusions --------------------------------------------------------------
# 1. *_test.cc — unit tests (matches MICROLITE_TEST_SRCS pattern).
list(FILTER TFLM_SOURCES EXCLUDE REGEX "_test\\.cc$")
# 2. Other test helpers / generators not in the kernel source list.
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/conv_test_common\\.cc$")
# 3. Benchmark binaries (MICROLITE_BENCHMARK_SRCS) — we keep our own main.
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/benchmarks/.*\\.cc$")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/tools/benchmarking/.*benchmark\\.cc$")

set(TFLM_SOURCES "${TFLM_SOURCES}" CACHE INTERNAL "TFLM source files")

# Include directories that downstream targets (and TFLM's own sources) need.
set(TFLM_INCLUDE_DIRS
  "${TFLM_ROOT}"
  CACHE INTERNAL "TFLM include directories"
)
