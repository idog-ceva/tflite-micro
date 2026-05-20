# third_party.cmake
#
# Fetches the third-party dependencies TFLM needs (flatbuffers, gemmlowp, ruy,
# kissfft) at the same versions pinned by tools/make/third_party_downloads.inc.
# Headers/sources land under the CMake build directory; nothing is downloaded
# into the source tree.

include(FetchContent)

# Where to apply patches from.
get_filename_component(_TFLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(_FLATBUFFERS_PATCH "${_TFLM_ROOT}/tensorflow/lite/micro/tools/make/flatbuffers.patch")
set(_KISSFFT_PATCH     "${_TFLM_ROOT}/third_party/kissfft/kissfft.patch")

# GNU `patch` is the most portable patch driver across hosts. We avoid
# `git apply` here because when the populated source dir sits inside an outer
# git working tree (as it does for in-repo builds — `_deps/<name>-src` is under
# the TFLM checkout), `git apply` interprets paths relative to the outer repo
# toplevel and silently skips all hunks.
find_program(PATCH_EXECUTABLE patch REQUIRED
  DOC "GNU patch (provided by 'patch' on Debian/Ubuntu, mingw-w64 'patch' on MSYS2)")

# --- flatbuffers -------------------------------------------------------------
# We only need flatbuffers' headers — skip its CMakeLists.txt (which would
# otherwise build flatc, the schema compiler, and a few other targets).
FetchContent_Declare(
  flatbuffers
  URL           https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.zip
  URL_HASH      MD5=023eca1e211d64007124420cd6be29c7
  PATCH_COMMAND ${PATCH_EXECUTABLE} -p1 -i "${_FLATBUFFERS_PATCH}"
  SOURCE_SUBDIR _no_cmake_here
)

# --- gemmlowp (header-only, fixed-point math) --------------------------------
FetchContent_Declare(
  gemmlowp
  URL      https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip
  URL_HASH MD5=7e8191b24853d75de2af87622ad293ba
)

# --- ruy (we only need profiler/instrumentation.h) ---------------------------
# ruy has a CMakeLists.txt at its root that pulls in a cpuinfo dependency. We
# only need a single header — skip add_subdirectory entirely by pointing
# SOURCE_SUBDIR at a non-existent sub-path.
FetchContent_Declare(
  ruy
  URL           https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip
  URL_HASH      MD5=abf7a91eb90d195f016ebe0be885bb6e
  SOURCE_SUBDIR _no_cmake_here
)

# --- kissfft -----------------------------------------------------------------
FetchContent_Declare(
  kissfft
  URL      https://github.com/mborgerding/kissfft/archive/refs/tags/v130.zip
  URL_HASH MD5=438ba1fef5783cc5f5f201395cc477ca
  PATCH_COMMAND ${PATCH_EXECUTABLE} -p1 -i "${_KISSFFT_PATCH}"
)

FetchContent_MakeAvailable(flatbuffers gemmlowp ruy kissfft)

# Export include directories. Order matters for kissfft because the wrappers
# in signal/src/kiss_fft_wrappers do `#include "kiss_fft.c"` and
# `#include "tools/kiss_fftr.c"` directly.
set(TFLM_THIRD_PARTY_INCLUDE_DIRS
  "${flatbuffers_SOURCE_DIR}/include"
  "${gemmlowp_SOURCE_DIR}"
  "${ruy_SOURCE_DIR}"
  "${kissfft_SOURCE_DIR}"
  CACHE INTERNAL "Third-party include directories"
)
