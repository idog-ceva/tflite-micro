cmake_minimum_required(VERSION 3.20)
project(model_runner LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Quiet the FetchContent download timestamp warning on CMake >= 3.24.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# --- Third-party dependencies (flatbuffers, gemmlowp, ruy, kissfft) ----------
include(cmake/third_party.cmake)

# --- TFLM source set ---------------------------------------------------------
include(cmake/tflm_sources.cmake)

# ---------------------------------------------------------------------------
# tflm_micro static library
# ---------------------------------------------------------------------------
add_library(tflm_micro STATIC ${TFLM_SOURCES})

target_include_directories(tflm_micro
  PUBLIC
    ${TFLM_INCLUDE_DIRS}
    ${TFLM_THIRD_PARTY_INCLUDE_DIRS}
)

target_compile_definitions(tflm_micro
  PUBLIC
    TF_LITE_STATIC_MEMORY
    TF_LITE_DISABLE_X86_NEON
    TF_LITE_USE_CTIME
)

# TFLM is widely warning-noisy under recent gcc/clang. Don't fail on it.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tflm_micro PRIVATE
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-sign-compare
    -Wno-deprecated-declarations
    -Wno-missing-field-initializers
    -ffunction-sections
    -fdata-sections
    -fno-rtti
    -fno-exceptions
    -fno-threadsafe-statics
  )
endif()

# ---------------------------------------------------------------------------
# model_runner executable
# ---------------------------------------------------------------------------
add_executable(model_runner
  src/main.cc
  src/model_runner.cc
  src/io_utils.cc
)

target_include_directories(model_runner PRIVATE src)
target_link_libraries(model_runner PRIVATE tflm_micro)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(model_runner PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
  )
endif()

# MinGW: link all runtime libraries statically so the .exe is self-contained.
# Plain -static (rather than just -static-libgcc/-static-libstdc++) is required
# to also pull in libwinpthread-1.dll, which MinGW-w64's POSIX-threaded
# libgcc/libstdc++ depend on; otherwise the .exe fails to run with a missing
# libwinpthread-1.dll error.
if(MINGW)
  target_link_options(model_runner PRIVATE
    -static
  )
endif()

message(STATUS "model_runner: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "model_runner: TFLM sources = ${TFLM_ROOT}")
list(LENGTH TFLM_SOURCES _n_tflm_sources)
message(STATUS "model_runner: ${_n_tflm_sources} TFLM .cc files in tflm_micro")

# Mirror compile_commands.json into the source directory so clangd's default
# discovery finds it without needing OS-specific paths in .clangd. The file is
# already in .gitignore.
add_custom_target(copy_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
  BYPRODUCTS "${CMAKE_SOURCE_DIR}/compile_commands.json"
  COMMENT "Updating ${CMAKE_SOURCE_DIR}/compile_commands.json for clangd"
  VERBATIM
)
add_dependencies(copy_compile_commands model_runner)
