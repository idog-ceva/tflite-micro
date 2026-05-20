#include "io_utils.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace model_runner {

namespace fs = std::filesystem;

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s for reading\n", path.c_str());
    return false;
  }
  const std::streamsize n = f.tellg();
  if (n < 0) return false;
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(n));
  if (n > 0 && !f.read(reinterpret_cast<char*>(out->data()), n)) {
    std::fprintf(stderr, "error: short read on %s\n", path.c_str());
    return false;
  }
  return true;
}

bool WriteFile(const std::string& path, const void* data, size_t n) {
  std::error_code ec;
  fs::path p(path);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path(), ec);
    // ec set if the path already exists — that's fine; we only error if the
    // subsequent open fails.
  }
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::fprintf(stderr, "error: cannot open %s for writing\n", path.c_str());
    return false;
  }
  if (n > 0) f.write(reinterpret_cast<const char*>(data), n);
  return static_cast<bool>(f);
}

void FillRandomBytes(void* data, size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  auto* bytes = static_cast<uint8_t*>(data);
  // Pull 32 bits at a time, slice into bytes.
  size_t i = 0;
  while (i + 4 <= n) {
    const uint32_t v = rng();
    bytes[i + 0] = static_cast<uint8_t>(v);
    bytes[i + 1] = static_cast<uint8_t>(v >> 8);
    bytes[i + 2] = static_cast<uint8_t>(v >> 16);
    bytes[i + 3] = static_cast<uint8_t>(v >> 24);
    i += 4;
  }
  if (i < n) {
    uint32_t v = rng();
    while (i < n) {
      bytes[i++] = static_cast<uint8_t>(v);
      v >>= 8;
    }
  }
}

void FillZeroBytes(void* data, size_t n) {
  std::memset(data, 0, n);
}

const char* DTypeName(TfLiteType t) {
  switch (t) {
    case kTfLiteNoType:        return "NONE";
    case kTfLiteFloat32:       return "FLOAT32";
    case kTfLiteInt32:         return "INT32";
    case kTfLiteUInt8:         return "UINT8";
    case kTfLiteInt64:         return "INT64";
    case kTfLiteString:        return "STRING";
    case kTfLiteBool:          return "BOOL";
    case kTfLiteInt16:         return "INT16";
    case kTfLiteComplex64:     return "COMPLEX64";
    case kTfLiteInt8:          return "INT8";
    case kTfLiteFloat16:       return "FLOAT16";
    case kTfLiteFloat64:       return "FLOAT64";
    case kTfLiteComplex128:    return "COMPLEX128";
    case kTfLiteUInt64:        return "UINT64";
    case kTfLiteResource:      return "RESOURCE";
    case kTfLiteVariant:       return "VARIANT";
    case kTfLiteUInt32:        return "UINT32";
    case kTfLiteUInt16:        return "UINT16";
    case kTfLiteInt4:          return "INT4";
    case kTfLiteBFloat16:      return "BFLOAT16";
    default:                   return "UNKNOWN";
  }
}

std::string ShapeToString(const TfLiteIntArray* dims) {
  if (dims == nullptr) return "[]";
  std::ostringstream s;
  s << "[";
  for (int i = 0; i < dims->size; ++i) {
    if (i) s << ",";
    s << dims->data[i];
  }
  s << "]";
  return s.str();
}

}  // namespace model_runner
