#ifndef MODEL_RUNNER_IO_UTILS_H_
#define MODEL_RUNNER_IO_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace model_runner {

// Reads the whole file into a vector of bytes. Returns false on any error.
bool ReadFile(const std::string& path, std::vector<uint8_t>* out);

// Writes a buffer to disk, creating parent directories as needed.
bool WriteFile(const std::string& path, const void* data, size_t n);

// Fills a buffer with deterministic pseudo-random bytes derived from `seed`.
void FillRandomBytes(void* data, size_t n, uint32_t seed);

// Fills a buffer with zeros.
void FillZeroBytes(void* data, size_t n);

// Returns a human-readable name for the dtype.
const char* DTypeName(TfLiteType t);

// Returns "[1,96,96,1]"-style shape string.
std::string ShapeToString(const TfLiteIntArray* dims);

}  // namespace model_runner

#endif  // MODEL_RUNNER_IO_UTILS_H_
