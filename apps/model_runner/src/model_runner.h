#ifndef MODEL_RUNNER_MODEL_RUNNER_H_
#define MODEL_RUNNER_MODEL_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "all_ops_resolver.h"

namespace model_runner {

// Wraps a MicroInterpreter with arena auto-grow.
//
// Lifecycle: construct, then call Load(). On success the model is parsed and
// AllocateTensors has been called with a tensor arena that fits; access inputs
// via Inputs()/Outputs() and run Invoke().
class Runner {
 public:
  struct Options {
    size_t arena_initial_bytes = 1u << 20;    // 1 MiB
    size_t arena_max_bytes     = 64u << 20;   // 64 MiB
  };

  Runner();
  ~Runner();

  // Loads the model bytes (caller must keep `model_data` alive for the
  // lifetime of this object — typically a std::vector owned by main).
  // Tries arena_initial_bytes, doubling on failure up to arena_max_bytes.
  bool Load(const uint8_t* model_data, size_t model_size, const Options& opts);

  // Runs Invoke() once. Returns true on success.
  bool Invoke();

  // Tensor accessors. Valid after Load() succeeds.
  size_t NumInputs() const;
  size_t NumOutputs() const;
  TfLiteTensor* Input(size_t i);
  TfLiteTensor* Output(size_t i);

  // Tensor names looked up from the model (the runtime TfLiteTensor under
  // TF_LITE_STATIC_MEMORY drops its `name` field). Empty string if missing.
  const char* InputName(size_t i) const;
  const char* OutputName(size_t i) const;

  size_t ArenaSize() const { return arena_.size(); }

 private:
  std::vector<uint8_t> arena_;
  std::unique_ptr<tflite::TflmOpResolver> resolver_;
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  const tflite::Model* model_ = nullptr;
};

}  // namespace model_runner

#endif  // MODEL_RUNNER_MODEL_RUNNER_H_
