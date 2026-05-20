#include "model_runner.h"

#include <algorithm>
#include <cstdio>

#include "tensorflow/lite/micro/micro_log.h"

namespace model_runner {

Runner::Runner() = default;
Runner::~Runner() = default;

bool Runner::Load(const uint8_t* model_data, size_t model_size,
                  const Options& opts) {
  if (model_data == nullptr || model_size == 0) {
    std::fprintf(stderr, "error: empty model buffer\n");
    return false;
  }

  model_ = tflite::GetModel(model_data);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    std::fprintf(stderr,
                 "error: model schema version %u != expected %d\n",
                 model_->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  resolver_ = std::make_unique<tflite::TflmOpResolver>();
  if (tflite::CreateOpResolver(*resolver_) != kTfLiteOk) {
    std::fprintf(stderr, "error: CreateOpResolver failed\n");
    return false;
  }

  // Auto-grow the arena until AllocateTensors succeeds or we hit the cap.
  // Always destroy the previous interpreter and clear the arena vector
  // *before* resizing — the old interpreter holds the previous arena pointer
  // and would access it during destruction.
  //
  // MicroInterpreter's constructor crashes if the arena is too small to even
  // hold the allocator's bookkeeping (it calls SingleArenaBufferAllocator::Create
  // which returns nullptr on failure but the caller doesn't check). Floor the
  // size at 4 KiB so the failure mode is always a clean AllocateTensors error.
  constexpr size_t kMinArena = 4096;
  size_t size = std::max(opts.arena_initial_bytes, kMinArena);
  while (true) {
    interpreter_.reset();
    arena_.clear();
    arena_.shrink_to_fit();
    arena_.assign(size, 0);
    interpreter_ = std::make_unique<tflite::MicroInterpreter>(
        model_, *resolver_, arena_.data(), arena_.size());
    if (interpreter_->AllocateTensors() == kTfLiteOk) {
      return true;
    }
    if (size >= opts.arena_max_bytes) {
      std::fprintf(stderr,
                   "error: AllocateTensors failed even at arena cap %zu bytes\n",
                   opts.arena_max_bytes);
      return false;
    }
    const size_t next = size * 2;
    size = next > opts.arena_max_bytes ? opts.arena_max_bytes : next;
    std::fprintf(stderr, "info: retrying with arena=%zu bytes\n", size);
  }
}

bool Runner::Invoke() {
  return interpreter_->Invoke() == kTfLiteOk;
}

size_t Runner::NumInputs() const { return interpreter_->inputs_size(); }
size_t Runner::NumOutputs() const { return interpreter_->outputs_size(); }
TfLiteTensor* Runner::Input(size_t i) { return interpreter_->input(i); }
TfLiteTensor* Runner::Output(size_t i) { return interpreter_->output(i); }

namespace {
const char* TensorName(const tflite::Model* model, int32_t tensor_idx) {
  const auto* sg = model->subgraphs()->Get(0);
  const auto* t = sg->tensors()->Get(tensor_idx);
  const auto* n = t->name();
  return n ? n->c_str() : "";
}
}  // namespace

const char* Runner::InputName(size_t i) const {
  return TensorName(model_, interpreter_->inputs().Get(i));
}
const char* Runner::OutputName(size_t i) const {
  return TensorName(model_, interpreter_->outputs().Get(i));
}

}  // namespace model_runner
