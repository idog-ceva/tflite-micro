// model_runner — load a .tflite, run it, write outputs.
//
// See PLAN.md for full design notes.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "io_utils.h"
#include "model_runner.h"

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string model;
  std::vector<std::string> inputs;
  std::string input_mode = "zero";  // "zero" or "random"
  uint32_t seed = 0;
  size_t arena = 1u << 20;          // 1 MiB
  size_t arena_max = 64u << 20;     // 64 MiB
  int iterations = 1;
  std::string output_dir = ".";
};

void PrintUsage() {
  std::fprintf(stderr,
    "usage: model_runner [OPTIONS] <model.tflite>\n"
    "\n"
    "  --input=FILE              raw binary, one per input tensor (repeatable)\n"
    "  --input-mode=zero|random  if no --input given (default: zero)\n"
    "  --seed=N                  RNG seed for random mode (default: 0)\n"
    "  --arena=BYTES             initial arena size (default: 1048576)\n"
    "  --arena-max=BYTES         arena auto-grow cap (default: 67108864)\n"
    "  --iterations=N            run inference N times (default: 1)\n"
    "  --output=DIR              write <model>_output_<i>.bin into DIR (default: .)\n"
    "  -h, --help                show this message\n");
}

bool ParseSize(const std::string& s, size_t* out) {
  try {
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos, 10);
    if (pos != s.size()) return false;
    *out = static_cast<size_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseArgs(int argc, char** argv, Args* out) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      PrintUsage();
      std::exit(0);
    }
    auto eq = a.find('=');
    auto key = (eq == std::string::npos) ? a : a.substr(0, eq);
    auto val = (eq == std::string::npos) ? std::string() : a.substr(eq + 1);

    if (key == "--input") {
      out->inputs.push_back(val);
    } else if (key == "--input-mode") {
      if (val != "zero" && val != "random") {
        std::fprintf(stderr, "error: --input-mode must be zero or random\n");
        return false;
      }
      out->input_mode = val;
    } else if (key == "--seed") {
      size_t v;
      if (!ParseSize(val, &v)) return false;
      out->seed = static_cast<uint32_t>(v);
    } else if (key == "--arena") {
      if (!ParseSize(val, &out->arena)) return false;
    } else if (key == "--arena-max") {
      if (!ParseSize(val, &out->arena_max)) return false;
    } else if (key == "--iterations") {
      size_t v;
      if (!ParseSize(val, &v)) return false;
      out->iterations = static_cast<int>(v);
    } else if (key == "--output") {
      out->output_dir = val;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "error: unknown flag '%s'\n", a.c_str());
      return false;
    } else {
      if (!out->model.empty()) {
        std::fprintf(stderr, "error: multiple positional model paths given\n");
        return false;
      }
      out->model = a;
    }
  }
  if (out->model.empty()) {
    std::fprintf(stderr, "error: missing <model.tflite>\n");
    return false;
  }
  if (out->iterations < 1) out->iterations = 1;
  return true;
}

void PrintTensor(const char* role, size_t i, const char* name,
                 const TfLiteTensor* t) {
  std::printf("  %s[%zu]: name=\"%s\" dtype=%s shape=%s bytes=%zu\n",
              role, i,
              name ? name : "",
              model_runner::DTypeName(t->type),
              model_runner::ShapeToString(t->dims).c_str(),
              t->bytes);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    PrintUsage();
    return 2;
  }

  std::vector<uint8_t> model_bytes;
  if (!model_runner::ReadFile(args.model, &model_bytes)) return 1;
  std::printf("model: %s (%zu bytes)\n", args.model.c_str(), model_bytes.size());

  model_runner::Runner runner;
  model_runner::Runner::Options opts;
  opts.arena_initial_bytes = args.arena;
  opts.arena_max_bytes = args.arena_max;
  if (!runner.Load(model_bytes.data(), model_bytes.size(), opts)) return 1;
  std::printf("arena: %zu bytes used\n", runner.ArenaSize());

  const size_t n_in  = runner.NumInputs();
  const size_t n_out = runner.NumOutputs();

  // --- Inputs ---
  if (!args.inputs.empty() && args.inputs.size() != n_in) {
    std::fprintf(stderr,
                 "error: model has %zu input(s) but %zu --input file(s) given\n",
                 n_in, args.inputs.size());
    return 1;
  }
  std::printf("inputs: %zu\n", n_in);
  for (size_t i = 0; i < n_in; ++i) {
    TfLiteTensor* t = runner.Input(i);
    PrintTensor("input", i, runner.InputName(i), t);
    if (!args.inputs.empty()) {
      std::vector<uint8_t> buf;
      if (!model_runner::ReadFile(args.inputs[i], &buf)) return 1;
      if (buf.size() != t->bytes) {
        std::fprintf(stderr,
                     "error: input[%zu] file is %zu bytes, tensor expects %zu\n",
                     i, buf.size(), t->bytes);
        return 1;
      }
      std::memcpy(t->data.raw, buf.data(), buf.size());
    } else if (args.input_mode == "random") {
      model_runner::FillRandomBytes(t->data.raw, t->bytes,
                                    args.seed + static_cast<uint32_t>(i));
    } else {
      model_runner::FillZeroBytes(t->data.raw, t->bytes);
    }
  }

  // --- Invoke ---
  using clock = std::chrono::steady_clock;
  double total_ms = 0.0;
  for (int i = 0; i < args.iterations; ++i) {
    const auto t0 = clock::now();
    if (!runner.Invoke()) {
      std::fprintf(stderr, "error: Invoke() failed on iteration %d\n", i);
      return 1;
    }
    const auto t1 = clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  std::printf("inference: %d iteration(s), avg=%.3f ms\n",
              args.iterations, total_ms / args.iterations);

  // --- Outputs ---
  std::error_code ec;
  fs::create_directories(args.output_dir, ec);
  const std::string model_name = fs::path(args.model).filename().string();
  std::printf("outputs: %zu\n", n_out);
  for (size_t i = 0; i < n_out; ++i) {
    TfLiteTensor* t = runner.Output(i);
    PrintTensor("output", i, runner.OutputName(i), t);
    fs::path p = fs::path(args.output_dir) /
                 (model_name + "_output_" + std::to_string(i) + ".bin");
    if (!model_runner::WriteFile(p.string(), t->data.raw, t->bytes)) return 1;
    std::printf("wrote: %s\n", p.string().c_str());
  }

  return 0;
}
