#ifndef MODEL_RUNNER_ALL_OPS_RESOLVER_H_
#define MODEL_RUNNER_ALL_OPS_RESOLVER_H_

// Reuse the all-ops resolver from the existing benchmarking tool — every
// builtin and signal op is registered there as MicroMutableOpResolver<117>.
// See tensorflow/lite/micro/tools/benchmarking/op_resolver.h for the full list
// of ops, and CreateOpResolver() to populate the resolver.
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"

#endif  // MODEL_RUNNER_ALL_OPS_RESOLVER_H_
