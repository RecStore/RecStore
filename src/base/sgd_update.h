#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#endif

namespace recstore {
inline bool SparseCommitFmaEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RECSTORE_SPARSE_COMMIT_FMA");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

inline void SgdFmaScalar(
    float* values,
    const float* grads,
    size_t count,
    float learning_rate,
    bool initialize) {
  for (size_t i = 0; i < count; ++i)
    values[i] =
        std::fma(-learning_rate, grads[i], initialize ? 0.0f : values[i]);
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
inline bool SgdFmaAvx2Supported() {
  static const bool supported = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return supported;
}

__attribute__((target("avx2,fma"), noinline)) inline void
SgdFmaAvx2(float* values,
           const float* grads,
           size_t count,
           float learning_rate,
           bool initialize) {
  const auto scale = _mm256_set1_ps(-learning_rate);
  size_t i         = 0;
  for (; count - i >= 8; i += 8) {
    const auto old =
        initialize ? _mm256_setzero_ps() : _mm256_loadu_ps(values + i);
    _mm256_storeu_ps(
        values + i, _mm256_fmadd_ps(scale, _mm256_loadu_ps(grads + i), old));
  }
  if (i < count)
    SgdFmaScalar(values + i, grads + i, count - i, learning_rate, initialize);
}
#endif

// Only the final independent-row SGD update changes; accumulation is external.
inline void
ApplySgdRow(float* values,
            const float* grads,
            size_t count,
            float learning_rate,
            bool initialize = false) {
  if (SparseCommitFmaEnabled()) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (SgdFmaAvx2Supported()) {
      SgdFmaAvx2(values, grads, count, learning_rate, initialize);
      return;
    }
#endif
    SgdFmaScalar(values, grads, count, learning_rate, initialize);
    return;
  }
  // Preserve both original expressions, including missing-row signed zero.
  if (initialize) {
#pragma omp simd
    for (size_t i = 0; i < count; ++i)
      values[i] = -learning_rate * grads[i];
  } else {
#pragma omp simd
    for (size_t i = 0; i < count; ++i)
      values[i] -= learning_rate * grads[i];
  }
}
} // namespace recstore
