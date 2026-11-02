#pragma once

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace recstore {

// Experimental FBGEMM 1.2.0 D128/world2 contract. Wire order is homogeneous
// little-endian <QHHHBB>; no key's descriptor/vectors may cross packet boundaries.
struct CanonicalSparseRecord {
  uint64_t key = 0;
  uint16_t count_rank0 = 0, count_rank1 = 0, vector_count = 0;
  uint8_t table_type = 0;  // 0: table-wise, 1: data-parallel.
  uint8_t reserved = 0;
};
static_assert(sizeof(CanonicalSparseRecord) == 16 &&
              offsetof(CanonicalSparseRecord, table_type) == 14,
              "CanonicalSparseRecord wire layout");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Canonical gradients require IEEE FP32");
inline constexpr size_t kCanonicalSparseDim = 128;

inline uint16_t CanonicalExpectedVectors(uint16_t c0, uint16_t c1,
                                         uint8_t type, uint32_t rank) {
  if (c0 > 1024 || c1 > 1024 || type > 1 || rank > 1 || !(c0 + c1))
    throw std::invalid_argument("Unsupported canonical gradient counts/type/rank");
  if (!(rank == 0 ? c0 : c1)) return 0;
  if (type == 1) return 1;
  size_t count = 0;
  const size_t n = size_t(c0) + c1;
  const auto leaf = [&](size_t lo, size_t hi) -> size_t {
    if (lo >= hi) return 0;
    if (hi <= c0) return rank == 0 ? 1 : 0;
    if (lo >= c0) return rank == 1 ? 1 : 0;
    return rank == 0 ? 1 : hi - c0;
  };
  if (n < 32) return static_cast<uint16_t>(leaf(0, n));
  for (size_t base = 0; base < n; base += 1024) {
    const size_t end = std::min(base + 1024, n);
    if (end <= c0) { count += rank == 0; continue; }
    if (base >= c0) { count += rank == 1; continue; }
    const size_t q = (end - base + 31) / 32;
    for (size_t j = 0; j < 32; ++j)
      count += leaf(base + j * q, std::min(base + (j + 1) * q, end));
  }
  return static_cast<uint16_t>(count);
}

inline void ValidateCanonicalSparseRecord(const CanonicalSparseRecord& r,
                                           uint32_t rank) {
  const auto expected = CanonicalExpectedVectors(
      r.count_rank0, r.count_rank1, r.table_type, rank);
  if (r.reserved || !expected || r.vector_count != expected)
    throw std::invalid_argument("Invalid canonical gradient record/vector count");
}

inline void ValidateCanonicalArithmeticEnvironment() {
  if (std::fegetround() != FE_TONEAREST)
    throw std::runtime_error("Canonical gradients require round-to-nearest");
#if defined(__SSE__)
  if (_mm_getcsr() & ((1u << 15) | (1u << 6)))
    throw std::runtime_error("Canonical CPU arithmetic requires FTZ/DAZ disabled");
#endif
}

// Disabled by default until the complete training lane has been validated.
inline bool CanonicalSimdEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RECSTORE_PS_CANONICAL_SIMD");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
inline bool CanonicalSimdSupported() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool supported = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return supported;
#else
  return false;
#endif
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
inline bool CanonicalEightFiniteAvx2(__m256 x) {
  const auto exponent = _mm256_set1_epi32(0x7f800000);
  return !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
      _mm256_and_si256(_mm256_castps_si256(x), exponent), exponent)));
}
// Integer exponent tests also work for byte-unaligned wire payloads.
__attribute__((target("avx2,fma"), noinline))
inline bool CanonicalAllFiniteAvx2(const void* values, size_t count) {
  const auto* bytes = static_cast<const char*>(values);
  size_t at = 0;
  for (; count - at >= 8; at += 8) {
    const auto bits = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(bytes + at * sizeof(float)));
    if (!CanonicalEightFiniteAvx2(_mm256_castsi256_ps(bits))) return false;
  }
  for (; at < count; ++at) {
    float value;
    std::memcpy(&value, bytes + at * sizeof(float), sizeof(value));
    if (!std::isfinite(value)) return false;
  }
  return true;
}
// SIMD spans independent columns; each column retains its original tree order.
__attribute__((target("avx2,fma"), noinline))
inline void CanonicalRowAddAvx2(float* dst, const float* src) {
  for (size_t col = 0; col < 128; col += 8) {
    const auto sum = _mm256_add_ps(_mm256_loadu_ps(dst + col), _mm256_loadu_ps(src + col));
    if (!CanonicalEightFiniteAvx2(sum))
      throw std::runtime_error("Canonical gradient reconstruction overflow");
    _mm256_storeu_ps(dst + col, sum);
  }
}
// Read-only precheck: the entire prefix still validates before any SGD write.
__attribute__((target("avx2,fma"), noinline))
inline void ValidateCanonicalSgdRowAvx2(const float* weights, const float* grads, float lr) {
  const auto scale = _mm256_set1_ps(-lr);
  for (size_t col = 0; col < 128; col += 8) {
    const auto w = _mm256_loadu_ps(weights + col), g = _mm256_loadu_ps(grads + col);
    if (!CanonicalEightFiniteAvx2(w) || !CanonicalEightFiniteAvx2(g) || !CanonicalEightFiniteAvx2(_mm256_fmadd_ps(scale, g, w)))
      throw std::runtime_error("Non-finite canonical SGD weight/result");
  }
}
#endif

inline bool CanonicalAllFinite(const void* values, size_t count) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (CanonicalSimdEnabled() && CanonicalSimdSupported())
    return CanonicalAllFiniteAvx2(values, count);
#endif
  const auto* bytes = static_cast<const char*>(values);
  for (size_t at = 0; at < count; ++at) {
    float value;
    std::memcpy(&value, bytes + at * sizeof(float), sizeof(value));
    if (!std::isfinite(value)) return false;
  }
  return true;
}

inline float CanonicalAdd(float a, float b) {
  // Materialize each FP32 addition; never reassociate or compile with fast-math.
  volatile float value = a + b;
  if (!std::isfinite(value))
    throw std::runtime_error("Canonical gradient reconstruction overflow");
  return value;
}

inline float CanonicalFlushSubnormal(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffu) < 0x00800000u) bits &= 0x80000000u;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline float CanonicalAtomicAdd(float a, float b) {
  // CUDA global atom.add.f32: signed FTZ for inputs and result, RNE addition.
  return CanonicalFlushSubnormal(CanonicalAdd(
      CanonicalFlushSubnormal(a), CanonicalFlushSubnormal(b)));
}

// Record pointers may be absent only when that rank's advertised count is zero.
// Input bytes can be unaligned. Rank-local vector order is CTA, then leaf, then
// rank1 raw crossing-leaf suffix occurrence order. TW inputs are predivided;
// DP inputs are local tree roots divided by world. No division occurs here.
inline void ReconstructCanonicalSparseGradient(
    const CanonicalSparseRecord* r0, const void* data0,
    const CanonicalSparseRecord* r1, const void* data1, float* output) {
  if ((!r0 && !r1) || !output)
    throw std::invalid_argument("Missing canonical gradient record/output");
  ValidateCanonicalArithmeticEnvironment();
  const auto& shape = r0 ? *r0 : *r1;
  if (r0) ValidateCanonicalSparseRecord(*r0, 0);
  if (r1) ValidateCanonicalSparseRecord(*r1, 1);
  if ((bool(r0) != bool(shape.count_rank0)) ||
      (bool(r1) != bool(shape.count_rank1)) ||
      (r0 && r1 && (r0->key != r1->key ||
                    r0->count_rank0 != r1->count_rank0 ||
                    r0->count_rank1 != r1->count_rank1 ||
                    r0->table_type != r1->table_type)))
    throw std::invalid_argument("Canonical rank record counts/type mismatch");
  if ((r0 && !data0) || (r1 && !data1))
    throw std::invalid_argument("Missing canonical gradient vectors");
  using Row = std::array<float, kCanonicalSparseDim>;
  const CanonicalSparseRecord* records[2] = {r0, r1};
  const char* data[2] = {static_cast<const char*>(data0), static_cast<const char*>(data1)};
  size_t used[2] = {0, 0};
  const auto take = [&](size_t rank, Row& row) {
    if (!records[rank] || used[rank] >= records[rank]->vector_count)
      throw std::invalid_argument("Truncated canonical gradient vector sequence");
    std::memcpy(row.data(), data[rank] + used[rank]++ * sizeof(Row), sizeof(Row));
    if (!CanonicalAllFinite(row.data(), row.size()))
      throw std::invalid_argument("Non-finite canonical gradient vector");
  };
  const auto add = [](Row& dst, const Row& src) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (CanonicalSimdEnabled() && CanonicalSimdSupported()) {
      CanonicalRowAddAvx2(dst.data(), src.data());
      return;
    }
#endif
    for (size_t col = 0; col < kCanonicalSparseDim; ++col)
      dst[col] = CanonicalAdd(dst[col], src[col]);
  };
  Row result{};
  if (shape.table_type == 1) {
    Row peer{};
    if (r0) take(0, result);
    if (r1) take(1, peer);
    add(result, peer); // Include the absent rank's explicit positive zero.
  } else {
    const size_t c0 = shape.count_rank0, n = c0 + shape.count_rank1;
    const auto leaf = [&](size_t lo, size_t hi, Row& out) {
      if (lo >= hi) return;
      if (hi <= c0) { take(0, out); return; }
      if (lo >= c0) { take(1, out); return; }
      take(0, out);
      Row raw;
      for (size_t i = c0; i < hi; ++i) { take(1, raw); add(out, raw); }
    };
    if (n < 32) {
      leaf(0, n, result);
    } else {
      std::array<Row, 2> roots{};
      size_t root_count = 0;
      for (size_t base = 0; base < n; base += 1024) {
        const size_t end = std::min(base + 1024, n);
        auto& root = roots[root_count++];
        if (end <= c0) { take(0, root); continue; }
        if (base >= c0) { take(1, root); continue; }
        const size_t q = (end - base + 31) / 32;
        std::array<Row, 32> leaves{};
        for (size_t j = 0; j < 32; ++j)
          leaf(base + j * q, std::min(base + (j + 1) * q, end), leaves[j]);
        for (size_t stride = 16; stride; stride /= 2)
          for (size_t j = 0; j < stride; ++j) add(leaves[j], leaves[j + stride]);
        root = leaves[0];
      }
      if (root_count == 1) result = roots[0];
      else
        for (const auto& root : roots)
          for (size_t col = 0; col < kCanonicalSparseDim; ++col)
            result[col] = CanonicalAtomicAdd(result[col], root[col]);
    }
  }
  for (size_t rank = 0; rank < 2; ++rank)
    if (used[rank] != (records[rank] ? records[rank]->vector_count : 0))
      throw std::invalid_argument("Unused canonical gradient vectors");
  std::memcpy(output, result.data(), sizeof(result));
}

// Called before any update in a prefix, after reading existing owned row views.
// The surrounding coordinator serializes commits; external writers are outside
// this experiment's contract. Missing rows are rejected by the CachePS wrapper.
inline void ValidateCanonicalSgdRow(const float* weights, const float* grads,
                                    float learning_rate) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (CanonicalSimdEnabled() && CanonicalSimdSupported()) {
    ValidateCanonicalSgdRowAvx2(weights, grads, learning_rate);
    return;
  }
#endif
  for (size_t col = 0; col < kCanonicalSparseDim; ++col)
    if (!std::isfinite(weights[col]) || !std::isfinite(grads[col]) ||
        !std::isfinite(std::fma(-learning_rate, grads[col], weights[col])))
      throw std::runtime_error("Non-finite canonical SGD weight/result");
}
} // namespace recstore
