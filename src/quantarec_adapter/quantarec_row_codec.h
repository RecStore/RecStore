#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace recstore {
namespace quantarec_adapter {

/// Fixed-row embedding layout helpers for Quantarec PS wire format (float32 rows).
inline int64_t FlatEmbeddingDim(const std::vector<int64_t>& embedding_shape) {
  int64_t dim = 1;
  for (int64_t d : embedding_shape) {
    dim *= d;
  }
  return dim;
}

inline void ZeroFillRow(float* dst, int64_t embedding_dim) {
  std::memset(dst, 0, static_cast<size_t>(embedding_dim) * sizeof(float));
}

}  // namespace quantarec_adapter
}  // namespace recstore
