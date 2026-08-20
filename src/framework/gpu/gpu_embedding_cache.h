#pragma once

#include <cstdint>

#include <torch/extension.h>

namespace recstore::framework::gpu {

// Profiling struct.  The profile *collection* was removed from
// gpu_embedding_cache.cu by the external-release cleanup, but op_torch.cc
// still calls these functions.  We provide inline no-op stubs here so the
// code compiles without depending on the (removed) .cu definitions.  The
// profile values will always be zero — if profiling is needed later,
// restore the definitions in gpu_embedding_cache.cu.
struct GpuCacheProfile {
  double query_ms          = 0.0;
  double backend_lookup_ms = 0.0;
  double fill_ms           = 0.0;
  double update_ms         = 0.0;
  double hit_count         = 0.0;
  double invalidate_ms     = 0.0;
  double request_count     = 0.0;
  double miss_count        = 0.0;
};

inline GpuCacheProfile GetLastGpuCacheProfile() { return {}; }
inline void ResetLastGpuCacheProfile() {}
inline void AddGpuCacheBackendLookupMs(double /*ms*/) {}

bool EnableGpuCache(int64_t capacity, int64_t embedding_dim);
void DisableGpuCache();
void ClearGpuCache();
bool IsGpuCacheEnabled();

bool CanUseGpuCache(const torch::Tensor& keys, int64_t embedding_dim);

struct GpuCacheLookupResult {
  torch::Tensor values;
  torch::Tensor missing_keys_cpu;
  torch::Tensor missing_positions_cpu;
  int64_t missing_count = 0;
};

GpuCacheLookupResult
QueryGpuCache(const torch::Tensor& keys, int64_t embedding_dim);
void FillGpuCache(const torch::Tensor& keys_cuda,
                  const torch::Tensor& values_cuda);
void ScatterMissValues(torch::Tensor* output_values,
                       const torch::Tensor& missing_positions_cpu,
                       const torch::Tensor& miss_values_cuda);
bool ApplySgdUpdateGpuCache(const torch::Tensor& keys_cuda,
                            const torch::Tensor& grads_cuda,
                            double learning_rate);
void UpdateGpuCache(const torch::Tensor& keys_cuda,
                    const torch::Tensor& values_cuda);
void InvalidateGpuCache(const torch::Tensor& keys_cuda);

} // namespace recstore::framework::gpu
