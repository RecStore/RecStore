#pragma once

#include <cstdint>

#include <torch/extension.h>

namespace recstore::framework::gpu {

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
// Best-effort in-place SGD: value -= lr * grad on present keys only.
// Missing keys are silently skipped — no Query, no missing report, no
// device synchronization, no cache invalidation.
void ApplySgdUpdateBestEffortGpuCache(const torch::Tensor& keys_cuda,
                                      const torch::Tensor& grads_cuda,
                                      double learning_rate);
void UpdateGpuCache(const torch::Tensor& keys_cuda,
                    const torch::Tensor& values_cuda);
void InvalidateGpuCache(const torch::Tensor& keys_cuda);

} // namespace recstore::framework::gpu
