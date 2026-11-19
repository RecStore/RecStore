#include "framework/pytorch/cache/cache_bindings.h"
#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

torch::Tensor
gpu_cache_lookup_flat_torch(const torch::Tensor& keys, int64_t dim) {
  return std::get<0>(CachedReadTorch(keys, dim));
}
std::tuple<torch::Tensor, torch::Tensor>
gpu_cache_lookup_flat_no_evict_torch(const torch::Tensor& keys, int64_t dim) {
  return CachedReadTorch(keys, dim, false, true);
}
void emb_write_values_torch(const torch::Tensor& keys,
                            const torch::Tensor& values) {
  CheckRows(keys, values);
  if (!keys.numel())
    return;
  auto cpu_keys = keys.cpu(), cpu_values = values.cpu();
  GetConcreteKVClientOp()->EmbWriteCached(
      ToRecTensor(cpu_keys, base::DataType::UINT64),
      ToRecTensor(cpu_values, base::DataType::FLOAT32),
      CacheActions(keys, values.size(1)),
      true);
}

bool enable_gpu_cache_torch(int64_t capacity, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return torch_cache::EnableGpuCache(capacity, embedding_dim);
#else
  (void)capacity;
  (void)embedding_dim;
  return false;
#endif
}

void disable_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  torch_cache::DisableGpuCache();
#endif
}

void clear_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  torch_cache::ResetGpuCache();
#endif
}

void prefill_gpu_cache_torch(const torch::Tensor& keys,
                             const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  CheckRows(keys, values, false);
  if (!keys.numel()) {
    return;
  }
  auto [k, v] = CudaCacheRows(keys, values);
  torch_cache::FillGpuCache(k, v);
#else
  (void)keys;
  (void)values;
#endif
}

torch::Tensor prefill_gpu_cache_no_evict_torch(const torch::Tensor& keys,
                                               const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  CheckRows(keys, values, false);
  if (!keys.numel()) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool));
  }
  auto [k, v] = CudaCacheRows(keys, values);
  return torch_cache::FillGpuCacheNoEvict(k, v);
#else
  (void)keys;
  (void)values;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

torch::Tensor contains_gpu_cache_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return torch_cache::ContainsGpuCache(keys);
#else
  (void)keys;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

int64_t get_gpu_cache_generation_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return static_cast<int64_t>(torch_cache::GetGpuCacheGeneration());
#else
  return 0;
#endif
}

void invalidate_gpu_cache_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  if (keys.numel() == 0) {
    return;
  }
  TORCH_CHECK(keys.is_cuda(), "invalidate_gpu_cache requires keys on CUDA");
  auto keys_cuda = keys;
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  torch_cache::InvalidateGpuCache(keys_cuda);
#else
  (void)keys;
#endif
}

torch::Tensor invalidate_gpu_cache_with_mask_torch(const torch::Tensor& keys) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  if (keys.numel() == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool));
  }
  TORCH_CHECK(keys.is_cuda(),
              "invalidate_gpu_cache_with_mask requires CUDA keys");
  auto keys_cuda = keys;
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  return torch_cache::InvalidateGpuCacheWithMask(keys_cuda);
#else
  (void)keys;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
gpu_cache_lookup_flat_assuming_hits_torch(const torch::Tensor& keys,
                                          int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(keys.is_cuda(),
              "gpu_cache_lookup_flat_assuming_hits requires CUDA keys");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");
  return torch_cache::LookupGpuCacheAssumingHits(keys, embedding_dim);
#else
  (void)keys;
  (void)embedding_dim;
  TORCH_CHECK(false, "GPU cache support is unavailable");
#endif
}

bool apply_sgd_update_gpu_cache_torch(const torch::Tensor& keys,
                                      const torch::Tensor& grads,
                                      double learning_rate) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  CheckRows(keys, grads, false);
  if (!keys.numel())
    return true;
  auto [k, v] = CudaCacheRows(keys, grads);
  return torch_cache::ApplySgdUpdateGpuCache(k, v, learning_rate);
#else
  (void)keys;
  (void)grads;
  (void)learning_rate;
  return false;
#endif
}

// Best-effort in-place SGD on the GPU cache: value -= lr * grad for keys
// present in the cache; missing keys silently skipped.  No Query, no
// missing report, no device synchronization.  This is the cache-side SGD
// primitive for controllers (BagPipe) that own their own cache policy.
void apply_sgd_update_gpu_cache_best_effort_torch(
    const torch::Tensor& keys,
    const torch::Tensor& grads,
    double learning_rate) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  CheckRows(keys, grads, false);
  if (!keys.numel()) {
    return;
  }
  auto [k, v] = CudaCacheRows(keys, grads);
  torch_cache::ApplySgdUpdateBestEffortGpuCache(k, v, learning_rate);
#else
  (void)keys;
  (void)grads;
  (void)learning_rate;
#endif
}

void set_gpu_cache_lookup_bypass_enabled_torch(bool enabled) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  g_cache_aware_op.SetBypassEnabled(enabled);
#else
  (void)enabled;
#endif
}

bool is_gpu_cache_lookup_bypass_enabled_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_cache_aware_op.BypassEnabled();
#else
  return false;
#endif
}

bool is_gpu_cache_lookup_bypassed_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_cache_aware_op.Bypassed();
#else
  return false;
#endif
}

void reset_gpu_cache_bypass_state_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  // Explicit policy reset: clear the latch on every thread, not just here.
  CacheAwareEpoch().fetch_add(1, std::memory_order_relaxed);
  g_cache_aware_op.Reset();
#endif
}

// ---- BagPipe-style GPU cache ops (query / update / invalidate / sgd) ----
std::tuple<torch::Tensor, torch::Tensor>
query_gpu_cache_torch(const torch::Tensor& keys, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (!torch_cache::IsGpuCacheEnabled() || keys.numel() == 0) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto dev  = keys.is_cuda() ? keys.device() : torch::kCPU;
    return {torch::empty({0, embedding_dim}, opts.device(dev)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))};
  }
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  auto keys_contig = keys.is_contiguous() ? keys : keys.contiguous();
  auto result      = torch_cache::QueryGpuCache(keys_contig, embedding_dim);
  return {result.values, result.missing_keys_cpu};
#else
  (void)keys;
  (void)embedding_dim;
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  return {torch::empty({0, 1}, opts),
          torch::empty({0}, opts.dtype(torch::kInt64))};
#endif
}

void update_gpu_cache_torch(const torch::Tensor& keys,
                            const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (!keys.numel())
    return;
  CheckRows(keys, values, false);
  auto [k, v] = CudaCacheRows(keys, values);
  torch_cache::UpdateGpuCache(k, v);
#else
  (void)keys;
  (void)values;
#endif
}

} // namespace

void RegisterGpuCacheOps(torch::Library& m) {
  m.def("gpu_cache_lookup_flat", gpu_cache_lookup_flat_torch);
  m.def("gpu_cache_lookup_flat_no_evict", gpu_cache_lookup_flat_no_evict_torch);
  m.def("emb_write_values", emb_write_values_torch);
  m.def("enable_gpu_cache", enable_gpu_cache_torch);
  m.def("disable_gpu_cache", disable_gpu_cache_torch);
  m.def("clear_gpu_cache", clear_gpu_cache_torch);
  m.def("prefill_gpu_cache", prefill_gpu_cache_torch);
  m.def("prefill_gpu_cache_no_evict", prefill_gpu_cache_no_evict_torch);
  m.def("contains_gpu_cache", contains_gpu_cache_torch);
  m.def("get_gpu_cache_generation", get_gpu_cache_generation_torch);
  m.def("invalidate_gpu_cache", invalidate_gpu_cache_torch);
  m.def("invalidate_gpu_cache_with_mask", invalidate_gpu_cache_with_mask_torch);
  m.def("gpu_cache_lookup_flat_assuming_hits",
        gpu_cache_lookup_flat_assuming_hits_torch);
  m.def("apply_sgd_update_gpu_cache", apply_sgd_update_gpu_cache_torch);
  m.def("apply_sgd_update_gpu_cache_best_effort",
        apply_sgd_update_gpu_cache_best_effort_torch);
  m.def("set_gpu_cache_lookup_bypass_enabled",
        set_gpu_cache_lookup_bypass_enabled_torch);
  m.def("is_gpu_cache_lookup_bypass_enabled",
        is_gpu_cache_lookup_bypass_enabled_torch);
  m.def("is_gpu_cache_lookup_bypassed", is_gpu_cache_lookup_bypassed_torch);
  m.def("reset_gpu_cache_bypass_state", reset_gpu_cache_bypass_state_torch);
  m.def("query_gpu_cache", query_gpu_cache_torch);
  m.def("update_gpu_cache", update_gpu_cache_torch);
}

} // namespace framework
} // namespace recstore
