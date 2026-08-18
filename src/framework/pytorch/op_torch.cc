#include <torch/extension.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include "base/tensor.h"
#include "framework/op.h"
#include "ps/local_shm/local_shm_client.h"
// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
#include <glog/logging.h>

#ifdef RECSTORE_ENABLE_GPU_CACHE
#  include "framework/gpu/gpu_embedding_cache.h"
#endif

#if __has_include(<cuda_runtime_api.h>)
#  include <ATen/cuda/CUDAContext.h>
#  include <c10/cuda/CUDAException.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <cuda_runtime_api.h>
#  define RECSTORE_HAS_CUDA_RUNTIME_API 1
#else
#  define RECSTORE_HAS_CUDA_RUNTIME_API 0
#endif

namespace recstore {
namespace framework {

namespace {

bool IsLocalFastPathBackend(const std::string& backend) {
  return backend == "local_shm" || backend == "hierkv";
}

#ifdef RECSTORE_ENABLE_GPU_CACHE
struct PendingGpuCacheUpdate {
  torch::Tensor keys;
  torch::Tensor grads;
};

std::mutex g_pending_gpu_cache_updates_mu;
std::unordered_map<uint64_t, PendingGpuCacheUpdate>
    g_pending_gpu_cache_updates;
#endif

#ifdef RECSTORE_ENABLE_GPU_CACHE
constexpr int64_t kGpuCacheBypassMinRows            = 1024;
constexpr int kGpuCacheLowHitLimit                  = 1;
constexpr double kGpuCacheLowHitRatio               = 0.05;
thread_local int g_gpu_cache_low_hit_streak         = 0;
thread_local bool g_gpu_cache_lookup_bypassed       = false;
thread_local bool g_gpu_cache_lookup_bypass_enabled = true;

void SafeClearGpuCacheNoThrow();

void ResetGpuCacheBypassState() {
  g_gpu_cache_low_hit_streak  = 0;
  g_gpu_cache_lookup_bypassed = false;
}

bool ShouldBypassGpuCacheLookup(int64_t num_keys) {
  return g_gpu_cache_lookup_bypass_enabled &&
         num_keys >= kGpuCacheBypassMinRows &&
         g_gpu_cache_low_hit_streak >= kGpuCacheLowHitLimit;
}

void RecordGpuCacheLookupOutcome(
    int64_t num_keys, double hit_count, double request_count) {
  if (num_keys < kGpuCacheBypassMinRows || request_count <= 0.0) {
    return;
  }
  const double hit_ratio = hit_count / request_count;
  if (hit_ratio < kGpuCacheLowHitRatio) {
    ++g_gpu_cache_low_hit_streak;
  } else {
    g_gpu_cache_low_hit_streak  = 0;
    g_gpu_cache_lookup_bypassed = false;
  }
}

bool ShouldBypassGpuCacheMaintenance(int64_t num_keys) {
  return g_gpu_cache_lookup_bypass_enabled &&
         num_keys >= kGpuCacheBypassMinRows && g_gpu_cache_lookup_bypassed;
}

void MarkGpuCacheLookupBypassed() {
  if (!g_gpu_cache_lookup_bypassed) {
    SafeClearGpuCacheNoThrow();
    g_gpu_cache_low_hit_streak = kGpuCacheLowHitLimit;
  }
  g_gpu_cache_lookup_bypassed = true;
}

void EnsureGpuCacheSafeForLookup() {
  if (g_gpu_cache_lookup_bypassed) {
    SafeClearGpuCacheNoThrow();
    ResetGpuCacheBypassState();
  }
}

void SafeClearGpuCacheNoThrow() {
  try {
    gpu::ClearGpuCache();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to clear GPU cache: " << e.what();
  } catch (...) {
    LOG(WARNING) << "Failed to clear GPU cache: unknown exception";
  }
}

void SetGpuCacheLookupBypassEnabled(bool enabled) {
  g_gpu_cache_lookup_bypass_enabled = enabled;
  if (!enabled) {
    ResetGpuCacheBypassState();
  }
}

void MaintainGpuCacheAfterUpdateNoThrow(const torch::Tensor& keys,
                                        const torch::Tensor& grads,
                                        int64_t embedding_dim) {
  (void)grads;
  if (!gpu::IsGpuCacheEnabled()) {
    return;
  }
  if (ShouldBypassGpuCacheMaintenance(keys.numel())) {
    return;
  }
  if (gpu::CanUseGpuCache(keys, embedding_dim)) {
    try {
      gpu::InvalidateGpuCache(keys);
      return;
    } catch (const std::exception& e) {
      LOG(WARNING) << "GPU cache invalidation failed after backend update "
                      "succeeded; clearing cache and continuing: "
                   << e.what();
    } catch (...) {
      LOG(WARNING) << "GPU cache invalidation failed after backend update "
                      "succeeded; clearing cache and continuing: "
                   << "unknown exception";
    }
  }
  SafeClearGpuCacheNoThrow();
}
#endif

} // namespace

static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

static torch::TensorOptions PinnedCpuOptions(torch::ScalarType dtype) {
  return torch::TensorOptions()
      .device(torch::kCPU)
      .dtype(dtype)
      .pinned_memory(true);
}

static torch::Tensor StageCudaTensorToPinnedCpu(const torch::Tensor& tensor,
                                                torch::ScalarType dtype) {
  auto cpu_tensor = torch::empty(tensor.sizes(), PinnedCpuOptions(dtype));
  cpu_tensor.copy_(tensor.to(dtype), /*non_blocking=*/false);
  return cpu_tensor;
}

static torch::Tensor
StageCudaTensorToPinnedCpuAsyncNoCast(const torch::Tensor& tensor) {
  auto cpu_tensor =
      torch::empty(tensor.sizes(), PinnedCpuOptions(tensor.scalar_type()));
  cpu_tensor.copy_(tensor, /*non_blocking=*/true);
  return cpu_tensor;
}

static void SynchronizeCurrentCudaStreamForTensor(const torch::Tensor& tensor) {
#if RECSTORE_HAS_CUDA_RUNTIME_API
  if (!tensor.is_cuda()) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(tensor.device());
  C10_CUDA_CHECK(
      cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
#else
  (void)tensor;
#endif
}

static bool EnsurePinnedLocalShmPayload(const void* ptr, std::size_t bytes) {
#if !RECSTORE_HAS_CUDA_RUNTIME_API
  (void)ptr;
  (void)bytes;
  return false;
#else
  if (ptr == nullptr || bytes == 0) {
    return false;
  }
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const std::size_t page_bytes = static_cast<std::size_t>(page_size);
  const uintptr_t raw_begin    = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t raw_end      = raw_begin + bytes;
  const uintptr_t page_begin =
      raw_begin & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const uintptr_t page_end =
      (raw_end + page_bytes - 1U) & ~(static_cast<uintptr_t>(page_bytes) - 1U);
  const std::size_t required_bytes =
      static_cast<std::size_t>(page_end - page_begin);

  static std::mutex mu;
  static std::unordered_map<uintptr_t, std::size_t> registered_bytes_by_base;
  std::lock_guard<std::mutex> guard(mu);
  const std::size_t existing_bytes = registered_bytes_by_base[page_begin];
  if (existing_bytes >= required_bytes) {
    return true;
  }

  void* register_ptr = reinterpret_cast<void*>(page_begin + existing_bytes);
  const std::size_t register_bytes = required_bytes - existing_bytes;
  const cudaError_t err =
      cudaHostRegister(register_ptr, register_bytes, cudaHostRegisterPortable);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    LOG(WARNING) << "cudaHostRegister failed for local_shm payload: "
                 << cudaGetErrorString(err)
                 << " base=" << reinterpret_cast<void*>(page_begin)
                 << " bytes=" << required_bytes;
    return false;
  }
  registered_bytes_by_base[page_begin] = required_bytes;
  return true;
#endif
}

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  bool is_cuda     = keys.is_cuda();
  auto orig_device = keys.device();

  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool can_use_gpu_cache = gpu::CanUseGpuCache(keys, embedding_dim);
  const bool bypass_gpu_cache_lookup =
      can_use_gpu_cache && ShouldBypassGpuCacheLookup(num_keys);
  if (bypass_gpu_cache_lookup) {
    MarkGpuCacheLookupBypassed();
  }
  if (can_use_gpu_cache && !bypass_gpu_cache_lookup) {
    EnsureGpuCacheSafeForLookup();
    try {
      auto cache_result = gpu::QueryGpuCache(keys, embedding_dim);
      RecordGpuCacheLookupOutcome(
          num_keys,
          static_cast<double>(num_keys - cache_result.missing_count),
          static_cast<double>(num_keys));
      if (cache_result.missing_count == 0) {
        return cache_result.values;
      }

      auto missing_cpu_values = torch::empty(
          {cache_result.missing_count, embedding_dim},
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      base::RecTensor rec_missing_keys = ToRecTensor(
          cache_result.missing_keys_cpu.contiguous(), base::DataType::UINT64);
      base::RecTensor rec_missing_values =
          ToRecTensor(missing_cpu_values, base::DataType::FLOAT32);
      op->EmbRead(rec_missing_keys, rec_missing_values);

      auto miss_keys_cuda =
          cache_result.missing_keys_cpu.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          missing_cpu_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "GPU cache emb_read failed; clearing cache and falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
      } catch (...) {
      LOG(WARNING)
          << "GPU cache emb_read failed; clearing cache and falling back: "
          << "unknown exception";
      SafeClearGpuCacheNoThrow();
      }
  }
#endif

  torch::Tensor cpu_keys = is_cuda ? keys.cpu() : keys;

  auto cpu_values = torch::empty(
      {num_keys, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  op->EmbRead(rec_keys, rec_values);

  if (is_cuda) {
    return cpu_values.to(orig_device);
  }
  return cpu_values;
}

static std::shared_ptr<KVClientOp> GetConcreteKVClientOp() {
  auto op    = GetKVClientOp();
  auto kv_op = std::dynamic_pointer_cast<KVClientOp>(op);
  TORCH_CHECK(kv_op != nullptr, "storage backend is not KVClientOp");
  return kv_op;
}

static torch::Tensor BackendLocalLookupFlat(
    const std::shared_ptr<KVClientOp>& kv_op,
    const torch::Tensor& cpu_keys,
    const torch::Device& result_device,
    bool result_on_cuda,
    int64_t embedding_dim) {
  const int64_t num_keys   = cpu_keys.size(0);
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  if (kv_op->CurrentPSBackend() != "local_shm") {
    auto cpu_values =
        result_on_cuda
            ? torch::empty({num_keys, embedding_dim},
                           PinnedCpuOptions(torch::kFloat32))
            : torch::empty({num_keys, embedding_dim},
                           torch::TensorOptions()
                               .device(torch::kCPU)
                               .dtype(torch::kFloat32));
    base::RecTensor rec_values =
        ToRecTensor(cpu_values, base::DataType::FLOAT32);
    kv_op->LocalLookupFlat(rec_keys, rec_values);
    if (result_on_cuda) {
      return cpu_values.to(result_device, /*non_blocking=*/true);
    }
    return cpu_values;
  }

  if (!result_on_cuda) {
    auto cpu_values = torch::empty(
        {num_keys, embedding_dim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    base::RecTensor rec_values =
        ToRecTensor(cpu_values, base::DataType::FLOAT32);
    kv_op->LocalLookupFlat(rec_keys, rec_values);
    return cpu_values;
  }

  LocalShmFlatGetHandle handle;
  TORCH_CHECK(
      kv_op->SubmitLocalLookupFlat(rec_keys, embedding_dim, &handle) == 0,
      "Failed to submit local_shm flat lookup.");
  const int wait_ret = kv_op->WaitLocalLookupFlat(&handle);
  if (wait_ret != 0) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(false, "Failed to wait for local_shm flat lookup.");
  }
  const float* payload_values = handle.values;
  const int64_t payload_rows  = handle.num_rows;
  const int64_t payload_dim   = handle.embedding_dim;
  const std::size_t payload_bytes =
      static_cast<std::size_t>(handle.output_bytes);
  const int64_t expected_bytes =
      num_keys * embedding_dim * static_cast<int64_t>(sizeof(float));
  if (payload_values == nullptr || payload_rows != num_keys ||
      payload_dim != embedding_dim ||
      static_cast<int64_t>(payload_bytes) != expected_bytes) {
    kv_op->ReleaseLocalLookupFlat(&handle);
    TORCH_CHECK(false,
                "local_shm flat lookup returned unexpected payload metadata.");
  }
  const bool payload_is_pinned =
      EnsurePinnedLocalShmPayload(payload_values, payload_bytes);
  if (payload_is_pinned) {
    try {
      LocalShmFlatGetHandle handle_for_release = handle;
      auto cpu_view                            = torch::from_blob(
          const_cast<float*>(payload_values),
          {num_keys, embedding_dim},
          [kv_op, handle_for_release](void* /*unused*/) mutable {
            kv_op->ReleaseLocalLookupFlat(&handle_for_release);
          },
          PinnedCpuOptions(torch::kFloat32));
      return cpu_view.to(result_device, /*non_blocking=*/true);
    } catch (...) {
      kv_op->ReleaseLocalLookupFlat(&handle);
      throw;
    }
  }

  auto cpu_values = torch::empty(
      {num_keys, embedding_dim}, PinnedCpuOptions(torch::kFloat32));
  std::memcpy(cpu_values.data_ptr<float>(), payload_values, payload_bytes);
  kv_op->ReleaseLocalLookupFlat(&handle);
  return cpu_values.to(result_device, /*non_blocking=*/true);
}

torch::Tensor
local_lookup_flat_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  const bool is_cuda = keys.is_cuda();
  auto orig_device   = keys.device();

  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(IsLocalFastPathBackend(kv_op->CurrentPSBackend()),
              "local_lookup_flat requires local_shm or hierkv backend, but "
              "current backend is ",
              kv_op->CurrentPSBackend());

  const int64_t num_keys = keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool can_use_gpu_cache = gpu::CanUseGpuCache(keys, embedding_dim);
  const bool bypass_gpu_cache_lookup =
      can_use_gpu_cache && ShouldBypassGpuCacheLookup(num_keys);
  if (bypass_gpu_cache_lookup) {
    MarkGpuCacheLookupBypassed();
  }
  if (can_use_gpu_cache && !bypass_gpu_cache_lookup) {
    EnsureGpuCacheSafeForLookup();
    try {
      auto cache_result = gpu::QueryGpuCache(keys, embedding_dim);
      RecordGpuCacheLookupOutcome(
          num_keys,
          static_cast<double>(num_keys - cache_result.missing_count),
          static_cast<double>(num_keys));
      if (cache_result.missing_count == 0) {
        return cache_result.values;
      }

      auto miss_values = BackendLocalLookupFlat(
          kv_op,
          cache_result.missing_keys_cpu.contiguous(),
          orig_device,
          /*result_on_cuda=*/false,
          embedding_dim);
      auto miss_keys_cuda =
          cache_result.missing_keys_cpu.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          miss_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "GPU cache lookup failed; clearing cache and falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
    } catch (...) {
      LOG(WARNING)
          << "GPU cache lookup failed; clearing cache and falling back: "
          << "unknown exception";
      SafeClearGpuCacheNoThrow();
    }
  }
#endif

  torch::Tensor cpu_keys = keys;
  if (is_cuda) {
    cpu_keys = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
  }

  return BackendLocalLookupFlat(
      kv_op, cpu_keys, orig_device, is_cuda, embedding_dim);
}


// GPU-cache-accelerated flat lookup that works with ANY backend (BRPC, GRPC,
// RDMA, local_shm).  Cache hits are served from the GPU cache; misses are
// fetched via EmbRead and filled back into the cache.  This is the forward
// path used by the BagPipe controller when the local_shm fast path is
// unavailable, so the GPU cache is actually queried instead of bypassed.
torch::Tensor
gpu_cache_lookup_flat_torch(const torch::Tensor& keys,
                            int64_t embedding_dim) {
  const bool is_cuda = keys.is_cuda();
  auto orig_device   = keys.device();

  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = keys.size(0);
  if (num_keys == 0) {
    return torch::empty(
        {0, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32));
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool can_use_gpu_cache = gpu::CanUseGpuCache(keys, embedding_dim);
  const bool bypass_gpu_cache_lookup =
      can_use_gpu_cache && ShouldBypassGpuCacheLookup(num_keys);
  if (bypass_gpu_cache_lookup) {
    MarkGpuCacheLookupBypassed();
  }
  if (can_use_gpu_cache && !bypass_gpu_cache_lookup) {
    EnsureGpuCacheSafeForLookup();
    try {
      auto cache_result = gpu::QueryGpuCache(keys, embedding_dim);
      RecordGpuCacheLookupOutcome(
          num_keys,
          static_cast<double>(num_keys - cache_result.missing_count),
          static_cast<double>(num_keys));
      if (cache_result.missing_count == 0) {
        return cache_result.values;
      }

      // Fetch misses via EmbRead (works with BRPC / GRPC / RDMA).
      auto miss_cpu_keys = cache_result.missing_keys_cpu.contiguous();
      const int64_t miss_count = miss_cpu_keys.size(0);
      auto miss_cpu_values = torch::empty(
          {miss_count, embedding_dim},
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      auto op = GetKVClientOp();
      base::RecTensor rec_miss_keys =
          ToRecTensor(miss_cpu_keys, base::DataType::UINT64);
      base::RecTensor rec_miss_values =
          ToRecTensor(miss_cpu_values, base::DataType::FLOAT32);
      op->EmbRead(rec_miss_keys, rec_miss_values);

      auto miss_keys_cuda =
          miss_cpu_keys.to(orig_device, /*non_blocking=*/false);
      auto miss_values_cuda =
          miss_cpu_values.to(orig_device, /*non_blocking=*/false);
      gpu::FillGpuCache(miss_keys_cuda, miss_values_cuda);
      gpu::ScatterMissValues(&cache_result.values,
                             cache_result.missing_positions_cpu,
                             miss_values_cuda);
      return cache_result.values;
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "gpu_cache_lookup_flat: cache lookup failed; falling back: "
          << e.what();
      SafeClearGpuCacheNoThrow();
    } catch (...) {
      LOG(WARNING)
          << "gpu_cache_lookup_flat: cache lookup failed; falling back";
      SafeClearGpuCacheNoThrow();
    }
  }
#endif

  // Fallback: direct EmbRead (no GPU cache).
  torch::Tensor cpu_keys = keys;
  if (is_cuda) {
    cpu_keys = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
  }
  auto op = GetKVClientOp();
  auto cpu_values = torch::empty(
      {cpu_keys.size(0), embedding_dim},
      is_cuda ? PinnedCpuOptions(torch::kFloat32)
              : torch::TensorOptions()
                    .device(torch::kCPU)
                    .dtype(torch::kFloat32));
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);
  op->EmbRead(rec_keys, rec_values);
  if (is_cuda) {
    return cpu_values.to(orig_device, /*non_blocking=*/true);
  }
  return cpu_values;
}

// Async prefetch: returns a unique prefetch id (uint64_t)
int64_t emb_prefetch_torch(const torch::Tensor& keys) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  auto op                = GetKVClientOp();
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  // Dummy values tensor (unused by backend prefetch implementation)
  auto dummy_vals = torch::empty({0, 0}, keys.options().dtype(torch::kFloat32));
  base::RecTensor rec_vals = ToRecTensor(dummy_vals, base::DataType::FLOAT32);
  uint64_t pid             = op->EmbPrefetch(rec_keys, rec_vals);
  return static_cast<int64_t>(pid);
}

// Wait for prefetch and return result tensor [N, embedding_dim] on CPU
torch::Tensor
emb_wait_result_torch(int64_t prefetch_id, int64_t embedding_dim) {
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");
  auto op = GetKVClientOp();
  op->WaitForPrefetch(static_cast<uint64_t>(prefetch_id));
  auto owned = std::make_shared<base::RecTensor>(
      std::vector<int64_t>{0, embedding_dim}, base::DataType::FLOAT32);
  op->GetPretchResult(static_cast<uint64_t>(prefetch_id), *owned);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const int64_t L = owned->dim() == 2 ? owned->shape(0) : 0;
  if (L == 0) {
    return torch::empty({0, embedding_dim}, options);
  }
  return torch::from_blob(
      owned->data(), {L, embedding_dim}, [owned](void*) {}, options);
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse "
      "optimizer.");
}

void emb_update_table_torch(const std::string& table_name,
                            const torch::Tensor& keys,
                            const torch::Tensor& grads) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and grads tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys  = keys;
  torch::Tensor cpu_grads = grads;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  if (grads.is_cuda()) {
    cpu_grads = grads.cpu();
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  op->EmbUpdate(table_name, rec_keys, rec_grads);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  MaintainGpuCacheAfterUpdateNoThrow(keys, grads, grads.size(1));
#endif
}

int64_t emb_update_async_torch(const std::string& table_name,
                               const torch::Tensor& keys,
                               const torch::Tensor& grads) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and grads tensors must have the same number of entries");

  torch::Tensor cpu_keys  = keys.is_cuda() ? keys.cpu() : keys;
  torch::Tensor cpu_grads = grads.is_cuda() ? grads.cpu() : grads;
  auto kv_op              = GetConcreteKVClientOp();
  TORCH_CHECK(kv_op->CurrentPSBackend() == "rdma",
              "emb_update_async requires the RDMA backend");
  const uint64_t update_id = kv_op->EmbUpdateAsync(
      table_name,
      ToRecTensor(cpu_keys, base::DataType::UINT64),
      ToRecTensor(cpu_grads, base::DataType::FLOAT32));

#ifdef RECSTORE_ENABLE_GPU_CACHE
  std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
  const auto [it, inserted] = g_pending_gpu_cache_updates.emplace(
      update_id, PendingGpuCacheUpdate{keys, grads});
  TORCH_CHECK(inserted, "Duplicate asynchronous RDMA update handle");
#endif
  return static_cast<int64_t>(update_id);
}

void emb_update_wait_torch(int64_t update_id) {
  TORCH_CHECK(update_id > 0, "update_id must be positive");
  auto kv_op = GetConcreteKVClientOp();
  try {
    kv_op->WaitForEmbUpdate(static_cast<uint64_t>(update_id));
  } catch (...) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
    {
      std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
      g_pending_gpu_cache_updates.erase(static_cast<uint64_t>(update_id));
    }
    SafeClearGpuCacheNoThrow();
#endif
    throw;
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  PendingGpuCacheUpdate pending;
  {
    std::lock_guard<std::mutex> guard(g_pending_gpu_cache_updates_mu);
    const auto it =
        g_pending_gpu_cache_updates.find(static_cast<uint64_t>(update_id));
    TORCH_CHECK(it != g_pending_gpu_cache_updates.end(),
                "Missing GPU cache state for asynchronous RDMA update");
    pending = std::move(it->second);
    g_pending_gpu_cache_updates.erase(it);
  }
  MaintainGpuCacheAfterUpdateNoThrow(
      pending.keys, pending.grads, pending.grads.size(1));
#endif
}

void local_update_flat_torch(const std::string& table_name,
                             const torch::Tensor& keys,
                             const torch::Tensor& grads) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and grads tensors must have the same number of entries");

  auto kv_op = GetConcreteKVClientOp();
  TORCH_CHECK(IsLocalFastPathBackend(kv_op->CurrentPSBackend()),
              "local_update_flat requires local_shm or hierkv backend, but "
              "current backend is ",
              kv_op->CurrentPSBackend());

  if (keys.size(0) == 0) {
    return;
  }

  torch::Tensor cpu_keys = keys;
  const bool can_async_stage_cuda =
      (keys.is_cuda() || grads.is_cuda()) &&
      (!keys.is_cuda() || !grads.is_cuda() || keys.device() == grads.device());
  bool staged_cuda_async = false;
  if (keys.is_cuda()) {
    if (can_async_stage_cuda) {
      cpu_keys          = StageCudaTensorToPinnedCpuAsyncNoCast(keys);
      staged_cuda_async = true;
    } else {
      cpu_keys = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
    }
  }
  torch::Tensor cpu_grads = grads;
  if (grads.is_cuda()) {
    if (can_async_stage_cuda) {
      cpu_grads         = StageCudaTensorToPinnedCpuAsyncNoCast(grads);
      staged_cuda_async = true;
    } else {
      cpu_grads = StageCudaTensorToPinnedCpu(grads, torch::kFloat32);
    }
  }
  if (staged_cuda_async) {
    SynchronizeCurrentCudaStreamForTensor(keys.is_cuda() ? keys : grads);
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  try {
    kv_op->LocalUpdateFlat(table_name, rec_keys, rec_grads);
  } catch (...) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
    if (gpu::IsGpuCacheEnabled()) {
      SafeClearGpuCacheNoThrow();
    }
#endif
    throw;
  }

#ifdef RECSTORE_ENABLE_GPU_CACHE
  MaintainGpuCacheAfterUpdateNoThrow(keys, grads, grads.size(1));
#endif
}

bool warmup_local_lookup_flat_cuda_region_torch() {
  auto kv_op                = GetConcreteKVClientOp();
  const void* payload_base  = nullptr;
  std::size_t payload_bytes = 0;
  if (!kv_op->GetLocalLookupFlatPayloadRegion(&payload_base, &payload_bytes)) {
    return false;
  }
  return EnsurePinnedLocalShmPayload(payload_base, payload_bytes);
}

int64_t init_embedding_table_torch(const std::string& table_name,
                                   int64_t num_embeddings,
                                   int64_t embedding_dim,
                                   int64_t table_id = 0) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");
  TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");
  TORCH_CHECK(table_id >= 0, "table_id must be non-negative");

  EmbeddingTableConfig cfg{};
  cfg.num_embeddings = static_cast<uint64_t>(num_embeddings);
  cfg.embedding_dim  = static_cast<uint64_t>(embedding_dim);
  cfg.table_id       = static_cast<uint64_t>(table_id);

  auto kv_op      = GetConcreteKVClientOp();
  const int64_t tag = kv_op->InitEmbeddingTable(table_name, cfg);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (tag >= 0 && gpu::IsGpuCacheEnabled()) {
    SafeClearGpuCacheNoThrow();
  }
#endif
  return tag;
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must have dtype float32");
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "Keys and Values tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  if (values.is_cuda()) {
    cpu_values = values.cpu();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  op->EmbWrite(rec_keys, rec_values);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (gpu::IsGpuCacheEnabled()) {
    SafeClearGpuCacheNoThrow();
  }
#endif
}


void emb_write_values_torch(const torch::Tensor& keys,
                            const torch::Tensor& values) {
  // Direct value-set to the PS for a subset of keys, with *per-key* GPU
  // cache invalidation (not a full clear).  Used by the BagPipe eviction
  // writeback path to push locally-updated cache values back to the PS
  // without disturbing other cached entries.  Mirrors emb_write_torch but
  // replaces SafeClearGpuCacheNoThrow() with InvalidateGpuCache(keys).
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must be float32");
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "Keys and Values tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  if (values.is_cuda()) {
    cpu_values = values.cpu();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  op->EmbWrite(rec_keys, rec_values);
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (gpu::IsGpuCacheEnabled()) {
    int64_t embedding_dim = values.size(1);
    if (keys.is_cuda() && gpu::CanUseGpuCache(keys, embedding_dim)) {
      try {
        gpu::InvalidateGpuCache(keys);
      } catch (...) {
        SafeClearGpuCacheNoThrow();
      }
    } else {
      SafeClearGpuCacheNoThrow();
    }
  }
#endif
}

void set_ps_config_torch(const std::string& host, int64_t port) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSConfig(host, static_cast<int>(port));
}

void set_ps_backend_torch(const std::string& backend) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSBackend(backend);
}

std::string current_ps_backend_torch() {
  auto kv_op = GetConcreteKVClientOp();
  return kv_op->CurrentPSBackend();
}

bool enable_gpu_cache_torch(int64_t capacity, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  const bool enabled = gpu::EnableGpuCache(capacity, embedding_dim);
  if (enabled) {
    ResetGpuCacheBypassState();
  }
  return enabled;
#else
  (void)capacity;
  (void)embedding_dim;
  return false;
#endif
}

void disable_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::DisableGpuCache();
  ResetGpuCacheBypassState();
#endif
}

void clear_gpu_cache_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ClearGpuCache();
  ResetGpuCacheBypassState();
#endif
}

void prefill_gpu_cache_torch(const torch::Tensor& keys,
                             const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  TORCH_CHECK(values.dim() == 2, "values must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must have dtype float32");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "keys and values must have the same number of rows");
  if (keys.numel() == 0) {
    return;
  }
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "prefill_gpu_cache requires keys or values on CUDA");
  const auto cache_device = values.is_cuda() ? values.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto values_cuda        = values.is_cuda() ? values : values.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!values_cuda.is_contiguous()) {
    values_cuda = values_cuda.contiguous();
  }
  gpu::FillGpuCache(keys_cuda, values_cuda);
#else
  (void)keys;
  (void)values;
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
  gpu::InvalidateGpuCache(keys_cuda);
#else
  (void)keys;
#endif
}

bool apply_sgd_update_gpu_cache_torch(const torch::Tensor& keys,
                                      const torch::Tensor& grads,
                                      double learning_rate) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  TORCH_CHECK(grads.dim() == 2, "grads must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "grads must have dtype float32");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "keys and grads must have the same number of rows");
  if (keys.numel() == 0) {
    return true;
  }
  TORCH_CHECK(keys.is_cuda() || grads.is_cuda(),
              "apply_sgd_update_gpu_cache requires keys or grads on CUDA");
  const auto cache_device = grads.is_cuda() ? grads.device() : keys.device();
  auto keys_cuda          = keys.is_cuda() ? keys : keys.to(cache_device);
  auto grads_cuda         = grads.is_cuda() ? grads : grads.to(cache_device);
  if (!keys_cuda.is_contiguous()) {
    keys_cuda = keys_cuda.contiguous();
  }
  if (!grads_cuda.is_contiguous()) {
    grads_cuda = grads_cuda.contiguous();
  }
  return gpu::ApplySgdUpdateGpuCache(keys_cuda, grads_cuda, learning_rate);
#else
  (void)keys;
  (void)grads;
  (void)learning_rate;
  return false;
#endif
}

void set_gpu_cache_lookup_bypass_enabled_torch(bool enabled) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  SetGpuCacheLookupBypassEnabled(enabled);
#else
  (void)enabled;
#endif
}

bool is_gpu_cache_lookup_bypass_enabled_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_gpu_cache_lookup_bypass_enabled;
#else
  return false;
#endif
}

bool is_gpu_cache_lookup_bypassed_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  return g_gpu_cache_lookup_bypassed;
#else
  return false;
#endif
}

void reset_gpu_cache_bypass_state_torch() {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  ResetGpuCacheBypassState();
#endif
}


// ---- BagPipe-style GPU cache ops (query / update / invalidate / sgd) ----

std::tuple<torch::Tensor, torch::Tensor>
query_gpu_cache_torch(const torch::Tensor& keys, int64_t embedding_dim) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (!gpu::IsGpuCacheEnabled() || keys.numel() == 0) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto dev  = keys.is_cuda() ? keys.device() : torch::kCPU;
    return {torch::empty({0, embedding_dim}, opts.device(dev)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64))};
  }
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "keys must have dtype int64");
  auto keys_contig = keys.is_contiguous() ? keys : keys.contiguous();
  auto result      = gpu::QueryGpuCache(keys_contig, embedding_dim);
  return {result.values, result.missing_keys_cpu};
#else
  (void)keys;
  (void)embedding_dim;
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  return {torch::empty({0, 1}, opts), torch::empty({0}, opts.dtype(torch::kInt64))};
#endif
}

void update_gpu_cache_torch(const torch::Tensor& keys,
                             const torch::Tensor& values) {
#ifdef RECSTORE_ENABLE_GPU_CACHE
  if (keys.numel() == 0) return;
  TORCH_CHECK(keys.dim() == 1, "keys must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must be int64");
  TORCH_CHECK(values.dim() == 2, "values must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must be float32");
  TORCH_CHECK(keys.size(0) == values.size(0), "row count mismatch");
  TORCH_CHECK(keys.is_cuda() || values.is_cuda(),
              "update_gpu_cache requires keys or values on CUDA");
  const auto dev         = values.is_cuda() ? values.device() : keys.device();
  auto keys_cuda         = keys.is_cuda() ? keys : keys.to(dev);
  auto values_cuda       = values.is_cuda() ? values : values.to(dev);
  if (!keys_cuda.is_contiguous()) keys_cuda = keys_cuda.contiguous();
  if (!values_cuda.is_contiguous()) values_cuda = values_cuda.contiguous();
  gpu::UpdateGpuCache(keys_cuda, values_cuda);
#else
  (void)keys;
  (void)values;
#endif
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("local_lookup_flat", local_lookup_flat_torch);
  m.def("gpu_cache_lookup_flat", gpu_cache_lookup_flat_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_update_table", emb_update_table_torch);
  m.def("emb_update_async", emb_update_async_torch);
  m.def("emb_update_wait", emb_update_wait_torch);
  m.def("local_update_flat", local_update_flat_torch);
  m.def("init_embedding_table", init_embedding_table_torch);
  m.def("emb_write", emb_write_torch);
  m.def("emb_write_values", emb_write_values_torch);
  m.def("emb_prefetch", emb_prefetch_torch);
  m.def("emb_wait_result", emb_wait_result_torch);
  m.def("set_ps_config", set_ps_config_torch);
  m.def("set_ps_backend", set_ps_backend_torch);
  m.def("current_ps_backend", current_ps_backend_torch);
  m.def("warmup_local_lookup_flat_cuda_region",
        warmup_local_lookup_flat_cuda_region_torch);
  m.def("enable_gpu_cache", enable_gpu_cache_torch);
  m.def("disable_gpu_cache", disable_gpu_cache_torch);
  m.def("clear_gpu_cache", clear_gpu_cache_torch);
  m.def("prefill_gpu_cache", prefill_gpu_cache_torch);
  m.def("invalidate_gpu_cache", invalidate_gpu_cache_torch);
  m.def("apply_sgd_update_gpu_cache", apply_sgd_update_gpu_cache_torch);
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
