#include "framework/gpu/gpu_embedding_cache.h"

#include <limits>
#include <memory>
#include <mutex>
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include "storage/gpu_cache/nv_gpu_cache.hpp"

namespace recstore::framework::gpu {
namespace {

using CacheImpl = gpu_cache::gpu_cache<int64_t,
                                       uint64_t,
                                       std::numeric_limits<int64_t>::max(),
                                       SET_ASSOCIATIVITY,
                                       SLAB_SIZE>;

std::mutex g_mu;
std::shared_ptr<CacheImpl> g_cache;
int64_t g_capacity = 0;
int64_t g_embedding_dim = 0;
int g_device_index = -1;
cudaEvent_t g_last_cache_event = nullptr;

void RequireCudaTensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void ValidateCacheMutationTensors(const torch::Tensor& keys,
                                  const torch::Tensor& values) {
  RequireCudaTensor(keys, "keys");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must have dtype int64");
  RequireCudaTensor(values, "values");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must have dtype float32");
  TORCH_CHECK(values.dim() == 2, "values must be a 2-D tensor");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "keys and values row count mismatch");
  TORCH_CHECK(values.size(1) == g_embedding_dim,
              "values embedding_dim mismatch");
}

void RequireCacheDevice(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(g_device_index >= 0, "gpu cache device is not initialized");
  TORCH_CHECK(tensor.device().index() == g_device_index,
              name,
              " must be on gpu cache device cuda:",
              g_device_index,
              ", got ",
              tensor.device());
}

void ReleaseCacheAfterStream(void* retained_cache) {
  delete static_cast<std::shared_ptr<CacheImpl>*>(retained_cache);
}

void RetainCacheUntilStreamCompletes(std::shared_ptr<CacheImpl> cache,
                                     cudaStream_t stream) {
  auto* retained_cache = new std::shared_ptr<CacheImpl>(std::move(cache));
  cudaError_t status =
      cudaLaunchHostFunc(stream, ReleaseCacheAfterStream, retained_cache);
  if (status != cudaSuccess) {
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    delete retained_cache;
    C10_CUDA_CHECK(status);
  }
}

void EnsureLastCacheEventLocked() {
  if (g_last_cache_event == nullptr) {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&g_last_cache_event,
                                            cudaEventDisableTiming));
  }
}

void WaitForPriorCacheOpOnStreamLocked(cudaStream_t stream) {
  if (g_last_cache_event != nullptr) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, g_last_cache_event, 0));
  }
}

void RecordLastCacheOpOnStreamLocked(cudaStream_t stream) {
  EnsureLastCacheEventLocked();
  C10_CUDA_CHECK(cudaEventRecord(g_last_cache_event, stream));
}

void SynchronizeLastCacheOpLocked() {
  if (g_last_cache_event != nullptr) {
    C10_CUDA_CHECK(cudaEventSynchronize(g_last_cache_event));
  }
}

void DestroyLastCacheEventLocked() {
  if (g_last_cache_event != nullptr) {
    C10_CUDA_CHECK(cudaEventDestroy(g_last_cache_event));
    g_last_cache_event = nullptr;
  }
}

void ResetCacheOnRecordedDeviceLocked() {
  if (g_cache && g_device_index >= 0) {
    c10::cuda::CUDAGuard device_guard(g_device_index);
    g_cache.reset();
    return;
  }
  g_cache.reset();
}

std::tuple<torch::Tensor, torch::Tensor>
UniqueKeysWithInverse(const torch::Tensor& keys) {
  // PyTorch 2.7 exposes one-dimensional unique with inverse through _unique2.
  // Keep this internal ATen dependency isolated so the version requirement has
  // a single call site if libtorch changes the generated C++ API.
  auto unique_result =
      at::_unique2(keys,
                   /*sorted=*/true,
                   /*return_inverse=*/true,
                   /*return_counts=*/false);
  return {std::get<0>(unique_result), std::get<1>(unique_result)};
}

struct GpuCacheAsyncLookup {
  torch::Tensor values;
  torch::Tensor missing_index;
  torch::Tensor missing_keys;
  torch::Tensor missing_len;
  std::shared_ptr<CacheImpl> cache;
  size_t* missing_len_device = nullptr;
  at::cuda::CUDAStream stream;
};

GpuCacheAsyncLookup SubmitGpuCacheLookupLocked(const torch::Tensor& keys,
                                               int64_t embedding_dim,
                                               const at::cuda::CUDAStream& stream,
                                               const std::shared_ptr<CacheImpl>& cache) {
  auto options = keys.options().dtype(torch::kFloat32);
  auto values = torch::empty({keys.numel(), embedding_dim}, options);
  auto missing_index =
      torch::empty({keys.numel()}, keys.options().dtype(torch::kInt64));
  auto missing_keys = torch::empty_like(keys);
  auto missing_len =
      torch::empty({1}, keys.options().dtype(torch::kInt64));
  auto* missing_len_device =
      reinterpret_cast<size_t*>(missing_len.data_ptr<int64_t>());

  WaitForPriorCacheOpOnStreamLocked(stream.stream());
  C10_CUDA_CHECK(
      cudaMemsetAsync(missing_len_device, 0, sizeof(size_t), stream.stream()));
  cache->Query(keys.data_ptr<int64_t>(),
               static_cast<size_t>(keys.numel()),
               values.data_ptr<float>(),
               reinterpret_cast<uint64_t*>(missing_index.data_ptr<int64_t>()),
               missing_keys.data_ptr<int64_t>(),
               missing_len_device,
               stream.stream());

  return GpuCacheAsyncLookup{
      values,
      missing_index,
      missing_keys,
      missing_len,
      cache,
      missing_len_device,
      stream,
  };
}

GpuCacheLookupResult MaterializeGpuCacheLookup(const GpuCacheAsyncLookup& lookup,
                                               int64_t num_keys) {
  size_t missing_count_host = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(&missing_count_host,
                                 lookup.missing_len_device,
                                 sizeof(size_t),
                                 cudaMemcpyDeviceToHost,
                                 lookup.stream.stream()));
  C10_CUDA_CHECK(cudaStreamSynchronize(lookup.stream.stream()));

  TORCH_CHECK(missing_count_host <= static_cast<size_t>(num_keys),
              "gpu cache returned invalid missing count");
  const int64_t missing_count = static_cast<int64_t>(missing_count_host);
  auto missing_keys_cpu =
      lookup.missing_keys.slice(0, 0, missing_count).to(torch::kCPU);
  auto missing_positions_cpu =
      lookup.missing_index.slice(0, 0, missing_count).to(torch::kCPU);

  return GpuCacheLookupResult{
      lookup.values,
      missing_keys_cpu,
      missing_positions_cpu,
      missing_count,
  };
}

} // namespace

bool EnableGpuCache(int64_t capacity, int64_t embedding_dim) {
  TORCH_CHECK(capacity > 0, "gpu cache capacity must be positive");
  TORCH_CHECK(embedding_dim > 0, "gpu cache embedding_dim must be positive");
  std::lock_guard<std::mutex> guard(g_mu);
  const int device_index = c10::cuda::current_device();
  if (g_cache && g_capacity == capacity && g_embedding_dim == embedding_dim &&
      g_device_index == device_index) {
    return true;
  }
  SynchronizeLastCacheOpLocked();
  ResetCacheOnRecordedDeviceLocked();
  {
    c10::cuda::CUDAGuard device_guard(device_index);
    DestroyLastCacheEventLocked();
    g_cache = std::make_shared<CacheImpl>(
        static_cast<size_t>(capacity), static_cast<size_t>(embedding_dim));
  }
  g_capacity = capacity;
  g_embedding_dim = embedding_dim;
  g_device_index = device_index;
  return true;
}

void DisableGpuCache() {
  std::lock_guard<std::mutex> guard(g_mu);
  SynchronizeLastCacheOpLocked();
  ResetCacheOnRecordedDeviceLocked();
  DestroyLastCacheEventLocked();
  g_capacity = 0;
  g_embedding_dim = 0;
  g_device_index = -1;
}

void ClearGpuCache() {
  std::lock_guard<std::mutex> guard(g_mu);
  if (!g_cache) {
      return;
  }
  TORCH_CHECK(g_capacity > 0, "gpu cache capacity must be positive");
  TORCH_CHECK(g_embedding_dim > 0, "gpu cache embedding_dim must be positive");
  TORCH_CHECK(g_device_index >= 0, "gpu cache device is not initialized");
  SynchronizeLastCacheOpLocked();
  ResetCacheOnRecordedDeviceLocked();
  {
    c10::cuda::CUDAGuard device_guard(g_device_index);
    DestroyLastCacheEventLocked();
    g_cache = std::make_shared<CacheImpl>(
        static_cast<size_t>(g_capacity), static_cast<size_t>(g_embedding_dim));
  }
}

bool IsGpuCacheEnabled() {
  std::lock_guard<std::mutex> guard(g_mu);
  return g_cache != nullptr;
}


bool CanUseGpuCache(const torch::Tensor& keys, int64_t embedding_dim) {
  std::lock_guard<std::mutex> guard(g_mu);
  return g_cache != nullptr && keys.is_cuda() && keys.is_contiguous() &&
         keys.scalar_type() == torch::kInt64 &&
         keys.device().index() == g_device_index &&
         embedding_dim == g_embedding_dim;
}

GpuCacheLookupResult QueryGpuCache(const torch::Tensor& keys,
                                   int64_t embedding_dim) {
  RequireCudaTensor(keys, "keys");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64, "keys must have dtype int64");

  c10::cuda::CUDAGuard device_guard(keys.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  std::optional<GpuCacheAsyncLookup> lookup;
  {
    std::lock_guard<std::mutex> guard(g_mu);
    TORCH_CHECK(g_cache != nullptr, "gpu cache is not enabled");
    TORCH_CHECK(embedding_dim == g_embedding_dim,
                "gpu cache embedding_dim mismatch");
    RequireCacheDevice(keys, "keys");
    auto cache = g_cache;
    lookup.emplace(
        SubmitGpuCacheLookupLocked(keys, embedding_dim, stream, cache));
    RecordLastCacheOpOnStreamLocked(stream.stream());
  }
  return MaterializeGpuCacheLookup(*lookup, keys.numel());
}

void FillGpuCache(const torch::Tensor& keys_cuda,
                  const torch::Tensor& values_cuda) {
  if (keys_cuda.numel() == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(keys_cuda.device());
  const auto stream     = at::cuda::getCurrentCUDAStream();
  std::shared_ptr<CacheImpl> cache;
  {
    std::lock_guard<std::mutex> guard(g_mu);
    if (!g_cache) {
      return;
    }
    ValidateCacheMutationTensors(keys_cuda, values_cuda);
    RequireCacheDevice(keys_cuda, "keys_cuda");
    RequireCacheDevice(values_cuda, "values_cuda");
    cache = g_cache;
    WaitForPriorCacheOpOnStreamLocked(stream.stream());
    cache->Replace(keys_cuda.data_ptr<int64_t>(),
                   static_cast<size_t>(keys_cuda.numel()),
                   values_cuda.data_ptr<float>(),
                   stream.stream());
    RetainCacheUntilStreamCompletes(cache, stream.stream());
    RecordLastCacheOpOnStreamLocked(stream.stream());
  }
}

void ScatterMissValues(torch::Tensor* output_values,
                       const torch::Tensor& missing_positions_cpu,
                       const torch::Tensor& miss_values_cuda) {
  if (miss_values_cuda.numel() == 0) {
    return;
  }
  TORCH_CHECK(output_values != nullptr, "output_values must not be null");
  RequireCudaTensor(*output_values, "output_values");
  RequireCudaTensor(miss_values_cuda, "miss_values_cuda");
  TORCH_CHECK(missing_positions_cpu.device().is_cpu(),
              "missing_positions_cpu must be a CPU tensor");
  TORCH_CHECK(missing_positions_cpu.scalar_type() == torch::kInt64,
              "missing_positions_cpu must have dtype int64");
  c10::cuda::CUDAGuard device_guard(output_values->device());
  auto positions_cuda =
      missing_positions_cpu.to(output_values->device(), /*non_blocking=*/true);
  output_values->index_copy_(0, positions_cuda.to(torch::kLong), miss_values_cuda);
}


void UpdateGpuCache(const torch::Tensor& keys_cuda,
                    const torch::Tensor& values_cuda) {
  if (keys_cuda.numel() == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(keys_cuda.device());
  const auto stream       = at::cuda::getCurrentCUDAStream();
  std::shared_ptr<CacheImpl> cache;
  {
    std::lock_guard<std::mutex> guard(g_mu);
    if (!g_cache) {
      return;
    }
    ValidateCacheMutationTensors(keys_cuda, values_cuda);
    RequireCacheDevice(keys_cuda, "keys_cuda");
    RequireCacheDevice(values_cuda, "values_cuda");
    cache = g_cache;
    WaitForPriorCacheOpOnStreamLocked(stream.stream());
    cache->Update(keys_cuda.data_ptr<int64_t>(),
                  static_cast<size_t>(keys_cuda.numel()),
                  values_cuda.data_ptr<float>(),
                  stream.stream());
    RetainCacheUntilStreamCompletes(cache, stream.stream());
    RecordLastCacheOpOnStreamLocked(stream.stream());
  }
}

void ApplySgdUpdateBestEffortGpuCache(const torch::Tensor& keys_cuda,
                                      const torch::Tensor& grads_cuda,
                                      double learning_rate) {
  if (keys_cuda.numel() == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(keys_cuda.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  std::shared_ptr<CacheImpl> cache;
  {
    std::lock_guard<std::mutex> guard(g_mu);
    if (!g_cache) {
      return;
    }
    ValidateCacheMutationTensors(keys_cuda, grads_cuda);
    RequireCacheDevice(keys_cuda, "keys_cuda");
    RequireCacheDevice(grads_cuda, "grads_cuda");
    cache = g_cache;
    WaitForPriorCacheOpOnStreamLocked(stream.stream());
    cache->ApplySgd(keys_cuda.data_ptr<int64_t>(),
                    static_cast<size_t>(keys_cuda.numel()),
                    grads_cuda.data_ptr<float>(),
                    static_cast<float>(learning_rate),
                    stream.stream());
    RetainCacheUntilStreamCompletes(cache, stream.stream());
    RecordLastCacheOpOnStreamLocked(stream.stream());
  }
}

void InvalidateGpuCache(const torch::Tensor& keys_cuda) {
  if (keys_cuda.numel() == 0) {
    return;
  }
  c10::cuda::CUDAGuard device_guard(keys_cuda.device());
  const auto stream           = at::cuda::getCurrentCUDAStream();
  std::shared_ptr<CacheImpl> cache;
  {
    std::lock_guard<std::mutex> guard(g_mu);
    if (!g_cache) {
      return;
    }
    RequireCudaTensor(keys_cuda, "keys_cuda");
    TORCH_CHECK(keys_cuda.scalar_type() == torch::kInt64,
                "keys_cuda must have dtype int64");
    RequireCacheDevice(keys_cuda, "keys_cuda");
    cache = g_cache;
    WaitForPriorCacheOpOnStreamLocked(stream.stream());
    cache->Remove(keys_cuda.data_ptr<int64_t>(),
                  static_cast<size_t>(keys_cuda.numel()),
                  stream.stream());
    RetainCacheUntilStreamCompletes(cache, stream.stream());
    RecordLastCacheOpOnStreamLocked(stream.stream());
  }
}

bool ApplySgdUpdateGpuCache(const torch::Tensor& keys_cuda,
                            const torch::Tensor& grads_cuda,
                            double learning_rate) {
  if (keys_cuda.numel() == 0) {
    return true;
  }
  RequireCudaTensor(keys_cuda, "keys_cuda");
  TORCH_CHECK(grads_cuda.dim() == 2, "grads_cuda must be a 2-D tensor");
  RequireCudaTensor(grads_cuda, "grads_cuda");
  TORCH_CHECK(grads_cuda.scalar_type() == torch::kFloat32,
              "grads_cuda must have dtype float32");
  TORCH_CHECK(keys_cuda.size(0) == grads_cuda.size(0),
              "keys_cuda and grads_cuda row count mismatch");
  const int64_t embedding_dim = grads_cuda.size(1);

  auto [unique_keys, inverse] = UniqueKeysWithInverse(keys_cuda);

  auto cached = QueryGpuCache(unique_keys, embedding_dim);
  if (cached.missing_count != 0) {
    InvalidateGpuCache(keys_cuda);
    return false;
  }

  auto grad_sums = torch::zeros_like(cached.values);
  grad_sums.index_add_(0, inverse, grads_cuda);
  auto updated_values = cached.values - grad_sums * learning_rate;
  UpdateGpuCache(unique_keys, updated_values.contiguous());
  return true;
}

} // namespace recstore::framework::gpu
