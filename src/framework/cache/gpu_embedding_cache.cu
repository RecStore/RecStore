#include "framework/cache/gpu_embedding_cache.h"
#include "framework/cache/cache_aware_op.h"
#include "storage/gpu_cache/nv_gpu_cache.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace recstore::framework::gpu {
namespace {
void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
class DeviceGuard {
public:
  explicit DeviceGuard(int device) {
    Check(cudaGetDevice(&previous_));
    Check(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous_); }

private:
  int previous_;
};

using CacheImpl =
    gpu_cache::gpu_cache<int64_t,
                         uint64_t,
                         std::numeric_limits<int64_t>::max(),
                         SET_ASSOCIATIVITY,
                         SLAB_SIZE>;
std::mutex g_mu;
std::unique_ptr<CacheImpl> g_cache;
int64_t g_capacity = 0, g_dim = 0;
int g_device          = -1;
cudaEvent_t g_last_op = nullptr;
// Bumped whenever resident rows stop being the ones a caller owns: enable,
// disable, clear and reset. Independent from CacheAwareEpoch(), which drives
// the bypass latch and must stay untouched by an internal fallback clear.
uint64_t g_generation = 0;

void ReleaseLocked() {
  if (!g_cache)
    return;
  DeviceGuard device(g_device);
  if (g_last_op) {
    Check(cudaEventSynchronize(g_last_op));
    Check(cudaEventDestroy(g_last_op));
    g_last_op = nullptr;
  }
  // Destroy on the host after all streams finish. Cache destructors call CUDA
  // and must never run from a cudaLaunchHostFunc callback.
  g_cache.reset();
}

template <typename F>
void WithCache(CacheContext context, int64_t dim, bool required, F action) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_cache) {
    Require(!required, "gpu cache is not enabled");
    return;
  }
  Require(context.device == g_device, "gpu cache device mismatch");
  Require(dim == 0 || dim == g_dim, "gpu cache embedding_dim mismatch");
  DeviceGuard device(context.device);
  if (!g_last_op)
    Check(cudaEventCreateWithFlags(&g_last_op, cudaEventDisableTiming));
  else
    Check(cudaStreamWaitEvent(context.stream, g_last_op, 0));
  try {
    action(*g_cache);
    Check(cudaEventRecord(g_last_op, context.stream));
  } catch (...) {
    // Even a failed operation may have launched work using cache storage.
    cudaStreamSynchronize(context.stream);
    throw;
  }
}

void CheckRows(
    const int64_t* keys, size_t count, const float* values, int64_t dim) {
  Require(dim > 0, "gpu cache embedding_dim must be positive");
  Require(!count || (keys && values), "gpu cache buffers must not be null");
}

size_t QueryLocked(CacheImpl& cache,
                   const int64_t* keys,
                   size_t count,
                   LookupBuffers out,
                   cudaStream_t stream) {
  Check(cudaMemsetAsync(out.missing_count, 0, sizeof(size_t), stream));
  if (count)
    cache.Query(
        keys,
        count,
        out.values,
        out.missing_positions,
        out.missing_keys,
        out.missing_count,
        stream);
  size_t missing = 0;
  Check(cudaMemcpyAsync(
      &missing,
      out.missing_count,
      sizeof(size_t),
      cudaMemcpyDeviceToHost,
      stream));
  Check(cudaStreamSynchronize(stream));
  Require(missing <= count, "gpu cache returned invalid missing count");
  return missing;
}

template <typename T>
T* Data(thrust::device_vector<T>& v) {
  return thrust::raw_pointer_cast(v.data());
}

// One thread per unique row/column, summing duplicates in input order. Keep
// multiply and subtract separate, as in the previous tensor implementation.
__global__ void SumGradientsAndUpdate(
    const int64_t* sorted_keys,
    const int64_t* order,
    size_t count,
    const int64_t* unique_keys,
    size_t unique_count,
    const float* grads,
    int64_t dim,
    float lr,
    float* values) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < unique_count * dim;
       i += size_t(blockDim.x) * gridDim.x) {
    const int64_t key = unique_keys[i / dim];
    size_t lo = 0, hi = count;
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (sorted_keys[mid] < key)
        lo = mid + 1;
      else
        hi = mid;
    }
    float sum = 0;
    for (size_t j = lo; j < count && sorted_keys[j] == key; ++j)
      sum += grads[order[j] * dim + i % dim];
    values[i] = __fsub_rn(values[i], __fmul_rn(sum, lr));
  }
}
} // namespace

bool EnableGpuCache(int64_t capacity, int64_t dim, int device) {
  Require(capacity > 0 && dim > 0,
          "gpu cache capacity and dim must be positive");
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_cache || capacity != g_capacity || dim != g_dim ||
      device != g_device) {
    // Validate the requested device before releasing the current cache.
    DeviceGuard guard(device);
    ReleaseLocked();
    g_cache    = std::make_unique<CacheImpl>(capacity, dim);
    g_capacity = capacity;
    g_dim      = dim;
    g_device   = device;
    // Bump the shared epoch so latches on other threads reset on their next
    // lookup, not only the calling thread's.
    ++g_generation;
    CacheAwareEpoch().fetch_add(1, std::memory_order_relaxed);
  }
  g_cache_aware_op.Reset();
  return true;
}
void DisableGpuCache() {
  std::lock_guard<std::mutex> lock(g_mu);
  ReleaseLocked();
  g_capacity = g_dim = 0;
  g_device           = -1;
  ++g_generation;
  CacheAwareEpoch().fetch_add(1, std::memory_order_relaxed);
  g_cache_aware_op.Reset();
}
void ClearGpuCache() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_cache)
    return;
  DeviceGuard guard(g_device);
  ReleaseLocked();
  g_cache = std::make_unique<CacheImpl>(g_capacity, g_dim);
  // Callers such as BagPipe track this counter to notice that someone else
  // wiped the rows they own. The latch epoch stays put: a fallback clear must
  // not re-enable a cache that tripped its bypass.
  ++g_generation;
}
void ResetGpuCache() {
  ClearGpuCache();
  // ClearGpuCache intentionally leaves the bypass latch alone (a fallback
  // clear must not re-enable a bypassed cache). An explicit reset does clear
  // it, on every thread.
  CacheAwareEpoch().fetch_add(1, std::memory_order_relaxed);
  g_cache_aware_op.Reset();
}
bool IsGpuCacheEnabled() {
  std::lock_guard<std::mutex> lock(g_mu);
  return bool(g_cache);
}
uint64_t GetGpuCacheGeneration() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_generation;
}
bool CanUseGpuCache(int device, int64_t dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_cache && device == g_device && dim == g_dim;
}

size_t QueryGpuCache(const int64_t* keys,
                     size_t count,
                     int64_t dim,
                     LookupBuffers out,
                     CacheContext context) {
  CheckRows(keys, count, out.values, dim);
  Require(out.missing_count &&
              (!count || (out.missing_positions && out.missing_keys)),
          "gpu cache miss buffers must not be null");
  size_t missing = 0;
  WithCache(context, dim, true, [&](CacheImpl& cache) {
    missing = QueryLocked(cache, keys, count, out, context.stream);
  });
  return missing;
}
void LookupGpuCacheAssumingHits(
    const int64_t* keys,
    size_t count,
    int64_t dim,
    float* values,
    bool* miss,
    CacheContext context) {
  CheckRows(keys, count, values, dim);
  Require(!count || miss, "gpu cache miss mask must not be null");
  WithCache(context, dim, true, [&](CacheImpl& cache) {
    if (count)
      cache.GetAssumingHits(keys, count, values, miss, context.stream);
  });
}
void ContainsGpuCache(
    const int64_t* keys, size_t count, bool* resident, CacheContext context) {
  Require(!count || (keys && resident), "gpu cache buffers must not be null");
  WithCache(context, 0, true, [&](CacheImpl& cache) {
    if (count)
      cache.Contains(keys, count, resident, context.stream);
  });
}
void FillGpuCache(const int64_t* keys,
                  size_t count,
                  const float* values,
                  int64_t dim,
                  CacheContext context) {
  CheckRows(keys, count, values, dim);
  WithCache(context, dim, false, [&](CacheImpl& cache) {
    if (count)
      cache.Replace(keys, count, values, context.stream);
  });
}
void FillGpuCacheNoEvict(
    const int64_t* keys,
    size_t count,
    const float* values,
    int64_t dim,
    bool* inserted,
    CacheContext context) {
  CheckRows(keys, count, values, dim);
  Require(!count || inserted, "gpu cache insert mask must not be null");
  WithCache(context, dim, true, [&](CacheImpl& cache) {
    if (count)
      cache.TryInsertNoEvict(keys, count, values, inserted, context.stream);
  });
}
void UpdateGpuCache(const int64_t* keys,
                    size_t count,
                    const float* values,
                    int64_t dim,
                    CacheContext context) {
  CheckRows(keys, count, values, dim);
  WithCache(context, dim, false, [&](CacheImpl& cache) {
    if (count)
      cache.Update(keys, count, values, context.stream);
  });
}
void InvalidateGpuCache(
    const int64_t* keys, size_t count, CacheContext context) {
  Require(!count || keys, "gpu cache keys must not be null");
  WithCache(context, 0, false, [&](CacheImpl& cache) {
    if (count)
      cache.Remove(keys, count, context.stream);
  });
}
void InvalidateGpuCacheWithMask(
    const int64_t* keys, size_t count, bool* removed, CacheContext context) {
  Require(!count || (keys && removed), "gpu cache buffers must not be null");
  WithCache(context, 0, true, [&](CacheImpl& cache) {
    if (count)
      cache.RemoveWithMask(keys, count, removed, context.stream);
  });
}
void ApplySgdUpdateBestEffortGpuCache(
    const int64_t* keys,
    size_t count,
    const float* grads,
    int64_t dim,
    double lr,
    CacheContext context) {
  CheckRows(keys, count, grads, dim);
  WithCache(context, dim, false, [&](CacheImpl& cache) {
    if (count)
      cache.ApplySgd(keys, count, grads, float(lr), context.stream);
  });
}
bool ApplySgdUpdateGpuCache(
    const int64_t* keys,
    size_t count,
    const float* grads,
    int64_t dim,
    double lr,
    CacheContext context) {
  CheckRows(keys, count, grads, dim);
  if (!count)
    return true;
  bool updated = false;
  WithCache(context, dim, true, [&](CacheImpl& cache) {
    auto policy = thrust::cuda::par.on(context.stream);
    thrust::device_vector<int64_t> sorted(count), order(count), unique(count);
    thrust::copy_n(policy, keys, count, sorted.begin());
    thrust::sequence(policy, order.begin(), order.end());
    thrust::stable_sort_by_key(
        policy, sorted.begin(), sorted.end(), order.begin());
    const size_t n =
        thrust::unique_copy(
            policy, sorted.begin(), sorted.end(), unique.begin()) -
        unique.begin();
    thrust::device_vector<float> values(n * dim);
    thrust::device_vector<uint64_t> positions(n);
    thrust::device_vector<int64_t> missing_keys(n);
    thrust::device_vector<size_t> missing_count(1);
    if (QueryLocked(
            cache,
            Data(unique),
            n,
            {Data(values),
             Data(positions),
             Data(missing_keys),
             Data(missing_count)},
            context.stream)) {
      cache.Remove(keys, count, context.stream);
    } else {
      const auto blocks =
          static_cast<unsigned>(std::min<size_t>((n * dim + 255) / 256, 65535));
      SumGradientsAndUpdate<<<blocks, 256, 0, context.stream>>>(
          Data(sorted),
          Data(order),
          count,
          Data(unique),
          n,
          grads,
          dim,
          float(lr),
          Data(values));
      Check(cudaGetLastError());
      cache.Update(Data(unique), n, Data(values), context.stream);
      updated = true;
    }
    Check(cudaStreamSynchronize(context.stream));
  });
  return updated;
}
} // namespace recstore::framework::gpu
