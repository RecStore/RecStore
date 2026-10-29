#include "framework/pytorch/op_flat.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

torch::Tensor
local_lookup_flat_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  return std::get<0>(CachedReadTorch(keys, embedding_dim, true));
}

void local_update_flat_torch(const std::string& table_name,
                             const torch::Tensor& keys,
                             const torch::Tensor& grads) {
  ResetLocalUpdateFlatProfile();
#ifdef RECSTORE_ENABLE_GPU_CACHE
  gpu::ResetLastGpuCacheProfile();
#endif
  const auto total_start = SteadyNow();
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
  kv_op->RequireLocalBackend();

  if (keys.size(0) == 0) {
    g_last_local_update_flat_profile[kUpdateTotalMs] = ElapsedMs(total_start);
    return;
  }

  torch::Tensor cpu_keys = keys;
  const bool can_async_stage_cuda =
      (keys.is_cuda() || grads.is_cuda()) &&
      (!keys.is_cuda() || !grads.is_cuda() || keys.device() == grads.device());
  bool staged_cuda_async = false;
  if (keys.is_cuda()) {
    const auto keys_stage_start = SteadyNow();
    if (can_async_stage_cuda) {
      cpu_keys          = StageCudaTensorToPinnedCpuAsyncNoCast(keys);
      staged_cuda_async = true;
    } else {
      cpu_keys = StageCudaTensorToPinnedCpu(keys, torch::kInt64);
    }
    g_last_local_update_flat_profile[kUpdateKeysStageMs] =
        ElapsedMs(keys_stage_start);
  }
  torch::Tensor cpu_grads = grads;
  if (grads.is_cuda()) {
    const auto grads_stage_start = SteadyNow();
    if (can_async_stage_cuda) {
      cpu_grads         = StageCudaTensorToPinnedCpuAsyncNoCast(grads);
      staged_cuda_async = true;
    } else {
      cpu_grads = StageCudaTensorToPinnedCpu(grads, torch::kFloat32);
    }
    g_last_local_update_flat_profile[kUpdateGradsStageMs] =
        ElapsedMs(grads_stage_start);
  }
  if (staged_cuda_async) {
    const auto stage_wait_start = SteadyNow();
    SynchronizeCurrentCudaStreamForTensor(keys.is_cuda() ? keys : grads);
    g_last_local_update_flat_profile[kUpdateStageWaitMs] =
        ElapsedMs(stage_wait_start);
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  const auto shm_call_start = SteadyNow();
  kv_op->EmbUpdateCached(
      table_name, rec_keys, rec_grads, CacheActions(keys, grads.size(1)), true);
  g_last_local_update_flat_profile[kUpdateShmCallMs] =
      ElapsedMs(shm_call_start);

  // Client-side split of the same round trip. The server half
  // (kUpdateServerBackendMs) comes back in the slot header, so the remainder
  // is slot acquire + copy + wake latency.
  const auto last_request = GetLastLocalShmRequestProfile();
  if (last_request.opcode == static_cast<uint32_t>(LocalOpcode::kUpdateFlat)) {
    g_last_local_update_flat_profile[kUpdateSlotAcquireMs] =
        last_request.acquire_slot_us / 1000.0;
    g_last_local_update_flat_profile[kUpdateCopyEnqueueMs] =
        last_request.enqueue_us / 1000.0;
    g_last_local_update_flat_profile[kUpdateWaitMs] =
        last_request.wait_us / 1000.0;
    g_last_local_update_flat_profile[kUpdateServerQueueWaitMs] =
        last_request.server_queue_wait_us / 1000.0;
    g_last_local_update_flat_profile[kUpdateServerBackendMs] =
        last_request.server_backend_us / 1000.0;
  }

  g_last_local_update_flat_profile[kUpdateTotalMs] = ElapsedMs(total_start);
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

} // namespace

void RegisterFlatOps(torch::Library& m) {
  m.def("local_lookup_flat", local_lookup_flat_torch);
  m.def("local_update_flat", local_update_flat_torch);
  m.def("warmup_local_lookup_flat_cuda_region",
        warmup_local_lookup_flat_cuda_region_torch);
}

} // namespace framework
} // namespace recstore
