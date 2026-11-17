#include "framework/op.h"
#include "framework/common/op_profile.h"

#ifndef USE_FAKE_KVCLIENT

#  include "framework/common/hierkv_local_runtime.h"
#  include "framework/common/local_shm_op_component.h"

#  include <stdexcept>

namespace recstore {

void KVClientOp::RequireLocalBackend() const {
  if (ps_backend_name_ != "local_shm" && !IsHierKVBackendName(ps_backend_name_))
    throw std::runtime_error(
        "Local flat ops require local_shm or hierkv backend, got " +
        ps_backend_name_);
}

std::shared_ptr<base::RecTensor> KVClientOp::AcquireLocalLookupFlat(
    const base::RecTensor& keys, int64_t embedding_dim) {
  using namespace framework;
  RequireLocalBackend();
  auto handle = std::make_unique<LocalShmFlatGetHandle>();
  auto start  = SteadyNow();
  if (SubmitLocalLookupFlat(keys, embedding_dim, handle.get()) != 0)
    throw std::runtime_error("Failed to submit local_shm flat lookup");
  g_last_local_lookup_flat_profile[kLookupSubmitMs] = ElapsedMs(start);
  // The lease releases the transport slot on every success/error path.
  auto lease = std::shared_ptr<LocalShmFlatGetHandle>(
      handle.release(),
      [client = ps_client_, backend = ps_backend_name_](auto* h) {
        ReleaseLocalShmLookupFlat(client, backend, h);
        delete h;
      });
  start = SteadyNow();
  if (WaitLocalLookupFlat(lease.get()) != 0)
    throw std::runtime_error("Failed to wait for local_shm flat lookup");
  g_last_local_lookup_flat_profile[kLookupWaitMs] = ElapsedMs(start);
  const int64_t rows                              = keys.shape(0);
  if (!lease->values || lease->num_rows != rows ||
      lease->embedding_dim != embedding_dim ||
      lease->output_bytes != rows * embedding_dim * sizeof(float))
    throw std::runtime_error(
        "Unexpected local_shm flat lookup payload metadata");
  return std::shared_ptr<base::RecTensor>(
      new base::RecTensor(
          const_cast<float*>(lease->values), {rows, embedding_dim}),
      [lease](auto* values) { delete values; });
}

void KVClientOp::LocalLookupFlat(const base::RecTensor& keys,
                                 base::RecTensor& values) {
  if (ps_backend_name_ != "local_shm" &&
      !IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "local_lookup_flat requires local_shm or hierkv "
        "backend, "
        "but current backend is " +
        ps_backend_name_);
  }
  if (IsHierKVBackendName(ps_backend_name_)) {
    EmbRead(keys, values);
    return;
  }
  LocalShmLookupFlat(ps_client_, ps_backend_name_, keys, values);
}

int KVClientOp::SubmitLocalLookupFlat(const base::RecTensor& keys,
                                      int64_t embedding_dim,
                                      LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "submit_local_lookup_flat is only supported by local_shm backend.");
  }
  return SubmitLocalShmLookupFlat(
      ps_client_, ps_backend_name_, keys, embedding_dim, handle);
}

int KVClientOp::WaitLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "wait_local_lookup_flat is only supported by local_shm backend.");
  }
  return WaitLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

void KVClientOp::ReleaseLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "release_local_lookup_flat is only supported by local_shm backend.");
  }
  ReleaseLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

bool KVClientOp::GetLocalLookupFlatPayloadRegion(const void** base,
                                                 std::size_t* bytes) {
  auto* local_client = GetLocalShmClientOrThrow(
      ps_client_, ps_backend_name_, "warmup_local_lookup_flat_cuda_region");
  return local_client->GetSlotPayloadRegion(base, bytes);
}

void KVClientOp::LocalUpdateFlat(
    const std::string& table_name,
    const base::RecTensor& keys,
    const base::RecTensor& grads,
    float learning_rate) {
  if (ps_backend_name_ != "local_shm" &&
      !IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "local_update_flat requires local_shm or hierkv "
        "backend, "
        "but current backend is " +
        ps_backend_name_);
  }
  if (IsHierKVBackendName(ps_backend_name_)) {
    EmbUpdate(table_name, keys, grads);
    return;
  }
  LocalShmUpdateFlat(
      ps_client_, ps_backend_name_, table_name, keys, grads, learning_rate);
}

} // namespace recstore

#endif // USE_FAKE_KVCLIENT
