#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>

#include "base/array.h"
#include "base/factory.h"
#include "base/json.h"
#include "ps/base/base_client.h"
#include "ps/local_shm/local_shm_region.h"

namespace recstore {

struct LocalShmFlatGetHandle {
  static constexpr uint32_t kInvalidSlotId =
      std::numeric_limits<uint32_t>::max();

  uint32_t slot_id      = kInvalidSlotId;
  uint64_t request_id   = 0;
  float* values         = nullptr;
  int64_t num_rows      = 0;
  int64_t embedding_dim = 0;
  uint64_t output_bytes = 0;
};

struct LocalShmRequestProfile {
  uint32_t opcode             = 0;
  double acquire_slot_us      = 0.0;
  double enqueue_us           = 0.0;
  double wait_us              = 0.0;
  double release_us           = 0.0;
  double request_total_us     = 0.0;
  double server_queue_wait_us = 0.0;
  double server_backend_us    = 0.0;
};

struct LocalShmActiveRequestProfile {
  std::chrono::steady_clock::time_point request_start{};
  uint32_t opcode        = 0;
  double acquire_slot_us = 0.0;
  double enqueue_us      = 0.0;
  double wait_us         = 0.0;
  bool active            = false;
};

class LocalShmPSClient : public BasePSClient {
public:
  explicit LocalShmPSClient(json config);
  ~LocalShmPSClient() override = default;

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   base::RecTensor& values) override;
  int GetParameterFlat(const base::ConstArray<uint64_t>& keys,
                       float* values,
                       int64_t num_rows,
                       int64_t embedding_dim);
  int SubmitGetParameterFlat(const base::ConstArray<uint64_t>& keys,
                             int64_t num_rows,
                             int64_t embedding_dim,
                             LocalShmFlatGetHandle* handle);
  int WaitGetParameterFlat(LocalShmFlatGetHandle* handle);
  void ReleaseGetParameterFlat(LocalShmFlatGetHandle* handle);
  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const base::RecTensor& values) override;
  int PutParameterFlat(const base::ConstArray<uint64_t>& keys,
                       const float* values,
                       int64_t num_rows,
                       int64_t embedding_dim);
  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const base::RecTensor& grads) override;
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim);
  int InitEmbeddingTable(const std::string& table_name,
                         const EmbeddingTableConfig& config) override;
  bool GetSlotPayloadRegion(const void** base, std::size_t* bytes) const;
  uint32_t CurrentReadyQueueId() const;
  LocalShmRequestProfile GetLastRequestProfile() const;

  void Command(PSCommand command) override;

  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         base::RecTensor& values) override;

private:
  int AcquireSlot();
  void ReleaseSlot(uint32_t slot_id);
  bool WaitForSlot(uint32_t slot_id, uint64_t request_id);
  uint64_t NextRequestId();
  uint32_t ResolveReadyQueueId(const json& config) const;
  void ResetActiveRequestProfile() const;
  void FinalizeActiveRequestProfile(
      const LocalShmSlotHeader* header,
      const std::chrono::steady_clock::time_point& request_start) const;

private:
  LocalShmRegion region_;
  std::string region_name_;
  uint32_t client_id_               = 0;
  uint32_t ready_queue_id_          = 0;
  bool thread_ready_queue_sharding_ = false;
  uint32_t timeout_ms_              = 30000;
  std::unordered_map<std::string, int64_t> table_embedding_dims_;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint64_t>> prefetch_map_;
};

} // namespace recstore
