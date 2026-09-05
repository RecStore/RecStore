#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/log.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/shard_routing.h"

namespace recstore {

// Owns distributed request submission, completion, assembly, and release.
class RdmaDistributedExecutor {
public:
  struct PrefetchResult {
    std::shared_ptr<std::vector<float>> buffer;
    const float* payload       = nullptr;
    std::size_t response_bytes = 0;
    std::int64_t key_count     = 0;
    std::int64_t embedding_dim = 0;
    std::int32_t status_code   = 0;
  };

  RdmaDistributedExecutor(std::vector<petps::PetPSClient*> clients,
                          int value_size,
                          int first_batch_id,
                          int batch_id_step);
  ~RdmaDistributedExecutor();

  int RegisterBatch(shard_routing::BatchRequest batch);
  int SubmitGetParameter(
      const std::vector<shard_routing::ShardChunk>& chunks,
      std::size_t total_key_count,
      float* values,
      bool is_async,
      int async_req_id,
      std::size_t max_in_flight);
  bool QueryBatchFinished(int batch_id);
  void WaitBatch(int batch_id);
  void ReleaseBatch(int batch_id);
  void WaitShardRpcs(
      const std::vector<shard_routing::PendingShardRpc>& shard_rpcs) const;
  void ReleaseShardRpcs(
      std::vector<shard_routing::PendingShardRpc>* shard_rpcs) const;
  int PutParameter(const std::vector<shard_routing::ShardChunk>& chunks,
                   const base::RecTensor& values) const;
  int UpdateParameter(const std::string& table_name,
                      const std::vector<shard_routing::ShardChunk>& chunks,
                      const base::RecTensor& grads) const;
  std::uint64_t SubmitUpdateParameterFlat(
      const std::string& table_name,
      base::ConstArray<std::uint64_t> keys,
      const float* grads,
      std::size_t embedding_dim,
      const std::vector<shard_routing::ShardChunk>& chunks,
      std::size_t max_in_flight);
  int WaitUpdateParameterFlat(std::uint64_t update_id);
  std::uint64_t
  SubmitPrefetch(const std::vector<shard_routing::ShardChunk>& chunks,
                 std::size_t key_count,
                 std::int64_t embedding_dim,
                 std::size_t max_in_flight);
  bool IsPrefetchDone(std::uint64_t prefetch_id);
  void WaitForPrefetch(std::uint64_t prefetch_id);
  std::int64_t PrefetchEmbeddingDim(std::uint64_t prefetch_id) const;
  PrefetchResult ReadPrefetch(std::uint64_t prefetch_id);
  void ReleasePrefetch(std::uint64_t prefetch_id);

private:
  struct PendingUpdate {
    std::vector<std::pair<int, int>> shard_rpcs;
    std::thread::id owner;
  };

  struct PendingPrefetch {
    std::shared_ptr<std::vector<float>> buffer;
    int batch_id               = 0;
    std::int64_t key_count     = 0;
    std::int64_t embedding_dim = 0;
  };

  petps::PetPSClient* ClientAt(int client_index) const;
  bool FinalizeBatch(shard_routing::BatchRequest* batch) const;
  const float* BorrowBatchResult(
      int batch_id, std::int32_t* status_code, std::size_t* response_bytes);
  PendingPrefetch GetPrefetch(std::uint64_t prefetch_id) const;
  void ErasePrefetch(std::uint64_t prefetch_id);
  int WaitUpdateRpcs(std::vector<std::pair<int, int>>* shard_rpcs) const;

  std::vector<petps::PetPSClient*> clients_;
  int value_size_       = 0;
  std::int64_t next_id_ = 0;
  int id_step_          = 0;
  mutable std::mutex batches_mu_;
  std::unordered_map<int, shard_routing::BatchRequest> batches_;
  std::mutex updates_mu_;
  std::uint64_t next_update_id_ = 1;
  std::unordered_map<std::uint64_t, PendingUpdate> pending_updates_;
  mutable std::mutex prefetches_mu_;
  std::uint64_t next_prefetch_id_ = 1;
  std::unordered_map<std::uint64_t, PendingPrefetch> prefetches_;
};

} // namespace recstore
