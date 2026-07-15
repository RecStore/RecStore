#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/json.h"
#include "ps/base/base_client.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/shard_routing.h"

namespace recstore {

struct EmbeddedRdmaClientIdentity {
  int client_index         = 0;
  int num_client_processes = 1;
  int global_id            = 0;
};

// Derives per-process RDMA mesh identity for embedded PyTorch clients.
EmbeddedRdmaClientIdentity ResolveEmbeddedRdmaClientIdentity(int num_shards);

void InitializeRdmaProcessRuntime();

class RDMAPSClientAdapter : public BasePSClient {
public:
  explicit RDMAPSClientAdapter(json config);
  ~RDMAPSClientAdapter() override = default;

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   float* values) override;
  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;
  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const std::vector<std::vector<float>>* grads) override;
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim) override;
  int InitEmbeddingTable(const std::string& table_name,
                         const EmbeddingTableConfig& config) override;
  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                        float* values) override;
  void Command(PSCommand command) override;
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         std::vector<std::vector<float>>* values) override;
  bool GetPrefetchResultFlat(uint64_t prefetch_id,
                             std::vector<float>* values,
                             int64_t* num_rows,
                             int64_t embedding_dim) override;

private:
  struct TableState {
    EmbeddingTableConfig config;
  };

  using PendingShardRpc = shard_routing::PendingShardRpc;
  using BatchRequest    = shard_routing::BatchRequest;
  using ShardChunk      = shard_routing::ShardChunk;

  struct PrefetchState {
    float* buffer          = nullptr;
    std::size_t buffer_id  = 0;
    int rpc_id             = -1;
    int64_t key_count      = 0;
    int64_t embedding_dim  = 0;
    bool borrowed_response = false;
    bool batch_response    = false;
  };

  void EnsureClientInitialized();
  void EnsureThreadInitialized();
  void EnsureTableReady(const std::string& table_name, int64_t embedding_dim);
  int64_t DefaultEmbeddingDimOrThrow() const;
  std::size_t MaxGetKeysPerRpc() const;
  std::size_t MaxPutKeysPerRpc() const;
  std::size_t MaxInFlightGetRpcs() const;
  std::vector<ShardChunk> BuildChunks(base::ConstArray<uint64_t> keys) const;
  void
  WaitShardRpcsCooperatively(const std::vector<PendingShardRpc>& shard_rpcs);
  int SubmitGetParameter(base::ConstArray<uint64_t> keys,
                         float* values,
                         bool isAsync,
                         int async_req_id);
  bool QueryRPCFinished(int rpc_id);
  void WaitRPCFinish(int rpc_id);
  void RevokeRPCResource(int rpc_id);
  float* AcquirePrefetchBuffer(std::size_t bytes, std::size_t* buffer_id);
  void ReleasePrefetchBuffer(std::size_t buffer_id);
  const float* BorrowPrefetchResult(const PrefetchState& state,
                                    std::int32_t* status_code,
                                    std::size_t* response_bytes);
  PrefetchState GetPrefetchState(uint64_t prefetch_id);
  void MarkPrefetchConsumed(uint64_t prefetch_id);

  json config_;
  std::mutex init_mu_;
  std::mutex thread_init_mu_;
  std::mutex state_mu_;
  bool initialized_ = false;
  std::unordered_set<std::thread::id> initialized_threads_;
  std::vector<std::unique_ptr<petps::PetPSClient>> shard_clients_;
  BaseParameterClient* client_ = nullptr;
  int num_shards_              = 1;
  std::string hash_method_     = "city_hash";
  std::unordered_map<int, int> shard_to_client_index_;
  int batch_rpc_id_acc_ = -1;
  mutable std::mutex batches_mu_;
  std::unordered_map<int, BatchRequest> batches_;
  std::unordered_map<std::string, TableState> tables_;
  std::vector<std::unique_ptr<char[]>> prefetch_buffers_;
  std::vector<std::size_t> prefetch_buffer_capacities_;
  std::vector<std::size_t> free_prefetch_buffer_ids_;
  std::unordered_map<uint64_t, PrefetchState> prefetches_;
  uint64_t next_prefetch_id_ = 1;
};

} // namespace recstore
