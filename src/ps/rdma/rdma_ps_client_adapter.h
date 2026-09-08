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
#include "ps/rdma/rdma_deployment.h"
#include "ps/rdma/rdma_distributed_executor.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/shard_routing.h"

namespace recstore {

struct EmbeddedRdmaClientIdentity {
  int client_index         = 0;
  int num_client_processes = 1;
  int global_id            = 0;
};

// Derives per-process RDMA mesh identity for embedded PyTorch clients.
EmbeddedRdmaClientIdentity
ResolveEmbeddedRdmaClientIdentity(int num_shards, int configured_num_clients);

void InitializeRdmaProcessRuntime();

class RDMAPSClientAdapter : public BasePSClient {
  // RdmaRawAccess exposes the raw async RPC lifecycle (submit → poll → wait →
  // revoke) to the low-level RDMA benchmarks without widening the production
  // BasePSClient surface.
  friend class RdmaRawAccess;

public:
  explicit RDMAPSClientAdapter(json config);
  ~RDMAPSClientAdapter() override = default;

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   base::RecTensor& values) override;
  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const base::RecTensor& values) override;
  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const base::RecTensor& grads) override;
  uint64_t SubmitUpdateParameterAsync(
      const std::string& table_name,
      const base::ConstArray<uint64_t>& keys,
      const base::RecTensor& grads) override;
  int WaitUpdateParameter(uint64_t update_id) override;
  int InitEmbeddingTable(const std::string& table_name,
                         const EmbeddingTableConfig& config) override;
  void Command(PSCommand command) override;
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         base::RecTensor& values) override;

private:
  struct TableState {
    EmbeddingTableConfig config;
    int tag = 0;
  };

  using ShardChunk = shard_routing::ShardChunk;

  void EnsureClientInitialized();
  void EnsureThreadInitialized();
  void EnsureTableReady(const std::string& table_name, int64_t embedding_dim);
  int64_t DefaultEmbeddingDimOrThrow() const;
  int64_t EmbeddingDimForKeys(base::ConstArray<uint64_t> keys) const;
  std::size_t MaxGetKeysPerRpc(int64_t embedding_dim) const;
  std::size_t MaxPutKeysPerRpc(int64_t embedding_dim) const;
  std::size_t MaxInFlightGetRpcs() const;
  std::vector<ShardChunk> BuildChunks(base::ConstArray<uint64_t> keys,
                                      std::size_t max_keys_per_rpc) const;
  int SubmitGetParameter(base::ConstArray<uint64_t> keys,
                         float* values,
                         bool isAsync,
                         int async_req_id,
                         int64_t embedding_dim);
  bool QueryRPCFinished(int rpc_id);
  void WaitRPCFinish(int rpc_id);
  void RevokeRPCResource(int rpc_id);

  json config_;
  std::mutex init_mu_;
  std::mutex thread_init_mu_;
  mutable std::mutex state_mu_;
  bool initialized_ = false;
  std::unordered_set<std::thread::id> initialized_threads_;
  ResolvedRdmaDeployment deployment_;
  std::vector<std::unique_ptr<petps::PetPSClient>> shard_clients_;
  petps::PetPSClient* client_ = nullptr;
  std::unordered_map<std::string, TableState> tables_;
  std::unordered_map<int, int64_t> tag_to_dim_;
  std::unique_ptr<RdmaDistributedExecutor> executor_;
};

// Friend accessor for the low-level RDMA benchmarks. Forwards the adapter's
// private raw-async RPC lifecycle so a benchmark can drive multi-shard async
// GETs (submit → poll → wait → revoke) directly, without the benchmark
// reimplementing routing/batch bookkeeping.
class RdmaRawAccess {
public:
  explicit RdmaRawAccess(RDMAPSClientAdapter* adapter) : adapter_(adapter) {}

  int SubmitGetParameter(base::ConstArray<uint64_t> keys,
                         float* values,
                         bool isAsync,
                         int async_req_id,
                         int64_t embedding_dim) {
    return adapter_->SubmitGetParameter(
        keys, values, isAsync, async_req_id, embedding_dim);
  }
  bool QueryRPCFinished(int rpc_id) {
    return adapter_->QueryRPCFinished(rpc_id);
  }
  void WaitRPCFinish(int rpc_id) { adapter_->WaitRPCFinish(rpc_id); }
  void RevokeRPCResource(int rpc_id) { adapter_->RevokeRPCResource(rpc_id); }

private:
  RDMAPSClientAdapter* adapter_;
};

} // namespace recstore
