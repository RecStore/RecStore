#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/log.h"
#include "ps/rdma/base_client.h"
#include "ps/rdma/rdma_status.h"
#include "ps/rdma/shard_routing.h"

class AllShardsParameterClientWrapper : public BaseParameterClient {
public:
  explicit AllShardsParameterClientWrapper(
      const std::vector<BaseParameterClient*>& clients,
      int num_shards,
      const std::string& hash_method    = "city_hash",
      const std::vector<int>& shard_ids = {});

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;

  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  void InitThread() override;
  void Barrier(const std::string& ss, int k) override;
  void* GetReceiveBuffer(size_t size) override;
  bool QueryRPCFinished(int rpc_id) override;
  void WaitRPCFinish(int rpc_id) override;
  void RevokeRPCResource(int rpc_id) override;
  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;
  int InitEmbeddingTable(const std::string& table_name,
                         std::uint64_t num_embeddings,
                         std::uint64_t embedding_dim) override;
  int UpdateParameter(const std::string& table_name,
                      base::ConstArray<uint64_t> keys,
                      const std::vector<std::vector<float>>* grads) override;

private:
  using PendingShardRpc = recstore::shard_routing::PendingShardRpc;
  using BatchRequest    = recstore::shard_routing::BatchRequest;
  using ShardChunk      = recstore::shard_routing::ShardChunk;

  std::vector<ShardChunk> BuildChunks(base::ConstArray<uint64_t> keys) const;
  void WaitShardRpcsCooperatively(
      const std::vector<PendingShardRpc>& shard_rpcs) const;

  std::vector<BaseParameterClient*> clients_; // Shard-local clients.
  int num_shards_; // Number of logical shards in the distributed layout.
  std::string hash_method_; // Hash method used to map keys to shards.
  std::unordered_map<int, int>
      shard_to_client_index_;          // Explicit shard -> client map.
  std::uint64_t batch_rpc_id_acc_ = 1; // Wrapper-local batch handle generator.
  mutable std::mutex batches_mu_; // Guards batch assembly and completion state.
  std::unordered_map<std::uint64_t, BatchRequest>
      batches_; // Outstanding wrapper batches.
};
