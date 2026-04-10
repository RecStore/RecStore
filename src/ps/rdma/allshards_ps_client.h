#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/log.h"
#include "base_client.h"

class AllShardsParameterClientWrapper : public BaseParameterClient {
public:
  explicit AllShardsParameterClientWrapper(
      const std::vector<BaseParameterClient*>& clients, int num_shards);

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

private:
  struct PendingShardRpc {
    int shard  = 0;
    int rpc_id = -1;
    std::vector<std::size_t> original_positions;
    void* recv_buffer     = nullptr;
    std::size_t key_count = 0;
  };

  struct BatchRequest {
    float* user_buffer = nullptr;
    bool assembled     = false;
    std::vector<PendingShardRpc> shard_rpcs;
  };

  struct ShardChunk {
    int shard = 0;
    std::vector<uint64_t> keys;
    std::vector<std::size_t> positions;
  };

  int PartitionKey(uint64_t key) const;
  std::vector<ShardChunk> BuildChunks(base::ConstArray<uint64_t> keys) const;
  void AssembleIfNeeded(BatchRequest* batch);

  std::vector<BaseParameterClient*> clients_;
  int num_shards_;
  std::uint64_t batch_rpc_id_acc_ = 1;
  std::unordered_map<std::uint64_t, BatchRequest> batches_;
};
