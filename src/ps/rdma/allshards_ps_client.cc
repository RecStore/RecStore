#include "allshards_ps_client.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "base/hash.h"
#include "ps/rdma/rdma_protocol.h"

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

AllShardsParameterClientWrapper::AllShardsParameterClientWrapper(
    const std::vector<BaseParameterClient*>& clients, int num_shards)
    : BaseParameterClient("", 0, 0),
      clients_(clients),
      num_shards_(num_shards) {
  CHECK_EQ(static_cast<int>(clients_.size()), num_shards_);
}

int AllShardsParameterClientWrapper::PartitionKey(uint64_t key) const {
  CHECK_GT(num_shards_, 0);
  return static_cast<int>(GetHash(key) % static_cast<uint64_t>(num_shards_));
}

std::vector<AllShardsParameterClientWrapper::ShardChunk>
AllShardsParameterClientWrapper::BuildChunks(
    base::ConstArray<uint64_t> keys) const {
  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::size_t>> shard_positions(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i]);
    shard_keys[shard].push_back(keys[i]);
    shard_positions[shard].push_back(i);
  }

  std::vector<ShardChunk> chunks;
  for (int shard = 0; shard < num_shards_; ++shard) {
    for (std::size_t offset = 0; offset < shard_keys[shard].size();
         offset += FLAGS_max_kv_num_per_request) {
      const std::size_t end = std::min(
          offset + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
          shard_keys[shard].size());
      ShardChunk chunk;
      chunk.shard = shard;
      chunk.keys.assign(
          shard_keys[shard].begin() + offset, shard_keys[shard].begin() + end);
      chunk.positions.assign(shard_positions[shard].begin() + offset,
                             shard_positions[shard].begin() + end);
      chunks.push_back(std::move(chunk));
    }
  }
  return chunks;
}

void AllShardsParameterClientWrapper::AssembleIfNeeded(BatchRequest* batch) {
  if (batch->assembled) {
    return;
  }

  const int embedding_dim = FLAGS_value_size / sizeof(float);
  for (const auto& pending : batch->shard_rpcs) {
    const float* shard_values = static_cast<const float*>(pending.recv_buffer);
    for (std::size_t i = 0; i < pending.original_positions.size(); ++i) {
      std::memcpy(
          batch->user_buffer + pending.original_positions[i] * embedding_dim,
          shard_values + i * embedding_dim,
          FLAGS_value_size);
    }
  }
  batch->assembled = true;
}

int AllShardsParameterClientWrapper::GetParameter(
    base::ConstArray<uint64_t> keys, std::vector<std::vector<float>>* values) {
  values->clear();
  if (keys.Size() == 0) {
    return 0;
  }

  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<float> flat(keys.Size() * embedding_dim + 1, 0.0f);
  int rpc_id = GetParameter(keys, flat.data(), false, 0);
  WaitRPCFinish(rpc_id);

  values->reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    std::vector<float> row(embedding_dim);
    std::memcpy(row.data(), flat.data() + i * embedding_dim, FLAGS_value_size);
    values->push_back(std::move(row));
  }
  RevokeRPCResource(rpc_id);
  return 0;
}

int AllShardsParameterClientWrapper::GetParameter(
    base::ConstArray<uint64_t> keys,
    float* values,
    bool isAsync,
    int async_req_id) {
  BatchRequest batch;
  batch.user_buffer = values;

  for (const auto& chunk : BuildChunks(keys)) {
    void* recv = clients_[chunk.shard]->GetReceiveBuffer(
        petps::FixedSlotResponseBytes(chunk.keys.size(), FLAGS_value_size));
    int rpc_id = clients_[chunk.shard]->GetParameter(
        base::ConstArray<uint64_t>(chunk.keys),
        static_cast<float*>(recv),
        isAsync,
        async_req_id);
    batch.shard_rpcs.push_back(PendingShardRpc{
        chunk.shard,
        rpc_id,
        chunk.positions,
        recv,
        chunk.keys.size(),
    });
  }

  const std::uint64_t batch_id = batch_rpc_id_acc_++;
  batches_[batch_id]           = std::move(batch);
  if (!isAsync) {
    WaitRPCFinish(static_cast<int>(batch_id));
  }
  return static_cast<int>(batch_id);
}

void AllShardsParameterClientWrapper::InitThread() {
  for (auto* client : clients_) {
    client->InitThread();
  }
}

void AllShardsParameterClientWrapper::Barrier(const std::string& ss, int k) {
  CHECK(!clients_.empty());
  clients_.front()->Barrier(ss, k);
}

void* AllShardsParameterClientWrapper::GetReceiveBuffer(size_t size) {
  return new char[size];
}

bool AllShardsParameterClientWrapper::QueryRPCFinished(int rpc_id) {
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    if (!clients_[pending.shard]->QueryRPCFinished(pending.rpc_id)) {
      return false;
    }
  }

  AssembleIfNeeded(&it->second);
  return true;
}

void AllShardsParameterClientWrapper::WaitRPCFinish(int rpc_id) {
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    clients_[pending.shard]->WaitRPCFinish(pending.rpc_id);
  }

  AssembleIfNeeded(&it->second);
}

void AllShardsParameterClientWrapper::RevokeRPCResource(int rpc_id) {
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    clients_[pending.shard]->RevokeRPCResource(pending.rpc_id);
  }

  batches_.erase(it);
}

int AllShardsParameterClientWrapper::PutParameter(
    const std::vector<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  CHECK_EQ(keys.size(), values.size());

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_values(num_shards_);

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const int shard = PartitionKey(keys[i]);
    shard_keys[shard].push_back(keys[i]);
    shard_values[shard].push_back(values[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    for (std::size_t offset = 0; offset < shard_keys[shard].size();
         offset += FLAGS_max_kv_num_per_request) {
      const std::size_t end = std::min(
          offset + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
          shard_keys[shard].size());
      std::vector<uint64_t> key_slice(
          shard_keys[shard].begin() + offset, shard_keys[shard].begin() + end);
      std::vector<std::vector<float>> value_slice(
          shard_values[shard].begin() + offset,
          shard_values[shard].begin() + end);
      int rc = clients_[shard]->PutParameter(key_slice, value_slice);
      if (rc != 0) {
        return rc;
      }
    }
  }

  return 0;
}
