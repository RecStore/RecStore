#include "allshards_ps_client.h"

#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ps/rdma/rdma_common.h"

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

AllShardsParameterClientWrapper::AllShardsParameterClientWrapper(
    const std::vector<BaseParameterClient*>& clients,
    int num_shards,
    const std::string& hash_method,
    const std::vector<int>& shard_ids)
    : BaseParameterClient("", 0, 0),
      clients_(clients),
      num_shards_(num_shards),
      hash_method_(hash_method) {
  CHECK_EQ(static_cast<int>(clients_.size()), num_shards_);
  if (!shard_ids.empty()) {
    CHECK_EQ(static_cast<int>(shard_ids.size()), num_shards_);
    for (int i = 0; i < num_shards_; ++i) {
      shard_to_client_index_[shard_ids[static_cast<std::size_t>(i)]] = i;
    }
  } else {
    for (int i = 0; i < num_shards_; ++i) {
      shard_to_client_index_[i] = i;
    }
  }
}

std::vector<AllShardsParameterClientWrapper::ShardChunk>
AllShardsParameterClientWrapper::BuildChunks(
    base::ConstArray<uint64_t> keys) const {
  return recstore::shard_routing::BuildChunks(
      keys,
      num_shards_,
      hash_method_,
      shard_to_client_index_,
      static_cast<std::size_t>(FLAGS_max_kv_num_per_request));
}

void AllShardsParameterClientWrapper::WaitShardRpcsCooperatively(
    const std::vector<PendingShardRpc>& shard_rpcs) const {
  using Coroutine = boost::coroutines2::coroutine<void>;
  std::vector<std::unique_ptr<Coroutine::pull_type>> waiters;
  waiters.reserve(shard_rpcs.size());
  for (const auto& pending : shard_rpcs) {
    waiters.emplace_back(std::make_unique<Coroutine::pull_type>(
        [this, pending](Coroutine::push_type& sink) {
          auto* client =
              clients_[static_cast<std::size_t>(pending.client_index)];
          while (!client->QueryRPCFinished(pending.rpc_id)) {
            sink();
          }
          client->WaitRPCFinish(pending.rpc_id);
        }));
  }

  while (!waiters.empty()) {
    for (auto it = waiters.begin(); it != waiters.end();) {
      auto& waiter = *it;
      if (*waiter) {
        (*waiter)();
      }
      if (!*waiter) {
        it = waiters.erase(it);
      } else {
        ++it;
      }
    }
    if (!waiters.empty()) {
      std::this_thread::yield();
    }
  }
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
  const auto* status_word =
      petps::FixedSlotStatusWord(flat.data(), keys.Size(), FLAGS_value_size);
  if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }

  petps::CopyFlatRowsToVectors(
      flat.data(),
      keys.Size(),
      static_cast<std::size_t>(embedding_dim),
      values);
  RevokeRPCResource(rpc_id);
  return 0;
}

int AllShardsParameterClientWrapper::GetParameter(
    base::ConstArray<uint64_t> keys,
    float* values,
    bool isAsync,
    int async_req_id) {
  BatchRequest batch;
  batch.user_buffer     = values;
  batch.total_key_count = keys.Size();
  auto* batch_status_word =
      petps::FixedSlotStatusWord(values, keys.Size(), FLAGS_value_size);
  *batch_status_word = static_cast<std::int32_t>(petps::RpcStatus::kPending);

  for (const auto& chunk : BuildChunks(keys)) {
    void* recv = clients_[chunk.client_index]->GetReceiveBuffer(
        chunk.keys.size() * static_cast<std::size_t>(FLAGS_value_size) +
        sizeof(std::int32_t));
    int rpc_id = clients_[chunk.client_index]->GetParameter(
        base::ConstArray<uint64_t>(chunk.keys),
        static_cast<float*>(recv),
        isAsync,
        async_req_id);
    batch.shard_rpcs.push_back(PendingShardRpc{
        chunk.shard_id,
        chunk.client_index,
        rpc_id,
        chunk.positions,
        recv,
        chunk.keys.size(),
    });
  }

  std::uint64_t batch_id = 0;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    batch_id = batch_rpc_id_acc_++;
    if (batch_id >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("allshards batch rpc id overflow int range: " +
                               std::to_string(batch_id));
    }
    batches_[batch_id] = std::move(batch);
  }
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
  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    if (!clients_[pending.client_index]->QueryRPCFinished(pending.rpc_id)) {
      return false;
    }
  }

  return recstore::shard_routing::FinalizeBatchIfNeeded(&it->second,
                                                        FLAGS_value_size);
}

void AllShardsParameterClientWrapper::WaitRPCFinish(int rpc_id) {
  std::vector<PendingShardRpc> shard_rpcs;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(rpc_id);
    CHECK(it != batches_.end());
    if (it->second.assembled) {
      return;
    }
    shard_rpcs = it->second.shard_rpcs;
  }

  WaitShardRpcsCooperatively(shard_rpcs);

  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(rpc_id);
    CHECK(it != batches_.end());
    recstore::shard_routing::FinalizeBatchIfNeeded(&it->second,
                                                   FLAGS_value_size);
  }
}

void AllShardsParameterClientWrapper::RevokeRPCResource(int rpc_id) {
  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    clients_[pending.client_index]->RevokeRPCResource(pending.rpc_id);
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
    const int shard =
        recstore::shard_routing::PartitionKey(keys[i], num_shards_, hash_method_);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_values[static_cast<std::size_t>(shard)].push_back(values[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    const int client_index = shard_to_client_index_.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
      const std::size_t end = std::min(
          offset + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
          shard_keys[static_cast<std::size_t>(shard)].size());
      std::vector<uint64_t> key_slice(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      std::vector<std::vector<float>> value_slice(
          shard_values[static_cast<std::size_t>(shard)].begin() + offset,
          shard_values[static_cast<std::size_t>(shard)].begin() + end);
      int rc = clients_[client_index]->PutParameter(key_slice, value_slice);
      if (rc != 0) {
        return rc;
      }
    }
  }

  return 0;
}

int AllShardsParameterClientWrapper::InitEmbeddingTable(
    const std::string& table_name,
    std::uint64_t num_embeddings,
    std::uint64_t embedding_dim) {
  for (auto* client : clients_) {
    const int rc =
        client->InitEmbeddingTable(table_name, num_embeddings, embedding_dim);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

int AllShardsParameterClientWrapper::UpdateParameter(
    const std::string& table_name,
    base::ConstArray<uint64_t> keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_grads(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard =
        recstore::shard_routing::PartitionKey(keys[i], num_shards_, hash_method_);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_grads[static_cast<std::size_t>(shard)].push_back((*grads)[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    if (shard_keys[static_cast<std::size_t>(shard)].empty()) {
      continue;
    }
    const int client_index = shard_to_client_index_.at(shard);
    const int rc           = clients_[client_index]->UpdateParameter(
        table_name,
        base::ConstArray<uint64_t>(shard_keys[static_cast<std::size_t>(shard)]),
        &shard_grads[static_cast<std::size_t>(shard)]);
    if (rc != 0) {
      return rc;
    }
  }

  return 0;
}
