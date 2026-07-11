#pragma once

// Shared shard-routing primitives for RDMA parameter clients.
//
// Both AllShardsParameterClientWrapper (allshards_ps_client.h) and
// RDMAPSClientAdapter (rdma_ps_client_adapter.h) fan a batch of keys out to
// per-shard RPCs and reassemble the responses into a single caller buffer.
// The structs and the partition/finalize logic below are identical between the
// two, so they live here instead of being copy-pasted.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/hash.h"
#include "base/log.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rdma_status.h"

namespace recstore {
namespace shard_routing {

// One in-flight per-shard GET RPC and where its rows land in the caller batch.
struct PendingShardRpc {
  int shard_id     = 0;  // Logical shard this RPC belongs to.
  int client_index = 0;  // Underlying client selected for this shard.
  int rpc_id       = -1; // RPC id returned by the shard-local client.
  std::vector<std::size_t> original_positions; // Positions in caller's batch.
  void* recv_buffer     = nullptr;             // Caller-visible response buffer.
  std::size_t key_count = 0;                   // Keys in this shard chunk.
};

// A caller batch spread across one or more PendingShardRpc.
struct BatchRequest {
  float* user_buffer          = nullptr; // Final output buffer owned by caller.
  bool assembled              = false;   // True once shard RPCs are merged.
  std::size_t total_key_count = 0;       // Total keys across all shards.
  std::int32_t status_code =
      static_cast<std::int32_t>(petps::RpcStatus::kPending);
  std::vector<PendingShardRpc> shard_rpcs; // One pending RPC per shard chunk.
};

// A contiguous group of keys routed to a single shard/client.
struct ShardChunk {
  int shard_id     = 0;               // Routed shard id.
  int client_index = 0;               // Client that serves this shard.
  std::vector<uint64_t> keys;         // Keys assigned to this shard chunk.
  std::vector<std::size_t> positions; // Original positions in caller input.
};

// Maps a key to a logical shard using the configured hash method.
inline int PartitionKey(uint64_t key,
                        int num_shards,
                        const std::string& hash_method) {
  CHECK_GT(num_shards, 0);
  if (hash_method == "city_hash") {
    return static_cast<int>(GetHash(key) % static_cast<uint64_t>(num_shards));
  }
  if (hash_method == "simple_mod") {
    return static_cast<int>(key % static_cast<uint64_t>(num_shards));
  }
  throw std::runtime_error("unsupported shard hash method: " + hash_method);
}

// Splits keys into per-shard chunks, each no larger than max_keys_per_rpc.
inline std::vector<ShardChunk> BuildChunks(
    base::ConstArray<uint64_t> keys,
    int num_shards,
    const std::string& hash_method,
    const std::unordered_map<int, int>& shard_to_client_index,
    std::size_t max_keys_per_rpc) {
  std::vector<std::vector<uint64_t>> shard_keys(num_shards);
  std::vector<std::vector<std::size_t>> shard_positions(num_shards);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i], num_shards, hash_method);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_positions[static_cast<std::size_t>(shard)].push_back(i);
  }

  std::vector<ShardChunk> chunks;
  for (int shard = 0; shard < num_shards; ++shard) {
    const int client_index = shard_to_client_index.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += max_keys_per_rpc) {
      const std::size_t end =
          std::min(offset + max_keys_per_rpc,
                   shard_keys[static_cast<std::size_t>(shard)].size());
      ShardChunk chunk;
      chunk.shard_id     = shard;
      chunk.client_index = client_index;
      chunk.keys.assign(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      chunk.positions.assign(
          shard_positions[static_cast<std::size_t>(shard)].begin() + offset,
          shard_positions[static_cast<std::size_t>(shard)].begin() + end);
      chunks.push_back(std::move(chunk));
    }
  }
  return chunks;
}

// Merges completed shard responses into batch->user_buffer and writes the
// trailing batch status word. Returns true iff the whole batch succeeded.
inline bool FinalizeBatchIfNeeded(BatchRequest* batch, int value_size) {
  if (batch == nullptr) {
    return false;
  }
  if (batch->assembled) {
    return batch->status_code ==
           static_cast<std::int32_t>(petps::RpcStatus::kOk);
  }

  batch->status_code = static_cast<std::int32_t>(petps::RpcStatus::kOk);
  for (const auto& pending : batch->shard_rpcs) {
    const auto* status_word = petps::FixedSlotStatusWord(
        pending.recv_buffer, pending.key_count, value_size);
    if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
      batch->status_code = *status_word;
      break;
    }
  }

  const int embedding_dim = value_size / sizeof(float);
  if (batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    for (const auto& pending : batch->shard_rpcs) {
      const float* shard_values =
          static_cast<const float*>(pending.recv_buffer);
      for (std::size_t i = 0; i < pending.original_positions.size(); ++i) {
        std::memcpy(
            batch->user_buffer + pending.original_positions[i] * embedding_dim,
            shard_values + i * embedding_dim,
            value_size);
      }
    }
  }

  auto* batch_status_word = reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(batch->user_buffer) +
      batch->total_key_count * static_cast<std::size_t>(value_size));
  *batch_status_word = batch->status_code;
  batch->assembled   = true;
  return batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk);
}

} // namespace shard_routing
} // namespace recstore
