#pragma once

// Shared shard-routing primitives for RDMA parameter clients.
//
// RDMAPSClientAdapter (rdma_ps_client_adapter.h) fans a batch of keys out to
// per-shard RPCs and reassembles the responses into a single caller buffer.
// These structs and the partition/finalize logic are shared by the adapter's
// sync and raw-async paths, so they live here instead of being copy-pasted.

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
  void* recv_buffer     = nullptr; // Caller-visible response buffer.
  std::size_t key_count = 0;       // Keys in this shard chunk.
  bool with_found       = false;   // Response carries a key_count-byte mask.
  bool released         = false;   // True once the shard RPC slot was returned.
};

// A caller batch spread across one or more PendingShardRpc.
struct BatchRequest {
  float* user_buffer = nullptr; // Final output buffer owned by caller.
  // Per-row hit mask for the whole batch; only set when with_found is true.
  std::uint8_t* user_found = nullptr;
  bool with_found          = false; // Chunks asked for hit masks.
  bool assembled           = false; // True once shard RPCs are merged.
  // The trailing status word lives just past the last row, which callers that
  // hand over their own exactly-sized buffer cannot host. Those callers read
  // the status from WaitBatch instead and turn this off.
  bool write_status_word      = true;
  std::size_t total_key_count = 0; // Total keys across all shards.
  int value_size              = 0; // Row bytes for this batch.
  std::int32_t status_code =
      static_cast<std::int32_t>(petps::RpcStatus::kPending);
  std::vector<std::vector<char>> receive_buffers;
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
inline int
PartitionKey(uint64_t key, int num_shards, const std::string& hash_method) {
  if (num_shards <= 0) {
    throw std::invalid_argument("num_shards must be positive");
  }
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
    std::size_t max_keys_per_rpc,
    bool interleave = false) {
  if (max_keys_per_rpc == 0) {
    throw std::invalid_argument("max_keys_per_rpc must be positive");
  }
  if (hash_method != "city_hash" && hash_method != "simple_mod") {
    throw std::invalid_argument(
        "unsupported shard hash method: " + hash_method);
  }
  std::vector<std::vector<uint64_t>> shard_keys(num_shards);
  std::vector<std::vector<std::size_t>> shard_positions(num_shards);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i], num_shards, hash_method);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_positions[static_cast<std::size_t>(shard)].push_back(i);
  }

  std::vector<std::vector<ShardChunk>> chunks_by_shard(num_shards);
  std::size_t total_chunks = 0;
  std::size_t max_chunks   = 0;
  for (int shard = 0; shard < num_shards; ++shard) {
    const int client_index     = shard_to_client_index.at(shard);
    const auto& keys_for_shard = shard_keys[static_cast<std::size_t>(shard)];
    const auto& positions_for_shard =
        shard_positions[static_cast<std::size_t>(shard)];
    auto& shard_chunks = chunks_by_shard[static_cast<std::size_t>(shard)];
    shard_chunks.reserve(
        (keys_for_shard.size() + max_keys_per_rpc - 1) / max_keys_per_rpc);
    for (std::size_t offset = 0; offset < keys_for_shard.size();
         offset += max_keys_per_rpc) {
      const std::size_t end =
          std::min(offset + max_keys_per_rpc, keys_for_shard.size());
      ShardChunk chunk;
      chunk.shard_id     = shard;
      chunk.client_index = client_index;
      chunk.keys.assign(
          keys_for_shard.begin() + offset, keys_for_shard.begin() + end);
      chunk.positions.assign(positions_for_shard.begin() + offset,
                             positions_for_shard.begin() + end);
      shard_chunks.push_back(std::move(chunk));
    }
    total_chunks += shard_chunks.size();
    max_chunks = std::max(max_chunks, shard_chunks.size());
  }

  std::vector<ShardChunk> chunks;
  chunks.reserve(total_chunks);
  if (!interleave) {
    for (auto& shard_chunks : chunks_by_shard) {
      for (auto& chunk : shard_chunks) {
        chunks.push_back(std::move(chunk));
      }
    }
    return chunks;
  }

  // Keep one request per shard in the submission window instead of filling it
  // with the first shard's chunks. The executor can then overlap shards even
  // when a single shard has more chunks than the window capacity.
  for (std::size_t round = 0; round < max_chunks; ++round) {
    for (auto& shard_chunks : chunks_by_shard) {
      if (round < shard_chunks.size()) {
        chunks.push_back(std::move(shard_chunks[round]));
      }
    }
  }
  return chunks;
}

inline std::int32_t
DecodeBatchStatus(const BatchRequest& batch, int value_size) {
  for (const auto& pending : batch.shard_rpcs) {
    const auto* status_word = petps::FixedSlotStatusWord(
        pending.recv_buffer, pending.key_count, value_size, pending.with_found);
    if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
      return *status_word;
    }
  }
  return static_cast<std::int32_t>(petps::RpcStatus::kOk);
}

inline void MergePendingRows(
    const PendingShardRpc& pending,
    const float* shard_values,
    float* user_buffer,
    std::uint8_t* user_found,
    int value_size) {
  const int embedding_dim = value_size / sizeof(float);
  const std::uint8_t* shard_found =
      pending.with_found
          ? petps::FixedSlotFoundMask(
                shard_values, pending.key_count, value_size)
          : nullptr;
  for (std::size_t i = 0; i < pending.original_positions.size(); ++i) {
    const std::size_t position = pending.original_positions[i];
    std::memcpy(user_buffer + position * embedding_dim,
                shard_values + i * embedding_dim,
                value_size);
    if (user_found != nullptr) {
      user_found[position] = shard_found[i];
    }
  }
}

inline void MergeBatchRows(BatchRequest* batch, int value_size) {
  for (const auto& pending : batch->shard_rpcs) {
    MergePendingRows(
        pending,
        static_cast<const float*>(pending.recv_buffer),
        batch->user_buffer,
        batch->user_found,
        value_size);
  }
}

inline void WriteBatchStatus(BatchRequest* batch, int value_size) {
  auto* batch_status_word = reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(batch->user_buffer) +
      batch->total_key_count *
          (static_cast<std::size_t>(value_size) + (batch->with_found ? 1 : 0)));
  *batch_status_word = batch->status_code;
}

} // namespace shard_routing
} // namespace recstore
