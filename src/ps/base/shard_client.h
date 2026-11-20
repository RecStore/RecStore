#pragma once

#include <cstdint>
#include <string>

#include "base/array.h"
#include "base/tensor.h"
#include "ps/base/base_client.h"

namespace recstore {

// Single-shard parameter-server client. One instance talks to one server
// shard; routing/fan-out/merge across shards lives in ShardedClient
// (dist_sharded_client.h). All methods return 0 on success and a negative
// value on error, except InitEmbeddingTable which returns the non-negative
// table tag.
class ShardClient : public SparseGradientCommitClient {
public:
  virtual ~ShardClient() = default;

  virtual int GetParameter(const base::ConstArray<uint64_t>& keys,
                           base::RecTensor& values) = 0;
  // Optional per-row hit mask for flat reads: found[row] is 1 when the key is
  // present and the row was filled. Backends that cannot report presence
  // return false and leave the caller's buffers untouched.
  virtual bool GetParameterFlatWithFound(
      const base::ConstArray<uint64_t>& keys,
      float* values,
      uint8_t* found,
      int64_t num_rows,
      int64_t embedding_dim) {
    (void)keys;
    (void)values;
    (void)found;
    (void)num_rows;
    (void)embedding_dim;
    return false;
  }
  virtual int PutParameter(const base::ConstArray<uint64_t>& keys,
                           const base::RecTensor& values)            = 0;
  virtual int UpdateParameter(const std::string& table_name,
                              const base::ConstArray<uint64_t>& keys,
                              const base::RecTensor& grads,
                              float learning_rate = 0.0f)            = 0;
  virtual int InitEmbeddingTable(const std::string& table_name,
                                 const EmbeddingTableConfig& config) = 0;
  virtual void Command(PSCommand command)                            = 0;

  virtual bool
  SaveCheckpoint(const std::string& path, const std::string& metadata) {
    (void)path;
    (void)metadata;
    return false;
  }
  virtual bool
  LoadCheckpoint(const std::string& path, const std::string& metadata) {
    (void)path;
    (void)metadata;
    return false;
  }

  virtual uint64_t
  PrefetchParameter(const base::ConstArray<uint64_t>& keys) = 0;
  virtual bool IsPrefetchDone(uint64_t prefetch_id)         = 0;
  virtual void WaitForPrefetch(uint64_t prefetch_id)        = 0;
  virtual bool
  GetPrefetchResult(uint64_t prefetch_id, base::RecTensor& values) = 0;
};

} // namespace recstore
