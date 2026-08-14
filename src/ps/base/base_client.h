#pragma once
#include <vector>
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

namespace recstore {
struct EmbeddingTableConfig {
  uint64_t num_embeddings = 0;
  uint64_t embedding_dim  = 0;
  uint64_t table_id       = 0;

  std::string Serialize() const {
    nlohmann::json payload{{"num_embeddings", num_embeddings},
                           {"embedding_dim", embedding_dim},
                           {"table_id", table_id}};
    return payload.dump();
  }
};

enum class PSCommand {
  CLEAR_PS,
  RELOAD_PS,
  LOAD_FAKE_DATA,
  DUMP_FAKE_DATA,
};

class BasePSClient {
  json json_config_;

public:
  explicit BasePSClient(json config) : json_config_(config) {}
  virtual ~BasePSClient() {}

  virtual int
  GetParameter(const base::ConstArray<uint64_t>& keys, float* values) = 0;

  virtual int PutParameter(const base::ConstArray<uint64_t>& keys,
                           const std::vector<std::vector<float>>& values)   = 0;
  virtual int UpdateParameter(const std::string& table_name,
                              const base::ConstArray<uint64_t>& keys,
                              const std::vector<std::vector<float>>* grads) = 0;
  virtual int UpdateParameterFlat(
      const std::string& table_name,
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim) = 0;

  virtual int InitEmbeddingTable(const std::string& table_name,
                                 const EmbeddingTableConfig& config) = 0;
  virtual int
  AsyncGetParameter(const base::ConstArray<uint64_t>& keys, float* values) = 0;

  virtual void Command(PSCommand command) = 0;

  virtual uint64_t
  PrefetchParameter(const base::ConstArray<uint64_t>& keys)               = 0;
  virtual bool IsPrefetchDone(uint64_t prefetch_id)                       = 0;
  virtual void WaitForPrefetch(uint64_t prefetch_id)                      = 0;
  virtual bool GetPrefetchResult(uint64_t prefetch_id,
                                 std::vector<std::vector<float>>* values) = 0;
  virtual bool GetPrefetchResultFlat(
      uint64_t prefetch_id,
      std::vector<float>* values,
      int64_t* num_rows,
      int64_t embedding_dim) = 0;
};

} // namespace recstore
