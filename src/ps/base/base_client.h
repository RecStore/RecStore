#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"
#include "base/tensor.h"

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

inline bool IsFloatEmbeddingValues(const base::RecTensor& values,
                                   int64_t num_keys) {
  if (values.dtype() != base::DataType::FLOAT32) {
    return false;
  }
  if (num_keys == 0) {
    return true;
  }
  return values.dim() == 2 && values.shape(0) == num_keys &&
         values.shape(1) > 0 && values.data() != nullptr;
}

inline bool EnsureEmbeddingOutput(base::RecTensor& values, int64_t num_keys) {
  if (values.dim() == 2 && values.shape(0) == 0 && values.shape(1) > 0 &&
      num_keys >= 0) {
    values = base::RecTensor({num_keys, values.shape(1)},
                             base::DataType::FLOAT32);
    return num_keys == 0 || values.data() != nullptr;
  }
  return IsFloatEmbeddingValues(values, num_keys);
}

inline void GatherEmbeddingRows(const base::RecTensor& src,
                                const std::vector<size_t>& src_indices,
                                base::RecTensor& dst) {
  const int64_t D = src.shape(1);
  const float* s  = src.data_as<float>();
  float* d        = dst.data_as<float>();
  for (size_t i = 0; i < src_indices.size(); ++i) {
    std::memcpy(d + i * D,
                s + src_indices[i] * D,
                static_cast<size_t>(D) * sizeof(float));
  }
}

inline void ScatterEmbeddingRows(const base::RecTensor& src,
                                 const std::vector<size_t>& dst_indices,
                                 base::RecTensor& dst) {
  const int64_t D = dst.shape(1);
  const float* s  = src.data_as<float>();
  float* d        = dst.data_as<float>();
  for (size_t i = 0; i < dst_indices.size(); ++i) {
    std::memcpy(d + dst_indices[i] * D,
                s + i * D,
                static_cast<size_t>(D) * sizeof(float));
  }
}

class BasePSClient {
  json json_config_;

public:
  explicit BasePSClient(json config) : json_config_(config) {}
  virtual ~BasePSClient() {}

  virtual int GetParameter(const base::ConstArray<uint64_t>& keys,
                           base::RecTensor& values) = 0;

  virtual int PutParameter(const base::ConstArray<uint64_t>& keys,
                           const base::RecTensor& values) = 0;
  virtual int UpdateParameter(const std::string& table_name,
                              const base::ConstArray<uint64_t>& keys,
                              const base::RecTensor& grads) = 0;

  virtual int InitEmbeddingTable(const std::string& table_name,
                                 const EmbeddingTableConfig& config) = 0;

  virtual void Command(PSCommand command) = 0;

  virtual uint64_t
  PrefetchParameter(const base::ConstArray<uint64_t>& keys) = 0;
  virtual bool IsPrefetchDone(uint64_t prefetch_id)         = 0;
  virtual void WaitForPrefetch(uint64_t prefetch_id)        = 0;
  virtual bool GetPrefetchResult(uint64_t prefetch_id,
                                 base::RecTensor& values)   = 0;

  // Asynchronous update. Backends without a native async path fail at
  // submit time so callers never treat handle 0 as in-flight work.
  virtual uint64_t SubmitUpdateParameterAsync(
      const std::string& table_name,
      const base::ConstArray<uint64_t>& keys,
      const base::RecTensor& grads) {
    (void)table_name;
    (void)keys;
    (void)grads;
    throw std::runtime_error("Async update is not supported by this backend");
  }
  virtual int WaitUpdateParameter(uint64_t update_id) {
    (void)update_id;
    throw std::runtime_error("Async update is not supported by this backend");
  }
};

} // namespace recstore
