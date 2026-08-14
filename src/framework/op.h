#pragma once

#include "base/tensor.h"
#include "framework/common/op_runtime_support.h"
#include "ps/base/base_client.h"
#include <cstddef>
#include <memory>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>

using base::RecTensor;

namespace recstore {
enum class InitStrategyType { Normal, Uniform, Xavier, Zero };
struct LocalShmFlatGetHandle;

struct InitStrategy {
  InitStrategy() = delete;
  InitStrategyType type;

  // Optional fields depending on type
  float mean  = 0.0f;
  float std   = 1.0f;
  float lower = -1.0f;
  float upper = 1.0f;

  InitStrategy(InitStrategyType t) : type(t) {}

  static InitStrategy Normal(float mean, float std) {
    InitStrategy s(InitStrategyType::Normal);
    s.mean = mean;
    s.std  = std;
    return s;
  }

  static InitStrategy Uniform(float lower, float upper) {
    InitStrategy s(InitStrategyType::Uniform);
    s.lower = lower;
    s.upper = upper;
    return s;
  }

  static InitStrategy Xavier() {
    return InitStrategy(InitStrategyType::Xavier);
  }
  static InitStrategy Zero() { return InitStrategy(InitStrategyType::Zero); }
};
class CommonOp {
public:
  // keys: uint64_t tensor with shape [N]
  // values: emb.dtype tensor with shape [N, D]

  CommonOp() = default;

  virtual void EmbInit(const RecTensor& keys, const RecTensor& init_values) = 0;
  virtual void EmbInit(const RecTensor& keys, const InitStrategy& strategy) = 0;

  // Core KV APIs (sync)
  virtual void
  EmbRead(const RecTensor& keys, RecTensor& values) = 0; // sync read
  virtual void
  EmbWrite(const RecTensor& keys, const RecTensor& values) = 0; // sync write

  virtual bool
  EmbExists(const RecTensor& keys) = 0; // not urgent, optional existence check
  virtual void
  EmbDelete(const RecTensor& keys) = 0; // not urgent, optional deletion

  // Optional Gradient Hook (can be omitted if optimizer is outside)
  virtual void
  EmbUpdate(const RecTensor& keys, const RecTensor& grads) = 0; // not urgent
  virtual void EmbUpdate(const std::string& table_name,
                         const RecTensor& keys,
                         const RecTensor& grads)           = 0;

  virtual int InitEmbeddingTable(const std::string& table_name,
                                 const EmbeddingTableConfig& config) = 0;

  // Prefetch & write (async)
  virtual uint64_t
  EmbPrefetch(const RecTensor& keys,
              const RecTensor& values) = 0; // async prefetch, returns a unique
                                            // ID to track the prefetch status.
  virtual bool IsPrefetchDone(
      uint64_t prefetch_id) = 0; // returns true if the prefetch identified by
                                 // prefetch_id is complete.
  virtual void WaitForPrefetch(
      uint64_t prefetch_id) = 0; // blocks until the prefetch identified by
                                 // prefetch_id is complete.
  virtual void GetPretchResult(uint64_t prefetch_id,
                               std::vector<std::vector<float>>* values) = 0;
  virtual void GetPretchResultFlat(
      uint64_t prefetch_id,
      std::vector<float>* values,
      int64_t* num_rows,
      int64_t embedding_dim) = 0;

  virtual uint64_t
  EmbWriteAsync(const RecTensor& keys,
                const RecTensor& values) = 0; // async write, returns a unique
                                              // ID to track the write status.
  virtual bool
  IsWriteDone(uint64_t write_id) = 0; // returns true if the asynchronous write
                                      // identified by write_id is complete.
  virtual void
  WaitForWrite(uint64_t write_id) = 0; // blocks until the asynchronous write
                                       // identified by write_id is complete.

  // Persistence
  virtual void SaveToFile(const std::string& path)   = 0; // not urgent
  virtual void LoadFromFile(const std::string& path) = 0; // not urgent

  virtual ~CommonOp() = default;
};

class KVClientOp : public CommonOp {
public:
  KVClientOp();

  void EmbInit(const base::RecTensor& keys,
               const base::RecTensor& init_values) override;
  void EmbInit(const base::RecTensor& keys,
               const InitStrategy& strategy) override;
  void EmbRead(const base::RecTensor& keys, base::RecTensor& values) override;
  void EmbWrite(const base::RecTensor& keys,
                const base::RecTensor& values) override;
  void EmbUpdate(const base::RecTensor& keys,
                 const base::RecTensor& grads) override;
  void EmbUpdate(const std::string& table_name,
                 const base::RecTensor& keys,
                 const base::RecTensor& grads) override;
  uint64_t EmbUpdateAsync(const std::string& table_name,
                          const base::RecTensor& keys,
                          const base::RecTensor& grads);
  void WaitForEmbUpdate(uint64_t update_id);
  int InitEmbeddingTable(const std::string& table_name,
                         const EmbeddingTableConfig& config) override;
  bool EmbExists(const base::RecTensor& keys) override;
  void EmbDelete(const base::RecTensor& keys) override;
  uint64_t EmbPrefetch(const base::RecTensor& keys,
                       const base::RecTensor& values) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  void GetPretchResult(uint64_t prefetch_id,
                       std::vector<std::vector<float>>* values) override;
  void GetPretchResultFlat(uint64_t prefetch_id,
                           std::vector<float>* values,
                           int64_t* num_rows,
                           int64_t embedding_dim) override;
  uint64_t EmbWriteAsync(const base::RecTensor& keys,
                         const base::RecTensor& values) override;
  bool IsWriteDone(uint64_t write_id) override;
  void WaitForWrite(uint64_t write_id) override;
  void SaveToFile(const std::string& path) override;
  void LoadFromFile(const std::string& path) override;
  void SetPSConfig(const std::string& host, int port);
  void SetPSBackend(const std::string& backend);
  std::string CurrentPSBackend() const;
  void LocalLookupFlat(const base::RecTensor& keys, base::RecTensor& values);
  int SubmitLocalLookupFlat(const base::RecTensor& keys,
                            int64_t embedding_dim,
                            LocalShmFlatGetHandle* handle);
  int WaitLocalLookupFlat(LocalShmFlatGetHandle* handle);
  void ReleaseLocalLookupFlat(LocalShmFlatGetHandle* handle);
  bool GetLocalLookupFlatPayloadRegion(const void** base, std::size_t* bytes);
  void LocalUpdateFlat(const std::string& table_name,
                       const base::RecTensor& keys,
                       const base::RecTensor& grads);

private:
  int64_t embedding_dim_;
  std::string ps_backend_name_ = "unknown";
  static BasePSClient* ps_client_;
  static std::unique_ptr<BasePSClient> ps_client_holder_;

#ifdef USE_FAKE_KVCLIENT
  std::unordered_map<uint64_t, std::vector<float>> store_;
  std::mutex mtx_;
  float learning_rate_;
  std::unordered_map<uint64_t, std::vector<std::vector<float>>>
      prefetch_results_;
  uint64_t next_prefetch_id_ = 1;
#endif
};

std::shared_ptr<CommonOp> GetKVClientOp();

} // namespace recstore
