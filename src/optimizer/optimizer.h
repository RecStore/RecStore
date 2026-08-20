#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "sparse_tensor.h"
#include "ps/base/base_client.h"
#include "ps/base/parameters.h"

using ::ParameterCompressReader;
using recstore::EmbeddingTableConfig;

class Optimizer {
protected:
  std::unordered_map<std::string, SparseTensor*> tensor_map_;

public:
  virtual ~Optimizer() {
    for (auto& pair : tensor_map_) {
      delete pair.second;
    }
  }

  virtual int Init(const std::vector<std::string> table_name,
                   const EmbeddingTableConfig& config,
                   BaseKV* base_kv) = 0;
  virtual int TensorsPerTable() const = 0;

  virtual void Update(std::string table,
                      const ParameterCompressReader* reader,
                      unsigned tid) = 0;
  virtual void UpdateFlat(
      std::string table,
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid) = 0;
};

class SGD : public Optimizer {
private:
  float learning_rate_;

public:
  explicit SGD(float lr = 0.01) : learning_rate_(lr) {}

  int Init(const std::vector<std::string> table_name,
           const EmbeddingTableConfig& config,
           BaseKV* base_kv) override;
  int TensorsPerTable() const override { return 1; }
  void Update(std::string table,
              const ParameterCompressReader* reader,
              unsigned tid) override;
  void UpdateFlat(std::string table,
                  const base::ConstArray<uint64_t>& keys,
                  const float* grads,
                  int64_t num_rows,
                  int64_t embedding_dim,
                  unsigned tid) override;
};

class AdaGrad : public Optimizer {
private:
  float learning_rate_;
  float epsilon_;

public:
  explicit AdaGrad(float lr = 0.01, float epsilon = 1e-10)
      : learning_rate_(lr), epsilon_(epsilon) {}

  int Init(const std::vector<std::string> table_name,
           const EmbeddingTableConfig& config,
           BaseKV* base_kv) override;
  int TensorsPerTable() const override { return 2; }
  void Update(std::string table,
              const ParameterCompressReader* reader,
              unsigned tid) override;
  void UpdateFlat(std::string table,
                  const base::ConstArray<uint64_t>& keys,
                  const float* grads,
                  int64_t num_rows,
                  int64_t embedding_dim,
                  unsigned tid) override;
};

class RowWiseAdaGrad : public Optimizer {
private:
  float learning_rate_;
  float epsilon_;

public:
  explicit RowWiseAdaGrad(float lr = 0.01, float epsilon = 1e-10)
      : learning_rate_(lr), epsilon_(epsilon) {}

  int Init(const std::vector<std::string> table_name,
           const EmbeddingTableConfig& config,
           BaseKV* base_kv) override;
  int TensorsPerTable() const override { return 2; }
  void Update(std::string table,
              const ParameterCompressReader* reader,
              unsigned tid) override;
  void UpdateFlat(std::string table,
                  const base::ConstArray<uint64_t>& keys,
                  const float* grads,
                  int64_t num_rows,
                  int64_t embedding_dim,
                  unsigned tid) override;
};

// Sparse AdamW keeps first/second moments and a persisted step counter in
// RecStore.  Updates are applied to rows present in the submitted sparse
// gradient (the same sparse visibility contract as the existing optimizers).
class AdamW : public Optimizer {
private:
  float learning_rate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  float weight_decay_;

  void UpdateRows(const std::string& table,
                  const uint64_t* keys,
                  const float* grads,
                  int64_t num_rows,
                  int64_t embedding_dim,
                  unsigned tid);

public:
  explicit AdamW(float lr           = 0.001,
                 float beta1        = 0.9,
                 float beta2        = 0.98,
                 float epsilon      = 1e-8,
                 float weight_decay = 0.0)
      : learning_rate_(lr),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay) {}

  int Init(const std::vector<std::string> table_name,
           const EmbeddingTableConfig& config,
           BaseKV* base_kv) override;
  int TensorsPerTable() const override { return 4; }
  void Update(std::string table,
              const ParameterCompressReader* reader,
              unsigned tid) override;
  void UpdateFlat(std::string table,
                  const base::ConstArray<uint64_t>& keys,
                  const float* grads,
                  int64_t num_rows,
                  int64_t embedding_dim,
                  unsigned tid) override;
};

std::unique_ptr<Optimizer> CreateOptimizer(const json& config);
