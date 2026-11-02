#include "framework/pytorch/op_base.h"

#include <torch/extension.h>

#include "framework/pytorch/op_internal.h"

namespace recstore {
namespace framework {
namespace {

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  return std::get<0>(CachedReadTorch(keys, embedding_dim));
}

void emb_read_into_torch(const torch::Tensor& keys,
                          const torch::Tensor& values) {
  CheckRows(keys, values);
  TORCH_CHECK(keys.device().is_cpu() && values.device().is_cpu(),
              "emb_read_into requires CPU keys and destination");
  TORCH_CHECK(values.size(1) > 0 && !values.requires_grad(),
              "emb_read_into requires positive dimension and no autograd");
  if (!keys.numel())
    return;
  auto k = ToRecTensor(keys, base::DataType::UINT64);
  auto v = ToRecTensor(values, base::DataType::FLOAT32);
  // Synchronous transport completion lets the caller own pinned staging.
  GetConcreteKVClientOp()->EmbRead(k, v);
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse "
      "optimizer.");
}

void emb_update_table_torch(const std::string& table_name,
                            const torch::Tensor& keys,
                            const torch::Tensor& grads) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  CheckRows(keys, grads);

  if (keys.size(0) == 0) {
    return;
  }

  torch::Tensor cpu_keys  = ToCpuTensor(keys);
  torch::Tensor cpu_grads = ToCpuTensor(grads);

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  GetConcreteKVClientOp()->EmbUpdateCached(
      table_name, rec_keys, rec_grads, CacheActions(keys, grads.size(1)));
}

int64_t emb_update_async_torch(const std::string& table_name,
                               const torch::Tensor& keys,
                               const torch::Tensor& grads) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  CheckRows(keys, grads);

  torch::Tensor cpu_keys   = ToCpuTensor(keys);
  torch::Tensor cpu_grads  = ToCpuTensor(grads);
  auto kv_op               = GetConcreteKVClientOp();
  const uint64_t update_id = kv_op->EmbUpdateAsync(
      table_name,
      ToRecTensor(cpu_keys, base::DataType::UINT64),
      ToRecTensor(cpu_grads, base::DataType::FLOAT32),
      CacheActions(keys, grads.size(1)));
  return static_cast<int64_t>(update_id);
}

void emb_update_wait_torch(int64_t update_id) {
  TORCH_CHECK(update_id > 0, "update_id must be positive");
  GetConcreteKVClientOp()->WaitForEmbUpdate(static_cast<uint64_t>(update_id));
}

int64_t init_embedding_table_torch(
    const std::string& table_name,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t table_id = 0) {
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");
  TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");
  TORCH_CHECK(table_id >= 0, "table_id must be non-negative");

  EmbeddingTableConfig cfg{};
  cfg.num_embeddings = static_cast<uint64_t>(num_embeddings);
  cfg.embedding_dim  = static_cast<uint64_t>(embedding_dim);
  cfg.table_id       = static_cast<uint64_t>(table_id);

  return GetConcreteKVClientOp()->InitEmbeddingTableCached(
      table_name, cfg, CacheActions());
}

bool save_checkpoint_torch(const std::string& path,
                           const std::string& metadata) {
  TORCH_CHECK(!path.empty(), "checkpoint path must be non-empty");
  TORCH_CHECK(!metadata.empty(), "checkpoint metadata must be non-empty");
  return GetKVClientOp()->SaveCheckpoint(path, metadata);
}

bool load_checkpoint_torch(const std::string& path,
                           const std::string& metadata) {
  TORCH_CHECK(!path.empty(), "checkpoint path must be non-empty");
  TORCH_CHECK(!metadata.empty(), "checkpoint metadata must be non-empty");
  return GetConcreteKVClientOp()->LoadCheckpointCached(
      path, metadata, CacheActions());
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  CheckRows(keys, values);

  if (keys.size(0) == 0) {
    return;
  }

  torch::Tensor cpu_keys   = ToCpuTensor(keys);
  torch::Tensor cpu_values = ToCpuTensor(values);

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  GetConcreteKVClientOp()->EmbWriteCached(
      rec_keys, rec_values, CacheActions(keys, values.size(1)));
}

void set_ps_config_torch(const std::string& host, int64_t port) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSConfig(host, static_cast<int>(port));
}

void set_ps_backend_torch(const std::string& backend) {
  auto kv_op = GetConcreteKVClientOp();
  kv_op->SetPSBackend(backend);
}

std::string current_ps_backend_torch() {
  auto kv_op = GetConcreteKVClientOp();
  return kv_op->CurrentPSBackend();
}

} // namespace

void RegisterBaseOps(torch::Library& m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_read_into(Tensor keys, Tensor(a!) values) -> ()", emb_read_into_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_update_table", emb_update_table_torch);
  m.def("emb_update_async", emb_update_async_torch);
  m.def("emb_update_wait", emb_update_wait_torch);
  m.def("init_embedding_table", init_embedding_table_torch);
  m.def("save_checkpoint", save_checkpoint_torch);
  m.def("load_checkpoint", load_checkpoint_torch);
  m.def("emb_write", emb_write_torch);
  m.def("set_ps_config", set_ps_config_torch);
  m.def("set_ps_backend", set_ps_backend_torch);
  m.def("current_ps_backend", current_ps_backend_torch);
}

} // namespace framework
} // namespace recstore
