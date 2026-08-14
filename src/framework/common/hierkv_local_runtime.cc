#include "framework/common/hierkv_local_runtime.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace recstore {

namespace {
constexpr float kHierKVLearningRate = 0.01f;

void ValidateKeys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " +
                                base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional, but has " +
                                std::to_string(keys.dim()) + " dimensions.");
  }
}

void ValidateEmbeddings(const base::RecTensor& embeddings,
                        const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " +
        base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " +
        std::to_string(embeddings.dim()) + " dimensions.");
  }
}
} // namespace

struct HierKVLocalRuntime::Impl {
  std::mutex mu;
  std::unordered_map<std::string, EmbeddingTableConfig> table_configs;
  std::unordered_map<uint64_t, std::vector<float>> store;
  std::unordered_map<uint64_t, std::vector<std::vector<float>>>
      prefetch_results;
  uint64_t next_prefetch_id = 1;
};

bool IsHierKVBackendName(const std::string& backend_name) {
  return backend_name == kHierKVBackendName;
}

HierKVLocalRuntime::Impl& HierKVLocalRuntime::impl() {
  static Impl runtime;
  return runtime;
}

bool HierKVLocalRuntime::InitEmbeddingTable(
    const std::string& table_name, const EmbeddingTableConfig& config) {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  auto it = state.table_configs.find(table_name);
  if (it != state.table_configs.end()) {
    if (it->second.embedding_dim != config.embedding_dim ||
        it->second.num_embeddings != config.num_embeddings) {
      throw std::runtime_error(
          "HierKV table already exists with different shape: " + table_name);
    }
    return true;
  }
  state.table_configs.emplace(table_name, config);
  return true;
}

void HierKVLocalRuntime::Write(const base::RecTensor& keys,
                               const base::RecTensor& values) {
  ValidateKeys(keys);
  ValidateEmbeddings(values, "Values");
  const int64_t num_rows = keys.shape(0);
  if (values.shape(0) != num_rows) {
    throw std::invalid_argument("HierKV write row count mismatch.");
  }
  const int64_t embedding_dim = values.shape(1);
  auto& state                 = impl();
  std::lock_guard<std::mutex> lock(state.mu);

  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* value_data  = values.data_as<float>();
  for (int64_t row = 0; row < num_rows; ++row) {
    const float* start = value_data + row * embedding_dim;
    state.store[key_data[row]].assign(start, start + embedding_dim);
  }
}

void HierKVLocalRuntime::Read(const base::RecTensor& keys,
                              base::RecTensor& values) {
  ValidateKeys(keys);
  ValidateEmbeddings(values, "Values");
  const int64_t num_rows = keys.shape(0);
  if (values.shape(0) != num_rows) {
    throw std::invalid_argument("HierKV read row count mismatch.");
  }
  const int64_t embedding_dim = values.shape(1);
  auto& state                 = impl();
  std::lock_guard<std::mutex> lock(state.mu);

  const uint64_t* key_data = keys.data_as<uint64_t>();
  float* out_data          = values.data_as<float>();
  for (int64_t row = 0; row < num_rows; ++row) {
    auto it        = state.store.find(key_data[row]);
    float* out_row = out_data + row * embedding_dim;
    if (it == state.store.end()) {
      std::fill(out_row, out_row + embedding_dim, 0.0f);
      continue;
    }
    if (static_cast<int64_t>(it->second.size()) != embedding_dim) {
      throw std::runtime_error("HierKV stored row dim mismatch for key " +
                               std::to_string(key_data[row]));
    }
    std::memcpy(out_row,
                it->second.data(),
                static_cast<size_t>(embedding_dim) * sizeof(float));
  }
}

void HierKVLocalRuntime::Update(const std::string& table_name,
                                const base::RecTensor& keys,
                                const base::RecTensor& grads) {
  ValidateKeys(keys);
  ValidateEmbeddings(grads, "Grads");
  const int64_t num_rows = keys.shape(0);
  if (grads.shape(0) != num_rows) {
    throw std::invalid_argument("HierKV update row count mismatch.");
  }
  const int64_t embedding_dim = grads.shape(1);
  auto& state                 = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  if (!table_name.empty()) {
    auto table_it = state.table_configs.find(table_name);
    if (table_it != state.table_configs.end() &&
        static_cast<int64_t>(table_it->second.embedding_dim) != embedding_dim) {
      throw std::runtime_error(
          "HierKV table dim mismatch for update: " + table_name);
    }
  }

  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* grad_data   = grads.data_as<float>();
  for (int64_t row = 0; row < num_rows; ++row) {
    auto& value = state.store[key_data[row]];
    if (value.empty()) {
      value.assign(static_cast<size_t>(embedding_dim), 0.0f);
    } else if (static_cast<int64_t>(value.size()) != embedding_dim) {
      throw std::runtime_error("HierKV stored row dim mismatch for key " +
                               std::to_string(key_data[row]));
    }
    for (int64_t col = 0; col < embedding_dim; ++col) {
      value[static_cast<size_t>(col)] -=
          kHierKVLearningRate * grad_data[row * embedding_dim + col];
    }
  }
}

uint64_t HierKVLocalRuntime::Prefetch(const base::RecTensor& keys,
                                      int64_t embedding_dim) {
  ValidateKeys(keys);
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  const uint64_t prefetch_id = state.next_prefetch_id++;
  auto& rows                 = state.prefetch_results[prefetch_id];
  const uint64_t* key_data   = keys.data_as<uint64_t>();
  const int64_t num_rows     = keys.shape(0);
  rows.resize(static_cast<size_t>(num_rows));
  for (int64_t row = 0; row < num_rows; ++row) {
    auto it = state.store.find(key_data[row]);
    if (it == state.store.end()) {
      rows[static_cast<size_t>(row)] =
          embedding_dim > 0
              ? std::vector<float>(static_cast<size_t>(embedding_dim), 0.0f)
              : std::vector<float>{};
    } else {
      rows[static_cast<size_t>(row)] = it->second;
    }
  }
  return prefetch_id;
}

bool HierKVLocalRuntime::IsPrefetchDone(uint64_t prefetch_id) {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  return state.prefetch_results.find(prefetch_id) !=
         state.prefetch_results.end();
}

void HierKVLocalRuntime::WaitForPrefetch(uint64_t prefetch_id) {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  if (state.prefetch_results.find(prefetch_id) ==
      state.prefetch_results.end()) {
    throw std::runtime_error(
        "unknown HierKV prefetch_id: " + std::to_string(prefetch_id));
  }
}

void HierKVLocalRuntime::ConsumePrefetch(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  auto& state = impl();
  std::lock_guard<std::mutex> lock(state.mu);
  auto it = state.prefetch_results.find(prefetch_id);
  if (it == state.prefetch_results.end()) {
    throw std::runtime_error(
        "unknown HierKV prefetch_id: " + std::to_string(prefetch_id));
  }
  *values = it->second;
  state.prefetch_results.erase(it);
}

void HierKVLocalRuntime::ConsumePrefetchFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  std::vector<std::vector<float>> rows;
  ConsumePrefetch(prefetch_id, &rows);
  *num_rows = static_cast<int64_t>(rows.size());
  values->assign(
      static_cast<size_t>(*num_rows) * static_cast<size_t>(embedding_dim),
      0.0f);
  for (int64_t row = 0; row < *num_rows; ++row) {
    const auto& src = rows[static_cast<size_t>(row)];
    const int64_t copy_dim =
        std::min<int64_t>(embedding_dim, static_cast<int64_t>(src.size()));
    if (copy_dim <= 0) {
      continue;
    }
    std::memcpy(values->data() + row * embedding_dim,
                src.data(),
                static_cast<size_t>(copy_dim) * sizeof(float));
  }
}

HierKVLocalRuntime& GetHierKVLocalRuntime() {
  static HierKVLocalRuntime runtime;
  return runtime;
}

} // namespace recstore
