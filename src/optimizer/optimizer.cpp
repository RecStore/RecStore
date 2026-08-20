#include "optimizer.h"
#include "ps/local_shm/local_shm_stage_report.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace {

std::vector<uint64_t> CollectReaderKeys(const ParameterCompressReader* reader) {
  const int size = reader->item_size();
  std::vector<uint64_t> keys;
  keys.reserve(size);
  for (int i = 0; i < size; ++i) {
    keys.push_back(reader->item(i)->key);
  }
  return keys;
}

void ValidateFlatUpdateArgs(const base::ConstArray<uint64_t>& keys,
                            const float* grads,
                            int64_t num_rows,
                            int64_t embedding_dim) {
  if (grads == nullptr) {
    throw std::runtime_error("UpdateFlat grads pointer is null");
  }
  if (num_rows < 0 || embedding_dim <= 0) {
    throw std::runtime_error("UpdateFlat invalid rows/dim");
  }
  if (keys.Size() != static_cast<size_t>(num_rows)) {
    throw std::runtime_error("UpdateFlat keys size mismatch");
  }
}

void CheckUpdateKeyTag(uint64_t key, TAG_TYPE expected, const std::string& table) {
  const TAG_TYPE got = ExtractKeyTag(key);
  if (got != expected) {
    throw std::runtime_error(
        "update key tag mismatch for table " + table + ": got " +
        std::to_string(static_cast<unsigned>(got)) + " expected " +
        std::to_string(static_cast<unsigned>(expected)));
  }
}

int InitOrReuseTensor(
    std::unordered_map<std::string, SparseTensor*>* tensor_map,
    const std::string& name,
    TensorType type,
    TAG_TYPE tag,
    std::vector<uint64_t> shape,
    BaseKV* base_kv) {
  auto it = tensor_map->find(name);
  if (it != tensor_map->end()) {
    if (it->second->Tag() != tag ||
        it->second->EmbeddingDim() != static_cast<int64_t>(shape[1])) {
      throw std::runtime_error(
          "table already registered with a different tag/dim: " + name);
    }
    return static_cast<int>(tag);
  }
  auto* tensor           = new SparseTensor();
  std::string mutable_name = name;
  tensor->init(mutable_name, type, tag, shape, base_kv);
  (*tensor_map)[name] = tensor;
  return static_cast<int>(tag);
}

} // namespace

std::unique_ptr<Optimizer> CreateOptimizer(const json& config) {
  if (!config.is_object()) {
    throw std::invalid_argument("cache_ps.optimizer must be an object");
  }
  const std::string type = config.value("type", "SGD");
  const float learning_rate = config.value("learning_rate", 0.01f);
  if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
    throw std::invalid_argument("cache_ps.optimizer.learning_rate is invalid");
  }
  if (type == "SGD") return std::make_unique<SGD>(learning_rate);
  if (type == "RowWiseAdagrad") {
    const float epsilon = config.value("epsilon", 1e-10f);
    if (!std::isfinite(epsilon) || epsilon < 0.0f) {
      throw std::invalid_argument("cache_ps.optimizer.epsilon is invalid");
    }
    return std::make_unique<RowWiseAdaGrad>(learning_rate, epsilon);
  }
  if (type == "AdamW") {
    const float beta1 = config.value("beta1", 0.9f);
    const float beta2 = config.value("beta2", 0.98f);
    const float epsilon = config.value("epsilon", 1e-8f);
    const float weight_decay = config.value("weight_decay", 0.0f);
    if (!std::isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !std::isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !std::isfinite(epsilon) || epsilon < 0.0f ||
        !std::isfinite(weight_decay) || weight_decay < 0.0f) {
      throw std::invalid_argument("cache_ps.optimizer AdamW parameters are invalid");
    }
    return std::make_unique<AdamW>(
        learning_rate, beta1, beta2, epsilon, weight_decay);
  }
  throw std::invalid_argument("Unsupported cache_ps.optimizer.type: " + type);
}

int SGD::Init(const std::vector<std::string> table_name,
              const EmbeddingTableConfig& config,
              BaseKV* base_kv) {
  LOG(INFO) << "SGD::Init called with " << table_name.size() << " table(s)";
  int param_tag = -1;
  const int k   = TensorsPerTable();
  for (const auto& name : table_name) {
    LOG(INFO) << "  Initializing table: '" << name << "' with shape ["
              << config.num_embeddings << ", " << config.embedding_dim
              << "] table_id=" << config.table_id;
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    param_tag                   = InitOrReuseTensor(
        &tensor_map_,
        name,
        PARAMETER,
        MakeTensorTag(config.table_id, 0, k),
        shape,
        base_kv);
  }
  LOG(INFO) << "SGD::Init completed. tensor_map_ now has " << tensor_map_.size()
            << " entries";
  return param_tag;
}

void SGD::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    throw std::runtime_error("Table not found: " + table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);
  if (!keys.empty()) {
    CheckUpdateKeyTag(keys[0], it->second->Tag(), table);
  }

  std::vector<base::ConstArray<float>> current_values;
  it->second->BatchGet(keys, &current_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0) {
      // If key not found, we fallback to Put to initialize it
      std::vector<float> zero_init(item->dim, 0.0f);
      for (int j = 0; j < item->dim; ++j) {
        zero_init[j] = -learning_rate_ * item->data()[j];
      }
      std::string val_str(
          (char*)zero_init.data(), zero_init.size() * sizeof(float));
      it->second->Put(item->key, val_str, tid);
      continue;
    }

    float* data = const_cast<float*>(current_values[i].Data());
    int dim     = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      data[j] -= learning_rate_ * item->data()[j];
    }
  }
}

void SGD::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    throw std::runtime_error("Table not found: " + table);
  }
  if (it->second->EmbeddingDim() != embedding_dim) {
    throw std::runtime_error(
        "SGD::UpdateFlat embedding_dim mismatch for table " + table);
  }
  if (keys.Size() > 0) {
    CheckUpdateKeyTag(keys[0], it->second->Tag(), table);
  }

  const auto direct_update_start = std::chrono::steady_clock::now();
  if (it->second->ApplySgdUpdateFlat(
          keys, grads, num_rows, embedding_dim, learning_rate_, tid)) {
    recstore::ReportLocalShmStageMetric(
        "sgd_update_direct_us",
        recstore::LocalShmElapsedUs(direct_update_start));
    return;
  }

  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  const auto batch_get_start = std::chrono::steady_clock::now();
  std::vector<base::ConstArray<float>> current_values;
  it->second->BatchGet(key_vec, &current_values, tid);
  recstore::ReportLocalShmStageMetric(
      "sgd_update_batch_get_us", recstore::LocalShmElapsedUs(batch_get_start));

  const auto apply_start = std::chrono::steady_clock::now();
  int64_t missing_rows   = 0;
  for (int64_t row = 0; row < num_rows; ++row) {
    const float* row_grad = grads + row * embedding_dim;
    const auto& current   = current_values[static_cast<size_t>(row)];
    if (current.Size() == 0) {
      ++missing_rows;
      std::vector<float> zero_init(static_cast<size_t>(embedding_dim), 0.0f);
      for (int64_t col = 0; col < embedding_dim; ++col) {
        zero_init[static_cast<size_t>(col)] = -learning_rate_ * row_grad[col];
      }
      std::string val_str(reinterpret_cast<char*>(zero_init.data()),
                          zero_init.size() * sizeof(float));
      it->second->Put(keys[static_cast<size_t>(row)], val_str, tid);
      continue;
    }
    if (static_cast<int64_t>(current.Size()) != embedding_dim) {
      throw std::runtime_error(
          "SGD::UpdateFlat embedding_dim mismatch for table " + table);
    }

    float* data = const_cast<float*>(current.Data());
#pragma omp simd
    for (int64_t col = 0; col < embedding_dim; ++col) {
      data[col] -= learning_rate_ * row_grad[col];
    }
  }
  recstore::ReportLocalShmStageMetric(
      "sgd_update_apply_us", recstore::LocalShmElapsedUs(apply_start));
  recstore::ReportLocalShmStageMetric(
      "sgd_update_missing_rows", static_cast<double>(missing_rows));
}

int AdaGrad::Init(const std::vector<std::string> table_name,
                  const EmbeddingTableConfig& config,
                  BaseKV* base_kv) {
  int param_tag = -1;
  const int k   = TensorsPerTable();
  for (const auto& name : table_name) {
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    param_tag                   = InitOrReuseTensor(
        &tensor_map_,
        name,
        PARAMETER,
        MakeTensorTag(config.table_id, 0, k),
        shape,
        base_kv);
    InitOrReuseTensor(
        &tensor_map_,
        name + "_accumulated_grad",
        MOMENT_1,
        MakeTensorTag(config.table_id, 1, k),
        shape,
        base_kv);
  }
  return param_tag;
}

void AdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Accumulated gradient table not found: " + acc_table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);
  if (!keys.empty()) {
    CheckUpdateKeyTag(keys[0], param_it->second->Tag(), table);
  }

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0 || acc_values[i].Size() == 0) {
      // Fallback to sequential initialization if not found
      // (This is rare in training but kept for robustness)
      continue;
    }

    float* param_data = const_cast<float*>(current_values[i].Data());
    float* acc_data   = const_cast<float*>(acc_values[i].Data());
    int dim           = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      acc_data[j] += item->data()[j] * item->data()[j];
      float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[j]) + epsilon_);
      param_data[j] -= adaptive_lr * item->data()[j];
    }
  }
}

void AdaGrad::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Accumulated gradient table not found: " + acc_table);
  }

  if (keys.Size() > 0) {
    CheckUpdateKeyTag(keys[0], param_it->second->Tag(), table);
  }
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(key_vec, &current_values, tid);
  acc_it->second->BatchGet(key_vec, &acc_values, tid);

  for (int64_t row = 0; row < num_rows; ++row) {
    const auto& current = current_values[static_cast<size_t>(row)];
    const auto& acc     = acc_values[static_cast<size_t>(row)];
    if (current.Size() == 0 || acc.Size() == 0) {
      continue;
    }
    if (static_cast<int64_t>(current.Size()) != embedding_dim ||
        static_cast<int64_t>(acc.Size()) != embedding_dim) {
      throw std::runtime_error(
          "AdaGrad::UpdateFlat embedding_dim mismatch for table " + table);
    }

    const float* row_grad = grads + row * embedding_dim;
    float* param_data     = const_cast<float*>(current.Data());
    float* acc_data       = const_cast<float*>(acc.Data());
#pragma omp simd
    for (int64_t col = 0; col < embedding_dim; ++col) {
      acc_data[col] += row_grad[col] * row_grad[col];
      float adaptive_lr =
          learning_rate_ / (std::sqrt(acc_data[col]) + epsilon_);
      param_data[col] -= adaptive_lr * row_grad[col];
    }
  }
}

int RowWiseAdaGrad::Init(const std::vector<std::string> table_name,
                         const EmbeddingTableConfig& config,
                         BaseKV* base_kv) {
  int param_tag = -1;
  const int k   = TensorsPerTable();
  for (const auto& name : table_name) {
    std::vector<uint64_t> shape     = {config.num_embeddings, config.embedding_dim};
    std::vector<uint64_t> acc_shape = {config.num_embeddings, 1};
    param_tag                       = InitOrReuseTensor(
        &tensor_map_,
        name,
        PARAMETER,
        MakeTensorTag(config.table_id, 0, k),
        shape,
        base_kv);
    InitOrReuseTensor(
        &tensor_map_,
        name + "_rowwise_accumulated_grad",
        MOMENT_1,
        MakeTensorTag(config.table_id, 1, k),
        acc_shape,
        base_kv);
  }
  return param_tag;
}

void RowWiseAdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_rowwise_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Row-wise accumulated gradient table not found: " + acc_table);
  }

  int size                   = reader->item_size();
  std::vector<uint64_t> keys = CollectReaderKeys(reader);
  if (!keys.empty()) {
    CheckUpdateKeyTag(keys[0], param_it->second->Tag(), table);
  }

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item           = reader->item(i);
    const auto& current        = current_values[static_cast<size_t>(i)];
    const auto& acc            = acc_values[static_cast<size_t>(i)];
    const int64_t expected_dim = param_it->second->EmbeddingDim();
    if (item->dim != expected_dim ||
        (current.Size() != 0 && current.Size() != expected_dim) ||
        (acc.Size() != 0 && acc.Size() != 1)) {
      throw std::runtime_error(
          "RowWiseAdaGrad::Update embedding_dim mismatch for table " + table);
    }
    const int dim = item->dim;

    float grad_square_mean = 0.0;
#pragma omp simd reduction(+ : grad_square_mean)
    for (int j = 0; j < dim; ++j) {
      grad_square_mean += item->data()[j] * item->data()[j];
    }
    grad_square_mean /= dim;

    float accumulated_grad = acc.Size() == 0 ? 0.0f : acc.Data()[0];
    accumulated_grad += grad_square_mean;

    const float adaptive_lr =
        learning_rate_ / (std::sqrt(accumulated_grad) + epsilon_);
    if (current.Size() == 0) {
      std::vector<float> initial_value(static_cast<size_t>(dim), 0.0f);
      for (int j = 0; j < dim; ++j) {
        initial_value[static_cast<size_t>(j)] = -adaptive_lr * item->data()[j];
      }
      const std::string value(
          reinterpret_cast<const char*>(initial_value.data()),
          initial_value.size() * sizeof(float));
      param_it->second->Put(item->key, value, tid);
    } else {
      float* param_data = const_cast<float*>(current.Data());
#pragma omp simd
      for (int j = 0; j < dim; ++j) {
        param_data[j] -= adaptive_lr * item->data()[j];
      }
    }

    if (acc.Size() == 0) {
      const std::string value(
          reinterpret_cast<const char*>(&accumulated_grad), sizeof(float));
      acc_it->second->Put(item->key, value, tid);
    } else {
      const_cast<float*>(acc.Data())[0] = accumulated_grad;
    }
  }
}

void RowWiseAdaGrad::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_rowwise_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Row-wise accumulated gradient table not found: " + acc_table);
  }

  if (keys.Size() > 0) {
    CheckUpdateKeyTag(keys[0], param_it->second->Tag(), table);
  }
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(key_vec, &current_values, tid);
  acc_it->second->BatchGet(key_vec, &acc_values, tid);

  for (int64_t row = 0; row < num_rows; ++row) {
    const auto& current = current_values[static_cast<size_t>(row)];
    const auto& acc     = acc_values[static_cast<size_t>(row)];
    if ((current.Size() != 0 &&
         static_cast<int64_t>(current.Size()) != embedding_dim) ||
        (acc.Size() != 0 && acc.Size() != 1)) {
      throw std::runtime_error(
          "RowWiseAdaGrad::UpdateFlat embedding_dim mismatch for table " +
          table);
    }

    const float* row_grad  = grads + row * embedding_dim;
    float grad_square_mean = 0.0f;
#pragma omp simd reduction(+ : grad_square_mean)
    for (int64_t col = 0; col < embedding_dim; ++col) {
      grad_square_mean += row_grad[col] * row_grad[col];
    }
    grad_square_mean /= static_cast<float>(embedding_dim);

    float accumulated_grad = acc.Size() == 0 ? 0.0f : acc.Data()[0];
    accumulated_grad += grad_square_mean;
    const float adaptive_lr =
        learning_rate_ / (std::sqrt(accumulated_grad) + epsilon_);

    if (current.Size() == 0) {
      std::vector<float> initial_value(
          static_cast<size_t>(embedding_dim), 0.0f);
      for (int64_t col = 0; col < embedding_dim; ++col) {
        initial_value[static_cast<size_t>(col)] = -adaptive_lr * row_grad[col];
      }
      const std::string value(
          reinterpret_cast<const char*>(initial_value.data()),
          initial_value.size() * sizeof(float));
      param_it->second->Put(keys[static_cast<size_t>(row)], value, tid);
    } else {
      float* param_data = const_cast<float*>(current.Data());
#pragma omp simd
      for (int64_t col = 0; col < embedding_dim; ++col) {
        param_data[col] -= adaptive_lr * row_grad[col];
      }
    }

    if (acc.Size() == 0) {
      const std::string value(
          reinterpret_cast<const char*>(&accumulated_grad), sizeof(float));
      acc_it->second->Put(keys[static_cast<size_t>(row)], value, tid);
    } else {
      const_cast<float*>(acc.Data())[0] = accumulated_grad;
    }
  }
}

namespace {

// The step is stored as a tagged scalar in its own table.  Keep its key away
// from normal embedding ids (the top 8 bits are reserved for TensorType).
constexpr uint64_t kAdamWStepKey = (std::numeric_limits<uint64_t>::max() >> 8);

} // namespace

int AdamW::Init(const std::vector<std::string> table_name,
                const EmbeddingTableConfig& config,
                BaseKV* base_kv) {
  for (const auto& name : table_name) {
    const std::vector<uint64_t> shape = {
        config.num_embeddings, config.embedding_dim};
    auto* param_tensor = new SparseTensor();
    auto mutable_name  = name;
    auto mutable_shape = shape;
    param_tensor->init(
        mutable_name, PARAMETER, PARAMETER, mutable_shape, base_kv);
    tensor_map_[name] = param_tensor;

    auto* first_moment           = new SparseTensor();
    const std::string first_name = name + "_adamw_m";
    auto mutable_first_name      = first_name;
    first_moment->init(
        mutable_first_name, MOMENT_1, MOMENT_1, mutable_shape, base_kv);
    tensor_map_[first_name] = first_moment;

    auto* second_moment           = new SparseTensor();
    const std::string second_name = name + "_adamw_v";
    auto mutable_second_name      = second_name;
    second_moment->init(
        mutable_second_name, MOMENT_2, MOMENT_2, mutable_shape, base_kv);
    tensor_map_[second_name] = second_moment;

    auto* step_tensor                = new SparseTensor();
    const std::string step_name      = name + "_adamw_step";
    auto mutable_step_name           = step_name;
    std::vector<uint64_t> step_shape = {1, 1};
    step_tensor->init(
        mutable_step_name, MOMENT_1, MOMENT_1, step_shape, base_kv);
    tensor_map_[step_name] = step_tensor;
  }
  return static_cast<int>(PARAMETER);
}

void AdamW::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }
  const int size    = reader->item_size();
  const int64_t dim = param_it->second->EmbeddingDim();
  std::vector<uint64_t> keys;
  std::vector<float> grads;
  keys.reserve(size);
  grads.reserve(static_cast<size_t>(size) * static_cast<size_t>(dim));
  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (item->dim != dim) {
      throw std::runtime_error(
          "AdamW::Update embedding_dim mismatch for table " + table);
    }
    keys.push_back(item->key);
    grads.insert(grads.end(), item->data(), item->data() + dim);
  }
  UpdateRows(table, keys.data(), grads.data(), size, dim, tid);
}

void AdamW::UpdateFlat(
    std::string table,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  ValidateFlatUpdateArgs(keys, grads, num_rows, embedding_dim);
  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }
  if (it->second->EmbeddingDim() != embedding_dim) {
    throw std::runtime_error(
        "AdamW::UpdateFlat embedding_dim mismatch for table " + table);
  }
  UpdateRows(table, keys.Data(), grads, num_rows, embedding_dim, tid);
}

void AdamW::UpdateRows(
    const std::string& table,
    const uint64_t* keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    unsigned tid) {
  auto param_it  = tensor_map_.find(table);
  auto first_it  = tensor_map_.find(table + "_adamw_m");
  auto second_it = tensor_map_.find(table + "_adamw_v");
  auto step_it   = tensor_map_.find(table + "_adamw_step");
  if (param_it == tensor_map_.end() || first_it == tensor_map_.end() ||
      second_it == tensor_map_.end() || step_it == tensor_map_.end()) {
    throw std::runtime_error("AdamW state table not found for table " + table);
  }

  std::string step_value;
  step_it->second->Get(kAdamWStepKey, step_value, tid);
  float step =
      step_value.empty() ? 0.0f : base::ConstArray<float>(step_value).Data()[0];
  step += 1.0f;
  const float bias1 = 1.0f - std::pow(beta1_, step);
  const float bias2 = 1.0f - std::pow(beta2_, step);
  if (!(bias1 > 0.0f) || !(bias2 > 0.0f)) {
    throw std::runtime_error("AdamW bias correction underflow");
  }

  std::vector<uint64_t> key_vec(keys, keys + num_rows);
  std::vector<base::ConstArray<float>> params;
  std::vector<base::ConstArray<float>> first;
  std::vector<base::ConstArray<float>> second;
  param_it->second->BatchGet(key_vec, &params, tid);
  first_it->second->BatchGet(key_vec, &first, tid);
  second_it->second->BatchGet(key_vec, &second, tid);

  const float decay       = 1.0f - learning_rate_ * weight_decay_;
  const float correction1 = 1.0f / bias1;
  const float correction2 = 1.0f / bias2;
  for (int64_t row = 0; row < num_rows; ++row) {
    const auto index      = static_cast<size_t>(row);
    const float* row_grad = grads + row * embedding_dim;
    std::vector<float> param(static_cast<size_t>(embedding_dim), 0.0f);
    std::vector<float> moment1(static_cast<size_t>(embedding_dim), 0.0f);
    std::vector<float> moment2(static_cast<size_t>(embedding_dim), 0.0f);
    if (params[index].Size() != 0) {
      if (params[index].Size() != embedding_dim) {
        throw std::runtime_error(
            "AdamW parameter dimension mismatch for table " + table);
      }
      std::copy(params[index].Data(),
                params[index].Data() + embedding_dim,
                param.begin());
    }
    if (first[index].Size() != 0) {
      if (first[index].Size() != embedding_dim) {
        throw std::runtime_error(
            "AdamW first moment dimension mismatch for table " + table);
      }
      std::copy(first[index].Data(),
                first[index].Data() + embedding_dim,
                moment1.begin());
    }
    if (second[index].Size() != 0) {
      if (second[index].Size() != embedding_dim) {
        throw std::runtime_error(
            "AdamW second moment dimension mismatch for table " + table);
      }
      std::copy(second[index].Data(),
                second[index].Data() + embedding_dim,
                moment2.begin());
    }
    for (int64_t col = 0; col < embedding_dim; ++col) {
      const float grad = row_grad[col];
      moment1[static_cast<size_t>(col)] =
          beta1_ * moment1[static_cast<size_t>(col)] + (1.0f - beta1_) * grad;
      moment2[static_cast<size_t>(col)] =
          beta2_ * moment2[static_cast<size_t>(col)] +
          (1.0f - beta2_) * grad * grad;
      const float m_hat = moment1[static_cast<size_t>(col)] * correction1;
      const float v_hat = moment2[static_cast<size_t>(col)] * correction2;
      param[static_cast<size_t>(col)] =
          decay * param[static_cast<size_t>(col)] -
          learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
    const std::string param_value(reinterpret_cast<const char*>(param.data()),
                                  param.size() * sizeof(float));
    const std::string first_value(reinterpret_cast<const char*>(moment1.data()),
                                  moment1.size() * sizeof(float));
    const std::string second_value(
        reinterpret_cast<const char*>(moment2.data()),
        moment2.size() * sizeof(float));
    param_it->second->Put(keys[index], param_value, tid);
    first_it->second->Put(keys[index], first_value, tid);
    second_it->second->Put(keys[index], second_value, tid);
  }
  const std::string next_step(
      reinterpret_cast<const char*>(&step), sizeof(float));
  step_it->second->Put(kAdamWStepKey, next_step, tid);
}
