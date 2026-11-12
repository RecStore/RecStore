#pragma once

#include <algorithm>
#include <atomic>
#include <boost/coroutine2/all.hpp>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <experimental/filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h" // NOLINT
#include "base/timer.h"
#include "parameters.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "optimizer/optimizer.h"
#include "ps/local_shm/local_shm_stage_report.h"
#include "ps/base/sparse_gradient_commit.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include "base/report/report_client.h"
#endif

using boost::coroutines2::coroutine;

static const int KEY_CNT = 12543670;

template <typename key_t>
struct TaskElement {
  TaskElement(const base::ConstArray<key_t>& keys,
              const base::MutableArray<ParameterPack>& packs,
              std::atomic_bool* promise)
      : keys(keys), packs(packs), promise(promise) {}

  TaskElement() {}

  base::ConstArray<key_t> keys;
  base::MutableArray<ParameterPack> packs;
  std::atomic_bool* promise;
};

class CachePS {
public:
  using key_t = uint64_t;

  struct FlatGetProfile {
    std::uint64_t batch_get_ns    = 0;
    std::uint64_t index_lookup_ns = 0;
    std::uint64_t zero_fill_ns    = 0;
    std::uint64_t row_copy_ns     = 0;
    std::uint64_t rows            = 0;
    std::uint64_t value_bytes     = 0;
    std::uint64_t missing_rows    = 0;
  };

  using DirectFixedRow    = BaseKV::DirectFixedRow;
  using RDMABackingRegion = BaseKV::RDMABackingRegion;

  CachePS(json config) {
    LOG(INFO) << "cache ps config: " << config.dump(2);
    if (config.contains("max_batch_keys_size")) {
      LOG(WARNING)
          << "cache_ps.max_batch_keys_size is deprecated and ignored; use "
             "distributed_client.max_keys_per_request for client batching";
    }
    const auto value_config =
        config.at("base_kv_config").value("value", json::object());
    const auto tiering = value_config.value("tiering", json::object());
    if (tiering.value("online_admission", false) &&
        config.value("optimizer", json::object()).value("type", "SGD") !=
            "SGD") {
      throw std::invalid_argument(
          "online tiered admission currently requires SGD");
    }
    BaseKVConfig kv_config;
    kv_config.num_threads_ = config["num_threads"].get<int>();
    kv_config.json_config_ = config["base_kv_config"];
    auto r                 = base::ResolveEngine(kv_config);
    base_kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        r.engine, r.cfg));
    optimizer_ = CreateOptimizer(config.value("optimizer", json::object()));
  }

  ~CachePS() {}

  std::string ExtraResultFields() const {
    base_kv_->DrainBackgroundWork();
    return base_kv_->ExtraResultFields();
  }

  bool Initialize(const std::vector<std::string>& model_config_path,
                  const std::vector<std::string>& emb_file_path) {
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(checkpoint_mu_);
    base_kv_->clear();
    active_checkpoint_identity_.clear();
    checkpoint_dirty_.store(
        base_kv_->CheckpointRecordCount() != 0, std::memory_order_relaxed);
  }

  void LoadFakeData(int64_t key_capacity, int value_size) {
    std::lock_guard<std::mutex> lock(checkpoint_mu_);
    checkpoint_dirty_.store(true, std::memory_order_relaxed);
    base_kv_->LoadFakeData(key_capacity, value_size);
  }

  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path) {
    // base_kv_->loadCkpt();
    // LoadFakeData(KEY_CNT);
    return true;
  }

  bool SaveCheckpoint(const std::string& path, const std::string& metadata) {
    const std::string identity = CheckpointIdentity(metadata);
    std::lock_guard<std::mutex> lock(checkpoint_mu_);
    if (!base_kv_->SaveCheckpoint(path, metadata)) {
      LOG(ERROR) << "Failed to save checkpoint: " << path;
      return false;
    }
    active_checkpoint_identity_ = identity;
    checkpoint_dirty_.store(false, std::memory_order_relaxed);
    return true;
  }

  bool LoadCheckpoint(const std::string& path,
                      const std::string& expected_metadata) {
    const std::string identity = CheckpointIdentity(expected_metadata);
    std::lock_guard<std::mutex> lock(checkpoint_mu_);
    if (!checkpoint_dirty_.load(std::memory_order_relaxed) &&
        active_checkpoint_identity_ == identity) {
      return true;
    }
    if (checkpoint_dirty_.load(std::memory_order_relaxed)) {
      throw std::runtime_error(
          "checkpoint load rejected: parameter server has unsaved updates");
    }
    if (!active_checkpoint_identity_.empty()) {
      throw std::runtime_error(
          "checkpoint load rejected: active checkpoint identity mismatch");
    }
    if (base_kv_->CheckpointRecordCount() != 0) {
      throw std::runtime_error(
          "checkpoint load rejected: parameter server is not fresh");
    }
    try {
      if (!base_kv_->LoadCheckpoint(path, expected_metadata)) {
        checkpoint_dirty_.store(
            base_kv_->CheckpointRecordCount() != 0, std::memory_order_relaxed);
        LOG(ERROR) << "Failed to load checkpoint: " << path;
        return false;
      }
    } catch (...) {
      checkpoint_dirty_.store(
          base_kv_->CheckpointRecordCount() != 0, std::memory_order_relaxed);
      throw;
    }
    active_checkpoint_identity_ = identity;
    checkpoint_dirty_.store(false, std::memory_order_relaxed);
    return true;
  }

  void PutSingleParameter(
      const uint64_t key, const void* data, const int dim, const int tid) {
    base_kv_->Put(key, std::string_view((char*)data, dim * sizeof(float)), tid);
    MarkCheckpointDirty();
  }

  void PutDenseParameterBatch(
      const uint64_t* keys,
      const float* values,
      int key_count,
      int embedding_dim,
      int tid) {
    if (key_count <= 0 || embedding_dim <= 0) {
      return;
    }
    base::ConstArray<uint64_t> key_array(keys, key_count);
    std::vector<base::ConstArray<float>> value_slices;
    value_slices.reserve(static_cast<std::size_t>(key_count));
    for (int i = 0; i < key_count; ++i) {
      value_slices.emplace_back(values + i * embedding_dim, embedding_dim);
    }
    base_kv_->BatchPut(key_array, &value_slices, tid);
    MarkCheckpointDirty();
  }

  void PutSingleParameter(const ParameterCompressItem* item, int tid) {
    auto key = item->key;
    auto dim = item->dim;
    base_kv_->Put(
        key, std::string_view((char*)item->data(), dim * sizeof(float)), tid);
    MarkCheckpointDirty();
  }

  void PutParameter(coroutine<void>::push_type& sink,
                    const ParameterCompressReader* reader,
                    int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back(
          (float*)reader->item(i)->data(), reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(sink, keys, &values, tid);
    MarkCheckpointDirty();
  }

  void PutParameter(const ParameterCompressReader* reader, int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back(
          (float*)reader->item(i)->data(), reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(keys, &values, tid);
    MarkCheckpointDirty();
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack& pack, int tid) {
    std::vector<uint64_t> keys = {key};
    base::ConstArray<uint64_t> keys_array(keys);
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(keys_array, &values, tid);
    base::ConstArray<float> value = values[0];

    if (value.Size() == 0) {
      pack.key      = key;
      pack.dim      = 0;
      pack.emb_data = nullptr;
      RECSTORE_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key      = key;
    pack.dim      = value.Size();
    pack.emb_data = value.Data();
    // LOG(ERROR) << "Get key " << key << " dim " << pack.dim;
    return true;
  }

  bool GetParameterRun2Completion(base::ConstArray<uint64_t> keys,
                                  std::vector<ParameterPack>& packs,
                                  int tid) {
#ifdef ENABLE_PERF_REPORT
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    const auto batch_get_start = std::chrono::steady_clock::now();
    std::vector<base::ConstArray<float>> values;
    base_kv_->BatchGet(keys, &values, tid);
    recstore::ReportLocalShmStageMetric(
        "cache_ps_get_batch_get_us",
        recstore::LocalShmElapsedUs(batch_get_start));

    const auto pack_build_start = std::chrono::steady_clock::now();
    for (int i = 0; i < keys.Size(); i++) {
      packs.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }
    recstore::ReportLocalShmStageMetric(
        "cache_ps_get_pack_us", recstore::LocalShmElapsedUs(pack_build_start));

#ifdef ENABLE_PERF_REPORT
    auto end_time = std::chrono::high_resolution_clock::now();
    double start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();

    std::string report_id = "cache_ps::GetParameterRun2Completion|" +
                            std::to_string(static_cast<uint64_t>(start_us));

    report("embread_stages",
           report_id.c_str(),
           "duration_us",
           static_cast<double>(duration));

    report("embread_stages",
           report_id.c_str(),
           "request_size",
           static_cast<double>(keys.Size()));

    std::string unique_id =
        "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
    FlameGraphData fg_data = {
        "cache_ps::GetParameterRun2Completion",
        start_us,
        3, // level
        static_cast<double>(duration),
        static_cast<double>(duration)};
    report_flame_graph("emb_read_flame_map", unique_id.c_str(), fg_data);
#endif
    return true;
  }

  bool GetParameterFlat(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t embedding_dim,
      int tid,
      FlatGetProfile* profile = nullptr) {
    if (values == nullptr) {
      LOG(ERROR) << "GetParameterFlat values pointer is null";
      return false;
    }
    if (num_rows < 0 || embedding_dim <= 0) {
      LOG(ERROR) << "GetParameterFlat invalid shape rows=" << num_rows
                 << " dim=" << embedding_dim;
      return false;
    }
    if (keys.Size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "GetParameterFlat keys size mismatch " << keys.Size()
                 << " vs " << num_rows;
      return false;
    }

    const auto batch_get_start = std::chrono::steady_clock::now();
    BaseKV::BatchGetFlatStats flat_stats;
    BaseKV::BatchGetFlatStats* flat_stats_ptr =
        profile != nullptr ? &flat_stats : nullptr;
    const bool flat_ok = base_kv_->BatchGetFlat(
        keys, values, num_rows, embedding_dim, tid, flat_stats_ptr);
    if (flat_ok) {
      if (profile != nullptr) {
        profile->batch_get_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast< std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - batch_get_start)
                .count());
        profile->rows = static_cast<std::uint64_t>(num_rows);
        profile->value_bytes =
            static_cast<std::uint64_t>(num_rows) *
            static_cast<std::uint64_t>(embedding_dim) * sizeof(float);
        profile->zero_fill_ns    = flat_stats.zero_fill_ns;
        profile->index_lookup_ns = flat_stats.index_lookup_ns;
        profile->row_copy_ns     = flat_stats.row_copy_ns;
        profile->missing_rows    = flat_stats.missing_rows;
      }
      recstore::ReportLocalShmStageMetric(
          "cache_ps_get_batch_get_us",
          recstore::LocalShmElapsedUs(batch_get_start));
      recstore::ReportLocalShmStageMetric("cache_ps_get_copy_us", 0);
      return true;
    }

    std::vector<base::ConstArray<float>> value_slices;
    value_slices.reserve(static_cast<std::size_t>(num_rows));
    base_kv_->BatchGet(keys, &value_slices, tid);
    if (profile != nullptr) {
      profile->batch_get_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - batch_get_start)
              .count());
      profile->rows = static_cast<std::uint64_t>(num_rows);
      profile->value_bytes =
          static_cast<std::uint64_t>(num_rows) *
          static_cast<std::uint64_t>(embedding_dim) * sizeof(float);
    }
    recstore::ReportLocalShmStageMetric(
        "cache_ps_get_batch_get_us",
        recstore::LocalShmElapsedUs(batch_get_start));
    if (value_slices.size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "GetParameterFlat BatchGet returned " << value_slices.size()
                 << " rows, expected " << num_rows;
      return false;
    }

    for (int64_t row = 0; row < num_rows; ++row) {
      const auto& slice = value_slices[static_cast<size_t>(row)];
      if (slice.Size() != 0 &&
          static_cast<int64_t>(slice.Size()) != embedding_dim) {
        LOG(ERROR) << "GetParameterFlat embedding_dim mismatch at row=" << row
                   << " key=" << keys[static_cast<size_t>(row)] << " expected="
                   << embedding_dim << " actual=" << slice.Size();
        return false;
      }
    }

    const auto row_copy_start          = std::chrono::steady_clock::now();
    std::uint64_t missing_zero_fill_ns = 0;
    for (int64_t row = 0; row < num_rows; ++row) {
      const auto& slice = value_slices[static_cast<size_t>(row)];
      if (slice.Size() > 0) {
        std::memcpy(values + row * embedding_dim,
                    slice.Data(),
                    static_cast<std::size_t>(embedding_dim) * sizeof(float));
      } else {
        const auto missing_zero_start = std::chrono::steady_clock::now();
        std::memset(values + row * embedding_dim,
                    0,
                    static_cast<std::size_t>(embedding_dim) * sizeof(float));
        if (profile != nullptr) {
          missing_zero_fill_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast< std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - missing_zero_start)
                  .count());
          ++profile->missing_rows;
        }
      }
    }
    if (profile != nullptr) {
      profile->zero_fill_ns = missing_zero_fill_ns;
      profile->row_copy_ns  = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - row_copy_start)
              .count());
    }
    recstore::ReportLocalShmStageMetric(
        "cache_ps_get_copy_us", recstore::LocalShmElapsedUs(row_copy_start));
    return true;
  }

  bool ProbeParameterIndex(base::ConstArray<uint64_t> keys,
                           int tid,
                           FlatGetProfile* profile = nullptr) {
    const auto batch_get_start = std::chrono::steady_clock::now();
    BaseKV::BatchGetFlatStats flat_stats;
    BaseKV::BatchGetFlatStats* flat_stats_ptr =
        profile != nullptr ? &flat_stats : nullptr;
    const bool ok = base_kv_->BatchGetIndexOnly(keys, tid, flat_stats_ptr);
    if (profile != nullptr) {
      profile->batch_get_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - batch_get_start)
              .count());
      profile->rows         = static_cast<std::uint64_t>(keys.Size());
      profile->value_bytes  = 0;
      profile->missing_rows = flat_stats.missing_rows;
    }
    return ok;
  }

  bool GetParameterDirectFixedRows(
      base::ConstArray<uint64_t> keys,
      int64_t num_rows,
      int64_t embedding_dim,
      int tid,
      std::vector<DirectFixedRow>* rows,
      FlatGetProfile* profile = nullptr) {
    const auto batch_get_start = std::chrono::steady_clock::now();
    BaseKV::BatchGetFlatStats flat_stats;
    BaseKV::BatchGetFlatStats* flat_stats_ptr =
        profile != nullptr ? &flat_stats : nullptr;
    const bool ok = base_kv_->BatchGetDirectFixedRows(
        keys, num_rows, embedding_dim, tid, rows, flat_stats_ptr);
    if (profile != nullptr) {
      profile->batch_get_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - batch_get_start)
              .count());
      profile->rows = static_cast<std::uint64_t>(num_rows);
      profile->value_bytes =
          static_cast<std::uint64_t>(num_rows) *
          static_cast<std::uint64_t>(embedding_dim) * sizeof(float);
      profile->zero_fill_ns    = flat_stats.zero_fill_ns;
      profile->index_lookup_ns = flat_stats.index_lookup_ns;
      profile->row_copy_ns     = 0;
      profile->missing_rows    = flat_stats.missing_rows;
    }
    return ok;
  }

  RDMABackingRegion GetRDMABackingRegion() const {
    return base_kv_->GetRDMABackingRegion();
  }

  bool GetParameterRun2Completion(
      coroutine<void>::push_type& sink,
      base::ConstArray<uint64_t> keys,
      std::vector<ParameterPack>& pack,
      int tid) {
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(sink, keys, &values, tid);

    for (int i = 0; i < keys.Size(); i++) {
      pack.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }
    return true;
  }

  /// optimizer interface

  void FailSparseGradientSession(const std::string& table,
                                 const std::string& payload) {
    sparse_gradient_commits_.FailSession(table, payload);
  }

  int64_t SparseGradientCommit(const std::string& table,
                               std::string payload,
                               unsigned tid,
                               int exact_owner_rank = -1) {
    recstore::SparseGradientCommitHeader h;
    try {
      h = recstore::DecodeSparseGradientCommit(payload);
      if (h.version == 4 &&
          (exact_owner_rank < 0 || exact_owner_rank >= 2 ||
           (h.rows && h.rank != static_cast<uint32_t>(exact_owner_rank))))
        throw std::invalid_argument(
            "Exact replacement requires the server-bound owner rank");
      const auto* sgd = dynamic_cast<SGD*>(optimizer_.get());
      if (!sgd || sgd->LearningRate() != h.learning_rate) {
        throw std::runtime_error("Sparse gradient commit requires matching "
                                 "client/PS SGD learning rates");
      }
      const auto config = table_configs_.find(table);
      if (config == table_configs_.end() ||
          config->second.embedding_dim != h.dim) {
        throw std::runtime_error(
            "Sparse gradient commit table or dimension mismatch");
      }
    } catch (...) {
      sparse_gradient_commits_.FailSession(table, payload);
      throw;
    }
    return sparse_gradient_commits_.Exchange(
        table,
        std::move(payload),
        [&](const std::vector<uint64_t>& keys,
            const std::vector<float>& grads) {
          if (h.version == 4) {
            // Experimental initialized DRAM namespace, with no external
            // writers. This lock excludes CachePS checkpoint/load/clear while
            // direct rows are validated and copied. The client must read old
            // weights only after the previous global commit; V4 does not prove
            // their freshness.
            std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mu_);
            const auto tag =
                MakeTensorTag(table_configs_.at(table).table_id, 0, 1);
            for (uint64_t key : keys)
              if (ExtractKeyTag(key) != tag)
                throw std::runtime_error(
                    "Exact replacement key/table tag mismatch");
            if (!recstore::CanonicalAllFinite(grads.data(), grads.size()))
              throw std::invalid_argument(
                  "Non-finite exact replacement candidate");
            const auto backing     = base_kv_->GetRDMABackingRegion();
            const size_t row_bytes = size_t(h.dim) * sizeof(float);
            if (!backing.data || backing.size < row_bytes)
              throw std::runtime_error(
                  "Exact replacement requires direct DRAM backing");
            std::vector<base::ConstArray<float>> current;
            base_kv_->BatchGet(base::ConstArray<uint64_t>(keys), &current, tid);
            if (current.size() != keys.size())
              throw std::runtime_error("Exact replacement row count mismatch");
            const auto begin = reinterpret_cast<uintptr_t>(backing.data);
            for (const auto& row : current) {
              const auto address = reinterpret_cast<uintptr_t>(row.Data());
              if (row.Size() != h.dim || address < begin ||
                  address - begin > backing.size - row_bytes)
                throw std::runtime_error(
                    "Exact replacement requires existing direct D128 rows");
            }
            // The whole owner step passes every check before any value is
            // written.
            for (size_t row = 0; row < keys.size(); ++row)
              std::memcpy(const_cast<float*>(current[row].Data()),
                          grads.data() + row * h.dim,
                          row_bytes);
            MarkCheckpointDirty();
            return;
          }
          if (!UpdateParameterFlat(
                  table,
                  base::ConstArray<uint64_t>(keys.data(), keys.size()),
                  grads.data(),
                  keys.size(),
                  h.dim,
                  tid)) {
            throw std::runtime_error("Sparse gradient commit optimizer failed");
          }
        },
        exact_owner_rank);
  }

  int InitTable(const std::string& table_name,
                uint64_t num_embeddings,
                uint64_t embedding_dim,
                uint64_t table_id = 0) {
    std::lock_guard<std::mutex> lock(checkpoint_mu_);
    EmbeddingTableConfig config{num_embeddings, embedding_dim, table_id};
    const auto existing = table_configs_.find(table_name);
    if (existing != table_configs_.end()) {
      const auto& old = existing->second;
      if (old.num_embeddings == num_embeddings &&
          old.embedding_dim == embedding_dim && old.table_id == table_id) {
        return optimizer_->Init({table_name}, config, base_kv_.get());
      }
      LOG(ERROR) << "Embedding table config mismatch for '" << table_name
                 << "': existing=[" << old.num_embeddings << ", "
                 << old.embedding_dim << "] requested=[" << num_embeddings
                 << ", " << embedding_dim << "]";
      return -1;
    }
    const int tag = optimizer_->Init({table_name}, config, base_kv_.get());
    table_configs_.emplace(table_name, config);
    return tag;
  }

  bool UpdateParameter(const std::string& table_name,
                       const ParameterCompressReader* reader,
                       unsigned tid,
                       float learning_rate = 0.0f) {
    if (!optimizer_) {
      LOG(ERROR) << "Optimizer not initialized. Please call InitTable first.";
      return false;
    }
    if (!CheckLearningRate(table_name, learning_rate)) {
      return false;
    }

    optimizer_->Update(table_name, reader, tid);
    MarkCheckpointDirty();
    return true;
  }

  bool UpdateParameterFlat(
      const std::string& table_name,
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid,
      float learning_rate = 0.0f) {
    if (grads == nullptr) {
      LOG(ERROR) << "UpdateParameterFlat grads pointer is null";
      return false;
    }
    if (num_rows < 0 || embedding_dim <= 0) {
      LOG(ERROR) << "UpdateParameterFlat invalid shape rows=" << num_rows
                 << " dim=" << embedding_dim;
      return false;
    }
    if (keys.Size() != static_cast<size_t>(num_rows)) {
      LOG(ERROR) << "UpdateParameterFlat keys size mismatch " << keys.Size()
                 << " vs " << num_rows;
      return false;
    }
    if (!optimizer_) {
      LOG(ERROR) << "Optimizer not initialized. Please call InitTable first.";
      return false;
    }
    if (!CheckLearningRate(table_name, learning_rate)) {
      return false;
    }
    optimizer_->UpdateFlat(
        table_name, keys, grads, num_rows, embedding_dim, tid);
    MarkCheckpointDirty();
    return true;
  }

private:
  // The server optimizer owns the learning rate. A client that advertises a
  // different rate must fail instead of being silently ignored.
  bool CheckLearningRate(const std::string& table_name, float client_lr) const {
    if (client_lr <= 0.0f) {
      return true;
    }
    const float server_lr = optimizer_->LearningRate();
    if (server_lr <= 0.0f) {
      return true;
    }
    const float scale = std::max(1.0f, std::abs(server_lr));
    if (std::abs(client_lr - server_lr) <= 1e-6f * scale) {
      return true;
    }
    LOG(ERROR) << "UpdateParameter learning rate mismatch for table "
               << table_name << ": client=" << client_lr
               << " server=" << server_lr;
    return false;
  }

  recstore::SparseGradientCommitCoordinator sparse_gradient_commits_;
  void MarkCheckpointDirty() {
    if (!checkpoint_dirty_.load(std::memory_order_relaxed)) {
      checkpoint_dirty_.store(true, std::memory_order_relaxed);
    }
  }

  static std::string CheckpointIdentity(const std::string& metadata) {
    const json parsed = json::parse(metadata);
    if (!parsed.is_object() || !parsed.contains("identity") ||
        !parsed["identity"].is_object() || !parsed.contains("checkpoint_id") ||
        !parsed["checkpoint_id"].is_string() ||
        parsed["checkpoint_id"].get<std::string>().empty()) {
      throw std::invalid_argument(
          "checkpoint metadata requires identity and non-empty checkpoint_id");
    }
    return parsed.dump();
  }

  std::unique_ptr<BaseKV> base_kv_;
  std::unique_ptr<Optimizer> optimizer_;
  std::mutex checkpoint_mu_;
  std::unordered_map<std::string, EmbeddingTableConfig> table_configs_;
  std::string active_checkpoint_identity_;
  std::atomic<bool> checkpoint_dirty_{false};
  std::atomic<bool> stopFlag_{false};
};
