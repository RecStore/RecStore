#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/factory.h"
#include "storage/index/dram/extendible_hash_index.h"
#include "storage/index/dram/pet_hash_index.h"
#include "storage/index/dram/unordered_map_index.h"
#include "storage/index/utils/hash.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/hybrid_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class KVEngineComposite : public BaseKV {
public:
  KVEngineComposite(std::unique_ptr<Index> index,
                    std::unique_ptr<ValueStore> value_store,
                    int num_threads = 0)
      : BaseKV(BaseKVConfig{}),
        index_(std::move(index)),
        value_store_(std::move(value_store)),
        num_threads_(std::max(num_threads, 1)),
        checkpoint_keys_by_thread_(static_cast<size_t>(num_threads_)) {}

  explicit KVEngineComposite(const BaseKVConfig& config) : BaseKV(config) {
    config_                      = config;
    const auto& j                = config.json_config_;
    const std::string index_type = j.at("index").at("type").get<std::string>();
    const std::string value_type = j.at("value").at("type").get<std::string>();
    using IF                     = base::Factory<Index, const BaseKVConfig&>;
    using VF = base::Factory<ValueStore, const BaseKVConfig&>;
    index_.reset(IF::NewInstance(index_type, config));
    value_store_.reset(VF::NewInstance(value_type, config));
    num_threads_ = config.num_threads_;
    checkpoint_keys_by_thread_.resize(
        static_cast<size_t>(std::max(num_threads_, 1)));
    default_value_size_hint_ =
        j.at("value").value("default_value_size_hint", 0);
    if (!index_ || !value_store_) {
      throw std::runtime_error("failed to create KVEngine components");
    }
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    (void)tid;
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    if (handle == kValueHandleNone) {
      value.clear();
      return;
    }
    if (const char* ptr = value_store_->DirectPtr(handle)) {
      value.resize(value_store_->SlotCapacity(handle));
      std::memcpy(value.data(), ptr, value.size());
      return;
    }
    value.resize(value_store_->SlotCapacity(handle));
    const size_t actual =
        value_store_->Read(handle, value.data(), value.size());
    value.resize(actual);
  }

  bool Exists(const uint64_t key, unsigned tid) override {
    (void)tid;
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    return handle != kValueHandleNone;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    PutInternal(key, value.data(), value.size(), tid, true);
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    if (values == nullptr || keys.Size() != static_cast<int>(values->size())) {
      LOG(FATAL) << "KVEngine::BatchPut size mismatch";
    }
    (void)tid;
    if (keys.Size() == 0) {
      return;
    }
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);

    std::unordered_set<uint64_t> seen_keys;
    seen_keys.reserve(static_cast<size_t>(keys.Size()));
    bool has_duplicate_key = false;
    for (int i = 0; i < keys.Size(); ++i) {
      if (!seen_keys.insert(keys[i]).second) {
        has_duplicate_key = true;
        break;
      }
    }
    if (has_duplicate_key) {
      for (int i = 0; i < keys.Size(); ++i) {
        const auto& item = (*values)[i];
        PutInternal(keys[i],
                    item.Data(),
                    static_cast<size_t>(item.Size()) * sizeof(float),
                    tid,
                    false);
      }
      return;
    }

    struct PutItem {
      uint64_t key = 0;
      ValueStore::WriteSpec spec{};
    };
    std::vector<PutItem> items;
    items.reserve(static_cast<size_t>(keys.Size()));

    for (int i = 0; i < keys.Size(); ++i) {
      const auto& item  = (*values)[i];
      const void* data  = item.Data();
      const size_t size = static_cast<size_t>(item.Size()) * sizeof(float);
      items.push_back(PutItem{keys[i], ValueStore::WriteSpec{data, size}});
    }

    std::vector<ValueStore::WriteSpec> specs;
    specs.reserve(items.size());
    for (const auto& item : items) {
      specs.push_back(item.spec);
    }
    const auto new_handles = value_store_->BatchAllocAndWrite(specs);
    if (new_handles.size() != items.size()) {
      LOG(FATAL) << "KVEngine::BatchPut allocation result size mismatch";
    }
    for (size_t i = 0; i < items.size(); ++i) {
      if (new_handles[i] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine batch value allocation failed, key="
                   << items[i].key << " size=" << items[i].spec.size;
      }
    }

    for (size_t i = 0; i < items.size(); ++i) {
      Value_t old_handle = index_->Put(items[i].key, new_handles[i], tid);
      if (old_handle != kValueHandleNone) {
        value_store_->Retire(old_handle);
      }
    }
    TrackKeys(keys, tid);
  }

  uint64_t BatchDelete(base::ConstArray<uint64_t> keys, unsigned tid) override {
    if (keys.Size() == 0) {
      return 0;
    }
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    return DeleteKeysLocked(keys, tid);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    values->resize(keys.Size());
    thread_local std::vector<Value_t> handles;
    thread_local std::vector<std::vector<float>> buffers;
    handles.assign(keys.Size(), kValueHandleNone);
    buffers.clear();
    buffers.resize(keys.Size());

    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    std::vector<uint64_t> batch_handles;
    std::vector<size_t> batch_indices;
    batch_handles.reserve(static_cast<size_t>(keys.Size()));
    batch_indices.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      if (const char* ptr = value_store_->DirectPtr(handles[i])) {
        const size_t bytes = value_store_->SlotCapacity(handles[i]);
        (*values)[i]       = base::ConstArray<float>(
            reinterpret_cast<float*>(const_cast<char*>(ptr)),
            bytes / sizeof(float));
        continue;
      }
      batch_handles.push_back(handles[i]);
      batch_indices.push_back(static_cast<size_t>(i));
    }

    std::vector<ValueStore::ReadResult> batch_results;
    if (!batch_handles.empty()) {
      value_store_->BatchRead(batch_handles, batch_results);
      if (batch_results.size() != batch_indices.size()) {
        LOG(FATAL) << "KVEngine::BatchGet read result size mismatch";
      }
      for (size_t i = 0; i < batch_indices.size(); ++i) {
        const size_t idx   = batch_indices[i];
        const auto& result = batch_results[i];
        buffers[idx].resize(result.data.size() / sizeof(float));
        if (!result.data.empty()) {
          std::memcpy(
              buffers[idx].data(), result.data.data(), result.data.size());
        }
        (*values)[idx] =
            base::ConstArray<float>(buffers[idx].data(), buffers[idx].size());
      }
    }
  }

  bool BatchGetFlat(base::ConstArray<uint64_t> keys,
                    float* values,
                    int64_t num_rows,
                    int64_t embedding_dim,
                    unsigned tid,
                    BatchGetFlatStats* stats = nullptr) override {
    if (values == nullptr || num_rows < 0 || embedding_dim <= 0 ||
        keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
    thread_local std::vector<Value_t> handles;
    thread_local std::vector<char> read_buffer;
    handles.assign(keys.Size(), kValueHandleNone);
    const auto index_lookup_start =
        stats != nullptr ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    if (stats != nullptr) {
      stats->index_lookup_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - index_lookup_start)
              .count());
    }

    std::uint64_t missing_zero_fill_ns = 0;
    std::uint64_t missing_rows         = 0;
    const auto row_copy_start =
        stats != nullptr ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    if (default_value_size_hint_ == row_bytes &&
        value_store_->ReadFlatFixedRows(
            handles.data(),
            static_cast<size_t>(num_rows),
            values,
            row_bytes,
            &missing_rows)) {
      if (stats != nullptr) {
        stats->zero_fill_ns = 0;
        stats->row_copy_ns  = static_cast<std::uint64_t>(
            std::chrono::duration_cast< std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - row_copy_start)
                .count());
        stats->missing_rows = missing_rows;
      }
      return true;
    }
    for (int64_t row = 0; row < num_rows; ++row) {
      const Value_t handle = handles[static_cast<size_t>(row)];
      float* dst           = values + row * embedding_dim;
      if (handle == kValueHandleNone) {
        const auto missing_zero_start =
            stats != nullptr ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
        std::memset(dst, 0, row_bytes);
        if (stats != nullptr) {
          missing_zero_fill_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast< std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - missing_zero_start)
                  .count());
        }
        ++missing_rows;
        continue;
      }
      if (const char* ptr = value_store_->DirectPtr(handle)) {
        if (default_value_size_hint_ != row_bytes) {
          const size_t slot_bytes = value_store_->SlotCapacity(handle);
          if (slot_bytes != row_bytes) {
            LOG(ERROR) << "KVEngine::BatchGetFlat row size mismatch row=" << row
                       << " key=" << keys[static_cast<int>(row)]
                       << " expected_bytes=" << row_bytes
                       << " actual_bytes=" << slot_bytes;
            return false;
          }
        }
        std::memcpy(dst, ptr, row_bytes);
      } else {
        read_buffer.resize(row_bytes + sizeof(float));
        const size_t actual =
            value_store_->Read(handle, read_buffer.data(), read_buffer.size());
        if (actual != row_bytes) {
          LOG(ERROR) << "KVEngine::BatchGetFlat read size mismatch row=" << row
                     << " key=" << keys[static_cast<int>(row)]
                     << " expected_bytes=" << row_bytes
                     << " actual_bytes=" << actual;
          return false;
        }
        std::memcpy(dst, read_buffer.data(), row_bytes);
      }
    }
    if (stats != nullptr) {
      stats->zero_fill_ns = missing_zero_fill_ns;
      stats->row_copy_ns  = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - row_copy_start)
              .count());
      stats->missing_rows = missing_rows;
    }
    return true;
  }

  bool BatchGetFlatRange(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t row_dim,
      int64_t value_offset,
      int64_t value_dim,
      unsigned tid,
      BatchGetFlatStats* stats = nullptr,
      bool collect_profile     = true) override {
    if (values == nullptr || num_rows < 0 || row_dim <= 0 || value_offset < 0 ||
        value_dim <= 0 || value_offset > row_dim ||
        value_dim > row_dim - value_offset ||
        keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }
    const size_t stored_row_bytes =
        static_cast<size_t>(row_dim) * sizeof(float);
    if (default_value_size_hint_ != stored_row_bytes) {
      return false;
    }
    const size_t source_offset_bytes =
        static_cast<size_t>(value_offset) * sizeof(float);
    const size_t output_row_bytes =
        static_cast<size_t>(value_dim) * sizeof(float);
    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    const bool profile_enabled = stats != nullptr && collect_profile;
    if (stats != nullptr) {
      stats->index_lookup_ns = 0;
      stats->zero_fill_ns    = 0;
      stats->row_copy_ns     = 0;
      stats->missing_rows    = 0;
    }
    const auto index_lookup_start =
        profile_enabled ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    if (profile_enabled) {
      stats->index_lookup_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - index_lookup_start)
              .count());
    }

    uint64_t missing_rows = 0;
    const auto row_copy_start =
        profile_enabled ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
    if (!value_store_->ReadFlatFixedRowSlices(
            handles.data(),
            static_cast<size_t>(num_rows),
            values,
            stored_row_bytes,
            source_offset_bytes,
            output_row_bytes,
            &missing_rows)) {
      return false;
    }
    if (stats != nullptr) {
      if (profile_enabled) {
        stats->row_copy_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - row_copy_start)
                .count());
      }
      stats->missing_rows = missing_rows;
    }
    return true;
  }

  bool BatchGetIndexedFlat(
      base::ConstArray<uint64_t> keys,
      int64_t* indices,
      float* values,
      bool* found,
      int64_t num_rows,
      int64_t value_dim,
      unsigned tid,
      BatchGetFlatStats* stats = nullptr,
      bool collect_profile     = true) override {
    if (stats != nullptr)
      *stats = BatchGetFlatStats{};
    if (num_rows < 0 || value_dim <= 0 ||
        keys.Size() != static_cast<size_t>(num_rows) ||
        static_cast<uint64_t>(value_dim) >
            ((std::numeric_limits<size_t>::max)() - sizeof(int64_t)) /
                sizeof(float) ||
        (num_rows != 0 && (keys.Data() == nullptr || indices == nullptr ||
                           values == nullptr || found == nullptr))) {
      return false;
    }
    const size_t value_bytes = static_cast<size_t>(value_dim) * sizeof(float);
    if (default_value_size_hint_ != sizeof(int64_t) + value_bytes ||
        static_cast<uint64_t>(num_rows) >
            (std::numeric_limits<size_t>::max)() / value_bytes) {
      return false;
    }
    if (num_rows == 0)
      return true;

    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    const bool profile = stats != nullptr && collect_profile;
    using Clock        = std::chrono::steady_clock;
    const auto start   = profile ? Clock::now() : Clock::time_point{};
    index_->BatchGet(keys, handles.data(), tid);
    const auto after_index = profile ? Clock::now() : Clock::time_point{};
    uint64_t missing       = 0;
    if (!value_store_->ReadIndexedFixedRows(
            handles.data(),
            static_cast<size_t>(num_rows),
            indices,
            values,
            found,
            value_bytes,
            &missing)) {
      return false;
    }
    if (stats != nullptr) {
      stats->missing_rows = missing;
      if (profile) {
        stats->index_lookup_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                after_index - start)
                .count());
        stats->row_copy_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - after_index)
                .count());
      }
    }
    return true;
  }

  bool BatchGetIndexOnly(base::ConstArray<uint64_t> keys,
                         unsigned tid,
                         BatchGetFlatStats* stats = nullptr) override {
    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    if (stats != nullptr) {
      std::uint64_t missing_rows = 0;
      for (const Value_t handle : handles) {
        if (handle == kValueHandleNone) {
          ++missing_rows;
        }
      }
      stats->missing_rows = missing_rows;
    }
    return true;
  }

  bool BatchGetDirectFixedRows(
      base::ConstArray<uint64_t> keys,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid,
      std::vector<DirectFixedRow>* rows,
      BatchGetFlatStats* stats = nullptr) override {
    if (rows == nullptr || num_rows < 0 || embedding_dim <= 0 ||
        keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
    if (default_value_size_hint_ != row_bytes) {
      return false;
    }
    thread_local std::vector<Value_t> handles;
    handles.assign(keys.Size(), kValueHandleNone);
    const auto index_lookup_start =
        stats != nullptr ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    if (keys.Size() > 0) {
      index_->BatchGet(keys, handles.data(), tid);
    }
    if (stats != nullptr) {
      stats->index_lookup_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast< std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - index_lookup_start)
              .count());
    }

    thread_local std::vector<ValueStore::DirectFixedRow> store_rows;
    store_rows.resize(static_cast<size_t>(num_rows));
    uint64_t missing_rows = 0;
    if (!value_store_->GetDirectFixedRows(
            handles.data(),
            static_cast<size_t>(num_rows),
            row_bytes,
            store_rows.data(),
            &missing_rows)) {
      return false;
    }
    rows->resize(store_rows.size());
    for (size_t i = 0; i < store_rows.size(); ++i) {
      (*rows)[i] = DirectFixedRow{
          store_rows[i].data, store_rows[i].size, store_rows[i].missing};
    }
    if (stats != nullptr) {
      stats->missing_rows = missing_rows;
    }
    return true;
  }

  RDMABackingRegion GetRDMABackingRegion() const override {
    if (!value_store_) {
      return {};
    }
    return RDMABackingRegion{
        value_store_->RDMABackingData(), value_store_->RDMABackingSize()};
  }

  bool ApplySgdUpdateFlat(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      uint8_t tag,
      unsigned tid) override {
    if (grads == nullptr || keys.Size() != static_cast<size_t>(num_rows) ||
        embedding_dim <= 0) {
      return false;
    }
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    const int tag_bits      = static_cast<int>(sizeof(tag) * 8);
    const int shift         = static_cast<int>(sizeof(uint64_t) * 8) - tag_bits;
    const uint64_t key_mask = ~0ULL >> tag_bits;
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);

    thread_local std::vector<uint64_t> tagged_keys;
    tagged_keys.resize(static_cast<size_t>(num_rows));
    for (int64_t r = 0; r < num_rows; ++r) {
      tagged_keys[static_cast<size_t>(r)] =
          (static_cast<uint64_t>(tag) << shift) |
          (keys[static_cast<size_t>(r)] & key_mask);
    }

    thread_local std::vector<DirectFixedRow> rows;
    if (!BatchGetDirectFixedRows(
            base::ConstArray<uint64_t>(tagged_keys),
            num_rows,
            embedding_dim,
            tid,
            &rows)) {
      return false;
    }

    std::vector<float> missing_row(static_cast<size_t>(embedding_dim));
    for (int64_t r = 0; r < num_rows; ++r) {
      const auto& row   = rows[static_cast<size_t>(r)];
      const float* grad = grads + r * embedding_dim;
      if (row.missing) {
        for (int64_t c = 0; c < embedding_dim; ++c) {
          missing_row[static_cast<size_t>(c)] = -learning_rate * grad[c];
        }
        PutInternal(tagged_keys[static_cast<size_t>(r)],
                    missing_row.data(),
                    row_bytes,
                    tid,
                    false);
        continue;
      }

      float* value = reinterpret_cast<float*>(const_cast<char*>(row.data));
#pragma omp simd
      for (int64_t c = 0; c < embedding_dim; ++c) {
        value[c] -= learning_rate * grad[c];
      }
    }
    return true;
  }

  bool ApplySgdUpdateFlatRange(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t row_dim,
      int64_t update_offset,
      int64_t update_dim,
      float learning_rate,
      unsigned tid) override {
    if (grads == nullptr || keys.Size() != static_cast<size_t>(num_rows) ||
        row_dim <= 0 || update_offset < 0 || update_dim <= 0 ||
        update_offset > row_dim || update_dim > row_dim - update_offset) {
      return false;
    }
    const size_t row_bytes = static_cast<size_t>(row_dim) * sizeof(float);
    if (default_value_size_hint_ != row_bytes) {
      return false;
    }

    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    thread_local std::vector<DirectFixedRow> rows;
    if (!BatchGetDirectFixedRows(keys, num_rows, row_dim, tid, &rows)) {
      return false;
    }
    for (int64_t row = 0; row < num_rows; ++row) {
      const auto& direct_row = rows[static_cast<size_t>(row)];
      if (direct_row.missing || direct_row.data == nullptr ||
          direct_row.size != row_bytes) {
        return false;
      }
      float* value =
          reinterpret_cast<float*>(const_cast<char*>(direct_row.data)) +
          update_offset;
      const float* grad = grads + row * update_dim;
#pragma omp simd
      for (int64_t column = 0; column < update_dim; ++column) {
        value[column] -= learning_rate * grad[column];
      }
    }
    return true;
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    BulkLoadRange(keys, value, 0);
  }

  void BulkLoadRange(base::ConstArray<uint64_t> keys,
                     const void* value,
                     unsigned tid,
                     BulkLoadStats* stats = nullptr) override {
    using clock             = std::chrono::steady_clock;
    const auto total_start  = clock::now();
    const auto& j           = config_.json_config_;
    const size_t value_size = j.at("value").value("default_value_size_hint", 0);
    if (value_size == 0) {
      LOG(FATAL) << "KVEngine::BulkLoad requires value_size hint";
    }
    if (keys.Size() == 0) {
      return;
    }
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    const char* data = reinterpret_cast<const char*>(value);
    std::vector<ValueStore::WriteSpec> specs;
    specs.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      specs.push_back(ValueStore::WriteSpec{data + i * value_size, value_size});
    }
    const auto prepare_end        = clock::now();
    std::vector<uint64_t> handles = value_store_->BatchAllocAndWrite(specs);
    const auto alloc_end          = clock::now();
    if (handles.size() != static_cast<size_t>(keys.Size())) {
      LOG(FATAL) << "KVEngine::BulkLoad allocation result size mismatch";
    }
    for (int i = 0; i < keys.Size(); ++i) {
      if (handles[static_cast<size_t>(i)] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine bulk value allocation failed, key=" << keys[i]
                   << " size=" << value_size;
      }
    }
    const auto validate_end = clock::now();
    index_->BatchPut(keys, handles.data(), tid);
    const auto index_end = clock::now();
    TrackKeys(keys, tid);
    const auto track_end = clock::now();
    if (stats != nullptr) {
      auto ns = [](clock::duration duration) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                .count());
      };
      stats->prepare_ns         = ns(prepare_end - total_start);
      stats->value_alloc_ns     = ns(alloc_end - prepare_end);
      stats->handle_validate_ns = ns(validate_end - alloc_end);
      stats->index_put_ns       = ns(index_end - validate_end);
      stats->key_track_ns       = ns(track_end - index_end);
      stats->total_ns           = ns(track_end - total_start);
    }
  }

  bool BulkLoadIndexedFloatRange(
      base::ConstArray<uint64_t> keys,
      const int64_t* row_indices,
      const float* values,
      int64_t num_rows,
      int64_t value_dim,
      unsigned tid,
      BulkLoadStats* stats = nullptr) override {
    using clock            = std::chrono::steady_clock;
    const auto total_start = clock::now();
    if (row_indices == nullptr || values == nullptr || num_rows < 0 ||
        value_dim <= 0 || keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }
    const size_t row_bytes =
        sizeof(int64_t) + static_cast<size_t>(value_dim) * sizeof(float);
    if (default_value_size_hint_ != row_bytes) {
      return false;
    }

    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    std::vector<uint64_t> handles(
        static_cast<size_t>(num_rows), kValueHandleNone);
    const auto prepare_end = clock::now();
    for (int64_t row_index = 0; row_index < num_rows; ++row_index) {
      const uint64_t handle = value_store_->Alloc(row_bytes);
      if (handle == kValueHandleNone) {
        LOG(FATAL) << "KVEngine indexed bulk value allocation failed, key="
                   << keys[static_cast<int>(row_index)]
                   << " size=" << row_bytes;
      }
      char* row = const_cast<char*>(value_store_->DirectPtr(handle));
      if (row == nullptr) {
        return false;
      }
      std::memcpy(row, row_indices + row_index, sizeof(int64_t));
      std::memcpy(row + sizeof(int64_t),
                  values + row_index * value_dim,
                  static_cast<size_t>(value_dim) * sizeof(float));
      handles[static_cast<size_t>(row_index)] = handle;
    }
    const auto alloc_end    = clock::now();
    const auto validate_end = alloc_end;
    index_->BatchPut(keys, handles.data(), tid);
    const auto index_end = clock::now();
    TrackKeys(keys, tid);
    const auto track_end = clock::now();
    if (stats != nullptr) {
      auto ns = [](clock::duration duration) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                .count());
      };
      stats->prepare_ns         = ns(prepare_end - total_start);
      stats->value_alloc_ns     = ns(alloc_end - prepare_end);
      stats->handle_validate_ns = ns(validate_end - alloc_end);
      stats->index_put_ns       = ns(index_end - validate_end);
      stats->key_track_ns       = ns(track_end - index_end);
      stats->total_ns           = ns(track_end - total_start);
    }
    return true;
  }

  bool SaveCheckpoint(const std::string& file,
                      const std::string& metadata) override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    if (file.empty()) {
      LOG(ERROR) << "KVEngine checkpoint path is empty";
      return false;
    }

    std::vector<uint64_t> keys = CollectCheckpointKeys();

    const std::filesystem::path checkpoint_path(file);
    std::filesystem::path temp_path(file);
    temp_path += ".tmp";
    std::error_code error;
    std::filesystem::remove(temp_path, error);

    std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      LOG(ERROR) << "Failed to open checkpoint temp file: " << temp_path;
      return false;
    }

    const uint64_t metadata_size = static_cast<uint64_t>(metadata.size());
    const uint64_t record_count  = static_cast<uint64_t>(keys.size());
    uint64_t checksum            = kCheckpointChecksumSeed;
    UpdateChecksum(&checksum, kCheckpointMagic.data(), kCheckpointMagic.size());
    UpdateChecksumPod(&checksum, kCheckpointVersion);
    UpdateChecksumPod(&checksum, metadata_size);
    UpdateChecksumPod(&checksum, record_count);
    UpdateChecksum(&checksum, metadata.data(), metadata.size());
    bool write_ok =
        WriteBytes(output, kCheckpointMagic.data(), kCheckpointMagic.size()) &&
        WritePod(output, kCheckpointVersion) &&
        WritePod(output, metadata_size) && WritePod(output, record_count) &&
        WriteBytes(output, metadata.data(), metadata.size());
    for (const uint64_t key : keys) {
      std::string value;
      Get(key, value, 0);
      if (!Exists(key, 0)) {
        LOG(ERROR) << "KVEngine checkpoint tracked key is missing: " << key;
        write_ok = false;
        break;
      }
      const uint64_t value_size = static_cast<uint64_t>(value.size());
      UpdateChecksumPod(&checksum, key);
      UpdateChecksumPod(&checksum, value_size);
      UpdateChecksum(&checksum, value.data(), value.size());
      write_ok =
          write_ok && WritePod(output, key) && WritePod(output, value_size) &&
          WriteBytes(output, value.data(), value.size());
      if (!write_ok) {
        break;
      }
    }
    write_ok = write_ok && WritePod(output, checksum);
    output.flush();
    write_ok = write_ok && output.good();
    output.close();
    write_ok = write_ok && !output.fail();
    if (!write_ok) {
      LOG(ERROR) << "Failed to write checkpoint temp file: " << temp_path;
      std::filesystem::remove(temp_path, error);
      return false;
    }

    error.clear();
    std::filesystem::rename(temp_path, checkpoint_path, error);
    if (error) {
      LOG(ERROR) << "Failed to publish checkpoint " << checkpoint_path << ": "
                 << error.message();
      std::filesystem::remove(temp_path, error);
      return false;
    }
    return true;
  }

  bool LoadCheckpoint(const std::string& file,
                      const std::string& expected_metadata) override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    if (HasCheckpointRecords()) {
      LOG(ERROR) << "KVEngine checkpoint load requires an empty engine";
      return false;
    }
    if (file.empty()) {
      LOG(ERROR) << "KVEngine checkpoint path is empty";
      return false;
    }

    std::error_code error;
    const std::uintmax_t file_size = std::filesystem::file_size(file, error);
    if (error || file_size > std::numeric_limits<uint64_t>::max()) {
      LOG(ERROR) << "Failed to inspect checkpoint file: " << file;
      return false;
    }
    uint64_t remaining = static_cast<uint64_t>(file_size);
    std::ifstream input(file, std::ios::binary);
    if (!input) {
      LOG(ERROR) << "Failed to open checkpoint file: " << file;
      return false;
    }

    std::array<char, kCheckpointMagic.size()> magic{};
    uint32_t version       = 0;
    uint64_t metadata_size = 0;
    uint64_t record_count  = 0;
    if (!ReadBytes(input, magic.data(), magic.size(), &remaining) ||
        !ReadPod(input, &version, &remaining) ||
        !ReadPod(input, &metadata_size, &remaining) ||
        !ReadPod(input, &record_count, &remaining) ||
        magic != kCheckpointMagic || version != kCheckpointVersion ||
        metadata_size > remaining ||
        metadata_size > std::numeric_limits<size_t>::max()) {
      LOG(ERROR) << "Invalid or truncated checkpoint header: " << file;
      return false;
    }

    std::string metadata(static_cast<size_t>(metadata_size), '\0');
    if (!ReadBytes(input, metadata.data(), metadata_size, &remaining)) {
      LOG(ERROR) << "Truncated checkpoint metadata: " << file;
      return false;
    }
    if (metadata != expected_metadata) {
      LOG(ERROR) << "Checkpoint metadata mismatch: " << file;
      return false;
    }
    if (remaining < sizeof(uint64_t)) {
      LOG(ERROR) << "Checkpoint checksum is missing: " << file;
      return false;
    }
    if (record_count > std::numeric_limits<size_t>::max() ||
        record_count >
            (remaining - sizeof(uint64_t)) / (2 * sizeof(uint64_t))) {
      LOG(ERROR) << "Invalid checkpoint record count: " << file;
      return false;
    }

    uint64_t checksum = kCheckpointChecksumSeed;
    UpdateChecksum(&checksum, magic.data(), magic.size());
    UpdateChecksumPod(&checksum, version);
    UpdateChecksumPod(&checksum, metadata_size);
    UpdateChecksumPod(&checksum, record_count);
    UpdateChecksum(&checksum, metadata.data(), metadata.size());
    std::vector<std::pair<uint64_t, std::string>> records;
    records.reserve(static_cast<size_t>(record_count));
    std::unordered_set<uint64_t> seen_keys;
    seen_keys.reserve(static_cast<size_t>(record_count));
    for (uint64_t i = 0; i < record_count; ++i) {
      uint64_t key        = 0;
      uint64_t value_size = 0;
      if (!ReadPod(input, &key, &remaining) ||
          !ReadPod(input, &value_size, &remaining) || value_size > remaining ||
          value_size > std::numeric_limits<size_t>::max() ||
          !seen_keys.insert(key).second) {
        LOG(ERROR) << "Invalid or truncated checkpoint record: " << file;
        return false;
      }
      std::string value(static_cast<size_t>(value_size), '\0');
      if (!ReadBytes(input, value.data(), value_size, &remaining)) {
        LOG(ERROR) << "Truncated checkpoint value: " << file;
        return false;
      }
      UpdateChecksumPod(&checksum, key);
      UpdateChecksumPod(&checksum, value_size);
      UpdateChecksum(&checksum, value.data(), value.size());
      records.emplace_back(key, std::move(value));
    }
    uint64_t saved_checksum = 0;
    if (!ReadPod(input, &saved_checksum, &remaining) || remaining != 0) {
      LOG(ERROR) << "Checkpoint contains trailing data: " << file;
      return false;
    }
    if (saved_checksum != checksum) {
      LOG(ERROR) << "Checkpoint checksum mismatch: " << file;
      return false;
    }

    for (const auto& record : records) {
      PutInternal(
          record.first, record.second.data(), record.second.size(), 0, false);
    }
    return true;
  }

  uint64_t CheckpointRecordCount() const override { return ActiveKeyCount(); }

  // CollectCheckpointKeys traverses checkpoint_keys_by_thread_ while the put
  // path (TrackKeys) push_backs into it under a *shared* lock, so collectors
  // must take the exclusive lock; a shared lock here races the writers.
  uint64_t ActiveKeyCount() const override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    return static_cast<uint64_t>(CollectCheckpointKeys().size());
  }

  std::vector<uint64_t> SnapshotKeys() const override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    return CollectCheckpointKeys();
  }

  void clear() override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    const std::vector<uint64_t> keys = CollectCheckpointKeys();
    if (!keys.empty()) {
      DeleteKeysLocked(base::ConstArray<uint64_t>(keys), 0);
    }
  }

  void Util() override {
    LOG(INFO) << "KVEngine index utilization=" << index_->Utilization()
              << " value=" << value_store_->GetInfo();
  }

  void DebugInfo() const override {
    index_->DebugInfo();
    LOG(INFO) << value_store_->GetInfo();
  }

  std::string ExtraResultFields() const override {
    return value_store_ ? value_store_->ExtraResultFields() : "";
  }

private:
  static bool WriteBytes(std::ofstream& output, const void* data, size_t size) {
    if (size >
        static_cast<size_t>((std::numeric_limits<std::streamsize>::max)())) {
      return false;
    }
    if (size != 0) {
      output.write(static_cast<const char*>(data),
                   static_cast<std::streamsize>(size));
    }
    return output.good();
  }

  template <typename T>
  static bool WritePod(std::ofstream& output, const T& value) {
    return WriteBytes(output, &value, sizeof(value));
  }

  static bool ReadBytes(
      std::ifstream& input, void* data, uint64_t size, uint64_t* remaining) {
    if (remaining == nullptr || size > *remaining ||
        size > static_cast<uint64_t>(
                   (std::numeric_limits<std::streamsize>::max)())) {
      return false;
    }
    if (size != 0) {
      input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
      if (!input) {
        return false;
      }
    }
    *remaining -= size;
    return true;
  }

  template <typename T>
  static bool ReadPod(std::ifstream& input, T* value, uint64_t* remaining) {
    return ReadBytes(input, value, sizeof(*value), remaining);
  }

  static void
  UpdateChecksum(uint64_t* checksum, const void* data, size_t size) {
    if (size != 0) {
      *checksum = xxhash(data, size, *checksum);
    }
  }

  template <typename T>
  static void UpdateChecksumPod(uint64_t* checksum, const T& value) {
    UpdateChecksum(checksum, &value, sizeof(value));
  }

  std::vector<uint64_t> CollectCheckpointKeys() const {
    size_t count = 0;
    for (const auto& keys : checkpoint_keys_by_thread_) {
      count += keys.size();
    }
    std::vector<uint64_t> keys;
    keys.reserve(count);
    for (const auto& thread_keys : checkpoint_keys_by_thread_) {
      keys.insert(keys.end(), thread_keys.begin(), thread_keys.end());
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
  }

  bool HasCheckpointRecords() const {
    return std::any_of(checkpoint_keys_by_thread_.begin(),
                       checkpoint_keys_by_thread_.end(),
                       [](const auto& keys) { return !keys.empty(); });
  }

  std::vector<uint64_t>& CheckpointKeysForThread(unsigned tid) {
    if (tid >= checkpoint_keys_by_thread_.size()) {
      LOG(FATAL) << "KVEngine checkpoint tid out of range: " << tid
                 << " >= " << checkpoint_keys_by_thread_.size();
    }
    return checkpoint_keys_by_thread_[tid];
  }

  void TrackKey(uint64_t key, unsigned tid) {
    CheckpointKeysForThread(tid).push_back(key);
  }

  void TrackKeys(base::ConstArray<uint64_t> keys, unsigned tid) {
    auto& thread_keys = CheckpointKeysForThread(tid);
    thread_keys.reserve(thread_keys.size() + static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      thread_keys.push_back(keys[i]);
    }
  }

  uint64_t DeleteKeysLocked(base::ConstArray<uint64_t> keys, unsigned tid) {
    (void)tid;
    std::unordered_set<uint64_t> deleted_keys;
    deleted_keys.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      Value_t handle = kValueHandleNone;
      index_->Get(keys[i], handle);
      if (handle == kValueHandleNone) {
        continue;
      }
      uint64_t key = keys[i];
      if (!index_->Delete(key)) {
        continue;
      }
      value_store_->Retire(handle);
      deleted_keys.insert(key);
    }
    if (!deleted_keys.empty()) {
      for (auto& tracked_keys : checkpoint_keys_by_thread_) {
        tracked_keys.erase(
            std::remove_if(
                tracked_keys.begin(),
                tracked_keys.end(),
                [&](uint64_t key) {
                  return deleted_keys.find(key) != deleted_keys.end();
                }),
            tracked_keys.end());
      }
    }
    return static_cast<uint64_t>(deleted_keys.size());
  }

  void PutInternal(uint64_t key,
                   const void* data,
                   size_t size,
                   unsigned tid,
                   bool emit_fence) {
    (void)tid;
    (void)emit_fence;
    Value_t new_handle = value_store_->AllocAndWrite(data, size);
    if (new_handle == kValueHandleNone) {
      LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                 << " size=" << size;
      return;
    }
    Value_t old_handle = index_->Put(key, new_handle, tid);
    if (old_handle != kValueHandleNone) {
      value_store_->Retire(old_handle);
    }
    TrackKey(key, tid);
  }

  inline static constexpr std::array<char, 8> kCheckpointMagic = {
      'R', 'S', 'K', 'V', 'C', 'P', '0', '1'};
  inline static constexpr uint32_t kCheckpointVersion = 2;
  inline static constexpr uint64_t kCheckpointChecksumSeed =
      0x9e3779b97f4a7c15ULL;

  BaseKVConfig config_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<ValueStore> value_store_;
  int num_threads_                = 0;
  size_t default_value_size_hint_ = 0;
  mutable std::shared_mutex checkpoint_mu_;
  std::vector<std::vector<uint64_t>> checkpoint_keys_by_thread_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
