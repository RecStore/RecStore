#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <folly/SharedMutex.h>

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
  struct OnlineSnapshot {
    uint64_t key;
    uint64_t handle;
    std::vector<char> data;
  };
  struct OnlineJob {
    uint64_t generation;
    std::vector<OnlineSnapshot> hot;
  };

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
    auto* hybrid = dynamic_cast<HybridValueStore*>(value_store_.get());
    const bool async_admission =
        j.at("value").contains("tiering") &&
        j.at("value").at("tiering").value("async_admission", false);
    if (async_admission &&
        (hybrid == nullptr || !hybrid->OnlinePlacementEnabled())) {
      throw std::invalid_argument("async admission requires online admission");
    }
    if (hybrid != nullptr && hybrid->OnlinePlacementEnabled()) {
      if (default_value_size_hint_ == 0 ||
          default_value_size_hint_ % sizeof(float) != 0) {
        throw std::invalid_argument(
            "online admission requires fixed-size float rows");
      }
      online_store_       = hybrid;
      online_residents_   = hybrid->DramProfileKeys();
      online_ghost_limit_ = std::max<size_t>(online_residents_.size() / 20, 1);
      online_async_       = async_admission;
      if (online_async_)
        background_thread_ = std::thread([this] { BackgroundLoop(); });
    }
  }

  ~KVEngineComposite() override {
    if (background_thread_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(background_mu_);
        background_stop_ = true;
      }
      background_cv_.notify_all();
      background_thread_.join();
      if (background_error_) {
        LOG(ERROR) << "online admission stopped after a background failure";
      }
    }
  }

  void DrainBackgroundWork() override {
    if (!online_async_)
      return;
    std::unique_lock<std::mutex> lock(background_mu_);
    background_cv_.wait(lock, [this] {
      return !background_active_ && !background_pending_;
    });
    if (background_error_)
      std::rethrow_exception(background_error_);
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    bool cold = false;
    std::vector<OnlineSnapshot> snapshots;
    uint64_t generation = 0;
    {
      auto online_lock = LockOnlineRead();
      Value_t handle   = kValueHandleNone;
      ReadValue(key, value, &cold, &handle);
      CaptureSnapshot(snapshots, key, handle, value.data(), value.size(), 1);
      generation = online_generation_;
    }
    if (online_store_ != nullptr && cold) {
      MaybeAdmit({key}, tid, 1, std::move(snapshots), generation);
    }
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
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
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
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);

    std::unordered_set<uint64_t> seen_keys;
    seen_keys.reserve(static_cast<size_t>(keys.Size()));
    bool has_duplicate_key = false;
    for (int i = 0; i < keys.Size(); ++i) {
      const auto& item = (*values)[i];
      ValidateOnlineRowSize(static_cast<size_t>(item.Size()) * sizeof(float));
      if (!seen_keys.insert(keys[i]).second) {
        has_duplicate_key = true;
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

    std::vector<uint64_t> batch_keys;
    std::vector<ValueStore::WriteSpec> specs;
    batch_keys.reserve(items.size());
    specs.reserve(items.size());
    for (const auto& item : items) {
      batch_keys.push_back(item.key);
      specs.push_back(item.spec);
    }
    auto new_handles = value_store_->BatchAllocAndWrite(batch_keys, specs);
    if (new_handles.size() != items.size()) {
      LOG(FATAL) << "KVEngine::BatchPut allocation result size mismatch";
    }
    for (size_t i = 0; i < items.size(); ++i) {
      if (new_handles[i] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine batch value allocation failed, key="
                   << items[i].key << " size=" << items[i].spec.size;
      }
    }

    std::vector<Value_t> old_handles(items.size(), kValueHandleNone);
    index_->BatchPutWithOld(
        base::ConstArray<uint64_t>(batch_keys),
        new_handles.data(),
        old_handles.data(),
        tid);
    for (size_t i = 0; i < items.size(); ++i) {
      const Value_t old_handle = old_handles[i];
      if (old_handle == kValueHandleNone) {
        TrackKey(items[i].key, tid);
      }
      if (old_handle != kValueHandleNone && old_handle != new_handles[i]) {
        value_store_->Retire(old_handle);
      }
    }
  }

  uint64_t BatchDelete(base::ConstArray<uint64_t> keys, unsigned tid) override {
    if (keys.Size() == 0) {
      return 0;
    }
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    return DeleteKeysLocked(keys, tid);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    auto online_lock = LockOnlineRead();
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
        const size_t bytes =
            online_store_ != nullptr
                ? default_value_size_hint_
                : value_store_->SlotCapacity(handles[i]);
        if (online_store_ != nullptr) {
          buffers[i].resize(bytes / sizeof(float));
          std::memcpy(buffers[i].data(), ptr, bytes);
          (*values)[i] = base::ConstArray<float>(buffers[i]);
        } else {
          (*values)[i] = base::ConstArray<float>(
              reinterpret_cast<float*>(const_cast<char*>(ptr)),
              bytes / sizeof(float));
        }
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
    if (online_store_ != nullptr && !batch_indices.empty()) {
      std::vector<uint64_t> cold_keys;
      std::vector<OnlineSnapshot> snapshots;
      for (size_t row : batch_indices) {
        cold_keys.push_back(keys[row]);
        CaptureSnapshot(
            snapshots,
            keys[row],
            handles[row],
            (*values)[row].Data(),
            (*values)[row].Size() * sizeof(float),
            keys.Size());
      }
      const uint64_t generation = online_generation_;
      online_lock.unlock();
      MaybeAdmit(cold_keys, tid, keys.Size(), std::move(snapshots), generation);
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
    auto online_lock       = LockOnlineRead();
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
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
    std::vector<uint64_t> batch_handles;
    std::vector<size_t> batch_rows;
    batch_handles.reserve(static_cast<size_t>(num_rows));
    batch_rows.reserve(static_cast<size_t>(num_rows));
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
        batch_handles.push_back(handle);
        batch_rows.push_back(static_cast<size_t>(row));
      }
    }
    if (!batch_handles.empty()) {
      std::vector<ValueStore::ReadResult> batch_results;
      value_store_->BatchRead(batch_handles, batch_results);
      if (batch_results.size() != batch_rows.size()) {
        LOG(ERROR) << "KVEngine::BatchGetFlat result size mismatch";
        return false;
      }
      for (size_t i = 0; i < batch_rows.size(); ++i) {
        const size_t row   = batch_rows[i];
        const auto& result = batch_results[i];
        if (result.data.size() != row_bytes) {
          LOG(ERROR) << "KVEngine::BatchGetFlat read size mismatch row=" << row
                     << " key=" << keys[static_cast<int>(row)]
                     << " expected_bytes=" << row_bytes
                     << " actual_bytes=" << result.data.size();
          return false;
        }
        std::memcpy(
            values + row * embedding_dim, result.data.data(), row_bytes);
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
    if (online_store_ != nullptr && !batch_rows.empty()) {
      std::vector<uint64_t> cold_keys;
      std::vector<OnlineSnapshot> snapshots;
      for (size_t row : batch_rows) {
        cold_keys.push_back(keys[row]);
        CaptureSnapshot(
            snapshots,
            keys[row],
            handles[row],
            values + row * embedding_dim,
            row_bytes,
            keys.Size());
      }
      const uint64_t generation = online_generation_;
      online_lock.unlock();
      MaybeAdmit(cold_keys, tid, keys.Size(), std::move(snapshots), generation);
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
    return BatchGetFlatRangeImpl(
        keys,
        values,
        num_rows,
        row_dim,
        value_offset,
        value_dim,
        tid,
        nullptr,
        stats,
        collect_profile);
  }

  bool BatchGetFlatRangeWithMissing(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t row_dim,
      int64_t value_offset,
      int64_t value_dim,
      unsigned tid,
      MissingPositionsOutput* missing,
      BatchGetFlatStats* stats = nullptr,
      bool collect_profile     = true) override {
    return BatchGetFlatRangeImpl(
        keys,
        values,
        num_rows,
        row_dim,
        value_offset,
        value_dim,
        tid,
        missing,
        stats,
        collect_profile);
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
    if (online_store_ != nullptr) {
      return false;
    }
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
      unsigned tid,
      UpdateFlatStats* stats = nullptr) override {
    if (grads == nullptr || keys.Size() != static_cast<size_t>(num_rows) ||
        embedding_dim <= 0) {
      return false;
    }
    using clock      = std::chrono::steady_clock;
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    const int tag_bits      = static_cast<int>(sizeof(tag) * 8);
    const int shift         = static_cast<int>(sizeof(uint64_t) * 8) - tag_bits;
    const uint64_t key_mask = ~0ULL >> tag_bits;
    const size_t row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);

    const auto tag_start = clock::now();
    thread_local std::vector<uint64_t> tagged_keys;
    tagged_keys.resize(static_cast<size_t>(num_rows));
    for (int64_t r = 0; r < num_rows; ++r) {
      tagged_keys[static_cast<size_t>(r)] =
          (static_cast<uint64_t>(tag) << shift) |
          (keys[static_cast<size_t>(r)] & key_mask);
    }
    if (stats != nullptr) {
      stats->tag_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              clock::now() - tag_start)
              .count());
    }

    thread_local std::vector<DirectFixedRow> rows;
    base::ConstArray<uint64_t> tagged(tagged_keys);
    BatchGetFlatStats get_stats;
    const auto batch_get_start = clock::now();
    if (!BatchGetDirectFixedRows(
            tagged,
            num_rows,
            embedding_dim,
            tid,
            &rows,
            stats != nullptr ? &get_stats : nullptr)) {
      std::vector<uint64_t> cold_keys;
      std::vector<OnlineSnapshot> snapshots;
      ApplySgdMixedRows(
          tagged_keys,
          grads,
          embedding_dim,
          learning_rate,
          tid,
          &cold_keys,
          &snapshots,
          stats);
      const uint64_t generation = online_generation_;
      checkpoint_lock.unlock();
      if (online_lock.owns_lock())
        online_lock.unlock();
      if (online_store_ != nullptr && !cold_keys.empty()) {
        MaybeAdmit(cold_keys,
                   tid,
                   tagged_keys.size(),
                   std::move(snapshots),
                   generation);
      }
      return true;
    }
    if (stats != nullptr) {
      stats->batch_get_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              clock::now() - batch_get_start)
              .count());
      stats->index_lookup_ns = get_stats.index_lookup_ns;
      stats->store_read_ns =
          stats->batch_get_ns > get_stats.index_lookup_ns
              ? stats->batch_get_ns - get_stats.index_lookup_ns
              : 0;
      stats->missing_rows = get_stats.missing_rows;
    }

    const auto loop_start = clock::now();
    std::vector<float> missing_row(static_cast<size_t>(embedding_dim));
    for (int64_t r = 0; r < num_rows; ++r) {
      const auto& row   = rows[static_cast<size_t>(r)];
      const float* grad = grads + r * embedding_dim;
      if (row.missing) {
#pragma omp simd
        for (int64_t c = 0; c < embedding_dim; ++c)
          missing_row[static_cast<size_t>(c)] = -learning_rate * grad[c];
        const auto miss_start =
            stats != nullptr ? clock::now() : clock::time_point{};
        PutInternalStats miss_stats;
        PutInternal(tagged_keys[static_cast<size_t>(r)],
                    missing_row.data(),
                    row_bytes,
                    tid,
                    false,
                    stats != nullptr ? &miss_stats : nullptr);
        if (stats != nullptr) {
          stats->missing_write_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  clock::now() - miss_start)
                  .count());
          stats->missing_alloc_ns += miss_stats.alloc_ns;
          stats->missing_index_ns += miss_stats.index_ns;
          stats->missing_track_ns += miss_stats.track_ns;
        }
        continue;
      }

      float* value = reinterpret_cast<float*>(const_cast<char*>(row.data));
#pragma omp simd
      for (int64_t c = 0; c < embedding_dim; ++c)
        value[c] -= learning_rate * grad[c];
    }
    if (stats != nullptr) {
      const auto loop_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              clock::now() - loop_start)
              .count());
      // The miss path is timed separately; subtract it so the hit write-back
      // bucket does not double count the inserts.
      stats->writeback_ns =
          loop_ns > stats->missing_write_ns
              ? loop_ns - stats->missing_write_ns
              : 0;
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
    if (online_store_ != nullptr) {
      return false;
    }
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
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    const char* data = reinterpret_cast<const char*>(value);
    std::vector<ValueStore::WriteSpec> specs;
    specs.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      specs.push_back(ValueStore::WriteSpec{data + i * value_size, value_size});
    }
    std::vector<uint64_t> bulk_keys;
    bulk_keys.reserve(static_cast<size_t>(keys.Size()));
    for (int i = 0; i < keys.Size(); ++i) {
      bulk_keys.push_back(keys[i]);
    }
    const auto prepare_end = clock::now();
    std::vector<uint64_t> handles =
        value_store_->BatchAllocAndWrite(bulk_keys, specs);
    const auto alloc_end = clock::now();
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
    if (online_store_ != nullptr) {
      return false;
    }
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
    DrainBackgroundWork();
    auto online_lock = LockOnlineWrite();
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
    const ValueStore::LayoutId layout_id = value_store_->GetLayoutId();
    UpdateChecksumPod(&checksum, layout_id.strategy);
    UpdateChecksumPod(&checksum, layout_id.checksum);
    UpdateChecksum(&checksum, metadata.data(), metadata.size());
    bool write_ok =
        WriteBytes(output, kCheckpointMagic.data(), kCheckpointMagic.size()) &&
        WritePod(output, kCheckpointVersion) &&
        WritePod(output, metadata_size) && WritePod(output, record_count) &&
        WritePod(output, layout_id.strategy) &&
        WritePod(output, layout_id.checksum) &&
        WriteBytes(output, metadata.data(), metadata.size());
    for (const uint64_t key : keys) {
      std::string value;
      ReadValue(key, value);
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
    DrainBackgroundWork();
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
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
        magic != kCheckpointMagic ||
        (version != kCheckpointVersion && version != 2) ||
        metadata_size > remaining ||
        metadata_size > std::numeric_limits<size_t>::max()) {
      LOG(ERROR) << "Invalid or truncated checkpoint header: " << file;
      return false;
    }
    ValueStore::LayoutId file_layout;
    if (version == kCheckpointVersion &&
        (!ReadPod(input, &file_layout.strategy, &remaining) ||
         !ReadPod(input, &file_layout.checksum, &remaining))) {
      LOG(ERROR) << "Truncated checkpoint layout identity: " << file;
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
    const ValueStore::LayoutId layout_id = value_store_->GetLayoutId();
    if (version == 2) {
      if (!layout_id.plain()) {
        LOG(ERROR) << "Checkpoint has no SSD layout identity: " << file;
        return false;
      }
    } else if (file_layout.strategy != layout_id.strategy ||
               file_layout.checksum != layout_id.checksum) {
      LOG(ERROR) << "Checkpoint SSD layout identity mismatch: " << file;
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
    if (version == kCheckpointVersion) {
      UpdateChecksumPod(&checksum, file_layout.strategy);
      UpdateChecksumPod(&checksum, file_layout.checksum);
    }
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
    DrainBackgroundWork();
    auto online_lock = LockOnlineWrite();
    BumpOnlineGeneration();
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
    auto online_lock = LockOnlineRead();
    const std::string fields =
        (value_store_ ? value_store_->ExtraResultFields() : "") +
        sgd_read_time_.Fields("tier_sgd_read") +
        sgd_write_time_.Fields("tier_sgd_write") +
        migration_read_time_.Fields("tier_migration_read") +
        migration_prepare_time_.Fields("tier_migration_prepare") +
        migration_publish_time_.Fields("tier_migration_publish");
    if (online_store_ == nullptr)
      return fields;
    return fields +
           " online_promoted_rows=" + std::to_string(online_promoted_rows_) +
           " online_migration_read_rows=" +
           std::to_string(online_migration_read_rows_) +
           " online_exchange_batches=" +
           std::to_string(online_exchange_batches_) +
           " online_capacity_rejects=" +
           std::to_string(online_capacity_rejects_) +
           " online_async=" + std::to_string(online_async_) +
           " online_submitted_jobs=" + std::to_string(online_submitted_jobs_) +
           " online_stale_jobs=" + std::to_string(online_stale_jobs_) +
           " online_queue_drops=" + std::to_string(online_queue_drops_) +
           " online_ghost_keys=" + std::to_string(online_ghost_.size()) +
           " online_ghost_limit=" + std::to_string(online_ghost_limit_);
  }

  void ResetStats() override {
    DrainBackgroundWork();
    auto online_lock = LockOnlineWrite();
    sgd_read_time_.Reset();
    sgd_write_time_.Reset();
    migration_read_time_.Reset();
    migration_prepare_time_.Reset();
    migration_publish_time_.Reset();
    if (online_store_ != nullptr) {
      online_promoted_rows_       = 0;
      online_migration_read_rows_ = 0;
      online_exchange_batches_    = 0;
      online_capacity_rejects_    = 0;
      online_submitted_jobs_ = online_stale_jobs_ = online_queue_drops_ = 0;
    }
    if (value_store_) {
      value_store_->ResetStats();
    }
  }

private:
  bool BatchGetFlatRangeImpl(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t row_dim,
      int64_t value_offset,
      int64_t value_dim,
      unsigned tid,
      MissingPositionsOutput* missing,
      BatchGetFlatStats* stats,
      bool collect_profile) {
    if (missing != nullptr) {
      missing->count = 0;
    }
    if (online_store_ != nullptr) {
      return false;
    }
    if (values == nullptr || num_rows < 0 || row_dim <= 0 || value_offset < 0 ||
        value_dim <= 0 || value_offset > row_dim ||
        value_dim > row_dim - value_offset ||
        keys.Size() != static_cast<size_t>(num_rows)) {
      return false;
    }

    if (missing != nullptr) {
      if (missing->capacity < static_cast<size_t>(num_rows) ||
          (num_rows > 0 && missing->positions == nullptr)) {
        return false;
      }
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

    if (num_rows == 0) {
      return true;
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

    if (missing != nullptr && missing_rows > 0) {
      size_t count = 0;
      for (size_t row = 0; row < handles.size(); ++row) {
        if (handles[row] == kValueHandleNone) {
          missing->positions[count] = row;
          ++count;
        }
      }
      missing->count = count;
    }

    return true;
  }

  struct TieringTime {
    std::atomic<uint64_t> total_ns{0}, max_ns{0}, calls{0};

    void Record(std::chrono::steady_clock::time_point begin) {
      const uint64_t elapsed = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - begin)
              .count());
      total_ns.fetch_add(elapsed, std::memory_order_relaxed);
      calls.fetch_add(1, std::memory_order_relaxed);
      uint64_t maximum = max_ns.load(std::memory_order_relaxed);
      while (maximum < elapsed &&
             !max_ns.compare_exchange_weak(
                 maximum, elapsed, std::memory_order_relaxed)) {
      }
    }

    void Reset() {
      total_ns.store(0, std::memory_order_relaxed);
      max_ns.store(0, std::memory_order_relaxed);
      calls.store(0, std::memory_order_relaxed);
    }

    std::string Fields(const std::string& name) const {
      return " " + name + "_ns=" + std::to_string(total_ns.load()) + " " +
             name + "_max_ns=" + std::to_string(max_ns.load()) + " " + name +
             "_calls=" + std::to_string(calls.load());
    }
  };

  void ApplySgdMixedRows(
      const std::vector<uint64_t>& keys,
      const float* grads,
      int64_t dim,
      float learning_rate,
      unsigned tid,
      std::vector<uint64_t>* cold_keys,
      std::vector<OnlineSnapshot>* snapshots,
      UpdateFlatStats* stats) {
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(float);
    ValidateOnlineRowSize(row_bytes);
    std::vector<uint64_t> unique_keys;
    std::vector<size_t> row_indices;
    std::unordered_map<uint64_t, size_t> positions;
    positions.reserve(keys.size());
    for (uint64_t key : keys) {
      const auto inserted = positions.emplace(key, unique_keys.size());
      if (inserted.second)
        unique_keys.push_back(key);
      row_indices.push_back(inserted.first->second);
    }
    std::vector<Value_t> handles(unique_keys.size(), kValueHandleNone);
    index_->BatchGet(
        base::ConstArray<uint64_t>(unique_keys), handles.data(), tid);
    std::vector<float*> pointers(unique_keys.size());
    std::vector<std::vector<float>> missing(unique_keys.size());
    std::vector<uint64_t> ssd_handles;
    std::vector<size_t> ssd_rows;
    for (size_t i = 0; i < unique_keys.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        missing[i].assign(static_cast<size_t>(dim), 0.0f);
        pointers[i] = missing[i].data();
      } else if (const char* ptr = value_store_->DirectPtr(handles[i])) {
        if (value_store_->SlotCapacity(handles[i]) < row_bytes) {
          throw std::runtime_error("SGD row dimension exceeds DRAM value size");
        }
        pointers[i] = reinterpret_cast<float*>(const_cast<char*>(ptr));
      } else {
        ssd_handles.push_back(handles[i]);
        ssd_rows.push_back(i);
        cold_keys->push_back(unique_keys[i]);
      }
    }
    std::vector<ValueStore::ReadResult> ssd_values;
    if (!ssd_handles.empty()) {
      const auto begin = std::chrono::steady_clock::now();
      value_store_->BatchRead(ssd_handles, ssd_values);
      sgd_read_time_.Record(begin);
    }
    if (ssd_values.size() != ssd_rows.size())
      throw std::runtime_error("SGD read count mismatch");
    std::vector<std::vector<float>> ssd_floats(ssd_rows.size());
    for (size_t i = 0; i < ssd_rows.size(); ++i) {
      if (ssd_values[i].data.size() != row_bytes)
        throw std::runtime_error("SGD SSD row size mismatch");
      ssd_floats[i].resize(static_cast<size_t>(dim));
      std::memcpy(ssd_floats[i].data(), ssd_values[i].data.data(), row_bytes);
      pointers[ssd_rows[i]] = ssd_floats[i].data();
    }
    // Duplicate keys share a destination and apply gradients in input order.
    for (size_t row = 0; row < keys.size(); ++row) {
      float* value      = pointers[row_indices[row]];
      const float* grad = grads + row * dim;
#pragma omp simd
      for (int64_t column = 0; column < dim; ++column) {
        value[column] -= learning_rate * grad[column];
      }
    }
    std::vector<ValueStore::WriteSpec> updates;
    for (const auto& value : ssd_floats)
      updates.push_back({value.data(), row_bytes});
    // BatchGet's SSD results are copies, so every modified cold row needs
    // explicit writeback before a subsequent read or promotion can see it.
    if (!ssd_handles.empty()) {
      const auto begin = std::chrono::steady_clock::now();
      value_store_->BatchWrite(ssd_handles, updates);
      sgd_write_time_.Record(begin);
    }
    for (size_t i = 0; i < unique_keys.size(); ++i) {
      if (handles[i] != kValueHandleNone)
        continue;
      PutInternal(unique_keys[i], missing[i].data(), row_bytes, tid, false);
      if (stats != nullptr)
        ++stats->missing_rows;
    }
    for (size_t i = 0; i < ssd_rows.size(); ++i) {
      const size_t row = ssd_rows[i];
      CaptureSnapshot(
          *snapshots,
          unique_keys[row],
          handles[row],
          ssd_floats[i].data(),
          row_bytes,
          keys.size());
    }
  }

  std::shared_lock<folly::SharedMutex> LockOnlineRead() const {
    std::shared_lock<folly::SharedMutex> lock(online_mu_, std::defer_lock);
    if (online_store_ != nullptr) {
      RethrowBackgroundError();
      lock.lock();
    }
    return lock;
  }

  std::unique_lock<folly::SharedMutex> LockOnlineWrite() const {
    std::unique_lock<folly::SharedMutex> lock(online_mu_, std::defer_lock);
    if (online_store_ != nullptr) {
      RethrowBackgroundError();
      lock.lock();
    }
    return lock;
  }

  void RethrowBackgroundError() const {
    if (!online_async_)
      return;
    std::lock_guard<std::mutex> lock(background_mu_);
    if (background_error_)
      std::rethrow_exception(background_error_);
  }

  void BumpOnlineGeneration() {
    if (online_store_ != nullptr)
      ++online_generation_;
  }

  static size_t AdmissionBudget(size_t rows) {
    return std::min<size_t>(256, 16 * std::max<size_t>(1, (rows + 499) / 500));
  }

  void CaptureSnapshot(
      std::vector<OnlineSnapshot>& snapshots,
      uint64_t key,
      uint64_t handle,
      const void* data,
      size_t size,
      size_t request_rows) const {
    if (!online_async_ || handle == kValueHandleNone ||
        snapshots.size() >= AdmissionBudget(request_rows) ||
        online_ghost_.count(key) == 0 ||
        value_store_->DirectPtr(handle) != nullptr)
      return;
    if (size != default_value_size_hint_)
      throw std::runtime_error("online snapshot size mismatch");
    for (const auto& snapshot : snapshots)
      if (snapshot.key == key)
        return;
    const char* begin = static_cast<const char*>(data);
    snapshots.push_back({key, handle, std::vector<char>(begin, begin + size)});
  }

  void BackgroundLoop() {
    for (;;) {
      std::unique_ptr<OnlineJob> job;
      {
        std::unique_lock<std::mutex> lock(background_mu_);
        background_cv_.wait(lock, [this] {
          return background_stop_ || background_pending_;
        });
        if (!background_pending_)
          return;
        job                = std::move(background_pending_);
        background_active_ = true;
      }
      try {
        RunBackgroundJob(*job);
      } catch (...) {
        std::lock_guard<std::mutex> lock(background_mu_);
        background_error_ = std::current_exception();
        background_pending_.reset();
        background_stop_ = true;
      }
      {
        std::lock_guard<std::mutex> lock(background_mu_);
        background_active_ = false;
      }
      background_cv_.notify_all();
    }
  }

  void RunBackgroundJob(OnlineJob& job) {
    std::vector<uint64_t> victim_keys, hot_keys, old_handles;
    std::vector<size_t> victim_slots;
    std::vector<std::vector<char>> victim_data;
    {
      auto lock = LockOnlineWrite();
      if (job.generation != online_generation_) {
        ++online_stale_jobs_;
        return;
      }
      job.hot.erase(std::remove_if(job.hot.begin(),
                                   job.hot.end(),
                                   [&](const auto& hot) {
                                     Value_t current = kValueHandleNone;
                                     index_->Get(hot.key, current, 0);
                                     return current != hot.handle;
                                   }),
                    job.hot.end());
      if (job.hot.empty()) {
        ++online_stale_jobs_;
        return;
      }
      for (size_t scanned = 0; scanned < online_residents_.size() &&
                               victim_keys.size() < job.hot.size();
           ++scanned) {
        const size_t slot  = online_cursor_++ % online_residents_.size();
        const uint64_t key = online_residents_[slot];
        Value_t handle     = kValueHandleNone;
        index_->Get(key, handle, 0);
        if (handle == kValueHandleNone)
          continue;
        const char* ptr = value_store_->DirectPtr(handle);
        if (ptr == nullptr)
          continue;
        victim_slots.push_back(slot);
        victim_keys.push_back(key);
        old_handles.push_back(handle);
        victim_data.emplace_back(ptr, ptr + default_value_size_hint_);
      }
    }
    const size_t count = victim_keys.size();
    if (count == 0)
      return;
    std::vector<ValueStore::WriteSpec> victim_specs, hot_specs;
    for (const auto& value : victim_data)
      victim_specs.push_back({value.data(), value.size()});
    for (size_t i = 0; i < count; ++i) {
      hot_keys.push_back(job.hot[i].key);
      old_handles.push_back(job.hot[i].handle);
      hot_specs.push_back({job.hot[i].data.data(), job.hot[i].data.size()});
    }
    // Private copies need no engine guard. A foreground mutation invalidates
    // the entire batch before it can publish either side of the exchange.
    const auto prepare_begin = std::chrono::steady_clock::now();
    auto replacements = online_store_->PrepareExchange(victim_specs, hot_specs);
    migration_prepare_time_.Record(prepare_begin);
    bool published = false;
    try {
      auto lock = LockOnlineWrite();
      if (replacements.empty()) {
        online_capacity_rejects_ += count;
        return;
      }
      if (job.generation != online_generation_) {
        ++online_stale_jobs_;
      } else {
        std::vector<uint64_t> keys = victim_keys;
        keys.insert(keys.end(), hot_keys.begin(), hot_keys.end());
        for (size_t i = 0; i < keys.size(); ++i) {
          Value_t current = kValueHandleNone;
          index_->Get(keys[i], current, 0);
          if (current != old_handles[i])
            throw std::logic_error("online generation missed a mutation");
        }
        std::vector<Value_t> retired(keys.size(), kValueHandleNone);
        const auto publish_begin = std::chrono::steady_clock::now();
        online_store_->ExchangeDramKeys(victim_keys, hot_keys);
        index_->BatchPutWithOld(
            base::ConstArray<uint64_t>(keys),
            replacements.data(),
            retired.data(),
            0);
        published = true;
        // A completed migration preserves values. Only foreground mutations
        // invalidate other snapshots; source handles detect moved rows.
        for (size_t i = 0; i < count; ++i) {
          CHECK_EQ(retired[i], old_handles[i]);
          CHECK_EQ(retired[i + count], old_handles[i + count]);
          online_residents_[victim_slots[i]] = hot_keys[i];
          online_ghost_.erase(hot_keys[i]);
        }
        for (uint64_t handle : retired)
          value_store_->Retire(handle);
        online_promoted_rows_ += count;
        ++online_exchange_batches_;
        migration_publish_time_.Record(publish_begin);
      }
    } catch (...) {
      if (!published)
        for (uint64_t handle : replacements)
          value_store_->Free(handle);
      throw;
    }
    if (!published)
      for (uint64_t handle : replacements)
        value_store_->Free(handle);
  }

  void ValidateOnlineRowSize(size_t size) const {
    if (online_store_ != nullptr && size != default_value_size_hint_) {
      throw std::invalid_argument(
          "online admission requires the configured fixed row size");
    }
  }

  void ReadValue(uint64_t key,
                 std::string& value,
                 bool* cold          = nullptr,
                 Value_t* out_handle = nullptr) {
    Value_t handle = kValueHandleNone;
    index_->Get(key, handle);
    if (out_handle != nullptr)
      *out_handle = handle;
    if (handle == kValueHandleNone) {
      value.clear();
      return;
    }
    if (const char* ptr = value_store_->DirectPtr(handle)) {
      value.resize(online_store_ != nullptr
                       ? default_value_size_hint_
                       : value_store_->SlotCapacity(handle));
      std::memcpy(value.data(), ptr, value.size());
      return;
    }
    if (cold != nullptr)
      *cold = true;
    value.resize(value_store_->SlotCapacity(handle));
    value.resize(value_store_->Read(handle, value.data(), value.size()));
  }

  void MaybeAdmit(const std::vector<uint64_t>& cold_keys,
                  unsigned tid,
                  size_t request_rows,
                  std::vector<OnlineSnapshot> snapshots = {},
                  uint64_t generation                   = 0) {
    auto online_lock = LockOnlineWrite();
    // LOCAL_SHM may combine many 500-row windows into a single request.
    const size_t budget = AdmissionBudget(request_rows);
    std::vector<uint64_t> hot_keys, hot_handles;
    std::unordered_set<uint64_t> seen;
    for (uint64_t key : cold_keys) {
      if (!seen.insert(key).second || online_store_->DramKeyEligible(key))
        continue;
      Value_t handle = kValueHandleNone;
      index_->Get(key, handle, tid);
      if (handle == kValueHandleNone ||
          value_store_->DirectPtr(handle) != nullptr)
        continue;
      if (online_ghost_.count(key) == 0) {
        const uint64_t token = ++online_ghost_token_;
        online_ghost_[key]   = token;
        online_ghost_fifo_.emplace_back(key, token);
        while (online_ghost_fifo_.size() > online_ghost_limit_) {
          const auto expired = online_ghost_fifo_.front();
          online_ghost_fifo_.pop_front();
          auto found = online_ghost_.find(expired.first);
          if (found != online_ghost_.end() && found->second == expired.second) {
            online_ghost_.erase(found);
          }
        }
        continue;
      }
      hot_keys.push_back(key);
      hot_handles.push_back(handle);
      if (hot_keys.size() == budget)
        break;
    }
    if (hot_keys.empty())
      return;
    if (online_async_) {
      if (generation != online_generation_) {
        ++online_stale_jobs_;
        return;
      }
      auto job        = std::make_unique<OnlineJob>();
      job->generation = generation;
      for (size_t i = 0; i < hot_keys.size(); ++i) {
        auto found = std::find_if(
            snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
              return snapshot.key == hot_keys[i] &&
                     snapshot.handle == hot_handles[i];
            });
        if (found != snapshots.end())
          job->hot.push_back(std::move(*found));
      }
      if (job->hot.empty())
        return;
      {
        std::lock_guard<std::mutex> lock(background_mu_);
        if (background_pending_ &&
            background_pending_->generation != generation) {
          background_pending_.reset();
          ++online_stale_jobs_;
        }
        if (background_pending_) {
          for (auto& hot : job->hot) {
            auto& pending = background_pending_->hot;
            if (pending.size() == 256) {
              ++online_queue_drops_;
              break;
            }
            if (std::none_of(
                    pending.begin(), pending.end(), [&](const auto& row) {
                      return row.key == hot.key;
                    })) {
              pending.push_back(std::move(hot));
            }
          }
        } else {
          background_pending_ = std::move(job);
          ++online_submitted_jobs_;
        }
      }
      background_cv_.notify_one();
      return;
    }

    std::vector<uint64_t> victim_keys, victim_handles;
    std::vector<size_t> victim_slots;
    std::vector<std::vector<char>> victim_data;
    for (size_t scanned = 0; scanned < online_residents_.size() &&
                             victim_keys.size() < hot_keys.size();
         ++scanned) {
      const size_t slot  = online_cursor_++ % online_residents_.size();
      const uint64_t key = online_residents_[slot];
      Value_t handle     = kValueHandleNone;
      index_->Get(key, handle, tid);
      if (handle == kValueHandleNone)
        continue;
      const char* ptr = value_store_->DirectPtr(handle);
      if (ptr == nullptr)
        continue;
      victim_slots.push_back(slot);
      victim_keys.push_back(key);
      victim_handles.push_back(handle);
      victim_data.emplace_back(ptr, ptr + default_value_size_hint_);
    }
    const size_t count = victim_keys.size();
    if (count == 0)
      return;
    hot_keys.resize(count);
    hot_handles.resize(count);
    std::vector<ValueStore::ReadResult> hot_data;
    online_migration_read_rows_ += count;
    // Re-read after acquiring the write lock: a sparse update may have
    // replaced the value between the foreground read and this admission.
    const auto read_begin = std::chrono::steady_clock::now();
    value_store_->BatchRead(hot_handles, hot_data);
    migration_read_time_.Record(read_begin);
    if (hot_data.size() != count)
      throw std::runtime_error("online read count mismatch");
    std::vector<uint64_t> keys = victim_keys;
    keys.insert(keys.end(), hot_keys.begin(), hot_keys.end());
    std::vector<ValueStore::WriteSpec> specs;
    for (const auto& value : victim_data)
      specs.push_back({value.data(), value.size()});
    for (const auto& value : hot_data) {
      if (value.data.size() != default_value_size_hint_) {
        throw std::runtime_error("online value size mismatch");
      }
      specs.push_back({value.data.data(), value.data.size()});
    }
    std::vector<Value_t> old(keys.size(), kValueHandleNone);
    online_store_->ExchangeDramKeys(victim_keys, hot_keys);
    std::vector<uint64_t> replacements;
    const auto prepare_begin = std::chrono::steady_clock::now();
    try {
      replacements = value_store_->BatchAllocAndWrite(keys, specs);
    } catch (...) {
      online_store_->ExchangeDramKeys(hot_keys, victim_keys);
      throw;
    }
    migration_prepare_time_.Record(prepare_begin);
    bool valid = replacements.size() == keys.size();
    for (size_t i = 0; i < replacements.size(); ++i) {
      valid =
          valid && replacements[i] != kValueHandleNone &&
          (value_store_->DirectPtr(replacements[i]) != nullptr) == (i >= count);
    }
    if (!valid) {
      for (uint64_t handle : replacements)
        value_store_->Free(handle);
      online_store_->ExchangeDramKeys(hot_keys, victim_keys);
      online_capacity_rejects_ += count;
      return;
    }
    const auto publish_begin = std::chrono::steady_clock::now();
    index_->BatchPutWithOld(
        base::ConstArray<uint64_t>(keys), replacements.data(), old.data(), tid);
    for (size_t i = 0; i < count; ++i) {
      CHECK_EQ(old[i], victim_handles[i]);
      CHECK_EQ(old[i + count], hot_handles[i]);
      online_residents_[victim_slots[i]] = hot_keys[i];
      online_ghost_.erase(hot_keys[i]);
    }
    // Both new copies are complete before publication. All readers hold a
    // shared guard through their copy, so retiring these handles is safe.
    for (uint64_t handle : old)
      value_store_->Retire(handle);
    online_promoted_rows_ += count;
    ++online_exchange_batches_;
    migration_publish_time_.Record(publish_begin);
  }

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

  struct PutInternalStats {
    std::uint64_t alloc_ns = 0;
    std::uint64_t index_ns = 0;
    std::uint64_t track_ns = 0;
  };

  void PutInternal(uint64_t key,
                   const void* data,
                   size_t size,
                   unsigned tid,
                   bool emit_fence,
                   PutInternalStats* stats = nullptr) {
    ValidateOnlineRowSize(size);
    (void)tid;
    (void)emit_fence;
    using clock = std::chrono::steady_clock;
    const auto alloc_start =
        stats != nullptr ? clock::now() : clock::time_point{};
    Value_t new_handle = value_store_->AllocAndWrite(key, data, size);
    if (new_handle == kValueHandleNone) {
      LOG(FATAL) << "KVEngine value allocation failed, key=" << key
                 << " size=" << size;
      return;
    }
    const auto index_start =
        stats != nullptr ? clock::now() : clock::time_point{};
    Value_t old_handle = index_->Put(key, new_handle, tid);
    if (old_handle != kValueHandleNone && old_handle != new_handle) {
      value_store_->Retire(old_handle);
    }
    const auto track_start =
        stats != nullptr ? clock::now() : clock::time_point{};
    // Checkpoint tracking already contains overwritten keys. Delete removes
    // them under the exclusive checkpoint lock, so reinsertion tracks again.
    if (old_handle == kValueHandleNone) {
      TrackKey(key, tid);
    }
    if (stats != nullptr) {
      const auto end = clock::now();
      stats->alloc_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              index_start - alloc_start)
              .count());
      stats->index_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              track_start - index_start)
              .count());
      stats->track_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              end - track_start)
              .count());
    }
  }

  inline static constexpr std::array<char, 8> kCheckpointMagic = {
      'R', 'S', 'K', 'V', 'C', 'P', '0', '1'};
  inline static constexpr uint32_t kCheckpointVersion = 3;
  inline static constexpr uint64_t kCheckpointChecksumSeed =
      0x9e3779b97f4a7c15ULL;

  BaseKVConfig config_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<ValueStore> value_store_;
  int num_threads_                = 0;
  size_t default_value_size_hint_ = 0;
  HybridValueStore* online_store_ = nullptr;
  // ponytail: serialize opt-in exchange and writes; shard this guard only
  // after measurements show contention on a changing working set.
  mutable folly::SharedMutex online_mu_;
  // ponytail: a single pending batch and one worker bound temporary payloads;
  // use per-row versions only if global-generation retries limit admission.
  bool online_async_          = false;
  uint64_t online_generation_ = 0;
  std::thread background_thread_;
  mutable std::mutex background_mu_;
  std::condition_variable background_cv_;
  std::unique_ptr<OnlineJob> background_pending_;
  bool background_active_ = false, background_stop_ = false;
  std::exception_ptr background_error_;
  std::vector<uint64_t> online_residents_;
  size_t online_cursor_        = 0;
  size_t online_ghost_limit_   = 0;
  uint64_t online_ghost_token_ = 0;
  std::unordered_map<uint64_t, uint64_t> online_ghost_;
  std::deque<std::pair<uint64_t, uint64_t>> online_ghost_fifo_;
  uint64_t online_promoted_rows_       = 0;
  uint64_t online_migration_read_rows_ = 0;
  uint64_t online_exchange_batches_    = 0;
  uint64_t online_capacity_rejects_    = 0;
  uint64_t online_submitted_jobs_ = 0, online_stale_jobs_ = 0,
           online_queue_drops_ = 0;
  TieringTime sgd_read_time_, sgd_write_time_;
  TieringTime migration_read_time_, migration_prepare_time_,
      migration_publish_time_;
  mutable std::shared_mutex checkpoint_mu_;
  std::vector<std::vector<uint64_t>> checkpoint_keys_by_thread_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
