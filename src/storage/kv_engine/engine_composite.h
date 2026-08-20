#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
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
        num_threads_(num_threads) {}

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
    TrackKeys(keys);
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

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
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
    std::vector<uint64_t> handles = value_store_->BatchAllocAndWrite(specs);
    if (handles.size() != static_cast<size_t>(keys.Size())) {
      LOG(FATAL) << "KVEngine::BulkLoad allocation result size mismatch";
    }
    for (int i = 0; i < keys.Size(); ++i) {
      if (handles[static_cast<size_t>(i)] == kValueHandleNone) {
        LOG(FATAL) << "KVEngine bulk value allocation failed, key=" << keys[i]
                   << " size=" << value_size;
      }
    }
    index_->BatchPut(keys, handles.data(), 0);
    TrackKeys(keys);
  }

  bool SaveCheckpoint(const std::string& file,
                      const std::string& metadata) override {
    std::unique_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    if (file.empty()) {
      LOG(ERROR) << "KVEngine checkpoint path is empty";
      return false;
    }

    std::vector<uint64_t> keys;
    {
      std::lock_guard<std::mutex> lock(checkpoint_keys_mu_);
      keys.assign(checkpoint_keys_.begin(), checkpoint_keys_.end());
    }
    std::sort(keys.begin(), keys.end());

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
    {
      std::lock_guard<std::mutex> lock(checkpoint_keys_mu_);
      if (!checkpoint_keys_.empty()) {
        LOG(ERROR) << "KVEngine checkpoint load requires an empty engine";
        return false;
      }
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

  uint64_t CheckpointRecordCount() const override {
    std::shared_lock<std::shared_mutex> checkpoint_lock(checkpoint_mu_);
    std::lock_guard<std::mutex> lock(checkpoint_keys_mu_);
    return static_cast<uint64_t>(checkpoint_keys_.size());
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

  void TrackKey(uint64_t key) {
    std::lock_guard<std::mutex> lock(checkpoint_keys_mu_);
    checkpoint_keys_.insert(key);
  }

  void TrackKeys(base::ConstArray<uint64_t> keys) {
    std::lock_guard<std::mutex> lock(checkpoint_keys_mu_);
    for (int i = 0; i < keys.Size(); ++i) {
      checkpoint_keys_.insert(keys[i]);
    }
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
    TrackKey(key);
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
  mutable std::mutex checkpoint_keys_mu_;
  std::unordered_set<uint64_t> checkpoint_keys_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineComposite, KVEngineComposite, const BaseKVConfig&);
