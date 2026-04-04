#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <mutex>
#include <string>
#include <shared_mutex>

#include "../dram/extendible_hash.h"
#include "base/factory.h"
#include "base_kv.h"
#include "storage/value/value.h"
#include "storage/value/pointer.h"

class KVEngineExtendibleHash : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;
  static constexpr size_t kLockStripeNum      = 4096;

public:
  KVEngineExtendibleHash(const BaseKVConfig& config) : BaseKV(config) {
    LOG(INFO)
        << "--------------init KVEngineExtendibleHash--------------------";

    std::string io_backend_type =
        config.json_config_.at("io_backend_type").get<std::string>();
    std::string value_db_path =
        config.json_config_.at("path").get<std::string>() + "/exthash_value.db";

    PageID_t value_offset = 1;
    if (io_backend_type == "SPDK") {
      value_offset =
          config.json_config_.value("spdk_value_offset", (uint64_t)1000000);
    }

    hash_table_ = new ExtendibleHash(config);
    value_storage_ =
        std::make_unique<SSDValueManager>(config, value_db_path, value_offset);

    LOG(INFO) << "After init KVEngineExtendibleHash";
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    Value_t raw = NONE;
    hash_table_->Get(key, raw, tid);
    if (raw == NONE) {
      value = {};
      return;
    }
    value = value_storage_->Read(UnifiedPointer::FromRaw(raw));
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    UnifiedPointer new_ptr = value_storage_->Write(value);
    hash_table_->Put(key, new_ptr.RawValue(), tid);
    // TODO: value_storage_->Free(UnifiedPointer::FromRaw(old_raw))
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    int size = keys.Size();

    // 查索引，收集有效的 UnifiedPointer
    std::vector<Value_t> raws(size, NONE);
    for (int i = 0; i < size; i++) {
      hash_table_->Get(keys[i], raws[i], tid);
    }

    std::vector<UnifiedPointer> ptrs;
    std::vector<int> valid_indices;
    ptrs.reserve(size);
    valid_indices.reserve(size);
    for (int i = 0; i < size; i++) {
      if (raws[i] != NONE) {
        ptrs.push_back(UnifiedPointer::FromRaw(raws[i]));
        valid_indices.push_back(i);
      }
    }

    // 批量读 value
    std::vector<std::string> read_results = value_storage_->BatchRead(ptrs);

    // 组装输出
    static thread_local std::vector<std::vector<float>> value_buffers;
    value_buffers.clear();
    value_buffers.resize(size);
    values->resize(size);
    int ri = 0;
    for (int i = 0; i < size; i++) {
      if (ri < (int)valid_indices.size() && valid_indices[ri] == i) {
        const std::string& s = read_results[ri];
        value_buffers[i].resize(s.size() / sizeof(float));
        memcpy(value_buffers[i].data(), s.data(), s.size());
        (*values)[i] = base::ConstArray<float>(
            value_buffers[i].data(), value_buffers[i].size());
        ri++;
      } else {
        (*values)[i] = base::ConstArray<float>();
      }
    }
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    int n = keys.Size();

    // Prepare a list of string_views for SSDValueManager batch write
    std::vector<std::string_view> value_views(n);
    for (int i = 0; i < n; i++) {
      value_views[i] =
          std::string_view(reinterpret_cast<const char*>((*values)[i].Data()),
                           (*values)[i].Size() * sizeof(float));
    }

    std::vector<UnifiedPointer> ptrs = value_storage_->BatchWrite(value_views);

    // Update index
    for (int i = 0; i < n; i++) {
      hash_table_->Put(keys[i], ptrs[i].RawValue(), tid);
    }
  }

  ~KVEngineExtendibleHash() {
    std::cout << "exit KVEngineExtendibleHash" << std::endl;
    if (hash_table_) {
      delete hash_table_;
      hash_table_ = nullptr;
    }
  }

private:
  std::shared_mutex& KeyMutex(uint64_t key) {
    return key_mutexes_[key & (kLockStripeNum - 1)];
  }

  ExtendibleHash* hash_table_;
  std::unique_ptr<SSDValueManager> value_storage_;
  std::array<std::shared_mutex, kLockStripeNum> key_mutexes_;
};

FACTORY_REGISTER(BaseKV,
                 KVEngineExtendibleHash,
                 KVEngineExtendibleHash,
                 const BaseKVConfig&);
