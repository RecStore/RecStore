#pragma once

#include <sys/user.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "../nvm/pet_kv/shm_common.h"
#include "../ssd/SSD_CCEH.h"
#include "base/factory.h"
#include "base_kv.h"
#include "../io_backend/io_backend.h"
#include "storage/value/value.h"
#include "storage/value/pointer.h"

class KVEngineCCEH : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineCCEH(const BaseKVConfig& config) : BaseKV(config) {
    queue_cnt_ = config.json_config_.at("queue_cnt").get<int>();
    std::string io_backend_type =
        config.json_config_.at("io_backend_type").get<std::string>();
    LOG(INFO) << "--------------init KVEngineCCEH--------------------";
    std::string index_path = config.json_config_.at("path").get<std::string>();
    std::string index_db_path = index_path + "/cceh_index.db";
    std::string value_db_path = index_path + "/cceh_value.db";

    PageID_t index_offset = 0;
    PageID_t value_offset = 1;
    if (io_backend_type == "SPDK") {
      index_offset =
          config.json_config_.value("spdk_index_offset", (uint64_t)0);
      value_offset =
          config.json_config_.value("spdk_value_offset", (uint64_t)1000000);
    }

    BaseKVConfig index_config                   = config;
    index_config.json_config_["file_path"]      = index_db_path;
    index_config.json_config_["page_id_offset"] = index_offset;
    hash_table_ = std::make_unique<SSDCCEH>(index_config);

    value_storage_ =
        std::make_unique<SSDValueManager>(config, value_db_path, value_offset);

    LOG(INFO) << "After init value and index io_backend";
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
    UnifiedPointer ptr = value_storage_->Write(value);
    hash_table_->Put(key, ptr.RawValue(), tid);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    values->clear();
    int size = keys.Size();
    std::vector<Value_t> vals(size, NONE);
    pending = 0;
    coros.clear();
    coros.reserve(size);
    for (size_t i = 0; i < size; i++) {
      auto k = keys[i];
      coros.push_back(std::make_unique<coroutine<void>::pull_type>(
          [this, &vals, i, k, tid](auto& yield) {
            hash_table_->Get(yield, i, k, vals[i], tid);
          }));
    }
    while (pending)
      hash_table_->io_backend->PollCompletion();

    // 把有效的 raw 值转成 UnifiedPointer
    std::vector<UnifiedPointer> ptrs;
    std::vector<int> valid_indices;
    ptrs.reserve(size);
    valid_indices.reserve(size);
    for (int i = 0; i < size; i++) {
      if (vals[i] != NONE) {
        ptrs.push_back(UnifiedPointer::FromRaw(vals[i]));
        valid_indices.push_back(i);
      }
    }

    // 批量读 value
    std::vector<std::string> read_results = value_storage_->BatchRead(ptrs);

    // 组装输出
    static thread_local std::vector<std::vector<float>> value_buffers;
    value_buffers.clear();
    value_buffers.resize(size);
    int ri = 0;
    for (int i = 0; i < size; i++) {
      if (ri < (int)valid_indices.size() && valid_indices[ri] == i) {
        const std::string& s = read_results[ri];
        value_buffers[i].resize(s.size() / sizeof(float));
        memcpy(value_buffers[i].data(), s.data(), s.size());
        values->emplace_back(value_buffers[i].data(), value_buffers[i].size());
        ri++;
      } else {
        values->emplace_back();
      }
    }
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    int n = keys.Size();

    // 准备 string_view 列表交给 SSDValueManager 批量写
    std::vector<std::string_view> value_views(n);
    for (int i = 0; i < n; i++)
      value_views[i] = std::string_view((const char*)(*values)[i].Data(),
                                        (*values)[i].Size() * sizeof(float));

    std::vector<UnifiedPointer> ptrs = value_storage_->BatchWrite(value_views);

    // 更新索引
    for (int i = 0; i < n; i++)
      hash_table_->Put(keys[i], ptrs[i].RawValue(), tid);
  }

  ~KVEngineCCEH() { std::cout << "exit KVEngineCCEH" << std::endl; }

private:
  std::string dict_pool_name_;
  int queue_cnt_;
  std::string type;
  std::unique_ptr<SSDCCEH> hash_table_;
  std::unique_ptr<SSDValueManager> value_storage_;
};

FACTORY_REGISTER(BaseKV, KVEngineCCEH, KVEngineCCEH, const BaseKVConfig&);
