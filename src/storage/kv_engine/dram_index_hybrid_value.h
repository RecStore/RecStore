#pragma once

#include <shared_mutex>
#include <string>
#include "base/factory.h"
#include "base_kv.h"
#include "src/storage/value/value.h"
#include "storage/hybrid/index.h"
#include "storage/hybrid/index_factory.h"

class KVEngineHybrid : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineHybrid(const BaseKVConfig& config)
      : BaseKV(config),
        valm(config.json_config_.at("path").get<std::string>() + "/shmvalue",
             config.json_config_.at("shmcapacity").get<size_t>(),
             config.json_config_.at("path").get<std::string>() + "/ssdvalue",
             config.json_config_.at("ssdcapacity").get<size_t>(),
             config) {
    // 创建 index
    using IndexF = base::Factory<Index, const BaseKVConfig&>;
    std::string index_type = config.json_config_.value("index_type", "ExtendibleHash");
    index_.reset(IndexF::NewInstance(index_type, config));
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    Value_t raw = 0;
    index_->Get(key, raw, tid);
    if (!raw) {
      value = {};
      return;
    }
    value = valm.RetrieveValue(raw, tid);
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    Value_t old_raw        = 0;
    index_->Get(key, old_raw, tid);
    UnifiedPointer new_ptr = valm.WriteValue(value);
    index_->Put(key, new_ptr.RawValue(), tid);
    // TODO: valm.Free(UnifiedPointer::FromRaw(old_raw)) 等 free 实现后加
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    std::unique_lock<std::shared_mutex> _(lock_);
    values->clear();
    values->reserve(keys.Size());

    storage_.clear();
    storage_.reserve(keys.Size());
    for (int k = 0; k < keys.Size(); k++) {
      Value_t raw = 0;
      index_->Get(keys[k], raw, tid);
      std::string temp_value = raw ? valm.RetrieveValue(raw, tid) : std::string();
      storage_.push_back(std::move(temp_value));

      const std::string& stored_value = storage_.back();
      if (stored_value.empty()) {
        values->push_back(base::ConstArray<float>(nullptr, 0));
      } else {
        const float* float_ptr = reinterpret_cast<const float*>(stored_value.data());
        size_t float_size      = stored_value.size() / sizeof(float);
        values->push_back(base::ConstArray<float>(float_ptr, float_size));
      }
    }
  }

  ~KVEngineHybrid() {
    std::cout << "exit KVEngineHybrid" << std::endl;
    storage_.clear();
  }

private:
  ValueManager valm;
  std::unique_ptr<Index> index_;
  mutable std::shared_mutex lock_;
  std::vector<std::string> storage_;
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid, const BaseKVConfig&);
