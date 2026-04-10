#pragma once

#include <shared_mutex>
#include <string>
#include "base/factory.h"
#include "base_kv.h"
#include "src/storage/hybrid/value.h"
#include "storage/index/index.h"
class KVEngineHybrid : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineHybrid(const BaseKVConfig& config)
      : BaseKV(config),
        valm(config.json_config_.at("path").get<std::string>() + "/shmvalue",
             config.json_config_.at("shmcapacity").get<size_t>(),
             config.json_config_.at("path").get<std::string>() + "/ssdvalue",
             config.json_config_.at("ssdcapacity").get<size_t>(),
             [&] {
               BaseKVConfig ic;
               ic.json_config_ = config.json_config_;
               return ic;
             }()) {}

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    value = valm.RetrieveValue(key, tid);
  }

  void GetIndex(uint64_t key, uint64_t& pointer, unsigned int tid) {
    valm.index->Get(key, pointer, tid);
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    valm.WriteValue(key, value, 0);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    std::unique_lock<std::shared_mutex> _(lock_);
    values->clear();
    values->reserve(keys.Size());

    // 清理上次调用的缓存，确保数据不会累积
    storage_.clear();
    storage_.reserve(keys.Size());
    for (int k = 0; k < keys.Size(); k++) {
      // 1. 从 ValueManager 获取原始字符串
      std::string temp_values = valm.RetrieveValue(keys[k], tid);

      // 2. 将字符串的副本存入成员变量 storage_，以确保其生命周期足够长，
      //    防止在函数返回后指针失效。
      storage_.push_back(std::move(temp_values));

      const std::string& stored_value = storage_.back();

      if (stored_value.empty()) {
        // 如果值为空，则返回一个空的 ConstArray
        values->push_back(base::ConstArray<float>(nullptr, 0));
      } else {
        // 3. 执行核心的 reinterpret_cast 逻辑来满足测试需求
        const char* char_ptr   = stored_value.data();
        const float* float_ptr = reinterpret_cast<const float*>(char_ptr);

        // 4. 计算出伪装后的 float 数组的大小
        size_t float_size = stored_value.size() / sizeof(float);

        // 5. 创建指向伪装数据的 ConstArray 并添加到结果中
        values->push_back(base::ConstArray<float>(float_ptr, float_size));
      }
    }
  }

  ~KVEngineHybrid() {
    std::cout << "exit KVEngineHybrid" << std::endl;
    for (int i = 0; i < storage_.size(); i++) {
      storage_[i].clear();
    }
  }

private:
  ValueManager valm;
  mutable std::shared_mutex lock_;
  std::vector<std::string> storage_;
  std::string dict_pool_name_;
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid, const BaseKVConfig&);
