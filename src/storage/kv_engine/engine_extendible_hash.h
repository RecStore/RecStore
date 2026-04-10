#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <mutex>
#include <string>
#include <shared_mutex>

#include "../index/dram_cceh/extendible_hash.h"
#include "storage/index/index.h"
#include "storage/nvm/pet_kv/shm_common.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/allocators/allocator_factory.h"
#include "memory/memory_factory.h"

class KVEngineExtendibleHash : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;
  static constexpr size_t kLockStripeNum      = 4096;

public:
  KVEngineExtendibleHash(const BaseKVConfig& config) : BaseKV(config) {
    LOG(INFO)
        << "--------------init KVEngineExtendibleHash--------------------";
    const std::string value_path =
        config.json_config_.at("path").get<std::string>() + "/value";
    std::string value_medium = config.json_config_.value("value_type", "DRAM");
    std::transform(
        value_medium.begin(),
        value_medium.end(),
        value_medium.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    uint64_t value_budget_bytes = 0;
    if (value_medium == "DRAM" && config.json_config_.contains("DRAM_SIZE")) {
      value_budget_bytes = config.json_config_.at("DRAM_SIZE").get<uint64_t>();
    } else if (value_medium == "SSD" &&
               config.json_config_.contains("SSD_SIZE")) {
      value_budget_bytes = config.json_config_.at("SSD_SIZE").get<uint64_t>();
    }
    if (value_budget_bytes == 0) {
      value_budget_bytes = static_cast<uint64_t>(std::llround(
          1.2 * config.json_config_.at("capacity").get<size_t>() *
          config.json_config_.at("value_size").get<size_t>()));
    }
    const auto cap_bytes = static_cast<int64>(value_budget_bytes);

    shm_malloc_ = base::allocators::CreateAllocator(
        config.json_config_,
        value_path,
        cap_bytes,
        config.json_config_.value("value_type", "DRAM"));

    if (!shm_malloc_)
      throw std::runtime_error("init shm malloc failed");

    value_size_ = config.json_config_.at("value_size").get<int>();

    hash_table_ = new ExtendibleHash(config);

    std::string path = config.json_config_.at("path").get<std::string>();

    // 初始化值存储区域
    uint64_t value_shm_size =
        config.json_config_.at("capacity").get<uint64_t>() *
        config.json_config_.at("value_size").get<uint64_t>();

    if (!valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize)) {
      base::file_util::Delete(path + "/valid", false);
      CHECK(
          valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize));
      shm_malloc_->Initialize();
    }
    LOG(INFO) << "After init: [shm_malloc] " << shm_malloc_->GetInfo();
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    std::shared_lock<std::shared_mutex> lk(KeyMutex(key));
    base::PetKVData shmkv_data;
    // std::shared_lock<std::shared_mutex> _(lock_);

    Key_t hash_key = key;
    Value_t read_value;
    hash_table_->Get(hash_key, read_value, tid);

    if (read_value == NONE) {
      value = std::string();
    } else {
      shmkv_data.data_value = read_value;
      char* data = shm_malloc_->GetMallocData(shmkv_data.shm_malloc_offset());
      if (data == nullptr) {
        value = std::string();
        return;
      }
#ifdef XMH_VARIABLE_SIZE_KV
      int size = shm_malloc_->GetMallocSize(shmkv_data.shm_malloc_offset());
#else
      int size = value_size_;
#endif
      value = std::string(data, size);
    }
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    base::PetKVData shmkv_data;
    char* sync_data = shm_malloc_->New(value.size());
    if (sync_data == nullptr) {
      LOG(ERROR) << "shm malloc failed (OOM?), key: " << key
                 << " size: " << value.size();
      return;
    }
    shmkv_data.SetShmMallocOffset(shm_malloc_->GetMallocOffset(sync_data));
    memcpy(sync_data, value.data(), value.size());
    _mm_mfence();
    asm volatile("" ::: "memory");

    std::unique_lock<std::shared_mutex> lk(KeyMutex(key));
    Key_t hash_key = key;
    Value_t old_val;
    hash_table_->Get(hash_key, old_val, tid);
    if (old_val != NONE) {
      base::PetKVData old_shm_data;
      old_shm_data.data_value = old_val;
      shm_malloc_->Free(
          shm_malloc_->GetMallocData(old_shm_data.shm_malloc_offset()));
    }
    hash_table_->Put(hash_key, shmkv_data.data_value, tid);
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    if (values == nullptr) {
      LOG(FATAL) << "BatchPut values is nullptr";
    }
    if (keys.Size() != static_cast<int>(values->size())) {
      LOG(FATAL) << "BatchPut size mismatch, keys: " << keys.Size()
                 << " values: " << values->size();
    }

#pragma omp parallel for num_threads(8) if (keys.Size() > 1024)
    for (int i = 0; i < keys.Size(); ++i) {
      const uint64_t key                  = keys[i];
      const base::ConstArray<float>& item = (*values)[i];
      const char* data_ptr                = (const char*)item.Data();
      const size_t data_size              = item.Size() * sizeof(float);

      if (data_ptr == nullptr || data_size == 0) {
        Put(key, std::string_view(), tid);
      } else {
        Put(key, std::string_view(data_ptr, data_size), tid);
      }
    }
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    values->resize(keys.Size());
#pragma omp parallel for num_threads(8) if (keys.Size() > 1024)
    for (int i = 0; i < (int)keys.Size(); ++i) {
      uint64_t k = keys[i];
      std::shared_lock<std::shared_mutex> lk(KeyMutex(k));
      base::PetKVData shmkv_data;
      Key_t hash_key = k;
      Value_t read_value;
      hash_table_->Get(hash_key, read_value, tid);

      if (read_value == NONE) {
        (*values)[i] = base::ConstArray<float>();
      } else {
        shmkv_data.data_value = read_value;
        char* data = shm_malloc_->GetMallocData(shmkv_data.shm_malloc_offset());
        if (data == nullptr) {
          (*values)[i] = base::ConstArray<float>();
          continue;
        }
#ifdef XMH_VARIABLE_SIZE_KV
        int size = shm_malloc_->GetMallocSize(shmkv_data.shm_malloc_offset());
#else
        int size = value_size_;
#endif
        (*values)[i] =
            base::ConstArray<float>((float*)data, size / sizeof(float));
      }
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
  std::string dict_pool_name_;
  int value_size_;
  std::unique_ptr<base::MallocApi> shm_malloc_;
  base::ShmFile valid_shm_file_;
  std::array<std::shared_mutex, kLockStripeNum> key_mutexes_;
};

FACTORY_REGISTER(BaseKV,
                 KVEngineExtendibleHash,
                 KVEngineExtendibleHash,
                 const BaseKVConfig&);
