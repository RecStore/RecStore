#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "base/factory.h"
#include "base_kv.h"
#include "storage/nvm/pet_kv/pet_kv.h"

DECLARE_int32(prefetch_method);

class KVEnginePetKV : public BaseKV {
public:
  explicit KVEnginePetKV(const BaseKVConfig& config) : BaseKV(config) {
    const std::string shm_path = config.json_config_["path"];
    const int shard_num        = config.json_config_.value("shard_num", 16);
    const int value_size = config.json_config_.at("value_size").get<int>();
    const int capacity   = config.json_config_.at("capacity").get<int>();
    constexpr int64_t kMinShardMemory = 2LL * 1024 * 1024;
    const int64_t value_capacity =
        config.json_config_.at("value_capacity").get<int64_t>();
    const int shard_capacity =
        std::max(1, (capacity + shard_num - 1) / shard_num);
    const int64_t shard_memory =
        std::max<int64_t>(kMinShardMemory, value_capacity / shard_num);
    shm_kv = std::make_unique<base::PetMultiKV>(
        shm_path, shard_num, shard_memory, shard_capacity, value_size);
  }

  void Get(const uint64_t key, std::string& value, unsigned t) override {
    auto kv_data = shm_kv->Get(key);
    if (kv_data.data)
      value = std::string(kv_data.data, kv_data.size);
  }

  void BatchGet(base::ConstArray<uint64> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned t) override {
    values->clear();
    if (FLAGS_prefetch_method == 0) {
      for (auto k : keys) {
        auto kv_data = shm_kv->Get(k);
#ifdef RPC_DEBUG
        CHECK_NE(kv_data.size, 0) << "empty kv, key is " << k;
#endif
        values->emplace_back(
            (float*)kv_data.data, kv_data.size / sizeof(float));
      }
    } else if (FLAGS_prefetch_method == 1) {
      shm_kv->BatchGet(keys, values);
    }
  }

  void
  Put(const uint64_t key, const std::string_view& value, unsigned t) override {
    CHECK(shm_kv->Update(key, value.data(), value.size()));
  }

  std::pair<uint64_t, uint64_t> RegisterPMAddr() const {
    return base::PMMmapRegisterCenter::GetInstance()->ForRDMAMemoryRegion();
  }

  void DebugInfo() const override { shm_kv->GetInfo(); }

  // PetKV lives in POSIX shared memory with no durable backing file, so it
  // cannot produce or consume checkpoints. Say so explicitly instead of
  // inheriting BaseKV's silent false.
  bool SaveCheckpoint(const std::string& file,
                      const std::string& metadata) override {
    (void)file;
    (void)metadata;
    LOG(ERROR) << "KVEnginePetKV does not support SaveCheckpoint: shared "
                  "memory is not durable; use a checkpoint-capable engine";
    return false;
  }

  bool LoadCheckpoint(const std::string& file,
                      const std::string& metadata) override {
    (void)file;
    (void)metadata;
    LOG(ERROR) << "KVEnginePetKV does not support LoadCheckpoint: shared "
                  "memory is not durable; use a checkpoint-capable engine";
    return false;
  }

  void clear() override {
    LOG(ERROR) << "KVEnginePetKV does not support clear(): recreate the "
                  "parameter server to reset the store";
  }

private:
  base::ScopedTempDir dir;
  std::unique_ptr<base::PetMultiKV> shm_kv;
};

FACTORY_REGISTER(BaseKV, KVEnginePetKV, KVEnginePetKV, const BaseKVConfig&);
