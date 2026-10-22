#pragma once

#include <memory>
#include <new>
#include <stdexcept>

#include "base/factory.h"
#include "storage/index/index.h"
#include "storage/nvm/pet_kv/pet_hash.h"

class DramPetHashIndex : public Index {
public:
  explicit DramPetHashIndex(const BaseKVConfig& config) : Index(config) {
    if (config.json_config_.contains("index")) {
      prefetch_distance_ =
          config.json_config_.at("index").value("prefetch_distance", 1);
    }
    if (prefetch_distance_ < 0 || prefetch_distance_ > 64) {
      throw std::invalid_argument("prefetch_distance must be in [0, 64]");
    }
    const uint64_t capacity =
        config.json_config_.at("capacity").get<uint64_t>();
    const size_t bytes =
        base::PetHash<Key_t, Value_t, false>::MemorySize(capacity);
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, bytes) != 0 || mem == nullptr) {
      throw std::bad_alloc();
    }
    impl_ = new (mem) base::PetHash<Key_t, Value_t, false>();
    impl_->Initialize(capacity);
  }

  ~DramPetHashIndex() override {
    if (impl_ != nullptr) {
      impl_->~PetHash<Key_t, Value_t, false>();
      std::free(impl_);
      impl_ = nullptr;
    }
  }

  void Get(Key_t key, Value_t& pointer, unsigned tid) override {
    (void)tid;
    auto [value, exists] = impl_->Get(key);
    pointer              = exists ? value : NONE;
  }

  Value_t Put(Key_t key, Value_t pointer, unsigned tid) override {
    (void)tid;
    Value_t old_handle = kValueHandleNone;
    impl_->Set(key, pointer, nullptr, false, &old_handle);
    return old_handle;
  }

  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    (void)tid;
    const int distance = prefetch_distance_;
    for (int i = 0; i < keys.Size(); ++i) {
      if (distance != 0 && i < keys.Size() - distance) {
        impl_->HintPrefetch(keys[i + distance]);
      }
      auto [value, exists] = impl_->Get(keys[i]);
      pointers[i]          = exists ? value : NONE;
    }
  }

  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override {
    for (int i = 0; i < keys.Size(); ++i) {
      Put(keys[i], pointers[i], tid);
    }
  }

  bool Delete(Key_t& key) override { return impl_->Delete(key); }
  size_t Capacity() override { return impl_->Capacity(); }

private:
  int prefetch_distance_                      = 1;
  base::PetHash<Key_t, Value_t, false>* impl_ = nullptr;
};

FACTORY_REGISTER(Index, DRAM_PET_HASH, DramPetHashIndex, const BaseKVConfig&);
