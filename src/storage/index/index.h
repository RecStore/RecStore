#pragma once
#include <boost/coroutine2/all.hpp>
#include <cstddef>
#include <glog/logging.h>
#include "base/array.h"
#include "base/log.h"
#include "utils/pair.h"
#include "storage/kv_engine/base_kv.h"

using boost::coroutines2::coroutine;

class Index {
public:
  virtual ~Index() { std::cout << "exit Index" << std::endl; }
  explicit Index(const BaseKVConfig& config){};

  virtual void Get(Key_t key, Value_t& pointer, unsigned tid) = 0;
  virtual void
  Get(coroutine<void>::push_type& sink,
      int index,
      Key_t key,
      Value_t& pointer,
      unsigned tid) {
    LOG(FATAL) << "not implemented";
  }
  virtual void Put(Key_t key, Value_t pointer, unsigned tid) = 0;
  virtual void
  Put(coroutine<void>::push_type& sink,
      int index,
      Key_t key,
      Value_t pointer,
      unsigned tid) {
    LOG(FATAL) << "not implemented";
  }
  virtual void
  BatchPut(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) = 0;
  virtual void
  BatchGet(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) = 0;
  virtual bool Delete(Key_t& key)                                         = 0;
  virtual double Utilization() { LOG(FATAL) << "not implemented"; }
  virtual size_t Capacity() { LOG(FATAL) << "not implemented"; }
  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) {
    LOG(FATAL) << "not implemented";
  };
  virtual void LoadFakeData(int64_t key_capacity, int value_size) {
    std::vector<uint64_t> keys;
    uint64_t* values =
        new uint64_t[value_size / sizeof(uint64_t) * key_capacity];
    keys.reserve(key_capacity);
    for (int64_t i = 0; i < key_capacity; i++) {
      keys.push_back(i);
      *(values + i)      = i;
      uint64_t ptr_value = *reinterpret_cast<const uint64_t*>(values + i);
    }
    this->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  };
  virtual void DebugInfo() const {}
};