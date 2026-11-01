#pragma once
#include <atomic>
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

  virtual void Get(Key_t key, Value_t& pointer) {
    Get(key, pointer, CurrentTid());
  }
  virtual void Get(Key_t key, Value_t& pointer, unsigned tid) = 0;
  virtual void GetAsync(coroutine<void>::push_type& sink,
                        int index,
                        Key_t key,
                        Value_t& pointer) {
    Get(sink, index, key, pointer, CurrentTid());
  }
  virtual void
  Get(coroutine<void>::push_type& sink,
      int index,
      Key_t key,
      Value_t& pointer,
      unsigned tid) {
    LOG(FATAL) << "not implemented";
  }
  virtual Value_t Put(Key_t key, Value_t pointer) {
    return Put(key, pointer, CurrentTid());
  }
  virtual Value_t Put(Key_t key, Value_t pointer, unsigned tid) = 0;
  virtual Value_t
  Put(coroutine<void>::push_type& sink,
      int index,
      Key_t key,
      Value_t pointer,
      unsigned tid) {
    LOG(FATAL) << "not implemented";
    return NONE;
  }
  virtual void BatchPut(base::ConstArray<Key_t> keys, Value_t* pointers) {
    BatchPut(keys, pointers, CurrentTid());
  }
  virtual void
  BatchPut(base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) = 0;
  virtual void BatchPutWithOld(base::ConstArray<Key_t> keys,
                               Value_t* pointers,
                               Value_t* old_pointers,
                               unsigned tid) {
    for (int i = 0; i < keys.Size(); ++i)
      old_pointers[i] = Put(keys[i], pointers[i], tid);
  }
  virtual void BatchGet(base::ConstArray<Key_t> keys, Value_t* pointers) {
    BatchGet(keys, pointers, CurrentTid());
  }
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

protected:
  static unsigned CurrentTid() {
    static std::atomic<unsigned> counter{0};
    thread_local unsigned tid = counter.fetch_add(1, std::memory_order_relaxed);
    return tid;
  }
};

static constexpr Value_t kValueHandleNone = NONE;
