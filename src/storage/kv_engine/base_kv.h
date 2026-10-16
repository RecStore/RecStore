#pragma once
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

using boost::coroutines2::coroutine;

// #define XMH_SIMPLE_MALLOC

struct BaseKVConfig {
  int num_threads_ = 0;
  json json_config_; // add your custom config in this field
};
/*
KVEngineComposite uses the nested configuration format:
{
  "engine_type": "KVEngineComposite",
  "capacity": 1000000,
  "index": {"type": "DRAM_EXTENDIBLE_HASH"},
  "value": {
    "type": "DRAM_VALUE_STORE",
    "path": "/tmp/recstore/value",
    "default_value_size_hint": 128,
    "dram_allocator": {
      "type": "PERSIST_LOOP_SLAB",
      "capacity_bytes": 128000000
    }
  }
}

ResolveEngine returns the matching factory name; engine_type defaults to
KVEngineComposite when omitted.
Field validation happens in each engine constructor. KVEngineComposite rejects
legacy top-level fields such as path/index_type/value_type and nested file_path.
*/

class BaseKV {
public:
  virtual ~BaseKV() {}

  explicit BaseKV(const BaseKVConfig& config){};

  virtual void Util() {
    std::cout << "BaseKV Util: no impl" << std::endl;
    return;
  }
  virtual std::string ExtraResultFields() const { return ""; }
  virtual void Get(const uint64_t key, std::string& value, unsigned tid) = 0;
  virtual bool Exists(const uint64_t key, unsigned tid) {
    std::string value;
    Get(key, value, tid);
    return !value.empty();
  }

  virtual void
  Put(const uint64_t key, const std::string_view& value, unsigned tid) = 0;

  virtual void BatchPut(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual void BatchPut(coroutine<void>::push_type& sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  };

  virtual uint64_t BatchDelete(base::ConstArray<uint64_t> keys, unsigned tid) {
    (void)keys;
    (void)tid;
    return 0;
  }

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) = 0;

  struct BatchGetFlatStats {
    std::uint64_t index_lookup_ns = 0;
    std::uint64_t zero_fill_ns    = 0;
    std::uint64_t row_copy_ns     = 0;
    std::uint64_t missing_rows    = 0;
  };

  struct BulkLoadStats {
    std::uint64_t prepare_ns         = 0;
    std::uint64_t value_alloc_ns     = 0;
    std::uint64_t handle_validate_ns = 0;
    std::uint64_t index_put_ns       = 0;
    std::uint64_t key_track_ns       = 0;
    std::uint64_t total_ns           = 0;
  };

  struct DirectFixedRow {
    const char* data = nullptr;
    size_t size      = 0;
    bool missing     = false;
  };

  struct RDMABackingRegion {
    char* data  = nullptr;
    size_t size = 0;
  };

  virtual bool BatchGetFlat(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid,
      BatchGetFlatStats* stats = nullptr) {
    return false;
  }

  virtual bool BatchGetFlatRange(
      base::ConstArray<uint64_t> keys,
      float* values,
      int64_t num_rows,
      int64_t row_dim,
      int64_t value_offset,
      int64_t value_dim,
      unsigned tid,
      BatchGetFlatStats* stats = nullptr,
      bool collect_profile = true) {
    return false;
  }

  virtual bool BatchGetIndexOnly(base::ConstArray<uint64_t> keys,
                                 unsigned tid,
                                 BatchGetFlatStats* stats = nullptr) {
    return false;
  }

  virtual bool BatchGetDirectFixedRows(
      base::ConstArray<uint64_t> keys,
      int64_t num_rows,
      int64_t embedding_dim,
      unsigned tid,
      std::vector<DirectFixedRow>* rows,
      BatchGetFlatStats* stats = nullptr) {
    return false;
  }

  virtual RDMABackingRegion GetRDMABackingRegion() const { return {}; }

  virtual void BatchGet(coroutine<void>::push_type& sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual bool ApplySgdUpdateFlat(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      uint8_t tag,
      unsigned tid) {
    return false;
  }

  virtual bool ApplySgdUpdateFlatRange(
      base::ConstArray<uint64_t> keys,
      const float* grads,
      int64_t num_rows,
      int64_t row_dim,
      int64_t update_offset,
      int64_t update_dim,
      float learning_rate,
      unsigned tid) {
    return false;
  }

  virtual void DebugInfo() const {}

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) {
    LOG(FATAL) << "not implemented";
  };

  virtual void BulkLoadRange(base::ConstArray<uint64_t> keys,
                             const void* value,
                             unsigned tid,
                             BulkLoadStats* stats = nullptr) {
    (void)tid;
    (void)stats;
    BulkLoad(keys, value);
  }

  virtual bool BulkLoadIndexedFloatRange(
      base::ConstArray<uint64_t> keys,
      const int64_t* row_indices,
      const float* values,
      int64_t num_rows,
      int64_t value_dim,
      unsigned tid,
      BulkLoadStats* stats = nullptr) {
    return false;
  }

  virtual void LoadFakeData(int64_t key_capacity, int value_size) {
    std::vector<uint64_t> keys;
    float* values = new float[value_size / sizeof(float) * key_capacity];
    keys.reserve(key_capacity);
    for (int64_t i = 0; i < key_capacity; i++) {
      keys.push_back(i);
    }
    this->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  };

  virtual void clear() {
    LOG(WARNING) << "clear() not fully implemented for this KV engine";
  };

  virtual bool
  SaveCheckpoint(const std::string& file, const std::string& metadata) {
    (void)file;
    (void)metadata;
    return false;
  }

  virtual bool LoadCheckpoint(const std::string& file,
                              const std::string& expected_metadata) {
    (void)file;
    (void)expected_metadata;
    return false;
  }

  virtual uint64_t CheckpointRecordCount() const { return 0; }

  virtual uint64_t ActiveKeyCount() const { return CheckpointRecordCount(); }

  virtual std::vector<uint64_t> SnapshotKeys() const { return {}; }

protected:
};
