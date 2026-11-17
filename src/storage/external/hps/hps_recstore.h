#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "third_party/HugeCTR/HugeCTR/include/hps/database_backend.hpp"

namespace recstore {
namespace storage {

struct HpsRecStoreBackendParams : public HugeCTR::VolatileBackendParams {
  std::string path;
  std::string index_type{"DRAM_EXTENDIBLE_HASH"};
  std::string value_store_type{"DRAM_VALUE_STORE"};
  std::string dram_allocator{"PERSIST_LOOP_SLAB"};
  uint64_t capacity{0};
  uint32_t value_size{0};
  uint64_t dram_capacity_bytes{0};
  uint64_t ssd_capacity_bytes{0};
  double tiered_high_watermark_ratio{0.0};
  std::string ssd_io_backend{"IOURING"};
  std::string ssd_value_file;
  int ssd_queue_depth{512};
  int num_threads{1};
};

template <typename Key>
class HpsRecStoreBackend final
    : public HugeCTR::VolatileBackend<Key, HpsRecStoreBackendParams> {
public:
  using Base = HugeCTR::VolatileBackend<Key, HpsRecStoreBackendParams>;

  explicit HpsRecStoreBackend(const HpsRecStoreBackendParams& params)
      : Base(params) {
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::BackendFromUseDram(
            params.value_store_type == "DRAM_VALUE_STORE" ||
            params.value_store_type == "TIERED_VALUE_STORE");

    BaseKVConfig cfg;
    cfg.num_threads_ = params.num_threads;
    for (unsigned tid = 0;
         tid < static_cast<unsigned>(std::max(1, params.num_threads));
         ++tid) {
      free_thread_slots_.push_back(tid);
    }
    const uint64_t value_size =
        params.value_size > 0 ? params.value_size : sizeof(float);
    fixed_row_bytes_ = value_size;
    const uint64_t dram_capacity =
        params.dram_capacity_bytes > 0
            ? params.dram_capacity_bytes
            : std::max<uint64_t>(1, params.capacity) * value_size * 6 / 5;
    const uint64_t ssd_slot_size = SsdSlotSize(value_size);
    const uint64_t ssd_capacity =
        params.ssd_capacity_bytes > 0
            ? params.ssd_capacity_bytes
            : std::max<uint64_t>(1, params.capacity) * ssd_slot_size * 6 / 5;
    const std::string ssd_value_file =
        params.ssd_value_file.empty()
            ? params.path + "_value.db"
            : params.ssd_value_file;
    cfg.json_config_ = {
        {"capacity", params.capacity},
        {"index", {{"type", params.index_type}}},
        {"value",
         {{"type", params.value_store_type},
          {"default_value_size_hint", value_size}}}};

    auto& value_config = cfg.json_config_["value"];
    if (params.value_store_type == "DRAM_VALUE_STORE" ||
        params.value_store_type == "TIERED_VALUE_STORE") {
      value_config["dram_allocator"] = {
          {"type", params.dram_allocator}, {"capacity_bytes", dram_capacity}};
      if (params.value_store_type == "DRAM_VALUE_STORE") {
        value_config["path"] = params.path;
      } else {
        value_config["dram_allocator"]["path"] = TieredDramPath(params.path);
      }
    }
    if (params.value_store_type == "SSD_VALUE_STORE" ||
        params.value_store_type == "TIERED_VALUE_STORE") {
      value_config["ssd_allocator"] = {
          {"type", "SSD_SLAB"},
          {"capacity_bytes", ssd_capacity},
          {"min_block_size", 128},
          {"max_block_size", 4096},
          {"io",
           {{"type", params.ssd_io_backend},
            {"queue_depth", params.ssd_queue_depth},
            {"base_offset_bytes", 4096}}}};
      if (params.value_store_type == "SSD_VALUE_STORE") {
        value_config["path"] = ssd_value_file;
      } else {
        value_config["ssd_allocator"]["path"] = ssd_value_file;
      }
    }
    if (params.value_store_type == "TIERED_VALUE_STORE") {
      value_config["tiering"] = {{"cache_policy", "LRU"}};
      if (params.tiered_high_watermark_ratio > 0.0) {
        value_config["tiering"]["high_watermark_ratio"] =
            params.tiered_high_watermark_ratio;
      }
    }

    auto resolved = base::ResolveEngine(cfg);
    kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        resolved.engine, resolved.cfg));
    if (!kv_) {
      throw std::runtime_error("failed to create RecStore BaseKV for HPS");
    }
  }

  const char* get_name() const override { return "RecStoreBackend"; }
  bool is_shared() const override { return false; }

  std::string extra_result_fields() const {
    return kv_ ? kv_->ExtraResultFields() : "";
  }

  size_t size(const std::string& table_name) const override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    const auto it = table_sizes_.find(table_name);
    return it == table_sizes_.end() ? 0 : it->second;
  }

  size_t contains(const std::string& table_name,
                  size_t num_keys,
                  const Key* keys,
                  const std::chrono::nanoseconds& time_budget) const override {
    (void)table_name;
    (void)time_budget;
    size_t hits = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (kv_->Exists(static_cast<uint64_t>(keys[i]), 0)) {
        ++hits;
      }
    }
    return hits;
  }

  size_t insert(const std::string& table_name,
                size_t num_pairs,
                const Key* keys,
                const char* values,
                uint32_t value_size,
                size_t value_stride) override {
    if (num_pairs == 0) {
      return 0;
    }
    if (value_size > value_stride) {
      throw std::invalid_argument(
          "HpsRecStoreBackend insert requires value_size <= value_stride");
    }
    std::unique_lock<std::shared_mutex> rows_lock(rows_mu_);
    if (value_size != fixed_row_bytes_) {
      fixed_rows_only_.store(false, std::memory_order_release);
    }
    auto& scratch = Scratch::Get();
    scratch.keys.resize(num_pairs);
    const size_t floats_per_value =
        (static_cast<size_t>(value_size) + sizeof(float) - 1) / sizeof(float);
    scratch.row_views.clear();
    scratch.row_views.reserve(num_pairs);
    if (value_size % sizeof(float) == 0) {
      scratch.insert_values.clear();
      for (size_t i = 0; i < num_pairs; ++i) {
        scratch.keys[i] = static_cast<uint64_t>(keys[i]);
        scratch.row_views.emplace_back(
            reinterpret_cast<const float*>(values + i * value_stride),
            floats_per_value);
      }
    } else {
      scratch.insert_values.assign(num_pairs * floats_per_value, 0.0f);
      for (size_t i = 0; i < num_pairs; ++i) {
        scratch.keys[i] = static_cast<uint64_t>(keys[i]);
        float* row      = scratch.insert_values.data() + i * floats_per_value;
        std::memcpy(row, values + i * value_stride, value_size);
        scratch.row_views.emplace_back(row, floats_per_value);
      }
    }
    ThreadSlot slot(*this);
    kv_->BatchPut(
        base::ConstArray<uint64_t>(scratch.keys), &scratch.row_views, slot.tid);
    {
      std::lock_guard<std::mutex> lock(table_sizes_mu_);
      table_sizes_[table_name] += num_pairs;
    }
    return num_pairs;
  }

  size_t fetch(const std::string& table_name,
               size_t num_keys,
               const Key* keys,
               char* values,
               size_t value_stride,
               const HugeCTR::DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override {
    return fetch_with_stats(
        table_name,
        num_keys,
        keys,
        values,
        value_stride,
        on_miss,
        time_budget,
        nullptr);
  }

  size_t fetch_with_stats(
      const std::string& table_name,
      size_t num_keys,
      const Key* keys,
      char* values,
      size_t value_stride,
      const HugeCTR::DatabaseMissCallback& on_miss,
      const std::chrono::nanoseconds& time_budget,
      BaseKV::BatchGetFlatStats* stats) {
    if (stats != nullptr)
      *stats = BaseKV::BatchGetFlatStats{};
    (void)table_name;
    (void)time_budget;
    if (num_keys == 0)
      return 0;
    auto& scratch = Scratch::Get();
    scratch.keys.resize(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
      scratch.keys[i] = static_cast<uint64_t>(keys[i]);
    }
    std::shared_lock<std::shared_mutex> rows_lock(rows_mu_);
    std::vector<size_t> misses;
    if (value_stride == fixed_row_bytes_ && value_stride % sizeof(float) == 0 &&
        reinterpret_cast<uintptr_t>(values) % alignof(float) == 0) {
      scratch.found.resize(num_keys);
      if (fixed_rows_only_.load(std::memory_order_acquire) &&
          kv_->BatchGetFlat(
              base::ConstArray<uint64_t>(scratch.keys),
              reinterpret_cast<float*>(values),
              num_keys,
              value_stride / sizeof(float),
              0,
              stats,
              scratch.found.data())) {
        size_t hits = 0;
        for (size_t i = 0; i < num_keys; ++i) {
          if (scratch.found[i])
            ++hits;
          else
            misses.push_back(i);
        }
        // Callbacks may reenter this backend and mutate its TLS scratch.
        rows_lock.unlock();
        for (size_t row : misses)
          on_miss(row);
        return hits;
      }
    }
    if (stats != nullptr)
      *stats = BaseKV::BatchGetFlatStats{};
    scratch.fetched_rows.clear();
    kv_->BatchGet(
        base::ConstArray<uint64_t>(scratch.keys), &scratch.fetched_rows, 0);

    const size_t hits = CopyRowsToOutput(
        scratch.fetched_rows, num_keys, values, value_stride, [&](size_t row) {
          misses.push_back(row);
        });
    rows_lock.unlock();
    for (size_t row : misses)
      on_miss(row);
    return hits;
  }

  size_t
  fetch(const std::string& table_name,
        size_t num_indices,
        const size_t* indices,
        const Key* keys,
        char* values,
        size_t value_stride,
        const HugeCTR::DatabaseMissCallback& on_miss,
        const std::chrono::nanoseconds& time_budget) override {
    (void)table_name;
    (void)time_budget;
    if (num_indices == 0) {
      return 0;
    }
    std::shared_lock<std::shared_mutex> rows_lock(rows_mu_);
    auto& scratch = Scratch::Get();
    scratch.keys.resize(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
      scratch.keys[i] = static_cast<uint64_t>(keys[indices[i]]);
    }
    scratch.fetched_rows.clear();
    kv_->BatchGet(
        base::ConstArray<uint64_t>(scratch.keys), &scratch.fetched_rows, 0);

    size_t hits = 0;
    std::vector<size_t> misses;
    for (size_t i = 0; i < num_indices; ++i) {
      const size_t output_index = indices[i];
      if (i < scratch.fetched_rows.size() &&
          scratch.fetched_rows[i].Data() != nullptr &&
          scratch.fetched_rows[i].Size() > 0) {
        CopyOneRow(scratch.fetched_rows[i],
                   values + output_index * value_stride,
                   value_stride);
        ++hits;
      } else {
        misses.push_back(output_index);
      }
    }
    rows_lock.unlock();
    for (size_t row : misses)
      on_miss(row);
    return hits;
  }

  size_t evict(const std::string& table_name) override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    table_sizes_.erase(table_name);
    return 0;
  }

  size_t evict(const std::string& table_name,
               size_t num_keys,
               const Key* keys) override {
    (void)table_name;
    (void)num_keys;
    (void)keys;
    return 0;
  }

  std::vector<std::string> find_tables(const std::string& model_name) override {
    std::lock_guard<std::mutex> lock(table_sizes_mu_);
    std::vector<std::string> tables;
    for (const auto& [name, _] : table_sizes_) {
      if (name.find(model_name) != std::string::npos) {
        tables.push_back(name);
      }
    }
    return tables;
  }

  size_t dump_bin(const std::string& table_name, std::ofstream& file) override {
    (void)table_name;
    (void)file;
    return 0;
  }

#ifdef HCTR_USE_ROCKS_DB
  size_t dump_sst(const std::string& table_name,
                  rocksdb::SstFileWriter& file) override {
    (void)table_name;
    (void)file;
    return 0;
  }
#endif

private:
  struct ThreadSlot {
    HpsRecStoreBackend& owner;
    unsigned tid;

    explicit ThreadSlot(HpsRecStoreBackend& backend) : owner(backend) {
      std::unique_lock<std::mutex> lock(owner.thread_slots_mu_);
      owner.thread_slots_cv_.wait(lock, [&] {
        return !owner.free_thread_slots_.empty();
      });
      tid = owner.free_thread_slots_.back();
      owner.free_thread_slots_.pop_back();
    }
    ~ThreadSlot() {
      {
        std::lock_guard<std::mutex> lock(owner.thread_slots_mu_);
        owner.free_thread_slots_.push_back(tid);
      }
      owner.thread_slots_cv_.notify_one();
    }
    ThreadSlot(const ThreadSlot&)            = delete;
    ThreadSlot& operator=(const ThreadSlot&) = delete;
  };

  std::mutex thread_slots_mu_;
  std::condition_variable thread_slots_cv_;
  std::vector<unsigned> free_thread_slots_;

  struct Scratch {
    std::vector<uint64_t> keys;
    std::vector<float> insert_values;
    std::vector<base::ConstArray<float>> row_views;
    std::vector<base::ConstArray<float>> fetched_rows;
    std::vector<uint8_t> found;

    static Scratch& Get() {
      thread_local Scratch scratch;
      return scratch;
    }
  };

  static void CopyOneRow(
      const base::ConstArray<float>& row, char* dst, size_t value_stride) {
    const size_t bytes =
        std::min(value_stride, static_cast<size_t>(row.Size()) * sizeof(float));
    if (bytes > 0) {
      std::memcpy(dst, row.Data(), bytes);
    }
    if (bytes < value_stride) {
      std::memset(dst + bytes, 0, value_stride - bytes);
    }
  }

  static size_t CopyRowsToOutput(
      const std::vector<base::ConstArray<float>>& rows,
      size_t num_keys,
      char* values,
      size_t value_stride,
      const HugeCTR::DatabaseMissCallback& on_miss) {
    size_t hits = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (i < rows.size() && rows[i].Data() != nullptr && rows[i].Size() > 0) {
        CopyOneRow(rows[i], values + i * value_stride, value_stride);
        ++hits;
      } else {
        on_miss(i);
      }
    }
    return hits;
  }

  static uint64_t SsdSlotSize(uint64_t value_size) {
    const uint64_t needed = value_size + 8;
    uint64_t slot         = 128;
    while (slot < needed && slot < 4096) {
      slot <<= 1;
    }
    if (slot < needed) {
      throw std::invalid_argument(
          "HpsRecStoreBackend SSD value_size exceeds max SSD block size");
    }
    return slot;
  }

  static std::string TieredDramPath(const std::string& path) {
    const std::string suffix = std::to_string(std::hash<std::string>{}(path));
    return "/dev/shm/recstore_hps_tiered_" + suffix + "_dram";
  }

  std::unique_ptr<BaseKV> kv_;
  size_t fixed_row_bytes_ = 0;
  std::shared_mutex rows_mu_;
  std::atomic<bool> fixed_rows_only_{true};
  mutable std::mutex table_sizes_mu_;
  std::unordered_map<std::string, size_t> table_sizes_;
};

} // namespace storage
} // namespace recstore
