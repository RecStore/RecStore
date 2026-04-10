#pragma once

#include <sys/user.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "../nvm/pet_kv/shm_common.h"
#include "../index/ssd_cceh/CCEH.h"
#include "base/factory.h"
#include "base_kv.h"
#include "../io_backend/io_backend.h"

static inline Value_t
PackPageValue(PageID_t start_page_id, uint16_t page_count) {
  return (uint64_t(page_count) << 48) | (start_page_id & 0x0000FFFFFFFFFFFF);
}
static inline void
UnpackPageValue(Value_t v, PageID_t& start_page_id, uint16_t& page_count) {
  page_count    = uint16_t(v >> 48);
  start_page_id = v & 0x0000FFFFFFFFFFFF;
}

class KVEngineCCEH : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineCCEH(const BaseKVConfig& config) : BaseKV(config) {
    queue_cnt_ = config.json_config_.at("queue_cnt").get<int>();
    std::string io_backend_type =
        config.json_config_.at("io_backend_type").get<std::string>();
    LOG(INFO) << "--------------init KVEngineCCEH--------------------";
    std::string index_path = config.json_config_.at("path").get<std::string>();
    std::string index_db_path = index_path + "/cceh_test.db";
    std::string value_path = config.json_config_.at("path").get<std::string>();
    std::string value_db_path = value_path + "/cceh_value.db";

    // Per-table LBA offsets for SPDK (raw NVMe device shared by all tables).
    // json_config_ may carry "spdk_index_offset" and "spdk_value_offset" set
    // by CCEHKVClientImpl::init_data to carve out non-overlapping zones.
    // For io_uring these are ignored (each table uses its own file).
    PageID_t index_offset = 0;
    PageID_t value_offset = 1; // io_uring: start at page 1 (page 0 unused)
    if (io_backend_type == "SPDK") {
      index_offset =
          config.json_config_.value("spdk_index_offset", (uint64_t)0);
      value_offset =
          config.json_config_.value("spdk_value_offset", (uint64_t)1000000);
    }
    BaseKVConfig index_config                   = config;
    BaseKVConfig value_config                   = config;
    index_config.json_config_["file_path"]      = index_db_path;
    value_config.json_config_["file_path"]      = value_db_path;
    index_config.json_config_["page_id_offset"] = index_offset;
    value_config.json_config_["page_id_offset"] = value_offset;

    hash_table_ = new CCEH(index_config);
    using IOF   = base::Factory<IOBackend, const BaseKVConfig&>;
    value_io_backend.reset(IOF::NewInstance(io_backend_type, value_config));
    value_io_backend->init();
    LOG(INFO) << "After init value and  index io_backend ";
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    Value_t v;
    hash_table_->Get(key, v, tid);
    if (v == NONE) {
      value = std::string();
      return;
    }
    PageID_t start_page_id;
    uint16_t page_count;
    UnpackPageValue(v, start_page_id, page_count);

    uint32_t actual_size = 0;
    uint32_t written     = 0;
    for (int i = 0; i < page_count; i++) {
      PageID_t pid = start_page_id + i;
      char* buffer = (char*)value_io_backend->GetPage(pid);
      if (i == 0) {
        memcpy(&actual_size, buffer, 4);
        value.resize(actual_size);
        uint32_t data_in_page =
            std::min((uint32_t)(PAGE_SIZE - 4), actual_size);
        memcpy(value.data(), buffer + 4, data_in_page);
        written += data_in_page;
      } else {
        uint32_t data_in_page =
            std::min((uint32_t)PAGE_SIZE, actual_size - written);
        memcpy(value.data() + written, buffer, data_in_page);
        written += data_in_page;
      }
      value_io_backend->Unpin(pid, buffer, false);
    }
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    uint32_t actual_size   = (uint32_t)value.size();
    uint16_t pages_needed  = (actual_size + 4 + PAGE_SIZE - 1) / PAGE_SIZE;
    PageID_t start_page_id = 0;
    uint32_t written       = 0;
    for (int i = 0; i < pages_needed; i++) {
      PageID_t page_id = value_io_backend->AllocatePage();
      if (i == 0)
        start_page_id = page_id;
      char* buffer = (char*)value_io_backend->GetPage(page_id);
      if (i == 0) {
        memcpy(buffer, &actual_size, 4);
        uint32_t data_in_page =
            std::min((uint32_t)(PAGE_SIZE - 4), actual_size);
        memcpy(buffer + 4, value.data(), data_in_page);
        written += data_in_page;
      } else {
        uint32_t data_in_page =
            std::min((uint32_t)PAGE_SIZE, actual_size - written);
        memcpy(buffer, value.data() + written, data_in_page);
        written += data_in_page;
      }
      value_io_backend->Unpin(page_id, buffer, true);
    }
    Key_t hash_key = key;
    hash_table_->Put(hash_key, PackPageValue(start_page_id, pages_needed), tid);
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
    // Collect valid keys and prepare read buffers
    std::vector<IOBackend::IOEntry> read_entries;
    std::vector<int> valid_indices; // which vals[] entries are valid
    read_entries.reserve(size);
    valid_indices.reserve(size);
    // Unpack each valid val to get (start_page_id, page_count)
    std::vector<uint16_t> entry_page_counts;
    entry_page_counts.reserve(size);
    for (int i = 0; i < size; i++) {
      if (vals[i] != NONE) {
        PageID_t start_page_id;
        uint16_t page_count;
        UnpackPageValue(vals[i], start_page_id, page_count);
        char* buffer = value_io_backend->AllocateBuffer(page_count);
        read_entries.push_back({start_page_id, buffer, page_count});
        entry_page_counts.push_back(page_count);
        valid_indices.push_back(i);
      }
    }
    // Batch read all values at once
    value_io_backend->BatchReadPages(read_entries);

    // Copy results into output
    static thread_local std::vector<std::vector<float>> value_buffers;
    value_buffers.clear();
    value_buffers.reserve(size);
    int ri = 0; // index into read_entries
    for (int i = 0; i < size; i++) {
      if (ri < (int)valid_indices.size() && valid_indices[ri] == i) {
        char* buf            = read_entries[ri].buffer;
        uint32_t actual_size = 0;
        memcpy(&actual_size, buf, 4);
        value_buffers.emplace_back(actual_size / sizeof(float));
        memcpy((char*)value_buffers.back().data(), buf + 4, actual_size);
        value_io_backend->FreeBuffer(buf);
        values->emplace_back(
            value_buffers.back().data(), value_buffers.back().size());
        ri++;
      } else {
        values->emplace_back();
        value_buffers.emplace_back();
      }
    }
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    int n = keys.Size();

    // Phase 1: compute per-key page counts and allocate page IDs
    std::vector<PageID_t> start_page_ids(n, NONE);
    std::vector<uint16_t> page_counts(n);
    for (int i = 0; i < n; i++) {
      uint32_t actual_size = (*values)[i].Size() * sizeof(float);
      page_counts[i]       = (actual_size + 4 + PAGE_SIZE - 1) / PAGE_SIZE;
      start_page_ids[i]    = value_io_backend->GetNextPageID();
      value_io_backend->SetNextPageID(start_page_ids[i] + page_counts[i]);
    }

    // Phase 2: prepare per-key buffers with 4-byte header and batch write
    std::vector<IOBackend::IOEntry> write_entries;
    write_entries.reserve(n);
    for (int j = 0; j < n; j++) {
      uint32_t actual_size = (*values)[j].Size() * sizeof(float);
      char* buffer         = value_io_backend->AllocateBuffer(page_counts[j]);
      memcpy(buffer, &actual_size, 4);
      memcpy(buffer + 4, (const char*)(*values)[j].Data(), actual_size);
      write_entries.push_back({start_page_ids[j], buffer, page_counts[j]});
    }
    value_io_backend->BatchWritePages(write_entries);
    for (auto& e : write_entries) {
      value_io_backend->FreeBuffer(e.buffer);
    }

    // Phase 3: CCEH index inserts with packed value
    for (int j = 0; j < n; j++) {
      Key_t hash_key = keys[j];
      hash_table_->Put(
          hash_key, PackPageValue(start_page_ids[j], page_counts[j]), tid);
    }
  }

  ~KVEngineCCEH() {
    std::cout << "exit KVEngineCCEH" << std::endl;
    if (hash_table_) {
      delete hash_table_;
      hash_table_ = nullptr;
    }
  }

private:
  CCEH* hash_table_;
  std::string dict_pool_name_;
  int queue_cnt_;
  std::string type;
  std::unique_ptr<IOBackend> value_io_backend;
};

FACTORY_REGISTER(BaseKV, KVEngineCCEH, KVEngineCCEH, const BaseKVConfig&);
