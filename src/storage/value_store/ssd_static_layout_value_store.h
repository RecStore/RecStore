#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <array>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "base/factory.h"
#include "storage/io_backend/io_backend.h"
#include "storage/ssd_layout/ssd_layout.h"
#include "storage/value_store/value_store.h"

class SsdStaticLayoutValueStore : public ValueStore {
public:
  struct Stats {
    uint64_t logical_rows    = 0;
    uint64_t physical_copies = 0;
    uint64_t page_reads      = 0;
    uint64_t requested_rows  = 0;
    uint64_t effective_rows  = 0;
    uint64_t bytes_read      = 0;
  };

  explicit SsdStaticLayoutValueStore(const BaseKVConfig& config) {
    const auto& value               = config.json_config_.at("value");
    const auto& layout              = value.at("ssd_layout");
    const std::string strategy_name = layout.at("strategy").get<std::string>();
    const auto strategy             = ssd_layout::ParseStrategy(strategy_name);
    if (strategy != ssd_layout::Strategy::kBandana &&
        strategy != ssd_layout::Strategy::kMaxEmbed) {
      throw std::invalid_argument(
          "SSD_STATIC_LAYOUT_VALUE_STORE requires bandana or maxembed");
    }
    const uint32_t row_size = static_cast<uint32_t>(
        value.at("default_value_size_hint").get<uint64_t>());
    artifact_ =
        ssd_layout::LoadArtifact(layout.at("mapping_path").get<std::string>());
    ssd_layout::CheckArtifact(artifact_, strategy, row_size);
    if (artifact_.page_size != PAGE_SIZE) {
      throw std::invalid_argument("SSD layout page size must equal PAGE_SIZE");
    }
    for (uint32_t index = 0; index < artifact_.keys.size(); ++index) {
      key_to_row_.emplace(artifact_.keys[index], index);
    }
    for (const auto& row : artifact_.copies) {
      copy_count_ += row.size();
    }

    const json io =
        value.contains("io")
            ? value.at("io")
            : value.value("ssd_allocator", json::object())
                  .value("io", json::object());
    BaseKVConfig io_config = config;
    auto& json             = io_config.json_config_;
    const uint64_t base_offset =
        io.value("base_offset_bytes", static_cast<uint64_t>(0));
    if (base_offset % PAGE_SIZE != 0) {
      throw std::invalid_argument(
          "SSD layout base offset must be page aligned");
    }
    page_id_offset_               = base_offset / PAGE_SIZE;
    json["io_backend_type"]       = io.value("type", "IOURING");
    json["file_path"]             = value.at("path").get<std::string>();
    json["queue_cnt"]             = io.value("queue_depth", 512);
    json["completion_poll_spins"] = io.value("completion_poll_spins", 8192);
    json["page_id_offset"]        = page_id_offset_;
    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    io_backend_.reset(IOF::NewInstance(json.at("io_backend_type"), io_config));
    io_backend_->init();
  }

  ssd_layout::Strategy strategy() const { return artifact_.strategy; }
  uint64_t checksum() const { return artifact_.checksum; }
  Stats stats() const {
    Stats out;
    out.logical_rows    = artifact_.keys.size();
    out.physical_copies = copy_count_;
    out.page_reads      = page_reads_.load(std::memory_order_relaxed);
    out.requested_rows  = requested_rows_.load(std::memory_order_relaxed);
    out.effective_rows  = effective_rows_.load(std::memory_order_relaxed);
    out.bytes_read      = bytes_read_.load(std::memory_order_relaxed);
    return out;
  }

  uint64_t Alloc(size_t) override { return RejectAlloc(); }
  uint64_t AllocAndWrite(const void*, size_t) override { return RejectAlloc(); }
  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteSpec>&) override {
    throw std::invalid_argument(
        "SSD_STATIC_LAYOUT_VALUE_STORE placement is fixed");
  }
  uint64_t AllocAndWrite(uint64_t key, const void* data, size_t size) override {
    const uint64_t handle = HandleForKey(key);
    Write(handle, data, size);
    return handle;
  }
  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<uint64_t>& keys,
                     const std::vector<WriteSpec>& specs) override {
    if (keys.size() != specs.size()) {
      throw std::invalid_argument("static SSD layout batch size mismatch");
    }
    std::vector<uint64_t> handles;
    handles.reserve(keys.size());
    for (uint64_t key : keys) {
      handles.push_back(HandleForKey(key));
    }
    BatchWrite(handles, specs);
    return handles;
  }
  ValueStore::LayoutId GetLayoutId() const override {
    return {static_cast<uint32_t>(artifact_.strategy), artifact_.checksum};
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    const auto& copies = CopiesOf(handle);
    if (size != artifact_.row_size) {
      throw std::invalid_argument("static SSD layout value size mismatch");
    }
    std::vector<uint64_t> pages;
    pages.reserve(copies.size());
    for (const auto& copy : copies) {
      pages.push_back(copy.page_id);
    }
    auto locked = LockPages(pages, true);
    for (const auto& copy : copies) {
      UpdatePage(copy, data, size);
    }
  }

  void BatchWrite(const std::vector<uint64_t>& handles,
                  const std::vector<WriteSpec>& specs) override {
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("static SSD layout batch size mismatch");
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      Write(handles[i], specs[i].data, specs[i].size);
    }
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    const auto& copies = CopiesOf(handle);
    if (buf_size < artifact_.row_size) {
      throw std::invalid_argument("static SSD layout read buffer is too small");
    }
    const auto& copy = copies.front();
    auto locked      = LockPages({copy.page_id}, false);
    ReadPage(copy, out_buf);
    requested_rows_.fetch_add(1, std::memory_order_relaxed);
    effective_rows_.fetch_add(1, std::memory_order_relaxed);
    return artifact_.row_size;
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& out_results) override {
    out_results.resize(handles.size());
    std::vector<char> done(handles.size(), 1);
    std::vector<const std::vector<ssd_layout::Location>*> copies(
        handles.size());
    size_t remaining = 0;
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
        continue;
      }
      copies[i] = &CopiesOf(handles[i]);
      done[i]   = 0;
      ++remaining;
      out_results[i].data.resize(artifact_.row_size);
    }
    requested_rows_.fetch_add(remaining, std::memory_order_relaxed);
    // ponytail: rescan candidate pages each round. Invert page->request if a
    // batch profile shows this scan above the SSD read.
    while (remaining > 0) {
      std::unordered_map<uint64_t, uint32_t> cover;
      for (size_t i = 0; i < done.size(); ++i) {
        if (done[i]) {
          continue;
        }
        for (const auto& copy : *copies[i]) {
          ++cover[copy.page_id];
        }
      }
      uint64_t best_page  = UINT64_MAX;
      uint32_t best_cover = 0;
      for (const auto& item : cover) {
        if (item.second > best_cover ||
            (item.second == best_cover && item.first < best_page)) {
          best_page  = item.first;
          best_cover = item.second;
        }
      }
      auto locked  = LockPages({best_page}, false);
      char* page   = ReadRawPage(best_page, true);
      uint32_t hit = 0;
      for (size_t i = 0; i < done.size(); ++i) {
        if (done[i]) {
          continue;
        }
        for (const auto& copy : *copies[i]) {
          if (copy.page_id != best_page) {
            continue;
          }
          std::memcpy(out_results[i].data.data(),
                      page + copy.byte_offset,
                      artifact_.row_size);
          done[i] = 1;
          --remaining;
          ++hit;
          break;
        }
      }
      io_backend_->FreeBuffer(page);
      effective_rows_.fetch_add(hit, std::memory_order_relaxed);
    }
  }

  void Free(uint64_t) override {}
  size_t SlotCapacity(uint64_t handle) const override {
    CopiesOf(handle);
    return artifact_.row_size;
  }

  void ResetStats() override {
    io_backend_->ResetReadStats();
    page_reads_.store(0, std::memory_order_relaxed);
    requested_rows_.store(0, std::memory_order_relaxed);
    effective_rows_.store(0, std::memory_order_relaxed);
    bytes_read_.store(0, std::memory_order_relaxed);
  }

  std::string ExtraResultFields() const override {
    const Stats snapshot = stats();
    const double rows_per_page =
        snapshot.page_reads == 0
            ? 0.0
            : static_cast<double>(snapshot.effective_rows) /
                  snapshot.page_reads;
    std::ostringstream out;
    out << " layout_strategy=" << ssd_layout::StrategyName(artifact_.strategy)
        << " layout_logical_rows=" << snapshot.logical_rows
        << " layout_physical_copies=" << snapshot.physical_copies
        << " layout_page_reads=" << snapshot.page_reads
        << " layout_requested_rows=" << snapshot.requested_rows
        << " layout_effective_rows=" << snapshot.effective_rows
        << " layout_rows_per_page=" << rows_per_page
        << " layout_bytes_read=" << snapshot.bytes_read;
    return out.str();
  }

private:
  struct PageLock {
    std::vector<std::shared_mutex*> mutexes;
    bool exclusive                       = false;
    PageLock()                           = default;
    PageLock(const PageLock&)            = delete;
    PageLock& operator=(const PageLock&) = delete;
    PageLock(PageLock&& other) noexcept : exclusive(other.exclusive) {
      mutexes.swap(other.mutexes);
    }
    ~PageLock() {
      for (auto it = mutexes.rbegin(); it != mutexes.rend(); ++it) {
        if (exclusive) {
          (*it)->unlock();
        } else {
          (*it)->unlock_shared();
        }
      }
    }
  };

  static uint64_t RejectAlloc() {
    throw std::invalid_argument(
        "SSD_STATIC_LAYOUT_VALUE_STORE placement is fixed");
  }

  uint64_t HandleForKey(uint64_t key) const {
    if (key > ssd_layout::kMaxKey) {
      throw std::invalid_argument("static SSD layout key is out of range");
    }
    return key + 1;
  }

  uint64_t DecodeKey(uint64_t handle) const {
    if (handle == kValueHandleNone || handle > ssd_layout::kMaxKey + 1) {
      throw std::invalid_argument("static SSD layout handle is invalid");
    }
    return handle - 1;
  }

  const std::vector<ssd_layout::Location>& CopiesOf(uint64_t handle) const {
    const auto found = key_to_row_.find(DecodeKey(handle));
    if (found == key_to_row_.end()) {
      throw std::invalid_argument("static SSD layout key is not mapped");
    }
    return artifact_.copies[found->second];
  }

  PageLock LockPages(std::vector<uint64_t> pages, bool exclusive) {
    std::vector<std::shared_mutex*> mutexes;
    mutexes.reserve(pages.size());
    for (uint64_t page : pages) {
      mutexes.push_back(&page_locks_[page % page_locks_.size()]);
    }
    std::sort(mutexes.begin(), mutexes.end());
    mutexes.erase(std::unique(mutexes.begin(), mutexes.end()), mutexes.end());
    PageLock locked;
    locked.exclusive = exclusive;
    locked.mutexes   = mutexes;
    for (auto* mutex : locked.mutexes) {
      if (exclusive) {
        mutex->lock();
      } else {
        mutex->lock_shared();
      }
    }
    return locked;
  }

  char* ReadRawPage(uint64_t page_id, bool count_read) {
    char* buffer = io_backend_->AllocateBuffer();
    io_backend_->BatchReadPages({{page_id + page_id_offset_, buffer, 1}});
    if (count_read) {
      page_reads_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(PAGE_SIZE, std::memory_order_relaxed);
    }
    return buffer;
  }

  void ReadPage(const ssd_layout::Location& copy, void* out) {
    char* page = ReadRawPage(copy.page_id, true);
    std::memcpy(out, page + copy.byte_offset, artifact_.row_size);
    io_backend_->FreeBuffer(page);
  }

  void
  UpdatePage(const ssd_layout::Location& copy, const void* data, size_t size) {
    char* page = ReadRawPage(copy.page_id, false);
    std::memcpy(page + copy.byte_offset, data, size);
    io_backend_->BatchWritePages({{copy.page_id + page_id_offset_, page, 1}});
    io_backend_->FreeBuffer(page);
  }

  ssd_layout::Artifact artifact_;
  std::unordered_map<uint64_t, uint32_t> key_to_row_;
  std::unique_ptr<IOBackend> io_backend_;
  uint64_t page_id_offset_ = 0;
  uint64_t copy_count_     = 0;
  std::array<std::shared_mutex, 64> page_locks_;
  std::atomic<uint64_t> page_reads_{0};
  std::atomic<uint64_t> requested_rows_{0};
  std::atomic<uint64_t> effective_rows_{0};
  std::atomic<uint64_t> bytes_read_{0};
};

FACTORY_REGISTER(ValueStore,
                 SSD_STATIC_LAYOUT_VALUE_STORE,
                 SsdStaticLayoutValueStore,
                 const BaseKVConfig&);
