#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/factory.h"
#include "storage/io_backend/io_backend.h"
#include "storage/value_store/ssd_value_store.h"

// Append-only update log in front of the in-place SSD store.
//
// A 128 B row update in the in-place store costs a 4 KiB page read plus a
// 4 KiB page write. The log turns it into a ~144 B append: entries are packed
// into 4 KiB pages and each page is written once, so one page write covers
// ~28 updates and no page read is needed. Reads check the log index first and
// fall back to the in-place store for rows never updated.
//
// Every entry is self-describing (handle + size), so a stale or reused log
// slot is rejected instead of returning another row's bytes. A read that hits
// the page still being appended flushes that page first; a read that races a
// fallback write revalidates the index and retries against the base store.
//
// Stage 1 has no segment recycling: when the configured capacity is exhausted,
// later updates fall back to the in-place write path and their log entry is
// invalidated. Segment compaction is the next stage.
class SsdLogValueStore : public ValueStore {
public:
  explicit SsdLogValueStore(const BaseKVConfig& config) : base_(config) {
    const auto& value      = config.json_config_.at("value");
    const auto& log        = value.at("write_log");
    const std::string path = log.at("path").get<std::string>();
    if (path.empty()) {
      throw std::invalid_argument("write_log.path must not be empty");
    }
    capacity_bytes_ = log.at("capacity_bytes").get<uint64_t>();
    segment_bytes_  = log.value("segment_bytes", uint64_t{8} << 20);
    if (capacity_bytes_ < PAGE_SIZE || segment_bytes_ < PAGE_SIZE ||
        segment_bytes_ % PAGE_SIZE != 0 ||
        capacity_bytes_ % segment_bytes_ != 0) {
      throw std::invalid_argument(
          "write_log capacity and segment bytes must be page aligned and "
          "capacity must be a multiple of the segment size");
    }
    segment_pages_ = segment_bytes_ / PAGE_SIZE;
    segment_count_ = capacity_bytes_ / segment_bytes_;

    const json io          = log.contains("io") ? log.at("io") : json::object();
    BaseKVConfig io_config = config;
    auto& io_json          = io_config.json_config_;
    io_json["io_backend_type"]       = io.value("type", "IOURING");
    io_json["file_path"]             = path;
    io_json["queue_cnt"]             = io.value("queue_depth", 512);
    io_json["completion_poll_spins"] = io.value("completion_poll_spins", 8192);
    io_json["page_id_offset"]        = 0;
    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    io_.reset(IOF::NewInstance(io_json.at("io_backend_type"), io_config));
    io_->init();
    page_buf_ = io_->AllocateBuffer();
    std::memset(page_buf_, 0, PAGE_SIZE);
  }

  ~SsdLogValueStore() override {
    if (page_buf_ != nullptr) {
      io_->FreeBuffer(page_buf_);
    }
  }

  uint64_t Alloc(size_t size) override {
    const uint64_t handle = base_.Alloc(size);
    if (handle != kValueHandleNone) {
      // Slot reuse must not inherit the previous row's log entry.
      Invalidate(handle);
    }
    return handle;
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    if (handle == kValueHandleNone) {
      return;
    }
    if (Append(handle, data, size)) {
      return;
    }
    Invalidate(handle);
    base_.Write(handle, data, size);
    fallbacks_.fetch_add(1, std::memory_order_relaxed);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle == kValueHandleNone) {
      return kValueHandleNone;
    }
    Write(handle, data, size);
    return handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    for (int attempt = 0; attempt < kReadAttempts; ++attempt) {
      Loc loc;
      if (!Lookup(handle, loc)) {
        break;
      }
      EnsureFlushed(loc);
      size_t actual = 0;
      if (!ReadEntry(loc, handle, out_buf, buf_size, &actual)) {
        break;
      }
      // A concurrent fallback write or slot reuse may have replaced the entry
      // while this read was in flight.
      if (StillCurrent(handle, loc)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        return actual;
      }
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    return base_.Read(handle, out_buf, buf_size);
  }

  void BatchWrite(const std::vector<uint64_t>& handles,
                  const std::vector<WriteSpec>& specs) override {
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("SsdLogValueStore::BatchWrite size mismatch");
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      Write(handles[i], specs[i].data, specs[i].size);
    }
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& out_results) override {
    out_results.resize(handles.size());
    std::vector<Loc> locs(handles.size());
    std::vector<char> in_log(handles.size(), 0);
    std::vector<uint64_t> pending_pages;
    std::unordered_set<uint64_t> seen_pages;
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
        continue;
      }
      if (Lookup(handles[i], locs[i])) {
        in_log[i]           = 1;
        const uint64_t page = AbsolutePage(locs[i]);
        if (seen_pages.insert(page).second) {
          pending_pages.push_back(page);
        }
      }
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      if (in_log[i]) {
        EnsureFlushed(locs[i]);
      }
    }

    // Read every distinct log page once, with all page I/O in flight.
    std::unordered_map<uint64_t, char*> page_buffers;
    std::vector<IOBackend::IOEntry> entries;
    for (uint64_t page : pending_pages) {
      char* buffer = io_->AllocateBuffer();
      page_buffers.emplace(page, buffer);
      entries.push_back({page, buffer, 1});
    }
    if (!entries.empty()) {
      io_->BatchReadPages(entries);
      pages_read_.fetch_add(entries.size(), std::memory_order_relaxed);
      bytes_read_.fetch_add(
          entries.size() * PAGE_SIZE, std::memory_order_relaxed);
    }

    std::vector<size_t> retry_rows;
    std::vector<uint64_t> retry_handles;
    for (size_t i = 0; i < handles.size(); ++i) {
      if (!in_log[i]) {
        continue;
      }
      const Loc& loc   = locs[i];
      const char* page = page_buffers.at(AbsolutePage(loc));
      const auto* header =
          reinterpret_cast<const EntryHeader*>(page + loc.byte_offset);
      const bool matches =
          header->handle == handles[i] && header->size == loc.size &&
          loc.byte_offset + kHeaderBytes + loc.size <= PAGE_SIZE;
      if (matches && StillCurrent(handles[i], loc)) {
        const size_t actual = loc.size;
        out_results[i].data.resize(actual);
        std::memcpy(out_results[i].data.data(),
                    page + loc.byte_offset + kHeaderBytes,
                    actual);
        hits_.fetch_add(1, std::memory_order_relaxed);
      } else {
        retry_rows.push_back(i);
        retry_handles.push_back(handles[i]);
      }
    }
    for (auto& entry : page_buffers) {
      io_->FreeBuffer(entry.second);
    }
    if (retry_handles.empty()) {
      return;
    }
    std::vector<ReadResult> base_results;
    base_.BatchRead(retry_handles, base_results);
    misses_.fetch_add(retry_handles.size(), std::memory_order_relaxed);
    for (size_t i = 0; i < retry_rows.size(); ++i) {
      out_results[retry_rows[i]].data = std::move(base_results[i].data);
    }
  }

  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteSpec>& specs) override {
    std::vector<uint64_t> handles;
    handles.reserve(specs.size());
    for (const auto& spec : specs) {
      handles.push_back(AllocAndWrite(spec.data, spec.size));
    }
    return handles;
  }

  void Free(uint64_t handle) override {
    if (handle == kValueHandleNone) {
      return;
    }
    Invalidate(handle);
    base_.Free(handle);
  }

  size_t SlotCapacity(uint64_t handle) const override {
    return base_.SlotCapacity(handle);
  }

  const char* DirectPtr(uint64_t handle) const override {
    (void)handle;
    // The logged value is not where the handle points, so zero-copy callers
    // must take the copy path.
    return nullptr;
  }

  ValueStore::LayoutId GetLayoutId() const override {
    return base_.GetLayoutId();
  }

  void ResetStats() override {
    base_.ResetStats();
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
    appends_.store(0, std::memory_order_relaxed);
    fallbacks_.store(0, std::memory_order_relaxed);
    pages_written_.store(0, std::memory_order_relaxed);
    bytes_written_.store(0, std::memory_order_relaxed);
    pages_read_.store(0, std::memory_order_relaxed);
    bytes_read_.store(0, std::memory_order_relaxed);
  }

  std::string ExtraResultFields() const override {
    std::ostringstream out;
    out << base_.ExtraResultFields()
        << " write_log_hits=" << hits_.load(std::memory_order_relaxed)
        << " write_log_misses=" << misses_.load(std::memory_order_relaxed)
        << " write_log_appends=" << appends_.load(std::memory_order_relaxed)
        << " write_log_fallbacks=" << fallbacks_.load(std::memory_order_relaxed)
        << " write_log_pages_written="
        << pages_written_.load(std::memory_order_relaxed)
        << " write_log_pages_read="
        << pages_read_.load(std::memory_order_relaxed)
        << " write_log_bytes_written="
        << bytes_written_.load(std::memory_order_relaxed)
        << " write_log_bytes_read="
        << bytes_read_.load(std::memory_order_relaxed)
        << " write_log_index_entries=" << IndexEntries()
        << " write_log_segments_used="
        << active_segment_.load(std::memory_order_relaxed)
        << " write_log_capacity_bytes=" << capacity_bytes_;
    return out.str();
  }

  std::string GetInfo() const override {
    return "SsdLog(" + ExtraResultFields() + ")";
  }

private:
  static constexpr size_t kHeaderBytes = 16;
  static constexpr size_t kShardCount  = 64;
  static constexpr int kReadAttempts   = 4;

  struct EntryHeader {
    uint64_t handle;
    uint32_t size;
    uint32_t reserved;
  };
  static_assert(sizeof(EntryHeader) == kHeaderBytes);

  struct Loc {
    uint32_t segment     = 0;
    uint32_t page        = 0;
    uint32_t byte_offset = 0;
    uint32_t size        = 0;
  };

  static size_t EntryBytes(size_t size) {
    return (kHeaderBytes + size + 7) & ~size_t{7};
  }

  static size_t ShardOf(uint64_t handle) {
    return static_cast<size_t>((handle * 0x9E3779B97F4A7C15ULL) >> 58);
  }

  uint64_t AbsolutePage(const Loc& loc) const {
    return static_cast<uint64_t>(loc.segment) * segment_pages_ + loc.page;
  }

  bool Lookup(uint64_t handle, Loc& loc) {
    const size_t shard_id = ShardOf(handle);
    std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
    const auto found = index_[shard_id].find(handle);
    if (found == index_[shard_id].end()) {
      return false;
    }
    loc = found->second;
    return true;
  }

  bool StillCurrent(uint64_t handle, const Loc& loc) {
    const size_t shard_id = ShardOf(handle);
    std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
    const auto found = index_[shard_id].find(handle);
    return found != index_[shard_id].end() &&
           found->second.segment == loc.segment &&
           found->second.page == loc.page &&
           found->second.byte_offset == loc.byte_offset &&
           found->second.size == loc.size;
  }

  void Invalidate(uint64_t handle) {
    const size_t shard_id = ShardOf(handle);
    std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
    index_[shard_id].erase(handle);
  }

  size_t IndexEntries() const {
    size_t total = 0;
    for (size_t i = 0; i < kShardCount; ++i) {
      std::lock_guard<std::mutex> lock(index_mu_[i]);
      total += index_[i].size();
    }
    return total;
  }

  // A read that arrives while its page is still in the append buffer must not
  // read the page from disk before it is written.
  void EnsureFlushed(const Loc& loc) {
    std::lock_guard<std::mutex> lock(append_mu_);
    if (loc.segment == cursor_segment_ && loc.page == cursor_page_) {
      FlushPageLocked();
    }
  }

  bool ReadEntry(const Loc& loc,
                 uint64_t handle,
                 void* out_buf,
                 size_t buf_size,
                 size_t* actual) {
    if (loc.byte_offset + kHeaderBytes + loc.size > PAGE_SIZE) {
      return false;
    }
    char* page    = io_->AllocateBuffer();
    size_t result = 0;
    bool matches  = false;
    try {
      io_->BatchReadPages({{AbsolutePage(loc), page, 1}});
      pages_read_.fetch_add(1, std::memory_order_relaxed);
      bytes_read_.fetch_add(PAGE_SIZE, std::memory_order_relaxed);
      const auto* header =
          reinterpret_cast<const EntryHeader*>(page + loc.byte_offset);
      matches = header->handle == handle && header->size == loc.size;
      if (matches) {
        result = std::min<size_t>(loc.size, buf_size);
        std::memcpy(out_buf, page + loc.byte_offset + kHeaderBytes, result);
      }
    } catch (...) {
      io_->FreeBuffer(page);
      throw;
    }
    io_->FreeBuffer(page);
    *actual = result;
    return matches;
  }

  bool Append(uint64_t handle, const void* data, size_t size) {
    const size_t entry_bytes = EntryBytes(size);
    if (entry_bytes > PAGE_SIZE) {
      return false;
    }
    std::lock_guard<std::mutex> lock(append_mu_);
    if (cursor_segment_ >= segment_count_) {
      return false;
    }
    if (entry_bytes > PAGE_SIZE - cursor_offset_) {
      if (!FlushPageLocked()) {
        return false;
      }
    }
    const size_t offset = cursor_offset_;
    EntryHeader header{handle, static_cast<uint32_t>(size), 0};
    std::memcpy(page_buf_ + offset, &header, sizeof(header));
    std::memcpy(page_buf_ + offset + sizeof(header), data, size);
    cursor_offset_ += entry_bytes;
    Loc loc;
    loc.segment     = cursor_segment_;
    loc.page        = cursor_page_;
    loc.byte_offset = static_cast<uint32_t>(offset);
    loc.size        = static_cast<uint32_t>(size);
    {
      const size_t shard_id = ShardOf(handle);
      std::lock_guard<std::mutex> index_lock(index_mu_[shard_id]);
      index_[shard_id][handle] = loc;
    }
    appends_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Writes the current page and advances the cursor. Returns false when no
  // segment remains; the page that was just written is still valid.
  bool FlushPageLocked() {
    const uint64_t page =
        static_cast<uint64_t>(cursor_segment_) * segment_pages_ + cursor_page_;
    io_->BatchWritePages({{page, page_buf_, 1}});
    pages_written_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(PAGE_SIZE, std::memory_order_relaxed);
    std::memset(page_buf_, 0, PAGE_SIZE);
    cursor_offset_ = 0;
    ++cursor_page_;
    if (cursor_page_ >= segment_pages_) {
      cursor_page_ = 0;
      ++cursor_segment_;
      active_segment_.store(cursor_segment_, std::memory_order_relaxed);
      if (cursor_segment_ >= segment_count_) {
        return false;
      }
    }
    return true;
  }

  SsdValueStore base_;
  std::unique_ptr<IOBackend> io_;
  char* page_buf_          = nullptr;
  uint64_t capacity_bytes_ = 0;
  uint64_t segment_bytes_  = 0;
  uint64_t segment_pages_  = 0;
  uint64_t segment_count_  = 0;

  std::mutex append_mu_;
  uint32_t cursor_segment_ = 0;
  uint32_t cursor_page_    = 0;
  uint32_t cursor_offset_  = 0;

  std::array<std::unordered_map<uint64_t, Loc>, kShardCount> index_;
  mutable std::array<std::mutex, kShardCount> index_mu_;

  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> appends_{0};
  std::atomic<uint64_t> fallbacks_{0};
  std::atomic<uint64_t> pages_written_{0};
  std::atomic<uint64_t> bytes_written_{0};
  std::atomic<uint64_t> pages_read_{0};
  std::atomic<uint64_t> bytes_read_{0};
  std::atomic<uint64_t> active_segment_{0};
};

FACTORY_REGISTER(
    ValueStore, SSD_LOG_VALUE_STORE, SsdLogValueStore, const BaseKVConfig&);
