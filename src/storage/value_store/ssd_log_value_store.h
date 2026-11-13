#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
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
// concurrent update, a compaction, or a fallback write revalidates the index
// and retries instead of trusting the in-place slot, which may never have been
// written for that row.
//
// Free keeps the index entry until the slot is allocated again: updates are
// copy-on-write, so a reader that fetched the handle just before the update
// must still see the previous value (the in-place slot keeps its bytes too).
//
// Stage 2 recycles segments. A sealed segment is compacted into one of the
// free segments: live entries are re-appended, their index entries move, and
// the old segment becomes free. Compaction runs only when one free segment is
// left, and the victim is the least-utilized sealed segment, so the copy cost
// stays bounded (utilization u copies u bytes to reclaim 1-u). If even the
// emptiest segment is above the configured utilization bound, the last
// segment is spent and later updates fall back to the in-place path.
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
    segment_pages_              = segment_bytes_ / PAGE_SIZE;
    segment_count_              = capacity_bytes_ / segment_bytes_;
    const double gc_utilization = log.value("gc_max_utilization", 0.9);
    if (!std::isfinite(gc_utilization) || gc_utilization <= 0.0 ||
        gc_utilization >= 1.0) {
      throw std::invalid_argument(
          "write_log.gc_max_utilization must be in (0, 1)");
    }
    gc_max_live_bytes_ = static_cast<uint64_t>(
        gc_utilization * static_cast<double>(segment_bytes_));

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

    segments_ = std::make_unique<Segment[]>(segment_count_);
    for (uint32_t i = 0; i < segment_count_; ++i) {
      free_.push_back(i);
    }
    free_segments_.store(free_.size(), std::memory_order_relaxed);
    const uint32_t first = free_.front();
    free_.pop_front();
    free_segments_.store(free_.size(), std::memory_order_relaxed);
    ActivateLocked(first);
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
        // The entry moved (concurrent update or compaction); look it up again.
        continue;
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
    std::vector<size_t> pending;
    std::vector<size_t> base_rows;
    pending.reserve(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
      } else {
        pending.push_back(i);
      }
    }

    for (int attempt = 0; attempt < kReadAttempts && !pending.empty();
         ++attempt) {
      std::vector<Loc> locs(pending.size());
      std::vector<size_t> log_positions;
      std::vector<size_t> unresolved;
      log_positions.reserve(pending.size());
      for (size_t k = 0; k < pending.size(); ++k) {
        if (Lookup(handles[pending[k]], locs[k])) {
          log_positions.push_back(k);
        } else {
          // The index entry is gone, so the in-place copy is authoritative.
          base_rows.push_back(pending[k]);
        }
      }
      if (log_positions.empty()) {
        break;
      }
      EnsureFlushedBatch(locs, log_positions);

      // Read every distinct log page once, with all page I/O in flight.
      std::unordered_map<uint64_t, char*> page_buffers;
      std::vector<IOBackend::IOEntry> entries;
      entries.reserve(log_positions.size());
      for (size_t k : log_positions) {
        const uint64_t page = AbsolutePage(locs[k]);
        if (page_buffers.find(page) != page_buffers.end()) {
          continue;
        }
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

      for (size_t k : log_positions) {
        const size_t row = pending[k];
        const Loc& loc   = locs[k];
        const char* page = page_buffers.at(AbsolutePage(loc));
        const auto* header =
            reinterpret_cast<const EntryHeader*>(page + loc.byte_offset);
        const bool matches =
            header->handle == handles[row] && header->size == loc.size &&
            loc.byte_offset + kHeaderBytes + loc.size <= PAGE_SIZE;
        // A concurrent update or compaction can move the entry between the
        // lookup and this check; re-look it up instead of reading the empty
        // in-place slot, which never received an update for this row.
        if (matches && StillCurrent(handles[row], loc)) {
          const size_t actual = loc.size;
          out_results[row].data.resize(actual);
          std::memcpy(out_results[row].data.data(),
                      page + loc.byte_offset + kHeaderBytes,
                      actual);
          hits_.fetch_add(1, std::memory_order_relaxed);
        } else {
          unresolved.push_back(row);
        }
      }
      for (auto& entry : page_buffers) {
        io_->FreeBuffer(entry.second);
      }
      pending.swap(unresolved);
    }

    base_rows.insert(base_rows.end(), pending.begin(), pending.end());
    if (base_rows.empty()) {
      return;
    }
    std::vector<uint64_t> base_handles;
    base_handles.reserve(base_rows.size());
    for (size_t row : base_rows) {
      base_handles.push_back(handles[row]);
    }
    std::vector<ReadResult> base_results;
    base_.BatchRead(base_handles, base_results);
    misses_.fetch_add(base_rows.size(), std::memory_order_relaxed);
    for (size_t i = 0; i < base_rows.size(); ++i) {
      out_results[base_rows[i]].data = std::move(base_results[i].data);
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
    // Keep the log entry. A reader that fetched this handle from the index
    // just before an update retired it must still see the previous value, and
    // the in-place slot also keeps its bytes after Free. Alloc() drops the
    // entry when the slot is handed out again.
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
    compactions_.store(0, std::memory_order_relaxed);
    compaction_bytes_.store(0, std::memory_order_relaxed);
    reclaimed_segments_.store(0, std::memory_order_relaxed);
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
        << (segment_count_ - free_segments_.load(std::memory_order_relaxed))
        << " write_log_segments_free="
        << free_segments_.load(std::memory_order_relaxed)
        << " write_log_compactions="
        << compactions_.load(std::memory_order_relaxed)
        << " write_log_compaction_bytes="
        << compaction_bytes_.load(std::memory_order_relaxed)
        << " write_log_segments_reclaimed="
        << reclaimed_segments_.load(std::memory_order_relaxed)
        << " write_log_exhausted="
        << (exhausted_.load(std::memory_order_relaxed) ? 1 : 0)
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
    // Bumped whenever a segment is reused, so a reader that raced a
    // compaction and a segment recycle cannot accept a same-shaped stale
    // entry from the previous life of that segment.
    uint32_t generation = 0;
  };

  struct Segment {
    enum State { kFree, kActive, kSealed, kCompacting };
    State state = kFree;
    std::atomic<uint64_t> live_bytes{0};
    std::atomic<uint32_t> epoch{0};
  };

  static size_t EntryBytes(size_t size) {
    return (kHeaderBytes + size + 7) & ~size_t{7};
  }

  static size_t ShardOf(uint64_t handle) {
    return static_cast<size_t>((handle * 0x9E3779B97F4A7C15ULL) >> 58);
  }

  uint64_t AbsolutePage(uint32_t segment, uint32_t page) const {
    return static_cast<uint64_t>(segment) * segment_pages_ + page;
  }

  uint64_t AbsolutePage(const Loc& loc) const {
    return AbsolutePage(loc.segment, loc.page);
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
    return found != index_[shard_id].end() && SameLoc(found->second, loc);
  }

  static bool SameLoc(const Loc& a, const Loc& b) {
    return a.segment == b.segment && a.page == b.page &&
           a.byte_offset == b.byte_offset && a.size == b.size &&
           a.generation == b.generation;
  }

  void Invalidate(uint64_t handle) {
    const size_t shard_id = ShardOf(handle);
    std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
    auto& shard      = index_[shard_id];
    const auto found = shard.find(handle);
    if (found == shard.end()) {
      return;
    }
    DropLiveBytes(found->second);
    shard.erase(found);
  }

  size_t IndexEntries() const {
    size_t total = 0;
    for (size_t i = 0; i < kShardCount; ++i) {
      std::lock_guard<std::mutex> lock(index_mu_[i]);
      total += index_[i].size();
    }
    return total;
  }

  // Per-segment live byte accounting follows the index: an entry contributes
  // EntryBytes(size) to exactly the segment its current index location names.
  void DropLiveBytes(const Loc& loc) {
    segments_[loc.segment].live_bytes.fetch_sub(
        EntryBytes(loc.size), std::memory_order_relaxed);
  }

  // Callers hold the index shard lock, so the previous location of a row and
  // its replacement move between segments atomically.
  void PublishLocked(uint64_t handle, const Loc& loc) {
    auto& shard      = index_[ShardOf(handle)];
    const auto found = shard.find(handle);
    if (found != shard.end()) {
      DropLiveBytes(found->second);
    }
    shard[handle] = loc;
    segments_[loc.segment].live_bytes.fetch_add(
        EntryBytes(loc.size), std::memory_order_relaxed);
  }

  bool PageNeedsFlush(const Loc& loc) const {
    return loc.segment == cursor_segment_.load(std::memory_order_acquire) &&
           loc.page == cursor_page_.load(std::memory_order_acquire);
  }

  // A read that arrives while its page is still in the append buffer must not
  // read the page from disk before it is written. The cursor check is
  // lock-free, so the common case (a page that was already flushed) does not
  // serialize readers on the append lock.
  void EnsureFlushed(const Loc& loc) {
    if (!PageNeedsFlush(loc)) {
      return;
    }
    std::lock_guard<std::mutex> lock(append_mu_);
    if (cursor_offset_ != 0) {
      FlushPageLocked();
    }
  }

  void EnsureFlushedBatch(const std::vector<Loc>& locs,
                          const std::vector<size_t>& positions) {
    bool flushed = false;
    for (size_t k : positions) {
      if (flushed || !PageNeedsFlush(locs[k])) {
        continue;
      }
      std::lock_guard<std::mutex> lock(append_mu_);
      if (cursor_offset_ != 0) {
        FlushPageLocked();
        flushed = true;
      }
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
    size_t offset = 0;
    if (!ReserveLocked(entry_bytes, &offset)) {
      return false;
    }
    WriteEntryLocked(offset, handle, data, size);
    const Loc loc         = MakeLoc(offset, size);
    const size_t shard_id = ShardOf(handle);
    {
      std::lock_guard<std::mutex> index_lock(index_mu_[shard_id]);
      PublishLocked(handle, loc);
    }
    appends_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Reserves room for one entry in the active page, flushing and advancing
  // the segment when the page is full. Returns false once the log has no
  // segment left to write into.
  bool ReserveLocked(size_t entry_bytes, size_t* offset) {
    if (exhausted_.load(std::memory_order_relaxed)) {
      return false;
    }
    // Compaction appends into the new segment while the flush runs, so the
    // page it leaves behind may already be too full for this entry.
    while (entry_bytes > PAGE_SIZE - cursor_offset_) {
      if (!FlushPageLocked()) {
        return false;
      }
    }
    *offset = cursor_offset_;
    cursor_offset_ += static_cast<uint32_t>(entry_bytes);
    return true;
  }

  void WriteEntryLocked(
      size_t offset, uint64_t handle, const void* data, size_t size) {
    EntryHeader header{handle, static_cast<uint32_t>(size), 0};
    std::memcpy(page_buf_ + offset, &header, sizeof(header));
    std::memcpy(page_buf_ + offset + sizeof(header), data, size);
  }

  Loc MakeLoc(size_t offset, size_t size) const {
    const uint32_t segment = cursor_segment_.load(std::memory_order_relaxed);
    Loc loc;
    loc.segment     = segment;
    loc.page        = cursor_page_.load(std::memory_order_relaxed);
    loc.byte_offset = static_cast<uint32_t>(offset);
    loc.size        = static_cast<uint32_t>(size);
    loc.generation  = segments_[segment].epoch.load(std::memory_order_relaxed);
    return loc;
  }

  // Writes the current page and advances the cursor. Returns false when no
  // segment remains; the page that was just written is still valid.
  bool FlushPageLocked() {
    if (cursor_offset_ == 0) {
      return !exhausted_.load(std::memory_order_relaxed);
    }
    const uint32_t segment = cursor_segment_.load(std::memory_order_relaxed);
    const uint32_t page    = cursor_page_.load(std::memory_order_relaxed);
    io_->BatchWritePages({{AbsolutePage(segment, page), page_buf_, 1}});
    pages_written_.fetch_add(1, std::memory_order_relaxed);
    bytes_written_.fetch_add(PAGE_SIZE, std::memory_order_relaxed);
    std::memset(page_buf_, 0, PAGE_SIZE);
    cursor_offset_ = 0;
    if (page + 1 >= segment_pages_) {
      cursor_page_.store(0, std::memory_order_release);
      segments_[segment].state = Segment::kSealed;
      return AdvanceSegmentLocked();
    }
    cursor_page_.store(page + 1, std::memory_order_release);
    return true;
  }

  bool AdvanceSegmentLocked() {
    ReclaimDeadLocked();
    if (free_.empty()) {
      exhausted_.store(true, std::memory_order_relaxed);
      return false;
    }
    // Keep one free segment while the active one fills: with a single free
    // segment left, compact a sealed segment into it instead of spending it.
    if (!compacting_ && free_.size() == 1 && CompactIntoFreeLocked()) {
      return true;
    }
    const uint32_t dest = free_.front();
    free_.pop_front();
    free_segments_.store(free_.size(), std::memory_order_relaxed);
    ActivateLocked(dest);
    return true;
  }

  void ReclaimDeadLocked() {
    for (uint32_t i = 0; i < segment_count_; ++i) {
      if (segments_[i].state != Segment::kSealed ||
          segments_[i].live_bytes.load(std::memory_order_relaxed) != 0) {
        continue;
      }
      segments_[i].state = Segment::kFree;
      free_.push_back(i);
    }
    free_segments_.store(free_.size(), std::memory_order_relaxed);
  }

  // Moves the least-utilized sealed segment into the last free segment, which
  // becomes the active one. Caller holds append_mu_ and free_ has one entry.
  bool CompactIntoFreeLocked() {
    const int64_t victim = PickVictimLocked();
    if (victim < 0) {
      return false;
    }
    const uint32_t dest = free_.front();
    free_.pop_front();
    free_segments_.store(free_.size(), std::memory_order_relaxed);
    ActivateLocked(dest);
    segments_[victim].state = Segment::kCompacting;
    const uint32_t victim_epoch =
        segments_[victim].epoch.load(std::memory_order_relaxed);
    compacting_ = true;

    char* page = io_->AllocateBuffer();
    try {
      for (uint32_t p = 0; p < segment_pages_; ++p) {
        io_->BatchReadPages({{AbsolutePage(victim, p), page, 1}});
        pages_read_.fetch_add(1, std::memory_order_relaxed);
        bytes_read_.fetch_add(PAGE_SIZE, std::memory_order_relaxed);
        size_t offset = 0;
        while (offset + kHeaderBytes <= PAGE_SIZE) {
          const auto* header =
              reinterpret_cast<const EntryHeader*>(page + offset);
          if (header->handle == kValueHandleNone || header->size == 0 ||
              offset + kHeaderBytes + header->size > PAGE_SIZE) {
            break;
          }
          Loc old_loc;
          old_loc.segment     = victim;
          old_loc.page        = p;
          old_loc.byte_offset = static_cast<uint32_t>(offset);
          old_loc.size        = header->size;
          old_loc.generation  = victim_epoch;
          if (AppendIfCurrentLocked(
                  header->handle, old_loc, page, offset, header->size)) {
            compaction_bytes_.fetch_add(
                EntryBytes(header->size), std::memory_order_relaxed);
          }
          offset += EntryBytes(header->size);
        }
      }
    } catch (...) {
      io_->FreeBuffer(page);
      compacting_             = false;
      segments_[victim].state = Segment::kSealed;
      throw;
    }
    io_->FreeBuffer(page);
    compacting_ = false;

    if (segments_[victim].live_bytes.load(std::memory_order_relaxed) != 0) {
      // Every live entry was copied or invalidated, so this is unreachable.
      // Keep the segment sealed rather than free space a reader may still
      // resolve; the next advance retries the compaction.
      segments_[victim].state = Segment::kSealed;
      return true;
    }
    segments_[victim].state = Segment::kFree;
    free_.push_back(victim);
    free_segments_.store(free_.size(), std::memory_order_relaxed);
    compactions_.fetch_add(1, std::memory_order_relaxed);
    reclaimed_segments_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  int64_t PickVictimLocked() const {
    int64_t best       = -1;
    uint64_t best_live = UINT64_MAX;
    for (uint32_t i = 0; i < segment_count_; ++i) {
      if (segments_[i].state != Segment::kSealed) {
        continue;
      }
      const uint64_t live =
          segments_[i].live_bytes.load(std::memory_order_relaxed);
      if (live == 0 || live > gc_max_live_bytes_) {
        continue;
      }
      if (live < best_live) {
        best_live = live;
        best      = i;
      }
    }
    return best;
  }

  // Copies one entry out of a segment being compacted. The index is checked
  // under its shard lock, so an entry freed while the victim page was being
  // read is skipped instead of resurrected.
  bool AppendIfCurrentLocked(
      uint64_t handle,
      const Loc& expect,
      const char* page,
      size_t entry_offset,
      size_t size) {
    // Most stale entries in a compacted segment are already dead. Reject them
    // before reserving space, or the destination fills with garbage and the
    // copy cannot finish.
    const size_t shard_id = ShardOf(handle);
    {
      std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
      const auto& shard = index_[shard_id];
      const auto found  = shard.find(handle);
      if (found == shard.end() || !SameLoc(found->second, expect)) {
        return false;
      }
    }

    size_t offset = 0;
    if (!ReserveLocked(EntryBytes(size), &offset)) {
      return false;
    }
    WriteEntryLocked(offset, handle, page + entry_offset + kHeaderBytes, size);
    const Loc loc = MakeLoc(offset, size);
    // Revalidate under the lock: a concurrent Free may have dropped the entry
    // while the victim page was being copied.
    std::lock_guard<std::mutex> lock(index_mu_[shard_id]);
    auto& shard      = index_[shard_id];
    const auto found = shard.find(handle);
    if (found == shard.end() || !SameLoc(found->second, expect)) {
      return false;
    }
    DropLiveBytes(found->second);
    shard[handle] = loc;
    segments_[loc.segment].live_bytes.fetch_add(
        EntryBytes(loc.size), std::memory_order_relaxed);
    return true;
  }

  void ActivateLocked(uint32_t segment) {
    cursor_segment_.store(segment, std::memory_order_release);
    cursor_page_.store(0, std::memory_order_relaxed);
    cursor_offset_           = 0;
    segments_[segment].state = Segment::kActive;
    segments_[segment].live_bytes.store(0, std::memory_order_relaxed);
    segments_[segment].epoch.fetch_add(1, std::memory_order_relaxed);
    std::memset(page_buf_, 0, PAGE_SIZE);
  }

  SsdValueStore base_;
  std::unique_ptr<IOBackend> io_;
  char* page_buf_             = nullptr;
  uint64_t capacity_bytes_    = 0;
  uint64_t segment_bytes_     = 0;
  uint64_t segment_pages_     = 0;
  uint64_t segment_count_     = 0;
  uint64_t gc_max_live_bytes_ = 0;

  std::mutex append_mu_;
  // Read without the append lock so a flushed page does not serialize readers.
  std::atomic<uint32_t> cursor_segment_{0};
  std::atomic<uint32_t> cursor_page_{0};
  uint32_t cursor_offset_ = 0;
  bool compacting_        = false;
  std::unique_ptr<Segment[]> segments_;
  std::deque<uint32_t> free_;

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
  std::atomic<uint64_t> compactions_{0};
  std::atomic<uint64_t> compaction_bytes_{0};
  std::atomic<uint64_t> reclaimed_segments_{0};
  std::atomic<uint64_t> free_segments_{0};
  std::atomic<bool> exhausted_{false};
};

FACTORY_REGISTER(
    ValueStore, SSD_LOG_VALUE_STORE, SsdLogValueStore, const BaseKVConfig&);
