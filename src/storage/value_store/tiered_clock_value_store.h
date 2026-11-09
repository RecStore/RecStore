#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <list>
#include <mutex>
#include <sstream>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "storage/value_store/dram_value_store.h"

// Handles contain a slot index and generation. Metadata slots have stable
// addresses and are recycled; readers never need a hash lookup or a shard lock.
class TieredClockValueStore final : public ValueStore {
public:
  TieredClockValueStore(
      DramValueStore* dram,
      ValueStore* ssd,
      uint64_t dram_limit,
      size_t shards,
      uint32_t min_misses,
      bool retain_backing_copies = true,
      bool frequency_admission = false)
      : dram_(dram),
        ssd_(ssd),
        limit_(dram_limit),
        shards_(shards),
        min_misses_(min_misses),
        fixed_(!ssd->GetLayoutId().plain()),
        retain_backing_(retain_backing_copies),
        frequency_admission_(frequency_admission),
        chunks_(new std::atomic<Chunk*>[kChunkCount]) {
    if (fixed_ && !retain_backing_)
      throw std::invalid_argument(
          "releasing CLOCK backing copies requires a plain SSD layout");
    for (size_t i = 0; i < kChunkCount; ++i)
      chunks_[i].store(nullptr);
  }

  ~TieredClockValueStore() override {
    for (size_t i = 0; i < kChunkCount; ++i)
      delete chunks_[i].load();
  }

  uint64_t Alloc(size_t size) override {
    if (fixed_)
      throw std::invalid_argument("CLOCK static placement requires a key");
    const size_t shard_id =
        allocation_shard_.fetch_add(1, std::memory_order_relaxed) %
        shards_.size();
    std::lock_guard<std::mutex> lock(shards_[shard_id].mutex);
    std::unique_lock<ValueMutex> entry_lock;
    Entry* entry = Create(shard_id, 0, false, size, entry_lock);
    return entry ? entry->handle : kValueHandleNone;
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      try {
        Write(handle, data, size);
      } catch (...) {
        Free(handle);
        throw;
      }
    }
    return handle;
  }

  uint64_t AllocAndWrite(uint64_t key, const void* data, size_t size) override {
    ValidateSize(key, size);
    const size_t shard_id = key % shards_.size();
    Shard& shard          = shards_[shard_id];
    for (;;) {
      std::unique_lock<std::mutex> shard_lock(shard.mutex);
      auto found = shard.keys.find(key);
      if (found != shard.keys.end()) {
        const uint64_t handle = found->second;
        Entry* entry          = Locate(handle);
        shard_lock.unlock();
        std::lock_guard<ValueMutex> lock(entry->mutex);
        if (!Matches(*entry, handle))
          continue;
        WriteLocked(*entry, data, size);
        return handle;
      }
      std::unique_lock<ValueMutex> entry_lock;
      Entry* entry = Create(shard_id, key, true, size, entry_lock);
      if (!entry)
        return kValueHandleNone;
      shard_lock.unlock();
      try {
        WriteLocked(*entry, data, size);
      } catch (...) {
        FreeLocked(*entry);
        throw;
      }
      return entry->handle;
    }
  }

  using ValueStore::BatchAllocAndWrite;
  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<uint64_t>& keys,
                     const std::vector<WriteSpec>& specs) override {
    if (keys.size() != specs.size())
      throw std::invalid_argument("CLOCK batch size mismatch");
    std::vector<uint64_t> handles;
    handles.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      handles.push_back(AllocAndWrite(keys[i], specs[i].data, specs[i].size));
    return handles;
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    Entry* entry = Locate(handle);
    if (!entry)
      return;
    std::lock_guard<ValueMutex> lock(entry->mutex);
    if (!Matches(*entry, handle))
      return;
    ValidateSize(entry->key, size);
    WriteLocked(*entry, data, size);
  }

  size_t Read(uint64_t handle, void* out, size_t capacity) override {
    Fast* fast = LocateFast(handle);
    if (!fast || !out)
      return 0;
    {
      ReadGuard guard(GetReader(), fast);
      if (fast->readable.load() == handle) {
        const size_t size = std::min(capacity, fast->size);
        std::memcpy(out, fast->direct, size);
        RecordHit(*fast, guard.reader);
        return size;
      }
    }
    Entry* entry = Locate(handle);
    std::lock_guard<ValueMutex> lock(entry->mutex);
    return Matches(*entry, handle) ? ReadLocked(*entry, out, capacity) : 0;
  }

  void ReadValue(uint64_t handle, std::string& out) override {
    Fast* fast = LocateFast(handle);
    if (!fast) {
      out.clear();
      return;
    }
    {
      ReadGuard guard(GetReader(), fast);
      if (fast->readable.load() == handle) {
        out.resize(fast->size);
        std::memcpy(out.data(), fast->direct, out.size());
        RecordHit(*fast, guard.reader);
        return;
      }
    }
    Entry* entry = Locate(handle);
    std::lock_guard<ValueMutex> lock(entry->mutex);
    if (!Matches(*entry, handle)) {
      out.clear();
      return;
    }
    out.resize(entry->size);
    out.resize(ReadLocked(*entry, out.data(), out.size()));
  }

  bool ReadFlatFixedRows(const uint64_t* handles,
                         size_t num_rows,
                         void* out_buf,
                         size_t row_bytes,
                         uint64_t* missing_rows) const override {
    if (!handles || !out_buf || row_bytes == 0)
      return false;
    auto* out        = static_cast<char*>(out_buf);
    uint64_t missing = 0;
    std::vector<uint64_t> pending;
    std::vector<size_t> rows;
    Reader& reader = GetReader();
    // Bound each read-side critical section so writers need wait for at most
    // 64 copies. One epoch publication covers all snapshots in that section.
    for (size_t begin = 0; begin < num_rows; begin += 64) {
      BatchReadGuard guard(reader);
      const size_t end = std::min(num_rows, begin + 64);
      for (size_t i = begin; i < end; ++i) {
        if (handles[i] == kValueHandleNone) {
          std::memset(out + i * row_bytes, 0, row_bytes);
          ++missing;
          continue;
        }
        if (Fast* fast = LocateFast(handles[i])) {
          if (fast->readable.load() == handles[i]) {
            if (fast->size != row_bytes)
              return false;
            std::memcpy(out + i * row_bytes, fast->direct, row_bytes);
            RecordHit(*fast, reader);
            continue;
          }
        }
        pending.push_back(handles[i]);
        rows.push_back(i);
      }
    }
    if (!pending.empty()) {
      std::vector<ReadResult> fetched;
      // Reading may promote SSD rows, just as the non-flat read API does.
      const_cast<TieredClockValueStore*>(this)->BatchRead(pending, fetched);
      for (size_t i = 0; i < rows.size(); ++i) {
        if (fetched[i].data.size() != row_bytes)
          return false;
        std::memcpy(
            out + rows[i] * row_bytes, fetched[i].data.data(), row_bytes);
      }
    }
    if (missing_rows)
      *missing_rows = missing;
    return true;
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& results) override {
    results.clear();
    results.resize(handles.size());
    // Copy resident rows before locking SSD misses: a slow SSD read must not
    // hold the locks of hot DRAM rows needed by other batches or writers.
    // Order remaining locks by metadata address, including recycled slots.
    constexpr size_t kBatch = 262144;
    for (size_t begin = 0; begin < handles.size(); begin += kBatch) {
      const size_t end = std::min(handles.size(), begin + kBatch);
      std::vector<Entry*> entries;
      std::vector<size_t> pending;
      for (size_t i = begin; i < end; ++i) {
        Fast* fast = LocateFast(handles[i]);
        if (!fast)
          continue;
        {
          ReadGuard guard(GetReader(), fast);
          if (fast->readable.load() == handles[i]) {
            auto& data = results[i].data;
            data.resize(fast->size);
            std::memcpy(data.data(), fast->direct, data.size());
            RecordHit(*fast, guard.reader);
            continue;
          }
        }
        pending.push_back(i);
        entries.push_back(Locate(handles[i]));
      }
      std::sort(entries.begin(), entries.end(), std::less<Entry*>{});
      entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
      std::vector<std::shared_lock<ValueMutex>> locks;
      for (Entry* entry : entries)
        locks.emplace_back(entry->mutex);
      std::vector<uint64_t> raw_handles, versions;
      std::vector<size_t> indices;
      for (size_t i : pending) {
        Entry* entry = Locate(handles[i]);
        if (!entry || !Matches(*entry, handles[i]) || !entry->initialized)
          continue;
        if (entry->ssd) {
          raw_handles.push_back(entry->raw);
          versions.push_back(entry->version);
          indices.push_back(i);
          ++entry->stats.misses;
          if (frequency_admission_)
            RecordAccess(GetReader());
        } else {
          auto& data = results[i].data;
          data.resize(entry->size);
          ReadLocked(*entry, data.data(), data.size());
        }
      }
      std::vector<ReadResult> fetched;
      if (!raw_handles.empty())
        ssd_->BatchRead(raw_handles, fetched);
      if (fetched.size() != indices.size())
        throw std::runtime_error("CLOCK batch result size mismatch");
      // Readers share miss locks during I/O. Recheck versions after acquiring
      // exclusive promotion locks, then group demand-driven cold writebacks.
      locks.clear();
      std::vector<Exchange> exchanges;
      try {
        exchanges.reserve(indices.size());
        std::unordered_set<Entry*> scheduled;
        scheduled.reserve(indices.size());
        for (size_t j = 0; j < indices.size(); ++j) {
          const size_t i = indices[j];
          Entry& entry   = *Locate(handles[i]);
          auto& data     = fetched[j].data;
          if (scheduled.count(&entry)) {
            Touch(*entry.fast);
            continue;
          }
          std::unique_lock<ValueMutex> lock(entry.mutex, std::try_to_lock);
          if (!lock)
            continue;
          if (Matches(entry, handles[i]) && entry.ssd &&
              entry.version == versions[j]) {
            const auto count = exchanges.size();
            FinishRead(entry, data.data(), data.size(), &exchanges, &scheduled);
            if (exchanges.size() != count) {
              exchanges.back().hot_lock = std::move(lock);
              scheduled.insert(&entry);
              scheduled.insert(exchanges.back().cold);
            }
          }
        }
        FlushExchanges(exchanges);
      } catch (const std::exception&) {
        for (auto& exchange : exchanges)
          if (exchange.hot->ssd)
            ++exchange.hot->stats.failures;
      }
      for (auto& exchange : exchanges)
        if (exchange.allocated)
          ssd_->Free(exchange.dest);
      for (size_t j = 0; j < indices.size(); ++j)
        results[indices[j]] = std::move(fetched[j]);
    }
  }

  void Free(uint64_t handle) override {
    Entry* entry = Locate(handle);
    if (!entry)
      return;
    std::lock_guard<ValueMutex> lock(entry->mutex);
    if (Matches(*entry, handle))
      FreeLocked(*entry);
  }

  size_t SlotCapacity(uint64_t handle) const override {
    Entry* entry = Locate(handle);
    if (!entry)
      return 0;
    std::lock_guard<ValueMutex> lock(entry->mutex);
    return Matches(*entry, handle) ? entry->size : 0;
  }

  LayoutId GetLayoutId() const override { return ssd_->GetLayoutId(); }

  std::string ExtraResultFields() const override {
    uint64_t dram = 0, ssd = 0, backed = 0;
    Counters stats;
    const uint64_t count = std::min<uint64_t>(next_slot_.load(), UINT32_MAX);
    for (uint64_t i = 0; i < count; ++i) {
      Entry* entry = Locate(i + 1);
      if (!entry)
        continue;
      std::lock_guard<ValueMutex> lock(entry->mutex);
      if (entry->alive) {
        entry->ssd ? ++ssd : ++dram;
        if (entry->backing != kValueHandleNone)
          ++backed;
      }
      stats.hits += entry->stats.hits;
      stats.misses += entry->stats.misses;
      stats.promotions += entry->stats.promotions;
      stats.demotions += entry->stats.demotions;
      stats.failures += entry->stats.failures;
      stats.rejects += entry->stats.rejects;
      stats.clean_evictions += entry->stats.clean_evictions;
      stats.writebacks += entry->stats.writebacks;
    }
    {
      std::lock_guard<std::mutex> lock(readers_mutex_);
      for (const auto& reader : readers_)
        stats.hits += reader->hits.load(std::memory_order_relaxed);
    }
    std::ostringstream out;
    out << " tiered_cache_policy=clock tiered_clock_promote_min_misses="
        << min_misses_ << " tiered_clock_retain_backing_copies="
        << retain_backing_ << " tiered_clock_frequency_admission="
        << frequency_admission_ << " tiered_dram_live_allocs=" << dram
        << " tiered_ssd_live_allocs=" << ssd
        << " tiered_live_allocs=" << dram + ssd << " tiered_ssd_live_ratio="
        << (dram + ssd ? double(ssd) / (dram + ssd) : 0)
        << " tiered_dram_bytes=" << dram_bytes_.load(std::memory_order_relaxed)
        << " tiered_dram_limit_bytes=" << limit_
        << " tiered_clock_backed_values=" << backed
        << " tiered_clock_metadata_slots=" << count << " tiered_clock_hits="
        << stats.hits << " tiered_clock_misses=" << stats.misses
        << " tiered_clock_promotions=" << stats.promotions
        << " tiered_clock_demotions=" << stats.demotions
        << " tiered_clock_failures=" << stats.failures
        << " tiered_clock_admission_rejects=" << stats.rejects
        << " tiered_clock_clean_evictions=" << stats.clean_evictions
        << " tiered_clock_writebacks=" << stats.writebacks;
    return out.str();
  }

private:
  struct Counters {
    std::atomic<uint64_t> hits{0}, misses{0};
    uint64_t promotions = 0, demotions = 0, failures = 0, rejects = 0,
             clean_evictions = 0, writebacks = 0;
  };
  struct Fast {
    std::atomic<uint64_t> readable{0};
    const char* direct = nullptr;
    size_t size        = 0;
    std::atomic<bool> referenced{false};
    std::atomic<uint8_t> heat{0};
    // Both modes fit in the original snapshot padding.
    std::atomic<uint32_t> frequency{0}; // Upper 24 bits: epoch; lower 8: count.
  };
  static_assert(sizeof(Fast) == 32, "compact resident snapshot");
  using ValueMutex = std::shared_mutex;
  struct Entry {
    ValueMutex mutex;
    Fast* fast         = nullptr;
    const char* direct = nullptr;
    size_t size        = 0;
    uint64_t backing   = kValueHandleNone;
    bool dirty         = false;
    uint64_t handle = 0, raw = 0, key = 0, version = 0;
    size_t capacity = 0, shard = 0;
    bool alive = false, ssd = false, keyed = false, initialized = false;
    uint32_t misses = 0;
    std::list<uint64_t>::iterator position;
    Counters stats; // Lifetime counters survive metadata-slot reuse.
  };
  struct Exchange {
    Entry* hot;
    Entry* cold;
    const void* data;
    uint64_t dest;
    bool allocated;
    bool writeback;
    std::unique_lock<ValueMutex> cold_lock;
    std::unique_lock<ValueMutex> hot_lock;
  };
  // Readers publish a hazard before checking readable. Writers revoke
  // readable, then wait for existing hazards before changing the DRAM bytes
  // or metadata. Sequential consistency closes the publication/check race.
  // No shared read-side lock or reference-count RMW touches a hot value.
  struct alignas(64) Reader {
    std::atomic<Fast*> active{nullptr};
    std::atomic<uint64_t> batch_epoch{0};
    std::atomic<uint64_t> hits{0};
    uint32_t accesses = 0;
    const std::thread::id owner = std::this_thread::get_id();
  };
  struct ReadGuard {
    Reader& reader;
    ReadGuard(Reader& state, Fast* entry) : reader(state) {
      reader.active.store(entry);
    }
    ~ReadGuard() { reader.active.store(nullptr, std::memory_order_release); }
  };
  struct BatchReadGuard {
    Reader& reader;
    explicit BatchReadGuard(Reader& state) : reader(state) {
      reader.batch_epoch.fetch_add(1);
    }
    ~BatchReadGuard() {
      reader.batch_epoch.fetch_add(1, std::memory_order_release);
    }
  };
  Reader& GetReader() const {
    struct Local {
      uint64_t owner = 0;
      Reader* reader = nullptr;
    };
    static thread_local Local local;
    if (local.owner != instance_) {
      std::lock_guard<std::mutex> lock(readers_mutex_);
      for (const auto& reader : readers_) {
        if (reader->owner == std::this_thread::get_id()) {
          local = {instance_, reader.get()};
          return *local.reader;
        }
      }
      readers_.push_back(std::make_unique<Reader>());
      local = {instance_, readers_.back().get()};
    }
    return *local.reader;
  }
  static constexpr uint32_t kHeatEpochMask = (1U << 24) - 1;
  static uint8_t HeatAt(uint32_t state, uint32_t epoch) {
    const uint32_t age = (epoch - (state >> 8)) & kHeatEpochMask;
    // A concurrent touch can publish the next epoch after a caller sampled it.
    return age > kHeatEpochMask / 2 ? state & UINT8_MAX
                                   : (state & UINT8_MAX) >> std::min(age, 8U);
  }
  void RecordAccess(Reader& reader) const {
    // Publish one shared clock update per 1024 reads, including SSD misses.
    if (++reader.accesses != 1024)
      return;
    reader.accesses = 0;
    const uint64_t accesses = heat_accesses_.fetch_add(1024,
                                                     std::memory_order_relaxed) + 1024;
    uint64_t next = next_heat_epoch_.load(std::memory_order_relaxed);
    if (accesses >= next && next_heat_epoch_.compare_exchange_strong(
            next,
            accesses + std::max<uint64_t>(1024, next_slot_.load(
                                                    std::memory_order_relaxed) * 4),
            std::memory_order_relaxed))
      heat_epoch_.fetch_add(1, std::memory_order_relaxed);
  }
  static void LegacyTouch(Fast& entry, uint8_t credit = 1) {
    const auto heat = entry.heat.load(std::memory_order_relaxed);
    if (heat != UINT8_MAX)
      entry.heat.store(std::min<unsigned>(UINT8_MAX, heat + credit),
                       std::memory_order_relaxed);
    if (!entry.referenced.load(std::memory_order_relaxed))
      entry.referenced.store(true, std::memory_order_relaxed);
  }
  void Touch(Fast& entry, uint8_t hot_credit = 1) const {
    if (!frequency_admission_) {
      LegacyTouch(entry, hot_credit);
      return;
    }
    uint32_t state = entry.frequency.load(std::memory_order_relaxed);
    for (;;) {
      uint32_t epoch = heat_epoch_.load(std::memory_order_relaxed) & kHeatEpochMask;
      if (((epoch - (state >> 8)) & kHeatEpochMask) > kHeatEpochMask / 2)
        epoch = state >> 8;
      const unsigned aged_heat = HeatAt(state, epoch);
      const unsigned credit = aged_heat < 64 ? 1 : hot_credit;
      if (credit == 0)
        break;
      const uint32_t heat = std::min<unsigned>(UINT8_MAX, aged_heat + credit);
      const uint32_t updated = (epoch << 8) | heat;
      if (state == updated || entry.frequency.compare_exchange_weak(
                                  state, updated, std::memory_order_relaxed))
        break;
    }
    if (!entry.referenced.load(std::memory_order_relaxed))
      entry.referenced.store(true, std::memory_order_relaxed);
  }
  void RecordHit(Fast& entry, Reader& reader) const {
    const uint64_t hits = reader.hits.load(std::memory_order_relaxed) + 1;
    reader.hits.store(hits, std::memory_order_relaxed);
    if (!frequency_admission_) {
      // Preserve the original sampled resident path without an access clock.
      if ((hits & 15) == 1)
        LegacyTouch(entry, 16);
      return;
    }
    RecordAccess(reader);
    // Low-frequency residents count every hit. Above that boundary, use the
    // upper Weyl bits so periodic batches do not always sample one position.
    const uint8_t hot_credit =
        ((hits * 0x9e3779b97f4a7c15ULL) >> 60) == 0 ? 16 : 0;
    Touch(entry, hot_credit);
  }
  void SuspendReads(Entry& entry) {
    if (entry.fast->readable.exchange(0) == 0)
      return;
    std::lock_guard<std::mutex> lock(readers_mutex_);
    for (const auto& reader : readers_) {
      const auto epoch = reader->batch_epoch.load();
      if (epoch & 1)
        while (reader->batch_epoch.load(std::memory_order_acquire) == epoch)
          std::this_thread::yield();
      while (reader->active.load() == entry.fast)
        std::this_thread::yield();
    }
  }
  struct ReadPublication {
    Entry& entry;
    ReadPublication(TieredClockValueStore& store, Entry& value) : entry(value) {
      store.SuspendReads(entry);
    }
    ~ReadPublication() {
      if (entry.alive && entry.initialized && !entry.ssd) {
        entry.fast->direct = entry.direct;
        entry.fast->size   = entry.size;
        entry.fast->readable.store(entry.handle);
      }
    }
  };
  struct alignas(64) Shard {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint64_t> keys;
    std::list<uint64_t> clock;
    std::vector<uint32_t> free_slots;
  };
  static constexpr uint64_t kChunkSize  = 1ULL << 16;
  static constexpr uint64_t kChunkCount = 1ULL << 16;
  struct Chunk {
    Fast fast[kChunkSize];
    Entry entries[kChunkSize];
  };

  Fast* LocateFast(uint64_t handle) const {
    const uint32_t slot = static_cast<uint32_t>(handle);
    if (slot == 0)
      return nullptr;
    Chunk* chunk =
        chunks_[(slot - 1) / kChunkSize].load(std::memory_order_acquire);
    return chunk ? chunk->fast + (slot - 1) % kChunkSize : nullptr;
  }
  Entry* Locate(uint64_t handle) const {
    const uint32_t slot = static_cast<uint32_t>(handle);
    if (slot == 0)
      return nullptr;
    Chunk* chunk =
        chunks_[(slot - 1) / kChunkSize].load(std::memory_order_acquire);
    return chunk ? chunk->entries + (slot - 1) % kChunkSize : nullptr;
  }
  static bool Matches(const Entry& entry, uint64_t handle) {
    return entry.alive && entry.handle == handle;
  }

  Entry* NewSlot(uint32_t& slot) {
    const uint64_t id = next_slot_.fetch_add(1, std::memory_order_relaxed);
    if (id >= UINT32_MAX)
      throw std::overflow_error("CLOCK metadata slots exhausted");
    slot                  = static_cast<uint32_t>(id + 1);
    const size_t chunk_id = id / kChunkSize;
    Chunk* chunk          = chunks_[chunk_id].load(std::memory_order_acquire);
    if (!chunk) {
      std::lock_guard<std::mutex> lock(directory_mutex_);
      chunk = chunks_[chunk_id].load(std::memory_order_relaxed);
      if (!chunk) {
        chunk = new Chunk;
        chunks_[chunk_id].store(chunk, std::memory_order_release);
      }
    }
    Entry* entry = chunk->entries + id % kChunkSize;
    entry->fast  = chunk->fast + id % kChunkSize;
    return entry;
  }

  void ValidateSize(uint64_t key, size_t size) const {
    if (fixed_ && (key == UINT64_MAX || ssd_->SlotCapacity(key + 1) != size))
      throw std::invalid_argument("CLOCK static row size/key mismatch");
  }

  uint64_t TryDram(size_t size) {
    uint64_t used = dram_bytes_.load(std::memory_order_relaxed);
    if (size > limit_ || used > limit_ - size)
      return kValueHandleNone;
    const uint64_t raw = dram_->Alloc(size);
    if (raw == kValueHandleNone)
      return raw;
    const size_t capacity = dram_->SlotCapacity(raw);
    while (capacity <= limit_ && used <= limit_ - capacity)
      if (dram_bytes_.compare_exchange_weak(
              used, used + capacity, std::memory_order_relaxed))
        return raw;
    dram_->Free(raw);
    return kValueHandleNone;
  }

  // Called with the shard lock. Only new/dead slots are locked in this order;
  // live-entry operations always release directory lookup locks first.
  Entry* Create(size_t shard_id,
                uint64_t key,
                bool keyed,
                size_t size,
                std::unique_lock<ValueMutex>& entry_lock) {
    Shard& shard = shards_[shard_id];
    uint32_t slot;
    Entry* entry;
    bool fresh = false;
    for (;;) {
      if (fresh || shard.free_slots.empty())
        entry = NewSlot(slot);
      else {
        slot = shard.free_slots.back();
        shard.free_slots.pop_back();
        entry = Locate(slot);
      }
      entry_lock = std::unique_lock<ValueMutex>(entry->mutex, std::try_to_lock);
      if (entry_lock)
        break;
      // A stale batch handle may still have this dead slot locked. Never wait
      // for a value lock while holding the CLOCK/key-directory lock.
      shard.free_slots.push_back(slot);
      fresh = true;
    }
    const uint64_t generation = (entry->handle >> 32) + 1;
    entry->handle             = (generation << 32) | slot;
    entry->key                = key;
    entry->keyed              = keyed;
    entry->size               = size;
    entry->shard              = shard_id;
    entry->backing            = kValueHandleNone;
    entry->dirty              = true;
    entry->initialized        = false;
    entry->fast->referenced   = false;
    entry->misses             = 0;
    entry->fast->heat         = 0;
    entry->fast->frequency    = 0;
    entry->raw                = TryDram(size);
    entry->ssd                = entry->raw == kValueHandleNone;
    if (entry->ssd)
      entry->raw = fixed_ ? key + 1 : ssd_->Alloc(size);
    if (entry->raw == kValueHandleNone) {
      if (generation < UINT32_MAX)
        shard.free_slots.push_back(slot);
      return nullptr;
    }
    if (entry->ssd)
      entry->backing = entry->raw;
    entry->capacity = (entry->ssd ? ssd_ : dram_)->SlotCapacity(entry->raw);
    entry->direct   = entry->ssd ? nullptr : dram_->DirectPtr(entry->raw);
    bool linked     = false;
    try {
      if (!entry->ssd) {
        shard.clock.push_back(entry->handle);
        entry->position = std::prev(shard.clock.end());
        linked          = true;
      }
      if (keyed)
        shard.keys.emplace(key, entry->handle);
    } catch (...) {
      if (linked)
        shard.clock.erase(entry->position);
      Release(*entry);
      if (generation < UINT32_MAX)
        shard.free_slots.push_back(slot);
      throw;
    }
    entry->alive = true;
    return entry;
  }

  void Release(const Entry& entry) {
    if (entry.ssd)
      ssd_->Free(entry.raw);
    else {
      dram_->Free(entry.raw);
      dram_bytes_.fetch_sub(entry.capacity, std::memory_order_relaxed);
      if (entry.backing != kValueHandleNone)
        ssd_->Free(entry.backing);
    }
  }

  void FreeLocked(Entry& entry) {
    ReadPublication publication(*this, entry);
    Shard& shard = shards_[entry.shard];
    std::lock_guard<std::mutex> lock(shard.mutex);
    Release(entry);
    if (!entry.ssd)
      shard.clock.erase(entry.position);
    if (entry.keyed)
      shard.keys.erase(entry.key);
    entry.alive = false;
    // Retire a slot permanently rather than wrap its generation (ABA).
    if ((entry.handle >> 32) < UINT32_MAX)
      shard.free_slots.push_back(static_cast<uint32_t>(entry.handle));
  }

  void WriteLocked(Entry& entry, const void* data, size_t size) {
    ReadPublication publication(*this, entry);
    if (size <= entry.capacity)
      (entry.ssd ? ssd_ : dram_)->Write(entry.raw, data, size);
    else {
      const uint64_t raw = ssd_->Alloc(size);
      if (raw == kValueHandleNone)
        throw std::runtime_error("CLOCK growth allocation failed");
      try {
        ssd_->Write(raw, data, size);
      } catch (...) {
        ssd_->Free(raw);
        throw;
      }
      std::lock_guard<std::mutex> lock(shards_[entry.shard].mutex);
      Release(entry);
      if (!entry.ssd)
        shards_[entry.shard].clock.erase(entry.position);
      entry.raw      = raw;
      entry.backing  = raw;
      entry.direct   = nullptr;
      entry.capacity = ssd_->SlotCapacity(raw);
      entry.ssd      = true;
    }
    entry.dirty            = !entry.ssd;
    entry.size             = size;
    entry.fast->referenced = entry.initialized;
    entry.initialized      = true;
    entry.misses           = 0;
    ++entry.version;
  }

  size_t ReadLocked(Entry& entry, void* out, size_t capacity) {
    if (!entry.initialized || capacity == 0)
      return 0;
    const size_t bytes = std::min(capacity, entry.size);
    if (!entry.ssd) {
      ++entry.stats.hits;
      if (frequency_admission_)
        RecordAccess(GetReader());
      Touch(*entry.fast);
      std::memcpy(out, entry.direct, bytes);
      return bytes;
    }
    ++entry.stats.misses;
    if (frequency_admission_)
      RecordAccess(GetReader());
    const size_t actual = ssd_->Read(entry.raw, out, bytes);
    FinishRead(entry, out, actual);
    return actual;
  }

  void FinishRead(Entry& entry,
                  const void* data,
                  size_t size,
                  std::vector<Exchange>* exchanges            = nullptr,
                  const std::unordered_set<Entry*>* scheduled = nullptr) {
    Touch(*entry.fast);
    if (entry.misses < min_misses_)
      ++entry.misses;
    if (entry.misses < min_misses_)
      ++entry.stats.rejects;
    else if (size == entry.size)
      Promote(entry, data, exchanges, scheduled);
  }

  void Promote(Entry& hot,
               const void* data,
               std::vector<Exchange>* exchanges            = nullptr,
               const std::unordered_set<Entry*>* scheduled = nullptr) {
    Shard& shard = shards_[hot.shard];
    try {
      const uint64_t free_dram = TryDram(hot.size);
      if (free_dram != kValueHandleNone) {
        ReadPublication publication(*this, hot);
        std::lock_guard<std::mutex> lock(shard.mutex);
        try {
          shard.clock.push_back(hot.handle);
        } catch (...) {
          dram_bytes_.fetch_sub(
              dram_->SlotCapacity(free_dram), std::memory_order_relaxed);
          dram_->Free(free_dram);
          throw;
        }
        dram_->Write(free_dram, data, hot.size);
        hot.direct           = dram_->DirectPtr(free_dram);
        hot.raw              = free_dram;
        hot.capacity         = dram_->SlotCapacity(free_dram);
        hot.position         = std::prev(shard.clock.end());
        hot.ssd              = false;
        hot.fast->referenced = true;
        hot.misses           = 0;
        ++hot.stats.promotions;
        ReleasePromotedBacking(hot);
        return;
      }
      std::unique_lock<std::mutex> shard_lock(shard.mutex);
      // ponytail: bounded, shard-local CLOCK. Add cross-shard balancing only
      // if traces show usable victims concentrated in other shards.
      const size_t scans = std::min<size_t>(64, shard.clock.size() * 2);
      for (size_t i = 0; i < scans; ++i) {
        auto position = shard.clock.begin();
        Entry& cold   = *Locate(*position);
        shard.clock.splice(shard.clock.end(), shard.clock, position);
        if (scheduled && scheduled->count(&cold))
          continue;
        std::unique_lock<ValueMutex> cold_lock(cold.mutex, std::try_to_lock);
        if (!cold_lock || !cold.initialized || cold.capacity < hot.size)
          continue;
        if (!frequency_admission_) {
          if (cold.fast->referenced) {
            cold.fast->referenced = false;
            cold.fast->heat.store(
                cold.fast->heat.load(std::memory_order_relaxed) / 2,
                std::memory_order_relaxed);
            continue;
          }
          if (cold.fast->heat.load(std::memory_order_relaxed) >=
              hot.fast->heat.load(std::memory_order_relaxed)) {
            cold.fast->heat.store(
                cold.fast->heat.load(std::memory_order_relaxed) / 2,
                std::memory_order_relaxed);
            continue;
          }
        } else {
          if (cold.fast->referenced.exchange(false, std::memory_order_relaxed))
            continue;
          // Age both tiers in the same access window. Failed admission scans
          // must not erase a resident's history or preserve stale SSD heat.
          const uint32_t epoch =
              heat_epoch_.load(std::memory_order_relaxed) & kHeatEpochMask;
          const unsigned cold_heat =
              HeatAt(cold.fast->frequency.load(std::memory_order_relaxed), epoch);
          const unsigned hot_heat =
              HeatAt(hot.fast->frequency.load(std::memory_order_relaxed), epoch);
          // Require a clear frequency advantage before paying for an exchange.
          // A threshold above 255 waits for aging rather than admitting a tie.
          if (hot_heat < cold_heat + std::max(1U, cold_heat / 4))
            continue;
        }
        // Keep only the two value locks during I/O. Other residents in this
        // shard remain readable/writable while the cold value is written.
        shard_lock.unlock();
        uint64_t dest        = cold.backing;
        const bool writeback = cold.dirty || dest == kValueHandleNone;
        const bool allocated =
            writeback && !fixed_ &&
            (dest == kValueHandleNone || ssd_->SlotCapacity(dest) < cold.size);
        if (writeback) {
          if (fixed_)
            dest = cold.key + 1;
          else if (allocated)
            dest = ssd_->Alloc(cold.size);
          if (dest == kValueHandleNone) {
            ++hot.stats.failures;
            return;
          }
        }
        Exchange exchange{
            &hot,
            &cold,
            data,
            dest,
            allocated,
            writeback,
            std::move(cold_lock)};
        if (exchanges) {
          // BatchRead reserves one slot per miss before entering this path.
          exchanges->push_back(std::move(exchange));
          return;
        }
        try {
          if (writeback)
            ssd_->Write(dest, cold.direct, cold.size);
          CommitExchange(exchange);
        } catch (...) {
          if (exchange.allocated)
            ssd_->Free(dest);
          throw;
        }
        return;
      }
    } catch (const std::exception&) {
      ++hot.stats.failures;
    }
  }

  void FlushExchanges(std::vector<Exchange>& exchanges) {
    std::vector<uint64_t> handles;
    std::vector<WriteSpec> specs;
    for (const auto& exchange : exchanges)
      if (exchange.writeback) {
        handles.push_back(exchange.dest);
        specs.push_back({exchange.cold->direct, exchange.cold->size});
      }
    if (!handles.empty())
      ssd_->BatchWrite(handles, specs);
    for (auto& exchange : exchanges)
      CommitExchange(exchange);
  }

  void CommitExchange(Exchange& exchange) {
    Entry& hot                = *exchange.hot;
    Entry& cold               = *exchange.cold;
    const size_t ssd_capacity = ssd_->SlotCapacity(exchange.dest);
    if (exchange.writeback) {
      if (exchange.allocated && cold.backing != kValueHandleNone)
        ssd_->Free(cold.backing);
      cold.backing       = exchange.dest;
      exchange.allocated = false;
      cold.dirty         = false;
      ++cold.stats.writebacks;
    } else {
      ++cold.stats.clean_evictions;
    }
    ReadPublication hot_publication(*this, hot);
    ReadPublication cold_publication(*this, cold);
    std::lock_guard<std::mutex> lock(shards_[hot.shard].mutex);
    const uint64_t raw    = cold.raw;
    const size_t capacity = cold.capacity;
    dram_->Write(raw, exchange.data, hot.size);
    cold.raw             = exchange.dest;
    cold.capacity        = ssd_capacity;
    cold.ssd             = true;
    cold.misses          = 0;
    hot.direct           = cold.direct;
    cold.direct          = nullptr;
    hot.raw              = raw;
    hot.capacity         = capacity;
    hot.ssd              = false;
    hot.fast->referenced = true;
    hot.misses           = 0;
    hot.position         = cold.position;
    *hot.position        = hot.handle;
    ++cold.stats.demotions;
    ++hot.stats.promotions;
    ReleasePromotedBacking(hot);
  }

  void ReleasePromotedBacking(Entry& hot) {
    // The new DRAM copy is authoritative before the SSD slot can be reused.
    // Keeping only SSD-resident values on disk reduces scattered cold reads,
    // at the cost of writing every value that is later evicted from DRAM.
    if (!retain_backing_ && hot.backing != kValueHandleNone) {
      ssd_->Free(hot.backing);
      hot.backing = kValueHandleNone;
    }
  }

  DramValueStore* dram_;
  ValueStore* ssd_;
  const uint64_t limit_;
  std::vector<Shard> shards_;
  const uint32_t min_misses_;
  const bool fixed_;
  const bool retain_backing_;
  const bool frequency_admission_;
  std::unique_ptr<std::atomic<Chunk*>[]> chunks_;
  std::mutex directory_mutex_;
  std::atomic<uint64_t> next_slot_{0}, allocation_shard_{0}, dram_bytes_{0};
  inline static std::atomic<uint64_t> next_instance_{1};
  const uint64_t instance_ = next_instance_.fetch_add(1);
  mutable std::atomic<uint64_t> heat_accesses_{0}, next_heat_epoch_{1024};
  mutable std::atomic<uint32_t> heat_epoch_{0};
  mutable std::mutex readers_mutex_;
  mutable std::vector<std::unique_ptr<Reader>> readers_;
};
