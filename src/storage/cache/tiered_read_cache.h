#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "storage/value_store/dram_value_store.h"

// Access-driven DRAM read cache for the tiered value store.
//
// The tiered store picks DRAM or SSD once, in Alloc(), so the DRAM layer ends
// up holding whatever was written first instead of what is read most. This
// cache keeps a second copy of hot SSD values in DRAM, keyed by the raw SSD
// handle, so a value that was placed on SSD can still be served from DRAM after
// it has been read once.
//
// It is sharded (one mutex, one LRU list and one byte cap per shard) because a
// single mutex turns every DRAM hit into a serialization point. A hit copies
// out while holding its shard lock: the copy is a memcpy of one value, and
// holding the lock is what makes evicting the same slot safe.
//
// The cap per shard is `min(configured budget, shared DRAM watermark - primary
// placement) / shards`, so the cache and primary placement together stay inside
// one watermark instead of each assuming it owns the whole DRAM layer. A full
// shard evicts its LRU entry before admitting a new one; refusing instead would
// freeze the cache on the first values it ever saw.
//
// `promote_min_misses` filters admission: with 2, a key is only copied into
// DRAM on its second miss. One-shot misses (a scan, a cold tail) would
// otherwise each cost a DRAM slot and evict a genuinely hot entry, which is
// what keeps the hit rate far below what the workload's frequency skew allows.
//
// Deliberately not built on `storage/cache/lru/lru_cache.h`: that module tracks
// keys only (no values, no byte budget, no atomic lookup-and-copy), so it
// cannot serve a read path.
class TieredReadCache {
public:
  TieredReadCache(DramValueStore* store,
                  size_t shard_count,
                  uint64_t budget_bytes,
                  uint64_t dram_limit_bytes                   = UINT64_MAX,
                  const std::atomic<uint64_t>* reserved_bytes = nullptr,
                  uint32_t promote_min_misses                 = 2)
      : store_(store),
        shards_(std::max<size_t>(1, shard_count)),
        budget_bytes_(budget_bytes),
        dram_limit_bytes_(dram_limit_bytes),
        reserved_bytes_(reserved_bytes),
        promote_min_misses_(std::max<uint32_t>(1, promote_min_misses)) {
    for (Shard& shard : shards_) {
      shard.admission.assign(kAdmissionSlots, AdmissionSlot{});
    }
  }

  // Returns true when `key` was cached, copying up to `buf_size` bytes out.
  bool CopyOut(uint64_t key, void* buf, size_t buf_size, size_t* copied) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto it = shard.map.find(key);
    if (it == shard.map.end()) {
      misses_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    const size_t n = std::min(buf_size, it->second.size);
    // Safe under the shard lock: an eviction of this key needs the same lock.
    store_->Read(it->second.dram_raw, buf, n);
    shard.order.splice(shard.order.begin(), shard.order, it->second.pos);
    hits_.fetch_add(1, std::memory_order_relaxed);
    if (copied != nullptr) {
      *copied = n;
    }
    return true;
  }

  // Caches a copy of an SSD value, evicting this shard's LRU entry when the
  // cache is already full. Silently does nothing when the value cannot fit in
  // one shard or when DRAM has no room left for the cache.
  void Promote(uint64_t key, const void* data, size_t size) {
    if (size == 0) {
      return;
    }
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto existing = shard.map.find(key);
    if (existing != shard.map.end()) {
      // Another thread promoted this key first; keep its copy, refresh recency.
      shard.order.splice(
          shard.order.begin(), shard.order, existing->second.pos);
      return;
    }
    if (promote_min_misses_ > 1) {
      // Count the miss and admit only once the key has proven it repeats. The
      // sample is a direct-mapped table: no allocation, no rehash and no
      // teardown on the miss path, which matters because with a flat workload
      // nearly every access lands here.
      AdmissionSlot& slot = AdmissionSlotOf(shard, key);
      if (slot.key != key) {
        slot.key   = key;
        slot.count = 1;
        admission_rejects_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (++slot.count < promote_min_misses_) {
        admission_rejects_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      slot.key   = 0;
      slot.count = 0;
    }
    // The budget is a share of DRAM; the total limit is what primary placement
    // and the cache together may occupy. Whichever binds first is the cap.
    const uint64_t shard_cap = EffectiveBudgetBytes() / shards_.size();
    if (size > shard_cap) {
      alloc_failures_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    // Free room before allocating: a full cache must replace, not refuse, or it
    // would keep the first values it ever saw and never admit a newer hot one.
    while (shard.bytes + size > shard_cap && !shard.order.empty()) {
      EraseLocked(shard, shard.map.find(shard.order.back()));
      evictions_.fetch_add(1, std::memory_order_relaxed);
    }
    const uint64_t dram_raw = store_->Alloc(size);
    if (dram_raw == kValueHandleNone) {
      alloc_failures_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    store_->Write(dram_raw, data, size);
    const size_t slot_bytes = store_->SlotCapacity(dram_raw);
    shard.order.push_front(key);
    shard.map.emplace(
        key, Entry{dram_raw, std::min(size, slot_bytes), shard.order.begin()});
    shard.bytes += slot_bytes;
    bytes_.fetch_add(slot_bytes, std::memory_order_relaxed);
    entries_.fetch_add(1, std::memory_order_relaxed);
    promotions_.fetch_add(1, std::memory_order_relaxed);
    TrimLocked(shard, shard_cap);
  }

  // Drops a cached copy (the SSD handle is being freed).
  void Invalidate(uint64_t key) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    // Also forget the miss count: the SSD slot gets reused, and a leftover
    // count would let a cold value inherit another value's history.
    AdmissionSlot& slot = AdmissionSlotOf(shard, key);
    if (slot.key == key) {
      slot = AdmissionSlot{};
    }
    EraseLocked(shard, shard.map.find(key));
  }

  // Keeps a cached copy consistent with an in-place write. The entry keeps the
  // byte count it was promoted with, so a cached read returns exactly what the
  // SSD read would have returned for the same slot.
  void WriteThrough(uint64_t key, const void* data, size_t size) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto it = shard.map.find(key);
    if (it == shard.map.end()) {
      return;
    }
    if (size > store_->SlotCapacity(it->second.dram_raw)) {
      // The cached slot is too small for the new value; dropping the copy is
      // cheaper than re-promoting with a bigger one.
      EraseLocked(shard, it);
      return;
    }
    store_->Write(it->second.dram_raw, data, size);
    shard.order.splice(shard.order.begin(), shard.order, it->second.pos);
  }

  uint64_t Bytes() const { return bytes_.load(std::memory_order_relaxed); }
  uint64_t Entries() const { return entries_.load(std::memory_order_relaxed); }
  uint64_t Hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t Misses() const { return misses_.load(std::memory_order_relaxed); }
  uint64_t Promotions() const {
    return promotions_.load(std::memory_order_relaxed);
  }
  uint64_t Evictions() const {
    return evictions_.load(std::memory_order_relaxed);
  }
  uint64_t AllocFailures() const {
    return alloc_failures_.load(std::memory_order_relaxed);
  }
  // Misses that were not admitted because the key had not repeated enough
  // times.
  uint64_t AdmissionRejects() const {
    return admission_rejects_.load(std::memory_order_relaxed);
  }
  uint32_t PromoteMinMisses() const { return promote_min_misses_; }
  uint64_t BudgetBytes() const { return budget_bytes_; }
  uint64_t DramLimitBytes() const { return dram_limit_bytes_; }
  size_t ShardCount() const { return shards_.size(); }

  // Budget actually in force: the configured budget, or what is left of the
  // shared DRAM limit once primary placement is accounted for, whichever is
  // smaller. It moves as primary placement grows.
  uint64_t EffectiveBudgetBytes() const {
    uint64_t cap = budget_bytes_;
    if (reserved_bytes_ != nullptr) {
      const uint64_t reserved =
          reserved_bytes_->load(std::memory_order_relaxed);
      const uint64_t allowed =
          dram_limit_bytes_ > reserved ? dram_limit_bytes_ - reserved : 0;
      cap = std::min(cap, allowed);
    }
    return cap;
  }

  // Frees every cached copy. Only safe once the store stops serving reads.
  void Clear() {
    for (Shard& shard : shards_) {
      std::lock_guard<std::mutex> lock(shard.mutex);
      std::fill(
          shard.admission.begin(), shard.admission.end(), AdmissionSlot{});
      while (!shard.order.empty()) {
        EraseLocked(shard, shard.map.find(shard.order.back()));
      }
    }
  }

private:
  struct Entry {
    uint64_t dram_raw = kValueHandleNone;
    size_t size       = 0;
    std::list<uint64_t>::iterator pos;
  };

  struct AdmissionSlot {
    uint64_t key   = 0;
    uint32_t count = 0;
  };

  struct Shard {
    std::mutex mutex;
    std::list<uint64_t> order;
    std::unordered_map<uint64_t, Entry> map;
    // Direct-mapped miss sample for `promote_min_misses_ > 1`: fixed size, no
    // allocation on the miss path. Collisions overwrite, which only costs
    // accuracy for keys that would not have repeated anyway.
    std::vector<AdmissionSlot> admission;
    size_t bytes = 0;
  };

  Shard& ShardOf(uint64_t key) {
    return shards_[static_cast<size_t>(key % shards_.size())];
  }

  AdmissionSlot& AdmissionSlotOf(Shard& shard, uint64_t key) {
    const size_t stride = (key * 0x9E3779B97F4A7C15ULL) >> 32;
    return shard.admission[stride % shard.admission.size()];
  }

  // Evicts this shard's LRU entries until it fits its share of the budget.
  void TrimLocked(Shard& shard, uint64_t shard_cap) {
    while (shard.bytes > shard_cap && !shard.order.empty()) {
      EraseLocked(shard, shard.map.find(shard.order.back()));
      evictions_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void EraseLocked(Shard& shard,
                   std::unordered_map<uint64_t, Entry>::iterator it) {
    if (it == shard.map.end()) {
      return;
    }
    const size_t slot_bytes = store_->SlotCapacity(it->second.dram_raw);
    store_->Free(it->second.dram_raw);
    shard.order.erase(it->second.pos);
    shard.map.erase(it);
    shard.bytes -= std::min(shard.bytes, slot_bytes);
    bytes_.fetch_sub(
        std::min<uint64_t>(Bytes(), slot_bytes), std::memory_order_relaxed);
    entries_.fetch_sub(1, std::memory_order_relaxed);
  }

  // Slots per shard in the direct-mapped admission sample. Sized well above the
  // number of entries a shard can hold, so a resident value's history is rarely
  // overwritten by a colliding key.
  static constexpr size_t kAdmissionSlots = 1024;

  DramValueStore* store_;
  std::vector<Shard> shards_;
  const uint64_t budget_bytes_;
  const uint64_t dram_limit_bytes_;
  const std::atomic<uint64_t>* reserved_bytes_;
  const uint32_t promote_min_misses_;
  std::atomic<uint64_t> bytes_{0};
  std::atomic<uint64_t> entries_{0};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> promotions_{0};
  std::atomic<uint64_t> evictions_{0};
  std::atomic<uint64_t> alloc_failures_{0};
  std::atomic<uint64_t> admission_rejects_{0};
};
