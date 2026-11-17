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
// Promotion is guarded by a write barrier. A reader captures a per-key token
// before it reads SSD and only inserts its copy if no Write, BatchWrite or Free
// bumped that key in the meantime. Without it, this interleaving leaves a stale
// value cached forever: reader fetches the old value, writer overwrites the
// slot (WriteThrough finds no entry, so it only bumps the token), reader
// inserts the old copy.
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
      shard.barriers.assign(kBarrierSlots, BarrierSlot{});
    }
  }

  // A miss optionally captures its promotion token under the same shard lock.
  // A hit copies up to `buf_size` bytes while eviction is excluded.
  bool CopyOut(uint64_t key,
               void* buf,
               size_t buf_size,
               size_t* copied,
               uint64_t* barrier = nullptr) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto it = shard.map.find(key);
    if (it == shard.map.end()) {
      ++shard.misses;
      if (barrier != nullptr) {
        *barrier = WriteBarrierLocked(shard, key);
      }
      return false;
    }
    const size_t n = std::min(buf_size, it->second.size);
    // Safe under the shard lock: an eviction of this key needs the same lock.
    store_->Read(it->second.dram_raw, buf, n);
    shard.order.splice(shard.order.begin(), shard.order, it->second.pos);
    ++shard.hits;
    if (copied != nullptr) {
      *copied = n;
    }
    return true;
  }

  // Token for the promotion barrier, captured before reading the SSD slot. Pass
  // it to Promote(); a copy is only admitted if nothing wrote to `key` since.
  uint64_t WriteBarrier(uint64_t key) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    return WriteBarrierLocked(shard, key);
  }

  // Caches a copy of an SSD value, evicting this shard's LRU entry when the
  // cache is already full. Silently does nothing when the value cannot fit in
  // one shard, when DRAM has no room left for the cache, or when `barrier` is
  // no longer the key's current write token (a write raced the SSD read).
  void Promote(uint64_t key, const void* data, size_t size, uint64_t barrier) {
    if (size == 0) {
      return;
    }
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    const BarrierSlot& barrier_slot = BarrierSlotOf(shard, key);
    if (barrier_slot.key != key || barrier_slot.seq != barrier) {
      ++shard.stale_promotions;
      return;
    }
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
        ++shard.admission_rejects;
        return;
      }
      if (++slot.count < promote_min_misses_) {
        ++shard.admission_rejects;
        return;
      }
      slot.key   = 0;
      slot.count = 0;
    }
    // The budget is a share of DRAM; the total limit is what primary placement
    // and the cache together may occupy. Whichever binds first is the cap.
    const uint64_t shard_cap = EffectiveBudgetBytes() / shards_.size();
    if (size > shard_cap) {
      ++shard.alloc_failures;
      return;
    }
    // Free room before allocating: a full cache must replace, not refuse, or it
    // would keep the first values it ever saw and never admit a newer hot one.
    while (shard.bytes + size > shard_cap && !shard.order.empty()) {
      EraseLocked(shard, shard.map.find(shard.order.back()));
      ++shard.evictions;
    }
    const uint64_t dram_raw = store_->Alloc(size);
    if (dram_raw == kValueHandleNone) {
      ++shard.alloc_failures;
      return;
    }
    store_->Write(dram_raw, data, size);
    const size_t slot_bytes = store_->SlotCapacity(dram_raw);
    shard.order.push_front(key);
    shard.map.emplace(
        key, Entry{dram_raw, std::min(size, slot_bytes), shard.order.begin()});
    shard.bytes += slot_bytes;
    ++shard.entries;
    ++shard.promotions;
    TrimLocked(shard, shard_cap);
  }

  // Drops a cached copy (the SSD handle is being freed).
  void Invalidate(uint64_t key) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    // A reader that already fetched this slot must not reinsert its copy after
    // the slot is handed out again.
    BumpBarrierLocked(shard, key);
    // Also forget the miss count: the SSD slot gets reused, and a leftover
    // count would let a cold value inherit another value's history.
    AdmissionSlot& slot = AdmissionSlotOf(shard, key);
    if (slot.key == key) {
      slot = AdmissionSlot{};
    }
    EraseLocked(shard, shard.map.find(key));
  }

  // Keeps a cached copy consistent with an in-place write. The entry tracks the
  // byte count it now holds, so a cached read returns exactly what the SSD read
  // would have returned for the same slot (a shrunken value must not expose the
  // tail of the previous one).
  void WriteThrough(uint64_t key, const void* data, size_t size) {
    Shard& shard = ShardOf(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    // Bump before touching the entry: the write may be visible in the SSD store
    // while a concurrent reader is still holding the old copy.
    BumpBarrierLocked(shard, key);
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
    it->second.size = size;
    shard.order.splice(shard.order.begin(), shard.order, it->second.pos);
  }

  uint64_t Bytes() const { return Sum(&Shard::bytes); }
  uint64_t Entries() const { return Sum(&Shard::entries); }
  uint64_t Hits() const { return Sum(&Shard::hits); }
  uint64_t Misses() const { return Sum(&Shard::misses); }
  uint64_t Promotions() const { return Sum(&Shard::promotions); }
  uint64_t Evictions() const { return Sum(&Shard::evictions); }
  uint64_t AllocFailures() const { return Sum(&Shard::alloc_failures); }
  // Misses that were not admitted because the key had not repeated enough
  // times.
  uint64_t AdmissionRejects() const { return Sum(&Shard::admission_rejects); }
  // Copies discarded because a write or free touched the key while they were
  // being read from SSD.
  uint64_t StalePromotions() const { return Sum(&Shard::stale_promotions); }
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

  struct BarrierSlot {
    uint64_t key = 0;
    uint64_t seq = 0;
  };

  struct alignas(64) Shard {
    mutable std::mutex mutex;
    std::list<uint64_t> order;
    std::unordered_map<uint64_t, Entry> map;
    // Direct-mapped miss sample for `promote_min_misses_ > 1`: fixed size, no
    // allocation on the miss path. Collisions overwrite, which only costs
    // accuracy for keys that would not have repeated anyway.
    std::vector<AdmissionSlot> admission;
    // Direct-mapped write tokens for promotion, same trade-off as `admission`:
    // a collision only causes a missed promotion (conservative), never a stale
    // one, because every assignment takes a fresh monotonically increasing seq.
    std::vector<BarrierSlot> barriers;
    uint64_t bytes             = 0;
    uint64_t entries           = 0;
    uint64_t next_barrier      = 0;
    uint64_t hits              = 0;
    uint64_t misses            = 0;
    uint64_t promotions        = 0;
    uint64_t evictions         = 0;
    uint64_t alloc_failures    = 0;
    uint64_t admission_rejects = 0;
    uint64_t stale_promotions  = 0;
  };

  Shard& ShardOf(uint64_t key) {
    return shards_[static_cast<size_t>(key % shards_.size())];
  }

  AdmissionSlot& AdmissionSlotOf(Shard& shard, uint64_t key) {
    const size_t stride = (key * 0x9E3779B97F4A7C15ULL) >> 32;
    return shard.admission[stride % shard.admission.size()];
  }

  BarrierSlot& BarrierSlotOf(Shard& shard, uint64_t key) {
    const size_t stride = (key * 0x9E3779B97F4A7C15ULL) >> 32;
    return shard.barriers[stride % shard.barriers.size()];
  }

  uint64_t Sum(uint64_t Shard::*field) const {
    uint64_t total = 0;
    for (const Shard& shard : shards_) {
      std::lock_guard<std::mutex> lock(shard.mutex);
      total += shard.*field;
    }
    return total;
  }

  uint64_t WriteBarrierLocked(Shard& shard, uint64_t key) {
    BarrierSlot& slot = BarrierSlotOf(shard, key);
    if (slot.key != key) {
      // Keys always use the same shard; a fresh shard sequence prevents ABA
      // when unrelated keys collide in the fixed-size token table.
      slot.key = key;
      slot.seq = ++shard.next_barrier;
    }
    return slot.seq;
  }

  void BumpBarrierLocked(Shard& shard, uint64_t key) {
    BarrierSlot& slot = BarrierSlotOf(shard, key);
    slot.key          = key;
    slot.seq          = ++shard.next_barrier;
  }

  // Evicts this shard's LRU entries until it fits its share of the budget.
  void TrimLocked(Shard& shard, uint64_t shard_cap) {
    while (shard.bytes > shard_cap && !shard.order.empty()) {
      EraseLocked(shard, shard.map.find(shard.order.back()));
      ++shard.evictions;
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
    --shard.entries;
  }

  // Slots per shard in the direct-mapped admission sample. Sized well above the
  // number of entries a shard can hold, so a resident value's history is rarely
  // overwritten by a colliding key.
  static constexpr size_t kAdmissionSlots = 1024;
  // Same sizing rule for the promotion write barriers.
  static constexpr size_t kBarrierSlots = 1024;

  DramValueStore* store_;
  std::vector<Shard> shards_;
  const uint64_t budget_bytes_;
  const uint64_t dram_limit_bytes_;
  const std::atomic<uint64_t>* reserved_bytes_;
  const uint32_t promote_min_misses_;
};
