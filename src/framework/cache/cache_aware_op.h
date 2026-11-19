#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <glog/logging.h>

namespace recstore::framework {

// Monotonic epoch bumped by GPU cache lifecycle calls that are meant to reset
// the bypass policy: enable (when the cache is rebuilt), disable and reset.
// CacheAwareOp is thread_local, so a reset performed on one thread must also
// clear the bypass latch left behind on other threads. An internal fallback
// clear does not bump it. SyncEpoch() only notices the bump on the next lookup
// of a thread, so this is best-effort, not a synchronization barrier.
inline std::atomic<uint64_t>& CacheAwareEpoch() {
  static std::atomic<uint64_t> epoch{0};
  return epoch;
}

// Callbacks adapt buffers/devices; ordering, bypass and fallback belong here.
struct CacheLookupActions {
  bool usable = false;
  std::function<int64_t()> query;
  std::function<void()> read_missing, fill, scatter, read_all, clear;
};

struct CacheUpdateActions {
  std::function<bool()> enabled, can_invalidate;
  std::function<void()> invalidate, clear;
};

class CacheAwareOp {
public:
  bool BypassEnabled() const { return bypass_enabled_; }
  bool Bypassed() const { return bypassed_; }
  void Reset() {
    low_hit_streak_      = 0;
    bypassed_            = false;
    probes_since_bypass_ = 0;
  }
  void SetBypassEnabled(bool enabled) {
    bypass_enabled_ = enabled;
    if (!enabled)
      Reset();
  }

  static void ClearNoThrow(const std::function<void()>& clear) {
    try {
      if (clear)
        clear();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to clear cache: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to clear cache";
    }
  }

  static void AfterReload(bool success, const CacheUpdateActions& io) {
    if (success && io.enabled && io.enabled())
      ClearNoThrow(io.clear);
  }

  void
  Lookup(int64_t rows, const CacheLookupActions& io, bool no_evict = false) {
    if (no_evict && !io.usable)
      throw std::runtime_error("No-eviction lookup requires an enabled GPU "
                               "cache on the keys' device");
    SyncEpoch();
    bool use_cache = io.usable;
    if (!no_evict && use_cache && bypass_enabled_ && bypassed_ &&
        rows >= kMinRows) {
      // Stay off the cache for a while, then probe again. Entering bypass never
      // clears the cache: the resident rows stay warm for the probe, so a
      // controller that disabled the policy keeps whatever it cached.
      if (++probes_since_bypass_ < kReprobeInterval) {
        use_cache = false;
      }
    }
    if (use_cache) {
      try {
        const auto missing = io.query();
        if (missing < 0 || missing > rows)
          throw std::runtime_error("Invalid cache miss count");
        if (rows >= kMinRows) {
          if (double(rows - missing) / rows < 0.05) {
            ++low_hit_streak_;
            if (low_hit_streak_ >= kLowHitStreakLimit) {
              // Trip on the next lookup, so this batch still served whatever
              // the cache had. Two consecutive low-hit batches are required:
              // a cold start must not disable the cache after one batch.
              bypassed_            = true;
              probes_since_bypass_ = 0;
            }
          } else {
            Reset();
          }
        }
        if (missing) {
          io.read_missing();
          io.fill();
          io.scatter();
        }
        return;
      } catch (...) {
        if (no_evict)
          throw;
        LOG(WARNING)
            << "Cache lookup failed; clearing cache and reading from backend";
        ClearNoThrow(io.clear);
      }
    }
    io.read_all();
  }

  // Bypass no longer clears the cache, so every write that reaches the cache
  // must invalidate here; there is no bulk clear left to hide a missed policy
  // hook. `rows` is kept for symmetry with Lookup and future heuristics.
  void AfterUpdate(int64_t rows,
                   const CacheUpdateActions& io,
                   bool force_invalidation = false) {
    (void)rows;
    if (!io.enabled || !io.enabled())
      return;
    SyncEpoch();
    // A controller with bypass disabled owns its cache coherence protocol.
    if (!force_invalidation && !bypass_enabled_)
      return;
    // Keep invalidating while bypassed: the cache is not cleared on bypass and
    // may be probed again, so it must not serve rows updated in the meantime.
    if (io.can_invalidate && io.can_invalidate()) {
      try {
        io.invalidate();
        return;
      } catch (...) {
        LOG(WARNING) << "Cache invalidation failed after backend update";
      }
    }
    ClearNoThrow(io.clear);
  }

  void Update(int64_t rows,
              const CacheUpdateActions& io,
              const std::function<void()>& update) {
    try {
      update();
    } catch (...) {
      ClearNoThrow(io.clear);
      throw;
    }
    AfterUpdate(rows, io);
  }

private:
  void SyncEpoch() {
    const uint64_t epoch = CacheAwareEpoch().load(std::memory_order_acquire);
    if (epoch != epoch_) {
      epoch_ = epoch;
      Reset();
    }
  }

  static constexpr int64_t kMinRows       = 1024;
  static constexpr int kLowHitStreakLimit = 2;
  static constexpr int kReprobeInterval   = 32;
  uint64_t epoch_                         = 0;
  bool bypass_enabled_ = true, bypassed_ = false;
  int low_hit_streak_      = 0;
  int probes_since_bypass_ = 0;
};

inline thread_local CacheAwareOp g_cache_aware_op;

// Opaque adapter callbacks retain the input buffers needed for invalidation.
// Transport completion must precede cache maintenance; failures clear the
// cache.
class PendingCacheUpdates {
public:
  void Add(uint64_t id, int64_t rows, CacheUpdateActions actions) {
    std::lock_guard<std::mutex> guard(mu_);
    if (!pending_.emplace(id, Pending{rows, std::move(actions)}).second)
      throw std::runtime_error("Duplicate asynchronous update handle");
  }
  void Wait(uint64_t id, const std::function<void()>& wait) {
    Pending pending;
    {
      std::lock_guard<std::mutex> guard(mu_);
      auto it = pending_.find(id);
      if (it == pending_.end())
        throw std::runtime_error("Unknown asynchronous update handle");
      pending = std::move(it->second);
      pending_.erase(it);
    }
    try {
      wait();
    } catch (...) {
      CacheAwareOp::ClearNoThrow(pending.actions.clear);
      throw;
    }
    g_cache_aware_op.AfterUpdate(pending.rows, pending.actions);
  }

private:
  struct Pending {
    int64_t rows = 0;
    CacheUpdateActions actions;
  };
  std::mutex mu_;
  std::unordered_map<uint64_t, Pending> pending_;
};

} // namespace recstore::framework
