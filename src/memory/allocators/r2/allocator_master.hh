#pragma once

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>

#ifndef JEMALLOC_NO_DEMANGLE
#define JEMALLOC_NO_DEMANGLE
#endif
#include <jemalloc/jemalloc.h>
#include "allocator.hh"

namespace r2 {

#ifndef container_of
#  define container_of(ptr, type, member)                                      \
    ((type*)((char*)(ptr)-offsetof(type, member)))
#endif

class AllocatorMaster {
public:
  AllocatorMaster() = default;

  ~AllocatorMaster() {
    shutting_down_.store(true, std::memory_order_release);

    auto& tls = tls_allocs();
    if (auto it = tls.find(this); it != tls.end()) {
      delete it->second;
      tls.erase(it);
    }

    (void)je_mallctl("thread.tcache.flush", nullptr, nullptr, nullptr, 0);
    std::vector<Entry> entries;
    {
      std::lock_guard<std::mutex> g(created_mu_);
      entries.swap(created_);
    }

    for (auto& e : entries) {
      if (e.tcache != 0) {
        size_t z = 0;
        (void)je_mallctl("tcache.destroy", &e.tcache, &z, nullptr, 0);
      }
    }

    for (auto& e : entries) {
      if (!e.owns_arena)
        continue;

      char purge[64];
      std::snprintf(purge, sizeof(purge), "arena.%u.purge", e.arena);
      (void)je_mallctl(purge, nullptr, nullptr, nullptr, 0);

      char d1[64];
      std::snprintf(d1, sizeof(d1), "arena.%u.destroy", e.arena);
      char d2[64];
      std::snprintf(d2, sizeof(d2), "arenas.%u.destroy", e.arena);
      int rc1 = je_mallctl(d1, nullptr, nullptr, nullptr, 0);
      int rc2 = (rc1 == 0) ? 0 : je_mallctl(d2, nullptr, nullptr, nullptr, 0);

      if (rc1 == 0 || rc2 == 0) {
        if (e.pack) {
          {
            std::lock_guard<std::mutex> gp(packs_mu());
            packs().erase(e.pack);
          }
          delete e.pack;
          e.pack = nullptr;
        }
      }
    }

    std::lock_guard<std::mutex> guard(lock_);
    start_addr_ = end_addr_ = heap_top_ = nullptr;
  }

  AllocatorMaster(const AllocatorMaster&)            = delete;
  AllocatorMaster& operator=(const AllocatorMaster&) = delete;
  AllocatorMaster(AllocatorMaster&&)                 = delete;
  AllocatorMaster& operator=(AllocatorMaster&&)      = delete;

  void init(char* mem, uint64_t mem_size) {
    std::lock_guard<std::mutex> guard(lock_);
    if (total_managed_mem() != 0)
      return;
    start_addr_ = mem;
    end_addr_   = mem + mem_size;
    heap_top_   = start_addr_;
  }

  Allocator* get_thread_allocator() {
    auto& tls = tls_allocs();
    if (auto it = tls.find(this); it != tls.end())
      return it->second;
    Allocator* al = get_allocator();
    if (!al) {
      // Fallback to an already-created arena so one failed arena creation
      // does not make all allocations on this thread fail.
      al = get_fallback_allocator();
    }
    if (!al)
      return nullptr;
    tls.emplace(this, al);
    return al;
  }

  Allocator* get_allocator() {
    if (shutting_down_.load(std::memory_order_acquire)) {
      LOG(INFO) << "has benn shutting down";
      return nullptr;
    }
    {
      std::lock_guard<std::mutex> g(lock_);
      if (memory_usblae() == 0) {
        // LOG(INFO) << "R2 allocator has no memory!";
        return nullptr;
      }
    }

    unsigned arena_id = 0, cache_id = 0;
    size_t olen = sizeof(unsigned);

    if (!ensure_shared_arena(&arena_id)) {
      return nullptr;
    }

    int e = je_mallctl("tcache.create", &cache_id, &olen, nullptr, 0);
    if (e != 0) {
      LOG(INFO) << "tcache.create fail!";
      return new Allocator(MALLOCX_ARENA(arena_id));
    }

    record_entry(arena_id, cache_id, /*pack=*/nullptr, /*owns_arena=*/false);
    return new Allocator(MALLOCX_ARENA(arena_id) | MALLOCX_TCACHE(cache_id));
  }

  uint64_t total_managed_mem() const {
    return static_cast<uint64_t>(end_addr_ - start_addr_);
  }

  uint64 memory_usblae() const {
    return (uint64_t)end_addr_ - (uint64_t)heap_top_;
  }

  bool within_range(ptr_t p) const {
    char* c = static_cast<char*>(p);
    return (c >= start_addr_) && (c < end_addr_);
  }

private:
  char* start_addr_ = nullptr;
  char* end_addr_   = nullptr;
  char* heap_top_   = nullptr;
  mutable std::mutex lock_;
  std::atomic<bool> shutting_down_{false};

  struct HooksPack {
    extent_hooks_t hooks{};
    AllocatorMaster* owner{nullptr};
  };

  struct Entry {
    unsigned arena{0};
    unsigned tcache{0};
    HooksPack* pack{nullptr};
    bool owns_arena{false};
  };

  void record_entry(
      unsigned arena, unsigned tcache, HooksPack* pack, bool owns_arena) {
    std::lock_guard<std::mutex> g(created_mu_);
    created_.push_back(Entry{arena, tcache, pack, owns_arena});
  }

  std::mutex created_mu_;
  std::vector<Entry> created_;
  bool shared_arena_created_{false};
  unsigned shared_arena_{0};

  bool ensure_shared_arena(unsigned* arena_id) {
    std::lock_guard<std::mutex> g(created_mu_);
    if (shared_arena_created_) {
      *arena_id = shared_arena_;
      return true;
    }

    HooksPack* pack          = new HooksPack();
    pack->owner              = this;
    pack->hooks.alloc        = &AllocatorMaster::extent_alloc_hook;
    pack->hooks.dalloc       = &AllocatorMaster::extent_dalloc_hook;
    pack->hooks.destroy      = &AllocatorMaster::extent_destroy_hook;
    pack->hooks.commit       = &AllocatorMaster::extent_commit_hook;
    pack->hooks.decommit     = &AllocatorMaster::extent_decommit_hook;
    pack->hooks.purge_lazy   = &AllocatorMaster::extent_purge_lazy_hook;
    pack->hooks.purge_forced = &AllocatorMaster::extent_purge_forced_hook;
    pack->hooks.split        = &AllocatorMaster::extent_split_hook;
    pack->hooks.merge        = &AllocatorMaster::extent_merge_hook;

    {
      std::lock_guard<std::mutex> gp(packs_mu());
      packs().emplace(pack);
    }

    unsigned new_arena_id     = 0;
    size_t olen               = sizeof(unsigned);
    extent_hooks_t* hooks_ptr = &pack->hooks;
    int e                     = je_mallctl(
        "arenas.create",
        &new_arena_id,
        &olen,
        &hooks_ptr,
        sizeof(extent_hooks_t*));
    if (e != 0) {
      LOG(INFO) << "arenas.create fail, err=" << e
                << " memory_usable=" << memory_usblae();
      {
        std::lock_guard<std::mutex> gp(packs_mu());
        packs().erase(pack);
      }
      delete pack;
      return false;
    }

    shared_arena_created_ = true;
    shared_arena_         = new_arena_id;
    created_.push_back(
        Entry{new_arena_id, /*tcache=*/0, pack, /*owns_arena=*/true});
    *arena_id = new_arena_id;
    return true;
  }

  Allocator* get_fallback_allocator() {
    std::lock_guard<std::mutex> g(created_mu_);
    if (!shared_arena_created_) {
      return nullptr;
    }
    return new Allocator(MALLOCX_ARENA(shared_arena_) | MALLOCX_TCACHE_NONE);
  }

  static std::unordered_set<HooksPack*>& packs() {
    static std::unordered_set<HooksPack*> s;
    return s;
  }
  static std::mutex& packs_mu() {
    static std::mutex m;
    return m;
  }

  static std::atomic<uint64_t>& zero_extent_requests() {
    static std::atomic<uint64_t> c{0};
    return c;
  }

  static thread_local std::unordered_map<const AllocatorMaster*, Allocator*>&
  tls_allocs() {
    static thread_local std::unordered_map<const AllocatorMaster*, Allocator*>
        m;
    return m;
  }

  static AllocatorMaster* owner_from_hooks(extent_hooks_t* eh) {
    HooksPack* pack = container_of(eh, HooksPack, hooks);
    return pack->owner;
  }

  static void* extent_alloc_hook(
      extent_hooks_t* eh,
      void*,
      size_t size,
      size_t alignment,
      bool* zero,
      bool*,
      unsigned) {
    AllocatorMaster* self = owner_from_hooks(eh);
    if (!self)
      return nullptr;

    std::lock_guard<std::mutex> guard(self->lock_);
    uintptr_t p     = reinterpret_cast<uintptr_t>(self->heap_top_);
    uintptr_t ret_u = (p + (alignment - 1)) & ~(alignment - 1);
    char* ret       = reinterpret_cast<char*>(ret_u);

    if (ret + size > self->end_addr_)
      return nullptr;

    self->heap_top_ = ret + size;
    if (*zero) {
      uint64_t n =
          zero_extent_requests().fetch_add(1, std::memory_order_relaxed) + 1;
      LOG(INFO) << "R2 extent_alloc_hook zero-filled extent, count=" << n
                << ", size=" << size;
      std::memset(ret, 0, size);
    }
    return ret;
  }

  static bool
  extent_dalloc_hook(extent_hooks_t*, void*, size_t, bool, unsigned) {
    return true;
  }
  static void
  extent_destroy_hook(extent_hooks_t*, void*, size_t, bool, unsigned) {}
  static bool
  extent_commit_hook(extent_hooks_t*, void*, size_t, size_t, size_t, unsigned) {
    return false;
  }
  static bool extent_decommit_hook(
      extent_hooks_t*, void*, size_t, size_t, size_t, unsigned) {
    return false;
  }
  static bool extent_purge_lazy_hook(
      extent_hooks_t*, void*, size_t, size_t, size_t, unsigned) {
    return true;
  }
  static bool extent_purge_forced_hook(
      extent_hooks_t*, void*, size_t, size_t, size_t, unsigned) {
    return true;
  }
  static bool extent_split_hook(
      extent_hooks_t*, void*, size_t, size_t, size_t, bool, unsigned) {
    return false;
  }
  static bool extent_merge_hook(
      extent_hooks_t*, void*, size_t, void*, size_t, bool, unsigned) {
    return false;
  }
};

} // namespace r2
