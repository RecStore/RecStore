#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <shared_mutex>
#include "../index.h"
#include "base/factory.h"
#include "storage/kv_engine/base_kv.h"
#include "../utils/util.h"
#define LSB

// Diagnostic counters for the insert path; only maintained when
// RECSTORE_EXTENDIBLE_HASH_COUNTERS is set (see extendible_hash.cpp).
struct ExtendibleHashInsertCounters {
  std::uint64_t insert_calls = 0;
  std::uint64_t split_calls  = 0;
  std::uint64_t split_ns     = 0;
};

void GetExtendibleHashInsertCounters(ExtendibleHashInsertCounters* out);

const size_t kMask  = 256 - 1;
const size_t kShift = 8;

struct Block {
  static const size_t kBlockSize = 256; // 4 - 1
  // static const size_t kBlockSize = 1024; // 16 - 1
  // static const size_t kBlockSize = 4*1024; // 64 - 1
  // static const size_t kBlockSize = 16*1024; // 256 - 1
  // static const size_t kBlockSize = 64*1024; // 1024 - 1
  // static const size_t kBlockSize = 256 * 1024; // 4096 - 1
  static const size_t kNumSlot = kBlockSize / sizeof(Pair);

  Block(void) : local_depth{0} {}

  Block(size_t depth) : local_depth{depth} {}

  ~Block(void) {}

  void* operator new(size_t size) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 64, size) != 0 || ret == nullptr)
      throw std::bad_alloc();
    LiveBlockCount().fetch_add(1, std::memory_order_relaxed);
    return ret;
  }

  void* operator new[](size_t size) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 64, size) != 0 || ret == nullptr)
      throw std::bad_alloc();
    return ret;
  }

  int Insert(Key_t&, Value_t, size_t, Value_t* old_value = nullptr);
  void Insert4split(Key_t&, Value_t);
  bool Put(Key_t&, Value_t, size_t);
  Block** Split(void);
  bool Delete(Key_t&);

  void operator delete[](void* p) noexcept { std::free(p); }
  void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
  void operator delete(void* p) noexcept {
    if (p != nullptr)
      LiveBlockCount().fetch_sub(1, std::memory_order_relaxed);
    std::free(p);
  }
  void operator delete(void* p, std::size_t) noexcept {
    if (p != nullptr)
      LiveBlockCount().fetch_sub(1, std::memory_order_relaxed);
    std::free(p);
  }

  // Number of live Block allocations. The index must release every block it
  // allocates; tests compare this against the unique blocks in the directory.
  static std::atomic<std::int64_t>& LiveBlockCount() {
    static std::atomic<std::int64_t> live{0};
    return live;
  }

  Pair _[kNumSlot];
  size_t local_depth;
  int64_t sema = 0;
  // Set for the whole split, before `sema` reaches -1. Inserters that see it
  // step back instead of taking another inserter slot, so the splitter can
  // drain the count; without it a stream of inserters can keep `sema` positive
  // and starve the split forever.
  std::atomic<bool> splitting{false};
  size_t pattern = 0;
  size_t numElem(void);
};

struct Directory {
  static const size_t kDefaultDirectorySize = 1024;
  Block** _;
  size_t capacity;
  bool lock;
  int sema = 0;

  Directory(void) {
    capacity = kDefaultDirectorySize;
    _        = new Block*[capacity];
    lock     = false;
    sema     = 0;
  }

  Directory(size_t size) {
    capacity = size;
    _        = new Block*[capacity];
    lock     = false;
    sema     = 0;
  }

  ~Directory(void) { delete[] _; }

  bool Acquire(void) {
    bool unlocked = false;
    return CAS(&lock, &unlocked, true);
  }

  bool Release(void) {
    bool locked = true;
    return CAS(&lock, &locked, false);
  }

  void SanityCheck(void*);
  void LSBUpdate(int, int, int, int, Block**);
};

class ExtendibleHash : public Index {
public:
  ExtendibleHash(const BaseKVConfig& config);
  ~ExtendibleHash();
  bool Delete(Key_t&) override;
  // Diagnostic: number of distinct blocks reachable from the directory.
  size_t UniqueBlocks() const;
  double Utilization(void) override;
  size_t Capacity(void) override;
  void Get(Key_t key, Value_t& value, unsigned tid) override;
  Value_t Put(Key_t key, Value_t value, unsigned tid) override;
  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override;
  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override;

  void operator delete(void* p) noexcept { std::free(p); }
  void operator delete(void* p, std::size_t) noexcept { std::free(p); }
  void* operator new(size_t size) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 64, size) != 0 || ret == nullptr)
      throw std::bad_alloc();
    return ret;
  }

private:
  // Stripe the directory so readers take one shared lock (no contention on a
  // single cacheline) while every structural writer takes all of them.
  static constexpr size_t kShardCount = 64;
  struct alignas(64) Shard {
    mutable std::shared_mutex mu;
  };

  std::shared_mutex& ShardFor(size_t index) {
    return shards_[index % kShardCount].mu;
  }

  Directory dir;
  size_t global_depth;
  mutable std::array<Shard, kShardCount> shards_;

  size_t DirectoryIndex(size_t key_hash) const;
  Value_t ExtractWithHash(Key_t& key, size_t key_hash);
  // Caller holds D (all shards shared, or one shard shared).
  Value_t ExtractWithHashUnlocked(Key_t& key, size_t key_hash);
  Value_t Extract(Key_t& key);
  Value_t Insert(Key_t& key, Value_t value);
  bool InsertOnly(Key_t& key, Value_t value);
  Value_t FindAnyway(Key_t&);
};
