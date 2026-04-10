#pragma once

#include <cstdlib>
#include <cstring>
#include <new>
#include "../index.h"
#include "base/factory.h"
#include "storage/kv_engine/base_kv.h"
#include "../utils/util.h"
#define LSB

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
    return ret;
  }

  void* operator new[](size_t size) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 64, size) != 0 || ret == nullptr)
      throw std::bad_alloc();
    return ret;
  }

  int Insert(Key_t&, Value_t, size_t);
  void Insert4split(Key_t&, Value_t);
  bool Put(Key_t&, Value_t, size_t);
  Block** Split(void);
  bool Delete(Key_t&);

  void operator delete[](void* p) noexcept { std::free(p); }
  void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
  void operator delete(void* p) noexcept { std::free(p); }
  void operator delete(void* p, std::size_t) noexcept { std::free(p); }

  Pair _[kNumSlot];
  size_t local_depth;
  int64_t sema   = 0;
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
  double Utilization(void) override;
  size_t Capacity(void) override;
  void Get(Key_t key, Value_t& value, unsigned tid) override;
  void Put(Key_t key, Value_t value, unsigned tid) override;
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
  Directory dir;
  size_t global_depth;

  Value_t Extract(Key_t& key);
  void Insert(Key_t& key, Value_t value);
  bool InsertOnly(Key_t& key, Value_t value);
  Value_t FindAnyway(Key_t&);
};

FACTORY_REGISTER(Index, DRAM, ExtendibleHash, const BaseKVConfig&);