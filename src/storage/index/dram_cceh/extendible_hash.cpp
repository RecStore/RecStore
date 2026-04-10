#include "extendible_hash.h"
#include "../utils/hash.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>

size_t lockCount  = 0;
size_t splitCount = 0;

static inline uint64_t mypow(uint64_t x, unsigned int y) noexcept {
  if (x == 2) {
    if (y >= 64)
      return std::numeric_limits<uint64_t>::max();
    return (1ULL << y);
  }

  if (x && ((x & (x - 1)) == 0)) {
    unsigned k     = __builtin_ctzll(x);
    unsigned shift = k * 1u * y;
    if (shift >= 64)
      return std::numeric_limits<uint64_t>::max();
    return (1ULL << shift);
  }

  uint64_t base = x, res = 1;
  unsigned e = y;
  while (e) {
    if (e & 1)
      res *= base;
    e >>= 1;
    if (e)
      base *= base;
  }
  return res;
}

size_t Block::numElem(void) {
  size_t sum = 0;
  for (unsigned i = 0; i < kNumSlot; ++i) {
    if (_[i].key != INVALID)
      sum++;
  }
  return sum;
}

int Block::Insert(Key_t& key, Value_t value, size_t key_hash) {
#ifdef INPLACE
  if (sema == -1)
    return -2;
#  ifdef LSB
  if ((key_hash & (size_t)mypow(2, local_depth) - 1) != pattern)
    return -2;
#  else
  if ((key_hash >> (8 * sizeof(key_hash) - local_depth)) != pattern)
    return -2;
#  endif
  auto lock = sema;
  int ret   = -1;
  if (lock < 0)
    return -2;
  while (!CAS(&sema, &lock, lock + 1)) {
    lock = sema;
    if (lock < 0)
      return -2;
  }
  Key_t LOCK = INVALID;
  for (unsigned i = 0; i < kNumSlot; ++i) {
#  ifdef LSB
    if ((h(&_[i].key, sizeof(Key_t)) & (size_t)mypow(2, local_depth) - 1) !=
        pattern) {
#  else
    if ((h(&_[i].key, sizeof(Key_t)) >> (8 * sizeof(key_hash) - local_depth)) !=
        pattern) {
#  endif
      _[i].key = INVALID;
      // auto invalid = _[slot].key;
      // CAS(&_[slot].key, &invalid, INVALID);
      // do I need clflush? i don't think so. as long as it is shared through
      // cache..
    }
    if (CAS(&_[i].key, &LOCK, SENTINEL)) {
      _[i].value = value;
      mfence();
      _[i].key = key;
      ret      = i;
      break;
    } else {
      LOCK = INVALID;
    }
  }
  lock = sema;
  while (!CAS(&sema, &lock, lock - 1)) {
    lock = sema;
  }
  return ret;
#else
  if (sema == -1)
    return -2;
#  ifdef LSB
  if ((key_hash & (size_t)mypow(2, local_depth) - 1) != pattern)
    return -2;
#  else
  if ((key_hash >> (8 * sizeof(key_hash) - local_depth)) != pattern)
    return -2;
#  endif
  auto lock = sema;
  if (lock < 0)
    return -2;
  int ret = -1;
  while (!CAS(&sema, &lock, lock + 1)) {
    lock = sema;
    if (lock < 0)
      return -2;
  }
  for (unsigned i = 0; i < kNumSlot; ++i) {
    if (_[i].key == key) {
      _[i].value = value;
      ret        = i;
      lock       = sema;
      while (!CAS(&sema, &lock, lock - 1)) {
        lock = sema;
      }
      return ret;
    }
  }
  Key_t LOCK = INVALID;
  for (unsigned i = 0; i < kNumSlot; ++i) {
    if (CAS(&_[i].key, &LOCK, SENTINEL)) {
      _[i].value = value;
      mfence();
      _[i].key = key;
      ret      = i;
      break;
    } else {
      LOCK = INVALID;
    }
  }
  lock = sema;
  while (!CAS(&sema, &lock, lock - 1)) {
    lock = sema;
  }
  return ret;
#endif
}

void Block::Insert4split(Key_t& key, Value_t value) {
  for (unsigned i = 0; i < kNumSlot; ++i) {
    if (_[i].key == INVALID) {
      _[i].key   = key;
      _[i].value = value;
      return;
    }
  }
}

Block** Block::Split(void) {
  using namespace std;
  int64_t lock = 0;
  if (!CAS(&sema, &lock, -1))
    return nullptr;

#ifdef INPLACE
  Block** split = new Block*[2];
  split[0]      = this;
  split[1]      = new Block(local_depth + 1);

  for (unsigned i = 0; i < kNumSlot; ++i) {
    auto key_hash = h(&_[i].key, sizeof(Key_t));
#  ifdef LSB
    if (key_hash & ((size_t)1 << local_depth)) {
#  else
    if (key_hash & ((size_t)1 << ((sizeof(Key_t) * 8 - local_depth - 1)))) {
#  endif
      split[1]->Insert4split(_[i].key, _[i].value);
    }
  }

  clflush((char*)split[1], sizeof(Block));
  local_depth = local_depth + 1;
  clflush((char*)&local_depth, sizeof(size_t));

  return split;
#else
  Block** split = new Block*[2];
  split[0]      = new Block(local_depth + 1);
  split[1]      = new Block(local_depth + 1);

  for (unsigned i = 0; i < kNumSlot; ++i) {
    auto key_hash = h(&_[i].key, sizeof(Key_t));
#  ifdef LSB
    if (key_hash & ((size_t)1 << (local_depth))) {
#  else
    if (key_hash & ((size_t)1 << ((sizeof(Key_t) * 8 - local_depth - 1)))) {
#  endif
      split[1]->Insert4split(_[i].key, _[i].value);
    } else {
      split[0]->Insert4split(_[i].key, _[i].value);
    }
  }

  clflush((char*)split[0], sizeof(Block));
  clflush((char*)split[1], sizeof(Block));

  return split;
#endif
}

bool Block::Delete(Key_t& key) {
  if (sema == -1)
    return false;

  int64_t lock = sema;
  if (lock < 0)
    return false;
  while (!CAS(&sema, &lock, lock + 1)) {
    lock = sema;
    if (lock < 0)
      return false;
  }

  bool found = false;
  for (unsigned i = 0; i < kNumSlot; ++i) {
    if (_[i].key == key) {
      _[i].key = INVALID;
      clflush((char*)&_[i].key, sizeof(Key_t));
      found = true;
      break;
    }
  }

  lock = sema;
  while (!CAS(&sema, &lock, lock - 1)) {
    lock = sema;
  }
  return found;
}

ExtendibleHash::ExtendibleHash(const BaseKVConfig& config)
    : Index(config),
      dir{[&config]() -> size_t {
        if (!config.json_config_.contains("initial_capacity")) {
          LOG(WARNING) << "ExtendibleHash config missing 'initial_capacity'";
          return 1;
        }
        const size_t init_cap =
            config.json_config_.at("initial_capacity").get<size_t>();
        if (init_cap == 0) {
          LOG(WARNING)
              << "ExtendibleHash config has invalid 'initial_capacity'=0";
          return 1;
        }
        return init_cap;
      }()},
      global_depth{0} {
  // Customize based on config, e.g.
  LOG(INFO) << "DRAM index init";
  global_depth = static_cast<size_t>(log2(dir.capacity));
  for (unsigned i = 0; i < dir.capacity; ++i) {
    dir._[i]          = new Block(global_depth);
    dir._[i]->pattern = i;
  }
}

ExtendibleHash::~ExtendibleHash() {}

void Directory::LSBUpdate(
    int local_depth, int global_depth, int dir_cap, int x, Block** s) {
  int depth_diff = global_depth - local_depth;
  if (depth_diff == 0) {
    if ((x % dir_cap) >= dir_cap / 2) {
      _[x - dir_cap / 2] = s[0];
      clflush((char*)&_[x - dir_cap / 2], sizeof(Block*));
      _[x] = s[1];
      clflush((char*)&_[x], sizeof(Block*));
    } else {
      _[x] = s[0];
      clflush((char*)&_[x], sizeof(Block*));
      _[x + dir_cap / 2] = s[1];
      clflush((char*)&_[x + dir_cap / 2], sizeof(Block*));
    }
  } else {
    if ((x % dir_cap) >= dir_cap / 2) {
      LSBUpdate(local_depth + 1, global_depth, dir_cap / 2, x - dir_cap / 2, s);
      LSBUpdate(local_depth + 1, global_depth, dir_cap / 2, x, s);
    } else {
      LSBUpdate(local_depth + 1, global_depth, dir_cap / 2, x, s);
      LSBUpdate(local_depth + 1, global_depth, dir_cap / 2, x + dir_cap / 2, s);
    }
  }
  return;
}

void ExtendibleHash::Insert(Key_t& key, Value_t value) {
  using namespace std;
  auto key_hash = h(&key, sizeof(key));

RETRY:
#ifdef LSB
  // auto x = (key_hash & (pow(2, global_depth)-1));
  auto x = (key_hash % dir.capacity);
#else
  auto x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
#endif
  auto target = dir._[x];
  auto ret    = target->Insert(key, value, key_hash);

  if (ret == -1) {
    Block** s = target->Split();
    if (s == nullptr) {
      // another thread is doing split
      goto RETRY;
    }

#ifdef LSB
    s[0]->pattern = (key_hash % (size_t)mypow(2, s[0]->local_depth - 1));
    s[1]->pattern = s[0]->pattern + (1 << (s[0]->local_depth - 1));
#else
    s[0]->pattern =
        (key_hash >> (8 * sizeof(key_hash) - s[0]->local_depth + 1)) << 1;
    s[1]->pattern =
        ((key_hash >> (8 * sizeof(key_hash) - s[1]->local_depth + 1)) << 1) + 1;
#endif

    // Directory management
    while (!dir.Acquire()) {
      // lockCount ++;
    }
    // dir.sema++;
    { // CRITICAL SECTION - directory update
      // auto prev = x;
#ifdef LSB
      x = (key_hash % dir.capacity);
#else
      x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
#endif
#ifdef INPLACE
      if (dir._[x]->local_depth - 1 < global_depth) { // normal split
#else
      if (dir._[x]->local_depth < global_depth) { // normal split
#endif
#ifdef LSB
        dir.LSBUpdate(s[0]->local_depth, global_depth, dir.capacity, x, s);
#else
        unsigned depth_diff = global_depth - s[0]->local_depth;
        if (depth_diff == 0) {
          if (x % 2 == 0) {
            dir._[x + 1] = s[1];
#  ifdef INPLACE
            clflush((char*)&dir._[x + 1], 8);
#  else
            mfence();
            dir._[x] = s[0];
            clflush((char*)&dir._[x], 16);
#  endif
          } else {
            dir._[x] = s[1];
#  ifdef INPLACE
            clflush((char*)&dir._[x], 8);
#  else
            mfence();
            dir._[x - 1] = s[0];
            clflush((char*)&dir._[x - 1], 16);
#  endif
          }
        } else {
          int chunk_size = mypow(2, global_depth - (s[0]->local_depth - 1));
          x              = x - (x % chunk_size);
          for (unsigned i = 0; i < chunk_size / 2; ++i) {
            dir._[x + chunk_size / 2 + i] = s[1];
          }
          clflush((char*)&dir._[x + chunk_size / 2],
                  sizeof(void*) * chunk_size / 2);
#  ifndef INPLACE
          for (unsigned i = 0; i < chunk_size / 2; ++i) {
            dir._[x + i] = s[0];
          }
          clflush((char*)&dir._[x], sizeof(void*) * chunk_size / 2);
#  endif
        }
#endif
      } else { // directory doubling
        auto d    = dir._;
        auto _dir = new Block*[dir.capacity * 2];
#ifdef LSB
        memcpy(_dir, d, sizeof(Block*) * dir.capacity);
        memcpy(_dir + dir.capacity, d, sizeof(Block*) * dir.capacity);
        _dir[x]                = s[0];
        _dir[x + dir.capacity] = s[1];
#else
        for (unsigned i = 0; i < dir.capacity; ++i) {
          if (i == x) {
            _dir[2 * i]     = s[0];
            _dir[2 * i + 1] = s[1];
          } else {
            _dir[2 * i]     = d[i];
            _dir[2 * i + 1] = d[i];
          }
        }
#endif
        clflush((char*)&dir._[0], sizeof(Block*) * dir.capacity);
        dir._ = _dir;
        clflush((char*)&dir._, sizeof(void*));
        dir.capacity *= 2;
        clflush((char*)&dir.capacity, sizeof(size_t));
        global_depth += 1;
        clflush((char*)&global_depth, sizeof(global_depth));
        delete[] d;
        // TODO: requiered to do this atomically
        // cout << x << " directory doubling " << target << " " << dir._[x]<<
        // endl;
      }
#ifdef INPLACE
      s[0]->sema = 0;
#endif
    } // End of critical section
    // dir.sema--;
    while (!dir.Release()) {
      // lockCount ++;
    }
    goto RETRY;
  } else if (ret == -2) {
    Insert(key, value);
  } else {
    clflush((char*)&dir._[x]->_[ret], sizeof(Pair));
  }
}

// This function does not allow resizing
bool ExtendibleHash::InsertOnly(Key_t& key, Value_t value) {
  auto key_hash = h(&key, sizeof(key));
#ifdef LSB
  auto x = (key_hash % dir.capacity);
#else
  auto x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
#endif

  auto ret = dir._[x]->Insert(key, value, key_hash);
  if (ret > -1) {
    clflush((char*)&dir._[x]->_[ret], sizeof(Pair));
    return true;
  }

  return false;
}

bool ExtendibleHash::Delete(Key_t& key) {
  auto key_hash = h(&key, sizeof(key));
#ifdef LSB
  auto x = (key_hash % dir.capacity);
#else
  auto x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
#endif

  Block* target = dir._[x];
  bool success  = target->Delete(key);

  if (!success && target->sema == -1) {
    std::this_thread::yield();
    x       = (key_hash % dir.capacity);
    success = dir._[x]->Delete(key);
  }
  return success;
}

Value_t ExtendibleHash::Extract(Key_t& key) {
  auto key_hash = h(&key, sizeof(key));
#ifdef LSB
  auto x = (key_hash % dir.capacity);
#else
  auto x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
#endif

  auto dir_ = dir._[x];

#ifdef INPLACE
  auto sema = dir._[x]->sema;
  while (!CAS(&dir._[x]->sema, &sema, sema + 1)) {
    sema = dir._[x]->sema;
  }
#endif

  for (unsigned i = 0; i < Block::kNumSlot; ++i) {
    if (dir_->_[i].key == key) {
#ifdef INPLACE
      sema = dir._[x]->sema;
      while (!CAS(&dir._[x]->sema, &sema, sema - 1)) {
        sema = dir._[x]->sema;
      }
#endif
      return dir_->_[i].value;
    }
    lockCount++;
  }

#ifdef INPLACE
  sema = dir._[x]->sema;
  while (!CAS(&dir._[x]->sema, &sema, sema - 1)) {
    sema = dir._[x]->sema;
  }
#endif
  return NONE;
}

// Debugging function
Value_t ExtendibleHash::FindAnyway(Key_t& key) {
  using namespace std;
  for (size_t i = 0; i < dir.capacity; ++i) {
    for (size_t j = 0; j < Block::kNumSlot; ++j) {
      if (dir._[i]->_[j].key == key) {
        auto key_hash = h(&key, sizeof(key));
        auto x        = (key_hash >> (8 * sizeof(key_hash) - global_depth));
        return dir._[i]->_[j].value;
      }
    }
  }
  return NONE;
}

// Not accurate
double ExtendibleHash::Utilization(void) {
  size_t sum = 0;
  std::unordered_map<Block*, bool> set;
  for (size_t i = 0; i < dir.capacity; ++i) {
    set[dir._[i]] = true;
  }
  for (auto& elem : set) {
    for (unsigned i = 0; i < Block::kNumSlot; ++i) {
      if (elem.first->_[i].key != INVALID)
        sum++;
    }
  }
  return ((double)sum) / ((double)set.size() * Block::kNumSlot) * 100.0;
}

void Directory::SanityCheck(void* addr) {
  using namespace std;
  for (unsigned i = 0; i < capacity; ++i) {
    if (_[i] == addr) {
      cout << i << " " << _[i]->sema << endl;
      exit(1);
    }
  }
}

size_t ExtendibleHash::Capacity(void) {
  std::unordered_map<Block*, bool> set;
  for (size_t i = 0; i < dir.capacity; ++i) {
    set[dir._[i]] = true;
  }
  return set.size() * Block::kNumSlot;
}

void ExtendibleHash::Get(Key_t key, Value_t& pointer, unsigned tid) {
  pointer = Extract(key);
}

void ExtendibleHash::Put(Key_t key, Value_t pointer, unsigned tid) {
  Insert(key, static_cast<Value_t>(pointer));
}

void ExtendibleHash::BatchGet(
    base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) {
  if (pointers == nullptr || keys.Size() == 0) {
    LOG(FATAL) << "Invalid pointers array or empty keys";
  }
  for (size_t i = 0; i < keys.Size(); ++i) {
    Key_t key = keys[i];
    Get(key, pointers[i], tid);
  }
}

void ExtendibleHash::BatchPut(
    base::ConstArray<Key_t> keys, Value_t* pointers, unsigned tid) {
  if (pointers == nullptr || keys.Size() == 0) {
    LOG(FATAL) << "Invalid pointers array or empty keys";
  }
  for (size_t i = 0; i < keys.Size(); ++i) {
    Key_t key = keys[i];
    Put(key, pointers[i], tid);
  }
}
