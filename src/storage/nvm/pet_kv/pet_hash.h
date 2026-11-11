#pragma once

#include <emmintrin.h> //NOLINT
#include <folly/GLog.h>
#include <nmmintrin.h> //NOLINT

#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/base.h"
#include "persistence.h"

namespace base {

#pragma pack(push)
#pragma pack(2)
template <class K, class T>
struct EntryT {
  EntryT() : first(), second() {}
  explicit EntryT(K s, T d) : first(s), second(d) {}
  explicit EntryT(K s) : first(s), second() {}
  K first;
  T second;
};
#pragma pack(pop)

class SparseMaskIter {
public:
  explicit SparseMaskIter(int mask) : mask_(mask) {}
  bool HasNext() const { return mask_ != 0; }
  unsigned Next() {
    unsigned i = __builtin_ctz(mask_);
    mask_ &= (mask_ - 1);
    return i;
  }
  uint32_t Val() const { return mask_; }

private:
  uint32_t mask_;
};

template <typename KeyT, typename ValueT, bool Persistence = false>
struct F14Chunk {
  typedef EntryT<KeyT, ValueT> ItemT;
  static constexpr int kMaxSize  = 14;
  static constexpr int kFullMask = (1 << kMaxSize) - 1;

  SparseMaskIter MatchTag(uint8_t tag) const {
    auto tag_vec = _mm_load_si128(static_cast<__m128i const*>( // NOLINT
        static_cast<const void*>(&tags_[0])));                 // NOLINT
    auto cmp_val = _mm_set1_epi8(tag);                         // NOLINT
    auto eq_vec  = _mm_cmpeq_epi8(tag_vec, cmp_val);           // NOLINT
    auto mask    = _mm_movemask_epi8(eq_vec) & kFullMask;      // NOLINT
    return SparseMaskIter{mask};
  }

  int Find(const KeyT& key, uint8_t tag) {
    auto it = MatchTag(tag);
    while (it.HasNext()) {
      auto id   = it.Next();
      auto pair = Get(id);
      if (LIKELY(key == pair->first)) {
        return id;
      }
    }
    return -1;
  }

  ItemT* Insert(const KeyT& key, const ValueT& value, const uint8_t sign) {
    CHECK_LT(Size(), kMaxSize);
    size_++;
    auto it = MatchTag(0);
    CHECK(it.HasNext());
    return Set(it.Next(), sign, ItemT{key, value});
  }

  ItemT* Set(size_t i, uint8_t tag, const ItemT& val) {
    value_[i] = val;
    IF_Persistence(clflushopt_range(&value_[i], sizeof(ItemT)););
    tags_[i] = tag;
    return value_ + i;
  }
  ItemT* Get(size_t i) { return value_ + i; }
  bool IncOverflow() {
    if (overflow_count_ == 255)
      return false;
    ++overflow_count_;
    return true;
  }
  void DecOverflow() {
    if (overflow_count_ != 255)
      --overflow_count_;
  }
  void Delete(size_t i) {
    tags_[i]        = 0;
    value_[i].first = -1;
    IF_Persistence(
        clflushopt_range(&value_[i].first, sizeof(value_[i].first)););
    size_--;
  }
  inline size_t OverflowCount() { return overflow_count_; }

  inline size_t Size() const { return size_; }
  bool InUse(size_t i) const { return tags_[i] != 0; }
  void Initialize() {
    for (int i = 0; i < kMaxSize; i++)
      value_[i].first = -1;
  }

  uint8_t tags_[kMaxSize];
  uint8_t size_;
  base::Lock lock_;
  std::atomic<uint8_t> overflow_count_;
  std::atomic<bool> is_migrating;
  ItemT value_[kMaxSize];
} __attribute__((__aligned__(64)));

// The basic structure is borrowed from F14 hash in Folly
template <typename KeyT,
          typename ValueT,
          bool Persistence = false,
          bool MM_PREFETCH = true>
class PetHash {
public:
  typedef EntryT<KeyT, ValueT> ItemT;
  typedef F14Chunk<KeyT, ValueT, Persistence> ChunkT;
  typedef std::function<bool(
      const KeyT&, const ValueT&, const bool& force_delete)>
      CheckFuncT;
  static constexpr size_t kChunkSize      = ChunkT::kMaxSize;
  static const uint32_t kSignBit          = 8;
  static const uint32_t kSignMask         = (1 << kSignBit) - 1;
  static const uint32_t kMaxLoadFactor    = 90;
  static constexpr size_t kSizeShardCount = Persistence ? 1 : 64;

  PetHash() = default;

  static int64_t OverFallChunkNum(uint64_t capacity) {
    return Next2Power((capacity + kChunkSize - 1) / kChunkSize);
    // return 2 * (capacity + kChunkSize - 1) / kChunkSize;
  }

  void Initialize(uint64_t capacity, bool ignore_load_factor = false) {
    if (!ignore_load_factor)
      capacity /= kMaxLoadFactor / 100.0;
    chunk_num_ = OverFallChunkNum(capacity);
    capacity_  = chunk_num_ * kChunkSize;
    ResetSize();
    memset(chunk_table_, 0, chunk_num_ * sizeof(ChunkT));
    for (uint64_t i = 0; i < chunk_num_; i++) {
      chunk_table_[i].Initialize();
    }
  }

  // Note(xieminhui) that mallocSetValid should be thread safe
  void Reload(uint64_t capacity,
              bool ignore_load_factor,
              std::function<void(KeyT, ValueT)> mallocSetValid,
              const int kRecoveryThread = 4) {
    CHECK(Persistence) << "If not PMEM KV, dont use Reload";
    if (!ignore_load_factor)
      capacity /= kMaxLoadFactor / 100.0;
    // chunk_num_ = Next2Power((capacity + kChunkSize - 1) / kChunkSize);
    chunk_num_        = OverFallChunkNum(capacity);
    capacity_         = chunk_num_ * kChunkSize;
    uint64_t old_size = Size();
    ResetSize();
    std::vector<std::thread> thread_pool;
    uint64_t block_num = chunk_num_ / kRecoveryThread + 1;
    for (int i = 0; i < kRecoveryThread; ++i) {
      uint64_t left  = i * block_num;
      uint64_t right = left + block_num;
      thread_pool.push_back(std::thread([&, left, right] {
        for (int chunk_id = left; chunk_id < right && chunk_id < chunk_num_;
             chunk_id++) {
          ChunkT& chunk = chunk_table_[chunk_id];
          chunk.size_   = 0;
          // reset inconsistent overflow_count_
          chunk.overflow_count_ = 0;
          for (int chunk_offset = 0; chunk_offset < ChunkT::kMaxSize;
               chunk_offset++) {
            if (chunk.value_[chunk_offset].first == -1)
              continue;
            chunk.size_++;
            ItemT& chunkItem = chunk.value_[chunk_offset];
            IncrementSize(chunkItem.first);
            uint64 hash_value = Hash(chunkItem.first);
            uint8 sign        = Sign(hash_value);
            if (UNLIKELY(chunk.tags_[chunk_offset] != sign)) {
              // inconsistent tags
              RECSTORE_LOG_EVERY_MS(WARNING, 2000)
                  << "inconsistent tag, key = " << chunkItem.first
                  << " , correct it";
              chunk.tags_[chunk_offset] = sign;
            }
          }
        }
      }));
    }
    for (auto& each : thread_pool)
      each.join();
    if (fabs((double)old_size - (double)Size()) >= 1000) {
      LOG(ERROR) << "Large Diff after reload.";
    }

    // 设置 overflowbit
    thread_pool.clear();
    for (int i = 0; i < kRecoveryThread; ++i) {
      uint64_t left  = i * block_num;
      uint64_t right = left + block_num;
      thread_pool.push_back(std::thread([&, left, right] {
        for (int chunk_id = left; chunk_id < right && chunk_id < chunk_num_;
             chunk_id++) {
          ChunkT& chunk = chunk_table_[chunk_id];
          for (int chunk_offset = 0; chunk_offset < ChunkT::kMaxSize;
               chunk_offset++) {
            ItemT& chunkItem = chunk.value_[chunk_offset];
            if (chunk.InUse(chunk_offset)) {
              mallocSetValid(chunkItem.first, chunkItem.second);
              uint64 hash_value = Hash(chunkItem.first);
              size_t step       = ProbeDelta(Sign(hash_value));
              uint64 pos        = hash_value & (chunk_num_ - 1);
              while (pos != chunk_id) {
                /* before
                 * chunk_table_[pos].IncOverflow();
                 *********************************
                 * after
                 * overflow_count_ will only incr. Thus we
                 * can use CAS to do lock free while avoid overflow
                 */
                uint8* prev_overflow_count = nullptr;
                uint8_t temp;
                do {
                  temp = chunk_table_[pos].overflow_count_;
                  if (temp == 255)
                    break;
                } while (
                    !chunk_table_[pos].overflow_count_.compare_exchange_strong(
                        temp, temp + 1)); // NOLINT
                pos = (pos + step) & (chunk_num_ - 1);
              }
            }
          }
        }
      }));
    }
    for (auto& each : thread_pool)
      each.join();
  }

  static uint64_t
  MemorySize(uint64_t capacity, bool ignore_load_factor = false) {
    if (!ignore_load_factor)
      capacity /= kMaxLoadFactor / 100.0;
    // auto chunk_table_num = Next2Power((capacity + kChunkSize - 1) /
    // kChunkSize);
    auto chunk_table_num = OverFallChunkNum(capacity);
    return sizeof(PetHash) + chunk_table_num * sizeof(ChunkT);
  }

  static uint64_t Next2Power(uint64_t size) {
    uint64_t new_size = 1;
    while (new_size < size)
      new_size <<= 1;
    return new_size;
  }

  bool Valid(int64_t memory_size) {
    if (capacity_ == 0 && memory_size > 0) {
      LOG(ERROR) << "PetHash invalid. capacity_ == 0";
      return false;
    }
    const uint64_t size = Size();
    if (size > capacity_) {
      LOG(ERROR) << "PetHash invalid. dict size_ > capacity_";
      return false;
    }
    if (chunk_num_ * kChunkSize != capacity_) {
      LOG(ERROR) << "chunk_num_ * kChunkSize != capacity_";
      return false;
    }
    if (memory_size != MemorySize(capacity_, true)) {
      if (capacity_ != 0)
        LOG(ERROR) << "PetHash invalid. memory_size != MemorySize(capacity_, "
                      "true), memory_size = "
                   << memory_size << ", capacity_ = " << capacity_;
      return false;
    }
    uint64_t total_num             = 0;
    uint64_t chunk_table_total_num = 0;
    KeyT key;
    for (uint64_t i = 0; i < capacity_; ++i) {
      if (IndexInUse(i)) {
        GetById(i, &key);
        auto [value, exists] = Get(key);
        if (!exists) {
          LOG(ERROR) << "PetHash invalid. GetById has key, but Get(key) "
                        "failed; key is "
                     << key;
          return false;
        }
        total_num++;
      }
    }
    for (int i = 0; i < chunk_num_; ++i) {
      chunk_table_total_num += chunk_table_[i].Size();
    }
    if (total_num != chunk_table_total_num)
      LOG(ERROR) << fmt::format(
          "PetHash invalid {} != {}", total_num, chunk_table_total_num);
    if (total_num != size)
      LOG(ERROR) << fmt::format("PetHash invalid {} != {}", total_num, size);
    return total_num == chunk_table_total_num && total_num == size;
  }

  void Debug() {
    LOG(INFO) << "capacity_: " << capacity_;
    LOG(INFO) << "chunk_num_: " << chunk_num_;
    LOG(INFO) << "Chunk size: " << sizeof(ChunkT);
    LOG(INFO) << "PetHash size: " << sizeof(PetHash);
  }

  std::pair<const ValueT, bool> Get(const KeyT& key) const {
    auto* p = (PetHash<KeyT, ValueT, Persistence, MM_PREFETCH>*)(this);
    auto [value, p_value, exist] =
        p->GetInternal(key, nullptr, nullptr, nullptr);
    return std::make_pair(value, exist);
  }

  std::pair<ValueT*, bool> GetReturnPtr(const KeyT& key) {
    auto [value, p_value, exist] = GetInternal(key, nullptr, nullptr, nullptr);
    return std::make_pair(p_value, exist);
  }

  inline void HintPrefetch(const KeyT key) {
    uint64_t hash_value = Hash(key);
    auto sign           = Sign(hash_value);
    size_t pos          = hash_value & (chunk_num_ - 1);
    auto chunk          = chunk_table_ + pos;
    _mm_prefetch((const char*)(chunk), _MM_HINT_T0);
    _mm_prefetch((const char*)(chunk->Get(0)), _MM_HINT_T0);
  }

  std::tuple<const ValueT, ValueT* const, bool>
  GetInternal(const KeyT& key,
              uint64* const item_pos     = nullptr,
              ChunkT** const chunk_pos   = nullptr,
              uint8_t* const in_chunk_id = nullptr) {
  RETRY:
    uint64_t hash_value = Hash(key);
    auto sign           = Sign(hash_value);
    size_t step         = ProbeDelta(sign);
    size_t pos          = hash_value & (chunk_num_ - 1);
    size_t id;
    bool touch_migrating_chunk = false;
    for (size_t i = 0; i < chunk_num_; ++i) {
      auto chunk = chunk_table_ + pos;
      if (MM_PREFETCH) {
        _mm_prefetch(
            static_cast<char const*>(static_cast<void const*>( // NOLINT
                chunk->Get(3))),
            _MM_HINT_T0); // NOLINT
      }
      if (UNLIKELY(chunk->is_migrating))
        touch_migrating_chunk = true;

      auto it = chunk->MatchTag(sign);
      while (it.HasNext()) {
        id       = it.Next();
        auto val = chunk->Get(id);
        // re-read the key to ensure no-one modify it
        if (LIKELY(key == val->first)) {
          if (item_pos != nullptr)
            *item_pos = pos * kChunkSize + id;
          if (chunk_pos != nullptr)
            *chunk_pos = chunk;
          if (in_chunk_id != nullptr)
            *in_chunk_id = id;
          return std::make_tuple(val->second, &val->second, true);
        }
      }
      if (LIKELY(chunk->OverflowCount() == 0))
        break;
      pos = (pos + step) & (chunk_num_ - 1);
    }
    if (UNLIKELY(touch_migrating_chunk))
      goto RETRY;
    return std::make_tuple(ValueT(), nullptr, false);
  }

  ChunkT* const NextChunk(ChunkT* const chunk, const int step) {
    auto p = chunk - chunk_table_;
    p      = (p + step + chunk_num_) % chunk_num_;
    return chunk_table_ + p;
  }
  ChunkT* const PrevChunk(ChunkT* const chunk, const int step) {
    auto p = chunk - chunk_table_;
    p      = (p - step + chunk_num_) % chunk_num_;
    return chunk_table_ + p;
  }

  // start -> mid ... -> mid -> end , [start, end]
  void ChainTraverseReverse(ChunkT* const start,
                            ChunkT* const end,
                            int step,
                            std::function<void(ChunkT* const)> func) {
    ChunkT* current_chunk = end;
    while (current_chunk != start) {
      func(current_chunk);
      current_chunk = PrevChunk(current_chunk, step);
    }
    func(current_chunk);
  }

  void ChainTraverse(ChunkT* const start,
                     ChunkT* const end,
                     int step,
                     std::function<void(ChunkT* const)> func) {
    ChunkT* current_chunk = end;
    while (current_chunk != start) {
      func(current_chunk);
      current_chunk = NextChunk(current_chunk, step);
    }
    func(current_chunk);
  }

  void HotnessAwareMigration(const base::ConstArray<KeyT> hot_keys) {
    for (auto key : hot_keys) {
      auto [home_chunk, home_chunk_id] = HomeChunk(key);
      auto sign                        = Sign(Hash(key));
      size_t hotkey_step               = ProbeDelta(sign);
      int id_in_chunk                  = home_chunk->Find(key, sign);
      if (id_in_chunk != -1) {
        // hot KV is already in its home chunk, continue
        continue;
      }
      ChunkT* current_chunk;
      uint8_t current_in_chunk_id;
      auto [value, p_value_no_use, exists] =
          GetInternal(key, nullptr, &current_chunk, &current_in_chunk_id);
      CHECK(exists);

      ChainTraverseReverse(
          home_chunk, current_chunk, hotkey_step, [](ChunkT* const each) {
            each->is_migrating = true;
          });

      // case 1: home_chunk has empty slot
      if (LIKELY(home_chunk->Size() != kChunkSize)) {
        // insert it to its home chunk
        home_chunk->Insert(key, value, sign);
        // remove it from its current chunk
        current_chunk->Delete(current_in_chunk_id);
        current_chunk->DecOverflow();
        ChainTraverseReverse(
            home_chunk,
            current_chunk,
            hotkey_step,
            [current_chunk](ChunkT* const each) {
              each->is_migrating = false;
            });
        continue;
      }
      // case 2: migration
      // Step 2.1: home_chunk select a victim
      KeyT victim_key     = -1;
      ValueT victim_value = -1;
      int victim_id       = 0;
      CHECK(home_chunk->InUse(victim_id));
      {
        auto item    = home_chunk->Get(victim_id);
        victim_key   = item->first;
        victim_value = item->second;
      }
      // Step 2.2: insert a victim to another chunk
      auto victim_sign     = Sign(Hash(victim_key));
      size_t victim_step   = ProbeDelta(victim_sign);
      size_t victim_pos    = (home_chunk_id + victim_step) & (chunk_num_ - 1);
      bool victim_migrated = false;
      auto chunk           = chunk_table_ + victim_pos;
      for (size_t i = 0; i < chunk_num_; ++i) {
        base::AutoLock lock(chunk->lock_);
        if (LIKELY(chunk->Size() != kChunkSize)) {
          chunk->Insert(victim_key, victim_value, victim_sign);
          // 不能把victim又插回他原来的home_chunk
          CHECK_NE(home_chunk, chunk);
          victim_migrated = true;
          break;
        }
        // 达到负载上限后需要删除才能插入
        if (UNLIKELY(Full())) {
          LOG(FATAL) << "todo: pet hash dict is full";
          // let the insert thread first eviction
          DeleteLocked(victim_pos, (sign + i) % kChunkSize);
        } else {
          chunk->IncOverflow();
          victim_pos = (victim_pos + victim_step) & (chunk_num_ - 1);
          chunk      = chunk_table_ + victim_pos;
        }
      }
      CHECK(victim_migrated);
      // Step 2.3: substitute victim with <key>
      home_chunk->IncOverflow();
      home_chunk->Set(victim_id, sign, ItemT{key, value});
      current_chunk->Delete(current_in_chunk_id);
      // all chunks along this chain should be reduced by 1
      // (current_chunk, home_chunk]
      ChainTraverseReverse(
          home_chunk,
          current_chunk,
          hotkey_step,
          [current_chunk](ChunkT* const each) {
            //  unlock
            each->is_migrating = false;
            //  unlock done
            if (each == current_chunk)
              return;
            each->DecOverflow();
          });
    }
  }

  void BatchSet(base::ConstArray<KeyT> keys,
                const ValueT* values,
                ValueT* old_values = nullptr) {
    for (int i = 0; i < keys.Size(); ++i) {
      if (i + 1 < keys.Size()) {
        HintPrefetch(keys[i + 1]);
      }
      Insert(keys[i],
             values[i],
             nullptr,
             false,
             old_values == nullptr ? nullptr : &old_values[i]);
    }
  }

  ValueT* Set(const KeyT& key,
              const ValueT& value,
              CheckFuncT check_func = nullptr,
              bool is_force         = false,
              ValueT* old_value_out = nullptr) {
    static_assert(sizeof(ValueT) <= sizeof(uint64_t),
                  "PetHash::Set CAS update expects a 64-bit-or-smaller value");
    if (old_value_out != nullptr) {
      *old_value_out = ValueT();
    }

  RETRY:
    uint64_t hash_value        = Hash(key);
    auto sign                  = Sign(hash_value);
    size_t step                = ProbeDelta(sign);
    size_t pos                 = hash_value & (chunk_num_ - 1);
    bool touch_migrating_chunk = false;

    for (size_t i = 0; i < chunk_num_; ++i) {
      auto chunk = chunk_table_ + pos;
      if (UNLIKELY(chunk->is_migrating)) {
        touch_migrating_chunk = true;
      }

      {
        auto it = chunk->MatchTag(sign);
        while (it.HasNext()) {
          auto id  = it.Next();
          auto val = chunk->Get(id);
          if (LIKELY(key == val->first)) {
            while (LIKELY(key == val->first)) {
              ValueT expected = val->second;
              ValueT desired  = value;
              if (__atomic_compare_exchange_n(
                      &val->second,
                      &expected,
                      desired,
                      false,
                      __ATOMIC_ACQ_REL,
                      __ATOMIC_ACQUIRE)) {
                if (old_value_out != nullptr) {
                  *old_value_out = expected;
                }
                IF_Persistence(clflushopt_range(&val->second, sizeof(uint64)););
                return &val->second;
              }
              __asm__ volatile("pause" ::: "memory");
            }
            goto RETRY;
          }
        }
      }

      if (LIKELY(chunk->OverflowCount() == 0)) {
        break;
      }
      pos = (pos + step) & (chunk_num_ - 1);
    }

    if (UNLIKELY(touch_migrating_chunk)) {
      goto RETRY;
    }
    return Insert(key, value, check_func, is_force, old_value_out);
  }

  std::pair<ChunkT*, size_t> HomeChunk(const KeyT& key) {
    uint64_t hash_value = Hash(key);
    size_t pos          = hash_value & (chunk_num_ - 1);
    auto chunk          = chunk_table_ + pos;
    return std::make_pair(chunk, pos);
  }

  ValueT* Insert(const KeyT& key,
                 const ValueT& value,
                 CheckFuncT check_func = nullptr,
                 bool force_insert     = false,
                 ValueT* old_value_out = nullptr) {
  RETRY:
    static_assert(sizeof(ValueT) <= sizeof(uint64_t),
                  "PetHash::Insert CAS update expects a 64-bit-or-smaller "
                  "value");
    if (old_value_out != nullptr) {
      *old_value_out = ValueT();
    }
    uint64_t hash_value           = Hash(key);
    auto sign                     = Sign(hash_value);
    size_t step                   = ProbeDelta(sign);
    auto [chunk, pos]             = HomeChunk(key);
    size_t overflowed_chunk_count = 0;
    std::vector<size_t> saturated_overflow_steps;
    auto rollback_overflow = [&] {
      size_t rollback_pos = hash_value & (chunk_num_ - 1);
      size_t saturated_id = 0;
      for (size_t overflow_step = 0; overflow_step < overflowed_chunk_count;
           ++overflow_step) {
        if (saturated_id < saturated_overflow_steps.size() &&
            saturated_overflow_steps[saturated_id] == overflow_step) {
          ++saturated_id;
        } else {
          chunk_table_[rollback_pos].DecOverflow();
        }
        rollback_pos = (rollback_pos + step) & (chunk_num_ - 1);
      }
    };

    for (size_t i = 0; i < chunk_num_; ++i) {
      base::AutoLock lock(chunk->lock_);
      if (UNLIKELY(chunk->is_migrating)) {
        rollback_overflow();
        goto RETRY;
      }
      // check whether KV expire
      if (UNLIKELY(check_func && chunk->Size() >= kChunkSize - 2)) {
        for (size_t id = 0; id < kChunkSize; ++id)
          if (chunk->InUse(id)) {
            auto it = chunk->Get(id);
            if (!check_func(it->first, it->second, false))
              DeleteLocked(pos, id);
          }
      }

      // re-search chunk
      {
        auto it = chunk->MatchTag(sign);
        while (it.HasNext()) {
          auto id  = it.Next();
          auto val = chunk->Get(id);
          if (LIKELY(key == val->first)) {
            rollback_overflow();
            while (LIKELY(key == val->first)) {
              ValueT expected = val->second;
              ValueT desired  = value;
              if (__atomic_compare_exchange(
                      &val->second,
                      &expected,
                      &desired,
                      false,
                      __ATOMIC_ACQ_REL,
                      __ATOMIC_ACQUIRE)) {
                if (old_value_out != nullptr) {
                  *old_value_out = expected;
                }
                IF_Persistence(clflushopt_range(&val->second, sizeof(uint64)););
                return &val->second;
              }
              __asm__ volatile("pause" ::: "memory");
            }
            goto RETRY;
          }
        }
      }

      if (LIKELY(chunk->Size() < kChunkSize)) {
        auto val = chunk->Insert(key, value, sign);
        IncrementSize(key);
        return &val->second;
      }
      if (UNLIKELY(Full())) {
        const uint64_t size = Size();
        RECSTORE_LOG_EVERY_MS(ERROR, 1000) << fmt::format(
            "Warning pethash dict full, {}/{}={}",
            size,
            capacity_,
            size / (double)capacity_);
        if (!force_insert) {
          LOG(FATAL) << "dict OOM";
          return nullptr;
        }
        auto it = chunk->Get((sign + i) % kChunkSize);
        if (check_func != nullptr)
          check_func(it->first, it->second, true);
        DeleteLocked(pos, (sign + i) % kChunkSize);
      } else {
        if (!chunk->IncOverflow()) {
          saturated_overflow_steps.push_back(overflowed_chunk_count);
        }
        ++overflowed_chunk_count;
        pos   = (pos + step) & (chunk_num_ - 1);
        chunk = chunk_table_ + pos;
      }
    }
    LOG(FATAL) << "dict OOM";
    return nullptr;
  }
  bool Delete(const KeyT& key) {
  RETRY:
    const uint64_t hash_value = Hash(key);
    const auto sign           = Sign(hash_value);
    const size_t step         = ProbeDelta(sign);
    size_t pos                = hash_value & (chunk_num_ - 1);
    for (size_t i = 0; i < chunk_num_; ++i) {
      auto* chunk = chunk_table_ + pos;
      base::AutoLock lock(chunk->lock_);
      if (UNLIKELY(chunk->is_migrating))
        goto RETRY;
      const int id = chunk->Find(key, sign);
      if (id != -1) {
        DeleteLocked(pos, id);
        return true;
      }
      if (chunk->OverflowCount() == 0)
        break;
      pos = (pos + step) & (chunk_num_ - 1);
    }
    return true;
  }

  ValueT* GetById(uint64_t pos, KeyT* key = nullptr) {
    auto it = chunk_table_[pos / kChunkSize].Get(pos % kChunkSize);
    if (key != nullptr)
      *key = it->first;
    return &it->second;
  }

  bool IndexInUse(uint64_t pos) {
    if (pos >= capacity_)
      return false;
    return chunk_table_[pos / kChunkSize].InUse(pos % kChunkSize);
  }

  int64_t Lookup(const KeyT& key) {
    uint64_t pos;
    auto [nouse0, nouse1, exists] = GetInternal(key, &pos);
    if (exists)
      return pos;
    return -1;
  }

  uint64_t Size() const {
    uint64_t size = 0;
    for (const auto& shard : size_shards_)
      size += shard.load(std::memory_order_relaxed);
    return size;
  }

  uint32_t Capacity() { return capacity_; }

  inline bool Full() { return Size() * 100 >= capacity_ * kMaxLoadFactor; }

private:
  // The caller holds the target chunk's lock.
  void DeleteLocked(size_t chunk_id, size_t chunk_offset) {
    const KeyT key            = chunk_table_[chunk_id].Get(chunk_offset)->first;
    const uint64_t hash_value = Hash(key);
    const size_t step         = ProbeDelta(Sign(hash_value));
    size_t pos                = hash_value & (chunk_num_ - 1);
    while (pos != chunk_id) {
      chunk_table_[pos].DecOverflow();
      pos = (pos + step) & (chunk_num_ - 1);
    }
    chunk_table_[chunk_id].Delete(chunk_offset);
    DecrementSize(key);
  }

  void ResetSize() {
    for (auto& shard : size_shards_)
      shard.store(0, std::memory_order_relaxed);
  }

  void IncrementSize(const KeyT& key) {
    size_shards_[Hash(key) & (kSizeShardCount - 1)].fetch_add(
        1, std::memory_order_relaxed);
  }

  void DecrementSize(const KeyT& key) {
    size_shards_[Hash(key) & (kSizeShardCount - 1)].fetch_sub(
        1, std::memory_order_relaxed);
  }

  inline size_t ProbeDelta(size_t key) const { return key * 2 | 1; }

  static inline uint8_t Sign(const uint64_t& key) { // NOLINT
    uint64_t c = _mm_crc32_u64(0, key);             // NOLINT
    // ??????
    // WARNING(xieminhui): diff from kuaishou
    // key += c;
    return ((c >> 24) | 0x80) & kSignMask;
  }

  static inline uint64_t Hash(uint64_t key) { return key * 0xc6a4a7935bd1e995; }

  uint64_t capacity_;
  uint64_t chunk_num_;
  std::atomic<uint64_t> size_shards_[kSizeShardCount];
  uint8_t not_use_[Persistence ? 64 - 3 * sizeof(uint64_t) : 0];
  ChunkT chunk_table_[0];
};

} // namespace base
