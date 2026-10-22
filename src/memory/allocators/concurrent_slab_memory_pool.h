#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/async_time.h"
#include "base/base.h"
#include "base/bitmap.h"
#include "base/factory.h"
#include "base/log.h"
#include "base/string.h"
#include <fmt/format.h>
#include "memory/malloc.h"
#include "memory/shm_file.h"

namespace base {

namespace concurrent_slab_detail {

inline uint64_t* BitmapWords(base::BitMap* bitmap) {
  return reinterpret_cast<uint64_t*>(
      reinterpret_cast<char*>(bitmap) + sizeof(base::BitMap));
}

inline int BitmapNumWords(int nr_entries) { return nr_entries / 64; }

inline int& AllocWordHintTls() {
  thread_local int hint = 0;
  return hint;
}

inline int
TryAllocFromBitmap(uint64_t* words, int n_words, int start_word = 0) {
  for (int w = 0; w < n_words; ++w) {
    const int i  = (start_word + w) % n_words;
    uint64_t old = words[i];
    while (true) {
      const uint64_t inv = ~old;
      if (!inv) {
        break;
      }
      const uint64_t pos     = static_cast<uint64_t>(__builtin_ctzll(inv));
      const uint64_t desired = old | (1ull << pos);
      if (__sync_bool_compare_and_swap(&words[i], old, desired)) {
        return i * 64 + static_cast<int>(pos);
      }
      old = words[i];
    }
  }
  return -1;
}

inline bool TryFreeBitmapBit(uint64_t* words, int bit) {
  const int i               = bit / 64;
  const int pos             = bit % 64;
  const uint64_t clear_mask = ~(1ull << pos);
  uint64_t old              = words[i];
  while (true) {
    if ((old & (1ull << pos)) == 0) {
      return false;
    }
    const uint64_t desired = old & clear_mask;
    if (__sync_bool_compare_and_swap(&words[i], old, desired)) {
      return true;
    }
    old = words[i];
  }
}

// Size classes for the legacy (8-byte metadata + payload) layout. The
// `8 + dim * 4` widths cover embedding rows: 8+80 serves a dim-20 row and
// 8+272 serves a dim-68 row, which would otherwise round up to 136 / 520.
inline const std::vector<int>& DefaultSlabSizeClasses() {
  static const std::vector<int> kClasses = {
      8 + 32, 8 + 64, 8 + 80, 8 + 128, 8 + 272, 8 + 512, 8 + 1024};
  return kClasses;
}

} // namespace concurrent_slab_detail

template <bool PERFECT_FIT_MOD = true>
class ConcurrentSlabMemoryPool : public MallocApi {
  static constexpr uint64_t kChunkSize = 1 * 1024 * 1024ULL;
  static constexpr int kMetaDataSize   = 8;

  class SlabChunk {
  public:
    SlabChunk(char* chunk_start, uint64_t chunk_id)
        : header_(reinterpret_cast<ChunkHeader*>(chunk_start)),
          data_(nullptr),
          chunk_id_(chunk_id),
          allocated_entries_(0),
          listed_in_partial_(false) {}

    void Initialize() {
      header_->slab_size_  = 0;
      header_->nr_entries_ = 0;
      data_                = nullptr;
      allocated_entries_.store(0, std::memory_order_relaxed);
      listed_in_partial_.store(false, std::memory_order_relaxed);
    }

    void Use(int slab_size) {
      CHECK_EQ(header_->slab_size_, 0)
          << "chunk " << chunk_id_ << " already bound to slab "
          << header_->slab_size_;

      int nr_entries = static_cast<int>(kChunkSize / slab_size);
      nr_entries -= (nr_entries % 64);

      while (ChunkHeader::HeaderSize(nr_entries) + nr_entries * slab_size >
             kChunkSize) {
        nr_entries -= 64;
      }
      CHECK_GT(nr_entries, 0);

      header_->slab_size_  = slab_size;
      header_->nr_entries_ = nr_entries;
      new (header_->bitmap_) base::BitMap(nr_entries);
      header_->bitmap_->Clear();
      data_ = reinterpret_cast<char*>(header_) + header_->HeaderSize();
      allocated_entries_.store(0, std::memory_order_relaxed);
      listed_in_partial_.store(false, std::memory_order_relaxed);
    }

    bool IsActivated() const { return header_->slab_size_ != 0; }

    bool Full() const {
      return allocated_entries_.load(std::memory_order_relaxed) >=
             header_->nr_entries_;
    }

    char* Malloc() {
      CHECK(IsActivated());
      const int nr_entries = header_->nr_entries_;
      uint64_t* words   = concurrent_slab_detail::BitmapWords(header_->bitmap_);
      const int n_words = concurrent_slab_detail::BitmapNumWords(nr_entries);
      int& word_hint    = concurrent_slab_detail::AllocWordHintTls();
      const int entry_id = concurrent_slab_detail::TryAllocFromBitmap(
          words, n_words, n_words > 0 ? word_hint % n_words : 0);
      if (entry_id < 0) {
        return nullptr;
      }
      word_hint = (entry_id / 64 + 1) % n_words;
      allocated_entries_.fetch_add(1, std::memory_order_relaxed);
      return Entry(entry_id);
    }

    // Returns true if the chunk was full before this free (now has free slots).
    bool Free(void* memory_data) {
      CHECK(IsActivated());
      const int nr_entries = header_->nr_entries_;
      const bool was_full =
          allocated_entries_.load(std::memory_order_relaxed) >= nr_entries;
      const int entry_id = EntryId(memory_data);
      uint64_t* words = concurrent_slab_detail::BitmapWords(header_->bitmap_);
      CHECK(concurrent_slab_detail::TryFreeBitmapBit(words, entry_id))
          << "double free or corrupt pointer in chunk " << chunk_id_;
      allocated_entries_.fetch_sub(1, std::memory_order_relaxed);
      return was_full;
    }

    bool TryMarkListedInPartial() {
      bool expected = false;
      return listed_in_partial_.compare_exchange_strong(
          expected, true, std::memory_order_relaxed);
    }

    void ClearListedInPartial() {
      listed_in_partial_.store(false, std::memory_order_relaxed);
    }

    int SlabSize() const { return header_->slab_size_; }

    int AllocatedEntryNumber() const {
      return allocated_entries_.load(std::memory_order_relaxed);
    }

    std::vector<char*> GetMallocedData() const {
      std::vector<char*> return_data;
      if (!IsActivated()) {
        return return_data;
      }
      for (int i = 0; i < header_->nr_entries_; ++i) {
        if (header_->bitmap_->Get(i)) {
          return_data.push_back(Entry(i));
        }
      }
      return return_data;
    }

    uint64_t chunk_id() const { return chunk_id_; }

  private:
    int EntryId(void* memory_data) const {
      return static_cast<int>(
          (reinterpret_cast<char*>(memory_data) - data_) / SlabSize());
    }

    char* Entry(int entry_id) const { return data_ + entry_id * SlabSize(); }

    struct ChunkHeader {
      int slab_size_  = 0;
      int nr_entries_ = 0;
      base::BitMap bitmap_[0];

      size_t HeaderSize() const {
        return sizeof(ChunkHeader) + base::BitMap::MemorySize(nr_entries_);
      }

      static size_t HeaderSize(int nr_entries) {
        return sizeof(ChunkHeader) + base::BitMap::MemorySize(nr_entries);
      }

      DISALLOW_COPY_AND_ASSIGN(ChunkHeader);
    };

    ChunkHeader* header_;
    char* data_;
    uint64_t chunk_id_;
    std::atomic<int> allocated_entries_;
    std::atomic<bool> listed_in_partial_;
    DISALLOW_COPY_AND_ASSIGN(SlabChunk);
  };

  struct SlabState {
    base::Lock lock;
    std::deque<SlabChunk*> partial;
    std::deque<SlabChunk*> active;
  };

  struct ThreadLocalPool {
    uint64_t generation = 0;
    std::unordered_map<int, SlabChunk*> hot_by_slab;
  };

  ThreadLocalPool& PoolTls() {
    thread_local ThreadLocalPool tls;
    // A thread may serve multiple allocators, and allocator addresses may be
    // reused after destruction. Never retain another instance's hot chunks.
    // Dropped hot chunks remain discoverable through their owner's active list.
    if (tls.generation != tls_generation_) {
      tls.hot_by_slab.clear();
      tls.generation = tls_generation_;
    }
    return tls;
  }

public:
  ConcurrentSlabMemoryPool(const std::string& filename,
                           int64 memory_size,
                           const std::vector<int>& slab_sizes)
      : allocated_slab_sizes_(slab_sizes.begin(), slab_sizes.end()) {
    if (base::file_util::PathExists(filename)) {
      CHECK(base::file_util::Delete(filename, false));
    }
    shm_file_ =
        ShmFile::New(ShmFile::ConfigForMedium("DRAM", filename, memory_size));
    CHECK(shm_file_) << filename << " " << memory_size;

    const int64 aligned_size =
        memory_size - (memory_size % static_cast<int64>(kChunkSize));
    nr_chunks_ = aligned_size / static_cast<int64>(kChunkSize);
    CHECK_GT(nr_chunks_, 0);

    chunks_.reserve(static_cast<size_t>(nr_chunks_));
    for (int64 i = 0; i < nr_chunks_; ++i) {
      auto* chunk = new SlabChunk(
          shm_file_->Data() + i * kChunkSize, static_cast<uint64_t>(i));
      chunk->Initialize();
      chunks_.push_back(chunk);
      free_chunks_.push_back(chunk);
    }

    for (int slab : slab_sizes) {
      (void)slab_states_[slab];
    }

    LOG(INFO) << "ConcurrentSlabMemoryPool: initialized " << nr_chunks_
              << " chunks of " << kChunkSize << " bytes.";
  }

  ~ConcurrentSlabMemoryPool() override {
    for (auto* chunk : chunks_) {
      delete chunk;
    }
  }

  char* New(int memory_size) override {
    if (PERFECT_FIT_MOD) {
      return NewInternal(memory_size);
    }
    auto iter = allocated_slab_sizes_.lower_bound(kMetaDataSize + memory_size);
    CHECK(iter != allocated_slab_sizes_.end());
    char* ptr                    = NewInternal(*iter);
    *reinterpret_cast<int*>(ptr) = memory_size;
    return ptr + kMetaDataSize;
  }

  bool Free(void* memory_data) override {
    if (!PERFECT_FIT_MOD) {
      memory_data = static_cast<char*>(memory_data) - kMetaDataSize;
    }

    const int64 chunk_id = GetChunkID(memory_data);
    if (chunk_id < 0 || chunk_id >= nr_chunks_) {
      return false;
    }
    SlabChunk* chunk     = chunks_[static_cast<size_t>(chunk_id)];
    const bool was_full  = chunk->Free(memory_data);
    const int slab_size  = chunk->SlabSize();
    ThreadLocalPool& tls = PoolTls();
    SlabChunk*& hot      = tls.hot_by_slab[slab_size];
    if (hot == nullptr && !chunk->Full()) {
      chunk->ClearListedInPartial();
      hot = chunk;
    } else if (was_full) {
      TryEnqueuePartial(slab_size, chunk);
    }
    total_malloc_.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }

  void GetMallocsAppend(std::vector<char*>* mallocs_data) const override {
    for (auto* chunk : chunks_) {
      for (char* each : chunk->GetMallocedData()) {
        if (PERFECT_FIT_MOD) {
          mallocs_data->push_back(each);
        } else {
          mallocs_data->push_back(each + kMetaDataSize);
        }
      }
    }
  }

  void GetMallocsAppend(std::vector<int64>* mallocs_offset) const override {
    std::vector<char*> mallocs_data;
    GetMallocsAppend(&mallocs_data);
    for (char* each : mallocs_data) {
      mallocs_offset->push_back(each - shm_file_->Data());
    }
  }

  void Initialize() override {
    for (auto* chunk : chunks_) {
      chunk->Initialize();
    }
    free_chunks_.clear();
    for (auto* chunk : chunks_) {
      free_chunks_.push_back(chunk);
    }
    for (auto& [slab, state] : slab_states_) {
      (void)slab;
      state.partial.clear();
      state.active.clear();
    }
    // Invalidate cached chunks on every worker, not just the resetting thread.
    tls_generation_ =
        next_tls_generation_.fetch_add(1, std::memory_order_relaxed);
    PoolTls().hot_by_slab.clear();
    total_malloc_.store(0, std::memory_order_relaxed);
  }

  char* BackingData() const override { return shm_file_->Data(); }
  int64 BackingSize() const override {
    return nr_chunks_ * static_cast<int64>(kChunkSize);
  }

  std::string GetInfo() const override {
    uint64_t malloced_bytes = 0;
    uint64_t malloced_chunk = 0;
    for (auto* each : chunks_) {
      if (each->IsActivated()) {
        malloced_bytes += static_cast<uint64_t>(each->AllocatedEntryNumber()) *
                          each->SlabSize();
        malloced_chunk++;
      }
    }
    return base::SFormat(
        "[ConcurrentSlabMemoryPool] "
        "Use {} of {} chunks, Util {} %",
        malloced_chunk,
        chunks_.size(),
        chunks_.empty() ? 0
                        : 100 * malloced_bytes / (kChunkSize * chunks_.size()));
  }

  char* GetMallocData(int64 offset) const override {
    return shm_file_->Data() + offset;
  }

  int GetMallocSize(int64 offset) const override {
    return GetMallocSize(GetMallocData(offset));
  }

  int64 GetMallocOffset(const char* data) const override {
    const auto ret = data - shm_file_->Data();
    CHECK_EQ((ret >> 3) << 3, ret);
    return ret;
  }

  int GetMallocSize(const char* data) const override {
    if (PERFECT_FIT_MOD) {
      return chunks_[static_cast<size_t>(GetChunkID(data))]->SlabSize();
    }
    return *reinterpret_cast<const int*>(data - kMetaDataSize);
  }

  uint64_t total_malloc() const override {
    return static_cast<uint64_t>(total_malloc_.load(std::memory_order_relaxed));
  }

  bool Healthy() const override { return true; }

  void AddMallocs4Recovery(int64_t /*shm_offset*/) override {
    RECSTORE_LOG_EVERY_MS(ERROR, 1000) << "AddMallocs4Recovery not implement";
  }

private:
  void AbandonHotChunk(int slab_size, SlabChunk*& hot) {
    SlabChunk* abandoned = hot;
    hot                  = nullptr;
    if (abandoned != nullptr && !abandoned->Full()) {
      TryEnqueuePartial(slab_size, abandoned);
    }
  }

  void TryEnqueuePartial(int slab_size, SlabChunk* chunk) {
    if (chunk == nullptr || chunk->Full() || !chunk->TryMarkListedInPartial()) {
      return;
    }
    auto state_it = slab_states_.find(slab_size);
    CHECK(state_it != slab_states_.end());
    base::AutoLock slab_lock(state_it->second.lock);
    if (chunk->Full()) {
      chunk->ClearListedInPartial();
      return;
    }
    state_it->second.partial.push_back(chunk);
  }

  char* NewInternal(int slab_size) {
    ThreadLocalPool& tls = PoolTls();
    SlabChunk*& hot      = tls.hot_by_slab[slab_size];

    while (true) {
      if (hot != nullptr && !hot->Full()) {
        char* ptr = hot->Malloc();
        if (ptr != nullptr) {
          total_malloc_.fetch_add(1, std::memory_order_relaxed);
          return ptr;
        }
        // Another thread filled the chunk after the Full() check.
        AbandonHotChunk(slab_size, hot);
      }

      SlabChunk* chunk = AcquireChunk(slab_size, &hot);
      char* ptr        = chunk->Malloc();
      if (ptr != nullptr) {
        total_malloc_.fetch_add(1, std::memory_order_relaxed);
        return ptr;
      }
      // The selected chunk became full after AcquireChunk() released the slab
      // lock, or Full() briefly lagged behind the bitmap state.
      if (hot == chunk) {
        AbandonHotChunk(slab_size, hot);
      }
    }
  }

  SlabChunk* AcquireChunk(int slab_size, SlabChunk** hot_slot) {
    auto state_it = slab_states_.find(slab_size);
    CHECK(state_it != slab_states_.end());
    SlabState& state = state_it->second;

    base::AutoLock slab_lock(state.lock);

    SlabChunk*& hot = *hot_slot;
    if (hot != nullptr && hot->Full()) {
      hot = nullptr;
    }

    while (!state.partial.empty()) {
      SlabChunk* chunk = state.partial.front();
      state.partial.pop_front();
      chunk->ClearListedInPartial();
      if (chunk->Full()) {
        continue;
      }
      chunk->ClearListedInPartial();
      hot = chunk;
      return chunk;
    }

    {
      base::AutoLock free_lock(free_chunks_lock_);
      if (!free_chunks_.empty()) {
        SlabChunk* chunk = free_chunks_.front();
        free_chunks_.pop_front();
        chunk->Use(slab_size);
        state.active.push_back(chunk);
        chunk->ClearListedInPartial();
        hot = chunk;
        return chunk;
      }
    }

    for (SlabChunk* chunk : state.active) {
      if (!chunk->Full()) {
        chunk->ClearListedInPartial();
        hot = chunk;
        return chunk;
      }
    }

    LOG(FATAL) << fmt::format(
        "ConcurrentSlabMemoryPool OOM, slab_size={}, total_malloc={}",
        slab_size,
        total_malloc_.load(std::memory_order_relaxed));
    return nullptr;
  }

  int64 GetChunkID(const void* data) const {
    return GetChunkID(static_cast<const char*>(data) - shm_file_->Data());
  }

  int64 GetChunkID(int64 offset) const {
    return offset / static_cast<int64>(kChunkSize);
  }

  std::unique_ptr<ShmFile> shm_file_;
  int64 nr_chunks_ = 0;
  std::vector<SlabChunk*> chunks_;
  std::deque<SlabChunk*> free_chunks_;
  base::Lock free_chunks_lock_;
  std::unordered_map<int, SlabState> slab_states_;
  std::set<int> allocated_slab_sizes_;
  std::atomic<int64> total_malloc_{0};
  inline static std::atomic<uint64_t> next_tls_generation_{1};
  uint64_t tls_generation_ =
      next_tls_generation_.fetch_add(1, std::memory_order_relaxed);
  DISALLOW_COPY_AND_ASSIGN(ConcurrentSlabMemoryPool);
};

class ConcurrentSlabMemoryPoolMalloc : public ConcurrentSlabMemoryPool<false> {
public:
  ConcurrentSlabMemoryPoolMalloc(const std::string& filename,
                                 int64 memory_size,
                                 const std::string& /*medium*/)
      : ConcurrentSlabMemoryPool<false>(
            filename,
            memory_size,
            concurrent_slab_detail::DefaultSlabSizeClasses()) {}
};

// Same pool, but the caller supplies the size-class list
// (value.dram_allocator.size_classes). The classes are exact minimum sizes:
// a request is served by the first class >= kMetaDataSize + request.
class ConcurrentSlabMemoryPoolSized : public ConcurrentSlabMemoryPool<false> {
public:
  ConcurrentSlabMemoryPoolSized(const std::string& filename,
                                int64 memory_size,
                                const std::string& /*medium*/,
                                const std::vector<int>& size_classes)
      : ConcurrentSlabMemoryPool<false>(
            filename,
            memory_size,
            size_classes.empty()
                ? concurrent_slab_detail::DefaultSlabSizeClasses()
                : size_classes) {}
};

FACTORY_REGISTER(MallocApi,
                 CONCURRENT_SLAB_MEMORY_POOL,
                 ConcurrentSlabMemoryPoolMalloc,
                 const std::string&,
                 int64,
                 const std::string&);

// The sized variant answers the same factory key but takes an explicit
// size-class list; DramValueStore picks it when
// value.dram_allocator.size_classes is set. The registration lives in its own
// namespace only because FACTORY_REGISTER names its static by key.
namespace {

FACTORY_REGISTER(MallocApi,
                 CONCURRENT_SLAB_MEMORY_POOL,
                 ConcurrentSlabMemoryPoolSized,
                 const std::string&,
                 int64,
                 const std::string&,
                 const std::vector<int>&);

} // namespace

} // namespace base
