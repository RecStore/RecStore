#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#include "storage/allocator/ssd/ssd_block_allocator.h"
#include "storage/io_backend/io_backend.h"

class SsdSlabAllocator : public SsdBlockAllocator {
public:
  SsdSlabAllocator(IOBackend* backend,
                   const std::vector<int>& size_classes,
                   uint64_t total_capacity_bytes,
                   uint64_t base_byte_offset)
      : backend_(backend) {
    if (backend_ == nullptr) {
      throw std::invalid_argument("SsdSlabAllocator backend is null");
    }
    size_classes_ =
        size_classes.empty() ? std::vector<int>{128, 256, 512, 1024, 4096}
                             : size_classes;
    std::sort(size_classes_.begin(), size_classes_.end());

    const uint64_t total_blocks = total_capacity_bytes / kBlockSize;
    global_free_blocks_.reserve(total_blocks);
    for (uint64_t i = 0; i < total_blocks; ++i) {
      global_free_blocks_.push_back(base_byte_offset + i * kBlockSize);
    }

    for (int cls : size_classes_) {
      auto pool       = std::make_unique<SlabPool>();
      pool->slot_size = static_cast<uint64_t>(cls) + kHeaderSize;
      if (pool->slot_size == 0 || kBlockSize < pool->slot_size) {
        throw std::invalid_argument(
            "SsdSlabAllocator invalid slot size vs block");
      }
      pool->slots_per_block = kBlockSize / pool->slot_size;
      slabs_.push_back(std::move(pool));
    }
  }

  uint64_t Alloc(size_t data_size) override {
    const uint16_t slab_idx = SelectSlab(data_size);
    auto& slab              = *slabs_.at(slab_idx);

    for (;;) {
      std::unique_lock<std::mutex> slab_lock(slab.mu);
      while (!slab.partial_blocks.empty()) {
        const uint64_t block_idx = slab.partial_blocks.back();
        Block& b                 = slab.blocks.at(block_idx);
        if (b.tombstone) {
          RemoveFromPartial(slab, block_idx);
          continue;
        }
        if (!HasSpace(b, slab)) {
          RemoveFromPartial(slab, block_idx);
          continue;
        }
        uint64_t slot_in_block = 0;
        if (!b.free_in_block.empty()) {
          slot_in_block = b.free_in_block.back();
          b.free_in_block.pop_back();
        } else {
          slot_in_block = b.cursor;
          b.cursor++;
        }
        b.used_count++;
        UpdatePartialAfterAlloc(slab, block_idx, b);
        const uint64_t local_id =
            block_idx * slab.slots_per_block + slot_in_block;
        return Encode(slab_idx, local_id);
      }

      slab_lock.unlock();

      // Serialize block claims. The winner publishes the claimed block into
      // partial_blocks before releasing this lock, so a waiter re-checks the
      // list instead of mistaking an in-flight claim for exhaustion.
      std::lock_guard<std::mutex> claim_lock(block_claim_mu_);
      slab_lock.lock();
      if (!slab.partial_blocks.empty()) {
        continue;
      }
      slab_lock.unlock();

      std::unique_lock<std::mutex> g_lock(global_mu_);
      if (global_free_blocks_.empty()) {
        return kInvalidHandle;
      }
      const uint64_t byte_off = global_free_blocks_.back();
      global_free_blocks_.pop_back();
      g_lock.unlock();

      slab_lock.lock();
      uint64_t block_idx = 0;
      Block nb;
      nb.byte_offset = byte_off;
      nb.cursor      = 0;
      nb.used_count  = 0;
      nb.tombstone   = false;

      if (!slab.tombstone_indices.empty()) {
        block_idx = slab.tombstone_indices.back();
        slab.tombstone_indices.pop_back();
        slab.blocks[block_idx] = std::move(nb);
      } else {
        block_idx = slab.blocks.size();
        slab.blocks.push_back(std::move(nb));
      }

      Block& b                     = slab.blocks[block_idx];
      const uint64_t slot_in_block = 0;
      b.cursor                     = 1;
      b.used_count                 = 1;
      UpdatePartialAfterAlloc(slab, block_idx, b);
      const uint64_t local_id =
          block_idx * slab.slots_per_block + slot_in_block;
      return Encode(slab_idx, local_id);
    }
  }

  void Free(uint64_t handle) override {
    uint16_t slab_idx;
    uint64_t local_slot_id;
    Decode(handle, slab_idx, local_slot_id);
    auto& slab                   = *slabs_.at(slab_idx);
    const uint64_t block_idx     = local_slot_id / slab.slots_per_block;
    const uint64_t slot_in_block = local_slot_id % slab.slots_per_block;

    std::unique_lock<std::mutex> slab_lock(slab.mu);
    if (block_idx >= slab.blocks.size()) {
      return;
    }
    Block& b = slab.blocks[block_idx];
    if (b.tombstone) {
      return;
    }
    b.used_count--;
    b.free_in_block.push_back(slot_in_block);
    UpdatePartialAfterFree(slab, block_idx, b);

    if (b.used_count == 0) {
      const uint64_t off               = b.byte_offset;
      slab.blocks[block_idx]           = Block{};
      slab.blocks[block_idx].tombstone = true;
      RemoveFromPartial(slab, block_idx);
      slab.tombstone_indices.push_back(block_idx);
      slab_lock.unlock();
      std::lock_guard<std::mutex> g_lock(global_mu_);
      global_free_blocks_.push_back(off);
    }
  }

  void Write(uint64_t handle, const void* data, size_t data_size) override {
    uint16_t slab_idx;
    uint64_t local_slot_id;
    Decode(handle, slab_idx, local_slot_id);
    const auto& slab = *slabs_.at(slab_idx);
    if (data_size + kHeaderSize > slab.slot_size) {
      throw std::invalid_argument("SsdSlabAllocator write exceeds slot size");
    }
    const uint64_t byte_off = ResolveSlotByteOffset(slab_idx, local_slot_id);
    WriteBytes(byte_off, data, data_size);
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    uint16_t slab_idx;
    uint64_t local_slot_id;
    Decode(handle, slab_idx, local_slot_id);
    const auto& slab        = *slabs_.at(slab_idx);
    const uint64_t byte_off = ResolveSlotByteOffset(slab_idx, local_slot_id);
    return ReadBytes(byte_off, slab.slot_size, out_buf, buf_size);
  }

  uint64_t AllocAndWrite(const void* data, size_t data_size) override {
    const uint64_t handle = Alloc(data_size);
    if (handle != kInvalidHandle) {
      try {
        Write(handle, data, data_size);
      } catch (...) {
        Free(handle);
        throw;
      }
    }
    return handle;
  }

  void BatchRead(const std::vector<ReadEntry>& entries,
                 std::vector<size_t>& out_sizes) override {
    out_sizes.assign(entries.size(), 0);
    struct PendingRead {
      uint64_t byte_off = 0;
      size_t buf_size = 0;
    };
    std::vector<IOBackend::IOEntry> io_entries;
    std::vector<PendingRead> pending;
    std::vector<uint64_t> pages;
    io_entries.reserve(entries.size());
    pending.reserve(entries.size());
    pages.reserve(entries.size() * 2);
    for (size_t i = 0; i < entries.size(); ++i) {
      uint16_t slab_idx;
      uint64_t local_slot_id;
      Decode(entries[i].handle, slab_idx, local_slot_id);
      const auto& slab        = *slabs_.at(slab_idx);
      const uint64_t byte_off = ResolveSlotByteOffset(slab_idx, local_slot_id);
      const PageID_t start    = byte_off / PAGE_SIZE;
      const uint64_t page_off = byte_off % PAGE_SIZE;
      const uint64_t total    = page_off + slab.slot_size;
      const uint64_t spans    = (total + PAGE_SIZE - 1) / PAGE_SIZE;
      pending.push_back(PendingRead{
          byte_off,
          entries[i].out_buf == nullptr ? 0 : slab.slot_size - kHeaderSize});
      for (uint64_t page = 0; page < spans; ++page) {
        pages.push_back(start + page);
      }
    }
    if (pages.empty()) {
      return;
    }
    SortUnique(pages);
    auto free_buffer = [this](char* ptr) { backend_->FreeBuffer(ptr); };
    std::unique_ptr<char, decltype(free_buffer)> buffer(
        backend_->AllocateBuffer(pages.size()), free_buffer);
    // Keep unique pages in physical order, so even a slot crossing a page
    // boundary is contiguous in the scratch buffer. Read each page only once.
    for (size_t i = 0; i < pages.size(); ++i) {
      if (!io_entries.empty() &&
          io_entries.back().page_id + io_entries.back().page_count == pages[i]) {
        ++io_entries.back().page_count;
      } else {
        io_entries.push_back({pages[i], buffer.get() + i * PAGE_SIZE, 1});
      }
    }
    // Readers share the page stripes with writers so a read cannot observe a
    // half written slot while its page is being read-modify-written.
    std::vector<std::shared_lock<std::shared_mutex>> locks;
    LockStripes(pages, locks);
    backend_->BatchReadPages(io_entries);
    for (size_t i = 0; i < pending.size(); ++i) {
      const auto& item = pending[i];
      const size_t page_index = static_cast<size_t>(
          std::lower_bound(pages.begin(), pages.end(), item.byte_off / PAGE_SIZE) -
          pages.begin());
      const char* src = buffer.get() + page_index * PAGE_SIZE +
                        item.byte_off % PAGE_SIZE;
      uint32_t n       = 0;
      std::memcpy(&n, src, sizeof(n));
      const size_t actual = std::min(item.buf_size, static_cast<size_t>(n));
      if (actual > 0) {
        std::memcpy(entries[i].out_buf,
                    src + kHeaderSize,
                    actual);
      }
      out_sizes[i] = actual;
    }
  }

  std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteEntry>& entries) override {
    std::vector<uint64_t> handles;
    handles.reserve(entries.size());
    // Length headers stay addressable for the whole batch: page patches hold
    // pointers into this storage until the write completes.
    std::vector<uint32_t> lengths(entries.size(), 0);
    std::vector<PagePatch> patches;
    patches.reserve(entries.size() * 2);
    try {
      for (size_t i = 0; i < entries.size(); ++i) {
        const uint64_t handle = Alloc(entries[i].data_size);
        handles.push_back(handle);
        if (handle == kInvalidHandle) {
          continue;
        }
        uint16_t slab_idx;
        uint64_t local_slot_id;
        Decode(handle, slab_idx, local_slot_id);
        const uint64_t byte_off = ResolveSlotByteOffset(slab_idx, local_slot_id);
        lengths[i]              = static_cast<uint32_t>(entries[i].data_size);
        AddPagePatch(patches, byte_off, &lengths[i], sizeof(lengths[i]));
        AddPagePatch(patches,
                     byte_off + kHeaderSize,
                     entries[i].data,
                     entries[i].data_size);
      }
      WritePages(patches);
    } catch (...) {
      for (uint64_t handle : handles) {
        if (handle != kInvalidHandle) Free(handle);
      }
      throw;
    }
    return handles;
  }

  void BatchWrite(const std::vector<UpdateEntry>& entries) override {
    std::vector<uint32_t> lengths(entries.size());
    std::vector<PagePatch> patches;
    patches.reserve(entries.size() * 2);
    for (size_t i = 0; i < entries.size(); ++i) {
      const auto& entry = entries[i];
      if (entry.data_size > SlotCapacity(entry.handle)) {
        throw std::invalid_argument("SSD batch update exceeds slot capacity");
      }
      uint16_t slab_idx;
      uint64_t local_slot_id;
      Decode(entry.handle, slab_idx, local_slot_id);
      const uint64_t offset = ResolveSlotByteOffset(slab_idx, local_slot_id);
      lengths[i] = static_cast<uint32_t>(entry.data_size);
      AddPagePatch(patches, offset, &lengths[i], sizeof(lengths[i]));
      AddPagePatch(patches, offset + kHeaderSize, entry.data, entry.data_size);
    }
    WritePages(patches);
  }

  size_t SlotCapacity(uint64_t handle) const override {
    uint16_t slab_idx;
    uint64_t local_slot_id;
    Decode(handle, slab_idx, local_slot_id);
    (void)local_slot_id;
    return slabs_.at(slab_idx)->slot_size - kHeaderSize;
  }

  /// For unit tests: physical chunk size when carving the backing store.
  static constexpr uint64_t kBlockSize = 1ULL << 26;

private:
  struct alignas(64) PageStripe {
    std::shared_mutex mu;
  };

  /// One per-page slice of a slot write. Patches are collected first and
  /// applied inside a single page read-modify-write cycle.
  struct PagePatch {
    uint64_t page   = 0; // absolute page id
    size_t offset   = 0; // byte offset inside the page
    const char* src = nullptr;
    size_t size     = 0;
  };

  struct Block {
    uint64_t byte_offset = 0;
    uint64_t cursor      = 0;
    uint64_t used_count  = 0;
    std::vector<uint64_t> free_in_block;
    bool tombstone = false;
  };

  struct SlabPool {
    uint64_t slot_size       = 0;
    uint64_t slots_per_block = 0;
    std::vector<Block> blocks;
    std::vector<uint64_t> tombstone_indices;
    std::vector<uint64_t> partial_blocks;
    mutable std::mutex mu;
  };

  static bool HasSpace(const Block& b, const SlabPool& slab) {
    return !b.free_in_block.empty() || b.cursor < slab.slots_per_block;
  }

  static void RemoveFromPartial(SlabPool& slab, uint64_t block_idx) {
    auto& v = slab.partial_blocks;
    v.erase(std::remove(v.begin(), v.end(), block_idx), v.end());
  }

  static void
  AddToPartialIfSpace(SlabPool& slab, uint64_t block_idx, const Block& b) {
    if (HasSpace(b, slab)) {
      if (std::find(slab.partial_blocks.begin(),
                    slab.partial_blocks.end(),
                    block_idx) == slab.partial_blocks.end()) {
        slab.partial_blocks.push_back(block_idx);
      }
    }
  }

  void
  UpdatePartialAfterAlloc(SlabPool& slab, uint64_t block_idx, const Block& b) {
    if (HasSpace(b, slab)) {
      AddToPartialIfSpace(slab, block_idx, b);
    } else {
      RemoveFromPartial(slab, block_idx);
    }
  }

  void
  UpdatePartialAfterFree(SlabPool& slab, uint64_t block_idx, const Block& b) {
    AddToPartialIfSpace(slab, block_idx, b);
  }

  uint64_t
  ResolveSlotByteOffset(uint16_t slab_idx, uint64_t local_slot_id) const {
    auto& slab = *slabs_.at(slab_idx);
    std::lock_guard<std::mutex> lock(slab.mu);
    const uint64_t block_idx     = local_slot_id / slab.slots_per_block;
    const uint64_t slot_in_block = local_slot_id % slab.slots_per_block;
    if (block_idx >= slab.blocks.size()) {
      return 0;
    }
    const Block& b = slab.blocks[block_idx];
    if (b.tombstone) {
      return 0;
    }
    return b.byte_offset + slot_in_block * slab.slot_size;
  }

  static uint64_t Encode(uint16_t slab_idx, uint64_t local_slot_id) {
    return (static_cast<uint64_t>(slab_idx) << 48) |
           ((local_slot_id + 1) & 0x0000FFFFFFFFFFFFULL);
  }

  static void
  Decode(uint64_t handle, uint16_t& slab_idx, uint64_t& local_slot_id) {
    slab_idx      = static_cast<uint16_t>(handle >> 48);
    local_slot_id = (handle & 0x0000FFFFFFFFFFFFULL) - 1;
  }

  static void SortUnique(std::vector<uint64_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  }

  /// Locks the stripes covering `pages`. Stripes are always taken in ascending
  /// order so concurrent readers and writers cannot deadlock.
  template <typename LockType>
  void LockStripes(std::vector<uint64_t> pages, std::vector<LockType>& locks) {
    for (auto& page : pages) {
      page %= kPageLockStripes;
    }
    SortUnique(pages);
    locks.reserve(locks.size() + pages.size());
    for (uint64_t stripe : pages) {
      locks.emplace_back(page_stripes_[stripe].mu);
    }
  }

  /// Splits `[byte_offset, byte_offset + size)` into per-page copy requests.
  /// The referenced source memory must outlive the matching WritePages call.
  void AddPagePatch(std::vector<PagePatch>& patches,
                    uint64_t byte_offset,
                    const void* data,
                    size_t size) {
    const char* src = static_cast<const char*>(data);
    while (size > 0) {
      const size_t page_off = static_cast<size_t>(byte_offset % PAGE_SIZE);
      const size_t chunk =
          std::min(size, static_cast<size_t>(PAGE_SIZE) - page_off);
      patches.push_back(
          PagePatch{byte_offset / PAGE_SIZE, page_off, src, chunk});
      byte_offset += chunk;
      src += chunk;
      size -= chunk;
    }
  }

  /// Applies a batch of page patches with one read-modify-write per page.
  ///
  /// Slots are packed, so several slots (and therefore several writers) can
  /// share a page. Patching per page instead of per slot keeps same-page
  /// updates from overwriting each other, and the stripe locks keep two
  /// threads from filling the same page image and clobbering one of the
  /// updates (the vanished-update race).
  void WritePages(const std::vector<PagePatch>& patches) {
    if (patches.empty()) {
      return;
    }
    std::vector<uint64_t> pages;
    pages.reserve(patches.size());
    for (const auto& patch : patches) {
      pages.push_back(patch.page);
    }
    SortUnique(pages);

    std::vector<std::unique_lock<std::shared_mutex>> locks;
    LockStripes(pages, locks);

    // Coalesce consecutive pages into a single I/O entry.
    struct PageRun {
      uint64_t first_page = 0;
      uint64_t page_count = 0;
      char* buf           = nullptr;
    };
    std::vector<PageRun> runs;
    std::vector<size_t> page_run(pages.size(), 0);
    for (size_t i = 0; i < pages.size(); ++i) {
      if (runs.empty() ||
          pages[i] != runs.back().first_page + runs.back().page_count) {
        runs.push_back(PageRun{pages[i], 0, nullptr});
      }
      ++runs.back().page_count;
      page_run[i] = runs.size() - 1;
    }
    std::vector<IOBackend::IOEntry> io_entries;
    io_entries.reserve(runs.size());
    // One owned buffer covers every run and is released on failures and
    // worker shutdown, without retaining a backend pointer in thread-local state.
    auto free_buffer = [this](char* ptr) { backend_->FreeBuffer(ptr); };
    std::unique_ptr<char, decltype(free_buffer)> buffer(
        backend_->AllocateBuffer(pages.size()), free_buffer);
    size_t cursor = 0;
    for (auto& run : runs) {
      run.buf = buffer.get() + cursor * PAGE_SIZE;
      cursor += run.page_count;
      io_entries.push_back({run.first_page, run.buf, run.page_count});
    }
    backend_->BatchReadPages(io_entries);
    for (const auto& patch : patches) {
      const size_t index = static_cast<size_t>(
          std::lower_bound(pages.begin(), pages.end(), patch.page) -
          pages.begin());
      const PageRun& run = runs[page_run[index]];
      char* dst =
          run.buf + (patch.page - run.first_page) * PAGE_SIZE + patch.offset;
      std::memcpy(dst, patch.src, patch.size);
    }
    backend_->BatchWritePages(io_entries);
  }

  uint16_t SelectSlab(size_t data_size) const {
    for (uint16_t i = 0; i < size_classes_.size(); ++i) {
      if (data_size <= static_cast<size_t>(size_classes_[i])) {
        return i;
      }
    }
    throw std::invalid_argument("SsdSlabAllocator value too large");
  }

  void WriteBytes(uint64_t byte_offset, const void* data, size_t data_size) {
    const uint32_t stored = static_cast<uint32_t>(data_size);
    std::vector<PagePatch> patches;
    patches.reserve(2);
    AddPagePatch(patches, byte_offset, &stored, sizeof(stored));
    AddPagePatch(patches, byte_offset + kHeaderSize, data, data_size);
    WritePages(patches);
  }

  size_t ReadBytes(uint64_t byte_offset,
                   uint64_t slot_size,
                   void* out_buf,
                   size_t buf_size) {
    const PageID_t start    = byte_offset / PAGE_SIZE;
    const uint64_t page_off = byte_offset % PAGE_SIZE;
    const uint64_t total    = page_off + slot_size;
    const uint64_t pages    = (total + PAGE_SIZE - 1) / PAGE_SIZE;
    auto free_buffer = [this](char* ptr) { backend_->FreeBuffer(ptr); };
    std::unique_ptr<char, decltype(free_buffer)> buffer(
        backend_->AllocateBuffer(pages), free_buffer);
    char* buf = buffer.get();
    std::vector<uint64_t> touched;
    touched.reserve(pages);
    for (uint64_t page = 0; page < pages; ++page) {
      touched.push_back(start + page);
    }
    std::vector<std::shared_lock<std::shared_mutex>> locks;
    LockStripes(touched, locks);
    backend_->BatchReadPages({{start, buf, pages}});
    uint32_t n = 0;
    std::memcpy(&n, buf + page_off, sizeof(n));
    const size_t actual = out_buf == nullptr ? 0 :
        std::min({buf_size, static_cast<size_t>(n),
                  static_cast<size_t>(slot_size - kHeaderSize)});
    if (actual != 0) {
      std::memcpy(out_buf, buf + page_off + kHeaderSize, actual);
    }
    return actual;
  }

  static constexpr uint64_t kHeaderSize      = 4;
  static constexpr uint64_t kPageLockStripes = 2048;
  IOBackend* backend_;
  std::vector<int> size_classes_;
  std::vector<std::unique_ptr<SlabPool>> slabs_;

  std::vector<uint64_t> global_free_blocks_;
  std::mutex global_mu_;
  std::mutex block_claim_mu_;

  std::array<PageStripe, kPageLockStripes> page_stripes_;
};

FACTORY_REGISTER(SsdBlockAllocator,
                 SSD_SLAB,
                 SsdSlabAllocator,
                 IOBackend*,
                 const std::vector<int>&,
                 uint64_t,
                 uint64_t);
