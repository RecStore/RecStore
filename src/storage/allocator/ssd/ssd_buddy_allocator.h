#pragma once

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "storage/allocator/ssd/ssd_block_allocator.h"
#include "storage/io_backend/io_backend.h"

class SsdBuddyAllocator : public SsdBlockAllocator {
public:
  SsdBuddyAllocator(IOBackend* backend,
                    int min_block_size,
                    int num_levels,
                    uint64_t capacity_bytes,
                    uint64_t base_byte_offset)
      : backend_(backend),
        min_block_size_(min_block_size),
        num_levels_(num_levels),
        total_min_blocks_(capacity_bytes / min_block_size),
        base_byte_offset_(base_byte_offset) {
    if (backend_ == nullptr || min_block_size_ <= 0 || num_levels_ <= 0) {
      throw std::invalid_argument("invalid SsdBuddyAllocator config");
    }
    free_lists_.resize(num_levels_);
    const uint64_t max_block_min_units = uint64_t{1} << (num_levels_ - 1);
    for (uint64_t block = 1; block + max_block_min_units <= total_min_blocks_;
         block += max_block_min_units) {
      free_lists_.back().insert(block);
    }
  }

  uint64_t Alloc(size_t data_size) override {
    std::lock_guard<std::mutex> lock(mu_);
    const int level      = LevelFor(data_size);
    const uint64_t block = AllocAtLevel(level);
    if (block != kInvalidHandle) {
      allocated_levels_.emplace(block, level);
    }
    return block;
  }

  void Free(uint64_t block_addr) override {
    if (block_addr == kInvalidHandle || block_addr == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = allocated_levels_.find(block_addr);
    if (found == allocated_levels_.end()) {
      return;
    }
    FreeAtLevel(block_addr, found->second);
    allocated_levels_.erase(found);
  }

  void Write(uint64_t block_addr, const void* data, size_t data_size) override {
    const int level = AllocationLevel(block_addr);
    const uint64_t capacity =
        (static_cast<uint64_t>(min_block_size_) << level) - kHeaderSize;
    if (data_size > capacity) {
      throw std::invalid_argument("SsdBuddyAllocator write exceeds block size");
    }
    WriteWithLevel(block_addr, level, data, data_size);
  }

  size_t Read(uint64_t block_addr, void* out_buf, size_t buf_size) override {
    std::lock_guard<std::mutex> lock(io_mu_);
    uint32_t data_size = 0;
    uint32_t level     = 0;
    ReadHeaderUnlocked(block_addr, data_size, level);
    const uint64_t block_size = static_cast<uint64_t>(min_block_size_) << level;
    const PageID_t start      = BlockByteOffset(block_addr) / PAGE_SIZE;
    const uint64_t page_off   = BlockByteOffset(block_addr) % PAGE_SIZE;
    const uint64_t pages = (page_off + block_size + PAGE_SIZE - 1) / PAGE_SIZE;
    char* buf            = backend_->AllocateBuffer(pages);
    backend_->BatchReadPages({{start, buf, pages}});
    const size_t actual = std::min(buf_size, static_cast<size_t>(data_size));
    std::memcpy(out_buf, buf + page_off + kHeaderSize, actual);
    backend_->FreeBuffer(buf);
    return actual;
  }

  uint64_t AllocAndWrite(const void* data, size_t data_size) override {
    const uint64_t handle = Alloc(data_size);
    if (handle != kInvalidHandle) {
      Write(handle, data, data_size);
    }
    return handle;
  }

  size_t SlotCapacity(uint64_t block_addr) const override {
    if (block_addr == 0 || block_addr == kInvalidHandle) {
      return 0;
    }
    return (static_cast<uint64_t>(min_block_size_)
            << AllocationLevel(block_addr)) -
           kHeaderSize;
  }

  static constexpr int kHeaderSize = 8;

private:
  int AllocationLevel(uint64_t block_addr) const {
    std::lock_guard<std::mutex> lock(mu_);
    return allocated_levels_.at(block_addr);
  }

  int LevelFor(size_t data_size) const {
    const uint64_t needed = static_cast<uint64_t>(data_size) + kHeaderSize;
    for (int level = 0; level < num_levels_; ++level) {
      if ((static_cast<uint64_t>(min_block_size_) << level) >= needed) {
        return level;
      }
    }
    throw std::invalid_argument("SsdBuddyAllocator value too large");
  }

  uint64_t AllocAtLevel(int level) {
    for (int cur = level; cur < num_levels_; ++cur) {
      if (free_lists_[cur].empty()) {
        continue;
      }
      uint64_t block = *free_lists_[cur].begin();
      free_lists_[cur].erase(block);
      while (cur > level) {
        --cur;
        const uint64_t buddy = block + (uint64_t{1} << cur);
        free_lists_[cur].insert(buddy);
      }
      return block;
    }
    return kInvalidHandle;
  }

  void FreeAtLevel(uint64_t block, int level) {
    while (level + 1 < num_levels_) {
      const uint64_t buddy = block ^ (uint64_t{1} << level);
      auto it              = free_lists_[level].find(buddy);
      if (it == free_lists_[level].end()) {
        break;
      }
      free_lists_[level].erase(it);
      block = std::min(block, buddy);
      ++level;
    }
    free_lists_[level].insert(block);
  }

  uint64_t BlockByteOffset(uint64_t block_addr) const {
    return base_byte_offset_ + block_addr * min_block_size_;
  }

  void WriteWithLevel(
      uint64_t block_addr, int level, const void* data, size_t data_size) {
    std::lock_guard<std::mutex> lock(io_mu_);
    const uint64_t block_size = static_cast<uint64_t>(min_block_size_) << level;
    const uint64_t byte_offset = BlockByteOffset(block_addr);
    const PageID_t start       = byte_offset / PAGE_SIZE;
    const uint64_t page_off    = byte_offset % PAGE_SIZE;
    const uint64_t pages = (page_off + block_size + PAGE_SIZE - 1) / PAGE_SIZE;
    char* buf            = backend_->AllocateBuffer(pages);
    backend_->BatchReadPages({{start, buf, pages}});
    const uint32_t n = static_cast<uint32_t>(data_size);
    const uint32_t l = static_cast<uint32_t>(level);
    std::memcpy(buf + page_off, &n, sizeof(n));
    std::memcpy(buf + page_off + sizeof(n), &l, sizeof(l));
    std::memcpy(buf + page_off + kHeaderSize, data, data_size);
    backend_->BatchWritePages({{start, buf, pages}});
    backend_->FreeBuffer(buf);
  }

  void ReadHeaderUnlocked(
      uint64_t block_addr, uint32_t& data_size, uint32_t& level) {
    const uint64_t byte_offset = BlockByteOffset(block_addr);
    const PageID_t start       = byte_offset / PAGE_SIZE;
    const uint64_t page_off    = byte_offset % PAGE_SIZE;
    char* buf                  = backend_->AllocateBuffer();
    backend_->ReadPage(start, buf);
    std::memcpy(&data_size, buf + page_off, sizeof(data_size));
    std::memcpy(&level, buf + page_off + sizeof(data_size), sizeof(level));
    backend_->FreeBuffer(buf);
  }

  IOBackend* backend_;
  const int min_block_size_;
  const int num_levels_;
  const uint64_t total_min_blocks_;
  const uint64_t base_byte_offset_;
  mutable std::mutex mu_;
  mutable std::mutex io_mu_;
  std::vector<std::unordered_set<uint64_t>> free_lists_;
  // Allocation sizes exist before the first write and survive shorter rewrites.
  std::unordered_map<uint64_t, int> allocated_levels_;
};

FACTORY_REGISTER(
    SsdBlockAllocator,
    SSD_BUDDY,
    SsdBuddyAllocator,
    IOBackend*,
    int,
    int,
    uint64_t,
    uint64_t);
