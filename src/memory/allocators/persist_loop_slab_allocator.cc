#include "memory/allocators/persist_loop_slab_allocator.h"

#include <algorithm>
#include <iostream>

DEFINE_double(shm_malloc_healthy_rate, 0.95, "shm malloc healthy rate");

DECLARE_double(shm_malloc_healthy_rate);

namespace base {
namespace {

uint64_t BitmapRangeMask(int bit_offset, int count) {
  return (~uint64{0} >> (64 - count)) << bit_offset;
}

int64 FindUsedBitmapBlock(const uint64* bits, int64 index, int remaining) {
  while (remaining > 0) {
    const int bit_offset = index & 63;
    const int count      = std::min(64 - bit_offset, remaining);
    const uint64 occupied =
        bits[index >> 6] & BitmapRangeMask(bit_offset, count);
    if (occupied != 0) {
      // All candidate starts through this occupied block must be skipped.
      return (index & ~int64{63}) + 63 - __builtin_clzll(occupied);
    }
    index += count;
    remaining -= count;
  }
  return -1;
}

void SetBitmapRange(uint64* bits, int64 index, int remaining) {
  while (remaining > 0) {
    const int bit_offset = index & 63;
    const int count      = std::min(64 - bit_offset, remaining);
    bits[index >> 6] |= BitmapRangeMask(bit_offset, count);
    index += count;
    remaining -= count;
  }
}

} // namespace

PersistLoopShmMalloc::PersistLoopShmMalloc(
    const std::string& filename, int64 memory_size, std::string medium) {
  block_num_ = 0;
  while (((block_num_ + 64) >> 3) + (block_num_ + 64) * sizeof(uint64) <=
         memory_size) {
    block_num_ += 64;
  }
  healthy_used_    = block_num_ * FLAGS_shm_malloc_healthy_rate;
  bool file_exists = base::file_util::PathExists(filename);
  shm_file_ =
      ShmFile::New(ShmFile::ConfigForMedium(medium, filename, memory_size));
  if (!shm_file_) {
    file_exists = false;
    CHECK(base::file_util::Delete(filename, false));
    shm_file_ =
        ShmFile::New(ShmFile::ConfigForMedium(medium, filename, memory_size));
    CHECK(shm_file_) << filename << " " << memory_size;
  }

  used_bits_    = reinterpret_cast<uint64*>(shm_file_->Data());
  data_         = shm_file_->Data() + (block_num_ >> 3);
  load_success_ = file_exists;
  Initialize();
  if (!file_exists) {
    LOG(INFO) << "PersistLoopShmMalloc: first initialization.";
  } else {
    LOG(INFO) << "PersistLoopShmMalloc: Recovery from shutdown.";
  }
}

void PersistLoopShmMalloc::AddMallocs4Recovery(int64_t shm_offset) {
  // NOTE(xieminhui) use atomic instr to reduce conflict between threads.
  // But the code below has redundancy with the function UseBlock
  __sync_fetch_and_add(&total_malloc_, 1); // NOLINT
  int64_t offset = shm_offset;
  CHECK_GE(offset, 0);
  CHECK_LT(offset, block_num_ * 8L);
  int memory_size      = GetMallocSize(offset);
  int memory_block_num = BlockNum(memory_size);

  __sync_fetch_and_add(&total_used_, memory_block_num + 1); // NOLINT
  int64_t sizeBlockIdx = BlockIndex(offset) - 1;

  for (int i = 1; i <= memory_block_num; i++) {
    int64 index = sizeBlockIdx + i;
    // Use Block [index]
    uint64 temp;
    do {
      temp = used_bits_[index >> 6];
    } while (!__sync_bool_compare_and_swap(
        &used_bits_[index >> 6], // NOLINT
        temp,
        temp | (1ul << (index & 63))));
  }
}

bool PersistLoopShmMalloc::UnusedMemoryValid(int64 block_index) const {
  int64 memory_block_num = BlockNum(Block()[block_index]);
  if (memory_block_num < 0) {
    LOG(ERROR) << "memory_block_num < 0";
    return false;
  }
  block_index++;
  if (block_index + memory_block_num > block_num_) {
    LOG(ERROR) << "block_index + memory_block_num > block_num_";
    return false;
  }
  while (memory_block_num > 0) {
    const int bit_offset = block_index & 63;
    const int count      = std::min<int64>(64 - bit_offset, memory_block_num);
    const uint64 mask    = BitmapRangeMask(bit_offset, count);
    if ((used_bits_[block_index >> 6] & mask) != mask) {
      return false;
    }
    block_index += count;
    memory_block_num -= count;
  }
  return true;
}

void PersistLoopShmMalloc::GetMallocsAppend(
    std::vector<char*>* mallocs_data) const {
  for (int64 i = 0; i < block_num_ - 1; ++i) {
    if (!Used(i) && Used(i + 1)) {
      CHECK_GT(Block()[i], 0) << "block id = " << i;
      mallocs_data->push_back(data_ + ((i + 1) << 3));
    }
  }
}

void PersistLoopShmMalloc::GetMallocsAppend(
    std::vector<int64>* mallocs_offset) const {
  for (int64 i = 0; i < block_num_ - 1; ++i) {
    if (!Used(i) && Used(i + 1)) {
      CHECK_GT(Block()[i], 0) << "block id = " << i;
      mallocs_offset->push_back((i + 1) << 3);
    }
  }
}

bool PersistLoopShmMalloc::MemoryValid() const {
  int64 block_index = 0;
  while (block_index < block_num_) {
    if (Used(block_index))
      return false;
    if (!UnusedMemoryValid(block_index))
      return false;
    block_index += 1 + BlockNum(Block()[block_index]);
  }
  return block_index == block_num_;
}

char* PersistLoopShmMalloc::New(int memory_size) {
  CHECK_GE(memory_size, 0);
  if (memory_size == 0) {
    return data_ + sizeof(uint64) * block_num_;
  }
  base::AutoLock lock(lock_);
  int memory_block_num  = BlockNum(memory_size);
  auto fast_malloc_list = fast_malloc_lists_.Get(memory_block_num);
  if (fast_malloc_list && enable_fast_malloc_) {
    // 先尝试从刚回收的内存块中分配
    while (fast_malloc_list->size()) {
      int64 malloc_block = fast_malloc_list->front();
      fast_malloc_list->pop_front();
      // Live headers have clear bits. Inspect the following bit too, unless
      // the payload ends at the pool boundary, to avoid consuming a header.
      const int check_blocks =
          std::min<int64>(memory_block_num + 2, block_num_ - malloc_block);
      if (FindUsedBitmapBlock(used_bits_, malloc_block, check_blocks) < 0) {
        Block()[malloc_block] = memory_size;
        SetBitmapRange(used_bits_, malloc_block + 1, memory_block_num);
        total_used_ += memory_block_num + 1;
        total_malloc_++;
        total_fast_malloc_.Inc(1);
        clflushopt_range(&Block()[malloc_block], sizeof(uint64));
        return data_ + sizeof(uint64) * (malloc_block + 1);
      }
    }
  }
  int64 offset       = 0;
  int64 malloc_block = last_malloc_block_;
  while (offset < block_num_) {
    if (malloc_block + memory_block_num + 1 <= block_num_) {
      const int check_blocks =
          std::min<int64>(memory_block_num + 2, block_num_ - malloc_block);
      const int64 occupied =
          FindUsedBitmapBlock(used_bits_, malloc_block, check_blocks);
      if (occupied < 0) {
        Block()[malloc_block] = memory_size;
        SetBitmapRange(used_bits_, malloc_block + 1, memory_block_num);
        last_malloc_block_ = (malloc_block + memory_block_num + 1) % block_num_;
        total_used_ += memory_block_num + 1;
        total_malloc_++;
        total_loop_malloc_.Inc(1);
        // persist the value first
        // clflushopt_range(sync_data - sizeof(uint64), size + sizeof(uint64));
        // warning(xieminhui) Now we only clflush the !!!!!value_len!!!!!
        clflushopt_range(&Block()[malloc_block], sizeof(uint64));
        return data_ + sizeof(uint64) * (malloc_block + 1);
      }
      const int64 advance = occupied - malloc_block + 1;
      malloc_block += advance;
      offset += advance;
    } else {
      offset += block_num_ - malloc_block;
      malloc_block = 0;
    }
  }
  // LOG(INFO) << "persist memory has no mem!";
  return NULL;
}

bool PersistLoopShmMalloc::Free(void* memory_data) {
  auto data    = reinterpret_cast<const char*>(memory_data);
  int64 offset = GetMallocOffset(data);
  if (offset < 0)
    return false;
  if (offset == sizeof(uint64) * block_num_)
    return true;
  base::AutoLock lock(lock_);
  int64 block_index = BlockIndex(offset) - 1;
  if (!Used(block_index) && UnusedMemoryValid(block_index)) {
    int memory_block_num = BlockNum(Block()[block_index]);
    if (enable_fast_malloc_) {
      auto fast_malloc_list = fast_malloc_lists_.Get(memory_block_num);
      if (!fast_malloc_list && fast_malloc_lists_.size() < max_fast_list_type) {
        fast_malloc_list = new std::deque<int64>();
        fast_malloc_lists_.Insert(memory_block_num, fast_malloc_list);
      }
      if (fast_malloc_list && fast_malloc_list->size() < max_fast_list_num) {
        fast_malloc_list->push_back(block_index);
      }
    }
    int64 index   = block_index;
    int remaining = memory_block_num + 1;
    while (remaining > 0) {
      const int bit_offset = index & 63;
      const int count      = std::min(64 - bit_offset, remaining);
      const uint64 mask    = BitmapRangeMask(bit_offset, count);
      used_bits_[index >> 6] &= ~mask;
      index += count;
      remaining -= count;
    }
    total_used_ -= memory_block_num + 1;
    total_malloc_--;
    return true;
  }
  // NOTE(xieminhui): this block already be freed by other thread

  // LOG(ERROR) << "block_index= " << block_index
  //            << ", Used(block_index)= " << Used(block_index)
  //            << ", UnusedMemoryValid(block_index)= "
  //            << UnusedMemoryValid(block_index);
  return false;
}

} // namespace base
