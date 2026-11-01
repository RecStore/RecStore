#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/allocators/persist_loop_slab_allocator.h"
#include "memory/shm_file.h"

namespace {

class PersistLoopSlabBitmapTest : public ::testing::Test {
protected:
  void SetUp() override {
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;
  }

  std::unique_ptr<base::PersistLoopShmMalloc>
  MakeAllocator(const std::string& name) {
    const std::string path =
        "/tmp/recstore-persist-bitmap-" + std::to_string(getpid()) + "-" + name;
    return std::make_unique<base::PersistLoopShmMalloc>(
        path, 64 * 1024, "DRAM");
  }
};

TEST_F(PersistLoopSlabBitmapTest, PreservesNeighborsAcrossEveryWordOffset) {
  for (int blocks : {63, 64, 65}) {
    for (int offset = 0; offset < 64; ++offset) {
      SCOPED_TRACE("blocks=" + std::to_string(blocks) +
                   " offset=" + std::to_string(offset));
      auto allocator =
          MakeAllocator(std::to_string(blocks) + "-" + std::to_string(offset));
      // Each allocation has one header block before its payload. Padding
      // places the target payload at every possible bit offset in a word.
      int padding_blocks = (offset + 62) % 64;
      if (padding_blocks == 0)
        padding_blocks = 64;
      const int padding_bytes = padding_blocks * 8;
      const int value_bytes   = blocks * 8;
      char* before            = allocator->New(padding_bytes);
      char* target            = allocator->New(value_bytes);
      char* after             = allocator->New(64);
      ASSERT_NE(before, nullptr);
      ASSERT_NE(target, nullptr);
      ASSERT_NE(after, nullptr);
      ASSERT_EQ((allocator->GetMallocOffset(target) / 8) % 64, offset);
      std::memset(before, 0x11, padding_bytes);
      std::memset(target, 0x22, value_bytes);
      std::memset(after, 0x33, 64);

      ASSERT_TRUE(allocator->Free(target));
      EXPECT_FALSE(allocator->Free(target));
      EXPECT_EQ(allocator->total_malloc(), 2);
      std::vector<char*> live;
      allocator->GetMallocsAppend(&live);
      ASSERT_EQ(live.size(), 2);
      EXPECT_NE(std::find(live.begin(), live.end(), before), live.end());
      EXPECT_NE(std::find(live.begin(), live.end(), after), live.end());

      char* reused = allocator->New(value_bytes);
      ASSERT_EQ(reused, target);
      EXPECT_EQ(allocator->GetMallocSize(reused), value_bytes);
      std::memset(reused, 0x44, value_bytes);
      EXPECT_TRUE(std::all_of(before, before + padding_bytes, [](char byte) {
        return byte == 0x11;
      }));
      EXPECT_TRUE(std::all_of(after, after + 64, [](char byte) {
        return byte == 0x33;
      }));
      EXPECT_EQ(allocator->total_malloc(), 3);
      ASSERT_TRUE(allocator->Free(before));
      ASSERT_TRUE(allocator->Free(reused));
      ASSERT_TRUE(allocator->Free(after));
      EXPECT_EQ(allocator->total_malloc(), 0);
      live.clear();
      allocator->GetMallocsAppend(&live);
      EXPECT_TRUE(live.empty());
    }
  }
}

TEST_F(PersistLoopSlabBitmapTest, ZeroSizeSentinelPreservesLiveAllocation) {
  auto allocator = MakeAllocator("sentinel");
  char* value    = allocator->New(512);
  ASSERT_NE(value, nullptr);
  std::memset(value, 0x55, 512);
  char* sentinel = allocator->New(0);
  EXPECT_EQ(sentinel, allocator->BackingData() + allocator->BackingSize());
  EXPECT_EQ(allocator->GetMallocSize(sentinel), 0);
  EXPECT_TRUE(allocator->Free(sentinel));
  EXPECT_TRUE(allocator->Free(sentinel));
  EXPECT_EQ(allocator->total_malloc(), 1);
  EXPECT_TRUE(
      std::all_of(value, value + 512, [](char byte) { return byte == 0x55; }));
  EXPECT_TRUE(allocator->Free(value));
  EXPECT_FALSE(allocator->Free(value));
  EXPECT_EQ(allocator->total_malloc(), 0);
}

TEST_F(PersistLoopSlabBitmapTest, RejectsStaleFreeListInsideLiveAllocation) {
  auto allocator = MakeAllocator("stale-free-list");
  char* first    = allocator->New(512);
  char* middle   = allocator->New(512);
  char* last     = allocator->New(512);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(middle, nullptr);
  ASSERT_NE(last, nullptr);
  // Force the next larger allocation to wrap into the three freed slots.
  const int tail_bytes = allocator->BackingSize() - (3 * 65 + 2) * 8;
  char* tail           = allocator->New(tail_bytes);
  ASSERT_NE(tail, nullptr);
  ASSERT_TRUE(allocator->Free(first));
  ASSERT_TRUE(allocator->Free(middle));
  ASSERT_TRUE(allocator->Free(last));
  char* larger = allocator->New(1536);
  ASSERT_EQ(larger, first);
  std::memset(larger, 0x5a, 1536);
  // The old middle entry is now inside a live payload, not free space.
  EXPECT_EQ(allocator->New(512), nullptr);
  EXPECT_TRUE(std::all_of(larger, larger + 1536, [](char byte) {
    return byte == 0x5a;
  }));
  EXPECT_EQ(allocator->total_malloc(), 2);
  EXPECT_TRUE(allocator->Free(larger));
  EXPECT_TRUE(allocator->Free(tail));
  EXPECT_EQ(allocator->total_malloc(), 0);
}

TEST_F(PersistLoopSlabBitmapTest, StaleEntryCannotConsumeLiveHeader) {
  auto allocator = MakeAllocator("live-header");
  char* first    = allocator->New(512);
  char* extra    = allocator->New(128);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(extra, nullptr);
  char* tail = allocator->New(allocator->BackingSize() - (65 + 17 + 2) * 8);
  ASSERT_NE(tail, nullptr);
  ASSERT_TRUE(allocator->Free(first));
  ASSERT_TRUE(allocator->Free(extra));
  char* prefix = allocator->New(504);
  ASSERT_EQ(prefix, first);
  char* live = allocator->New(64);
  ASSERT_EQ(live, first + 512);
  std::memset(live, 0x33, 64);
  ASSERT_TRUE(allocator->Free(prefix));
  // All bits in the stale range are clear, but its last bit is a live header.
  EXPECT_EQ(allocator->New(512), nullptr);
  EXPECT_EQ(allocator->GetMallocSize(live), 64);
  EXPECT_TRUE(
      std::all_of(live, live + 64, [](char byte) { return byte == 0x33; }));
  EXPECT_TRUE(allocator->Free(live));
  EXPECT_TRUE(allocator->Free(tail));
  EXPECT_EQ(allocator->total_malloc(), 0);
}

TEST_F(PersistLoopSlabBitmapTest, PoolEndDoesNotReadPayloadAsBitmap) {
  auto allocator = MakeAllocator("pool-end");
  char* first    = allocator->New(8);
  ASSERT_NE(first, nullptr);
  const int tail_bytes = allocator->BackingSize() - 3 * 8;
  char* tail           = allocator->New(tail_bytes);
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(tail + tail_bytes,
            allocator->BackingData() + allocator->BackingSize());
  ASSERT_TRUE(allocator->Free(first));
  first = allocator->New(7);
  ASSERT_NE(first, nullptr);
  // An odd first length sets the bit immediately beyond the bitmap storage.
  ASSERT_TRUE(allocator->Free(tail));
  EXPECT_EQ(allocator->New(tail_bytes), tail);
  EXPECT_TRUE(allocator->Free(tail));
  EXPECT_TRUE(allocator->Free(first));
  EXPECT_EQ(allocator->total_malloc(), 0);
}

TEST_F(PersistLoopSlabBitmapTest, MixedSizesMatchOccupiedRangeModel) {
  auto allocator = MakeAllocator("mixed-sizes");
  struct Live {
    char* data;
    int size;
    char pattern;
  };
  std::vector<Live> live;
  std::vector<bool> occupied(allocator->BackingSize() / 8, false);
  std::mt19937 random(20261029);
  auto release = [&](size_t index) {
    const auto value = live[index];
    EXPECT_EQ(allocator->GetMallocSize(value.data), value.size);
    EXPECT_TRUE(
        std::all_of(value.data, value.data + value.size, [&](char byte) {
          return byte == value.pattern;
        }));
    EXPECT_TRUE(allocator->Free(value.data));
    const auto start = allocator->GetMallocOffset(value.data) / 8 - 1;
    const int blocks = (value.size + 7) / 8 + 1;
    std::fill(
        occupied.begin() + start, occupied.begin() + start + blocks, false);
    live.erase(live.begin() + index);
  };
  for (int iteration = 0; iteration < 5000; ++iteration) {
    SCOPED_TRACE(iteration);
    if (!live.empty() && random() % 100 < 35) {
      release(random() % live.size());
    } else {
      const int size   = 1 + random() % 1024;
      const int blocks = (size + 7) / 8 + 1;
      char* data       = allocator->New(size);
      if (!data) {
        EXPECT_EQ(
            std::search_n(occupied.begin(), occupied.end(), blocks, false),
            occupied.end());
        continue;
      }
      const auto start = allocator->GetMallocOffset(data) / 8 - 1;
      ASSERT_GE(start, 0);
      ASSERT_LE(start + blocks, occupied.size());
      ASSERT_TRUE(std::none_of(occupied.begin() + start,
                               occupied.begin() + start + blocks,
                               [](bool used) { return used; }));
      std::fill(
          occupied.begin() + start, occupied.begin() + start + blocks, true);
      const char pattern = 1 + random() % 127;
      std::memset(data, pattern, size);
      live.push_back({data, size, pattern});
    }
    EXPECT_EQ(allocator->total_malloc(), live.size());
    EXPECT_EQ(allocator->total_used(),
              std::count(occupied.begin(), occupied.end(), true));
  }
  while (!live.empty())
    release(live.size() - 1);
  EXPECT_EQ(allocator->total_malloc(), 0);
  EXPECT_EQ(allocator->total_used(), 0);
}

} // namespace
