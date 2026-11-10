#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "storage/allocator/ssd/ssd_block_allocator.h"
#include "storage/allocator/ssd/ssd_buddy_allocator.h"
#include "storage/allocator/ssd/ssd_slab_allocator.h"
#include "storage/io_backend/io_backend.h"

thread_local int pending = 0;
thread_local std::vector<std::unique_ptr<coroutine<void>::pull_type>> coros;

namespace {

class InMemoryIOBackend : public IOBackend {
public:
  InMemoryIOBackend() : IOBackend(MakeConfig()) {}

  void init() override {}

  std::chrono::microseconds read_delay{0};
  bool fail_writes = false;

  void BatchWritePages(const std::vector<IOEntry>& entries) override {
    if (fail_writes) throw std::runtime_error("injected SSD write failure");
    IOBackend::BatchWritePages(entries);
  }

  void*
  GetPage(coroutine<void>::push_type&, uint64_t, PageID_t page_id) override {
    auto* page = new char[PAGE_SIZE];
    ReadPageSync(page_id, page);
    return page;
  }

  void* GetPage(PageID_t page_id) override {
    auto* page = new char[PAGE_SIZE];
    ReadPageSync(page_id, page);
    return page;
  }

  void Unpin(coroutine<void>::push_type&,
             uint64_t,
             PageID_t page_id,
             void* page_data,
             bool is_dirty) override {
    Unpin(page_id, page_data, is_dirty);
  }

  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) override {
    if (is_dirty) {
      WritePageSync(page_id, static_cast<char*>(page_data));
    }
    delete[] static_cast<char*>(page_data);
  }

  char* AllocateBuffer() override { return AllocateBuffer(1); }

  char* AllocateBuffer(uint64_t page_count) override {
    return new char[page_count * PAGE_SIZE]();
  }

  void FreeBuffer(char* buf) override { delete[] buf; }

  void submit() override {}
  void PollCompletion() override {}

protected:
  void ReadPageAsync(coroutine<void>::push_type&,
                     uint64_t,
                     PageID_t page_id,
                     char* buffer) override {
    ReadPageSync(page_id, buffer);
  }

  void ReadPageSync(PageID_t page_id, char* buffer) override {
    if (read_delay.count() > 0) {
      std::this_thread::sleep_for(read_delay);
    }
    std::lock_guard<std::mutex> lock(page_mu_);
    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
      std::memset(buffer, 0, PAGE_SIZE);
      return;
    }
    std::memcpy(buffer, it->second.data(), PAGE_SIZE);
  }

  void WritePageAsync(coroutine<void>::push_type&,
                      uint64_t,
                      PageID_t page_id,
                      char* buffer) override {
    WritePageSync(page_id, buffer);
  }

  void WritePageSync(PageID_t page_id, char* buffer) override {
    std::lock_guard<std::mutex> lock(page_mu_);
    auto& page = pages_[page_id];
    if (page.empty()) {
      page.resize(PAGE_SIZE, 0);
    }
    std::memcpy(page.data(), buffer, PAGE_SIZE);
  }

private:
  static BaseKVConfig MakeConfig() {
    BaseKVConfig cfg;
    cfg.json_config_ = {{"page_id_offset", 1}, {"queue_cnt", 16}};
    return cfg;
  }

  std::unordered_map<PageID_t, std::vector<char>> pages_;
  std::mutex page_mu_;
};

int BuddyLevelForRequest(size_t data_size, int min_block_size, int num_levels) {
  const uint64_t needed =
      static_cast<uint64_t>(data_size) + SsdBuddyAllocator::kHeaderSize;
  for (int level = 0; level < num_levels; ++level) {
    if ((static_cast<uint64_t>(min_block_size) << level) >= needed) {
      return level;
    }
  }
  throw std::invalid_argument("request too large for buddy test parameters");
}

size_t ExpectedBuddyAllocCount(size_t data_size,
                               int min_block_size,
                               int num_levels,
                               uint64_t capacity_bytes) {
  const uint64_t total_min_blocks =
      capacity_bytes / static_cast<uint64_t>(min_block_size);
  const uint64_t top_block_units = uint64_t{1} << (num_levels - 1);
  size_t top_blocks              = 0;
  for (uint64_t block = 1; block + top_block_units <= total_min_blocks;
       block += top_block_units) {
    ++top_blocks;
  }
  const int request_level =
      BuddyLevelForRequest(data_size, min_block_size, num_levels);
  const size_t split_factor = size_t{1} << (num_levels - 1 - request_level);
  return top_blocks * split_factor;
}

size_t ExpectedSlabAllocCountByGlobalPool(
    uint64_t total_capacity_bytes, uint64_t block_size, uint64_t slot_size) {
  return static_cast<size_t>(
      (total_capacity_bytes / block_size) * (block_size / slot_size));
}

class DummyBlockAllocator : public SsdBlockAllocator {
public:
  uint64_t Alloc(size_t data_size) override {
    (void)data_size;
    return kInvalidHandle;
  }
  void Free(uint64_t) override {}
  void Write(uint64_t handle, const void* data, size_t data_size) override {
    storage_[handle] =
        std::vector<char>(static_cast<const char*>(data),
                          static_cast<const char*>(data) + data_size);
  }
  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    const auto it = storage_.find(handle);
    if (it == storage_.end()) {
      return 0;
    }
    const size_t n = std::min(buf_size, it->second.size());
    std::memcpy(out_buf, it->second.data(), n);
    return n;
  }
  uint64_t AllocAndWrite(const void* data, size_t data_size) override {
    const uint64_t handle = next_++;
    Write(handle, data, data_size);
    return handle;
  }
  size_t SlotCapacity(uint64_t handle) const override {
    const auto it = storage_.find(handle);
    return it == storage_.end() ? 0 : it->second.size();
  }

private:
  uint64_t next_ = 1;
  std::unordered_map<uint64_t, std::vector<char>> storage_;
};

TEST(SsdBlockAllocatorTest, BatchHelpersRoundTrip) {
  DummyBlockAllocator alloc;
  const std::array<std::string, 3> payloads = {"aa", "bbbb", "cccccc"};
  std::vector<SsdBlockAllocator::WriteEntry> writes;
  writes.reserve(payloads.size());
  for (const auto& payload : payloads) {
    writes.push_back({payload.data(), payload.size()});
  }

  const auto handles = alloc.BatchAllocAndWrite(writes);
  ASSERT_EQ(handles.size(), payloads.size());

  std::array<std::string, 3> outputs;
  std::vector<SsdBlockAllocator::ReadEntry> reads;
  std::vector<std::vector<char>> bufs(payloads.size());
  for (size_t i = 0; i < handles.size(); ++i) {
    bufs[i].resize(payloads[i].size());
    reads.push_back({handles[i], bufs[i].data()});
  }
  std::vector<size_t> out_sizes;
  alloc.BatchRead(reads, out_sizes);
  ASSERT_EQ(out_sizes.size(), payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    EXPECT_EQ(out_sizes[i], payloads[i].size());
    EXPECT_EQ(std::string(bufs[i].begin(), bufs[i].end()), payloads[i]);
  }
}

TEST(SsdBuddyAllocatorTest, AllocWriteReadAndReuse) {
  InMemoryIOBackend backend;
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/128,
      /*num_levels=*/5,
      /*capacity_bytes=*/128 * 32,
      /*base_byte_offset=*/0);

  const std::string data = "buddy-allocator-payload";
  const uint64_t h1      = alloc.AllocAndWrite(data.data(), data.size());
  ASSERT_NE(h1, SsdBlockAllocator::kInvalidHandle);
  EXPECT_GE(alloc.SlotCapacity(h1), data.size());

  std::vector<char> out(data.size(), 0);
  const size_t n = alloc.Read(h1, out.data(), out.size());
  EXPECT_EQ(n, data.size());
  EXPECT_EQ(std::string(out.begin(), out.end()), data);

  alloc.Free(h1);
  const uint64_t h2 = alloc.Alloc(data.size());
  ASSERT_NE(h2, SsdBlockAllocator::kInvalidHandle);
  EXPECT_EQ(h2, h1);
}

// Buddy blocks are 256 B here, so 16 of them share one 4 KiB page. Unlike the
// slab allocator, buddy serializes every device read and write behind io_mu_,
// so its page read-modify-write cycle is already atomic; this test pins that
// invariant so a future per-page stripe refactor cannot silently drop updates.
TEST(SsdBuddyAllocatorTest, ConcurrentWritesToSlotsSharingPagesKeepUpdates) {
  InMemoryIOBackend backend;
  // Widen the read->write window so unserialized writers would race.
  backend.read_delay = std::chrono::microseconds(200);
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/128,
      /*num_levels=*/6,
      /*capacity_bytes=*/32768,
      /*base_byte_offset=*/0);

  constexpr size_t kThreads   = 8;
  constexpr size_t kPerThread = 8;
  constexpr size_t kValueSize = 128;
  constexpr size_t kRounds    = 3;
  constexpr int kBaseByte     = 1;

  std::vector<uint64_t> handles(kThreads * kPerThread);
  for (auto& handle : handles) {
    handle = alloc.Alloc(kValueSize);
    ASSERT_NE(handle, SsdBlockAllocator::kInvalidHandle);
  }

  for (size_t round = 0; round < kRounds; ++round) {
    std::atomic<size_t> ready{0};
    std::vector<std::thread> writers;
    writers.reserve(kThreads);
    for (size_t t = 0; t < kThreads; ++t) {
      writers.emplace_back([&, t] {
        const std::string payload(
            kValueSize, static_cast<char>(kBaseByte + round * kThreads + t));
        ++ready;
        while (ready.load() < kThreads) {
        }
        for (size_t i = 0; i < kPerThread; ++i) {
          alloc.Write(
              handles[t * kPerThread + i], payload.data(), payload.size());
        }
      });
    }
    for (auto& writer : writers) {
      writer.join();
    }

    backend.read_delay = std::chrono::microseconds(0);
    for (size_t t = 0; t < kThreads; ++t) {
      const std::string expected(
          kValueSize, static_cast<char>(kBaseByte + round * kThreads + t));
      for (size_t i = 0; i < kPerThread; ++i) {
        std::vector<char> out(kValueSize, 0);
        const size_t slot = t * kPerThread + i;
        ASSERT_EQ(alloc.Read(handles[slot], out.data(), out.size()), kValueSize)
            << "round=" << round << " slot=" << slot;
        EXPECT_EQ(std::string(out.begin(), out.end()), expected)
            << "round=" << round << " slot=" << slot;
      }
    }
    backend.read_delay = std::chrono::microseconds(200);
  }
}

TEST(SsdBuddyAllocatorTest, OversizedValueThrows) {
  InMemoryIOBackend backend;
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/64,
      /*num_levels=*/3,
      /*capacity_bytes=*/64 * 16,
      /*base_byte_offset=*/0);
  EXPECT_THROW((void)alloc.Alloc(1024), std::invalid_argument);
}

TEST(SsdBuddyAllocatorTest, CapacityExhaustionAndReuseAfterFree) {
  InMemoryIOBackend backend;
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/64,
      /*num_levels=*/2,
      /*capacity_bytes=*/64 * 4,
      /*base_byte_offset=*/0);
  std::vector<uint64_t> handles;
  while (true) {
    const uint64_t handle = alloc.Alloc(16);
    if (handle == SsdBlockAllocator::kInvalidHandle) {
      break;
    }
    handles.push_back(handle);
  }
  ASSERT_FALSE(handles.empty());

  alloc.Free(handles.back());
  const uint64_t reused = alloc.Alloc(16);
  ASSERT_NE(reused, SsdBlockAllocator::kInvalidHandle);
}

TEST(SsdBuddyAllocatorTest, AllocatableCapacityMatchesExpected) {
  constexpr int kMinBlockSize       = 64;
  constexpr int kNumLevels          = 4;
  constexpr uint64_t kCapacityBytes = kMinBlockSize * 64;
  constexpr size_t kRequestBytes    = 48;

  InMemoryIOBackend backend;
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/kMinBlockSize,
      /*num_levels=*/kNumLevels,
      /*capacity_bytes=*/kCapacityBytes,
      /*base_byte_offset=*/0);

  const size_t expected = ExpectedBuddyAllocCount(
      kRequestBytes, kMinBlockSize, kNumLevels, kCapacityBytes);
  ASSERT_GT(expected, 0U);

  size_t actual = 0;
  while (alloc.Alloc(kRequestBytes) != SsdBlockAllocator::kInvalidHandle) {
    ++actual;
  }
  EXPECT_EQ(actual, expected);
}

TEST(SsdBuddyAllocatorTest, OneGiBWith128BAllocHasExpectedUsableCapacity) {
  constexpr uint64_t kOneGiB  = 1ULL << 30;
  constexpr int kMinBlockSize = 128;
  constexpr int kNumLevels =
      6; // min=128, max=4096 => 128,256,512,1024,2048,4096
  constexpr size_t kRequestBytes = 128;

  InMemoryIOBackend backend;
  SsdBuddyAllocator alloc(
      &backend,
      /*min_block_size=*/kMinBlockSize,
      /*num_levels=*/kNumLevels,
      /*capacity_bytes=*/kOneGiB,
      /*base_byte_offset=*/0);

  const size_t expected_alloc_count = ExpectedBuddyAllocCount(
      kRequestBytes, kMinBlockSize, kNumLevels, kOneGiB);
  ASSERT_EQ(expected_alloc_count, 4194288U);

  size_t actual_alloc_count = 0;
  while (alloc.Alloc(kRequestBytes) != SsdBlockAllocator::kInvalidHandle) {
    ++actual_alloc_count;
  }
  EXPECT_EQ(actual_alloc_count, expected_alloc_count);

  const uint64_t usable_payload_bytes = actual_alloc_count * kRequestBytes;
  EXPECT_EQ(usable_payload_bytes, 536868864ULL);
}

TEST(SsdSlabAllocatorTest, AllocWriteReadAndReuse) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{64, 128},
      /*total_capacity_bytes=*/SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);
  const std::string data = "slab-data";
  const uint64_t h1      = alloc.AllocAndWrite(data.data(), data.size());
  ASSERT_NE(h1, SsdBlockAllocator::kInvalidHandle);
  EXPECT_GE(alloc.SlotCapacity(h1), data.size());

  std::vector<char> out(data.size(), 0);
  const size_t n = alloc.Read(h1, out.data(), out.size());
  EXPECT_EQ(n, data.size());
  EXPECT_EQ(std::string(out.begin(), out.end()), data);

  alloc.Free(h1);
  const uint64_t h2 = alloc.Alloc(data.size());
  ASSERT_NE(h2, SsdBlockAllocator::kInvalidHandle);
  EXPECT_EQ(h2, h1);
}

TEST(SsdSlabAllocatorTest, WriteLargerThanSlotThrows) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{64},
      /*total_capacity_bytes=*/SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);
  const uint64_t handle = alloc.Alloc(32);
  ASSERT_NE(handle, SsdBlockAllocator::kInvalidHandle);
  std::array<char, 80> oversized{};
  EXPECT_THROW(alloc.Write(handle, oversized.data(), oversized.size()),
               std::invalid_argument);
}

TEST(SsdSlabAllocatorTest, CapacityExhaustionAndReuseAfterFree) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{4096},
      /*total_capacity_bytes=*/SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);
  std::vector<uint64_t> handles;
  while (true) {
    const uint64_t handle = alloc.Alloc(2048);
    if (handle == SsdBlockAllocator::kInvalidHandle) {
      break;
    }
    handles.push_back(handle);
  }
  ASSERT_FALSE(handles.empty());

  alloc.Free(handles.back());
  const uint64_t h3 = alloc.Alloc(2048);
  ASSERT_NE(h3, SsdBlockAllocator::kInvalidHandle);
}

TEST(SsdSlabAllocatorTest, OneGiBWith4KiBAllocSingleClassEfficiency) {
  constexpr uint64_t kOneGiB     = 1ULL << 30;
  constexpr size_t kRequestBytes = 4096;
  constexpr int kClass           = 4096;
  constexpr uint64_t kSlotSize   = static_cast<uint64_t>(kClass) + 4; // header

  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{kClass},
      /*total_capacity_bytes=*/kOneGiB,
      /*base_byte_offset=*/0);

  const size_t expected_count = ExpectedSlabAllocCountByGlobalPool(
      kOneGiB, SsdSlabAllocator::kBlockSize, kSlotSize);
  ASSERT_GT(expected_count, 0U);

  size_t actual_count = 0;
  while (alloc.Alloc(kRequestBytes) != SsdBlockAllocator::kInvalidHandle) {
    ++actual_count;
  }
  EXPECT_EQ(actual_count, expected_count);

  const uint64_t payload = actual_count * kRequestBytes;
  EXPECT_EQ(payload, static_cast<uint64_t>(expected_count) * kRequestBytes);
}

// With the global block pool, whichever slab class allocates first can consume
// all currently free global blocks.
TEST(SsdSlabAllocatorTest,
     OneGiBWith4KiBAllocDefaultClassesGlobalPoolEfficiency) {
  constexpr uint64_t kOneGiB             = 1ULL << 30;
  constexpr size_t kRequestBytes         = 4096;
  constexpr int kClass                   = 4096;
  constexpr uint64_t kSlotSize           = static_cast<uint64_t>(kClass) + 4;
  const std::vector<int> kDefaultClasses = {128, 256, 512, 1024, 4096};

  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/kDefaultClasses,
      /*total_capacity_bytes=*/kOneGiB,
      /*base_byte_offset=*/0);

  const size_t expected_count = ExpectedSlabAllocCountByGlobalPool(
      kOneGiB, SsdSlabAllocator::kBlockSize, kSlotSize);
  ASSERT_GT(expected_count, 0U);

  size_t actual_count = 0;
  while (alloc.Alloc(kRequestBytes) != SsdBlockAllocator::kInvalidHandle) {
    ++actual_count;
  }
  EXPECT_EQ(actual_count, expected_count);

  const uint64_t payload = actual_count * kRequestBytes;
  EXPECT_EQ(payload, static_cast<uint64_t>(expected_count) * kRequestBytes);
}

TEST(SsdSlabAllocatorTest, GlobalPoolReleaseAcrossSlabs) {
  InMemoryIOBackend backend;
  constexpr uint64_t cap = 2 * SsdSlabAllocator::kBlockSize;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{4096, 8192},
      /*total_capacity_bytes=*/cap,
      /*base_byte_offset=*/0);

  const uint64_t slot_small = static_cast<uint64_t>(4096) + 4;
  const size_t slots_per_block =
      static_cast<size_t>(SsdSlabAllocator::kBlockSize / slot_small);
  std::vector<uint64_t> small_handles;
  small_handles.reserve(slots_per_block);
  for (size_t i = 0; i < slots_per_block; ++i) {
    const uint64_t h = alloc.Alloc(2048);
    ASSERT_NE(h, SsdBlockAllocator::kInvalidHandle);
    small_handles.push_back(h);
  }
  // Must request >4096 bytes so allocation goes to the 8192 size-class slab.
  const uint64_t large_first = alloc.Alloc(4097);
  ASSERT_NE(large_first, SsdBlockAllocator::kInvalidHandle);

  for (uint64_t h : small_handles) {
    alloc.Free(h);
  }

  const uint64_t large_second = alloc.Alloc(4097);
  ASSERT_NE(large_second, SsdBlockAllocator::kInvalidHandle);
}

TEST(SsdSlabAllocatorTest, TombstoneReuseKeepsHandlesStable) {
  InMemoryIOBackend backend;
  constexpr uint64_t cap = 2 * SsdSlabAllocator::kBlockSize;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{4096},
      /*total_capacity_bytes=*/cap,
      /*base_byte_offset=*/0);

  const uint64_t slot_small = static_cast<uint64_t>(4096) + 4;
  const size_t slots_per_block =
      static_cast<size_t>(SsdSlabAllocator::kBlockSize / slot_small);
  std::vector<uint64_t> block0;
  block0.reserve(slots_per_block);
  for (size_t i = 0; i < slots_per_block; ++i) {
    const uint64_t h = alloc.Alloc(2048);
    ASSERT_NE(h, SsdBlockAllocator::kInvalidHandle);
    block0.push_back(h);
  }
  const uint64_t h_block1 = alloc.Alloc(2048);
  ASSERT_NE(h_block1, SsdBlockAllocator::kInvalidHandle);
  const std::string payload = "stable-handle-check";
  alloc.Write(h_block1, payload.data(), payload.size());

  for (uint64_t h : block0) {
    alloc.Free(h);
  }

  for (size_t i = 0; i < slots_per_block; ++i) {
    const uint64_t h = alloc.Alloc(2048);
    ASSERT_NE(h, SsdBlockAllocator::kInvalidHandle);
  }

  std::vector<char> out(payload.size());
  const size_t n = alloc.Read(h_block1, out.data(), out.size());
  EXPECT_EQ(n, payload.size());
  EXPECT_EQ(std::string(out.begin(), out.end()), payload);
}

TEST(SsdSlabAllocatorTest, AllocSpansMultipleBlocks) {
  InMemoryIOBackend backend;
  constexpr uint64_t cap = 2 * SsdSlabAllocator::kBlockSize;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{4096},
      /*total_capacity_bytes=*/cap,
      /*base_byte_offset=*/0);

  const uint64_t slot_small = static_cast<uint64_t>(4096) + 4;
  const size_t slots_per_block =
      static_cast<size_t>(SsdSlabAllocator::kBlockSize / slot_small);

  std::vector<uint64_t> handles;
  handles.reserve(slots_per_block + 1);
  for (size_t i = 0; i <= slots_per_block; ++i) {
    const uint64_t h = alloc.Alloc(2048);
    ASSERT_NE(h, SsdBlockAllocator::kInvalidHandle);
    handles.push_back(h);
  }

  const std::string a = "block0-mark";
  const std::string b = "block1-mark";
  alloc.Write(handles[0], a.data(), a.size());
  alloc.Write(handles[slots_per_block], b.data(), b.size());

  std::vector<char> buf(std::max(a.size(), b.size()));
  ASSERT_EQ(alloc.Read(handles[0], buf.data(), a.size()), a.size());
  EXPECT_EQ(std::string(buf.data(), buf.data() + a.size()), a);
  ASSERT_EQ(alloc.Read(handles[slots_per_block], buf.data(), b.size()),
            b.size());
  EXPECT_EQ(std::string(buf.data(), buf.data() + b.size()), b);
}

// Regression for #273: slots are packed, so a slot can start near the end of a
// page and continue into the next one. Patching per _slot_ made the following
// slot's page write revert the straddling slot's tail.
TEST(SsdSlabAllocatorTest, BatchAllocAndWriteKeepsSlotsCrossingPageBoundaries) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{128},
      /*total_capacity_bytes=*/SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);

  // 128 B value + 4 B header = 132 B slots, 31 per 4 KiB page.
  constexpr size_t kValueSize = 128;
  constexpr size_t kSlots     = 96;

  std::vector<std::string> values;
  std::vector<SsdBlockAllocator::WriteEntry> entries;
  values.reserve(kSlots);
  entries.reserve(kSlots);
  for (size_t i = 0; i < kSlots; ++i) {
    values.emplace_back(kValueSize, static_cast<char>(i + 1));
    entries.push_back({values.back().data(), kValueSize});
  }

  const std::vector<uint64_t> handles = alloc.BatchAllocAndWrite(entries);
  ASSERT_EQ(handles.size(), kSlots);
  for (size_t i = 0; i < kSlots; ++i) {
    ASSERT_NE(handles[i], SsdBlockAllocator::kInvalidHandle);
    std::vector<char> out(kValueSize, 0);
    ASSERT_EQ(alloc.Read(handles[i], out.data(), out.size()), kValueSize)
        << "slot=" << i;
    EXPECT_EQ(std::string(out.begin(), out.end()), values[i]) << "slot=" << i;
  }

  // A batch shares page images across rows, including duplicate handles and
  // the values that straddle page boundaries.
  std::vector<std::array<char, kValueSize>> buffers(kSlots + 1);
  std::vector<SsdBlockAllocator::ReadEntry> reads;
  for (size_t i = 0; i < kSlots; ++i) {
    reads.push_back({handles[i], buffers[i].data()});
  }
  reads.push_back({handles[30], buffers.back().data()});
  backend.ResetReadStats();
  std::vector<size_t> sizes;
  alloc.BatchRead(reads, sizes);
  EXPECT_EQ(backend.GetReadStats().page_reads, 4U);
  ASSERT_EQ(sizes.size(), kSlots + 1);
  for (size_t i = 0; i < kSlots; ++i) {
    EXPECT_EQ(sizes[i], kValueSize);
    EXPECT_EQ(std::string(buffers[i].data(), kValueSize), values[i]);
  }
  EXPECT_EQ(std::string(buffers.back().data(), kValueSize), values[30]);
  alloc.BatchRead({}, sizes);
  EXPECT_TRUE(sizes.empty());

  std::vector<SsdBlockAllocator::UpdateEntry> updates;
  for (size_t i = 0; i < kSlots; ++i) {
    values[i].assign(kValueSize, static_cast<char>(i + 7));
    updates.push_back({handles[i], values[i].data(), kValueSize});
  }
  backend.ResetReadStats();
  alloc.BatchWrite(updates);
  EXPECT_EQ(backend.GetReadStats().page_reads, 4U);
  alloc.BatchRead(reads, sizes);
  for (size_t i = 0; i < kSlots; ++i) {
    EXPECT_EQ(std::string(buffers[i].data(), sizes[i]), values[i]);
  }
}

// Regression for #273: single-key writes used to be a page-level
// read-modify-write with no serialization, so writers whose slots shared a page
// lost each other's updates.
TEST(SsdSlabAllocatorTest, ConcurrentWritesToSlotsSharingPagesKeepUpdates) {
  InMemoryIOBackend backend;
  // Widen the read->write window so racing writers reliably read the same page
  // image unless the page is serialized.
  backend.read_delay = std::chrono::microseconds(200);
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{128},
      /*total_capacity_bytes=*/SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);

  constexpr size_t kThreads   = 8;
  constexpr size_t kPerThread = 8;
  constexpr size_t kValueSize = 128;
  constexpr size_t kRounds    = 3;
  constexpr int kBaseByte     = 1;

  std::vector<uint64_t> handles(kThreads * kPerThread);
  for (auto& handle : handles) {
    handle = alloc.Alloc(kValueSize);
    ASSERT_NE(handle, SsdBlockAllocator::kInvalidHandle);
  }
  // 64 slots * 132 B span less than three pages, so the threads necessarily
  // share pages.

  for (size_t round = 0; round < kRounds; ++round) {
    std::atomic<size_t> ready{0};
    std::vector<std::thread> writers;
    writers.reserve(kThreads);
    for (size_t t = 0; t < kThreads; ++t) {
      writers.emplace_back([&, t] {
        const std::string payload(
            kValueSize, static_cast<char>(kBaseByte + round * kThreads + t));
        ++ready;
        while (ready.load() < kThreads) {
        }
        for (size_t i = 0; i < kPerThread; ++i) {
          alloc.Write(
              handles[t * kPerThread + i], payload.data(), payload.size());
        }
      });
    }
    for (auto& writer : writers) {
      writer.join();
    }

    backend.read_delay = std::chrono::microseconds(0);
    for (size_t t = 0; t < kThreads; ++t) {
      const std::string expected(
          kValueSize, static_cast<char>(kBaseByte + round * kThreads + t));
      for (size_t i = 0; i < kPerThread; ++i) {
        std::vector<char> out(kValueSize, 0);
        const size_t slot = t * kPerThread + i;
        ASSERT_EQ(alloc.Read(handles[slot], out.data(), out.size()), kValueSize)
            << "round=" << round << " slot=" << slot;
        EXPECT_EQ(std::string(out.begin(), out.end()), expected)
            << "round=" << round << " slot=" << slot;
      }
    }
    backend.read_delay = std::chrono::microseconds(200);
  }
}

// Same race on the bulk insert path: several loaders allocate and write packed
// slots concurrently, so their page read-modify-write cycles overlap.
TEST(SsdSlabAllocatorTest, ConcurrentBatchAllocAndWriteKeepsEverySlot) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(
      &backend,
      /*size_classes=*/{64},
      /*total_capacity_bytes=*/2 * SsdSlabAllocator::kBlockSize,
      /*base_byte_offset=*/0);

  // 64 B value + 4 B header = 68 B slots, 60 per page.
  constexpr size_t kValueSize = 64;
  constexpr size_t kThreads   = 8;
  constexpr size_t kPerThread = 128;

  std::atomic<size_t> ready{0};
  std::vector<std::thread> loaders;
  std::vector<std::vector<uint64_t>> thread_handles(kThreads);
  std::vector<std::string> thread_payloads(kThreads);
  for (size_t t = 0; t < kThreads; ++t) {
    thread_payloads[t] = std::string(kValueSize, static_cast<char>(t + 1));
  }
  backend.read_delay = std::chrono::microseconds(200);
  for (size_t t = 0; t < kThreads; ++t) {
    loaders.emplace_back([&, t] {
      std::vector<SsdBlockAllocator::WriteEntry> entries(
          kPerThread,
          SsdBlockAllocator::WriteEntry{
              thread_payloads[t].data(), thread_payloads[t].size()});
      ++ready;
      while (ready.load() < kThreads) {
      }
      thread_handles[t] = alloc.BatchAllocAndWrite(entries);
    });
  }
  for (auto& loader : loaders) {
    loader.join();
  }
  backend.read_delay = std::chrono::microseconds(0);

  for (size_t t = 0; t < kThreads; ++t) {
    ASSERT_EQ(thread_handles[t].size(), kPerThread) << "thread=" << t;
    for (size_t i = 0; i < kPerThread; ++i) {
      const uint64_t handle = thread_handles[t][i];
      ASSERT_NE(handle, SsdBlockAllocator::kInvalidHandle)
          << "thread=" << t << " slot=" << i;
      std::vector<char> out(kValueSize, 0);
      ASSERT_EQ(alloc.Read(handle, out.data(), out.size()), kValueSize)
          << "thread=" << t << " slot=" << i;
      EXPECT_EQ(std::string(out.begin(), out.end()), thread_payloads[t])
          << "thread=" << t << " slot=" << i;
    }
  }
}

} // namespace

TEST(SsdSlabAllocatorTest, FailedSingleWriteReleasesItsSlot) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(&backend, {128}, 64ULL << 20, 0);
  std::array<char, 128> payload{};
  const uint64_t expected = alloc.Alloc(payload.size());
  ASSERT_NE(expected, SsdBlockAllocator::kInvalidHandle);
  alloc.Free(expected);
  backend.fail_writes = true;
  EXPECT_THROW(alloc.AllocAndWrite(payload.data(), payload.size()), std::runtime_error);
  backend.fail_writes = false;
  const uint64_t reused = alloc.AllocAndWrite(payload.data(), payload.size());
  EXPECT_EQ(reused, expected);
  alloc.Free(reused);
}

TEST(SsdSlabAllocatorTest, CorruptLengthCannotReadTheNextSlot) {
  InMemoryIOBackend backend;
  SsdSlabAllocator alloc(&backend, {128}, 64ULL << 20, 0);
  std::array<char, 128> payload;
  payload.fill('v');
  const uint64_t handle = alloc.AllocAndWrite(payload.data(), payload.size());
  ASSERT_NE(handle, SsdBlockAllocator::kInvalidHandle);
  char* page = static_cast<char*>(backend.GetPage(0));
  const uint32_t corrupt_length = 129;
  std::memcpy(page, &corrupt_length, sizeof(corrupt_length));
  backend.Unpin(0, page, true);
  std::array<char, 256> output;
  output.fill('x');
  EXPECT_EQ(alloc.Read(handle, output.data(), output.size()), payload.size());
  EXPECT_EQ(std::memcmp(output.data(), payload.data(), payload.size()), 0);
  EXPECT_EQ(output[payload.size()], 'x');
  EXPECT_EQ(alloc.Read(handle, nullptr, output.size()), 0U);
  alloc.Free(handle);
}
