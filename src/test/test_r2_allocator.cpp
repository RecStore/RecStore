#include <jemalloc/jemalloc.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "base/base.h"
#include "memory/allocators/r2/allocator_master.hh"
#include "memory/allocators/r2_slab_allocator.h"

namespace {

TEST(R2AllocatorTest, AllocationSizeAndFreeUseConfiguredJemallocPrefix) {
  // jemalloc is included first to exercise the public aliases' include order.
  std::vector<char> backing(16ULL << 20);
  r2::AllocatorMaster master;
  master.init(backing.data(), backing.size());
  auto* allocator = master.get_thread_allocator();
  ASSERT_NE(allocator, nullptr);
  void* allocation = allocator->alloc(128);
  ASSERT_NE(allocation, nullptr);
  EXPECT_TRUE(master.within_range(allocation));
  EXPECT_GE(je_malloc_usable_size(allocation), 128U);
  allocator->dealloc(allocation);
}

TEST(R2AllocatorTest, SmallConcurrentAllocationsFitInTightBackingMemory) {
  constexpr int kThreads   = 16;
  constexpr int kIters     = 10000;
  constexpr int kAllocSize = 128;
  constexpr uint64_t kRequestedBytes =
      static_cast<uint64_t>(kThreads) * kIters * kAllocSize;
  constexpr uint64_t kBackingBytes = kRequestedBytes * 3 / 2;

  std::vector<char> backing(kBackingBytes);
  r2::AllocatorMaster master;
  master.init(backing.data(), backing.size());

  std::atomic<uint64_t> failed_allocations{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      r2::Allocator* allocator = master.get_thread_allocator();
      if (!allocator) {
        failed_allocations.fetch_add(kIters, std::memory_order_relaxed);
        return;
      }
      for (int i = 0; i < kIters; ++i) {
        if (!allocator->alloc(kAllocSize)) {
          failed_allocations.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(failed_allocations.load(), 0);
  EXPECT_LE(master.total_managed_mem() - master.memory_usblae(), kBackingBytes);
}

} // namespace
