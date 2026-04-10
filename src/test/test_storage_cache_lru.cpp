#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "storage/cache/cache_factory.h"
#include "storage/cache/lru/lru_cache.h"

TEST(StorageLRUCacheTest, EvictOrderRespectsRecentAccess) {
  LRUCachePolicy<uint64_t> cache(8);
  cache.OnInsert(1);
  cache.OnInsert(2);
  cache.OnInsert(3);
  cache.OnAccess(1);
  cache.OnAccess(2);

  const std::vector<uint64_t> evicted = cache.Evict(2);
  ASSERT_EQ(evicted.size(), 2u);
  EXPECT_EQ(evicted[0], 3u);
  EXPECT_EQ(evicted[1], 1u);
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_EQ(cache.Size(), 1u);
}

TEST(StorageLRUCacheTest, EvictBoundaryCases) {
  LRUCachePolicy<uint64_t> cache(4);
  cache.OnInsert(10);
  cache.OnInsert(11);
  cache.OnInsert(12);

  EXPECT_TRUE(cache.Evict(0).empty());
  const std::vector<uint64_t> evicted = cache.Evict(100);
  EXPECT_EQ(evicted.size(), 3u);
  EXPECT_EQ(cache.Size(), 0u);
}

TEST(StorageLRUCacheTest, InsertKeepsCapacityBound) {
  constexpr size_t kCapacity = 3;
  LRUCachePolicy<uint64_t> cache(kCapacity);
  cache.OnInsert(1);
  cache.OnInsert(2);
  cache.OnInsert(3);
  cache.OnInsert(4);

  EXPECT_EQ(cache.Size(), kCapacity);
  EXPECT_FALSE(cache.Contains(1));
  EXPECT_TRUE(cache.Contains(4));
}

TEST(StorageLRUCacheTest, ConcurrentInsertAccessEvictIsThreadSafe) {
  constexpr size_t kCapacity          = 1024;
  constexpr int kThreadNum            = 8;
  constexpr int kOpsPerThread         = 4000;
  constexpr int kManualEvictBatchSize = 8;

  LRUCachePolicy<uint64_t> cache(kCapacity);
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreadNum);

  for (int t = 0; t < kThreadNum; ++t) {
    threads.emplace_back([&, t] {
      while (!start.load(std::memory_order_acquire)) {
      }
      const uint64_t base = static_cast<uint64_t>(t) << 32;
      for (int i = 0; i < kOpsPerThread; ++i) {
        const uint64_t key = base + static_cast<uint64_t>(i);
        cache.OnInsert(key);
        cache.OnAccess(key);
        if ((i % 97) == 0) {
          (void)cache.Evict(kManualEvictBatchSize);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_LE(cache.Size(), kCapacity);
}

TEST(StorageLRUCacheTest, FactoryCreatesPolicyFromNameCaseInsensitive) {
  auto policy = storage::cache::CreateCachePolicy<uint64_t>("lru", 16);
  ASSERT_NE(policy, nullptr);
  policy->OnInsert(100);
  policy->OnAccess(100);
  EXPECT_EQ(policy->Size(), 1u);
}

TEST(StorageLRUCacheTest, FactoryRejectsUnsupportedPolicy) {
  EXPECT_THROW(storage::cache::CreateCachePolicy<uint64_t>("fifo", 16),
               std::invalid_argument);
}
