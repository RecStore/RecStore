#include <algorithm>

#include "memory/allocators/concurrent_slab_memory_pool.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace {

constexpr int kNumThreads               = 32;
constexpr int kAllocsPerThread          = 200;
constexpr int kMultiSlabAllocsPerThread = 40;
constexpr int64_t kPoolBytes            = 256 * 1024 * 1024;

std::string TempPoolPath(const std::string& name) {
  return "/tmp/recstore_concurrent_slab_pool_" + name;
}

std::vector<int> FactoryPayloadSizes() { return {32, 64, 128, 512, 1024}; }

} // namespace

class ConcurrentSlabMemoryPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    path_ = TempPoolPath("gtest");
    base::file_util::Delete(path_, false);
    pool_ = std::make_unique<base::ConcurrentSlabMemoryPoolMalloc>(
        path_, kPoolBytes, "DRAM");
  }

  void TearDown() override {
    pool_.reset();
    base::file_util::Delete(path_, false);
  }

  std::string path_;
  std::unique_ptr<base::ConcurrentSlabMemoryPoolMalloc> pool_;
};

TEST_F(ConcurrentSlabMemoryPoolTest, FactoryCreatesInstance) {
  auto* from_factory = base::
      Factory<base::MallocApi, const std::string&, int64, const std::string&>::
          NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                      path_ + "_factory",
                      8 * 1024 * 1024,
                      "DRAM");
  ASSERT_NE(from_factory, nullptr);
  char* p = from_factory->New(32);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(from_factory->GetMallocSize(p), 32);
  EXPECT_TRUE(from_factory->Free(p));
  delete from_factory;
  base::file_util::Delete(path_ + "_factory", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SingleSlabMultithreadedNewFree) {
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &failures]() {
      std::vector<char*> ptrs;
      ptrs.reserve(kAllocsPerThread);
      for (int i = 0; i < kAllocsPerThread; ++i) {
        char* p = pool_->New(64);
        if (p == nullptr) {
          failures.fetch_add(1);
          return;
        }
        if (pool_->GetMallocSize(p) != 64) {
          failures.fetch_add(1);
          return;
        }
        ptrs.push_back(p);
      }
      for (char* p : ptrs) {
        if (!pool_->Free(p)) {
          failures.fetch_add(1);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, MultiSlabPerThreadHotChunks) {
  const auto payload_sizes = FactoryPayloadSizes();
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &payload_sizes, &failures]() {
      std::vector<char*> ptrs;
      ptrs.reserve(kMultiSlabAllocsPerThread * payload_sizes.size());
      for (int i = 0; i < kMultiSlabAllocsPerThread; ++i) {
        for (int payload : payload_sizes) {
          char* p = pool_->New(payload);
          if (p == nullptr) {
            failures.fetch_add(1);
            return;
          }
          if (pool_->GetMallocSize(p) != payload) {
            failures.fetch_add(1);
            return;
          }
          ptrs.push_back(p);
        }
      }
      for (char* p : ptrs) {
        if (!pool_->Free(p)) {
          failures.fetch_add(1);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, AllocatesBeyondSingleHotChunk) {
  std::vector<char*> ptrs;
  ptrs.reserve(9000);

  for (int i = 0; i < 9000; ++i) {
    char* p = pool_->New(128);
    ASSERT_NE(p, nullptr) << "allocation " << i;
    ASSERT_EQ(pool_->GetMallocSize(p), 128);
    ptrs.push_back(p);
  }

  for (char* p : ptrs) {
    ASSERT_TRUE(pool_->Free(p));
  }
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, ConcurrentAllocatesBeyondSingleChunk) {
  constexpr int kThreads = 32;
  constexpr int kAllocs  = 400;

  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::vector<char*>> ptrs(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([this, t, &start, &failures, &ptrs]() {
      ptrs[t].reserve(kAllocs);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kAllocs; ++i) {
        char* p = pool_->New(128);
        if (p == nullptr || pool_->GetMallocSize(p) != 128) {
          failures.fetch_add(1);
          return;
        }
        ptrs[t].push_back(p);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(failures.load(), 0);
  for (auto& local : ptrs) {
    for (char* p : local) {
      ASSERT_TRUE(pool_->Free(p));
    }
  }
  EXPECT_EQ(pool_->total_malloc(), 0u);
}

TEST_F(ConcurrentSlabMemoryPoolTest, AllocatedPointersDoNotOverlap) {
  std::set<char*> unique_ptrs;
  std::mutex ptr_mu;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, &unique_ptrs, &ptr_mu]() {
      std::vector<char*> local;
      for (int i = 0; i < 50; ++i) {
        local.push_back(pool_->New(128));
      }
      {
        std::lock_guard<std::mutex> lock(ptr_mu);
        for (char* p : local) {
          EXPECT_TRUE(unique_ptrs.insert(p).second) << "duplicate allocation";
        }
      }
      for (char* p : local) {
        pool_->Free(p);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

TEST_F(ConcurrentSlabMemoryPoolTest, DefaultClassesIncludeEmbeddingWidths) {
  auto* pool = base::
      Factory<base::MallocApi, const std::string&, int64, const std::string&>::
          NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                      path_ + "_widths",
                      8 * 1024 * 1024,
                      "DRAM");
  ASSERT_NE(pool, nullptr);

  // 8 + dim * 4 classes keep dim 20 / dim 68 rows on an exact size class; both
  // would otherwise round up to 136 / 520 and waste most of the slot.
  char* row80 = pool->New(80);
  ASSERT_NE(row80, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row80), 80);

  char* row272 = pool->New(272);
  ASSERT_NE(row272, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row272), 272);

  EXPECT_TRUE(pool->Free(row80));
  EXPECT_TRUE(pool->Free(row272));
  delete pool;
  base::file_util::Delete(path_ + "_widths", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SizedFactoryUsesConfiguredClasses) {
  auto* pool = base::Factory<
      base::MallocApi,
      const std::string&,
      int64,
      const std::string&,
      const std::vector<int>&>::NewInstance("CONCURRENT_SLAB_MEMORY_POOL",
                                            path_ + "_sized",
                                            8 * 1024 * 1024,
                                            "DRAM",
                                            std::vector<int>{8 + 96});
  ASSERT_NE(pool, nullptr);

  char* row96 = pool->New(96);
  ASSERT_NE(row96, nullptr);
  EXPECT_EQ(pool->GetMallocSize(row96), 96);
  EXPECT_TRUE(pool->Free(row96));
  delete pool;
  base::file_util::Delete(path_ + "_sized", false);
}

TEST_F(ConcurrentSlabMemoryPoolTest, SizedFactorySupportsRowsBeyondDefaultCap) {
  // The fixed default table stops at 8+1024, so rows above 1024 bytes have no
  // class and hit a CHECK failure at the first insert. An explicit list is the
  // supported way to serve wider rows.
  const std::vector<int> classes{8 + 32, 8 + 2048};
  auto* pool =
      base::Factory<base::MallocApi,
                    const std::string&,
                    int64,
                    const std::string&,
                    const std::vector<int>&>::NewInstance(
          "CONCURRENT_SLAB_MEMORY_POOL",
          path_ + "_wide",
          64 * 1024 * 1024,
          "DRAM",
          classes);
  ASSERT_NE(pool, nullptr);

  char* wide = pool->New(1032);
  ASSERT_NE(wide, nullptr);
  EXPECT_EQ(pool->GetMallocSize(wide), 1032);
  EXPECT_TRUE(pool->Free(wide));
  delete pool;
  base::file_util::Delete(path_ + "_wide", false);
}
