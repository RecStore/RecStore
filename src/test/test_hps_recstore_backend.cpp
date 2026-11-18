#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "storage/external/hps/hps_recstore.h"
#include "test_io_uring_helper.h"

namespace {

constexpr uint64_t kTestSsdCapacityBytes = 64ULL * 1024ULL * 1024ULL;

recstore::storage::HpsRecStoreBackendParams
MakeParams(const std::string& path) {
  recstore::storage::HpsRecStoreBackendParams params;
  params.path                = path;
  params.capacity            = 16;
  params.value_size          = sizeof(float) * 2;
  params.num_threads         = 1;
  params.index_type          = "DRAM_EXTENDIBLE_HASH";
  params.value_store_type    = "DRAM_VALUE_STORE";
  params.dram_allocator      = "PERSIST_LOOP_SLAB";
  params.dram_capacity_bytes = 4096;
  return params;
}

void AssertInsertFetchAndMiss(
    recstore::storage::HpsRecStoreBackend<long long>* backend) {
  const std::string table = "model.table";
  const long long keys[]  = {1, 2};
  const float values[]    = {1.0f, 1.5f, 2.0f, 2.5f};

  EXPECT_EQ(backend->insert(table,
                            2,
                            keys,
                            reinterpret_cast<const char*>(values),
                            sizeof(float) * 2,
                            sizeof(float) * 2),
            2);

  const long long query[] = {2, 3, 1};
  float out[]             = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  std::vector<size_t> misses;
  const size_t hits = backend->fetch(
      table,
      3,
      query,
      reinterpret_cast<char*>(out),
      sizeof(float) * 2,
      [&](size_t index) {
        misses.push_back(index);
        out[index * 2]     = 0.0f;
        out[index * 2 + 1] = 0.0f;
      },
      std::chrono::nanoseconds::zero());

  EXPECT_EQ(hits, 2);
  ASSERT_EQ(misses, std::vector<size_t>({1}));
  EXPECT_FLOAT_EQ(out[0], 2.0f);
  EXPECT_FLOAT_EQ(out[1], 2.5f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
  EXPECT_FLOAT_EQ(out[4], 1.0f);
  EXPECT_FLOAT_EQ(out[5], 1.5f);
}

void AssertIndirectFetchAndMiss(
    recstore::storage::HpsRecStoreBackend<long long>* backend) {
  const std::string table = "model.indirect_table";
  const long long keys[]  = {10, 20, 30};
  const float values[]    = {10.0f, 10.5f, 20.0f, 20.5f, 30.0f, 30.5f};

  EXPECT_EQ(backend->insert(table,
                            3,
                            keys,
                            reinterpret_cast<const char*>(values),
                            sizeof(float) * 2,
                            sizeof(float) * 2),
            3);

  const long long query[] = {10, 40, 20, 50, 30};
  const size_t indices[]  = {4, 1, 2};
  float out[10]           = {};
  std::vector<size_t> miss;
  const size_t hits = backend->fetch(
      table,
      3,
      indices,
      query,
      reinterpret_cast<char*>(out),
      sizeof(float) * 2,
      [&](size_t index) { miss.push_back(index); },
      std::chrono::nanoseconds::zero());

  EXPECT_EQ(hits, 2);
  ASSERT_EQ(miss, std::vector<size_t>({1}));
  EXPECT_FLOAT_EQ(out[8], 30.0f);
  EXPECT_FLOAT_EQ(out[9], 30.5f);
  EXPECT_FLOAT_EQ(out[2], 0.0f);
  EXPECT_FLOAT_EQ(out[3], 0.0f);
  EXPECT_FLOAT_EQ(out[4], 20.0f);
  EXPECT_FLOAT_EQ(out[5], 20.5f);
}

} // namespace

TEST(HpsRecStoreBackendTest, InsertFetchAndMissCallback) {
  const std::string path = "/dev/shm/test_hps_recstore_backend_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);

  recstore::storage::HpsRecStoreBackend<long long> backend(MakeParams(path));
  AssertInsertFetchAndMiss(&backend);
  AssertIndirectFetchAndMiss(&backend);

  std::filesystem::remove_all(path);
}

TEST(HpsRecStoreBackendTest, SsdValueStoreInsertFetchAndMissCallback) {
  std::string reason;
  if (!test_utils::CanUseIoUring(&reason))
    GTEST_SKIP() << reason;

  const std::string path = "/tmp/test_hps_recstore_backend_ssd_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  auto params               = MakeParams(path);
  params.value_store_type   = "SSD_VALUE_STORE";
  params.ssd_capacity_bytes = kTestSsdCapacityBytes;
  params.ssd_value_file     = path + "/value_pages.db";
  params.ssd_io_backend     = "IOURING";
  params.ssd_queue_depth    = 64;
  recstore::storage::HpsRecStoreBackend<long long> backend(params);
  AssertInsertFetchAndMiss(&backend);

  std::filesystem::remove_all(path);
}

TEST(HpsRecStoreBackendTest, TieredValueStoreInsertFetchAndMissCallback) {
  std::string reason;
  if (!test_utils::CanUseIoUring(&reason))
    GTEST_SKIP() << reason;

  const std::string path = "/tmp/test_hps_recstore_backend_tiered_" +
                           std::to_string(static_cast<long long>(getpid()));
  const std::string dram_path =
      "/dev/shm/recstore_hps_tiered_" +
      std::to_string(std::hash<std::string>{}(path)) + "_dram";
  std::filesystem::remove_all(path);
  std::filesystem::remove_all(dram_path);
  std::filesystem::create_directories(path);

  auto params                = MakeParams(path);
  params.value_store_type    = "TIERED_VALUE_STORE";
  params.dram_capacity_bytes = sizeof(float) * 2;
  params.ssd_capacity_bytes  = kTestSsdCapacityBytes;
  params.ssd_value_file      = path + "/tiered_value_pages.db";
  params.ssd_io_backend      = "IOURING";
  params.ssd_queue_depth     = 64;
  recstore::storage::HpsRecStoreBackend<long long> backend(params);
  AssertInsertFetchAndMiss(&backend);

  std::filesystem::remove_all(path);
  std::filesystem::remove_all(dram_path);
}

TEST(HpsRecStoreBackendTest,
     FlatFetchPreservesZeroValuesDuplicatesAndMissCallbacks) {
  auto params =
      MakeParams("/dev/shm/test_hps_flat_" + std::to_string(getpid()));
  params.index_type = "DRAM_PET_HASH";
  recstore::storage::HpsRecStoreBackend<long long> backend(params);
  const long long keys[] = {1, 2};
  const float values[]   = {0, 0, 3, 4};
  ASSERT_EQ(backend.insert(
                "table", 2, keys, reinterpret_cast<const char*>(values), 8, 8),
            2);
  const long long query[] = {2, 99, 1, 2};
  float output[8];
  std::fill_n(output, 8, -1);
  std::vector<size_t> misses;
  EXPECT_EQ(
      backend.fetch(
          "table",
          4,
          query,
          reinterpret_cast<char*>(output),
          8,
          [&](size_t row) {
            misses.push_back(row);
            output[row * 2] = output[row * 2 + 1] = 77;
            const float short_value               = 9;
            EXPECT_EQ(
                backend.insert("table",
                               1,
                               query + row,
                               reinterpret_cast<const char*>(&short_value),
                               4,
                               4),
                1);
          },
          std::chrono::nanoseconds::zero()),
      3);
  EXPECT_EQ(misses, std::vector<size_t>({1}));
  const float expected[] = {3, 4, 77, 77, 0, 0, 3, 4};
  EXPECT_EQ(std::memcmp(output, expected, sizeof(expected)), 0);
  float inserted[2] = {-1, -1};
  EXPECT_EQ(
      backend.fetch(
          "table",
          1,
          query + 1,
          reinterpret_cast<char*>(inserted),
          8,
          [](size_t) { ADD_FAILURE() << "missing callback insert"; },
          std::chrono::nanoseconds::zero()),
      1);
  EXPECT_FLOAT_EQ(inserted[0], 9);
  EXPECT_FLOAT_EQ(inserted[1], 0);
  const long long absent = 1000;
  char padded[14];
  EXPECT_EQ(
      backend.fetch(
          "table",
          1,
          &absent,
          padded + 1,
          13,
          [&](size_t) {
            const float replacement[] = {5, 6};
            backend.insert(
                "table",
                1,
                &absent,
                reinterpret_cast<const char*>(replacement),
                8,
                8);
          },
          std::chrono::nanoseconds::zero()),
      0);
  EXPECT_EQ(
      backend.fetch(
          "table",
          1,
          &absent,
          reinterpret_cast<char*>(inserted),
          8,
          [](size_t) { ADD_FAILURE(); },
          std::chrono::nanoseconds::zero()),
      1);
  EXPECT_FLOAT_EQ(inserted[0], 5);
  EXPECT_FLOAT_EQ(inserted[1], 6);
}

TEST(HpsRecStoreBackendTest, FlatFetchFallsBackForPaddedUnalignedAndShortRows) {
  auto params =
      MakeParams("/dev/shm/test_hps_flat_fallback_" + std::to_string(getpid()));
  recstore::storage::HpsRecStoreBackend<long long> backend(params);
  const long long keys[] = {1, 2};
  const float values[]   = {1, 2, 3, 4};
  ASSERT_EQ(backend.insert(
                "table", 2, keys, reinterpret_cast<const char*>(values), 8, 8),
            2);
  std::vector<char> output(32, static_cast<char>(0x55));
  EXPECT_EQ(
      backend.fetch(
          "table",
          2,
          keys,
          output.data() + 1,
          13,
          [](size_t) { ADD_FAILURE() << "unexpected miss"; },
          std::chrono::nanoseconds::zero()),
      2);
  for (size_t row = 0; row < 2; ++row) {
    EXPECT_EQ(std::memcmp(output.data() + 1 + row * 13, values + row * 2, 8),
              0);
    for (size_t byte = 8; byte < 13; ++byte)
      EXPECT_EQ(output[1 + row * 13 + byte], 0);
  }
  EXPECT_EQ(output[0], 0x55);
  EXPECT_EQ(output[27], 0x55);
  const float short_row = 9;
  ASSERT_EQ(
      backend.insert(
          "table", 1, keys, reinterpret_cast<const char*>(&short_row), 4, 4),
      1);
  float padded[] = {-1, -1};
  EXPECT_EQ(
      backend.fetch(
          "table",
          1,
          keys,
          reinterpret_cast<char*>(padded),
          8,
          [](size_t) { ADD_FAILURE() << "unexpected miss"; },
          std::chrono::nanoseconds::zero()),
      1);
  EXPECT_FLOAT_EQ(padded[0], 9);
  EXPECT_FLOAT_EQ(padded[1], 0);
}

TEST(HpsRecStoreBackendTest, ConcurrentInsertsReuseWorkerSlots) {
  auto params =
      MakeParams("/dev/shm/test_hps_concurrent_" + std::to_string(getpid()));
  constexpr int workers      = 8;
  constexpr int per_worker   = 4096;
  params.capacity            = workers * per_worker;
  params.num_threads         = 4;
  params.dram_capacity_bytes = 4 * 1024 * 1024;
  recstore::storage::HpsRecStoreBackend<long long> backend(params);
  for (int round = 0; round < 2; ++round) {
    std::vector<std::thread> threads;
    for (int worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
        for (int start = 0; start < per_worker; start += 32) {
          long long keys[32];
          float values[64];
          for (int i = 0; i < 32; ++i) {
            keys[i]           = worker * per_worker + start + i;
            values[2 * i]     = static_cast<float>(keys[i]);
            values[2 * i + 1] = static_cast<float>(round);
          }
          EXPECT_EQ(backend.insert("table",
                                   32,
                                   keys,
                                   reinterpret_cast<const char*>(values),
                                   8,
                                   8),
                    32);
        }
      });
    }
    for (auto& thread : threads)
      thread.join();
    const long long keys[] = {0, per_worker, workers * per_worker - 1};
    float output[6]{};
    EXPECT_EQ(
        backend.fetch(
            "table",
            3,
            keys,
            reinterpret_cast<char*>(output),
            8,
            [](size_t) { ADD_FAILURE() << "missing inserted key"; },
            std::chrono::nanoseconds::zero()),
        3);
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(output[2 * i], static_cast<float>(keys[i]));
      EXPECT_FLOAT_EQ(output[2 * i + 1], static_cast<float>(round));
    }
  }
}
