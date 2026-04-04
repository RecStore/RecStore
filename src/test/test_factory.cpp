// ======== 笛卡尔积参数化测试 ========

#include <tuple>
#include <cmath>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/dram_index_ssd_value.h"

#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"

class KVEngineCartesianTest
    : public ::testing::TestWithParam<
          std::tuple<const char*, const char*, const char*>> {
public:
  static std::string Sanitize(std::string s) {
    for (char& c : s)
      if (!std::isalnum((unsigned char)c))
        c = '_';
    return s;
  }

protected:
  std::string CreateFixedLengthValue(const std::string& base_value) {
    std::string value = base_value;
    value.resize(128);
    return value;
  }

  static const char*
  ExpectedEngine(const std::string& idx, const std::string& val) {
    if (val == "HYBRID")
      return "KVEngineHybrid";
    if (idx == "SSD")
      return "KVEngineCCEH";
    return "KVEngineExtendibleHash";
  }

  static const char* AllocatorTypeFromImpl(const std::string& impl) {
    if (impl == "PersistLoopShmMalloc")
      return "PERSIST_LOOP_SLAB";
    if (impl == "R2ShmMalloc")
      return "R2_SLAB";
    return "PERSIST_LOOP_SLAB";
  }

  void SetUp() override {
    const char *idx_c, *val_c, *mm_c;
    std::tie(idx_c, val_c, mm_c) = GetParam();
    index_type_                  = idx_c;
    value_type_                  = val_c;
    mem_mgr_                     = mm_c;

    test_dir_ = "/tmp/test_kv_engine_cartesian_" + std::to_string(getpid()) +
                "_" + index_type_ + "_" + value_type_ + "_" + mem_mgr_;
    std::filesystem::create_directories(test_dir_);

    base::PMMmapRegisterCenter::GetConfig().use_dram = true;
    base::PMMmapRegisterCenter::GetConfig().numa_id  = 0;

    cfg_.num_threads_     = 8;
    const size_t capacity = 1000000;
    const int value_sz    = 128;

    cfg_.json_config_ = {
        {"path", test_dir_},
        {"index_type", index_type_},
        {"value_type", value_type_},
        {"value_size", value_sz},
        {"capacity", capacity},
        {"value_memory_management", mem_mgr_},
        {"allocator_type", AllocatorTypeFromImpl(mem_mgr_)}};

    // CCEH-backed index paths require explicit IO backend settings.
    if (index_type_ == "SSD") {
      cfg_.json_config_["io_backend_type"] = "IOURING";
      cfg_.json_config_["queue_cnt"]       = 512;
      cfg_.json_config_["page_id_offset"]  = 0;
      // Needed when ValueManager instantiates SSD index directly (HYBRID
      // value).
      cfg_.json_config_["file_path"] = test_dir_ + "/index_cceh.db";
    }

    if (value_type_ == "HYBRID") {
      cfg_.json_config_["DRAM_SIZE"] = capacity * value_sz / 2;
      cfg_.json_config_["SSD_SIZE"]  = capacity * value_sz;
    }

    auto r       = base::ResolveEngine(cfg_);
    engine_name_ = r.engine;
    ASSERT_EQ(engine_name_,
              std::string(ExpectedEngine(index_type_, value_type_)))
        << "selector derived mismatch for (" << index_type_ << ","
        << value_type_ << "," << mem_mgr_ << ")";

    try {
      kv_engine_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          engine_name_, r.cfg));
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Create engine failed for mem_mgr=" << mem_mgr_ << " : "
                   << e.what();
    }
    if (!kv_engine_) {
      GTEST_SKIP() << "Engine '" << engine_name_ << "' or allocator '"
                   << mem_mgr_ << "' not registered/linked.";
    }
  }

  void TearDown() override {
    kv_engine_.reset();
    std::filesystem::remove_all(test_dir_);
  }

  class SimpleBarrier {
  public:
    explicit SimpleBarrier(int count) : count_(count), current_(0) {}

    void wait() {
      std::unique_lock<std::mutex> lock(mutex_);
      ++current_;
      if (current_ == count_) {
        condition_.notify_all();
      } else {
        condition_.wait(lock, [this] { return current_ == count_; });
      }
    }

  private:
    int count_;
    int current_;
    std::mutex mutex_;
    std::condition_variable condition_;
  };

  std::string test_dir_;
  BaseKVConfig cfg_;
  std::unique_ptr<BaseKV> kv_engine_;
  std::string index_type_, value_type_, mem_mgr_, engine_name_;
};

TEST_P(KVEngineCartesianTest, BasicPutAndGet) {
  uint64_t key      = 123;
  std::string value = CreateFixedLengthValue("test_value_123");
  std::string retrieved_value;

  kv_engine_->Put(key, value, 0);

  kv_engine_->Get(key, retrieved_value, 0);

  EXPECT_EQ(retrieved_value, value);
}

TEST_P(KVEngineCartesianTest, GetNonExistentKey) {
  uint64_t key = 999;
  std::string retrieved_value;

  kv_engine_->Get(key, retrieved_value, 0);

  EXPECT_TRUE(retrieved_value.empty());
}

TEST_P(KVEngineCartesianTest, KeyOverwrite) {
  uint64_t key       = 100;
  std::string value1 = CreateFixedLengthValue("initial_value");
  std::string value2 = CreateFixedLengthValue("updated_value");
  std::string retrieved_value;

  kv_engine_->Put(key, value1, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value1);

  kv_engine_->Put(key, value2, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value2);
}

TEST_P(KVEngineCartesianTest, MultiplePutAndGet) {
  const int num_pairs = 50;
  std::vector<std::pair<uint64_t, std::string>> test_data;

  for (int i = 0; i < num_pairs; i++) {
    test_data.emplace_back(
        i, CreateFixedLengthValue("value_" + std::to_string(i)));
  }

  for (const auto& pair : test_data) {
    kv_engine_->Put(pair.first, pair.second, 0);
  }

  for (const auto& pair : test_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}
TEST_P(KVEngineCartesianTest, BatchGet) {
  const int num_keys = 1000;
  std::vector<uint64_t> keys;
  std::vector<std::string> expected_values;

  for (int i = 0; i < num_keys; i++) {
    keys.push_back(i);
    expected_values.push_back(
        CreateFixedLengthValue("batch_value_" + std::to_string(i)));
    kv_engine_->Put(i, expected_values[i], 0);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  EXPECT_EQ(batch_values.size(), num_keys);

  for (int i = 0; i < num_keys; i++) {
    if (batch_values[i].Size() > 0) {
      std::string retrieved_value((char*)batch_values[i].Data(),
                                  batch_values[i].Size() * sizeof(float));
      size_t null_pos = retrieved_value.find('\0');
      if (null_pos != std::string::npos) {
        retrieved_value = retrieved_value.substr(0, null_pos);
      }

      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ(retrieved_value, expected_original) << "Failed for key " << i;
    }
  }
}

TEST_P(KVEngineCartesianTest, BatchGetNonExistentKeys) {
  std::vector<uint64_t> keys = {999, 1000, 1001};
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  EXPECT_EQ(batch_values.size(), 3);
  for (const auto& value : batch_values) {
    EXPECT_EQ(value.Size(), 0);
  }
}

TEST_P(KVEngineCartesianTest, BatchGetMixedKeys) {
  kv_engine_->Put(1, CreateFixedLengthValue("value_1"), 0);
  kv_engine_->Put(3, CreateFixedLengthValue("value_3"), 0);
  kv_engine_->Put(5, CreateFixedLengthValue("value_5"), 0);

  std::vector<uint64_t> keys = {1, 2, 3, 4, 5, 6};
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  EXPECT_EQ(batch_values.size(), 6);

  EXPECT_GT(batch_values[0].Size(), 0); // key 1 exists
  EXPECT_EQ(batch_values[1].Size(), 0); // key 2 doesn't exist
  EXPECT_GT(batch_values[2].Size(), 0); // key 3 exists
  EXPECT_EQ(batch_values[3].Size(), 0); // key 4 doesn't exist
  EXPECT_GT(batch_values[4].Size(), 0); // key 5 exists
  EXPECT_EQ(batch_values[5].Size(), 0); // key 6 doesn't exist
}

TEST_P(KVEngineCartesianTest, BoundaryValues) {
  uint64_t key1           = 1;
  std::string empty_value = CreateFixedLengthValue("");
  std::string retrieved_value;

  kv_engine_->Put(key1, empty_value, 0);
  kv_engine_->Get(key1, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, empty_value);

  uint64_t key2          = 2;
  std::string long_value = CreateFixedLengthValue(std::string(100, 'x'));
  kv_engine_->Put(key2, long_value, 0);
  kv_engine_->Get(key2, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, long_value);

  uint64_t key3             = 3;
  std::string special_value = CreateFixedLengthValue("Hello\nWorld\t\0Test");
  kv_engine_->Put(key3, special_value, 0);
  kv_engine_->Get(key3, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, special_value);
}

TEST_P(KVEngineCartesianTest, SpecialKeys) {
  std::string test_value = CreateFixedLengthValue("test_value");
  std::string retrieved_value;

  kv_engine_->Put(0, test_value, 0);
  kv_engine_->Get(0, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, test_value);

  uint64_t large_key = UINT64_MAX - 1000;
  kv_engine_->Put(large_key, test_value, 0);
  kv_engine_->Get(large_key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, test_value);
}

TEST_P(KVEngineCartesianTest, RandomData) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> key_dist(1, 1000);
  std::uniform_int_distribution<int> value_length_dist(1, 50);

  const int num_operations = 1000;
  std::unordered_map<uint64_t, std::string> expected_data;

  for (int i = 0; i < num_operations; i++) {
    uint64_t key     = key_dist(gen);
    int value_length = value_length_dist(gen);

    std::string base_value;
    for (int j = 0; j < value_length; j++) {
      base_value += static_cast<char>('a' + (gen() % 26));
    }
    std::string value = CreateFixedLengthValue(base_value);

    kv_engine_->Put(key, value, 0);
    expected_data[key] = value;
  }

  for (const auto& pair : expected_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}

TEST_P(KVEngineCartesianTest, PerformanceTest) {
  const int num_operations = 1000;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; i++) {
    std::string value =
        CreateFixedLengthValue("performance_test_value_" + std::to_string(i));
    kv_engine_->Put(i, value, 0);
  }

  auto insert_end_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty())
        << "Failed to get value for key " << i;
    std::string expected_prefix = "performance_test_value_" + std::to_string(i);
    EXPECT_TRUE(retrieved_value.find(expected_prefix) != std::string::npos)
        << "Retrieved value doesn't contain expected prefix for key " << i;
  }

  auto get_end_time = std::chrono::high_resolution_clock::now();

  auto insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      insert_end_time - start_time);
  auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      get_end_time - insert_end_time);

  std::cout << "KVEngineExtendibleHash Performance Results for "
            << num_operations << " operations:\n";
  std::cout << "Insert time: " << insert_duration.count() << " microseconds\n";
  std::cout << "Get time: " << get_duration.count() << " microseconds\n";
  std::cout << "Insert throughput: "
            << (num_operations * 1000000.0 / insert_duration.count())
            << " ops/sec\n";
  std::cout << "Get throughput: "
            << (num_operations * 1000000.0 / get_duration.count())
            << " ops/sec\n";
}

TEST_P(KVEngineCartesianTest, StressTest) {
  const int num_operations = 10000;

  for (int i = 0; i < num_operations; i++) {
    std::string base_value = "stress_test_value_" + std::to_string(i) + "_" +
                             std::string(20, 'x'); // 较长的值
    std::string value = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty()) << "Failed for key " << i;
    EXPECT_TRUE(
        retrieved_value.find("stress_test_value_" + std::to_string(i)) !=
        std::string::npos);
  }
}

TEST_P(KVEngineCartesianTest, ConcurrentPutTest) {
  const int num_threads           = 16;
  const int operations_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> failed_operations(0);

  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this, t, operations_per_thread, &barrier, &failed_operations]() {
          barrier.wait();

          for (int i = 0; i < operations_per_thread; i++) {
            uint64_t key = t * operations_per_thread + i;
            std::string base_value =
                "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
            std::string value = CreateFixedLengthValue(base_value);

            try {
              kv_engine_->Put(key, value, 0);
            } catch (const std::exception& e) {
              failed_operations++;
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failed_operations.load(), 0);

  for (int t = 0; t < num_threads; t++) {
    for (int i = 0; i < operations_per_thread; i++) {
      uint64_t key = t * operations_per_thread + i;
      std::string retrieved_value;
      kv_engine_->Get(key, retrieved_value, 0);
      EXPECT_FALSE(retrieved_value.empty()) << "Failed for key " << key;
      std::string expected_prefix =
          "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
      EXPECT_TRUE(retrieved_value.find(expected_prefix) != std::string::npos)
          << "Value mismatch for key " << key;
    }
  }
}

TEST_P(KVEngineCartesianTest, ConcurrentGetTest) {
  const int num_data         = 200;
  const int num_threads      = 16;
  const int reads_per_thread = 1000;

  for (int i = 0; i < num_data; i++) {
    std::string base_value = "concurrent_get_value_" + std::to_string(i);
    std::string value      = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  std::vector<std::thread> threads;
  std::atomic<int> successful_reads(0);
  std::atomic<int> failed_reads(0);

  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         reads_per_thread,
         num_data,
         &barrier,
         &successful_reads,
         &failed_reads]() {
          barrier.wait();

          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> dist(0, num_data - 1);

          for (int i = 0; i < reads_per_thread; i++) {
            uint64_t key = dist(gen);
            std::string retrieved_value;

            try {
              kv_engine_->Get(key, retrieved_value, 0);
              if (!retrieved_value.empty()) {
                successful_reads++;
              } else {
                failed_reads++;
              }
            } catch (const std::exception& e) {
              failed_reads++;
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_GT(successful_reads.load(), 0);
  EXPECT_EQ(failed_reads.load(), 0);
  EXPECT_EQ(successful_reads.load(), num_threads * reads_per_thread);
}

TEST_P(KVEngineCartesianTest, ConcurrentReadWriteTest) {
  const int num_threads           = 16;
  const int operations_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> successful_operations(0);
  std::atomic<int> failed_operations(0);

  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         operations_per_thread,
         &barrier,
         &successful_operations,
         &failed_operations]() {
          barrier.wait();

          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> op_dist(0, 1); // 0: Put, 1: Get
          std::uniform_int_distribution<uint64_t> key_dist(0, 199);

          for (int i = 0; i < operations_per_thread; i++) {
            uint64_t key = key_dist(gen);
            bool is_put  = op_dist(gen) == 0;

            try {
              if (is_put) {
                std::string base_value = "mixed_thread_" + std::to_string(t) +
                                         "_value_" + std::to_string(i);
                std::string value = CreateFixedLengthValue(base_value);
                kv_engine_->Put(key, value, 0);
                successful_operations++;
              } else {
                std::string retrieved_value;
                kv_engine_->Get(key, retrieved_value, 0);
                successful_operations++;
              }
            } catch (const std::exception& e) {
              failed_operations++;
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failed_operations.load(), 0);
  EXPECT_EQ(successful_operations.load(), num_threads * operations_per_thread);
}

TEST_P(KVEngineCartesianTest, ConcurrentBatchGetTest) {
  const int num_data    = 100;
  const int num_threads = 16;
  const int batch_size  = 10;

  for (int i = 0; i < num_data; i++) {
    std::string base_value = "batch_get_value_" + std::to_string(i);
    std::string value      = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  std::vector<std::thread> threads;
  std::atomic<int> successful_batches(0);
  std::atomic<int> failed_batches(0);

  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         batch_size,
         num_data,
         &barrier,
         &successful_batches,
         &failed_batches]() {
          barrier.wait();

          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> dist(0, num_data - 1);

          for (int batch = 0; batch < 5; batch++) {
            std::vector<uint64_t> keys;
            for (int i = 0; i < batch_size; i++) {
              keys.push_back(dist(gen));
            }

            try {
              base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
              std::vector<base::ConstArray<float>> batch_values;
              kv_engine_->BatchGet(keys_array, &batch_values, 0);

              if (batch_values.size() == keys.size()) {
                successful_batches++;
              } else {
                failed_batches++;
              }
            } catch (const std::exception& e) {
              failed_batches++;
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_GT(successful_batches.load(), 0);
  EXPECT_EQ(failed_batches.load(), 0);
}

TEST_P(KVEngineCartesianTest, DataConsistencyTest) {
  const int num_threads     = 16;
  const int num_keys        = 1000;
  const int updates_per_key = 10;

  std::vector<std::thread> threads;
  std::atomic<int> total_updates(0);

  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this, t, num_keys, updates_per_key, &barrier, &total_updates]() {
          barrier.wait();

          for (int update = 0; update < updates_per_key; update++) {
            for (int key = 0; key < num_keys; key++) {
              std::string base_value =
                  "consistency_thread_" + std::to_string(t) + "_update_" +
                  std::to_string(update) + "_key_" + std::to_string(key);
              std::string value = CreateFixedLengthValue(base_value);

              try {
                kv_engine_->Put(key, value, 0);
                total_updates++;
              } catch (const std::exception& e) {
              }
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int valid_keys = 0;
  for (int key = 0; key < num_keys; key++) {
    std::string retrieved_value;
    kv_engine_->Get(key, retrieved_value, 0);
    if (!retrieved_value.empty()) {
      valid_keys++;
      EXPECT_TRUE(
          retrieved_value.find("consistency_thread_") != std::string::npos)
          << "Invalid value for key " << key;
    }
  }

  EXPECT_GT(valid_keys, num_keys / 2);
  EXPECT_GT(total_updates.load(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombos,
    KVEngineCartesianTest,
    ::testing::Combine(
        ::testing::Values("DRAM", "SSD"),           // index_type
        ::testing::Values("DRAM", "SSD", "HYBRID"), // value_type
        ::testing::Values("R2ShmMalloc", "PersistLoopShmMalloc")),
    [](const testing::TestParamInfo<KVEngineCartesianTest::ParamType>& info) {
      auto idx = std::get<0>(info.param);
      auto val = std::get<1>(info.param);
      auto mm  = std::get<2>(info.param);
      std::string name =
          KVEngineCartesianTest::Sanitize(idx) + "_" +
          KVEngineCartesianTest::Sanitize(val) + "_" +
          KVEngineCartesianTest::Sanitize(mm);
      return name;
    });
