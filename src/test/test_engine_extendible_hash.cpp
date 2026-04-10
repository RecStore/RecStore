#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <random>
#include <string>
#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/engine_extendible_hash.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"

class KVEngineExtendibleHashTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 创建临时测试目录
    test_dir_ =
        "/tmp/test_kv_engine_extendible_hash_" + std::to_string(getpid());
    std::filesystem::create_directories(test_dir_);

    // 配置使用DRAM而不是持久内存
    base::PMMmapRegisterCenter::GetConfig().use_dram = true;
    base::PMMmapRegisterCenter::GetConfig().numa_id  = 0;

    // 创建配置
    config_.num_threads_ = 16;
    config_.json_config_ = {
        {"path", test_dir_},
        {"capacity", 1000000},
        {"value_size", 128},
        {"value_type", "SSD"},
        {"index_type", "DRAM"},
        {"initial_capacity", 16},
        {"value_memory_management", "R2ShmMalloc"} // R2ShmMalloc
    };

    auto r = base::ResolveEngine(config_);
    kv_engine_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        r.engine, r.cfg));
    // KVEngineCCEH
  }

  void TearDown() override {
    // 清理测试
    kv_engine_.reset();

    // 删除临时测试目录
    std::filesystem::remove_all(test_dir_);
  }

  // 辅助函数：创建固定长度的value
  std::string CreateFixedLengthValue(const std::string& base_value) {
    std::string value = base_value;
    value.resize(128); // 确保value长度为128字节
    return value;
  }

  // 简单的barrier实现，用于同步多线程测试
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
  BaseKVConfig config_;
  std::unique_ptr<BaseKV> kv_engine_;
};

// 基本的Put和Get测试
TEST_F(KVEngineExtendibleHashTest, BasicPutAndGet) {
  uint64_t key      = 123;
  std::string value = CreateFixedLengthValue("test_value_123");
  std::string retrieved_value;

  // 测试Put操作
  kv_engine_->Put(key, value, 0);

  // 测试Get操作
  kv_engine_->Get(key, retrieved_value, 0);

  EXPECT_EQ(retrieved_value, value);
}

// 测试不存在的键
TEST_F(KVEngineExtendibleHashTest, GetNonExistentKey) {
  uint64_t key = 999;
  std::string retrieved_value;

  kv_engine_->Get(key, retrieved_value, 0);

  EXPECT_TRUE(retrieved_value.empty());
}

// 测试键值覆盖
TEST_F(KVEngineExtendibleHashTest, KeyOverwrite) {
  uint64_t key       = 100;
  std::string value1 = CreateFixedLengthValue("initial_value");
  std::string value2 = CreateFixedLengthValue("updated_value");
  std::string retrieved_value;

  // 插入初始值
  kv_engine_->Put(key, value1, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value1);

  // 覆盖值
  kv_engine_->Put(key, value2, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value2);
}

// 测试多个键值对
TEST_F(KVEngineExtendibleHashTest, MultiplePutAndGet) {
  const int num_pairs = 50;
  std::vector<std::pair<uint64_t, std::string>> test_data;

  // 准备测试数据
  for (int i = 0; i < num_pairs; i++) {
    test_data.emplace_back(
        i, CreateFixedLengthValue("value_" + std::to_string(i)));
  }

  // 插入数据
  for (const auto& pair : test_data) {
    kv_engine_->Put(pair.first, pair.second, 0);
  }

  // 验证数据
  for (const auto& pair : test_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}

// 测试BatchPut功能
TEST_F(KVEngineExtendibleHashTest, BatchPutThenBatchGet) {
  const int num_keys = 128;
  std::vector<uint64_t> keys;
  keys.reserve(num_keys);

  std::vector<std::vector<float>> source_values(num_keys);
  std::vector<base::ConstArray<float>> batch_put_values;
  batch_put_values.reserve(num_keys);

  for (int i = 0; i < num_keys; ++i) {
    keys.push_back(i + 10000);
    source_values[i].resize(32); // 32 * sizeof(float) == 128 bytes
    for (int j = 0; j < 32; ++j) {
      source_values[i][j] = static_cast<float>(i * 100 + j);
    }
    batch_put_values.emplace_back(
        source_values[i].data(), static_cast<int>(source_values[i].size()));
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
  kv_engine_->BatchPut(keys_array, &batch_put_values, 0);

  std::vector<base::ConstArray<float>> batch_get_values;
  kv_engine_->BatchGet(keys_array, &batch_get_values, 0);

  ASSERT_EQ(batch_get_values.size(), num_keys);
  for (int i = 0; i < num_keys; ++i) {
    ASSERT_EQ(batch_get_values[i].Size(),
              static_cast<int>(source_values[i].size()))
        << "Failed for key " << keys[i];
    for (int j = 0; j < batch_get_values[i].Size(); ++j) {
      EXPECT_FLOAT_EQ(batch_get_values[i][j], source_values[i][j])
          << "Failed for key " << keys[i] << ", index " << j;
    }
  }
}

// 测试BatchGet功能
TEST_F(KVEngineExtendibleHashTest, BatchGet) {
  const int num_keys = 1000;
  std::vector<uint64_t> keys;
  std::vector<std::string> expected_values;

  // 准备测试数据
  for (int i = 0; i < num_keys; i++) {
    keys.push_back(i);
    expected_values.push_back(
        CreateFixedLengthValue("batch_value_" + std::to_string(i)));
    kv_engine_->Put(i, expected_values[i], 0);
  }

  // 创建keys数组
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  // 执行BatchGet
  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  // 验证结果
  EXPECT_EQ(batch_values.size(), num_keys);

  for (int i = 0; i < num_keys; i++) {
    if (batch_values[i].Size() > 0) {
      // 将float数组转换回字符串进行比较
      std::string retrieved_value((char*)batch_values[i].Data(),
                                  batch_values[i].Size() * sizeof(float));
      // 由于存储的是字符串，我们需要截断到实际字符串长度
      size_t null_pos = retrieved_value.find('\0');
      if (null_pos != std::string::npos) {
        retrieved_value = retrieved_value.substr(0, null_pos);
      }

      // 创建期望值的原始字符串（不包含填充）
      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ(retrieved_value, expected_original) << "Failed for key " << i;
    }
  }
}

// 测试BatchGet中不存在的键
TEST_F(KVEngineExtendibleHashTest, BatchGetNonExistentKeys) {
  std::vector<uint64_t> keys = {999, 1000, 1001};
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  // 验证所有不存在的键都返回空数组
  EXPECT_EQ(batch_values.size(), 3);
  for (const auto& value : batch_values) {
    EXPECT_EQ(value.Size(), 0);
  }
}

// 测试混合存在和不存在的键的BatchGet
TEST_F(KVEngineExtendibleHashTest, BatchGetMixedKeys) {
  // 插入一些数据
  kv_engine_->Put(1, CreateFixedLengthValue("value_1"), 0);
  kv_engine_->Put(3, CreateFixedLengthValue("value_3"), 0);
  kv_engine_->Put(5, CreateFixedLengthValue("value_5"), 0);

  // 查询混合键（包含存在和不存在的）
  std::vector<uint64_t> keys = {1, 2, 3, 4, 5, 6};
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  EXPECT_EQ(batch_values.size(), 6);

  // 验证存在的键有值，不存在的键为空
  EXPECT_GT(batch_values[0].Size(), 0); // key 1 exists
  EXPECT_EQ(batch_values[1].Size(), 0); // key 2 doesn't exist
  EXPECT_GT(batch_values[2].Size(), 0); // key 3 exists
  EXPECT_EQ(batch_values[3].Size(), 0); // key 4 doesn't exist
  EXPECT_GT(batch_values[4].Size(), 0); // key 5 exists
  EXPECT_EQ(batch_values[5].Size(), 0); // key 6 doesn't exist
}

// 测试边界值
TEST_F(KVEngineExtendibleHashTest, BoundaryValues) {
  // 测试空字符串
  uint64_t key1           = 1;
  std::string empty_value = CreateFixedLengthValue("");
  std::string retrieved_value;

  kv_engine_->Put(key1, empty_value, 0);
  kv_engine_->Get(key1, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, empty_value);

  // 测试长字符串
  uint64_t key2          = 2;
  std::string long_value = CreateFixedLengthValue(std::string(100, 'x'));
  kv_engine_->Put(key2, long_value, 0);
  kv_engine_->Get(key2, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, long_value);

  // 测试包含特殊字符的字符串
  uint64_t key3             = 3;
  std::string special_value = CreateFixedLengthValue("Hello\nWorld\t\0Test");
  kv_engine_->Put(key3, special_value, 0);
  kv_engine_->Get(key3, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, special_value);
}

// 测试特殊键值
TEST_F(KVEngineExtendibleHashTest, SpecialKeys) {
  std::string test_value = CreateFixedLengthValue("test_value");
  std::string retrieved_value;

  // 测试0键
  kv_engine_->Put(0, test_value, 0);
  kv_engine_->Get(0, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, test_value);

  // 测试大键值
  uint64_t large_key = UINT64_MAX - 1000;
  kv_engine_->Put(large_key, test_value, 0);
  kv_engine_->Get(large_key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, test_value);
}

// 随机数据测试
TEST_F(KVEngineExtendibleHashTest, RandomData) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> key_dist(1, 1000);
  std::uniform_int_distribution<int> value_length_dist(1, 50);

  const int num_operations = 1000;
  std::unordered_map<uint64_t, std::string> expected_data;

  for (int i = 0; i < num_operations; i++) {
    uint64_t key     = key_dist(gen);
    int value_length = value_length_dist(gen);

    // 生成随机值
    std::string base_value;
    for (int j = 0; j < value_length; j++) {
      base_value += static_cast<char>('a' + (gen() % 26));
    }
    std::string value = CreateFixedLengthValue(base_value);

    kv_engine_->Put(key, value, 0);
    expected_data[key] = value; // 记录最后插入的值
  }

  // 验证所有最终的键值对
  for (const auto& pair : expected_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}

// 性能测试
TEST_F(KVEngineExtendibleHashTest, PerformanceTest) {
  const int num_operations = 1000;

  auto start_time = std::chrono::high_resolution_clock::now();

  // 插入操作
  for (int i = 0; i < num_operations; i++) {
    std::string value =
        CreateFixedLengthValue("performance_test_value_" + std::to_string(i));
    kv_engine_->Put(i, value, 0);
  }

  auto insert_end_time = std::chrono::high_resolution_clock::now();

  // 查找操作
  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty())
        << "Failed to get value for key " << i;
    // 验证值是否包含预期的内容
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

// 压力测试
TEST_F(KVEngineExtendibleHashTest, StressTest) {
  const int num_operations = 10000;

  // 大量插入操作
  for (int i = 0; i < num_operations; i++) {
    std::string base_value = "stress_test_value_" + std::to_string(i) + "_" +
                             std::string(20, 'x'); // 较长的值
    std::string value = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  // 验证所有数据
  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty()) << "Failed for key " << i;
    EXPECT_TRUE(
        retrieved_value.find("stress_test_value_" + std::to_string(i)) !=
        std::string::npos);
  }
}

// 多线程并发Put测试
TEST_F(KVEngineExtendibleHashTest, ConcurrentPutTest) {
  const int num_threads           = 16;
  const int operations_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> failed_operations(0);

  SimpleBarrier barrier(num_threads);

  // 创建多个线程同时进行Put操作
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this, t, operations_per_thread, &barrier, &failed_operations]() {
          barrier.wait(); // 等待所有线程就绪

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

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证所有数据都成功插入
  EXPECT_EQ(failed_operations.load(), 0);

  // 验证所有插入的数据都能正确读取
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

// 多线程并发Get测试
TEST_F(KVEngineExtendibleHashTest, ConcurrentGetTest) {
  const int num_data         = 200;
  const int num_threads      = 16;
  const int reads_per_thread = 1000;

  // 预先插入数据
  for (int i = 0; i < num_data; i++) {
    std::string base_value = "concurrent_get_value_" + std::to_string(i);
    std::string value      = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  std::vector<std::thread> threads;
  std::atomic<int> successful_reads(0);
  std::atomic<int> failed_reads(0);

  SimpleBarrier barrier(num_threads);

  // 创建多个线程同时进行Get操作
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         reads_per_thread,
         num_data,
         &barrier,
         &successful_reads,
         &failed_reads]() {
          barrier.wait(); // 等待所有线程就绪

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

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证读取结果
  EXPECT_GT(successful_reads.load(), 0);
  EXPECT_EQ(failed_reads.load(), 0);
  EXPECT_EQ(successful_reads.load(), num_threads * reads_per_thread);
}

// 多线程混合读写测试
TEST_F(KVEngineExtendibleHashTest, ConcurrentReadWriteTest) {
  const int num_threads           = 16;
  const int operations_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> successful_operations(0);
  std::atomic<int> failed_operations(0);

  SimpleBarrier barrier(num_threads);

  // 创建多个线程同时进行读写操作
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         operations_per_thread,
         &barrier,
         &successful_operations,
         &failed_operations]() {
          barrier.wait(); // 等待所有线程就绪

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

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证操作结果
  EXPECT_EQ(failed_operations.load(), 0);
  EXPECT_EQ(successful_operations.load(), num_threads * operations_per_thread);
}

// 多线程BatchGet测试
TEST_F(KVEngineExtendibleHashTest, ConcurrentBatchGetTest) {
  const int num_data    = 100;
  const int num_threads = 16;
  const int batch_size  = 10;

  // 预先插入数据
  for (int i = 0; i < num_data; i++) {
    std::string base_value = "batch_get_value_" + std::to_string(i);
    std::string value      = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i, value, 0);
  }

  std::vector<std::thread> threads;
  std::atomic<int> successful_batches(0);
  std::atomic<int> failed_batches(0);

  SimpleBarrier barrier(num_threads);

  // 创建多个线程同时进行BatchGet操作
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         batch_size,
         num_data,
         &barrier,
         &successful_batches,
         &failed_batches]() {
          barrier.wait(); // 等待所有线程就绪

          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> dist(0, num_data - 1);

          // 执行5次BatchGet操作
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

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证BatchGet结果
  EXPECT_GT(successful_batches.load(), 0);
  EXPECT_EQ(failed_batches.load(), 0);
}

// 数据一致性测试
TEST_F(KVEngineExtendibleHashTest, DataConsistencyTest) {
  const int num_threads     = 16;
  const int num_keys        = 1000;
  const int updates_per_key = 10;

  std::vector<std::thread> threads;
  std::atomic<int> total_updates(0);

  SimpleBarrier barrier(num_threads);

  // 创建多个线程对同一组键进行更新
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this, t, num_keys, updates_per_key, &barrier, &total_updates]() {
          barrier.wait(); // 等待所有线程就绪

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
                // 忽略异常，继续测试
              }
            }
          }
        });
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证所有键都存在且有值
  int valid_keys = 0;
  for (int key = 0; key < num_keys; key++) {
    std::string retrieved_value;
    kv_engine_->Get(key, retrieved_value, 0);
    if (!retrieved_value.empty()) {
      valid_keys++;
      // 验证值包含预期的前缀
      EXPECT_TRUE(
          retrieved_value.find("consistency_thread_") != std::string::npos)
          << "Invalid value for key " << key;
    }
  }

  // 验证大部分键都有值
  EXPECT_GT(valid_keys, num_keys / 2);
  EXPECT_GT(total_updates.load(), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
