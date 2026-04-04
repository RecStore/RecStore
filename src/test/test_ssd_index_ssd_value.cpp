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

#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/ssd_index_ssd_value.h"

class SSDIndexSSDValueTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = "/tmp/test_ssd_index_ssd_value_" + std::to_string(getpid());
    std::filesystem::create_directories(test_dir_);
    base::PMMmapRegisterCenter::GetConfig().use_dram = true;
    base::PMMmapRegisterCenter::GetConfig().numa_id  = 0;
    config_.num_threads_                             = 16;
    config_.json_config_                             = {
        {"path", test_dir_},
        {"capacity", 100000},
        {"value_size", 128},
        {"io_backend_type", "IOURING"},
        {"queue_cnt", 512},
        {"page_id_offset", 0},
        {"file_path", test_dir_ + "/cceh.db"}};
    kv_engine_ = std::make_unique<KVEngineCCEH>(config_);
  }

  void TearDown() override {
    kv_engine_.reset();
    std::filesystem::remove_all(test_dir_);
  }

  std::string CreateFixedLengthValue(const std::string& base_value) {
    std::string value = base_value;
    value.resize(128);
    return value;
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
  BaseKVConfig config_;
  std::unique_ptr<KVEngineCCEH> kv_engine_;
};

// 基本的Put和Get测试
TEST_F(SSDIndexSSDValueTest, BasicPutAndGet) {
  uint64_t key      = 123;
  std::string value = CreateFixedLengthValue("test_value_123");
  std::string retrieved_value;
  kv_engine_->Put(key, value, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value);
}

// 测试多个键值对
TEST_F(SSDIndexSSDValueTest, MultiplePutAndGet) {
  const int num_pairs = 500;
  std::vector<std::pair<uint64_t, std::string>> test_data;
  for (int i = 1; i <= num_pairs; i++)
    test_data.emplace_back(
        i, CreateFixedLengthValue("value_" + std::to_string(i)));
  for (const auto& pair : test_data)
    kv_engine_->Put(pair.first, pair.second, 0);
  for (const auto& pair : test_data) {
    std::string retrieved_value;
    kv_engine_->Get(pair.first, retrieved_value, 0);
    EXPECT_EQ(retrieved_value, pair.second) << "Failed for key " << pair.first;
  }
}

// // 测试BatchGet功能
TEST_F(SSDIndexSSDValueTest, BatchGet) {
  const int num_keys = 512;
  int cnt            = 0;
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
  EXPECT_EQ(batch_values.size(), num_keys) << "Failed size\n";
  for (int i = 0; i < num_keys; i++) {
    if (batch_values[i].Size() > 0) {
      std::string retrieved_value((char*)batch_values[i].Data(),
                                  batch_values[i].Size() * sizeof(float));
      size_t null_pos = retrieved_value.find('\0');
      if (null_pos != std::string::npos)
        retrieved_value = retrieved_value.substr(0, null_pos);
      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ(retrieved_value, expected_original) << "Failed for key " << i;
    } else {
      std::string expected_original = "batch_value_" + std::to_string(i);
      EXPECT_EQ("", expected_original) << "Failed for key " << i;
    }
  }
}

TEST_F(SSDIndexSSDValueTest, ConcurrentBatchGet) {
  const int num_keys_per_thread = 512;
  const int num_threads         = 16;
  const int total_keys          = num_keys_per_thread * num_threads;
  for (int i = 0; i < total_keys; i++) {
    std::string value =
        CreateFixedLengthValue("concurrent_value_" + std::to_string(i));
    kv_engine_->Put(i, value, 0);
  }
  std::vector<std::vector<std::string>> thread_results(num_threads);
  std::vector<std::string> thread_errors(num_threads);
  std::vector<std::thread> threads;
  SimpleBarrier barrier(num_threads);
  for (int tid = 0; tid < num_threads; tid++) {
    threads.emplace_back([&, tid]() {
      try {
        barrier.wait();
        std::vector<uint64_t> keys;
        for (int i = tid * num_keys_per_thread;
             i < (tid + 1) * num_keys_per_thread;
             i++)
          keys.push_back(i);
        base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
        std::vector<base::ConstArray<float>> batch_values;
        kv_engine_->BatchGet(keys_array, &batch_values, 0);
        for (int i = 0; i < num_keys_per_thread; i++) {
          if (batch_values[i].Size() > 0) {
            std::string retrieved_value((char*)batch_values[i].Data(),
                                        batch_values[i].Size() * sizeof(float));
            size_t null_pos = retrieved_value.find('\0');
            if (null_pos != std::string::npos)
              retrieved_value = retrieved_value.substr(0, null_pos);
            thread_results[tid].push_back(retrieved_value);
          } else {
            thread_results[tid].push_back("");
          }
        }
      } catch (const std::exception& e) {
        thread_errors[tid] = e.what();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (int tid = 0; tid < num_threads; tid++) {
    EXPECT_TRUE(thread_errors[tid].empty())
        << "Thread " << tid << " error: " << thread_errors[tid];
    EXPECT_EQ(thread_results[tid].size(), num_keys_per_thread)
        << "Thread " << tid << " result count mismatch";

    for (int i = 0; i < num_keys_per_thread; i++) {
      int global_key       = tid * num_keys_per_thread + i;
      std::string expected = "concurrent_value_" + std::to_string(global_key);
      EXPECT_EQ(thread_results[tid][i], expected)
          << "Thread " << tid << " key " << global_key << " value mismatch";
    }
  }
}

// 多线程并发BatchPut测试
TEST_F(SSDIndexSSDValueTest, ConcurrentBatchPut) {
  const int num_threads     = 16;
  const int keys_per_thread = 512;
  const int floats_per_key  = 128 / sizeof(float);
  std::vector<std::thread> threads;
  std::atomic<int> failed_batches(0);
  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         t,
         keys_per_thread,
         floats_per_key,
         &barrier,
         &failed_batches]() {
          barrier.wait();
          std::vector<uint64_t> keys(keys_per_thread);
          std::vector<std::vector<float>> write_data(keys_per_thread);
          std::vector<base::ConstArray<float>> values_in(keys_per_thread);

          for (int i = 0; i < keys_per_thread; i++) {
            keys[i] = 1000000 + t * keys_per_thread + i;
            write_data[i].resize(floats_per_key);
            for (int j = 0; j < floats_per_key; j++)
              write_data[i][j] = t * 1000.0f + i * 10.0f + j;
            values_in[i] =
                base::ConstArray<float>(write_data[i].data(), floats_per_key);
          }

          try {
            base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
            kv_engine_->BatchPut(keys_array, &values_in, 0);
          } catch (const std::exception&) {
            failed_batches++;
          }
        });
  }

  for (auto& t : threads)
    t.join();

  EXPECT_EQ(failed_batches.load(), 0);

  // 验证所有线程写入的数据都正确
  for (int t = 0; t < num_threads; t++) {
    std::vector<uint64_t> keys(keys_per_thread);
    std::vector<std::vector<float>> expected(keys_per_thread);
    for (int i = 0; i < keys_per_thread; i++) {
      keys[i] = 1000000 + t * keys_per_thread + i;
      expected[i].resize(floats_per_key);
      for (int j = 0; j < floats_per_key; j++)
        expected[i][j] = t * 1000.0f + i * 10.0f + j;
    }
    base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
    std::vector<base::ConstArray<float>> batch_values;
    kv_engine_->BatchGet(keys_array, &batch_values, 0);

    ASSERT_EQ((int)batch_values.size(), keys_per_thread);
    for (int i = 0; i < keys_per_thread; i++) {
      ASSERT_EQ(batch_values[i].Size(), floats_per_key)
          << "Thread " << t << " key " << keys[i] << " dim mismatch";
      for (int j = 0; j < floats_per_key; j++) {
        EXPECT_FLOAT_EQ(batch_values[i].Data()[j], expected[i][j])
            << "Thread " << t << " key " << keys[i] << " float[" << j
            << "] mismatch";
      }
    }
  }
}

// BatchPut写入 + BatchGet读回，用float数据做roundtrip验证
TEST_F(SSDIndexSSDValueTest, BatchPutAndBatchGet) {
  const int num_keys = 256;
  const int floats_per_key = 128 / sizeof(float); // value_size=128 → 32 floats

  // 构造每个key对应的float数据，key i 的第 j 个float = i * 100.0f + j
  std::vector<std::vector<float>> write_data(num_keys);
  for (int i = 0; i < num_keys; i++) {
    write_data[i].resize(floats_per_key);
    for (int j = 0; j < floats_per_key; j++) {
      write_data[i][j] = i * 100.0f + j;
    }
  }

  // 准备 keys 数组
  std::vector<uint64_t> keys(num_keys);
  for (int i = 0; i < num_keys; i++) {
    keys[i] = i + 10000; // 用 10000 开头的key，避免和其他测试冲突
  }

  // 构造 BatchPut 需要的 vector<ConstArray<float>>
  std::vector<base::ConstArray<float>> values_in(num_keys);
  for (int i = 0; i < num_keys; i++) {
    values_in[i] =
        base::ConstArray<float>(write_data[i].data(), write_data[i].size());
  }

  // 调用 BatchPut 写入
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
  kv_engine_->BatchPut(keys_array, &values_in, 0);

  // 调用 BatchGet 读回
  std::vector<base::ConstArray<float>> values_out;
  kv_engine_->BatchGet(keys_array, &values_out, 0);

  // 验证返回的key数量
  ASSERT_EQ(values_out.size(), num_keys);

  // 逐key逐float比对
  for (int i = 0; i < num_keys; i++) {
    ASSERT_GT(values_out[i].Size(), 0)
        << "Key " << keys[i] << " returned empty";
    ASSERT_EQ(values_out[i].Size(), floats_per_key)
        << "Key " << keys[i] << " dim mismatch";
    for (int j = 0; j < floats_per_key; j++) {
      EXPECT_FLOAT_EQ(values_out[i].Data()[j], write_data[i][j])
          << "Key " << keys[i] << " float[" << j << "] mismatch";
    }
  }
}

// 测试BatchPut覆盖写
TEST_F(SSDIndexSSDValueTest, BatchPutOverwrite) {
  const int num_keys       = 100;
  const int floats_per_key = 128 / sizeof(float);

  std::vector<uint64_t> keys(num_keys);
  for (int i = 0; i < num_keys; i++)
    keys[i] = i + 1100000;
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());

  // 第一次写入
  std::vector<std::vector<float>> write_data1(num_keys);
  std::vector<base::ConstArray<float>> values_in1(num_keys);
  for (int i = 0; i < num_keys; i++) {
    write_data1[i].resize(floats_per_key);
    for (int j = 0; j < floats_per_key; j++)
      write_data1[i][j] = i * 1.0f + j;
    values_in1[i] =
        base::ConstArray<float>(write_data1[i].data(), floats_per_key);
  }
  kv_engine_->BatchPut(keys_array, &values_in1, 0);

  // 第二次覆盖写入
  std::vector<std::vector<float>> write_data2(num_keys);
  std::vector<base::ConstArray<float>> values_in2(num_keys);
  for (int i = 0; i < num_keys; i++) {
    write_data2[i].resize(floats_per_key);
    for (int j = 0; j < floats_per_key; j++)
      write_data2[i][j] = i * 200.0f + j;
    values_in2[i] =
        base::ConstArray<float>(write_data2[i].data(), floats_per_key);
  }
  kv_engine_->BatchPut(keys_array, &values_in2, 0);

  // 验证读回的是第二次写入的值
  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);

  ASSERT_EQ((int)batch_values.size(), num_keys);
  for (int i = 0; i < num_keys; i++) {
    ASSERT_EQ(batch_values[i].Size(), floats_per_key);
    for (int j = 0; j < floats_per_key; j++) {
      EXPECT_FLOAT_EQ(batch_values[i].Data()[j], write_data2[i][j])
          << "Key " << keys[i] << " float[" << j
          << "] mismatch after overwrite";
    }
  }
}

// 测试不同 value size 下 Put/Get 的正确性
// 覆盖场景：小于一页、恰好一页、跨两页、跨多页
TEST_F(SSDIndexSSDValueTest, VariableValueSize_PutGet) {
  // 每组：(floats数量, 说明)
  // PAGE_SIZE=4096，头4字节存长度，所以：
  //   <= 1023 floats (4092B) → 1页
  //   == 1024 floats (4096B) → 需要2页（4092+4）
  //   == 2048 floats (8192B) → 2页
  //   == 12800 floats (51200B) → 13页，对应 ml20m 场景
  const std::vector<std::pair<int, std::string>> cases = {
      {16, "small 64B"},
      {32, "default 128B"},
      {1023, "exactly fills one page"},
      {1024, "just spills to two pages"},
      {2048, "two full pages"},
      {12800, "ml20m scenario 51200B"},
  };

  for (const auto& [num_floats, desc] : cases) {
    uint64_t key = 90000 + num_floats;

    // 构造数据：第 j 个 float = num_floats * 1.0f + j
    std::vector<float> write_data(num_floats);
    for (int j = 0; j < num_floats; j++)
      write_data[j] = num_floats * 1.0f + j;

    std::string value_in(reinterpret_cast<const char*>(write_data.data()),
                         num_floats * sizeof(float));
    kv_engine_->Put(key, value_in, 0);

    std::string value_out;
    kv_engine_->Get(key, value_out, 0);

    ASSERT_EQ(value_out.size(), (size_t)(num_floats * sizeof(float)))
        << "[" << desc << "] size mismatch";

    const float* out_ptr = reinterpret_cast<const float*>(value_out.data());
    for (int j = 0; j < num_floats; j++) {
      EXPECT_FLOAT_EQ(out_ptr[j], write_data[j])
          << "[" << desc << "] float[" << j << "] mismatch";
    }
  }
}

// 测试 BatchPut/BatchGet 在混合 value size 下的正确性
// 同一个 batch 里每个 key 的 value 大小不同
TEST_F(SSDIndexSSDValueTest, VariableValueSize_BatchPutBatchGet) {
  // 每个 key 用不同的 floats 数量，覆盖单页、跨页等场景
  const std::vector<int> sizes_per_key = {
      16, 32, 512, 1023, 1024, 1025, 2048, 4096, 12800};
  const int num_keys = sizes_per_key.size();

  std::vector<uint64_t> keys(num_keys);
  std::vector<std::vector<float>> write_data(num_keys);
  std::vector<base::ConstArray<float>> values_in(num_keys);

  for (int i = 0; i < num_keys; i++) {
    keys[i] = 80000 + i;
    int nf  = sizes_per_key[i];
    write_data[i].resize(nf);
    for (int j = 0; j < nf; j++)
      write_data[i][j] = i * 1000.0f + j;
    values_in[i] = base::ConstArray<float>(write_data[i].data(), nf);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_keys);
  kv_engine_->BatchPut(keys_array, &values_in, 0);

  std::vector<base::ConstArray<float>> values_out;
  kv_engine_->BatchGet(keys_array, &values_out, 0);

  ASSERT_EQ((int)values_out.size(), num_keys);
  for (int i = 0; i < num_keys; i++) {
    int nf = sizes_per_key[i];
    ASSERT_EQ(values_out[i].Size(), nf)
        << "Key " << keys[i] << " (size=" << nf << ") dim mismatch";
    for (int j = 0; j < nf; j++) {
      EXPECT_FLOAT_EQ(values_out[i].Data()[j], write_data[i][j])
          << "Key " << keys[i] << " float[" << j << "] mismatch";
    }
  }
}

TEST_F(SSDIndexSSDValueTest, UpdateOverwritesExistingValue) {
  uint64_t key         = 424242;
  std::string original = CreateFixedLengthValue("original_value");
  std::string updated  = CreateFixedLengthValue("updated_value");
  std::string retrieved;

  kv_engine_->Put(key, original, 0);
  kv_engine_->Get(key, retrieved, 0);
  EXPECT_EQ(retrieved, original);

  kv_engine_->Put(key, updated, 0);
  kv_engine_->Get(key, retrieved, 0);
  EXPECT_EQ(retrieved, updated);
}

// 测试不存在的键
TEST_F(SSDIndexSSDValueTest, GetNonExistentKey) {
  uint64_t key = 999;
  std::string retrieved_value;
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_TRUE(retrieved_value.empty());
}

// 测试BatchGet中不存在的键
TEST_F(SSDIndexSSDValueTest, BatchGetNonExistentKeys) {
  std::vector<uint64_t> keys = {999999, 1000000, 1000001};
  base::ConstArray<uint64_t> keys_array(keys.data(), keys.size());
  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(keys_array, &batch_values, 0);
  EXPECT_EQ(batch_values.size(), 3);
  for (const auto& value : batch_values) {
    EXPECT_EQ(value.Size(), 0);
  }
}

// 测试混合存在和不存在的键的BatchGet
TEST_F(SSDIndexSSDValueTest, BatchGetMixedKeys) {
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

// 测试边界值
TEST_F(SSDIndexSSDValueTest, BoundaryValues) {
  std::string retrieved_value;

  uint64_t key1           = 700001;
  std::string empty_value = CreateFixedLengthValue("");
  kv_engine_->Put(key1, empty_value, 0);
  kv_engine_->Get(key1, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, empty_value);

  uint64_t key2          = 700002;
  std::string long_value = CreateFixedLengthValue(std::string(100, 'x'));
  kv_engine_->Put(key2, long_value, 0);
  kv_engine_->Get(key2, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, long_value);

  uint64_t key3             = 700003;
  std::string special_value = CreateFixedLengthValue("Hello\nWorld\t\0Test");
  kv_engine_->Put(key3, special_value, 0);
  kv_engine_->Get(key3, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, special_value);
}

// 测试特殊键值
TEST_F(SSDIndexSSDValueTest, SpecialKeys) {
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

// 随机数据测试
TEST_F(SSDIndexSSDValueTest, RandomData) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> key_dist(1, 1000);
  std::uniform_int_distribution<int> value_length_dist(1, 50);

  const int num_operations = 1000;
  std::unordered_map<uint64_t, std::string> expected_data;

  for (int i = 0; i < num_operations; i++) {
    uint64_t key     = key_dist(gen) + 500000;
    int value_length = value_length_dist(gen);
    std::string base_value;
    for (int j = 0; j < value_length; j++)
      base_value += static_cast<char>('a' + (gen() % 26));
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

// 性能测试
TEST_F(SSDIndexSSDValueTest, PerformanceTest) {
  const int num_operations = 1000;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; i++) {
    std::string value =
        CreateFixedLengthValue("performance_test_value_" + std::to_string(i));
    kv_engine_->Put(i + 200000, value, 0);
  }

  auto insert_end_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i + 200000, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty()) << "Failed for key " << i;
    std::string expected_prefix = "performance_test_value_" + std::to_string(i);
    EXPECT_TRUE(retrieved_value.find(expected_prefix) != std::string::npos)
        << "Retrieved value doesn't contain expected prefix for key " << i;
  }

  auto get_end_time = std::chrono::high_resolution_clock::now();

  auto insert_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      insert_end_time - start_time);
  auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      get_end_time - insert_end_time);

  std::cout << "SSDIndexSSDValue Performance Results for " << num_operations
            << " operations:\n";
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
TEST_F(SSDIndexSSDValueTest, StressTest) {
  const int num_operations = 10000;

  for (int i = 0; i < num_operations; i++) {
    std::string base_value =
        "stress_test_value_" + std::to_string(i) + "_" + std::string(20, 'x');
    std::string value = CreateFixedLengthValue(base_value);
    kv_engine_->Put(i + 300000, value, 0);
  }

  for (int i = 0; i < num_operations; i++) {
    std::string retrieved_value;
    kv_engine_->Get(i + 300000, retrieved_value, 0);
    EXPECT_FALSE(retrieved_value.empty()) << "Failed for key " << i;
    EXPECT_TRUE(
        retrieved_value.find("stress_test_value_" + std::to_string(i)) !=
        std::string::npos);
  }
}

// 多线程并发Put测试
TEST_F(SSDIndexSSDValueTest, ConcurrentPutTest) {
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
            uint64_t key = 400000 + t * operations_per_thread + i;
            std::string base_value =
                "thread_" + std::to_string(t) + "_value_" + std::to_string(i);
            std::string value = CreateFixedLengthValue(base_value);
            try {
              kv_engine_->Put(key, value, 0);
            } catch (const std::exception&) {
              failed_operations++;
            }
          }
        });
  }

  for (auto& thread : threads)
    thread.join();

  EXPECT_EQ(failed_operations.load(), 0);

  for (int t = 0; t < num_threads; t++) {
    for (int i = 0; i < operations_per_thread; i++) {
      uint64_t key = 400000 + t * operations_per_thread + i;
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
TEST_F(SSDIndexSSDValueTest, ConcurrentGetTest) {
  const int num_data         = 200;
  const int num_threads      = 16;
  const int reads_per_thread = 1000;

  for (int i = 0; i < num_data; i++) {
    std::string value =
        CreateFixedLengthValue("concurrent_get_value_" + std::to_string(i));
    kv_engine_->Put(i + 600000, value, 0);
  }

  std::vector<std::thread> threads;
  std::atomic<int> successful_reads(0);
  std::atomic<int> failed_reads(0);
  SimpleBarrier barrier(num_threads);

  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back(
        [this,
         reads_per_thread,
         num_data,
         &barrier,
         &successful_reads,
         &failed_reads]() {
          barrier.wait();
          std::mt19937 gen(std::random_device{}());
          std::uniform_int_distribution<int> dist(0, num_data - 1);

          for (int i = 0; i < reads_per_thread; i++) {
            uint64_t key = dist(gen) + 600000;
            std::string retrieved_value;
            try {
              kv_engine_->Get(key, retrieved_value, 0);
              if (!retrieved_value.empty())
                successful_reads++;
              else
                failed_reads++;
            } catch (const std::exception&) {
              failed_reads++;
            }
          }
        });
  }

  for (auto& thread : threads)
    thread.join();

  EXPECT_GT(successful_reads.load(), 0);
  EXPECT_EQ(failed_reads.load(), 0);
  EXPECT_EQ(successful_reads.load(), num_threads * reads_per_thread);
}

// 多线程混合读写测试
TEST_F(SSDIndexSSDValueTest, ConcurrentReadWriteTest) {
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
          std::mt19937 gen(std::random_device{}());
          std::uniform_int_distribution<int> op_dist(0, 1);
          std::uniform_int_distribution<uint64_t> key_dist(700000, 700199);

          for (int i = 0; i < operations_per_thread; i++) {
            uint64_t key = key_dist(gen);
            try {
              if (op_dist(gen) == 0) {
                std::string base_value = "mixed_thread_" + std::to_string(t) +
                                         "_value_" + std::to_string(i);
                kv_engine_->Put(key, CreateFixedLengthValue(base_value), 0);
              } else {
                std::string retrieved_value;
                kv_engine_->Get(key, retrieved_value, 0);
              }
              successful_operations++;
            } catch (const std::exception&) {
              failed_operations++;
            }
          }
        });
  }

  for (auto& thread : threads)
    thread.join();

  EXPECT_EQ(failed_operations.load(), 0);
  EXPECT_EQ(successful_operations.load(), num_threads * operations_per_thread);
}

// 数据一致性测试
TEST_F(SSDIndexSSDValueTest, DataConsistencyTest) {
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
                kv_engine_->Put(key + 800000, value, 0);
                total_updates++;
              } catch (const std::exception&) {
              }
            }
          }
        });
  }

  for (auto& thread : threads)
    thread.join();

  int valid_keys = 0;
  for (int key = 0; key < num_keys; key++) {
    std::string retrieved_value;
    kv_engine_->Get(key + 800000, retrieved_value, 0);
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
