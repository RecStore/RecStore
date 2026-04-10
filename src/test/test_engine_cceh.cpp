#include <condition_variable>
#include <filesystem>
#include <gtest/gtest.h>
#include <mutex>
#include <string>

#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/engine_cceh.h"

namespace {
std::string GetDirectIOTestDir() {
  const std::filesystem::path dir =
      std::filesystem::current_path() /
      ("test_kv_engine_cceh_" + std::to_string(getpid()));
  return dir.string();
}
} // namespace

class KVEngineCCEHTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_dir_ = GetDirectIOTestDir();
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
TEST_F(KVEngineCCEHTest, BasicPutAndGet) {
  uint64_t key      = 123;
  std::string value = CreateFixedLengthValue("test_value_123");
  std::string retrieved_value;
  kv_engine_->Put(key, value, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value);
}

// 测试多个键值对
TEST_F(KVEngineCCEHTest, MultiplePutAndGet) {
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
TEST_F(KVEngineCCEHTest, BatchGet) {
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

TEST_F(KVEngineCCEHTest, ConcurrentBatchGet) {
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

// BatchPut写入 + BatchGet读回，用float数据做roundtrip验证
TEST_F(KVEngineCCEHTest, BatchPutAndBatchGet) {
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

// 测试不同 value size 下 Put/Get 的正确性
// 覆盖场景：小于一页、恰好一页、跨两页、跨多页
TEST_F(KVEngineCCEHTest, VariableValueSize_PutGet) {
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
TEST_F(KVEngineCCEHTest, VariableValueSize_BatchPutBatchGet) {
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

TEST_F(KVEngineCCEHTest, UpdateOverwritesExistingValue) {
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
