// ======== 笛卡尔积参数化测试 ========

#include <tuple>
#include <cmath>

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
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

#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "test_io_uring_helper.h"

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

  static bool IsHeavyTestName(const std::string& test_name) {
    static const std::vector<std::string> kHeavyPrefixes = {
        "RandomData",
        "PerformanceTest",
        "StressTest",
        "ConcurrentPutTest",
        "ConcurrentGetTest",
        "ConcurrentReadWriteTest",
        "ConcurrentBatchGetTest",
        "DataConsistencyTest"};
    for (const auto& prefix : kHeavyPrefixes) {
      if (test_name.rfind(prefix, 0) == 0) {
        return true;
      }
    }
    return false;
  }

  static bool IsSlabAllocatorForHeavyScenarios(const std::string& allocator) {
    return allocator == "PERSIST_LOOP_SLAB" ||
           allocator == "CONCURRENT_SLAB_MEMORY_POOL";
  }

  static bool RunAllEnabled() {
    const char* value = std::getenv("RUN_ALL");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }

  static bool ShouldRunHeavyScenario(const std::string& idx,
                                     const std::string& val,
                                     const std::string& allocator) {
    if (RunAllEnabled())
      return true;
    if (idx == "DRAM_EXTENDIBLE_HASH" && val == "DRAM_VALUE_STORE")
      return true;
    if (!IsSlabAllocatorForHeavyScenarios(allocator))
      return false;
    return idx == "DRAM_UNORDERED_MAP" || idx == "DRAM_PET_HASH";
  }

protected:
  std::string CreateFixedLengthValue(const std::string& base_value) {
    std::string value = base_value;
    value.resize(128);
    return value;
  }

  void SetUp() override {
    const char *idx_c, *val_c, *allocator_c;
    std::tie(idx_c, val_c, allocator_c) = GetParam();
    index_type_                         = idx_c;
    value_type_                         = val_c;
    allocator_type_                     = allocator_c;

    const auto* test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info != nullptr && IsHeavyTestName(test_info->name()) &&
        !ShouldRunHeavyScenario(index_type_, value_type_, allocator_type_)) {
      GTEST_SKIP() << "Skip heavy scenario for (" << index_type_ << ","
                   << value_type_ << "," << allocator_type_ << ")";
    }

    const std::string test_name =
        "test_kv_engine_cartesian_" + std::to_string(getpid()) + "_" +
        index_type_ + "_" + value_type_ + "_" + allocator_type_;
    dram_test_dir_ = "/dev/shm/" + test_name;
    ssd_test_dir_  = "/tmp/" + test_name;
    std::filesystem::create_directories(dram_test_dir_);
    std::filesystem::create_directories(ssd_test_dir_);

    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;

    cfg_.num_threads_     = 8;
    const size_t capacity = 1000000;
    const int value_sz    = 128;

    if ((index_type_ == "SSD" || index_type_ == "SSD_EXTENDIBLE_HASH") &&
        value_type_ != "SSD_VALUE_STORE") {
      GTEST_SKIP() << "Skip invalid SSD index combo with " << value_type_;
    }

    if (NeedsIoUring(index_type_, value_type_)) {
      std::string reason;
      if (!test_utils::CanUseIoUring(&reason)) {
        GTEST_SKIP() << reason;
      }
    }

    cfg_.json_config_ = BuildConfig(capacity, value_sz);

    auto r       = base::ResolveEngine(cfg_);
    engine_name_ = r.engine;
    ASSERT_EQ(engine_name_, std::string("KVEngineComposite"))
        << "selector derived mismatch for (" << index_type_ << ","
        << value_type_ << "," << allocator_type_ << ")";

    try {
      kv_engine_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          engine_name_, r.cfg));
    } catch (const std::exception& e) {
      if (RunAllEnabled()) {
        FAIL() << "Create engine failed for allocator=" << allocator_type_
               << " : " << e.what();
      } else {
        GTEST_SKIP() << "Create engine failed for allocator=" << allocator_type_
                     << " : " << e.what();
      }
    }
    if (!kv_engine_) {
      if (RunAllEnabled()) {
        FAIL() << "Engine '" << engine_name_ << "' or allocator '"
               << allocator_type_ << "' not registered/linked.";
      } else {
        GTEST_SKIP() << "Engine '" << engine_name_ << "' or allocator '"
                     << allocator_type_ << "' not registered/linked.";
      }
    }
  }

  void TearDown() override {
    kv_engine_.reset();
    std::filesystem::remove_all(dram_test_dir_);
    std::filesystem::remove_all(ssd_test_dir_);
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

  std::string dram_test_dir_;
  std::string ssd_test_dir_;
  BaseKVConfig cfg_;
  std::unique_ptr<BaseKV> kv_engine_;
  std::string index_type_, value_type_, allocator_type_, engine_name_;

private:
  static bool IsSsdIndex(const std::string& idx) {
    return idx == "SSD" || idx == "SSD_EXTENDIBLE_HASH";
  }

  static bool NeedsIoUring(const std::string& idx, const std::string& val) {
    return IsSsdIndex(idx) || val == "SSD_VALUE_STORE" ||
           val == "TIERED_VALUE_STORE";
  }

  json BuildConfig(size_t capacity, int value_sz) const {
    json j = {{"engine_type", "KVEngineComposite"},
              {"capacity", capacity},
              {"index", {{"type", index_type_}}},
              {"value",
               {{"type", value_type_}, {"default_value_size_hint", value_sz}}}};
    if (value_type_ == "DRAM_VALUE_STORE") {
      j["value"]["path"]           = dram_test_dir_ + "/value";
      j["value"]["dram_allocator"] = {
          {"type", allocator_type_},
          {"capacity_bytes", capacity * static_cast<size_t>(value_sz)}};
    } else if (value_type_ == "SSD_VALUE_STORE") {
      j["value"]["ssd_allocator"] = {
          {"type", "SSD_BUDDY"},
          {"capacity_bytes", capacity * static_cast<size_t>(value_sz)},
          {"min_block_size", 128},
          {"max_block_size", 65536},
          {"io",
           {{"type", "IOURING"},
            {"queue_depth", 512},
            {"base_offset_bytes", 4096}}}};
      j["value"]["path"] = ssd_test_dir_ + "/value_pages.db";
    } else if (value_type_ == "TIERED_VALUE_STORE") {
      j["value"]["dram_allocator"] = {
          {"type", allocator_type_},
          {"capacity_bytes", capacity * static_cast<size_t>(value_sz) / 2},
          {"path", dram_test_dir_ + "/dram"}};
      j["value"]["ssd_allocator"] = {
          {"type", "SSD_BUDDY"},
          {"capacity_bytes", capacity * static_cast<size_t>(value_sz)},
          {"min_block_size", 128},
          {"max_block_size", 65536},
          {"path", ssd_test_dir_ + "/tiered_value_pages.db"},
          {"io",
           {{"type", "IOURING"},
            {"queue_depth", 512},
            {"base_offset_bytes", 4096}}}};
      j["value"]["tiering"] = {{"cache_policy", "LRU"}};
    }
    return j;
  }
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

TEST_P(KVEngineCartesianTest, BatchGetFlatCopiesRowsAndZerosMissing) {
  const std::vector<uint64_t> put_keys       = {11, 13};
  std::vector<std::vector<float>> put_values = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {10.0f, 20.0f, 30.0f, 40.0f}};
  std::vector<base::ConstArray<float>> put_slices;
  put_slices.reserve(put_values.size());
  for (const auto& row : put_values) {
    put_slices.emplace_back(row.data(), static_cast<int>(row.size()));
  }
  kv_engine_->BatchPut(
      base::ConstArray<uint64_t>(put_keys.data(), put_keys.size()),
      &put_slices,
      0);

  const std::vector<uint64_t> get_keys = {11, 12, 13};
  std::vector<float> out(get_keys.size() * 4, -1.0f);
  BaseKV::BatchGetFlatStats stats;
  ASSERT_TRUE(kv_engine_->BatchGetFlat(
      base::ConstArray<uint64_t>(get_keys.data(), get_keys.size()),
      out.data(),
      static_cast<int64_t>(get_keys.size()),
      4,
      0,
      &stats));

  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 2.0f);
  EXPECT_FLOAT_EQ(out[2], 3.0f);
  EXPECT_FLOAT_EQ(out[3], 4.0f);
  EXPECT_FLOAT_EQ(out[4], 0.0f);
  EXPECT_FLOAT_EQ(out[5], 0.0f);
  EXPECT_FLOAT_EQ(out[6], 0.0f);
  EXPECT_FLOAT_EQ(out[7], 0.0f);
  EXPECT_FLOAT_EQ(out[8], 10.0f);
  EXPECT_FLOAT_EQ(out[9], 20.0f);
  EXPECT_FLOAT_EQ(out[10], 30.0f);
  EXPECT_FLOAT_EQ(out[11], 40.0f);
  EXPECT_EQ(stats.missing_rows, 1);
}

TEST_P(KVEngineCartesianTest, BatchGetFlatRejectsMismatchedDim) {
  const std::vector<uint64_t> keys            = {21};
  std::vector<float> row                      = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<base::ConstArray<float>> slices = {
      base::ConstArray<float>(row.data(), static_cast<int>(row.size()))};
  kv_engine_->BatchPut(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), &slices, 0);

  std::vector<float> out(3, -1.0f);
  EXPECT_FALSE(kv_engine_->BatchGetFlat(
      base::ConstArray<uint64_t>(keys.data(), keys.size()),
      out.data(),
      1,
      3,
      0));
}

TEST(KVEngineFlatReadTest, DramRowsSupportNonPowerOfTwoByteWidths) {
  constexpr int64_t kDim = 15;
  const std::string value_path =
      "/dev/shm/test_kv_engine_flat_non_power_" + std::to_string(getpid());
  std::filesystem::create_directories(value_path);

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::Backend::kAnonymousDram;
  BaseKVConfig config;
  config.num_threads_ = 1;
  config.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 128},
      {"index", {{"type", "DRAM_PET_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", value_path},
        {"default_value_size_hint", kDim * sizeof(float)},
        {"dram_allocator",
         {{"type", "CONCURRENT_SLAB_MEMORY_POOL"},
          {"capacity_bytes", 64ULL << 20}}}}}};
  auto resolved = base::ResolveEngine(config);
  std::unique_ptr<BaseKV> engine(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
  ASSERT_NE(engine, nullptr);

  const std::vector<uint64_t> put_keys = {101, 202};
  std::vector<float> first(kDim);
  std::vector<float> second(kDim);
  for (int64_t column = 0; column < kDim; ++column) {
    first[static_cast<size_t>(column)] = static_cast<float>(column + 1);
    second[static_cast<size_t>(column)] = static_cast<float>(100 + column);
  }
  std::vector<base::ConstArray<float>> rows = {
      base::ConstArray<float>(first.data(), static_cast<int>(first.size())),
      base::ConstArray<float>(second.data(), static_cast<int>(second.size()))};
  engine->BatchPut(base::ConstArray<uint64_t>(put_keys), &rows, 0);

  const std::vector<uint64_t> get_keys = {202, 303, 101};
  std::vector<float> output(get_keys.size() * kDim, -1.0F);
  BaseKV::BatchGetFlatStats stats;
  ASSERT_TRUE(engine->BatchGetFlat(base::ConstArray<uint64_t>(get_keys),
                                   output.data(), get_keys.size(), kDim, 0,
                                   &stats));
  EXPECT_EQ(stats.missing_rows, 1);
  EXPECT_EQ(std::vector<float>(output.begin(), output.begin() + kDim), second);
  EXPECT_EQ(std::vector<float>(output.begin() + kDim,
                               output.begin() + 2 * kDim),
            std::vector<float>(kDim, 0.0F));
  EXPECT_EQ(std::vector<float>(output.begin() + 2 * kDim, output.end()), first);

  engine.reset();
  std::filesystem::remove_all(value_path);
}

TEST(KVEngineFlatReadTest, DramRowsSupportDirectFixedSlices) {
  constexpr int64_t kDim = 16;
  constexpr int64_t kMetadataFloats = sizeof(int64_t) / sizeof(float);
  constexpr int64_t kStoredDim = kMetadataFloats + kDim;
  const std::string value_path =
      "/dev/shm/test_kv_engine_flat_slice_" + std::to_string(getpid());
  std::filesystem::create_directories(value_path);

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::Backend::kAnonymousDram;
  BaseKVConfig config;
  config.num_threads_ = 1;
  config.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 128},
      {"index", {{"type", "DRAM_PET_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", value_path},
        {"default_value_size_hint", kStoredDim * sizeof(float)},
        {"dram_allocator",
         {{"type", "CONCURRENT_SLAB_MEMORY_POOL"},
          {"capacity_bytes", 64ULL << 20}}}}}};
  auto resolved = base::ResolveEngine(config);
  std::unique_ptr<BaseKV> engine(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
  ASSERT_NE(engine, nullptr);

  const std::vector<uint64_t> put_keys = {101, 202};
  std::vector<float> first(kStoredDim);
  std::vector<float> second(kStoredDim);
  const int64_t first_index = 17;
  const int64_t second_index = 29;
  std::memcpy(first.data(), &first_index, sizeof(first_index));
  std::memcpy(second.data(), &second_index, sizeof(second_index));
  for (int64_t column = 0; column < kDim; ++column) {
    first[static_cast<size_t>(kMetadataFloats + column)] =
        static_cast<float>(column + 1);
    second[static_cast<size_t>(kMetadataFloats + column)] =
        static_cast<float>(100 + column);
  }
  std::vector<base::ConstArray<float>> rows = {
      base::ConstArray<float>(first.data(), static_cast<int>(first.size())),
      base::ConstArray<float>(second.data(), static_cast<int>(second.size()))};
  engine->BatchPut(base::ConstArray<uint64_t>(put_keys), &rows, 0);

  const std::vector<uint64_t> get_keys = {202, 303, 101};
  std::vector<float> output(get_keys.size() * kDim, -1.0F);
  BaseKV::BatchGetFlatStats stats;
  ASSERT_TRUE(engine->BatchGetFlatRange(
      base::ConstArray<uint64_t>(get_keys), output.data(), get_keys.size(),
      kStoredDim, kMetadataFloats, kDim, 0, &stats));
  EXPECT_EQ(stats.missing_rows, 1);
  EXPECT_EQ(std::vector<float>(output.begin(), output.begin() + kDim),
            std::vector<float>(second.begin() + kMetadataFloats, second.end()));
  EXPECT_EQ(std::vector<float>(output.begin() + kDim,
                               output.begin() + 2 * kDim),
            std::vector<float>(kDim, 0.0F));
  EXPECT_EQ(std::vector<float>(output.begin() + 2 * kDim, output.end()),
            std::vector<float>(first.begin() + kMetadataFloats, first.end()));
  EXPECT_FALSE(engine->BatchGetFlatRange(
      base::ConstArray<uint64_t>(get_keys), output.data(), get_keys.size(),
      kStoredDim, kStoredDim, kDim, 0));

  engine.reset();
  std::filesystem::remove_all(value_path);
}

TEST_P(KVEngineCartesianTest, ApplySgdUpdateFlatUpdatesAndInitializesRows) {
  if (value_type_ != "DRAM_VALUE_STORE") {
    GTEST_SKIP() << "Direct flat SGD update requires DRAM_VALUE_STORE";
  }

  constexpr int64_t kDim     = 32;
  std::vector<uint64_t> keys = {101, 202};
  std::vector<float> initial(kDim, 1.0f);
  kv_engine_->Put(
      keys[0],
      std::string_view(reinterpret_cast<const char*>(initial.data()),
                       initial.size() * sizeof(float)),
      0);

  std::vector<float> grads(keys.size() * kDim, 2.0f);
  ASSERT_TRUE(kv_engine_->ApplySgdUpdateFlat(
      base::ConstArray<uint64_t>(keys),
      grads.data(),
      keys.size(),
      kDim,
      0.1f,
      0,
      0));

  for (size_t row = 0; row < keys.size(); ++row) {
    std::string value;
    kv_engine_->Get(keys[row], value, 0);
    ASSERT_EQ(value.size(), initial.size() * sizeof(float));
    const auto* floats   = reinterpret_cast<const float*>(value.data());
    const float expected = row == 0 ? 0.8f : -0.2f;
    for (int64_t col = 0; col < kDim; ++col) {
      EXPECT_FLOAT_EQ(floats[col], expected);
    }
  }
}

TEST_P(KVEngineCartesianTest, BatchPutMixedOverwriteAndRealloc) {
  const std::vector<uint64_t> keys = {7001, 7002, 7003, 7004};
  std::vector<std::vector<float>> initial_values(
      keys.size(), std::vector<float>(16, 1.0f));
  std::vector<base::ConstArray<float>> initial_slices;
  initial_slices.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    for (size_t j = 0; j < initial_values[i].size(); ++j) {
      initial_values[i][j] = static_cast<float>(i * 10 + j);
    }
    initial_slices.emplace_back(
        initial_values[i].data(), static_cast<int>(initial_values[i].size()));
  }
  kv_engine_->BatchPut(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), &initial_slices, 0);

  std::vector<std::vector<float>> updated_values(keys.size());
  updated_values[0].resize(16); // overwrite path
  updated_values[1].resize(48); // realloc path
  updated_values[2].resize(16); // overwrite path
  updated_values[3].resize(64); // realloc path
  for (size_t i = 0; i < updated_values.size(); ++i) {
    for (size_t j = 0; j < updated_values[i].size(); ++j) {
      updated_values[i][j] = static_cast<float>(1000 + i * 100 + j);
    }
  }
  std::vector<base::ConstArray<float>> updated_slices;
  updated_slices.reserve(keys.size());
  for (size_t i = 0; i < updated_values.size(); ++i) {
    updated_slices.emplace_back(
        updated_values[i].data(), static_cast<int>(updated_values[i].size()));
  }
  kv_engine_->BatchPut(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), &updated_slices, 0);

  std::vector<base::ConstArray<float>> out_values;
  kv_engine_->BatchGet(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), &out_values, 0);
  ASSERT_EQ(out_values.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_EQ(out_values[i].Size(), static_cast<int>(updated_values[i].size()));
    for (int j = 0; j < out_values[i].Size(); ++j) {
      EXPECT_FLOAT_EQ(out_values[i][j], updated_values[i][j])
          << "key=" << keys[i] << " idx=" << j;
    }
  }
}

TEST_P(KVEngineCartesianTest, BatchGetRepeatedKeysKeepsConsistentValues) {
  std::vector<uint64_t> put_keys                  = {8101, 8102};
  std::vector<std::vector<float>> put_values_data = {
      std::vector<float>(32, 0.0f), std::vector<float>(32, 0.0f)};
  for (int i = 0; i < 32; ++i) {
    put_values_data[0][i] = static_cast<float>(i + 1);
    put_values_data[1][i] = static_cast<float>(200 + i);
  }
  std::vector<base::ConstArray<float>> put_values = {
      base::ConstArray<float>(
          put_values_data[0].data(), put_values_data[0].size()),
      base::ConstArray<float>(
          put_values_data[1].data(), put_values_data[1].size())};
  kv_engine_->BatchPut(
      base::ConstArray<uint64_t>(put_keys.data(), put_keys.size()),
      &put_values,
      0);

  std::vector<uint64_t> get_keys = {8102, 8101, 8102, 8101, 8101};
  std::vector<base::ConstArray<float>> out_values;
  kv_engine_->BatchGet(
      base::ConstArray<uint64_t>(get_keys.data(), get_keys.size()),
      &out_values,
      0);
  ASSERT_EQ(out_values.size(), get_keys.size());

  for (size_t i = 0; i < get_keys.size(); ++i) {
    const auto& expected =
        (get_keys[i] == 8101) ? put_values_data[0] : put_values_data[1];
    ASSERT_EQ(out_values[i].Size(), static_cast<int>(expected.size()));
    for (int j = 0; j < out_values[i].Size(); ++j) {
      EXPECT_FLOAT_EQ(out_values[i][j], expected[j])
          << "key=" << get_keys[i] << " idx=" << j;
    }
  }
}

TEST_P(KVEngineCartesianTest, BulkLoadThenGetAndBatchGet) {
  constexpr int kRows         = 8;
  constexpr int kFloatsPerRow = 32;
  std::vector<uint64_t> keys;
  keys.reserve(kRows);
  std::vector<float> values(kRows * kFloatsPerRow);
  for (int row = 0; row < kRows; ++row) {
    keys.push_back(static_cast<uint64_t>(9001 + row));
    for (int col = 0; col < kFloatsPerRow; ++col) {
      values[row * kFloatsPerRow + col] = static_cast<float>(row * 100 + col);
    }
  }

  kv_engine_->BulkLoad(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), values.data());

  for (int row = 0; row < kRows; ++row) {
    std::string out;
    kv_engine_->Get(keys[row], out, 0);
    ASSERT_EQ(out.size(), kFloatsPerRow * sizeof(float));
    EXPECT_EQ(std::memcmp(
                  out.data(), values.data() + row * kFloatsPerRow, out.size()),
              0)
        << "row=" << row;
  }

  std::vector<base::ConstArray<float>> out_values;
  kv_engine_->BatchGet(
      base::ConstArray<uint64_t>(keys.data(), keys.size()), &out_values, 0);
  ASSERT_EQ(out_values.size(), keys.size());
  for (int row = 0; row < kRows; ++row) {
    ASSERT_EQ(out_values[row].Size(), kFloatsPerRow);
    for (int col = 0; col < kFloatsPerRow; ++col) {
      EXPECT_FLOAT_EQ(out_values[row][col], values[row * kFloatsPerRow + col])
          << "row=" << row << " col=" << col;
    }
  }
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

  const int num_operations = 200;
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
  const int num_operations = 200;

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

  std::cout << "KVEngineComposite Performance Results for " << num_operations
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

TEST_P(KVEngineCartesianTest, StressTest) {
  const int num_operations = 2000;

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
  const int num_threads           = 8;
  const int operations_per_thread = 200;
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
              kv_engine_->Put(key, value, t);
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
  const int num_threads      = 8;
  const int reads_per_thread = 200;

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
  const int num_threads           = 8;
  const int operations_per_thread = 200;
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
                kv_engine_->Put(key, value, t);
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
  const int num_threads = 8;
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
  const int num_threads     = 8;
  const int num_keys        = 200;
  const int updates_per_key = 3;

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
                kv_engine_->Put(key, value, t);
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

#if defined(RECSTORE_TEST_ENABLE_FASTERKV_ENGINE) ||                           \
    defined(RECSTORE_TEST_ENABLE_HPS_ENGINE)

class KVEngineExternalEngineTest
    : public ::testing::TestWithParam<const char*> {
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

  void SetUp() override {
    engine_type_ = GetParam();
    test_dir_    = "/tmp/test_kv_engine_external_" + engine_type_ + "_" +
                std::to_string(getpid());
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directories(test_dir_);

    cfg_.num_threads_ = 8;
    cfg_.json_config_ = {
        {"engine_type", engine_type_},
        {"path", test_dir_},
        {"capacity", 1000000},
        {"value_size", 128}};

    auto resolved = base::ResolveEngine(cfg_);
    engine_name_  = resolved.engine;
    ASSERT_EQ(engine_name_, engine_type_);

    try {
      kv_engine_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          engine_name_, resolved.cfg));
    } catch (const std::exception& e) {
      FAIL() << "Create external engine failed for " << engine_type_ << " : "
             << e.what();
    }
    ASSERT_NE(kv_engine_, nullptr);
  }

  void TearDown() override {
    kv_engine_.reset();
    std::filesystem::remove_all(test_dir_);
  }

  std::string test_dir_;
  BaseKVConfig cfg_;
  std::unique_ptr<BaseKV> kv_engine_;
  std::string engine_type_;
  std::string engine_name_;
};

TEST_P(KVEngineExternalEngineTest, BasicPutAndGet) {
  uint64_t key      = 123;
  std::string value = CreateFixedLengthValue("external_test_value_123");
  std::string retrieved_value;

  kv_engine_->Put(key, value, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value);
}

TEST_P(KVEngineExternalEngineTest, GetNonExistentKey) {
  std::string retrieved_value;
  kv_engine_->Get(999999, retrieved_value, 0);
  EXPECT_TRUE(retrieved_value.empty());
}

TEST_P(KVEngineExternalEngineTest, KeyOverwrite) {
  uint64_t key       = 456;
  std::string value1 = CreateFixedLengthValue("external_value1");
  std::string value2 = CreateFixedLengthValue("external_value2");
  std::string retrieved_value;

  kv_engine_->Put(key, value1, 0);
  kv_engine_->Put(key, value2, 0);
  kv_engine_->Get(key, retrieved_value, 0);
  EXPECT_EQ(retrieved_value, value2);
}

TEST_P(KVEngineExternalEngineTest, MultiplePutAndGet) {
  const int num_keys = 100;
  for (int i = 0; i < num_keys; i++) {
    std::string value =
        CreateFixedLengthValue("external_key_" + std::to_string(i));
    kv_engine_->Put(i, value, 0);
  }
  for (int i = 0; i < num_keys; i++) {
    std::string expected =
        CreateFixedLengthValue("external_key_" + std::to_string(i));
    std::string retrieved;
    kv_engine_->Get(i, retrieved, 0);
    EXPECT_EQ(retrieved, expected) << "Mismatch for key " << i;
  }
}

TEST_P(KVEngineExternalEngineTest, BatchGet) {
  std::vector<uint64_t> keys;
  for (int i = 0; i < 10; i++) {
    keys.push_back(i);
    kv_engine_->Put(i, CreateFixedLengthValue("batch_" + std::to_string(i)), 0);
  }

  base::ConstArray<uint64_t> key_view(keys.data(), keys.size());
  std::vector<base::ConstArray<float>> values;
  kv_engine_->BatchGet(key_view, &values, 0);

  ASSERT_EQ(values.size(), keys.size());
  const int floats_per_row = 128 / static_cast<int>(sizeof(float));
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_EQ(values[i].Size(), floats_per_row);
  }
}

TEST_P(KVEngineExternalEngineTest, BatchGetFlatUnsupported) {
  std::vector<uint64_t> keys = {1, 2};
  for (uint64_t key : keys) {
    kv_engine_->Put(
        key, CreateFixedLengthValue("flat_" + std::to_string(key)), 0);
  }

  std::vector<float> flat_values(keys.size() * 32);
  EXPECT_FALSE(kv_engine_->BatchGetFlat(
      base::ConstArray<uint64_t>(keys.data(), keys.size()),
      flat_values.data(),
      static_cast<int64_t>(keys.size()),
      32,
      0));
}

TEST_P(KVEngineExternalEngineTest, BulkLoadThenGetAndBatchGet) {
  const int num_keys = 50;
  std::vector<uint64_t> keys(num_keys);
  std::vector<std::string> values(num_keys);
  std::vector<char> bulk_data(static_cast<size_t>(num_keys) * 128);

  for (int i = 0; i < num_keys; i++) {
    keys[i]   = i + 1000;
    values[i] = CreateFixedLengthValue("bulk_" + std::to_string(i));
    std::memcpy(
        bulk_data.data() + static_cast<size_t>(i) * 128, values[i].data(), 128);
  }

  base::ConstArray<uint64_t> key_view(keys.data(), keys.size());
  kv_engine_->BulkLoad(key_view, bulk_data.data());

  for (int i = 0; i < num_keys; i++) {
    std::string retrieved;
    kv_engine_->Get(keys[i], retrieved, 0);
    EXPECT_EQ(retrieved, values[i]);
  }

  std::vector<base::ConstArray<float>> batch_values;
  kv_engine_->BatchGet(key_view, &batch_values, 0);
  ASSERT_EQ(batch_values.size(), keys.size());
}

#  ifdef RECSTORE_TEST_ENABLE_FASTERKV_ENGINE
INSTANTIATE_TEST_SUITE_P(
    FasterKV,
    KVEngineExternalEngineTest,
    ::testing::Values("KVEngineFasterKV"),
    [](const testing::TestParamInfo<KVEngineExternalEngineTest::ParamType>&
           info) { return KVEngineExternalEngineTest::Sanitize(info.param); });
#  endif

#  ifdef RECSTORE_TEST_ENABLE_HPS_ENGINE
INSTANTIATE_TEST_SUITE_P(
    HPSRocksDB,
    KVEngineExternalEngineTest,
    ::testing::Values("KVEngineHPSRocksDB"),
    [](const testing::TestParamInfo<KVEngineExternalEngineTest::ParamType>&
           info) { return KVEngineExternalEngineTest::Sanitize(info.param); });
#  endif

#endif // RECSTORE_TEST_ENABLE_FASTERKV_ENGINE ||
       // RECSTORE_TEST_ENABLE_HPS_ENGINE

INSTANTIATE_TEST_SUITE_P(
    AllCombos,
    KVEngineCartesianTest,
    ::testing::Combine(
        ::testing::Values(
            "DRAM_EXTENDIBLE_HASH", "DRAM_UNORDERED_MAP", "DRAM_PET_HASH"),
        ::testing::Values(
            "DRAM_VALUE_STORE", "SSD_VALUE_STORE", "TIERED_VALUE_STORE"),
        ::testing::Values(
            "PERSIST_LOOP_SLAB", "R2_SLAB", "CONCURRENT_SLAB_MEMORY_POOL")),
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

TEST(KVEngineNestedConfigTest, DramIndexDramValueSmoke) {
  const std::string test_dir =
      "/dev/shm/test_kv_engine_nested_" + std::to_string(getpid());
  std::filesystem::remove_all(test_dir);
  std::filesystem::create_directories(test_dir);

  BaseKVConfig cfg;
  cfg.num_threads_ = 4;
  cfg.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 1024},
      {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", test_dir + "/value"},
        {"default_value_size_hint", 128},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1024 * 128}}}}}};

  auto resolved = base::ResolveEngine(cfg);
  ASSERT_EQ(resolved.engine, "KVEngineComposite");
  std::unique_ptr<BaseKV> kv(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));

  std::string value = "nested-config-value";
  value.resize(128);
  kv->Put(7, value, 0);

  std::string out;
  kv->Get(7, out, 0);
  EXPECT_EQ(out, value);

  std::string overwrite = "nested-config-overwrite";
  overwrite.resize(128);
  kv->Put(7, overwrite, 0);
  kv->Get(7, out, 0);
  EXPECT_EQ(out, overwrite);

  kv.reset();
  std::filesystem::remove_all(test_dir);
}

namespace {

std::unique_ptr<BaseKV> CreateCompositeEngine(const BaseKVConfig& cfg) {
  auto resolved = base::ResolveEngine(cfg);
  EXPECT_EQ(resolved.engine, "KVEngineComposite");
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

} // namespace

TEST(KVEngineCompositeConfigTest,
     ResolveEngineDefaultsToCompositeWithoutEngineType) {
  BaseKVConfig cfg;
  cfg.json_config_ = {{"capacity", 1024},
                      {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
                      {"value", {{"type", "DRAM_VALUE_STORE"}}}};
  auto resolved    = base::ResolveEngine(cfg);
  EXPECT_EQ(resolved.engine, "KVEngineComposite");
}

TEST(KVEngineCompositeConfigTest, DramValueStoreAllowsAnonymousPathInSelector) {
  BaseKVConfig cfg;
  cfg.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 1024},
      {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", ""},
        {"default_value_size_hint", 128},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1024 * 128}}}}}};

  EXPECT_NO_THROW(base::ResolveEngine(cfg));
  EXPECT_THROW(CreateCompositeEngine(cfg), std::invalid_argument);
}

TEST(KVEngineCompositeConfigTest, DramValueStoreAcceptsDevShmPath) {
  BaseKVConfig cfg;
  cfg.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 1024},
      {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", "/dev/shm/kv/value"},
        {"default_value_size_hint", 128},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1024 * 128}}}}}};

  EXPECT_NO_THROW(CreateCompositeEngine(cfg));
}
