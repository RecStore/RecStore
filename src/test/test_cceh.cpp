#include "storage/index/ssd_cceh/CCEH.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <thread>
#include <vector>

namespace {
std::string GetDirectIOTestFilePath() {
  const std::filesystem::path dir =
      std::filesystem::current_path() / "test_cceh_data";
  return (dir / "test_cceh.db").string();
}
} // namespace

BaseKVConfig config{
    0,
    {{"io_backend_type", "IOURING"},
     {"page_id_offset", 0},
     {"queue_cnt", 512},
     {"file_path", GetDirectIOTestFilePath()}}};

class CCEHTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::filesystem::path file_path =
        config.json_config_.at("file_path").get<std::string>();
    std::filesystem::create_directories(file_path.parent_path());
    std::remove(config.json_config_.at("file_path").get<std::string>().c_str());
  }
  void TearDown() override {
    std::remove(config.json_config_.at("file_path").get<std::string>().c_str());
  }
};

TEST_F(CCEHTest, SimpleInsertAndGet) {
  CCEH cceh(config);

  Key_t key     = 100;
  Value_t value = 200;
  cceh.Put(key, value, 0);

  Value_t ret_val;
  cceh.Get(key, ret_val, 0);
  EXPECT_EQ(ret_val, value);

  Key_t not_exist_key = 101;
  cceh.Get(not_exist_key, ret_val, 0);
  EXPECT_EQ(ret_val, NONE);
}

TEST_F(CCEHTest, SplitTest) {
  CCEH cceh(config);

  const int num_to_insert = 10000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    keys.push_back(key);
    cceh.Put(key, key * 2, 0);
  }

  for (const auto& key : keys) {
    Value_t ret_val;
    cceh.Get(key, ret_val, 0);
    EXPECT_EQ(ret_val, key * 2);
  }
}

TEST_F(CCEHTest, DirectoryExpansionTest) {
  CCEH cceh(config);

  const int num_to_insert = 100000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i * 3;
    keys.push_back(key);
    cceh.Put(key, key * 2, 0);
  }

  for (const auto& key : keys) {
    Value_t ret_val;
    cceh.Get(key, ret_val, 0);
    if (ret_val != key * 2) {
      EXPECT_EQ(ret_val, key * 2) << "Failed for key: " << key;
    }
  }
}

TEST_F(CCEHTest, ConcurrentInsertTest) {
  CCEH cceh(config);

  const int kNumThreads       = 64;
  const int kInsertsPerThread = 1000;
  std::vector<std::thread> threads;

  auto inserter_func = [&](int thread_id) {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      Key_t key = thread_id * kInsertsPerThread + i;
      cceh.Put(key, key * 2, 0);
    }
  };

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(inserter_func, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  // Verification
  for (int i = 0; i < kNumThreads * kInsertsPerThread; ++i) {
    Key_t key = i;
    Value_t ret_val;
    cceh.Get(key, ret_val, 0);
    EXPECT_EQ(ret_val, key * 2);
  }
}
