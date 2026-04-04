#include "storage/ssd/SSD_CCEH.h"
#include "gtest/gtest.h"
#include <thread>
#include <vector>

BaseKVConfig config{
    0,
    {{"io_backend_type", "IOURING"},
     {"page_id_offset", 0},
     {"queue_cnt", 512},
     {"file_path", "/tmp/test_cceh.db"}}};

class SSDCCEHTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::remove(config.json_config_.at("file_path").get<std::string>().c_str());
  }
  void TearDown() override {
    std::remove(config.json_config_.at("file_path").get<std::string>().c_str());
  }
};

TEST_F(SSDCCEHTest, SimpleInsertAndGet) {
  SSDCCEH ssd_cceh(config);

  Key_t key     = 100;
  Value_t value = 200;
  ssd_cceh.Put(key, value, 0);

  Value_t ret_val;
  ssd_cceh.Get(key, ret_val, 0);
  EXPECT_EQ(ret_val, value);

  Key_t not_exist_key = 101;
  ssd_cceh.Get(not_exist_key, ret_val, 0);
  EXPECT_EQ(ret_val, NONE);
}

TEST_F(SSDCCEHTest, SplitTest) {
  SSDCCEH ssd_cceh(config);

  const int num_to_insert = 10000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i;
    keys.push_back(key);
    ssd_cceh.Put(key, key * 2, 0);
  }

  for (const auto& key : keys) {
    Value_t ret_val;
    ssd_cceh.Get(key, ret_val, 0);
    EXPECT_EQ(ret_val, key * 2);
  }
}

TEST_F(SSDCCEHTest, DirectoryExpansionTest) {
  SSDCCEH ssd_cceh(config);

  const int num_to_insert = 100000;
  std::vector<Key_t> keys;
  for (int i = 0; i < num_to_insert; ++i) {
    Key_t key = i * 3;
    keys.push_back(key);
    ssd_cceh.Put(key, key * 2, 0);
  }

  for (const auto& key : keys) {
    Value_t ret_val;
    ssd_cceh.Get(key, ret_val, 0);
    if (ret_val != key * 2) {
      EXPECT_EQ(ret_val, key * 2) << "Failed for key: " << key;
    }
  }
}

TEST_F(SSDCCEHTest, ConcurrentInsertTest) {
  SSDCCEH ssd_cceh(config);

  const int kNumThreads       = 64;
  const int kInsertsPerThread = 1000;
  std::vector<std::thread> threads;

  auto inserter_func = [&](int thread_id) {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      Key_t key = thread_id * kInsertsPerThread + i;
      ssd_cceh.Put(key, key * 2, 0);
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
    ssd_cceh.Get(key, ret_val, 0);
    EXPECT_EQ(ret_val, key * 2);
  }
}