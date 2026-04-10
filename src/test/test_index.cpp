#include "storage/index/dram_cceh/extendible_hash.h"
#include <gtest/gtest.h>
#include <vector>

class IndexTest : public ::testing::Test {
protected:
  void SetUp() override {
    BaseKVConfig config{0, {{"initial_capacity", 16}}};
    index_ = new ExtendibleHash(config); // 初始化为 ExtendibleHash 实例
  }

  void TearDown() override { delete index_; }

  Index* index_;
};

// 基本 Put 和 Get 测试
TEST_F(IndexTest, BasicPutAndGet) {
  uint64_t key   = 123;
  uint64_t value = 456;
  unsigned tid   = 0;

  index_->Put(key, value, tid); // 插入
  uint64_t retrieved;
  index_->Get(key, retrieved, tid); // 检索

  EXPECT_EQ(retrieved, value)
      << "Failed to retrieve correct value for key " << key;
}

// 测试不存在的键
TEST_F(IndexTest, GetNonExistentKey) {
  uint64_t key = 999;
  uint64_t retrieved;
  unsigned tid = 0;

  index_->Get(key, retrieved, tid);

  EXPECT_EQ(retrieved, 0)
      << "Non-existent key should return 0"; // 假设 0 为无效
}

// 测试键值覆盖
TEST_F(IndexTest, PutOverwrite) {
  uint64_t key    = 100;
  uint64_t value1 = 200;
  uint64_t value2 = 300;
  unsigned tid    = 0;

  index_->Put(key, value1, tid);
  index_->Put(key, value2, tid);

  uint64_t retrieved;
  index_->Get(key, retrieved, tid);
  EXPECT_EQ(retrieved, value2) << "Failed to overwrite value for key " << key;
}

// 测试 BatchGet
TEST_F(IndexTest, BatchGetWithoutSink) {
  const int num_pairs = 50;
  std::vector<uint64_t> keys(num_pairs);
  std::vector<uint64_t> values(num_pairs);
  unsigned tid = 0;

  // 插入数据
  for (int i = 0; i < num_pairs; i++) {
    keys[i]   = i + 2000;
    values[i] = i * 30;
    index_->Put(keys[i], values[i], tid);
  }

  base::ConstArray<uint64_t> keys_array(keys.data(), num_pairs);
  std::vector<uint64_t> retrieved_values(num_pairs);
  index_->BatchGet(keys_array, retrieved_values.data(), tid);

  for (int i = 0; i < num_pairs; i++) {
    EXPECT_EQ(retrieved_values[i], values[i]) << "Failed for key " << keys[i];
  }
}