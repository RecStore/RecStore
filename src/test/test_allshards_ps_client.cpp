#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "base/hash.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

DEFINE_int32(value_size, 16, "");
DEFINE_int32(max_kv_num_per_request, 4, "");

namespace {

class FakeShardClient : public BaseParameterClient {
public:
  explicit FakeShardClient(int shard)
      : BaseParameterClient("127.0.0.1", 1234, shard) {}

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override {
    values->clear();
    for (auto key : keys) {
      values->push_back(storage_[key]);
    }
    return 0;
  }

  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override {
    (void)isAsync;
    (void)async_req_id;
    request_sizes_.push_back(keys.Size());

    const int dim = FLAGS_value_size / sizeof(float);
    for (int i = 0; i < keys.Size(); ++i) {
      const auto& row = storage_[keys[i]];
      std::memcpy(values + i * dim, row.data(), FLAGS_value_size);
    }

    auto* status = reinterpret_cast<std::int32_t*>(
        reinterpret_cast<char*>(values) + keys.Size() * FLAGS_value_size);
    *status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    return next_rpc_id_++;
  }

  void InitThread() override {}

  void* GetReceiveBuffer(size_t size) override {
    buffers_.emplace_back(size);
    return buffers_.back().data();
  }

  bool QueryRPCFinished(int rpc_id) override {
    (void)rpc_id;
    return true;
  }

  void WaitRPCFinish(int rpc_id) override { (void)rpc_id; }

  void RevokeRPCResource(int rpc_id) override { (void)rpc_id; }

  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override {
    for (std::size_t i = 0; i < keys.size(); ++i) {
      storage_[keys[i]] = values[i];
    }
    return 0;
  }

  const std::vector<int>& request_sizes() const { return request_sizes_; }

  std::unordered_map<uint64_t, std::vector<float>> storage_;

private:
  int next_rpc_id_ = 1;
  std::vector<int> request_sizes_;
  std::vector<std::vector<char>> buffers_;
};

std::vector<uint64_t> SelectKeysForShard(int shard, int count, int num_shards) {
  std::vector<uint64_t> keys;
  for (uint64_t key = 1; static_cast<int>(keys.size()) < count; ++key) {
    if (static_cast<int>(GetHash(key) % static_cast<uint64_t>(num_shards)) ==
        shard) {
      keys.push_back(key);
    }
  }
  return keys;
}

TEST(AllShardsClientTest, SplitsLargeRequestsWithoutDroppingKeys) {
  FLAGS_value_size             = 16;
  FLAGS_max_kv_num_per_request = 2;

  FakeShardClient shard0(0);
  FakeShardClient shard1(1);

  auto shard0_keys = SelectKeysForShard(0, 5, 2);
  auto shard1_keys = SelectKeysForShard(1, 2, 2);

  for (uint64_t key : shard0_keys) {
    shard0.storage_[key] = {static_cast<float>(key), 0, 0, 0};
  }
  for (uint64_t key : shard1_keys) {
    shard1.storage_[key] = {static_cast<float>(key), 0, 0, 0};
  }

  std::vector<uint64_t> keys = shard0_keys;
  keys.insert(keys.end(), shard1_keys.begin(), shard1_keys.end());

  std::vector<BaseParameterClient*> clients = {&shard0, &shard1};
  AllShardsParameterClientWrapper wrapper(clients, 2);
  wrapper.InitThread();

  std::vector<float> output(keys.size() * 4 + 1, 0.0f);
  int rpc_id = wrapper.GetParameter(
      base::ConstArray<uint64_t>(keys), output.data(), false, 0);
  wrapper.WaitRPCFinish(rpc_id);

  EXPECT_EQ(shard0.request_sizes(), std::vector<int>({2, 2, 1}));
  EXPECT_EQ(shard1.request_sizes(), std::vector<int>({2}));

  for (std::size_t i = 0; i < keys.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i * 4], static_cast<float>(keys[i]));
  }

  wrapper.RevokeRPCResource(rpc_id);
}

TEST(AllShardsClientTest, RoutesPutByShard) {
  FLAGS_value_size             = 16;
  FLAGS_max_kv_num_per_request = 4;

  FakeShardClient shard0(0);
  FakeShardClient shard1(1);
  std::vector<BaseParameterClient*> clients = {&shard0, &shard1};
  AllShardsParameterClientWrapper wrapper(clients, 2);

  auto shard0_keys           = SelectKeysForShard(0, 2, 2);
  auto shard1_keys           = SelectKeysForShard(1, 2, 2);
  std::vector<uint64_t> keys = {
      shard1_keys[0],
      shard0_keys[0],
      shard1_keys[1],
      shard0_keys[1],
  };
  std::vector<std::vector<float>> values = {
      {1, 1, 1, 1},
      {2, 2, 2, 2},
      {3, 3, 3, 3},
      {4, 4, 4, 4},
  };

  ASSERT_EQ(wrapper.PutParameter(keys, values), 0);
  EXPECT_EQ(shard0.storage_.size(), 2u);
  EXPECT_EQ(shard1.storage_.size(), 2u);
  EXPECT_FLOAT_EQ(shard0.storage_[shard0_keys[0]][0], 2.0f);
  EXPECT_FLOAT_EQ(shard1.storage_[shard1_keys[0]][0], 1.0f);
}

} // namespace
