#include <folly/init/Init.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "base/array.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

DECLARE_int32(value_size);

namespace {

std::vector<std::vector<float>>
MakeValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row;
    row.reserve(embedding_dim);
    for (int d = 0; d < embedding_dim; ++d) {
      row.push_back(static_cast<float>(key * 10 + d));
    }
    values.push_back(std::move(row));
  }
  return values;
}

void ExpectFlatSlots(const float* buffer,
                     const std::vector<std::vector<float>>& expected,
                     int embedding_dim) {
  for (std::size_t row = 0; row < expected.size(); ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      EXPECT_FLOAT_EQ(buffer[row * embedding_dim + col], expected[row][col]);
    }
  }
}

petps::PetPSClient& SingleShardClient() {
  static auto* client = []() {
    auto* created = new petps::PetPSClient("127.0.0.1", 1234, 0);
    created->InitThread();
    return created;
  }();
  return *client;
}

} // namespace

TEST(PetPSIntegrationTest, PutGetRoundTripSingleShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {101, 102, 103};
  auto values                     = MakeValues(keys, embedding_dim);

  ASSERT_EQ(client.PutParameter(keys, values), 0);

  void* recv_buffer =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys),
      static_cast<float*>(recv_buffer),
      false);
  client.WaitRPCFinish(rpc_id);

  ExpectFlatSlots(static_cast<float*>(recv_buffer), values, embedding_dim);
  client.RevokeRPCResource(rpc_id);
}

TEST(PetPSIntegrationTest, MissingKeysReturnZeroSlots) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {999001, 999002};
  void* recv_buffer =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys),
      static_cast<float*>(recv_buffer),
      false);
  client.WaitRPCFinish(rpc_id);

  const float* values = static_cast<float*>(recv_buffer);
  for (std::size_t i = 0; i < keys.size() * embedding_dim; ++i) {
    EXPECT_FLOAT_EQ(values[i], 0.0f);
  }
  client.RevokeRPCResource(rpc_id);
}

TEST(PetPSIntegrationTest, PutGetRoundTripMultiShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);

  auto shard0 = std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, 0);
  auto shard1 = std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, 1);

  shard0->InitThread();
  shard1->InitThread();

  std::vector<BaseParameterClient*> clients = {shard0.get(), shard1.get()};
  AllShardsParameterClientWrapper wrapper(clients, 2);

  std::vector<std::uint64_t> keys = {1, 2, 3, 4, 5, 6};
  auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(wrapper.PutParameter(keys, values), 0);

  std::vector<float> output(keys.size() * embedding_dim + 1, 0.0f);
  int rpc_id = wrapper.GetParameter(
      base::ConstArray<std::uint64_t>(keys), output.data(), false, 0);
  wrapper.WaitRPCFinish(rpc_id);

  ExpectFlatSlots(output.data(), values, embedding_dim);
  wrapper.RevokeRPCResource(rpc_id);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
