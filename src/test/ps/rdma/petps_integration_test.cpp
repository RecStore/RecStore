#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <vector>

#include "base/array.h"
#include "base/tensor.h"
#include "benchmark/ps/rdma_rc_transport_benchmark_values.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/rdma_protocol.h"

DECLARE_int32(value_size);
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(rdma_rc_qps_per_client_per_shard);
DECLARE_int32(rdma_rc_slots_per_qp);
DECLARE_string(rdma_get_response_mode);

namespace {

base::RecTensor
MakeValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  base::RecTensor values(
      {static_cast<int64_t>(keys.size()), embedding_dim},
      base::DataType::FLOAT32);
  float* dst = values.data_as<float>();
  for (std::size_t row = 0; row < keys.size(); ++row) {
    for (int d = 0; d < embedding_dim; ++d) {
      dst[row * static_cast<std::size_t>(embedding_dim) + d] =
          static_cast<float>(keys[row] * 10 + d);
    }
  }
  return values;
}

void ExpectFlatSlots(const float* buffer,
                     const base::RecTensor& expected,
                     int embedding_dim) {
  const float* src = expected.data_as<float>();
  const int64_t rows = expected.shape(0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      EXPECT_FLOAT_EQ(buffer[row * embedding_dim + col],
                      src[row * embedding_dim + col]);
    }
  }
}

base::RecTensor
MakeHashedValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  base::RecTensor values(
      {static_cast<int64_t>(keys.size()), embedding_dim},
      base::DataType::FLOAT32);
  float* dst = values.data_as<float>();
  for (std::size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      dst[row * static_cast<std::size_t>(embedding_dim) + col] =
          recstore::benchmark::MakeHashedValue(keys[row], col);
    }
  }
  return values;
}

void ExpectHashedFlatSlots(const float* buffer,
                           const std::vector<std::uint64_t>& keys,
                           int embedding_dim) {
  for (std::size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      const float expected =
          recstore::benchmark::MakeHashedValue(keys[row], col);
      const float actual = buffer[row * embedding_dim + col];
      EXPECT_EQ(recstore::benchmark::FloatBits(actual),
                recstore::benchmark::FloatBits(expected))
          << "row=" << row << " key=" << keys[row] << " col=" << col;
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

TEST(PetPSIntegrationTest, UpdateGetRoundTripSingleShard) {
  auto& client = SingleShardClient();

  std::vector<std::uint64_t> keys = {401, 402};
  std::vector<float> initial_flat = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> grads_flat = {
      0.5f, 1.0f, 1.5f, 2.0f, 1.0f, 0.5f, 2.0f, 1.5f};
  const std::vector<float> expected = {
      0.995f, 1.99f, 2.985f, 3.98f, 4.99f, 5.995f, 6.98f, 7.985f};
  base::RecTensor initial(initial_flat.data(), {2, 4});
  base::RecTensor grads(grads_flat.data(), {2, 4});

  ASSERT_EQ(client.InitEmbeddingTable("table_update", 128, 4), 0);
  ASSERT_EQ(client.PutParameter(keys, initial), 0);
  ASSERT_EQ(client.UpdateParameter(
                "table_update", base::ConstArray<std::uint64_t>(keys), grads),
            0);

  base::RecTensor actual({2, 4}, base::DataType::FLOAT32);
  ASSERT_EQ(client.GetParameter(base::ConstArray<std::uint64_t>(keys), actual),
            0);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual.data_as<float>()[i], expected[i]);
  }
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

TEST(PetPSIntegrationTest, HashedValueBatchGetTransferSingleShard) {
  ASSERT_EQ(FLAGS_value_size % static_cast<int>(sizeof(float)), 0);
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys;
  keys.reserve(500);
  for (std::uint64_t i = 0; i < 500; ++i) {
    keys.push_back(4000001ULL + i);
  }
  auto values = MakeHashedValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  const std::string previous_response_mode = FLAGS_rdma_get_response_mode;
  FLAGS_rdma_get_response_mode             = "staging_copy";
  std::vector<float> output(
      keys.size() * static_cast<std::size_t>(embedding_dim) + 1, 0.0f);
  int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys), output.data(), false);
  client.WaitRPCFinish(rpc_id);

  ExpectHashedFlatSlots(output.data(), keys, embedding_dim);
  const auto* status = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(output.data()) +
      keys.size() * static_cast<std::size_t>(FLAGS_value_size));
  EXPECT_EQ(*status, static_cast<std::int32_t>(petps::RpcStatus::kOk));
  client.RevokeRPCResource(rpc_id);
  FLAGS_rdma_get_response_mode = previous_response_mode;
}

TEST(PetPSIntegrationTest, AdapterSplitGetRoundTripMultiShard) {
  const int embedding_dim       = FLAGS_value_size / sizeof(float);
  json config                   = json::object();
  config["cache_ps"]["ps_type"] = "RDMA";
  config["cache_ps"]["base_kv_config"]["value"]["default_value_size_hint"] =
      FLAGS_value_size;
  config["client"] = json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}};
  config["distributed_client"] = {
      {"num_shards", 2},
      {"hash_method", "simple_mod"},
      {"max_keys_per_request", 2},
      {"servers",
       json::array(
           {json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}},
            json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 1}}})},
  };
  recstore::RDMAPSClientAdapter adapter(config);

  std::vector<std::uint64_t> keys;
  keys.reserve(10);
  for (std::uint64_t key = 0; key < 10; ++key) {
    keys.push_back(5000000ULL + key);
  }
  auto values = MakeValues(keys, embedding_dim);

  ASSERT_EQ(adapter.PutParameter(base::ConstArray<std::uint64_t>(keys), values),
            0);

  std::vector<float> output(
      keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
  base::RecTensor output_t(
      output.data(),
      {static_cast<int64_t>(keys.size()), embedding_dim});
  ASSERT_EQ(adapter.GetParameter(
                base::ConstArray<std::uint64_t>(keys), output_t),
            0);

  ExpectFlatSlots(output.data(), values, embedding_dim);
}

TEST(PetPSIntegrationTest, AdapterFlatUpdateRoundTripMultiShard) {
  const int embedding_dim       = FLAGS_value_size / sizeof(float);
  json config                   = json::object();
  config["cache_ps"]["ps_type"] = "RDMA";
  config["cache_ps"]["base_kv_config"]["value"]["default_value_size_hint"] =
      FLAGS_value_size;
  config["client"] = json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}};
  config["distributed_client"] = {
      {"num_shards", 2},
      {"hash_method", "simple_mod"},
      {"max_keys_per_request", 2},
      {"servers",
       json::array(
           {json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}},
            json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 1}}})},
  };
  recstore::RDMAPSClientAdapter adapter(config);
  ASSERT_EQ(adapter.InitEmbeddingTable(
                "flat_update",
                recstore::EmbeddingTableConfig{10000000,
                                               static_cast<uint64_t>(embedding_dim)}),
            0);

  std::vector<std::uint64_t> keys;
  std::vector<float> grads;
  keys.reserve(10);
  grads.reserve(10 * static_cast<std::size_t>(embedding_dim));
  for (std::uint64_t row = 0; row < 10; ++row) {
    keys.push_back(6000000ULL + row);
    for (int col = 0; col < embedding_dim; ++col) {
      grads.push_back(static_cast<float>(row * embedding_dim + col + 1));
    }
  }

  base::RecTensor grads_t(
      grads.data(),
      {static_cast<int64_t>(keys.size()), embedding_dim});
  const uint64_t update_id = adapter.SubmitUpdateParameterAsync(
      "flat_update", base::ConstArray<std::uint64_t>(keys), grads_t);
  ASSERT_GT(update_id, 0);
  ASSERT_EQ(adapter.WaitUpdateParameter(update_id), 0);
  EXPECT_THROW(adapter.WaitUpdateParameter(update_id), std::runtime_error);

  std::vector<float> output(grads.size(), 0.0f);
  base::RecTensor output_t(
      output.data(),
      {static_cast<int64_t>(keys.size()), embedding_dim});
  ASSERT_EQ(adapter.GetParameter(
                base::ConstArray<std::uint64_t>(keys), output_t),
            0);
  for (std::size_t index = 0; index < grads.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], -0.01f * grads[index]);
  }
}

TEST(PetPSIntegrationTest, ExhaustedQpPoolFailsLoudly) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {701, 702};
  auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  const int slot_count =
      FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp;
  ASSERT_GT(slot_count, 0);

  std::vector<void*> recv_buffers;
  std::vector<int> rpc_ids;
  recv_buffers.reserve(static_cast<std::size_t>(slot_count));
  rpc_ids.reserve(static_cast<std::size_t>(slot_count));

  for (int i = 0; i < slot_count; ++i) {
    void* recv_buffer =
        client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
    recv_buffers.push_back(recv_buffer);
    rpc_ids.push_back(client.GetParameter(
        base::ConstArray<std::uint64_t>(keys),
        static_cast<float*>(recv_buffer),
        true));
  }

  void* overflow_recv =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  EXPECT_THROW(client.GetParameter(base::ConstArray<std::uint64_t>(keys),
                                   static_cast<float*>(overflow_recv),
                                   true),
               std::runtime_error);

  for (int rpc_id : rpc_ids) {
    client.WaitRPCFinish(rpc_id);
    client.RevokeRPCResource(rpc_id);
  }
}

TEST(PetPSIntegrationTest, RepeatedPutGetStressSingleShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);

  for (int round = 0; round < 50; ++round) {
    std::vector<std::uint64_t> keys;
    keys.reserve(8);
    const std::uint64_t base =
        1000000ULL + static_cast<std::uint64_t>(client_id) * 100000ULL +
        static_cast<std::uint64_t>(round) * 100ULL;
    for (std::uint64_t i = 0; i < 8; ++i) {
      keys.push_back(base + i);
    }
    auto values = MakeValues(keys, embedding_dim);

    ASSERT_EQ(client.PutParameter(keys, values), 0) << "round=" << round;

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
}

TEST(PetPSIntegrationTest, RepeatedPutGetStressMultiShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);

  json config                   = json::object();
  config["cache_ps"]["ps_type"] = "RDMA";
  config["cache_ps"]["base_kv_config"]["value"]["default_value_size_hint"] =
      FLAGS_value_size;
  config["client"] = json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}};
  config["distributed_client"] = {
      {"num_shards", 2},
      {"hash_method", "simple_mod"},
      {"max_keys_per_request", 16},
      {"servers",
       json::array(
           {json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}},
            json{{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 1}}})},
  };
  recstore::RDMAPSClientAdapter adapter(config);

  for (int round = 0; round < 50; ++round) {
    std::vector<std::uint64_t> keys;
    keys.reserve(16);
    const std::uint64_t base =
        2000000ULL + static_cast<std::uint64_t>(client_id) * 100000ULL +
        static_cast<std::uint64_t>(round) * 100ULL;
    for (std::uint64_t i = 0; i < 16; ++i) {
      keys.push_back(base + i);
    }
    auto values = MakeValues(keys, embedding_dim);

    ASSERT_EQ(
        adapter.PutParameter(base::ConstArray<std::uint64_t>(keys), values), 0)
        << "round=" << round;

    std::vector<float> output(
        keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
    base::RecTensor output_t(
        output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
    ASSERT_EQ(adapter.GetParameter(base::ConstArray<std::uint64_t>(keys),
                                   output_t),
              0)
        << "round=" << round;

    ExpectFlatSlots(output.data(), values, embedding_dim);
  }
}

TEST(PetPSIntegrationTest, AsyncGetPrefetchStressSingleShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);

  const int prefetch_count = std::min(
      FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp, 8);
  ASSERT_GT(prefetch_count, 1);

  std::vector<std::vector<std::uint64_t>> request_keys;
  std::vector<base::RecTensor> expected_values;
  request_keys.reserve(static_cast<std::size_t>(prefetch_count));
  expected_values.reserve(static_cast<std::size_t>(prefetch_count));

  for (int request = 0; request < prefetch_count; ++request) {
    std::vector<std::uint64_t> keys;
    keys.reserve(4);
    const std::uint64_t base =
        3000000ULL + static_cast<std::uint64_t>(client_id) * 100000ULL +
        static_cast<std::uint64_t>(request) * 100ULL;
    for (std::uint64_t i = 0; i < 4; ++i) {
      keys.push_back(base + i);
    }
    auto values = MakeValues(keys, embedding_dim);
    ASSERT_EQ(client.PutParameter(keys, values), 0) << "request=" << request;
    request_keys.push_back(std::move(keys));
    expected_values.push_back(std::move(values));
  }

  std::vector<std::vector<float>> recv_buffers;
  std::vector<int> rpc_ids;
  recv_buffers.reserve(static_cast<std::size_t>(prefetch_count));
  rpc_ids.reserve(static_cast<std::size_t>(prefetch_count));

  for (int request = 0; request < prefetch_count; ++request) {
    recv_buffers.emplace_back(
        request_keys[static_cast<std::size_t>(request)].size() *
                static_cast<std::size_t>(embedding_dim) +
            1,
        0.0f);
    rpc_ids.push_back(client.GetParameter(
        base::ConstArray<std::uint64_t>(
            request_keys[static_cast<std::size_t>(request)]),
        recv_buffers.back().data(),
        true));
  }

  for (int request = 0; request < prefetch_count; ++request) {
    client.WaitRPCFinish(rpc_ids[request]);
    ExpectFlatSlots(recv_buffers[static_cast<std::size_t>(request)].data(),
                    expected_values[static_cast<std::size_t>(request)],
                    embedding_dim);
    const auto* status = reinterpret_cast<const std::int32_t*>(
        reinterpret_cast<const char*>(
            recv_buffers[static_cast<std::size_t>(request)].data()) +
        request_keys[static_cast<std::size_t>(request)].size() *
            static_cast<std::size_t>(FLAGS_value_size));
    EXPECT_EQ(*status, static_cast<std::int32_t>(petps::RpcStatus::kOk));
    client.RevokeRPCResource(rpc_ids[request]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
