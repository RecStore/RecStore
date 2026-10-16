#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/tensor.h"
#include "base/json.h"
#include "benchmark/ps/rdma_rc_transport_benchmark_values.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/rdma_protocol.h"

DECLARE_int32(value_size);
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(rdma_rc_client_id_base);
DECLARE_int32(rdma_rc_qps_per_client_per_shard);
DECLARE_int32(rdma_rc_slots_per_qp);
DECLARE_string(rdma_get_response_mode);
DECLARE_bool(rdma_lane_rebuild_enabled);

namespace {

base::RecTensor
MakeValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  base::RecTensor values({static_cast<int64_t>(keys.size()), embedding_dim},
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

void ExpectFlatSlots(
    const float* buffer, const base::RecTensor& expected, int embedding_dim) {
  const float* src   = expected.data_as<float>();
  const int64_t rows = expected.shape(0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      EXPECT_FLOAT_EQ(
          buffer[row * embedding_dim + col], src[row * embedding_dim + col]);
    }
  }
}

base::RecTensor
MakeHashedValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  base::RecTensor values({static_cast<int64_t>(keys.size()), embedding_dim},
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

recstore::ResolvedRdmaFabric TestFabric() {
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  if (config_path == nullptr || *config_path == '\0') {
    throw std::runtime_error(
        "RECSTORE_CONFIG is required for RDMA integration");
  }
  const auto deployment =
      recstore::ParseResolvedRdmaDeploymentConfig(ParseFile2Json(config_path));
  return recstore::LocalRdmaFabric(deployment, FLAGS_global_id);
}

petps::PetPSClient& SingleShardClient() {
  static auto* client = []() {
    const auto fabric = TestFabric();
    const int logical_client_id =
        FLAGS_rdma_rc_client_id_base >= 0
            ? FLAGS_rdma_rc_client_id_base
            : fabric.logical_id;
    auto* created =
        new petps::PetPSClient("127.0.0.1", 1234, 0, logical_client_id);
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
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  json config = ParseFile2Json(config_path);
  ASSERT_TRUE(config.contains("distributed_client"));
  recstore::RDMAPSClientAdapter adapter(config);
  ASSERT_EQ(adapter.InitEmbeddingTable(
                "split_get",
                recstore::EmbeddingTableConfig{
                    10000000, static_cast<uint64_t>(embedding_dim)}),
            0);

  std::vector<std::uint64_t> keys;
  keys.reserve(1000);
  for (std::uint64_t key = 0; key < 1000; ++key) {
    keys.push_back(5000000ULL + key);
  }
  auto values = MakeValues(keys, embedding_dim);

  ASSERT_EQ(adapter.PutParameter(base::ConstArray<std::uint64_t>(keys), values),
            0);

  std::vector<float> output(
      keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
  base::RecTensor output_t(
      output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
  ASSERT_EQ(
      adapter.GetParameter(base::ConstArray<std::uint64_t>(keys), output_t), 0);

  ExpectFlatSlots(output.data(), values, embedding_dim);
}

TEST(PetPSIntegrationTest, AdapterFlatUpdateRoundTripMultiShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  json config = ParseFile2Json(config_path);
  recstore::RDMAPSClientAdapter adapter(config);
  ASSERT_EQ(adapter.InitEmbeddingTable(
                "flat_update",
                recstore::EmbeddingTableConfig{
                    10000000, static_cast<uint64_t>(embedding_dim)}),
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
      grads.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
  const uint64_t update_id = adapter.SubmitUpdateParameterAsync(
      "flat_update", base::ConstArray<std::uint64_t>(keys), grads_t);
  ASSERT_GT(update_id, 0);
  ASSERT_EQ(adapter.WaitUpdateParameter(update_id), 0);
  EXPECT_THROW(adapter.WaitUpdateParameter(update_id), std::runtime_error);

  std::vector<float> output(grads.size(), 0.0f);
  base::RecTensor output_t(
      output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
  ASSERT_EQ(
      adapter.GetParameter(base::ConstArray<std::uint64_t>(keys), output_t), 0);
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
  const auto fabric = TestFabric();
  const int logical_client_id =
      FLAGS_rdma_rc_client_id_base >= 0
          ? FLAGS_rdma_rc_client_id_base
          : fabric.logical_id;

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
  config["rdma_deployment"] = {
      {"deployment_id", "rdma-multishard-inline-test"},
      {"epoch", 1},
      {"protocol_version", 1},
      {"num_clients", FLAGS_num_client_processes},
      {"nodes", json::array()},
  };
  for (int shard = 0; shard < 2; ++shard) {
    config["rdma_deployment"]["nodes"].push_back(json{
        {"node_id", shard},
        {"role", "server"},
        {"device", "mlx5_0"},
        {"port", 1},
        {"gid_index", 0},
        {"mode", "ib"}});
  }
  for (int client = 0; client < FLAGS_num_client_processes; ++client) {
    config["rdma_deployment"]["nodes"].push_back(json{
        {"node_id", 2 + client},
        {"role", "client"},
        {"device", "mlx5_0"},
        {"port", 1},
        {"gid_index", 0},
        {"mode", "ib"}});
  }
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
    ASSERT_EQ(
        adapter.GetParameter(base::ConstArray<std::uint64_t>(keys), output_t),
        0)
        << "round=" << round;

    ExpectFlatSlots(output.data(), values, embedding_dim);
  }
}

// Lane/QP rebuild: with --rdma_lane_rebuild_enabled on both ends, forcing a
// rebuild of a healthy lane's QP must be acknowledged by the peer (the
// bilateral generation handshake) and the lane must keep serving afterwards.
// Skips unless the flag is set, so the default suites keep the terminal-dirty
// behavior.
TEST(PetPSIntegrationTest, LaneRebuildResumesServiceSingleShard) {
  if (!FLAGS_rdma_lane_rebuild_enabled) {
    GTEST_SKIP() << "requires --rdma_lane_rebuild_enabled on server and client";
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {7600001, 7600002, 7600003};
  auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  const int lanes = FLAGS_rdma_rc_qps_per_client_per_shard;
  ASSERT_GT(lanes, 0);

  const auto expect_get_ok = [&]() {
    void* recv_buffer =
        client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
    const int rpc_id = client.GetParameter(
        base::ConstArray<std::uint64_t>(keys),
        static_cast<float*>(recv_buffer),
        false);
    client.WaitRPCFinish(rpc_id);
    ExpectFlatSlots(static_cast<float*>(recv_buffer), values, embedding_dim);
    client.RevokeRPCResource(rpc_id);
  };

  ASSERT_EQ(client.RebuildAllLanesForTest(), lanes)
      << "the server did not acknowledge every lane rebuild";
  expect_get_ok();

  // A second round proves the generations keep advancing in lockstep.
  ASSERT_EQ(client.RebuildAllLanesForTest(), lanes);
  expect_get_ok();
}

// Client fast-fail scenario. Needs an external parent that freezes shard 1
// after the warm-up marker and unfreezes it once every lane is dirty, so it
// skips unless RECSTORE_RDMA_FAST_FAIL_TEST is set.
namespace {
constexpr char kFastFailWarmMarker[]  = "RECSTORE_FAST_FAIL_WARM_READY";
constexpr char kFastFailDirtyMarker[] = "RECSTORE_FAST_FAIL_ALL_LANES_DIRTY";
// An immediately failed request only loads an atomic and throws; anything close
// to the probe delay means the lane was not actually retired.
constexpr std::int64_t kInstantFailUs = 100000;
} // namespace

TEST(PetPSIntegrationTest, ShardUnreachableRetiresLanesAndFailsFast) {
  if (std::getenv("RECSTORE_RDMA_FAST_FAIL_TEST") == nullptr) {
    GTEST_SKIP()
        << "requires RECSTORE_RDMA_FAST_FAIL_TEST and a driving parent";
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);
  const auto fabric = TestFabric();
  const int logical_client_id =
      FLAGS_rdma_rc_client_id_base >= 0
          ? FLAGS_rdma_rc_client_id_base
          : fabric.logical_id;
  petps::PetPSClient healthy("127.0.0.1", 1234, 0, logical_client_id);
  petps::PetPSClient frozen("127.0.0.1", 1234, 1, logical_client_id);
  healthy.InitThread();
  frozen.InitThread();

  const std::vector<std::uint64_t> healthy_keys = {8000001, 8000002, 8000003};
  const std::vector<std::uint64_t> frozen_keys  = {9000001, 9000002, 9000003};
  const auto healthy_values = MakeValues(healthy_keys, embedding_dim);
  const auto frozen_values  = MakeValues(frozen_keys, embedding_dim);
  ASSERT_EQ(healthy.PutParameter(healthy_keys, healthy_values), 0);
  ASSERT_EQ(frozen.PutParameter(frozen_keys, frozen_values), 0);

  const auto get_matches = [&](petps::PetPSClient& client,
                               const std::vector<std::uint64_t>& keys,
                               const base::RecTensor& expected) {
    std::vector<float> output(
        keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
    base::RecTensor output_tensor(
        output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
    if (client.GetParameter(
            base::ConstArray<std::uint64_t>(keys), output_tensor) != 0) {
      return false;
    }
    return std::equal(output.begin(), output.end(), expected.data_as<float>());
  };

  // The parent freezes shard 1 once it sees this marker.
  std::printf("%s\n", kFastFailWarmMarker);
  std::fflush(stdout);

  // Phase 1: the next request has to die through the slow path with the
  // dedicated error instead of sitting on the full L3 wait budget.
  const auto phase1_start = std::chrono::steady_clock::now();
  bool saw_unreachable    = false;
  std::string first_error;
  while (!saw_unreachable && std::chrono::steady_clock::now() - phase1_start <
                                 std::chrono::seconds(60)) {
    try {
      EXPECT_TRUE(get_matches(frozen, frozen_keys, frozen_values));
    } catch (const petps::ShardUnreachableError& error) {
      saw_unreachable = true;
      first_error     = error.what();
    } catch (const std::exception& error) {
      FAIL() << "expected the slow-path verdict, got: " << error.what();
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_TRUE(saw_unreachable) << "shard 1 never produced a verdict";
  EXPECT_NE(first_error.find("unreachable"), std::string::npos) << first_error;

  // Phase 2: retire the remaining lanes. Attempts that still own a live lane
  // fail through the slow path; once every lane is dirty the request must fail
  // immediately, without a probe and without waiting.
  const int lanes = FLAGS_rdma_rc_qps_per_client_per_shard;
  ASSERT_GT(lanes, 1);
  std::int64_t instant_us = -1;
  int attempts            = 1;
  for (; attempts <= lanes + 4 && instant_us < 0; ++attempts) {
    const auto attempt_start = std::chrono::steady_clock::now();
    EXPECT_THROW(get_matches(frozen, frozen_keys, frozen_values),
                 petps::ShardUnreachableError);
    const std::int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - attempt_start)
            .count();
    if (elapsed_us < kInstantFailUs) {
      instant_us = elapsed_us;
    }
    // Isolation: the healthy shard must keep serving while its peer retires.
    EXPECT_TRUE(get_matches(healthy, healthy_keys, healthy_values));
  }
  EXPECT_GE(instant_us, 0) << "no instant failure after " << attempts - 1
                           << " attempts (lanes=" << lanes << ")";
  EXPECT_LE(attempts, lanes + 1)
      << "retired more lanes than the pool has: " << attempts - 1;

  // The parent brings shard 1 back once every lane is dirty.
  std::printf("%s\n", kFastFailDirtyMarker);
  std::fflush(stdout);

  // Phase 3: the control plane reports the shard serving again, but with lane
  // rebuild disabled the retired lanes must keep failing fast instead of
  // re-using slots the server may still read.
  std::this_thread::sleep_for(std::chrono::seconds(8));
  const auto post_start = std::chrono::steady_clock::now();
  EXPECT_THROW(get_matches(frozen, frozen_keys, frozen_values),
               petps::ShardUnreachableError);
  const std::int64_t post_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - post_start)
          .count();
  EXPECT_LT(post_us, kInstantFailUs)
      << "a retired lane was re-armed without a rebuild";
  EXPECT_TRUE(get_matches(healthy, healthy_keys, healthy_values));
}

// End-to-end lane rebuild recovery: an external parent SIGSTOPs shard 1 after
// the warm marker and SIGCONTs it later; with lane rebuild enabled the client
// must retire the frozen lanes, then re-arm and serve them again once the shard
// resumes (the bilateral generation handshake rebuilds both QPs).
namespace {
constexpr char kRebuildWarmMarker[] = "RECSTORE_REBUILD_WARM_READY";
} // namespace

TEST(PetPSIntegrationTest, LaneRebuildRecoversAfterFreeze) {
  if (!FLAGS_rdma_lane_rebuild_enabled) {
    GTEST_SKIP() << "requires --rdma_lane_rebuild_enabled on server and client";
  }
  if (std::getenv("RECSTORE_RDMA_REBUILD_RECOVERY_TEST") == nullptr) {
    GTEST_SKIP()
        << "requires RECSTORE_RDMA_REBUILD_RECOVERY_TEST and a driving "
           "parent";
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);
  const auto fabric = TestFabric();
  const int logical_client_id =
      FLAGS_rdma_rc_client_id_base >= 0
          ? FLAGS_rdma_rc_client_id_base
          : fabric.logical_id;
  petps::PetPSClient client("127.0.0.1", 1234, 1, logical_client_id);
  client.InitThread();
  // Also pair with shard 0: a server transport connects to every client that
  // uses its shard, and shard 0 also hosts the control plane whose metadata
  // exchange this test depends on.
  petps::PetPSClient healthy("127.0.0.1", 1234, 0, logical_client_id);
  healthy.InitThread();

  const std::vector<std::uint64_t> keys = {7700001, 7700002, 7700003};
  const auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);
  const std::vector<std::uint64_t> healthy_keys = {7800001, 7800002, 7800003};
  ASSERT_EQ(healthy.PutParameter(
                healthy_keys, MakeValues(healthy_keys, embedding_dim)),
            0);

  const auto get_matches = [&]() {
    std::vector<float> output(
        keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
    base::RecTensor output_tensor(
        output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
    if (client.GetParameter(
            base::ConstArray<std::uint64_t>(keys), output_tensor) != 0) {
      return false;
    }
    return std::equal(output.begin(), output.end(), values.data_as<float>());
  };
  ASSERT_TRUE(get_matches());

  // The parent freezes shard 1 once it sees this marker, then unfreezes it.
  std::printf("%s\n", kRebuildWarmMarker);
  std::fflush(stdout);

  const auto start = std::chrono::steady_clock::now();
  bool saw_failure = false;
  bool recovered   = false;
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(120)) {
    try {
      if (get_matches() && saw_failure) {
        recovered = true;
        break;
      }
    } catch (const std::exception&) {
      saw_failure = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  EXPECT_TRUE(saw_failure) << "the freeze never produced a failure";
  EXPECT_TRUE(recovered) << "the lane never recovered after the freeze lifted";
}

// Device-layer scenario: an external parent SIGKILLs shard 1 so its QP is
// destroyed and verbs retry exhaustion surfaces as a CQ completion error on the
// client. The client must classify that error, retire the lane and fail fast
// without the health slow path (the driver disables it with
// --rdma-health-probe-after-ms=0) and far below the wait-timeout budget.
namespace {
constexpr char kKilledShardWarmMarker[] = "RECSTORE_KILLED_SHARD_WARM_READY";
constexpr char kKilledShardDirtyMarker[] =
    "RECSTORE_KILLED_SHARD_ALL_LANES_DIRTY";
// A request posted against a destroyed QP fails when verbs exhausts its retry
// budget (measured ~4.2s on this mlx5 loopback with timeout=14 x retry_cnt=7),
// and the client harvests the lane CQ every 50ms; the budget below has to be
// far under the 60s L3 default but well above the verbs window.
constexpr std::int64_t kCompletionErrorFailBudgetUs = 8000000;
} // namespace

TEST(PetPSIntegrationTest, KilledShardRetiresLanesThroughCompletionError) {
  if (std::getenv("RECSTORE_RDMA_DEVICE_FAIL_TEST") == nullptr) {
    GTEST_SKIP() << "requires RECSTORE_RDMA_DEVICE_FAIL_TEST and a driving "
                    "parent";
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  const int client_id     = FLAGS_global_id - FLAGS_num_server_processes;
  ASSERT_GE(client_id, 0);
  const auto fabric = TestFabric();
  const int logical_client_id =
      FLAGS_rdma_rc_client_id_base >= 0
          ? FLAGS_rdma_rc_client_id_base
          : fabric.logical_id;
  petps::PetPSClient healthy("127.0.0.1", 1234, 0, logical_client_id);
  petps::PetPSClient killed("127.0.0.1", 1234, 1, logical_client_id);
  healthy.InitThread();
  killed.InitThread();

  const std::vector<std::uint64_t> healthy_keys = {8100001, 8100002, 8100003};
  const std::vector<std::uint64_t> killed_keys  = {9100001, 9100002, 9100003};
  const auto healthy_values = MakeValues(healthy_keys, embedding_dim);
  const auto killed_values  = MakeValues(killed_keys, embedding_dim);
  ASSERT_EQ(healthy.PutParameter(healthy_keys, healthy_values), 0);
  ASSERT_EQ(killed.PutParameter(killed_keys, killed_values), 0);

  const auto get_matches = [&](petps::PetPSClient& client,
                               const std::vector<std::uint64_t>& keys,
                               const base::RecTensor& expected) {
    std::vector<float> output(
        keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
    base::RecTensor output_tensor(
        output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
    if (client.GetParameter(
            base::ConstArray<std::uint64_t>(keys), output_tensor) != 0) {
      return false;
    }
    return std::equal(output.begin(), output.end(), expected.data_as<float>());
  };

  // The parent SIGKILLs shard 1 once it sees this marker. Wait for the kill to
  // land first: a request whose Writes were already ACKed when the peer died
  // gets no completion error at all (the QP is idle and verbs has no signal),
  // which is the health slow path's job. The device layer is exercised by the
  // next request, whose WRITEs hit the destroyed QP and retry out.
  std::printf("%s\n", kKilledShardWarmMarker);
  std::fflush(stdout);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  // Phase 1: the next request has to die on the classified completion error in
  // roughly the verbs retry window, not on the L3 wait budget.
  const auto phase1_start = std::chrono::steady_clock::now();
  bool saw_unreachable    = false;
  std::string first_error;
  std::int64_t first_fail_us = -1;
  while (!saw_unreachable && std::chrono::steady_clock::now() - phase1_start <
                                 std::chrono::seconds(60)) {
    const auto attempt_start = std::chrono::steady_clock::now();
    try {
      EXPECT_TRUE(get_matches(killed, killed_keys, killed_values));
    } catch (const petps::ShardUnreachableError& error) {
      saw_unreachable = true;
      first_error     = error.what();
    } catch (const std::exception& error) {
      FAIL() << "expected the device-layer verdict, got: " << error.what();
      return;
    }
    if (saw_unreachable) {
      first_fail_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - attempt_start)
              .count();
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_TRUE(saw_unreachable) << "shard 1 never produced a verdict";
  EXPECT_NE(first_error.find("unreachable"), std::string::npos) << first_error;
  EXPECT_NE(first_error.find("completion failed"), std::string::npos)
      << first_error;
  EXPECT_LT(first_fail_us, kCompletionErrorFailBudgetUs)
      << "the completion error was not classified fast enough: "
      << first_fail_us << "us; " << first_error;

  // Phase 2: retire the remaining lanes the same way; once every lane is dirty
  // the request must fail immediately, and shard 0 must stay unaffected.
  const int lanes = FLAGS_rdma_rc_qps_per_client_per_shard;
  ASSERT_GT(lanes, 1);
  std::int64_t instant_us = -1;
  int attempts            = 1;
  for (; attempts <= lanes + 4 && instant_us < 0; ++attempts) {
    const auto attempt_start = std::chrono::steady_clock::now();
    EXPECT_THROW(get_matches(killed, killed_keys, killed_values),
                 petps::ShardUnreachableError);
    const std::int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - attempt_start)
            .count();
    if (elapsed_us < kInstantFailUs) {
      instant_us = elapsed_us;
    }
    EXPECT_TRUE(get_matches(healthy, healthy_keys, healthy_values));
  }
  EXPECT_GE(instant_us, 0) << "no instant failure after " << attempts - 1
                           << " attempts (lanes=" << lanes << ")";
  EXPECT_LE(attempts, lanes + 1)
      << "retired more lanes than the pool has: " << attempts - 1;

  // The parent stops the run once this marker appears.
  std::printf("%s\n", kKilledShardDirtyMarker);
  std::fflush(stdout);
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
