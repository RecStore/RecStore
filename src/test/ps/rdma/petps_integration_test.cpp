#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "base/array.h"
#include "base/tensor.h"
#include "base/json.h"
#include "benchmark/ps/rdma_rc_transport_benchmark_values.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/shard_routing.h"

DECLARE_int32(value_size);
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(rdma_rc_num_logical_clients);
DECLARE_int32(rdma_rc_client_id_base);
DECLARE_int32(rdma_rc_qps_per_client_per_shard);
DECLARE_int32(rdma_rc_server_get_workers);
DECLARE_int32(rdma_rc_server_get_worker_hold_us);
DECLARE_int32(rdma_lane_rebuild_max_attempts);
DECLARE_int32(rdma_lane_rebuild_cooldown_ms);
DECLARE_int32(rdma_lane_rebuild_wait_ms);
DECLARE_int32(rdma_rc_slots_per_qp);
DECLARE_int32(rdma_wait_timeout_ms);
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

// The client id this process must use: an explicit override wins, otherwise the
// id the local fabric assigned to it. Always parses the config so a missing
// RECSTORE_CONFIG still fails loudly.
int ActiveLogicalClientId() {
  const auto fabric = TestFabric();
  return FLAGS_rdma_rc_client_id_base >= 0
           ? FLAGS_rdma_rc_client_id_base
           : fabric.logical_id;
}

// Reads `keys` back and compares them with `expected`; false when the RPC
// itself failed, so callers can retry a lane that is not serviceable yet.
bool GetMatches(petps::PetPSClient& client,
                const std::vector<std::uint64_t>& keys,
                const base::RecTensor& expected,
                int embedding_dim) {
  std::vector<float> output(
      keys.size() * static_cast<std::size_t>(embedding_dim), 0.0f);
  base::RecTensor output_tensor(
      output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
  if (client.GetParameter(
          base::ConstArray<std::uint64_t>(keys), output_tensor) != 0) {
    return false;
  }
  return std::equal(output.begin(), output.end(), expected.data_as<float>());
}

petps::PetPSClient& SingleShardClient() {
  static auto* client = []() {
    auto* created =
        new petps::PetPSClient("127.0.0.1", 1234, 0, ActiveLogicalClientId());
    created->InitThread();
    return created;
  }();
  return *client;
}

} // namespace

namespace petps {

// Test-only entry into PetPSClient's lane rebuild path (declared as a friend in
// petps_client.h, so the production API stays free of test seams). Forcing a
// rebuild of healthy lanes has no production caller; the integration test uses
// it to drive the bilateral handshake deterministically.
struct PetPSClientTestAccess {
  static int RebuildAllLanes(PetPSClient& client) {
    std::lock_guard<std::mutex> guard(client.mu_);
    int recovered = 0;
    for (std::size_t qp_index = 0; qp_index < client.qps_.size(); ++qp_index) {
      if (client.qps_[qp_index].dead.load(std::memory_order_relaxed)) {
        continue;
      }
      if (client.TryRebuildLaneLocked(static_cast<int>(qp_index))) {
        ++recovered;
      }
    }
    return recovered;
  }
};

} // namespace petps

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

  // The response must report success, not just carry the right bytes: a handler
  // failure completes the RPC with an error status and leaves the payload
  // empty.
  const auto* status_word =
      petps::FixedSlotStatusWord(recv_buffer, keys.size(), FLAGS_value_size);
  EXPECT_EQ(*status_word, static_cast<std::int32_t>(petps::RpcStatus::kOk));
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

TEST(PetPSIntegrationTest, SparseGradientCommitCommitSingleShard) {
  auto& client               = SingleShardClient();
  const std::string table    = "sparse_gradient_commit";
  std::vector<uint64_t> keys = {601, 602};
  std::vector<float> initial(8, 1.0f), gradients(8, 2.0f);
  base::RecTensor values(initial.data(), {2, 4});
  base::RecTensor grads(gradients.data(), {2, 4});
  ASSERT_EQ(client.InitEmbeddingTable(table, 1024, 4), 0);
  ASSERT_EQ(client.PutParameter(keys, values), 0);
  recstore::SparseGradientCommitHeader header;
  header.session = 7123;
  header.world   = 2;
  header.divisor = 2;
  header.dim     = 4;
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), -1);
  base::RecTensor actual({2, 4}, base::DataType::FLOAT32);
  ASSERT_EQ(client.GetParameter(keys, actual), 0);
  for (int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(actual.data_as<float>()[i], 1.0f);
  header.rank = 1;
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), 0);
  // Repeating a committed submission must not apply SGD again.
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), 0);
  header.kind = 1;
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), 0);
  ASSERT_EQ(client.GetParameter(keys, actual), 0);
  for (int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(actual.data_as<float>()[i], .98f);
  header.kind = 2;
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), 0);
  header.rank = 0;
  EXPECT_EQ(client.SparseGradientCommit(table, header, keys, grads), 0);
}

namespace {

constexpr int kOwnerTestDim      = 128;
constexpr uint64_t kOwnerTestTag = 5;

petps::PetPSClient& OwnerShardClient(int shard) {
  static const auto clients = [] {
    std::array<std::unique_ptr<petps::PetPSClient>, 2> result;
    for (int owner = 0; owner < 2; ++owner) {
      result[owner] = std::make_unique<petps::PetPSClient>(
          "127.0.0.1", 1234, owner, ActiveLogicalClientId());
      result[owner]->InitThread();
    }
    return result;
  }();
  return *clients.at(shard);
}

std::vector<uint64_t>
OwnerKeys(int shard,
          size_t count,
          uint64_t first = 100,
          uint64_t tag   = kOwnerTestTag) {
  std::vector<uint64_t> keys;
  for (uint64_t row = first; keys.size() < count; ++row) {
    const uint64_t encoded = (tag << 56) | row;
    if (recstore::shard_routing::PartitionKey(encoded, 2, "city_hash") == shard)
      keys.push_back(encoded);
  }
  return keys;
}

recstore::SparseGradientCommitHeader
OwnerHeader(uint64_t session, int rank, int step = 0, int kind = 0) {
  recstore::SparseGradientCommitHeader h;
  h.version = 4;
  h.session = session;
  h.rank    = rank;
  h.world   = 2;
  h.dim     = kOwnerTestDim;
  h.divisor = 1;
  h.step    = step;
  h.kind    = kind;
  return h;
}

std::string OwnerPacket(recstore::SparseGradientCommitHeader h,
                        const std::vector<uint64_t>& keys = {},
                        const std::vector<float>& values  = {}) {
  if (values.size() != keys.size() * kOwnerTestDim)
    throw std::invalid_argument("Owner test packet shape mismatch");
  h.rows = keys.size();
  return recstore::PackSparseGradientCommit(h, keys.data(), values.data());
}

std::vector<float> OwnerBitValues(size_t rows) {
  const uint32_t patterns[] = {
      0x00000000u,
      0x80000000u,
      0x00000001u,
      0x80000001u,
      0x00800000u,
      0x80800000u,
      0x7f7fffffu,
      0xff7fffffu,
      0x3f800001u,
      0xbf800001u};
  std::vector<float> values(rows * kOwnerTestDim);
  for (size_t i = 0; i < values.size(); ++i) {
    const auto bits = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
    std::memcpy(&values[i], &bits, sizeof(bits));
  }
  return values;
}

void OwnerInit(const std::string& table,
               const std::array<std::vector<uint64_t>, 2>& keys) {
  for (int shard = 0; shard < 2; ++shard) {
    auto& client = OwnerShardClient(shard);
    ASSERT_EQ(
        client.InitEmbeddingTable(table, 1024, kOwnerTestDim, kOwnerTestTag),
        static_cast<int>(kOwnerTestTag));
    std::vector<float> initial(keys[shard].size() * kOwnerTestDim, 1.25f);
    base::RecTensor values(
        initial.data(),
        {static_cast<int64_t>(keys[shard].size()), kOwnerTestDim});
    ASSERT_EQ(client.PutParameter(keys[shard], values), 0);
  }
}

void ExpectOwnerBits(int shard,
                     const std::vector<uint64_t>& keys,
                     const std::vector<float>& expected) {
  ASSERT_EQ(expected.size(), keys.size() * kOwnerTestDim);
  base::RecTensor actual({static_cast<int64_t>(keys.size()), kOwnerTestDim},
                         base::DataType::FLOAT32);
  ASSERT_EQ(OwnerShardClient(shard).GetParameter(keys, actual), 0);
  for (size_t i = 0; i < expected.size(); ++i)
    ASSERT_EQ(recstore::benchmark::FloatBits(actual.data_as<float>()[i]),
              recstore::benchmark::FloatBits(expected[i]))
        << "element=" << i;
}

void RequireOwnerDeployment() {
  ASSERT_EQ(FLAGS_num_server_processes, 2);
  ASSERT_EQ(FLAGS_value_size, kOwnerTestDim * sizeof(float));
  const auto deployment = recstore::ParseResolvedRdmaDeploymentConfig(
      ParseFile2Json(std::getenv("RECSTORE_CONFIG")));
  ASSERT_EQ(deployment.num_shards, 2);
  ASSERT_EQ(deployment.hash_method, "city_hash");
}

} // namespace

TEST(PetPSIntegrationTest, OwnerExactBitsAndEmptyOwnerMultiShard) {
  ASSERT_NO_FATAL_FAILURE(RequireOwnerDeployment());
  const std::string table = "owner_exact_bits";
  const std::array<std::vector<uint64_t>, 2> keys{
      OwnerKeys(0, 2), OwnerKeys(1, 2)};
  ASSERT_NO_FATAL_FAILURE(OwnerInit(table, keys));
  const auto candidates = OwnerBitValues(2);
  // Every physical shard waits for both ranks, including the empty non-owner.
  for (int shard = 0; shard < 2; ++shard) {
    auto& client     = OwnerShardClient(shard);
    const auto empty = OwnerPacket(OwnerHeader(21001, 1 - shard));
    EXPECT_EQ(client.ExchangeSparseGradientCommit(table, empty), -1);
    ExpectOwnerBits(
        shard, keys[shard], std::vector<float>(2 * kOwnerTestDim, 1.25f));
    const auto packet =
        OwnerPacket(OwnerHeader(21001, shard), keys[shard], candidates);
    ASSERT_EQ(client.ExchangeSparseGradientCommit(table, packet), 0);
    ASSERT_EQ(client.ExchangeSparseGradientCommit(table, packet), 0);
    ASSERT_EQ(client.ExchangeSparseGradientCommit(table, empty), 0);
    ExpectOwnerBits(shard, keys[shard], candidates);
  }
  // The next step has no rows on owner 0; its two empty markers still commit.
  for (int shard = 0; shard < 2; ++shard) {
    auto& client = OwnerShardClient(shard);
    EXPECT_EQ(client.ExchangeSparseGradientCommit(
                  table, OwnerPacket(OwnerHeader(21001, 1 - shard, 1))),
              0);
    const std::vector<uint64_t> selected =
        shard ? keys[shard] : std::vector<uint64_t>{};
    const std::vector<float> next(selected.size() * kOwnerTestDim, -2.5f);
    ASSERT_EQ(
        client.ExchangeSparseGradientCommit(
            table, OwnerPacket(OwnerHeader(21001, shard, 1), selected, next)),
        1);
    ExpectOwnerBits(shard, keys[shard], shard ? next : candidates);
    for (int rank = 0; rank < 2; ++rank)
      EXPECT_EQ(client.ExchangeSparseGradientCommit(
                    table, OwnerPacket(OwnerHeader(21001, rank, 1, 2))),
                1);
  }
}

TEST(PetPSIntegrationTest, OwnerWrongRankOrShardMultiShard) {
  ASSERT_NO_FATAL_FAILURE(RequireOwnerDeployment());
  const std::array<std::vector<uint64_t>, 2> keys{
      OwnerKeys(0, 1), OwnerKeys(1, 1)};
  for (int wrong_shard = 0; wrong_shard < 2; ++wrong_shard) {
    const std::string table = "owner_wrong_" + std::to_string(wrong_shard);
    ASSERT_NO_FATAL_FAILURE(OwnerInit(table, keys));
    auto& client           = OwnerShardClient(0);
    const uint64_t session = 21100 + wrong_shard;
    ASSERT_EQ(client.ExchangeSparseGradientCommit(
                  table, OwnerPacket(OwnerHeader(session, 1))),
              -1);
    // Send raw packets so the server, not the client's adapter validator,
    // rejects them.
    const auto h   = OwnerHeader(session, wrong_shard ? 0 : 1);
    const auto bad = OwnerPacket(h, keys[wrong_shard], OwnerBitValues(1));
    EXPECT_THROW(client.ExchangeSparseGradientCommit(table, bad),
                 std::runtime_error);
    EXPECT_THROW(client.ExchangeSparseGradientCommit(
                     table, OwnerPacket(OwnerHeader(session, 0, 0, 1))),
                 std::runtime_error);
    for (int shard = 0; shard < 2; ++shard)
      ExpectOwnerBits(
          shard, keys[shard], std::vector<float>(kOwnerTestDim, 1.25f));
  }
}

TEST(PetPSIntegrationTest, OwnerLateInvalidStickyMultiShard) {
  ASSERT_NO_FATAL_FAILURE(RequireOwnerDeployment());
  const std::array<std::vector<uint64_t>, 2> keys{
      OwnerKeys(0, 2), OwnerKeys(1, 1)};
  for (int nonfinite = 0; nonfinite < 2; ++nonfinite) {
    const std::string table = "owner_late_bad_" + std::to_string(nonfinite);
    ASSERT_NO_FATAL_FAILURE(OwnerInit(table, keys));
    auto& client           = OwnerShardClient(0);
    const uint64_t session = 21200 + nonfinite;
    ASSERT_EQ(client.ExchangeSparseGradientCommit(
                  table, OwnerPacket(OwnerHeader(session, 1))),
              -1);
    auto h           = OwnerHeader(session, 0);
    h.chunks         = 2;
    const auto first = OwnerPacket(h, {keys[0][0]}, OwnerBitValues(1));
    ASSERT_EQ(client.ExchangeSparseGradientCommit(table, first), -1);
    ASSERT_EQ(client.ExchangeSparseGradientCommit(table, first), -1);
    ExpectOwnerBits(0, keys[0], std::vector<float>(2 * kOwnerTestDim, 1.25f));
    h.chunk             = 1;
    const auto last_key = nonfinite ? keys[0][1] : OwnerKeys(0, 1, 1000000)[0];
    auto last           = OwnerPacket(h, {last_key}, OwnerBitValues(1));
    if (nonfinite) {
      const uint32_t nan = 0x7fc00001u;
      std::memcpy(last.data() + sizeof(h) + sizeof(uint64_t) +
                      (kOwnerTestDim - 1) * sizeof(float),
                  &nan,
                  sizeof(nan));
    }
    EXPECT_THROW(client.ExchangeSparseGradientCommit(table, last),
                 std::runtime_error);
    // Failed metadata/payload or apply precheck is sticky; no earlier row was
    // changed.
    EXPECT_THROW(client.ExchangeSparseGradientCommit(table, first),
                 std::runtime_error);
    EXPECT_THROW(client.ExchangeSparseGradientCommit(
                     table, OwnerPacket(OwnerHeader(session, 0, 0, 1))),
                 std::runtime_error);
    for (int shard = 0; shard < 2; ++shard)
      ExpectOwnerBits(
          shard,
          keys[shard],
          std::vector<float>(keys[shard].size() * kOwnerTestDim, 1.25f));
    if (!nonfinite)
      ExpectOwnerBits(0, {last_key}, std::vector<float>(kOwnerTestDim, 0.0f));
  }
}

TEST(PetPSIntegrationTest, OwnerRetryAndVersionMultiShard) {
  ASSERT_NO_FATAL_FAILURE(RequireOwnerDeployment());
  const std::array<std::vector<uint64_t>, 2> keys{
      OwnerKeys(0, 1), OwnerKeys(1, 1)};
  const std::string table = "owner_retry";
  ASSERT_NO_FATAL_FAILURE(OwnerInit(table, keys));
  auto& client      = OwnerShardClient(0);
  const auto values = OwnerBitValues(1);
  const auto packet = OwnerPacket(OwnerHeader(21300, 0), keys[0], values);
  ASSERT_EQ(client.ExchangeSparseGradientCommit(table, packet), -1);
  ASSERT_EQ(client.ExchangeSparseGradientCommit(table, packet), -1);
  const auto empty = OwnerPacket(OwnerHeader(21300, 1));
  ASSERT_EQ(client.ExchangeSparseGradientCommit(table, empty), 0);
  ASSERT_EQ(client.ExchangeSparseGradientCommit(table, packet), 0);
  ExpectOwnerBits(0, keys[0], values);
  auto conflicting = values;
  conflicting[0] =
      -0.0f; // Numeric equality cannot hide a changed retry payload.
  EXPECT_THROW(
      client.ExchangeSparseGradientCommit(
          table, OwnerPacket(OwnerHeader(21300, 0), keys[0], conflicting)),
      std::runtime_error);
  EXPECT_THROW(client.ExchangeSparseGradientCommit(table, empty),
               std::runtime_error);
  ExpectOwnerBits(0, keys[0], values);
  // No v1/v3 fallback or unknown wire version may enter an active v4 session.
  for (uint32_t version : {1u, 3u, 5u}) {
    const auto name = "owner_version_" + std::to_string(version);
    ASSERT_NO_FATAL_FAILURE(OwnerInit(name, keys));
    const uint64_t session = 21400 + version;
    ASSERT_EQ(client.ExchangeSparseGradientCommit(
                  name, OwnerPacket(OwnerHeader(session, 1))),
              -1);
    auto h       = OwnerHeader(session, 0);
    auto invalid = OwnerPacket(h);
    h.version    = version;
    std::memcpy(invalid.data(), &h, sizeof(h));
    EXPECT_THROW(client.ExchangeSparseGradientCommit(name, invalid),
                 std::runtime_error);
    EXPECT_THROW(client.ExchangeSparseGradientCommit(
                     name, OwnerPacket(OwnerHeader(session, 0, 0, 1))),
                 std::runtime_error);
    ExpectOwnerBits(0, keys[0], std::vector<float>(kOwnerTestDim, 1.25f));
  }
}

TEST(PetPSIntegrationTest, OwnerAdapterAbortReachesAllShardsWindowOne) {
  ASSERT_NO_FATAL_FAILURE(RequireOwnerDeployment());
  ASSERT_EQ(FLAGS_num_client_processes, 1);
  ASSERT_EQ(FLAGS_rdma_rc_num_logical_clients, 2);
  ASSERT_EQ(FLAGS_rdma_rc_qps_per_client_per_shard, 1);
  ASSERT_EQ(FLAGS_rdma_rc_slots_per_qp, 1);
  const int observer_id = ActiveLogicalClientId();
  ASSERT_GE(observer_id, 0);
  ASSERT_LT(observer_id, 2);
  const std::string table = "owner_abort_all_shards";
  const std::array<std::vector<uint64_t>, 2> keys{
      OwnerKeys(0, 1), OwnerKeys(1, 1)};
  json config = ParseFile2Json(std::getenv("RECSTORE_CONFIG"));
  ASSERT_EQ(recstore::ParseResolvedRdmaDeploymentConfig(config).num_clients, 1);
  // The raw per-shard observers and adapter must not replace each other's QPs.
  config["rdma_logical_client_id"] = 1 - observer_id;
  recstore::RDMAPSClientAdapter adapter(config);
  // One OS client exposes two logical lanes. Each server becomes ready only
  // after both lanes connect, so initialize both IDs concurrently in shard
  // order.
  auto observer_ready = std::async(std::launch::async, [] {
    for (int shard = 0; shard < 2; ++shard)
      (void)OwnerShardClient(shard);
  });
  std::exception_ptr initialization_error;
  int adapter_tag = -1;
  try {
    adapter_tag =
        adapter.InitEmbeddingTable(table, {1024, kOwnerTestDim, kOwnerTestTag});
  } catch (...) {
    initialization_error = std::current_exception();
  }
  // Always join the observer before throwing or making a fatal assertion.
  try {
    observer_ready.get();
  } catch (...) {
    if (!initialization_error)
      initialization_error = std::current_exception();
  }
  if (initialization_error)
    std::rethrow_exception(initialization_error);
  ASSERT_EQ(adapter_tag, static_cast<int>(kOwnerTestTag));
  ASSERT_NO_FATAL_FAILURE(OwnerInit(table, keys));
  // Check again after adapter initialization, which also reads runtime options.
  ASSERT_EQ(FLAGS_rdma_rc_qps_per_client_per_shard, 1);
  ASSERT_EQ(FLAGS_rdma_rc_slots_per_qp, 1);
  base::RecTensor empty({0, kOwnerTestDim}, base::DataType::FLOAT32);
  auto header = OwnerHeader(21500, 0);
  ASSERT_EQ(adapter.SparseGradientCommit(table, header, {}, empty), -1);
  for (int shard = 0; shard < 2; ++shard)
    ASSERT_EQ(OwnerShardClient(shard).ExchangeSparseGradientCommit(
                  table, OwnerPacket(OwnerHeader(21500, 0, 0, 1))),
              -1);

  header.kind = 3;
  // Abort deliberately returns an error even on a healthy server. With a
  // one-RPC window the old fail-fast loop never attempted the second shard.
  EXPECT_THROW(adapter.SparseGradientCommit(table, header, {}, empty),
               std::runtime_error);
  for (int shard = 0; shard < 2; ++shard) {
    EXPECT_THROW(OwnerShardClient(shard).ExchangeSparseGradientCommit(
                     table, OwnerPacket(OwnerHeader(21500, 0, 0, 1))),
                 std::runtime_error)
        << "shard=" << shard;
    ExpectOwnerBits(
        shard, keys[shard], std::vector<float>(kOwnerTestDim, 1.25f));
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

TEST(PetPSIntegrationTest, AdapterSparseGradientCommitWithOutstandingPrefetch) {
  json config = ParseFile2Json(std::getenv("RECSTORE_CONFIG"));
  config["distributed_client"]["max_keys_per_request"] = 10000;
  recstore::RDMAPSClientAdapter adapter(config);
  constexpr int rows = 100000; // Multiple training packets per shard.
  constexpr int dim  = 4;
  ASSERT_EQ(adapter.InitEmbeddingTable("training", {rows, dim}), 0);
  std::vector<uint64_t> keys(rows);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<float> initial(rows * dim, 1.0f), gradients(rows * dim, 2.0f);
  base::RecTensor values(initial.data(), {rows, dim});
  base::RecTensor grads(gradients.data(), {rows, dim});
  ASSERT_EQ(adapter.PutParameter(keys, values), 0);
  std::vector<uint64_t> probe{0, 7, 777, rows - 1};
  auto pref0 = adapter.PrefetchParameter(probe);
  auto pref1 = adapter.PrefetchParameter(probe);
  recstore::SparseGradientCommitHeader header;
  header.session = 8123;
  header.world   = 2;
  header.divisor = 2;
  header.dim     = dim;
  EXPECT_EQ(adapter.SparseGradientCommit("training", header, keys, grads), -1);
  header.rank = 1;
  EXPECT_EQ(adapter.SparseGradientCommit("training", header, keys, grads), 0);
  for (auto handle : {pref0, pref1}) {
    base::RecTensor old({0, dim}, base::DataType::FLOAT32);
    adapter.WaitForPrefetch(handle);
    EXPECT_TRUE(adapter.GetPrefetchResult(handle, old));
    EXPECT_EQ(old.shape(0), probe.size());
    for (int i = 0; i < probe.size() * dim; ++i)
      EXPECT_FLOAT_EQ(old.data_as<float>()[i], 1.0f);
  }
  base::RecTensor actual({rows, dim}, base::DataType::FLOAT32);
  ASSERT_EQ(adapter.GetParameter(keys, actual), 0);
  for (int i = 0; i < rows * dim; ++i)
    EXPECT_FLOAT_EQ(actual.data_as<float>()[i], .98f);
  header.step = 1;
  header.rank = 0;
  base::RecTensor empty({0, dim}, base::DataType::FLOAT32);
  EXPECT_EQ(adapter.SparseGradientCommit("training", header, {}, empty), 0);
  header.rank = 1;
  EXPECT_EQ(adapter.SparseGradientCommit("training", header, keys, grads), 1);
  ASSERT_EQ(adapter.GetParameter(keys, actual), 0);
  for (int i = 0; i < rows * dim; ++i)
    EXPECT_NEAR(actual.data_as<float>()[i], .97f, 1e-6);
}

TEST(PetPSIntegrationTest, AdapterWaitTimeoutReleasesSynchronousBatch) {
  if (std::getenv("RECSTORE_RDMA_ADAPTER_WAIT_TIMEOUT_TEST") == nullptr) {
    GTEST_SKIP()
        << "requires the adapter timeout driver and delayed GET workers";
  }
  ASSERT_EQ(FLAGS_num_server_processes, 1);
  ASSERT_EQ(FLAGS_rdma_rc_qps_per_client_per_shard, 2);
  ASSERT_EQ(FLAGS_rdma_rc_slots_per_qp, 1);
  ASSERT_EQ(FLAGS_rdma_wait_timeout_ms, 100);
  ASSERT_GE(FLAGS_rdma_rc_server_get_worker_hold_us, 2000000);
  ASSERT_FALSE(FLAGS_rdma_lane_rebuild_enabled);
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  recstore::RDMAPSClientAdapter adapter(ParseFile2Json(config_path));
  const int dim = FLAGS_value_size / sizeof(float);
  ASSERT_EQ(
      adapter.InitEmbeddingTable("adapter_timeout", {1024, uint64_t(dim)}), 0);
  const std::vector<uint64_t> keys{711};
  const auto values = MakeValues(keys, dim);
  ASSERT_EQ(adapter.PutParameter(keys, values), 0);

  const auto sync_start = std::chrono::steady_clock::now();
  {
    base::RecTensor output({1, dim}, base::DataType::FLOAT32);
    try {
      adapter.GetParameter(keys, output);
      FAIL() << "the delayed GET did not time out";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("wait timeout"),
                std::string::npos)
          << error.what();
    }
  }
  EXPECT_LT(std::chrono::steady_clock::now() - sync_start,
            std::chrono::milliseconds(1500));

  // This drains registered batches before handling empty keys. The failed
  // synchronous batch must be gone before its caller's output is destroyed.
  base::RecTensor empty({0, dim}, base::DataType::FLOAT32);
  ASSERT_EQ(adapter.GetParameter({}, empty), 0);

  // Only QP 0 timed out above; QP 1 must accept this single-chunk prefetch.
  const auto handle = adapter.PrefetchParameter(keys);
  ASSERT_GT(handle, 0);
  const auto prefetch_start = std::chrono::steady_clock::now();
  try {
    adapter.WaitForPrefetch(handle);
    FAIL() << "the delayed prefetch did not time out";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("wait timeout"), std::string::npos)
        << error.what();
  }
  EXPECT_LT(std::chrono::steady_clock::now() - prefetch_start,
            std::chrono::milliseconds(1500));

  // The failed prefetch still owns its private buffer until adapter teardown.
  // A following synchronous drain must report the retired lane promptly.
  const auto drain_start = std::chrono::steady_clock::now();
  EXPECT_THROW(adapter.GetParameter({}, empty), petps::ShardUnreachableError);
  EXPECT_LT(std::chrono::steady_clock::now() - drain_start,
            std::chrono::milliseconds(1500));
}

TEST(PetPSIntegrationTest, AdapterMultiChunkTimeoutRetiresAllSubmittedLanes) {
  if (std::getenv("RECSTORE_RDMA_ADAPTER_WAIT_TIMEOUT_TEST") == nullptr) {
    GTEST_SKIP()
        << "requires the adapter timeout driver and delayed GET workers";
  }
  ASSERT_EQ(FLAGS_num_server_processes, 1);
  ASSERT_EQ(FLAGS_rdma_rc_qps_per_client_per_shard, 2);
  ASSERT_EQ(FLAGS_rdma_rc_slots_per_qp, 1);
  ASSERT_EQ(FLAGS_rdma_wait_timeout_ms, 100);
  ASSERT_GE(FLAGS_rdma_rc_server_get_worker_hold_us, 2000000);
  ASSERT_FALSE(FLAGS_rdma_lane_rebuild_enabled);
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  const json config = ParseFile2Json(config_path);
  ASSERT_EQ(config["distributed_client"]["max_keys_per_request"].get<int>(), 1);
  recstore::RDMAPSClientAdapter adapter(config);
  const int dim = FLAGS_value_size / sizeof(float);
  ASSERT_EQ(adapter.InitEmbeddingTable("window_timeout", {1024, uint64_t(dim)}),
            0);
  const std::vector<uint64_t> keys{711, 712};
  const auto values = MakeValues(keys, dim);
  ASSERT_EQ(adapter.PutParameter(keys, values), 0);

  const auto wait_start = std::chrono::steady_clock::now();
  {
    base::RecTensor output({2, dim}, base::DataType::FLOAT32);
    try {
      adapter.GetParameter(keys, output);
      FAIL() << "the delayed two-chunk GET did not time out";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("wait timeout"),
                std::string::npos)
          << error.what();
    }
  }
  EXPECT_LT(std::chrono::steady_clock::now() - wait_start,
            std::chrono::milliseconds(1500));
  base::RecTensor empty({0, dim}, base::DataType::FLOAT32);
  ASSERT_EQ(adapter.GetParameter({}, empty), 0);

  // Both delayed RPCs still have remote work pending. Neither lane may be
  // reused.
  const std::vector<uint64_t> probe{711};
  const auto submit_start = std::chrono::steady_clock::now();
  EXPECT_THROW(adapter.PrefetchParameter(probe), petps::ShardUnreachableError);
  EXPECT_LT(std::chrono::steady_clock::now() - submit_start,
            std::chrono::milliseconds(500));
}

TEST(PetPSIntegrationTest, AdapterPrefetchOverCapacityReverseConsumption) {
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  recstore::RDMAPSClientAdapter adapter(ParseFile2Json(config_path));
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  ASSERT_GT(FLAGS_rdma_rc_qps_per_client_per_shard, 0);
  ASSERT_GT(FLAGS_rdma_rc_slots_per_qp, 0);
  const std::size_t request_count =
      static_cast<std::size_t>(FLAGS_rdma_rc_qps_per_client_per_shard) *
          static_cast<std::size_t>(FLAGS_rdma_rc_slots_per_qp) +
      1;
  ASSERT_EQ(adapter.InitEmbeddingTable(
                "prefetch_capacity",
                {4 * request_count, static_cast<uint64_t>(embedding_dim)}),
            0);

  for (int round = 0; round < 2; ++round) {
    SCOPED_TRACE(round);
    std::vector<std::vector<uint64_t>> request_keys;
    std::vector<uint64_t> handles;
    for (std::size_t request = 0; request < request_count; ++request) {
      const uint64_t first_key = 2 * (round * request_count + request);
      request_keys.push_back({first_key, first_key + 1});
      const auto values = MakeHashedValues(request_keys.back(), embedding_dim);
      ASSERT_EQ(adapter.PutParameter(request_keys.back(), values), 0);
    }
    // Exceed the physical slot capacity without consuming any handles.
    for (const auto& keys : request_keys) {
      handles.push_back(adapter.PrefetchParameter(keys));
    }
    for (std::size_t remaining = request_count; remaining > 0; --remaining) {
      const std::size_t request = remaining - 1;
      SCOPED_TRACE(request);
      base::RecTensor actual({0, embedding_dim}, base::DataType::FLOAT32);
      ASSERT_TRUE(adapter.GetPrefetchResult(handles[request], actual));
      ASSERT_EQ(actual.shape(0), 2);
      ASSERT_EQ(actual.shape(1), embedding_dim);
      ExpectHashedFlatSlots(
          actual.data_as<float>(), request_keys[request], embedding_dim);
    }
    // The second round must reuse slots without returning the first round's
    // data.
  }
}

TEST(PetPSIntegrationTest, AdapterWritesWithOutstandingPrefetch) {
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  const int capacity =
      FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp;
  if (capacity < 2) {
    GTEST_SKIP()
        << "the baseline needs a spare slot to write before consuming GETs";
  }
  recstore::RDMAPSClientAdapter adapter(ParseFile2Json(config_path));
  const int dim = FLAGS_value_size / sizeof(float);
  ASSERT_EQ(adapter.InitEmbeddingTable(
                "prefetch_writes", {2, static_cast<uint64_t>(dim)}),
            0);
  const std::vector<uint64_t> keys{0, 1};
  std::vector<float> initial(2 * dim, 1.0f), replacement(2 * dim, 3.0f),
      gradients(2 * dim, 2.0f);
  base::RecTensor old_values(initial.data(), {2, dim});
  base::RecTensor new_values(replacement.data(), {2, dim});
  base::RecTensor grads(gradients.data(), {2, dim});
  for (int use_update = 0; use_update < 2; ++use_update) {
    SCOPED_TRACE(use_update);
    ASSERT_EQ(adapter.PutParameter(keys, old_values), 0);
    std::vector<uint64_t> handles;
    for (int request = 0; request < capacity; ++request) {
      handles.push_back(adapter.PrefetchParameter(keys));
    }
    // Finish the reads while retaining their slots and public handles.
    for (const auto handle : handles) {
      adapter.WaitForPrefetch(handle);
    }
    if (use_update) {
      const auto update =
          adapter.SubmitUpdateParameterAsync("prefetch_writes", keys, grads);
      ASSERT_EQ(adapter.WaitUpdateParameter(update), 0);
    } else {
      ASSERT_EQ(adapter.PutParameter(keys, new_values), 0);
    }
    for (auto handle = handles.rbegin(); handle != handles.rend(); ++handle) {
      base::RecTensor actual({0, dim}, base::DataType::FLOAT32);
      ASSERT_TRUE(adapter.GetPrefetchResult(*handle, actual));
      ASSERT_EQ(actual.shape(0), 2);
      ASSERT_EQ(actual.shape(1), dim);
      ExpectFlatSlots(actual.data_as<float>(), old_values, dim);
    }
    base::RecTensor actual({2, dim}, base::DataType::FLOAT32);
    ASSERT_EQ(adapter.GetParameter(keys, actual), 0);
    for (int value = 0; value < 2 * dim; ++value) {
      EXPECT_FLOAT_EQ(actual.data_as<float>()[value], use_update ? .98f : 3.0f);
    }
  }
}

TEST(PetPSIntegrationTest, AdapterPrefetchWithPendingAsyncUpdate) {
  const char* config_path = std::getenv("RECSTORE_CONFIG");
  ASSERT_NE(config_path, nullptr);
  const json config = ParseFile2Json(config_path);
  const int capacity =
      FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp;
  if (config["distributed_client"]["num_shards"].get<int>() != 1 ||
      capacity < 2) {
    GTEST_SKIP() << "requires one shard and at least two slots";
  }
  recstore::RDMAPSClientAdapter adapter(config);
  const int dim = FLAGS_value_size / sizeof(float);
  ASSERT_EQ(
      adapter.InitEmbeddingTable(
          "pending_update",
          {static_cast<uint64_t>(capacity + 1), static_cast<uint64_t>(dim)}),
      0);
  std::vector<uint64_t> keys;
  for (int key = 0; key <= capacity; ++key) {
    keys.push_back(static_cast<uint64_t>(key));
  }
  std::vector<float> initial(keys.size() * dim, 1.0f), gradients(dim, 2.0f);
  base::RecTensor values(initial.data(), {capacity + 1, dim});
  base::RecTensor grads(gradients.data(), {1, dim});
  ASSERT_EQ(adapter.PutParameter(keys, values), 0);
  const std::vector<uint64_t> update_keys{0};
  // One tail UPDATE keeps a physical slot until WaitUpdateParameter consumes
  // it.
  const auto update =
      adapter.SubmitUpdateParameterAsync("pending_update", update_keys, grads);
  std::vector<uint64_t> handles;
  for (int key = 1; key <= capacity; ++key) {
    const std::vector<uint64_t> read_keys{static_cast<uint64_t>(key)};
    handles.push_back(adapter.PrefetchParameter(read_keys));
  }
  ASSERT_EQ(adapter.WaitUpdateParameter(update), 0);
  for (auto handle = handles.rbegin(); handle != handles.rend(); ++handle) {
    base::RecTensor actual({0, dim}, base::DataType::FLOAT32);
    ASSERT_TRUE(adapter.GetPrefetchResult(*handle, actual));
    ASSERT_EQ(actual.shape(0), 1);
    ASSERT_EQ(actual.shape(1), dim);
    for (int value = 0; value < dim; ++value) {
      EXPECT_FLOAT_EQ(actual.data_as<float>()[value], 1.0f);
    }
  }
  base::RecTensor actual({1, dim}, base::DataType::FLOAT32);
  ASSERT_EQ(adapter.GetParameter(update_keys, actual), 0);
  for (int value = 0; value < dim; ++value) {
    EXPECT_FLOAT_EQ(actual.data_as<float>()[value], .98f);
  }
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
  const int logical_client_id = ActiveLogicalClientId();

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

  ASSERT_EQ(petps::PetPSClientTestAccess::RebuildAllLanes(client), lanes)
      << "the server did not acknowledge every lane rebuild";
  expect_get_ok();

  // A second round proves the generations keep advancing in lockstep.
  ASSERT_EQ(petps::PetPSClientTestAccess::RebuildAllLanes(client), lanes);
  expect_get_ok();
}

// Regression test for the review finding: an async RPC in flight when its lane
// is rebuilt must fail through QueryRPCFinished()/WaitRPCFinish() instead of
// reporting "finished". Dropping the pending entry made both look successful,
// so the caller decoded an untouched (or zero-filled) buffer as the result.
TEST(PetPSIntegrationTest, RebuildDropsInflightAsyncRpcRepro) {
  if (!FLAGS_rdma_lane_rebuild_enabled) {
    GTEST_SKIP() << "requires --rdma_lane_rebuild_enabled on server and client";
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys;
  for (int i = 0; i < 64; ++i) {
    keys.push_back(8800001ULL + static_cast<std::uint64_t>(i));
  }
  auto values = MakeHashedValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  void* recv_buffer =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  float* buf = static_cast<float*>(recv_buffer);
  const std::size_t floats =
      keys.size() * static_cast<std::size_t>(embedding_dim);
  // Sentinel pre-fill: distinguishes "response never arrived" from zeros.
  for (std::size_t i = 0; i <= floats; ++i) {
    buf[i] = -1.0f;
  }

  const int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys), buf, /*isAsync=*/true);
  const int lanes   = FLAGS_rdma_rc_qps_per_client_per_shard;
  const int rebuilt = petps::PetPSClientTestAccess::RebuildAllLanes(client);
  printf("RECSTORE_REPRO rpc_id=%d rebuilt=%d/%d\n", rpc_id, rebuilt, lanes);

  const std::int32_t status_word = static_cast<std::int32_t>(buf[floats]);
  std::size_t sentinel_left      = 0;
  std::size_t correct            = 0;
  for (std::size_t i = 0; i < floats; ++i) {
    if (buf[i] == -1.0f) {
      ++sentinel_left;
    } else if (buf[i] == values.data_as<float>()[i]) {
      ++correct;
    }
  }
  printf("RECSTORE_REPRO status_word=%d sentinel_left=%zu correct=%zu/%zu\n",
         status_word,
         sentinel_left,
         correct,
         floats);

  // The rebuild discarded the in-flight response. Every waiter path must
  // report the lane failure; a missing pending entry (the old behaviour) made
  // QueryRPCFinished() return true and WaitRPCFinish() return silently, so a
  // zero-filled recv buffer decoded as RpcStatus::kOk.
  EXPECT_THROW(client.QueryRPCFinished(rpc_id), petps::ShardUnreachableError);
  EXPECT_THROW(client.WaitRPCFinish(rpc_id), petps::ShardUnreachableError);
  EXPECT_EQ(sentinel_left, floats);
  EXPECT_EQ(correct, 0u);
  client.RevokeRPCResource(rpc_id);
  printf("RECSTORE_REPRO failed_as_expected\n");
}

// Regression test for the review finding: server-side QP rebuild (RESET + CQ
// drain on the polling thread) must not race get-payload worker threads posting
// payload WRs on the same lane QP. Needs --rdma_rc_server_get_workers > 0 on
// the server; before the fence, ~5 in 9k GETs returned rc=0 with all-zero
// payload.
TEST(PetPSIntegrationTest, RebuildRacesGetPayloadWorkersRepro) {
  if (!FLAGS_rdma_lane_rebuild_enabled) {
    GTEST_SKIP() << "requires --rdma_lane_rebuild_enabled on server and client";
  }
  printf("RECSTORE_REPRO2 client_sees_get_workers=%d (server-side flag)\n",
         FLAGS_rdma_rc_server_get_workers);
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys;
  for (int i = 0; i < 256; ++i) {
    keys.push_back(8900001ULL + static_cast<std::uint64_t>(i));
  }
  auto values = MakeHashedValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  // Sustained collision: raise the rebuild budget so lanes never go dead, then
  // hammer GETs (through the server's payload workers) while continuously
  // forcing lane rebuilds. Each GET pre-fills its buffer with a sentinel so a
  // truncated/missing row is distinguishable from a correct one.
  FLAGS_rdma_lane_rebuild_max_attempts = 1000000;
  FLAGS_rdma_lane_rebuild_cooldown_ms  = 0;
  FLAGS_rdma_lane_rebuild_wait_ms      = 50;

  std::atomic<bool> stop{false};
  std::atomic<long> gets_ok{0};
  std::atomic<long> gets_failed{0};
  std::atomic<long> wrong_responses{0};
  std::atomic<long> rebuild_rounds{0};
  std::mutex print_mu;

  std::thread rebuilder([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      petps::PetPSClientTestAccess::RebuildAllLanes(client);
      rebuild_rounds.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  const auto getter = [&]() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline &&
           !stop.load(std::memory_order_relaxed)) {
      std::vector<float> output(
          keys.size() * static_cast<std::size_t>(embedding_dim), -1.0f);
      base::RecTensor output_tensor(
          output.data(), {static_cast<int64_t>(keys.size()), embedding_dim});
      int rc = -1;
      try {
        rc = client.GetParameter(
            base::ConstArray<std::uint64_t>(keys), output_tensor);
      } catch (const std::exception& error) {
        gets_failed.fetch_add(1, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> guard(print_mu);
        printf("RECSTORE_REPRO2 get_threw: %s\n", error.what());
        continue;
      }
      if (rc != 0) {
        gets_failed.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      const float* got  = output_tensor.data_as<float>();
      std::size_t wrong = 0;
      for (std::size_t i = 0; i < output.size(); ++i) {
        if (got[i] != values.data_as<float>()[i]) {
          ++wrong;
        }
      }
      if (wrong > 0) {
        wrong_responses.fetch_add(1, std::memory_order_relaxed);
        const std::lock_guard<std::mutex> guard(print_mu);
        printf("RECSTORE_REPRO2 KOK_OVER_MISSING_PAYLOAD rc=%d wrong=%zu/%zu "
               "first_bad=%f expected=%f\n",
               rc,
               wrong,
               output.size(),
               got[0],
               values.data_as<float>()[0]);
      } else {
        gets_ok.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::vector<std::thread> getters;
  for (int i = 0; i < 4; ++i) {
    getters.emplace_back(getter);
  }
  for (auto& thread : getters) {
    thread.join();
  }
  stop.store(true, std::memory_order_relaxed);
  rebuilder.join();
  printf("RECSTORE_REPRO2 summary gets_ok=%ld gets_failed=%ld "
         "wrong_responses=%ld rebuild_rounds=%ld\n",
         gets_ok.load(),
         gets_failed.load(),
         wrong_responses.load(),
         rebuild_rounds.load());
  // A rebuild may legitimately fail an in-flight GET (the caller retries), but
  // it must never return rc=0 with payload that the client did not receive.
  EXPECT_EQ(wrong_responses.load(), 0);
  if (FLAGS_rdma_rc_server_get_worker_hold_us == 0) {
    // Only the un-stalled lane must actually serve GETs. The fence-proof lane
    // deliberately stalls the payload workers, so its GETs mostly fail on
    // purpose; there the driver asserts the fence counters instead, and this
    // test only guards against a wrong response leaking through.
    EXPECT_GT(gets_ok.load(), 0);
  }
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
  const int logical_client_id = ActiveLogicalClientId();
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
      EXPECT_TRUE(
          GetMatches(frozen, frozen_keys, frozen_values, embedding_dim));
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
    EXPECT_THROW(GetMatches(frozen, frozen_keys, frozen_values, embedding_dim),
                 petps::ShardUnreachableError);
    const std::int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - attempt_start)
            .count();
    if (elapsed_us < kInstantFailUs) {
      instant_us = elapsed_us;
    }
    // Isolation: the healthy shard must keep serving while its peer retires.
    EXPECT_TRUE(
        GetMatches(healthy, healthy_keys, healthy_values, embedding_dim));
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
  EXPECT_THROW(GetMatches(frozen, frozen_keys, frozen_values, embedding_dim),
               petps::ShardUnreachableError);
  const std::int64_t post_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - post_start)
          .count();
  EXPECT_LT(post_us, kInstantFailUs)
      << "a retired lane was re-armed without a rebuild";
  EXPECT_TRUE(GetMatches(healthy, healthy_keys, healthy_values, embedding_dim));
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
  const int logical_client_id = ActiveLogicalClientId();
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

  ASSERT_TRUE(GetMatches(client, keys, values, embedding_dim));
  // The parent freezes shard 1 once it sees this marker, then unfreezes it.
  std::printf("%s\n", kRebuildWarmMarker);
  std::fflush(stdout);

  const auto start = std::chrono::steady_clock::now();
  bool saw_failure = false;
  bool recovered   = false;
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(120)) {
    try {
      if (GetMatches(client, keys, values, embedding_dim) && saw_failure) {
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
  const int logical_client_id = ActiveLogicalClientId();
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
      EXPECT_TRUE(
          GetMatches(killed, killed_keys, killed_values, embedding_dim));
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
    EXPECT_THROW(GetMatches(killed, killed_keys, killed_values, embedding_dim),
                 petps::ShardUnreachableError);
    const std::int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - attempt_start)
            .count();
    if (elapsed_us < kInstantFailUs) {
      instant_us = elapsed_us;
    }
    EXPECT_TRUE(
        GetMatches(healthy, healthy_keys, healthy_values, embedding_dim));
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
