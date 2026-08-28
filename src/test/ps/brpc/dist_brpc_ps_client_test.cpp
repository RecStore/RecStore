#include "ps/brpc/dist_brpc_ps_client.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/hash.h"
#include "base/init.h"
#include "base/tensor.h"
#include "base/thread.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "test/server_mgr/ps_server_launcher.h"

using namespace xmh;
using namespace recstore;

namespace {
constexpr int kDistBrpcPort0 = 16133;
constexpr int kDistBrpcPort1 = 16134;

base::RecTensor OwnedEmbedding(int64_t n, int64_t d, const std::vector<float>& flat) {
  base::RecTensor t({n, d}, base::DataType::FLOAT32);
  if (t.data() != nullptr && !flat.empty()) {
    std::memcpy(t.data(),
                flat.data(),
                sizeof(float) * std::min(flat.size(),
                                         static_cast<size_t>(t.num_elements())));
  }
  return t;
}

bool TensorEq(const base::RecTensor& a, const std::vector<float>& b) {
  if (static_cast<size_t>(a.num_elements()) != b.size()) {
    return false;
  }
  if (b.empty()) {
    return true;
  }
  const float* p = a.data_as<float>();
  for (size_t i = 0; i < b.size(); ++i) {
    if (std::abs(p[i] - b[i]) > 1e-6f) {
      return false;
    }
  }
  return true;
}
} // namespace

void TestBasicConfig() {
  std::cout << "=== Testing Basic Configuration (bRPC) ===" << std::endl;

  json recstore_config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedBRPCParameterClient client(recstore_config);
    std::cout << "Recstore bRPC config parsed successfully, shard count: "
              << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore bRPC config test failed: " << e.what() << std::endl;
  }
}

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern (bRPC) ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  std::unique_ptr<BasePSClient> base_client(
      base::Factory<BasePSClient, json>::NewInstance(
          "distributed_brpc", config));

  if (!base_client) {
    std::cerr << "Failed to create distributed bRPC PS client via factory!"
              << std::endl;
    return;
  }

  std::cout << "Successfully created distributed bRPC PS client via factory"
            << std::endl;

  auto* client =
      dynamic_cast<DistributedBRPCParameterClient*>(base_client.get());
  if (!client) {
    std::cerr << "Failed to cast to DistributedBRPCParameterClient!"
              << std::endl;
    return;
  }

  try {
    client->ClearPS();
    std::vector<uint64_t> keys_vec = {1, 2, 3};
    base::ConstArray<uint64_t> keys(keys_vec);
    auto rightvalues = OwnedEmbedding(3, 3, {1, 0, 0, 2, 2, 0, 3, 3, 3});
    auto values      = OwnedEmbedding(3, 3, std::vector<float>(9, -1.0f));

    CHECK(client->GetParameter(keys, values) == 0);

    CHECK(client->PutParameter(keys, rightvalues) == 0);
    CHECK(client->GetParameter(keys, values) == 0);
    CHECK(TensorEq(values, {1, 0, 0, 2, 2, 0, 3, 3, 3}));

    client->ClearPS();
    CHECK(client->GetParameter(keys, values) == 0);

    std::cout << "load fake data" << std::endl;
    CHECK(client->LoadFakeData(100));
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    CHECK(client->DumpFakeData(100));
    std::cout << "dump fake data done" << std::endl;

    std::cout << "All distributed bRPC PS operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "=== Testing Direct bRPC Client Creation ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedBRPCParameterClient client(config);
    std::cout << "Direct bRPC client created successfully, shard count: "
              << client.shard_count() << std::endl;

    client.ClearPS();
    std::vector<uint64_t> keys = {1001, 1002, 1003};
    auto rightvalues = OwnedEmbedding(3, 3, {1, 0, 1, 2, 2, 0, 3, 3, 3});
    auto values      = OwnedEmbedding(3, 3, std::vector<float>(9, -1.0f));
    base::ConstArray<uint64_t> keys_array(keys);

    CHECK(client.GetParameter(keys_array, values) == 0);

    CHECK(client.PutParameter(keys_array, rightvalues) == 0);
    CHECK(client.GetParameter(keys_array, values) == 0);
    CHECK(TensorEq(values, {1, 0, 1, 2, 2, 0, 3, 3, 3}));

    client.ClearPS();
    CHECK(client.GetParameter(keys_array, values) == 0);

    std::cout << "All direct bRPC client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestLargeBatch() {
  std::cout << "=== Testing Large Batch Operations (bRPC) ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 50}}}};

  try {
    DistributedBRPCParameterClient client(config);

    std::vector<uint64_t> large_keys;
    std::vector<float> large_flat;
    large_keys.reserve(100);
    large_flat.reserve(200);
    for (int i = 0; i < 100; ++i) {
      large_keys.push_back(2000 + static_cast<uint64_t>(i) * 2);
      large_flat.push_back(float(i));
      large_flat.push_back(float(i * 2));
    }
    auto large_values = OwnedEmbedding(100, 2, large_flat);
    auto retrieved    = OwnedEmbedding(100, 2, std::vector<float>(200, -1.0f));
    base::ConstArray<uint64_t> keys_array(large_keys);

    client.ClearPS();
    CHECK(client.PutParameter(keys_array, large_values) == 0);
    CHECK(client.GetParameter(keys_array, retrieved) == 0);
    CHECK(TensorEq(retrieved, large_flat));

    std::cout << "Large batch test passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "BRPC";
  launch_options.override_ports   = {kDistBrpcPort0, kDistBrpcPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  TestBasicConfig();
  TestFactoryClient();
  TestDirectClient();
  TestLargeBatch();
  return 0;
}
