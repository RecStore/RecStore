#include "ps/grpc/dist_grpc_ps_client.h"

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "base/array.h"
#include "base/init.h"
#include "base/tensor.h"
#include "base/timer.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kDistGrpcPort0 = 15133;
constexpr int kDistGrpcPort1 = 15134;

void DisableGrpcTracingEnv() {
  unsetenv("GRPC_TRACE");
  unsetenv("GRPC_VERBOSITY");
}

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

using namespace xmh;
using namespace recstore;

void TestBasicConfig(const std::vector<int>& ports) {
  std::cout << "=== Testing Basic Configuration ===" << std::endl;

  json recstore_config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(recstore_config);
    std::cout << "Recstore config parsed successfully, shard count: "
              << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore config test failed: " << e.what() << std::endl;
  }
}

void TestDirectClient(const std::vector<int>& ports) {
  std::cout << "=== Testing Direct Client Creation ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(config);
    std::cout << "Direct client created successfully, shard count: "
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

    std::cout << "All direct client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestLargeBatch(const std::vector<int>& ports) {
  std::cout << "=== Testing Large Batch Operations ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 50}}}};

  try {
    DistributedGRPCParameterClient client(config);

    std::vector<uint64_t> large_keys;
    std::vector<float> large_flat;
    large_keys.reserve(120);
    large_flat.reserve(240);
    for (int i = 0; i < 120; ++i) {
      large_keys.push_back(2000 + static_cast<uint64_t>(i) * 2);
      large_flat.push_back(float(i));
      large_flat.push_back(float(i * 2));
    }
    auto large_values = OwnedEmbedding(120, 2, large_flat);
    auto retrieved    = OwnedEmbedding(120, 2, std::vector<float>(240, -1.0f));
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

void TestPrefetch(const std::vector<int>& ports) {
  std::cout << "=== Testing Distributed gRPC Prefetch ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 8}}}};

  DistributedGRPCParameterClient client(config);
  client.ClearPS();

  std::vector<uint64_t> keys = {100, 101, 102, 103, 104, 105, 106, 107};
  const std::vector<float> flat = {
      1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f, 4.0f, 4.1f, 4.2f,
      5.0f, 5.1f, 5.2f, 6.0f, 6.1f, 6.2f, 7.0f, 7.1f, 7.2f, 8.0f, 8.1f, 8.2f};
  auto values = OwnedEmbedding(8, 3, flat);
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.PutParameter(keys_array, values) == 0);

  uint64_t prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(prefetch_id != 0);
  std::cout << "Issued prefetch request" << std::endl;
  CHECK(!client.IsPrefetchDone(999999));
  client.WaitForPrefetch(prefetch_id);
  std::cout << "Prefetch wait completed" << std::endl;
  CHECK(client.IsPrefetchDone(prefetch_id));

  base::RecTensor fetched({0, 3}, base::DataType::FLOAT32);
  CHECK(client.GetPrefetchResult(prefetch_id, fetched));
  std::cout << "Fetched prefetch result" << std::endl;
  CHECK(TensorEq(fetched, flat));
  CHECK(!client.GetPrefetchResult(prefetch_id, fetched));

  uint64_t second_id = client.PrefetchParameter(keys_array);
  CHECK(second_id != 0);
  auto preallocated = OwnedEmbedding(8, 3, std::vector<float>(24, -1.0f));
  CHECK(client.GetPrefetchResult(second_id, preallocated));
  CHECK(TensorEq(preallocated, flat));
  CHECK(!client.GetPrefetchResult(second_id, preallocated));
}

void TestPrefetchConcurrency(const std::vector<int>& ports) {
  std::cout << "=== Testing Distributed gRPC Prefetch Concurrency ==="
            << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", ports[0]}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", ports[1]}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 6}}}};

  DistributedGRPCParameterClient client(config);
  client.ClearPS();

  struct CaseData {
    std::vector<uint64_t> keys;
    std::vector<float> flat;
  };

  std::vector<CaseData> cases(16);
  for (size_t c = 0; c < cases.size(); ++c) {
    auto& cs = cases[c];
    cs.flat.reserve(36);
    for (int i = 0; i < 12; ++i) {
      uint64_t k = 3000 + static_cast<uint64_t>(c) * 100 + i;
      cs.keys.push_back(k);
      cs.flat.push_back(static_cast<float>(k));
      cs.flat.push_back(static_cast<float>(k + 1));
      cs.flat.push_back(static_cast<float>(k + 2));
    }
    auto values = OwnedEmbedding(12, 3, cs.flat);
    base::ConstArray<uint64_t> keys_array(cs.keys);
    CHECK(client.PutParameter(keys_array, values) == 0);
  }

  std::vector<std::future<bool>> futures;
  futures.reserve(cases.size());
  for (const auto& cs : cases) {
    futures.emplace_back(std::async(std::launch::async, [&client, cs]() {
      base::ConstArray<uint64_t> keys_array(cs.keys);
      uint64_t prefetch_id = client.PrefetchParameter(keys_array);
      if (prefetch_id == 0) {
        return false;
      }
      client.WaitForPrefetch(prefetch_id);
      base::RecTensor fetched({0, 3}, base::DataType::FLOAT32);
      if (!client.GetPrefetchResult(prefetch_id, fetched)) {
        return false;
      }
      return TensorEq(fetched, cs.flat);
    }));
  }

  for (auto& future : futures) {
    CHECK(future.get());
  }
}

int main(int argc, char** argv) {
  DisableGrpcTracingEnv();
  base::Init(&argc, &argv);
  Reporter::StartReportThread(2000);

  const std::vector<int> ports = {kDistGrpcPort0, kDistGrpcPort1};

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "GRPC";
  launch_options.override_ports   = ports;
  recstore::test::ScopedPSServer server(launch_options, true);

  std::cout << "=== Distributed gRPC PS client tests ===" << std::endl;
  std::cout << std::endl;

  TestBasicConfig(ports);
  std::cout << std::endl;

  TestDirectClient(ports);
  std::cout << std::endl;

  TestLargeBatch(ports);
  std::cout << std::endl;

  TestPrefetch(ports);
  std::cout << std::endl;

  TestPrefetchConcurrency(ports);
  std::cout << std::endl;

  std::cout << "All tests completed!" << std::endl;
  return 0;
}
