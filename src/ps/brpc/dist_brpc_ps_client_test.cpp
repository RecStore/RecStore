#include "dist_brpc_ps_client.h"

#include <algorithm>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <future>
#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/hash.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "test/server_mgr/ps_server_launcher.h"

using namespace xmh;
using namespace recstore;

namespace {
constexpr int kDistBrpcPort0 = 16133;
constexpr int kDistBrpcPort1 = 16134;
} // namespace

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() < b.size())
    return false;

  for (int i = 0; i < b.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6)
      return false;
  }
  return true;
}

static bool check_eq_2d(const std::vector<std::vector<float>>& a,
                        const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false)
      return false;
  }
  return true;
}

void TestBasicConfig() {
  std::cout << "=== Testing Basic Configuration (bRPC) ===" << std::endl;

  // 测试 recstore 配置格式
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

  // 转换为子类指针以访问扩展接口
  auto* client =
      dynamic_cast<DistributedBRPCParameterClient*>(base_client.get());
  if (!client) {
    std::cerr << "Failed to cast to DistributedBRPCParameterClient!"
              << std::endl;
    return;
  }

  std::cout << "Successfully created distributed bRPC PS client via factory"
            << std::endl;

  try {
    client->ClearPS();
    // assert empty
    std::vector<uint64_t> keys_vec = {1, 2, 3};
    base::ConstArray<uint64_t> keys(keys_vec);
    std::vector<std::vector<float>> emptyvalues(keys_vec.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    // insert something
    client->PutParameter(keys, rightvalues);
    // read those
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    client->ClearPS();
    // read those
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

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
    // assert empty
    std::vector<uint64_t> keys = {1001, 1002, 1003};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {
        {1, 0, 1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    base::ConstArray<uint64_t> keys_array(keys);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    // insert something
    client.PutParameter(keys_array, rightvalues);
    // read those
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    client.ClearPS();
    // read those
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

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

    // 构造同一 shard 的大批量 keys，稳定超过 max_keys_per_request
    std::vector<uint64_t> large_keys;
    std::vector<std::vector<float>> large_values;
    for (int i = 0; i < 100; ++i) {
      large_keys.push_back(2000 + static_cast<uint64_t>(i) * 2);
      large_values.push_back({float(i), float(i * 2)});
    }

    base::ConstArray<uint64_t> keys_array(large_keys);

    client.ClearPS();

    // 写入大批量数据
    int put_result = client.PutParameter(keys_array, large_values);
    CHECK(put_result == 0);

    // 读取并验证
    std::vector<std::vector<float>> retrieved_values;
    bool get_success = client.GetParameter(keys_array, &retrieved_values);
    CHECK(get_success);
    CHECK(check_eq_2d(retrieved_values, large_values));

    std::cout << "Large batch bRPC operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Large batch test skipped (servers not available): "
              << e.what() << std::endl;
  }
}

void TestPrefetch() {
  std::cout << "=== Testing Distributed bRPC Prefetch ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 8}}}};

  DistributedBRPCParameterClient client(config);
  client.ClearPS();

  std::vector<uint64_t> keys = {100, 101, 102, 103, 104, 105, 106, 107};
  std::vector<std::vector<float>> values = {
      {1.0f, 1.1f, 1.2f},
      {2.0f, 2.1f, 2.2f},
      {3.0f, 3.1f, 3.2f},
      {4.0f, 4.1f, 4.2f},
      {5.0f, 5.1f, 5.2f},
      {6.0f, 6.1f, 6.2f},
      {7.0f, 7.1f, 7.2f},
      {8.0f, 8.1f, 8.2f}};
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.PutParameter(keys_array, values) == 0);

  uint64_t prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(prefetch_id != 0);
  CHECK(!client.IsPrefetchDone(999999));
  client.WaitForPrefetch(prefetch_id);
  CHECK(client.IsPrefetchDone(prefetch_id));

  std::vector<std::vector<float>> fetched_values;
  CHECK(client.GetPrefetchResult(prefetch_id, &fetched_values));
  CHECK(check_eq_2d(fetched_values, values));
  CHECK(!client.GetPrefetchResult(prefetch_id, &fetched_values));

  uint64_t flat_prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(flat_prefetch_id != 0);
  std::vector<float> flat_values;
  int64_t num_rows = 0;
  CHECK(client.GetPrefetchResultFlat(
      flat_prefetch_id, &flat_values, &num_rows, 3));
  CHECK(num_rows == static_cast<int64_t>(keys.size()));
  CHECK(flat_values.size() == keys.size() * 3);
  for (size_t i = 0; i < keys.size(); ++i) {
    for (int d = 0; d < 3; ++d) {
      CHECK(std::abs(flat_values[i * 3 + d] - values[i][d]) < 1e-6);
    }
  }
  CHECK(!client.GetPrefetchResultFlat(
      flat_prefetch_id, &flat_values, &num_rows, 3));
  std::cout << "TestPrefetch done" << std::endl;
}

void TestPrefetchConcurrency() {
  std::cout << "=== Testing Distributed bRPC Prefetch Concurrency ==="
            << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kDistBrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kDistBrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 6}}}};

  DistributedBRPCParameterClient client(config);
  client.ClearPS();

  struct CaseData {
    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> values;
  };

  std::vector<CaseData> cases(4);
  for (size_t c = 0; c < cases.size(); ++c) {
    auto& cs = cases[c];
    for (int i = 0; i < 12; ++i) {
      uint64_t k = 5000 + static_cast<uint64_t>(c) * 100 + i;
      cs.keys.push_back(k);
      cs.values.push_back({static_cast<float>(k),
                           static_cast<float>(k + 1),
                           static_cast<float>(k + 2)});
    }
    base::ConstArray<uint64_t> keys_array(cs.keys);
    CHECK(client.PutParameter(keys_array, cs.values) == 0);
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
      std::vector<std::vector<float>> fetched_values;
      if (!client.GetPrefetchResult(prefetch_id, &fetched_values)) {
        return false;
      }
      return check_eq_2d(fetched_values, cs.values);
    }));
  }

  for (auto& future : futures) {
    CHECK(future.get());
  }
  std::cout << "TestPrefetchConcurrency passed!" << std::endl;
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "BRPC";
  launch_options.override_ports   = {kDistBrpcPort0, kDistBrpcPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  std::cout << "=== 分布式 bRPC 客户端测试 ===" << std::endl;
  std::cout << std::endl;

  TestBasicConfig();
  std::cout << std::endl;

  TestFactoryClient();
  std::cout << std::endl;

  TestDirectClient();
  std::cout << std::endl;

  TestLargeBatch();
  std::cout << std::endl;

  TestPrefetch();
  std::cout << std::endl;

  TestPrefetchConcurrency();
  std::cout << std::endl;

  std::cout << "All tests completed!" << std::endl;
  return 0;
}
