#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>

#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "brpc_ps_client.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kBrpcTestPort0 = 16123;
constexpr int kBrpcTestPort1 = 16124;
} // namespace

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  std::cout << "a: ";
  for (auto& v : a) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  std::cout << "b: ";
  for (auto& v : b) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  if (a.size() < b.size())
    return false;

  for (int i = 0; i < b.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6)
      return false;
  }
  return true;
}

static bool check_eq_2d(std::vector<std::vector<float>>& a,
                        const std::vector<std::vector<float>>& b) {
  a.resize(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    a[i].resize(b[i].size());
  }
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false)
      return false;
  }
  return true;
}

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern (bRPC) ===" << std::endl;

  json config = {{"host", "127.0.0.1"}, {"port", kBrpcTestPort0}, {"shard", 1}};

  std::unique_ptr<recstore::BasePSClient> client(
      base::Factory<recstore::BasePSClient, json>::NewInstance("brpc", config));

  if (!client) {
    std::cerr << "Failed to create bRPC PS client via factory!" << std::endl;
    return;
  }

  std::cout << "Successfully created bRPC PS client via factory" << std::endl;

  auto brpc_client = dynamic_cast<BRPCParameterClient*>(client.get());
  if (brpc_client) {
    // brpc_client->ClearPS();
    // assert empty
    std::vector<uint64_t> keys = {1, 2, 3};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    // insert something
    brpc_client->PutParameter(keys, rightvalues);
    std::cout << "put parameter done" << std::endl;
    // read those
    brpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    brpc_client->ClearPS();
    // read those
    brpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "load fake data" << std::endl;
    CHECK(brpc_client->LoadFakeData(100));
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    CHECK(brpc_client->DumpFakeData(100));
    std::cout << "dump fake data done" << std::endl;

    std::cout << "All bRPC operations passed!" << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "\n=== Testing Direct bRPC Client Creation ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", kBrpcTestPort0, 1);

  client.ClearPS();
  // assert empty
  std::vector<uint64_t> keys = {1, 2, 3};
  std::vector<std::vector<float>> emptyvalues(keys.size());
  std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
  std::vector<std::vector<float>> values;

  base::ConstArray<uint64_t> keys_array(keys);
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, emptyvalues));

  // insert something
  client.PutParameter(keys, rightvalues);
  // read those
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, rightvalues));

  // clear all
  client.ClearPS();
  // read those
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, emptyvalues));

  std::cout << "All direct bRPC client operations passed!" << std::endl;
}

void TestPrefetch() {
  std::cout << "\n=== Testing bRPC Prefetch ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", kBrpcTestPort0, 1);

  client.ClearPS();

  // 准备测试数据
  std::vector<uint64_t> keys             = {100, 101, 102, 103, 104};
  std::vector<std::vector<float>> values = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}};

  // 先写入数据
  client.PutParameter(keys, values);

  // 测试 prefetch
  base::ConstArray<uint64_t> keys_array(keys);
  uint64_t prefetch_id = client.PrefetchParameter(keys_array);

  if (prefetch_id != 0) {
    client.WaitForPrefetch(prefetch_id);

    std::vector<std::vector<float>> fetched_values;
    if (client.GetPrefetchResult(prefetch_id, &fetched_values)) {
      CHECK(check_eq_2d(fetched_values, values));
      std::cout << "Prefetch test passed!" << std::endl;
    } else {
      std::cout << "Failed to get prefetch result" << std::endl;
    }
  } else {
    std::cout << "Prefetch not supported or failed" << std::endl;
  }

  client.ClearPS();
}

void TestAsyncReadWriteConcurrency() {
  std::cout << "\n=== Testing bRPC Async Read/Write Concurrency ==="
            << std::endl;

  BRPCParameterClient client("127.0.0.1", kBrpcTestPort0, 1);
  CHECK(client.ClearPS());

  struct CaseData {
    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> values;
    std::vector<float> flat_values;
  };

  constexpr int kCaseNum     = 4;
  constexpr int kRowsPerCase = 12;
  constexpr int kDim         = 4;

  std::vector<CaseData> cases(kCaseNum);
  for (int c = 0; c < kCaseNum; ++c) {
    auto& cs = cases[c];
    cs.keys.reserve(kRowsPerCase);
    cs.values.reserve(kRowsPerCase);
    cs.flat_values.reserve(kRowsPerCase * kDim);
    for (int i = 0; i < kRowsPerCase; ++i) {
      uint64_t key = 20000 + static_cast<uint64_t>(c * 100 + i);
      cs.keys.push_back(key);
      std::vector<float> embedding = {
          static_cast<float>(key),
          static_cast<float>(key + 1),
          static_cast<float>(key + 2),
          static_cast<float>(key + 3)};
      cs.values.push_back(embedding);
      cs.flat_values.insert(
          cs.flat_values.end(), embedding.begin(), embedding.end());
    }
  }

  std::vector<uint64_t> write_ids;
  write_ids.reserve(cases.size());
  for (auto& cs : cases) {
    base::RecTensor key_tensor(
        cs.keys.data(), {static_cast<int64_t>(cs.keys.size())});
    base::RecTensor value_tensor(
        cs.flat_values.data(),
        {static_cast<int64_t>(cs.keys.size()), static_cast<int64_t>(kDim)});
    uint64_t write_id = client.EmbWriteAsync(key_tensor, value_tensor);
    CHECK(write_id != 0);
    write_ids.push_back(write_id);
  }
  for (uint64_t write_id : write_ids) {
    client.WaitForWrite(write_id);
  }

  std::vector<uint64_t> prefetch_ids;
  prefetch_ids.reserve(cases.size());
  for (const auto& cs : cases) {
    base::ConstArray<uint64_t> keys_array(cs.keys);
    uint64_t prefetch_id = client.PrefetchParameter(keys_array);
    CHECK(prefetch_id != 0);
    prefetch_ids.push_back(prefetch_id);
  }

  for (size_t i = 0; i < cases.size(); ++i) {
    client.WaitForPrefetch(prefetch_ids[i]);
    std::vector<std::vector<float>> fetched_values;
    CHECK(client.GetPrefetchResult(prefetch_ids[i], &fetched_values));
    CHECK(check_eq_2d(fetched_values, cases[i].values));
    CHECK(static_cast<int>(fetched_values.size()) == kRowsPerCase);
    for (const auto& row : fetched_values) {
      CHECK(static_cast<int>(row.size()) == kDim);
    }
  }

  CHECK(client.ClearPS());
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "BRPC";
  launch_options.override_ports   = {kBrpcTestPort0, kBrpcTestPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  std::cout << "=== bRPC 参数服务器客户端测试 ===" << std::endl;
  std::cout << std::endl;

  try {
    TestFactoryClient();
    TestDirectClient();
    TestPrefetch();
    TestAsyncReadWriteConcurrency();

    std::cout << "\n所有 bRPC 测试通过！" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
