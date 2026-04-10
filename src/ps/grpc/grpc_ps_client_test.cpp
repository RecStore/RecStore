#include <folly/executors/CPUThreadPoolExecutor.h>

#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "grpc_ps_client.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kGrpcTestPort = 15123;
}

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return false;

  for (int i = 0; i < a.size(); i++) {
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
  std::cout << "=== Testing Factory Pattern ===" << std::endl;

  json config = {{"host", "127.0.0.1"}, {"port", kGrpcTestPort}, {"shard", 0}};

  std::unique_ptr<recstore::BasePSClient> client(
      base::Factory<recstore::BasePSClient, json>::NewInstance("grpc", config));

  if (!client) {
    std::cerr << "Failed to create PS client via factory!" << std::endl;
    return;
  }

  std::cout << "Successfully created PS client via factory" << std::endl;

  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(1, 200LL * 1e6);

  auto grpc_client = dynamic_cast<GRPCParameterClient*>(client.get());
  if (grpc_client) {
    std::vector<uint64_t> keys = {1, 2, 3};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    // insert something
    grpc_client->PutParameter(keys, rightvalues);

    grpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    grpc_client->ClearPS();
    // read those
    grpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "load fake data" << std::endl;
    CHECK(grpc_client->LoadFakeData(100));
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    CHECK(grpc_client->DumpFakeData(100));
    std::cout << "dump fake data done" << std::endl;
  }
}

void TestAsyncReadWriteConcurrency() {
  std::cout << "\n=== Testing gRPC Async Read/Write Concurrency ==="
            << std::endl;

  GRPCParameterClient client("127.0.0.1", kGrpcTestPort, 1);
  CHECK(client.ClearPS());

  struct CaseData {
    std::vector<uint64_t> keys;
    std::vector<std::vector<float>> values;
  };

  constexpr int kCaseNum     = 4;
  constexpr int kRowsPerCase = 12;
  constexpr int kDim         = 4;

  std::vector<CaseData> cases(kCaseNum);
  for (int c = 0; c < kCaseNum; ++c) {
    auto& cs = cases[c];
    cs.keys.reserve(kRowsPerCase);
    cs.values.reserve(kRowsPerCase);
    for (int i = 0; i < kRowsPerCase; ++i) {
      uint64_t key = 10000 + static_cast<uint64_t>(c * 100 + i);
      cs.keys.push_back(key);
      cs.values.push_back(
          {static_cast<float>(key),
           static_cast<float>(key + 1),
           static_cast<float>(key + 2),
           static_cast<float>(key + 3)});
    }
  }

  std::vector<uint64_t> write_ids;
  write_ids.reserve(cases.size());
  for (const auto& cs : cases) {
    base::ConstArray<uint64_t> keys_array(cs.keys);
    uint64_t write_id = client.EmbWriteAsync(keys_array, cs.values);
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
  launch_options.override_ps_type = "GRPC";
  launch_options.override_ports   = {kGrpcTestPort, kGrpcTestPort + 1};
  recstore::test::ScopedPSServer server(launch_options, true);

  TestFactoryClient();
  TestAsyncReadWriteConcurrency();

  std::cout << "\n=== Testing Original Implementation ===" << std::endl;

  // GRPCParameterClient client("127.0.0.1", 15000, 1);
  // std::random_device
  //     rd;  // Will be used to obtain a seed for the random number engine
  // std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
  // rd() std::uniform_int_distribution<> distrib(1, 200LL * 1e6);

  // // while (1) {
  // //   int perf_count = 500;
  // //   std::vector<uint64_t> keys(perf_count);
  // //   for (int i = 0; i < perf_count; i++) {
  // //     keys[i] = distrib(gen);
  // //   }
  // //   std::vector<std::vector<float>> values;
  // //   xmh::Timer timer_client("client get");
  // //   ConstArray<uint64_t> keys_array(keys);
  // //   client.GetParameter(keys_array, &values);
  // //   timer_client.end();
  // // }

  // client.ClearPS();
  // // assert empty
  // std::vector<uint64_t> keys = {1, 2, 3};
  // std::vector<std::vector<float>> emptyvalues(keys.size());
  // std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
  // std::vector<std::vector<float>> values;
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, emptyvalues));

  // // insert something
  // client.PutParameter(keys, rightvalues);
  // // read those
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, rightvalues));

  // // clear all
  // client.ClearPS();
  // // read those
  // client.GetParameter(keys, &values);
  // CHECK(check_eq_2d(values, emptyvalues));

  return 0;
}
