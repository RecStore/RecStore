#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/tensor.h"
#include "base/thread.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "ps/brpc/brpc_ps_client.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kBrpcTestPort0 = 16123;
constexpr int kBrpcTestPort1 = 16124;

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

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern (bRPC) ===" << std::endl;

  json config = {{"host", "127.0.0.1"}, {"port", kBrpcTestPort0}, {"shard", 1}};

  BRPCParameterClient client(config);
  std::vector<uint64_t> keys = {1, 2, 3};
  auto rightvalues           = OwnedEmbedding(3, 3, {1, 0, 0, 2, 2, 0, 3, 3, 3});
  auto values                = OwnedEmbedding(3, 3, std::vector<float>(9, -1.0f));
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.PutParameter(keys, rightvalues));
  std::cout << "put parameter done" << std::endl;
  CHECK(client.GetParameter(keys_array, values) == 0);
  CHECK(TensorEq(values, {1, 0, 0, 2, 2, 0, 3, 3, 3}));

  client.ClearPS();
  CHECK(client.GetParameter(keys_array, values) == 0);

  std::cout << "load fake data" << std::endl;
  CHECK(client.LoadFakeData(100));
  std::cout << "load fake data done" << std::endl;
  std::cout << "dump fake data" << std::endl;
  CHECK(client.DumpFakeData(100));
  std::cout << "dump fake data done" << std::endl;

  std::cout << "All bRPC operations passed!" << std::endl;
}

void TestDirectClient() {
  std::cout << "\n=== Testing Direct bRPC Client Creation ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", kBrpcTestPort0, 1);

  client.ClearPS();
  std::vector<uint64_t> keys = {1, 2, 3};
  auto rightvalues           = OwnedEmbedding(3, 3, {1, 0, 0, 2, 2, 0, 3, 3, 3});
  auto values                = OwnedEmbedding(3, 3, std::vector<float>(9, -1.0f));
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.GetParameter(keys_array, values) == 0);

  CHECK(client.PutParameter(keys, rightvalues));
  CHECK(client.GetParameter(keys_array, values) == 0);
  CHECK(TensorEq(values, {1, 0, 0, 2, 2, 0, 3, 3, 3}));

  client.ClearPS();
  CHECK(client.GetParameter(keys_array, values) == 0);

  std::cout << "All direct bRPC client operations passed!" << std::endl;
}

void TestPrefetch() {
  std::cout << "\n=== Testing bRPC Prefetch ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", kBrpcTestPort0, 1);
  client.ClearPS();

  std::vector<uint64_t> keys = {100, 101, 102, 103, 104};
  auto values =
      OwnedEmbedding(5, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});

  CHECK(client.PutParameter(keys, values));
  base::ConstArray<uint64_t> keys_array(keys);
  uint64_t prefetch_id = client.PrefetchParameter(keys_array);

  if (prefetch_id != 0) {
    client.WaitForPrefetch(prefetch_id);
    base::RecTensor fetched({0, 2}, base::DataType::FLOAT32);
    if (client.GetPrefetchResult(prefetch_id, fetched)) {
      CHECK(TensorEq(fetched, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}));
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
    base::RecTensor values;
  };

  constexpr int kCaseNum     = 4;
  constexpr int kRowsPerCase = 12;
  constexpr int kDim         = 4;

  std::vector<CaseData> cases(kCaseNum);
  for (int c = 0; c < kCaseNum; ++c) {
    auto& cs = cases[c];
    cs.keys.reserve(kRowsPerCase);
    std::vector<float> flat;
    flat.reserve(kRowsPerCase * kDim);
    for (int i = 0; i < kRowsPerCase; ++i) {
      uint64_t key = 20000 + static_cast<uint64_t>(c * 100 + i);
      cs.keys.push_back(key);
      flat.push_back(static_cast<float>(key));
      flat.push_back(static_cast<float>(key + 1));
      flat.push_back(static_cast<float>(key + 2));
      flat.push_back(static_cast<float>(key + 3));
    }
    cs.values = OwnedEmbedding(kRowsPerCase, kDim, flat);
  }

  for (auto& cs : cases) {
    CHECK(client.PutParameter(cs.keys, cs.values));
  }

  std::vector<uint64_t> prefetch_ids;
  prefetch_ids.reserve(cases.size());
  for (auto& cs : cases) {
    base::ConstArray<uint64_t> keys_array(cs.keys);
    uint64_t prefetch_id = client.PrefetchParameter(keys_array);
    CHECK(prefetch_id != 0);
    prefetch_ids.push_back(prefetch_id);
  }

  for (size_t i = 0; i < cases.size(); ++i) {
    client.WaitForPrefetch(prefetch_ids[i]);
    base::RecTensor fetched({0, kDim}, base::DataType::FLOAT32);
    CHECK(client.GetPrefetchResult(prefetch_ids[i], fetched));
    CHECK(fetched.shape(0) == kRowsPerCase);
    CHECK(fetched.shape(1) == kDim);
    CHECK(TensorEq(fetched,
                   {cases[i].values.data_as<float>(),
                    cases[i].values.data_as<float>() +
                        cases[i].values.num_elements()}));
  }

  CHECK(client.ClearPS());
}

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "BRPC";
  launch_options.override_ports   = {kBrpcTestPort0, kBrpcTestPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  std::cout << "=== bRPC parameter server client tests ===" << std::endl;
  std::cout << std::endl;

  try {
    TestFactoryClient();
    TestDirectClient();
    TestPrefetch();
    TestAsyncReadWriteConcurrency();

    std::cout << "\nAll bRPC tests passed." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
