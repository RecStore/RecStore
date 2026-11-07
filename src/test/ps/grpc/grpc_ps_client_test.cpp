#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "ps/grpc/grpc_ps_client.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kGrpcTestPort0 = 15123;
constexpr int kGrpcTestPort1 = 15124;

base::RecTensor
OwnedEmbedding(int64_t n, int64_t d, const std::vector<float>& flat) {
  base::RecTensor t({n, d}, base::DataType::FLOAT32);
  if (t.data() != nullptr && !flat.empty()) {
    std::memcpy(
        t.data(),
        flat.data(),
        sizeof(float) *
            std::min(flat.size(), static_cast<size_t>(t.num_elements())));
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

// Binds an ephemeral loopback port and releases it, so the client can be
// pointed at an address that refuses connections.
int FindUnusedTcpPort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port        = 0;
  CHECK(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  socklen_t len = sizeof(addr);
  CHECK(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}
} // namespace

void TestFactoryClient(int grpc_port) {
  std::cout << "=== Testing Factory Pattern ===" << std::endl;

  json config = {{"host", "127.0.0.1"}, {"port", grpc_port}, {"shard", 0}};

  GRPCParameterClient client(config);
  std::vector<uint64_t> keys = {1, 2, 3};
  auto rightvalues = OwnedEmbedding(3, 3, {1, 0, 0, 2, 2, 0, 3, 3, 3});
  auto values      = OwnedEmbedding(3, 3, std::vector<float>(9, -1.0f));
  base::ConstArray<uint64_t> keys_array(keys);

  CHECK(client.PutParameter(keys, rightvalues));
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
}

void TestAsyncReadWriteConcurrency(int grpc_port) {
  std::cout << "\n=== Testing gRPC Async Read/Write Concurrency ==="
            << std::endl;

  GRPCParameterClient client("127.0.0.1", grpc_port, 1);
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
      uint64_t key = 10000 + static_cast<uint64_t>(c * 100 + i);
      cs.keys.push_back(key);
      flat.push_back(static_cast<float>(key));
      flat.push_back(static_cast<float>(key + 1));
      flat.push_back(static_cast<float>(key + 2));
      flat.push_back(static_cast<float>(key + 3));
    }
    cs.values = OwnedEmbedding(kRowsPerCase, kDim, flat);
  }

  for (const auto& cs : cases) {
    CHECK(client.PutParameter(cs.keys, cs.values));
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
    base::RecTensor fetched({0, kDim}, base::DataType::FLOAT32);
    CHECK(client.GetPrefetchResult(prefetch_ids[i], fetched));
    CHECK(fetched.shape(0) == kRowsPerCase);
    CHECK(fetched.shape(1) == kDim);
    CHECK(TensorEq(
        fetched,
        {cases[i].values.data_as<float>(),
         cases[i].values.data_as<float>() + cases[i].values.num_elements()}));
  }

  CHECK(client.ClearPS());
}

// The server keeps ownership of the learning rate; a client that advertises a
// different rate must get a failure instead of a silent server-side update.
void TestLearningRateGuard(int grpc_port) {
  std::cout << "\n=== Testing gRPC Learning Rate Guard ===" << std::endl;

  GRPCParameterClient client("127.0.0.1", grpc_port, 0);
  CHECK(client.InitEmbeddingTable("lr_guard", {128, 4}) >= 0);

  std::vector<uint64_t> keys = {1, 2};
  std::vector<float> flat(8, 0.1f);
  base::RecTensor grads(flat.data(), {2, 4});
  base::ConstArray<uint64_t> keys_array(keys);

  // The test server uses the default SGD learning rate (0.01).
  CHECK(client.UpdateParameter("lr_guard", keys_array, grads, 0.01f) == 0);
  CHECK(client.UpdateParameter("lr_guard", keys_array, grads, 0.02f) == -1);
  // 0 means "not advertised" and keeps the legacy behaviour.
  CHECK(client.UpdateParameter("lr_guard", keys_array, grads) == 0);
}

void TestMultiBatchGet(int grpc_port) {
  std::cout << "\n=== Testing gRPC Multi-Batch Get ===" << std::endl;

  GRPCParameterClient client("127.0.0.1", grpc_port, 0);
  CHECK(client.ClearPS());

  constexpr int kRows = MAX_PARAMETER_BATCH + 1;
  constexpr int kDim  = 2;
  std::vector<uint64_t> keys(kRows);
  std::vector<float> flat(static_cast<size_t>(kRows) * kDim);
  for (int i = 0; i < kRows; ++i) {
    keys[i]            = 200000 + static_cast<uint64_t>(i);
    flat[i * kDim]     = static_cast<float>(i);
    flat[i * kDim + 1] = static_cast<float>(i) + 0.5f;
  }

  auto values = OwnedEmbedding(kRows, kDim, flat);
  CHECK(client.PutParameter(keys, values));

  auto fetched =
      OwnedEmbedding(kRows, kDim, std::vector<float>(flat.size(), -1.0f));
  base::ConstArray<uint64_t> keys_array(keys);
  CHECK(client.GetParameter(keys_array, fetched) == 0);
  CHECK(TensorEq(fetched, flat));

  CHECK(client.ClearPS());
}

// A dead peer must surface as an error instead of zero-filled embeddings.
void TestRpcFailureReporting() {
  std::cout << "\n=== Testing gRPC Failure Reporting ===" << std::endl;

  GRPCParameterClient client("127.0.0.1", FindUnusedTcpPort(), 0);

  std::vector<uint64_t> keys = {1, 2, 3};
  base::ConstArray<uint64_t> keys_array(keys);
  auto values = OwnedEmbedding(3, 4, std::vector<float>(12, -1.0f));
  CHECK(client.GetParameter(keys_array, values) == -1);

  const uint64_t prefetch_id = client.PrefetchParameter(keys_array);
  CHECK(prefetch_id != 0);
  client.WaitForPrefetch(prefetch_id);
  CHECK(client.IsPrefetchDone(prefetch_id));
  base::RecTensor fetched({0, 4}, base::DataType::FLOAT32);
  CHECK(!client.GetPrefetchResult(prefetch_id, fetched));
}

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "GRPC";
  launch_options.override_ports   = {kGrpcTestPort0, kGrpcTestPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  TestFactoryClient(kGrpcTestPort0);
  TestAsyncReadWriteConcurrency(kGrpcTestPort0);
  TestMultiBatchGet(kGrpcTestPort0);
  TestRpcFailureReporting();
  TestLearningRateGuard(kGrpcTestPort0);
  return 0;
}
