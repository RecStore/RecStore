#include <folly/init/Init.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "base/array.h"
#include "benchmark/ps_transport_benchmark_config.h"
#include "framework/ps_client_factory.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

DEFINE_string(transport, "rdma", "rdma|grpc|brpc");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 25000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_int32(iterations, 100, "number of get/put iterations");
DECLARE_int32(value_size);

namespace {

std::vector<uint64_t> MakeKeys() { return {1001, 1002, 1003, 1004}; }

std::vector<std::vector<float>> MakeValues(const std::vector<uint64_t>& keys) {
  const int dim = FLAGS_value_size / sizeof(float);
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row(dim, static_cast<float>(key));
    values.push_back(std::move(row));
  }
  return values;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);

  const std::string transport = NormalizeBenchmarkTransport(FLAGS_transport);
  const auto keys             = MakeKeys();
  const auto values           = MakeValues(keys);
  const auto key_array        = base::ConstArray<uint64_t>(keys);

  if (transport == "RDMA") {
    if (FLAGS_num_shards == 1) {
      petps::PetPSClient client(FLAGS_host, FLAGS_port, 0);
      client.InitThread();

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < FLAGS_iterations; ++i) {
        client.PutParameter(keys, values);
        void* recv_buffer =
            client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
        int rpc_id = client.GetParameter(
            key_array, static_cast<float*>(recv_buffer), false, 0);
        client.WaitRPCFinish(rpc_id);
        client.RevokeRPCResource(rpc_id);
      }
      auto end = std::chrono::steady_clock::now();
      std::cout << "transport=RDMA elapsed_us="
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       end - start)
                       .count()
                << std::endl;
      return 0;
    }

    std::vector<std::unique_ptr<petps::PetPSClient>> owned;
    std::vector<BaseParameterClient*> clients;
    for (int shard = 0; shard < FLAGS_num_shards; ++shard) {
      owned.push_back(
          std::make_unique<petps::PetPSClient>(FLAGS_host, FLAGS_port, shard));
      owned.back()->InitThread();
      clients.push_back(owned.back().get());
    }

    AllShardsParameterClientWrapper client(clients, FLAGS_num_shards);
    client.InitThread();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < FLAGS_iterations; ++i) {
      client.PutParameter(keys, values);
      std::vector<float> output(
          keys.size() * (FLAGS_value_size / sizeof(float)) + 1, 0.0f);
      int rpc_id = client.GetParameter(key_array, output.data(), false, 0);
      client.WaitRPCFinish(rpc_id);
      client.RevokeRPCResource(rpc_id);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "transport=RDMA elapsed_us="
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end - start)
                     .count()
              << std::endl;
    return 0;
  }

  auto config = BuildRpcBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
  std::unique_ptr<recstore::BasePSClient> client(
      recstore::CreateFrameworkPSClient(config));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < FLAGS_iterations; ++i) {
    client->PutParameter(key_array, values);
    if (BenchmarkUsesVectorGet(transport)) {
      auto* brpc_client = dynamic_cast<BRPCParameterClient*>(client.get());
      CHECK_NE(brpc_client, nullptr);
      std::vector<std::vector<float>> output;
      brpc_client->GetParameter(key_array, &output);
    } else {
      std::vector<float> output(
          keys.size() * (FLAGS_value_size / sizeof(float)), 0.0f);
      client->GetParameter(key_array, output.data());
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "transport=" << transport << " elapsed_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start)
                   .count()
            << std::endl;
  return 0;
}
