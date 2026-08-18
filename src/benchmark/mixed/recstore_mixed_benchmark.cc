#include <folly/init/Init.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <cstdint>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/tensor.h"
#include "benchmark/ps/ps_transport_benchmark_config.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"

DEFINE_string(transport, "grpc", "grpc|brpc|rdma|local_shm");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 15000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_int32(iterations, 100, "number of mixed read/update iterations");
DEFINE_int32(rounds, 1, "number of measured rounds");
DEFINE_int32(warmup_rounds, 0, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 128, "number of keys per iteration");
DEFINE_int32(embedding_dim, 128, "embedding dimension");
DEFINE_int64(num_embeddings, 1024, "number of embeddings preloaded into table");
DEFINE_int32(init_chunk_size, 4096, "initialization chunk size");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DEFINE_string(table_name, "default", "embedding table name");
DEFINE_double(update_scale, 0.001, "gradient scale per key");

namespace {

std::string NormalizeMixedTransport(std::string transport) {
  std::transform(
      transport.begin(),
      transport.end(),
      transport.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (transport == "RDMA" || transport == "GRPC" || transport == "BRPC" ||
      transport == "LOCAL_SHM") {
    return transport;
  }
  throw std::invalid_argument("Unsupported transport: " + transport);
}

nlohmann::json BuildMixedBenchmarkConfig(
    const std::string& transport, const std::string& host, int port) {
  const auto normalized = NormalizeMixedTransport(transport);
  if (normalized == "LOCAL_SHM") {
    return {{"cache_ps", {{"ps_type", "LOCAL_SHM"}}},
            {"client", {{"host", host}, {"port", port}, {"shard", 0}}},
            {"region_name", "recstore_local_ps"}};
  }
  return BuildRpcBenchmarkConfig(normalized, host, port);
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

double PercentileUs(std::vector<int64_t> samples, double ratio) {
  CHECK(!samples.empty());
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(std::min<int64_t>(
      samples.size() - 1,
      static_cast<int64_t>(std::ceil(ratio * samples.size()) - 1)));
  return static_cast<double>(samples[idx]);
}

void PrintSummary(const std::string& transport,
                  const std::string& phase,
                  const std::vector<int64_t>& elapsed_us_samples,
                  int iterations_per_round,
                  int batch_keys,
                  int64_t num_embeddings) {
  if (elapsed_us_samples.empty()) {
    return;
  }

  const double total_us = std::accumulate(
      elapsed_us_samples.begin(), elapsed_us_samples.end(), 0.0);
  const double mean_us =
      total_us / static_cast<double>(elapsed_us_samples.size());
  const double p50_us = PercentileUs(elapsed_us_samples, 0.50);
  const double p95_us = PercentileUs(elapsed_us_samples, 0.95);
  const double p99_us = PercentileUs(elapsed_us_samples, 0.99);

  const double mixed_ops_per_round =
      static_cast<double>(iterations_per_round) * 2.0; // read + update
  const double key_ops_per_round =
      mixed_ops_per_round * static_cast<double>(batch_keys);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec =
      (mixed_ops_per_round * total_rounds) / (total_us / 1e6);
  const double key_ops_per_sec =
      (key_ops_per_round * total_rounds) / (total_us / 1e6);

  std::cout << "system=RecStore transport=" << transport << " phase=" << phase
            << " summary rounds=" << elapsed_us_samples.size() << " iterations="
            << iterations_per_round << " batch_keys=" << batch_keys
            << " num_embeddings=" << num_embeddings
            << " elapsed_us_mean=" << mean_us << " elapsed_us_p50=" << p50_us
            << " elapsed_us_p95=" << p95_us << " elapsed_us_p99=" << p99_us
            << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

void PrintInitSummary(const std::string& transport,
                      int64_t num_embeddings,
                      int init_chunk_size,
                      int64_t elapsed_us) {
  const double elapsed_s = static_cast<double>(elapsed_us) / 1e6;
  const double embeddings_per_sec =
      elapsed_s > 0.0 ? static_cast<double>(num_embeddings) / elapsed_s : 0.0;
  std::cout << "system=RecStore transport=" << transport
            << " phase=init summary rounds=1 iterations=1 batch_keys=0"
            << " num_embeddings=" << num_embeddings << " elapsed_us_mean="
            << elapsed_us << " elapsed_us_p50=" << elapsed_us
            << " elapsed_us_p95=" << elapsed_us << " elapsed_us_p99="
            << elapsed_us << " ops_per_sec=" << embeddings_per_sec
            << " key_ops_per_sec=" << embeddings_per_sec
            << " init_chunk_size=" << init_chunk_size << std::endl;
}

std::vector<uint64_t> MakeSequentialKeys(int64_t begin, int key_count) {
  CHECK_GE(begin, 0) << "key begin must be non-negative";
  CHECK_GT(key_count, 0) << "sequential key count must be positive";
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(key_count));
  for (int i = 0; i < key_count; ++i) {
    keys.push_back(static_cast<uint64_t>(1001 + begin + i));
  }
  return keys;
}

std::vector<uint64_t>
MakeBatchKeys(int64_t num_embeddings, int batch_keys, int64_t batch_offset) {
  CHECK_GT(num_embeddings, 0) << "--num_embeddings must be positive";
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    const int64_t idx = (batch_offset + i) % num_embeddings;
    keys.push_back(static_cast<uint64_t>(1001 + idx));
  }
  return keys;
}

base::RecTensor
MakeValues(const std::vector<uint64_t>& keys, int embedding_dim) {
  base::RecTensor values(
      {static_cast<int64_t>(keys.size()), embedding_dim},
      base::DataType::FLOAT32);
  float* dst = values.data_as<float>();
  for (size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      dst[row * static_cast<size_t>(embedding_dim) + static_cast<size_t>(col)] =
          static_cast<float>(keys[row]);
    }
  }
  return values;
}

std::vector<float> MakeFlatGradients(
    const std::vector<uint64_t>& keys, int embedding_dim, double update_scale) {
  std::vector<float> grads(
      static_cast<std::size_t>(keys.size()) *
          static_cast<std::size_t>(embedding_dim),
      0.0f);
  for (std::size_t row = 0; row < keys.size(); ++row) {
    const float value = static_cast<float>(keys[row] * update_scale);
    for (int col = 0; col < embedding_dim; ++col) {
      grads[row * static_cast<std::size_t>(embedding_dim) +
            static_cast<std::size_t>(col)] = value;
    }
  }
  return grads;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);

  const std::string transport   = NormalizeMixedTransport(FLAGS_transport);
  const std::string report_mode = FLAGS_report_mode;
  CHECK(report_mode == "summary" || report_mode == "per_round" ||
        report_mode == "both")
      << "Invalid --report_mode: " << report_mode
      << ", expected summary|per_round|both";
  CHECK_GT(FLAGS_embedding_dim, 0) << "--embedding_dim must be positive";
  CHECK_GT(FLAGS_num_embeddings, 0) << "--num_embeddings must be positive";
  CHECK_GT(FLAGS_init_chunk_size, 0) << "--init_chunk_size must be positive";
  CHECK_GE(FLAGS_num_embeddings, FLAGS_batch_keys)
      << "--num_embeddings must be >= --batch_keys";

  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;

  auto config = BuildMixedBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
  std::unique_ptr<recstore::BasePSClient> client(recstore::CreatePSClient(
      recstore::ResolvePSClientOptionsFromFrameworkConfig(config)));

  CHECK_EQ(client->InitEmbeddingTable(
               FLAGS_table_name,
               recstore::EmbeddingTableConfig{
                   static_cast<uint64_t>(FLAGS_num_embeddings),
                   static_cast<uint64_t>(FLAGS_embedding_dim)}),
           0)
      << transport << " InitEmbeddingTable failed";

  const auto init_start = std::chrono::steady_clock::now();
  for (int64_t begin = 0; begin < FLAGS_num_embeddings;
       begin += FLAGS_init_chunk_size) {
    const int chunk_size = static_cast<int>(
        std::min<int64_t>(FLAGS_init_chunk_size, FLAGS_num_embeddings - begin));
    const auto init_keys   = MakeSequentialKeys(begin, chunk_size);
    const auto init_values = MakeValues(init_keys, FLAGS_embedding_dim);
    CHECK(BenchmarkWriteSucceeded(
        transport,
        client->PutParameter(
            base::ConstArray<uint64_t>(init_keys), init_values)))
        << transport
        << " PutParameter failed during initialization at begin=" << begin;
  }
  const auto init_end = std::chrono::steady_clock::now();
  const int64_t init_elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          init_end - init_start)
          .count();

  std::vector<int64_t> warmup_samples_us;
  std::vector<int64_t> measure_samples_us;
  warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  measure_samples_us.reserve(std::max(0, FLAGS_rounds));

  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < FLAGS_warmup_rounds;
    auto start           = std::chrono::steady_clock::now();
    for (int i = 0; i < FLAGS_iterations; ++i) {
      const int64_t batch_offset =
          (static_cast<int64_t>(round) * FLAGS_iterations + i) *
          FLAGS_batch_keys;
      const auto keys =
          MakeBatchKeys(FLAGS_num_embeddings, FLAGS_batch_keys, batch_offset);
      const auto grads =
          MakeFlatGradients(keys, FLAGS_embedding_dim, FLAGS_update_scale);
      const auto key_array = base::ConstArray<uint64_t>(keys);
      base::RecTensor output(
          {static_cast<int64_t>(keys.size()), FLAGS_embedding_dim},
          base::DataType::FLOAT32);
      CHECK(BenchmarkReadSucceeded(
          transport, client->GetParameter(key_array, output)))
          << transport << " GetParameter failed at iteration=" << i;
      base::RecTensor grads_t(
          const_cast<float*>(grads.data()),
          {static_cast<int64_t>(keys.size()), FLAGS_embedding_dim});
      CHECK_EQ(client->UpdateParameter(FLAGS_table_name, key_array, grads_t), 0)
          << transport << " UpdateParameter failed at iteration=" << i;
    }
    auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? warmup_samples_us : measure_samples_us).push_back(elapsed_us);
    if (ShouldPrintPerRound(report_mode)) {
      std::cout << "system=RecStore transport=" << transport
                << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
                << (is_warmup ? (round + 1) : (round - FLAGS_warmup_rounds + 1))
                << "/" << (is_warmup ? FLAGS_warmup_rounds : FLAGS_rounds)
                << " elapsed_us=" << elapsed_us << std::endl;
    }
  }

  if (ShouldPrintSummary(report_mode)) {
    PrintInitSummary(transport,
                     FLAGS_num_embeddings,
                     FLAGS_init_chunk_size,
                     init_elapsed_us);
    PrintSummary(
        transport,
        "warmup",
        warmup_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        FLAGS_num_embeddings);
    PrintSummary(
        transport,
        "measure",
        measure_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        FLAGS_num_embeddings);
  }
  return 0;
}
