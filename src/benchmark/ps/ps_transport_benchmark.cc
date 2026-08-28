#include <folly/init/Init.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/bind_core.h"
#include "benchmark/ps/ps_transport_benchmark_config.h"
#include "framework/common/ps_client_config_adapter.h"
#include "ps/base/base_client.h"
#include "ps/client_factory.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/brpc/dist_brpc_ps_client.h"
#include "ps/grpc/dist_grpc_ps_client.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rc_options.h"
#include "ps/rdma/rdma_status.h"

DEFINE_string(transport, "rdma", "rdma|grpc|brpc");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 25000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_string(config_path, "", "full RecStore config path for PS client");
DEFINE_int32(iterations, 100, "number of get/put iterations");
DEFINE_int32(rounds, 1, "number of measured rounds");
DEFINE_int32(warmup_rounds, 0, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 4, "number of keys per put/get RPC");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DEFINE_string(workload, "micro", "benchmark workload: micro|transactions");
DEFINE_string(mode,
              "fetch",
              "transactions mode: fetch|insert|mixed|fetch_insert");
DEFINE_int64(record_count, 1000000, "record count for transactions workload");
DEFINE_int32(thread_num, 16, "worker thread count for transactions workload");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");
DEFINE_string(distribution, "uniform", "uniform|zipfian");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_int32(read_ratio, 100, "read percentage for mixed mode");
DEFINE_uint64(seed, 0x9e3779b97f4a7c15ULL, "base random seed");
DEFINE_bool(skip_load, false, "skip transactions preload phase");
DEFINE_bool(load_only, false, "run transactions preload phase and exit");
DEFINE_int32(prefetch_depth,
             0,
             "fetch-only prefetch pipeline depth for transactions; "
             "0 uses the RDMA default depth");
DEFINE_bool(transaction_profile,
            false,
            "print per-thread transaction timing breakdown");
DEFINE_bool(rdma_direct_async_fetch,
            false,
            "RDMA fetch-only transactions use petps::PetPSClient async GET "
            "with preallocated output buffers, bypassing the PS prefetch "
            "adapter state and result vector copy");
DEFINE_bool(verify_deterministic_values,
            false,
            "write key-derived rows during preload and verify fetched values");
DEFINE_int32(rdma_logical_client_id,
             -1,
             "Benchmark-only logical RDMA client id override for this process");
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(value_size);

namespace {

struct PhaseStats {
  uint64_t batches = 0;
  uint64_t key_ops = 0;
};

struct LocalShmTransportStats {
  uint32_t opcode             = 0;
  double acquire_slot_us      = 0.0;
  double enqueue_us           = 0.0;
  double wait_us              = 0.0;
  double release_us           = 0.0;
  double request_total_us     = 0.0;
  double server_queue_wait_us = 0.0;
  double server_backend_us    = 0.0;
  uint64_t samples            = 0;
};

using LocalShmTransportStatsByOpcode =
    std::map<uint32_t, LocalShmTransportStats>;

void AddLocalShmTransportSample(
    LocalShmTransportStats* stats,
    const recstore::LocalShmRequestProfile& profile) {
  if (stats == nullptr) {
    return;
  }
  stats->opcode = profile.opcode;
  stats->acquire_slot_us += profile.acquire_slot_us;
  stats->enqueue_us += profile.enqueue_us;
  stats->wait_us += profile.wait_us;
  stats->release_us += profile.release_us;
  stats->request_total_us += profile.request_total_us;
  stats->server_queue_wait_us += profile.server_queue_wait_us;
  stats->server_backend_us += profile.server_backend_us;
  ++stats->samples;
}

const char* LocalOpcodeLabel(uint32_t opcode) {
  switch (static_cast<recstore::LocalOpcode>(opcode)) {
  case recstore::LocalOpcode::kInitTable:
    return "INIT_TABLE";
  case recstore::LocalOpcode::kGet:
    return "GET";
  case recstore::LocalOpcode::kPut:
    return "PUT";
  case recstore::LocalOpcode::kUpdateFlat:
    return "UPDATE_FLAT";
  default:
    return "UNKNOWN";
  }
}

struct TransactionProfileStats {
  uint64_t make_keys_ns = 0;
  uint64_t submit_ns    = 0;
  uint64_t consume_ns   = 0;
  uint64_t wait_ns      = 0;
  uint64_t copy_ns      = 0;
  uint64_t iterations   = 0;
};

class FastRandom {
public:
  explicit FastRandom(uint64_t seed)
      : state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t Next() {
    uint64_t x = state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state_ = x;
    return x * 2685821657736338717ULL;
  }

  double Uniform01() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }
  uint64_t Uniform(uint64_t n) { return n == 0 ? 0 : Next() % n; }

private:
  uint64_t state_;
};

class KeyGenerator {
public:
  KeyGenerator(std::string distribution,
               uint64_t record_count,
               double alpha,
               uint64_t seed)
      : distribution_(std::move(distribution)),
        record_count_(record_count),
        alpha_(alpha),
        rng_(seed) {
    if (record_count_ == 0) {
      throw std::invalid_argument("record_count must be positive");
    }
    if (distribution_ == "zipfian") {
      if (std::abs(alpha_ - 1.0) < 1e-9) {
        log_n_ = std::log(static_cast<double>(record_count_));
      } else {
        pow_n_ = std::pow(static_cast<double>(record_count_), 1.0 - alpha_);
      }
    } else if (distribution_ != "uniform") {
      throw std::invalid_argument("distribution must be uniform or zipfian");
    }
  }

  uint64_t NextKey() {
    if (distribution_ == "uniform") {
      return rng_.Uniform(record_count_) + 1;
    }
    return NextZipfian() + 1;
  }

  uint64_t NextUint(uint64_t n) { return rng_.Uniform(n); }

private:
  uint64_t NextZipfian() {
    const double u = std::max(rng_.Uniform01(), 1e-12);
    double rank    = 1.0;
    if (std::abs(alpha_ - 1.0) < 1e-9) {
      rank = std::exp(u * log_n_);
    } else {
      rank = std::pow(1.0 + u * (pow_n_ - 1.0), 1.0 / (1.0 - alpha_));
    }
    uint64_t key = static_cast<uint64_t>(rank);
    if (key >= record_count_) {
      key = record_count_ - 1;
    }
    return key;
  }

  std::string distribution_;
  uint64_t record_count_;
  double alpha_;
  double pow_n_ = 1.0;
  double log_n_ = 0.0;
  FastRandom rng_;
};

std::vector<uint64_t> MakeKeys(int batch_keys) {
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    keys.push_back(static_cast<uint64_t>(1001 + i));
  }
  return keys;
}

base::RecTensor MakeValues(const std::vector<uint64_t>& keys) {
  const int dim = FLAGS_value_size / sizeof(float);
  base::RecTensor values(
      {static_cast<int64_t>(keys.size()), dim}, base::DataType::FLOAT32);
  float* dst = values.data_as<float>();
  for (size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < dim; ++col) {
      dst[row * static_cast<size_t>(dim) + static_cast<size_t>(col)] =
          static_cast<float>(keys[row]);
    }
  }
  return values;
}

std::vector<float> MakeFlatValues(size_t rows, int dim, int seed) {
  std::vector<float> values(rows * static_cast<size_t>(dim));
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((static_cast<int>(i) + seed) % 101);
  }
  return values;
}

float DeterministicValueForKey(uint64_t key, int column) {
  return static_cast<float>(
      (key % 1000003ULL) * 0.001 + static_cast<double>(column) * 0.0001);
}

std::vector<float>
MakeDeterministicFlatValues(const std::vector<uint64_t>& keys, int dim) {
  std::vector<float> values(keys.size() * static_cast<size_t>(dim));
  for (size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < dim; ++col) {
      values[row * static_cast<size_t>(dim) + static_cast<size_t>(col)] =
          DeterministicValueForKey(keys[row], col);
    }
  }
  return values;
}

void VerifyDeterministicFlatValues(const std::vector<uint64_t>& keys,
                                   const std::vector<float>& output,
                                   int dim) {
  CHECK_GE(output.size(), keys.size() * static_cast<size_t>(dim));
  for (size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < dim; ++col) {
      const float expected = DeterministicValueForKey(keys[row], col);
      const float actual =
          output[row * static_cast<size_t>(dim) + static_cast<size_t>(col)];
      CHECK(std::abs(actual - expected) <= 1e-6f)
          << "deterministic value mismatch key=" << keys[row] << " col=" << col
          << " expected=" << expected << " actual=" << actual;
    }
  }
}

bool ClientReturnsZeroOnSuccess(recstore::BasePSClient* client) {
  (void)client;
  return true;
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

void MaybePrintPerRound(
    const std::string& transport,
    const std::string& op,
    const std::string& report_mode,
    bool is_warmup,
    int round,
    int warmup_rounds,
    int rounds,
    int64_t elapsed_us) {
  if (!ShouldPrintPerRound(report_mode)) {
    return;
  }
  std::cout << "transport=" << transport << " op=" << op
            << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
            << (is_warmup ? (round + 1) : (round - warmup_rounds + 1)) << "/"
            << (is_warmup ? warmup_rounds : rounds)
            << " elapsed_us=" << elapsed_us << std::endl;
}

template <typename IterationFn>
void RunOperationRounds(
    const std::string& transport,
    const std::string& op,
    int total_rounds,
    int warmup_rounds,
    int rounds,
    int iterations,
    const std::string& report_mode,
    IterationFn&& run_iteration,
    std::vector<int64_t>* warmup_samples_us,
    std::vector<int64_t>* measure_samples_us) {
  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < warmup_rounds;
    auto start           = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      run_iteration(i);
    }
    auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? *warmup_samples_us : *measure_samples_us)
        .push_back(elapsed_us);
    MaybePrintPerRound(
        transport,
        op,
        report_mode,
        is_warmup,
        round,
        warmup_rounds,
        rounds,
        elapsed_us);
  }
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

void PrintSummary(
    const std::string& transport,
    const std::string& op,
    const std::string& phase,
    const std::vector<int64_t>& elapsed_us_samples,
    int iterations_per_round,
    int batch_keys,
    std::size_t keys_per_iteration) {
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

  const double ops_per_round = static_cast<double>(iterations_per_round);
  const double key_ops_per_round =
      ops_per_round * static_cast<double>(keys_per_iteration);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec  = (ops_per_round * total_rounds) / (total_us / 1e6);
  const double key_ops_per_sec =
      (key_ops_per_round * total_rounds) / (total_us / 1e6);

  std::cout << "transport=" << transport << " op=" << op << " phase=" << phase
            << " summary rounds=" << elapsed_us_samples.size()
            << " iterations=" << iterations_per_round
            << " batch_keys=" << batch_keys << " elapsed_us_mean=" << mean_us
            << " elapsed_us_p50=" << p50_us << " elapsed_us_p95=" << p95_us
            << " elapsed_us_p99=" << p99_us << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

nlohmann::json LoadClientConfig(const std::string& transport) {
  auto attach_logical_client_id = [](nlohmann::json* config) {
    if (FLAGS_rdma_logical_client_id >= 0) {
      (*config)["rdma_logical_client_id"] = FLAGS_rdma_logical_client_id;
    }
  };
  if (!FLAGS_config_path.empty()) {
    std::ifstream in(FLAGS_config_path);
    CHECK(in.good()) << "failed to open --config_path=" << FLAGS_config_path;
    nlohmann::json config;
    in >> config;
    if (NormalizeBenchmarkTransport(transport) == "RDMA") {
      attach_logical_client_id(&config);
    }
    return config;
  }
  nlohmann::json config =
      BuildRpcBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
  if (NormalizeBenchmarkTransport(transport) == "RDMA") {
    attach_logical_client_id(&config);
  }
  return config;
}

std::unique_ptr<recstore::BasePSClient>
CreateBenchmarkClient(const std::string& transport) {
  auto config           = LoadClientConfig(transport);
  const auto normalized = NormalizeBenchmarkTransport(transport);
  if (FLAGS_num_shards > 1 && normalized == "BRPC") {
    return std::make_unique<recstore::DistributedBRPCParameterClient>(config);
  }
  if (FLAGS_num_shards > 1 && normalized == "GRPC") {
    return std::make_unique<recstore::DistributedGRPCParameterClient>(config);
  }
  return recstore::CreatePSClient(
      recstore::ResolvePSClientOptionsFromFrameworkConfig(config));
}

bool PutFlat(recstore::BasePSClient* client,
             const std::string& transport,
             const std::vector<uint64_t>& keys,
             const std::vector<float>& flat_values,
             int dim) {
  base::RecTensor values(const_cast<float*>(flat_values.data()),
                         {static_cast<int64_t>(keys.size()), dim});
  return BenchmarkWriteSucceeded(
      transport,
      client->PutParameter(base::ConstArray<uint64_t>(keys), values),
      ClientReturnsZeroOnSuccess(client));
}

bool GetFlat(recstore::BasePSClient* client,
             const std::string& transport,
             const std::vector<uint64_t>& keys,
             std::vector<float>* output) {
  const auto key_array = base::ConstArray<uint64_t>(keys);
  const int dim        = FLAGS_value_size / sizeof(float);
  const size_t needed  = keys.size() * static_cast<size_t>(dim);
  if (output->size() < needed) {
    output->assign(needed, 0.0f);
  }
  base::RecTensor values(output->data(),
                         {static_cast<int64_t>(keys.size()), dim});
  return BenchmarkReadSucceeded(
      transport,
      client->GetParameter(key_array, values),
      ClientReturnsZeroOnSuccess(client));
}

void AccumulateLocalShmTransportStats(
    recstore::BasePSClient* client,
    LocalShmTransportStats* stats,
    LocalShmTransportStatsByOpcode* by_opcode = nullptr) {
  if (stats == nullptr && by_opcode == nullptr) {
    return;
  }
  auto* local_client = dynamic_cast<recstore::LocalShmPSClient*>(client);
  if (local_client == nullptr) {
    return;
  }
  const auto profile = local_client->GetLastRequestProfile();
  AddLocalShmTransportSample(stats, profile);
  if (by_opcode != nullptr) {
    AddLocalShmTransportSample(&(*by_opcode)[profile.opcode], profile);
  }
}

void PrintLocalShmTransportStats(const char* phase,
                                 const LocalShmTransportStats& stats) {
  if (stats.samples == 0) {
    return;
  }
  const double samples = static_cast<double>(stats.samples);
  std::cout
      << "PS_LOCAL_SHM_PROFILE phase=" << phase << " samples=" << stats.samples
      << " acquire_slot_us_mean=" << (stats.acquire_slot_us / samples)
      << " enqueue_us_mean=" << (stats.enqueue_us / samples)
      << " wait_us_mean=" << (stats.wait_us / samples)
      << " release_us_mean=" << (stats.release_us / samples)
      << " request_total_us_mean=" << (stats.request_total_us / samples)
      << " server_queue_wait_us_mean=" << (stats.server_queue_wait_us / samples)
      << " server_backend_us_mean=" << (stats.server_backend_us / samples)
      << " opcode=" << LocalOpcodeLabel(stats.opcode) << std::endl;
}

void PrintLocalShmTransportStatsByOpcode(
    const char* phase, const LocalShmTransportStatsByOpcode& by_opcode) {
  for (const auto& [opcode, stats] : by_opcode) {
    if (stats.samples == 0) {
      continue;
    }
    const double samples = static_cast<double>(stats.samples);
    std::cout << "PS_LOCAL_SHM_PROFILE_OPCODE phase=" << phase << " opcode="
              << LocalOpcodeLabel(opcode) << " samples=" << stats.samples
              << " acquire_slot_us_mean=" << (stats.acquire_slot_us / samples)
              << " enqueue_us_mean=" << (stats.enqueue_us / samples)
              << " wait_us_mean=" << (stats.wait_us / samples)
              << " release_us_mean=" << (stats.release_us / samples)
              << " request_total_us_mean=" << (stats.request_total_us / samples)
              << " server_queue_wait_us_mean="
              << (stats.server_queue_wait_us / samples)
              << " server_backend_us_mean="
              << (stats.server_backend_us / samples) << std::endl;
  }
}

bool PrefetchFlat(recstore::BasePSClient* client,
                  const std::vector<uint64_t>& keys,
                  uint64_t* prefetch_id) {
  if (prefetch_id == nullptr) {
    return false;
  }
  *prefetch_id = client->PrefetchParameter(base::ConstArray<uint64_t>(keys));
  return *prefetch_id != 0;
}

bool ConsumePrefetchFlat(
    recstore::BasePSClient* client,
    uint64_t prefetch_id,
    std::vector<float>* output,
    int64_t* num_rows,
    int dim) {
  if (output == nullptr || num_rows == nullptr) {
    return false;
  }
  base::RecTensor fetched({0, dim}, base::DataType::FLOAT32);
  if (!client->GetPrefetchResult(prefetch_id, fetched)) {
    return false;
  }
  *num_rows = fetched.shape(0);
  if (fetched.data() == nullptr || fetched.num_elements() == 0) {
    output->clear();
    return true;
  }
  output->assign(fetched.data_as<float>(),
                 fetched.data_as<float>() + fetched.num_elements());
  return true;
}

int64_t NsSince(std::chrono::steady_clock::time_point start,
                std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

using BenchmarkClient = std::unique_ptr<recstore::BasePSClient>;

struct DirectRdmaClient {
  std::unique_ptr<recstore::RDMAPSClientAdapter> adapter;
  std::unique_ptr<recstore::RdmaRawAccess> raw;
};

DirectRdmaClient
CreateDirectRdmaClientFromConfig(const nlohmann::json& config) {
  DirectRdmaClient direct;
  direct.adapter = std::make_unique<recstore::RDMAPSClientAdapter>(config);
  direct.raw = std::make_unique<recstore::RdmaRawAccess>(direct.adapter.get());
  return direct;
}

PhaseStats LoadRecords(
    const std::string& transport,
    int load_threads,
    int dim,
    std::vector<BenchmarkClient>* reusable_clients            = nullptr,
    LocalShmTransportStats* local_shm_stats                   = nullptr,
    LocalShmTransportStatsByOpcode* local_shm_stats_by_opcode = nullptr) {
  if (reusable_clients != nullptr) {
    CHECK_EQ(static_cast<int>(reusable_clients->size()), load_threads);
  }
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(load_threads);
  const uint64_t record_count = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t per_thread =
      (record_count + static_cast<uint64_t>(load_threads) - 1) /
      static_cast<uint64_t>(load_threads);

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      auto owned_client =
          reusable_clients == nullptr
              ? CreateBenchmarkClient(transport)
              : nullptr;
      recstore::BasePSClient* client =
          reusable_clients == nullptr
              ? owned_client.get()
              : reusable_clients->at(static_cast<std::size_t>(tid)).get();
      std::vector<uint64_t> keys;
      keys.reserve(static_cast<size_t>(FLAGS_batch_keys));
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread + 1;
      const uint64_t end   = std::min(record_count + 1, begin + per_thread);
      PhaseStats local;
      for (uint64_t key = begin; key < end;) {
        keys.clear();
        while (key < end &&
               keys.size() < static_cast<size_t>(FLAGS_batch_keys)) {
          keys.push_back(key++);
        }
        std::vector<float> values =
            FLAGS_verify_deterministic_values
                ? MakeDeterministicFlatValues(keys, dim)
                : MakeFlatValues(keys.size(), dim, tid);
        CHECK(PutFlat(client, transport, keys, values, dim))
            << transport << " preload PutParameter failed";
        AccumulateLocalShmTransportStats(
            client, local_shm_stats, local_shm_stats_by_opcode);
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  return total;
}

PhaseStats RunTransactions(
    const std::string& transport,
    int dim,
    std::vector<BenchmarkClient>* reusable_clients            = nullptr,
    LocalShmTransportStats* local_shm_stats                   = nullptr,
    LocalShmTransportStatsByOpcode* local_shm_stats_by_opcode = nullptr) {
  if (reusable_clients != nullptr) {
    CHECK_EQ(static_cast<int>(reusable_clients->size()), FLAGS_thread_num);
  }
  const bool fetch_only   = FLAGS_mode == "fetch";
  const bool insert_only  = FLAGS_mode == "insert";
  const bool mixed        = FLAGS_mode == "mixed";
  const bool fetch_insert = FLAGS_mode == "fetch_insert";
  CHECK(fetch_only || insert_only || mixed || fetch_insert)
      << "mode must be fetch|insert|mixed|fetch_insert";

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      auto owned_client =
          reusable_clients == nullptr
              ? CreateBenchmarkClient(transport)
              : nullptr;
      recstore::BasePSClient* client =
          reusable_clients == nullptr
              ? owned_client.get()
              : reusable_clients->at(static_cast<std::size_t>(tid)).get();
      KeyGenerator key_gen(
          FLAGS_distribution,
          static_cast<uint64_t>(FLAGS_record_count),
          FLAGS_zipfian_alpha,
          FLAGS_seed + static_cast<uint64_t>(tid));
      std::vector<uint64_t> keys(static_cast<size_t>(FLAGS_batch_keys));
      std::vector<float> values =
          MakeFlatValues(static_cast<size_t>(FLAGS_batch_keys), dim, tid);
      std::vector<float> output(
          static_cast<size_t>(FLAGS_batch_keys) * static_cast<size_t>(dim));
      PhaseStats local;
      while (!start.load(std::memory_order_acquire)) {
      }
      while (!stop.load(std::memory_order_relaxed)) {
        for (auto& key : keys) {
          key = key_gen.NextKey();
        }
        const bool do_fetch =
            fetch_only || fetch_insert ||
            (mixed &&
             static_cast<int>(key_gen.NextUint(100)) < FLAGS_read_ratio);
        if (do_fetch) {
          CHECK(GetFlat(client, transport, keys, &output))
              << transport << " GetParameter failed";
          if (FLAGS_verify_deterministic_values) {
            VerifyDeterministicFlatValues(keys, output, dim);
          }
          AccumulateLocalShmTransportStats(
              client, local_shm_stats, local_shm_stats_by_opcode);
        }
        if (insert_only || fetch_insert || (mixed && !do_fetch)) {
          CHECK(PutFlat(client, transport, keys, values, dim))
              << transport << " PutParameter failed";
          AccumulateLocalShmTransportStats(
              client, local_shm_stats, local_shm_stats_by_opcode);
        }
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }

  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::seconds(FLAGS_running_seconds));
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  return total;
}

PhaseStats RunPrefetchFetchTransactions(
    const std::string& transport,
    int dim,
    int prefetch_depth,
    std::vector<BenchmarkClient>* reusable_clients = nullptr) {
  CHECK_GT(prefetch_depth, 0);
  CHECK_EQ(FLAGS_mode, "fetch")
      << "--prefetch_depth currently supports fetch mode only";
  if (reusable_clients != nullptr) {
    CHECK_EQ(static_cast<int>(reusable_clients->size()), FLAGS_thread_num);
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);
  std::vector<TransactionProfileStats> profile_stats(FLAGS_thread_num);

  struct PendingFetch {
    uint64_t prefetch_id = 0;
    std::vector<uint64_t> keys;
  };

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      auto owned_client =
          reusable_clients == nullptr
              ? CreateBenchmarkClient(transport)
              : nullptr;
      recstore::BasePSClient* client =
          reusable_clients == nullptr
              ? owned_client.get()
              : reusable_clients->at(static_cast<std::size_t>(tid)).get();
      CHECK_EQ(client->InitEmbeddingTable(
                   "benchmark",
                   recstore::EmbeddingTableConfig{
                       static_cast<uint64_t>(FLAGS_record_count),
                       static_cast<uint64_t>(dim)}),
               0);
      KeyGenerator key_gen(
          FLAGS_distribution,
          static_cast<uint64_t>(FLAGS_record_count),
          FLAGS_zipfian_alpha,
          FLAGS_seed + static_cast<uint64_t>(tid));
      std::deque<PendingFetch> pending;
      std::vector<float> output;
      int64_t num_rows = 0;
      PhaseStats local;
      TransactionProfileStats local_profile;

      auto make_keys = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        std::vector<uint64_t> keys(static_cast<size_t>(FLAGS_batch_keys));
        for (auto& key : keys) {
          key = key_gen.NextKey();
        }
        const auto end = std::chrono::steady_clock::now();
        local_profile.make_keys_ns +=
            static_cast<uint64_t>(NsSince(begin, end));
        return keys;
      };
      auto submit_one = [&]() {
        PendingFetch fetch;
        fetch.keys              = make_keys();
        const auto submit_begin = std::chrono::steady_clock::now();
        CHECK(PrefetchFlat(client, fetch.keys, &fetch.prefetch_id))
            << transport << " PrefetchParameter failed";
        const auto submit_end = std::chrono::steady_clock::now();
        local_profile.submit_ns +=
            static_cast<uint64_t>(NsSince(submit_begin, submit_end));
        pending.push_back(std::move(fetch));
      };
      auto consume_front = [&]() {
        PendingFetch fetch = std::move(pending.front());
        pending.pop_front();
        const auto consume_begin = std::chrono::steady_clock::now();
        CHECK(ConsumePrefetchFlat(
            client, fetch.prefetch_id, &output, &num_rows, dim))
            << transport << " GetPrefetchResult failed";
        const auto consume_end = std::chrono::steady_clock::now();
        CHECK_EQ(num_rows, static_cast<int64_t>(fetch.keys.size()));
        if (FLAGS_verify_deterministic_values) {
          VerifyDeterministicFlatValues(fetch.keys, output, dim);
        }
        local_profile.consume_ns +=
            static_cast<uint64_t>(NsSince(consume_begin, consume_end));
        local_profile.wait_ns +=
            static_cast<uint64_t>(NsSince(consume_begin, consume_end));
        ++local_profile.iterations;
        ++local.batches;
        local.key_ops += fetch.keys.size();
      };

      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < prefetch_depth; ++i) {
        submit_one();
      }
      while (!stop.load(std::memory_order_relaxed)) {
        consume_front();
        submit_one();
      }
      while (!pending.empty()) {
        consume_front();
      }
      stats[tid]         = local;
      profile_stats[tid] = local_profile;
    });
  }

  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::seconds(FLAGS_running_seconds));
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  if (FLAGS_transaction_profile) {
    TransactionProfileStats total_profile;
    for (const auto& each : profile_stats) {
      total_profile.make_keys_ns += each.make_keys_ns;
      total_profile.submit_ns += each.submit_ns;
      total_profile.consume_ns += each.consume_ns;
      total_profile.wait_ns += each.wait_ns;
      total_profile.copy_ns += each.copy_ns;
      total_profile.iterations += each.iterations;
    }
    const double denom =
        total_profile.iterations == 0
            ? 1.0
            : static_cast<double>(total_profile.iterations);
    std::cout << "PS_BENCHMARK_PROFILE phase=run transport=" << transport
              << " mode=" << FLAGS_mode << " prefetch_depth=" << prefetch_depth
              << " batches=" << total_profile.iterations << " make_keys_avg_ns="
              << static_cast<double>(total_profile.make_keys_ns) / denom
              << " submit_avg_ns="
              << static_cast<double>(total_profile.submit_ns) / denom
              << " consume_avg_ns="
              << static_cast<double>(total_profile.consume_ns) / denom
              << " wait_plus_result_avg_ns="
              << static_cast<double>(total_profile.wait_ns) / denom
              << std::endl;
  }
  return total;
}

PhaseStats RunRdmaDirectAsyncFetchTransactions(int dim, int prefetch_depth) {
  CHECK_GT(prefetch_depth, 0);
  CHECK_EQ(FLAGS_mode, "fetch")
      << "--rdma_direct_async_fetch currently supports fetch mode only";
  CHECK_EQ(FLAGS_thread_num, 1)
      << "RDMA direct async fetch supports one worker thread per benchmark "
         "process; scale with client processes";

  const nlohmann::json config = LoadClientConfig("RDMA");
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);
  std::vector<TransactionProfileStats> profile_stats(FLAGS_thread_num);

  struct DirectSlot {
    int rpc_id = -1;
    std::vector<uint64_t> keys;
    std::vector<float> output;
  };

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      DirectRdmaClient direct     = CreateDirectRdmaClientFromConfig(config);
      recstore::RdmaRawAccess* raw = direct.raw.get();
      CHECK_NE(raw, nullptr);
      KeyGenerator key_gen(
          FLAGS_distribution,
          static_cast<uint64_t>(FLAGS_record_count),
          FLAGS_zipfian_alpha,
          FLAGS_seed + static_cast<uint64_t>(tid));

      const std::size_t response_floats =
          static_cast<std::size_t>(FLAGS_batch_keys) *
              static_cast<std::size_t>(dim) +
          1;
      std::vector<DirectSlot> slots(static_cast<std::size_t>(prefetch_depth));
      for (auto& slot : slots) {
        slot.keys.resize(static_cast<std::size_t>(FLAGS_batch_keys));
        slot.output.resize(response_floats);
      }

      PhaseStats local;
      TransactionProfileStats local_profile;

      auto fill_keys = [&](std::vector<uint64_t>* keys) {
        const auto begin = std::chrono::steady_clock::now();
        for (auto& key : *keys) {
          key = key_gen.NextKey();
        }
        const auto end = std::chrono::steady_clock::now();
        local_profile.make_keys_ns +=
            static_cast<uint64_t>(NsSince(begin, end));
      };

      auto submit_slot = [&](DirectSlot* slot) {
        fill_keys(&slot->keys);
        const auto submit_begin = std::chrono::steady_clock::now();
        slot->rpc_id            = raw->SubmitGetParameter(
            base::ConstArray<uint64_t>(slot->keys),
            slot->output.data(),
            true,
            0,
            dim);
        const auto submit_end = std::chrono::steady_clock::now();
        local_profile.submit_ns +=
            static_cast<uint64_t>(NsSince(submit_begin, submit_end));
      };

      auto consume_slot = [&](DirectSlot* slot) {
        const auto consume_begin = std::chrono::steady_clock::now();
        raw->WaitRPCFinish(slot->rpc_id);
        const auto* status_word = petps::FixedSlotStatusWord(
            slot->output.data(), slot->keys.size(), FLAGS_value_size);
        CHECK_EQ(*status_word, static_cast<std::int32_t>(petps::RpcStatus::kOk))
            << "RDMA direct async fetch failed with status=" << *status_word;
        if (FLAGS_verify_deterministic_values) {
          VerifyDeterministicFlatValues(slot->keys, slot->output, dim);
        }
        raw->RevokeRPCResource(slot->rpc_id);
        const auto consume_end = std::chrono::steady_clock::now();
        local_profile.consume_ns +=
            static_cast<uint64_t>(NsSince(consume_begin, consume_end));
        local_profile.wait_ns +=
            static_cast<uint64_t>(NsSince(consume_begin, consume_end));
        ++local_profile.iterations;
        ++local.batches;
        local.key_ops += slot->keys.size();
      };

      while (!start.load(std::memory_order_acquire)) {
      }
      for (auto& slot : slots) {
        submit_slot(&slot);
      }
      std::size_t next_slot = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        DirectSlot& slot = slots[next_slot];
        consume_slot(&slot);
        submit_slot(&slot);
        next_slot = (next_slot + 1) % slots.size();
      }
      for (auto& slot : slots) {
        consume_slot(&slot);
      }

      stats[tid]         = local;
      profile_stats[tid] = local_profile;
    });
  }

  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::seconds(FLAGS_running_seconds));
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  if (FLAGS_transaction_profile) {
    TransactionProfileStats total_profile;
    for (const auto& each : profile_stats) {
      total_profile.make_keys_ns += each.make_keys_ns;
      total_profile.submit_ns += each.submit_ns;
      total_profile.consume_ns += each.consume_ns;
      total_profile.wait_ns += each.wait_ns;
      total_profile.iterations += each.iterations;
    }
    const double denom =
        total_profile.iterations == 0
            ? 1.0
            : static_cast<double>(total_profile.iterations);
    std::cout
        << "PS_BENCHMARK_PROFILE phase=run transport=RDMA"
        << " mode=" << FLAGS_mode << " prefetch_depth=" << prefetch_depth
        << " direct_async_fetch=1"
        << " batches=" << total_profile.iterations << " make_keys_avg_ns="
        << static_cast<double>(total_profile.make_keys_ns) / denom
        << " submit_avg_ns="
        << static_cast<double>(total_profile.submit_ns) / denom
        << " consume_avg_ns="
        << static_cast<double>(total_profile.consume_ns) / denom
        << " wait_plus_result_avg_ns="
        << static_cast<double>(total_profile.wait_ns) / denom << std::endl;
  }
  return total;
}

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

void PrintTransactionResult(const char* phase,
                            const std::string& transport,
                            const PhaseStats& stats,
                            double seconds) {
  const double batch_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.batches) / seconds : 0.0;
  const double key_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.key_ops) / seconds : 0.0;
  std::cout << "PS_BENCHMARK_RESULT phase=" << phase
            << " transport=" << transport << " mode=" << FLAGS_mode
            << " distribution=" << FLAGS_distribution << " zipfian_alpha="
            << FLAGS_zipfian_alpha << " threads=" << FLAGS_thread_num
            << " batch_size=" << FLAGS_batch_keys
            << " records=" << FLAGS_record_count << " runtime_s=" << seconds
            << " batches=" << stats.batches << " key_ops=" << stats.key_ops
            << " throughput_batches_sec=" << batch_ops_sec
            << " throughput_keys_sec=" << key_ops_sec << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);

  const std::string transport = NormalizeBenchmarkTransport(FLAGS_transport);
  if (transport == "RDMA") {
    base::global_socket_id = FLAGS_rdma_rc_client_numa_id;
  }
  const std::string report_mode = FLAGS_report_mode;
  CHECK(report_mode == "summary" || report_mode == "per_round" ||
        report_mode == "both")
      << "Invalid --report_mode: " << report_mode
      << ", expected summary|per_round|both";
  CHECK_GT(FLAGS_value_size, 0) << "--value_size must be positive";
  CHECK_EQ(FLAGS_value_size % static_cast<int>(sizeof(float)), 0)
      << "--value_size must be divisible by sizeof(float)";
  const int dim = FLAGS_value_size / sizeof(float);
  CHECK_GE(FLAGS_prefetch_depth, 0) << "--prefetch_depth must be non-negative";
  if (FLAGS_prefetch_depth > 0) {
    CHECK_EQ(FLAGS_workload, "transactions")
        << "--prefetch_depth is only valid for transactions workload";
    CHECK_EQ(FLAGS_mode, "fetch")
        << "--prefetch_depth currently supports fetch mode only";
  }
  if (FLAGS_rdma_direct_async_fetch) {
    CHECK_EQ(FLAGS_workload, "transactions")
        << "--rdma_direct_async_fetch is only valid for transactions workload";
    CHECK_EQ(transport, "RDMA")
        << "--rdma_direct_async_fetch is only valid for RDMA";
    CHECK_EQ(FLAGS_mode, "fetch")
        << "--rdma_direct_async_fetch currently supports fetch mode only";
  }

  if (FLAGS_workload == "transactions") {
    CHECK_GT(FLAGS_record_count, 0);
    CHECK_GT(FLAGS_batch_keys, 0);
    CHECK_GT(FLAGS_thread_num, 0);
    CHECK_GT(FLAGS_running_seconds, 0);
    const int load_threads =
        FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;
    std::vector<BenchmarkClient> reusable_clients;
    if (transport == "RDMA") {
      CHECK_EQ(load_threads, FLAGS_thread_num)
          << "RDMA transactions must reuse the same client for load and run";
      reusable_clients.reserve(static_cast<std::size_t>(FLAGS_thread_num));
      for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
        FLAGS_rdma_logical_client_id =
            FLAGS_rdma_rc_client_id_base >= 0
                ? FLAGS_rdma_rc_client_id_base + tid
                : FLAGS_global_id - FLAGS_num_server_processes + tid;
        reusable_clients.push_back(CreateBenchmarkClient(transport));
      }
      FLAGS_rdma_logical_client_id = -1;
    }
    if (!FLAGS_skip_load) {
      LocalShmTransportStats load_transport_stats;
      LocalShmTransportStatsByOpcode load_transport_stats_by_opcode;
      const auto load_begin = std::chrono::steady_clock::now();
      const PhaseStats load = LoadRecords(
          transport,
          load_threads,
          dim,
          reusable_clients.empty() ? nullptr : &reusable_clients,
          &load_transport_stats,
          &load_transport_stats_by_opcode);
      const auto load_end = std::chrono::steady_clock::now();
      PrintTransactionResult(
          "load", transport, load, SecondsSince(load_begin, load_end));
      if (transport == "LOCAL_SHM") {
        PrintLocalShmTransportStats("load", load_transport_stats);
        PrintLocalShmTransportStatsByOpcode(
            "load", load_transport_stats_by_opcode);
      }
    }
    if (FLAGS_load_only) {
      return 0;
    }
    if (FLAGS_rdma_direct_async_fetch) {
      reusable_clients.clear();
    }

    LocalShmTransportStats run_transport_stats;
    LocalShmTransportStatsByOpcode run_transport_stats_by_opcode;
    const auto run_begin = std::chrono::steady_clock::now();
    const int effective_prefetch_depth =
        FLAGS_prefetch_depth > 0
            ? FLAGS_prefetch_depth
            : (transport == "RDMA" && FLAGS_mode == "fetch" ? 16 : 0);
    if (transport == "RDMA" && FLAGS_mode == "fetch" &&
        effective_prefetch_depth > 0) {
      CHECK_GT(FLAGS_rdma_rc_qps_per_client_per_shard, 0)
          << "--rdma_rc_qps_per_client_per_shard must be positive";
      CHECK_GT(FLAGS_rdma_rc_slots_per_qp, 0)
          << "--rdma_rc_slots_per_qp must be positive";
      const int slot_capacity =
          FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp;
      CHECK_LE(effective_prefetch_depth, slot_capacity)
          << "RDMA fetch prefetch depth exceeds RC slot capacity; "
          << "prefetch_depth=" << effective_prefetch_depth
          << ", qps_per_client_per_shard="
          << FLAGS_rdma_rc_qps_per_client_per_shard
          << ", slots_per_qp=" << FLAGS_rdma_rc_slots_per_qp;
    }
    const PhaseStats run =
        FLAGS_rdma_direct_async_fetch
            ? RunRdmaDirectAsyncFetchTransactions(
                  dim, std::max(1, effective_prefetch_depth))
        : effective_prefetch_depth > 0
            ? RunPrefetchFetchTransactions(
                  transport,
                  dim,
                  effective_prefetch_depth,
                  reusable_clients.empty() ? nullptr : &reusable_clients)
            : RunTransactions(
                  transport,
                  dim,
                  reusable_clients.empty() ? nullptr : &reusable_clients,
                  &run_transport_stats,
                  &run_transport_stats_by_opcode);
    const auto run_end = std::chrono::steady_clock::now();
    PrintTransactionResult(
        "run", transport, run, SecondsSince(run_begin, run_end));
    if (transport == "LOCAL_SHM") {
      PrintLocalShmTransportStats("run", run_transport_stats);
      PrintLocalShmTransportStatsByOpcode("run", run_transport_stats_by_opcode);
    }
    return 0;
  }
  CHECK_EQ(FLAGS_workload, "micro") << "workload must be micro|transactions";

  const auto keys        = MakeKeys(FLAGS_batch_keys);
  const auto values      = MakeValues(keys);
  const auto key_array   = base::ConstArray<uint64_t>(keys);
  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;

  if (transport == "RDMA") {
    if (FLAGS_num_shards == 1) {
      petps::PetPSClient client(FLAGS_host, FLAGS_port, 0);
      client.InitThread();
      void* recv_buffer =
          client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
      std::vector<int64_t> put_warmup_samples_us;
      std::vector<int64_t> put_measure_samples_us;
      put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
      put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
      RunOperationRounds(
          "RDMA",
          "put",
          total_rounds,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          FLAGS_iterations,
          report_mode,
          [&](int iteration) {
            CHECK_EQ(client.PutParameter(keys, values), 0)
                << "RDMA PutParameter failed at iteration=" << iteration;
          },
          &put_warmup_samples_us,
          &put_measure_samples_us);

      std::vector<int64_t> get_warmup_samples_us;
      std::vector<int64_t> get_measure_samples_us;
      get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
      get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
      RunOperationRounds(
          "RDMA",
          "get",
          total_rounds,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          FLAGS_iterations,
          report_mode,
          [&](int iteration) {
            int rpc_id = client.GetParameter(
                key_array, static_cast<float*>(recv_buffer), false, 0);
            client.WaitRPCFinish(rpc_id);
            client.RevokeRPCResource(rpc_id);
            (void)iteration;
          },
          &get_warmup_samples_us,
          &get_measure_samples_us);
      if (ShouldPrintSummary(report_mode)) {
        PrintSummary(
            "RDMA",
            "put",
            "warmup",
            put_warmup_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "put",
            "measure",
            put_measure_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "get",
            "warmup",
            get_warmup_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
        PrintSummary(
            "RDMA",
            "get",
            "measure",
            get_measure_samples_us,
            FLAGS_iterations,
            FLAGS_batch_keys,
            keys.size());
      }
      return 0;
    }

    nlohmann::json config      = nlohmann::json::object();
    config["client"] = {{"host", FLAGS_host}, {"port", FLAGS_port}, {"shard", 0}};
    nlohmann::json servers     = nlohmann::json::array();
    for (int shard = 0; shard < FLAGS_num_shards; ++shard) {
      servers.push_back(
          {{"host", FLAGS_host}, {"port", FLAGS_port}, {"shard", shard}});
    }
    config["distributed_client"] = {
        {"num_shards", FLAGS_num_shards},
        {"hash_method", "city_hash"},
        {"servers", servers},
    };
    auto adapter = std::make_unique<recstore::RDMAPSClientAdapter>(config);
    auto raw     = std::make_unique<recstore::RdmaRawAccess>(adapter.get());
    std::vector<int64_t> put_warmup_samples_us;
    std::vector<int64_t> put_measure_samples_us;
    put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
    put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
    RunOperationRounds(
        "RDMA",
        "put",
        total_rounds,
        FLAGS_warmup_rounds,
        FLAGS_rounds,
        FLAGS_iterations,
        report_mode,
        [&](int iteration) {
          CHECK_EQ(adapter->PutParameter(key_array, values), 0)
              << "RDMA(all-shards) PutParameter failed at iteration="
              << iteration;
        },
        &put_warmup_samples_us,
        &put_measure_samples_us);

    std::vector<int64_t> get_warmup_samples_us;
    std::vector<int64_t> get_measure_samples_us;
    get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
    get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
    RunOperationRounds(
        "RDMA",
        "get",
        total_rounds,
        FLAGS_warmup_rounds,
        FLAGS_rounds,
        FLAGS_iterations,
        report_mode,
        [&](int iteration) {
          const int dim = FLAGS_value_size / sizeof(float);
          std::vector<float> output(
              keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
          int rpc_id = raw->SubmitGetParameter(
              key_array, output.data(), false, 0, dim);
          raw->WaitRPCFinish(rpc_id);
          raw->RevokeRPCResource(rpc_id);
          (void)iteration;
        },
        &get_warmup_samples_us,
        &get_measure_samples_us);
    if (ShouldPrintSummary(report_mode)) {
      PrintSummary(
          "RDMA",
          "put",
          "warmup",
          put_warmup_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "put",
          "measure",
          put_measure_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "get",
          "warmup",
          get_warmup_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
      PrintSummary(
          "RDMA",
          "get",
          "measure",
          get_measure_samples_us,
          FLAGS_iterations,
          FLAGS_batch_keys,
          keys.size());
    }
    return 0;
  }

  std::unique_ptr<recstore::BasePSClient> client =
      CreateBenchmarkClient(transport);

  std::vector<int64_t> put_warmup_samples_us;
  std::vector<int64_t> put_measure_samples_us;
  put_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  put_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
  RunOperationRounds(
      transport,
      "put",
      total_rounds,
      FLAGS_warmup_rounds,
      FLAGS_rounds,
      FLAGS_iterations,
      report_mode,
      [&](int iteration) {
        const int ret = client->PutParameter(key_array, values);
        CHECK(BenchmarkWriteSucceeded(
            transport, ret, ClientReturnsZeroOnSuccess(client.get())))
            << transport << " PutParameter failed at iteration=" << iteration;
      },
      &put_warmup_samples_us,
      &put_measure_samples_us);

  std::vector<int64_t> get_warmup_samples_us;
  std::vector<int64_t> get_measure_samples_us;
  get_warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  get_measure_samples_us.reserve(std::max(0, FLAGS_rounds));
  RunOperationRounds(
      transport,
      "get",
      total_rounds,
      FLAGS_warmup_rounds,
      FLAGS_rounds,
      FLAGS_iterations,
      report_mode,
      [&](int iteration) {
        std::vector<float> output(
            keys.size() * (FLAGS_value_size / sizeof(float)), 0.0f);
        base::RecTensor out(
            output.data(),
            {static_cast<int64_t>(keys.size()),
             FLAGS_value_size / static_cast<int>(sizeof(float))});
        const int ret = client->GetParameter(key_array, out);
        CHECK(BenchmarkReadSucceeded(
            transport, ret, ClientReturnsZeroOnSuccess(client.get())))
            << transport << " GetParameter failed at iteration=" << iteration;
      },
      &get_warmup_samples_us,
      &get_measure_samples_us);
  if (ShouldPrintSummary(report_mode)) {
    PrintSummary(
        transport,
        "put",
        "warmup",
        put_warmup_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "put",
        "measure",
        put_measure_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "get",
        "warmup",
        get_warmup_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
    PrintSummary(
        transport,
        "get",
        "measure",
        get_measure_samples_us,
        FLAGS_iterations,
        FLAGS_batch_keys,
        keys.size());
  }
  return 0;
}
