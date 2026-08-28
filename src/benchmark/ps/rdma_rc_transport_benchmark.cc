#include <folly/init/Init.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/tensor.h"
#include "base/bind_core.h"
#include "benchmark/ps/rdma_rc_transport_benchmark_values.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "ps/rdma/rc_options.h"

DEFINE_int32(num_shards, 1, "number of RDMA RC shards");
DEFINE_int32(iterations, 100, "number of iterations per round");
DEFINE_int32(rounds, 5, "number of measured rounds");
DEFINE_int32(warmup_rounds, 1, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 16, "number of keys per request");
DEFINE_int32(get_ratio, 95, "GET percentage for mixed operation");
DEFINE_int32(async_depth, 1, "outstanding async GET depth for async_stream op");
DEFINE_string(op, "all", "operation: all|put|get|async_get|async_stream|mixed");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DEFINE_bool(verify_values,
            false,
            "verify GET response values against initialized test data");
DEFINE_int32(verify_value_row_stride,
             1,
             "row stride for --verify_values; 1 verifies every returned row");
DECLARE_int32(value_size);

namespace {

constexpr const char* kTransportName = "RDMA_RC";

struct BenchmarkInput {
  std::vector<std::uint64_t> keys;
  base::RecTensor values;
  base::ConstArray<std::uint64_t> key_array;
};

std::vector<std::uint64_t> MakeKeys(int batch_keys) {
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<std::uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    keys.push_back(static_cast<std::uint64_t>(1000001 + i));
  }
  return keys;
}

base::RecTensor MakeValues(const std::vector<std::uint64_t>& keys) {
  const int dim = FLAGS_value_size / sizeof(float);
  CHECK_GT(dim, 0) << "--value_size must be at least sizeof(float)";
  base::RecTensor values(
      {static_cast<int64_t>(keys.size()), dim}, base::DataType::FLOAT32);
  float* dst = values.data_as<float>();
  for (size_t row = 0; row < keys.size(); ++row) {
    for (int col = 0; col < dim; ++col) {
      dst[row * static_cast<size_t>(dim) + static_cast<size_t>(col)] =
          recstore::benchmark::MakeHashedValue(keys[row], col);
    }
  }
  return values;
}

BenchmarkInput MakeInput() {
  BenchmarkInput input;
  input.keys      = MakeKeys(FLAGS_batch_keys);
  input.values    = MakeValues(input.keys);
  input.key_array = base::ConstArray<std::uint64_t>(input.keys);
  return input;
}

bool ShouldRunOp(const std::string& requested, const std::string& op) {
  return requested == "all" || requested == op;
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

void MaybePrintAsyncProgress(
    bool is_warmup,
    int round,
    int warmup_rounds,
    int measure_rounds,
    const char* stage,
    int completed,
    int requests_per_round,
    int submitted,
    std::chrono::steady_clock::time_point start) {
  const int progress_step   = std::max(1, requests_per_round / 10);
  const int completed_count = completed + 1;
  if (completed_count != requests_per_round &&
      (completed_count % progress_step) != 0) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const int64_t elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(now - start)
          .count();
  std::cout << "transport=" << kTransportName << " op=async_stream progress"
            << " stage=" << stage
            << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
            << (is_warmup ? (round + 1) : (round - warmup_rounds + 1)) << "/"
            << (is_warmup ? warmup_rounds : measure_rounds)
            << " completed=" << completed_count << "/" << requests_per_round
            << " submitted=" << submitted
            << " elapsed_ms=" << (elapsed_us / 1000.0) << std::endl;
}

bool MixedIterationIsGet(int iteration) {
  if (FLAGS_get_ratio == 100) {
    return true;
  }
  if (FLAGS_get_ratio == 0) {
    return false;
  }
  const int put_ratio = 100 - FLAGS_get_ratio;
  const int period    = std::max(1, 100 / put_ratio);
  return (iteration + 1) % period != 0;
}

void ValidateFlags() {
  CHECK_GT(FLAGS_num_shards, 0);
  CHECK_GT(FLAGS_iterations, 0);
  CHECK_GE(FLAGS_warmup_rounds, 0);
  CHECK_GT(FLAGS_rounds, 0);
  CHECK_GT(FLAGS_batch_keys, 0);
  CHECK(FLAGS_op == "all" || FLAGS_op == "put" || FLAGS_op == "get" ||
        FLAGS_op == "async_get" || FLAGS_op == "async_stream" ||
        FLAGS_op == "mixed")
      << "Invalid --op: " << FLAGS_op;
  CHECK_GT(FLAGS_async_depth, 0);
  CHECK_GE(FLAGS_get_ratio, 0);
  CHECK_LE(FLAGS_get_ratio, 100);
  CHECK(FLAGS_report_mode == "summary" || FLAGS_report_mode == "per_round" ||
        FLAGS_report_mode == "both")
      << "Invalid --report_mode: " << FLAGS_report_mode;
  CHECK_GT(FLAGS_verify_value_row_stride, 0);
  if (FLAGS_op == "async_stream") {
    if (FLAGS_rdma_rc_qps_per_client_per_shard <= 0) {
      throw std::runtime_error(
          "--rdma_rc_qps_per_client_per_shard must be positive");
    }
    if (FLAGS_rdma_rc_slots_per_qp <= 0) {
      throw std::runtime_error("--rdma_rc_slots_per_qp must be positive");
    }
    const int capacity =
        FLAGS_rdma_rc_qps_per_client_per_shard * FLAGS_rdma_rc_slots_per_qp;
    if (capacity < FLAGS_async_depth) {
      throw std::runtime_error(
          "async_stream requires --rdma_rc_qps_per_client_per_shard * "
          "--rdma_rc_slots_per_qp >= --async_depth to avoid exhausting the "
          "client slot pool at startup (qps_per_client_per_shard=" +
          std::to_string(FLAGS_rdma_rc_qps_per_client_per_shard) +
          ", slots_per_qp=" + std::to_string(FLAGS_rdma_rc_slots_per_qp) +
          ", async_depth=" + std::to_string(FLAGS_async_depth) + ")");
    }
  }
}

void VerifyFlatOutput(const BenchmarkInput& input,
                      const std::vector<float>& output,
                      const std::string& op,
                      int iteration,
                      int output_index = 0) {
  if (!FLAGS_verify_values) {
    return;
  }
  const int dim = FLAGS_value_size / sizeof(float);
  const std::size_t value_count =
      input.keys.size() * static_cast<std::size_t>(dim);
  CHECK_GE(output.size(), value_count + 1);
  const auto* status = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(output.data()) +
      input.keys.size() * static_cast<std::size_t>(FLAGS_value_size));
  CHECK_EQ(*status, static_cast<std::int32_t>(petps::RpcStatus::kOk))
      << "GET status mismatch op=" << op << " iteration=" << iteration
      << " output_index=" << output_index;
  const std::size_t row_stride =
      static_cast<std::size_t>(FLAGS_verify_value_row_stride);
  for (std::size_t row = 0; row < input.keys.size(); row += row_stride) {
    for (int col = 0; col < dim; ++col) {
      const float actual = output[row * static_cast<std::size_t>(dim) +
                                  static_cast<std::size_t>(col)];
      const float expected =
          recstore::benchmark::MakeHashedValue(input.keys[row], col);
      CHECK_EQ(recstore::benchmark::FloatBits(actual),
               recstore::benchmark::FloatBits(expected))
          << "GET value mismatch op=" << op << " iteration=" << iteration
          << " output_index=" << output_index << " row=" << row << " key="
          << input.keys[row] << " col=" << col << " actual=" << actual
          << " expected=" << expected << " actual_bits=0x" << std::hex
          << recstore::benchmark::FloatBits(actual) << " expected_bits=0x"
          << recstore::benchmark::FloatBits(expected) << std::dec;
    }
  }
}

void MaybePrintPerRound(
    const std::string& op,
    bool is_warmup,
    int round,
    int warmup_rounds,
    int measure_rounds,
    int64_t elapsed_us) {
  if (!ShouldPrintPerRound(FLAGS_report_mode)) {
    return;
  }
  std::cout << "transport=" << kTransportName << " op=" << op
            << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
            << (is_warmup ? (round + 1) : (round - warmup_rounds + 1)) << "/"
            << (is_warmup ? warmup_rounds : measure_rounds)
            << " elapsed_us=" << elapsed_us << std::endl;
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

void PrintSummary(const std::string& op,
                  const std::string& phase,
                  const std::vector<int64_t>& elapsed_us_samples,
                  int iterations_per_round,
                  std::size_t keys_per_iteration) {
  if (elapsed_us_samples.empty()) {
    return;
  }

  const double total_us = std::accumulate(
      elapsed_us_samples.begin(), elapsed_us_samples.end(), 0.0);
  const double mean_us =
      total_us / static_cast<double>(elapsed_us_samples.size());
  const double p50_us       = PercentileUs(elapsed_us_samples, 0.50);
  const double p95_us       = PercentileUs(elapsed_us_samples, 0.95);
  const double p99_us       = PercentileUs(elapsed_us_samples, 0.99);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec =
      (static_cast<double>(iterations_per_round) * total_rounds) /
      (total_us / 1e6);
  const double key_ops_per_sec =
      (static_cast<double>(iterations_per_round) *
       static_cast<double>(keys_per_iteration) * total_rounds) /
      (total_us / 1e6);

  std::cout << "transport=" << kTransportName << " op=" << op << " phase="
            << phase << " summary rounds=" << elapsed_us_samples.size()
            << " iterations=" << iterations_per_round << " batch_keys="
            << FLAGS_batch_keys << " elapsed_us_mean=" << mean_us
            << " elapsed_us_p50=" << p50_us << " elapsed_us_p95=" << p95_us
            << " elapsed_us_p99=" << p99_us << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

template <typename IterationFn>
void RunOperation(const std::string& op,
                  IterationFn&& run_iteration,
                  std::size_t keys_per_iteration) {
  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;
  std::vector<int64_t> warmup_samples_us;
  std::vector<int64_t> measure_samples_us;
  warmup_samples_us.reserve(static_cast<std::size_t>(FLAGS_warmup_rounds));
  measure_samples_us.reserve(static_cast<std::size_t>(FLAGS_rounds));

  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < FLAGS_warmup_rounds;
    const auto start     = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < FLAGS_iterations; ++iteration) {
      run_iteration(iteration);
    }
    const auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? warmup_samples_us : measure_samples_us).push_back(elapsed_us);
    MaybePrintPerRound(
        op, is_warmup, round, FLAGS_warmup_rounds, FLAGS_rounds, elapsed_us);
  }

  if (ShouldPrintSummary(FLAGS_report_mode)) {
    PrintSummary(
        op, "warmup", warmup_samples_us, FLAGS_iterations, keys_per_iteration);
    PrintSummary(op,
                 "measure",
                 measure_samples_us,
                 FLAGS_iterations,
                 keys_per_iteration);
  }
}

void RunAsyncStreamOperation(recstore::RdmaRawAccess* raw,
                             const BenchmarkInput& input) {
  const int dim                = FLAGS_value_size / sizeof(float);
  const int total_rounds       = FLAGS_warmup_rounds + FLAGS_rounds;
  const int requests_per_round = FLAGS_iterations * FLAGS_async_depth;
  const std::size_t output_size =
      input.keys.size() * static_cast<std::size_t>(dim) + 1;
  const std::string op = "async_stream" + std::to_string(FLAGS_async_depth);

  std::vector<std::vector<float>> outputs;
  outputs.reserve(static_cast<std::size_t>(FLAGS_async_depth));
  for (int i = 0; i < FLAGS_async_depth; ++i) {
    outputs.emplace_back(output_size, 0.0f);
  }
  std::vector<int> rpc_ids(static_cast<std::size_t>(FLAGS_async_depth), 0);
  std::vector<int> request_ids(static_cast<std::size_t>(FLAGS_async_depth), 0);

  std::vector<int64_t> warmup_samples_us;
  std::vector<int64_t> measure_samples_us;
  warmup_samples_us.reserve(static_cast<std::size_t>(FLAGS_warmup_rounds));
  measure_samples_us.reserve(static_cast<std::size_t>(FLAGS_rounds));

  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < FLAGS_warmup_rounds;
    const auto start     = std::chrono::steady_clock::now();

    int submitted     = 0;
    const int initial = std::min(FLAGS_async_depth, requests_per_round);
    for (; submitted < initial; ++submitted) {
      const int slot = submitted % FLAGS_async_depth;
      rpc_ids[static_cast<std::size_t>(slot)] = raw->SubmitGetParameter(
          input.key_array,
          outputs[static_cast<std::size_t>(slot)].data(),
          true,
          0,
          dim);
      request_ids[static_cast<std::size_t>(slot)] = submitted;

      MaybePrintAsyncProgress(
          is_warmup,
          round,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          "submit",
          submitted,
          requests_per_round,
          submitted + 1,
          start);
    }

    for (int completed = 0; completed < requests_per_round; ++completed) {
      const int slot   = completed % FLAGS_async_depth;
      const int rpc_id = rpc_ids[static_cast<std::size_t>(slot)];
      raw->WaitRPCFinish(rpc_id);
      VerifyFlatOutput(
          input,
          outputs[static_cast<std::size_t>(slot)],
          "async_stream",
          request_ids[static_cast<std::size_t>(slot)],
          slot);
      raw->RevokeRPCResource(rpc_id);

      if (submitted < requests_per_round) {
        rpc_ids[static_cast<std::size_t>(slot)] = raw->SubmitGetParameter(
            input.key_array,
            outputs[static_cast<std::size_t>(slot)].data(),
            true,
            0,
            dim);
        request_ids[static_cast<std::size_t>(slot)] = submitted;
        ++submitted;
      }

      MaybePrintAsyncProgress(
          is_warmup,
          round,
          FLAGS_warmup_rounds,
          FLAGS_rounds,
          "complete",
          completed,
          requests_per_round,
          submitted,
          start);
    }

    const auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? warmup_samples_us : measure_samples_us).push_back(elapsed_us);
    MaybePrintPerRound(
        op, is_warmup, round, FLAGS_warmup_rounds, FLAGS_rounds, elapsed_us);
  }

  if (ShouldPrintSummary(FLAGS_report_mode)) {
    PrintSummary(
        op, "warmup", warmup_samples_us, requests_per_round, input.keys.size());
    PrintSummary(op,
                 "measure",
                 measure_samples_us,
                 requests_per_round,
                 input.keys.size());
  }
}

class BenchmarkClient {
public:
  explicit BenchmarkClient(int num_shards) {
    CHECK_GT(num_shards, 0);
    json config      = json::object();
    config["client"] = {{"host", "127.0.0.1"}, {"port", 1234}, {"shard", 0}};
    json servers     = json::array();
    for (int shard = 0; shard < num_shards; ++shard) {
      servers.push_back(
          {{"host", "127.0.0.1"}, {"port", 1234}, {"shard", shard}});
    }
    config["distributed_client"] = {
        {"num_shards", num_shards},
        {"hash_method", "city_hash"},
        {"servers", servers},
    };
    adapter_ = std::make_unique<recstore::RDMAPSClientAdapter>(config);
    raw_     = std::make_unique<recstore::RdmaRawAccess>(adapter_.get());
  }

  recstore::RDMAPSClientAdapter* get() const { return adapter_.get(); }
  recstore::RdmaRawAccess* raw() const { return raw_.get(); }

private:
  std::unique_ptr<recstore::RDMAPSClientAdapter> adapter_;
  std::unique_ptr<recstore::RdmaRawAccess> raw_;
};

} // namespace

int main(int argc, char** argv) {
  try {
    folly::Init init(&argc, &argv);
    ValidateFlags();
    base::global_socket_id = FLAGS_rdma_rc_client_numa_id;

    BenchmarkInput input = MakeInput();
    BenchmarkClient benchmark_client(FLAGS_num_shards);
    recstore::RDMAPSClientAdapter* adapter = benchmark_client.get();
    recstore::RdmaRawAccess* raw = benchmark_client.raw();
    CHECK_NE(adapter, nullptr);
    CHECK_NE(raw, nullptr);

    CHECK_EQ(adapter->PutParameter(input.key_array, input.values), 0)
        << "initial RDMA RC PutParameter failed";

    if (ShouldRunOp(FLAGS_op, "put")) {
      RunOperation(
          "put",
          [&](int iteration) {
            CHECK_EQ(adapter->PutParameter(input.key_array, input.values), 0)
                << "RDMA RC put failed at iteration=" << iteration;
          },
          input.keys.size());
    }

    if (ShouldRunOp(FLAGS_op, "get")) {
      const int dim = FLAGS_value_size / sizeof(float);
      std::vector<float> output(
          input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
      RunOperation(
          "get",
          [&](int iteration) {
            int rpc_id = raw->SubmitGetParameter(
                input.key_array, output.data(), false, 0, dim);
            raw->WaitRPCFinish(rpc_id);
            VerifyFlatOutput(input, output, "get", iteration);
            raw->RevokeRPCResource(rpc_id);
            (void)iteration;
          },
          input.keys.size());
    }

    if (ShouldRunOp(FLAGS_op, "async_get")) {
      const int dim = FLAGS_value_size / sizeof(float);
      std::vector<float> output(
          input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
      RunOperation(
          "async_get",
          [&](int iteration) {
            int rpc_id = raw->SubmitGetParameter(
                input.key_array, output.data(), true, 0, dim);
            raw->WaitRPCFinish(rpc_id);
            VerifyFlatOutput(input, output, "async_get", iteration);
            raw->RevokeRPCResource(rpc_id);
            (void)iteration;
          },
          input.keys.size());
    }

    if (ShouldRunOp(FLAGS_op, "async_stream")) {
      RunAsyncStreamOperation(raw, input);
    }

    if (ShouldRunOp(FLAGS_op, "mixed")) {
      const int dim = FLAGS_value_size / sizeof(float);
      std::vector<float> output(
          input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
      RunOperation(
          "mixed_get" + std::to_string(FLAGS_get_ratio) + "_put" +
              std::to_string(100 - FLAGS_get_ratio),
          [&](int iteration) {
            if (MixedIterationIsGet(iteration)) {
              int rpc_id = raw->SubmitGetParameter(
                  input.key_array, output.data(), false, 0, dim);
              raw->WaitRPCFinish(rpc_id);
              VerifyFlatOutput(input, output, "mixed", iteration);
              raw->RevokeRPCResource(rpc_id);
            } else {
              CHECK_EQ(adapter->PutParameter(input.key_array, input.values), 0)
                  << "RDMA RC mixed put failed at iteration=" << iteration;
            }
          },
          input.keys.size());
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
