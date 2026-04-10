#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "grpc_ps_client.h"

DEFINE_string(host, "127.0.0.1", "PS server host");
DEFINE_int32(port, 15000, "PS server port");
DEFINE_int32(warmup, 10, "Warmup iterations");
DEFINE_int32(iters, 100, "Measured iterations per payload size");

static const int64_t kSizes[] = {
    4 * 1024,        //   4 KB
    64 * 1024,       //  64 KB
    512 * 1024,      // 512 KB
    4 * 1024 * 1024, //   4 MB
};

static double now_us() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() /
         1e3;
}

template <typename Fn>
static void
RunBench(const char* label, int64_t size_bytes, int warmup, int iters, Fn fn) {
  for (int i = 0; i < warmup; i++)
    fn(size_bytes);

  std::vector<double> latencies(iters);
  double t_start = now_us();
  for (int i = 0; i < iters; i++) {
    double t0 = now_us();
    fn(size_bytes);
    latencies[i] = now_us() - t0;
  }
  double elapsed_us = now_us() - t_start;

  std::sort(latencies.begin(), latencies.end());
  double avg_us = elapsed_us / iters;
  double p50_us = latencies[iters * 50 / 100];
  double p99_us = latencies[iters * 99 / 100];
  double tput_mb =
      (static_cast<double>(size_bytes) * iters / (1024.0 * 1024.0)) /
      (elapsed_us / 1e6);

  printf("  [%-4s] %6.0f KB  avg=%8.1f us  p50=%8.1f us  p99=%8.1f us  "
         "tput=%8.1f MB/s\n",
         label,
         size_bytes / 1024.0,
         avg_us,
         p50_us,
         p99_us,
         tput_mb);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  GRPCParameterClient client(FLAGS_host, FLAGS_port, /*shard=*/0);

  printf("PS Network Benchmark  server=%s:%d  warmup=%d  iters=%d\n\n",
         FLAGS_host.c_str(),
         FLAGS_port,
         FLAGS_warmup,
         FLAGS_iters);

  printf("LoadFakeData  (download: server -> client)\n");
  for (int64_t sz : kSizes)
    RunBench("load", sz, FLAGS_warmup, FLAGS_iters, [&](int64_t n) {
      client.LoadFakeData(n);
    });

  printf("\nDumpFakeData  (upload: client -> server)\n");
  for (int64_t sz : kSizes)
    RunBench("dump", sz, FLAGS_warmup, FLAGS_iters, [&](int64_t n) {
      client.DumpFakeData(n);
    });

  return 0;
}
