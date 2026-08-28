#include <folly/init/Init.h>

#include "base/factory.h"
#include "base/timer.h"
#include "base/zipf.h"
#include "benchmark/ps/benchmark_client_common.h"
#include "ps/base/Postoffice.h"

DEFINE_int32(thread_num, 1, "client thread num");
DEFINE_int32(batch_read_count, 300, "");
DEFINE_int32(async_req_num, 1, "");
DEFINE_int32(read_ratio, 100, "read ratio 0~100");
DEFINE_string(dataset, "zipfian", "zipfian/dataset");
DEFINE_double(zipf_theta, 0.99, "");
DEFINE_int64(key_space_m, 100, "request key space in million");
DECLARE_int32(value_size);

std::atomic<bool> stop{false};

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv);
  xmh::Reporter::StartReportThread();
  BenchmarkClientCommon::BenchmarkClientCommonArgs args;
  args.thread_count_     = FLAGS_thread_num;
  args.async_req_num_    = FLAGS_async_req_num;
  args.batch_read_count_ = FLAGS_batch_read_count;
  args.key_space_M_      = FLAGS_key_space_m;
  args.zipf_theta_       = FLAGS_zipf_theta;
  args.value_size_       = FLAGS_value_size;
  args.dataset_          = FLAGS_dataset;
  args.read_ratio_       = FLAGS_read_ratio;

  std::unique_ptr<BenchmarkClientCommon> benchmark_client_;
  benchmark_client_.reset(new BenchmarkClientCommon(args));
  benchmark_client_->Run();
  return 0;
}
