#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/resource.h>

#include "base/bind_core.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/log.h"
#include "memory/shm_file.h"
#include "storage/io_backend/force_link.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"
#include "storage/ssd_layout/ssd_layout.h"

// Defined in kv_engine_register.cc; see the call site for why it exists.
extern "C" void RecStoreForceLinkKVEngineRegistrations();

DEFINE_string(
    dram_path,
    "",
    "DRAM data directory; empty only works with anonymous DRAM allocators");
DEFINE_string(ssd_path, "", "SSD data directory");
DEFINE_string(index_type, "DRAM_EXTENDIBLE_HASH", "index.type");
DEFINE_string(value_store_type, "DRAM_VALUE_STORE", "value.type");
DEFINE_string(engine_class, "KVEngineComposite", "BaseKV factory class name");
DEFINE_string(fasterkv_storage,
              "memory",
              "KVEngineFasterKV storage backend: memory|ssd");
DEFINE_string(
    fasterkv_log_path,
    "",
    "KVEngineFasterKV SSD log directory; defaults to ssd_path/fasterkv-log");
DEFINE_int64(
    fasterkv_hlog_memory_bytes,
    0,
    "KVEngineFasterKV hybrid log memory bytes; 0 uses backend default");
DEFINE_double(
    fasterkv_mutable_fraction,
    0.0,
    "KVEngineFasterKV hybrid log mutable fraction; 0 uses backend default");
DEFINE_int64(fasterkv_read_cache_bytes,
             0,
             "KVEngineFasterKV read cache bytes; 0 disables read cache");
DEFINE_string(dram_allocator,
              "CONCURRENT_SLAB_MEMORY_POOL",
              "value.dram_allocator.type");
DEFINE_string(ssd_io_backend, "IOURING", "SSD IO backend");
DEFINE_int32(ssd_queue_depth, 512, "SSD IO queue depth");
DEFINE_int64(dram_capacity_bytes, 0, "override DRAM allocator capacity bytes");
DEFINE_int64(ssd_capacity_bytes, 0, "override SSD allocator capacity bytes");
DEFINE_string(tiering_cache_policy,
              "none",
              "value.tiering.cache_policy for TIERED_VALUE_STORE: none|lru");
DEFINE_double(tiering_cache_budget_ratio,
              0.05,
              "read cache budget as a share of the tiered DRAM layer; measured "
              "on e2e, 0.05 leaves primary placement almost intact, 0.5 halves "
              "it and loses 47%");
DEFINE_int32(tiering_cache_shards, 16, "read cache shard count");
DEFINE_int32(tiering_promote_min_misses,
             2,
             "misses before a value is admitted into the read cache; 2 filters "
             "one-shot misses out (measured +21% on zipfian, flat on uniform)");

DEFINE_int64(record_count, 100000000, "YCSB record count");
DEFINE_string(workload,
              "c",
              "YCSB workload: a/b/c or workloada/workloadb/workloadc");
DEFINE_string(distribution, "uniform", "key distribution: uniform/zipfian/trace");
DEFINE_string(read_trace_path, "", "text query log for distribution=trace");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_bool(
    zipfian_scramble,
    false,
    "permute the zipfian rank->key mapping so hot ranks are not low key "
    "ids; used to test frequency-aware placement without the "
    "sequential-load bias");
DEFINE_string(read_mode, "exists", "read mode: exists/get/batch_get_flat");
DEFINE_int32(batch_keys, 500, "keys per BatchGetFlat operation");
DEFINE_int32(load_batch_keys, 1, "keys per load BatchPut; 1 uses Put");
DEFINE_string(tiered_dram_keys_path, "", "sorted native uint64 DRAM key profile");
DEFINE_bool(tiered_online_admission, false, "enable bounded second-miss tiered admission");
DEFINE_bool(tiered_async_admission, false, "prepare online tier exchanges in a bounded background queue");
DEFINE_bool(load, true, "run load phase");
DEFINE_bool(run, true, "run transaction phase");
DEFINE_bool(print_util, false, "print KVEngine utilization after load");

DEFINE_int32(thread_num, 16, "worker thread count");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(value_size, 128, "value size bytes");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");
DEFINE_bool(
    report_latency,
    false,
    "record per-iteration latency (one batch_get_flat call counts once) "
    "and print p50/p90/p99/max");
DEFINE_bool(report_stages,
            false,
            "collect BatchGetFlat stage stats (index/zero-fill/row-copy); adds "
            "clock calls to the measured path");
DEFINE_string(ssd_layout_strategy, "", "plain, bandana, or maxembed");
DEFINE_string(ssd_layout_mapping, "", "precomputed SSD layout artifact");
DEFINE_string(ssd_layout_query, "", "text query log replayed after load");

namespace {

struct PhaseStats {
  uint64_t total_ops  = 0;
  uint64_t read_ops   = 0;
  uint64_t update_ops = 0;

  // Linear 100 ns buckets up to ~102 us, with everything slower folded into the
  // last bucket. That resolution separates p50 from p99 in the sub-10 us range
  // where these engines live; max_latency_ns keeps the exact tail.
  static constexpr uint64_t kLatencyBucketNs   = 100;
  static constexpr std::size_t kLatencyBuckets = 1024;
  std::array<uint64_t, kLatencyBuckets> latency_ns{};
  uint64_t max_latency_ns = 0;
  BaseKV::BatchGetFlatStats get_stats;

  void RecordLatency(uint64_t ns) {
    const auto bucket =
        std::min<uint64_t>(ns / kLatencyBucketNs, kLatencyBuckets - 1);
    ++latency_ns[static_cast<std::size_t>(bucket)];
    max_latency_ns = std::max(max_latency_ns, ns);
  }
};

struct WorkloadMix {
  int read_percent = 100;
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
               double zipfian_alpha,
               uint64_t seed,
               bool scramble = false,
               const std::vector<uint32_t>* trace = nullptr)
      : distribution_(std::move(distribution)),
        record_count_(record_count),
        alpha_(zipfian_alpha),
        rng_(seed),
        trace_(trace) {
    if (record_count_ == 0) {
      throw std::invalid_argument("record_count must be positive");
    }
    if (distribution_ == "trace") {
      if (trace_ == nullptr || trace_->empty()) {
        throw std::invalid_argument("trace distribution requires nonempty keys");
      }
      trace_index_ = rng_.Uniform(trace_->size());
    } else if (distribution_ == "zipfian") {
      if (alpha_ < 0.0) {
        throw std::invalid_argument("zipfian_alpha must be non-negative");
      }
      if (std::abs(alpha_ - 1.0) < 1e-9) {
        log_n_ = std::log(static_cast<double>(record_count_));
      } else {
        pow_n_ = std::pow(static_cast<double>(record_count_), 1.0 - alpha_);
      }
      if (scramble) {
        scramble_stride_ = ChooseCoprimeStride(record_count_);
      }
    } else if (distribution_ != "uniform") {
      throw std::invalid_argument("distribution must be uniform, zipfian, or trace");
    }
  }

  uint64_t NextKey() {
    if (trace_ != nullptr) {
      const uint64_t key = (*trace_)[trace_index_++];
      if (trace_index_ == trace_->size()) {
        trace_index_ = 0;
      }
      return key;
    }
    if (distribution_ == "uniform") {
      return rng_.Uniform(record_count_);
    }
    return NextZipfian();
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
    if (scramble_stride_ != 0) {
      key = (key * scramble_stride_) % record_count_;
    }
    return key;
  }

  static uint64_t ChooseCoprimeStride(uint64_t n) {
    static constexpr uint64_t kPreferredStrides[] = {
        104729, 99991, 65537, 10007, 97};
    for (uint64_t candidate : kPreferredStrides) {
      if (candidate < n && std::gcd(candidate, n) == 1) {
        return candidate;
      }
    }
    for (uint64_t candidate = n / 2 + 1; candidate > 1; --candidate) {
      if (std::gcd(candidate, n) == 1) {
        return candidate;
      }
    }
    return 0;
  }

  std::string distribution_;
  uint64_t record_count_;
  double alpha_;
  double pow_n_             = 1.0;
  double log_n_             = 0.0;
  uint64_t scramble_stride_ = 0;
  FastRandom rng_;
  const std::vector<uint32_t>* trace_ = nullptr;
  size_t trace_index_ = 0;
};

std::string NormalizeWorkload(std::string workload) {
  std::transform(workload.begin(), workload.end(), workload.begin(), ::tolower);
  if (workload == "workloada")
    return "a";
  if (workload == "workloadb")
    return "b";
  if (workload == "workloadc")
    return "c";
  return workload;
}

WorkloadMix GetWorkloadMix(const std::string& workload) {
  if (workload == "a")
    return WorkloadMix{50};
  if (workload == "b")
    return WorkloadMix{95};
  if (workload == "c")
    return WorkloadMix{100};
  throw std::invalid_argument(
      "workload must be a/b/c or workloada/workloadb/workloadc");
}

bool IsSsdIndexType(const std::string& type) {
  return type == "SSD" || type == "SSD_EXTENDIBLE_HASH";
}

bool HasDramValueStore(const std::string& type) {
  return type == "DRAM_VALUE_STORE" || type == "TIERED_VALUE_STORE";
}

bool HasSsdValueStore(const std::string& type) {
  return type == "SSD_VALUE_STORE" || type == "SSD_STATIC_LAYOUT_VALUE_STORE" ||
         type == "TIERED_VALUE_STORE";
}

bool IsDirectExternalEngine(const std::string& engine_class) {
  return engine_class == "KVEngineFasterKV" ||
         engine_class == "KVEngineHPSHashMap" ||
         engine_class == "KVEngineHPSRocksDB";
}

bool IsLegacyDirectEngine(const std::string& engine_class) {
  return IsDirectExternalEngine(engine_class) ||
         engine_class == "KVEnginePetKV";
}

BaseKVConfig BuildConfig() {
  BaseKVConfig config;
  const uint64_t capacity = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t value_slot_bytes =
      static_cast<uint64_t>(std::max(FLAGS_value_size, 1)) + sizeof(uint64_t);
  const uint64_t value_capacity           = capacity * value_slot_bytes * 1.2;
  constexpr uint64_t kMinSsdCapacityBytes = 256ULL * 1024ULL * 1024ULL;
  if (FLAGS_engine_class == "KVEngineFasterKV") {
    if (FLAGS_fasterkv_storage != "memory" && FLAGS_fasterkv_storage != "ssd") {
      throw std::invalid_argument("fasterkv_storage must be memory or ssd");
    }
    if (FLAGS_fasterkv_mutable_fraction < 0.0 ||
        FLAGS_fasterkv_mutable_fraction > 1.0) {
      throw std::invalid_argument(
          "fasterkv_mutable_fraction must be in [0, 1]");
    }
    config.json_config_ = {
        {"engine_type", "KVEngineFasterKV"},
        {"capacity", capacity},
        {"path", FLAGS_ssd_path.empty() ? FLAGS_dram_path : FLAGS_ssd_path},
        {"value_size", FLAGS_value_size},
        {"fasterkv", {{"storage", FLAGS_fasterkv_storage}}}};
    if (FLAGS_fasterkv_storage == "ssd") {
      const std::string log_path =
          FLAGS_fasterkv_log_path.empty()
              ? FLAGS_ssd_path + "/fasterkv-log"
              : FLAGS_fasterkv_log_path;
      config.json_config_["fasterkv"]["log_path"] = log_path;
    }
    if (FLAGS_fasterkv_hlog_memory_bytes > 0) {
      config.json_config_["fasterkv"]["hlog_memory_bytes"] =
          static_cast<uint64_t>(FLAGS_fasterkv_hlog_memory_bytes);
    }
    if (FLAGS_fasterkv_mutable_fraction > 0.0) {
      config.json_config_["fasterkv"]["mutable_fraction"] =
          FLAGS_fasterkv_mutable_fraction;
    }
    if (FLAGS_fasterkv_read_cache_bytes > 0) {
      config.json_config_["fasterkv"]["read_cache_bytes"] =
          static_cast<uint64_t>(FLAGS_fasterkv_read_cache_bytes);
    }
    config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
    return config;
  } else if (FLAGS_engine_class == "KVEngineHPSHashMap" ||
             FLAGS_engine_class == "KVEngineHPSRocksDB") {
    const std::string data_path =
        FLAGS_ssd_path.empty() ? FLAGS_dram_path : FLAGS_ssd_path;
    if (data_path.empty()) {
      throw std::invalid_argument(
          "dram_path or ssd_path must be set for " + FLAGS_engine_class);
    }
    config.json_config_ = {
        {"engine_type", FLAGS_engine_class},
        {"capacity", capacity},
        {"path", data_path},
        {"value_size", FLAGS_value_size}};
    config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
    return config;
  } else if (FLAGS_engine_class == "KVEnginePetKV") {
    if (FLAGS_dram_path.empty()) {
      throw std::invalid_argument("dram_path must be set for KVEnginePetKV");
    }
    config.json_config_ = {
        {"capacity", capacity},
        {"path", FLAGS_dram_path},
        {"value_size", FLAGS_value_size},
        {"value_capacity", value_capacity}};
    config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
    return config;
  }

  const bool ssd_index = IsSsdIndexType(FLAGS_index_type);

  config.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", capacity},
      {"index", {{"type", FLAGS_index_type}}},
      {"value",
       {{"type", FLAGS_value_store_type},
        {"default_value_size_hint", FLAGS_value_size}}}};

  if (ssd_index) {
    config.json_config_["index"]["path"] = FLAGS_ssd_path + "/index.db";
    config.json_config_["index"]["io"]   = {
        {"type", FLAGS_ssd_io_backend},
        {"queue_depth", FLAGS_ssd_queue_depth},
        {"base_offset_bytes", 0}};
  }

  uint64_t dram_capacity =
      FLAGS_dram_capacity_bytes > 0
          ? static_cast<uint64_t>(FLAGS_dram_capacity_bytes)
          : value_capacity;
  const uint64_t ssd_capacity =
      FLAGS_ssd_capacity_bytes > 0
          ? static_cast<uint64_t>(FLAGS_ssd_capacity_bytes)
          : std::max(value_capacity, kMinSsdCapacityBytes);

  if (FLAGS_value_store_type == "DRAM_VALUE_STORE") {
    if (!FLAGS_dram_path.empty()) {
      config.json_config_["value"]["path"] = FLAGS_dram_path + "/value";
    }
    config.json_config_["value"]["dram_allocator"] = {
        {"type", FLAGS_dram_allocator}, {"capacity_bytes", dram_capacity}};
    // TODO(xieminhui): fix BUG when R2_ALLOC is used, dram_capacity is not
    // enough, should be 10 times of value_capacity
    if (FLAGS_dram_allocator == "R2_ALLOC")
      dram_capacity *= 10;
  } else if (FLAGS_value_store_type == "SSD_VALUE_STORE") {
    config.json_config_["value"]["path"] = FLAGS_ssd_path + "/value.db";
    config.json_config_["value"]["ssd_allocator"] = {
        {"type", "SSD_SLAB"},
        {"capacity_bytes", ssd_capacity},
        {"min_block_size", 128},
        {"max_block_size", 4096},
        {"io",
         {{"type", FLAGS_ssd_io_backend},
          {"queue_depth", FLAGS_ssd_queue_depth},
          {"base_offset_bytes", 4096}}}};
  } else if (FLAGS_value_store_type == "TIERED_VALUE_STORE") {
    constexpr double dram_ratio = 0.1;
    dram_capacity               = dram_capacity * dram_ratio;

    config.json_config_["value"]["dram_allocator"] = {
        {"type", FLAGS_dram_allocator},
        {"capacity_bytes", dram_capacity},
        {"path", FLAGS_dram_path + "/dram"}};
    config.json_config_["value"]["ssd_allocator"] = {
        {"type", "SSD_SLAB"},
        {"capacity_bytes", ssd_capacity},
        {"min_block_size", 128},
        {"max_block_size", 4096},
        {"path", FLAGS_ssd_path + "/ssd.db"},
        {"io",
         {{"type", FLAGS_ssd_io_backend},
          {"queue_depth", FLAGS_ssd_queue_depth},
          {"base_offset_bytes", 4096}}}};
    config.json_config_["value"]["tiering"] = {
        {"cache_policy", FLAGS_tiering_cache_policy},
        {"cache_budget_ratio", FLAGS_tiering_cache_budget_ratio},
        {"cache_shards", FLAGS_tiering_cache_shards},
        {"promote_min_misses", FLAGS_tiering_promote_min_misses}};
    if (!FLAGS_tiered_dram_keys_path.empty()) {
      config.json_config_["value"]["tiering"]["dram_keys_path"] =
          FLAGS_tiered_dram_keys_path;
    }
    if (FLAGS_tiered_online_admission) {
      config.json_config_["value"]["tiering"]["online_admission"] = true;
      config.json_config_["value"]["tiering"]["high_watermark_ratio"] = 0.90;
    }
    if (FLAGS_tiered_async_admission) {
      if (!FLAGS_tiered_online_admission) throw std::invalid_argument("async admission requires online admission");
      config.json_config_["value"]["tiering"]["async_admission"] = true;
    }
  }
  if (!FLAGS_ssd_layout_strategy.empty() &&
      FLAGS_ssd_layout_strategy != "plain") {
    if (FLAGS_ssd_layout_mapping.empty()) {
      throw std::invalid_argument("ssd_layout_mapping is required");
    }
    if (FLAGS_value_store_type != "TIERED_VALUE_STORE") {
      config.json_config_["value"]["type"] = "SSD_STATIC_LAYOUT_VALUE_STORE";
      config.json_config_["value"]["path"] = FLAGS_ssd_path + "/value.db";
      config.json_config_["value"]["io"]   = {
          {"type", FLAGS_ssd_io_backend},
          {"queue_depth", FLAGS_ssd_queue_depth},
          {"base_offset_bytes", 0}};
    }
    config.json_config_["value"]["ssd_layout"] = {
        {"strategy", FLAGS_ssd_layout_strategy},
        {"mapping_path", FLAGS_ssd_layout_mapping}};
  }
  LOG(INFO) << config.json_config_.dump(2);
  config.num_threads_ = std::max(FLAGS_thread_num, FLAGS_load_thread_num);
  return config;
}

PhaseStats LoadRecords(BaseKV* kv, int load_threads, uint64_t record_count) {
  if (FLAGS_load_batch_keys <= 0 ||
      (FLAGS_load_batch_keys > 1 && FLAGS_value_size % sizeof(float) != 0)) {
    throw std::invalid_argument("invalid load_batch_keys or non-float row size");
  }
  std::vector<std::thread> threads;
  std::vector<uint64_t> counts(load_threads, 0);
  const uint64_t per_thread = (record_count + load_threads - 1) / load_threads;

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([kv, tid, per_thread, record_count, &counts]() {
      base::auto_bind_core();
      std::string value(FLAGS_value_size, static_cast<char>('a' + (tid % 26)));
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread;
      const uint64_t end   = std::min(record_count, begin + per_thread);
      if (FLAGS_load_batch_keys > 1) {
        std::vector<uint64_t> keys;
        std::vector<base::ConstArray<float>> values;
        for (uint64_t key = begin; key < end;) {
          const size_t count = std::min<uint64_t>(FLAGS_load_batch_keys, end - key);
          keys.resize(count);
          std::iota(keys.begin(), keys.end(), key);
          values.assign(count, base::ConstArray<float>(
                                   reinterpret_cast<float*>(value.data()),
                                   FLAGS_value_size / sizeof(float)));
          kv->BatchPut(base::ConstArray<uint64_t>(keys), &values, tid);
          key += count;
          counts[tid] += count;
        }
        return;
      }
      for (uint64_t key = begin; key < end; ++key) {
        kv->Put(key, std::string_view(value.data(), value.size()), tid);
        ++counts[tid];
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats stats;
  for (uint64_t count : counts) {
    stats.total_ops += count;
    stats.update_ops += count;
  }
  return stats;
}

PhaseStats RunTransactions(
    BaseKV* kv,
    const std::string& workload,
    const std::string& distribution,
    int threads_num,
    uint64_t record_count,
    int seconds) {
  const WorkloadMix mix         = GetWorkloadMix(workload);
  const bool use_exists         = FLAGS_read_mode == "exists";
  const bool use_batch_get_flat = FLAGS_read_mode == "batch_get_flat";
  if (!use_exists && FLAGS_read_mode != "get" && !use_batch_get_flat) {
    throw std::invalid_argument(
        "read_mode must be exists, get, or batch_get_flat");
  }
  if (FLAGS_batch_keys <= 0) {
    throw std::invalid_argument("batch_keys must be positive");
  }
  if (use_batch_get_flat && FLAGS_value_size % sizeof(float) != 0) {
    throw std::invalid_argument(
        "batch_get_flat requires value_size to be a multiple of sizeof(float)");
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<uint32_t> trace_keys;
  if (distribution == "trace") {
    auto graph = ssd_layout::ReadQueryLog(FLAGS_read_trace_path, false);
    if (graph.vertex_count != record_count || graph.pins.empty()) {
      throw std::invalid_argument("trace record count mismatch or empty trace");
    }
    trace_keys = std::move(graph.pins);
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(threads_num);
  std::exception_ptr first_exception;
  std::mutex exception_mu;

  for (int tid = 0; tid < threads_num; ++tid) {
    threads.emplace_back([&, tid]() {
      try {
        base::auto_bind_core();
        KeyGenerator key_gen(
            distribution,
            record_count,
            FLAGS_zipfian_alpha,
            0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(tid),
            FLAGS_zipfian_scramble,
            distribution == "trace" ? &trace_keys : nullptr);
        std::string value(
            FLAGS_value_size, static_cast<char>('A' + (tid % 26)));
        std::string read_value;
        std::vector<uint64_t> batch_keys(static_cast<size_t>(FLAGS_batch_keys));
        std::vector<float> flat_values(static_cast<size_t>(FLAGS_batch_keys) *
                                       static_cast<size_t>(embedding_dim));
        std::vector<base::ConstArray<float>> write_values(
            static_cast<size_t>(FLAGS_batch_keys),
            base::ConstArray<float>(reinterpret_cast<float*>(value.data()),
                                    embedding_dim));
        PhaseStats local;
        while (!start.load(std::memory_order_acquire)) {
        }
        const bool sample_latency = FLAGS_report_latency;
        const auto clock          = std::chrono::steady_clock::now;
        while (!stop.load(std::memory_order_relaxed)) {
          const auto iter_begin =
              sample_latency ? clock()
                             : std::chrono::steady_clock::time_point{};
          auto record_latency = [&] {
            if (sample_latency) {
              local.RecordLatency(static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      clock() - iter_begin)
                      .count()));
            }
          };
          const bool do_read =
              static_cast<int>(key_gen.NextUint(100)) < mix.read_percent;
          if (do_read) {
            if (use_batch_get_flat) {
              for (int i = 0; i < FLAGS_batch_keys; ++i) {
                batch_keys[static_cast<size_t>(i)] = key_gen.NextKey();
              }
              // The engine assigns the stats struct per call, so accumulate a
              // scratch copy instead of passing the running total.
              BaseKV::BatchGetFlatStats scratch;
              const bool ok = kv->BatchGetFlat(
                  base::ConstArray<uint64_t>(
                      batch_keys.data(), batch_keys.size()),
                  flat_values.data(),
                  FLAGS_batch_keys,
                  embedding_dim,
                  tid,
                  FLAGS_report_stages ? &scratch : nullptr);
              if (FLAGS_report_stages) {
                local.get_stats.index_lookup_ns += scratch.index_lookup_ns;
                local.get_stats.zero_fill_ns += scratch.zero_fill_ns;
                local.get_stats.row_copy_ns += scratch.row_copy_ns;
                local.get_stats.missing_rows += scratch.missing_rows;
              }
              if (!ok) {
                throw std::runtime_error(
                    "read_mode=batch_get_flat is unsupported by this KV "
                    "engine");
              }
              local.read_ops += static_cast<uint64_t>(FLAGS_batch_keys);
              local.total_ops += static_cast<uint64_t>(FLAGS_batch_keys);
              record_latency();
              continue;
            }
            const uint64_t key = key_gen.NextKey();
            if (use_exists) {
              (void)kv->Exists(key, tid);
            } else {
              kv->Get(key, read_value, tid);
            }
            ++local.read_ops;
          } else {
            if (use_batch_get_flat) {
              for (auto& key : batch_keys) {
                key = key_gen.NextKey();
              }
              kv->BatchPut(base::ConstArray<uint64_t>(batch_keys),
                           &write_values, tid);
              local.update_ops += batch_keys.size();
              local.total_ops += batch_keys.size();
              continue;
            }
            const uint64_t key = key_gen.NextKey();
            kv->Put(key, std::string_view(value.data(), value.size()), tid);
            ++local.update_ops;
          }
          ++local.total_ops;
          record_latency();
        }
        stats[tid] = local;
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(exception_mu);
          if (!first_exception) {
            first_exception = std::current_exception();
          }
        }
        stop.store(true, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  sleep(seconds);
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
  kv->DrainBackgroundWork();

  PhaseStats total;
  for (const auto& each : stats) {
    total.total_ops += each.total_ops;
    total.read_ops += each.read_ops;
    total.update_ops += each.update_ops;
    for (std::size_t bucket = 0; bucket < each.latency_ns.size(); ++bucket) {
      total.latency_ns[bucket] += each.latency_ns[bucket];
    }
    total.get_stats.index_lookup_ns += each.get_stats.index_lookup_ns;
    total.get_stats.zero_fill_ns += each.get_stats.zero_fill_ns;
    total.get_stats.row_copy_ns += each.get_stats.row_copy_ns;
    total.get_stats.missing_rows += each.get_stats.missing_rows;
    total.max_latency_ns = std::max(total.max_latency_ns, each.max_latency_ns);
  }
  return total;
}

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

namespace {

double BucketUpperBoundNs(std::size_t bucket) {
  // Linear buckets: report the bucket's upper bound in ns.
  return static_cast<double>((bucket + 1) * PhaseStats::kLatencyBucketNs - 1);
}

void PrintLatencyPercentiles(const PhaseStats& stats) {
  uint64_t samples = 0;
  for (const uint64_t count : stats.latency_ns) {
    samples += count;
  }
  if (samples == 0) {
    return;
  }
  auto percentile_ns = [&](double fraction) {
    const auto target = static_cast<uint64_t>(
        std::ceil(fraction * static_cast<double>(samples)));
    uint64_t seen = 0;
    for (std::size_t bucket = 0; bucket < stats.latency_ns.size(); ++bucket) {
      seen += stats.latency_ns[bucket];
      if (seen >= target) {
        return BucketUpperBoundNs(bucket);
      }
    }
    return BucketUpperBoundNs(stats.latency_ns.size() - 1);
  };
  const double p50 = percentile_ns(0.50);
  const double p90 = percentile_ns(0.90);
  const double p99 = percentile_ns(0.99);
  const double max = static_cast<double>(stats.max_latency_ns);
  std::printf(
      "YCSB_LATENCY samples=%lu p50_us=%.3f p90_us=%.3f p99_us=%.3f "
      "max_us=%.3f\n",
      samples,
      p50 / 1000.0,
      p90 / 1000.0,
      p99 / 1000.0,
      max / 1000.0);
}

void PrintStageBreakdown(const PhaseStats& stats) {
  const auto& get = stats.get_stats;
  const uint64_t total =
      get.index_lookup_ns + get.zero_fill_ns + get.row_copy_ns;
  auto percent = [total](uint64_t value) {
    return total == 0
             ? 0.0
             : 100.0 * static_cast<double>(value) / static_cast<double>(total);
  };
  std::printf(
      "YCSB_STAGES batch_get_total_us=%llu index_lookup_us=%llu (%.1f%%) "
      "zero_fill_us=%llu (%.1f%%) row_copy_us=%llu (%.1f%%) "
      "missing_rows=%llu\n",
      static_cast<unsigned long long>(total / 1000),
      static_cast<unsigned long long>(get.index_lookup_ns / 1000),
      percent(get.index_lookup_ns),
      static_cast<unsigned long long>(get.zero_fill_ns / 1000),
      percent(get.zero_fill_ns),
      static_cast<unsigned long long>(get.row_copy_ns / 1000),
      percent(get.row_copy_ns),
      static_cast<unsigned long long>(get.missing_rows));
}

} // namespace

void ReplayQueryLog(BaseKV* kv) {
  if (FLAGS_ssd_layout_query.empty()) {
    return;
  }
  const auto graph = ssd_layout::ReadQueryLog(FLAGS_ssd_layout_query, false);
  uint64_t queries = 0;
  uint64_t rows    = 0;
  kv->ResetStats();
  const auto begin = std::chrono::steady_clock::now();
  for (size_t query = 0; query + 1 < graph.xpins.size(); ++query) {
    std::vector<uint64_t> keys;
    keys.reserve(graph.xpins[query + 1] - graph.xpins[query]);
    keys.insert(keys.end(),
                graph.pins.begin() + graph.xpins[query],
                graph.pins.begin() + graph.xpins[query + 1]);
    if (keys.empty()) {
      continue;
    }
    std::vector<base::ConstArray<float>> values;
    kv->BatchGet(base::ConstArray<uint64_t>(keys), &values, 0);
    ++queries;
    rows += keys.size();
  }
  const double seconds = SecondsSince(begin, std::chrono::steady_clock::now());
  std::printf("LAYOUT_RESULT queries=%lu rows=%lu seconds=%.6f%s\n",
              queries,
              rows,
              seconds,
              kv->ExtraResultFields().c_str());
}

} // namespace

int RunBenchmark(int argc, char* argv[]) {
  // Static builds drop the registration translation unit unless it is
  // referenced; shared builds get it through --no-as-needed.
  RecStoreForceLinkKVEngineRegistrations();
  ForceLinkIOBackends();
  base::Init(&argc, &argv);

  if (FLAGS_record_count <= 0) {
    LOG(FATAL) << "record_count must be positive";
  }
  if (FLAGS_thread_num <= 0) {
    LOG(FATAL) << "thread_num must be positive";
  }
  if (FLAGS_value_size <= 0) {
    LOG(FATAL) << "value_size must be positive";
  }
  if (FLAGS_running_seconds <= 0) {
    LOG(FATAL) << "running_seconds must be positive";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      IsSsdIndexType(FLAGS_index_type) && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for SSD index";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      HasSsdValueStore(FLAGS_value_store_type) && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for SSD/TIERED value store";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      !FLAGS_ssd_layout_strategy.empty() &&
      FLAGS_ssd_layout_strategy != "plain" && FLAGS_ssd_path.empty()) {
    LOG(FATAL) << "ssd_path must be set for a static SSD layout";
  }
  if (!IsLegacyDirectEngine(FLAGS_engine_class) &&
      FLAGS_value_store_type == "TIERED_VALUE_STORE" &&
      FLAGS_dram_path.empty()) {
    LOG(FATAL) << "dram_path must be set for TIERED value store";
  }
  if (FLAGS_engine_class == "KVEngineFasterKV" &&
      FLAGS_fasterkv_storage == "ssd" && FLAGS_ssd_path.empty() &&
      FLAGS_fasterkv_log_path.empty()) {
    LOG(FATAL) << "ssd_path or fasterkv_log_path must be set for "
                  "KVEngineFasterKV fasterkv_storage=ssd";
  }

  const std::string workload = NormalizeWorkload(FLAGS_workload);
  (void)GetWorkloadMix(workload);
  const int load_threads =
      FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(
          HasDramValueStore(FLAGS_value_store_type) ||
          IsLegacyDirectEngine(FLAGS_engine_class));

  BaseKVConfig config = BuildConfig();
  std::unique_ptr<BaseKV> kv;
  if (FLAGS_engine_class == "KVEngine") {
    auto resolved = base::ResolveEngine(config);
    kv.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        resolved.engine, resolved.cfg));
  } else {
    kv.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
        FLAGS_engine_class, config));
  }
  if (!kv) {
    LOG(FATAL) << "failed to create KVEngine";
  }

  if (FLAGS_load) {
    const auto begin      = std::chrono::steady_clock::now();
    PhaseStats load_stats = LoadRecords(
        kv.get(), load_threads, static_cast<uint64_t>(FLAGS_record_count));
    const auto end       = std::chrono::steady_clock::now();
    const double seconds = SecondsSince(begin, end);
    const double throughput =
        seconds > 0.0 ? static_cast<double>(load_stats.total_ops) / seconds
                      : 0.0;
    std::printf(
        "YCSB_LOAD_RESULT records=%ld threads=%d seconds=%.6f ops=%lu "
        "throughput_ops_sec=%.6f%s\n",
        FLAGS_record_count,
        load_threads,
        seconds,
        load_stats.total_ops,
        throughput,
        kv->ExtraResultFields().c_str());
    if (FLAGS_print_util) {
      kv->Util();
    }
  }

  if (FLAGS_run) {
    kv->ResetStats();
    const auto begin     = std::chrono::steady_clock::now();
    PhaseStats run_stats = RunTransactions(
        kv.get(),
        workload,
        FLAGS_distribution,
        FLAGS_thread_num,
        static_cast<uint64_t>(FLAGS_record_count),
        FLAGS_running_seconds);
    const auto end       = std::chrono::steady_clock::now();
    const double seconds = SecondsSince(begin, end);
    const double throughput =
        seconds > 0.0 ? static_cast<double>(run_stats.total_ops) / seconds
                      : 0.0;
    struct rusage usage {};
    const long peak_rss_kb = getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
    std::printf(
        "YCSB_RESULT workload=%s distribution=%s zipfian_alpha=%.6f "
        "read_mode=%s threads=%d records=%ld runtime_s=%.6f ops=%lu "
        "throughput_ops_sec=%.6f read_ops=%lu update_ops=%lu process_peak_rss_kb=%ld%s\n",
        workload.c_str(),
        FLAGS_distribution.c_str(),
        FLAGS_zipfian_alpha,
        FLAGS_read_mode.c_str(),
        FLAGS_thread_num,
        FLAGS_record_count,
        seconds,
        run_stats.total_ops,
        throughput,
        run_stats.read_ops,
        run_stats.update_ops,
        peak_rss_kb,
        kv->ExtraResultFields().c_str());
    if (FLAGS_report_latency) {
      PrintLatencyPercentiles(run_stats);
    }
    if (FLAGS_report_stages) {
      if (FLAGS_read_mode == "batch_get_flat") {
        PrintStageBreakdown(run_stats);
      } else {
        std::printf("YCSB_STAGES skipped=1 reason=read_mode_has_no_"
                    "batch_get_flat_stats\n");
      }
    }
  }

  ReplayQueryLog(kv.get());
  LOG(INFO) << "gracefully exit";
  return 0;
}

int main(int argc, char* argv[]) {
  try {
    return RunBenchmark(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR benchmark_kv_engine: %s\n", e.what());
    return 1;
  }
}
