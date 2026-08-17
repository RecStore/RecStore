#include <folly/init/Init.h>

#include <boost/coroutine2/all.hpp>

#include <atomic>
#include <array>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/bind_core.h"
#include "base/config.h"
#include "base/log.h"
#include "base/timer.h"
#include "memory/shm_file.h"
#include "ps/rdma/rdma_common.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/rdma/control_plane.h"
#include "ps/rdma/rc_options.h"
#include "ps/rdma/rc_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

DEFINE_string(config_path, "", "config file path");
DEFINE_int32(thread_num, 1, "RC write poll thread count");
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DEFINE_bool(use_dram, false, "unused compatibility flag");
DEFINE_int32(numa_id, 0, "NUMA node id for mmap and core binding");

namespace {

using petps::Exchange;
using petps::NamespaceToken;
using petps::NowNs;

constexpr std::size_t kMaxDirectSgesPerWr = 32;

bool ShouldTraceRdmaGet() {
  static const bool enabled = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::uint64_t RdmaGetTraceInterval() {
  static const std::uint64_t interval = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE_INTERVAL");
    if (env == nullptr) {
      return std::uint64_t{5000};
    }
    const auto parsed =
        static_cast<std::uint64_t>(std::strtoull(env, nullptr, 10));
    return parsed == 0 ? std::uint64_t{5000} : parsed;
  }();
  return interval;
}

std::string TimestampNow() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::to_string(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

int ResolveShardId(const nlohmann::json& config) {
  const int default_shard = FLAGS_global_id;
  if (!config.contains("cache_ps") || !config["cache_ps"].is_object()) {
    return default_shard;
  }
  const auto& cache_ps = config["cache_ps"];
  if (cache_ps.contains("servers") && cache_ps["servers"].is_array()) {
    for (const auto& server : cache_ps["servers"]) {
      if (server.value("shard", -1) == FLAGS_global_id) {
        return server.value("shard", default_shard);
      }
    }
  }
  return default_shard;
}

void NormalizeDramValuePath(nlohmann::json* base_kv_config) {
  if (base_kv_config == nullptr || !base_kv_config->is_object()) {
    return;
  }
  if (!base_kv_config->contains("value") ||
      !(*base_kv_config)["value"].is_object()) {
    return;
  }
  auto& value_cfg = (*base_kv_config)["value"];
  const std::string value_type =
      value_cfg.value("type", std::string("DRAM_VALUE_STORE"));
  if (value_type != "DRAM_VALUE_STORE") {
    return;
  }
  const std::string path = value_cfg.value("path", std::string());
  if (path.empty() || path.rfind("/dev/shm", 0) == 0) {
    return;
  }
  value_cfg["path"] = "/dev/shm/recstore_rdma_rc_" + TimestampNow() + "/value";
}

class PetPSServer {
public:
  PetPSServer(CachePS* cache_ps,
              int thread_count,
              int shard_id,
              const std::string& namespace_token)
      : cache_ps_(cache_ps),
        thread_count_(thread_count),
        shard_id_(shard_id),
        control_plane_client_(petps::RdmaControlPlaneEndpoint{
            FLAGS_rdma_control_plane_host,
            FLAGS_rdma_control_plane_port,
            FLAGS_rdma_control_plane_timeout_ms,
        }) {
    petps::RcTransportConfig config;
    config.shard_id = shard_id_;
    config.num_clients =
        FLAGS_rdma_rc_num_logical_clients >= 0
            ? FLAGS_rdma_rc_num_logical_clients
            : FLAGS_num_client_processes;
    config.qps_per_client_per_shard = FLAGS_rdma_rc_qps_per_client_per_shard;
    config.slots_per_qp             = FLAGS_rdma_rc_slots_per_qp;
    config.request_slot_bytes =
        static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes);
    config.response_slot_bytes =
        static_cast<std::size_t>(FLAGS_rdma_rc_response_slot_bytes);
    config.control_plane_host       = FLAGS_rdma_control_plane_host;
    config.control_plane_port       = FLAGS_rdma_control_plane_port;
    config.control_plane_timeout_ms = FLAGS_rdma_control_plane_timeout_ms;
    config.namespace_token          = namespace_token;
    // The transport constructor waits for every logical client's metadata.
    // Publish readiness first so a launcher that gates clients on server-ready
    // cannot deadlock with the constructor.
    control_plane_client_.PublishServerReady(FLAGS_global_id);
    ready_published_.store(true, std::memory_order_release);
    LOG(INFO) << "component=rdma_control_plane event=server_ready_published"
              << " server_id=" << FLAGS_global_id
              << " host=" << FLAGS_rdma_control_plane_host
              << " port=" << FLAGS_rdma_control_plane_port;
    transport_ = std::make_unique<petps::RcShardServerTransport>(config);
    const auto backing = cache_ps_->GetRDMABackingRegion();
    if (backing.data != nullptr && backing.size > 0) {
      transport_->RegisterLocalMemoryRegion(backing.data, backing.size);
      LOG(INFO) << "component=rdma_rc_server event=value_region_registered"
                << " bytes=" << backing.size;
    } else {
      LOG(INFO) << "component=rdma_rc_server event=value_region_unavailable";
    }
    last_seq_.assign(
        static_cast<std::size_t>(transport_->TotalSlots()), std::uint64_t{0});
    inflight_seq_.assign(
        static_cast<std::size_t>(transport_->TotalSlots()), std::uint64_t{0});
    get_payload_worker_count_ = FLAGS_rdma_rc_server_get_workers;
    if (get_payload_worker_count_ < 0) {
      LOG(FATAL) << "--rdma_rc_server_get_workers must be non-negative";
    }
    poller_profiles_.reserve(
        static_cast<std::size_t>(std::max(1, thread_count_)));
    for (int i = 0; i < std::max(1, thread_count_); ++i) {
      poller_profiles_.emplace_back(std::make_unique<PollerProfile>());
    }
    get_payload_completions_.resize(
        static_cast<std::size_t>(std::max(1, thread_count_)));
  }

  void Run() {
    StartGetPayloadWorkers();
    for (int i = 0; i < thread_count_; ++i) {
      threads_.emplace_back(&PetPSServer::PollingThread, this, i);
    }
  }

private:
  struct GetPayloadTask {
    int slot           = -1;
    int client_id      = -1;
    int qp_index       = -1;
    int slot_in_qp     = -1;
    int poll_thread_id = -1;
    std::uint64_t seq  = 0;
    petps::RequestDescriptor descriptor{};
    const char* payload = nullptr;
    petps::RcShardServerTransport::ResponseView response{};
  };

  struct GetPayloadCompletion {
    int slot           = -1;
    int client_id      = -1;
    int qp_index       = -1;
    int slot_in_qp     = -1;
    int poll_thread_id = -1;
    std::uint64_t seq  = 0;
    petps::RcShardServerTransport::ResponseView response{};
    bool payload_written_direct = false;
  };

  struct ProfileCounters {
    std::atomic<std::uint64_t> scan_rounds{0};
    std::atomic<std::uint64_t> scanned_slots{0};
    std::atomic<std::uint64_t> ready_slots{0};
    std::atomic<std::uint64_t> not_ready_slots{0};
    std::atomic<std::uint64_t> zero_seq_ready{0};
    std::atomic<std::uint64_t> duplicate_seq_ready{0};
    std::atomic<std::uint64_t> inflight_seq_ready{0};
    std::atomic<std::uint64_t> empty_scan_rounds{0};
    std::atomic<std::uint64_t> max_ready_per_round{0};
    std::atomic<std::uint64_t> handled_get{0};
    std::atomic<std::uint64_t> handled_put{0};
    std::atomic<std::uint64_t> handled_update{0};
    std::atomic<std::uint64_t> handled_init{0};
    std::atomic<std::uint64_t> invalid_descriptor{0};
    std::atomic<std::uint64_t> wrong_shard{0};
    std::atomic<std::uint64_t> handle_get_ns{0};
    std::atomic<std::uint64_t> get_batch_get_ns{0};
    std::atomic<std::uint64_t> get_index_lookup_ns{0};
    std::atomic<std::uint64_t> get_zero_fill_ns{0};
    std::atomic<std::uint64_t> get_row_copy_ns{0};
    std::atomic<std::uint64_t> get_rows{0};
    std::atomic<std::uint64_t> get_value_bytes{0};
    std::atomic<std::uint64_t> get_missing_rows{0};
    std::atomic<std::uint64_t> get_direct_sg{0};
    std::atomic<std::uint64_t> get_direct_sg_fallback{0};
    std::atomic<std::uint64_t> get_direct_sg_ns{0};
    std::atomic<std::uint64_t> get_direct_sg_wr{0};
    std::atomic<std::uint64_t> handle_put_ns{0};
    std::atomic<std::uint64_t> handle_update_ns{0};
    std::atomic<std::uint64_t> handle_init_ns{0};
    std::atomic<std::uint64_t> complete_response_ns{0};
    std::atomic<std::uint64_t> poll_loop_ns{0};
    std::atomic<std::uint64_t> next_report_ns{0};
  };

  struct PollerProfile {
    std::atomic<std::uint64_t> scan_rounds{0};
    std::atomic<std::uint64_t> scanned_slots{0};
    std::atomic<std::uint64_t> ready_slots{0};
    std::atomic<std::uint64_t> not_ready_slots{0};
    std::atomic<std::uint64_t> duplicate_seq_ready{0};
    std::atomic<std::uint64_t> inflight_seq_ready{0};
    std::atomic<std::uint64_t> handled_get{0};
    std::atomic<std::uint64_t> poll_loop_ns{0};
  };

  static void
  UpdateMax(std::atomic<std::uint64_t>* value, std::uint64_t candidate) {
    std::uint64_t current = value->load(std::memory_order_relaxed);
    while (candidate > current &&
           !value->compare_exchange_weak(
               current, candidate, std::memory_order_relaxed)) {
    }
  }

  void MaybeReportProfile(int thread_id) {
    if (FLAGS_rdma_rc_profile_interval_ms <= 0 || thread_id != 0) {
      return;
    }
    const std::uint64_t now = NowNs();
    const std::uint64_t interval =
        static_cast<std::uint64_t>(FLAGS_rdma_rc_profile_interval_ms) * 1000000;
    std::uint64_t expected =
        profile_.next_report_ns.load(std::memory_order_relaxed);
    if (expected == 0) {
      profile_.next_report_ns.compare_exchange_strong(
          expected, now + interval, std::memory_order_relaxed);
      return;
    }
    if (now < expected ||
        !profile_.next_report_ns.compare_exchange_strong(
            expected, now + interval, std::memory_order_relaxed)) {
      return;
    }

    const std::uint64_t scan_rounds     = Exchange(&profile_.scan_rounds);
    const std::uint64_t scanned_slots   = Exchange(&profile_.scanned_slots);
    const std::uint64_t ready_slots     = Exchange(&profile_.ready_slots);
    const std::uint64_t not_ready_slots = Exchange(&profile_.not_ready_slots);
    const std::uint64_t zero_seq_ready  = Exchange(&profile_.zero_seq_ready);
    const std::uint64_t duplicate_seq_ready =
        Exchange(&profile_.duplicate_seq_ready);
    const std::uint64_t inflight_seq_ready =
        Exchange(&profile_.inflight_seq_ready);
    const std::uint64_t empty_scan_rounds =
        Exchange(&profile_.empty_scan_rounds);
    const std::uint64_t max_ready_per_round =
        Exchange(&profile_.max_ready_per_round);
    const std::uint64_t handled_get    = Exchange(&profile_.handled_get);
    const std::uint64_t handled_put    = Exchange(&profile_.handled_put);
    const std::uint64_t handled_update = Exchange(&profile_.handled_update);
    const std::uint64_t handled_init   = Exchange(&profile_.handled_init);
    const std::uint64_t complete_count =
        handled_get + handled_put + handled_update + handled_init;
    const std::uint64_t handle_get_ns    = Exchange(&profile_.handle_get_ns);
    const std::uint64_t get_batch_get_ns = Exchange(&profile_.get_batch_get_ns);
    const std::uint64_t get_index_lookup_ns =
        Exchange(&profile_.get_index_lookup_ns);
    const std::uint64_t get_zero_fill_ns = Exchange(&profile_.get_zero_fill_ns);
    const std::uint64_t get_row_copy_ns  = Exchange(&profile_.get_row_copy_ns);
    const std::uint64_t get_rows         = Exchange(&profile_.get_rows);
    const std::uint64_t get_value_bytes  = Exchange(&profile_.get_value_bytes);
    const std::uint64_t get_missing_rows = Exchange(&profile_.get_missing_rows);
    const std::uint64_t get_direct_sg    = Exchange(&profile_.get_direct_sg);
    const std::uint64_t get_direct_sg_ns = Exchange(&profile_.get_direct_sg_ns);
    const std::uint64_t handle_put_ns    = Exchange(&profile_.handle_put_ns);
    const std::uint64_t handle_update_ns = Exchange(&profile_.handle_update_ns);
    const std::uint64_t handle_init_ns   = Exchange(&profile_.handle_init_ns);
    const std::uint64_t complete_response_ns =
        Exchange(&profile_.complete_response_ns);
    const std::uint64_t poll_loop_ns = Exchange(&profile_.poll_loop_ns);
    std::uint64_t poller_min_get   = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t poller_max_get   = 0;
    int poller_min_get_thread      = -1;
    int poller_max_get_thread      = -1;
    std::uint64_t poller_total_get = 0;
    std::uint64_t poller_active    = 0;
    for (std::size_t i = 0; i < poller_profiles_.size(); ++i) {
      auto& poller                           = *poller_profiles_[i];
      const std::uint64_t poller_get         = Exchange(&poller.handled_get);
      const std::uint64_t poller_scan_rounds = Exchange(&poller.scan_rounds);
      const std::uint64_t poller_scanned_slots =
          Exchange(&poller.scanned_slots);
      const std::uint64_t poller_ready_slots = Exchange(&poller.ready_slots);
      const std::uint64_t poller_not_ready_slots =
          Exchange(&poller.not_ready_slots);
      const std::uint64_t poller_duplicate_seq_ready =
          Exchange(&poller.duplicate_seq_ready);
      const std::uint64_t poller_inflight_seq_ready =
          Exchange(&poller.inflight_seq_ready);
      const std::uint64_t poller_poll_loop_ns = Exchange(&poller.poll_loop_ns);
      if (poller_get > 0) {
        ++poller_active;
      }
      poller_total_get += poller_get;
      if (poller_get < poller_min_get) {
        poller_min_get        = poller_get;
        poller_min_get_thread = static_cast<int>(i);
      }
      if (poller_get > poller_max_get) {
        poller_max_get        = poller_get;
        poller_max_get_thread = static_cast<int>(i);
      }
      std::cout
          << "component=rdma_rc_server_poller_profile"
          << " shard=" << shard_id_ << " thread_id=" << i << " scan_rounds="
          << poller_scan_rounds << " scanned_slots=" << poller_scanned_slots
          << " ready_slots=" << poller_ready_slots << " scan_hit_pct="
          << (poller_scanned_slots == 0
                  ? 0.0
                  : 100.0 * static_cast<double>(poller_ready_slots) /
                        static_cast<double>(poller_scanned_slots))
          << " not_ready_slots=" << poller_not_ready_slots
          << " duplicate_seq_ready=" << poller_duplicate_seq_ready
          << " inflight_seq_ready=" << poller_inflight_seq_ready
          << " handled_get=" << poller_get << " poll_loop_avg_ns="
          << (poller_scan_rounds == 0
                  ? 0
                  : poller_poll_loop_ns / poller_scan_rounds)
          << std::endl;
    }
    if (poller_min_get == std::numeric_limits<std::uint64_t>::max()) {
      poller_min_get = 0;
    }
    std::cout
        << "component=rdma_rc_server_profile"
        << " shard=" << shard_id_ << " threads=" << thread_count_
        << " scan_rounds=" << scan_rounds << " scanned_slots=" << scanned_slots
        << " ready_slots=" << ready_slots << " not_ready_slots="
        << not_ready_slots << " zero_seq_ready=" << zero_seq_ready
        << " duplicate_seq_ready=" << duplicate_seq_ready
        << " inflight_seq_ready=" << inflight_seq_ready
        << " empty_scan_rounds=" << empty_scan_rounds << " scan_hit_pct="
        << (scanned_slots == 0 ? 0.0
                               : 100.0 * static_cast<double>(ready_slots) /
                                     static_cast<double>(scanned_slots))
        << " ready_round_pct="
        << (scan_rounds == 0
                ? 0.0
                : 100.0 * static_cast<double>(scan_rounds - empty_scan_rounds) /
                      static_cast<double>(scan_rounds))
        << " avg_ready_per_round="
        << (scan_rounds == 0 ? 0.0
                             : static_cast<double>(ready_slots) /
                                   static_cast<double>(scan_rounds))
        << " max_ready_per_round=" << max_ready_per_round
        << " handled_get=" << handled_get << " handled_put=" << handled_put
        << " handled_update=" << handled_update
        << " handled_init=" << handled_init
        << " invalid_descriptor=" << Exchange(&profile_.invalid_descriptor)
        << " wrong_shard=" << Exchange(&profile_.wrong_shard)
        << " handle_get_avg_ns="
        << (handled_get == 0 ? 0 : handle_get_ns / handled_get)
        << " get_batch_get_avg_ns="
        << (handled_get == 0 ? 0 : get_batch_get_ns / handled_get)
        << " get_index_lookup_avg_ns="
        << (handled_get == 0 ? 0 : get_index_lookup_ns / handled_get)
        << " get_zero_fill_avg_ns="
        << (handled_get == 0 ? 0 : get_zero_fill_ns / handled_get)
        << " get_row_copy_avg_ns="
        << (handled_get == 0 ? 0 : get_row_copy_ns / handled_get)
        << " get_rows=" << get_rows << " get_value_bytes=" << get_value_bytes
        << " get_missing_rows=" << get_missing_rows
        << " get_direct_sg=" << get_direct_sg << " get_direct_sg_fallback="
        << Exchange(&profile_.get_direct_sg_fallback)
        << " get_direct_sg_avg_ns="
        << (get_direct_sg == 0 ? 0 : get_direct_sg_ns / get_direct_sg)
        << " get_direct_sg_wr=" << Exchange(&profile_.get_direct_sg_wr)
        << " handle_put_avg_ns="
        << (handled_put == 0 ? 0 : handle_put_ns / handled_put)
        << " handle_update_avg_ns="
        << (handled_update == 0 ? 0 : handle_update_ns / handled_update)
        << " handle_init_avg_ns="
        << (handled_init == 0 ? 0 : handle_init_ns / handled_init)
        << " complete_response_avg_ns="
        << (complete_count == 0 ? 0 : complete_response_ns / complete_count)
        << " poll_loop_avg_ns="
        << (scan_rounds == 0 ? 0 : poll_loop_ns / scan_rounds)
        << " poller_active=" << poller_active << " poller_total_get="
        << poller_total_get << " poller_min_get=" << poller_min_get
        << " poller_min_get_thread=" << poller_min_get_thread
        << " poller_max_get=" << poller_max_get
        << " poller_max_get_thread=" << poller_max_get_thread << std::endl;
  }

  bool GetPayloadOffloadEnabled() const {
    return get_payload_worker_count_ > 0;
  }

  std::size_t MaxGetPayloadQueueDepth() const {
    return static_cast<std::size_t>(std::max(1, transport_->TotalSlots()));
  }

  void StartGetPayloadWorkers() {
    if (!GetPayloadOffloadEnabled()) {
      return;
    }
    for (int worker_id = 0; worker_id < get_payload_worker_count_;
         ++worker_id) {
      get_payload_workers_.emplace_back(
          &PetPSServer::GetPayloadWorkerLoop, this, worker_id);
    }
    LOG(INFO) << "component=rdma_rc_server event=get_payload_workers_started"
              << " count=" << get_payload_worker_count_;
  }

  void BindServerCore(int core_index) {
    base::bind_core_with_env_offset(core_index);
  }

  bool EnqueueGetPayloadTask(const GetPayloadTask& task) {
    std::lock_guard<std::mutex> guard(get_payload_mu_);
    if (get_payload_tasks_.size() >= MaxGetPayloadQueueDepth()) {
      return false;
    }
    get_payload_tasks_.push_back(task);
    get_payload_cv_.notify_one();
    return true;
  }

  std::size_t PollThreadIndex(int poll_thread_id) const {
    return static_cast<std::size_t>(poll_thread_id);
  }

  bool TryPopGetPayloadCompletion(int poll_thread_id,
                                  GetPayloadCompletion* completion) {
    std::lock_guard<std::mutex> guard(get_payload_mu_);
    auto& completions =
        get_payload_completions_.at(PollThreadIndex(poll_thread_id));
    if (completions.empty()) {
      return false;
    }
    *completion = completions.front();
    completions.pop_front();
    return true;
  }

  void PushGetPayloadCompletion(const GetPayloadCompletion& completion) {
    std::lock_guard<std::mutex> guard(get_payload_mu_);
    get_payload_completions_.at(PollThreadIndex(completion.poll_thread_id))
        .push_back(completion);
  }

  void AccumulateFlatGetProfile(const CachePS::FlatGetProfile& get_profile) {
    profile_.get_batch_get_ns.fetch_add(
        get_profile.batch_get_ns, std::memory_order_relaxed);
    profile_.get_index_lookup_ns.fetch_add(
        get_profile.index_lookup_ns, std::memory_order_relaxed);
    profile_.get_zero_fill_ns.fetch_add(
        get_profile.zero_fill_ns, std::memory_order_relaxed);
    profile_.get_row_copy_ns.fetch_add(
        get_profile.row_copy_ns, std::memory_order_relaxed);
    profile_.get_rows.fetch_add(get_profile.rows, std::memory_order_relaxed);
    profile_.get_value_bytes.fetch_add(
        get_profile.value_bytes, std::memory_order_relaxed);
    profile_.get_missing_rows.fetch_add(
        get_profile.missing_rows, std::memory_order_relaxed);
  }

  void GetPayloadWorkerLoop(int worker_id) {
    BindServerCore(thread_count_ + worker_id);
    LOG(INFO) << "component=rdma_rc_server event=get_payload_worker_ready"
              << " worker_id=" << worker_id;
    while (true) {
      GetPayloadTask task;
      {
        std::unique_lock<std::mutex> lock(get_payload_mu_);
        get_payload_cv_.wait(lock, [this] {
          return !get_payload_tasks_.empty();
        });
        task = get_payload_tasks_.front();
        get_payload_tasks_.pop_front();
      }

      const bool profile_enabled = FLAGS_rdma_rc_profile_interval_ms > 0;
      const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
      const bool payload_written_direct   = HandleGet(
          task.descriptor,
          task.payload,
          &task.response,
          worker_id,
          task.slot_in_qp);
      if (profile_enabled) {
        profile_.handled_get.fetch_add(1, std::memory_order_relaxed);
        profile_.handle_get_ns.fetch_add(
            NowNs() - handle_start_ns, std::memory_order_relaxed);
      }
      const GetPayloadCompletion completion{
          task.slot,
          task.client_id,
          task.qp_index,
          task.slot_in_qp,
          task.poll_thread_id,
          task.seq,
          task.response,
          payload_written_direct,
      };
      PushGetPayloadCompletion(completion);
    }
  }

  void CompleteResponseForSlot(
      int slot,
      int client_id,
      int qp_index,
      int slot_in_qp,
      const petps::RcShardServerTransport::ResponseView& response,
      std::uint64_t seq,
      bool profile_enabled) {
    std::atomic_thread_fence(std::memory_order_release);
    const std::uint64_t complete_start_ns = profile_enabled ? NowNs() : 0;
    transport_->CompleteResponse(
        client_id, qp_index, slot_in_qp, response, seq);
    if (profile_enabled) {
      profile_.complete_response_ns.fetch_add(
          NowNs() - complete_start_ns, std::memory_order_relaxed);
    }
    VLOG(1) << "component=rdma_rc_server event=complete shard=" << shard_id_
            << " slot=" << slot << " client_id=" << client_id
            << " qp=" << qp_index << " seq=" << seq
            << " status=" << response.status->status
            << " response_bytes=" << response.status->response_bytes;
    last_seq_[static_cast<std::size_t>(slot)] = seq;
    if (GetPayloadOffloadEnabled()) {
      inflight_seq_[static_cast<std::size_t>(slot)] = 0;
    }
  }

  void CompleteResponseStatusOnlyForSlot(
      int slot,
      int client_id,
      int qp_index,
      int slot_in_qp,
      const petps::RcShardServerTransport::ResponseView& response,
      std::uint64_t seq,
      bool profile_enabled) {
    std::atomic_thread_fence(std::memory_order_release);
    const std::uint64_t complete_start_ns = profile_enabled ? NowNs() : 0;
    transport_->CompleteResponseStatusOnly(
        client_id, qp_index, slot_in_qp, response, seq);
    if (profile_enabled) {
      profile_.complete_response_ns.fetch_add(
          NowNs() - complete_start_ns, std::memory_order_relaxed);
    }
    VLOG(1) << "component=rdma_rc_server event=complete_direct shard="
            << shard_id_ << " slot=" << slot << " client_id=" << client_id
            << " qp=" << qp_index << " seq=" << seq
            << " status=" << response.status->status
            << " response_bytes=" << response.status->response_bytes;
    last_seq_[static_cast<std::size_t>(slot)] = seq;
    if (GetPayloadOffloadEnabled()) {
      inflight_seq_[static_cast<std::size_t>(slot)] = 0;
    }
  }

  void DrainGetPayloadCompletions(int poll_thread_id, bool profile_enabled) {
    GetPayloadCompletion completion;
    while (TryPopGetPayloadCompletion(poll_thread_id, &completion)) {
      if (completion.payload_written_direct) {
        CompleteResponseStatusOnlyForSlot(
            completion.slot,
            completion.client_id,
            completion.qp_index,
            completion.slot_in_qp,
            completion.response,
            completion.seq,
            profile_enabled);
      } else {
        CompleteResponseForSlot(
            completion.slot,
            completion.client_id,
            completion.qp_index,
            completion.slot_in_qp,
            completion.response,
            completion.seq,
            profile_enabled);
      }
    }
  }

  bool HandleGetDirectSg(
      const petps::RequestDescriptor& descriptor,
      base::ConstArray<std::uint64_t> keys,
      petps::RcShardServerTransport::ResponseView* response,
      int thread_id,
      int slot_in_qp,
      CachePS::FlatGetProfile* get_profile) {
    if (descriptor.response_bytes == 0 || descriptor.embedding_dim == 0) {
      return false;
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(descriptor.embedding_dim) * sizeof(float);
    if (row_bytes == 0 ||
        descriptor.response_bytes !=
            descriptor.key_count * static_cast<std::uint32_t>(row_bytes)) {
      return false;
    }

    thread_local std::vector<CachePS::DirectFixedRow> rows;
    rows.clear();
    const std::uint64_t direct_start_ns =
        FLAGS_rdma_rc_profile_interval_ms > 0 ? NowNs() : 0;
    const bool ok = cache_ps_->GetParameterDirectFixedRows(
        keys,
        descriptor.key_count,
        descriptor.embedding_dim,
        thread_id,
        &rows,
        get_profile);
    if (!ok || rows.size() != descriptor.key_count) {
      return false;
    }
    std::uint64_t response_offset = 0;
    std::uint64_t wr_count        = 0;
    for (std::size_t row = 0; row < rows.size();) {
      std::array<petps::RawVerbsSge, kMaxDirectSgesPerWr> sges{};
      std::size_t sge_count = 0;
      std::size_t row_count = 0;
      for (; row < rows.size(); ++row) {
        const auto& ref = rows[row];
        if (ref.missing || ref.data == nullptr || ref.size != row_bytes) {
          return false;
        }
        if (sge_count > 0) {
          auto& last = sges[sge_count - 1];
          const char* last_end =
              static_cast<const char*>(last.data) + last.bytes;
          if (last_end == ref.data) {
            last.bytes += row_bytes;
            ++row_count;
            continue;
          }
        }
        if (sge_count == kMaxDirectSgesPerWr) {
          break;
        }
        sges[sge_count++] = petps::RawVerbsSge{ref.data, row_bytes};
        ++row_count;
      }
      const std::uint64_t bytes =
          static_cast<std::uint64_t>(row_count * row_bytes);
      transport_->WriteResponsePayloadSg(
          descriptor.client_id,
          descriptor.qp_index,
          slot_in_qp,
          base::ConstArray<petps::RawVerbsSge>(
              sges.data(), static_cast<int>(sge_count)),
          response_offset,
          bytes);
      response_offset += bytes;
      ++wr_count;
    }
    response->status->status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    response->status->response_bytes =
        static_cast<std::uint32_t>(descriptor.response_bytes);
    if (FLAGS_rdma_rc_profile_interval_ms > 0) {
      profile_.get_direct_sg.fetch_add(1, std::memory_order_relaxed);
      profile_.get_direct_sg_ns.fetch_add(
          NowNs() - direct_start_ns, std::memory_order_relaxed);
      profile_.get_direct_sg_wr.fetch_add(wr_count, std::memory_order_relaxed);
      if (get_profile != nullptr) {
        AccumulateFlatGetProfile(*get_profile);
      }
    }
    return true;
  }

  bool HandleGet(const petps::RequestDescriptor& descriptor,
                 const char* payload,
                 petps::RcShardServerTransport::ResponseView* response,
                 int thread_id,
                 int slot_in_qp) {
    if (FLAGS_rdma_rc_fake_get_mode == "status_only") {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kOk);
      response->status->response_bytes = 0;
      return false;
    }
    if (FLAGS_rdma_rc_fake_get_mode == "payload_memset") {
      std::memset(response->payload, 0, descriptor.response_bytes);
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kOk);
      response->status->response_bytes =
          static_cast<std::uint32_t>(descriptor.response_bytes);
      return false;
    }
    if (FLAGS_rdma_rc_fake_get_mode == "index_only") {
      base::ConstArray<std::uint64_t> keys(
          reinterpret_cast<const std::uint64_t*>(payload),
          descriptor.key_count);
      CachePS::FlatGetProfile get_profile;
      CachePS::FlatGetProfile* get_profile_ptr =
          FLAGS_rdma_rc_profile_interval_ms > 0 ? &get_profile : nullptr;
      const bool ok =
          cache_ps_->ProbeParameterIndex(keys, thread_id, get_profile_ptr);
      if (get_profile_ptr != nullptr) {
        AccumulateFlatGetProfile(get_profile);
      }
      response->status->status = static_cast<std::int32_t>(
          ok ? petps::RpcStatus::kOk : petps::RpcStatus::kValueSizeMismatch);
      response->status->response_bytes = 0;
      return false;
    }
    if (FLAGS_rdma_rc_fake_get_mode != "none" &&
        !FLAGS_rdma_rc_fake_get_mode.empty()) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return false;
    }

    base::ConstArray<std::uint64_t> keys(
        reinterpret_cast<const std::uint64_t*>(payload), descriptor.key_count);
    CachePS::FlatGetProfile get_profile;
    CachePS::FlatGetProfile* get_profile_ptr =
        FLAGS_rdma_rc_profile_interval_ms > 0 ? &get_profile : nullptr;
    if ((descriptor.flags & petps::kRcFlagGetDirectSg) != 0) {
      const bool direct_ok = HandleGetDirectSg(
          descriptor, keys, response, thread_id, slot_in_qp, get_profile_ptr);
      if (direct_ok) {
        return true;
      }
      if (FLAGS_rdma_rc_profile_interval_ms > 0) {
        profile_.get_direct_sg_fallback.fetch_add(1, std::memory_order_relaxed);
      }
      if ((descriptor.flags & petps::kRcFlagGetAllowFallbackCopy) == 0) {
        response->status->status =
            static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
        response->status->response_bytes = 0;
        return false;
      }
    }
    const bool ok = cache_ps_->GetParameterFlat(
        keys,
        reinterpret_cast<float*>(response->payload),
        descriptor.key_count,
        descriptor.embedding_dim,
        thread_id,
        get_profile_ptr);
    if (get_profile_ptr != nullptr) {
      AccumulateFlatGetProfile(get_profile);
    }
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kValueSizeMismatch);
    response->status->response_bytes =
        static_cast<std::uint32_t>(descriptor.response_bytes);
    return false;
  }

  void HandlePut(const petps::RequestDescriptor& descriptor,
                 const char* payload,
                 petps::RcShardServerTransport::ResponseView* response,
                 int thread_id) {
    const auto* reader =
        reinterpret_cast<const ParameterCompressReader*>(payload);
    if (!reader->Valid(static_cast<int>(descriptor.payload_bytes))) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }
    for (int i = 0; i < reader->item_size(); ++i) {
      cache_ps_->PutSingleParameter(reader->item(i), thread_id);
    }
    response->status->status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    response->status->response_bytes = 0;
  }

  void HandleUpdate(const petps::RequestDescriptor& descriptor,
                    const char* payload,
                    petps::RcShardServerTransport::ResponseView* response,
                    int thread_id) {
    const std::string_view table_name = petps::DescriptorTableName(descriptor);
    if (table_name.empty()) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    const auto* reader =
        reinterpret_cast<const ParameterCompressReader*>(payload);
    if (!reader->Valid(static_cast<int>(descriptor.payload_bytes))) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    const bool ok = cache_ps_->UpdateParameter(
        std::string(table_name), reader, static_cast<unsigned>(thread_id));
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kInvalidPayload);
    response->status->response_bytes = 0;
  }

  void HandleUpdateFlat(const petps::RequestDescriptor& descriptor,
                        const char* payload,
                        petps::RcShardServerTransport::ResponseView* response,
                        int thread_id) {
    const std::string_view table_name = petps::DescriptorTableName(descriptor);
    const std::size_t expected_bytes = petps::FlatUpdatePayloadBytes(
        descriptor.key_count, descriptor.embedding_dim);
    if (table_name.empty() || descriptor.key_count == 0 || expected_bytes == 0 ||
        descriptor.payload_bytes != expected_bytes) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    const std::size_t key_bytes =
        static_cast<std::size_t>(descriptor.key_count) * sizeof(std::uint64_t);
    const auto* keys = reinterpret_cast<const std::uint64_t*>(payload);
    const auto* grads = reinterpret_cast<const float*>(payload + key_bytes);
    const bool ok = cache_ps_->UpdateParameterFlat(
        std::string(table_name),
        base::ConstArray<std::uint64_t>(keys, descriptor.key_count),
        grads,
        descriptor.key_count,
        descriptor.embedding_dim,
        static_cast<unsigned>(thread_id));
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kInvalidPayload);
    response->status->response_bytes = 0;
  }

  void HandleInitTable(const petps::RequestDescriptor& descriptor,
                       const char* payload,
                       petps::RcShardServerTransport::ResponseView* response) {
    const std::string_view table_name = petps::DescriptorTableName(descriptor);
    if (table_name.empty() ||
        descriptor.payload_bytes != petps::InitTablePayloadBytes()) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    std::uint64_t num_embeddings = 0;
    std::uint64_t embedding_dim  = 0;
    std::memcpy(&num_embeddings, payload, sizeof(num_embeddings));
    std::memcpy(&embedding_dim,
                payload + sizeof(num_embeddings),
                sizeof(embedding_dim));
    const bool ok = cache_ps_->InitTable(
        std::string(table_name), num_embeddings, embedding_dim);
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kInvalidPayload);
    response->status->response_bytes = 0;
  }

  void MaybePublishServerReady() {
    const int started =
        started_threads_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (started != thread_count_ ||
        ready_published_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    control_plane_client_.PublishServerReady(FLAGS_global_id);
    LOG(INFO) << "component=rdma_control_plane event=server_ready_published"
              << " server_id=" << FLAGS_global_id
              << " host=" << FLAGS_rdma_control_plane_host
              << " port=" << FLAGS_rdma_control_plane_port;
  }

  void PollingThread(int thread_id) {
    BindServerCore(thread_id);
    LOG(INFO) << "component=rdma_server event=polling_thread_ready thread_id="
              << thread_id;
    MaybePublishServerReady();
    const int coroutines_per_thread =
        std::max(1, FLAGS_rdma_rc_server_coroutines_per_thread);
    LOG(INFO) << "component=rdma_rc_server event=polling_thread_mode"
              << " thread_id=" << thread_id
              << " coroutines_per_thread=" << coroutines_per_thread;
    if (coroutines_per_thread > 1) {
      RunCoroutinePollingThread(thread_id, coroutines_per_thread);
      return;
    }
    while (true) {
      const bool profile_enabled        = FLAGS_rdma_rc_profile_interval_ms > 0;
      const std::uint64_t poll_start_ns = profile_enabled ? NowNs() : 0;
      std::uint64_t scanned_slots       = 0;
      std::uint64_t ready_slots         = 0;
      DrainGetPayloadCompletions(thread_id, profile_enabled);
      ScanAssignedSlots(
          thread_id,
          /*worker_id=*/0,
          /*worker_count=*/1,
          profile_enabled,
          &scanned_slots,
          &ready_slots);
      DrainGetPayloadCompletions(thread_id, profile_enabled);
      if (profile_enabled) {
        profile_.scan_rounds.fetch_add(1, std::memory_order_relaxed);
        profile_.scanned_slots.fetch_add(
            scanned_slots, std::memory_order_relaxed);
        if (ready_slots == 0) {
          profile_.empty_scan_rounds.fetch_add(1, std::memory_order_relaxed);
        }
        UpdateMax(&profile_.max_ready_per_round, ready_slots);
        const std::uint64_t poll_loop_ns = NowNs() - poll_start_ns;
        profile_.poll_loop_ns.fetch_add(
            poll_loop_ns, std::memory_order_relaxed);
        auto& poller =
            *poller_profiles_.at(static_cast<std::size_t>(thread_id));
        poller.scan_rounds.fetch_add(1, std::memory_order_relaxed);
        poller.scanned_slots.fetch_add(
            scanned_slots, std::memory_order_relaxed);
        poller.ready_slots.fetch_add(ready_slots, std::memory_order_relaxed);
        poller.poll_loop_ns.fetch_add(poll_loop_ns, std::memory_order_relaxed);
        MaybeReportProfile(thread_id);
      }
      std::this_thread::yield();
    }
  }

  bool ProcessSlot(int slot, int thread_id, bool profile_enabled) {
    int client_id  = -1;
    int qp_index   = -1;
    int slot_in_qp = -1;
    transport_->DecodeSlotIndex(slot, &client_id, &qp_index, &slot_in_qp);
    auto* commit = transport_->RequestCommitAt(slot);
    if (commit->state.load(std::memory_order_acquire) != petps::kRcSlotReady) {
      if (profile_enabled) {
        profile_.not_ready_slots.fetch_add(1, std::memory_order_relaxed);
        poller_profiles_.at(static_cast<std::size_t>(thread_id))
            ->not_ready_slots.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    const std::uint64_t seq = commit->seq.load(std::memory_order_acquire);
    if (seq == 0) {
      if (profile_enabled) {
        profile_.zero_seq_ready.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    if (seq == last_seq_[static_cast<std::size_t>(slot)]) {
      if (profile_enabled) {
        profile_.duplicate_seq_ready.fetch_add(1, std::memory_order_relaxed);
        poller_profiles_.at(static_cast<std::size_t>(thread_id))
            ->duplicate_seq_ready.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    if (GetPayloadOffloadEnabled() &&
        seq == inflight_seq_[static_cast<std::size_t>(slot)]) {
      if (profile_enabled) {
        profile_.inflight_seq_ready.fetch_add(1, std::memory_order_relaxed);
        poller_profiles_.at(static_cast<std::size_t>(thread_id))
            ->inflight_seq_ready.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    if (profile_enabled) {
      profile_.ready_slots.fetch_add(1, std::memory_order_relaxed);
    }

    auto* descriptor = transport_->RequestDescriptorAt(slot);
    std::string error;
    if (!petps::ValidateRequestDescriptor(
            *descriptor,
            transport_->config().request_slot_bytes,
            transport_->config().response_slot_bytes,
            &error)) {
      LOG(ERROR) << "component=rdma_rc_server event=invalid_descriptor"
                 << " shard=" << shard_id_ << " slot=" << slot
                 << " thread_id=" << thread_id << " seq=" << seq
                 << " descriptor_seq=" << descriptor->seq
                 << " client_id=" << descriptor->client_id
                 << " qp=" << descriptor->qp_index << " op=" << descriptor->op
                 << " key_count=" << descriptor->key_count
                 << " payload_bytes=" << descriptor->payload_bytes
                 << " response_bytes=" << descriptor->response_bytes
                 << " error=\"" << error << "\"";
      if (profile_enabled) {
        profile_.invalid_descriptor.fetch_add(1, std::memory_order_relaxed);
      }
      last_seq_[static_cast<std::size_t>(slot)] = seq;
      commit->state.store(0, std::memory_order_release);
      return true;
    }
    if (descriptor->client_id != static_cast<std::uint32_t>(client_id) ||
        descriptor->qp_index != static_cast<std::uint32_t>(qp_index)) {
      LOG(ERROR) << "component=rdma_rc_server event=slot_descriptor_mismatch"
                 << " shard=" << shard_id_ << " slot=" << slot
                 << " thread_id=" << thread_id
                 << " slot_client_id=" << client_id << " slot_qp=" << qp_index
                 << " descriptor_client_id=" << descriptor->client_id
                 << " descriptor_qp=" << descriptor->qp_index << " seq=" << seq;
      if (profile_enabled) {
        profile_.invalid_descriptor.fetch_add(1, std::memory_order_relaxed);
      }
      last_seq_[static_cast<std::size_t>(slot)] = seq;
      commit->state.store(0, std::memory_order_release);
      return true;
    }

    auto response =
        transport_->OpenClientResponse(client_id, qp_index, slot_in_qp);
    const char* payload = transport_->RequestPayloadAt(slot);
    VLOG(1) << "component=rdma_rc_server event=consume shard=" << shard_id_
            << " slot=" << slot << " client_id=" << descriptor->client_id
            << " qp=" << descriptor->qp_index << " seq=" << seq << " op="
            << descriptor->op << " key_count=" << descriptor->key_count
            << " payload_bytes=" << descriptor->payload_bytes
            << " response_bytes=" << descriptor->response_bytes;
    response.status->status =
        static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
    response.status->response_bytes = 0;

    if (descriptor->shard_id != static_cast<std::uint32_t>(shard_id_)) {
      LOG(ERROR) << "component=rdma_rc_server event=wrong_shard"
                 << " expected_shard=" << shard_id_
                 << " actual_shard=" << descriptor->shard_id << " slot=" << slot
                 << " client_id=" << descriptor->client_id
                 << " qp=" << descriptor->qp_index << " seq=" << seq << " op="
                 << descriptor->op << " key_count=" << descriptor->key_count;
      if (profile_enabled) {
        profile_.wrong_shard.fetch_add(1, std::memory_order_relaxed);
      }
      response.status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kWrongShard);
    } else if (descriptor->op ==
               static_cast<std::uint16_t>(petps::RcOp::kGet)) {
      if (GetPayloadOffloadEnabled()) {
        const GetPayloadTask task{
            slot,
            client_id,
            qp_index,
            slot_in_qp,
            thread_id,
            seq,
            *descriptor,
            payload,
            response,
        };
        if (!EnqueueGetPayloadTask(task)) {
          return false;
        }
        inflight_seq_[static_cast<std::size_t>(slot)] = seq;
        return true;
      } else {
        const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
        const bool payload_written_direct =
            HandleGet(*descriptor, payload, &response, thread_id, slot_in_qp);
        if (profile_enabled) {
          profile_.handled_get.fetch_add(1, std::memory_order_relaxed);
          profile_.handle_get_ns.fetch_add(
              NowNs() - handle_start_ns, std::memory_order_relaxed);
          poller_profiles_.at(static_cast<std::size_t>(thread_id))
              ->handled_get.fetch_add(1, std::memory_order_relaxed);
        }
        if (payload_written_direct) {
          CompleteResponseStatusOnlyForSlot(
              slot,
              client_id,
              qp_index,
              slot_in_qp,
              response,
              seq,
              profile_enabled);
          return true;
        }
      }
    } else if (descriptor->op ==
               static_cast<std::uint16_t>(petps::RcOp::kPut)) {
      const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
      HandlePut(*descriptor, payload, &response, thread_id);
      if (profile_enabled) {
        profile_.handled_put.fetch_add(1, std::memory_order_relaxed);
        profile_.handle_put_ns.fetch_add(
            NowNs() - handle_start_ns, std::memory_order_relaxed);
      }
    } else if (descriptor->op ==
               static_cast<std::uint16_t>(petps::RcOp::kUpdate)) {
      const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
      HandleUpdate(*descriptor, payload, &response, thread_id);
      if (profile_enabled) {
        profile_.handled_update.fetch_add(1, std::memory_order_relaxed);
        profile_.handle_update_ns.fetch_add(
            NowNs() - handle_start_ns, std::memory_order_relaxed);
      }
    } else if (descriptor->op ==
               static_cast<std::uint16_t>(petps::RcOp::kUpdateFlat)) {
      const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
      HandleUpdateFlat(*descriptor, payload, &response, thread_id);
      if (profile_enabled) {
        profile_.handled_update.fetch_add(1, std::memory_order_relaxed);
        profile_.handle_update_ns.fetch_add(
            NowNs() - handle_start_ns, std::memory_order_relaxed);
      }
    } else if (descriptor->op ==
               static_cast<std::uint16_t>(petps::RcOp::kInitTable)) {
      const std::uint64_t handle_start_ns = profile_enabled ? NowNs() : 0;
      HandleInitTable(*descriptor, payload, &response);
      if (profile_enabled) {
        profile_.handled_init.fetch_add(1, std::memory_order_relaxed);
        profile_.handle_init_ns.fetch_add(
            NowNs() - handle_start_ns, std::memory_order_relaxed);
      }
    }

    CompleteResponseForSlot(
        slot, client_id, qp_index, slot_in_qp, response, seq, profile_enabled);
    return true;
  }

  void ScanAssignedSlots(
      int thread_id,
      int worker_id,
      int worker_count,
      bool profile_enabled,
      std::uint64_t* scanned_slots,
      std::uint64_t* ready_slots) {
    const int qp_count     = transport_->config().qps_per_client_per_shard;
    const int slots_per_qp = transport_->config().slots_per_qp;
    const int num_clients  = transport_->config().num_clients;
    const int lane_slots   = num_clients * slots_per_qp;
    for (int qp_index = thread_id; qp_index < qp_count;
         qp_index += thread_count_) {
      for (int lane_slot = worker_id; lane_slot < lane_slots;
           lane_slot += worker_count) {
        const int client_id  = lane_slot / slots_per_qp;
        const int slot_in_qp = lane_slot % slots_per_qp;
        const int slot_index =
            transport_->SlotIndex(client_id, qp_index, slot_in_qp);
        ++(*scanned_slots);
        if (ProcessSlot(slot_index, thread_id, profile_enabled)) {
          ++(*ready_slots);
        }
      }
    }
  }

  void CoroutineSlotScanner(
      boost::coroutines2::coroutine<void>::push_type& sink,
      int thread_id,
      int worker_id,
      int worker_count) {
    while (true) {
      const bool profile_enabled        = FLAGS_rdma_rc_profile_interval_ms > 0;
      const std::uint64_t poll_start_ns = profile_enabled ? NowNs() : 0;
      std::uint64_t scanned_slots       = 0;
      std::uint64_t ready_slots         = 0;
      DrainGetPayloadCompletions(thread_id, profile_enabled);
      ScanAssignedSlots(
          thread_id,
          worker_id,
          worker_count,
          profile_enabled,
          &scanned_slots,
          &ready_slots);
      DrainGetPayloadCompletions(thread_id, profile_enabled);
      if (profile_enabled) {
        profile_.scan_rounds.fetch_add(1, std::memory_order_relaxed);
        profile_.scanned_slots.fetch_add(
            scanned_slots, std::memory_order_relaxed);
        if (ready_slots == 0) {
          profile_.empty_scan_rounds.fetch_add(1, std::memory_order_relaxed);
        }
        UpdateMax(&profile_.max_ready_per_round, ready_slots);
        const std::uint64_t poll_loop_ns = NowNs() - poll_start_ns;
        profile_.poll_loop_ns.fetch_add(
            poll_loop_ns, std::memory_order_relaxed);
        auto& poller =
            *poller_profiles_.at(static_cast<std::size_t>(thread_id));
        poller.scan_rounds.fetch_add(1, std::memory_order_relaxed);
        poller.scanned_slots.fetch_add(
            scanned_slots, std::memory_order_relaxed);
        poller.ready_slots.fetch_add(ready_slots, std::memory_order_relaxed);
        poller.poll_loop_ns.fetch_add(poll_loop_ns, std::memory_order_relaxed);
      }
      sink();
    }
  }

  void RunCoroutinePollingThread(int thread_id, int coroutines_per_thread) {
    using Coroutine = boost::coroutines2::coroutine<void>;
    std::vector<std::unique_ptr<Coroutine::pull_type>> coroutines;
    coroutines.reserve(static_cast<std::size_t>(coroutines_per_thread));
    for (int coroutine_id = 0; coroutine_id < coroutines_per_thread;
         ++coroutine_id) {
      coroutines.emplace_back(std::make_unique<Coroutine::pull_type>(
          [this, thread_id, coroutine_id, coroutines_per_thread](
              Coroutine::push_type& sink) {
            CoroutineSlotScanner(
                sink, thread_id, coroutine_id, coroutines_per_thread);
          }));
    }
    while (true) {
      for (auto& coroutine : coroutines) {
        (*coroutine)();
      }
      MaybeReportProfile(thread_id);
      std::this_thread::yield();
    }
  }

  CachePS* cache_ps_ = nullptr;
  int thread_count_  = 1;
  int shard_id_      = 0;
  std::unique_ptr<petps::RcShardServerTransport> transport_;
  petps::RdmaControlPlaneClient control_plane_client_;
  std::vector<std::thread> threads_;
  std::vector<std::uint64_t> last_seq_;
  std::vector<std::uint64_t> inflight_seq_;
  std::vector<std::unique_ptr<PollerProfile>> poller_profiles_;
  int get_payload_worker_count_ = 0;
  std::vector<std::thread> get_payload_workers_;
  std::mutex get_payload_mu_;
  std::condition_variable get_payload_cv_;
  std::deque<GetPayloadTask> get_payload_tasks_;
  std::vector<std::deque<GetPayloadCompletion>> get_payload_completions_;
  std::atomic<int> started_threads_{0};
  std::atomic<bool> ready_published_{false};
  ProfileCounters profile_;
};

std::atomic<bool> g_stop_requested{false};

void HandleStopSignal(int) { g_stop_requested.store(true); }

} // namespace

int main(int argc, char* argv[]) {
  // Keep crash dumps for real fatals; let SIGTERM/SIGINT exit cleanly.
  folly::InitOptions init_options;
  init_options.fatalSignals(
      (1UL << SIGSEGV) | (1UL << SIGILL) | (1UL << SIGFPE) | (1UL << SIGABRT) |
      (1UL << SIGBUS) | (1UL << SIGQUIT));
  folly::init(&argc, &argv, init_options);
  std::signal(SIGTERM, HandleStopSignal);
  std::signal(SIGINT, HandleStopSignal);

  if (ShouldTraceRdmaGet()) {
    std::cerr << "component=rdma_get_trace side=server event=enabled interval="
              << RdmaGetTraceInterval() << std::endl;
  }
  xmh::Reporter::StartReportThread();

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(FLAGS_use_dram);
  base::PMMmapRegisterCenter::GetConfig().numa_id = FLAGS_numa_id;

  base::global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  const std::string config_path =
      FLAGS_config_path.empty()
          ? base::ResolveRecStoreConfigPath().string()
          : FLAGS_config_path;
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    LOG(FATAL) << "Cannot open config file: " << config_path;
  }

  nlohmann::json config;
  config_file >> config;
  if (config.contains("cache_ps") && config["cache_ps"].is_object() &&
      config["cache_ps"].contains("base_kv_config")) {
    NormalizeDramValuePath(&config["cache_ps"]["base_kv_config"]);
  }
  if (config.contains("distributed_client") &&
      config["distributed_client"].is_object() &&
      config["distributed_client"].contains("base_kv_config")) {
    NormalizeDramValuePath(&config["distributed_client"]["base_kv_config"]);
  }
  std::unique_ptr<petps::RdmaControlPlaneServer> control_plane_server;
  if (FLAGS_global_id == 0) {
    control_plane_server = std::make_unique<petps::RdmaControlPlaneServer>(
        petps::RdmaControlPlaneEndpoint{
            FLAGS_rdma_control_plane_host,
            FLAGS_rdma_control_plane_port,
            FLAGS_rdma_control_plane_timeout_ms,
        });
    control_plane_server->Start();
    LOG(INFO) << "component=rdma_control_plane event=listening"
              << " server_id=0"
              << " host=" << FLAGS_rdma_control_plane_host
              << " port=" << FLAGS_rdma_control_plane_port;
  }
  auto cache_ps      = std::make_unique<CachePS>(config["cache_ps"]);
  const int shard_id = ResolveShardId(config);
  auto ps            = std::make_unique<PetPSServer>(
      cache_ps.get(), FLAGS_thread_num, shard_id, NamespaceToken());
  ps->Run();
  while (!g_stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  LOG(INFO) << "component=petps_server event=shutdown";
  // ponytail: process exit; PetPSServer's joinable poll threads would
  // std::terminate in ~vector<thread>. Upgrade: PetPSServer::Stop()+join.
  std::_Exit(0);
}
