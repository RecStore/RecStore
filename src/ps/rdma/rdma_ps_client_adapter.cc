#include "ps/rdma/rdma_ps_client_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>

#include <folly/portability/GFlags.h>
#include <folly/init/Init.h>

#include "framework/common/ps_client_config_adapter.h"
#include "optimizer/sparse_tensor.h"
#include "ps/base/config.h"
#include "ps/base/parameters.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rc_options.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DECLARE_int32(rdma_rc_client_id_base);
DECLARE_int32(rdma_rc_num_logical_clients);
DECLARE_int32(rdma_control_plane_timeout_ms);
DECLARE_string(rdma_get_response_mode);
DECLARE_string(rdma_transport_mode);
DEFINE_string(rdma_transport_mode, "rc_write", "RDMA transport mode: rc_write");
DEFINE_bool(rdma_adapter_skip_prefetch_result_copy,
            false,
            "Benchmark-only option to skip copying RDMA prefetch results into "
            "the GetPrefetchResult output tensor");

namespace recstore {

namespace detail {

bool TryParseIntEnv(const char* env_name, int* parsed_value) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || *value == '\0') {
    return false;
  }
  char* end         = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return false;
  }
  *parsed_value = static_cast<int>(parsed);
  return true;
}

} // namespace detail

namespace {

bool AdapterProfileEnabled() {
  const char* value = std::getenv("RECSTORE_RDMA_ADAPTER_PROFILE");
  return value != nullptr && std::string(value) != "0";
}

void SetIntFlagFromEnv(const char* env_name, int32_t* flag_value) {
  int parsed = 0;
  if (detail::TryParseIntEnv(env_name, &parsed)) {
    *flag_value = static_cast<int32_t>(parsed);
  }
}

void ApplyRdmaFlagsFromEnv() {
  if (const char* value = std::getenv("RECSTORE_RDMA_RC_NAMESPACE")) {
    FLAGS_rdma_rc_namespace = value;
  }
  if (const char* value = std::getenv("RECSTORE_RDMA_CONTROL_PLANE_HOST")) {
    FLAGS_rdma_control_plane_host = value;
  }
  if (const char* value = std::getenv("RECSTORE_RDMA_GET_RESPONSE_MODE")) {
    const std::string mode(value);
    if (mode != "direct_sg" && mode != "staging_copy") {
      throw std::runtime_error(
          "RECSTORE_RDMA_GET_RESPONSE_MODE must be direct_sg or staging_copy");
    }
    FLAGS_rdma_get_response_mode = mode;
  }
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_CONTROL_PLANE_PORT", &FLAGS_rdma_control_plane_port);
  SetIntFlagFromEnv("RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS",
                    &FLAGS_rdma_control_plane_timeout_ms);
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_WAIT_TIMEOUT_MS", &FLAGS_rdma_wait_timeout_ms);
  SetIntFlagFromEnv("RECSTORE_RDMA_RC_QPS_PER_CLIENT_PER_SHARD",
                    &FLAGS_rdma_rc_qps_per_client_per_shard);
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_RC_SLOTS_PER_QP", &FLAGS_rdma_rc_slots_per_qp);
  SetIntFlagFromEnv("RECSTORE_RDMA_RC_SERVER_COROUTINES_PER_THREAD",
                    &FLAGS_rdma_rc_server_coroutines_per_thread);
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_RC_SERVER_GET_WORKERS", &FLAGS_rdma_rc_server_get_workers);
}

std::int64_t NsSince(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

int ValueSizeHintFromBaseKvConfig(const json& base_kv_config,
                                  int fallback_value_size) {
  if (!base_kv_config.is_object()) {
    return fallback_value_size;
  }
  if (!base_kv_config.contains("value") ||
      !base_kv_config["value"].is_object()) {
    return fallback_value_size;
  }
  return base_kv_config["value"].value(
      "default_value_size_hint", fallback_value_size);
}

std::vector<std::string> ReadProcessArgv() {
  std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
  std::vector<std::string> argv;
  if (!cmdline.is_open()) {
    return argv;
  }

  std::string current;
  char ch = '\0';
  while (cmdline.get(ch)) {
    if (ch == '\0') {
      if (!current.empty()) {
        argv.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    argv.push_back(current);
  }
  return argv;
}
} // namespace

EmbeddedRdmaClientIdentity
ResolveEmbeddedRdmaClientIdentity(int num_shards, int configured_num_clients) {
  if (num_shards <= 0) {
    throw std::runtime_error("embedded RDMA num_shards must be positive");
  }
  if (configured_num_clients <= 0) {
    throw std::runtime_error(
        "embedded RDMA configured_num_clients must be positive");
  }

  int client_index = 0;
  if (!detail::TryParseIntEnv("RECSTORE_RDMA_OS_CLIENT_INDEX", &client_index) &&
      !detail::TryParseIntEnv("RANK", &client_index)) {
    detail::TryParseIntEnv("LOCAL_RANK", &client_index);
  }

  int num_client_processes = 1;
  if (!detail::TryParseIntEnv(
          "RECSTORE_RDMA_NUM_CLIENT_PROCESSES", &num_client_processes) &&
      !detail::TryParseIntEnv("WORLD_SIZE", &num_client_processes)) {
    detail::TryParseIntEnv("LOCAL_WORLD_SIZE", &num_client_processes);
  }

  if (client_index < 0) {
    throw std::runtime_error("embedded RDMA client index must be non-negative");
  }
  if (num_client_processes <= 0) {
    throw std::runtime_error(
        "embedded RDMA num_client_processes must be positive");
  }
  if (client_index >= num_client_processes) {
    throw std::runtime_error(
        "embedded RDMA client index out of range for num_client_processes");
  }
  if (num_client_processes != configured_num_clients) {
    throw std::runtime_error("embedded RDMA num_client_processes must match "
                             "rdma_deployment.num_clients");
  }

  EmbeddedRdmaClientIdentity identity;
  identity.client_index         = client_index;
  identity.num_client_processes = num_client_processes;
  identity.global_id            = num_shards + client_index;
  return identity;
}

std::vector<RDMAPSClientAdapter::ShardChunk> RDMAPSClientAdapter::BuildChunks(
    base::ConstArray<uint64_t> keys, std::size_t max_keys_per_rpc) const {
  return shard_routing::BuildChunks(
      keys,
      deployment_.num_shards,
      deployment_.hash_method,
      deployment_.shard_to_endpoint_index,
      max_keys_per_rpc);
}

void InitializeRdmaProcessRuntime() {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    // Python entrypoints pass application CLI flags that are not gflags.
    // Passing them to folly::init makes gflags abort before the RDMA client can
    // start.
    std::vector<std::string> argv_strings = {"recstore_rdma_client"};
    std::vector<char*> argv_storage;
    argv_storage.reserve(argv_strings.size() + 1);
    for (auto& arg : argv_strings) {
      argv_storage.push_back(arg.data());
    }
    argv_storage.push_back(nullptr);

    int argc    = static_cast<int>(argv_strings.size());
    char** argv = argv_storage.data();
    folly::init(&argc, &argv);
    ApplyRdmaFlagsFromEnv();
  });
}

RDMAPSClientAdapter::RDMAPSClientAdapter(json config)
    : BasePSClient(config), config_(std::move(config)),
      split_lanes_([] {
        const char* value = std::getenv("RECSTORE_RDMA_SPLIT_LANES");
        return value != nullptr && std::strcmp(value, "1") == 0;
      }()) {}

void RDMAPSClientAdapter::EnsureClientInitialized() {
  std::lock_guard<std::mutex> guard(init_mu_);
  if (initialized_) {
    return;
  }

  const json cache_ps_cfg =
      config_.contains("cache_ps") ? config_["cache_ps"] : json::object();
  const json client_cfg =
      config_.contains("client") ? config_["client"] : json::object();
  const json dist_cfg = ResolveFrameworkDistributedClientConfig(config_);

  deployment_ = ParseResolvedRdmaDeploymentConfig(config_);

  if (FLAGS_global_id < deployment_.num_shards) {
    const auto identity = ResolveEmbeddedRdmaClientIdentity(
        deployment_.num_shards, deployment_.num_clients);
    FLAGS_num_server_processes = deployment_.num_shards;
    FLAGS_num_client_processes = identity.num_client_processes;
    FLAGS_global_id            = identity.global_id;
    if (FLAGS_rdma_rc_num_logical_clients < 0) {
      FLAGS_rdma_rc_num_logical_clients = identity.num_client_processes;
    }
    if (FLAGS_rdma_rc_client_id_base < 0) {
      FLAGS_rdma_rc_client_id_base = identity.client_index;
    }
  }
  if (FLAGS_num_server_processes != deployment_.num_shards) {
    throw std::runtime_error(
        "RDMA num_server_processes must match distributed_client.num_shards");
  }
  if (FLAGS_num_client_processes != deployment_.num_clients) {
    throw std::runtime_error(
        "RDMA num_client_processes must match rdma_deployment.num_clients");
  }
  if (FLAGS_global_id < 0 ||
      FLAGS_global_id >= deployment_.num_shards + deployment_.num_clients) {
    throw std::runtime_error(
        "RDMA global_id is outside the resolved deployment node range");
  }
  FLAGS_value_size =
      cache_ps_cfg.contains("base_kv_config")
          ? ValueSizeHintFromBaseKvConfig(
                cache_ps_cfg["base_kv_config"], FLAGS_value_size)
          : FLAGS_value_size;
  FLAGS_max_kv_num_per_request = deployment_.max_keys_per_request;
  if (const char* mode = std::getenv("RECSTORE_RDMA_TRANSPORT_MODE")) {
    FLAGS_rdma_transport_mode = mode;
  }

  const int logical_client_id =
      config_.value("rdma_logical_client_id", FLAGS_rdma_rc_client_id_base);

  if (split_lanes_ && FLAGS_rdma_rc_qps_per_client_per_shard < 2) {
    throw std::invalid_argument("RECSTORE_RDMA_SPLIT_LANES requires at least two QPs");
  }
  shard_clients_.clear();
  client_ = nullptr;

  if (deployment_.num_shards <= 1) {
    shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
        client_cfg.value("host", std::string("127.0.0.1")),
        client_cfg.value("port", 25000),
        client_cfg.value("shard", 0),
        logical_client_id));
    client_ = shard_clients_.front().get();
  } else {
    for (const auto& endpoint : deployment_.endpoints) {
      shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
          endpoint.host, endpoint.port, endpoint.shard, logical_client_id));
    }
  }

  if (split_lanes_) {
    for (auto& shard_client : shard_clients_) {
      shard_client->EnableSplitLanes();
    }
    LOG(WARNING) << "component=rdma_adapter event=split_lanes_enabled"
                 << " get_qps=" << FLAGS_rdma_rc_qps_per_client_per_shard - 1
                 << " commit_qps=1 contract=fully_initialized_ps_pipeline"
                 << " all_ranks_same_pipeline=required missing_rows=forbidden"
                 << " external_writers_or_checkpoints=forbidden"
                 << " dirty_reads_require_global_commit=true";
  }
  executor_ = std::make_unique<RdmaDistributedExecutor>(
      [&]() {
        std::vector<petps::PetPSClient*> clients;
        clients.reserve(shard_clients_.size());
        for (auto& client : shard_clients_)
          clients.push_back(client.get());
        return clients;
      }(),
      FLAGS_value_size,
      -1,
      -1);
  initialized_ = true;
}

void RDMAPSClientAdapter::EnsureThreadInitialized() {
  EnsureClientInitialized();
  const std::thread::id tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> guard(thread_init_mu_);
  if (initialized_threads_.find(tid) != initialized_threads_.end()) {
    return;
  }

  if (deployment_.num_shards <= 1) {
    if (client_ != nullptr) {
      client_->InitThread();
    }
  } else {
    for (auto& shard_client : shard_clients_) {
      shard_client->InitThread();
    }
  }

  initialized_threads_.insert(tid);
}

void RDMAPSClientAdapter::EnsureTableReady(const std::string& table_name,
                                           int64_t embedding_dim) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = tables_.find(table_name);
  if (it == tables_.end()) {
    throw std::runtime_error("RDMA table is not initialized: " + table_name);
  }
  if (static_cast<int64_t>(it->second.config.embedding_dim) != embedding_dim) {
    throw std::runtime_error(
        "RDMA embedding dimension mismatch for table " + table_name);
  }
}

int64_t RDMAPSClientAdapter::DefaultEmbeddingDimOrThrow() const {
  std::lock_guard<std::mutex> guard(state_mu_);
  if (tables_.empty()) {
    throw std::runtime_error(
        "RDMA table metadata is empty; call InitEmbeddingTable first");
  }
  return static_cast<int64_t>(tables_.begin()->second.config.embedding_dim);
}

int64_t RDMAPSClientAdapter::EmbeddingDimForKeys(
    base::ConstArray<uint64_t> keys) const {
  if (keys.Size() == 0) {
    return DefaultEmbeddingDimOrThrow();
  }
  const int tag = static_cast<int>(ExtractKeyTag(keys[0]));
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = tag_to_dim_.find(tag);
  if (it == tag_to_dim_.end()) {
    throw std::runtime_error("unknown RDMA key tag: " + std::to_string(tag));
  }
  return it->second;
}

std::size_t RDMAPSClientAdapter::MaxGetKeysPerRpc(int64_t embedding_dim) const {
  const std::size_t value_size =
      static_cast<std::size_t>(embedding_dim) * sizeof(float);
  const std::size_t response_limited = petps::GetKeysPerRpcByResponseBudget(
      value_size,
      static_cast<std::size_t>(FLAGS_rdma_rc_mtu_bytes),
      static_cast<std::size_t>(FLAGS_rdma_rc_target_response_mtu));
  const std::size_t request_limited =
      petps::PutPayloadBudget(
          static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes)) /
      sizeof(std::uint64_t);
  std::size_t limit = static_cast<std::size_t>(FLAGS_max_kv_num_per_request);
  if (response_limited > 0) {
    limit = std::min(limit, response_limited);
  }
  if (request_limited > 0) {
    limit = std::min(limit, request_limited);
  }
  return std::max<std::size_t>(limit, 1);
}

std::size_t RDMAPSClientAdapter::MaxPutKeysPerRpc(int64_t embedding_dim) const {
  const std::size_t payload_budget = petps::PutPayloadBudget(
      static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes));
  const std::size_t bytes_per_row =
      sizeof(ParameterCompressItem) +
      static_cast<std::size_t>(embedding_dim) * sizeof(float) + sizeof(int);
  std::size_t limit = static_cast<std::size_t>(FLAGS_max_kv_num_per_request);
  if (payload_budget > sizeof(int) && bytes_per_row > 0) {
    const std::size_t request_limited =
        (payload_budget - sizeof(int)) / bytes_per_row;
    if (request_limited > 0) {
      limit = std::min(limit, request_limited);
    }
  }
  return std::max<std::size_t>(limit, 1);
}

std::size_t RDMAPSClientAdapter::MaxInFlightGetRpcs() const {
  const std::size_t qps = static_cast<std::size_t>(
      std::max(FLAGS_rdma_rc_qps_per_client_per_shard, 1));
  const std::size_t slots =
      static_cast<std::size_t>(std::max(FLAGS_rdma_rc_slots_per_qp, 1));
  return std::max<std::size_t>((qps - (split_lanes_ ? 1 : 0)) * slots, 1);
}

std::size_t RDMAPSClientAdapter::MaxInFlightCommitRpcs() const {
  // Conservative global window: no shard can exceed its reserved QP slots,
  // including uneven sharding and empty-rank markers.
  return split_lanes_ ? static_cast<std::size_t>(
                            std::max(FLAGS_rdma_rc_slots_per_qp, 1))
                      : MaxInFlightGetRpcs();
}

bool RDMAPSClientAdapter::QueryRPCFinished(int rpc_id) {
  return executor_->QueryBatchFinished(rpc_id);
}

void RDMAPSClientAdapter::WaitRPCFinish(int rpc_id) {
  executor_->WaitBatch(rpc_id);
}

void RDMAPSClientAdapter::RevokeRPCResource(int rpc_id) {
  executor_->ReleaseBatch(rpc_id);
}

int RDMAPSClientAdapter::SubmitGetParameter(
    base::ConstArray<uint64_t> keys,
    float* values,
    bool isAsync,
    int async_req_id,
    int64_t embedding_dim,
    bool write_status_word) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    if (write_status_word) {
      auto* status = reinterpret_cast<std::int32_t*>(values);
      *status      = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    }
    return 0;
  }
  const int value_size =
      static_cast<int>(static_cast<std::size_t>(embedding_dim) * sizeof(float));
  return executor_->SubmitGetParameter(
      BuildChunks(keys, MaxGetKeysPerRpc(embedding_dim)),
      keys.Size(),
      value_size,
      values,
      isAsync,
      async_req_id,
      MaxInFlightGetRpcs(),
      write_status_word);
}

int RDMAPSClientAdapter::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      base::RecTensor& values) {
  EnsureThreadInitialized();
  std::lock_guard<std::mutex> io_guard(io_mu_);
  executor_->DrainBatchSlots();
  if (!IsFloatEmbeddingValues(values, static_cast<int64_t>(keys.Size()))) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }
  const int64_t embedding_dim = values.shape(1);
  const int value_size =
      static_cast<int>(static_cast<std::size_t>(embedding_dim) * sizeof(float));

  // The shard merge writes every requested row, so submit the caller's tensor
  // as the batch destination instead of round-tripping through a scratch
  // buffer. `values` has no room for the trailing status word, so read the
  // batch verdict from WaitBatch.
  const int rpc_id = SubmitGetParameter(
      keys,
      values.data_as<float>(),
      /*isAsync=*/false,
      /*async_req_id=*/0,
      embedding_dim,
      /*write_status_word=*/false);
  const bool ok = executor_->WaitBatch(rpc_id);
  RevokeRPCResource(rpc_id);
  return ok ? 0 : -1;
}

int RDMAPSClientAdapter::PutParameter(const base::ConstArray<uint64_t>& keys,
                                      const base::RecTensor& values) {
  if (!IsFloatEmbeddingValues(values, static_cast<int64_t>(keys.Size()))) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }
  EnsureThreadInitialized();
  std::scoped_lock lifecycle_guard(io_mu_, commit_mu_);
  executor_->DrainBatchSlots();
  return executor_->PutParameter(
      BuildChunks(keys, MaxPutKeysPerRpc(values.shape(1))), values);
}

int64_t RDMAPSClientAdapter::SparseGradientCommit(
    const std::string& table,
    SparseGradientCommitHeader h,
    const base::ConstArray<uint64_t>& keys,
    const base::RecTensor& grads) {
  EnsureClientInitialized();
  EnsureThreadInitialized();
  EnsureTableReady(table, h.dim);
  std::lock_guard<std::mutex> io_guard(split_lanes_ ? commit_mu_ : io_mu_);
  if (!split_lanes_) {
    executor_->DrainBatchSlots();
  }
  if (h.kind == 0 && (grads.dim() != 2 || grads.shape(1) != h.dim ||
                      !IsFloatEmbeddingValues(grads, keys.Size()))) {
    throw std::invalid_argument("Invalid RDMA sparse gradients");
  }
  if (h.version == 4) {
    DecodeSparseGradientCommit(PackSparseGradientCommit(h));
    if (deployment_.num_shards != 2 || deployment_.hash_method != "city_hash" ||
        keys.Size() > 2 * 26 * 1024 ||
        (h.kind != 0 && (keys.Size() || grads.shape(0))))
      throw std::invalid_argument("Unsupported owner replacement deployment or bounds");
    // Validate every candidate before the first RPC can publish any row.
    for (size_t i = 0; i < keys.Size(); ++i) {
      if ((i && keys[i - 1] >= keys[i]) ||
          shard_routing::PartitionKey(keys[i], 2, deployment_.hash_method) != h.rank)
        throw std::invalid_argument("Owner replacement requires sorted owned keys");
    }
    if (!CanonicalAllFinite(grads.data_as<float>(), keys.Size() * h.dim))
      throw std::invalid_argument("Nonfinite owner candidate");
  }
  if (h.version == 4 && h.kind == 3) {
    // Abort replies intentionally fail. An early shard error must not prevent
    // the remaining owners from receiving the same sticky-failure notification.
    std::exception_ptr first_error;
    int64_t committed = std::numeric_limits<int64_t>::max();
    const auto packet = PackSparseGradientCommit(h);
    for (auto& client : shard_clients_) {
      try {
        const auto rpc = client->SubmitSparseGradientCommit(table, packet);
        committed = std::min(committed, client->WaitSparseGradientCommit(rpc));
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    }
    if (first_error) std::rethrow_exception(first_error);
    return committed;
  }
  const size_t raw_limit = shard_clients_.front()->SparseGradientCommitPayloadLimit();
  const size_t limit = h.version == 4 ? std::min(size_t{1} << 20, raw_limit) : raw_limit;
  const size_t row_bytes = 8 + size_t(h.dim) * 4;
  if (limit < sizeof(h) + row_bytes)
    throw std::runtime_error("Gradient row exceeds RDMA slot");
  std::vector<std::vector<ShardChunk>> shards(shard_clients_.size());
  if (h.kind == 0) {
    for (auto& chunk : BuildChunks(keys, (limit - sizeof(h)) / row_bytes)) {
      shards[chunk.client_index].push_back(std::move(chunk));
    }
  }
  size_t rounds = 1;
  for (auto& shard : shards) {
    if (shard.empty())
      shard.emplace_back(); // Empty-rank marker for every PS.
    rounds = std::max(rounds, shard.size());
    if (h.version == 4 && shard.size() > 64)
      throw std::invalid_argument("Owner replacement exceeds 64 chunks");
  }
  std::vector<int64_t> committed(shards.size(), -1);
  std::deque<std::pair<size_t, int>> pending;
  auto drain_one = [&] {
    const auto [client, rpc] = pending.front();
    pending.pop_front();
    committed[client] =
        std::max(committed[client],
                 shard_clients_[client]->WaitSparseGradientCommit(rpc));
  };
  try {
    // Round-robin chunks across shards, with a bounded window of outstanding
    // RC writes. Gather directly into each packet, without a full shard copy.
    for (size_t round = 0; round < rounds; ++round) {
      for (size_t client = 0; client < shards.size(); ++client) {
        if (round >= shards[client].size())
          continue;
        const auto& chunk = shards[client][round];
        h.chunk           = round;
        h.chunks          = shards[client].size();
        h.rows            = chunk.keys.size();
        std::string packet(sizeof(h) + h.rows * row_bytes, '\0');
        std::memcpy(packet.data(), &h, sizeof(h));
        if (h.rows) {
          std::memcpy(packet.data() + sizeof(h), chunk.keys.data(), h.rows * 8);
          char* dst = packet.data() + sizeof(h) + h.rows * 8;
          for (size_t row = 0; row < h.rows; ++row) {
            std::memcpy(dst + row * h.dim * 4,
                        grads.data_as<float>() + chunk.positions[row] * h.dim,
                        h.dim * 4);
          }
        }
        pending.emplace_back(
            client,
            shard_clients_[client]->SubmitSparseGradientCommit(table, packet));
        if (pending.size() >= MaxInFlightCommitRpcs())
          drain_one();
      }
    }
    while (!pending.empty())
      drain_one();
  } catch (...) {
    while (!pending.empty()) {
      try {
        drain_one();
      } catch (...) {
      }
    }
    throw;
  }
  return *std::min_element(committed.begin(), committed.end());
}

int64_t RDMAPSClientAdapter::SparseCanonicalGradientCommit(
    const std::string& table, SparseGradientCommitHeader h,
    const std::vector<CanonicalSparseRecord>& records,
    const base::RecTensor& grads) {
  EnsureClientInitialized();
  EnsureThreadInitialized();
  EnsureTableReady(table, h.dim);
  std::lock_guard<std::mutex> io_guard(split_lanes_ ? commit_mu_ : io_mu_);
  if (!split_lanes_)
    executor_->DrainBatchSlots();
  if (h.version != 3 || h.rows || h.chunk || h.chunks != 1 ||
      grads.dim() != 2 || grads.dtype() != base::DataType::FLOAT32 ||
      grads.shape(1) != 128 || (h.kind != 0 && (!records.empty() || grads.shape(0))))
    throw std::invalid_argument("Invalid canonical RDMA submission");
  // Validate all metadata and construct all packets before posting any update.
  DecodeSparseGradientCommit(PackSparseGradientCommit(h));
  std::vector<size_t> offsets(records.size() + 1, 0);
  std::vector<std::vector<size_t>> positions(shard_clients_.size());
  for (size_t i = 0; i < records.size(); ++i) {
    const auto& record = records[i];
    if (record.reserved || (i && records[i - 1].key >= record.key) ||
        !record.vector_count || record.vector_count != CanonicalExpectedVectors(
            record.count_rank0, record.count_rank1, record.table_type, h.rank))
      throw std::invalid_argument("Invalid canonical RDMA record");
    offsets[i + 1] = offsets[i] + record.vector_count;
    const auto shard = shard_routing::PartitionKey(
        record.key, deployment_.num_shards, deployment_.hash_method);
    positions.at(deployment_.shard_to_endpoint_index.at(shard)).push_back(i);
  }
  if (offsets.back() != static_cast<size_t>(grads.shape(0)))
    throw std::invalid_argument("Canonical RDMA vector count mismatch");
  const auto* values = grads.data_as<float>();
  if (!CanonicalAllFinite(values, offsets.back() * 128))
    throw std::invalid_argument("Nonfinite canonical RDMA gradient");

  std::vector<std::vector<std::string>> packets(shard_clients_.size());
  size_t rounds = 1;
  for (size_t client = 0; client < packets.size(); ++client) {
    const size_t limit = std::min(size_t{1} << 20,
        shard_clients_[client]->SparseGradientCommitPayloadLimit());
    if (limit < sizeof(h))
      throw std::runtime_error("Canonical header exceeds RDMA slot");
    const auto& selected = positions[client];
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t begin = 0, bytes = sizeof(h);
    for (size_t i = 0; i < selected.size(); ++i) {
      const size_t record_bytes = sizeof(CanonicalSparseRecord) +
          size_t(records[selected[i]].vector_count) * 128 * sizeof(float);
      if (record_bytes > limit - sizeof(h))
        throw std::runtime_error("Canonical key record exceeds RDMA slot");
      if (bytes + record_bytes > limit) {
        ranges.emplace_back(begin, i);
        begin = i;
        bytes = sizeof(h);
      }
      bytes += record_bytes;
    }
    if (begin < selected.size() || ranges.empty())
      ranges.emplace_back(begin, selected.size());
    if (ranges.size() > 64)
      throw std::runtime_error("Canonical RDMA submission exceeds 64 chunks");
    h.chunks = ranges.size();
    for (size_t chunk = 0; chunk < ranges.size(); ++chunk) {
      const auto [first, end] = ranges[chunk];
      size_t vectors = 0;
      for (size_t i = first; i < end; ++i)
        vectors += records[selected[i]].vector_count;
      h.chunk = chunk;
      h.rows = end - first;
      std::string packet(sizeof(h) + h.rows * sizeof(CanonicalSparseRecord) +
                         vectors * 128 * sizeof(float), '\0');
      std::memcpy(packet.data(), &h, sizeof(h));
      char* descriptor = packet.data() + sizeof(h);
      char* output = descriptor + h.rows * sizeof(CanonicalSparseRecord);
      for (size_t i = first; i < end; ++i) {
        const size_t position = selected[i];
        const auto& record = records[position];
        std::memcpy(descriptor, &record, sizeof(record));
        descriptor += sizeof(record);
        const size_t value_bytes = size_t(record.vector_count) * 128 * sizeof(float);
        std::memcpy(output, values + offsets[position] * 128, value_bytes);
        output += value_bytes;
      }
      DecodeSparseGradientCommit(packet);
      packets[client].push_back(std::move(packet));
    }
    rounds = std::max(rounds, packets[client].size());
  }
  std::vector<int64_t> committed(packets.size(), -1);
  std::deque<std::pair<size_t, int>> pending;
  auto drain_one = [&] {
    const auto [client, rpc] = pending.front();
    pending.pop_front();
    committed[client] = std::max(committed[client],
        shard_clients_[client]->WaitSparseGradientCommit(rpc));
  };
  try {
    for (size_t round = 0; round < rounds; ++round) {
      for (size_t client = 0; client < packets.size(); ++client) {
        if (round >= packets[client].size())
          continue;
        pending.emplace_back(client, shard_clients_[client]->SubmitSparseGradientCommit(
            table, packets[client][round]));
        if (pending.size() >= MaxInFlightCommitRpcs())
          drain_one();
      }
    }
    while (!pending.empty())
      drain_one();
  } catch (...) {
    while (!pending.empty()) {
      try { drain_one(); } catch (...) {}
    }
    throw;
  }
  return *std::min_element(committed.begin(), committed.end());
}

int RDMAPSClientAdapter::UpdateParameter(const std::string& table_name,
                                         const base::ConstArray<uint64_t>& keys,
                                         const base::RecTensor& grads) {
  if (keys.Size() == 0) {
    return 0;
  }
  return WaitUpdateParameter(
      SubmitUpdateParameterAsync(table_name, keys, grads));
}

uint64_t RDMAPSClientAdapter::SubmitUpdateParameterAsync(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const base::RecTensor& grads) {
  if (split_lanes_) {
    throw std::logic_error("Legacy async update is unsupported with split RDMA lanes");
  }
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA update requires at least one key");
  }
  if (!IsFloatEmbeddingValues(grads, static_cast<int64_t>(keys.Size()))) {
    throw std::invalid_argument("RDMA update has invalid rows or gradients");
  }
  const int64_t embedding_dim = grads.shape(1);
  EnsureTableReady(table_name, embedding_dim);
  EnsureThreadInitialized();
  std::lock_guard<std::mutex> io_guard(io_mu_);
  executor_->DrainBatchSlots();
  return executor_->SubmitUpdateParameterFlat(
      table_name,
      keys,
      grads.data_as<float>(),
      static_cast<std::size_t>(embedding_dim),
      BuildChunks(keys, MaxPutKeysPerRpc(embedding_dim)),
      MaxInFlightGetRpcs());
}

int RDMAPSClientAdapter::WaitUpdateParameter(uint64_t update_id) {
  return executor_->WaitUpdateParameterFlat(update_id);
}

int RDMAPSClientAdapter::InitEmbeddingTable(
    const std::string& table_name, const EmbeddingTableConfig& config) {
  EnsureThreadInitialized();
  std::scoped_lock lifecycle_guard(io_mu_, commit_mu_);
  executor_->DrainBatchSlots();
  int tag = -1;
  for (auto& shard_client : shard_clients_) {
    const int rc = shard_client->InitEmbeddingTable(
        table_name,
        config.num_embeddings,
        config.embedding_dim,
        config.table_id);
    if (rc < 0) {
      return rc;
    }
    if (tag < 0) {
      tag = rc;
    } else if (tag != rc) {
      return -1;
    }
  }
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto [it, inserted] =
      tables_.emplace(table_name, TableState{config, tag});
  if (!inserted && (it->second.config.embedding_dim != config.embedding_dim ||
                    it->second.config.num_embeddings != config.num_embeddings ||
                    it->second.tag != tag)) {
    return -1;
  }
  tag_to_dim_[tag] = static_cast<int64_t>(config.embedding_dim);
  return tag;
}

void RDMAPSClientAdapter::Command(PSCommand) {
  EnsureThreadInitialized();
  std::scoped_lock lifecycle_guard(io_mu_, commit_mu_);
  if (deployment_.num_shards <= 1) {
    if (client_ == nullptr) {
      throw std::runtime_error("RDMA adapter has no initialized client");
    }
    client_->Barrier("rdma_command", 0);
    return;
  }
  if (shard_clients_.empty()) {
    throw std::runtime_error("RDMA adapter has no initialized clients");
  }
  shard_clients_.front()->Barrier("rdma_command", 0);
}

uint64_t
RDMAPSClientAdapter::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
  if (split_lanes_) {
    throw std::logic_error("Legacy prefetch is unsupported with split RDMA lanes");
  }
  EnsureThreadInitialized();
  std::lock_guard<std::mutex> io_guard(io_mu_);
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA prefetch requires at least one key");
  }
  const int64_t embedding_dim = EmbeddingDimForKeys(keys);
  const int value_size =
      static_cast<int>(static_cast<std::size_t>(embedding_dim) * sizeof(float));
  const auto chunks = BuildChunks(keys, MaxGetKeysPerRpc(embedding_dim));
  const std::size_t max_in_flight = MaxInFlightGetRpcs();
  if (!executor_->HasGetSlotCapacity(chunks, max_in_flight)) {
    executor_->DrainBatchSlots();
  }
  return executor_->SubmitPrefetch(
      chunks, keys.Size(), embedding_dim, value_size, max_in_flight);
}

bool RDMAPSClientAdapter::IsPrefetchDone(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  return executor_->IsPrefetchDone(prefetch_id);
}

void RDMAPSClientAdapter::WaitForPrefetch(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  executor_->WaitForPrefetch(prefetch_id);
}

bool RDMAPSClientAdapter::GetPrefetchResult(uint64_t prefetch_id,
                                            base::RecTensor& values) {
  // Keep borrowed MR payloads live until the consumer has copied them.
  std::lock_guard<std::mutex> io_guard(io_mu_);
  const auto result  = executor_->ReadPrefetch(prefetch_id);
  const bool discard = values.data() == nullptr && values.dim() == 0;
  if (!discard && !EnsureEmbeddingOutput(values, result.key_count)) {
    executor_->ReleasePrefetch(prefetch_id);
    return false;
  }
  if (!discard && values.shape(1) != result.embedding_dim) {
    executor_->ReleasePrefetch(prefetch_id);
    return false;
  }
  const bool ok =
      result.status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk);
  if (ok && !discard && !FLAGS_rdma_adapter_skip_prefetch_result_copy &&
      result.payload != nullptr && result.response_bytes > 0) {
    const std::size_t expected_bytes =
        static_cast<std::size_t>(result.key_count) *
        static_cast<std::size_t>(result.embedding_dim) * sizeof(float);
    if (result.response_bytes < expected_bytes) {
      executor_->ReleasePrefetch(prefetch_id);
      return false;
    }
    std::memcpy(values.data_as<float>(), result.payload, expected_bytes);
  }
  executor_->ReleasePrefetch(prefetch_id);
  return ok;
}

} // namespace recstore
