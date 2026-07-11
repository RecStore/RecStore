#include "ps/rdma/rdma_ps_client_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include <folly/portability/GFlags.h>
#include <folly/init/Init.h>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/base/config.h"
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
DECLARE_string(rdma_transport_mode);
DEFINE_string(rdma_transport_mode, "rc_write", "RDMA transport mode: rc_write");
DEFINE_bool(rdma_adapter_skip_prefetch_result_copy,
            false,
            "Benchmark-only option to skip copying RDMA prefetch results into "
            "the GetPrefetchResultFlat output vector");

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
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_CONTROL_PLANE_PORT", &FLAGS_rdma_control_plane_port);
  SetIntFlagFromEnv(
      "RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS",
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

EmbeddedRdmaClientIdentity ResolveEmbeddedRdmaClientIdentity(int num_shards) {
  if (num_shards <= 0) {
    throw std::runtime_error("embedded RDMA num_shards must be positive");
  }

  int client_index = 0;
  if (!detail::TryParseIntEnv("RECSTORE_RDMA_OS_CLIENT_INDEX", &client_index) &&
      !detail::TryParseIntEnv("RANK", &client_index)) {
    detail::TryParseIntEnv("LOCAL_RANK", &client_index);
  }

  int num_client_processes = 1;
  if (!detail::TryParseIntEnv("RECSTORE_RDMA_NUM_CLIENT_PROCESSES",
                              &num_client_processes) &&
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

  EmbeddedRdmaClientIdentity identity;
  identity.client_index         = client_index;
  identity.num_client_processes = num_client_processes;
  identity.global_id            = num_shards + client_index;
  return identity;
}

std::vector<RDMAPSClientAdapter::ShardChunk>
RDMAPSClientAdapter::BuildChunks(base::ConstArray<uint64_t> keys) const {
  return shard_routing::BuildChunks(keys,
                                    num_shards_,
                                    hash_method_,
                                    shard_to_client_index_,
                                    MaxGetKeysPerRpc());
}

void RDMAPSClientAdapter::WaitShardRpcsCooperatively(
    const std::vector<PendingShardRpc>& shard_rpcs) {
  std::vector<bool> finished(shard_rpcs.size(), false);
  std::size_t remaining = shard_rpcs.size();
  while (remaining > 0) {
    bool made_progress = false;
    for (std::size_t i = 0; i < shard_rpcs.size(); ++i) {
      if (finished[i]) {
        continue;
      }
      const auto& pending = shard_rpcs[i];
      auto& client =
          shard_clients_[static_cast<std::size_t>(pending.client_index)];
      if (!client->QueryRPCFinished(pending.rpc_id)) {
        continue;
      }
      client->WaitRPCFinish(pending.rpc_id);
      finished[i]   = true;
      made_progress = true;
      --remaining;
    }
    if (!made_progress) {
      std::this_thread::yield();
    }
  }
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
    : BasePSClient(config), config_(std::move(config)) {}

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

  num_shards_  = dist_cfg.value("num_shards", 1);
  hash_method_ = dist_cfg.value("hash_method", "city_hash");
  if (FLAGS_global_id < num_shards_) {
    const auto identity = ResolveEmbeddedRdmaClientIdentity(num_shards_);
    FLAGS_num_server_processes = num_shards_;
    FLAGS_num_client_processes = identity.num_client_processes;
    FLAGS_global_id            = identity.global_id;
    if (FLAGS_rdma_rc_num_logical_clients < 0) {
      FLAGS_rdma_rc_num_logical_clients = identity.num_client_processes;
    }
    if (FLAGS_rdma_rc_client_id_base < 0) {
      FLAGS_rdma_rc_client_id_base = identity.client_index;
    }
  } else if (FLAGS_num_server_processes != num_shards_) {
    throw std::runtime_error(
        "RDMA num_server_processes must match distributed_client.num_shards");
  }
  FLAGS_value_size =
      cache_ps_cfg.contains("base_kv_config")
          ? ValueSizeHintFromBaseKvConfig(
                cache_ps_cfg["base_kv_config"], FLAGS_value_size)
          : FLAGS_value_size;
  FLAGS_max_kv_num_per_request =
      dist_cfg.value("max_keys_per_request", FLAGS_max_kv_num_per_request);
  if (const char* mode = std::getenv("RECSTORE_RDMA_TRANSPORT_MODE")) {
    FLAGS_rdma_transport_mode = mode;
  }

  const int logical_client_id =
      config_.value("rdma_logical_client_id", FLAGS_rdma_rc_client_id_base);

  shard_clients_.clear();
  shard_to_client_index_.clear();
  client_ = nullptr;

  if (num_shards_ <= 1) {
    shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
        client_cfg.value("host", std::string("127.0.0.1")),
        client_cfg.value("port", 25000),
        client_cfg.value("shard", 0),
        logical_client_id));
    client_                   = shard_clients_.front().get();
    shard_to_client_index_[0] = 0;
  } else {
    const auto servers_it = dist_cfg.find("servers");
    if (servers_it == dist_cfg.end() || !servers_it->is_array() ||
        servers_it->empty()) {
      throw std::runtime_error(
          "RDMA distributed_client.servers must be provided for multi-shard "
          "configuration");
    }

    CHECK_EQ(static_cast<int>(servers_it->size()), num_shards_)
        << "RDMA distributed_client.servers size must equal num_shards";
    for (const auto& server : *servers_it) {
      const int shard = server.value("shard", -1);
      if (shard < 0) {
        throw std::runtime_error(
            "RDMA distributed_client.servers[].shard must be explicit");
      }
      shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
          server.value("host", std::string("127.0.0.1")),
          server.value("port", 25000),
          shard,
          logical_client_id));
      shard_to_client_index_[shard] =
          static_cast<int>(shard_clients_.size() - 1);
    }
  }

  initialized_ = true;
}

void RDMAPSClientAdapter::EnsureThreadInitialized() {
  EnsureClientInitialized();
  const std::thread::id tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> guard(thread_init_mu_);
  if (initialized_threads_.find(tid) != initialized_threads_.end()) {
    return;
  }

  if (num_shards_ <= 1) {
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
  if (tables_.empty()) {
    throw std::runtime_error(
        "RDMA table metadata is empty; call InitEmbeddingTable first");
  }
  return static_cast<int64_t>(tables_.begin()->second.config.embedding_dim);
}

std::size_t RDMAPSClientAdapter::MaxGetKeysPerRpc() const {
  const std::size_t response_limited = petps::GetKeysPerRpcByResponseBudget(
      static_cast<std::size_t>(FLAGS_value_size),
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

std::size_t RDMAPSClientAdapter::MaxPutKeysPerRpc() const {
  const std::size_t payload_budget = petps::PutPayloadBudget(
      static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes));
  const std::size_t embedding_dim =
      static_cast<std::size_t>(DefaultEmbeddingDimOrThrow());
  const std::size_t bytes_per_row =
      sizeof(ParameterCompressItem) + embedding_dim * sizeof(float) +
      sizeof(int);
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
  return std::max<std::size_t>(qps * slots, 1);
}

RDMAPSClientAdapter::PrefetchState
RDMAPSClientAdapter::GetPrefetchState(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = prefetches_.find(prefetch_id);
  if (it == prefetches_.end()) {
    throw std::runtime_error(
        "Unknown RDMA prefetch id: " + std::to_string(prefetch_id));
  }
  return it->second;
}

void RDMAPSClientAdapter::MarkPrefetchConsumed(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = prefetches_.find(prefetch_id);
  if (it == prefetches_.end()) {
    return;
  }
  free_prefetch_buffer_ids_.push_back(it->second.buffer_id);
  prefetches_.erase(it);
}

bool RDMAPSClientAdapter::QueryRPCFinished(int rpc_id) {
  if (rpc_id >= 0 && num_shards_ <= 1) {
    return client_ != nullptr ? client_->QueryRPCFinished(rpc_id) : true;
  }

  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    if (!shard_clients_[static_cast<std::size_t>(pending.client_index)]
             ->QueryRPCFinished(pending.rpc_id)) {
      return false;
    }
  }

  return shard_routing::FinalizeBatchIfNeeded(&it->second, FLAGS_value_size);
}

void RDMAPSClientAdapter::WaitRPCFinish(int rpc_id) {
  if (rpc_id >= 0 && num_shards_ <= 1) {
    if (client_ != nullptr) {
      client_->WaitRPCFinish(rpc_id);
    }
    return;
  }

  std::vector<PendingShardRpc> shard_rpcs;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(rpc_id);
    CHECK(it != batches_.end());
    if (it->second.assembled) {
      return;
    }
    shard_rpcs = it->second.shard_rpcs;
  }

  WaitShardRpcsCooperatively(shard_rpcs);

  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(rpc_id);
    CHECK(it != batches_.end());
    shard_routing::FinalizeBatchIfNeeded(&it->second, FLAGS_value_size);
  }
}

void RDMAPSClientAdapter::RevokeRPCResource(int rpc_id) {
  if (rpc_id >= 0 && num_shards_ <= 1) {
    if (client_ != nullptr) {
      client_->RevokeRPCResource(rpc_id);
    }
    return;
  }

  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    shard_clients_[static_cast<std::size_t>(pending.client_index)]
        ->RevokeRPCResource(pending.rpc_id);
  }

  batches_.erase(it);
}

float* RDMAPSClientAdapter::AcquirePrefetchBuffer(std::size_t bytes,
                                                  std::size_t* buffer_id) {
  if (buffer_id == nullptr) {
    throw std::invalid_argument("RDMA prefetch buffer_id must not be null");
  }
  std::lock_guard<std::mutex> guard(state_mu_);
  while (!free_prefetch_buffer_ids_.empty()) {
    const std::size_t id = free_prefetch_buffer_ids_.back();
    free_prefetch_buffer_ids_.pop_back();
    if (id < prefetch_buffer_capacities_.size() &&
        prefetch_buffer_capacities_[id] >= bytes) {
      *buffer_id = id;
      return reinterpret_cast<float*>(prefetch_buffers_[id].get());
    }
  }

  const std::size_t id = prefetch_buffers_.size();
  prefetch_buffers_.push_back(std::make_unique<char[]>(bytes));
  prefetch_buffer_capacities_.push_back(bytes);
  *buffer_id = id;
  return reinterpret_cast<float*>(prefetch_buffers_.back().get());
}

void RDMAPSClientAdapter::ReleasePrefetchBuffer(std::size_t buffer_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  if (buffer_id < prefetch_buffers_.size()) {
    free_prefetch_buffer_ids_.push_back(buffer_id);
  }
}

const float* RDMAPSClientAdapter::BorrowPrefetchResult(
    const PrefetchState& state,
    std::int32_t* status_code,
    std::size_t* response_bytes) {
  if (!state.borrowed_response || client_ == nullptr) {
    return nullptr;
  }
  auto* pet_client = dynamic_cast<petps::PetPSClient*>(client_);
  if (pet_client == nullptr) {
    return nullptr;
  }
  std::size_t key_count = 0;
  const float* payload  = pet_client->BorrowGetResultPayload(
      state.rpc_id, &key_count, response_bytes, status_code);
  if (payload == nullptr ||
      key_count != static_cast<std::size_t>(state.key_count)) {
    return nullptr;
  }
  return payload;
}

int RDMAPSClientAdapter::SubmitGetParameter(
    base::ConstArray<uint64_t> keys,
    float* values,
    bool isAsync,
    int async_req_id) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    auto* status =
        reinterpret_cast<std::int32_t*>(reinterpret_cast<char*>(values));
    *status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    return 0;
  }

  BatchRequest batch;
  batch.user_buffer     = values;
  batch.total_key_count = keys.Size();
  auto* batch_status_word =
      petps::FixedSlotStatusWord(values, keys.Size(), FLAGS_value_size);
  *batch_status_word = static_cast<std::int32_t>(petps::RpcStatus::kPending);

  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    const std::size_t max_keys_per_rpc = MaxGetKeysPerRpc();
    const std::size_t max_in_flight    = MaxInFlightGetRpcs();
    const std::size_t total_keys       = keys.Size();
    if (total_keys <= max_keys_per_rpc) {
      return client_->GetParameter(keys, values, isAsync, async_req_id);
    }
    std::vector<PendingShardRpc> window;
    window.reserve(max_in_flight);
    auto drain_and_release_window = [this, &window, &batch]() {
      // Large model batches can split into more GET RPCs than the RC slot pool.
      // Keep submission bounded by waiting and freeing each window before
      // acquiring more slots.
      for (const auto& pending : window) {
        client_->WaitRPCFinish(pending.rpc_id);
      }
      for (const auto& pending : window) {
        batch.shard_rpcs.push_back(pending);
        client_->RevokeRPCResource(pending.rpc_id);
      }
      window.clear();
    };
    for (std::size_t offset = 0; offset < total_keys;
         offset += max_keys_per_rpc) {
      const std::size_t end = std::min(offset + max_keys_per_rpc, total_keys);
      std::vector<uint64_t> key_slice;
      key_slice.reserve(end - offset);
      std::vector<std::size_t> positions;
      positions.reserve(end - offset);
      for (std::size_t i = offset; i < end; ++i) {
        key_slice.push_back(keys[i]);
        positions.push_back(i);
      }
      void* recv = client_->GetReceiveBuffer(
          key_slice.size() * static_cast<std::size_t>(FLAGS_value_size) +
          sizeof(std::int32_t));
      const int rpc_id = client_->GetParameter(
          base::ConstArray<uint64_t>(key_slice),
          static_cast<float*>(recv),
          isAsync,
          async_req_id);
      window.push_back(PendingShardRpc{
          0,
          0,
          rpc_id,
          std::move(positions),
          recv,
          key_slice.size(),
      });
      if (window.size() >= max_in_flight) {
        drain_and_release_window();
      }
    }
    if (!window.empty()) {
      drain_and_release_window();
    }
  } else {
    const std::size_t max_in_flight = MaxInFlightGetRpcs();
    std::vector<PendingShardRpc> window;
    window.reserve(max_in_flight);
    auto drain_and_release_window = [this, &window, &batch]() {
      WaitShardRpcsCooperatively(window);
      for (const auto& pending : window) {
        batch.shard_rpcs.push_back(pending);
        shard_clients_[static_cast<std::size_t>(pending.client_index)]
            ->RevokeRPCResource(pending.rpc_id);
      }
      window.clear();
    };
    for (const auto& chunk : BuildChunks(keys)) {
      BaseParameterClient* client = shard_clients_[chunk.client_index].get();
      void* recv                  = client->GetReceiveBuffer(
          chunk.keys.size() * static_cast<std::size_t>(FLAGS_value_size) +
          sizeof(std::int32_t));
      const int rpc_id = client->GetParameter(
          base::ConstArray<uint64_t>(chunk.keys),
          static_cast<float*>(recv),
          isAsync,
          async_req_id);
      window.push_back(PendingShardRpc{
          chunk.shard_id,
          chunk.client_index,
          rpc_id,
          chunk.positions,
          recv,
          chunk.keys.size(),
      });
      if (window.size() >= max_in_flight) {
        drain_and_release_window();
      }
    }
    if (!window.empty()) {
      drain_and_release_window();
    }
  }

  int batch_id = 0;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    batch_id = batch_rpc_id_acc_--;
    if (batch_id >= 0) {
      throw std::runtime_error("rdma batch rpc id exhausted negative range");
    }
    batches_[batch_id] = std::move(batch);
  }
  if (!isAsync) {
    WaitRPCFinish(batch_id);
  }
  return batch_id;
}

int RDMAPSClientAdapter::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      float* values) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    return 0;
  }

  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  float* recv = nullptr;
  if (num_shards_ > 1) {
    if (shard_clients_.empty()) {
      return -1;
    }
    recv = static_cast<float*>(
        shard_clients_.front()->GetReceiveBuffer(response_bytes));
  } else {
    recv = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
  }

  const int rpc_id = SubmitGetParameter(keys, recv, false, 0);
  WaitRPCFinish(rpc_id);
  const auto* status_word =
      petps::FixedSlotStatusWord(recv, keys.Size(), FLAGS_value_size);
  if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }

  std::memcpy(
      values, recv, keys.Size() * static_cast<std::size_t>(FLAGS_value_size));
  RevokeRPCResource(rpc_id);
  return 0;
}

int RDMAPSClientAdapter::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  EnsureThreadInitialized();
  if (keys.Size() != values.size()) {
    return -1;
  }
  const std::size_t max_keys_per_rpc = MaxPutKeysPerRpc();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    const std::size_t key_count = static_cast<std::size_t>(keys.Size());
    for (std::size_t offset = 0; offset < key_count;
         offset += max_keys_per_rpc) {
      const std::size_t end = std::min(offset + max_keys_per_rpc, key_count);
      std::vector<uint64_t> key_slice;
      key_slice.reserve(end - offset);
      for (std::size_t i = offset; i < end; ++i) {
        key_slice.push_back(keys[i]);
      }
      std::vector<std::vector<float>> value_slice(
          values.begin() + static_cast<std::ptrdiff_t>(offset),
          values.begin() + static_cast<std::ptrdiff_t>(end));
      const int rc = client_->PutParameter(key_slice, value_slice);
      if (rc != 0) {
        return rc;
      }
    }
    return 0;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_values(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard =
        shard_routing::PartitionKey(keys[i], num_shards_, hash_method_);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_values[static_cast<std::size_t>(shard)].push_back(values[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    const int client_index = shard_to_client_index_.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += max_keys_per_rpc) {
      const std::size_t end = std::min(
          offset + max_keys_per_rpc,
          shard_keys[static_cast<std::size_t>(shard)].size());
      std::vector<uint64_t> key_slice(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      std::vector<std::vector<float>> value_slice(
          shard_values[static_cast<std::size_t>(shard)].begin() + offset,
          shard_values[static_cast<std::size_t>(shard)].begin() + end);
      int rc =
          shard_clients_[static_cast<std::size_t>(client_index)]->PutParameter(
              key_slice, value_slice);
      if (rc != 0) {
        return rc;
      }
    }
  }
  return 0;
}

int RDMAPSClientAdapter::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (grads->empty()) {
    return 0;
  }
  EnsureThreadInitialized();
  const std::size_t max_keys_per_rpc = MaxPutKeysPerRpc();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    const std::size_t key_count = static_cast<std::size_t>(keys.Size());
    for (std::size_t offset = 0; offset < key_count;
         offset += max_keys_per_rpc) {
      const std::size_t end = std::min(offset + max_keys_per_rpc, key_count);
      std::vector<uint64_t> key_slice;
      key_slice.reserve(end - offset);
      for (std::size_t i = offset; i < end; ++i) {
        key_slice.push_back(keys[i]);
      }
      std::vector<std::vector<float>> grad_slice(
          grads->begin() + static_cast<std::ptrdiff_t>(offset),
          grads->begin() + static_cast<std::ptrdiff_t>(end));
      const int rc = client_->UpdateParameter(
          table_name, base::ConstArray<uint64_t>(key_slice), &grad_slice);
      if (rc != 0) {
        return rc;
      }
    }
    return 0;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_grads(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard =
        shard_routing::PartitionKey(keys[i], num_shards_, hash_method_);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_grads[static_cast<std::size_t>(shard)].push_back((*grads)[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    if (shard_keys[static_cast<std::size_t>(shard)].empty()) {
      continue;
    }
    const int client_index = shard_to_client_index_.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += max_keys_per_rpc) {
      const std::size_t end = std::min(
          offset + max_keys_per_rpc,
          shard_keys[static_cast<std::size_t>(shard)].size());
      std::vector<uint64_t> key_slice(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      std::vector<std::vector<float>> grad_slice(
          shard_grads[static_cast<std::size_t>(shard)].begin() + offset,
          shard_grads[static_cast<std::size_t>(shard)].begin() + end);
      const int rc =
          shard_clients_[static_cast<std::size_t>(client_index)]->UpdateParameter(
              table_name, base::ConstArray<uint64_t>(key_slice), &grad_slice);
      if (rc != 0) {
        return rc;
      }
    }
  }
  return 0;
}

int RDMAPSClientAdapter::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  EnsureTableReady(table_name, embedding_dim);
  if (num_rows == 0) {
    return 0;
  }
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != static_cast<std::size_t>(num_rows)) {
    return -1;
  }
  std::vector<std::vector<float>> updated;
  updated.reserve(static_cast<std::size_t>(num_rows));
  for (int64_t row = 0; row < num_rows; ++row) {
    std::vector<float> values(static_cast<std::size_t>(embedding_dim), 0.0f);
    for (int64_t col = 0; col < embedding_dim; ++col) {
      const std::size_t idx =
          static_cast<std::size_t>(row * embedding_dim + col);
      values[static_cast<std::size_t>(col)] = grads[idx];
    }
    updated.push_back(std::move(values));
  }

  return UpdateParameter(table_name, keys, &updated);
}

int RDMAPSClientAdapter::InitEmbeddingTable(
    const std::string& table_name, const EmbeddingTableConfig& config) {
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    const int init_rc = client_->InitEmbeddingTable(
        table_name, config.num_embeddings, config.embedding_dim);
    if (init_rc != 0) {
      return init_rc;
    }
  } else {
    for (auto& shard_client : shard_clients_) {
      const int rc = shard_client->InitEmbeddingTable(
          table_name, config.num_embeddings, config.embedding_dim);
      if (rc != 0) {
        return rc;
      }
    }
  }

  std::lock_guard<std::mutex> guard(state_mu_);
  const auto [it, inserted] = tables_.emplace(table_name, TableState{config});
  if (!inserted) {
    if (it->second.config.embedding_dim != config.embedding_dim ||
        it->second.config.num_embeddings != config.num_embeddings) {
      return -1;
    }
  }
  return 0;
}

int RDMAPSClientAdapter::AsyncGetParameter(const base::ConstArray<uint64_t>&,
                                           float*) {
  throw std::runtime_error(
      "RDMA adapter AsyncGetParameter not implemented yet");
}

void RDMAPSClientAdapter::Command(PSCommand) {
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
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
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA prefetch requires at least one key");
  }

  const int64_t embedding_dim = DefaultEmbeddingDimOrThrow();
  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  std::size_t buffer_id = 0;
  const bool borrow_single_shard_response =
      num_shards_ <= 1 && keys.Size() <= MaxGetKeysPerRpc();
  const bool batch_response = !borrow_single_shard_response;
  float* buffer             = AcquirePrefetchBuffer(response_bytes, &buffer_id);
  auto* status_word         = petps::FixedSlotStatusWord(
      buffer, static_cast<std::size_t>(keys.Size()), FLAGS_value_size);
  *status_word = static_cast<std::int32_t>(petps::RpcStatus::kPending);

  int rpc_id = -1;
  try {
    rpc_id = SubmitGetParameter(keys, buffer, true, 0);
  } catch (...) {
    ReleasePrefetchBuffer(buffer_id);
    throw;
  }

  std::lock_guard<std::mutex> guard(state_mu_);
  const uint64_t prefetch_id = next_prefetch_id_++;
  prefetches_.emplace(
      prefetch_id,
      PrefetchState{
          buffer,
          buffer_id,
          rpc_id,
          static_cast<int64_t>(keys.Size()),
          embedding_dim,
          borrow_single_shard_response,
          batch_response,
      });
  return prefetch_id;
}

bool RDMAPSClientAdapter::IsPrefetchDone(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  const PrefetchState state = GetPrefetchState(prefetch_id);
  return QueryRPCFinished(state.rpc_id);
}

void RDMAPSClientAdapter::WaitForPrefetch(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  const PrefetchState state = GetPrefetchState(prefetch_id);
  try {
    WaitRPCFinish(state.rpc_id);
  } catch (...) {
    RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    throw;
  }
}

bool RDMAPSClientAdapter::GetPrefetchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  if (values == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  std::vector<float> flat;
  int64_t num_rows = 0;
  if (!GetPrefetchResultFlat(
          prefetch_id, &flat, &num_rows, state.embedding_dim)) {
    return false;
  }

  petps::CopyFlatRowsToVectors(
      flat.data(),
      static_cast<std::size_t>(num_rows),
      static_cast<std::size_t>(state.embedding_dim),
      values);
  return true;
}

bool RDMAPSClientAdapter::GetPrefetchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  if (values == nullptr || num_rows == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  if (embedding_dim != state.embedding_dim) {
    return false;
  }

  const bool profile_enabled = AdapterProfileEnabled();
  const auto wait_begin      = std::chrono::steady_clock::now();
  std::int32_t status_code   = static_cast<std::int32_t>(petps::RpcStatus::kOk);
  std::size_t response_bytes = 0;
  const float* result_payload =
      BorrowPrefetchResult(state, &status_code, &response_bytes);
  if (result_payload == nullptr) {
    WaitForPrefetch(prefetch_id);
    const auto* status_word = petps::FixedSlotStatusWord(
        state.buffer,
        static_cast<std::size_t>(state.key_count),
        FLAGS_value_size);
    status_code    = *status_word;
    response_bytes = static_cast<std::size_t>(state.key_count) *
                     static_cast<std::size_t>(FLAGS_value_size);
    result_payload = state.buffer;
  }
  const auto wait_end = std::chrono::steady_clock::now();
  if (status_code != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    return false;
  }

  const std::size_t value_count =
      static_cast<std::size_t>(state.key_count) *
      static_cast<std::size_t>(state.embedding_dim);
  const auto assign_begin = std::chrono::steady_clock::now();
  if (FLAGS_rdma_adapter_skip_prefetch_result_copy) {
    values->clear();
  } else if (response_bytes == 0) {
    values->clear();
  } else {
    values->resize(value_count);
    if (value_count > 0) {
      const std::size_t expected_bytes = value_count * sizeof(values->front());
      if (response_bytes < expected_bytes) {
        RevokeRPCResource(state.rpc_id);
        MarkPrefetchConsumed(prefetch_id);
        return false;
      }
      std::memcpy(values->data(), result_payload, expected_bytes);
    }
  }
  const auto assign_end   = std::chrono::steady_clock::now();
  *num_rows               = state.key_count;
  const auto revoke_begin = std::chrono::steady_clock::now();
  RevokeRPCResource(state.rpc_id);
  MarkPrefetchConsumed(prefetch_id);
  const auto revoke_end = std::chrono::steady_clock::now();
  if (profile_enabled) {
    static std::atomic<std::uint64_t> count{0};
    static std::atomic<std::uint64_t> wait_ns{0};
    static std::atomic<std::uint64_t> assign_ns{0};
    static std::atomic<std::uint64_t> revoke_ns{0};
    const std::uint64_t current = count.fetch_add(1) + 1;
    wait_ns.fetch_add(
        static_cast<std::uint64_t>(NsSince(wait_begin, wait_end)));
    assign_ns.fetch_add(
        static_cast<std::uint64_t>(NsSince(assign_begin, assign_end)));
    revoke_ns.fetch_add(
        static_cast<std::uint64_t>(NsSince(revoke_begin, revoke_end)));
    if (current == 1 || current % 512 == 0) {
      const double denom = static_cast<double>(current);
      std::cout
          << "component=rdma_adapter_prefetch_profile"
          << " batches=" << current
          << " wait_avg_ns=" << static_cast<double>(wait_ns.load()) / denom
          << " assign_avg_ns=" << static_cast<double>(assign_ns.load()) / denom
          << " revoke_avg_ns=" << static_cast<double>(revoke_ns.load()) / denom
          << " value_count=" << value_count << std::endl;
    }
  }
  return true;
}

} // namespace recstore
