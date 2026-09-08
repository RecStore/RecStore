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

std::vector<RDMAPSClientAdapter::ShardChunk>
RDMAPSClientAdapter::BuildChunks(base::ConstArray<uint64_t> keys,
                                 std::size_t max_keys_per_rpc) const {
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

  shard_clients_.clear();
  client_ = nullptr;

  if (deployment_.num_shards <= 1) {
    shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
        client_cfg.value("host", std::string("127.0.0.1")),
        client_cfg.value("port", 25000),
        client_cfg.value("shard", 0),
        logical_client_id));
    client_                   = shard_clients_.front().get();
  } else {
    for (const auto& endpoint : deployment_.endpoints) {
      shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
          endpoint.host,
          endpoint.port,
          endpoint.shard,
          logical_client_id));
    }
  }

  executor_ = std::make_unique<RdmaDistributedExecutor>(
      [&]() {
        std::vector<petps::PetPSClient*> clients;
        clients.reserve(shard_clients_.size());
        for (auto& client : shard_clients_) clients.push_back(client.get());
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
  return std::max<std::size_t>(qps * slots, 1);
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
    int64_t embedding_dim) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    auto* status = reinterpret_cast<std::int32_t*>(values);
    *status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
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
      MaxInFlightGetRpcs());
}

int RDMAPSClientAdapter::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      base::RecTensor& values) {
  EnsureThreadInitialized();
  if (!IsFloatEmbeddingValues(values, static_cast<int64_t>(keys.Size()))) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }
  const int64_t embedding_dim = values.shape(1);
  const int value_size =
      static_cast<int>(static_cast<std::size_t>(embedding_dim) * sizeof(float));

  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), value_size);
  std::vector<float> recv_storage(
      (response_bytes + sizeof(float) - 1) / sizeof(float), 0.0f);
  float* recv = recv_storage.data();

  const int rpc_id =
      SubmitGetParameter(keys, recv, false, 0, embedding_dim);
  WaitRPCFinish(rpc_id);
  const auto* status_word =
      petps::FixedSlotStatusWord(recv, keys.Size(), value_size);
  if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }

  std::memcpy(
      values.data_as<float>(),
      recv,
      keys.Size() * static_cast<std::size_t>(value_size));
  RevokeRPCResource(rpc_id);
  return 0;
}

int RDMAPSClientAdapter::PutParameter(
    const base::ConstArray<uint64_t>& keys, const base::RecTensor& values) {
  if (!IsFloatEmbeddingValues(values, static_cast<int64_t>(keys.Size()))) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }
  EnsureThreadInitialized();
  return executor_->PutParameter(
      BuildChunks(keys, MaxPutKeysPerRpc(values.shape(1))), values);
}

int RDMAPSClientAdapter::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const base::RecTensor& grads) {
  return WaitUpdateParameter(
      SubmitUpdateParameterAsync(table_name, keys, grads));
}

uint64_t RDMAPSClientAdapter::SubmitUpdateParameterAsync(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const base::RecTensor& grads) {
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA update requires at least one key");
  }
  if (!IsFloatEmbeddingValues(grads, static_cast<int64_t>(keys.Size()))) {
    throw std::invalid_argument("RDMA update has invalid rows or gradients");
  }
  const int64_t embedding_dim = grads.shape(1);
  EnsureTableReady(table_name, embedding_dim);
  EnsureThreadInitialized();
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
  int tag = -1;
  for (auto& shard_client : shard_clients_) {
    const int rc = shard_client->InitEmbeddingTable(
        table_name, config.num_embeddings, config.embedding_dim, config.table_id);
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
  if (!inserted &&
      (it->second.config.embedding_dim != config.embedding_dim ||
       it->second.config.num_embeddings != config.num_embeddings ||
       it->second.tag != tag)) {
    return -1;
  }
  tag_to_dim_[tag] = static_cast<int64_t>(config.embedding_dim);
  return tag;
}

void RDMAPSClientAdapter::Command(PSCommand) {
  EnsureThreadInitialized();
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
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA prefetch requires at least one key");
  }
  const int64_t embedding_dim = EmbeddingDimForKeys(keys);
  const int value_size =
      static_cast<int>(static_cast<std::size_t>(embedding_dim) * sizeof(float));
  return executor_->SubmitPrefetch(
      BuildChunks(keys, MaxGetKeysPerRpc(embedding_dim)),
      keys.Size(),
      embedding_dim,
      value_size,
      MaxInFlightGetRpcs());
}

bool RDMAPSClientAdapter::IsPrefetchDone(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  return executor_->IsPrefetchDone(prefetch_id);
}

void RDMAPSClientAdapter::WaitForPrefetch(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  executor_->WaitForPrefetch(prefetch_id);
}

bool RDMAPSClientAdapter::GetPrefetchResult(
    uint64_t prefetch_id, base::RecTensor& values) {
  const auto result = executor_->ReadPrefetch(prefetch_id);
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
