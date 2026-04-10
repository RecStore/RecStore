#include "dist_grpc_ps_client.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <thread>
#include <fstream>

#include "base/factory.h"
#include "base/log.h"
#include "base/timer.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::Status;
using recstoreps::CommandRequest;
using recstoreps::CommandResponse;
using recstoreps::GetParameterRequest;
using recstoreps::GetParameterResponse;
using recstoreps::InitEmbeddingTableRequest;
using recstoreps::InitEmbeddingTableResponse;
using recstoreps::PSCommand;
using recstoreps::PutParameterRequest;
using recstoreps::PutParameterResponse;
using recstoreps::UpdateParameterRequest;
using recstoreps::UpdateParameterResponse;

namespace recstore {

FACTORY_REGISTER(
    BasePSClient, distributed_grpc, DistributedGRPCParameterClient, json);

DistributedGRPCParameterClient::DistributedGRPCParameterClient(json config)
    : BasePSClient(config) {
  json client_config;

  if (config.contains("distributed_client")) {
    LOG(INFO) << "Detected recstore config format, extracting "
                 "distributed_client section";
    client_config = config["distributed_client"];
  } else {
    LOG(FATAL)
        << "Invalid config format. Expected either recstore config with "
           "'distributed_client' section "
        << "or direct client config with 'servers' and 'num_shards' fields";
  }

  // 验证必要字段
  if (!client_config.contains("servers") ||
      !client_config["servers"].is_array()) {
    LOG(FATAL)
        << "Missing or invalid 'servers' field in distributed client config";
  }

  if (!client_config.contains("num_shards") ||
      !client_config["num_shards"].is_number_integer()) {
    LOG(FATAL)
        << "Missing or invalid 'num_shards' field in distributed client config";
  }

  num_shards_           = client_config["num_shards"].get<int>();
  max_keys_per_request_ = client_config.value("max_keys_per_request", 500);
  hash_method_          = client_config.value("hash_method", "city_hash");
  if (max_keys_per_request_ <= 0) {
    LOG(FATAL) << "Invalid max_keys_per_request: " << max_keys_per_request_
               << ", must be > 0";
  }

  // 解析服务器配置
  auto servers = client_config["servers"];
  server_configs_.reserve(servers.size());

  for (size_t i = 0; i < servers.size(); ++i) {
    const auto& server = servers[i];
    if (!server.contains("host") || !server.contains("port") ||
        !server.contains("shard")) {
      LOG(FATAL) << "Server config " << i
                 << " missing required fields (host, port, shard)";
    }

    ServerConfig cfg;
    cfg.host  = server["host"].get<std::string>();
    cfg.port  = server["port"].get<int>();
    cfg.shard = server["shard"].get<int>();

    server_configs_.push_back(cfg);
    shard_to_client_index_[cfg.shard] = i;
  }

  if (server_configs_.size() != static_cast<size_t>(num_shards_)) {
    LOG(WARNING) << "Number of servers (" << server_configs_.size()
                 << ") doesn't match num_shards (" << num_shards_ << ")";
  }

  partitioned_key_buffer_.resize(num_shards_);
  key_index_mapping_.resize(num_shards_);

  InitializeClients();

  LOG(INFO) << "Initialized DistributedGRPCParameterClient with " << num_shards_
            << " shards, hash method: " << hash_method_;
}

DistributedGRPCParameterClient::~DistributedGRPCParameterClient() {}

void DistributedGRPCParameterClient::InitializeClients() {
  clients_.clear();
  clients_.reserve(server_configs_.size());

  for (const auto& server_config : server_configs_) {
    // 为每个服务器创建独立的配置
    json client_config = {{"host", server_config.host},
                          {"port", server_config.port},
                          {"shard", server_config.shard}};

    auto client = std::make_unique<GRPCParameterClient>(client_config);
    clients_.push_back(std::move(client));

    LOG(INFO) << "Created gRPC client for shard " << server_config.shard
              << " at " << server_config.host << ":" << server_config.port;
  }
}

int DistributedGRPCParameterClient::GetShardId(uint64_t key) const {
  if (hash_method_ == "city_hash") {
    return GetHash(key) % num_shards_;
  } else if (hash_method_ == "simple_mod") {
    return key % num_shards_;
  } else {
    LOG(ERROR) << "Unknown hash method: " << hash_method_
               << ", using city_hash";
    return GetHash(key) % num_shards_;
  }
}

void DistributedGRPCParameterClient::PartitionKeys(
    const base::ConstArray<uint64_t>& keys,
    std::vector<std::vector<uint64_t>>& partitioned_keys) const {
  for (auto& partition : partitioned_key_buffer_) {
    partition.clear();
  }
  for (auto& mapping : key_index_mapping_) {
    mapping.clear();
  }

  for (size_t i = 0; i < keys.Size(); ++i) {
    uint64_t key = keys[i];
    int shard_id = GetShardId(key);

    partitioned_key_buffer_[shard_id].push_back(key);
    key_index_mapping_[shard_id].push_back(i);
  }

  partitioned_keys = partitioned_key_buffer_;
}

bool DistributedGRPCParameterClient::GetParameter(
    const base::ConstArray<uint64_t>& keys,
    std::vector<std::vector<float>>* values) {
  if (keys.Size() == 0) {
    values->clear();
    return true;
  }

  xmh::Timer timer("DistributedGRPCParameterClient::GetParameter");

  std::vector<std::vector<uint64_t>> partitioned_keys;
  PartitionKeys(keys, partitioned_keys);

  std::vector<std::future<int>> futures;
  std::vector<std::vector<std::vector<float>>> partitioned_results(num_shards_);

  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (partitioned_keys[shard_id].empty()) {
      continue;
    }

    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      return false;
    }

    int client_index = it->second;
    auto* client     = clients_[client_index].get();

    // 异步请求
    futures.push_back(std::async(
        std::launch::async, [=, &partitioned_keys, &partitioned_results]() {
          const auto& shard_keys_vec = partitioned_keys[shard_id];
          auto& shard_result_vec     = partitioned_results[shard_id];
          shard_result_vec.clear();
          shard_result_vec.reserve(shard_keys_vec.size());

          for (size_t start = 0; start < shard_keys_vec.size();
               start += static_cast<size_t>(max_keys_per_request_)) {
            size_t end =
                std::min(start + static_cast<size_t>(max_keys_per_request_),
                         shard_keys_vec.size());
            base::ConstArray<uint64_t> shard_chunk(
                shard_keys_vec.data() + start, static_cast<int>(end - start));
            std::vector<std::vector<float>> chunk_result;
            if (!client->GetParameter(shard_chunk, &chunk_result)) {
              return 0;
            }
            shard_result_vec.insert(shard_result_vec.end(),
                                    chunk_result.begin(),
                                    chunk_result.end());
          }
          return 1;
        }));
  }

  // 3. 等待所有请求完成
  for (auto& future : futures) {
    if (!future.get()) {
      LOG(ERROR) << "Failed to get parameters from one of the shards";
      return false;
    }
  }

  // 4. 合并结果
  MergeResults(keys, partitioned_results, values);

  return true;
}

void DistributedGRPCParameterClient::MergeResults(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<std::vector<float>>>& partitioned_results,
    std::vector<std::vector<float>>* values) const {
  values->clear();
  values->resize(keys.Size());

  // 重建key -> index映射
  std::unordered_map<uint64_t, size_t> key_to_result_index;
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      if (i < partitioned_results[shard_id].size()) {
        (*values)[original_index] = partitioned_results[shard_id][i];
      }
    }
  }
}

void DistributedGRPCParameterClient::MergeResultsToArray(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<std::vector<float>>>& partitioned_results,
    float* values) const {
  int emb_dim = 0;
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (!partitioned_results[shard_id].empty() &&
        !partitioned_results[shard_id][0].empty()) {
      emb_dim = partitioned_results[shard_id][0].size();
      break;
    }
  }

  if (emb_dim == 0) {
    LOG(WARNING) << "No valid embeddings found";
    return;
  }

  // 合并结果到连续内存
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      if (i < partitioned_results[shard_id].size()) {
        const auto& embedding = partitioned_results[shard_id][i];
        std::copy(embedding.begin(),
                  embedding.end(),
                  values + original_index * emb_dim);
      }
    }
  }
}

// 实现BasePSClient接口
int DistributedGRPCParameterClient::GetParameter(
    const base::ConstArray<uint64_t>& keys, float* values) {
  std::vector<std::vector<float>> result_vectors;
  bool success = GetParameter(keys, &result_vectors);

  if (!success) {
    return -1;
  }

  if (keys.Size() == 0) {
    return 0;
  }
  int emb_dim = 0;
  for (const auto& row : result_vectors) {
    if (!row.empty()) {
      emb_dim = static_cast<int>(row.size());
      break;
    }
  }
  if (emb_dim == 0) {
    LOG(WARNING) << "No valid embeddings found";
    return 0;
  }

  for (size_t i = 0; i < result_vectors.size(); ++i) {
    const auto& row = result_vectors[i];
    if (row.empty()) {
      continue;
    }
    std::copy(row.begin(), row.end(), values + i * emb_dim);
  }
  return 0;
}

int DistributedGRPCParameterClient::AsyncGetParameter(
    const base::ConstArray<uint64_t>& keys, float* values) {
  return GetParameter(keys, values);
}

int DistributedGRPCParameterClient::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  if (keys.Size() != values.size()) {
    LOG(ERROR) << "Keys and values size mismatch: " << keys.Size() << " vs "
               << values.size();
    return -1;
  }

  std::vector<std::vector<uint64_t>> partitioned_keys;
  PartitionKeys(keys, partitioned_keys);

  std::vector<std::vector<std::vector<float>>> partitioned_values(num_shards_);
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      partitioned_values[shard_id].push_back(values[original_index]);
    }
  }

  // 并发put到各个分片
  std::vector<std::future<int>> futures;

  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (partitioned_keys[shard_id].empty()) {
      continue;
    }

    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      return -1;
    }

    int client_index = it->second;
    auto* client     = clients_[client_index].get();

    futures.push_back(std::async(
        std::launch::async, [=, &partitioned_keys, &partitioned_values]() {
          const auto& shard_keys_vec = partitioned_keys[shard_id];
          const auto& shard_vals_vec = partitioned_values[shard_id];
          for (size_t start = 0; start < shard_keys_vec.size();
               start += static_cast<size_t>(max_keys_per_request_)) {
            size_t end =
                std::min(start + static_cast<size_t>(max_keys_per_request_),
                         shard_keys_vec.size());
            base::ConstArray<uint64_t> shard_chunk(
                shard_keys_vec.data() + start, static_cast<int>(end - start));
            std::vector<std::vector<float>> value_chunk(
                shard_vals_vec.begin() + start, shard_vals_vec.begin() + end);
            if (client->PutParameter(shard_chunk, value_chunk) != 1) {
              return 0;
            }
          }
          return 1;
        }));
  }

  // 等待所有请求完成
  for (auto& future : futures) {
    if (future.get() != 1) {
      LOG(ERROR) << "Failed to put parameters to one of the shards";
      return -1;
    }
  }

  return 0;
}

void DistributedGRPCParameterClient::Command(PSCommand command) {
  std::vector<std::future<void>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(std::launch::async, [&client, command]() {
      client->Command(command);
    }));
  }

  for (auto& future : futures) {
    future.wait();
  }
}

bool DistributedGRPCParameterClient::ClearPS() {
  std::vector<std::future<bool>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(std::launch::async, [&client]() {
      return client->ClearPS();
    }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

bool DistributedGRPCParameterClient::LoadFakeData(int64_t n) {
  std::vector<std::future<bool>> futures;
  for (auto& client : clients_) {
    GRPCParameterClient* raw = client.get();
    futures.push_back(std::async(std::launch::async, [raw, n]() {
      return raw->LoadFakeData(n);
    }));
  }
  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }
  return all_success;
}

bool DistributedGRPCParameterClient::DumpFakeData(int64_t n) {
  std::vector<std::future<bool>> futures;
  for (auto& client : clients_) {
    GRPCParameterClient* raw = client.get();
    futures.push_back(std::async(std::launch::async, [raw, n]() {
      return raw->DumpFakeData(n);
    }));
  }
  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }
  return all_success;
}

bool DistributedGRPCParameterClient::LoadCkpt(
    const std::vector<std::string>& model_config_path,
    const std::vector<std::string>& emb_file_path) {
  std::vector<std::future<bool>> futures;

  for (auto& client : clients_) {
    futures.push_back(std::async(
        std::launch::async, [&client, &model_config_path, &emb_file_path]() {
          return client->LoadCkpt(model_config_path, emb_file_path);
        }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

int DistributedGRPCParameterClient::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    LOG(ERROR) << "UpdateParameter grads pointer is null";
    return -1;
  }
  if (keys.Size() != grads->size()) {
    LOG(ERROR) << "UpdateParameter keys/grads size mismatch: " << keys.Size()
               << " vs " << grads->size();
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> partitioned_keys;
  PartitionKeys(keys, partitioned_keys);

  std::vector<std::vector<std::vector<float>>> partitioned_grads(num_shards_);
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    for (size_t i = 0; i < key_index_mapping_[shard_id].size(); ++i) {
      size_t original_index = key_index_mapping_[shard_id][i];
      partitioned_grads[shard_id].push_back((*grads)[original_index]);
    }
  }

  std::vector<std::future<int>> futures;
  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (partitioned_keys[shard_id].empty()) {
      continue;
    }
    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      return -1;
    }
    int client_index = it->second;
    auto* client     = clients_[client_index].get();

    futures.push_back(std::async(
        std::launch::async, [=, &partitioned_keys, &partitioned_grads]() {
          const auto& shard_keys_vec  = partitioned_keys[shard_id];
          const auto& shard_grads_vec = partitioned_grads[shard_id];
          for (size_t start = 0; start < shard_keys_vec.size();
               start += static_cast<size_t>(max_keys_per_request_)) {
            size_t end =
                std::min(start + static_cast<size_t>(max_keys_per_request_),
                         shard_keys_vec.size());
            base::ConstArray<uint64_t> shard_chunk(
                shard_keys_vec.data() + start, static_cast<int>(end - start));
            std::vector<std::vector<float>> grad_chunk(
                shard_grads_vec.begin() + start, shard_grads_vec.begin() + end);
            if (client->UpdateParameter(table_name, shard_chunk, &grad_chunk) !=
                0) {
              return -1;
            }
          }
          return 0;
        }));
  }

  for (auto& future : futures) {
    if (future.get() != 0) {
      LOG(ERROR) << "Failed to update parameters on one of the shards";
      return -1;
    }
  }

  return 0;
}

int DistributedGRPCParameterClient::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  if (grads == nullptr) {
    LOG(ERROR) << "UpdateParameterFlat grads pointer is null";
    return -1;
  }
  if (num_rows < 0 || embedding_dim <= 0) {
    LOG(ERROR) << "UpdateParameterFlat invalid shape: rows=" << num_rows
               << " dim=" << embedding_dim;
    return -1;
  }
  if (keys.Size() != static_cast<size_t>(num_rows)) {
    LOG(ERROR) << "UpdateParameterFlat keys/grads size mismatch: "
               << keys.Size() << " vs " << num_rows;
    return -1;
  }

  std::vector<std::vector<float>> row_grads;
  row_grads.reserve(static_cast<size_t>(num_rows));
  for (int64_t i = 0; i < num_rows; ++i) {
    const float* row = grads + i * embedding_dim;
    row_grads.emplace_back(row, row + embedding_dim);
  }
  return UpdateParameter(table_name, keys, &row_grads);
}

int DistributedGRPCParameterClient::InitEmbeddingTable(
    const std::string& table_name,
    const recstore::EmbeddingTableConfig& config) {
  std::vector<std::future<int>> futures;
  for (auto& client : clients_) {
    futures.push_back(
        std::async(std::launch::async, [&client, &table_name, &config]() {
          return client->InitEmbeddingTable(table_name, config);
        }));
  }

  for (auto& future : futures) {
    if (future.get() != 0) {
      LOG(ERROR) << "InitEmbeddingTable failed on one of the shards";
      return -1;
    }
  }

  return 0;
}

// Prefetch 接口实现
uint64_t DistributedGRPCParameterClient::PrefetchParameter(
    const base::ConstArray<uint64_t>& keys) {
  auto cleanup_state = [this](const DistPrefetchState& state) {
    for (const auto& shard_state : state.shard_states) {
      if (shard_state.client_index < 0 ||
          shard_state.client_index >= static_cast<int>(clients_.size())) {
        continue;
      }
      auto* client = clients_[shard_state.client_index].get();
      for (uint64_t child_prefetch_id : shard_state.child_prefetch_ids) {
        client->WaitForPrefetch(child_prefetch_id);
        std::vector<std::vector<float>> tmp;
        client->GetPrefetchResult(child_prefetch_id, &tmp);
      }
    }
  };

  if (keys.Size() == 0) {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    uint64_t prefetch_id          = next_prefetch_id_++;
    auto state                    = std::make_shared<DistPrefetchState>();
    state->total_keys             = 0;
    prefetch_states_[prefetch_id] = state;
    return prefetch_id;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<size_t>> shard_indices(num_shards_);
  for (size_t i = 0; i < keys.Size(); ++i) {
    const int shard_id = GetShardId(keys[i]);
    shard_keys[shard_id].push_back(keys[i]);
    shard_indices[shard_id].push_back(i);
  }

  auto state        = std::make_shared<DistPrefetchState>();
  state->total_keys = keys.Size();

  for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
    if (shard_keys[shard_id].empty()) {
      continue;
    }

    auto it = shard_to_client_index_.find(shard_id);
    if (it == shard_to_client_index_.end()) {
      LOG(ERROR) << "No client found for shard " << shard_id;
      cleanup_state(*state);
      return 0;
    }

    DistPrefetchShardState shard_state;
    shard_state.shard_id         = shard_id;
    shard_state.client_index     = it->second;
    shard_state.original_indices = std::move(shard_indices[shard_id]);

    const auto& skeys = shard_keys[shard_id];
    for (size_t start = 0; start < skeys.size();
         start += static_cast<size_t>(max_keys_per_request_)) {
      size_t end = std::min(
          start + static_cast<size_t>(max_keys_per_request_), skeys.size());
      base::ConstArray<uint64_t> chunk(
          skeys.data() + start, static_cast<int>(end - start));
      uint64_t child_prefetch_id =
          clients_[shard_state.client_index]->PrefetchParameter(chunk);
      if (child_prefetch_id == 0) {
        LOG(ERROR) << "PrefetchParameter failed for shard " << shard_id;
        cleanup_state(*state);
        return 0;
      }
      shard_state.child_prefetch_ids.push_back(child_prefetch_id);
      shard_state.chunk_sizes.push_back(static_cast<int>(end - start));
    }
    state->shard_states.push_back(std::move(shard_state));
  }

  std::lock_guard<std::mutex> lk(prefetch_mu_);
  uint64_t prefetch_id          = next_prefetch_id_++;
  prefetch_states_[prefetch_id] = std::move(state);
  return prefetch_id;
}

bool DistributedGRPCParameterClient::IsPrefetchDone(uint64_t prefetch_id) {
  std::shared_ptr<DistPrefetchState> state;
  {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    auto it = prefetch_states_.find(prefetch_id);
    if (it == prefetch_states_.end()) {
      LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
      return false;
    }
    state = it->second;
  }

  for (const auto& shard_state : state->shard_states) {
    auto* client = clients_[shard_state.client_index].get();
    for (uint64_t child_prefetch_id : shard_state.child_prefetch_ids) {
      if (!client->IsPrefetchDone(child_prefetch_id)) {
        return false;
      }
    }
  }
  return true;
}

void DistributedGRPCParameterClient::WaitForPrefetch(uint64_t prefetch_id) {
  std::shared_ptr<DistPrefetchState> state;
  {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    auto it = prefetch_states_.find(prefetch_id);
    if (it == prefetch_states_.end()) {
      LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
      return;
    }
    state = it->second;
  }

  for (const auto& shard_state : state->shard_states) {
    auto* client = clients_[shard_state.client_index].get();
    for (uint64_t child_prefetch_id : shard_state.child_prefetch_ids) {
      client->WaitForPrefetch(child_prefetch_id);
    }
  }
}

bool DistributedGRPCParameterClient::GetPrefetchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  if (values == nullptr) {
    LOG(ERROR) << "GetPrefetchResult output pointer is null";
    return false;
  }

  std::shared_ptr<DistPrefetchState> state;
  {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    auto it = prefetch_states_.find(prefetch_id);
    if (it == prefetch_states_.end()) {
      LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
      return false;
    }
    state = it->second;
  }

  // Ensure all child RPCs are completed before consuming payloads.
  WaitForPrefetch(prefetch_id);

  values->clear();
  values->resize(state->total_keys);

  bool ok_all = true;
  for (const auto& shard_state : state->shard_states) {
    auto* client        = clients_[shard_state.client_index].get();
    size_t shard_offset = 0;
    for (size_t i = 0; i < shard_state.child_prefetch_ids.size(); ++i) {
      std::vector<std::vector<float>> chunk_values;
      if (!client->GetPrefetchResult(
              shard_state.child_prefetch_ids[i], &chunk_values)) {
        ok_all = false;
        break;
      }
      const int expected =
          (i < shard_state.chunk_sizes.size()
               ? shard_state.chunk_sizes[i]
               : -1);
      if (expected >= 0 && static_cast<int>(chunk_values.size()) != expected) {
        LOG(ERROR) << "Prefetch chunk size mismatch: got "
                   << chunk_values.size() << ", expected " << expected;
        ok_all = false;
        break;
      }
      for (const auto& row : chunk_values) {
        if (shard_offset >= shard_state.original_indices.size()) {
          LOG(ERROR) << "Prefetch result overflow in shard "
                     << shard_state.shard_id;
          ok_all = false;
          break;
        }
        (*values)[shard_state.original_indices[shard_offset++]] = row;
      }
      if (!ok_all) {
        break;
      }
    }
    if (!ok_all) {
      break;
    }
  }

  {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    prefetch_states_.erase(prefetch_id);
  }
  return ok_all;
}

bool DistributedGRPCParameterClient::GetPrefetchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  if (values == nullptr || num_rows == nullptr) {
    LOG(ERROR) << "GetPrefetchResultFlat output pointer is null";
    return false;
  }
  if (embedding_dim <= 0) {
    LOG(ERROR) << "GetPrefetchResultFlat invalid embedding_dim: "
               << embedding_dim;
    return false;
  }

  std::vector<std::vector<float>> merged_values;
  if (!GetPrefetchResult(prefetch_id, &merged_values)) {
    return false;
  }

  *num_rows = static_cast<int64_t>(merged_values.size());
  values->assign(
      static_cast<size_t>(*num_rows) * static_cast<size_t>(embedding_dim),
      0.0f);
  for (size_t i = 0; i < merged_values.size(); ++i) {
    const auto& row = merged_values[i];
    if (row.empty()) {
      continue;
    }
    const int64_t copy_d =
        std::min<int64_t>(embedding_dim, static_cast<int64_t>(row.size()));
    std::memcpy(values->data() + i * static_cast<size_t>(embedding_dim),
                row.data(),
                static_cast<size_t>(copy_d) * sizeof(float));
  }
  return true;
}

} // namespace recstore
