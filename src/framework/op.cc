#include "framework/op.h"
#include "ps/grpc/grpc_ps_client.h"
#include "base/factory.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <memory>
#include <numeric>
#include <thread>
#include <cstdlib>
#include <string>
#include <fstream>

#include "base/tensor.h"
#include "op.h"
#include <glog/logging.h>
#ifdef ENABLE_PERF_REPORT
#  include "base/report/report_client.h"
#endif

namespace recstore {

json load_config_from_file(const std::string& config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + config_path);
  }

  std::string content(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  try {
    return json::parse(content);
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "Failed to parse config file: " + std::string(e.what()));
  }
}

void validate_keys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " +
                                base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional, but has " +
                                std::to_string(keys.dim()) + " dimensions.");
  }
}

void validate_embeddings(const base::RecTensor& embeddings,
                         const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " +
        base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " +
        std::to_string(embeddings.dim()) + " dimensions.");
  }
  // No fixed embedding dimension check for mock.
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const base::RecTensor& init_values) {
  EmbWrite(keys, init_values);
}

void KVClientOp::EmbDelete(const base::RecTensor& keys) {
  throw std::runtime_error("Not impl");
}
bool KVClientOp::EmbExists(const base::RecTensor& keys) {
  throw std::runtime_error("Not impl");
}

void KVClientOp::WaitForWrite(uint64_t write_id) {
  throw std::runtime_error("Not impl");
}
void KVClientOp::SaveToFile(const std::string& path) {
  throw std::runtime_error("Not impl");
}
void KVClientOp::LoadFromFile(const std::string& path) {
  throw std::runtime_error("Not impl");
}

uint64_t KVClientOp::EmbWriteAsync(const base::RecTensor& keys,
                                   const base::RecTensor& values) {
  throw std::runtime_error("Not impl");
}

std::shared_ptr<CommonOp> GetKVClientOp() {
  static std::shared_ptr<CommonOp> instance;
  static std::once_flag once_flag;
  std::call_once(once_flag, []() {
    instance = std::make_shared<KVClientOp>();
  });
  return instance;
}

} // namespace recstore

#ifndef USE_FAKE_KVCLIENT

#  include "ps/grpc/grpc_ps_client.h"
namespace recstore {

BasePSClient* create_ps_client_from_config(const json& config) {
  std::string ps_type = "GRPC";
  try {
    if (config.contains("cache_ps") && config["cache_ps"].contains("ps_type")) {
      ps_type = config["cache_ps"]["ps_type"].get<std::string>();
    }
  } catch (...) {
  }

  if (false && config.contains("distributed_client")) {
    std::string type_key =
        (ps_type == "BRPC") ? "distributed_brpc" : "distributed_grpc";
    BasePSClient* client =
        base::Factory<BasePSClient, json>::NewInstance(type_key, config);
    if (client)
      return client;
  }

  json client_config;
  if (config.contains("client")) {
    client_config = config["client"];
  } else {
    client_config = json{{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}};
  }

  std::string type_key = (ps_type == "BRPC") ? "brpc" : "grpc";
  BasePSClient* client =
      base::Factory<BasePSClient, json>::NewInstance(type_key, client_config);
  if (client == nullptr) {
    return new GRPCParameterClient(client_config);
  }
  return client;
}

json GetGlobalConfig() {
  try {
    auto current_path = std::filesystem::current_path();
    LOG(INFO) << "Current working directory: " << current_path.string();

    std::filesystem::path config_path;
    bool config_found = false;

    for (auto p = current_path; p.has_parent_path(); p = p.parent_path()) {
      if (std::filesystem::exists(p / "recstore_config.json")) {
        config_path  = p / "recstore_config.json";
        config_found = true;
        LOG(INFO) << "Found config file at: " << config_path.string();
        break;
      }
    }

    if (!config_found) {
      throw std::runtime_error(
          "Could not find 'recstore_config.json' in current or any parent "
          "directory starting from: " +
          current_path.string());
    }

    std::ifstream test_file(config_path);
    if (!test_file.good()) {
      throw std::runtime_error(
          "Config file not found: " + config_path.string() +
          ". Please ensure recstore_config.json exists "
          "in the project root directory.");
    }
    test_file.close();

    return load_config_from_file(config_path);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to load config: " << std::string(e.what());
    return json::object();
  }
}

void ConfigureLogging() {
  static std::once_flag log_init_flag;
  std::call_once(log_init_flag, []() {
    std::cerr << "[Debug] ConfigureLogging called. Setting flags." << std::endl;
    FLAGS_log_dir         = "/tmp";
    FLAGS_alsologtostderr = false;
    FLAGS_logtostderr     = false;
    FLAGS_stderrthreshold = google::ERROR;

    google::SetLogDestination(google::INFO, "/tmp/recstore_op_layer_INFO_");
    google::SetLogDestination(
        google::WARNING, "/tmp/recstore_op_layer_WARNING_");
    google::SetLogDestination(google::ERROR, "/tmp/recstore_op_layer_ERROR_");

    google::InitGoogleLogging("recstore_op_layer");
  });
}

KVClientOp::KVClientOp() {
  ConfigureLogging();

  if (!ps_client_) {
    try {
      json config = GetGlobalConfig();
      ps_client_  = create_ps_client_from_config(config);

      LOG(INFO) << "PS client initialized successfully.";
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to initialize PS client: " << std::string(e.what());
      throw;
    }
  }
}

BasePSClient* KVClientOp::ps_client_ = nullptr;

void KVClientOp::SetPSConfig(const std::string& host, int port) {
  if (ps_client_) {
    delete ps_client_;
    ps_client_ = nullptr;
  }

  json file_config = GetGlobalConfig();
  int final_port   = port;
  if (final_port <= 0) {
    if (file_config.contains("client") &&
        file_config["client"].contains("port")) {
      final_port = file_config["client"]["port"].get<int>();
    } else if (file_config.contains("cache_ps") &&
               file_config["cache_ps"].contains("servers") &&
               file_config["cache_ps"]["servers"].is_array() &&
               !file_config["cache_ps"]["servers"].empty()) {
      final_port = file_config["cache_ps"]["servers"][0]["port"].get<int>();
    } else {
      final_port = 15000;
    }
  }

  std::string final_host = host;
  if (final_host.empty()) {
    final_host = "127.0.0.1";
  }

  json config = file_config;
  if (!config.contains("client")) {
    config["client"] = json::object();
  }
  config["client"]["host"]  = final_host;
  config["client"]["port"]  = final_port;
  config["client"]["shard"] = 0;

  ps_client_ = create_ps_client_from_config(config);
  LOG(INFO) << "Re-initialized PS client with host=" << final_host
            << " port=" << final_port;
}

void KVClientOp::EmbRead(const RecTensor& keys, RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id =
      "op::EmbRead|" + std::to_string(static_cast<uint64_t>(start_us));
  std::string unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
#  endif

  LOG(INFO) << "EmbRead: keys.shape=" << keys.shape(0) << ", values.shape=["
            << values.shape(0) << ", " << values.shape(1) << "]";
  LOG(INFO) << "EmbRead: keys.data=" << keys.data_as<uint64_t>()
            << ", values.data=" << values.data_as<float>();
  if (keys.shape(0) > 0) {
    std::ostringstream oss;
    oss << "EmbRead: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.shape(0)); ++i)
      oss << keys.data_as<uint64_t>()[i] << ", ";
    LOG(INFO) << oss.str();
  }
  if (values.shape(0) > 0) {
    std::ostringstream oss;
    oss << "EmbRead: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.shape(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.shape(1)); ++j) {
        oss << values.data_as<float>()[i * values.shape(1) + j] << ", ";
      }
      oss << "] ";
    }
    LOG(INFO) << oss.str();
  }
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  float* values_data = values.data_as<float>();

  const int64_t D    = values.shape(1);
  const size_t total = static_cast<size_t>(L) * static_cast<size_t>(D);
  std::fill_n(values_data, total, 0.0f);

  bool success = ps_client_->GetParameter(keys_array, values_data);
  if (!success) {
    throw std::runtime_error("Failed to read embeddings from PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  std::string op_latency_key =
      "EmbRead|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "request_size",
         static_cast<double>(keys.shape(0)));

  FlameGraphData op_data = {
      "op::EmbRead",
      start_us,
      0, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), op_data);
#  endif
}

void KVClientOp::EmbUpdate(const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  EmbUpdate("default", keys, grads);
}

void KVClientOp::EmbUpdate(const std::string& table_name,
                           const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time         = std::chrono::high_resolution_clock::now();
  const uint64_t trace_id = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count());
  struct TraceGuard {
    explicit TraceGuard(uint64_t new_trace_id)
        : previous_trace_id_(recstore::g_trace_id) {
      recstore::g_trace_id = new_trace_id;
    }
    ~TraceGuard() { recstore::g_trace_id = previous_trace_id_; }
    uint64_t previous_trace_id_;
  } trace_guard(trace_id);
#  endif

  int64_t validate_done_us = 0;
  validate_keys(keys);
  validate_embeddings(grads, "Grads");

  const int64_t L = keys.shape(0);
  if (grads.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but grads has length " + std::to_string(grads.shape(0)));
  }

  const int64_t D = grads.shape(1);
  if (D <= 0) {
    throw std::invalid_argument(
        "Invalid grad dimension D: " + std::to_string(D));
  }

#  ifdef ENABLE_PERF_REPORT
  auto validate_done_time = std::chrono::high_resolution_clock::now();
  validate_done_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          validate_done_time - start_time)
          .count();
#  endif

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);

  const float* grads_data = grads.data_as<float>();
  int ret =
      ps_client_->UpdateParameterFlat(table_name, keys_array, grads_data, L, D);
  if (ret != 0) {
    throw std::runtime_error("Failed to update embeddings via PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbUpdate|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));

  std::string update_stage_id =
      "op_client::EmbUpdate|" + std::to_string(trace_id);
  report("embupdate_stages",
         update_stage_id.c_str(),
         "op_total_us",
         static_cast<double>(duration));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "op_validate_us",
         static_cast<double>(validate_done_us));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "request_size",
         static_cast<double>(L));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "embedding_dim",
         static_cast<double>(D));
#  endif
}

bool KVClientOp::InitEmbeddingTable(const std::string& table_name,
                                    const EmbeddingTableConfig& config) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#  endif
  int ret = ps_client_->InitEmbeddingTable(table_name, config);
#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "InitEmbeddingTable|" + std::to_string(static_cast<uint64_t>(start_us));
  // report(table_name, key, metric_name, value)
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));
#  endif
  return ret == 0;
}

void KVClientOp::EmbWrite(const RecTensor& keys, const RecTensor& values) {
  if (ps_client_ == nullptr) {
    throw std::runtime_error("PS client is not initialized. Please call "
                             "KVClientOp::SetPSClient() first.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#  endif

  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L          = keys.shape(0);
  const auto& values_shape = values.shape();
  if (values.shape(0) != L) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(L) +
        " but values has length " + std::to_string(values.shape(0)));
  }
  const int64_t D = values.shape(1);

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  const float* values_data = values.data_as<float>();

  const int64_t total_values = L * D;
  if (values_shape[0] * values_shape[1] != total_values) {
    throw std::invalid_argument(
        "Values total elements mismatch: expected " +
        std::to_string(total_values) + ", but got " +
        std::to_string(values_shape[0] * values_shape[1]));
  }

  if (D <= 0) {
    throw std::invalid_argument(
        "Invalid embedding dimension D: " + std::to_string(D));
  }

  std::vector<std::vector<float>> values_vector;
  values_vector.reserve(L);
  for (int64_t i = 0; i < L; ++i) {
    std::vector<float> row(D);
    std::memcpy(row.data(), values_data + i * D, D * sizeof(float));
    asm volatile("" ::: "memory");
    _mm_mfence();
    values_vector.push_back(std::move(row));
  }

  LOG(INFO) << "=== Keys Array Info ===";
  LOG(INFO) << "Keys size: " << L;
  if (L > 0) {
    std::ostringstream keys_stream;
    keys_stream << "First 3 keys: ";
    for (int64_t i = 0; i < std::min(L, static_cast<int64_t>(3)); ++i) {
      keys_stream << keys_array[i] << " ";
    }
    LOG(INFO) << keys_stream.str();
  }

  LOG(INFO) << "=== Values Vector Info ===";
  LOG(INFO) << "Values total elements: " << total_values;
  LOG(INFO) << "Embedding dimension D: " << D;
  if (L > 0 && D > 0) {
    std::ostringstream values_stream;
    values_stream << "First 3 embeddings (each first 3 items): ";
    for (int64_t i = 0; i < std::min(L, static_cast<int64_t>(3)); ++i) {
      values_stream << "[";
      for (int64_t j = 0; j < std::min(D, static_cast<int64_t>(3)); ++j) {
        values_stream << values_vector[i][j] << " ";
      }
      values_stream << "] ";
    }
    LOG(INFO) << values_stream.str();
  }

  bool success = ps_client_->PutParameter(keys_array, values_vector);
  if (!success) {
    throw std::runtime_error("Failed to write embeddings to PS client.");
  }

#  ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbWrite|" + std::to_string(static_cast<uint64_t>(start_us));
  report("op_latency",
         op_latency_key.c_str(),
         "recstore_us",
         static_cast<double>(duration));
#  endif
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const InitStrategy& strategy) {
  validate_keys(keys);
}

uint64_t
KVClientOp::EmbPrefetch(const base::RecTensor& keys, const RecTensor& values) {
  const uint64_t* keys_data = keys.data_as<uint64_t>();
  int64_t L                 = keys.shape(0);
  base::ConstArray<uint64_t> keys_array(keys_data, L);
  return ps_client_->PrefetchParameter(keys_array);
}

bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
  return ps_client_->IsPrefetchDone(prefetch_id);
}

void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
  ps_client_->WaitForPrefetch(prefetch_id);
}

void KVClientOp::GetPretchResult(uint64_t prefetch_id,
                                 std::vector<std::vector<float>>* values) {
  ps_client_->GetPrefetchResult(prefetch_id, values);
}

void KVClientOp::GetPretchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  ps_client_->GetPrefetchResultFlat(
      prefetch_id, values, num_rows, embedding_dim);
}

bool KVClientOp::IsWriteDone(uint64_t write_id) {
  // return ps_client_->IsWriteDone(write_id);
  throw std::runtime_error("Not impl");
}

namespace testing {} // namespace testing

} // namespace recstore

#else

#  include "op_mock.cc"

#endif // USE_FAKE_KVCLIENT
