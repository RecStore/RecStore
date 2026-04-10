#include "brpc_ps_client.h"

#include <brpc/channel.h>
#include <fmt/core.h>

#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/flatc.h"
#include "base/log.h"
#include "base/timer.h"
#include "ps/base/parameters.h"
#include "ps_brpc.pb.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include "base/report/report_client.h"
#endif

using recstoreps_brpc::CommandRequest;
using recstoreps_brpc::CommandResponse;
using recstoreps_brpc::GetParameterRequest;
using recstoreps_brpc::GetParameterResponse;
using recstoreps_brpc::InitEmbeddingTableRequest;
using recstoreps_brpc::InitEmbeddingTableResponse;
using recstoreps_brpc::PSCommand;
using recstoreps_brpc::PutParameterRequest;
using recstoreps_brpc::PutParameterResponse;
using recstoreps_brpc::UpdateParameterRequest;
using recstoreps_brpc::UpdateParameterResponse;

namespace {

const ParameterCompressReader* ExtractGetResponseReader(
    const brpc::Controller& cntl,
    const GetParameterResponse& response,
    std::string* payload_storage,
    int* payload_size) {
  if (!cntl.response_attachment().empty()) {
    payload_storage->clear();
    cntl.response_attachment().copy_to(payload_storage);
    *payload_size = payload_storage->size();
    return reinterpret_cast<const ParameterCompressReader*>(
        payload_storage->data());
  }

  *payload_size = response.parameter_value().size();
  return reinterpret_cast<const ParameterCompressReader*>(
      response.parameter_value().data());
}

} // namespace

namespace {

int BuildUpdateBlocksFromFlat(
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    ParameterCompressor* compressor) {
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

  for (int64_t i = 0; i < num_rows; ++i) {
    ParameterPack pack;
    pack.key      = keys[static_cast<size_t>(i)];
    pack.dim      = embedding_dim;
    pack.emb_data = grads + i * embedding_dim;
    compressor->AddItem(pack, nullptr);
  }
  return 0;
}

} // namespace

DEFINE_int32(brpc_timeout_ms, 5000, "brpc request timeout in milliseconds");
DEFINE_int32(brpc_max_retry, 3, "brpc max retry times");
DEFINE_bool(parameter_client_random_init_brpc, false, "");

// 新的构造函数，接收 json 配置参数
BRPCParameterClient::BRPCParameterClient(json config)
    : recstore::BasePSClient(config) {
  host_       = config.value("host", "localhost");
  port_       = config.value("port", 15000);
  shard_      = config.value("shard", 0);
  timeout_ms_ = config.value("timeout_ms", FLAGS_brpc_timeout_ms);
  max_retry_  = config.value("max_retry", FLAGS_brpc_max_retry);

  Initialize();

  // 初始化 bRPC channel
  channel_ = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = timeout_ms_;
  options.max_retry  = max_retry_;

  std::string server_addr = fmt::format("{}:{}", host_, port_);
  if (channel_->Init(server_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to initialize bRPC channel to " << server_addr;
  } else {
    LOG(INFO) << "Initialized bRPC PS Client Shard " << shard_ << " at "
              << server_addr;
  }
}

// 保留原有的构造函数以保持向后兼容
BRPCParameterClient::BRPCParameterClient(
    const std::string& host, int port, int shard)
    : recstore::BasePSClient(
          json{{"host", host}, {"port", port}, {"shard", shard}}),
      host_(host),
      port_(port),
      shard_(shard),
      timeout_ms_(FLAGS_brpc_timeout_ms),
      max_retry_(FLAGS_brpc_max_retry) {
  Initialize();

  channel_ = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = timeout_ms_;
  options.max_retry  = max_retry_;

  std::string server_addr = fmt::format("{}:{}", host, port);
  if (channel_->Init(server_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "Failed to initialize bRPC channel to " << server_addr;
  } else {
    LOG(INFO) << "Initialized bRPC PS Client Shard " << shard_ << " at "
              << server_addr;
  }
}

bool BRPCParameterClient::Initialize() { return true; }

int BRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      float* values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (FLAGS_parameter_client_random_init_brpc) {
    CHECK(0) << "todo implement";
    return true;
  }

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH_BRPC - 1) / MAX_PARAMETER_BATCH_BRPC;
  std::vector<GetParameterRequest> requests(request_num);
  std::vector<GetParameterResponse> responses(request_num);
  std::vector<brpc::Controller> controllers(request_num);
  std::vector<int> key_sizes;

  // 创建 stub
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());

#ifdef ENABLE_PERF_REPORT
  auto wait_start_time = std::chrono::high_resolution_clock::now();
#endif

  // 发送异步请求
  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH_BRPC, ++index) {
    int key_size =
        std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH_BRPC);
    key_sizes.push_back(key_size);

    controllers[index].request_attachment().append(
        reinterpret_cast<const char*>(&keys[start]),
        sizeof(uint64_t) * key_size);

    google::protobuf::Closure* done = brpc::NewCallback([]() { /* 空回调 */ });
    stub.GetParameter(
        &controllers[index], &requests[index], &responses[index], done);
  }

  // 等待所有请求完成
  for (int i = 0; i < request_num; ++i) {
    brpc::Join(controllers[i].call_id());
    if (controllers[i].Failed()) {
      LOG(ERROR) << "bRPC GetParameter failed: " << controllers[i].ErrorText();
      return false;
    }
  }

#ifdef ENABLE_PERF_REPORT
  auto wait_end_time = std::chrono::high_resolution_clock::now();
  auto wait_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          wait_end_time - wait_start_time)
          .count();
  double wait_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          wait_start_time.time_since_epoch())
          .count();
  std::string wait_label =
      "brpc_client::RPC_Call_And_Wait_Shard" + std::to_string(shard_);
  FlameGraphData wait_fg = {
      wait_label,
      wait_start_us,
      2, // level
      static_cast<double>(wait_duration),
      static_cast<double>(wait_duration)};
  std::string unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(wait_start_us));
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), wait_fg);

  double start_us_for_rpc =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_rpc =
      "brpc_client::GetParameter|" +
      std::to_string(static_cast<uint64_t>(start_us_for_rpc));
  report("embread_stages",
         report_id_for_rpc.c_str(),
         "rpc_duration_us",
         static_cast<double>(wait_duration));

  auto deserialize_start_time = std::chrono::high_resolution_clock::now();
#endif

  // 解析结果
  size_t get_embedding_acc = 0;
  int old_dimension        = -1;
  std::string payload_storage;

  for (int i = 0; i < responses.size(); ++i) {
    auto& response   = responses[i];
    int key_size     = key_sizes[i];
    int payload_size = 0;
    auto parameters  = ExtractGetResponseReader(
        controllers[i], response, &payload_storage, &payload_size);

    if (parameters == nullptr || !parameters->Valid(payload_size)) {
      LOG(ERROR) << "GetParameter invalid payload: " << payload_size;
      return false;
    }

    if (parameters->size != key_size) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size(); ++index) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        if (old_dimension == -1)
          old_dimension = item->dim;
        CHECK_EQ(item->dim, old_dimension);
        std::copy_n(
            item->embedding, item->dim, values + item->dim * get_embedding_acc);
      } else {
        FB_LOG_EVERY_MS(ERROR, 2000)
            << "error; not find key " << keys[get_embedding_acc] << " in ps";
      }
      get_embedding_acc++;
    }
  }

#ifdef ENABLE_PERF_REPORT
  auto deserialize_end_time = std::chrono::high_resolution_clock::now();
  auto deserialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          deserialize_end_time - deserialize_start_time)
          .count();
  double deserialize_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          deserialize_start_time.time_since_epoch())
          .count();
  std::string des_label =
      "brpc_client::Deserialize_Shard" + std::to_string(shard_);
  FlameGraphData des_fg = {
      des_label,
      deserialize_start_us,
      2, // level
      static_cast<double>(deserialize_duration),
      static_cast<double>(deserialize_duration)};
  std::string des_unique_id =
      "embread_debug|" +
      std::to_string(static_cast<uint64_t>(deserialize_start_us));
  report_flame_graph("emb_read_flame_map", des_unique_id.c_str(), des_fg);

  double start_us_for_des =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_des =
      "brpc_client::GetParameter|" +
      std::to_string(static_cast<uint64_t>(start_us_for_des));
  report("embread_stages",
         report_id_for_des.c_str(),
         "deserialize_duration_us",
         static_cast<double>(deserialize_duration));
#endif

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_client_latency",
         "GetParameter",
         "latency_us",
         static_cast<double>(duration));

  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  FlameGraphData fg_data = {
      "brpc_client::GetParameter",
      start_us,
      1, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};

  std::string report_id = "brpc_client::GetParameter|" +
                          std::to_string(static_cast<uint64_t>(start_us));

  report("embread_stages",
         report_id.c_str(),
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "request_size",
         static_cast<double>(keys.Size()));

  std::string final_unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
  report_flame_graph("emb_read_flame_map", final_unique_id.c_str(), fg_data);
#endif

  return true;
}

int BRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      std::vector<std::vector<float>>* values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (FLAGS_parameter_client_random_init_brpc) {
    values->clear();
    values->reserve(keys.Size());
    for (size_t i = 0; i < keys.Size(); i++)
      values->emplace_back(std::vector<float>(128, 0.1));
    return true;
  }

  values->clear();
  values->reserve(keys.Size());

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH_BRPC - 1) / MAX_PARAMETER_BATCH_BRPC;
  std::vector<GetParameterRequest> requests(request_num);
  std::vector<GetParameterResponse> responses(request_num);
  std::vector<brpc::Controller> controllers(request_num);
  std::vector<int> key_sizes;

  recstoreps_brpc::ParameterService_Stub stub(channel_.get());

#ifdef ENABLE_PERF_REPORT
  auto wait_start_time = std::chrono::high_resolution_clock::now();
#endif

  // 发送异步请求
  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH_BRPC, ++index) {
    int key_size =
        std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH_BRPC);
    key_sizes.push_back(key_size);

    controllers[index].request_attachment().append(
        reinterpret_cast<const char*>(&keys[start]),
        sizeof(uint64_t) * key_size);

    google::protobuf::Closure* done = brpc::NewCallback([]() { /* 空回调 */ });
    stub.GetParameter(
        &controllers[index], &requests[index], &responses[index], done);
  }

  // 等待所有请求完成
  for (int i = 0; i < request_num; ++i) {
    brpc::Join(controllers[i].call_id());
    if (controllers[i].Failed()) {
      LOG(ERROR) << "bRPC GetParameter failed: " << controllers[i].ErrorText();
      return false;
    }
  }

#ifdef ENABLE_PERF_REPORT
  auto wait_end_time = std::chrono::high_resolution_clock::now();
  auto wait_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          wait_end_time - wait_start_time)
          .count();
  double wait_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          wait_start_time.time_since_epoch())
          .count();
  std::string wait_label =
      "brpc_client::RPC_Call_And_Wait_Shard" + std::to_string(shard_);
  FlameGraphData wait_fg = {
      wait_label,
      wait_start_us,
      2, // level
      static_cast<double>(wait_duration),
      static_cast<double>(wait_duration)};
  std::string unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(wait_start_us));
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), wait_fg);

  double start_us_for_rpc =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_rpc =
      "brpc_client::GetParameter_Vec|" +
      std::to_string(static_cast<uint64_t>(start_us_for_rpc));

  report("embread_stages",
         report_id_for_rpc.c_str(),
         "rpc_duration_us",
         static_cast<double>(wait_duration));

  auto deserialize_start_time = std::chrono::high_resolution_clock::now();
#endif

  // 解析结果
  std::string payload_storage;
  for (int i = 0; i < responses.size(); ++i) {
    auto& response   = responses[i];
    int key_size     = key_sizes[i];
    int payload_size = 0;
    auto parameters  = ExtractGetResponseReader(
        controllers[i], response, &payload_storage, &payload_size);

    if (parameters == nullptr || !parameters->Valid(payload_size)) {
      LOG(ERROR) << "GetParameter(vector) invalid payload: " << payload_size;
      return false;
    }

    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size(); ++index) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        values->emplace_back(
            std::vector<float>(item->embedding, item->embedding + item->dim));
      } else {
        values->emplace_back(std::vector<float>(0));
      }
    }
  }

#ifdef ENABLE_PERF_REPORT
  auto deserialize_end_time = std::chrono::high_resolution_clock::now();
  auto deserialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          deserialize_end_time - deserialize_start_time)
          .count();
  double deserialize_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          deserialize_start_time.time_since_epoch())
          .count();
  std::string des_label =
      "brpc_client::Deserialize_Shard" + std::to_string(shard_);
  FlameGraphData des_fg = {
      des_label,
      deserialize_start_us,
      2, // level
      static_cast<double>(deserialize_duration),
      static_cast<double>(deserialize_duration)};
  std::string des_unique_id =
      "embread_debug|" +
      std::to_string(static_cast<uint64_t>(deserialize_start_us));
  report_flame_graph("emb_read_flame_map", des_unique_id.c_str(), des_fg);

  double start_us_for_des =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_des =
      "brpc_client::GetParameter_Vec|" +
      std::to_string(static_cast<uint64_t>(start_us_for_des));
  report("embread_stages",
         report_id_for_des.c_str(),
         "deserialize_duration_us",
         static_cast<double>(deserialize_duration));
#endif

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_client_latency",
         "GetParameter",
         "latency_us",
         static_cast<double>(duration));

  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  FlameGraphData fg_data = {
      "brpc_client::GetParameter_Vec",
      start_us,
      1, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};

  std::string report_id = "brpc_client::GetParameter_Vec|" +
                          std::to_string(static_cast<uint64_t>(start_us));

  report("embread_stages",
         report_id.c_str(),
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "request_size",
         static_cast<double>(keys.Size()));

  std::string final_unique_id =
      "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
  report_flame_graph("emb_read_flame_map", final_unique_id.c_str(), fg_data);
#endif

  return true;
}

static void OnPrefetchDone(BrpcPrefetchBatch* batch) {
  batch->completed_count_++;
}

static void OnPrewriteDone(BrpcPrewriteBatch* batch) {
  batch->completed_count_++;
}

uint64_t
BRPCParameterClient::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
  uint64_t prefetch_id = next_prefetch_id_++;
  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH_BRPC - 1) / MAX_PARAMETER_BATCH_BRPC;

  // 在 map 中构造对象，以保证地址稳定
  auto it = prefetch_batches_.emplace(prefetch_id, request_num).first;
  struct BrpcPrefetchBatch* pb = &it->second;

  recstoreps_brpc::ParameterService_Stub stub(channel_.get());

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH_BRPC, ++index) {
    int key_size =
        std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH_BRPC);
    pb->key_sizes_[index] = key_size;

    GetParameterRequest request;

    pb->controllers_[index] = std::make_unique<brpc::Controller>();
    pb->controllers_[index]->request_attachment().append(
        reinterpret_cast<const char*>(&keys[start]),
        sizeof(uint64_t) * key_size);

    google::protobuf::Closure* done = brpc::NewCallback(OnPrefetchDone, pb);
    stub.GetParameter(
        pb->controllers_[index].get(), &request, &pb->responses_[index], done);
  }

  return prefetch_id;
}

bool BRPCParameterClient::IsPrefetchDone(uint64_t prefetch_id) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }

  auto& pb = it->second;

  return pb.completed_count_ == pb.batch_size_;
}

void BRPCParameterClient::WaitForPrefetch(uint64_t prefetch_id) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return;
  }
  auto& pb = it->second;
  for (int i = 0; i < pb.batch_size_; ++i) {
    if (pb.controllers_[i]) {
      brpc::Join(pb.controllers_[i]->call_id());
    }
  }
  pb.completed_count_ = pb.batch_size_;
}

bool BRPCParameterClient::GetPrefetchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }

  auto& pb        = it->second;
  int request_num = pb.batch_size_;

  values->clear();
  int keys_size = 0;
  for (const auto& size : pb.key_sizes_) {
    keys_size += size;
  }
  values->reserve(keys_size);

  for (int i = 0; i < request_num; ++i) {
    if (pb.controllers_[i]->Failed()) {
      LOG(ERROR) << "Prefetch request failed: "
                 << pb.controllers_[i]->ErrorText();
      return false;
    }

    auto& response = pb.responses_[i];
    int key_size   = pb.key_sizes_[i];
    std::string payload_storage;
    int payload_size = 0;
    auto parameters  = ExtractGetResponseReader(
        *pb.controllers_[i], response, &payload_storage, &payload_size);

    if (parameters == nullptr || !parameters->Valid(payload_size)) {
      LOG(ERROR) << "Prefetch invalid payload: " << payload_size;
      return false;
    }

    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size(); ++index) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        values->emplace_back(
            std::vector<float>(item->embedding, item->embedding + item->dim));
      } else {
        values->emplace_back(std::vector<float>(0));
      }
    }
  }

  // 清理
  prefetch_batches_.erase(it);

  return true;
}

bool BRPCParameterClient::GetPrefetchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }
  if (values == nullptr || num_rows == nullptr) {
    LOG(ERROR) << "GetPrefetchResultFlat output pointer is null";
    return false;
  }

  auto& pb        = it->second;
  int request_num = pb.batch_size_;
  int total_keys  = 0;
  for (const auto& size : pb.key_sizes_) {
    total_keys += size;
  }

  *num_rows = static_cast<int64_t>(total_keys);
  values->assign(
      static_cast<size_t>(*num_rows) * static_cast<size_t>(embedding_dim),
      0.0f);

  size_t row_offset = 0;
  for (int i = 0; i < request_num; ++i) {
    if (pb.controllers_[i]->Failed()) {
      LOG(ERROR) << "Prefetch request failed: "
                 << pb.controllers_[i]->ErrorText();
      return false;
    }

    auto& response = pb.responses_[i];
    int key_size   = pb.key_sizes_[i];
    std::string payload_storage;
    int payload_size = 0;
    auto parameters  = ExtractGetResponseReader(
        *pb.controllers_[i], response, &payload_storage, &payload_size);

    if (parameters == nullptr || !parameters->Valid(payload_size)) {
      LOG(ERROR) << "Prefetch invalid payload: " << payload_size;
      return false;
    }

    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size();
         ++index, ++row_offset) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        const int64_t copy_d =
            std::min<int64_t>(embedding_dim, static_cast<int64_t>(item->dim));
        std::memcpy(values->data() + row_offset * embedding_dim,
                    item->embedding,
                    static_cast<size_t>(copy_d) * sizeof(float));
      }
    }
  }

  prefetch_batches_.erase(it);
  return true;
}

bool BRPCParameterClient::ClearPS() {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::CLEAR_PS);

  brpc::Controller cntl;
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.Command(&cntl, &request, &response, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "bRPC Command failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool BRPCParameterClient::LoadFakeData(int64_t data) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::LOAD_FAKE_DATA);
  request.add_arg1(&data, sizeof(int64_t));

  brpc::Controller cntl;
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.Command(&cntl, &request, &response, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "bRPC LoadFakeData failed: " << cntl.ErrorText();
    return false;
  }
  if (response.reply().size() != static_cast<size_t>(data)) {
    LOG(ERROR) << "bRPC LoadFakeData reply size mismatch: expected " << data
               << ", got " << response.reply().size();
    return false;
  }
  return true;
}

bool BRPCParameterClient::DumpFakeData(int64_t n) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::DUMP_FAKE_DATA);
  request.add_arg1(&n, sizeof(int64_t));

  brpc::Controller cntl;
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.Command(&cntl, &request, &response, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "bRPC DumpFakeData failed: " << cntl.ErrorText();
    return false;
  }
  if (response.reply() != "ok") {
    LOG(ERROR) << "bRPC DumpFakeData unexpected reply: " << response.reply();
    return false;
  }
  return true;
}

bool BRPCParameterClient::LoadCkpt(
    const std::vector<std::string>& model_config_path,
    const std::vector<std::string>& emb_file_path) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::RELOAD_PS);

  for (auto& each : model_config_path) {
    request.add_arg1(each);
  }
  for (auto& each : emb_file_path) {
    request.add_arg2(each);
  }

  brpc::Controller cntl;
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.Command(&cntl, &request, &response, nullptr);

  if (cntl.Failed()) {
    LOG(ERROR) << "bRPC LoadCkpt failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool BRPCParameterClient::PutParameter(
    const std::vector<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  recstoreps_brpc::ParameterService_Stub stub(channel_.get());

  for (int start = 0, index = 0; start < keys.size();
       start += MAX_PARAMETER_BATCH_BRPC, ++index) {
    int key_size =
        std::min((int)(keys.size() - start), MAX_PARAMETER_BATCH_BRPC);

    PutParameterRequest request;
    PutParameterResponse response;
    ParameterCompressor compressor;

    for (int i = start; i < start + key_size; i++) {
      auto each_key   = keys[i];
      auto& embedding = values[i];
      ParameterPack parameter_pack;
      parameter_pack.key      = each_key;
      parameter_pack.dim      = embedding.size();
      parameter_pack.emb_data = embedding.data();
      compressor.AddItem(parameter_pack, nullptr);
    }

    brpc::Controller cntl;
    compressor.AppendToIOBuf(&cntl.request_attachment());
    stub.PutParameter(&cntl, &request, &response, nullptr);

    if (cntl.Failed()) {
      LOG(ERROR) << "bRPC PutParameter failed: " << cntl.ErrorText();
      return false;
    }
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_client_latency",
         "PutParameter",
         "latency_us",
         static_cast<double>(duration));
#endif

  return true;
}

int BRPCParameterClient::AsyncGetParameter(
    const base::ConstArray<uint64_t>& keys, float* values) {
  return GetParameter(keys, values);
}

int BRPCParameterClient::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  bool success = PutParameter(key_vec, values);
  return success ? 1 : 0;
}

void BRPCParameterClient::Command(recstore::PSCommand command) {
  switch (command) {
  case recstore::PSCommand::CLEAR_PS:
    ClearPS();
    break;
  case recstore::PSCommand::RELOAD_PS:
    LOG(WARNING) << "RELOAD_PS command requires additional parameters";
    break;
  case recstore::PSCommand::LOAD_FAKE_DATA: {
    int64_t fake_data = 1000;
    LoadFakeData(fake_data);
  } break;
  case recstore::PSCommand::DUMP_FAKE_DATA: {
    DumpFakeData(4096);
  } break;
  default:
    LOG(ERROR) << "Unknown PS command: " << static_cast<int>(command);
    break;
  }
}

int BRPCParameterClient::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
#ifdef ENABLE_PERF_REPORT
  auto start_time         = std::chrono::high_resolution_clock::now();
  const uint64_t trace_id = recstore::g_trace_id;
#endif
  if (grads == nullptr) {
    LOG(ERROR) << "UpdateParameter grads pointer is null";
    return -1;
  }
  if (keys.Size() != grads->size()) {
    LOG(ERROR) << "UpdateParameter keys/grads size mismatch: " << keys.Size()
               << " vs " << grads->size();
    return -1;
  }

  ParameterCompressor compressor;
  for (size_t i = 0; i < keys.Size(); ++i) {
    ParameterPack pack;
    pack.key      = keys[i];
    pack.dim      = grads->at(i).size();
    pack.emb_data = grads->at(i).data();
    compressor.AddItem(pack, nullptr);
  }
#ifdef ENABLE_PERF_REPORT
  auto serialize_done_time = std::chrono::high_resolution_clock::now();
#endif
  if (keys.Size() == 0) {
    LOG(WARNING) << "UpdateParameter no gradients to send";
    return 0;
  }

  UpdateParameterRequest request;
  UpdateParameterResponse response;
  request.set_table_name(table_name);

  brpc::Controller cntl;
#ifdef ENABLE_PERF_REPORT
  if (trace_id != 0) {
    cntl.http_request().SetHeader(
        "x-recstore-trace-id", std::to_string(trace_id));
  }
  auto rpc_start_time = std::chrono::high_resolution_clock::now();
#endif
  compressor.AppendToIOBuf(&cntl.request_attachment());
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.UpdateParameter(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UpdateParameter RPC failed: " << cntl.ErrorText();
    return -1;
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  auto serialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          serialize_done_time - start_time)
          .count();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - rpc_start_time)
          .count();
  report("ps_client_latency",
         "UpdateParameter",
         "latency_us",
         static_cast<double>(duration));

  const uint64_t effective_trace_id =
      trace_id == 0
          ? static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    start_time.time_since_epoch())
                    .count())
          : trace_id;
  std::string stage_id =
      "brpc_client::EmbUpdate|" + std::to_string(effective_trace_id);
  report("embupdate_stages",
         stage_id.c_str(),
         "client_serialize_us",
         static_cast<double>(serialize_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_rpc_us",
         static_cast<double>(rpc_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_total_us",
         static_cast<double>(duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_request_size",
         static_cast<double>(keys.Size()));
#endif

  return response.success() ? 0 : -1;
}

int BRPCParameterClient::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
#ifdef ENABLE_PERF_REPORT
  auto start_time         = std::chrono::high_resolution_clock::now();
  const uint64_t trace_id = recstore::g_trace_id;
#endif
  if (keys.Size() == 0) {
    return 0;
  }

  ParameterCompressor compressor;
  if (BuildUpdateBlocksFromFlat(
          keys, grads, num_rows, embedding_dim, &compressor) != 0) {
    return -1;
  }
#ifdef ENABLE_PERF_REPORT
  auto serialize_done_time = std::chrono::high_resolution_clock::now();
#endif

  UpdateParameterRequest request;
  UpdateParameterResponse response;
  request.set_table_name(table_name);

  brpc::Controller cntl;
#ifdef ENABLE_PERF_REPORT
  if (trace_id != 0) {
    cntl.http_request().SetHeader(
        "x-recstore-trace-id", std::to_string(trace_id));
  }
  auto rpc_start_time = std::chrono::high_resolution_clock::now();
#endif
  compressor.AppendToIOBuf(&cntl.request_attachment());
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.UpdateParameter(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UpdateParameterFlat RPC failed: " << cntl.ErrorText();
    return -1;
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  auto serialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          serialize_done_time - start_time)
          .count();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - rpc_start_time)
          .count();
  report("ps_client_latency",
         "UpdateParameterFlat",
         "latency_us",
         static_cast<double>(duration));

  const uint64_t effective_trace_id =
      trace_id == 0
          ? static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    start_time.time_since_epoch())
                    .count())
          : trace_id;
  std::string stage_id =
      "brpc_client::EmbUpdate|" + std::to_string(effective_trace_id);
  report("embupdate_stages",
         stage_id.c_str(),
         "client_serialize_us",
         static_cast<double>(serialize_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_rpc_us",
         static_cast<double>(rpc_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_total_us",
         static_cast<double>(duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_request_size",
         static_cast<double>(num_rows));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_embedding_dim",
         static_cast<double>(embedding_dim));
#endif

  return response.success() ? 0 : -1;
}

int BRPCParameterClient::InitEmbeddingTable(
    const std::string& table_name,
    const recstore::EmbeddingTableConfig& config) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  InitEmbeddingTableRequest request;
  InitEmbeddingTableResponse response;
  request.set_table_name(table_name);
  request.set_config_payload(config.Serialize());

  brpc::Controller cntl;
  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  stub.InitEmbeddingTable(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "InitEmbeddingTable RPC failed: " << cntl.ErrorText();
    return -1;
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_client_latency",
         "InitEmbeddingTable",
         "latency_us",
         static_cast<double>(duration));
#endif

  return response.success() ? 0 : -1;
}

uint64_t BRPCParameterClient::EmbWriteAsync(const base::RecTensor& keys,
                                            const base::RecTensor& values) {
  if (keys.dtype() != base::DataType::UINT64 || keys.dim() != 1) {
    LOG(ERROR) << "EmbWriteAsync expects keys as 1D UINT64 tensor, got dtype="
               << base::DataTypeToString(keys.dtype())
               << ", dim=" << keys.dim();
    return 0;
  }
  if (values.dtype() != base::DataType::FLOAT32 || values.dim() != 2) {
    LOG(ERROR)
        << "EmbWriteAsync expects values as 2D FLOAT32 tensor, got dtype="
        << base::DataTypeToString(values.dtype()) << ", dim=" << values.dim();
    return 0;
  }
  if (values.shape(0) != keys.shape(0)) {
    LOG(ERROR) << "EmbWriteAsync row mismatch: keys=" << keys.shape(0)
               << ", values=" << values.shape(0);
    return 0;
  }
  if (values.shape(1) <= 0) {
    LOG(ERROR) << "EmbWriteAsync invalid embedding dim: " << values.shape(1);
    return 0;
  }

  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* value_data  = values.data_as<float>();
  int64_t key_count        = keys.shape(0);
  int64_t emb_dim          = values.shape(1);
  if (key_count == 0) {
    return 0;
  }

  uint64_t prewrite_id = next_prewrite_id_++;
  int request_num =
      (static_cast<int>(key_count) + MAX_PARAMETER_BATCH_BRPC - 1) /
      MAX_PARAMETER_BATCH_BRPC;

  auto it = prewrite_batches_.emplace(prewrite_id, request_num).first;
  struct BrpcPrewriteBatch* pb = &it->second;

  recstoreps_brpc::ParameterService_Stub stub(channel_.get());
  for (int start = 0, index = 0; start < key_count;
       start += MAX_PARAMETER_BATCH_BRPC, ++index) {
    int key_size =
        std::min(static_cast<int>(key_count - start), MAX_PARAMETER_BATCH_BRPC);
    pb->key_sizes_[index]   = key_size;
    pb->controllers_[index] = std::make_unique<brpc::Controller>();

    ParameterCompressor compressor;
    for (int i = 0; i < key_size; ++i) {
      int64_t row = start + i;
      ParameterPack parameter_pack;
      parameter_pack.key      = key_data[row];
      parameter_pack.dim      = emb_dim;
      parameter_pack.emb_data = value_data + row * emb_dim;
      compressor.AddItem(parameter_pack, nullptr);
    }

    compressor.AppendToIOBuf(&pb->controllers_[index]->request_attachment());
    google::protobuf::Closure* done = brpc::NewCallback(OnPrewriteDone, pb);
    stub.PutParameter(
        pb->controllers_[index].get(),
        &pb->requests_[index],
        &pb->responses_[index],
        done);
  }

  return prewrite_id;
}

bool BRPCParameterClient::IsWriteDone(uint64_t write_id) {
  auto it = prewrite_batches_.find(write_id);
  if (it == prewrite_batches_.end()) {
    LOG(ERROR) << "Invalid prewrite_id: " << write_id;
    return false;
  }
  auto& pb = it->second;
  return pb.completed_count_ == pb.batch_size_;
}

void BRPCParameterClient::WaitForWrite(uint64_t write_id) {
  auto it = prewrite_batches_.find(write_id);
  if (it == prewrite_batches_.end()) {
    LOG(ERROR) << "Invalid prewrite_id: " << write_id;
    return;
  }
  auto& pb = it->second;
  for (int i = 0; i < pb.batch_size_; ++i) {
    if (!pb.controllers_[i]) {
      continue;
    }
    brpc::Join(pb.controllers_[i]->call_id());
    if (pb.controllers_[i]->Failed()) {
      LOG(ERROR) << "Async PutParameter failed: "
                 << pb.controllers_[i]->ErrorText();
    }
  }
  pb.completed_count_ = pb.batch_size_;
  prewrite_batches_.erase(it);
}

// 注册 BRPCParameterClient 到工厂
using BasePSClient = recstore::BasePSClient;
FACTORY_REGISTER(BasePSClient, brpc, BRPCParameterClient, json);
