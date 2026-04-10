#include "grpc_ps_client.h"

#include <fmt/core.h>
#include <grpcpp/grpcpp.h>

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
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"

#ifdef ENABLE_PERF_REPORT
#  include "base/report/report_client.h"
#  include <chrono>
#endif

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

DEFINE_int32(get_parameter_threads, 4, "get clients per shard");
DEFINE_bool(parameter_client_random_init, false, "");

// 新的构造函数，接收 json 配置参数
/*
可用下面的代码来读取配置文件
std::ifstream config_file(FLAGS_config_path);
  nlohmann::json ex;
  config_file >> ex;
  json client_config = ex["client"];

*/
GRPCParameterClient::GRPCParameterClient(json config)
    : recstore::BasePSClient(config) {
  // 从json配置中提取参数
  host_       = config.value("host", "localhost");
  port_       = config.value("port", 15000);
  shard_      = config.value("shard", 0);
  nr_clients_ = FLAGS_get_parameter_threads;
  Initialize();

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  args.SetMaxSendMessageSize(-1);

  channel_ = grpc::CreateCustomChannel(
      fmt::format("{}:{}", host_, port_),
      grpc::InsecureChannelCredentials(),
      args);

  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

// 保留原有的构造函数以保持向后兼容
GRPCParameterClient::GRPCParameterClient(
    const std::string& host, int port, int shard)
    : recstore::BasePSClient(
          json{{"host", host}, {"port", port}, {"shard", shard}}),
      host_(host),
      port_(port),
      shard_(shard),
      nr_clients_(FLAGS_get_parameter_threads) {
  Initialize();

  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  args.SetMaxSendMessageSize(-1);

  channel_ = grpc::CreateCustomChannel(
      fmt::format("{}:{}", host, port),
      grpc::InsecureChannelCredentials(),
      args);

  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      float* values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (FLAGS_parameter_client_random_init) {
    CHECK(0) << "todo implement";
    return true;
  }

  get_param_key_sizes_.clear();
  get_param_status_.clear();
  get_param_requests_.clear();
  get_param_responses_.clear();
  get_param_resonse_readers_.clear();
  get_param_contexts_.clear();

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;
  get_param_status_.resize(request_num);
  get_param_requests_.resize(request_num);
  get_param_responses_.resize(request_num);
  get_param_contexts_.resize(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    get_param_key_sizes_.emplace_back(key_size);
    auto& status   = get_param_status_[index];
    auto& request  = get_param_requests_[index];
    auto& response = get_param_responses_[index];
    request.set_keys(reinterpret_cast<const char*>(&keys[start]),
                     sizeof(uint64_t) * key_size);
    // rpc
    // grpc::ClientContext context;
    if (!get_param_contexts_[index]) {
      get_param_contexts_[index] = std::make_unique<grpc::ClientContext>();
    }
    std::unique_ptr<ClientAsyncResponseReader<GetParameterResponse>> rpc =
        stubs_[0]->AsyncGetParameter(
            get_param_contexts_[index].get(), request, &cq);
    rpc->Finish(&response, &status, reinterpret_cast<void*>(index));
  }
  int get = 0;
  while (get != request_num) {
    void* got_tag;
    bool ok = false;
    cq.Next(&got_tag, &ok);
    if (!ok) {
      LOG(ERROR) << "error";
    }
    get++;
  }
#ifdef ENABLE_PERF_REPORT
  auto after_rpc_time = std::chrono::high_resolution_clock::now();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          after_rpc_time - start_time)
          .count();
  double start_us_for_rpc =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_rpc =
      "grpc_client::GetParameter|" +
      std::to_string(static_cast<uint64_t>(start_us_for_rpc));
  report("embread_stages",
         report_id_for_rpc.c_str(),
         "rpc_duration_us",
         static_cast<double>(rpc_duration));
#endif
  size_t get_embedding_acc = 0;
  int old_dimension        = -1;

  for (int i = 0; i < get_param_responses_.size(); ++i) {
    auto& response  = get_param_responses_[i];
    int key_size    = get_param_key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader*>(
        response.parameter_value().data());

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
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();

  auto deserialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - after_rpc_time)
          .count();

  report("embread_stages",
         "grpc_client::GetParameter",
         "deserialize_duration_us",
         static_cast<double>(deserialize_duration));

  report("embread_stages",
         "grpc_client::GetParameter",
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         "grpc_client::GetParameter",
         "request_size",
         static_cast<double>(keys.Size()));

  FlameGraphData grpc_client_data = {
      "grpc_ps_client::GetParameter",
      start_us,
      1, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};

  std::string unique_id = "embread_debug";
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), grpc_client_data);
#endif

  return true;
}

int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      std::vector<std::vector<float>>* values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (FLAGS_parameter_client_random_init) {
    values->clear();
    values->reserve(keys.Size());
    for (size_t i = 0; i < keys.Size(); i++)
      values->emplace_back(std::vector<float>(128, 0.1));

    return true;
  }

  values->clear();
  get_param_key_sizes_.clear();
  get_param_status_.clear();
  get_param_requests_.clear();
  get_param_responses_.clear();
  get_param_resonse_readers_.clear();
  get_param_contexts_.clear();

  values->reserve(keys.Size());

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;

  get_param_status_.resize(request_num);
  get_param_requests_.resize(request_num);
  get_param_responses_.resize(request_num);
  get_param_contexts_.resize(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    get_param_key_sizes_.emplace_back(key_size);
    auto& status   = get_param_status_[index];
    auto& request  = get_param_requests_[index];
    auto& response = get_param_responses_[index];
    request.set_keys(reinterpret_cast<const char*>(&keys[start]),
                     sizeof(uint64_t) * key_size);
    // rpc
    // grpc::ClientContext context;
    if (!get_param_contexts_[index]) {
      get_param_contexts_[index] = std::make_unique<grpc::ClientContext>();
    }
    get_param_resonse_readers_.emplace_back(stubs_[0]->AsyncGetParameter(
        get_param_contexts_[index].get(), request, &cq));
    auto& rpc = get_param_resonse_readers_.back();
    // GetParameter(&context, request, &response);
    rpc->Finish(&response, &status, reinterpret_cast<void*>(index));
  }

  int get = 0;
  while (get != request_num) {
    void* got_tag;
    bool ok = false;
    cq.Next(&got_tag, &ok);
    if (unlikely(!ok)) {
      LOG(ERROR) << "error";
    }
    get++;
  }

#ifdef ENABLE_PERF_REPORT
  auto after_rpc_time = std::chrono::high_resolution_clock::now();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          after_rpc_time - start_time)
          .count();
  double start_us_for_rpc =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string report_id_for_rpc =
      "grpc_client::GetParameter|" +
      std::to_string(static_cast<uint64_t>(start_us_for_rpc));
  report("embread_stages",
         report_id_for_rpc.c_str(),
         "rpc_duration_us",
         static_cast<double>(rpc_duration));
#endif

  for (int i = 0; i < get_param_responses_.size(); ++i) {
    auto& response  = get_param_responses_[i];
    int key_size    = get_param_key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader*>(
        response.parameter_value().data());

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
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();

  auto deserialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - after_rpc_time)
          .count();

  report("embread_stages",
         "grpc_client::GetParameter",
         "deserialize_duration_us",
         static_cast<double>(deserialize_duration));

  report("embread_stages",
         "grpc_client::GetParameter",
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         "grpc_client::GetParameter",
         "request_size",
         static_cast<double>(keys.Size()));

  FlameGraphData grpc_client_data = {
      "grpc_ps_client::GetParameter",
      start_us,
      1, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};

  std::string unique_id = "embread_debug";
  report_flame_graph("emb_read_flame_map", unique_id.c_str(), grpc_client_data);
#endif

  return true;
}

// return prefetch id
uint64_t
GRPCParameterClient::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
  uint64_t prefetch_id = next_prefetch_id_++;
  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;

  struct PrefetchBatch pb(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    pb.key_sizes_[index] = key_size;
    auto& status         = pb.status_[index];
    if (!pb.contexts_[index]) {
      pb.contexts_[index] = std::make_unique<grpc::ClientContext>();
    }
    auto& request  = pb.requests_[index];
    auto& response = pb.responses_[index];
    request.set_keys(reinterpret_cast<const char*>(&keys[start]),
                     sizeof(uint64_t) * key_size);
    // rpc
    // grpc::ClientContext context;
    pb.response_readers_.emplace_back(stubs_[0]->AsyncGetParameter(
        pb.contexts_[index].get(), request, pb.cqs_.get()));
    auto& rpc = pb.response_readers_.back();
    // GetParameter(&context, request, &response);
    rpc->Finish(&response, &status, reinterpret_cast<void*>(index));
  }
  prefetch_batches_.emplace(prefetch_id, std::move(pb));

  return prefetch_id;
}

bool GRPCParameterClient::IsPrefetchDone(uint64_t prefetch_id) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }
  auto& pb        = it->second;
  int request_num = pb.batch_size_;
  int get         = 0;

  if (pb.completed_count_ == pb.batch_size_) {
    return true;
  }

  void* got_tag = nullptr;
  bool ok       = false;
  auto deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(0);
  for (;;) {
    auto status = pb.cqs_->AsyncNext(&got_tag, &ok, deadline);
    if (status == grpc::CompletionQueue::NextStatus::GOT_EVENT) {
      if (unlikely(!ok)) {
        LOG(ERROR) << "CompletionQueue returned not ok for prefetch";
      }
      pb.completed_count_++;
      if (pb.completed_count_ == pb.batch_size_)
        break;
      deadline =
          std::chrono::system_clock::now() + std::chrono::milliseconds(0);
      continue;
    } else if (status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
      break;
    } else {
      LOG(ERROR) << "CompletionQueue shutdown during prefetch";
      break;
    }
  }
  return (pb.completed_count_ == pb.batch_size_);
}

void GRPCParameterClient::WaitForPrefetch(uint64_t prefetch_id) {
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return;
  }
  auto& pb      = it->second;
  void* got_tag = nullptr;
  bool ok       = false;
  while (pb.completed_count_ < pb.batch_size_) {
    auto status = pb.cqs_->Next(&got_tag, &ok);
    if (!status) {
      LOG(ERROR) << "CompletionQueue shutdown while waiting prefetch";
      break;
    }
    if (unlikely(!ok)) {
      LOG(ERROR) << "CompletionQueue returned not ok for prefetch";
    }
    pb.completed_count_++;
  }
}

bool GRPCParameterClient::GetPrefetchResult(
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
    auto& response  = pb.responses_[i];
    int key_size    = pb.key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader*>(
        response.parameter_value().data());

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

  return true;
}

bool GRPCParameterClient::GetPrefetchResultFlat(
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
    auto& response  = pb.responses_[i];
    int key_size    = pb.key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader*>(
        response.parameter_value().data());

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

  return true;
}

bool GRPCParameterClient::ClearPS() {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::CLEAR_PS);
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

// Read n bytes from the server. The server does not access storage;
// it generates data randomly instead.
bool GRPCParameterClient::LoadFakeData(int64_t n) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::LOAD_FAKE_DATA);
  request.add_arg1(&n, sizeof(int64_t));
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC LoadFakeData failed: " << status.error_code() << " "
               << status.error_message();
    return false;
  }
  if (response.reply().size() != static_cast<size_t>(n)) {
    LOG(ERROR) << "gRPC LoadFakeData reply size mismatch: expected " << n
               << ", got " << response.reply().size();
    return false;
  }
  return true;
}

// Write n bytes(random generated) into the server
bool GRPCParameterClient::DumpFakeData(int64_t n) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::DUMP_FAKE_DATA);
  request.add_arg1(&n, sizeof(int64_t));
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC DumpFakeData failed: " << status.error_code() << " "
               << status.error_message();
    return false;
  }
  if (response.reply() != "ok") {
    LOG(ERROR) << "gRPC DumpFakeData unexpected reply: " << response.reply();
    return false;
  }
  return true;
}

bool GRPCParameterClient::LoadCkpt(
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
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

bool GRPCParameterClient::PutParameter(
    const std::vector<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  if (keys.size() != values.size()) {
    LOG(ERROR) << "PutParameter keys/values size mismatch: " << keys.size()
               << " vs " << values.size();
    return false;
  }
  for (int start = 0, index = 0; start < keys.size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.size() - start), MAX_PARAMETER_BATCH);
    PutParameterRequest request;
    PutParameterResponse response;
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    for (int i = start; i < start + key_size; i++) {
      auto each_key   = keys[i];
      auto& embedding = values[i];
      ParameterPack parameter_pack;
      parameter_pack.key      = each_key;
      parameter_pack.dim      = embedding.size();
      parameter_pack.emb_data = embedding.data();
      compressor.AddItem(parameter_pack, &blocks);
    }
    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    request.mutable_parameter_value()->swap(blocks[0]);
    grpc::ClientContext context;
    grpc::Status status = stubs_[0]->PutParameter(&context, request, &response);
    if (status.ok()) {
      continue;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return false;
    }
  }
  return true;
}

int GRPCParameterClient::UpdateParameter(
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
  compressor.ToBlock(request.mutable_gradients());
  if (request.gradients().empty()) {
    LOG(WARNING) << "UpdateParameter no serialized gradients payload";
    return 0;
  }

  grpc::ClientContext context;
#ifdef ENABLE_PERF_REPORT
  if (trace_id != 0) {
    context.AddMetadata("x-recstore-trace-id", std::to_string(trace_id));
  }
  auto rpc_start_time = std::chrono::high_resolution_clock::now();
#endif
  grpc::Status status =
      stubs_[0]->UpdateParameter(&context, request, &response);
#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto serialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          serialize_done_time - start_time)
          .count();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - rpc_start_time)
          .count();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  std::string stage_id =
      "grpc_client::EmbUpdate|" +
      std::to_string(
          trace_id == 0
              ? static_cast<uint64_t>(
                    std::chrono::duration_cast< std::chrono::microseconds>(
                        start_time.time_since_epoch())
                        .count())
              : trace_id);
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
         static_cast<double>(total_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_request_size",
         static_cast<double>(keys.Size()));
#endif
  if (!status.ok()) {
    LOG(ERROR) << "UpdateParameter RPC failed: " << status.error_message();
    return -1;
  }
  return response.success() ? 0 : -1;
}

int GRPCParameterClient::UpdateParameterFlat(
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
  compressor.ToBlock(request.mutable_gradients());
  if (request.gradients().empty()) {
    return 0;
  }

  grpc::ClientContext context;
#ifdef ENABLE_PERF_REPORT
  if (trace_id != 0) {
    context.AddMetadata("x-recstore-trace-id", std::to_string(trace_id));
  }
  auto rpc_start_time = std::chrono::high_resolution_clock::now();
#endif
  grpc::Status status =
      stubs_[0]->UpdateParameter(&context, request, &response);
#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto serialize_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          serialize_done_time - start_time)
          .count();
  auto rpc_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - rpc_start_time)
          .count();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  std::string stage_id =
      "grpc_client::EmbUpdate|" +
      std::to_string(
          trace_id == 0
              ? static_cast<uint64_t>(
                    std::chrono::duration_cast< std::chrono::microseconds>(
                        start_time.time_since_epoch())
                        .count())
              : trace_id);
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
         static_cast<double>(total_duration));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_request_size",
         static_cast<double>(num_rows));
  report("embupdate_stages",
         stage_id.c_str(),
         "client_embedding_dim",
         static_cast<double>(embedding_dim));
#endif
  if (!status.ok()) {
    LOG(ERROR) << "UpdateParameterFlat RPC failed: " << status.error_message();
    return -1;
  }
  return response.success() ? 0 : -1;
}

int GRPCParameterClient::InitEmbeddingTable(
    const std::string& table_name,
    const recstore::EmbeddingTableConfig& config) {
  InitEmbeddingTableRequest request;
  InitEmbeddingTableResponse response;
  request.set_table_name(table_name);
  request.set_config_payload(config.Serialize());

  grpc::ClientContext context;
  grpc::Status status =
      stubs_[0]->InitEmbeddingTable(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "InitEmbeddingTable RPC failed: " << status.error_message();
    return -1;
  }
  return response.success() ? 0 : -1;
}

// 实现 BasePSClient 的纯虚函数
// int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
// float* values) {
//   return GetParameter(ConstArray<uint64_t>(keys.Data(), keys.Size()), values)
//   ? 0 : -1;
// }

int GRPCParameterClient::AsyncGetParameter(
    const base::ConstArray<uint64_t>& keys, float* values) {
  return GetParameter(keys, values);
}

int GRPCParameterClient::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  bool success = PutParameter(key_vec, values);
  if (!success) {
    LOG(ERROR) << "PutParameter batch failed";
  }
  return success ? 1 : 0;
}

void GRPCParameterClient::Command(recstore::PSCommand command) {
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

uint64_t GRPCParameterClient::EmbWriteAsync(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  uint64_t prewrite_id = next_prewrite_id_++;
  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;

  struct PrewriteBatch pb(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    pb.key_sizes_[index] = key_size;
    auto& status         = pb.status_[index];
    if (!pb.contexts_[index]) {
      pb.contexts_[index] = std::make_unique<grpc::ClientContext>();
    }
    auto& request  = pb.requests_[index];
    auto& response = pb.responses_[index];

    // 将键值对打包
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    for (int i = start; i < start + key_size; i++) {
      auto each_key   = keys[i];
      auto& embedding = values[i];
      ParameterPack parameter_pack;
      parameter_pack.key      = each_key;
      parameter_pack.dim      = embedding.size();
      parameter_pack.emb_data = embedding.data();
      compressor.AddItem(parameter_pack, &blocks);
    }
    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);

    request.mutable_parameter_value()->swap(blocks[0]);

    // 发送异步请求
    pb.response_readers_.emplace_back(stubs_[0]->AsyncPutParameter(
        pb.contexts_[index].get(), request, pb.cqs_.get()));

    // 异步调用，设置回调
    auto& rpc = pb.response_readers_.back();
    rpc->Finish(&response, &status, reinterpret_cast<void*>(index));
  }

  // 将批次信息存储到 prewrite_batches_ 中
  prewrite_batches_.emplace(prewrite_id, std::move(pb));
  return prewrite_id;
}

bool GRPCParameterClient::IsWriteDone(uint64_t write_id) {
  LOG(ERROR) << "IsWriteDone not implemented!";
  auto it = prewrite_batches_.find(write_id);
  if (it == prewrite_batches_.end()) {
    LOG(ERROR) << "Invalid prewrite_id: " << write_id;
    return false;
  }
  auto& pb = it->second;
  return (pb.completed_count_ == pb.batch_size_);
}

void GRPCParameterClient::WaitForWrite(uint64_t write_id) {
  auto it = prewrite_batches_.find(write_id);
  if (it == prewrite_batches_.end()) {
    LOG(ERROR) << "Invalid prewrite_id: " << write_id;
    return;
  }
  auto& pb = it->second;
  while (pb.completed_count_ < pb.batch_size_) {
    void* got_tag = nullptr;
    bool ok       = false;
    if (!pb.cqs_->Next(&got_tag, &ok)) {
      break;
    }
    if (unlikely(!ok)) {
      LOG(ERROR) << "Completion queue returned not ok for write";
      continue;
    }
    pb.completed_count_++;
  }
}

// 注册 GRPCParameterClient 到工厂
using BasePSClient = recstore::BasePSClient;
FACTORY_REGISTER(BasePSClient, grpc, GRPCParameterClient, json);
