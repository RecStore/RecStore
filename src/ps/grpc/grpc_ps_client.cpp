#include "grpc_ps_client.h"

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
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

constexpr int kDefaultRpcDeadlineMs   = 15000;
constexpr int kPrefetchPollIntervalMs = 200;
constexpr int kPrefetchMaxIdleRounds  = 150; // 200 ms * 150 = 30 s

void SetRpcDeadline(grpc::ClientContext* context,
                    int timeout_ms = kDefaultRpcDeadlineMs) {
  context->set_deadline(
      std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
}

// Records the first failure of a prefetch batch. Later errors are ignored so
// the reported cause stays the one that actually invalidated the responses.
void MarkPrefetchFailed(PrefetchBatch* batch, const std::string& message) {
  if (!batch->failed_) {
    batch->failed_ = true;
    batch->error_  = message;
  }
  LOG(ERROR) << "Prefetch failed: " << message;
}

// Consumes one completion event from a prefetch batch. Both transport
// failures (ok == false) and non-OK RPC statuses invalidate the batch.
void ConsumePrefetchCompletion(PrefetchBatch* batch, void* tag, bool ok) {
  const auto index = reinterpret_cast<intptr_t>(tag);
  if (index < 0 || index >= batch->batch_size_) {
    MarkPrefetchFailed(
        batch, fmt::format("unexpected completion tag {}", index));
    return;
  }
  if (!ok) {
    MarkPrefetchFailed(
        batch, fmt::format("RPC for request {} did not complete", index));
  } else if (!batch->status_[index].ok()) {
    const grpc::Status& status = batch->status_[index];
    MarkPrefetchFailed(
        batch,
        fmt::format("RPC for request {} failed: {} {}",
                    index,
                    static_cast<int>(status.error_code()),
                    status.error_message()));
  }
  batch->completed_count_++;
}

} // namespace

DEFINE_int32(get_parameter_threads, 4, "get clients per shard");
DEFINE_bool(parameter_client_random_init, false, "");

// New constructor that takes JSON config.
/*
Example: load config from file
std::ifstream config_file(FLAGS_config_path);
  nlohmann::json ex;
  config_file >> ex;
  json client_config = ex["client"];

*/
GRPCParameterClient::GRPCParameterClient(json config) {
  // Extract fields from JSON config
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
  auto* raw_cq = new grpc::CompletionQueue();
  cq_.reset(raw_cq);

  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

// Legacy constructor for backward compatibility
GRPCParameterClient::GRPCParameterClient(
    const std::string& host, int port, int shard)
    : host_(host),
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
  auto* raw_cq = new grpc::CompletionQueue();
  cq_.reset(raw_cq);

  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      base::RecTensor& values) {
#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (!recstore::IsFloatEmbeddingValues(
          values, static_cast<int64_t>(keys.Size()))) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }
  const int64_t D = values.shape(1);
  float* dst      = values.data_as<float>();
  std::memset(dst, 0, keys.Size() * static_cast<size_t>(D) * sizeof(float));

  if (FLAGS_parameter_client_random_init) {
    std::fill(dst, dst + keys.Size() * static_cast<size_t>(D), 0.1f);
    return 0;
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
    SetRpcDeadline(get_param_contexts_[index].get());
    get_param_resonse_readers_.emplace_back(stubs_[0]->AsyncGetParameter(
        get_param_contexts_[index].get(), request, cq_.get()));
    auto& rpc = get_param_resonse_readers_.back();
    // GetParameter(&context, request, &response);
    rpc->Finish(&response,
                &status,
                reinterpret_cast<void*>(static_cast<intptr_t>(index)));
  }

  bool transport_ok = true;
  for (int completed = 0; completed < request_num; ++completed) {
    void* got_tag = nullptr;
    bool ok       = false;
    if (!cq_->Next(&got_tag, &ok)) {
      LOG(ERROR) << "GetParameter completion queue shut down after "
                 << completed << "/" << request_num << " responses";
      return -1;
    }
    const auto index = reinterpret_cast<intptr_t>(got_tag);
    if (index < 0 || index >= request_num) {
      LOG(ERROR) << "GetParameter unexpected completion tag " << index;
      return -1;
    }
    if (!ok) {
      LOG(ERROR) << "GetParameter RPC did not complete for request " << index
                 << ": " << get_param_status_[index].error_code() << " "
                 << get_param_status_[index].error_message();
      transport_ok = false;
    }
  }
  if (!transport_ok) {
    return -1;
  }
  for (int i = 0; i < request_num; ++i) {
    if (!get_param_status_[i].ok()) {
      LOG(ERROR) << "GetParameter RPC failed for request " << i << ": "
                 << get_param_status_[i].error_code() << " "
                 << get_param_status_[i].error_message();
      return -1;
    }
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

  size_t row_offset = 0;
  for (int i = 0; i < get_param_responses_.size(); ++i) {
    auto& response             = get_param_responses_[i];
    int key_size               = get_param_key_sizes_[i];
    const std::string& payload = response.parameter_value();
    // The server emits one compressed item per requested key (missing rows
    // carry dim == 0), so an empty payload for a non-empty request means the
    // response is truncated or the peer is not the expected PS.
    if (payload.empty()) {
      LOG(ERROR) << "GetParameter empty payload for request " << i << " ("
                 << key_size << " keys)";
      return -1;
    }
    auto parameters =
        reinterpret_cast<const ParameterCompressReader*>(payload.data());
    if (!parameters->Valid(static_cast<int>(payload.size()))) {
      LOG(ERROR) << "GetParameter invalid payload for request " << i << ": "
                 << payload.size() << " bytes";
      return -1;
    }
    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return -1;
    }

    CopyCompressItemsToFlat(parameters, dst, D, row_offset);
    row_offset += static_cast<size_t>(key_size);
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

  return 0;
}

// return prefetch id
uint64_t
GRPCParameterClient::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
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
    SetRpcDeadline(pb.contexts_[index].get());
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
  uint64_t prefetch_id = 0;
  {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    prefetch_id = next_prefetch_id_++;
    prefetch_batches_.emplace(prefetch_id, std::move(pb));
  }

  return prefetch_id;
}

bool GRPCParameterClient::IsPrefetchDone(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> lk(prefetch_mu_);
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }
  auto& pb = it->second;
  if (pb.completed_count_ >= pb.batch_size_) {
    return true;
  }

  for (;;) {
    void* got_tag = nullptr;
    bool ok       = false;
    auto deadline = std::chrono::system_clock::now();
    auto status   = pb.cqs_->AsyncNext(&got_tag, &ok, deadline);
    if (status == grpc::CompletionQueue::NextStatus::GOT_EVENT) {
      ConsumePrefetchCompletion(&pb, got_tag, ok);
      if (pb.completed_count_ >= pb.batch_size_) {
        break;
      }
      continue;
    }
    if (status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
      MarkPrefetchFailed(&pb, "completion queue shut down during prefetch");
    }
    break;
  }
  // A failed batch is terminal as well. Otherwise callers polling for
  // completion would spin forever on a batch that can never succeed.
  return pb.failed_ || pb.completed_count_ >= pb.batch_size_;
}

void GRPCParameterClient::WaitForPrefetch(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> lk(prefetch_mu_);
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return;
  }
  auto& pb        = it->second;
  int idle_rounds = 0;
  while (pb.completed_count_ < pb.batch_size_) {
    void* got_tag = nullptr;
    bool ok       = false;
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::milliseconds(kPrefetchPollIntervalMs);
    auto status = pb.cqs_->AsyncNext(&got_tag, &ok, deadline);
    if (status == grpc::CompletionQueue::NextStatus::GOT_EVENT) {
      idle_rounds = 0;
      ConsumePrefetchCompletion(&pb, got_tag, ok);
      continue;
    }
    if (status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
      if (++idle_rounds >= kPrefetchMaxIdleRounds) {
        MarkPrefetchFailed(
            &pb,
            fmt::format("wait timed out for prefetch_id {} "
                        "({}/{} responses)",
                        prefetch_id,
                        pb.completed_count_,
                        pb.batch_size_));
        break;
      }
      continue;
    }
    MarkPrefetchFailed(
        &pb, "completion queue shut down while waiting prefetch");
    break;
  }
}

bool GRPCParameterClient::GetPrefetchResult(uint64_t prefetch_id,
                                            base::RecTensor& values) {
  // Callers of the single-shard API are not required to call
  // WaitForPrefetch first; make sure every response has been consumed before
  // inspecting statuses and payloads.
  WaitForPrefetch(prefetch_id);

  std::lock_guard<std::mutex> lk(prefetch_mu_);
  auto it = prefetch_batches_.find(prefetch_id);
  if (it == prefetch_batches_.end()) {
    LOG(ERROR) << "Invalid prefetch_id: " << prefetch_id;
    return false;
  }
  auto& pb        = it->second;
  int request_num = pb.batch_size_;

  if (pb.failed_) {
    LOG(ERROR) << "GetPrefetchResult failed for prefetch_id " << prefetch_id
               << ": " << pb.error_;
    return false;
  }

  int keys_size = 0;
  for (const auto& size : pb.key_sizes_) {
    keys_size += size;
  }
  const bool discard = values.data() == nullptr && values.dim() == 0;
  if (!discard && !recstore::EnsureEmbeddingOutput(
                      values, static_cast<int64_t>(keys_size))) {
    return false;
  }
  const int64_t D = discard ? 0 : values.shape(1);
  float* dst      = discard ? nullptr : values.data_as<float>();
  if (!discard && dst != nullptr && keys_size > 0) {
    std::memset(dst,
                0,
                static_cast<size_t>(keys_size) * static_cast<size_t>(D) *
                    sizeof(float));
  }
  size_t row_offset = 0;

  for (int i = 0; i < request_num; ++i) {
    auto& response             = pb.responses_[i];
    int key_size               = pb.key_sizes_[i];
    const std::string& payload = response.parameter_value();
    if (payload.empty()) {
      LOG(ERROR) << "Prefetch empty payload for request " << i << " ("
                 << key_size << " keys)";
      return false;
    }
    auto parameters =
        reinterpret_cast<const ParameterCompressReader*>(payload.data());
    if (!parameters->Valid(static_cast<int>(payload.size()))) {
      LOG(ERROR) << "Prefetch invalid payload for request " << i << ": "
                 << payload.size() << " bytes";
      return false;
    }
    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    if (!discard) {
      CopyCompressItemsToFlat(parameters, dst, D, row_offset);
      row_offset += static_cast<size_t>(key_size);
    }
  }

  return true;
}

bool GRPCParameterClient::ClearPS() {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::CLEAR_PS);
  grpc::ClientContext context;
  SetRpcDeadline(&context);
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC ClearPS failed: " << status.error_code() << " "
               << status.error_message();
  }
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
  SetRpcDeadline(&context);
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
  SetRpcDeadline(&context);
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
  SetRpcDeadline(&context, 30000);
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

bool GRPCParameterClient::SaveCheckpoint(const std::string& path,
                                         const std::string& metadata) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::SAVE_CHECKPOINT);
  request.add_arg1(path);
  request.add_arg2(metadata);
  grpc::ClientContext context;
  SetRpcDeadline(&context, 600000);
  const grpc::Status status = stubs_[0]->Command(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC checkpoint save failed: " << status.error_code() << " "
               << status.error_message();
  }
  return status.ok();
}

bool GRPCParameterClient::LoadCheckpoint(const std::string& path,
                                         const std::string& metadata) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::LOAD_CHECKPOINT);
  request.add_arg1(path);
  request.add_arg2(metadata);
  grpc::ClientContext context;
  SetRpcDeadline(&context, 600000);
  const grpc::Status status = stubs_[0]->Command(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC checkpoint load failed: " << status.error_code() << " "
               << status.error_message();
  }
  return status.ok();
}

bool GRPCParameterClient::PutParameter(const std::vector<uint64_t>& keys,
                                       const base::RecTensor& values) {
  if (!recstore::IsFloatEmbeddingValues(
          values, static_cast<int64_t>(keys.size()))) {
    LOG(ERROR) << "PutParameter keys/values size mismatch: " << keys.size();
    return false;
  }
  const int64_t D  = keys.empty() ? 0 : values.shape(1);
  const float* src = keys.empty() ? nullptr : values.data_as<float>();
  for (int start = 0, index = 0; start < keys.size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.size() - start), MAX_PARAMETER_BATCH);
    PutParameterRequest request;
    PutParameterResponse response;
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    CompressEmbeddingRows(
        &compressor,
        keys.data() + start,
        src + start * D,
        key_size,
        D,
        &blocks);
    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    request.mutable_parameter_value()->swap(blocks[0]);
    grpc::ClientContext context;
    SetRpcDeadline(&context);
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
    const base::RecTensor& grads,
    float learning_rate) {
#ifdef ENABLE_PERF_REPORT
  auto start_time         = std::chrono::high_resolution_clock::now();
  const uint64_t trace_id = recstore::g_trace_id;
#endif
  if (!recstore::IsFloatEmbeddingValues(
          grads, static_cast<int64_t>(keys.Size()))) {
    LOG(ERROR) << "UpdateParameter keys/grads size mismatch: " << keys.Size();
    return -1;
  }

  ParameterCompressor compressor;
  if (keys.Size() > 0) {
    CompressEmbeddingRows(
        &compressor,
        keys.Data(),
        grads.data_as<float>(),
        keys.Size(),
        grads.shape(1));
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
  request.set_learning_rate(learning_rate);
  compressor.ToBlock(request.mutable_gradients());
  if (request.gradients().empty()) {
    LOG(WARNING) << "UpdateParameter no serialized gradients payload";
    return 0;
  }

  grpc::ClientContext context;
  SetRpcDeadline(&context);
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
  return response.success() ? response.tag() : -1;
}

// BasePSClient pure virtual implementations
// int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys,
// float* values) {
//   return GetParameter(ConstArray<uint64_t>(keys.Data(), keys.Size()), values)
//   ? 0 : -1;
// }

int GRPCParameterClient::PutParameter(const base::ConstArray<uint64_t>& keys,
                                      const base::RecTensor& values) {
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  bool success = PutParameter(key_vec, values);
  if (!success) {
    LOG(ERROR) << "PutParameter batch failed";
  }
  return success ? 0 : -1;
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
  case recstore::PSCommand::SAVE_CHECKPOINT:
  case recstore::PSCommand::LOAD_CHECKPOINT:
    LOG(WARNING) << "Checkpoint command requires path and metadata";
    break;
  default:
    LOG(ERROR) << "Unknown PS command: " << static_cast<int>(command);
    break;
  }
}

int64_t GRPCParameterClient::ExchangeSparseGradientCommit(
    const std::string& table, const std::string& payload) {
  recstoreps::SparseGradientCommitRequest request;
  recstoreps::SparseGradientCommitResponse response;
  request.set_table_name(table);
  request.set_payload(payload);
  grpc::ClientContext context;
  SetRpcDeadline(&context);
  const auto status =
      stubs_[0]->SparseGradientCommit(&context, request, &response);
  if (!status.ok() || !response.error().empty()) {
    throw std::runtime_error(
        "SparseGradientCommit: " +
        (status.ok() ? response.error() : status.error_message()));
  }
  return response.committed_step();
}
