#include "brpc_ps_server.h"

#include <brpc/server.h>
#include <fmt/core.h>
#include <gflags/gflags.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/base.h"
#include "base/factory.h"
#include "base/flatc.h"
#include "base/log.h"
#include "base/timer.h"
#include "ps/base/base_ps_server.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/base/parameters.h"
#include "ps_brpc.pb.h"
#include "recstore_config.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include <cstdlib>
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

DEFINE_string(brpc_config_path,
              RECSTORE_PATH "/recstore_config.json",
              "config file path");
DEFINE_int32(brpc_server_port, 15000, "bRPC server port");
DEFINE_int32(brpc_server_num_threads,
             0,
             "Number of threads for bRPC server, 0 means auto");

namespace recstore {

namespace {

void AppendShardSuffixIfPresent(
    nlohmann::json& config_node, const char* key, int shard_id) {
  if (!config_node.contains(key) || !config_node[key].is_string()) {
    return;
  }
  config_node[key] =
      config_node[key].get<std::string>() + "_" + std::to_string(shard_id);
}

bool ExtractPayloadBytes(
    const brpc::Controller* cntl,
    const std::string& proto_bytes,
    std::string* payload_storage,
    const char** payload_data,
    int* payload_size) {
  if (!cntl->request_attachment().empty()) {
    payload_storage->clear();
    cntl->request_attachment().copy_to(payload_storage);
    *payload_data = payload_storage->data();
    *payload_size = payload_storage->size();
    return true;
  }
  if (!proto_bytes.empty()) {
    *payload_data = proto_bytes.data();
    *payload_size = proto_bytes.size();
    return true;
  }
  *payload_data = nullptr;
  *payload_size = 0;
  return false;
}

} // namespace

BRPCParameterServiceImpl::BRPCParameterServiceImpl(CachePS* cache_ps)
    : cache_ps_(cache_ps) {
  start_time_ = std::chrono::steady_clock::now();
}

void BRPCParameterServiceImpl::ResetMetrics() {
  total_get_requests_ = 0;
  total_put_requests_ = 0;
  total_get_keys_     = 0;
  total_put_keys_     = 0;
  total_get_bytes_    = 0;
  total_put_bytes_    = 0;
  start_time_         = std::chrono::steady_clock::now();
}

void BRPCParameterServiceImpl::PrintMetrics(const std::string& table_name,
                                            const std::string& unique_id) {
  auto now         = std::chrono::steady_clock::now();
  double elapsed_s = std::chrono::duration<double>(now - start_time_).count();
  if (elapsed_s > 0) {
    double overall_qps =
        (total_get_requests_ + total_put_requests_) / elapsed_s;
    double overall_throughput_mbps =
        ((total_get_bytes_ + total_put_bytes_) / 1024.0 / 1024.0) / elapsed_s;

    // Report QPS and throughput metrics
    // report(table_name.c_str(), unique_id.c_str(), "overall_qps",
    // overall_qps); report(table_name.c_str(),
    //        unique_id.c_str(),
    //        "overall_throughput_mbps",
    //        overall_throughput_mbps);
  }
}

void BRPCParameterServiceImpl::GetParameter(
    google::protobuf::RpcController* controller,
    const GetParameterRequest* request,
    GetParameterResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

#ifdef ENABLE_PERF_REPORT
  auto start_time       = std::chrono::high_resolution_clock::now();
  uint64_t trace_id     = cntl->log_id();
  std::string unique_id = "embread_debug" + std::to_string(trace_id);
#endif
  std::string keys_storage;
  const char* keys_data = nullptr;
  int keys_size         = 0;
  ExtractPayloadBytes(
      cntl, request->keys(), &keys_storage, &keys_data, &keys_size);
  base::ConstArray<uint64_t> keys_array;
  keys_array.SetData(keys_data, keys_size);
  if (keys_size % static_cast<int>(sizeof(uint64_t)) != 0) {
    LOG(ERROR) << "GetParameter invalid keys payload size=" << keys_size;
    return;
  }
  bool isPerf = request->has_perf() && request->perf();

  if (isPerf) {
    xmh::PerfCounter::Record("PS Get Keys", keys_array.Size());
  }

  xmh::Timer timer_ps_get_req("PS GetParameter Req");
  ParameterCompressor compressor;

  FB_LOG_EVERY_MS(INFO, 1000)
      << "[bRPC PS] Getting " << keys_array.Size() << " keys";

  int total_dim = 0;

#ifdef ENABLE_PERF_REPORT
  auto cache_loop_start = std::chrono::high_resolution_clock::now();
#endif
  std::vector<ParameterPack> packs;
  packs.reserve(keys_array.Size());
  cache_ps_->GetParameterRun2Completion(keys_array, packs, 0);

  for (auto& pack : packs) {
    compressor.AddItem(pack, nullptr);
    total_dim += pack.dim;
  }
#ifdef ENABLE_PERF_REPORT
  auto cache_loop_end = std::chrono::high_resolution_clock::now();
  auto cache_loop_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          cache_loop_end - cache_loop_start)
          .count();
  double cache_loop_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          cache_loop_start.time_since_epoch())
          .count();

  std::string report_id_cache =
      "brpc_server::GetParameter|" +
      std::to_string(static_cast<uint64_t>(cache_loop_start_us));
  report("embread_stages",
         report_id_cache.c_str(),
         "cache_lookup_us",
         static_cast<double>(cache_loop_duration));

  FlameGraphData cache_loop_fg = {
      "brpc_server::CacheGet_Loop",
      cache_loop_start_us,
      4, // level
      static_cast<double>(cache_loop_duration),
      static_cast<double>(cache_loop_duration)};
  if (trace_id != 0) {
    std::string cache_unique_id =
        "embread_debug|" +
        std::to_string(static_cast<uint64_t>(cache_loop_start_us));
    report_flame_graph(
        "emb_read_flame_map", cache_unique_id.c_str(), cache_loop_fg);
  }

  auto toblock_start = std::chrono::high_resolution_clock::now();
#endif

  compressor.AppendToIOBuf(&cntl->response_attachment());

#ifdef ENABLE_PERF_REPORT
  auto toblock_end = std::chrono::high_resolution_clock::now();
  auto toblock_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          toblock_end - toblock_start)
          .count();
  double toblock_start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          toblock_start.time_since_epoch())
          .count();
  FlameGraphData toblock_fg = {
      "brpc_server::Compressor_ToBlock",
      toblock_start_us,
      4, // level
      static_cast<double>(toblock_duration),
      static_cast<double>(toblock_duration)};
  if (trace_id != 0) {
    std::string toblock_unique_id =
        "embread_debug|" +
        std::to_string(static_cast<uint64_t>(toblock_start_us));
    report_flame_graph(
        "emb_read_flame_map", toblock_unique_id.c_str(), toblock_fg);
  }
#endif

  total_get_requests_++;
  total_get_keys_ += keys_array.Size();
  total_get_bytes_ += total_dim * sizeof(float);

  if (isPerf) {
    timer_ps_get_req.end();
  } else {
    timer_ps_get_req.destroy();
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_server_latency",
         "GetParameter",
         "latency_us",
         static_cast<double>(duration));

  double start_us_for_key =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbRead|" + std::to_string(static_cast<uint64_t>(start_us_for_key));
  report("op_latency",
         op_latency_key.c_str(),
         "recserver_us",
         static_cast<double>(duration));

  double start_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();

  std::string report_id = "brpc_server::GetParameter|" +
                          std::to_string(static_cast<uint64_t>(start_us));

  report("embread_stages",
         report_id.c_str(),
         "duration_us",
         static_cast<double>(duration));

  report("embread_stages",
         report_id.c_str(),
         "request_size",
         static_cast<double>(keys_array.Size()));

  FlameGraphData fg_data = {
      "brpc_server::GetParameter",
      start_us,
      3, // level
      static_cast<double>(duration),
      static_cast<double>(duration)};
  if (trace_id != 0) {
    std::string req_unique_id =
        "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
    report_flame_graph("emb_read_flame_map", req_unique_id.c_str(), fg_data);
  }
#endif
}

void BRPCParameterServiceImpl::Command(
    google::protobuf::RpcController* controller,
    const CommandRequest* request,
    CommandResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

  if (request->command() == recstoreps_brpc::PSCommand::CLEAR_PS) {
    LOG(WARNING) << "[PS Command] Clear All";
    cache_ps_->Clear();
  } else if (request->command() == recstoreps_brpc::PSCommand::RELOAD_PS) {
    LOG(WARNING) << "[PS Command] Reload PS";
    CHECK_NE(request->arg1().size(), 0);
    CHECK_NE(request->arg2().size(), 0);
    CHECK_EQ(request->arg1().size(), 1);
    LOG(WARNING) << "model_config_path = " << request->arg1()[0];
    for (int i = 0; i < request->arg2().size(); i++) {
      LOG(WARNING) << fmt::format("emb_file {}: {}", i, request->arg2()[i]);
    }
    std::vector<std::string> arg1;
    for (auto& each : request->arg1()) {
      arg1.push_back(each);
    }
    std::vector<std::string> arg2;
    for (auto& each : request->arg2()) {
      arg2.push_back(each);
    }
    cache_ps_->Initialize(arg1, arg2);
  } else if (request->command() == recstoreps_brpc::PSCommand::LOAD_FAKE_DATA) {
    if (request->arg1_size() != 1 ||
        static_cast<size_t>(request->arg1(0).size()) != sizeof(int64_t)) {
      LOG(ERROR) << "LOAD_FAKE_DATA: arg1 must be one " << sizeof(int64_t)
                 << "-byte int64_t (requested reply payload size)";
      cntl->SetFailed(EINVAL, "LOAD_FAKE_DATA invalid arg1 size");
      return;
    }
    int64_t payload_bytes = 0;
    std::memcpy(&payload_bytes, request->arg1(0).data(), sizeof(int64_t));
    if (payload_bytes < 0) {
      LOG(ERROR) << "LOAD_FAKE_DATA: payload_bytes must be non-negative, got "
                 << payload_bytes;
      cntl->SetFailed(
          EINVAL, "LOAD_FAKE_DATA payload_bytes must be non-negative");
      return;
    }
    constexpr int64_t kMaxReplyPayload = 16 * 1024 * 1024;
    if (payload_bytes > kMaxReplyPayload) {
      LOG(ERROR) << "LOAD_FAKE_DATA: payload_bytes " << payload_bytes
                 << " exceeds cap " << kMaxReplyPayload;
      cntl->SetFailed(EINVAL, "LOAD_FAKE_DATA payload too large");
      return;
    }
    std::string fake(static_cast<size_t>(payload_bytes), '\xab');
    response->set_reply(std::move(fake));
  } else if (request->command() == recstoreps_brpc::PSCommand::DUMP_FAKE_DATA) {
    if (request->arg1_size() != 1 ||
        static_cast<size_t>(request->arg1(0).size()) != sizeof(int64_t)) {
      LOG(ERROR) << "DUMP_FAKE_DATA: arg1 must be one " << sizeof(int64_t)
                 << "-byte int64_t (payload bytes n)";
      cntl->SetFailed(EINVAL, "DUMP_FAKE_DATA invalid arg1 size");
      return;
    }
    int64_t n = 0;
    std::memcpy(&n, request->arg1(0).data(), sizeof(int64_t));
    if (n <= 0) {
      LOG(ERROR) << "DUMP_FAKE_DATA: n must be positive";
      cntl->SetFailed(EINVAL, "DUMP_FAKE_DATA n must be positive");
      return;
    }
    if (n % static_cast<int64_t>(sizeof(float)) != 0) {
      LOG(ERROR) << "DUMP_FAKE_DATA: n must be a multiple of " << sizeof(float);
      cntl->SetFailed(
          EINVAL, "DUMP_FAKE_DATA n must be multiple of sizeof(float)");
      return;
    }
    constexpr int64_t kMaxDumpBytes = 64 * 1024 * 1024;
    if (n > kMaxDumpBytes) {
      LOG(ERROR) << "DUMP_FAKE_DATA: n exceeds cap " << kMaxDumpBytes;
      cntl->SetFailed(EINVAL, "DUMP_FAKE_DATA n exceeds cap");
      return;
    }
    response->set_reply("ok");
  } else {
    LOG(FATAL) << "invalid command";
  }
}

void BRPCParameterServiceImpl::PutParameter(
    google::protobuf::RpcController* controller,
    const PutParameterRequest* request,
    PutParameterResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

  std::string payload_storage;
  const char* payload_data = nullptr;
  int payload_size         = 0;
  if (!ExtractPayloadBytes(
          cntl,
          request->parameter_value(),
          &payload_storage,
          &payload_data,
          &payload_size)) {
    LOG(ERROR) << "PutParameter empty payload";
    return;
  }

#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  const ParameterCompressReader* reader =
      reinterpret_cast<const ParameterCompressReader*>(payload_data);
  if (!reader->Valid(payload_size)) {
    LOG(ERROR) << "PutParameter invalid payload, size=" << payload_size;
    return;
  }
  int size             = reader->item_size();
  uint64_t total_bytes = 0;

  for (int i = 0; i < size; i++) {
    cache_ps_->PutSingleParameter(reader->item(i), 0);
    total_bytes += reader->item(i)->dim * sizeof(float);
  }

  total_put_requests_++;
  total_put_keys_ += size;
  total_put_bytes_ += total_bytes;

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_server_latency",
         "PutParameter",
         "latency_us",
         static_cast<double>(duration));

  double start_us_for_key =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbWrite|" + std::to_string(static_cast<uint64_t>(start_us_for_key));
  report("op_latency",
         op_latency_key.c_str(),
         "recserver_us",
         static_cast<double>(duration));
#endif
}

void BRPCParameterServiceImpl::UpdateParameter(
    google::protobuf::RpcController* controller,
    const UpdateParameterRequest* request,
    UpdateParameterResponse* reply,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);

#ifdef ENABLE_PERF_REPORT
  auto start_time   = std::chrono::high_resolution_clock::now();
  uint64_t trace_id = 0;
  const std::string* header_trace =
      cntl->http_request().GetHeader("x-recstore-trace-id");
  if (header_trace != nullptr && !header_trace->empty()) {
    trace_id = static_cast<uint64_t>(
        std::strtoull(header_trace->c_str(), nullptr, 10));
  }
#endif
  bool success = false;
  int size     = 0;
#ifdef ENABLE_PERF_REPORT
  auto before_cache_update_time = std::chrono::high_resolution_clock::now();
#endif

  try {
    const std::string& table_name = request->table_name();

    std::string payload_storage;
    const char* payload_data = nullptr;
    int payload_size         = 0;
    if (!ExtractPayloadBytes(
            cntl,
            request->gradients(),
            &payload_storage,
            &payload_data,
            &payload_size)) {
      throw std::runtime_error("UpdateParameter empty gradients payload");
    }

    const ParameterCompressReader* reader =
        reinterpret_cast<const ParameterCompressReader*>(payload_data);
    if (!reader->Valid(payload_size)) {
      throw std::runtime_error("UpdateParameter invalid gradients payload");
    }
    size = reader->item_size();

#ifdef ENABLE_PERF_REPORT
    before_cache_update_time = std::chrono::high_resolution_clock::now();
#endif
    success = cache_ps_->UpdateParameter(table_name, reader, 0);

    FB_LOG_EVERY_MS(INFO, 2000)
        << "UpdateParameter: table=" << table_name << ", keys=" << size;

    reply->set_success(success);
  } catch (const std::exception& e) {
    LOG(ERROR) << "UpdateParameter error: " << e.what();
    reply->set_success(false);
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_server_latency",
         "UpdateParameter",
         "latency_us",
         static_cast<double>(duration));

  double start_us_for_key =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "EmbUpdate|" + std::to_string(static_cast<uint64_t>(start_us_for_key));
  report("op_latency",
         op_latency_key.c_str(),
         "recserver_us",
         static_cast<double>(duration));

  auto backend_update_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - before_cache_update_time)
          .count();
  const uint64_t effective_trace_id =
      trace_id == 0 ? static_cast<uint64_t>(start_us_for_key) : trace_id;
  std::string update_stage_id =
      "brpc_server::EmbUpdate|" + std::to_string(effective_trace_id);
  report("embupdate_stages",
         update_stage_id.c_str(),
         "server_total_us",
         static_cast<double>(duration));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "server_backend_update_us",
         static_cast<double>(backend_update_duration));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "server_request_size",
         static_cast<double>(size));
  report("embupdate_stages",
         update_stage_id.c_str(),
         "server_success",
         success ? 1.0 : 0.0);
#endif
}

void BRPCParameterServiceImpl::InitEmbeddingTable(
    google::protobuf::RpcController* controller,
    const InitEmbeddingTableRequest* request,
    InitEmbeddingTableResponse* reply,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

#ifdef ENABLE_PERF_REPORT
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  try {
    if (request->has_config_payload()) {
      auto payload            = request->config_payload();
      nlohmann::json cfg      = nlohmann::json::parse(payload);
      uint64_t num_embeddings = cfg.value("num_embeddings", 0);
      uint64_t embedding_dim  = cfg.value("embedding_dim", 0);
      FB_LOG_EVERY_MS(INFO, 2000)
          << "InitEmbeddingTable: table=" << request->table_name()
          << ", num_embeddings=" << num_embeddings
          << ", embedding_dim=" << embedding_dim;

      bool init_success = cache_ps_->InitTable(
          request->table_name(), num_embeddings, embedding_dim);
      reply->set_success(init_success);
    } else {
      LOG(WARNING) << "InitEmbeddingTable called without config_payload";
      reply->set_success(false);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "InitEmbeddingTable error: " << e.what();
    reply->set_success(false);
  }

#ifdef ENABLE_PERF_REPORT
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time)
          .count();
  report("ps_server_latency",
         "InitEmbeddingTable",
         "latency_us",
         static_cast<double>(duration));

  double start_us_for_key =
      std::chrono::duration_cast<std::chrono::microseconds>(
          start_time.time_since_epoch())
          .count();
  std::string op_latency_key =
      "InitEmbeddingTable|" +
      std::to_string(static_cast<uint64_t>(start_us_for_key));
  report("op_latency",
         op_latency_key.c_str(),
         "recserver_us",
         static_cast<double>(duration));
#endif
}

class BRPCParameterServer : public BaseParameterServer {
public:
  BRPCParameterServer() = default;

  void Run() {
    // 检查是否配置了多分片
    int num_shards = 1; // 默认单分片
    if (config_["cache_ps"].contains("num_shards")) {
      num_shards = config_["cache_ps"]["num_shards"];
    }

    if (num_shards > 1) {
      // 多服务器启动逻辑
      std::cout
          << "Starting distributed parameter server (bRPC), number of shards: "
          << num_shards << std::endl;

      if (!config_["cache_ps"].contains("servers")) {
        LOG(FATAL) << "配置了 num_shards > 1 但缺少 servers 配置";
        return;
      }

      auto servers = config_["cache_ps"]["servers"];
      if (servers.size() != num_shards) {
        LOG(FATAL) << "servers 配置数量 (" << servers.size()
                   << ") 与 num_shards (" << num_shards << ") 不匹配";
        return;
      }

      std::vector<std::thread> server_threads;

      for (auto& server_config : servers) {
        server_threads.emplace_back([this, server_config]() {
          std::string host = server_config["host"];
          int port         = server_config["port"];
          int shard        = server_config["shard"];

          std::string server_address = host + ":" + std::to_string(port);

          nlohmann::json shard_config = config_["cache_ps"];
          if (shard_config.contains("base_kv_config") &&
              shard_config["base_kv_config"].is_object()) {
            auto& base_kv_config = shard_config["base_kv_config"];
            AppendShardSuffixIfPresent(base_kv_config, "path", shard);
            AppendShardSuffixIfPresent(base_kv_config, "file_path", shard);
            LOG(INFO) << "bRPC shard " << shard
                      << " using base_kv_config: " << base_kv_config.dump();
          }

          auto cache_ps = std::make_unique<CachePS>(shard_config);
          auto service =
              std::make_unique<BRPCParameterServiceImpl>(cache_ps.get());

          brpc::Server server;
          brpc::ServerOptions options;
          options.num_threads = FLAGS_brpc_server_num_threads;

          if (server.AddService(
                  service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
            LOG(ERROR) << "Failed to add service!";
            return;
          }

          if (server.Start(server_address.c_str(), &options) != 0) {
            LOG(ERROR) << "Failed to start bRPC server at " << server_address;
            return;
          }

          std::cout << "bRPC Server shard " << shard << " listening on "
                    << server_address << std::endl;
          server.RunUntilAskedToQuit();
        });
      }

      // 等待所有服务器线程
      for (auto& t : server_threads) {
        t.join();
      }
    } else {
      // 单服务器启动逻辑
      std::cout << "启动单参数服务器 (bRPC)" << std::endl;
      std::string server_address =
          "0.0.0.0:" + std::to_string(FLAGS_brpc_server_port);
      auto cache_ps = std::make_unique<CachePS>(config_["cache_ps"]);
      auto service = std::make_unique<BRPCParameterServiceImpl>(cache_ps.get());

      std::atomic<bool> metrics_running{true};
      std::thread metrics_thread([&service, &metrics_running]() {
        while (metrics_running) {
          std::this_thread::sleep_for(std::chrono::seconds(10));
          service->PrintMetrics();
          service->ResetMetrics();
        }
      });

      brpc::Server server;
      brpc::ServerOptions options;
      options.num_threads = FLAGS_brpc_server_num_threads;

      if (server.AddService(service.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
          0) {
        LOG(ERROR) << "Failed to add service!";
        metrics_running = false;
        if (metrics_thread.joinable()) {
          metrics_thread.join();
        }
        return;
      }

      if (server.Start(server_address.c_str(), &options) != 0) {
        LOG(ERROR) << "Failed to start bRPC server at " << server_address;
        metrics_running = false;
        if (metrics_thread.joinable()) {
          metrics_thread.join();
        }
        return;
      }

      std::cout << "bRPC Server listening on " << server_address << std::endl;
      server.RunUntilAskedToQuit();

      metrics_running = false;
      if (metrics_thread.joinable()) {
        metrics_thread.join();
      }
    }
  }
};

FACTORY_REGISTER(BaseParameterServer, BRPCParameterServer, BRPCParameterServer);

} // namespace recstore

#ifndef RECSTORE_NO_SERVER_MAIN
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::ifstream config_file(FLAGS_brpc_config_path);
  nlohmann::json ex;
  config_file >> ex;

  recstore::BRPCParameterServer ps;
  std::cout << "bRPC Parameter server config: " << ex.dump(2) << std::endl;
  ps.Init(ex);
  ps.Run();

  return 0;
}
#endif
