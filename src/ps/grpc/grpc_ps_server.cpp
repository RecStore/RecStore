#include <folly/executors/CPUThreadPoolExecutor.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "base/array.h"
#include "base/base.h"
#include "base/flatc.h"
#include "base/timer.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"
#include "ps/base/base_ps_server.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/base/parameters.h"
#include "recstore_config.h"

#ifdef ENABLE_PERF_REPORT
#  include <chrono>
#  include <cstdlib>

#  include "base/report/report_client.h"
#else
#  include "../report_client.h"
#endif

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
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

DEFINE_string(config_path,
              RECSTORE_PATH "/recstore_config.json",
              "config file path");

class ParameterServiceImpl final
    : public recstoreps::ParameterService::Service {
public:
  ParameterServiceImpl(CachePS* cache_ps) {
    cache_ps_   = cache_ps;
    start_time_ = std::chrono::steady_clock::now();
  }
  void ResetMetrics() {
    total_get_requests_ = 0;
    total_put_requests_ = 0;
    total_get_keys_     = 0;
    total_put_keys_     = 0;
    total_get_bytes_    = 0;
    total_put_bytes_    = 0;
    start_time_         = std::chrono::steady_clock::now();
  }
  void PrintMetrics(const std::string& table_name = "grpc_ps_server_metrics",
                    const std::string& unique_id  = "default_server") {
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

private:
  Status GetParameter(ServerContext* context,
                      const GetParameterRequest* request,
                      GetParameterResponse* reply) override {
#ifdef ENABLE_PERF_REPORT
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    base::ConstArray<uint64_t> keys_array(request->keys());
    bool isPerf = request->has_perf() && request->perf();
    if (isPerf) {
      xmh::PerfCounter::Record("PS Get Keys", keys_array.Size());
    }
    xmh::Timer timer_ps_get_req("PS GetParameter Req");
    ParameterCompressor compressor(std::numeric_limits<int>::max());
    std::vector<std::string> blocks;
    FB_LOG_EVERY_MS(INFO, 1000)
        << "[PS] Getting " << keys_array.Size() << " keys";
    int total_dim = 0;
#ifdef ENABLE_PERF_REPORT
    auto cache_start_time = std::chrono::high_resolution_clock::now();
#endif
    std::vector<ParameterPack> packs;
    packs.reserve(keys_array.Size());
    cache_ps_->GetParameterRun2Completion(keys_array, packs, 0);

    for (auto& pack : packs) {
      compressor.AddItem(pack, &blocks);
      total_dim += pack.dim;
    }
#ifdef ENABLE_PERF_REPORT
    auto cache_end_time = std::chrono::high_resolution_clock::now();
    auto cache_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            cache_end_time - cache_start_time)
            .count();
    double start_us_for_cache =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    std::string report_id_for_cache =
        "grpc_server::GetParameter|" +
        std::to_string(static_cast<uint64_t>(start_us_for_cache));
    report("embread_stages",
           report_id_for_cache.c_str(),
           "cache_lookup_us",
           static_cast<double>(cache_duration));
#endif

    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    reply->mutable_parameter_value()->swap(blocks[0]);
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
    double start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();

    std::string report_id = "grpc_server::GetParameter|" +
                            std::to_string(static_cast<uint64_t>(start_us));

    std::string op_latency_key =
        "EmbRead|" + std::to_string(static_cast<uint64_t>(start_us));
    report("op_latency",
           op_latency_key.c_str(),
           "recserver_us",
           static_cast<double>(duration));

    report("embread_stages",
           report_id.c_str(),
           "duration_us",
           static_cast<double>(duration));

    report("embread_stages",
           report_id.c_str(),
           "request_size",
           static_cast<double>(keys_array.Size()));

    std::string unique_id =
        "embread_debug|" + std::to_string(static_cast<uint64_t>(start_us));
    FlameGraphData grpc_server_data = {
        "grpc_ps_server::GetParameter",
        start_us,
        2, // level
        static_cast<double>(duration),
        static_cast<double>(duration)};
    report_flame_graph(
        "emb_read_flame_map", unique_id.c_str(), grpc_server_data);
#endif

    return Status::OK;
  }

  Status Command(ServerContext* context,
                 const CommandRequest* request,
                 CommandResponse* reply) override {
    if (request->command() == PSCommand::CLEAR_PS) {
      LOG(WARNING) << "[PS Command] Clear All";
      cache_ps_->Clear();
    } else if (request->command() == PSCommand::RELOAD_PS) {
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
    } else if (request->command() == PSCommand::LOAD_FAKE_DATA) {
      if (request->arg1_size() != 1 ||
          static_cast<size_t>(request->arg1(0).size()) != sizeof(int64_t)) {
        LOG(ERROR) << "LOAD_FAKE_DATA: arg1 must be one " << sizeof(int64_t)
                   << "-byte int64_t (requested reply payload size)";
        return Status(grpc::StatusCode::INVALID_ARGUMENT,
                      "LOAD_FAKE_DATA invalid arg1 size");
      }
      int64_t payload_bytes = 0;
      std::memcpy(&payload_bytes, request->arg1(0).data(), sizeof(int64_t));
      if (payload_bytes < 0) {
        LOG(ERROR) << "LOAD_FAKE_DATA: payload_bytes must be non-negative, got "
                   << payload_bytes;
        return Status(grpc::StatusCode::INVALID_ARGUMENT,
                      "payload_bytes must be non-negative");
      }
      constexpr int64_t kMaxReplyPayload = 16 * 1024 * 1024;
      if (payload_bytes > kMaxReplyPayload) {
        LOG(ERROR) << "LOAD_FAKE_DATA: payload_bytes " << payload_bytes
                   << " exceeds cap " << kMaxReplyPayload;
        return Status(grpc::StatusCode::INVALID_ARGUMENT, "payload too large");
      }
      std::string fake(static_cast<size_t>(payload_bytes), '\xab');
      reply->set_reply(std::move(fake));
    } else if (request->command() == PSCommand::DUMP_FAKE_DATA) {
      if (request->arg1_size() != 1 ||
          static_cast<size_t>(request->arg1(0).size()) != sizeof(int64_t)) {
        LOG(ERROR) << "DUMP_FAKE_DATA: arg1 must be one " << sizeof(int64_t)
                   << "-byte int64_t (payload bytes n)";
        return Status(grpc::StatusCode::INVALID_ARGUMENT,
                      "DUMP_FAKE_DATA invalid arg1 size");
      }
      int64_t n = 0;
      std::memcpy(&n, request->arg1(0).data(), sizeof(int64_t));
      if (n <= 0) {
        LOG(ERROR) << "DUMP_FAKE_DATA: n must be positive";
        return Status(grpc::StatusCode::INVALID_ARGUMENT,
                      "DUMP_FAKE_DATA n must be positive");
      }
      if (n % static_cast<int64_t>(sizeof(float)) != 0) {
        LOG(ERROR) << "DUMP_FAKE_DATA: n must be a multiple of "
                   << sizeof(float);
        return Status(grpc::StatusCode::INVALID_ARGUMENT,
                      "DUMP_FAKE_DATA n must be multiple of sizeof(float)");
      }
      constexpr int64_t kMaxDumpBytes = 64 * 1024 * 1024;
      if (n > kMaxDumpBytes) {
        LOG(ERROR) << "DUMP_FAKE_DATA: n exceeds cap " << kMaxDumpBytes;
        return Status(
            grpc::StatusCode::INVALID_ARGUMENT, "DUMP_FAKE_DATA n exceeds cap");
      }
      // Receive fake data payload (used for write bandwidth benchmarking)
      reply->set_reply("ok");
    } else {
      LOG(FATAL) << "invalid command";
    }
    return Status::OK;
  }

  Status PutParameter(ServerContext* context,
                      const PutParameterRequest* request,
                      PutParameterResponse* reply) override {
#ifdef ENABLE_PERF_REPORT
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    const ParameterCompressReader* reader =
        reinterpret_cast<const ParameterCompressReader*>(
            request->parameter_value().data());
    int size = reader->item_size();
    LOG(INFO) << "[PS] PutParameter: " << size << " keys";
    uint64_t total_bytes = 0;

    for (int i = 0; i < size; i++) {
      cache_ps_->PutSingleParameter(reader->item(i), 0);
      total_bytes += reader->item(i)->dim * sizeof(float);
    }
    LOG(INFO) << "[PS] PutParameter done: " << size << " keys";
    total_put_requests_++;
    total_put_keys_ += size;
    total_put_bytes_ += total_bytes;

#ifdef ENABLE_PERF_REPORT
    auto end_time = std::chrono::high_resolution_clock::now();
    double start_us_for_key =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();
    std::string op_latency_key =
        "EmbWrite|" + std::to_string(static_cast<uint64_t>(start_us_for_key));
    report("op_latency",
           op_latency_key.c_str(),
           "recserver_us",
           static_cast<double>(duration));
#endif

    return Status::OK;
  }

  Status UpdateParameter(ServerContext* context,
                         const UpdateParameterRequest* request,
                         UpdateParameterResponse* reply) override {
#ifdef ENABLE_PERF_REPORT
    auto start_time   = std::chrono::high_resolution_clock::now();
    uint64_t trace_id = 0;
    const auto trace_it =
        context->client_metadata().find("x-recstore-trace-id");
    if (trace_it != context->client_metadata().end()) {
      std::string trace_id_str(
          trace_it->second.data(), trace_it->second.length());
      trace_id = static_cast<uint64_t>(
          std::strtoull(trace_id_str.c_str(), nullptr, 10));
    }
#endif
    bool success = false;
    int size     = 0;
    std::string table_name;
#ifdef ENABLE_PERF_REPORT
    auto before_cache_update_time = std::chrono::high_resolution_clock::now();
#endif
    try {
      table_name = request->table_name();
      const ParameterCompressReader* reader =
          reinterpret_cast<const ParameterCompressReader*>(
              request->gradients().data());
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
    double start_us_for_key =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
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
        "grpc_server::EmbUpdate|" + std::to_string(effective_trace_id);
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

    return Status::OK;
  }

  Status InitEmbeddingTable(ServerContext* context,
                            const InitEmbeddingTableRequest* request,
                            InitEmbeddingTableResponse* reply) override {
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
    double start_us_for_key =
        std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch())
            .count();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time)
            .count();
    std::string op_latency_key =
        "InitEmbeddingTable|" +
        std::to_string(static_cast<uint64_t>(start_us_for_key));
    report("op_latency",
           op_latency_key.c_str(),
           "recserver_us",
           static_cast<double>(duration));
#endif

    return Status::OK;
  }

private:
  CachePS* cache_ps_;
  std::atomic<uint64_t> total_get_requests_{0};
  std::atomic<uint64_t> total_put_requests_{0};
  std::atomic<uint64_t> total_get_keys_{0};
  std::atomic<uint64_t> total_put_keys_{0};
  std::atomic<uint64_t> total_get_bytes_{0};
  std::atomic<uint64_t> total_put_bytes_{0};
  std::chrono::steady_clock::time_point start_time_;
};

namespace recstore {
class GRPCParameterServer : public BaseParameterServer {
public:
  GRPCParameterServer() = default;

  void Run() {
    // 检查是否配置了多分片
    int num_shards = 1; // 默认单分片
    if (config_["cache_ps"].contains("num_shards")) {
      num_shards = config_["cache_ps"]["num_shards"];
    }

    if (num_shards > 1) {
      // 多服务器启动逻辑
      std::cout << "Starting distributed parameter server (gRPC), number "
                   "of shards: "
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
          try {
            std::string host = server_config["host"];
            int port         = server_config["port"];
            int shard        = server_config["shard"];

            std::string server_address = host + ":" + std::to_string(port);

            nlohmann::json shard_config = config_["cache_ps"];
            if (shard_config.contains("base_kv_config") &&
                shard_config["base_kv_config"].contains("path")) {
              std::string original_path =
                  shard_config["base_kv_config"]["path"];
              shard_config["base_kv_config"]["path"] =
                  original_path + "_" + std::to_string(shard);
              LOG(INFO) << "Shard " << shard << " using data path: "
                        << shard_config["base_kv_config"]["path"];
            }

            auto cache_ps = std::make_unique<CachePS>(shard_config);
            ParameterServiceImpl service(cache_ps.get());

            grpc::EnableDefaultHealthCheckService(true);
            grpc::reflection::InitProtoReflectionServerBuilderPlugin();
            ServerBuilder builder;
            builder.AddListeningPort(
                server_address, grpc::InsecureServerCredentials());
            builder.RegisterService(&service);
            builder.SetMaxReceiveMessageSize(-1); // Unlimited
            builder.SetMaxSendMessageSize(-1);    // Unlimited
            std::unique_ptr<Server> server(builder.BuildAndStart());

            if (!server) {
              std::string err_msg = fmt::format(
                  "FATAL: Failed to start gRPC server shard {} "
                  "on {}. "
                  "Port might be in use or invalid "
                  "configuration. "
                  "Check if port {} is already occupied.",
                  shard,
                  server_address,
                  port);
              std::cerr << err_msg << std::endl;
              LOG(FATAL) << err_msg;
              return;
            }
            std::cout << "Server shard " << shard << " listening on "
                      << server_address << std::endl;
            server->Wait();
          } catch (const std::exception& e) {
            std::cerr << "FATAL: Uncaught exception in shard thread: "
                      << e.what() << std::endl;
            LOG(FATAL) << "Uncaught exception in shard thread: " << e.what();
          } catch (...) {
            std::cerr << "FATAL: Unknown exception in shard thread"
                      << std::endl;
            LOG(FATAL) << "Unknown exception in shard thread";
          }
        });
      }

      // 等待所有服务器线程
      for (auto& t : server_threads) {
        t.join();
      }
    } else {
      // 单服务器启动逻辑
      std::cout << "启动单参数服务器" << std::endl;
      std::string server_address("0.0.0.0:15000");
      auto cache_ps = std::make_unique<CachePS>(config_["cache_ps"]);
      ParameterServiceImpl service(cache_ps.get());

      std::atomic<bool> metrics_running{true};
      std::thread metrics_thread([&service, &metrics_running]() {
        while (metrics_running) {
          std::this_thread::sleep_for(std::chrono::seconds(10));
          service.PrintMetrics();
          service.ResetMetrics();
        }
      });

      grpc::EnableDefaultHealthCheckService(true);
      grpc::reflection::InitProtoReflectionServerBuilderPlugin();
      ServerBuilder builder;
      builder.AddListeningPort(
          server_address, grpc::InsecureServerCredentials());
      builder.RegisterService(&service);
      builder.SetMaxReceiveMessageSize(-1); // Unlimited
      builder.SetMaxSendMessageSize(-1);    // Unlimited
      std::unique_ptr<Server> server(builder.BuildAndStart());
      std::cerr << "sever built succesfully" << std::endl;
      if (!server) {
        std::string err_msg = fmt::format(
            "FATAL: Failed to start gRPC server on {}. "
            "Port might be in use or invalid configuration.",
            server_address);
        std::cerr << err_msg << std::endl;
        LOG(FATAL) << err_msg;
        metrics_running = false;
        if (metrics_thread.joinable()) {
          metrics_thread.join();
        }
        return;
      }
      std::cout << "Server listening on " << server_address << std::endl;
      server->Wait();

      metrics_running = false;
      if (metrics_thread.joinable()) {
        metrics_thread.join();
      }
    }
  }
};

FACTORY_REGISTER(BaseParameterServer, GRPCParameterServer, GRPCParameterServer);

} // namespace recstore

#ifndef RECSTORE_NO_SERVER_MAIN
int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);
  std::ifstream config_file(FLAGS_config_path);
  nlohmann::json ex;
  config_file >> ex;
  recstore::GRPCParameterServer ps;
  std::cout << "Parameter server config: " << ex.dump(2) << std::endl;
  ps.Init(ex);
  ps.Run();
  return 0;
}
#endif
