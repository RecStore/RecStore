#pragma once

#include <brpc/server.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "base/json.h"
#include "ps/base/cache_ps_impl.h"
#include "ps_brpc.pb.h"

namespace recstore {

// bRPC parameter service implementation
class BRPCParameterServiceImpl : public recstoreps_brpc::ParameterService {
public:
  explicit BRPCParameterServiceImpl(CachePS* cache_ps, int shard_id = 0);

  virtual ~BRPCParameterServiceImpl() = default;

  void SparseGradientCommit(google::protobuf::RpcController*,
                            const recstoreps_brpc::SparseGradientCommitRequest*,
                            recstoreps_brpc::SparseGradientCommitResponse*,
                            google::protobuf::Closure*) override;

  void GetParameter(google::protobuf::RpcController* controller,
                    const recstoreps_brpc::GetParameterRequest* request,
                    recstoreps_brpc::GetParameterResponse* response,
                    google::protobuf::Closure* done) override;

  void Command(google::protobuf::RpcController* controller,
               const recstoreps_brpc::CommandRequest* request,
               recstoreps_brpc::CommandResponse* response,
               google::protobuf::Closure* done) override;

  void PutParameter(google::protobuf::RpcController* controller,
                    const recstoreps_brpc::PutParameterRequest* request,
                    recstoreps_brpc::PutParameterResponse* response,
                    google::protobuf::Closure* done) override;

  void UpdateParameter(google::protobuf::RpcController* controller,
                       const recstoreps_brpc::UpdateParameterRequest* request,
                       recstoreps_brpc::UpdateParameterResponse* response,
                       google::protobuf::Closure* done) override;

  void InitEmbeddingTable(
      google::protobuf::RpcController* controller,
      const recstoreps_brpc::InitEmbeddingTableRequest* request,
      recstoreps_brpc::InitEmbeddingTableResponse* response,
      google::protobuf::Closure* done) override;

  void ResetMetrics();
  void PrintMetrics(const std::string& table_name = "brpc_ps_server_metrics",
                    const std::string& unique_id  = "default_server");

private:
  CachePS* cache_ps_;
  int shard_id_;
  std::atomic<uint64_t> total_get_requests_{0};
  std::atomic<uint64_t> total_put_requests_{0};
  std::atomic<uint64_t> total_get_keys_{0};
  std::atomic<uint64_t> total_put_keys_{0};
  std::atomic<uint64_t> total_get_bytes_{0};
  std::atomic<uint64_t> total_put_bytes_{0};
  std::chrono::steady_clock::time_point start_time_;
};

std::vector<nlohmann::json>
SelectBRPCShardConfigs(const nlohmann::json& cache_ps_config,
                       const std::optional<int>& local_shard_id);

} // namespace recstore
