#pragma once

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

#include "base/array.h"
#include "base/flatc.h"
#include "base/json.h"
#include "base/tensor.h"
#include "ps/base/base_client.h"
#include "ps/base/parameters.h"
#include "ps_brpc.pb.h"

using json = nlohmann::json;

static const int MAX_PARAMETER_BATCH_BRPC = 2000;

// bRPC 版本的 Prefetch 批次结构
struct BrpcPrefetchBatch {
  BrpcPrefetchBatch(int request_num) {
    batch_size_ = request_num;
    key_sizes_.resize(request_num);
    responses_.resize(request_num);
    controllers_.resize(request_num);
    completed_count_ = 0;
  }

  BrpcPrefetchBatch(BrpcPrefetchBatch&& other) noexcept
      : key_sizes_(std::move(other.key_sizes_)),
        responses_(std::move(other.responses_)),
        controllers_(std::move(other.controllers_)),
        batch_size_(other.batch_size_),
        completed_count_(other.completed_count_.load()) {
    other.batch_size_ = 0;
  }

  BrpcPrefetchBatch(const BrpcPrefetchBatch&)            = delete;
  BrpcPrefetchBatch& operator=(const BrpcPrefetchBatch&) = delete;

  std::vector<int> key_sizes_;
  std::vector<recstoreps_brpc::GetParameterResponse> responses_;
  std::vector<std::unique_ptr<brpc::Controller>> controllers_;
  int batch_size_;
  std::atomic<int> completed_count_;
};

struct BrpcPrewriteBatch {
  BrpcPrewriteBatch(int request_num) {
    batch_size_ = request_num;
    key_sizes_.resize(request_num);
    requests_.resize(request_num);
    responses_.resize(request_num);
    controllers_.resize(request_num);
    completed_count_ = 0;
  }

  BrpcPrewriteBatch(BrpcPrewriteBatch&& other) noexcept
      : key_sizes_(std::move(other.key_sizes_)),
        requests_(std::move(other.requests_)),
        responses_(std::move(other.responses_)),
        controllers_(std::move(other.controllers_)),
        batch_size_(other.batch_size_),
        completed_count_(other.completed_count_.load()) {
    other.batch_size_ = 0;
  }

  BrpcPrewriteBatch(const BrpcPrewriteBatch&)            = delete;
  BrpcPrewriteBatch& operator=(const BrpcPrewriteBatch&) = delete;

  std::vector<int> key_sizes_;
  std::vector<recstoreps_brpc::PutParameterRequest> requests_;
  std::vector<recstoreps_brpc::PutParameterResponse> responses_;
  std::vector<std::unique_ptr<brpc::Controller>> controllers_;
  int batch_size_;
  std::atomic<int> completed_count_;
};

class BRPCParameterClient : public recstore::BasePSClient {
public:
  // 新的构造函数，接收 json 配置参数
  explicit BRPCParameterClient(json config);

  // 保留原有的构造函数以保持向后兼容
  explicit BRPCParameterClient(const std::string& host, int port, int shard);

  ~BRPCParameterClient() {}

  // 实现 BasePSClient 的纯虚函数
  virtual int
  GetParameter(const base::ConstArray<uint64_t>& keys, float* values) override;

  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                        float* values) override;

  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;

  void Command(recstore::PSCommand command) override;

  // 原有的接口方法
  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   std::vector<std::vector<float>>* values);

  inline int shard() const { return shard_; }

  bool ClearPS();

  bool LoadFakeData(int64_t data);

  bool DumpFakeData(int64_t n);

  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path);

  bool PutParameter(const std::vector<uint64_t>& keys,
                    const std::vector<std::vector<float>>& values);

  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const std::vector<std::vector<float>>* grads);
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim) override;

  int InitEmbeddingTable(const std::string& table_name,
                         const recstore::EmbeddingTableConfig& config);

  // Prefetch 接口
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys);
  bool IsPrefetchDone(uint64_t prefetch_id);
  void WaitForPrefetch(uint64_t prefetch_id);
  bool GetPrefetchResult(uint64_t prefetch_id,
                         std::vector<std::vector<float>>* values);
  bool GetPrefetchResultFlat(uint64_t prefetch_id,
                             std::vector<float>* values,
                             int64_t* num_rows,
                             int64_t embedding_dim) override;

  virtual uint64_t
  EmbWriteAsync(const base::RecTensor& keys, const base::RecTensor& values);
  virtual bool IsWriteDone(uint64_t write_id);
  virtual void WaitForWrite(uint64_t write_id);

protected:
  bool Initialize();

  std::string host_;
  int port_;
  int shard_;
  int timeout_ms_;
  int max_retry_;

  // bRPC channel
  std::shared_ptr<brpc::Channel> channel_;

  std::vector<float> cache_;
  std::vector<int32_t> offset_;

private:
  std::unordered_map<uint64_t, struct BrpcPrefetchBatch> prefetch_batches_;
  std::unordered_map<uint64_t, struct BrpcPrewriteBatch> prewrite_batches_;
  uint64_t next_prefetch_id_ = 1;
  uint64_t next_prewrite_id_ = 1;
};
