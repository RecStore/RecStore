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
#include "ps/base/shard_client.h"
#include "ps/base/parameters.h"
#include "ps_brpc.pb.h"

using json = nlohmann::json;

// Increased from 2000 to 65536 to reduce per-call protobuf message and
// brpc::Controller allocation overhead. The server max_batch_keys_size is
// also 65536, so this stays within the single-request limit.
static const int MAX_PARAMETER_BATCH_BRPC = 65536;

// Prefetch batch structure for bRPC
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

class BRPCParameterClient : public recstore::ShardClient {
public:
  // New constructor with JSON config
  explicit BRPCParameterClient(json config);

  // Legacy constructor for backward compatibility
  explicit BRPCParameterClient(const std::string& host, int port, int shard);

  ~BRPCParameterClient() {}

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   base::RecTensor& values) override;

  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const base::RecTensor& values) override;

  void Command(recstore::PSCommand command) override;

  inline int shard() const { return shard_; }

  bool ClearPS();

  bool LoadFakeData(int64_t data);

  bool DumpFakeData(int64_t n);

  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path);

  bool PutParameter(const std::vector<uint64_t>& keys,
                    const base::RecTensor& values);

  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const base::RecTensor& grads) override;

  int InitEmbeddingTable(const std::string& table_name,
                         const recstore::EmbeddingTableConfig& config) override;

  // Prefetch API
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         base::RecTensor& values) override;

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
  uint64_t next_prefetch_id_ = 1;
};
