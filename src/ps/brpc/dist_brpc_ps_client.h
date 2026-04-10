#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/hash.h"
#include "base/json.h"
#include "base/log.h"
#include "ps/base/base_client.h"
#include "brpc_ps_client.h"

using json = nlohmann::json;

namespace recstore {

/**
 * @brief 分布式 bRPC 参数服务器客户端
 *
 * 支持多对多的连接模式，通过 hash 函数将 key 路由到对应的服务器。
 * 配置通过 JSON 文件指定服务器列表和 hash 分区方法。
 */
class DistributedBRPCParameterClient : public BasePSClient {
public:
  explicit DistributedBRPCParameterClient(json config);

  ~DistributedBRPCParameterClient();

  // 实现 BasePSClient 的纯虚函数
  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   float* values) override;

  int AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                        float* values) override;

  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;

  void Command(PSCommand command) override;

  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const std::vector<std::vector<float>>* grads) override;
  int UpdateParameterFlat(const std::string& table_name,
                          const base::ConstArray<uint64_t>& keys,
                          const float* grads,
                          int64_t num_rows,
                          int64_t embedding_dim) override;

  int InitEmbeddingTable(const std::string& table_name,
                         const recstore::EmbeddingTableConfig& config) override;

  // Prefetch 接口实现
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool GetPrefetchResult(uint64_t prefetch_id,
                         std::vector<std::vector<float>>* values) override;
  bool GetPrefetchResultFlat(uint64_t prefetch_id,
                             std::vector<float>* values,
                             int64_t* num_rows,
                             int64_t embedding_dim) override;

  // 扩展接口
  bool GetParameter(const base::ConstArray<uint64_t>& keys,
                    std::vector<std::vector<float>>* values);

  bool ClearPS();

  bool LoadFakeData(int64_t n);
  bool DumpFakeData(int64_t n);
  bool LoadCkpt(const std::vector<std::string>& model_config_path,
                const std::vector<std::string>& emb_file_path);

  int shard_count() const { return num_shards_; }

private:
  struct DistPrefetchShardState {
    int shard_id     = -1;
    int client_index = -1;
    std::vector<size_t> original_indices;
    std::vector<uint64_t> child_prefetch_ids;
    std::vector<int> chunk_sizes;
  };

  struct DistPrefetchState {
    size_t total_keys = 0;
    std::vector<DistPrefetchShardState> shard_states;
  };

  int GetShardId(uint64_t key) const;

  void InitializeClients();

  void
  PartitionKeys(const base::ConstArray<uint64_t>& keys,
                std::vector<std::vector<uint64_t>>& partitioned_keys) const;

  void MergeResults(
      const base::ConstArray<uint64_t>& keys,
      const std::vector<std::vector<std::vector<float>>>& partitioned_results,
      std::vector<std::vector<float>>* values) const;

  void MergeResultsToArray(
      const base::ConstArray<uint64_t>& keys,
      const std::vector<std::vector<std::vector<float>>>& partitioned_results,
      float* values) const;

private:
  // 配置信息
  int num_shards_;
  int max_keys_per_request_;
  std::string hash_method_;

  // 服务器配置
  struct ServerConfig {
    std::string host;
    int port;
    int shard;
  };
  std::vector<ServerConfig> server_configs_;

  // bRPC 客户端实例
  std::vector<std::unique_ptr<BRPCParameterClient>> clients_;

  // 分片到客户端的映射
  std::unordered_map<int, int> shard_to_client_index_;

  // 分区缓冲区
  mutable std::vector<std::vector<uint64_t>> partitioned_key_buffer_;
  mutable std::vector<std::vector<size_t>> key_index_mapping_;

  std::mutex prefetch_mu_;
  std::unordered_map<uint64_t, std::shared_ptr<DistPrefetchState>>
      prefetch_states_;
  uint64_t next_prefetch_id_ = 1;
};

} // namespace recstore
