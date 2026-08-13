#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"
#include "ps/rdma/base_client.h"
#include "ps/rdma/rc_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

namespace petps {

class PetPSClient : public BaseParameterClient {
public:
  explicit PetPSClient(const std::string& host, int port, int shard);
  PetPSClient(
      const std::string& host, int port, int shard, int logical_client_id);
  ~PetPSClient() override;

  void Barrier(const std::string& ss, int k) override;
  void InitThread() override;

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;
  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  std::size_t ResponseBufferBytes(std::size_t key_count) const;

  void* GetReceiveBuffer(size_t size) override;
  // Return a GetReceiveBuffer() result to the pool once the owning batch has
  // fully consumed it (after FinalizeBatchIfNeeded). Buffers not registered in
  // the pool (user-provided direct write targets) are ignored.
  void ReturnGetReceiveBuffer(const float* buffer);
  const float* BorrowGetResultPayload(
      int rpc_id,
      std::size_t* key_count,
      std::size_t* response_bytes,
      std::int32_t* status_code);
  bool QueryRPCFinished(int rpc_id) override;
  void WaitRPCFinish(int rpc_id) override;
  void RevokeRPCResource(int rpc_id) override;

  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;
  int InitEmbeddingTable(const std::string& table_name,
                         std::uint64_t num_embeddings,
                         std::uint64_t embedding_dim) override;
  int UpdateParameter(const std::string& table_name,
                      base::ConstArray<uint64_t> keys,
                      const std::vector<std::vector<float>>* grads) override;
  int SubmitUpdateParameterFlat(const std::string& table_name,
                                base::ConstArray<uint64_t> keys,
                                const float* grads,
                                std::size_t embedding_dim);
  int SubmitUpdateParameterFlatGather(
      const std::string& table_name,
      const std::uint64_t* keys,
      const float* grads,
      std::size_t num_rows,
      std::size_t embedding_dim,
      const std::size_t* row_indices,
      std::size_t row_count);
  int WaitUpdateParameter(int rpc_id);
  int FakePutParameter(base::ConstArray<uint64_t> keys, float* values) override;

private:
  struct SlotContext {
    RcClientQpView view;            // Local request/response slot view.
    std::uint64_t next_seq = 1;     // Monotonic request sequence for this lane.
    bool busy              = false; // True while one RPC is in flight.
  };

  struct QpContext {
    int qp_index = 0;
    std::vector<SlotContext> slots;
  };

  struct PendingRpc {
    int qp_index               = -1;      // Lane used for this pending RPC.
    int slot_in_qp             = -1;      // Logical slot within the lane.
    int slot_index             = -1;      // Global request slot index.
    std::uint64_t seq          = 0;       // Sequence number written into slot.
    float* recv_buffer         = nullptr; // User-visible response buffer.
    std::size_t key_count      = 0;       // Number of keys in this RPC.
    std::size_t response_bytes = 0;       // Dense response payload bytes.
  };

  struct ProfileCounters {
    std::atomic<std::uint64_t> acquire_qp_count{0};
    std::atomic<std::uint64_t> acquire_qp_failures{0};
    std::atomic<std::uint64_t> submit_rpc_count{0};
    std::atomic<std::uint64_t> wait_rpc_count{0};
    std::atomic<std::uint64_t> revoke_rpc_count{0};
    std::atomic<std::uint64_t> submit_request_ns{0};
    std::atomic<std::uint64_t> wait_status_ns{0};
    std::atomic<std::uint64_t> copy_response_ns{0};
    std::atomic<std::uint64_t> revoke_resource_ns{0};
    std::atomic<std::uint64_t> response_bytes_copied{0};
    std::atomic<std::uint64_t> pending_rpc_peak{0};
    std::atomic<std::uint64_t> pending_rpc_samples{0};
    std::atomic<std::uint64_t> pending_rpc_sum{0};
    std::atomic<std::uint64_t> pending_rpc_last{0};
    std::atomic<std::uint64_t> next_report_ns{0};
  };

  void InitializeTransport();
  struct SlotHandle {
    int qp_index   = -1;
    int slot_in_qp = -1;
  };
  SlotHandle AcquireIdleSlot();
  SlotContext& SlotAt(int qp_index, int slot_in_qp);
  const SlotContext& SlotAt(int qp_index, int slot_in_qp) const;
  void EnsureThreadInitializedLocked() const;
  bool PendingRpcLocked(int rpc_id, PendingRpc* pending) const;
  bool RequestPayloadFitsSlot(std::size_t payload_bytes) const;
  float* AllocateStatusReceiveBufferLocked();
  void MaybeReportProfile();
  void FillGetDescriptor(RequestDescriptor* descriptor,
                         std::uint64_t seq,
                         std::size_t key_count,
                         std::size_t response_bytes,
                         const RcClientQpView& view) const;
  void FillPutDescriptor(RequestDescriptor* descriptor,
                         std::uint64_t seq,
                         std::size_t key_count,
                         std::size_t payload_bytes,
                         const RcClientQpView& view) const;
  void FillUpdateDescriptor(
      RequestDescriptor* descriptor,
      std::uint64_t seq,
      std::size_t key_count,
      std::size_t payload_bytes,
      const std::string& table_name,
      const RcClientQpView& view) const;
  void FillUpdateFlatDescriptor(
      RequestDescriptor* descriptor,
      std::uint64_t seq,
      std::size_t key_count,
      std::size_t payload_bytes,
      std::size_t embedding_dim,
      const std::string& table_name,
      const RcClientQpView& view) const;
  void FillInitTableDescriptor(RequestDescriptor* descriptor,
                               std::uint64_t seq,
                               const std::string& table_name,
                               const RcClientQpView& view) const;
  int SubmitRpcLocked(
      SlotContext* slot,
      const RequestDescriptor& descriptor,
      const void* payload,
      std::size_t payload_bytes,
      float* recv_buffer,
      std::size_t key_count,
      std::size_t response_bytes,
      bool is_async);

  std::string namespace_token_; // Shared-memory namespace token.
  int explicit_client_id_ = -1; // Optional logical client id override.
  int client_id_          = -1; // Logical client id derived from global id.
  RcTransportConfig config_;    // Transport slot sizing and shard config.
  std::unique_ptr<RcShardClientTransport> transport_; // Slot transport owner.
  std::vector<QpContext> qps_; // One context per client-side QP lane.
  std::vector<std::vector<char>>
      receive_buffers_; // All heap-backed response buffers ever allocated.
                       // Element data() pointers stay stable for the buffer
                       // lifetime, so they can be pooled by index.
  std::vector<std::size_t>
      receive_buffer_free_; // Indices into receive_buffers_ idle for reuse
                            // (LIFO). Avoids a fresh mmap + zero-fill +
                            // page-fault every batch on the submit path.
  std::unordered_map<const char*, std::size_t>
      receive_buffer_index_; // data() pointer -> receive_buffers_ index, used
                             // to return a buffer to the pool at revoke time.
  std::unordered_map<int, PendingRpc> pending_rpcs_; // In-flight RPC table.
  std::mutex mu_; // Guards transport setup and pending RPC state.
  std::atomic<int> next_rpc_id_{1}; // Monotonic RPC handle generator.
  ProfileCounters profile_;
  bool thread_initialized_ = false; // Set after InitThread has run.
};

FACTORY_REGISTER(BaseParameterClient,
                 PetPSClient,
                 PetPSClient,
                 const std::string&,
                 int,
                 int);

} // namespace petps
