#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/log.h"
#include "ps/base/shard_client.h"
#include "ps/rdma/control_plane.h"
#include "ps/rdma/rc_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

namespace petps {

class PetPSClient : public recstore::ShardClient {
public:
  explicit PetPSClient(const std::string& host, int port, int shard);
  PetPSClient(
      const std::string& host, int port, int shard, int logical_client_id);
  ~PetPSClient() override;

  void Barrier(const std::string& ss, int k);
  void InitThread();

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   base::RecTensor& values) override;
  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id  = 0,
                   int embedding_dim = 0);

  std::size_t ResponseBufferBytes(std::size_t key_count) const;

  void* GetReceiveBuffer(size_t size);
  // Return a GetReceiveBuffer() result to the pool once the owning batch has
  // fully consumed it (after FinalizeBatchIfNeeded). Buffers not registered in
  // the pool (user-provided direct write targets) are ignored.
  void ReturnGetReceiveBuffer(const float* buffer);
  const float* BorrowGetResultPayload(
      int rpc_id,
      std::size_t* key_count,
      std::size_t* response_bytes,
      std::int32_t* status_code);
  bool QueryRPCFinished(int rpc_id);
  void WaitRPCFinish(int rpc_id);
  void RevokeRPCResource(int rpc_id);

  int PutParameter(const std::vector<uint64_t>& keys,
                   const base::RecTensor& values);
  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const base::RecTensor& values) override;
  int InitEmbeddingTable(const std::string& table_name,
                         std::uint64_t num_embeddings,
                         std::uint64_t embedding_dim,
                         std::uint64_t table_id = 0);
  int InitEmbeddingTable(const std::string& table_name,
                         const recstore::EmbeddingTableConfig& config) override;
  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const base::RecTensor& grads) override;
  void Command(recstore::PSCommand command) override;
  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override;
  bool IsPrefetchDone(uint64_t prefetch_id) override;
  void WaitForPrefetch(uint64_t prefetch_id) override;
  bool
  GetPrefetchResult(uint64_t prefetch_id, base::RecTensor& values) override;
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
  int FakePutParameter(base::ConstArray<uint64_t> keys, float* values);

private:
  struct SlotContext {
    RcClientQpView view;            // Local request/response slot view.
    std::uint64_t next_seq = 1;     // Monotonic request sequence for this lane.
    bool busy              = false; // True while one RPC is in flight.
  };

  struct QpContext {
    QpContext()                            = default;
    QpContext(const QpContext&)            = delete;
    QpContext& operator=(const QpContext&) = delete;
    QpContext(QpContext&& other) noexcept
        : qp_index(other.qp_index),
          slots(std::move(other.slots)),
          dirty(other.dirty.load(std::memory_order_relaxed)) {}

    int qp_index = 0;
    std::vector<SlotContext> slots;
    // P1b whole-lane dirty marker: a lane that lost a completion or whose
    // shard was judged dead never hands its slots to new requests and fails
    // its waiters fast (docs/plans/rdma-reliability-architecture.md §4.1).
    std::atomic<bool> dirty{false};
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
  // Waits for the slot status word, running the P1b slow-path health probe
  // after the configured delay and failing fast with ShardUnreachableError
  // once the lane is dirty.
  std::int32_t
  WaitForStatusWord(int qp_index, const StatusWord* status, std::uint64_t seq);
  // Queries the control plane about this shard at most once per probe
  // interval; when the verdict is Dead the lane is retired and
  // ShardUnreachableError is thrown.
  void MaybeProbeShardHealth(int qp_index, std::int64_t waited_ms);
  // Marks the lane as unusable for good: its slots are never reused and its
  // waiters fail fast. `reason` is wait_timeout, health_dead or cq_<class>.
  void MarkLaneDirty(int qp_index,
                     const char* reason,
                     const RdmaServerHealthSnapshot* snapshot,
                     std::int64_t waited_ms);
  // Retires the lane on a CQ completion error and throws: peer-side failures
  // (retry exhaustion, flushes) surface as ShardUnreachableError, local or
  // rejected operations keep their typed raw-verbs error.
  [[noreturn]] void FailLaneOnCompletionError(
      int qp_index,
      const RawVerbsCompletionError& error,
      std::int64_t waited_ms);
  RdmaControlPlaneClient MakeControlPlaneClient() const;
  SlotContext& SlotAt(int qp_index, int slot_in_qp);
  const SlotContext& SlotAt(int qp_index, int slot_in_qp) const;
  void EnsureThreadInitializedLocked() const;
  bool PendingRpcLocked(int rpc_id, PendingRpc* pending) const;
  bool RequestPayloadFitsSlot(std::size_t payload_bytes) const;
  float* AllocateStatusReceiveBufferLocked();
  void MaybeReportProfile();
  void FillGetDescriptor(
      RequestDescriptor* descriptor,
      std::uint64_t seq,
      std::size_t key_count,
      std::size_t response_bytes,
      std::size_t value_size,
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
  // Requires `mu_` (every call site holds it). Never waits for the RPC here:
  // the status wait must run outside `mu_` because WaitForStatusWord harvests
  // the lane CQ under that mutex.
  int SubmitRpcLocked(
      SlotContext* slot,
      const RequestDescriptor& descriptor,
      const void* payload,
      std::size_t payload_bytes,
      float* recv_buffer,
      std::size_t key_count,
      std::size_t response_bytes);

  struct PrefetchState {
    int rpc_id            = -1;      // In-flight async GET handle.
    float* recv_buffer    = nullptr; // Response buffer for the async GET.
    std::size_t key_count = 0;
    int64_t embedding_dim = 0;
  };

  std::string namespace_token_; // Shared-memory namespace token.
  int shard_              = 0;
  int explicit_client_id_ = -1; // Optional logical client id override.
  int client_id_          = -1; // Logical client id derived from global id.
  RcTransportConfig config_;    // Transport slot sizing and shard config.
  std::unique_ptr<RcShardClientTransport> transport_; // Slot transport owner.
  // Control-plane connection shared by the init handshake and the P1b
  // slow-path health probes.
  std::unique_ptr<RdmaControlPlaneClient> control_plane_;
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
  std::mutex prefetch_mu_;          // Guards the prefetch handle table.
  std::unordered_map<uint64_t, PrefetchState> prefetches_;
  uint64_t next_prefetch_id_ = 1;
  ProfileCounters profile_;
  bool thread_initialized_ = false; // Set after InitThread has run.
  // P1b slow path: shared by all lanes because shard health is per-shard.
  RdmaSlowPathProbeGate health_probe_gate_;
  std::atomic<std::uint64_t> health_probe_failures_{0};
  std::atomic<std::uint64_t> health_probe_dead_verdicts_{0};
};

} // namespace petps
