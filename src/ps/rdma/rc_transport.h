#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/array.h"
#include "ps/rdma/raw_verbs_transport.h"
#include "ps/rdma/rdma_deployment.h"
#include "ps/rdma/rdma_protocol.h"

namespace petps {

struct RcTransportConfig {
  int node_id        = -1; // Dense deployment node id for this process.
  int num_servers    = 0;  // Number of server nodes in the deployment.
  int num_os_clients = 0;  // Number of client processes in the deployment.
  int shard_id       = 0;  // Logical shard served by this transport.
  int client_id      = -1; // Logical client id for response slot selection.
  int num_clients    = 1;  // Total client count expected by the server.
  int qps_per_client_per_shard   = 32; // Number of lanes per client per shard.
  int slots_per_qp               = 1; // Logical slots multiplexed on each lane.
  std::size_t request_slot_bytes = 1 << 20;  // Bytes per server request slot.
  std::size_t response_slot_bytes = 1 << 20; // Bytes per client response slot.
  std::string control_plane_host  = "127.0.0.1";
  int control_plane_port          = 25100;
  int control_plane_timeout_ms    = 30000;
  std::string namespace_token =
      "default"; // Shared-memory namespace for this run.
  std::string deployment_id;
  std::uint64_t deployment_epoch = 0;
  std::uint32_t protocol_version = recstore::kRdmaDeploymentProtocolVersion;
  std::string configuration_digest;
  std::string fabric_digest;
  recstore::ResolvedRdmaFabric fabric;
};

struct RcClientQpView {
  int qp_index                  = 0; // Lane index local to the client.
  int slot_in_qp                = 0; // Logical slot index within the lane.
  int slot_index                = 0; // Global request slot index on the server.
  void* request_slot            = nullptr; // Base address of the request slot.
  RequestDescriptor* descriptor = nullptr;
  char* payload          = nullptr; // Request payload region after descriptor.
  CommitWord* commit     = nullptr; // Commit word at end of request slot.
  void* response_slot    = nullptr; // Base address of the client response slot.
  char* response_payload = nullptr; // Response payload region before status.
  StatusWord* status     = nullptr; // Status word at end of response slot.
};

class RcShardClientTransport {
public:
  explicit RcShardClientTransport(const RcTransportConfig& config);
  ~RcShardClientTransport();

  RcShardClientTransport(const RcShardClientTransport&)            = delete;
  RcShardClientTransport& operator=(const RcShardClientTransport&) = delete;

  RcClientQpView OpenQp(int qp_index);
  RcClientQpView OpenSlot(int qp_index, int slot_in_qp);
  void SubmitRequest(const RcClientQpView& view,
                     const RequestDescriptor& descriptor,
                     const void* payload,
                     std::size_t payload_bytes);
  void ClearRequestSlot(const RcClientQpView& view);
  std::size_t request_slot_bytes() const { return config_.request_slot_bytes; }
  std::size_t response_slot_bytes() const {
    return config_.response_slot_bytes;
  }
  const RcTransportConfig& config() const { return config_; }

private:
  struct Lane {
    std::unique_ptr<RawVerbsTransport>
        verbs; // RC QP and registered memory for this lane.
    void* lane_base =
        nullptr; // Local registered region for all slots in this lane.
    std::vector<std::uint8_t> submit_completion_pending;
    std::vector<std::uint8_t> submit_completion_ready;
  };

  Lane& LaneAt(int qp_index);
  const Lane& LaneAt(int qp_index) const;

  RcTransportConfig config_;
  int server_node_id_ = 0; // Global node id of the target shard server.
  std::vector<Lane> lanes_;
};

class RcShardServerTransport {
public:
  explicit RcShardServerTransport(const RcTransportConfig& config);
  ~RcShardServerTransport();

  RcShardServerTransport(const RcShardServerTransport&)            = delete;
  RcShardServerTransport& operator=(const RcShardServerTransport&) = delete;

  int TotalSlots() const;
  void RegisterLocalMemoryRegion(void* base, std::size_t bytes);
  int SlotIndex(int client_id, int qp_index, int slot_in_qp) const;
  void DecodeSlotIndex(
      int slot_index, int* client_id, int* qp_index, int* slot_in_qp) const;
  void* RequestSlot(int slot_index) const;
  RequestDescriptor* RequestDescriptorAt(int slot_index) const;
  char* RequestPayloadAt(int slot_index) const;
  CommitWord* RequestCommitAt(int slot_index) const;

  struct ResponseView {
    void* slot         = nullptr; // Base address of the client response slot.
    char* payload      = nullptr; // Response payload region.
    StatusWord* status = nullptr; // Final completion word for this response.
  };

  ResponseView OpenClientResponse(int client_id, int qp_index, int slot_in_qp);
  void CompleteResponse(int client_id,
                        int qp_index,
                        int slot_in_qp,
                        const ResponseView& response,
                        std::uint64_t seq);
  void WriteResponsePayloadSg(
      int client_id,
      int qp_index,
      int slot_in_qp,
      base::ConstArray<RawVerbsSge> sges,
      std::uint64_t response_offset,
      std::uint64_t bytes);
  void CompleteResponseStatusOnly(
      int client_id,
      int qp_index,
      int slot_in_qp,
      const ResponseView& response,
      std::uint64_t seq);
  const RcTransportConfig& config() const { return config_; }

private:
  struct Lane {
    std::unique_ptr<RawVerbsTransport>
        verbs;                     // RC QP and registered memory for this lane.
    void* request_slots = nullptr; // Local registered slots for all clients.
    std::vector<void*> response_staging; // Per-client-per-slot registered
                                         // response staging slots.
    std::vector<std::uint8_t> response_completion_pending;
    std::vector<std::uint8_t> response_completion_ready;
  };

  Lane& LaneAt(int client_id, int qp_index);
  const Lane& LaneAt(int client_id, int qp_index) const;

  RcTransportConfig config_; // Transport sizing and namespace config.
  std::vector<Lane> lanes_;  // One verbs RC lane per qp_index.
};

} // namespace petps
