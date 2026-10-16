#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <infiniband/verbs.h>

#include "base/array.h"
#include "ps/rdma/global_address.h"
#include "ps/rdma/rdma_deployment.h"

namespace petps {

class RdmaControlPlaneClient;

struct RawVerbsConfig {
  int global_id    = 0;
  int local_lane   = 0;
  int remote_lane  = 0;
  int only_node_id = -1; // Optional single peer node filter.
  int num_servers  = 1;
  int num_clients  = 1;
  std::string device_name;
  std::uint8_t port_num                = 0;
  recstore::RdmaFabricMode fabric_mode = recstore::RdmaFabricMode::kIb;
  int gid_index                        = -1;
  std::uint8_t hop_limit               = 0;
  std::uint8_t traffic_class           = 0;
  std::uint32_t flow_label             = 0;
  std::string deployment_id;
  std::uint64_t deployment_epoch = 0;
  std::uint32_t protocol_version = recstore::kRdmaDeploymentProtocolVersion;
  std::string configuration_digest;
  std::string fabric_digest;
  std::uint32_t max_inline_data         = 0;
  bool connect_to_servers               = true;
  bool connect_to_clients               = true;
  std::size_t local_region_bytes        = 128 * 1024 * 1024;
  std::uint64_t local_base_addr         = 0;
  std::string control_plane_host        = "127.0.0.1";
  int control_plane_port                = 25100;
  int control_plane_timeout_ms          = 30000;
  std::uint64_t allocation_start_offset = 0;
  std::uint64_t reserved_region_offset  = 0;
  std::uint64_t reserved_region_bytes   = 0;
};

inline std::string RawVerbsMetaKey(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane) {
  return "raw-verbs-meta-" + std::to_string(publisher_node_id) + "-lane-" +
         std::to_string(publisher_lane) + "-to-" +
         std::to_string(receiver_node_id) + "-lane-" +
         std::to_string(receiver_lane);
}

inline bool
ShouldRawVerbsConnectToNode(const RawVerbsConfig& config, int node_id) {
  if (node_id == config.global_id) {
    return false;
  }
  if (node_id < 0 || node_id >= config.num_servers + config.num_clients) {
    return false;
  }
  if (config.only_node_id >= 0 && node_id != config.only_node_id) {
    return false;
  }
  const bool is_server = node_id < config.num_servers;
  return is_server ? config.connect_to_servers : config.connect_to_clients;
}

struct RawVerbsReservedRegion {
  std::uint64_t offset = 0;
  std::uint64_t bytes  = 0;
};

class RawVerbsRegionAllocator {
public:
  explicit RawVerbsRegionAllocator(std::uint64_t limit_bytes,
                                   std::uint64_t allocation_start_offset = 0)
      : limit_bytes_(limit_bytes), allocation_offset_(allocation_start_offset) {
    if (allocation_start_offset > limit_bytes) {
      throw std::runtime_error(
          "raw verbs allocation start outside local memory");
    }
  }

  void SetReservedRegion(RawVerbsReservedRegion reserved) {
    if (reserved.bytes != 0) {
      const std::uint64_t reserved_end = reserved.offset + reserved.bytes;
      if (reserved_end < reserved.offset || reserved_end > limit_bytes_) {
        throw std::runtime_error(
            "raw verbs reserved region outside local memory");
      }
    }
    reserved_                        = reserved;
    const std::uint64_t reserved_end = reserved.offset + reserved.bytes;
    if (reserved.bytes != 0 && allocation_offset_ >= reserved.offset &&
        allocation_offset_ < reserved_end) {
      allocation_offset_ = reserved_end;
    }
  }

  std::uint64_t Allocate(std::size_t bytes) {
    const std::uint64_t aligned        = Align(bytes);
    std::uint64_t offset               = allocation_offset_;
    const std::uint64_t reserved_begin = reserved_.offset;
    const std::uint64_t reserved_end   = reserved_.offset + reserved_.bytes;
    if (reserved_.bytes != 0) {
      if (offset >= reserved_begin && offset < reserved_end) {
        offset = reserved_end;
      } else if (offset < reserved_begin && offset + aligned > reserved_begin) {
        offset = reserved_end;
      }
    }
    if (offset + aligned < offset || offset + aligned > limit_bytes_) {
      throw std::runtime_error("raw verbs registered region exhausted");
    }
    allocation_offset_ = offset + aligned;
    return offset;
  }

  std::uint64_t Checkpoint() const { return allocation_offset_; }

  void Restore(std::uint64_t checkpoint) {
    if (checkpoint > limit_bytes_) {
      throw std::runtime_error(
          "raw verbs allocation checkpoint outside local memory");
    }
    allocation_offset_ = checkpoint;
  }

  std::uint64_t current_offset() const { return allocation_offset_; }

private:
  static std::uint64_t Align(std::size_t bytes) {
    return (static_cast<std::uint64_t>(bytes) + 63) & ~std::uint64_t{63};
  }

  std::uint64_t limit_bytes_       = 0;
  std::uint64_t allocation_offset_ = 0;
  RawVerbsReservedRegion reserved_{};
};

class RawVerbsRegionAllocatorScope {
public:
  explicit RawVerbsRegionAllocatorScope(RawVerbsRegionAllocator* allocator)
      : allocator_(allocator),
        checkpoint_(allocator != nullptr ? allocator->Checkpoint() : 0) {}

  ~RawVerbsRegionAllocatorScope() {
    if (allocator_ != nullptr) {
      allocator_->Restore(checkpoint_);
    }
  }

  RawVerbsRegionAllocatorScope(const RawVerbsRegionAllocatorScope&) = delete;
  RawVerbsRegionAllocatorScope&
  operator=(const RawVerbsRegionAllocatorScope&) = delete;

private:
  RawVerbsRegionAllocator* allocator_ = nullptr;
  std::uint64_t checkpoint_           = 0;
};

struct RawVerbsRemoteMemory {
  std::uint16_t node_id   = 0;
  std::uint64_t base_addr = 0;
  std::uint32_t rkey      = 0;
};

struct RawVerbsCompletion {
  std::uint64_t wr_id    = 0;
  std::uint32_t imm_data = 0;
  bool has_imm           = false;
  ibv_wc_opcode opcode   = IBV_WC_SEND;
};

// Classification of a non-successful CQ completion (device-layer failure
// semantics). Only the peer-side classes say the remote node stopped
// answering; local classes mean the fault is ours and a QP rebuild would not
// help.
enum class RawVerbsErrorClass : std::uint8_t {
  kPeerUnreachable = 0, // Retry / RNR retry exhausted: no ACK from the peer.
  kPeerRejected    = 1, // The peer answered but rejected the operation.
  kLocalError      = 2, // Local length/protection/QP-operation fault.
  kFlushed         = 3, // The WR was flushed from a QP that was already dead.
  kOther           = 4,
};

inline const char* RawVerbsErrorClassToString(RawVerbsErrorClass error_class) {
  switch (error_class) {
  case RawVerbsErrorClass::kPeerUnreachable:
    return "peer_unreachable";
  case RawVerbsErrorClass::kPeerRejected:
    return "peer_rejected";
  case RawVerbsErrorClass::kLocalError:
    return "local_error";
  case RawVerbsErrorClass::kFlushed:
    return "flushed";
  case RawVerbsErrorClass::kOther:
    return "other";
  }
  return "other";
}

inline RawVerbsErrorClass
ClassifyRawVerbsCompletionStatus(ibv_wc_status status) {
  switch (status) {
  case IBV_WC_RETRY_EXC_ERR:
  case IBV_WC_RNR_RETRY_EXC_ERR:
    return RawVerbsErrorClass::kPeerUnreachable;
  case IBV_WC_REM_INV_REQ_ERR:
  case IBV_WC_REM_ACCESS_ERR:
  case IBV_WC_REM_OP_ERR:
    return RawVerbsErrorClass::kPeerRejected;
  case IBV_WC_LOC_LEN_ERR:
  case IBV_WC_LOC_QP_OP_ERR:
  case IBV_WC_LOC_PROT_ERR:
  case IBV_WC_LOC_ACCESS_ERR:
  case IBV_WC_LOC_RDD_VIOL_ERR:
    return RawVerbsErrorClass::kLocalError;
  case IBV_WC_WR_FLUSH_ERR:
    return RawVerbsErrorClass::kFlushed;
  default:
    return RawVerbsErrorClass::kOther;
  }
}

// True when the completion error means the shard cannot be reached on this
// lane: the peer stopped answering, or the lane was already flushed dead.
inline bool IsRawVerbsUnreachableError(RawVerbsErrorClass error_class) {
  return error_class == RawVerbsErrorClass::kPeerUnreachable ||
         error_class == RawVerbsErrorClass::kFlushed;
}

struct RawVerbsCompletionError {
  ibv_wc_status status           = IBV_WC_SUCCESS;
  std::uint64_t wr_id            = 0;
  ibv_wc_opcode opcode           = IBV_WC_SEND;
  RawVerbsErrorClass error_class = RawVerbsErrorClass::kOther;
};

// Human-readable form, e.g. "raw verbs CQ error: retry exceeded
// class=peer_unreachable wr_id=0 opcode=0".
std::string
DescribeRawVerbsCompletionError(const RawVerbsCompletionError& error);

class RawVerbsCompletionException : public std::runtime_error {
public:
  explicit RawVerbsCompletionException(const RawVerbsCompletionError& error,
                                       const std::string& context = "");
  const RawVerbsCompletionError& error() const { return error_; }

private:
  RawVerbsCompletionError error_;
};

enum class RawVerbsPollOutcome : std::uint8_t {
  kEmpty,      // No completion available.
  kCompletion, // A successful completion was returned.
  kError,      // A completion error was returned instead of thrown.
};

struct RawVerbsSge {
  const void* data  = nullptr;
  std::size_t bytes = 0;
};

inline constexpr int kRawVerbsPollBatchSize = 16;

class RawVerbsCompletionBatchCursor {
public:
  bool HasCachedCompletion() const { return current_ < size_; }

  void Reset(ibv_wc* entries, int size) {
    entries_ = entries;
    size_    = size;
    current_ = 0;
  }

  ibv_wc* TakeCachedCompletion() {
    if (!HasCachedCompletion()) {
      return nullptr;
    }
    return entries_ + current_++;
  }

private:
  ibv_wc* entries_ = nullptr;
  int size_        = 0;
  int current_     = 0;
};

struct RawVerbsNodeMeta {
  std::string deployment_id;
  std::string configuration_digest;
  std::string fabric_digest;
  std::uint16_t node_id          = 0;
  std::int32_t logical_id        = -1;
  std::uint16_t lid              = 0;
  std::uint32_t qpn              = 0;
  std::uint32_t psn              = 3185;
  std::uint32_t rkey             = 0;
  std::uint64_t base_addr        = 0;
  std::uint8_t gid[16]           = {};
  std::uint32_t protocol_version = recstore::kRdmaDeploymentProtocolVersion;
  std::uint64_t deployment_epoch = 0;
  std::uint8_t node_role         = 0;
  std::uint8_t port_num          = 0;
  std::uint8_t link_layer        = 0;
  std::int32_t gid_index         = -1;
  std::uint8_t active_mtu        = 0;
  std::uint8_t fabric_mode       = 0;
  // This lane's rebuild generation. Bumped before the QP is republished so the
  // peer can observe "this side rebuilt" and re-pair the QP.
  std::uint64_t lane_generation = 0;
};

class RawVerbsTransport {
public:
  explicit RawVerbsTransport(const RawVerbsConfig& config);
  ~RawVerbsTransport();

  RawVerbsTransport(const RawVerbsTransport&)            = delete;
  RawVerbsTransport& operator=(const RawVerbsTransport&) = delete;

  void RegisterMemoryRegion(void* base, std::size_t bytes);
  void* AllocateRegistered(std::size_t bytes);
  std::uint64_t SaveAllocationState() const;
  void RestoreAllocationState(std::uint64_t checkpoint);
  GlobalAddress LocalAddress(void* ptr) const;
  void* LocalPointer(GlobalAddress address) const;

  void Publish();
  void Connect();
  void PublishAndConnect();
  RawVerbsNodeMeta LocalMeta() const;

  void Write(const void* local,
             GlobalAddress remote,
             std::size_t bytes,
             std::uint64_t wr_id,
             bool signaled);
  void WriteSg(base::ConstArray<RawVerbsSge> sges,
               GlobalAddress remote,
               std::uint64_t wr_id,
               bool signaled);
  void WriteWithImm(const void* local,
                    GlobalAddress remote,
                    std::size_t bytes,
                    std::uint32_t imm_data,
                    std::uint64_t wr_id,
                    bool signaled);
  void Read(void* local,
            GlobalAddress remote,
            std::size_t bytes,
            std::uint64_t wr_id,
            bool signaled);
  void SendDoorbell(
      std::uint16_t node_id, std::uint32_t imm_data, std::uint64_t wr_id);
  bool Poll(RawVerbsCompletion* completion, int timeout_ms);
  // Non-blocking variant of Poll() for wait loops that must notice a broken QP
  // without waiting for the next drain: a completion error is returned through
  // `error` (which must not be null) instead of being thrown.
  RawVerbsPollOutcome
  TryPoll(RawVerbsCompletion* completion, RawVerbsCompletionError* error);
  // Rebuilds the QP for one peer node after a device-layer failure, reusing the
  // PD / MR / registered memory and the peer metadata captured by Connect():
  // reset -> init -> RTR -> RTS. The reset flushes every outstanding WR on the
  // QP, so those completions are dropped from the CQ afterwards and the receive
  // queue is re-armed because reset flushes it too. Only peer-side failures
  // (retry exhaustion / flush) are worth rebuilding; callers must decide that
  // from RawVerbsErrorClass. The QP is a shared resource, so a rebuild is only
  // half of recovery: the peer must rebuild its side too, which the caller
  // coordinates through the control plane. Must not run concurrently with
  // submits or polls on the same lane.
  //
  // `cq_exclusive` states that this QP is the only one posting to the
  // transport's CQ (true on the client side, and for a server lane that serves
  // a single OS client). The post-reset drain runs only then, because draining
  // a shared CQ would also discard the other peer QPs' completions; when false
  // the caller must not rebuild in the first place (see
  // RcShardServerTransport::RebuildClientLaneIfStale).
  void RebuildQp(std::uint16_t node_id, bool cq_exclusive = true);
  // Lane generation handshake. The lane generation is the local side's rebuild
  // counter toward `node_id`; `AdvanceLaneGeneration` bumps it and republishes
  // the metadata so the peer can observe the change. `LaneGeneration` reads
  // that counter back; the peer's counter is read from the meta returned by
  // RefreshPeerMeta().
  std::uint64_t AdvanceLaneGeneration(std::uint16_t node_id);
  std::uint64_t LaneGeneration(std::uint16_t node_id) const;
  // Re-reads `node_id`'s published metadata (GetMeta), validates it, and stores
  // it for a subsequent RebuildQp(). Used to learn a peer's lane generation and
  // to pick up addressing changes if the peer process restarted.
  RawVerbsNodeMeta RefreshPeerMeta(std::uint16_t node_id);
  std::uint32_t max_inline_data(std::uint16_t node_id) const;

private:
  struct Impl;
  ibv_mr* FindLocalMr(const void* ptr, std::size_t bytes) const;
  // Lazily opens (and caches) the one control-plane client this transport owns.
  // Publish()/Connect()/RefreshPeerMeta() share it so a lane uses a single
  // channel instead of a fresh connection per metadata RPC.
  RdmaControlPlaneClient& ControlPlane();
  std::unique_ptr<Impl> impl_;
};

class RawVerbsTransportAllocationScope {
public:
  explicit RawVerbsTransportAllocationScope(RawVerbsTransport* transport)
      : transport_(transport),
        checkpoint_(
            transport != nullptr ? transport->SaveAllocationState() : 0) {}

  ~RawVerbsTransportAllocationScope() {
    if (transport_ != nullptr) {
      transport_->RestoreAllocationState(checkpoint_);
    }
  }

  RawVerbsTransportAllocationScope(const RawVerbsTransportAllocationScope&) =
      delete;
  RawVerbsTransportAllocationScope&
  operator=(const RawVerbsTransportAllocationScope&) = delete;

private:
  RawVerbsTransport* transport_ = nullptr;
  std::uint64_t checkpoint_     = 0;
};

} // namespace petps
