#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ps/rdma/raw_verbs_transport.h"

namespace grpc {
class Server;
}

namespace petps {

class RdmaControlPlaneService;

struct RdmaControlPlaneEndpoint {
  std::string host = "127.0.0.1";
  int port         = 25100;
  int timeout_ms   = 30000;
  std::string deployment_id;
  std::uint64_t deployment_epoch = 0;
  std::uint32_t protocol_version = recstore::kRdmaDeploymentProtocolVersion;
  std::string configuration_digest;
  std::string fabric_digest;
  // Optional on client-only endpoints; the server uses it to bound READY ids.
  int num_servers = 0;
  // Control-plane health monitor thresholds (server side only). Defaults match
  // the L2 budget in docs/plans/rdma-reliability-architecture.md. Record GC is
  // off by default so P1a stays observation-only; P1b re-decides the policy.
  int health_stale_ms            = 3000;
  int health_dead_ms             = 9000;
  int health_record_retention_ms = 0;
};

inline constexpr std::uint32_t kRawVerbsMetadataMagic  = 0x52444d41;
inline constexpr std::size_t kMaxRawVerbsMetadataBytes = 4096;

// Per-shard liveness used by the control-plane health monitor. This is an
// observation-only signal in P1a: it is logged and counted, but it does not
// change request handling (see docs/plans/rdma-reliability-architecture.md).
enum class RdmaServerHealth : std::uint8_t {
  kServing = 0,
  kSuspect = 1,
  kDead    = 2,
};

inline const char* RdmaServerHealthToString(RdmaServerHealth health) {
  switch (health) {
  case RdmaServerHealth::kServing:
    return "serving";
  case RdmaServerHealth::kSuspect:
    return "suspect";
  case RdmaServerHealth::kDead:
    return "dead";
  }
  return "unknown";
}

// Pure staleness evaluation, shared by the monitor thread and unit tests. A
// record that has never been touched (default last_seen) is treated as serving
// so a freshly created record is never reported dead.
inline RdmaServerHealth EvaluateServerHealth(
    std::chrono::steady_clock::time_point last_seen,
    std::chrono::steady_clock::time_point now,
    int stale_ms,
    int dead_ms) {
  if (last_seen == std::chrono::steady_clock::time_point{}) {
    return RdmaServerHealth::kServing;
  }
  const auto age_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_seen)
          .count();
  if (dead_ms > 0 && age_ms >= dead_ms) {
    return RdmaServerHealth::kDead;
  }
  if (stale_ms > 0 && age_ms >= stale_ms) {
    return RdmaServerHealth::kSuspect;
  }
  return RdmaServerHealth::kServing;
}

// Metadata keys carried on the Probe RPC so heartbeat payloads do not need a
// proto change.
inline constexpr char kHeartbeatNodeIdMetadata[]   = "x-recstore-rdma-node-id";
inline constexpr char kHeartbeatInstanceMetadata[] = "x-recstore-rdma-instance";
inline constexpr char kHeartbeatDeploymentIdMetadata[] =
    "x-recstore-rdma-deployment-id";
inline constexpr char kHeartbeatDeploymentEpochMetadata[] =
    "x-recstore-rdma-deployment-epoch";

std::string EncodeRawVerbsNodeMeta(const RawVerbsNodeMeta& meta);
RawVerbsNodeMeta DecodeRawVerbsNodeMeta(const std::string& payload);
void ValidateRawVerbsPeerMeta(const RawVerbsConfig& local,
                              int expected_node_id,
                              const RawVerbsNodeMeta& remote);

class RdmaControlPlaneClient {
public:
  explicit RdmaControlPlaneClient(RdmaControlPlaneEndpoint endpoint);

  void PublishMeta(int publisher_node_id,
                   int publisher_lane,
                   int receiver_node_id,
                   int receiver_lane,
                   const RawVerbsNodeMeta& meta) const;
  RawVerbsNodeMeta
  GetMeta(int publisher_node_id,
          int publisher_lane,
          int receiver_node_id,
          int receiver_lane,
          int timeout_ms = -1) const;
  // `instance_nonce` identifies the serving process; when non-empty it is
  // recorded in the READY record so a restart can be told apart from the same
  // process (docs/plans/rdma-reliability-architecture.md §3.7).
  void PublishServerReady(int server_id,
                          const std::string& instance_nonce = "") const;
  void PublishServerDraining(int server_id) const;
  // Lightweight liveness heartbeat. P1a: the server records it as `last_seen`.
  void SendHeartbeat(int server_id, const std::string& instance_nonce) const;
  void WaitServer(int server_id, int timeout_ms = -1) const;
  void WaitServerReady(int num_servers, int timeout_ms = -1) const;

private:
  RdmaControlPlaneEndpoint endpoint_;
};

class RdmaControlPlaneServer {
public:
  explicit RdmaControlPlaneServer(RdmaControlPlaneEndpoint endpoint);
  ~RdmaControlPlaneServer();

  void Start();
  void Stop();

  // P1a observability (docs/plans/rdma-reliability-architecture.md §3.7). These
  // read-only accessors let tests and future P1b callers inspect health without
  // touching the monitor internals.
  std::uint64_t heartbeats_received() const {
    return heartbeats_received_.load(std::memory_order_relaxed);
  }
  std::uint64_t health_transitions() const {
    return health_transitions_.load(std::memory_order_relaxed);
  }
  // Health of a ready record evaluated at `now` (kServing when unknown).
  RdmaServerHealth
  ServerHealth(int server_id, std::chrono::steady_clock::time_point now) const;
  // Latest instance nonce observed for a ready record ("" when unknown).
  std::string ServerInstanceNonce(int server_id) const;

private:
  friend class RdmaControlPlaneService;

  struct MetaKey {
    std::string deployment_id;
    std::uint64_t deployment_epoch = 0;
    int publisher_node_id          = 0;
    int publisher_lane             = 0;
    int receiver_node_id           = 0;
    int receiver_lane              = 0;

    bool operator==(const MetaKey& other) const {
      return deployment_id == other.deployment_id &&
             deployment_epoch == other.deployment_epoch &&
             publisher_node_id == other.publisher_node_id &&
             publisher_lane == other.publisher_lane &&
             receiver_node_id == other.receiver_node_id &&
             receiver_lane == other.receiver_lane;
    }
  };

  struct MetaKeyHash {
    std::size_t operator()(const MetaKey& key) const;
  };

  struct ReadyKey {
    std::string deployment_id;
    std::uint64_t deployment_epoch = 0;
    int server_id                  = 0;

    bool operator==(const ReadyKey& other) const {
      return deployment_id == other.deployment_id &&
             deployment_epoch == other.deployment_epoch &&
             server_id == other.server_id;
    }
  };

  struct ReadyKeyHash {
    std::size_t operator()(const ReadyKey& key) const;
  };

  struct ReadyRecord {
    enum class State : std::uint8_t { kServing, kDraining };

    std::uint32_t protocol_version = 0;
    std::string configuration_digest;
    std::string fabric_digest;
    State state = State::kServing;

    // P1a liveness fields. `last_seen` is refreshed by PublishServerReady and
    // by heartbeat Probes; `instance_nonce` identifies the serving process so a
    // restart can be told apart from the same process (see §3.7).
    std::chrono::steady_clock::time_point last_seen{};
    std::string instance_nonce;
    RdmaServerHealth observed_health = RdmaServerHealth::kServing;
  };

  int ReadyCount(const std::string& deployment_id,
                 std::uint64_t deployment_epoch,
                 std::uint32_t protocol_version,
                 const std::string& configuration_digest,
                 const std::string& fabric_digest) const;
  bool HasReadyContractMismatch(
      const std::string& deployment_id,
      std::uint64_t deployment_epoch,
      std::uint32_t protocol_version,
      const std::string& configuration_digest,
      const std::string& fabric_digest) const;
  bool HasDrainingServer(const std::string& deployment_id,
                         std::uint64_t deployment_epoch,
                         int num_servers) const;

  RdmaControlPlaneEndpoint endpoint_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<MetaKey, RawVerbsNodeMeta, MetaKeyHash> metadata_;
  std::unordered_map<ReadyKey, ReadyRecord, ReadyKeyHash> ready_servers_;
  std::atomic<bool> stop_requested_{false};
  std::unique_ptr<RdmaControlPlaneService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::thread health_monitor_;
  std::condition_variable health_cv_;
  std::atomic<std::uint64_t> heartbeats_received_{0};
  std::atomic<std::uint64_t> health_transitions_{0};

  void HealthMonitorLoop();
  void RecordHeartbeat(const std::string& deployment_id,
                       std::uint64_t deployment_epoch,
                       int server_id,
                       const std::string& instance_nonce);
};

} // namespace petps
