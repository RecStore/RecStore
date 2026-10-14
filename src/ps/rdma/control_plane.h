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

// Snapshot returned by the GetServerHealth RPC and the matching server-side
// accessor. `known=false` means shard 0 has no READY record for that shard, so
// the data path must stay on its own budgets instead of failing the request
// (P1b slow-path fast-fail; docs/plans/rdma-reliability-architecture.md §3.7).
struct RdmaServerHealthSnapshot {
  bool known                    = false;
  RdmaServerHealth health       = RdmaServerHealth::kServing;
  std::int64_t last_seen_age_ms = -1;
  std::string instance_nonce;
};

// Slow-path health probe policy (P1b). A client that has been waiting longer
// than the probe threshold asks the control plane about its shard; the query is
// throttled so many waiters share at most one RPC per probe interval
// (docs/plans/rdma-reliability-architecture.md §3.7).
enum class RdmaSlowPathVerdict {
  kKeepWaiting,      // serving / suspect / unknown: stay on the L0..L3 budgets
  kShardUnreachable, // the control plane judged the shard dead
};

inline RdmaSlowPathVerdict ClassifyShardHealthSnapshot(
    bool query_ok, const RdmaServerHealthSnapshot& snapshot) {
  // Never fail a request on an inconclusive query: an unreachable control plane
  // or a shard without a READY record falls back to the data-path budgets
  // instead of producing a false verdict.
  if (!query_ok || !snapshot.known) {
    return RdmaSlowPathVerdict::kKeepWaiting;
  }
  return snapshot.health == RdmaServerHealth::kDead
           ? RdmaSlowPathVerdict::kShardUnreachable
           : RdmaSlowPathVerdict::kKeepWaiting;
}

// Throttles slow-path probes: at most one in-flight query, and at most one
// query per `ttl_ms`. A caller that cannot get the gate skips its own probe
// instead of blocking (the verdict is re-read after the TTL); the probe
// timestamp is taken before the RPC so a slow or failing control plane is
// throttled as well.
class RdmaSlowPathProbeGate {
public:
  // Returns true when the caller owns the probe and must call EndProbe().
  bool TryBeginProbe(std::chrono::steady_clock::time_point now, int ttl_ms) {
    std::lock_guard<std::mutex> guard(mu_);
    if (probing_) {
      skipped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (has_probe_ && ttl_ms > 0 &&
        now - last_probe_ < std::chrono::milliseconds(ttl_ms)) {
      skipped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    probing_    = true;
    has_probe_  = true;
    last_probe_ = now;
    return true;
  }

  void EndProbe() {
    std::lock_guard<std::mutex> guard(mu_);
    probing_ = false;
    probes_.fetch_add(1, std::memory_order_relaxed);
  }

  std::uint64_t probes() const {
    return probes_.load(std::memory_order_relaxed);
  }
  std::uint64_t skipped() const {
    return skipped_.load(std::memory_order_relaxed);
  }

private:
  mutable std::mutex mu_;
  bool probing_   = false;
  bool has_probe_ = false;
  std::chrono::steady_clock::time_point last_probe_{};
  std::atomic<std::uint64_t> probes_{0};
  std::atomic<std::uint64_t> skipped_{0};
};

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

// Opaque per-client gRPC channel/stub cache; defined in control_plane.cc. Kept
// out of the header so this file does not pull in the generated stubs, and
// shared between copies of a client so they reuse one connection.
struct RdmaControlPlaneChannelCache;

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
  // Queries shard health from the shard 0 control plane. Never throws: RPC
  // failures are reported as `query_ok=false` with an unknown snapshot so the
  // client slow path can fall back to L0/L3 instead of failing the request.
  RdmaServerHealthSnapshot GetServerHealth(
      int server_id, int timeout_ms = -1, bool* query_ok = nullptr) const;
  void WaitServer(int server_id, int timeout_ms = -1) const;
  void WaitServerReady(int num_servers, int timeout_ms = -1) const;

  // Number of gRPC channels this client has opened. It stays 1 no matter how
  // many RPCs are sent, which is what keeps the per-second shard heartbeats off
  // a per-heartbeat connect (P1b §15-5).
  std::uint64_t channels_opened() const;

private:
  RdmaControlPlaneEndpoint endpoint_;
  std::shared_ptr<RdmaControlPlaneChannelCache> channels_;
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
  // Heartbeats dropped for a bad identity: malformed metadata, another
  // deployment/epoch, or an out-of-range shard id (P1b §15-6).
  std::uint64_t heartbeats_rejected() const {
    return heartbeats_rejected_.load(std::memory_order_relaxed);
  }
  // Heartbeats for a shard without a READY record; READY stays the only way to
  // become serving, so these never create or revive a record.
  std::uint64_t heartbeats_unknown_shard() const {
    return heartbeats_unknown_shard_.load(std::memory_order_relaxed);
  }
  std::uint64_t health_transitions() const {
    return health_transitions_.load(std::memory_order_relaxed);
  }
  // Health of a ready record evaluated at `now` (kServing when unknown).
  RdmaServerHealth
  ServerHealth(int server_id, std::chrono::steady_clock::time_point now) const;
  // Fresh snapshot for the GetServerHealth RPC: `known=false` when there is no
  // READY record, otherwise the health evaluated at query time plus the
  // last-seen age and instance nonce.
  RdmaServerHealthSnapshot ServerHealthSnapshot(int server_id) const;
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
  std::atomic<std::uint64_t> heartbeats_rejected_{0};
  std::atomic<std::uint64_t> heartbeats_unknown_shard_{0};
  std::atomic<std::uint64_t> health_transitions_{0};

  void HealthMonitorLoop();
  // Records an accepted heartbeat. The caller has already validated the
  // heartbeat identity against this endpoint.
  void RecordHeartbeat(int server_id, const std::string& instance_nonce);
};

} // namespace petps
