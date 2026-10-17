#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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
  // Control-plane health evaluation thresholds (server side only).
  int health_stale_ms = 3000;
  int health_dead_ms  = 9000;
};

inline constexpr std::uint32_t kRawVerbsMetadataMagic  = 0x52444d41;
inline constexpr std::size_t kMaxRawVerbsMetadataBytes = 4096;

// Per-shard liveness reported by GetServerHealth: heartbeats feed `last_seen`
// and a query evaluates it against the stale/dead thresholds.
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
// the data path must stay on its own budgets instead of failing the request.
struct RdmaServerHealthSnapshot {
  bool known                    = false;
  RdmaServerHealth health       = RdmaServerHealth::kServing;
  std::int64_t last_seen_age_ms = -1;
};

// Slow-path health probe policy. A client that has been waiting longer than the
// probe threshold asks the control plane about its shard; the query is
// throttled so many waiters share at most one RPC per probe interval.
// True when the control plane positively judged the shard dead. An inconclusive
// query never fails a request: an unreachable control plane and a shard without
// a READY record both report `known=false`, and then the request stays on its
// data-path wait budgets instead of producing a false verdict.
inline bool IsShardUnreachable(const RdmaServerHealthSnapshot& snapshot) {
  return snapshot.known && snapshot.health == RdmaServerHealth::kDead;
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
      return false;
    }
    if (last_probe_ != std::chrono::steady_clock::time_point{} && ttl_ms > 0 &&
        now - last_probe_ < std::chrono::milliseconds(ttl_ms)) {
      return false;
    }
    probing_    = true;
    last_probe_ = now;
    return true;
  }

  void EndProbe() {
    std::lock_guard<std::mutex> guard(mu_);
    probing_ = false;
  }

private:
  mutable std::mutex mu_;
  bool probing_ = false;
  std::chrono::steady_clock::time_point last_probe_{};
};

std::string EncodeRawVerbsNodeMeta(const RawVerbsNodeMeta& meta);
RawVerbsNodeMeta DecodeRawVerbsNodeMeta(const std::string& payload);
void ValidateRawVerbsPeerMeta(const RawVerbsConfig& local,
                              int expected_node_id,
                              const RawVerbsNodeMeta& remote);

// Opaque per-client gRPC channel/stub cache; defined in control_plane.cc. Kept
// out of the header so this file does not pull in the generated stubs.
struct RdmaControlPlaneChannelCache;

class RdmaControlPlaneClient {
public:
  explicit RdmaControlPlaneClient(RdmaControlPlaneEndpoint endpoint);
  // Out of line: `channels_` is an incomplete type here.
  ~RdmaControlPlaneClient();

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
  void PublishServerReady(int server_id) const;
  void PublishServerDraining(int server_id) const;
  // Lightweight liveness heartbeat; the server records it as `last_seen`.
  void SendHeartbeat(int server_id) const;
  // Queries shard health from the shard 0 control plane. Never throws: RPC
  // failures (and shards without a READY record) come back as `known=false`, so
  // the client slow path falls back to its wait budgets instead of failing.
  RdmaServerHealthSnapshot
  GetServerHealth(int server_id, int timeout_ms = -1) const;
  void WaitServer(int server_id, int timeout_ms = -1) const;
  void WaitServerReady(int num_servers, int timeout_ms = -1) const;

private:
  RdmaControlPlaneEndpoint endpoint_;
  std::unique_ptr<RdmaControlPlaneChannelCache> channels_;
};

class RdmaControlPlaneServer {
public:
  explicit RdmaControlPlaneServer(RdmaControlPlaneEndpoint endpoint);
  ~RdmaControlPlaneServer();

  void Start();
  void Stop();

  // Fresh snapshot for the GetServerHealth RPC: `known=false` when there is no
  // READY record, otherwise the health evaluated at query time plus the
  // last-seen age.
  RdmaServerHealthSnapshot ServerHealthSnapshot(int server_id) const;

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

    // `last_seen` is refreshed by PublishServerReady and by heartbeat Probes.
    std::chrono::steady_clock::time_point last_seen{};
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

  // Records an accepted heartbeat. The caller has already validated the
  // heartbeat identity against this endpoint.
  void RecordHeartbeat(int server_id);
};

} // namespace petps
