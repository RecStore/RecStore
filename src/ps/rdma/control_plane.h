#pragma once

#include <atomic>
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
};

inline constexpr std::uint32_t kRawVerbsMetadataMagic  = 0x52444d41;
inline constexpr std::size_t kMaxRawVerbsMetadataBytes = 4096;

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
  void PublishServerReady(int server_id) const;
  void PublishServerDraining(int server_id) const;
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
  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<MetaKey, RawVerbsNodeMeta, MetaKeyHash> metadata_;
  std::unordered_map<ReadyKey, ReadyRecord, ReadyKeyHash> ready_servers_;
  std::atomic<bool> stop_requested_{false};
  std::unique_ptr<RdmaControlPlaneService> service_;
  std::unique_ptr<grpc::Server> server_;
};

} // namespace petps
