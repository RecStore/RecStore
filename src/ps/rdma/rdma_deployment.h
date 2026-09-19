#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/json.h"

namespace recstore {

inline constexpr std::uint32_t kRdmaDeploymentProtocolVersion = 1;

enum class RdmaFabricMode { kIb, kRocEv1, kRocEv2 };
enum class RdmaNodeRole { kServer, kClient };

struct ResolvedRdmaFabric {
  std::string deployment_id;
  std::uint64_t deployment_epoch = 0;
  std::uint32_t protocol_version = kRdmaDeploymentProtocolVersion;
  std::string configuration_digest;
  std::string fabric_digest;
  RdmaFabricMode mode = RdmaFabricMode::kIb;
  std::string device;
  int port                 = 0;
  int gid_index            = -1;
  int hop_limit            = 0;
  int traffic_class        = 0;
  std::uint32_t flow_label = 0;
  RdmaNodeRole role        = RdmaNodeRole::kServer;
  int logical_id           = -1;
};

struct ResolvedRdmaEndpoint {
  std::string host;
  int port  = 0;
  int shard = -1;
};

// Canonical RDMA topology after framework-level config normalization.
struct ResolvedRdmaDeployment {
  std::string deployment_id;
  std::uint64_t epoch            = 0;
  std::uint32_t protocol_version = kRdmaDeploymentProtocolVersion;
  std::string configuration_digest;
  std::string fabric_digest;
  int num_shards = 0;
  std::string hash_method;
  int max_keys_per_request = 0;
  std::vector<ResolvedRdmaEndpoint> endpoints;
  std::unordered_map<int, int> shard_to_endpoint_index;
  std::unordered_map<int, ResolvedRdmaFabric> fabrics;
  int num_clients = 0;
};

ResolvedRdmaDeployment
ParseResolvedRdmaDeployment(const json& distributed_client);

ResolvedRdmaDeployment ParseResolvedRdmaDeploymentConfig(const json& config);
const ResolvedRdmaFabric&
LocalRdmaFabric(const ResolvedRdmaDeployment& deployment, int node_id);

} // namespace recstore
