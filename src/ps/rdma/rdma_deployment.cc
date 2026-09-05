#include "ps/rdma/rdma_deployment.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace recstore {

namespace {

std::string StableDigest(const json& value) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : value.dump()) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

json CanonicalDeploymentConfig(const json& distributed_client,
                               const json& rdma_deployment) {
  json canonical_client = distributed_client;
  auto& servers         = canonical_client["servers"];
  std::sort(
      servers.begin(), servers.end(), [](const json& lhs, const json& rhs) {
        return lhs.at("shard").get<int>() < rhs.at("shard").get<int>();
      });

  json canonical_rdma = rdma_deployment;
  auto& nodes         = canonical_rdma["nodes"];
  std::sort(nodes.begin(), nodes.end(), [](const json& lhs, const json& rhs) {
    return lhs.at("node_id").get<int>() < rhs.at("node_id").get<int>();
  });
  return {
      {"distributed_client", std::move(canonical_client)},
      {"rdma_deployment", std::move(canonical_rdma)},
  };
}

} // namespace

ResolvedRdmaDeployment
ParseResolvedRdmaDeployment(const json& distributed_client) {
  if (!distributed_client.is_object()) {
    throw std::invalid_argument("RDMA distributed_client must be an object");
  }
  if (!distributed_client.contains("num_shards") ||
      !distributed_client["num_shards"].is_number_integer()) {
    throw std::invalid_argument("RDMA distributed_client.num_shards must be an "
                                "explicit positive integer");
  }
  const int num_shards = distributed_client["num_shards"].get<int>();
  if (num_shards <= 0) {
    throw std::invalid_argument("RDMA distributed_client.num_shards must be an "
                                "explicit positive integer");
  }
  if (!distributed_client.contains("hash_method") ||
      !distributed_client["hash_method"].is_string()) {
    throw std::invalid_argument(
        "RDMA distributed_client.hash_method must be explicit");
  }
  const std::string hash_method =
      distributed_client["hash_method"].get<std::string>();
  if (hash_method != "city_hash" && hash_method != "simple_mod") {
    throw std::invalid_argument("unsupported RDMA hash method: " + hash_method);
  }
  if (!distributed_client.contains("max_keys_per_request") ||
      !distributed_client["max_keys_per_request"].is_number_integer() ||
      distributed_client["max_keys_per_request"].get<int>() <= 0) {
    throw std::invalid_argument("RDMA distributed_client.max_keys_per_request "
                                "must be explicit and positive");
  }
  if (!distributed_client.contains("servers") ||
      !distributed_client["servers"].is_array() ||
      distributed_client["servers"].size() !=
          static_cast<std::size_t>(num_shards)) {
    throw std::invalid_argument("RDMA distributed_client.servers must be an "
                                "array with one entry per shard");
  }

  ResolvedRdmaDeployment deployment;
  deployment.num_shards  = num_shards;
  deployment.hash_method = hash_method;
  deployment.max_keys_per_request =
      distributed_client["max_keys_per_request"].get<int>();
  deployment.endpoints.reserve(distributed_client["servers"].size());
  std::vector<bool> seen_shards(static_cast<std::size_t>(num_shards), false);
  for (const auto& server : distributed_client["servers"]) {
    if (!server.is_object()) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers entries must be objects");
    }
    if (!server.contains("host") || !server["host"].is_string() ||
        server["host"].get<std::string>().empty()) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers[].host must be a non-empty string");
    }
    if (!server.contains("port") || !server["port"].is_number_integer()) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers[].port must be explicit");
    }
    const int port = server["port"].get<int>();
    if (port < 1 || port > 65535) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers[].port must be in [1, 65535]");
    }
    if (!server.contains("shard") || !server["shard"].is_number_integer()) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers[].shard must be explicit");
    }
    const int shard = server["shard"].get<int>();
    if (shard < 0 || shard >= num_shards) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers[].shard is out of range");
    }
    if (seen_shards[static_cast<std::size_t>(shard)]) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers contains duplicate shard");
    }
    const std::string host   = server["host"].get<std::string>();
    const int endpoint_index = static_cast<int>(deployment.endpoints.size());
    deployment.endpoints.push_back({host, port, shard});
    deployment.shard_to_endpoint_index.emplace(shard, endpoint_index);
    seen_shards[static_cast<std::size_t>(shard)] = true;
  }
  for (bool seen : seen_shards) {
    if (!seen) {
      throw std::invalid_argument(
          "RDMA distributed_client.servers must cover every shard");
    }
  }
  return deployment;
}

namespace {
RdmaFabricMode ParseMode(const json& node) {
  const auto mode = node.value("mode", std::string());
  if (mode == "ib")
    return RdmaFabricMode::kIb;
  if (mode == "rocev1")
    return RdmaFabricMode::kRocEv1;
  if (mode == "rocev2")
    return RdmaFabricMode::kRocEv2;
  throw std::invalid_argument("RDMA fabric mode must be ib, rocev1, or rocev2");
}
} // namespace

ResolvedRdmaDeployment ParseResolvedRdmaDeploymentConfig(const json& config) {
  if (!config.contains("distributed_client") ||
      !config.contains("rdma_deployment")) {
    throw std::invalid_argument("RDMA config requires rdma_deployment");
  }
  auto deployment =
      ParseResolvedRdmaDeployment(config.at("distributed_client"));
  const auto& rdma = config.at("rdma_deployment");
  if (!rdma.contains("deployment_id") ||
      !rdma.at("deployment_id").is_string() ||
      rdma.at("deployment_id").get<std::string>().empty()) {
    throw std::invalid_argument(
        "rdma_deployment.deployment_id must be explicit and non-empty");
  }
  if (!rdma.contains("epoch") || !rdma.at("epoch").is_number_integer() ||
      rdma.at("epoch").get<std::int64_t>() <= 0) {
    throw std::invalid_argument(
        "rdma_deployment.epoch must be explicit and positive");
  }
  if (!rdma.contains("protocol_version") ||
      !rdma.at("protocol_version").is_number_integer() ||
      rdma.at("protocol_version").get<std::int64_t>() <= 0) {
    throw std::invalid_argument(
        "rdma_deployment.protocol_version must be explicit and positive");
  }
  deployment.deployment_id = rdma.at("deployment_id").get<std::string>();
  deployment.epoch         = rdma.at("epoch").get<std::uint64_t>();
  deployment.protocol_version =
      rdma.at("protocol_version").get<std::uint32_t>();
  if (deployment.protocol_version != kRdmaDeploymentProtocolVersion) {
    throw std::invalid_argument("unsupported RDMA deployment protocol_version");
  }
  if (!rdma.contains("num_clients") ||
      !rdma.at("num_clients").is_number_integer() ||
      rdma.at("num_clients").get<int>() <= 0)
    throw std::invalid_argument(
        "rdma_deployment.num_clients must be explicit and positive");
  deployment.num_clients = rdma.at("num_clients").get<int>();
  if (!rdma.contains("nodes") || !rdma.at("nodes").is_array() ||
      rdma.at("nodes").empty()) {
    throw std::invalid_argument(
        "rdma_deployment.nodes must be a non-empty array");
  }
  const auto& nodes = rdma.at("nodes");
  for (const auto& node : nodes) {
    if (!node.is_object() || !node.contains("node_id") ||
        !node.contains("role") || !node.contains("device") ||
        !node.contains("port") || !node.contains("gid_index")) {
      throw std::invalid_argument(
          "fabric node requires node_id, role, device, port, gid_index");
    }
    if (!node.at("node_id").is_number_integer() ||
        !node.at("role").is_string() || !node.at("device").is_string() ||
        !node.at("port").is_number_integer() ||
        !node.at("gid_index").is_number_integer()) {
      throw std::invalid_argument("RDMA fabric node fields have invalid types");
    }
    const int id = node.at("node_id").get<int>();
    if (!deployment.fabrics.emplace(id, ResolvedRdmaFabric{}).second) {
      throw std::invalid_argument("duplicate RDMA fabric node_id");
    }
    auto& fabric            = deployment.fabrics.at(id);
    fabric.deployment_id    = deployment.deployment_id;
    fabric.deployment_epoch = deployment.epoch;
    fabric.protocol_version = deployment.protocol_version;
    const auto role         = node.at("role").get<std::string>();
    if (role != "server" && role != "client") {
      throw std::invalid_argument("RDMA fabric role must be server or client");
    }
    const bool server = role == "server";
    fabric.role       = server ? RdmaNodeRole::kServer : RdmaNodeRole::kClient;
    fabric.logical_id = server ? id : id - deployment.num_shards;
    if ((server && (id >= deployment.num_shards || fabric.logical_id < 0)) ||
        (!server && (id < deployment.num_shards || fabric.logical_id < 0 ||
                     fabric.logical_id >= deployment.num_clients)))
      throw std::invalid_argument("RDMA fabric node_id/role is inconsistent");
    fabric.mode      = ParseMode(node);
    fabric.device    = node.at("device").get<std::string>();
    fabric.port      = node.at("port").get<int>();
    fabric.gid_index = node.at("gid_index").get<int>();
    if (fabric.device.empty() || fabric.port < 1 || fabric.port > 255 ||
        fabric.gid_index < 0 || id < 0) {
      throw std::invalid_argument("RDMA fabric device/port/gid_index invalid");
    }
    if (fabric.mode == RdmaFabricMode::kRocEv2) {
      if (!node.contains("hop_limit") || !node.contains("traffic_class") ||
          !node.contains("flow_label")) {
        throw std::invalid_argument(
            "RoCE v2 requires hop_limit, traffic_class, flow_label");
      }
      fabric.hop_limit     = node.at("hop_limit").get<int>();
      fabric.traffic_class = node.at("traffic_class").get<int>();
      fabric.flow_label    = node.at("flow_label").get<std::uint32_t>();
      if (fabric.hop_limit < 1 || fabric.hop_limit > 255 ||
          fabric.traffic_class < 0 || fabric.traffic_class > 255 ||
          fabric.flow_label > 0xfffff) {
        throw std::invalid_argument("RoCE v2 network fields are out of range");
      }
    }
  }
  if (deployment.fabrics.size() !=
      static_cast<std::size_t>(deployment.num_shards + deployment.num_clients))
    throw std::invalid_argument(
        "RDMA fabric nodes must cover every server and client");
  const json canonical =
      CanonicalDeploymentConfig(config.at("distributed_client"), rdma);
  deployment.configuration_digest = StableDigest(canonical);
  deployment.fabric_digest =
      StableDigest(canonical.at("rdma_deployment").at("nodes"));
  for (auto& [node_id, fabric] : deployment.fabrics) {
    (void)node_id;
    fabric.configuration_digest = deployment.configuration_digest;
    fabric.fabric_digest        = deployment.fabric_digest;
  }
  return deployment;
}

const ResolvedRdmaFabric&
LocalRdmaFabric(const ResolvedRdmaDeployment& deployment, int node_id) {
  const auto it = deployment.fabrics.find(node_id);
  if (it == deployment.fabrics.end()) {
    throw std::invalid_argument(
        "missing RDMA fabric for local node_id=" + std::to_string(node_id));
  }
  return it->second;
}

} // namespace recstore
