#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "base/factory.h"
#include "base/json.h"
#include "ps/base/base_client.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/grpc/grpc_ps_client.h"

namespace recstore {

inline std::string NormalizePSType(std::string ps_type) {
  std::transform(
      ps_type.begin(), ps_type.end(), ps_type.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
      });
  return ps_type;
}

inline std::string ResolveFrameworkPSType(const json& config) {
  std::string ps_type = "GRPC";
  if (config.contains("cache_ps") && config["cache_ps"].contains("ps_type")) {
    ps_type = config["cache_ps"]["ps_type"].get<std::string>();
  }

  const std::string normalized = NormalizePSType(ps_type);
  if (normalized == "GRPC" || normalized == "BRPC") {
    return normalized;
  }
  if (normalized == "RDMA") {
    throw std::invalid_argument(
        "KVClientOp does not support RDMA directly. Use src/ps/rdma clients "
        "or add a dedicated BasePSClient adapter.");
  }

  throw std::invalid_argument("Unknown ps_type for KVClientOp: " + ps_type);
}

inline json ResolveFrameworkClientConfig(const json& config) {
  if (config.contains("client")) {
    return config["client"];
  }
  return json{{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}};
}

inline BasePSClient* CreateFrameworkPSClient(const json& config) {
  const std::string ps_type  = ResolveFrameworkPSType(config);
  const json client_config   = ResolveFrameworkClientConfig(config);
  const std::string type_key = (ps_type == "BRPC") ? "brpc" : "grpc";

  BasePSClient* client =
      base::Factory<BasePSClient, json>::NewInstance(type_key, client_config);
  if (client != nullptr) {
    return client;
  }

  if (ps_type == "GRPC") {
    return new GRPCParameterClient(client_config);
  }

  if (ps_type == "BRPC") {
    return new BRPCParameterClient(client_config);
  }

  throw std::runtime_error(
      "Failed to create framework PS client for type " + ps_type);
}

} // namespace recstore
