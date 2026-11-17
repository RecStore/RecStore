#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include "base/json.h"

namespace recstore {

struct ShardEndpoint {
  std::string host;
  int port  = 0;
  int shard = 0;
};

inline std::string FormatShardEndpoint(const ShardEndpoint& endpoint) {
  return endpoint.host + ":" + std::to_string(endpoint.port);
}

// Resolves the listen endpoint for a single-shard server. Returns nullopt when
// cache_ps.servers is absent, so the caller can fall back to the legacy
// default address. Throws when servers is present but the requested shard (or
// shard 0 when no local shard id is given) is missing or malformed.
inline std::optional<ShardEndpoint>
SelectSingleShardEndpoint(const nlohmann::json& cache_ps_config,
                          const std::optional<int>& local_shard_id) {
  if (!cache_ps_config.contains("servers") ||
      !cache_ps_config["servers"].is_array() ||
      cache_ps_config["servers"].empty()) {
    return std::nullopt;
  }

  const int wanted = local_shard_id.value_or(0);
  for (const auto& server : cache_ps_config["servers"]) {
    if (!server.contains("shard") || !server["shard"].is_number_integer() ||
        server["shard"].get<int>() != wanted) {
      continue;
    }
    if (!server.contains("host") || !server["host"].is_string() ||
        !server.contains("port") || !server["port"].is_number_integer()) {
      throw std::runtime_error(
          "cache_ps.servers entry for shard " + std::to_string(wanted) +
          " must have a string host and an integer port");
    }
    ShardEndpoint endpoint;
    endpoint.host  = server["host"].get<std::string>();
    endpoint.port  = server["port"].get<int>();
    endpoint.shard = wanted;
    return endpoint;
  }

  throw std::runtime_error(
      "cache_ps.servers has no entry for shard " + std::to_string(wanted));
}

} // namespace recstore
