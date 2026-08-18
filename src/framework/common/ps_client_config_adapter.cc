#include "framework/common/ps_client_config_adapter.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace recstore {

std::string NormalizePSType(std::string ps_type) {
  std::transform(
      ps_type.begin(), ps_type.end(), ps_type.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
      });
  return ps_type;
}

PSClientType ResolveFrameworkPSClientType(const json& config) {
  std::string ps_type = "GRPC";
  if (config.contains("cache_ps") && config["cache_ps"].contains("ps_type")) {
    ps_type = config["cache_ps"]["ps_type"].get<std::string>();
  }

  const std::string normalized = NormalizePSType(ps_type);
  if (normalized == "GRPC") {
    return PSClientType::kGrpc;
  }
  if (normalized == "BRPC") {
    return PSClientType::kBrpc;
  }
  if (normalized == "RDMA") {
    return PSClientType::kRdma;
  }
  if (normalized == "LOCAL_SHM") {
    return PSClientType::kLocalShm;
  }

  throw std::invalid_argument("Unknown ps_type for KVClientOp: " + ps_type);
}

json ResolveFrameworkDistributedClientConfig(const json& config) {
  json resolved = json::object();

  const json* distributed_client = nullptr;
  if (config.contains("distributed_client") &&
      config["distributed_client"].is_object()) {
    distributed_client = &config["distributed_client"];
    resolved           = *distributed_client;
  }

  const json* cache_ps = nullptr;
  if (config.contains("cache_ps") && config["cache_ps"].is_object()) {
    cache_ps = &config["cache_ps"];
  }

  const auto copy_field_if_missing = [&](const char* field) {
    if (!resolved.contains(field) && cache_ps != nullptr &&
        cache_ps->contains(field)) {
      resolved[field] = (*cache_ps)[field];
    }
  };

  copy_field_if_missing("servers");
  copy_field_if_missing("num_shards");
  copy_field_if_missing("hash_method");
  copy_field_if_missing("max_keys_per_request");

  if (!resolved.contains("num_shards") && resolved.contains("servers") &&
      resolved["servers"].is_array()) {
    resolved["num_shards"] = resolved["servers"].size();
  }

  return resolved;
}

bool HasFrameworkDistributedClientConfig(const json& config) {
  const json distributed_client =
      ResolveFrameworkDistributedClientConfig(config);
  return distributed_client.contains("servers") &&
         distributed_client["servers"].is_array() &&
         !distributed_client["servers"].empty() &&
         distributed_client.contains("num_shards") &&
         distributed_client["num_shards"].is_number_integer();
}

json ResolveFrameworkPSClientTransportConfig(const json& config) {
  const PSClientType ps_type = ResolveFrameworkPSClientType(config);
  if (ps_type == PSClientType::kLocalShm) {
    if (config.contains("local_shm")) {
      return config["local_shm"];
    }
    return json{{"region_name", "recstore_local_ps"},
                {"slot_count", 64},
                {"slot_buffer_bytes", 8 * 1024 * 1024},
                {"client_timeout_ms", 30000}};
  }

  if (config.contains("client")) {
    return config["client"];
  }
  return json{{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 0}};
}

PSClientCreateOptions
ResolvePSClientOptionsFromFrameworkConfig(const json& config) {
  json raw_config = config;
  const PSClientType type = ResolveFrameworkPSClientType(config);
  const json transport_config = ResolveFrameworkPSClientTransportConfig(config);

  json distributed_client = ResolveFrameworkDistributedClientConfig(config);
  const bool has_servers = distributed_client.contains("servers") &&
                           distributed_client["servers"].is_array() &&
                           !distributed_client["servers"].empty();
  if (!has_servers &&
      (type == PSClientType::kGrpc || type == PSClientType::kBrpc)) {
    // Single-server deployment: synthesize a one-shard distributed_client so
    // grpc/brpc always route through the sharded client (num_shards == 1).
    distributed_client = {
        {"servers",
         json::array({{{"host", transport_config.value("host", "127.0.0.1")},
                       {"port", transport_config.value("port", 15000)},
                       {"shard", transport_config.value("shard", 0)}}})},
        {"num_shards", 1},
    };
  }
  if (!distributed_client.empty()) {
    raw_config["distributed_client"] = distributed_client;
  }

  return PSClientCreateOptions{.type             = type,
                               .transport_config = transport_config,
                               .raw_config       = raw_config};
}

} // namespace recstore
