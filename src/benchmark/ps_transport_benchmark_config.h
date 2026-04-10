#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "base/json.h"

inline std::string NormalizeBenchmarkTransport(std::string transport) {
  std::transform(
      transport.begin(),
      transport.end(),
      transport.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (transport == "RDMA" || transport == "GRPC" || transport == "BRPC") {
    return transport;
  }
  throw std::invalid_argument("Unsupported transport: " + transport);
}

inline nlohmann::json BuildRpcBenchmarkConfig(
    const std::string& transport, const std::string& host, int port) {
  return {
      {"cache_ps", {{"ps_type", NormalizeBenchmarkTransport(transport)}}},
      {"client", {{"host", host}, {"port", port}, {"shard", 0}}},
  };
}

inline bool BenchmarkRequiresManagedReceiveBuffer(const std::string& transport,
                                                  int num_shards) {
  return NormalizeBenchmarkTransport(transport) == "RDMA" && num_shards == 1;
}

inline bool BenchmarkUsesVectorGet(const std::string& transport) {
  return NormalizeBenchmarkTransport(transport) == "BRPC";
}
