#pragma once

// Parse-only helpers for cache_ps.checkpoint. They stay header-inline so
// lightweight tests can use them without linking the engine.

#include <stdexcept>
#include <string>

#include "base/json.h"

namespace recstore {

// Key layout the optimizer writes into the KV engine. Bump it whenever a tensor
// tag or key encoding changes: a shard written under another layout is refused
// at load time, because its optimizer slots would otherwise be read as missing
// rows and re-initialised from zero.
inline constexpr int kCheckpointOptimizerLayout = 1;

// Validate client-supplied checkpoint metadata and bind it to this shard. The
// layout marker is server-authoritative: a client cannot claim a layout, and a
// shard without the marker predates the current key layout.
inline std::string
BindCheckpointMetadata(const std::string& metadata, int shard_id) {
  nlohmann::json parsed = nlohmann::json::parse(metadata);
  if (!parsed.is_object() || !parsed.contains("identity") ||
      !parsed["identity"].is_object() || !parsed.contains("checkpoint_id") ||
      !parsed["checkpoint_id"].is_string() ||
      parsed["checkpoint_id"].get<std::string>().empty()) {
    throw std::invalid_argument(
        "checkpoint metadata requires identity and non-empty checkpoint_id");
  }
  if (shard_id < 0) {
    throw std::invalid_argument("checkpoint shard_id must be non-negative");
  }
  parsed["optimizer_layout"] = kCheckpointOptimizerLayout;
  parsed["shard_id"]         = shard_id;
  return parsed.dump();
}

struct CheckpointRestoreConfig {
  // Directory holding a committed manifest.json plus one file per shard.
  // Empty disables startup restore.
  std::string restore_dir;
  // false (on_missing=start_empty) starts an empty shard when no manifest is
  // present; true (on_missing=fail) treats that as a startup error.
  bool fail_if_missing = true;
};

// Parse the optional cache_ps.checkpoint block. Throws std::invalid_argument
// on a malformed block so a typo cannot silently disable restore.
inline CheckpointRestoreConfig
ParseCheckpointRestoreConfig(const nlohmann::json& cache_ps) {
  CheckpointRestoreConfig config;
  if (!cache_ps.is_object() || !cache_ps.contains("checkpoint")) {
    return config;
  }
  const auto& node = cache_ps.at("checkpoint");
  if (!node.is_object()) {
    throw std::invalid_argument("cache_ps.checkpoint must be an object");
  }
  if (node.contains("restore_dir")) {
    if (!node["restore_dir"].is_string()) {
      throw std::invalid_argument(
          "cache_ps.checkpoint.restore_dir must be a string");
    }
    config.restore_dir = node["restore_dir"].get<std::string>();
  }
  if (node.contains("on_missing")) {
    if (!node["on_missing"].is_string()) {
      throw std::invalid_argument(
          "cache_ps.checkpoint.on_missing must be a string");
    }
    const std::string value = node["on_missing"].get<std::string>();
    if (value == "fail") {
      config.fail_if_missing = true;
    } else if (value == "start_empty") {
      config.fail_if_missing = false;
    } else {
      throw std::invalid_argument(
          "cache_ps.checkpoint.on_missing must be fail or start_empty");
    }
  }
  return config;
}

} // namespace recstore
