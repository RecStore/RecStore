#pragma once

// Parse-only helpers for cache_ps.checkpoint. They stay header-inline so
// lightweight tests can use them without linking the engine.

#include <stdexcept>
#include <string>

#include "base/json.h"

namespace recstore {

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
