#pragma once

// Startup restore helpers that need the full CachePS definition. Server
// entry points include this after cache_ps_impl.h; the parse-only helpers live
// in checkpoint_restore.h so lightweight tests can link without the engine.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "base/log.h"
#include "ps/base/cache_ps_impl.h"
#include "storage/kv_engine/checkpoint_format.h"
#include "ps/base/checkpoint_restore.h"

namespace recstore {
namespace detail {

constexpr int kCheckpointManifestSchemaVersion =
    checkpoint_format::kManifestSchemaVersion;

inline std::string CheckpointShardFileName(int shard_id) {
  return "recstore-shard-" + std::to_string(shard_id) + ".bin";
}

inline std::filesystem::path ManifestPath(const std::filesystem::path& dir) {
  return dir / checkpoint_format::kManifestName;
}

} // namespace detail

// Restore this shard from <restore_dir>/manifest.json before the server starts
// listening. Returns true when a checkpoint was loaded. A missing manifest with
// on_missing=start_empty returns false with an empty `error`; any real failure
// returns false with `error` filled.
inline bool RestoreCheckpointOnStartup(
    CachePS* cache_ps,
    int shard_id,
    const CheckpointRestoreConfig& config,
    std::string* error) {
  if (config.restore_dir.empty()) {
    return false;
  }
  const std::filesystem::path root(config.restore_dir);

  // restore_dir may name the checkpoint root (newest committed version wins)
  // or a single version directory.
  std::filesystem::path directory     = root;
  std::filesystem::path manifest_path = detail::ManifestPath(root);
  std::error_code exists_error;
  if (!std::filesystem::exists(manifest_path, exists_error)) {
    std::optional<std::filesystem::path> newest;
    int64_t newest_step = -1;
    std::filesystem::file_time_type newest_time{};
    std::error_code iter_error;
    for (const auto& entry :
         std::filesystem::directory_iterator(root, iter_error)) {
      if (iter_error || !entry.is_directory()) {
        continue;
      }
      const std::string name = entry.path().filename().string();
      if (name.empty() || name.front() == '.') {
        continue;
      }
      const auto candidate = detail::ManifestPath(entry.path());
      if (!std::filesystem::exists(candidate, exists_error)) {
        continue;
      }
      try {
        std::ifstream probe(candidate);
        json probe_manifest;
        probe >> probe_manifest;
        const int64_t step = probe_manifest.value("step", int64_t{0});
        std::error_code time_error;
        const auto written =
            std::filesystem::last_write_time(candidate, time_error);
        if (!newest.has_value() || step > newest_step ||
            (step == newest_step && written > newest_time)) {
          newest      = entry.path();
          newest_step = step;
          newest_time = written;
        }
      } catch (const std::exception&) {
        LOG(WARNING) << "Ignoring unreadable checkpoint manifest " << candidate;
      }
    }
    if (!newest.has_value()) {
      const std::string message =
          "no committed checkpoint under: " + root.string();
      if (config.fail_if_missing) {
        *error = message;
        return false;
      }
      LOG(WARNING) << message << "; starting shard " << shard_id << " empty";
      return false;
    }
    directory     = *newest;
    manifest_path = detail::ManifestPath(directory);
  }

  std::ifstream input(manifest_path);
  if (!input) {
    *error = "checkpoint manifest is not readable: " + manifest_path.string();
    return false;
  }

  try {
    json manifest;
    input >> manifest;
    if (!manifest.is_object() ||
        manifest.value("schema_version", 0) !=
            detail::kCheckpointManifestSchemaVersion ||
        !manifest.contains("checkpoint_id") ||
        !manifest["checkpoint_id"].is_string() ||
        !manifest.contains("identity") || !manifest["identity"].is_object() ||
        !manifest.contains("files") || !manifest["files"].is_array() ||
        manifest["files"].empty()) {
      *error = "invalid checkpoint manifest: " + manifest_path.string();
      return false;
    }

    const std::string file_name = detail::CheckpointShardFileName(shard_id);
    const json* shard_entry     = nullptr;
    for (const auto& entry : manifest["files"]) {
      if (entry.is_object() && entry.value("name", "") == file_name) {
        shard_entry = &entry;
        break;
      }
    }
    if (shard_entry == nullptr || !shard_entry->contains("size") ||
        !(*shard_entry)["size"].is_number_integer()) {
      *error = "checkpoint does not contain shard " + std::to_string(shard_id) +
               ": " + manifest_path.string();
      return false;
    }

    const std::filesystem::path file_path = directory / file_name;
    std::error_code size_error;
    const auto actual_size = std::filesystem::file_size(file_path, size_error);
    const auto expected_size = (*shard_entry)["size"].get<std::uint64_t>();
    if (size_error || actual_size != expected_size) {
      *error = "checkpoint shard file is missing or truncated: " +
               file_path.string();
      return false;
    }

    // Same binding as the SAVE/LOAD RPC path, so a shard written under a
    // different optimizer key layout fails here instead of restoring with its
    // optimizer slots treated as missing rows.
    const std::string metadata = BindCheckpointMetadata(
        json{{"checkpoint_id", manifest["checkpoint_id"]},
             {"identity", manifest["identity"]}}
            .dump(),
        shard_id);
    if (!cache_ps->LoadCheckpoint(file_path.string(), metadata)) {
      *error = "checkpoint load failed: " + file_path.string();
      return false;
    }
    LOG(INFO) << "Restored shard " << shard_id << " from " << file_path
              << " (checkpoint " << manifest["checkpoint_id"].get<std::string>()
              << ", manifest step " << manifest.value("step", 0) << ")";
    return true;
  } catch (const json::exception& e) {
    *error = std::string("invalid checkpoint manifest: ") + e.what();
    return false;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

// Convenience for the server entry points: parse cache_ps.checkpoint, restore
// this shard, and LOG(FATAL) when a configured restore fails. Returns true when
// a checkpoint was loaded.
inline bool RestoreCheckpointFromServerConfig(
    CachePS* cache_ps, int shard_id, const nlohmann::json& cache_ps_config) {
  CheckpointRestoreConfig config;
  try {
    config = ParseCheckpointRestoreConfig(cache_ps_config);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Invalid cache_ps.checkpoint configuration: " << e.what();
  }
  std::string error;
  const bool restored =
      RestoreCheckpointOnStartup(cache_ps, shard_id, config, &error);
  if (!error.empty()) {
    LOG(FATAL) << "Failed to restore shard " << shard_id
               << " from checkpoint: " << error;
  }
  return restored;
}

} // namespace recstore
