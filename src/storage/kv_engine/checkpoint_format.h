#pragma once

// Byte layout of a KV engine checkpoint shard. Shared by the engine that writes
// it and by offline tools that rewrite it (checkpoint_migrate), so the two
// copies of the format cannot drift apart.
//
// v3 shard layout, little endian:
//
//   char     magic[8]          "RSKVCP01"
//   uint32_t version           3
//   uint64_t metadata_size
//   uint64_t record_count
//   uint32_t layout_strategy   value store layout identity (v3 only)
//   uint64_t layout_checksum   value store layout identity (v3 only)
//   char     metadata[metadata_size]   server-generated JSON
//   record_count x {
//     uint64_t key
//     uint64_t value_size
//     char     value[value_size]
//   }
//   uint64_t checksum
//
// The checksum folds the header, the metadata, and every key/value pair, in
// that order, through `storage/index/utils/hash.h::xxhash`, seeded with
// kChecksumSeed. Empty byte ranges are skipped.

#include <array>
#include <cstdint>

namespace recstore::checkpoint_format {

inline constexpr std::array<char, 8> kMagic = {
    'R', 'S', 'K', 'V', 'C', 'P', '0', '1'};
inline constexpr uint32_t kVersion = 3;
// v2 predates the value store layout identity and is only loadable by engines
// with the plain layout; checkpoint_migrate refuses it.
inline constexpr uint32_t kLegacyVersion = 2;
inline constexpr uint64_t kChecksumSeed  = 0x9e3779b97f4a7c15ULL;

// Name of the committed-manifest file inside a checkpoint version directory.
inline constexpr const char* kManifestName  = "manifest.json";
inline constexpr int kManifestSchemaVersion = 1;

} // namespace recstore::checkpoint_format
