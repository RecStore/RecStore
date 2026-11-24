#pragma once

// Offline migration of KV engine checkpoints written before the optimizer key
// layout carried `optimizer_layout` in its metadata.
//
// A shard is a header, one metadata JSON, a list of (key, value) records, and a
// trailing checksum; the metadata length sits in the header and the metadata
// and every record feed the checksum, so migrating a shard rewrites the file.
// `storage/kv_engine/checkpoint_format.h` holds the byte layout.
//
// Two things change, and only the second one is obvious:
//
//   1. The metadata gains `optimizer_layout`.
//   2. AdamW's keys move: the old layout tagged them with the bare role
//      (0 = parameter, 1 = first moment, 2 = second moment) and stored the step
//      under the sentinel id with the first-moment tag (`AdamW::kStepKey`). The
//      new layout uses `table_id * 4 + role` with the step at role 3.
//
// The client picks a shard with `hash(key) % num_shards`, so moving a key also
// moves the shard that owns it: the migration repartitions every record instead
// of rewriting shard files one by one. The AdamW step scalar is not routed by
// the client (each server keeps its own copy under a server-local key), so one
// step row is synthesised per output shard instead of being moved.
//
// SGD / RowWiseAdagrad already used table-tagged keys, so their keys and hashes
// are unchanged and the repartition puts every record back where it came from.
//
// The AdamW mapping is only meaningful when a single table wrote the shard.
// Builds before #411 could register a second AdamW table, and those rows were
// already overwritten in the source checkpoint; the tool cannot detect that
// from the file, so the caller has to know the run. Per-shard step values must
// agree, otherwise the source shards were updated a different number of times
// and the merged counter would mis-correct at least one of them.

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace recstore::checkpoint_migrate {

// Tensor counts of the server-side optimizers. `optimizer.h` owns the values;
// test_checkpoint_migrate asserts these stay equal.
inline constexpr int kSgdTensorsPerTable            = 1;
inline constexpr int kRowWiseAdagradTensorsPerTable = 2;
inline constexpr int kAdamWTensorsPerTable          = 4;
// Key id of AdamW's persisted step scalar, tagged with role 3.
inline constexpr uint64_t kAdamWStepKeyId =
    (std::numeric_limits<uint64_t>::max() >> 8);

enum class Optimizer { kSgd, kRowWiseAdagrad, kAdamW };

// Must match `distributed_client.hash_method` of the run that will read the
// migrated checkpoint.
enum class HashMethod { kCityHash, kSimpleMod };

struct Options {
  Optimizer optimizer = Optimizer::kSgd;
  // AdamW only: the backend table_id the old shard belongs to. Old tags did not
  // record it, so it cannot be recovered from the file.
  uint64_t table_id = 0;
  // SGD / RowWiseAdagrad only, optional: every key tag must decode to one of
  // these table ids.
  std::vector<uint64_t> table_ids;
  // 0 = take the shard count from the checkpoint. Override when the source
  // checkpoint is incomplete or the new run uses a different shard count.
  int num_shards  = 0;
  HashMethod hash = HashMethod::kCityHash;
  // Re-read every migrated shard and verify its checksum before the manifest is
  // committed. Costs a second I/O pass over the checkpoint.
  bool verify = true;
};

struct ShardReport {
  std::string name;
  int shard_id           = -1;
  uint64_t records       = 0;
  uint64_t step_rows     = 0;
  uint64_t bytes_read    = 0;
  uint64_t bytes_written = 0;
  // Tag histogram before and after migration, keyed by the upper key byte.
  std::map<unsigned, uint64_t> tags_before;
  std::map<unsigned, uint64_t> tags_after;
};

// Migrate one committed checkpoint version directory (the one holding
// manifest.json). Every shard is rewritten, repartitioned, and verified before
// the manifest is committed, so an interrupted run leaves a directory that
// readers ignore.
bool MigrateVersionDir(const std::string& input_dir,
                       const std::string& output_dir,
                       const Options& options,
                       std::vector<ShardReport>* reports,
                       std::string* error);

// Migrate a version directory or a root directory whose immediate
// subdirectories are versions.
bool MigrateCheckpoint(const std::string& input_path,
                       const std::string& output_path,
                       const Options& options,
                       std::vector<ShardReport>* reports,
                       std::string* error);

const char* OptimizerName(Optimizer optimizer);

} // namespace recstore::checkpoint_migrate
