#include "ps/checkpoint/checkpoint_migrate.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "base/hash.h"
#include "base/json.h"
#include "ps/base/checkpoint_restore.h"
#include "storage/index/utils/hash.h"
#include "storage/kv_engine/checkpoint_format.h"

namespace recstore::checkpoint_migrate {
namespace {

namespace fs = std::filesystem;

constexpr int kTagShift             = 64 - 8;
constexpr uint64_t kIdMask          = (1ULL << kTagShift) - 1;
constexpr uint64_t kMaxMetadataSize = 16 * 1024 * 1024;
constexpr int64_t kMaxShardId       = 999999999;

struct ShardHeader {
  uint32_t version         = 0;
  uint64_t metadata_size   = 0;
  uint64_t record_count    = 0;
  uint32_t layout_strategy = 0;
  uint64_t layout_checksum = 0;
  std::string metadata;
};

// Same fold order as KVEngineComposite's UpdateChecksum: empty ranges are
// skipped, everything else folds into the running value.
class Checksum {
public:
  void Add(const void* data, size_t size) {
    if (size != 0) {
      value_ = xxhash(data, size, value_);
    }
  }

  template <typename T>
  void AddPod(const T& value) {
    Add(&value, sizeof(value));
  }

  uint64_t value() const { return value_; }

private:
  uint64_t value_ = checkpoint_format::kChecksumSeed;
};

[[noreturn]] void Throw(const std::string& message) {
  throw std::runtime_error(message);
}

// "recstore-shard-<N>.bin" -> N, or -1 when the name is not a shard name.
int ShardIdOf(const std::string& name) {
  const std::string prefix = "recstore-shard-";
  const std::string suffix = ".bin";
  if (name.size() <= prefix.size() + suffix.size() ||
      name.compare(0, prefix.size(), prefix) != 0 ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return -1;
  }
  const std::string digits =
      name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (digits.empty() || digits.size() > 9) {
    return -1;
  }
  for (const char c : digits) {
    if (c < '0' || c > '9') {
      return -1;
    }
  }
  const int64_t value = std::stoll(digits);
  return value <= kMaxShardId ? static_cast<int>(value) : -1;
}

std::string ShardFileName(int shard_id) {
  return "recstore-shard-" + std::to_string(shard_id) + ".bin";
}

ShardHeader ReadHeader(std::ifstream& input, const std::string& path) {
  ShardHeader header;
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != checkpoint_format::kMagic) {
    Throw("not a RecStore checkpoint shard: " + path);
  }
  input.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
  if (!input) {
    Throw("truncated checkpoint header: " + path);
  }
  if (header.version != checkpoint_format::kVersion) {
    Throw("checkpoint format v" + std::to_string(header.version) +
          " is not supported (only v" +
          std::to_string(checkpoint_format::kVersion) + "): " + path);
  }
  input.read(reinterpret_cast<char*>(&header.metadata_size),
             sizeof(header.metadata_size));
  input.read(reinterpret_cast<char*>(&header.record_count),
             sizeof(header.record_count));
  input.read(reinterpret_cast<char*>(&header.layout_strategy),
             sizeof(header.layout_strategy));
  input.read(reinterpret_cast<char*>(&header.layout_checksum),
             sizeof(header.layout_checksum));
  if (!input) {
    Throw("truncated checkpoint header: " + path);
  }
  if (header.metadata_size > kMaxMetadataSize) {
    Throw("checkpoint metadata is implausibly large: " + path);
  }
  header.metadata.resize(static_cast<size_t>(header.metadata_size));
  input.read(header.metadata.data(),
             static_cast<std::streamsize>(header.metadata.size()));
  if (!input) {
    Throw("truncated checkpoint metadata: " + path);
  }
  return header;
}

void FoldHeader(Checksum* checksum, const ShardHeader& header) {
  checksum->Add(
      checkpoint_format::kMagic.data(), checkpoint_format::kMagic.size());
  checksum->AddPod(header.version);
  checksum->AddPod(header.metadata_size);
  checksum->AddPod(header.record_count);
  checksum->AddPod(header.layout_strategy);
  checksum->AddPod(header.layout_checksum);
  checksum->Add(header.metadata.data(), header.metadata.size());
}

void WriteHeader(std::ofstream& output, const ShardHeader& header) {
  output.write(checkpoint_format::kMagic.data(),
               static_cast<std::streamsize>(checkpoint_format::kMagic.size()));
  output.write(reinterpret_cast<const char*>(&header.version),
               sizeof(header.version));
  output.write(reinterpret_cast<const char*>(&header.metadata_size),
               sizeof(header.metadata_size));
  output.write(reinterpret_cast<const char*>(&header.record_count),
               sizeof(header.record_count));
  output.write(reinterpret_cast<const char*>(&header.layout_strategy),
               sizeof(header.layout_strategy));
  output.write(reinterpret_cast<const char*>(&header.layout_checksum),
               sizeof(header.layout_checksum));
  output.write(header.metadata.data(),
               static_cast<std::streamsize>(header.metadata.size()));
}

uint64_t TensorsPerTable(Optimizer optimizer) {
  switch (optimizer) {
  case Optimizer::kSgd:
    return kSgdTensorsPerTable;
  case Optimizer::kRowWiseAdagrad:
    return kRowWiseAdagradTensorsPerTable;
  case Optimizer::kAdamW:
    return kAdamWTensorsPerTable;
  }
  Throw("unknown optimizer");
}

// Maps one source key into the current layout and validates it against the
// declared optimizer. `step_rows` counts AdamW step scalars, so a shard holding
// two of them is refused instead of silently merged.
uint64_t MapKey(uint64_t key, const Options& options, uint64_t* step_rows) {
  const uint64_t tag = key >> kTagShift;
  const uint64_t id  = key & kIdMask;
  if (options.optimizer == Optimizer::kAdamW) {
    if (tag >= static_cast<uint64_t>(kAdamWTensorsPerTable)) {
      Throw("key tag " + std::to_string(tag) +
            " is not a pre-#400 AdamW role tag (0..2); the shard was not "
            "written by AdamW");
    }
    const bool is_step = tag == 1 && id == kAdamWStepKeyId;
    if (is_step && ++*step_rows > 1) {
      Throw("shard holds more than one AdamW step scalar");
    }
    const uint64_t role = is_step ? 3 : tag;
    const uint64_t target =
        options.table_id * static_cast<uint64_t>(kAdamWTensorsPerTable) + role;
    if (target > 0xFF) {
      Throw("table_id " + std::to_string(options.table_id) +
            " does not fit the 8-bit key tag");
    }
    return (target << kTagShift) | id;
  }

  if (id == kAdamWStepKeyId) {
    Throw(std::string("shard holds an AdamW step scalar, so it was not written "
                      "by ") +
          OptimizerName(options.optimizer) +
          "; rerun with --optimizer AdamW --table-id <id>");
  }
  if (!options.table_ids.empty()) {
    const uint64_t tensors  = TensorsPerTable(options.optimizer);
    const uint64_t table_id = tag / tensors;
    if (std::find(
            options.table_ids.begin(), options.table_ids.end(), table_id) ==
        options.table_ids.end()) {
      Throw("key tag " + std::to_string(tag) + " decodes to table_id " +
            std::to_string(table_id) + ", which is not in --table-ids");
    }
  }
  return key;
}

bool IsStepKey(uint64_t key) {
  return (key >> kTagShift) == 1 && (key & kIdMask) == kAdamWStepKeyId;
}

uint64_t NewStepKey(uint64_t table_id) {
  return ((table_id * static_cast<uint64_t>(kAdamWTensorsPerTable) + 3)
          << kTagShift) |
         kAdamWStepKeyId;
}

// The client routes by the parameter key: `Init` returns the parameter tag and
// the Python client encodes every id with it. Moment and accumulator rows are
// written by the server into the engine that received the update, so they must
// follow the parameter row rather than hash by their own tag.
uint64_t ParamKeyOf(uint64_t mapped_key, const Options& options) {
  const uint64_t tensors = TensorsPerTable(options.optimizer);
  const uint64_t tag     = mapped_key >> kTagShift;
  const uint64_t id      = mapped_key & kIdMask;
  return (((tag / tensors) * tensors) << kTagShift) | id;
}

int TargetShard(uint64_t mapped_key, const Options& options, int num_shards) {
  const uint64_t key    = ParamKeyOf(mapped_key, options);
  const uint64_t shards = static_cast<uint64_t>(num_shards);
  if (options.hash == HashMethod::kSimpleMod) {
    return static_cast<int>(key % shards);
  }
  return static_cast<int>(GetHash(key) % shards);
}

std::string StripShardId(const std::string& metadata, const std::string& path) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(metadata);
  } catch (const nlohmann::json::exception& e) {
    Throw("checkpoint metadata is not JSON (" + std::string(e.what()) +
          "): " + path);
  }
  if (!parsed.is_object() || !parsed.contains("checkpoint_id") ||
      !parsed.contains("identity")) {
    Throw("checkpoint metadata lacks checkpoint_id or identity: " + path);
  }
  if (parsed.contains("optimizer_layout")) {
    const auto layout = parsed["optimizer_layout"];
    Throw("shard already records optimizer_layout=" +
          (layout.is_number_integer() ? std::to_string(layout.get<int64_t>())
                                      : layout.dump()) +
          ", nothing to migrate: " + path);
  }
  parsed.erase("shard_id");
  return parsed.dump();
}

std::string
MigrateMetadata(const std::string& canonical_without_shard_id, int shard_id) {
  nlohmann::json parsed = nlohmann::json::parse(canonical_without_shard_id);
  parsed["optimizer_layout"] = kCheckpointOptimizerLayout;
  parsed["shard_id"]         = shard_id;
  return parsed.dump();
}

void ReadValue(std::ifstream& input,
               uint64_t value_size,
               std::string* value,
               uint64_t* bytes_read) {
  value->resize(static_cast<size_t>(value_size));
  input.read(value->data(), static_cast<std::streamsize>(value_size));
  if (!input) {
    Throw("truncated checkpoint record value");
  }
  if (bytes_read != nullptr) {
    *bytes_read += value_size;
  }
}

// Re-reads a migrated shard and checks metadata, record count, and checksum so
// a bad write cannot look like a good migration.
void VerifyShard(
    const std::string& path, const std::string& metadata, uint64_t records) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Throw("cannot reopen migrated shard: " + path);
  }
  const ShardHeader header = ReadHeader(input, path);
  if (header.metadata != metadata) {
    Throw("migrated metadata did not round trip: " + path);
  }
  if (header.record_count != records) {
    Throw("migrated record count changed: " + path);
  }
  Checksum checksum;
  FoldHeader(&checksum, header);
  for (uint64_t index = 0; index < header.record_count; ++index) {
    uint64_t key        = 0;
    uint64_t value_size = 0;
    input.read(reinterpret_cast<char*>(&key), sizeof(key));
    input.read(reinterpret_cast<char*>(&value_size), sizeof(value_size));
    if (!input) {
      Throw("migrated record header is truncated: " + path);
    }
    checksum.AddPod(key);
    checksum.AddPod(value_size);
    std::string value;
    ReadValue(input, value_size, &value, nullptr);
    checksum.Add(value.data(), value.size());
  }
  uint64_t stored = 0;
  input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (!input || stored != checksum.value()) {
    Throw("migrated checksum mismatch: " + path);
  }
}

nlohmann::json ReadManifest(const fs::path& dir) {
  const fs::path path = dir / checkpoint_format::kManifestName;
  std::ifstream input(path);
  if (!input) {
    Throw("checkpoint manifest is missing: " + path.string());
  }
  nlohmann::json manifest;
  try {
    input >> manifest;
  } catch (const nlohmann::json::exception& e) {
    Throw("checkpoint manifest is not JSON (" + std::string(e.what()) +
          "): " + path.string());
  }
  if (!manifest.is_object() ||
      manifest.value("schema_version", 0) !=
          checkpoint_format::kManifestSchemaVersion ||
      !manifest.contains("checkpoint_id") || !manifest.contains("identity") ||
      !manifest.contains("files") || !manifest["files"].is_array() ||
      manifest["files"].empty()) {
    Throw("invalid checkpoint manifest: " + path.string());
  }
  return manifest;
}

void WriteManifest(const fs::path& dir, const nlohmann::json& manifest) {
  const fs::path path = dir / checkpoint_format::kManifestName;
  const fs::path temp = path.string() + ".tmp";
  {
    std::ofstream output(temp, std::ios::trunc);
    if (!output) {
      Throw("cannot write migrated manifest: " + temp.string());
    }
    // Match recstore.checkpoint's on-disk style: sorted keys, 2-space indent.
    output << manifest.dump(2) << "\n";
    output.flush();
    if (!output.good()) {
      Throw("failed to write migrated manifest: " + temp.string());
    }
  }
  std::error_code error;
  fs::rename(temp, path, error);
  if (error) {
    fs::remove(temp, error);
    Throw("failed to publish migrated manifest: " + path.string());
  }
}

struct SourceShard {
  int id = -1;
  std::string name;
  ShardHeader header;
};

struct TargetWriter {
  int shard_id = -1;
  fs::path temp_path;
  fs::path final_path;
  std::ofstream output;
  Checksum checksum;
  ShardHeader header;
  ShardReport report;
};

// First pass: count how many records land in each output shard and collect the
// AdamW step scalars. Values are skipped, not read.
void CountTargets(const fs::path& input_dir,
                  const std::vector<SourceShard>& sources,
                  const Options& options,
                  int num_shards,
                  std::vector<uint64_t>* counts,
                  std::vector<std::string>* step_values) {
  counts->assign(static_cast<size_t>(num_shards), 0);
  for (const auto& source : sources) {
    const fs::path path = input_dir / source.name;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      Throw("cannot open checkpoint shard: " + path.string());
    }
    const ShardHeader header = ReadHeader(input, path.string());
    uint64_t step_rows       = 0;
    for (uint64_t index = 0; index < header.record_count; ++index) {
      uint64_t key        = 0;
      uint64_t value_size = 0;
      input.read(reinterpret_cast<char*>(&key), sizeof(key));
      input.read(reinterpret_cast<char*>(&value_size), sizeof(value_size));
      if (!input) {
        Throw("truncated checkpoint record header: " + path.string());
      }
      const bool is_step =
          options.optimizer == Optimizer::kAdamW && IsStepKey(key);
      if (is_step) {
        if (++step_rows > 1) {
          Throw("shard holds more than one AdamW step scalar: " +
                path.string());
        }
        if (value_size != sizeof(float)) {
          Throw("AdamW step scalar is not a float: " + path.string());
        }
        std::string value;
        ReadValue(input, value_size, &value, nullptr);
        step_values->push_back(value);
        continue;
      }
      const uint64_t mapped = MapKey(key, options, &step_rows);
      ++(*counts)[static_cast<size_t>(
          TargetShard(mapped, options, num_shards))];
      input.seekg(static_cast<std::streamoff>(value_size), std::ios::cur);
      if (!input) {
        Throw("truncated checkpoint record value: " + path.string());
      }
    }
    uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input) {
      Throw("checkpoint checksum is missing: " + path.string());
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
      Throw("trailing bytes after checkpoint checksum: " + path.string());
    }
  }
}

void WriteTargets(
    const fs::path& input_dir,
    const fs::path& output_dir,
    const std::vector<SourceShard>& sources,
    const std::string& canonical_metadata,
    const Options& options,
    int num_shards,
    const std::vector<uint64_t>& counts,
    const std::string& step_value,
    std::vector<TargetWriter>* writers) {
  writers->clear();
  writers->reserve(static_cast<size_t>(num_shards));
  for (int shard_id = 0; shard_id < num_shards; ++shard_id) {
    TargetWriter writer;
    writer.shard_id   = shard_id;
    writer.final_path = output_dir / ShardFileName(shard_id);
    writer.temp_path  = writer.final_path.string() + ".tmp";
    if (fs::exists(writer.final_path)) {
      Throw("refusing to overwrite existing file: " +
            writer.final_path.string());
    }
    writer.header               = sources.front().header;
    writer.header.metadata      = MigrateMetadata(canonical_metadata, shard_id);
    writer.header.metadata_size = writer.header.metadata.size();
    writer.header.record_count  = counts[static_cast<size_t>(shard_id)];

    fs::remove(writer.temp_path);
    writer.output.open(writer.temp_path, std::ios::binary | std::ios::trunc);
    if (!writer.output) {
      Throw("cannot create migrated shard: " + writer.temp_path.string());
    }
    WriteHeader(writer.output, writer.header);
    FoldHeader(&writer.checksum, writer.header);
    writer.report.name     = writer.final_path.filename().string();
    writer.report.shard_id = shard_id;
    writers->push_back(std::move(writer));
  }

  for (const auto& source : sources) {
    const fs::path path = input_dir / source.name;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      Throw("cannot open checkpoint shard: " + path.string());
    }
    const ShardHeader header = ReadHeader(input, path.string());
    Checksum source_checksum;
    FoldHeader(&source_checksum, header);
    uint64_t step_rows = 0;
    for (uint64_t index = 0; index < header.record_count; ++index) {
      uint64_t key        = 0;
      uint64_t value_size = 0;
      input.read(reinterpret_cast<char*>(&key), sizeof(key));
      input.read(reinterpret_cast<char*>(&value_size), sizeof(value_size));
      if (!input) {
        Throw("truncated checkpoint record header: " + path.string());
      }
      std::string value;
      ReadValue(input, value_size, &value, nullptr);
      source_checksum.AddPod(key);
      source_checksum.AddPod(value_size);
      source_checksum.Add(value.data(), value.size());
      if (options.optimizer == Optimizer::kAdamW && IsStepKey(key)) {
        continue; // Re-created per output shard below.
      }
      const uint64_t mapped = MapKey(key, options, &step_rows);
      const int target      = TargetShard(mapped, options, num_shards);
      TargetWriter& writer  = (*writers)[static_cast<size_t>(target)];
      writer.checksum.AddPod(mapped);
      writer.checksum.AddPod(value_size);
      writer.checksum.Add(value.data(), value.size());
      writer.output.write(
          reinterpret_cast<const char*>(&mapped), sizeof(mapped));
      writer.output.write(
          reinterpret_cast<const char*>(&value_size), sizeof(value_size));
      writer.output.write(
          value.data(), static_cast<std::streamsize>(value.size()));
      if (!writer.output) {
        Throw("failed to write migrated shard: " + writer.temp_path.string());
      }
      ++writer.report.records;
      writer.report.bytes_written +=
          sizeof(mapped) + sizeof(value_size) + value.size();
      ++writer.report.tags_before[static_cast<unsigned>(key >> kTagShift)];
      ++writer.report.tags_after[static_cast<unsigned>(mapped >> kTagShift)];
    }
    uint64_t stored = 0;
    input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!input || stored != source_checksum.value()) {
      Throw("source checkpoint checksum mismatch (truncated or corrupted): " +
            path.string());
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
      Throw("trailing bytes after checkpoint checksum: " + path.string());
    }
  }

  if (!step_value.empty()) {
    // The step scalar is a server-local key: every shard keeps its own copy, so
    // the migration writes the common value into each output shard instead of
    // repartitioning the source rows.
    const uint64_t key = NewStepKey(options.table_id);
    for (auto& writer : *writers) {
      const uint64_t value_size = step_value.size();
      writer.checksum.AddPod(key);
      writer.checksum.AddPod(value_size);
      writer.checksum.Add(step_value.data(), step_value.size());
      writer.output.write(reinterpret_cast<const char*>(&key), sizeof(key));
      writer.output.write(
          reinterpret_cast<const char*>(&value_size), sizeof(value_size));
      writer.output.write(
          step_value.data(), static_cast<std::streamsize>(step_value.size()));
      if (!writer.output) {
        Throw("failed to write the AdamW step scalar into " +
              writer.temp_path.string());
      }
      ++writer.report.records;
      ++writer.report.step_rows;
      writer.report.bytes_written +=
          sizeof(key) + sizeof(value_size) + step_value.size();
      ++writer.report.tags_after[static_cast<unsigned>(key >> kTagShift)];
    }
  }

  for (auto& writer : *writers) {
    const uint64_t checksum = writer.checksum.value();
    writer.output.write(
        reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    writer.output.flush();
    if (!writer.output.good()) {
      Throw("failed to flush migrated shard: " + writer.temp_path.string());
    }
    writer.output.close();
    if (writer.output.fail()) {
      Throw("failed to close migrated shard: " + writer.temp_path.string());
    }
    std::error_code error;
    fs::rename(writer.temp_path, writer.final_path, error);
    if (error) {
      fs::remove(writer.temp_path, error);
      Throw("failed to publish migrated shard: " + writer.final_path.string());
    }
    writer.report.bytes_written = fs::file_size(writer.final_path);
  }
}

} // namespace

const char* OptimizerName(Optimizer optimizer) {
  switch (optimizer) {
  case Optimizer::kSgd:
    return "SGD";
  case Optimizer::kRowWiseAdagrad:
    return "RowWiseAdagrad";
  case Optimizer::kAdamW:
    return "AdamW";
  }
  return "unknown";
}

bool MigrateVersionDir(const std::string& input_dir_path,
                       const std::string& output_dir_path,
                       const Options& options,
                       std::vector<ShardReport>* reports,
                       std::string* error) {
  if (reports == nullptr || error == nullptr) {
    return false;
  }
  try {
    const fs::path input_dir(input_dir_path);
    const fs::path output_dir(output_dir_path);
    if (!fs::is_directory(input_dir)) {
      Throw("checkpoint input is not a directory: " + input_dir.string());
    }
    if (fs::exists(output_dir)) {
      Throw("refusing to overwrite existing directory: " + output_dir.string());
    }
    if (options.optimizer == Optimizer::kAdamW && options.table_id > 63) {
      Throw("--table-id must be <= 63 for the 8-bit key tag");
    }
    if (options.num_shards < 0 || options.num_shards > 4096) {
      Throw("--shards is out of range");
    }

    nlohmann::json manifest = ReadManifest(input_dir);

    std::map<int, std::string> shard_names;
    for (const auto& entry : manifest["files"]) {
      if (!entry.is_object() || !entry.contains("name") ||
          !entry["name"].is_string()) {
        Throw("invalid manifest file entry in " + input_dir.string());
      }
      const std::string name = entry["name"].get<std::string>();
      const int shard_id     = ShardIdOf(name);
      if (shard_id >= 0) {
        shard_names[shard_id] = name;
      }
    }
    if (shard_names.empty()) {
      Throw("checkpoint holds no shard files: " + input_dir.string());
    }
    const int num_shards =
        options.num_shards > 0 ? options.num_shards
                               : static_cast<int>(shard_names.size());
    for (int shard_id = 0; shard_id < num_shards; ++shard_id) {
      if (shard_names.find(shard_id) == shard_names.end()) {
        Throw("checkpoint is missing shard " + std::to_string(shard_id) +
              " of " + std::to_string(num_shards));
      }
    }
    if (static_cast<int>(shard_names.size()) != num_shards) {
      Throw("checkpoint holds " + std::to_string(shard_names.size()) +
            " shard files but " + std::to_string(num_shards) +
            " are expected; pass --shards to match the new run");
    }

    std::vector<SourceShard> sources;
    sources.reserve(shard_names.size());
    for (const auto& [shard_id, name] : shard_names) {
      const fs::path path = input_dir / name;
      std::ifstream input(path, std::ios::binary);
      if (!input) {
        Throw("cannot open checkpoint shard: " + path.string());
      }
      SourceShard source;
      source.id     = shard_id;
      source.name   = name;
      source.header = ReadHeader(input, path.string());
      source.header.metadata =
          StripShardId(source.header.metadata, path.string());
      sources.push_back(std::move(source));
    }
    for (const auto& source : sources) {
      if (source.header.metadata != sources.front().header.metadata) {
        Throw("shards disagree on checkpoint metadata: " + source.name);
      }
    }

    std::vector<uint64_t> counts;
    std::vector<std::string> step_values;
    CountTargets(
        input_dir, sources, options, num_shards, &counts, &step_values);

    std::string step_value;
    if (!step_values.empty()) {
      step_value = step_values.front();
      for (const auto& value : step_values) {
        if (value != step_value) {
          Throw("AdamW per-shard step counters disagree (" +
                std::to_string(step_values.size()) +
                " shards); the run was updated unevenly and cannot be merged");
        }
      }
      for (auto& count : counts) {
        ++count; // the synthesised step row
      }
    }

    fs::create_directories(output_dir);
    for (const auto& dir_entry : fs::directory_iterator(input_dir)) {
      const std::string name = dir_entry.path().filename().string();
      if (name == checkpoint_format::kManifestName ||
          name.rfind(".staging-", 0) == 0) {
        continue;
      }
      bool listed = false;
      for (const auto& entry : manifest["files"]) {
        listed = listed || entry.value("name", "") == name;
      }
      if (!listed) {
        Throw("checkpoint directory holds an unlisted file '" + name +
              "'; refusing to drop it");
      }
    }
    for (const auto& entry : manifest["files"]) {
      const std::string name = entry["name"].get<std::string>();
      if (ShardIdOf(name) >= 0) {
        continue;
      }
      const fs::path source = input_dir / name;
      if (!fs::exists(source)) {
        Throw("manifest lists a missing file: " + source.string());
      }
      fs::copy_file(source, output_dir / name, fs::copy_options::none);
    }

    std::vector<TargetWriter> writers;
    WriteTargets(
        input_dir,
        output_dir,
        sources,
        sources.front().header.metadata,
        options,
        num_shards,
        counts,
        step_value,
        &writers);

    for (auto& writer : writers) {
      if (options.verify) {
        VerifyShard(writer.final_path.string(),
                    writer.header.metadata,
                    writer.header.record_count);
      }
      reports->push_back(writer.report);
    }

    // Manifest last: a half-migrated directory has no manifest, so readers
    // ignore it exactly like an interrupted save.
    for (auto& entry : manifest["files"]) {
      const std::string name = entry["name"].get<std::string>();
      const int shard_id     = ShardIdOf(name);
      if (shard_id < 0) {
        continue;
      }
      entry["name"] = ShardFileName(shard_id);
      entry["size"] = fs::file_size(output_dir / ShardFileName(shard_id));
    }
    nlohmann::json provenance = {
        {"optimizer", OptimizerName(options.optimizer)},
        {"source_layout", "pre-optimizer_layout"},
        {"tool", "checkpoint_migrate"}};
    if (options.optimizer == Optimizer::kAdamW) {
      provenance["table_id"] = options.table_id;
    }
    if (!manifest.contains("extra") || !manifest["extra"].is_object()) {
      manifest["extra"] = nlohmann::json::object();
    }
    manifest["extra"]["layout_migration"] = provenance;
    WriteManifest(output_dir, manifest);
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

bool MigrateCheckpoint(const std::string& input_path,
                       const std::string& output_path,
                       const Options& options,
                       std::vector<ShardReport>* reports,
                       std::string* error) {
  if (reports == nullptr || error == nullptr) {
    return false;
  }
  try {
    const fs::path input(input_path);
    if (!fs::is_directory(input)) {
      Throw("checkpoint input is not a directory: " + input.string());
    }
    if (fs::exists(fs::path(output_path))) {
      Throw("refusing to overwrite existing path: " + output_path);
    }
    if (fs::exists(input / checkpoint_format::kManifestName)) {
      return MigrateVersionDir(
          input_path, output_path, options, reports, error);
    }

    std::vector<fs::path> versions;
    for (const auto& entry : fs::directory_iterator(input)) {
      if (!entry.is_directory()) {
        continue;
      }
      const std::string name = entry.path().filename().string();
      if (!name.empty() && name.front() == '.') {
        continue;
      }
      if (fs::exists(entry.path() / checkpoint_format::kManifestName)) {
        versions.push_back(entry.path());
      }
    }
    if (versions.empty()) {
      Throw("no committed checkpoint (manifest.json) under: " + input.string());
    }
    std::sort(versions.begin(), versions.end());
    for (const auto& version : versions) {
      if (!MigrateVersionDir(
              version.string(),
              (fs::path(output_path) / version.filename()).string(),
              options,
              reports,
              error)) {
        return false;
      }
    }
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

} // namespace recstore::checkpoint_migrate
