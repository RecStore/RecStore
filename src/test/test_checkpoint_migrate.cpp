#include "ps/checkpoint/checkpoint_migrate.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "base/hash.h"
#include "base/json.h"
#include "memory/shm_file.h"
#include "optimizer/optimizer.h"
#include "ps/base/checkpoint_restore.h"
#include "storage/kv_engine/engine_composite.h"

namespace {

namespace fs  = std::filesystem;
namespace mig = recstore::checkpoint_migrate;

// The migrator keeps its own copy of the optimizer layout constants so it stays
// out of the optimizer headers; these asserts are what keeps the copies honest.
static_assert(mig::kAdamWTensorsPerTable == AdamW::kTensorsPerTable);
static_assert(mig::kAdamWStepKeyId == AdamW::kStepKey);
static_assert(mig::kSgdTensorsPerTable == 1);
static_assert(mig::kRowWiseAdagradTensorsPerTable == 2);

constexpr uint64_t kTagShift = 56;
constexpr uint64_t kIdMask   = (1ULL << kTagShift) - 1;
constexpr uint64_t kStepId   = AdamW::kStepKey;

uint64_t Tagged(uint64_t tag, uint64_t id) {
  return (tag << kTagShift) | (id & kIdMask);
}

class CheckpointMigrateTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() /
            ("recstore-checkpoint-migrate-" + std::to_string(getpid()));
    fs::remove_all(root_);
    fs::create_directories(root_);
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;
  }

  void TearDown() override { fs::remove_all(root_); }

  std::unique_ptr<KVEngineComposite> MakeEngine(const std::string& name) {
    const fs::path value_path = root_ / name / "value";
    fs::create_directories(value_path.parent_path());
    BaseKVConfig config;
    config.num_threads_ = 2;
    config.json_config_ = {
        {"engine_type", "KVEngineComposite"},
        {"capacity", 1024},
        {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
        {"value",
         {{"type", "DRAM_VALUE_STORE"},
          {"path", value_path.string()},
          {"default_value_size_hint", 4 * sizeof(float)},
          {"dram_allocator",
           {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1024 * 1024}}}}}};
    return std::make_unique<KVEngineComposite>(config);
  }

  static std::string FloatBytes(float value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(value));
  }

  static float ReadFloat(BaseKV* engine, uint64_t key) {
    std::string value;
    engine->Get(key, value, 0);
    if (value.size() != sizeof(float)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    float result = 0.0f;
    std::memcpy(&result, value.data(), sizeof(result));
    return result;
  }

  std::string ClientMetadata() const {
    const nlohmann::json identity = {
        {"schema_version", 1}, {"ps_type", "GRPC"}, {"table", {{"name", "t"}}}};
    return nlohmann::json{
        {"checkpoint_id", "step-00000007"}, {"identity", identity}}
        .dump();
  }

  // Metadata a pre-#400 server stored inside a shard: the client payload with
  // the server's shard_id, and no layout marker.
  static std::string
  LegacyMetadata(const std::string& client_metadata, int shard_id) {
    nlohmann::json parsed = nlohmann::json::parse(client_metadata);
    parsed["shard_id"]    = shard_id;
    return parsed.dump();
  }

  // Build a legacy checkpoint version directory: shard files written by the
  // real engine plus the client-style manifest.
  fs::path MakeLegacyCheckpoint(
      const std::string& name,
      const std::vector<
          std::pair<int, std::vector<std::pair<uint64_t, float>>>>& shards) {
    const fs::path dir = root_ / name;
    fs::create_directories(dir);
    const std::string client_metadata = ClientMetadata();
    nlohmann::json files              = nlohmann::json::array();
    for (const auto& [shard_id, rows] : shards) {
      auto engine = MakeEngine(name + "-" + std::to_string(shard_id));
      for (const auto& [key, value] : rows) {
        engine->Put(key, FloatBytes(value), 0);
      }
      const std::string file_name =
          "recstore-shard-" + std::to_string(shard_id) + ".bin";
      if (!engine->SaveCheckpoint((dir / file_name).string(),
                                  LegacyMetadata(client_metadata, shard_id))) {
        throw std::runtime_error("failed to write fixture shard " + file_name);
      }
      files.push_back(
          {{"name", file_name}, {"size", fs::file_size(dir / file_name)}});
    }
    const nlohmann::json manifest = {
        {"schema_version", 1},
        {"checkpoint_id", "step-00000007"},
        {"step", 7},
        {"identity", nlohmann::json::parse(client_metadata)["identity"]},
        {"files", files}};
    std::ofstream output(dir / "manifest.json");
    output << manifest.dump(2) << "\n";
    output.close();
    return dir;
  }

  std::unique_ptr<KVEngineComposite>
  LoadShard(const fs::path& dir, int shard_id, const std::string& name) {
    auto engine = MakeEngine(name);
    const std::string file =
        "recstore-shard-" + std::to_string(shard_id) + ".bin";
    EXPECT_TRUE(engine->LoadCheckpoint(
        (dir / file).string(),
        recstore::BindCheckpointMetadata(ClientMetadata(), shard_id)));
    return engine;
  }

  fs::path root_;
};

TEST_F(CheckpointMigrateTest, MigratesSgdShardWithoutChangingKeys) {
  const auto legacy = MakeLegacyCheckpoint(
      "sgd", {{0, {{Tagged(0, 42), 1.5f}}}, {1, {{Tagged(1, 43), 2.5f}}}});

  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  options.table_ids = {0, 1};
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "sgd-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;
  ASSERT_EQ(reports.size(), 2u);

  auto shard0 = LoadShard(migrated, 0, "sgd0-target");
  auto shard1 = LoadShard(migrated, 1, "sgd1-target");
  EXPECT_FLOAT_EQ(ReadFloat(shard0.get(), Tagged(0, 42)), 1.5f);
  EXPECT_FLOAT_EQ(ReadFloat(shard1.get(), Tagged(1, 43)), 2.5f);
}

TEST_F(CheckpointMigrateTest, MigratesAdamWShardOntoTableTaggedKeys) {
  // One shard keeps the layout simple; the repartition below covers routing.
  const auto legacy = MakeLegacyCheckpoint(
      "adamw",
      {{0,
        {{Tagged(0, 7), 1.0f},
         {Tagged(1, 7), 2.0f},
         {Tagged(2, 7), 3.0f},
         {Tagged(1, kStepId), 4.0f}}}});

  mig::Options options;
  options.optimizer  = mig::Optimizer::kAdamW;
  options.table_id   = 2;
  options.num_shards = 1;
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "adamw-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;
  ASSERT_EQ(reports.size(), 1u);
  EXPECT_EQ(reports[0].step_rows, 1u);
  EXPECT_EQ(reports[0].tags_after.at(8), 1u);  // 4 * 2 + 0
  EXPECT_EQ(reports[0].tags_after.at(9), 1u);  // first moment
  EXPECT_EQ(reports[0].tags_after.at(10), 1u); // second moment
  EXPECT_EQ(reports[0].tags_after.at(11), 1u); // step

  auto engine = LoadShard(migrated, 0, "adamw-target");
  EXPECT_FLOAT_EQ(ReadFloat(engine.get(), Tagged(8, 7)), 1.0f);
  EXPECT_FLOAT_EQ(ReadFloat(engine.get(), Tagged(9, 7)), 2.0f);
  EXPECT_FLOAT_EQ(ReadFloat(engine.get(), Tagged(10, 7)), 3.0f);
  EXPECT_FLOAT_EQ(ReadFloat(engine.get(), Tagged(11, kStepId)), 4.0f);
  // The old key space is gone.
  EXPECT_TRUE(std::isnan(ReadFloat(engine.get(), Tagged(0, 7))));
  EXPECT_TRUE(std::isnan(ReadFloat(engine.get(), Tagged(1, kStepId))));
}

TEST_F(CheckpointMigrateTest, RepartitionsRecordsForTheNewKeys) {
  // Four ids, all on shard 0 in the legacy layout. With table_id 2 the tagged
  // keys hash elsewhere, so the migration must redistribute them across both
  // shards - a per-file rewrite would leave rows on a shard the client never
  // queries for them.
  std::vector<std::pair<uint64_t, float>> rows;
  for (uint64_t id = 0; id < 4; ++id) {
    rows.emplace_back(Tagged(0, id), static_cast<float>(id) + 0.5f);
  }
  const auto legacy = MakeLegacyCheckpoint("repartition", {{0, rows}, {1, {}}});

  mig::Options options;
  options.optimizer = mig::Optimizer::kAdamW;
  options.table_id  = 2;
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "repartition-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;
  ASSERT_EQ(reports.size(), 2u);

  int per_shard[2] = {static_cast<int>(reports[0].records),
                      static_cast<int>(reports[1].records)};
  int expected[2]  = {0, 0};
  for (uint64_t id = 0; id < 4; ++id) {
    ++expected[GetHash(Tagged(8, id)) % 2];
  }
  EXPECT_EQ(per_shard[0], expected[0]);
  EXPECT_EQ(per_shard[1], expected[1]);

  // Reading the row through the shard the new key hashes to must find it.
  auto shard0 = LoadShard(migrated, 0, "repartition0");
  auto shard1 = LoadShard(migrated, 1, "repartition1");
  for (uint64_t id = 0; id < 4; ++id) {
    const uint64_t key = Tagged(8, id);
    const auto value   = GetHash(key) % 2 == 0 ? ReadFloat(shard0.get(), key)
                                               : ReadFloat(shard1.get(), key);
    EXPECT_FLOAT_EQ(value, static_cast<float>(id) + 0.5f);
  }
}

TEST_F(CheckpointMigrateTest, KeepsRowWiseAdagradAccumulatorWithItsParameter) {
  // RowWiseAdagrad keeps the accumulator under its own tag, written by the
  // server, so it must stay in the parameter row's shard.
  std::vector<std::pair<uint64_t, float>> rows;
  for (uint64_t id = 0; id < 4; ++id) {
    rows.emplace_back(Tagged(2, id), 1.0f + static_cast<float>(id));
    rows.emplace_back(Tagged(3, id), 5.0f + static_cast<float>(id));
  }
  const auto legacy = MakeLegacyCheckpoint("rwadagrad", {{0, rows}, {1, {}}});

  mig::Options options;
  options.optimizer = mig::Optimizer::kRowWiseAdagrad;
  options.table_ids = {1};
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "rwadagrad-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;

  auto shard0 = LoadShard(migrated, 0, "rwadagrad0");
  auto shard1 = LoadShard(migrated, 1, "rwadagrad1");
  for (uint64_t id = 0; id < 4; ++id) {
    const bool on_zero = GetHash(Tagged(2, id)) % 2 == 0;
    BaseKV* engine     = on_zero ? shard0.get() : shard1.get();
    EXPECT_FLOAT_EQ(ReadFloat(engine, Tagged(2, id)), 1.0f + id);
    EXPECT_FLOAT_EQ(ReadFloat(engine, Tagged(3, id)), 5.0f + id);
  }
}

TEST_F(CheckpointMigrateTest, KeepsAdamWTensorsWithTheirParameterRow) {
  // Moment rows are written by the server into the engine that received the
  // update, so they must follow the parameter row's shard instead of hashing by
  // their own tag. Routing them by tag would hide every moment from the shard
  // that actually reads it.
  std::vector<std::pair<uint64_t, float>> rows;
  for (uint64_t id = 0; id < 4; ++id) {
    rows.emplace_back(Tagged(0, id), 1.0f + static_cast<float>(id));
    rows.emplace_back(Tagged(1, id), 2.0f + static_cast<float>(id));
    rows.emplace_back(Tagged(2, id), 3.0f + static_cast<float>(id));
  }
  const auto legacy = MakeLegacyCheckpoint("follow", {{0, rows}, {1, {}}});

  mig::Options options;
  options.optimizer = mig::Optimizer::kAdamW;
  options.table_id  = 2;
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "follow-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;

  auto shard0 = LoadShard(migrated, 0, "follow0");
  auto shard1 = LoadShard(migrated, 1, "follow1");
  for (uint64_t id = 0; id < 4; ++id) {
    const uint64_t param_key = Tagged(8, id);
    const bool on_zero       = GetHash(param_key) % 2 == 0;
    BaseKV* engine           = on_zero ? shard0.get() : shard1.get();
    BaseKV* other            = on_zero ? shard1.get() : shard0.get();
    EXPECT_FLOAT_EQ(ReadFloat(engine, Tagged(8, id)), 1.0f + id);
    EXPECT_FLOAT_EQ(ReadFloat(engine, Tagged(9, id)), 2.0f + id);
    EXPECT_FLOAT_EQ(ReadFloat(engine, Tagged(10, id)), 3.0f + id);
    EXPECT_TRUE(std::isnan(ReadFloat(other, Tagged(9, id))));
    EXPECT_TRUE(std::isnan(ReadFloat(other, Tagged(10, id))));
  }
}

TEST_F(CheckpointMigrateTest, RefusesAdamWShardDeclaredAsSgd) {
  const auto legacy = MakeLegacyCheckpoint(
      "adamw-for-sgd",
      {{0, {{Tagged(0, 7), 1.0f}, {Tagged(1, kStepId), 4.0f}}}});

  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  std::vector<mig::ShardReport> reports;
  std::string error;
  EXPECT_FALSE(mig::MigrateVersionDir(
      legacy.string(), (root_ / "out").string(), options, &reports, &error));
  EXPECT_NE(error.find("AdamW"), std::string::npos) << error;
}

TEST_F(CheckpointMigrateTest, RefusesAlreadyMigratedShard) {
  const auto legacy =
      MakeLegacyCheckpoint("once", {{0, {{Tagged(0, 3), 1.0f}}}});
  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "once-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;

  EXPECT_FALSE(mig::MigrateVersionDir(
      migrated.string(),
      (root_ / "twice").string(),
      options,
      &reports,
      &error));
  EXPECT_NE(error.find("already records optimizer_layout"), std::string::npos)
      << error;
}

TEST_F(CheckpointMigrateTest, RefusesTableIdThatOverflowsTheTag) {
  const auto legacy = MakeLegacyCheckpoint(
      "overflow", {{0, {{Tagged(0, 1), 1.0f}, {Tagged(1, kStepId), 1.0f}}}});
  mig::Options options;
  options.optimizer = mig::Optimizer::kAdamW;
  options.table_id  = 64;
  std::vector<mig::ShardReport> reports;
  std::string error;
  EXPECT_FALSE(mig::MigrateCheckpoint(
      legacy.string(), (root_ / "out").string(), options, &reports, &error));
  EXPECT_NE(error.find("--table-id"), std::string::npos) << error;
}

TEST_F(CheckpointMigrateTest, RefusesShardsWithDivergedStepCounters) {
  const auto legacy = MakeLegacyCheckpoint(
      "diverged",
      {{0, {{Tagged(0, 1), 1.0f}, {Tagged(1, kStepId), 3.0f}}},
       {1, {{Tagged(0, 2), 1.0f}, {Tagged(1, kStepId), 2.0f}}}});
  mig::Options options;
  options.optimizer = mig::Optimizer::kAdamW;
  options.table_id  = 1;
  std::vector<mig::ShardReport> reports;
  std::string error;
  EXPECT_FALSE(mig::MigrateVersionDir(
      legacy.string(),
      (root_ / "diverged-out").string(),
      options,
      &reports,
      &error));
  EXPECT_NE(error.find("step counters disagree"), std::string::npos) << error;
}

TEST_F(CheckpointMigrateTest, RefusesCorruptedSourceChecksum) {
  const auto legacy =
      MakeLegacyCheckpoint("corrupt", {{0, {{Tagged(0, 9), 1.0f}}}});
  {
    // Flip a bit inside the value, leaving the trailing checksum intact.
    std::fstream file(legacy / "recstore-shard-0.bin",
                      std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.good());
    file.seekp(-static_cast<std::streamoff>(sizeof(uint64_t) + 1),
               std::ios::end);
    char byte = 0;
    file.read(&byte, 1);
    byte ^= 0x1;
    file.seekp(-1, std::ios::cur);
    file.write(&byte, 1);
  }

  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  std::vector<mig::ShardReport> reports;
  std::string error;
  // Migration itself only verifies the input checksum in the counting pass when
  // values are skipped; run with verification to surface the corruption.
  EXPECT_FALSE(mig::MigrateVersionDir(
      legacy.string(),
      (root_ / "corrupt-out").string(),
      options,
      &reports,
      &error));
  EXPECT_NE(error.find("checksum"), std::string::npos) << error;
}

TEST_F(CheckpointMigrateTest, UpdatesManifestAfterMigration) {
  const auto legacy =
      MakeLegacyCheckpoint("manifest", {{0, {{Tagged(0, 11), 7.0f}}}});
  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  std::vector<mig::ShardReport> reports;
  std::string error;
  const fs::path migrated = root_ / "manifest-migrated";
  ASSERT_TRUE(mig::MigrateVersionDir(
      legacy.string(), migrated.string(), options, &reports, &error))
      << error;

  std::ifstream input(migrated / "manifest.json");
  ASSERT_TRUE(input.good());
  nlohmann::json manifest;
  input >> manifest;
  EXPECT_EQ(manifest["checkpoint_id"], "step-00000007");
  EXPECT_EQ(manifest["files"][0]["size"].get<uint64_t>(),
            fs::file_size(migrated / "recstore-shard-0.bin"));
  EXPECT_EQ(manifest["extra"]["layout_migration"]["optimizer"], "SGD");

  auto engine = LoadShard(migrated, 0, "manifest-target");
  EXPECT_FLOAT_EQ(ReadFloat(engine.get(), Tagged(0, 11)), 7.0f);
}

TEST_F(CheckpointMigrateTest, RefusesOutputThatAlreadyExists) {
  const auto legacy =
      MakeLegacyCheckpoint("exists", {{0, {{Tagged(0, 1), 1.0f}}}});
  mig::Options options;
  options.optimizer = mig::Optimizer::kSgd;
  std::vector<mig::ShardReport> reports;
  std::string error;
  EXPECT_FALSE(mig::MigrateVersionDir(
      legacy.string(), legacy.string(), options, &reports, &error));
  EXPECT_NE(error.find("refusing to overwrite"), std::string::npos) << error;
}

} // namespace
