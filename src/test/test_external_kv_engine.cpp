#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "base/factory.h"
#include "base/json.h"
#include "storage/kv_engine/base_kv.h"
#include "storage/kv_engine/engine_factory.h"
#include "storage/kv_engine/engine_selector.h"

namespace {

constexpr size_t kValueSize = 128;

BaseKVConfig MakeExternalEngineConfig(const std::string& engine_type,
                                      const std::string& path) {
  BaseKVConfig config;
  config.num_threads_ = 4;
  config.json_config_ = {
      {"engine_type", engine_type},
      {"path", path},
      {"capacity", 1024},
      {"value_size", kValueSize},
      {"max_batch_size", 16}};
  return config;
}

void AddEngineSpecificConfig(BaseKVConfig* config,
                             const std::string& engine_type) {
#ifdef RECSTORE_TEST_ENABLE_DETABLE_ENGINE
  if (engine_type == "KVEngineDETable") {
    config->json_config_["detable"] = {
        {"library_path", RECSTORE_TEST_DYNAMIC_KV_PLUGIN_PATH},
        {"block_size", 64}};
  }
#else
  (void)config;
  (void)engine_type;
#endif
}

std::unique_ptr<BaseKV> CreateEngine(const std::string& engine_type) {
  const std::string path = "/tmp/test_external_kv_engine_" + engine_type + "_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  BaseKVConfig config = MakeExternalEngineConfig(engine_type, path);
  AddEngineSpecificConfig(&config, engine_type);
  base::EngineResolved resolved;
  EXPECT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_EQ(resolved.engine, engine_type);
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

std::unique_ptr<BaseKV> CreateFasterKvSsdEngine() {
  const std::string path = "/tmp/test_external_kv_engine_fasterkv_ssd_" +
                           std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  BaseKVConfig config = MakeExternalEngineConfig("KVEngineFasterKV", path);
  config.json_config_["fasterkv"] = {
      {"storage", "ssd"},
      {"log_path", path + "/fasterkv-log"},
      {"hlog_memory_bytes", 1ULL << 30},
      {"mutable_fraction", 0.5}};

  base::EngineResolved resolved;
  EXPECT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_EQ(resolved.engine, "KVEngineFasterKV");
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

std::unique_ptr<BaseKV>
CreateEngineFromRecstoreConfigFile(const std::string& engine_type) {
  const std::string path =
      "/tmp/test_external_kv_engine_config_" + engine_type + "_" +
      std::to_string(static_cast<long long>(getpid()));
  const std::string config_path = path + ".json";
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);

  json recstore_config = {
      {"cache_ps",
       {{"num_threads", 4},
        {"ps_type", "GRPC"},
        {"base_kv_config",
         {{"engine_type", engine_type},
          {"path", path},
          {"capacity", 1024},
          {"value_size", kValueSize},
          {"max_batch_size", 16},
          {"table_name", "default"}}}}}};

  if (engine_type == "KVEngineDETable") {
#ifdef RECSTORE_TEST_ENABLE_DETABLE_ENGINE
    recstore_config["cache_ps"]["base_kv_config"]["detable"] = {
        {"library_path", RECSTORE_TEST_DYNAMIC_KV_PLUGIN_PATH},
        {"block_size", 64}};
#endif
  }

  {
    std::ofstream out(config_path);
    out << recstore_config.dump(2);
  }

  std::ifstream in(config_path);
  json loaded;
  in >> loaded;

  BaseKVConfig config;
  config.num_threads_ = loaded.at("cache_ps").at("num_threads").get<int>();
  config.json_config_ = loaded.at("cache_ps").at("base_kv_config");

  base::EngineResolved resolved;
  EXPECT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_EQ(resolved.engine, engine_type);
  return std::unique_ptr<BaseKV>(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
}

void AssertBasicPutGet(BaseKV* kv) {
  std::string value = "external_engine_value";
  value.resize(kValueSize, '\0');

  kv->Put(7, value, 0);

  std::string out;
  kv->Get(7, out, 0);
  EXPECT_EQ(out, value);

  std::string miss;
  kv->Get(99, miss, 0);
  EXPECT_TRUE(miss.empty());
}

void AssertBatchPutGet(BaseKV* kv) {
  constexpr int kRows = 3;
  constexpr int kDim  = static_cast<int>(kValueSize / sizeof(float));

  std::vector<uint64_t> keys = {101, 102, 103};
  std::vector<std::vector<float>> rows(kRows, std::vector<float>(kDim));
  std::vector<base::ConstArray<float>> views;
  views.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kDim; ++j) {
      rows[i][j] = static_cast<float>(i * 100 + j);
    }
    views.emplace_back(rows[i].data(), rows[i].size());
  }

  base::ConstArray<uint64_t> key_view(keys.data(), keys.size());
  kv->BatchPut(key_view, &views, 0);

  std::vector<base::ConstArray<float>> out;
  kv->BatchGet(key_view, &out, 0);

  ASSERT_EQ(out.size(), keys.size());
  for (int i = 0; i < kRows; ++i) {
    ASSERT_EQ(out[i].Size(), kDim);
    for (int j = 0; j < kDim; ++j) {
      EXPECT_FLOAT_EQ(out[i][j], rows[i][j]);
    }
  }
}

void AssertFactoryEngine(const std::string& engine_type) {
  auto kv = CreateEngine(engine_type);
  ASSERT_NE(kv, nullptr);
  AssertBasicPutGet(kv.get());
  AssertBatchPutGet(kv.get());
}

void AssertConfigFileEngine(const std::string& engine_type) {
  auto kv = CreateEngineFromRecstoreConfigFile(engine_type);
  ASSERT_NE(kv, nullptr);
  AssertBasicPutGet(kv.get());
}

} // namespace

TEST(ExternalKVEngineSelectorTest,
     MissingEngineTypeDefaultsToCompositeNotExternal) {
  BaseKVConfig config = MakeExternalEngineConfig("KVEngineFasterKV", "/tmp/x");
  config.json_config_.erase("engine_type");
  auto resolved = base::ResolveEngine(config);
  EXPECT_EQ(resolved.engine, "KVEngineComposite");
}

TEST(ExternalKVEngineSelectorTest, RejectsRemovedExternalEngineTypeField) {
  BaseKVConfig config = MakeExternalEngineConfig("KVEngineFasterKV", "/tmp/x");
  config.json_config_["external_engine_type"] = "KVEngineFasterKV";
  EXPECT_THROW(base::ResolveEngine(config), std::invalid_argument);
}

TEST(ExternalKVEngineSelectorTest, RejectsUnknownEngineType) {
  BaseKVConfig config = MakeExternalEngineConfig("KVEngineFasterKV", "/tmp/x");
  config.json_config_["engine_type"] = "KVEngineUnknown";
  EXPECT_THROW(base::ResolveEngine(config), std::invalid_argument);
}

#ifdef RECSTORE_TEST_ENABLE_DETABLE_ENGINE
TEST(ExternalKVEngineFactoryTest, DETableEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineDETable");
}

TEST(ExternalKVEngineFactoryTest, DETableEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineDETable");
}

TEST(ExternalKVEngineFactoryTest, DETableEngineSupportsFlatBatchAndClear) {
  auto kv = CreateEngine("KVEngineDETable");
  constexpr int kDim = static_cast<int>(kValueSize / sizeof(float));
  std::vector<uint64_t> keys = {1, 2};
  std::vector<float> first(kDim, 1.0F);
  std::vector<float> second(kDim, 2.0F);
  std::vector<base::ConstArray<float>> rows;
  rows.emplace_back(first.data(), first.size());
  rows.emplace_back(second.data(), second.size());
  kv->BatchPut(base::ConstArray<uint64_t>(keys), &rows, 0);

  std::vector<uint64_t> lookup = {2, 3};
  std::vector<float> output(lookup.size() * kDim, -1.0F);
  BaseKV::BatchGetFlatStats stats;
  ASSERT_TRUE(kv->BatchGetFlat(base::ConstArray<uint64_t>(lookup),
                               output.data(),
                               lookup.size(),
                               kDim,
                               0,
                               &stats));
  EXPECT_EQ(stats.missing_rows, 1);
  EXPECT_FLOAT_EQ(output[0], 2.0F);
  EXPECT_FLOAT_EQ(output[kDim], 0.0F);

  kv->clear();
  EXPECT_FALSE(kv->Exists(1, 0));
}

TEST(ExternalKVEngineFactoryTest, DETableEngineRequiresLibraryPath) {
  BaseKVConfig config = MakeExternalEngineConfig("KVEngineDETable", "/tmp/x");
  config.json_config_["detable"] = {{"library_path", ""}};
  auto resolved = base::ResolveEngine(config);
  EXPECT_THROW(std::unique_ptr<BaseKV>(
                   base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
                       resolved.engine, resolved.cfg)),
               std::invalid_argument);
}
#endif

#ifdef RECSTORE_TEST_ENABLE_FASTERKV_ENGINE
TEST(ExternalKVEngineFactoryTest, FasterKVEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineFasterKV");
}

TEST(ExternalKVEngineFactoryTest, FasterKVEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineFasterKV");
}

TEST(ExternalKVEngineFactoryTest, FasterKVEngineSupportsSsdLogConfig) {
  auto kv = CreateFasterKvSsdEngine();
  ASSERT_NE(kv, nullptr);
  AssertBasicPutGet(kv.get());
  AssertBatchPutGet(kv.get());
}

TEST(ExternalKVEngineFactoryTest,
     FasterKVEngineRejectsSsdWithoutLogPathOrPath) {
  BaseKVConfig config = MakeExternalEngineConfig("KVEngineFasterKV", "");
  config.json_config_["fasterkv"] = {{"storage", "ssd"}};

  base::EngineResolved resolved;
  ASSERT_NO_THROW(resolved = base::ResolveEngine(config));
  EXPECT_THROW(std::unique_ptr<BaseKV>(
                   base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
                       resolved.engine, resolved.cfg)),
               std::invalid_argument);
}
#endif

#ifdef RECSTORE_TEST_ENABLE_HPS_ENGINE
TEST(ExternalKVEngineFactoryTest, HpsHashMapEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineHPSHashMap");
}

TEST(ExternalKVEngineFactoryTest, HpsRocksDBEngineUsesBaseKVInterface) {
  AssertFactoryEngine("KVEngineHPSRocksDB");
}

TEST(ExternalKVEngineFactoryTest, HpsHashMapEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineHPSHashMap");
}

TEST(ExternalKVEngineFactoryTest, HpsRocksDBEngineCanBeSelectedByConfigFile) {
  AssertConfigFileEngine("KVEngineHPSRocksDB");
}
#endif
