#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/shm_file.h"
#include "storage/index/utils/hash.h"
#include "storage/io_backend/force_link.h"
#include "storage/io_backend/io_backend.h"
#include "storage/cache/tiered_read_cache.h"
#include "storage/kv_engine/engine_composite.h"
#include "storage/ssd_layout/ssd_layout.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/hybrid_value_store.h"
#include "storage/value_store/ssd_static_layout_value_store.h"
#include "storage/value_store/ssd_value_store.h"

namespace {

std::atomic<uint64_t> g_read_entries{0};

class CountingIOBackend : public IOBackend {
public:
  explicit CountingIOBackend(const BaseKVConfig& config) : IOBackend(config) {}
  void init() override {}
  void*
  GetPage(coroutine<void>::push_type&, uint64_t, PageID_t page_id) override {
    return GetPage(page_id);
  }
  void* GetPage(PageID_t page_id) override {
    auto* page = new char[PAGE_SIZE];
    ReadPageSync(page_id, page);
    return page;
  }
  void Unpin(coroutine<void>::push_type&,
             uint64_t,
             PageID_t page_id,
             void* page,
             bool dirty) override {
    Unpin(page_id, page, dirty);
  }
  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) override {
    if (is_dirty) {
      WritePageSync(page_id, static_cast<char*>(page_data));
    }
    delete[] static_cast<char*>(page_data);
  }
  char* AllocateBuffer() override { return AllocateBuffer(1); }
  char* AllocateBuffer(uint64_t page_count) override {
    return new char[page_count * PAGE_SIZE]();
  }
  void FreeBuffer(char* buf) override { delete[] buf; }
  void submit() override {}
  void PollCompletion() override {}
  void BatchReadPages(const std::vector<IOEntry>& entries) override {
    g_read_entries.fetch_add(entries.size(), std::memory_order_relaxed);
    IOBackend::BatchReadPages(entries);
  }

protected:
  void ReadPageAsync(coroutine<void>::push_type&,
                     uint64_t,
                     PageID_t page_id,
                     char* buffer) override {
    ReadPageSync(page_id, buffer);
  }
  void ReadPageSync(PageID_t page_id, char* buffer) override {
    auto found = pages_.find(page_id);
    if (found == pages_.end()) {
      std::memset(buffer, 0, PAGE_SIZE);
      return;
    }
    std::memcpy(buffer, found->second.data(), PAGE_SIZE);
  }
  void WritePageAsync(coroutine<void>::push_type&,
                      uint64_t,
                      PageID_t page_id,
                      char* buffer) override {
    WritePageSync(page_id, buffer);
  }
  void WritePageSync(PageID_t page_id, char* buffer) override {
    auto& page = pages_[page_id];
    if (page.empty()) {
      page.resize(PAGE_SIZE, 0);
    }
    std::memcpy(page.data(), buffer, PAGE_SIZE);
  }

private:
  std::unordered_map<PageID_t, std::vector<char>> pages_;
};

FACTORY_REGISTER(IOBackend, INMEMORY, CountingIOBackend, const BaseKVConfig&);

class StaticLayoutValueTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("recstore-static-value-" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    g_read_entries.store(0);
  }
  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path Save(const ssd_layout::Artifact& artifact) {
    const auto path = root_ / "layout.bin";
    ssd_layout::SaveArtifact(ssd_layout::Normalize(artifact), path.string());
    return path;
  }

  BaseKVConfig
  Config(const std::filesystem::path& mapping,
         const char* strategy,
         const char* value_type = "SSD_STATIC_LAYOUT_VALUE_STORE") {
    BaseKVConfig config;
    config.num_threads_ = 2;
    config.json_config_ = {
        {"engine_type", "KVEngineComposite"},
        {"capacity", 16},
        {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
        {"value",
         {{"type", value_type},
          {"path", (root_ / "value.db").string()},
          {"default_value_size_hint", 4},
          {"io",
           {{"type", "INMEMORY"},
            {"queue_depth", 4},
            {"base_offset_bytes", 0}}},
          {"ssd_layout",
           {{"strategy", strategy}, {"mapping_path", mapping.string()}}}}}};
    return config;
  }

  std::filesystem::path root_;
};

ssd_layout::Artifact Bandana() {
  ssd_layout::Artifact artifact;
  artifact.strategy  = ssd_layout::Strategy::kBandana;
  artifact.page_size = 4096;
  artifact.row_size  = 4;
  artifact.keys      = {0, 1, 2, 3, 4, 5};
  artifact.copies    = {
      {{0, 0}}, {{0, 4}}, {{0, 8}}, {{1, 0}}, {{1, 4}}, {{1, 8}}};
  return artifact;
}

ssd_layout::Artifact MaxEmbed() {
  auto artifact     = Bandana();
  artifact.strategy = ssd_layout::Strategy::kMaxEmbed;
  for (uint32_t slot = 0; slot < artifact.keys.size(); ++slot) {
    artifact.copies[slot].push_back({2, slot * 4});
  }
  return artifact;
}

TEST_F(StaticLayoutValueTest, KeyHandleRoundTripAndFixedPlacement) {
  auto store = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  const char zero[]                = {'z', 'e', 'r', 'o'};
  const char max_key_is_separate[] = {'n', 'o', 'p', 'e'};
  store->Write(1, zero, 4);
  std::string out(4, '\0');
  EXPECT_EQ(store->Read(1, out.data(), out.size()), 4U);
  EXPECT_EQ(out, std::string(zero, 4));
  store->Free(1);
  EXPECT_EQ(store->Read(1, out.data(), out.size()), 4U);
  EXPECT_THROW(store->Alloc(4), std::invalid_argument);
  EXPECT_THROW(store->AllocAndWrite(zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(99, zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(1ULL << 63, zero, 4), std::invalid_argument);
  EXPECT_THROW(store->Write(1, max_key_is_separate, 3), std::invalid_argument);
  EXPECT_THROW(SsdStaticLayoutValueStore(Config(Save(Bandana()), "plain")),
               std::invalid_argument);

  ssd_layout::Artifact wide = Bandana();
  wide.keys                 = {ssd_layout::kMaxKey};
  wide.copies               = {{{0, 0}}};
  auto max_store            = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(wide), "bandana"));
  max_store->Write(ssd_layout::kMaxKey + 1, zero, 4);
  EXPECT_EQ(max_store->Read(ssd_layout::kMaxKey + 1, out.data(), out.size()),
            4U);
}

TEST_F(StaticLayoutValueTest, BatchReadUsesFewerPagesThanPlain) {
  const char payload[] = {'a', 'b', 'c', 'd'};
  BaseKVConfig plain_config;
  plain_config.json_config_ = {
      {"value",
       {{"type", "SSD_VALUE_STORE"},
        {"path", (root_ / "plain.db").string()},
        {"ssd_allocator",
         {{"type", "SSD_SLAB"},
          {"capacity_bytes", 64ULL << 20},
          {"size_classes", {128}},
          {"io",
           {{"type", "INMEMORY"},
            {"queue_depth", 4},
            {"base_offset_bytes", 0}}}}}}}};
  SsdValueStore plain(plain_config);
  std::vector<uint64_t> plain_handles;
  for (int i = 0; i < 6; ++i) {
    plain_handles.push_back(plain.AllocAndWrite(payload, 4));
  }
  plain.ResetStats();
  g_read_entries.store(0);
  std::vector<ValueStore::ReadResult> plain_out;
  plain.BatchRead(plain_handles, plain_out);
  const uint64_t plain_reads = g_read_entries.load();
  EXPECT_NE(plain.ExtraResultFields().find("layout_page_reads=6"),
            std::string::npos);
  EXPECT_NE(plain.ExtraResultFields().find("layout_bytes_read=24576"),
            std::string::npos);

  auto bandana = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  auto maxembed = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(MaxEmbed()), "maxembed"));
  std::vector<uint64_t> handles;
  std::vector<ValueStore::WriteSpec> specs;
  for (uint64_t key = 0; key < 6; ++key) {
    handles.push_back(key + 1);
    specs.push_back({payload, 4});
  }
  bandana->BatchWrite(handles, specs);
  maxembed->BatchWrite(handles, specs);
  std::vector<ValueStore::ReadResult> bandana_out;
  std::vector<ValueStore::ReadResult> max_out;
  bandana->BatchRead(handles, bandana_out);
  maxembed->BatchRead(handles, max_out);
  EXPECT_LT(maxembed->stats().page_reads, bandana->stats().page_reads);
  EXPECT_LT(bandana->stats().page_reads, plain_reads);
  ASSERT_EQ(bandana_out.size(), 6U);
  EXPECT_EQ(bandana_out.front().data, std::vector<char>(payload, payload + 4));
  EXPECT_EQ(max_out.back().data, bandana_out.back().data);

  std::vector<uint64_t> repeated = {1, 1, 1};
  std::vector<ValueStore::ReadResult> repeated_out;
  const auto before = maxembed->stats().page_reads;
  maxembed->BatchRead(repeated, repeated_out);
  EXPECT_EQ(maxembed->stats().page_reads, before + 1);
  EXPECT_EQ(repeated_out[0].data, repeated_out[2].data);
}

TEST_F(StaticLayoutValueTest, HybridWatermarkAndReplicaUpdate) {
  auto mapping        = Save(MaxEmbed());
  BaseKVConfig config = Config(mapping, "maxembed", "TIERED_VALUE_STORE");
  config.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "dram").string()}};
  config.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  config.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 0.0}};
  KVEngineComposite engine(config);
  for (uint64_t key = 0; key < 6; ++key) {
    engine.Put(key, std::string(4, static_cast<char>('A' + key)), 0);
  }
  const std::string updated(4, 'Z');
  engine.Put(2, updated, 0);
  std::string actual;
  engine.Get(2, actual, 0);
  EXPECT_EQ(actual, updated);
  const std::vector<uint64_t> keys = {0, 1, 2, 3, 4, 5};
  std::vector<base::ConstArray<float>> values;
  engine.BatchGet(base::ConstArray<uint64_t>(keys), &values, 0);
  ASSERT_EQ(values.size(), keys.size());
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(values[2].Data()), 4),
            updated);

  config.json_config_["value"]["tiering"] = {
      {"high_watermark_ratio", 4.0 / static_cast<double>(1 << 20)}};
  HybridValueStore hybrid(config);
  const char payload[4] = {'a', 'a', 'a', 'a'};
  const uint64_t first  = hybrid.AllocAndWrite(0, payload, 4);
  const uint64_t second = hybrid.AllocAndWrite(1, payload, 4);
  EXPECT_EQ(first & (1ULL << 63), 0U);
  EXPECT_NE(second & (1ULL << 63), 0U);
}

TEST_F(StaticLayoutValueTest, HybridValidatesStaticRowsBeforeDramPlacement) {
  BaseKVConfig config =
      Config(Save(Bandana()), "bandana", "TIERED_VALUE_STORE");
  config.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "dram").string()}};
  config.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  config.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 1.0}};
  HybridValueStore hybrid(config);
  const char payload[4] = {'a', 'b', 'c', 'd'};
  EXPECT_THROW(hybrid.AllocAndWrite(99, payload, sizeof(payload)),
               std::invalid_argument);
  EXPECT_THROW(hybrid.AllocAndWrite(0, payload, sizeof(payload) - 1),
               std::invalid_argument);
}

TEST_F(StaticLayoutValueTest, StaticStoreUsesIoDefaultsWhenConfigIsOmitted) {
  BaseKVConfig config = Config(Save(Bandana()), "bandana");
  config.json_config_["value"].erase("io");
  ForceLinkIOBackends();
  EXPECT_NO_THROW({ SsdStaticLayoutValueStore store(config); });
}

TEST_F(StaticLayoutValueTest, BatchGetFlatUsesGroupedStaticReads) {
  const auto mapping               = Save(Bandana());
  const std::vector<uint64_t> keys = {0, 1, 2};
  const std::array<float, 3> input = {1.0F, 2.0F, 3.0F};
  auto exercise                    = [&](BaseKVConfig config) {
    KVEngineComposite engine(config);
    for (size_t i = 0; i < keys.size(); ++i) {
      engine.Put(
          keys[i],
          std::string(reinterpret_cast<const char*>(&input[i]), sizeof(float)),
          0);
    }
    g_read_entries.store(0);
    std::array<float, 3> output{};
    EXPECT_TRUE(engine.BatchGetFlat(
        base::ConstArray<uint64_t>(keys), output.data(), 3, 1, 0));
    EXPECT_EQ(output, input);
    EXPECT_EQ(g_read_entries.load(), 1U);
  };

  exercise(Config(mapping, "bandana"));

  BaseKVConfig hybrid = Config(mapping, "bandana", "TIERED_VALUE_STORE");
  hybrid.json_config_["value"]["dram_allocator"] = {
      {"type", "PERSIST_LOOP_SLAB"},
      {"capacity_bytes", 1 << 20},
      {"path", (root_ / "flat-dram").string()}};
  hybrid.json_config_["value"]["ssd_allocator"] = {
      {"type", "SSD_SLAB"},
      {"capacity_bytes", 64ULL << 20},
      {"path", (root_ / "flat-ssd.db").string()},
      {"io",
       {{"type", "INMEMORY"}, {"queue_depth", 4}, {"base_offset_bytes", 0}}}};
  hybrid.json_config_["value"]["tiering"] = {{"high_watermark_ratio", 0.0}};
  exercise(std::move(hybrid));
}

TEST_F(StaticLayoutValueTest, ConcurrentPageUpdatesDoNotTear) {
  auto store = std::make_unique<SsdStaticLayoutValueStore>(
      Config(Save(Bandana()), "bandana"));
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 2; ++thread) {
    threads.emplace_back([&, thread] {
      const std::string value(4, static_cast<char>('A' + thread));
      for (int i = 0; i < 100; ++i) {
        store->Write(
            static_cast<uint64_t>(thread + 1), value.data(), value.size());
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::string left(4, '\0');
  std::string right(4, '\0');
  store->Read(1, left.data(), left.size());
  store->Read(2, right.data(), right.size());
  EXPECT_EQ(left, std::string(4, 'A'));
  EXPECT_EQ(right, std::string(4, 'B'));
}

void WriteLegacyCheckpoint(const std::filesystem::path& path,
                           const std::string& metadata) {
  const char magic[]           = {'R', 'S', 'K', 'V', 'C', 'P', '0', '1'};
  const uint32_t version       = 2;
  const uint64_t metadata_size = metadata.size();
  const uint64_t record_count  = 0;
  uint64_t checksum            = 0x9e3779b97f4a7c15ULL;
  auto mix                     = [&](const void* data, size_t size) {
    if (size != 0) {
      checksum = xxhash(data, size, checksum);
    }
  };
  mix(magic, sizeof(magic));
  mix(&version, sizeof(version));
  mix(&metadata_size, sizeof(metadata_size));
  mix(&record_count, sizeof(record_count));
  mix(metadata.data(), metadata.size());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(magic, sizeof(magic));
  output.write(reinterpret_cast<const char*>(&version), sizeof(version));
  output.write(reinterpret_cast<const char*>(&metadata_size),
               sizeof(metadata_size));
  output.write(reinterpret_cast<const char*>(&record_count),
               sizeof(record_count));
  output.write(metadata.data(), metadata.size());
  output.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
}

TEST_F(StaticLayoutValueTest, CheckpointIdentityRoundTripAndMismatch) {
  const std::string metadata = "layout-v3";
  const auto checkpoint      = root_ / "layout.ckpt";
  auto source =
      std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  source->Put(0, std::string("abcd"), 0);
  source->Put(5, std::string("wxyz"), 0);
  ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), metadata));

  auto restored =
      std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), metadata));
  std::string actual;
  restored->Get(0, actual, 0);
  EXPECT_EQ(actual, "abcd");
  restored->Get(5, actual, 0);
  EXPECT_EQ(actual, "wxyz");

  auto mismatched =
      std::make_unique<KVEngineComposite>(Config(Save(MaxEmbed()), "maxembed"));
  EXPECT_FALSE(mismatched->LoadCheckpoint(checkpoint.string(), metadata));
  mismatched->Get(0, actual, 0);
  EXPECT_TRUE(actual.empty());

  const auto legacy = root_ / "legacy.ckpt";
  WriteLegacyCheckpoint(legacy, metadata);
  BaseKVConfig dram;
  dram.num_threads_ = 1;
  dram.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 16},
      {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "dram-value").string()},
        {"default_value_size_hint", 4},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1 << 20}}}}}};
  KVEngineComposite plain(dram);
  EXPECT_TRUE(plain.LoadCheckpoint(legacy.string(), metadata));
  auto rejected =
      std::make_unique<KVEngineComposite>(Config(Save(Bandana()), "bandana"));
  EXPECT_FALSE(rejected->LoadCheckpoint(legacy.string(), metadata));
  rejected->Get(0, actual, 0);
  EXPECT_TRUE(actual.empty());
}

// ── Tiered read cache ───────────────────────────────────────────────────────
//
// The cache is opt-in through `value.tiering.cache_policy`, so these tests turn
// it on explicitly. They drive HybridValueStore/TieredReadCache directly: no
// KVEngine, no real disk (the SSD store uses the in-memory IO backend above).

constexpr size_t kCachedValueSize  = 512;
constexpr uint64_t kCacheDramBytes = 1 << 20;

BaseKVConfig TieredStoreConfig(const std::filesystem::path& root,
                               const std::string& name,
                               const json& tiering) {
  BaseKVConfig config;
  config.num_threads_ = 2;
  config.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", 16},
      {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
      {"value",
       {{"type", "TIERED_VALUE_STORE"},
        {"path", (root / (name + "-value.db")).string()},
        {"default_value_size_hint", kCachedValueSize},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root / (name + "-dram")).string()}}},
        {"ssd_allocator",
         {{"type", "SSD_SLAB"},
          {"capacity_bytes", 64ULL << 20},
          {"path", (root / (name + "-ssd.db")).string()},
          {"io",
           {{"type", "INMEMORY"},
            {"queue_depth", 4},
            {"base_offset_bytes", 0}}}}},
        {"tiering", tiering}}}};
  return config;
}

struct SsdValue {
  uint64_t handle = 0;
  std::string payload;
};

// Allocates values until `want` of them landed on SSD. Primary placement stops
// at high_watermark * (capacity - cache budget), so this needs a few hundred
// iterations; everything stays allocated, which is what fills the DRAM layer.
std::vector<SsdValue> FillUntilSsd(HybridValueStore& store, size_t want) {
  std::vector<SsdValue> out;
  for (uint64_t i = 0; i < 100000 && out.size() < want; ++i) {
    const std::string payload(
        kCachedValueSize, static_cast<char>('A' + (i % 26)));
    const uint64_t handle = store.AllocAndWrite(payload.data(), payload.size());
    if (handle == kValueHandleNone) {
      break;
    }
    if ((handle & (1ULL << 63)) != 0) {
      out.push_back({handle, payload});
    }
  }
  return out;
}

double TieredField(const std::string& info, const std::string& key) {
  const auto pos = info.find(key + "=");
  if (pos == std::string::npos) {
    return -1.0;
  }
  return std::stod(info.substr(pos + key.size() + 1));
}

json LruTieringConfig(
    double budget_ratio, int shards, int promote_min_misses = 1) {
  return {{"cache_policy", "lru"},
          {"cache_budget_ratio", budget_ratio},
          {"cache_shards", shards},
          {"promote_min_misses", promote_min_misses}};
}

// Promotes with the write token that is current right now, which is what a
// read-path caller captures before it starts the SSD read.
void PromoteNow(TieredReadCache& cache,
                uint64_t key,
                const std::string& value) {
  cache.Promote(key, value.data(), value.size(), cache.WriteBarrier(key));
}

TEST_F(StaticLayoutValueTest, TieredReadCachePromotesOnFirstRead) {
  HybridValueStore store(
      TieredStoreConfig(root_, "promote", LruTieringConfig(0.5, 4)));
  const auto ssd = FillUntilSsd(store, 1);
  ASSERT_EQ(ssd.size(), 1U);
  const uint64_t handle = ssd.front().handle;

  std::string out(kCachedValueSize, '\0');
  EXPECT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  EXPECT_EQ(out, ssd.front().payload);

  std::string info = store.ExtraResultFields();
  EXPECT_NE(info.find("tiered_cache_policy=lru"), std::string::npos);
  EXPECT_EQ(TieredField(info, "tiered_cache_misses"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_entries"), 1.0);

  EXPECT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  EXPECT_EQ(out, ssd.front().payload);
  info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits"), 1.0);
  EXPECT_DOUBLE_EQ(TieredField(info, "tiered_cache_hit_rate"), 0.5);
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 1.0);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheWriteThroughKeepsHitsCoherent) {
  HybridValueStore store(
      TieredStoreConfig(root_, "writethrough", LruTieringConfig(0.5, 4)));
  const auto ssd = FillUntilSsd(store, 1);
  ASSERT_EQ(ssd.size(), 1U);
  const uint64_t handle = ssd.front().handle;

  std::string out(kCachedValueSize, '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  ASSERT_EQ(out, ssd.front().payload);

  const std::string updated(kCachedValueSize, 'Z');
  store.Write(handle, updated.data(), updated.size());
  EXPECT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  EXPECT_EQ(out, updated);

  const std::string info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 1.0);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheInvalidatesOnFree) {
  HybridValueStore store(
      TieredStoreConfig(root_, "invalidate", LruTieringConfig(0.5, 4)));
  const auto ssd = FillUntilSsd(store, 1);
  ASSERT_EQ(ssd.size(), 1U);
  const uint64_t handle = ssd.front().handle;

  std::string out(kCachedValueSize, '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_cache_entries"),
            1.0);

  store.Free(handle);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_cache_entries"),
            0.0);

  // The freed SSD slot can be handed out again; a stale cached copy must not
  // resurface through the new handle.
  const std::string fresh(kCachedValueSize, 'Q');
  const uint64_t reused = store.AllocAndWrite(fresh.data(), fresh.size());
  ASSERT_NE(reused, kValueHandleNone);
  EXPECT_EQ(store.Read(reused, out.data(), out.size()), kCachedValueSize);
  EXPECT_EQ(out, fresh);
}

TEST_F(StaticLayoutValueTest, TieredReadCachePromotesValuesSmallerThanSlot) {
  HybridValueStore store(
      TieredStoreConfig(root_, "smaller", LruTieringConfig(0.5, 4)));
  const std::string payload(300, 'S');
  uint64_t handle = kValueHandleNone;
  for (uint64_t i = 0; i < 100000 && handle == kValueHandleNone; ++i) {
    const uint64_t candidate =
        store.AllocAndWrite(payload.data(), payload.size());
    ASSERT_NE(candidate, kValueHandleNone);
    if ((candidate & (1ULL << 63)) != 0) {
      handle = candidate;
    }
  }
  ASSERT_NE(handle, kValueHandleNone);
  // A 300-byte value lives in a 512-byte slab slot, so its length differs from
  // the slot capacity: that must not block promotion.
  ASSERT_NE(store.SlotCapacity(handle), payload.size());

  std::string out(512, '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), payload.size());
  EXPECT_EQ(out.substr(0, payload.size()), payload);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_cache_promotions"),
            1.0);

  ASSERT_EQ(store.Read(handle, out.data(), out.size()), payload.size());
  EXPECT_EQ(out.substr(0, payload.size()), payload);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_cache_hits"), 1.0);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheWriteThroughTracksShrunkValue) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "shrink-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "shrink-dram").string()}}}}}};
  DramValueStore dram(config);
  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);

  TieredReadCache cache(
      &dram, 1, 4 * slot, UINT64_MAX, nullptr, /*promote_min_misses=*/1);
  const std::string wide(kCachedValueSize, 'w');
  PromoteNow(cache, 5, wide);

  const std::string narrow(128, 'n');
  cache.WriteThrough(5, narrow.data(), narrow.size());

  std::string out(kCachedValueSize, '\0');
  size_t copied = 0;
  ASSERT_TRUE(cache.CopyOut(5, out.data(), out.size(), &copied));
  // Must report the new length, not the promoted one, and leave no tail of the
  // previous value.
  EXPECT_EQ(copied, narrow.size());
  EXPECT_EQ(out.substr(0, copied), narrow);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheRejectsPromotionRacingWrite) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "race-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "race-dram").string()}}}}}};
  DramValueStore dram(config);
  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);

  TieredReadCache cache(
      &dram, 1, 4 * slot, UINT64_MAX, nullptr, /*promote_min_misses=*/1);
  const std::string old_value(kCachedValueSize, 'o');
  const std::string new_value(kCachedValueSize, 'n');

  // Reader captures the token, writer overwrites the slot (no entry yet, so
  // WriteThrough only bumps), then the reader tries to insert its old copy.
  const uint64_t stale = cache.WriteBarrier(11);
  cache.WriteThrough(11, new_value.data(), new_value.size());
  cache.Promote(11, old_value.data(), old_value.size(), stale);
  EXPECT_EQ(cache.Entries(), 0U);
  EXPECT_EQ(cache.StalePromotions(), 1U);

  std::string out(kCachedValueSize, '\0');
  size_t copied = 0;
  EXPECT_FALSE(cache.CopyOut(11, out.data(), out.size(), &copied));

  // A read that starts after the write is admitted normally.
  PromoteNow(cache, 11, new_value);
  ASSERT_TRUE(cache.CopyOut(11, out.data(), out.size(), &copied));
  EXPECT_EQ(out, new_value);

  // The same guard covers a freed slot: an in-flight read must not resurrect a
  // copy under a handle that has been recycled.
  const uint64_t freed = cache.WriteBarrier(12);
  cache.Invalidate(12);
  cache.Promote(12, old_value.data(), old_value.size(), freed);
  EXPECT_EQ(cache.Entries(), 1U);
  EXPECT_EQ(cache.StalePromotions(), 2U);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheStaysWithinBudget) {
  const double ratio = 0.5;
  HybridValueStore store(
      TieredStoreConfig(root_, "budget", LruTieringConfig(ratio, 4)));
  const auto ssd = FillUntilSsd(store, 1200);
  ASSERT_EQ(ssd.size(), 1200U);

  std::string out(kCachedValueSize, '\0');
  for (const auto& value : ssd) {
    ASSERT_EQ(store.Read(value.handle, out.data(), out.size()),
              kCachedValueSize);
    ASSERT_EQ(out, value.payload);
  }

  const std::string info = store.ExtraResultFields();
  const double budget    = TieredField(info, "tiered_cache_budget_bytes");
  const double bytes     = TieredField(info, "tiered_cache_bytes");
  EXPECT_DOUBLE_EQ(budget, ratio * static_cast<double>(kCacheDramBytes));
  EXPECT_LE(bytes, budget);
  EXPECT_GT(TieredField(info, "tiered_cache_evictions"), 0.0);

  // The most recently promoted values are still resident, so re-reading them
  // must hit even though the working set is larger than the budget.
  for (size_t i = ssd.size() - 100; i < ssd.size(); ++i) {
    ASSERT_EQ(store.Read(ssd[i].handle, out.data(), out.size()),
              kCachedValueSize);
    ASSERT_EQ(out, ssd[i].payload);
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_cache_hits"), 100.0);
  EXPECT_LE(TieredField(store.ExtraResultFields(), "tiered_cache_bytes"),
            budget);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheEvictsLeastRecentlyUsed) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "lru-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "lru-dram").string()}}}}}};
  DramValueStore dram(config);

  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);
  ASSERT_GE(slot, kCachedValueSize);

  // One shard and a three-slot budget makes the eviction order observable.
  TieredReadCache cache(
      &dram,
      1,
      3 * slot,
      UINT64_MAX,
      nullptr,
      /*promote_min_misses=*/1);
  const std::string a(kCachedValueSize, 'a');
  const std::string b(kCachedValueSize, 'b');
  const std::string c(kCachedValueSize, 'c');
  const std::string d(kCachedValueSize, 'd');
  PromoteNow(cache, 1, a);
  PromoteNow(cache, 2, b);
  PromoteNow(cache, 3, c);

  std::string out(kCachedValueSize, '\0');
  size_t copied = 0;
  ASSERT_TRUE(cache.CopyOut(1, out.data(), out.size(), &copied));
  ASSERT_EQ(out, a);

  PromoteNow(cache, 4, d);
  EXPECT_EQ(cache.Evictions(), 1U);
  EXPECT_FALSE(cache.CopyOut(2, out.data(), out.size(), &copied));
  EXPECT_TRUE(cache.CopyOut(1, out.data(), out.size(), &copied));
  EXPECT_EQ(out, a);
  EXPECT_TRUE(cache.CopyOut(3, out.data(), out.size(), &copied));
  EXPECT_EQ(out, c);
  EXPECT_TRUE(cache.CopyOut(4, out.data(), out.size(), &copied));
  EXPECT_EQ(out, d);
  EXPECT_LE(cache.Bytes(), cache.BudgetBytes());
}

TEST_F(StaticLayoutValueTest, TieredReadCacheSkipsOversizedValues) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "oversize-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "oversize-dram").string()}}}}}};
  DramValueStore dram(config);
  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);

  TieredReadCache cache(
      &dram,
      1,
      slot,
      UINT64_MAX,
      nullptr,
      /*promote_min_misses=*/1);
  const std::string big(4 * slot, 'x');
  PromoteNow(cache, 7, big);
  EXPECT_EQ(cache.Entries(), 0U);
  EXPECT_EQ(cache.Bytes(), 0U);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheRespectsTotalDramLimit) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "limit-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "limit-dram").string()}}}}}};
  DramValueStore dram(config);
  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);

  std::atomic<uint64_t> reserved{0};
  const std::string value(kCachedValueSize, 'v');
  // Budget is generous; the shared watermark is what caps the cache at two
  // entries, so the third promotion has to evict rather than refuse.
  TieredReadCache cache(
      &dram,
      1,
      16 * slot,
      2 * slot,
      &reserved,
      /*promote_min_misses=*/1);
  PromoteNow(cache, 1, value);
  PromoteNow(cache, 2, value);
  EXPECT_EQ(cache.Entries(), 2U);

  PromoteNow(cache, 3, value);
  EXPECT_EQ(cache.Entries(), 2U);
  EXPECT_EQ(cache.Evictions(), 1U);
  EXPECT_EQ(cache.AllocFailures(), 0U);
  EXPECT_LE(cache.Bytes(), 2 * slot);
  size_t copied = 0;
  std::string out(kCachedValueSize, '\0');
  EXPECT_FALSE(cache.CopyOut(1, out.data(), out.size(), &copied));
  EXPECT_TRUE(cache.CopyOut(3, out.data(), out.size(), &copied));

  // Primary placement holding all the room leaves the cache no capacity, so a
  // promotion is refused instead of evicting into an allocator that is full.
  reserved.store(2 * slot);
  EXPECT_EQ(cache.EffectiveBudgetBytes(), 0U);
  PromoteNow(cache, 4, value);
  EXPECT_EQ(cache.Entries(), 2U);
  EXPECT_EQ(cache.AllocFailures(), 1U);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheAdmitsOnSecondMiss) {
  // `promote_min_misses=2` is the admission filter: a key that is missed once
  // (a scan, a cold tail) must not take a DRAM slot away from a hot entry.
  HybridValueStore store(TieredStoreConfig(
      root_, "admit", LruTieringConfig(0.5, 4, /*promote_min_misses=*/2)));
  const auto ssd = FillUntilSsd(store, 1);
  ASSERT_EQ(ssd.size(), 1U);
  const uint64_t handle = ssd.front().handle;

  std::string out(kCachedValueSize, '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  ASSERT_EQ(out, ssd.front().payload);
  std::string info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 0.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_entries"), 0.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_admission_rejects"), 1.0);

  // Second miss: the key proved it repeats, so it is admitted.
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  ASSERT_EQ(out, ssd.front().payload);
  info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_entries"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_admission_rejects"), 1.0);

  // Third read hits, and no further admission work happens.
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), kCachedValueSize);
  info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits"), 1.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 1.0);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheAdmissionSampleStaysBounded) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", (root_ / "bounded-dram").string()},
        {"dram_allocator",
         {{"type", "PERSIST_LOOP_SLAB"},
          {"capacity_bytes", kCacheDramBytes},
          {"path", (root_ / "bounded-dram").string()}}}}}};
  DramValueStore dram(config);
  const uint64_t probe = dram.Alloc(kCachedValueSize);
  ASSERT_NE(probe, kValueHandleNone);
  const size_t slot = dram.SlotCapacity(probe);
  dram.Free(probe);

  const std::string value(kCachedValueSize, 'v');
  // Room for one entry, admission needs two hits: nothing is ever resident, so
  // the miss sample is the only structure that can grow.
  TieredReadCache cache(&dram, 1, slot, UINT64_MAX, nullptr, 2);
  for (uint64_t key = 0; key < 5000; ++key) {
    PromoteNow(cache, key, value);
  }
  EXPECT_EQ(cache.Entries(), 0U);
  EXPECT_EQ(cache.Promotions(), 0U);
  EXPECT_EQ(cache.AdmissionRejects(), 5000U);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheBatchReadPromotesAndHits) {
  HybridValueStore store(
      TieredStoreConfig(root_, "batch", LruTieringConfig(0.5, 4)));
  const auto ssd = FillUntilSsd(store, 2);
  ASSERT_EQ(ssd.size(), 2U);

  const std::vector<uint64_t> handles = {
      ssd[0].handle, ssd[1].handle, ssd[0].handle};
  std::vector<ValueStore::ReadResult> results;
  store.BatchRead(handles, results);
  ASSERT_EQ(results.size(), handles.size());
  EXPECT_EQ(std::string(results[0].data.begin(), results[0].data.end()),
            ssd[0].payload);
  EXPECT_EQ(std::string(results[1].data.begin(), results[1].data.end()),
            ssd[1].payload);
  EXPECT_EQ(std::string(results[2].data.begin(), results[2].data.end()),
            ssd[0].payload);

  std::string info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 2.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_entries"), 2.0);

  store.BatchRead(handles, results);
  ASSERT_EQ(results.size(), handles.size());
  EXPECT_EQ(std::string(results[0].data.begin(), results[0].data.end()),
            ssd[0].payload);
  info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits"), 3.0);
  EXPECT_EQ(TieredField(info, "tiered_cache_promotions"), 2.0);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheIsOffUnlessConfigured) {
  HybridValueStore store(TieredStoreConfig(root_, "nocache", json::object()));
  const auto ssd = FillUntilSsd(store, 1);
  ASSERT_EQ(ssd.size(), 1U);

  std::string out(kCachedValueSize, '\0');
  EXPECT_EQ(store.Read(ssd.front().handle, out.data(), out.size()),
            kCachedValueSize);
  EXPECT_EQ(out, ssd.front().payload);

  const std::string info = store.ExtraResultFields();
  EXPECT_NE(info.find("tiered_cache_policy=none"), std::string::npos);
  EXPECT_EQ(info.find("tiered_cache_hits"), std::string::npos);
  // Without a cache the whole DRAM layer stays available to primary placement.
  EXPECT_EQ(TieredField(info, "tiered_dram_placement_bytes"),
            static_cast<double>(kCacheDramBytes));
}

TEST_F(StaticLayoutValueTest, TieredReadCacheRejectsOutOfRangeKnobs) {
  // A negative count used to wrap through the integer cast: -1 shards became
  // ~SIZE_MAX (std::length_error at construction) and -1 promote_min_misses
  // became ~4.29e9, i.e. a cache that silently never admits anything.
  const json negative_shards = {{"cache_policy", "lru"}, {"cache_shards", -1}};
  EXPECT_THROW(
      HybridValueStore(TieredStoreConfig(root_, "badshards", negative_shards)),
      std::invalid_argument);
  const json zero_shards = {{"cache_policy", "lru"}, {"cache_shards", 0}};
  EXPECT_THROW(
      HybridValueStore(TieredStoreConfig(root_, "zeroshards", zero_shards)),
      std::invalid_argument);
  const json negative_misses = {
      {"cache_policy", "lru"}, {"promote_min_misses", -1}};
  EXPECT_THROW(
      HybridValueStore(TieredStoreConfig(root_, "badmisses", negative_misses)),
      std::invalid_argument);
  const json zero_misses = {{"cache_policy", "lru"}, {"promote_min_misses", 0}};
  EXPECT_THROW(
      HybridValueStore(TieredStoreConfig(root_, "zeromisses", zero_misses)),
      std::invalid_argument);
}

TEST_F(StaticLayoutValueTest, TieredReadCacheConcurrentReadsStayCoherent) {
  HybridValueStore store(
      TieredStoreConfig(root_, "concurrent", LruTieringConfig(0.5, 8)));
  const auto ssd = FillUntilSsd(store, 4);
  ASSERT_EQ(ssd.size(), 4U);

  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  constexpr int kThreads = 4;
  constexpr int kReads   = 200;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&store, &ssd, &mismatches, t] {
      std::string out(kCachedValueSize, '\0');
      for (int i = 0; i < kReads; ++i) {
        const auto& value = ssd[static_cast<size_t>(t + i) % ssd.size()];
        if (store.Read(value.handle, out.data(), out.size()) !=
                kCachedValueSize ||
            out != value.payload) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(mismatches.load(), 0);

  const std::string info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits") +
                TieredField(info, "tiered_cache_misses"),
            static_cast<double>(kThreads * kReads));
  EXPECT_LE(TieredField(info, "tiered_cache_bytes"),
            TieredField(info, "tiered_cache_budget_bytes"));
}

TEST_F(StaticLayoutValueTest, TieredReadCacheConcurrentWriteStaysCoherent) {
  // Readers and a writer share one SSD handle while the cache is too small to
  // hold the working set, so the read path keeps falling back to SSD and
  // re-promoting. That is the interleaving the write barrier exists for: a
  // reader that fetched the pre-write value must not insert it after the write
  // landed, or the cache serves a stale copy until the key is written again.
  // TieredReadCacheRejectsPromotionRacingWrite pins the interleaving
  // deterministically; this test runs the same guard from two real threads.
  constexpr size_t kHandles  = 200;
  constexpr int kReaders     = 4;
  constexpr int kWriterLoops = 20000;
  HybridValueStore store(
      TieredStoreConfig(root_, "concurrent-write", LruTieringConfig(0.02, 4)));
  const auto ssd = FillUntilSsd(store, kHandles);
  ASSERT_EQ(ssd.size(), kHandles);

  const std::string first(kCachedValueSize, 'x');
  const std::string second(kCachedValueSize, 'y');
  const std::string last(kCachedValueSize, 'z');

  std::atomic<bool> stopped{false};
  std::atomic<int> mismatches{0};
  std::atomic<int> reads_done{0};
  std::atomic<int> readers_ready{0};

  std::vector<std::thread> readers;
  readers.reserve(kReaders);
  for (int t = 0; t < kReaders; ++t) {
    readers.emplace_back([&, t] {
      std::string out(kCachedValueSize, '\0');
      int reads = 0;
      readers_ready.fetch_add(1, std::memory_order_release);
      for (int i = 0; i < 1000000 && !stopped.load(std::memory_order_acquire);
           ++i) {
        const size_t index = static_cast<size_t>(t + i) % ssd.size();
        const bool whole =
            store.Read(ssd[index].handle, out.data(), out.size()) ==
            kCachedValueSize;
        // A read racing a write may see either value, but never a mixture of
        // the two (the store writes whole rows under a page lock).
        const bool coherent =
            index == 0 ? (out == ssd[0].payload || out == first ||
                          out == second || out == last)
                       : out == ssd[index].payload;
        if (!whole || !coherent) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
        ++reads;
      }
      reads_done.fetch_add(reads, std::memory_order_relaxed);
    });
  }

  std::thread writer([&] {
    // Start writing only once the readers are running, so the two paths are
    // guaranteed to overlap instead of racing thread startup.
    while (readers_ready.load(std::memory_order_acquire) < kReaders) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kWriterLoops; ++i) {
      const std::string& value = (i % 2 == 0) ? first : second;
      store.Write(ssd.front().handle, value.data(), value.size());
    }
    // Leave behind a value that differs from both racing ones, then let the
    // readers drain so that no read is still in flight afterwards.
    store.Write(ssd.front().handle, last.data(), last.size());
    stopped.store(true, std::memory_order_release);
  });
  writer.join();
  for (auto& thread : readers) {
    thread.join();
  }
  EXPECT_EQ(mismatches.load(), 0);
  EXPECT_GT(reads_done.load(), 0);

  // Every read performs exactly one cache lookup, whichever way it ends.
  const std::string info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_cache_hits") +
                TieredField(info, "tiered_cache_misses"),
            static_cast<double>(reads_done.load()));
  // The reads must have gone back to SSD and re-promoted, otherwise this is
  // just the read-only hit test again.
  EXPECT_GT(TieredField(info, "tiered_cache_misses"), 0.0);
  EXPECT_GT(TieredField(info, "tiered_cache_promotions"), 0.0);

  // The writer's last value must be what the store serves now; a promotion
  // that outlived its barrier would have left the previous copy cached.
  std::string out(kCachedValueSize, '\0');
  ASSERT_EQ(store.Read(ssd.front().handle, out.data(), out.size()),
            kCachedValueSize);
  EXPECT_EQ(out, last);
}

} // namespace
