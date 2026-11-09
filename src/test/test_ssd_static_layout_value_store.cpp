#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <chrono>
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

std::atomic<uint64_t> g_read_entries{0}, g_completed_writes{0};
std::atomic<uint64_t> g_buffer_allocations{0}, g_buffer_frees{0};
std::atomic<bool> g_fail_next_write{false};
std::atomic<bool> g_fail_next_read{false};
std::atomic<bool> g_pause_next_write{false}, g_write_paused{false}, g_release_write{false};
std::atomic<bool> g_pause_next_read{false}, g_read_paused{false},
    g_release_read{false};

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
    auto* buffer = new char[page_count * PAGE_SIZE]();
    ++g_buffer_allocations;
    return buffer;
  }
  void FreeBuffer(char* buf) override {
    ++g_buffer_frees;
    delete[] buf;
  }
  void submit() override {}
  void PollCompletion() override {}
  void BatchReadPages(const std::vector<IOEntry>& entries) override {
    const bool fail = g_fail_next_read.exchange(false);
    if (g_pause_next_read.exchange(false)) {
      g_read_paused.store(true);
      while (!g_release_read.load())
        std::this_thread::yield();
    }
    g_read_entries.fetch_add(entries.size(), std::memory_order_relaxed);
    if (fail)
      throw std::runtime_error("injected read failure");
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
    std::lock_guard<std::mutex> lock(pages_mutex_);
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
    if (g_fail_next_write.exchange(false)) throw std::runtime_error("injected write failure");
    if (g_pause_next_write.exchange(false)) {
      g_write_paused.store(true);
      while (!g_release_write.load()) std::this_thread::yield();
    }
    std::lock_guard<std::mutex> lock(pages_mutex_);
    auto& page = pages_[page_id];
    if (page.empty()) {
      page.resize(PAGE_SIZE, 0);
    }
    std::memcpy(page.data(), buffer, PAGE_SIZE);
    ++g_completed_writes;
  }

private:
  std::mutex pages_mutex_;
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
    g_completed_writes.store(0);
    g_fail_next_write.store(false);
    g_fail_next_read.store(false);
    g_pause_next_write.store(false);
    g_write_paused.store(false);
    g_release_write.store(false);
    g_pause_next_read.store(false);
    g_read_paused.store(false);
    g_release_read.store(false);
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

TEST_F(StaticLayoutValueTest, BatchReadCoalescesPagesAcrossLayouts) {
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
  EXPECT_EQ(plain_reads, 1U);
  EXPECT_NE(plain.ExtraResultFields().find("layout_page_reads=1"),
            std::string::npos);
  EXPECT_NE(plain.ExtraResultFields().find("layout_bytes_read=4096"),
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

TEST_F(StaticLayoutValueTest, SlabBatchReadSharesPagesAndPreservesRowOrder) {
  BaseKVConfig config;
  config.json_config_ = {{"queue_cnt", 4}, {"page_id_offset", 0}};
  CountingIOBackend backend(config);
  SsdSlabAllocator slab(&backend, {128}, 64ULL << 20, 0);
  std::vector<uint64_t> handles;
  std::vector<std::string> values;
  for (size_t i = 0; i < 80; ++i) {
    values.emplace_back(128, static_cast<char>(i));
    handles.push_back(
        slab.AllocAndWrite(values.back().data(), values.back().size()));
  }
  // Slot 31 starts four bytes before a page boundary. Its value crosses pages.
  const std::vector<size_t> rows{31, 0, 62, 32, 79, 31};
  std::vector<std::array<char, 128>> out(rows.size());
  std::vector<SsdBlockAllocator::ReadEntry> entries;
  for (size_t i = 0; i < rows.size(); ++i) {
    entries.push_back({handles[rows[i]], out[i].data()});
  }
  std::vector<size_t> sizes;
  g_read_entries.store(0);
  backend.ResetReadStats();
  slab.BatchRead(entries, sizes);
  EXPECT_EQ(g_read_entries.load(), 1U);
  EXPECT_EQ(backend.GetReadStats().page_reads, 3U);
  ASSERT_EQ(sizes.size(), rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    ASSERT_EQ(sizes[i], values[rows[i]].size());
    EXPECT_EQ(std::string(out[i].data(), sizes[i]), values[rows[i]]);
  }
  values[31] = "short";
  slab.Write(handles[31], values[31].data(), values[31].size());
  slab.BatchRead(entries, sizes);
  EXPECT_EQ(std::string(out.front().data(), sizes.front()), values[31]);
  EXPECT_EQ(std::string(out.back().data(), sizes.back()), values[31]);
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
  uint64_t stale = 0;
  std::string out(kCachedValueSize, '\0');
  size_t copied = 0;
  ASSERT_FALSE(cache.CopyOut(11, out.data(), out.size(), &copied, &stale));
  cache.WriteThrough(11, new_value.data(), new_value.size());
  cache.Promote(11, old_value.data(), old_value.size(), stale);
  EXPECT_EQ(cache.Entries(), 0U);
  EXPECT_EQ(cache.StalePromotions(), 1U);

  EXPECT_FALSE(cache.CopyOut(11, out.data(), out.size(), &copied));

  // A read that starts after the write is admitted normally.
  PromoteNow(cache, 11, new_value);
  ASSERT_TRUE(cache.CopyOut(11, out.data(), out.size(), &copied));
  EXPECT_EQ(out, new_value);

  // The same guard covers a freed slot: an in-flight read must not resurrect a
  // copy under a handle that has been recycled.
  uint64_t freed = 0;
  ASSERT_FALSE(cache.CopyOut(12, out.data(), out.size(), &copied, &freed));
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

TEST_F(StaticLayoutValueTest,
       TieredReadCacheDoesNotDisplaceAFittingWorkingSet) {
  auto tiering                   = LruTieringConfig(0.5, 4);
  const uint64_t expected        = kCacheDramBytes * 4 / 5;
  tiering["expected_dram_bytes"] = expected;
  HybridValueStore store(TieredStoreConfig(root_, "fits", tiering));
  const std::string value(kCachedValueSize, 'f');
  const uint64_t first = store.AllocAndWrite(value.data(), value.size());
  ASSERT_NE(first, kValueHandleNone);
  const size_t rows = expected / store.SlotCapacity(first);
  for (size_t i = 1; i < rows; ++i) {
    const uint64_t handle = store.AllocAndWrite(value.data(), value.size());
    ASSERT_NE(handle, kValueHandleNone);
    ASSERT_NE(store.DirectPtr(handle), nullptr);
  }
  EXPECT_NE(store.ExtraResultFields().find("tiered_cache_policy=none"),
            std::string::npos);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_ssd_live_allocs"),
            0);
  EXPECT_EQ(
      TieredField(store.ExtraResultFields(), "tiered_dram_placement_bytes"),
      kCacheDramBytes);
  // An underestimated working set still spills safely through the SSD path.
  const auto spilled = FillUntilSsd(store, 1);
  ASSERT_EQ(spilled.size(), 1U);
  std::string out(kCachedValueSize, '\0');
  EXPECT_EQ(store.Read(spilled[0].handle, out.data(), out.size()), out.size());
  EXPECT_EQ(out, spilled[0].payload);

  for (int64_t estimate : {int64_t{0}, int64_t{kCacheDramBytes}}) {
    tiering["expected_dram_bytes"] = estimate;
    HybridValueStore cached(
        TieredStoreConfig(root_, std::to_string(estimate), tiering));
    EXPECT_NE(cached.ExtraResultFields().find("tiered_cache_policy=lru"),
              std::string::npos);
  }
  tiering["expected_dram_bytes"] = -1;
  EXPECT_THROW(
      HybridValueStore(TieredStoreConfig(root_, "negative-fit", tiering)),
      std::invalid_argument);
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

} // namespace



namespace {
BaseKVConfig ClockConfig(const std::filesystem::path& root,
                         const std::string& name,
                         size_t slots    = 2,
                         size_t row_size = kCachedValueSize,
                         bool frequency_admission = false) {
  auto config = TieredStoreConfig(
      root,
      name,
      {{"cache_policy", "clock"},
       {"cache_shards", 1},
       {"promote_min_misses", 2},
       {"clock_frequency_admission", frequency_admission}});
  auto probe_config = config;
  probe_config.json_config_["value"]["path"] =
      (root / (name + "-probe")).string();
  DramValueStore probe(probe_config);
  const auto handle     = probe.Alloc(row_size);
  const size_t capacity = probe.SlotCapacity(handle);
  config.json_config_["value"]["tiering"]["high_watermark_ratio"] =
      double(capacity * slots) / kCacheDramBytes;
  config.json_config_["value"]["ssd_allocator"]["capacity_bytes"] =
      256ULL << 20;
  config.json_config_["value"]["default_value_size_hint"] = row_size;
  return config;
}
// Separate candidate pages from the new victim backing page in paused-I/O tests.
std::vector<SsdValue> FillClockValues(HybridValueStore& store, size_t count) {
  std::vector<SsdValue> values;
  for (size_t i = 0; i < count; ++i) {
    std::string payload(kCachedValueSize, static_cast<char>('A' + i % 26));
    const auto handle = store.AllocAndWrite(i, payload.data(), payload.size());
    values.push_back({handle, std::move(payload)});
  }
  return values;
}

bool WaitClockIdle(HybridValueStore& store) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  do {
    if (TieredField(store.ExtraResultFields(), "tiered_clock_exchange_pending") == 0)
      return true;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

struct ReleaseClockWrite {
  ~ReleaseClockWrite() { g_release_write.store(true); }
};

bool PauseClockExchange(HybridValueStore& store, const SsdValue& candidate) {
  g_write_paused.store(false);
  g_release_write.store(false);
  g_pause_next_write.store(true);
  auto read = std::async(std::launch::async, [&] {
    std::string out;
    store.ReadValue(candidate.handle, out);
    return out;
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!g_write_paused.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  const bool paused = g_write_paused.load();
  const auto state = read.wait_for(std::chrono::milliseconds(200));
  if (!paused || state != std::future_status::ready)
    g_release_write.store(true);
  EXPECT_TRUE(paused);
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(read.get(), candidate.payload);
  return paused && state == std::future_status::ready;
}

} // namespace

TEST_F(StaticLayoutValueTest, ClockExchangesColdDramWithHotSsd) {
  auto config = ClockConfig(root_, "clock");
  config.json_config_["value"]["tiering"]["expected_dram_bytes"] = 1;
  config.json_config_["value"]["tiering"]["cache_budget_ratio"]  = 0.0;
  HybridValueStore store(config);
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b'),
      c(kCachedValueSize, 'c');
  auto ha           = store.AllocAndWrite(0, a.data(), a.size());
  auto hb           = store.AllocAndWrite(1, b.data(), b.size());
  auto hc           = store.AllocAndWrite(2, c.data(), c.size());
  const auto before = store.ExtraResultFields();
  ASSERT_EQ(TieredField(before, "tiered_dram_live_allocs"), 2);
  ASSERT_EQ(TieredField(before, "tiered_ssd_live_allocs"), 1);
  EXPECT_EQ(store.DirectPtr(ha), nullptr);
  std::string out;
  store.ReadValue(ha, out); // Protect A; B has not been read.
  EXPECT_EQ(out, a);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, c);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"),
            0);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, c);
  auto after = store.ExtraResultFields();
  EXPECT_EQ(TieredField(after, "tiered_clock_demotions"), 1);
  EXPECT_EQ(TieredField(after, "tiered_clock_promotions"), 1);
  EXPECT_EQ(TieredField(after, "tiered_dram_bytes"),
            TieredField(before, "tiered_dram_bytes"));
  const auto reads = g_read_entries.load();
  store.ReadValue(ha, out);
  EXPECT_EQ(out, a);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, c);
  EXPECT_EQ(g_read_entries.load(), reads);
  store.ReadValue(hb, out); // The original B handle now reads its SSD location.
  EXPECT_EQ(out, b);
  EXPECT_GT(g_read_entries.load(), reads);
  store.Free(hb);
  store.Free(hb);
  store.ReadValue(hb, out);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_live_allocs"), 2);
}

TEST_F(StaticLayoutValueTest,
       ClockBatchAndVariableSizeUpdatesKeepLogicalHandles) {
  HybridValueStore store(ClockConfig(root_, "clock-batch"));
  const std::string value(kCachedValueSize, 'x');
  const std::vector<uint64_t> keys{0, 1, 2, 3};
  std::vector<ValueStore::WriteSpec> specs(4, {value.data(), value.size()});
  const auto handles = store.BatchAllocAndWrite(keys, specs);
  const std::vector<uint64_t> reads{
      handles[3], handles[0], 0, handles[3], handles[2]};
  std::vector<ValueStore::ReadResult> results;
  for (int i = 0; i < 3; ++i) {
    store.BatchRead(reads, results);
    ASSERT_EQ(results.size(), reads.size());
    for (size_t j = 0; j < reads.size(); ++j) {
      EXPECT_EQ(std::string(results[j].data.begin(), results[j].data.end()),
                j == 2 ? "" : value);
    }
  }
  const std::string large(1500, 'L'), short_value(17, 's');
  EXPECT_EQ(store.AllocAndWrite(3, large.data(), large.size()), handles[3]);
  std::string out;
  store.ReadValue(handles[3], out);
  EXPECT_EQ(out, large);
  store.BatchWrite({handles[3], handles[0]},
                   {{short_value.data(), short_value.size()},
                    {short_value.data(), short_value.size()}});
  store.ReadValue(handles[3], out);
  EXPECT_EQ(out, short_value);
  store.ReadValue(handles[0], out);
  EXPECT_EQ(out, short_value);
  const auto info = store.ExtraResultFields();
  EXPECT_LE(TieredField(info, "tiered_dram_bytes"),
            TieredField(info, "tiered_dram_limit_bytes"));
}

TEST_F(StaticLayoutValueTest, ClockFailedDemotionLeavesBothValuesIntact) {
  HybridValueStore store(ClockConfig(root_, "clock-failure", 1));
  const std::string cold(kCachedValueSize, 'c'), hot(kCachedValueSize, 'h');
  const auto hc = store.AllocAndWrite(0, cold.data(), cold.size());
  const auto hh = store.AllocAndWrite(1, hot.data(), hot.size());
  std::string out;
  store.ReadValue(hh, out);
  g_fail_next_write.store(true);
  EXPECT_NO_THROW(store.ReadValue(hh, out));
  EXPECT_EQ(out, hot);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 1);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"),
            0);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, cold);
  // Reading the resident also heats it; retry until the SSD value wins
  // admission.
  for (int i = 0;
       i < 16 &&
       TieredField(store.ExtraResultFields(), "tiered_clock_demotions") == 0;
       ++i)
    store.ReadValue(hh, out);
  EXPECT_EQ(out, hot);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"),
            1);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, cold);
}

TEST_F(StaticLayoutValueTest, ClockConcurrentEngineReadsUpdatesAndExchanges) {
  auto config                     = ClockConfig(root_, "clock-concurrent", 2, kCachedValueSize, true);
  config.num_threads_             = 5;
  config.json_config_["capacity"] = 32;
  KVEngineComposite engine(config);
  for (uint64_t key = 0; key < 8; ++key)
    engine.Put(key, std::string(512, 'a'), 0);
  std::atomic<bool> start{false}, failed{false};
  std::atomic<bool> ready_for_updates{false};
  std::vector<std::thread> threads;
  threads.emplace_back([&] {
    while (!start.load())
      std::this_thread::yield();
    // Readers first exercise replacement at the original fixed size. Growth
    // can otherwise remove all residents before any reader is scheduled.
    while (!ready_for_updates.load())
      std::this_thread::yield();
    for (int i = 0; i < 300; ++i) {
      const std::string value(i % 2 ? 128 : 1024, i % 2 ? 'b' : 'c');
      engine.Put(i % 8, value, 0);
    }
  });
  for (unsigned tid = 1; tid < 5; ++tid) {
    threads.emplace_back([&, tid] {
      while (!start.load())
        std::this_thread::yield();
      std::string out;
      for (int i = 0; i < 500; ++i) {
        if (i == 100)
          ready_for_updates.store(true);
        // Short hot phases make replacement useful before growth starts.
        const uint64_t key = i < 100 ? 2 + i / 25 : (i + tid) % 8;
        engine.Get(key, out, tid);
        if (out.empty() ||
            (out.size() != 128 && out.size() != 512 && out.size() != 1024) ||
            out.find_first_not_of(out[0]) != std::string::npos)
          failed.store(true);
      }
    });
  }
  start.store(true);
  for (auto& thread : threads)
    thread.join();
  EXPECT_FALSE(failed.load());
  EXPECT_GT(TieredField(engine.ExtraResultFields(), "tiered_clock_demotions"),
            0);
  std::string out;
  engine.Put(0, std::string(512, 'z'), 0);
  engine.Get(0, out, 1);
  EXPECT_EQ(out, std::string(512, 'z'));
}

TEST_F(StaticLayoutValueTest, ClockStaticRowsExchangeAndUpdateReplicas) {
  auto config = ClockConfig(root_, "clock-static", 1, 4);
  config.json_config_["value"]["ssd_layout"] = {
      {"strategy", "maxembed"}, {"mapping_path", Save(MaxEmbed()).string()}};
  auto invalid = config;
  invalid.json_config_["value"]["tiering"]["retain_backing_copies"] = false;
  EXPECT_THROW(HybridValueStore rejected(invalid), std::invalid_argument);
  HybridValueStore store(config);
  const std::string a = "aaaa", b = "bbbb", updated = "zzzz";
  auto ha = store.AllocAndWrite(0, a.data(), a.size());
  auto hb = store.AllocAndWrite(1, b.data(), b.size());
  EXPECT_THROW(store.AllocAndWrite(99, a.data(), a.size()),
               std::invalid_argument);
  EXPECT_THROW(store.AllocAndWrite(0, a.data(), 3), std::invalid_argument);
  std::string out;
  store.ReadValue(hb, out);
  store.ReadValue(hb, out);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"),
            1);
  EXPECT_EQ(store.AllocAndWrite(1, updated.data(), updated.size()), hb);
  store.ReadValue(ha, out);
  store.ReadValue(ha, out);
  EXPECT_EQ(out, a);
  std::vector<ValueStore::ReadResult> values;
  store.BatchRead({hb, hb, ha}, values);
  EXPECT_EQ(std::string(values[0].data.begin(), values[0].data.end()), updated);
  EXPECT_EQ(values[0].data, values[1].data);
}

TEST_F(StaticLayoutValueTest, ClockPartialReadNeverPromotesTruncatedValues) {
  HybridValueStore store(ClockConfig(root_, "clock-partial", 1));
  const std::string value(kCachedValueSize, 'p');
  store.AllocAndWrite(0, value.data(), value.size());
  const auto hot = store.AllocAndWrite(1, value.data(), value.size());
  char small[17];
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(store.Read(hot, small, sizeof(small)), sizeof(small));
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
            0);
  std::string out;
  store.ReadValue(hot, out);
  EXPECT_EQ(out, value);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
            1);
  store.ReadValue(hot, out);
  EXPECT_EQ(out, value);
}

TEST_F(StaticLayoutValueTest, ClockSsdExhaustionDoesNotLoseColdDram) {
  class LimitedSsd : public SsdValueStore {
  public:
    using SsdValueStore::SsdValueStore;
    bool full = false;
    uint64_t Alloc(size_t size) override {
      return full ? kValueHandleNone : SsdValueStore::Alloc(size);
    }
  };
  auto config = ClockConfig(root_, "clock-full", 1);
  DramValueStore dram(config);
  LimitedSsd ssd(config);
  const auto limit = static_cast<uint64_t>(
      config.json_config_["value"]["tiering"]["high_watermark_ratio"]
          .get<double>() *
      kCacheDramBytes);
  TieredClockValueStore store(&dram, &ssd, limit, 1, 2);
  const std::string cold(kCachedValueSize, 'c'), hot(kCachedValueSize, 'h');
  const auto hc = store.AllocAndWrite(0, cold.data(), cold.size());
  const auto hh = store.AllocAndWrite(1, hot.data(), hot.size());
  std::string out;
  store.ReadValue(hh, out);
  ssd.full = true;
  store.ReadValue(hh, out);
  EXPECT_EQ(out, hot);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 1);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"),
            0);
  store.ReadValue(hc, out);
  EXPECT_EQ(out, cold);
}

TEST_F(StaticLayoutValueTest, ClockFlatReadsAndCheckpointSurviveExchanges) {
  auto config = ClockConfig(root_, "clock-flat", 1, sizeof(float), true);
  KVEngineComposite engine(config);
  const std::vector<uint64_t> keys{0, 1, 2};
  const std::array<float, 3> values{1.5F, 2.5F, 3.5F};
  for (size_t i = 0; i < keys.size(); ++i)
    engine.Put(
        keys[i],
        std::string(reinterpret_cast<const char*>(&values[i]), sizeof(float)),
        0);
  // Repeated equal-frequency batches need not exchange tiers. Make an SSD
  // row genuinely hot before testing mixed reads and checkpoint restoration.
  const std::vector<uint64_t> hot_key{1};
  float hot_value = 0;
  for (int i = 0; i < 64; ++i)
    ASSERT_TRUE(engine.BatchGetFlat(
        base::ConstArray<uint64_t>(hot_key), &hot_value, 1, 1, 0));
  EXPECT_EQ(hot_value, values[1]);
  std::array<float, 3> out{};
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(engine.BatchGetFlat(
        base::ConstArray<uint64_t>(keys), out.data(), 3, 1, 0));
    EXPECT_EQ(out, values);
  }
  EXPECT_GT(TieredField(engine.ExtraResultFields(), "tiered_clock_demotions"),
            0);
  const auto path = (root_ / "clock.ckpt").string();
  ASSERT_TRUE(engine.SaveCheckpoint(path, "clock"));
  KVEngineComposite restored(
      ClockConfig(root_, "clock-restored", 1, sizeof(float)));
  ASSERT_TRUE(restored.LoadCheckpoint(path, "clock"));
  EXPECT_TRUE(restored.BatchGetFlat(
      base::ConstArray<uint64_t>(keys), out.data(), 3, 1, 0));
  EXPECT_EQ(out, values);
}

TEST_F(StaticLayoutValueTest, ClockDirectBulkFallbackReleasesUnpublishedRows) {
  auto config =
      ClockConfig(root_, "clock-bulk", 2, sizeof(int64_t) + sizeof(float));
  KVEngineComposite engine(config);
  const std::vector<uint64_t> keys{0, 1};
  const int64_t indices[]{0, 1};
  const float values[]{1.0F, 2.0F};
  EXPECT_FALSE(engine.BulkLoadIndexedFloatRange(
      base::ConstArray<uint64_t>(keys), indices, values, 2, 1, 0));
  EXPECT_EQ(TieredField(engine.ExtraResultFields(), "tiered_live_allocs"), 0);
  EXPECT_EQ(TieredField(engine.ExtraResultFields(), "tiered_dram_bytes"), 0);
}


TEST_F(StaticLayoutValueTest, ClockReusedMetadataRejectsStaleHandles) {
  HybridValueStore store(ClockConfig(root_, "clock-reuse", 1));
  const std::string value(kCachedValueSize, 'r');
  uint64_t old = store.AllocAndWrite(0, value.data(), value.size());
  for (int i = 0; i < 100; ++i) {
    store.Free(old);
    const auto current = store.AllocAndWrite(0, value.data(), value.size());
    EXPECT_NE(current, old);
    EXPECT_EQ(static_cast<uint32_t>(current), static_cast<uint32_t>(old));
    store.Write(old, "bad", 3);
    store.Free(old);
    std::vector<ValueStore::ReadResult> results;
    store.BatchRead({old, current, old, current}, results);
    EXPECT_TRUE(results[0].data.empty());
    EXPECT_TRUE(results[2].data.empty());
    EXPECT_EQ(std::string(results[1].data.begin(), results[1].data.end()), value);
    EXPECT_EQ(results[1].data, results[3].data);
    old = current;
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_metadata_slots"), 1);
}

TEST_F(StaticLayoutValueTest, ClockRingUnlinksAndReusesAllPositions) {
  for (bool frequency : {false, true}) {
    SCOPED_TRACE(frequency);
    auto config = ClockConfig(root_, frequency ? "clock-ring-frequency"
                                              : "clock-ring-legacy",
                              4, kCachedValueSize, frequency);
    config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
    HybridValueStore store(config);
    std::array<uint64_t, 6> handles;
    std::array<std::string, 6> values;
    for (size_t i = 0; i < handles.size(); ++i) {
      values[i] = std::string(kCachedValueSize, 'a' + i);
      handles[i] = store.AllocAndWrite(i, values[i].data(), values[i].size());
    }
    const auto budget = TieredField(store.ExtraResultFields(),
                                   "tiered_dram_bytes");
    uint64_t next_key = handles.size();
    std::string out;
    // Remove a middle resident, the head, and the newly appended tail.
    for (size_t index : {1U, 0U, 0U}) {
      const uint64_t retired = handles[index];
      store.Free(retired);
      values[index] = std::string(kCachedValueSize, 'a' + next_key);
      handles[index] = store.AllocAndWrite(
          next_key++, values[index].data(), values[index].size());
      EXPECT_NE(handles[index], retired);
      EXPECT_EQ(static_cast<uint32_t>(handles[index]),
                static_cast<uint32_t>(retired));
      store.Free(retired);
      store.ReadValue(handles[index], out);
      EXPECT_EQ(out, values[index]);
      const auto info = store.ExtraResultFields();
      EXPECT_EQ(TieredField(info, "tiered_dram_live_allocs"), 4);
      EXPECT_EQ(TieredField(info, "tiered_ssd_live_allocs"), 2);
      EXPECT_EQ(TieredField(info, "tiered_dram_bytes"), budget);
    }
    for (size_t index : {4U, 5U}) {
      const auto before = TieredField(store.ExtraResultFields(),
                                     "tiered_clock_promotions");
      for (int retry = 0; retry < 64; ++retry) {
        store.ReadValue(handles[index], out);
        if (TieredField(store.ExtraResultFields(), "tiered_clock_promotions") !=
            before)
          break;
      }
      EXPECT_EQ(out, values[index]);
      EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
                before + 1);
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      store.ReadValue(handles[i], out);
      EXPECT_EQ(out, values[i]);
      EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_dram_bytes"),
                budget);
    }
    for (uint64_t handle : handles)
      store.Free(handle);
    const auto empty = store.ExtraResultFields();
    EXPECT_EQ(TieredField(empty, "tiered_live_allocs"), 0);
    EXPECT_EQ(TieredField(empty, "tiered_dram_bytes"), 0);
    EXPECT_EQ(TieredField(empty, "tiered_clock_failures"), 0);
    // The final resident's removal must leave a reusable empty ring.
    const auto fresh = store.AllocAndWrite(100, values[0].data(), values[0].size());
    store.ReadValue(fresh, out);
    EXPECT_EQ(out, values[0]);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_metadata_slots"),
              handles.size());
    store.Free(fresh);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_live_allocs"), 0);
  }
}

TEST_F(StaticLayoutValueTest, ClockCountersSurviveReaderExitAndSlotReuse) {
  for (bool frequency : {false, true}) {
    SCOPED_TRACE(frequency);
    auto config = ClockConfig(root_, frequency ? "clock-count-frequency"
                                              : "clock-count-legacy",
                              2, kCachedValueSize, frequency);
    config.json_config_["value"]["tiering"]["promote_min_misses"] = 1000;
    HybridValueStore store(config);
    const std::string value(kCachedValueSize, 'c');
    std::array<uint64_t, 3> handles;
    for (size_t i = 0; i < handles.size(); ++i)
      handles[i] = store.AllocAndWrite(i, value.data(), value.size());
    std::thread reader([&] {
      std::string out;
      store.ReadValue(handles[0], out);
      EXPECT_EQ(out, value);
      std::array<char, kCachedValueSize> bytes{};
      EXPECT_EQ(store.Read(handles[1], bytes.data(), bytes.size()), value.size());
      EXPECT_EQ(std::string(bytes.data(), bytes.size()), value);
      std::vector<ValueStore::ReadResult> results;
      store.BatchRead({handles[0], handles[2], handles[0], handles[2],
                       kValueHandleNone}, results);
      ASSERT_EQ(results.size(), 5);
      for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
                  value);
      EXPECT_TRUE(results[4].data.empty());
      // Cross two 64-row guard boundaries with duplicate hits and misses.
      std::vector<uint64_t> batch(130, handles[1]);
      for (size_t i : {0U, 63U, 64U, 129U})
        batch[i] = handles[2];
      batch[1] = batch[128] = kValueHandleNone;
      std::vector<char> flat(batch.size() * value.size());
      uint64_t missing = 0;
      ASSERT_TRUE(store.ReadFlatFixedRows(batch.data(), batch.size(), flat.data(),
                                         value.size(), &missing));
      EXPECT_EQ(missing, 2);
      for (size_t i = 0; i < batch.size(); ++i)
        EXPECT_EQ(std::string(flat.data() + i * value.size(), value.size()),
                  i == 1 || i == 128 ? std::string(value.size(), '\0') : value);
    });
    reader.join();
    for (bool free_slots : {false, true}) {
      if (free_slots)
        for (uint64_t handle : handles)
          store.Free(handle);
      const auto info = store.ExtraResultFields();
      EXPECT_EQ(TieredField(info, "tiered_clock_hits"), 128);
      EXPECT_EQ(TieredField(info, "tiered_clock_misses"), 6);
      EXPECT_EQ(TieredField(info, "tiered_clock_admission_rejects"), 6);
      EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 0);
      EXPECT_EQ(TieredField(info, "tiered_clock_writebacks"), 0);
    }
    const auto reused = store.AllocAndWrite(7, value.data(), value.size());
    std::string out;
    store.ReadValue(reused, out);
    EXPECT_EQ(out, value);
    store.Free(reused);
    const auto info = store.ExtraResultFields();
    EXPECT_EQ(TieredField(info, "tiered_clock_hits"), 129);
    EXPECT_EQ(TieredField(info, "tiered_clock_misses"), 6);
    EXPECT_EQ(TieredField(info, "tiered_live_allocs"), 0);
    EXPECT_EQ(TieredField(info, "tiered_clock_metadata_slots"), handles.size());
  }
}

TEST_F(StaticLayoutValueTest, ClockExchangeQueueRejectsNonpositiveLimits) {
  for (const char* field : {"clock_exchange_queue_entries",
                            "clock_exchange_queue_bytes"}) {
    for (int64_t limit : {0, -1}) {
      SCOPED_TRACE(field);
      SCOPED_TRACE(limit);
      auto config = ClockConfig(root_, "clock-invalid-queue", 1);
      config.json_config_["value"]["tiering"][field] = limit;
      EXPECT_THROW(HybridValueStore rejected(config), std::invalid_argument);
    }
  }
}

TEST_F(StaticLayoutValueTest, ClockSampledHeatProtectsPeriodicResidents) {
  for (bool cost_aware : {false, true}) {
    for (bool flat : {false, true}) {
      SCOPED_TRACE(cost_aware);
      SCOPED_TRACE(flat);
      auto config = ClockConfig(root_, "clock-periodic", 16);
      config.json_config_["value"]["tiering"]["clock_cost_aware"] = cost_aware;
      HybridValueStore store(config);
      const auto values = FillClockValues(store, 17);
      std::array<uint64_t, 16> handles;
      for (size_t key = 0; key < handles.size(); ++key)
        handles[key] = values[key].handle;
      std::vector<char> buffer(handles.size() * kCachedValueSize);
      std::string out;
      // A fixed 1-in-16 sample heats only the first position in this workload.
      // Every resident is repeatedly used and should resist a rare SSD visitor.
      for (int round = 0; round < 128; ++round) {
        if (flat) {
          uint64_t missing = 0;
          ASSERT_TRUE(store.ReadFlatFixedRows(handles.data(), handles.size(),
                                             buffer.data(), kCachedValueSize,
                                             &missing));
          ASSERT_EQ(missing, 0);
          for (size_t key = 0; key < handles.size(); ++key)
            ASSERT_EQ(std::string(buffer.data() + key * kCachedValueSize,
                                  kCachedValueSize), values[key].payload);
        } else {
          for (size_t key = 0; key < handles.size(); ++key) {
            store.ReadValue(handles[key], out);
            ASSERT_EQ(out, values[key].payload);
          }
        }
      }
      const auto reads = g_read_entries.load();
      for (int access = 0; access < 2; ++access) {
        store.ReadValue(values[16].handle, out);
        EXPECT_EQ(out, values[16].payload);
      }
      EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
      for (size_t key = 0; key < handles.size(); ++key) {
        store.ReadValue(handles[key], out);
        EXPECT_EQ(out, values[key].payload);
      }
      EXPECT_EQ(g_read_entries.load(), reads + 2);
    }
  }
}

TEST_F(StaticLayoutValueTest, ClockCostAwareRequiresRepeatedDemandForEveryVictim) {
  auto config = ClockConfig(root_, "clock-cost-dirty", 1);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  HybridValueStore dirty(config);
  const auto values = FillClockValues(dirty, 2);
  std::string out;
  dirty.ReadValue(values[1].handle, out);
  EXPECT_EQ(out, values[1].payload);
  EXPECT_EQ(TieredField(dirty.ExtraResultFields(), "tiered_clock_promotions"), 0);
  dirty.ReadValue(values[1].handle, out);
  EXPECT_EQ(TieredField(dirty.ExtraResultFields(), "tiered_clock_promotions"), 1);
  EXPECT_EQ(TieredField(dirty.ExtraResultFields(), "tiered_clock_writebacks"), 1);

  config = ClockConfig(root_, "clock-cost-clean", 1);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  HybridValueStore clean(config);
  auto backed = FillClockValues(clean, 2);
  clean.Free(backed[0].handle);
  clean.ReadValue(backed[1].handle, out);
  clean.ReadValue(backed[1].handle, out); // Promote into free DRAM, keeping backing.
  const auto candidate = clean.AllocAndWrite(2, values[0].payload.data(),
                                           values[0].payload.size());
  clean.ReadValue(candidate, out);
  EXPECT_EQ(TieredField(clean.ExtraResultFields(), "tiered_clock_promotions"), 1);
  clean.ReadValue(candidate, out);
  EXPECT_EQ(out, values[0].payload);
  const auto info = clean.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 2);
  EXPECT_EQ(TieredField(info, "tiered_clock_clean_evictions"), 1);
  EXPECT_EQ(TieredField(info, "tiered_clock_writebacks"), 0);
}

TEST_F(StaticLayoutValueTest, ClockCostAwarePrefersCleanWhenVictimHeatMatches) {
  auto config = ClockConfig(root_, "clock-cost-tie", 2);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 16;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 4);
  std::string out;
  for (int i = 0; i < 16; ++i)
    store.ReadValue(values[2].handle, out);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"), 1);
  // The first sampled DRAM hit gives the dirty resident the same heat (16)
  // as the clean newcomer. Both lose their reference bit and halve to 8
  // before becoming eligible, so clean backing only breaks a genuine tie.
  const auto before_hit = g_read_entries.load();
  store.ReadValue(values[1].handle, out);
  ASSERT_EQ(out, values[1].payload);
  ASSERT_EQ(g_read_entries.load(), before_hit);
  for (int i = 0; i < 16; ++i)
    store.ReadValue(values[3].handle, out);
  EXPECT_EQ(out, values[3].payload);
  const auto info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 2);
  EXPECT_EQ(TieredField(info, "tiered_clock_writebacks"), 1);
  EXPECT_EQ(TieredField(info, "tiered_clock_clean_evictions"), 1);
  const auto reads = g_read_entries.load();
  store.ReadValue(values[1].handle, out);
  EXPECT_EQ(out, values[1].payload);
  store.ReadValue(values[3].handle, out);
  EXPECT_EQ(out, values[3].payload);
  EXPECT_EQ(g_read_entries.load(), reads);
}

TEST_F(StaticLayoutValueTest, ClockCostLookaheadPreservesLegacyHeat) {
  auto config = ClockConfig(root_, "clock-lookahead-heat", 2);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 4);
  std::string out;
  for (int access = 0; access < 32; ++access) {
    store.ReadValue(values[1].handle, out);
    ASSERT_EQ(out, values[1].payload);
  }
  // Key 0 is already eligible. Looking beyond it must not clear key 1's
  // reference or halve its heat while choosing a cheaper victim.
  for (int access = 0; access < 2; ++access) {
    store.ReadValue(values[2].handle, out);
    ASSERT_EQ(out, values[2].payload);
  }
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  for (int access = 0; access < 256; ++access) {
    store.ReadValue(values[2].handle, out);
    ASSERT_EQ(out, values[2].payload);
  }
  // Partial reads build demand without triggering a promotion scan. The
  // next full read can expose heat lost during the earlier lookahead.
  char prefix = 0;
  for (int access = 0; access < 6; ++access) {
    ASSERT_EQ(store.Read(values[3].handle, &prefix, 1), size_t{1});
    ASSERT_EQ(prefix, values[3].payload.front());
  }
  store.ReadValue(values[3].handle, out);
  EXPECT_EQ(out, values[3].payload);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  const auto reads = g_read_entries.load();
  store.ReadValue(values[1].handle, out);
  EXPECT_EQ(out, values[1].payload);
  EXPECT_EQ(g_read_entries.load(), reads);

  // Primary CLOCK scans still age residents as the working set changes.
  for (int access = 0; access < 16; ++access) {
    store.ReadValue(values[3].handle, out);
    ASSERT_EQ(out, values[3].payload);
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 2);
  const auto adapted_reads = g_read_entries.load();
  store.ReadValue(values[3].handle, out);
  EXPECT_EQ(out, values[3].payload);
  EXPECT_EQ(g_read_entries.load(), adapted_reads);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 0);
}

TEST_F(StaticLayoutValueTest, ClockCostLookaheadPreservesFrequencyReference) {
  auto config = ClockConfig(root_, "clock-lookahead-reference", 2,
                            kCachedValueSize, true);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 4);
  std::string out;
  for (int access = 0; access < 32; ++access) {
    store.ReadValue(values[0].handle, out);
    ASSERT_EQ(out, values[0].payload);
  }
  for (int access = 0; access < 2; ++access) {
    store.ReadValue(values[2].handle, out);
    ASSERT_EQ(out, values[2].payload);
  }
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"), 1);
  // The bootstrap scan consumed dirty key 0's reference. Newly promoted
  // clean key 2 still has its second chance, which cost lookahead must keep.
  char prefix = 0;
  for (int access = 0; access < 48; ++access) {
    ASSERT_EQ(store.Read(values[3].handle, &prefix, 1), size_t{1});
    ASSERT_EQ(prefix, values[3].payload.front());
  }
  store.ReadValue(values[3].handle, out);
  EXPECT_EQ(out, values[3].payload);
  const auto info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 2);
  EXPECT_EQ(TieredField(info, "tiered_clock_writebacks"), 2);
  EXPECT_EQ(TieredField(info, "tiered_clock_clean_evictions"), 0);
  const auto reads = g_read_entries.load();
  store.ReadValue(values[2].handle, out);
  EXPECT_EQ(out, values[2].payload);
  for (int access = 0; access < 16; ++access) {
    store.ReadValue(values[3].handle, out);
    ASSERT_EQ(out, values[3].payload);
  }
  EXPECT_EQ(g_read_entries.load(), reads);

  // A later return to the previous hot key must also adapt online.
  for (int access = 0; access < 64; ++access) {
    store.ReadValue(values[0].handle, out);
    ASSERT_EQ(out, values[0].payload);
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 3);
  const auto adapted_reads = g_read_entries.load();
  store.ReadValue(values[0].handle, out);
  EXPECT_EQ(out, values[0].payload);
  EXPECT_EQ(g_read_entries.load(), adapted_reads);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 0);
}

TEST_F(StaticLayoutValueTest, ClockCostAwareRechecksMissGateAfterDemotion) {
  for (bool asynchronous : {false, true}) {
    SCOPED_TRACE(asynchronous);
    auto config = ClockConfig(root_, asynchronous ? "clock-readmit-async"
                                                  : "clock-readmit-sync", 1);
    config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
    config.json_config_["value"]["tiering"]["clock_async_exchange"] = asynchronous;
    HybridValueStore store(config);
    const auto values = FillClockValues(store, 2);
    std::string out;
    // Admit a key, then demote it through a clean swap.
    for (int access = 0; access < 2; ++access) {
      store.ReadValue(values[1].handle, out);
      ASSERT_EQ(out, values[1].payload);
      if (asynchronous)
        ASSERT_TRUE(WaitClockIdle(store));
    }
    ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
    ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"), 1);
    for (int access = 0; access < 2; ++access) {
      store.ReadValue(values[0].handle, out);
      ASSERT_EQ(out, values[0].payload);
      if (asynchronous)
        ASSERT_TRUE(WaitClockIdle(store));
    }
    const auto before = store.ExtraResultFields();
    ASSERT_EQ(TieredField(before, "tiered_clock_promotions"), 2);
    const auto reads = g_read_entries.load();
    const auto misses = TieredField(before, "tiered_clock_misses");
    // Cost ranking must not bypass the per-residency miss gate.
    store.ReadValue(values[1].handle, out);
    EXPECT_EQ(out, values[1].payload);
    if (asynchronous)
      ASSERT_TRUE(WaitClockIdle(store));
    const auto first = store.ExtraResultFields();
    EXPECT_EQ(TieredField(first, "tiered_clock_promotions"), 2);
    EXPECT_EQ(TieredField(first, "tiered_clock_misses"), misses + 1);
    EXPECT_EQ(TieredField(first, "tiered_clock_admission_rejects"),
              TieredField(before, "tiered_clock_admission_rejects") + 1);
    store.ReadValue(values[1].handle, out);
    EXPECT_EQ(out, values[1].payload);
    if (asynchronous)
      ASSERT_TRUE(WaitClockIdle(store));
    const auto second = store.ExtraResultFields();
    EXPECT_EQ(TieredField(second, "tiered_clock_promotions"), 3);
    EXPECT_EQ(TieredField(second, "tiered_clock_misses"), misses + 2);
    EXPECT_EQ(TieredField(second, "tiered_clock_writebacks"), 1);
    EXPECT_EQ(g_read_entries.load(), reads + 2);
    store.ReadValue(values[1].handle, out);
    EXPECT_EQ(out, values[1].payload);
    EXPECT_EQ(g_read_entries.load(), reads + 2);
  }
}

TEST_F(StaticLayoutValueTest, ClockCostAwareDoesNotStrandUnusedDirtyResidents) {
  for (bool asynchronous : {false, true}) {
    SCOPED_TRACE(asynchronous);
    auto config = ClockConfig(root_, asynchronous ? "clock-fairness-async"
                                                  : "clock-fairness-sync", 4);
    config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
    config.json_config_["value"]["tiering"]["clock_async_exchange"] = asynchronous;
    HybridValueStore store(config);
    const auto values = FillClockValues(store, 9);
    store.Free(values[0].handle);
    std::string out;
    for (int access = 0; access < 2; ++access) {
      store.ReadValue(values[4].handle, out);
      ASSERT_EQ(out, values[4].payload);
      if (asynchronous)
        ASSERT_TRUE(WaitClockIdle(store));
    }
    const auto before = store.ExtraResultFields();
    ASSERT_EQ(TieredField(before, "tiered_dram_live_allocs"), 4);
    ASSERT_EQ(TieredField(before, "tiered_ssd_live_allocs"), 4);
    ASSERT_EQ(TieredField(before, "tiered_clock_promotions"), 1);
    ASSERT_EQ(TieredField(before, "tiered_clock_writebacks"), 0);
    const auto initial_reads = g_read_entries.load();
    // Three unused dirty residents must yield capacity to the new working set.
    // A hotter clean resident must not protect unused dirty slots.
    for (int round = 0; round < 128; ++round) {
      for (size_t key = 5; key < 9; ++key) {
        for (int access = 0; access < 2; ++access) {
          store.ReadValue(values[key].handle, out);
          ASSERT_EQ(out, values[key].payload);
          if (asynchronous)
            ASSERT_TRUE(WaitClockIdle(store));
        }
      }
    }
    const auto settled = store.ExtraResultFields();
    SCOPED_TRACE(::testing::Message()
                 << "promotions="
                 << TieredField(settled, "tiered_clock_promotions") -
                        TieredField(before, "tiered_clock_promotions")
                 << " clean_evictions="
                 << TieredField(settled, "tiered_clock_clean_evictions")
                 << " writebacks="
                 << TieredField(settled, "tiered_clock_writebacks")
                 << " read_entries=" << g_read_entries.load() - initial_reads);
    EXPECT_GE(TieredField(settled, "tiered_clock_writebacks"), 3);
    EXPECT_EQ(TieredField(settled, "tiered_dram_bytes"),
              TieredField(before, "tiered_dram_bytes"));
    EXPECT_EQ(TieredField(settled, "tiered_clock_failures"), 0);
    const auto reads = g_read_entries.load();
    for (int repeat = 0; repeat < 8; ++repeat)
      for (size_t key = 5; key < 9; ++key) {
        store.ReadValue(values[key].handle, out);
        EXPECT_EQ(out, values[key].payload);
      }
    if (asynchronous)
      ASSERT_TRUE(WaitClockIdle(store));
    EXPECT_EQ(g_read_entries.load(), reads);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_misses"),
              TieredField(settled, "tiered_clock_misses"));
    for (size_t key = 1; key < 5; ++key) {
      store.ReadValue(values[key].handle, out);
      EXPECT_EQ(out, values[key].payload);
    }
    if (asynchronous)
      ASSERT_TRUE(WaitClockIdle(store));
  }
}

TEST_F(StaticLayoutValueTest, ClockAsyncReturnsBeforeWriteAndDestructorDrainsQueue) {
  auto config = ClockConfig(root_, "clock-async-drain", 2);
  config.json_config_["value"]["tiering"]["clock_async_exchange"] = true;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  auto store = std::make_unique<HybridValueStore>(config);
  const auto values = FillClockValues(*store, 66);
  ReleaseClockWrite release;
  ASSERT_TRUE(PauseClockExchange(*store, values[2]));
  auto next = std::async(std::launch::async, [&] {
    std::string out;
    // Put this reader's sampled heat on the victim already leaving DRAM.
    store->ReadValue(values[0].handle, out);
    EXPECT_EQ(out, values[0].payload);
    store->ReadValue(values[1].handle, out);
    EXPECT_EQ(out, values[1].payload);
    store->ReadValue(values[10].handle, out);
    return out;
  });
  const auto state = next.wait_for(std::chrono::milliseconds(200));
  if (state != std::future_status::ready)
    g_release_write.store(true);
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(next.get(), values[10].payload);
  if (state != std::future_status::ready)
    return;
  EXPECT_EQ(TieredField(store->ExtraResultFields(), "tiered_clock_exchange_pending"), 2);
  const auto writes = g_completed_writes.load();
  auto shutdown = std::async(std::launch::async, [&] { store.reset(); });
  EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  g_release_write.store(true);
  EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(3)), std::future_status::ready);
  shutdown.get();
  // Both the active and queued job had dirty victims and must finish on shutdown.
  EXPECT_GE(g_completed_writes.load() - writes, 2);
}

TEST_F(StaticLayoutValueTest, ClockAsyncBoundsActiveAndQueuedEntriesAndBytes) {
  for (bool byte_limit : {false, true}) {
    SCOPED_TRACE(byte_limit);
    auto config = ClockConfig(root_, byte_limit ? "clock-async-bytes"
                                                : "clock-async-entries", 2);
    auto& tiering = config.json_config_["value"]["tiering"];
    tiering["clock_async_exchange"] = true;
    tiering["promote_min_misses"] = 1;
    tiering["clock_exchange_queue_entries"] = byte_limit ? 8 : 2;
    tiering["clock_exchange_queue_bytes"] = byte_limit ? 2 * kCachedValueSize
                                                       : 4 * 1024 * 1024;
    HybridValueStore store(config);
    const auto values = FillClockValues(store, 66);
    ReleaseClockWrite release;
    ASSERT_TRUE(PauseClockExchange(store, values[2]));
    auto submit = std::async(std::launch::async, [&] {
      for (size_t i : {10U, 10U, 18U}) {
        std::string out;
        store.ReadValue(values[i].handle, out);
        EXPECT_EQ(out, values[i].payload);
      }
    });
    const auto state = submit.wait_for(std::chrono::milliseconds(200));
    if (state != std::future_status::ready)
      g_release_write.store(true);
    EXPECT_EQ(state, std::future_status::ready);
    submit.get();
    if (state != std::future_status::ready)
      return;
    const auto info = store.ExtraResultFields();
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_pending"), 2);
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_inflight"), 1);
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_pending_bytes"),
              2 * kCachedValueSize);
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_enqueued"), 2);
    EXPECT_GE(TieredField(info, "tiered_clock_exchange_deduplicated"), 1);
    EXPECT_GE(TieredField(info, "tiered_clock_exchange_dropped"), 1);
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_peak_entries"), 2);
    EXPECT_EQ(TieredField(info, "tiered_clock_exchange_peak_bytes"),
              2 * kCachedValueSize);
    g_release_write.store(true);
    ASSERT_TRUE(WaitClockIdle(store));
    EXPECT_EQ(TieredField(store.ExtraResultFields(),
                         "tiered_clock_exchange_pending_bytes"), 0);
  }
}

TEST_F(StaticLayoutValueTest, ClockAsyncOversizeValueStillReturnsWithoutExchange) {
  auto config = ClockConfig(root_, "clock-async-oversize", 2);
  auto& tiering = config.json_config_["value"]["tiering"];
  tiering["clock_async_exchange"] = true;
  tiering["promote_min_misses"] = 1;
  tiering["clock_exchange_queue_bytes"] = kCachedValueSize - 1;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 3);
  std::string out;
  store.ReadValue(values[2].handle, out);
  EXPECT_EQ(out, values[2].payload);
  const auto info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_clock_exchange_enqueued"), 0);
  EXPECT_EQ(TieredField(info, "tiered_clock_exchange_pending"), 0);
  EXPECT_EQ(TieredField(info, "tiered_clock_exchange_dropped"), 1);
  EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 0);
  EXPECT_EQ(TieredField(info, "tiered_clock_writebacks"), 0);
}

TEST_F(StaticLayoutValueTest, ClockAsyncRejectsQueuedWritesAndReusedHandles) {
  auto config = ClockConfig(root_, "clock-async-stale", 2);
  config.json_config_["value"]["tiering"]["clock_async_exchange"] = true;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 66);
  const std::string updated(kCachedValueSize, 'u'), replacement(kCachedValueSize, 'r');
  uint64_t reused = 0;
  ReleaseClockWrite release;
  ASSERT_TRUE(PauseClockExchange(store, values[2]));
  auto changes = std::async(std::launch::async, [&] {
    std::string out;
    for (size_t i : {10U, 18U}) {
      store.ReadValue(values[i].handle, out);
      EXPECT_EQ(out, values[i].payload);
    }
    store.Write(values[10].handle, updated.data(), updated.size());
    store.Free(values[18].handle);
    reused = store.AllocAndWrite(100, replacement.data(), replacement.size());
  });
  const auto state = changes.wait_for(std::chrono::milliseconds(200));
  if (state != std::future_status::ready)
    g_release_write.store(true);
  EXPECT_EQ(state, std::future_status::ready);
  changes.get();
  if (state != std::future_status::ready)
    return;
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_exchange_pending"), 3);
  EXPECT_NE(reused, values[18].handle);
  EXPECT_EQ(static_cast<uint32_t>(reused), static_cast<uint32_t>(values[18].handle));
  g_release_write.store(true);
  ASSERT_TRUE(WaitClockIdle(store));
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_exchange_stale"), 2);
  std::string out;
  store.ReadValue(values[10].handle, out);
  EXPECT_EQ(out, updated);
  store.ReadValue(values[18].handle, out);
  EXPECT_TRUE(out.empty());
  store.ReadValue(reused, out);
  EXPECT_EQ(out, replacement);
  ASSERT_TRUE(WaitClockIdle(store));
  const auto info = store.ExtraResultFields();
  EXPECT_EQ(TieredField(info, "tiered_live_allocs"), values.size());
  EXPECT_LE(TieredField(info, "tiered_dram_bytes"),
            TieredField(info, "tiered_dram_limit_bytes"));
}

TEST_F(StaticLayoutValueTest, ClockAsyncWriteFailurePreservesLatestValueAndWorker) {
  auto config = ClockConfig(root_, "clock-async-failure", 2);
  config.json_config_["value"]["tiering"]["clock_async_exchange"] = true;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 66);
  const std::string updated(kCachedValueSize, 'z');
  store.Write(values[0].handle, updated.data(), updated.size());
  const auto before = store.ExtraResultFields();
  g_fail_next_write.store(true);
  std::string out;
  store.ReadValue(values[2].handle, out);
  EXPECT_EQ(out, values[2].payload);
  ASSERT_TRUE(WaitClockIdle(store));
  EXPECT_FALSE(g_fail_next_write.exchange(false));
  auto info = store.ExtraResultFields();
  EXPECT_GE(TieredField(info, "tiered_clock_exchange_failures"), 1);
  EXPECT_EQ(TieredField(info, "tiered_clock_promotions"), 0);
  EXPECT_EQ(TieredField(info, "tiered_clock_backed_values"),
            TieredField(before, "tiered_clock_backed_values"));
  EXPECT_EQ(TieredField(info, "tiered_dram_bytes"),
            TieredField(before, "tiered_dram_bytes"));
  store.ReadValue(values[0].handle, out);
  EXPECT_EQ(out, updated);
  store.ReadValue(values[2].handle, out); // The same worker must accept a retry.
  EXPECT_EQ(out, values[2].payload);
  ASSERT_TRUE(WaitClockIdle(store));
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  store.ReadValue(values[0].handle, out);
  EXPECT_EQ(out, updated);
  ASSERT_TRUE(WaitClockIdle(store));
}

TEST_F(StaticLayoutValueTest, ClockAsyncCostAwareAdaptsToBothHotsetDirections) {
  auto config = ClockConfig(root_, "clock-async-drift", 4, kCachedValueSize, true);
  config.json_config_["value"]["tiering"]["clock_cost_aware"] = true;
  config.json_config_["value"]["tiering"]["clock_async_exchange"] = true;
  HybridValueStore store(config);
  const auto values = FillClockValues(store, 8);
  std::string out;
  for (int repeat = 0; repeat < 16; ++repeat)
    for (size_t i = 0; i < 4; ++i)
      store.ReadValue(values[i].handle, out);
  for (size_t begin : {4U, 0U}) {
    SCOPED_TRACE(begin);
    for (int repeat = 0; repeat < 1024; ++repeat)
      for (size_t i = begin; i < begin + 4; ++i) {
        store.ReadValue(values[i].handle, out);
        EXPECT_EQ(out, values[i].payload);
      }
    ASSERT_TRUE(WaitClockIdle(store));
    const auto reads = g_read_entries.load();
    for (size_t i = begin; i < begin + 4; ++i) {
      store.ReadValue(values[i].handle, out);
      EXPECT_EQ(out, values[i].payload);
    }
    EXPECT_EQ(g_read_entries.load(), reads);
    const auto info = store.ExtraResultFields();
    EXPECT_EQ(TieredField(info, "tiered_dram_live_allocs"), 4);
    EXPECT_EQ(TieredField(info, "tiered_ssd_live_allocs"), 4);
    EXPECT_LE(TieredField(info, "tiered_dram_bytes"),
              TieredField(info, "tiered_dram_limit_bytes"));
    EXPECT_EQ(TieredField(info, "tiered_clock_failures"), 0);
  }
  EXPECT_GE(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 8);
}

TEST_F(StaticLayoutValueTest, ClockDemotionIoDoesNotBlockUnrelatedDramRead) {
  HybridValueStore store(ClockConfig(root_, "clock-nonblocking", 2));
  const std::string value(kCachedValueSize, 'q');
  const auto first = store.AllocAndWrite(0, value.data(), value.size());
  store.AllocAndWrite(1, value.data(), value.size());
  const auto hot = store.AllocAndWrite(2, value.data(), value.size());
  std::string out;
  store.ReadValue(first, out);
  store.ReadValue(hot, out);
  g_pause_next_write.store(true);
  auto promotion = std::async(std::launch::async, [&] {
    std::string data;
    store.ReadValue(hot, data);
    return data;
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!g_write_paused.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(g_write_paused.load());
  auto resident_read = std::async(std::launch::async, [&] {
    std::string data;
    store.ReadValue(first, data);
    return data;
  });
  const auto state = resident_read.wait_for(std::chrono::milliseconds(200));
  g_release_write.store(true);
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(resident_read.get(), value);
  EXPECT_EQ(promotion.get(), value);
}


TEST_F(StaticLayoutValueTest, ClockProtectsFrequentResidentsAndAdaptsToNewHotKey) {
  HybridValueStore store(ClockConfig(root_, "clock-heat", 1, kCachedValueSize, true));
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto ha = store.AllocAndWrite(0, a.data(), a.size());
  const auto hb = store.AllocAndWrite(1, b.data(), b.size());
  std::string out;
  for (int i = 0; i < 20; ++i) store.ReadValue(ha, out);
  store.ReadValue(hb, out);
  store.ReadValue(hb, out);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"), 0);
  // The new working set must earn more heat than the old one.
  for (int i = 0; i < 128; ++i) store.ReadValue(hb, out);
  EXPECT_EQ(out, b);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_demotions"), 1);
  store.ReadValue(ha, out);
  EXPECT_EQ(out, a);
}

TEST_F(StaticLayoutValueTest, ClockDefaultsToLegacyAdmission) {
  auto config = ClockConfig(root_, "clock-default-legacy", 1);
  config.json_config_["value"]["tiering"].erase("clock_frequency_admission");
  HybridValueStore store(config);
  EXPECT_EQ(TieredField(store.ExtraResultFields(),
                       "tiered_clock_frequency_admission"), 0);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_cost_aware"), 0);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_async_exchange"), 0);
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto resident = store.AllocAndWrite(0, a.data(), a.size());
  const auto candidate = store.AllocAndWrite(1, b.data(), b.size());
  std::string out;
  for (int i = 0; i < 20; ++i)
    store.ReadValue(resident, out);
  for (int i = 0; i < 2; ++i)
    store.ReadValue(candidate, out);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  // Legacy scans halve resident heat, so a new hot key can replace it before
  // accumulating the 25 credits required by frequency admission in this case.
  for (int i = 0; i < 10; ++i)
    store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
}

TEST_F(StaticLayoutValueTest, ClockFrequencyAdmissionFlagReachesEngineFactory) {
  for (const bool enabled : {false, true}) {
    auto config = ClockConfig(root_, enabled ? "clock-opt-in" : "clock-opt-out",
                              1, kCachedValueSize, enabled);
    KVEngineComposite engine(config);
    EXPECT_EQ(TieredField(engine.ExtraResultFields(),
                         "tiered_clock_frequency_admission"), enabled);
    const std::string value(kCachedValueSize, 'f');
    engine.Put(0, value, 0);
    std::string out;
    engine.Get(0, out, 0);
    EXPECT_EQ(out, value);
  }
}

TEST_F(StaticLayoutValueTest, ClockAdmissionRequiresFrequencyAdvantage) {
  HybridValueStore store(
      ClockConfig(root_, "clock-admission-margin", 1, kCachedValueSize, true));
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto resident = store.AllocAndWrite(0, a.data(), a.size());
  const auto candidate = store.AllocAndWrite(1, b.data(), b.size());
  std::string out;
  for (int i = 0; i < 20; ++i)
    store.ReadValue(resident, out);
  for (int i = 0; i < 24; ++i)
    store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  // A slight lead must not churn placement; a 25% lead can replace the old
  // resident without waiting for an aging epoch.
  store.ReadValue(candidate, out);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  const auto reads = g_read_entries.load();
  store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  EXPECT_EQ(g_read_entries.load(), reads);
}

TEST_F(StaticLayoutValueTest, ClockAgedHotResidentReturnsToExactCounting) {
  HybridValueStore store(
      ClockConfig(root_, "clock-hot-count-aging", 2, kCachedValueSize, true));
  const std::string a(kCachedValueSize, 'a'), d(kCachedValueSize, 'd'),
      b(kCachedValueSize, 'b');
  const auto resident = store.AllocAndWrite(0, a.data(), a.size());
  const auto driver = store.AllocAndWrite(1, d.data(), d.size());
  const auto candidate = store.AllocAndWrite(2, b.data(), b.size());
  std::string out;
  for (int i = 0; i < 100; ++i)
    store.ReadValue(resident, out);
  // Age the formerly hot resident through eight epochs while another value
  // stays hot. Subsequent low-frequency reads must count individually again.
  for (int i = 0; i < 8192; ++i)
    store.ReadValue(driver, out);
  for (int i = 0; i < 20; ++i)
    store.ReadValue(resident, out);
  EXPECT_EQ(out, a);
  for (int i = 0; i < 24; ++i)
    store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  store.ReadValue(candidate, out);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  const auto reads = g_read_entries.load();
  store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  store.ReadValue(driver, out);
  EXPECT_EQ(out, d);
  EXPECT_EQ(g_read_entries.load(), reads);
}

TEST_F(StaticLayoutValueTest, ClockRejectedAdmissionsDoNotEraseResidentHeat) {
  HybridValueStore store(
      ClockConfig(root_, "clock-scan-heat", 1, kCachedValueSize, true));
  const std::string value(kCachedValueSize, 'h');
  const auto hot = store.AllocAndWrite(0, value.data(), value.size());
  std::vector<uint64_t> cold;
  for (uint64_t key = 1; key <= 64; ++key)
    cold.push_back(store.AllocAndWrite(key, value.data(), value.size()));
  std::string out;
  for (int i = 0; i < 128; ++i)
    store.ReadValue(hot, out);
  for (const auto handle : cold) {
    store.ReadValue(handle, out);
    store.ReadValue(handle, out);
    ASSERT_EQ(out, value);
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  const auto reads = g_read_entries.load();
  store.ReadValue(hot, out);
  EXPECT_EQ(out, value);
  EXPECT_EQ(g_read_entries.load(), reads);
}

TEST_F(StaticLayoutValueTest, ClockSeparatedMissesCanReplaceUnusedResident) {
  HybridValueStore store(
      ClockConfig(root_, "clock-separated-misses", 2, kCachedValueSize, true));
  const std::string a(kCachedValueSize, 'a'), d(kCachedValueSize, 'd'),
      b(kCachedValueSize, 'b');
  const auto unused = store.AllocAndWrite(0, a.data(), a.size());
  const auto driver = store.AllocAndWrite(1, d.data(), d.size());
  const auto candidate = store.AllocAndWrite(2, b.data(), b.size());
  std::string out;
  store.ReadValue(candidate, out);
  ASSERT_EQ(out, b);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  // The first miss's heat expires, but the admission miss count survives.
  // A second separated miss must still displace a never-read resident.
  for (int i = 0; i < 8192; ++i)
    store.ReadValue(driver, out);
  store.ReadValue(candidate, out);
  ASSERT_EQ(out, b);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
  const auto reads = g_read_entries.load();
  store.ReadValue(candidate, out);
  EXPECT_EQ(out, b);
  store.ReadValue(driver, out);
  EXPECT_EQ(out, d);
  EXPECT_EQ(g_read_entries.load(), reads);
  store.ReadValue(unused, out);
  EXPECT_EQ(out, a);
  EXPECT_GT(g_read_entries.load(), reads);
}

TEST_F(StaticLayoutValueTest, ClockSsdHistoryAgesWithResidentReads) {
  HybridValueStore store(
      ClockConfig(root_, "clock-ssd-aging", 2, kCachedValueSize, true));
  const std::string value(kCachedValueSize, 'h');
  const auto first = store.AllocAndWrite(0, value.data(), value.size());
  const auto second = store.AllocAndWrite(1, value.data(), value.size());
  const auto old_ssd = store.AllocAndWrite(2, value.data(), value.size());
  std::string out;
  for (int i = 0; i < 4096; ++i) {
    store.ReadValue(first, out);
    store.ReadValue(second, out);
  }
  for (int i = 0; i < 64; ++i)
    store.ReadValue(old_ssd, out);
  ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  // Both tiers age while another value is read. Refresh the second resident
  // before revisiting the old SSD candidate so only recent demand competes.
  for (int i = 0; i < 8192; ++i)
    store.ReadValue(first, out);
  for (int i = 0; i < 64; ++i)
    store.ReadValue(second, out);
  store.ReadValue(old_ssd, out);
  store.ReadValue(old_ssd, out);
  EXPECT_EQ(out, value);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
}

TEST_F(StaticLayoutValueTest, ClockConcurrentBatchesSurviveWritesAndSlotReuse) {
  auto config = ClockConfig(root_, "clock-batch-race", 8, kCachedValueSize, true);
  config.json_config_["value"]["tiering"]["cache_shards"] = 4;
  HybridValueStore store(config);
  const std::string initial(512, 'a');
  std::vector<uint64_t> handles;
  for (uint64_t key = 0; key < 16; ++key)
    handles.push_back(store.AllocAndWrite(key, initial.data(), initial.size()));
  std::atomic<uint64_t> current{handles[0]};
  std::atomic<bool> start{false}, failed{false};
  std::vector<std::thread> threads;
  threads.emplace_back([&] {
    while (!start.load()) std::this_thread::yield();
    for (int i = 0; i < 300; ++i) {
      const std::string value(i % 2 ? 128 : 1024, i % 2 ? 'b' : 'c');
      store.Free(current.load());
      current.store(store.AllocAndWrite(0, value.data(), value.size()));
      store.AllocAndWrite(1, value.data(), value.size());
    }
  });
  for (int tid = 0; tid < 3; ++tid) {
    threads.emplace_back([&] {
      while (!start.load()) std::this_thread::yield();
      for (int i = 0; i < 300; ++i) {
        auto batch = handles;
        batch.push_back(current.load());
        std::vector<ValueStore::ReadResult> results;
        store.BatchRead(batch, results);
        for (const auto& result : results) {
          const auto& data = result.data;
          if (data.empty()) continue;
          if ((data.size() != 128 && data.size() != 512 && data.size() != 1024) ||
              !std::all_of(data.begin(), data.end(), [&](char c) { return c == data[0]; }))
            failed.store(true);
        }
      }
    });
  }
  start.store(true);
  for (auto& thread : threads) thread.join();
  EXPECT_FALSE(failed.load());
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_live_allocs"), 16);
}

TEST_F(StaticLayoutValueTest, ClockFlatReadsMixTiersAndRejectStaleOrWrongSize) {
  HybridValueStore store(ClockConfig(root_, "clock-flat", 8));
  std::vector<uint64_t> handles;
  std::vector<std::string> values;
  for (uint64_t key = 0; key < 32; ++key) {
    values.emplace_back(512, 'A' + key);
    handles.push_back(store.AllocAndWrite(key, values.back().data(), 512));
  }
  handles.push_back(handles[0]);
  values.push_back(values[0]);
  handles.push_back(kValueHandleNone);
  values.emplace_back(512, '\0');
  std::vector<char> output(handles.size() * 512);
  uint64_t missing = 0;
  for (int repeat = 0; repeat < 5; ++repeat) {
    ASSERT_TRUE(store.ReadFlatFixedRows(handles.data(), handles.size(), output.data(), 512, &missing));
    EXPECT_EQ(missing, 1);
    for (size_t i = 0; i < handles.size(); ++i)
      EXPECT_EQ(std::string(output.data() + i * 512, 512), values[i]);
  }
  EXPECT_FALSE(store.ReadFlatFixedRows(handles.data(), handles.size(), output.data(), 128, &missing));
  store.Free(handles[0]);
  store.AllocAndWrite(0, values[0].data(), 512);
  EXPECT_FALSE(store.ReadFlatFixedRows(handles.data(), handles.size(), output.data(), 512, &missing));
}

TEST_F(StaticLayoutValueTest, ClockBatchSsdWaitDoesNotBlockResidentUpdate) {
  HybridValueStore store(ClockConfig(root_, "clock-batch-resident", 1));
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto resident = store.AllocAndWrite(0, a.data(), a.size());
  const auto ssd      = store.AllocAndWrite(1, a.data(), a.size());
  g_pause_next_read.store(true);
  auto read = std::async(std::launch::async, [&] {
    std::vector<ValueStore::ReadResult> results;
    store.BatchRead({resident, ssd}, results);
    return results;
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!g_read_paused.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(g_read_paused.load());
  auto write       = std::async(std::launch::async, [&] {
    store.Write(resident, b.data(), b.size());
  });
  const auto state = write.wait_for(std::chrono::milliseconds(200));
  g_release_read.store(true);
  EXPECT_EQ(state, std::future_status::ready);
  write.get();
  const auto results = read.get();
  ASSERT_EQ(results.size(), 2);
  for (const auto& result : results)
    EXPECT_EQ(std::string(result.data.begin(), result.data.end()), a);
  std::string current;
  store.ReadValue(resident, current);
  EXPECT_EQ(current, b);
}

TEST_F(StaticLayoutValueTest, ClockBatchCoalescesDemandWriteback) {
  auto config = ClockConfig(root_, "clock-batch-writeback", 32);
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  std::vector<uint64_t> handles;
  for (int i = 0; i < 64; ++i) {
    const std::string value(kCachedValueSize, 'A' + i);
    handles.push_back(store.AllocAndWrite(i, value.data(), value.size()));
  }
  const std::vector<uint64_t> hot(handles.begin() + 32, handles.end());
  std::vector<ValueStore::ReadResult> results;
  const auto before = g_read_entries.load();
  store.BatchRead(hot, results);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"),
            32);
  // One grouped miss read plus one grouped read-modify-write of packed slots.
  EXPECT_LE(g_read_entries.load() - before, 4);
  ASSERT_EQ(results.size(), hot.size());
  for (size_t i = 0; i < results.size(); ++i)
    EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
              std::string(kCachedValueSize, 'A' + i + 32));
  for (size_t i = 0; i < handles.size(); ++i) {
    std::string value;
    store.ReadValue(handles[i], value);
    EXPECT_EQ(value, std::string(kCachedValueSize, 'A' + i));
  }
}

TEST_F(StaticLayoutValueTest, ClockBatchReadersShareSsdValues) {
  HybridValueStore store(ClockConfig(root_, "clock-shared-read", 0));
  const std::string value(kCachedValueSize, 's');
  const auto handle = store.AllocAndWrite(0, value.data(), value.size());
  auto read         = [&] {
    std::vector<ValueStore::ReadResult> results;
    store.BatchRead({handle}, results);
    return std::string(results[0].data.begin(), results[0].data.end());
  };
  g_pause_next_read.store(true);
  auto first = std::async(std::launch::async, read);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!g_read_paused.load() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(g_read_paused.load());
  auto second      = std::async(std::launch::async, read);
  const auto state = second.wait_for(std::chrono::milliseconds(200));
  g_release_read.store(true);
  EXPECT_EQ(state, std::future_status::ready);
  EXPECT_EQ(first.get(), value);
  EXPECT_EQ(second.get(), value);
}

TEST_F(StaticLayoutValueTest, ClockBatchDemandSurvivesPromotionLockContention) {
  for (bool frequency : {false, true}) {
    SCOPED_TRACE(frequency);
    auto config = ClockConfig(root_, frequency ? "clock-demand-frequency"
                                               : "clock-demand-legacy",
                              1, kCachedValueSize, frequency);
    config.json_config_["value"]["tiering"]["promote_min_misses"] = 3;
    HybridValueStore store(config);
    const auto values = FillClockValues(store, 2);
    ASSERT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_entry_bytes"),
              160);
    g_read_paused.store(false);
    g_release_read.store(false);
    g_pause_next_read.store(true);
    auto first = std::async(std::launch::async, [&] {
      std::vector<ValueStore::ReadResult> results;
      store.BatchRead({values[1].handle}, results);
      return results;
    });
    struct ReleaseReadOnExit {
      ~ReleaseReadOnExit() { g_release_read.store(true); }
    } release;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!g_read_paused.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    ASSERT_TRUE(g_read_paused.load());

    // The first reader keeps its shared value lock. The second can finish
    // its small SSD read, but cannot acquire an exclusive promotion lock.
    auto second = std::async(std::launch::async, [&] {
      std::vector<ValueStore::ReadResult> results;
      store.BatchRead({values[1].handle, values[1].handle}, results);
      return results;
    });
    const auto state = second.wait_for(std::chrono::seconds(3));
    if (state != std::future_status::ready)
      g_release_read.store(true);
    const auto completed = second.get();
    EXPECT_EQ(state, std::future_status::ready);
    ASSERT_EQ(completed.size(), 2);
    for (const auto& result : completed)
      EXPECT_EQ(std::string(result.data.begin(), result.data.end()),
                values[1].payload);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_misses"), 3);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);

    g_release_read.store(true);
    const auto resumed = first.get();
    ASSERT_EQ(resumed.size(), 1);
    EXPECT_EQ(std::string(resumed[0].data.begin(), resumed[0].data.end()),
              values[1].payload);
    // All three successful demands must count, including the two whose
    // promotion locks were busy. The last reader can now admit the hot key.
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
    const auto reads = g_read_entries.load();
    std::string value;
    store.ReadValue(values[1].handle, value);
    EXPECT_EQ(value, values[1].payload);
    EXPECT_EQ(g_read_entries.load(), reads);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 0);
  }
}

TEST_F(StaticLayoutValueTest, ClockFailedBatchReadDoesNotEarnAdmissionCredit) {
  for (bool frequency : {false, true}) {
    SCOPED_TRACE(frequency);
    HybridValueStore store(ClockConfig(
        root_, frequency ? "clock-failed-demand-frequency"
                         : "clock-failed-demand-legacy",
        1, kCachedValueSize, frequency));
    const auto values = FillClockValues(store, 2);
    std::vector<ValueStore::ReadResult> results;
    g_fail_next_read.store(true);
    EXPECT_THROW(store.BatchRead({values[1].handle}, results), std::runtime_error);
    for (int successful_reads = 1; successful_reads <= 2; ++successful_reads) {
      store.BatchRead({values[1].handle}, results);
      ASSERT_EQ(results.size(), 1);
      EXPECT_EQ(std::string(results[0].data.begin(), results[0].data.end()),
                values[1].payload);
      EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
                successful_reads == 2 ? 1 : 0);
    }
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_misses"), 3);
    EXPECT_EQ(TieredField(store.ExtraResultFields(),
                         "tiered_clock_admission_rejects"), 1);
  }
}

TEST_F(StaticLayoutValueTest, ClockBatchDemandResetsOnWriteAndSlotReuse) {
  for (bool frequency : {false, true}) {
    SCOPED_TRACE(frequency);
    auto config = ClockConfig(root_, frequency ? "clock-demand-reset-frequency"
                                               : "clock-demand-reset-legacy",
                              1, kCachedValueSize, frequency);
    config.json_config_["value"]["tiering"]["promote_min_misses"] = 3;
    HybridValueStore store(config);
    const auto values = FillClockValues(store, 2);
    std::vector<ValueStore::ReadResult> results;
    store.BatchRead({values[1].handle, values[1].handle}, results);
    ASSERT_EQ(results.size(), 2);
    for (const auto& result : results)
      EXPECT_EQ(std::string(result.data.begin(), result.data.end()),
                values[1].payload);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);

    const std::string updated(kCachedValueSize, 'w');
    store.Write(values[1].handle, updated.data(), updated.size());
    store.BatchRead({values[1].handle}, results);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(std::string(results[0].data.begin(), results[0].data.end()), updated);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);

    store.Free(values[1].handle);
    const std::string replacement(kCachedValueSize, 'r');
    const auto reused = store.AllocAndWrite(2, replacement.data(),
                                           replacement.size());
    ASSERT_NE(reused, values[1].handle);
    // The stale logical handle earns no credit, and each valid duplicate
    // earns exactly one. Neither the write nor slot reuse retains old credit.
    store.BatchRead({values[1].handle, reused, reused}, results);
    ASSERT_EQ(results.size(), 3);
    EXPECT_TRUE(results[0].data.empty());
    for (size_t i = 1; i < results.size(); ++i)
      EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
                replacement);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
    store.BatchRead({reused}, results);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(std::string(results[0].data.begin(), results[0].data.end()),
              replacement);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 1);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 0);
  }
}

TEST_F(StaticLayoutValueTest, SsdBatchWriteReleasesBuffersOnIoFailure) {
  SsdValueStore store(TieredStoreConfig(root_, "batch-write-failure", {{"cache_policy", "none"}}));
  const std::string value(512, 'a');
  std::vector<ValueStore::WriteSpec> specs(40, {value.data(), value.size()});
  const auto handles = store.BatchAllocAndWrite(specs);
  for (bool read_failure : {true, false}) {
    const auto allocated = g_buffer_allocations.load();
    const auto freed = g_buffer_frees.load();
    g_fail_next_read.store(read_failure);
    g_fail_next_write.store(!read_failure);
    EXPECT_THROW(store.BatchWrite({handles.front(), handles.back()}, {specs.front(), specs.back()}), std::runtime_error);
    EXPECT_EQ(g_buffer_allocations.load() - allocated, 2);
    EXPECT_EQ(g_buffer_frees.load() - freed, 2);
  }
}

TEST_F(StaticLayoutValueTest, SsdLargeBatchesShareOnlyInflightReadsAndErrors) {
  for (bool fail : {false, true}) {
    SsdValueStore store(TieredStoreConfig(
        root_,
        fail ? "join-failure" : "join-success",
        {{"cache_policy", "none"}}));
    std::vector<std::string> values;
    std::vector<ValueStore::WriteSpec> specs;
    values.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
      values.emplace_back(kCachedValueSize, 'A' + i % 26);
      specs.push_back({values.back().data(), values.back().size()});
    }
    const auto handles = store.BatchAllocAndWrite(specs);
    auto read          = [&](size_t start, size_t count) {
      std::vector<ValueStore::ReadResult> results;
      store.BatchRead({handles.begin() + start, handles.begin() + start + count}, results);
      return results;
    };
    auto verify = [&](const std::vector<ValueStore::ReadResult>& results, size_t start, size_t count) {
      ASSERT_EQ(results.size(), count);
      for (size_t i = 0; i < results.size(); ++i)
        EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
                  values[start + i]);
    };
    g_read_paused.store(false);
    g_release_read.store(false);
    g_pause_next_read.store(true);
    g_fail_next_read.store(fail);
    const auto before = g_read_entries.load();
    auto first        = std::async(std::launch::async, read, 0, 7500);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!g_read_paused.load() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_TRUE(g_read_paused.load());
    auto second = std::async(std::launch::async, read, 2500, 7500);
    deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    uint64_t joined = 0;
    while ((joined = TieredField(
                store.ExtraResultFields(), "ssd_joined_read_pages")) == 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    g_release_read.store(true);
    EXPECT_GE(joined, 512);
    if (fail) {
      EXPECT_THROW(first.get(), std::runtime_error);
      EXPECT_THROW(second.get(), std::runtime_error);
    } else {
      verify(first.get(), 0, 7500);
      verify(second.get(), 2500, 7500);
    }
    EXPECT_EQ(g_read_entries.load() - before, 2);
    EXPECT_EQ(TieredField(store.ExtraResultFields(), "ssd_inflight_read_pages"),
              0);
    const auto completed = g_read_entries.load();
    verify(read(0, handles.size()), 0, handles.size());
    EXPECT_EQ(g_read_entries.load() - completed, 1);
  }
}

TEST_F(StaticLayoutValueTest, ClockFailedBatchWritebackKeepsBothTiersIntact) {
  auto config = ClockConfig(root_, "clock-batch-write-failure", 32);
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  std::vector<uint64_t> handles;
  for (int i = 0; i < 64; ++i) {
    const std::string value(kCachedValueSize, 'A' + i);
    handles.push_back(store.AllocAndWrite(i, value.data(), value.size()));
  }
  const std::vector<uint64_t> hot(handles.begin() + 32, handles.end());
  std::vector<ValueStore::ReadResult> results;
  g_fail_next_write.store(true);
  EXPECT_NO_THROW(store.BatchRead(hot, results));
  EXPECT_FALSE(g_fail_next_write.load());
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
            0);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"),
            32);
  ASSERT_EQ(results.size(), hot.size());
  for (size_t i = 0; i < results.size(); ++i)
    EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
              std::string(kCachedValueSize, 'A' + i + 32));
  for (size_t i = 0; i < handles.size(); ++i) {
    std::string value;
    store.ReadValue(handles[i], value);
    EXPECT_EQ(value, std::string(kCachedValueSize, 'A' + i));
  }
  for (auto handle : handles)
    store.Free(handle);
  EXPECT_EQ(
      TieredField(store.ExtraResultFields(), "tiered_clock_backed_values"), 0);
}

TEST_F(StaticLayoutValueTest, ClockCanReleasePromotedBackingCopies) {
  auto config = ClockConfig(root_, "clock-exclusive", 4, kCachedValueSize, true);
  config.json_config_["value"]["tiering"]["retain_backing_copies"] = false;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  std::vector<uint64_t> handles;
  std::vector<std::string> values;
  for (uint64_t key = 0; key < 24; ++key) {
    values.emplace_back(kCachedValueSize, 'A' + key);
    handles.push_back(store.AllocAndWrite(key, values.back().data(), values.back().size()));
  }
  for (int repeat = 0; repeat < 8; ++repeat) {
    // Create an actual hot candidate instead of requiring equal-frequency churn.
    for (int access = 0; access < 4; ++access) {
      std::string value;
      store.ReadValue(handles[repeat + 4], value);
      EXPECT_EQ(value, values[repeat + 4]);
    }
    std::vector<ValueStore::ReadResult> results;
    store.BatchRead(handles, results);
    ASSERT_EQ(results.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i)
      EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()), values[i]);
    const auto info = store.ExtraResultFields();
    EXPECT_EQ(TieredField(info, "tiered_clock_backed_values"), 20);
    EXPECT_EQ(TieredField(info, "tiered_clock_clean_evictions"), 0);
    EXPECT_EQ(TieredField(info, "tiered_clock_failures"), 0);
    EXPECT_LE(TieredField(info, "tiered_dram_bytes"), TieredField(info, "tiered_dram_limit_bytes"));
    values[repeat] = std::string(kCachedValueSize, 'z');
    store.Write(handles[repeat], values[repeat].data(), values[repeat].size());
  }
  EXPECT_GT(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    std::string value;
    store.ReadValue(handles[i], value);
    EXPECT_EQ(value, values[i]);
    store.Free(handles[i]);
  }
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_live_allocs"), 0);
}

TEST_F(StaticLayoutValueTest, ClockReleasedBackingSurvivesFailedEviction) {
  auto config = ClockConfig(root_, "clock-exclusive-failure", 1, kCachedValueSize, true);
  config.json_config_["value"]["tiering"]["retain_backing_copies"] = false;
  config.json_config_["value"]["tiering"]["promote_min_misses"] = 1;
  HybridValueStore store(config);
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto ha = store.AllocAndWrite(a.data(), a.size());
  const auto hb = store.AllocAndWrite(b.data(), b.size());
  std::string value;
  store.ReadValue(hb, value);
  ASSERT_EQ(value, b);
  g_fail_next_write.store(true);
  for (int retry = 0; retry < 32 && g_fail_next_write.load(); ++retry)
    store.ReadValue(ha, value);
  EXPECT_EQ(value, a);
  EXPECT_FALSE(g_fail_next_write.load());
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 1);
  store.ReadValue(hb, value);
  EXPECT_EQ(value, b);
  for (int retry = 0; retry < 16; ++retry)
    store.ReadValue(ha, value);
  EXPECT_EQ(value, a);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"), 2);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_backed_values"), 1);
}

TEST_F(StaticLayoutValueTest,
       ClockCleanEvictionsSkipWritesAndDirtyValuesWriteBack) {
  HybridValueStore store(ClockConfig(root_, "clock-clean", 1));
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b');
  const auto ha = store.AllocAndWrite(0, a.data(), a.size());
  const auto hb = store.AllocAndWrite(1, b.data(), b.size());
  std::string out;
  auto promote = [&](uint64_t handle) {
    const auto before =
        TieredField(store.ExtraResultFields(), "tiered_clock_promotions");
    for (int i = 0; i < 32 && TieredField(store.ExtraResultFields(),
                                          "tiered_clock_promotions") == before;
         ++i)
      store.ReadValue(handle, out);
    EXPECT_GT(TieredField(store.ExtraResultFields(), "tiered_clock_promotions"),
              before);
  };
  promote(hb); // The initial DRAM value needs its first SSD backing copy.
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"),
            1);
  g_fail_next_write.store(true);
  promote(ha); // B already has an unchanged SSD copy: no write is issued.
  EXPECT_EQ(out, a);
  EXPECT_TRUE(g_fail_next_write.exchange(false));
  EXPECT_EQ(
      TieredField(store.ExtraResultFields(), "tiered_clock_clean_evictions"),
      1);
  const std::string updated(kCachedValueSize, 'z');
  store.Write(ha, updated.data(), updated.size());
  promote(hb);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_writebacks"),
            2);
  store.ReadValue(ha, out);
  EXPECT_EQ(out, updated);
  store.Free(ha);
  store.Free(hb);
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_live_allocs"), 0);
}

TEST_F(StaticLayoutValueTest,
       ClockFastReadersSurviveInPlaceWritesAndMetadataReuse) {
  HybridValueStore store(ClockConfig(root_, "clock-hazards", 1));
  const std::string initial(512, 'a');
  std::atomic<uint64_t> current{
      store.AllocAndWrite(0, initial.data(), initial.size())};
  std::atomic<bool> start{false}, done{false}, failed{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 8; ++i)
    readers.emplace_back([&, i] {
      std::string out;
      std::array<uint64_t, 128> handles;
      std::array<char, 128 * 512> flat;
      while (!start.load())
        std::this_thread::yield();
      while (!done.load()) {
        if (i & 1) {
          handles.fill(current.load());
          if (store.ReadFlatFixedRows(handles.data(), handles.size(), flat.data(), 512, nullptr))
            for (size_t row = 0; row < handles.size(); ++row)
              if (!std::all_of(flat.begin() + row * 512, flat.begin() + (row + 1) * 512,
                               [&](char c) { return c == flat[row * 512]; }))
                failed.store(true);
          continue;
        }
        store.ReadValue(current.load(), out);
        if (!out.empty() &&
            (out.size() != 512 ||
             out.find_first_not_of(out[0]) != std::string::npos))
          failed.store(true);
      }
    });
  start.store(true);
  for (int i = 0; i < 1000; ++i) {
    const std::string value(512, i % 2 ? 'b' : 'c');
    if (i % 10 == 0) {
      store.Free(current.load());
      current.store(store.AllocAndWrite(0, value.data(), value.size()));
    } else
      store.Write(current.load(), value.data(), value.size());
  }
  done.store(true);
  for (auto& thread : readers)
    thread.join();
  EXPECT_FALSE(failed.load());
}

TEST_F(StaticLayoutValueTest, ClockFastReadersSeeWholeVariableSizeValues) {
  HybridValueStore store(ClockConfig(root_, "clock-short-snapshots", 1));
  const std::string initial(512, 'a');
  const auto handle = store.AllocAndWrite(0, initial.data(), initial.size());
  const std::array<size_t, 9> sizes{1, 7, 8, 9, 17, 31, 127, 511, 512};
  std::atomic<bool> start{false}, done{false}, failed{false};
  std::vector<std::thread> readers;
  for (int i = 0; i < 8; ++i)
    readers.emplace_back([&] {
      std::string out;
      while (!start.load())
        std::this_thread::yield();
      while (!done.load()) {
        store.ReadValue(handle, out);
        if (std::find(sizes.begin(), sizes.end(), out.size()) == sizes.end() ||
            out.find_first_not_of(out[0]) != std::string::npos)
          failed.store(true);
      }
    });
  start.store(true);
  for (int i = 0; i < 5000; ++i) {
    const std::string value(sizes[i % sizes.size()], 'b' + i % 10);
    store.Write(handle, value.data(), value.size());
  }
  done.store(true);
  for (auto& thread : readers)
    thread.join();
  EXPECT_FALSE(failed.load());
  EXPECT_LE(TieredField(store.ExtraResultFields(), "tiered_dram_bytes"),
            TieredField(store.ExtraResultFields(), "tiered_dram_limit_bytes"));
}

TEST_F(StaticLayoutValueTest, AdaptiveSizesSmallCacheToKnownOverflow) {
  auto config = TieredStoreConfig(
      root_,
      "adaptive-small",
      {{"cache_policy", "adaptive"},
       {"cache_budget_ratio", 0.0},
       {"cache_shards", 1},
       {"expected_dram_bytes",
        static_cast<int64_t>(kCacheDramBytes * 0.85) + kCachedValueSize}});
  HybridValueStore store(config);
  const std::string value(kCachedValueSize, 's');
  const auto handle = store.AllocAndWrite(0, value.data(), value.size());
  EXPECT_NE(store.DirectPtr(handle), nullptr);
  const auto info = store.ExtraResultFields();
  EXPECT_NE(info.find("tiered_cache_policy=lru"), std::string::npos);
  EXPECT_GT(TieredField(info, "tiered_cache_budget_bytes"), 0);
  EXPECT_LT(TieredField(info, "tiered_cache_budget_bytes"),
            kCacheDramBytes / 100);
}

TEST_F(StaticLayoutValueTest, AdaptiveFlatReadsMixDramCachedSsdAndMissingRows) {
  auto config = TieredStoreConfig(
      root_, "adaptive-flat",
      {{"cache_policy", "adaptive"}, {"cache_shards", 1},
       {"expected_dram_bytes", static_cast<int64_t>(kCacheDramBytes * 0.85) + kCachedValueSize}});
  HybridValueStore store(config);
  const std::string resident(kCachedValueSize, 'r');
  const auto dram = store.AllocAndWrite(resident.data(), resident.size());
  const auto ssd = FillUntilSsd(store, 2);
  ASSERT_EQ(ssd.size(), 2);
  const std::array<uint64_t, 5> handles{dram, ssd[0].handle, kValueHandleNone, ssd[1].handle, ssd[0].handle};
  std::vector<std::string> expected{resident, ssd[0].payload, std::string(kCachedValueSize, '\0'), ssd[1].payload, ssd[0].payload};
  std::array<char, 5 * kCachedValueSize> output;
  auto verify = [&] {
    uint64_t missing = 0;
    ASSERT_TRUE(store.ReadFlatFixedRows(handles.data(), handles.size(), output.data(), kCachedValueSize, &missing));
    EXPECT_EQ(missing, 1);
    for (size_t i = 0; i < handles.size(); ++i)
      EXPECT_EQ(std::string(output.data() + i * kCachedValueSize, kCachedValueSize), expected[i]);
  };
  for (int repeat = 0; repeat < 3; ++repeat)
    verify();
  EXPECT_GT(TieredField(store.ExtraResultFields(), "tiered_cache_hits"), 0);
  expected[1] = expected[4] = std::string(kCachedValueSize, 'z');
  store.Write(ssd[0].handle, expected[1].data(), expected[1].size());
  verify();
  const std::string short_value(128, 's');
  store.Write(ssd[0].handle, short_value.data(), short_value.size());
  EXPECT_FALSE(store.ReadFlatFixedRows(handles.data(), handles.size(), output.data(), kCachedValueSize, nullptr));
}

TEST_F(StaticLayoutValueTest, AdaptiveUsesClockForLargeOrUnknownWorkingSets) {
  for (int64_t expected : {int64_t(0), int64_t(kCacheDramBytes * 2)}) {
    auto config = TieredStoreConfig(
        root_,
        "adaptive-" + std::to_string(expected),
        {{"cache_policy", "adaptive"}, {"expected_dram_bytes", expected}});
    HybridValueStore store(config);
    EXPECT_NE(store.ExtraResultFields().find("tiered_cache_policy=clock"),
              std::string::npos);
  }
  auto config = TieredStoreConfig(
      root_,
      "adaptive-fits",
      {{"cache_policy", "adaptive"},
       {"expected_dram_bytes", kCachedValueSize}});
  HybridValueStore store(config);
  EXPECT_NE(store.ExtraResultFields().find("tiered_cache_policy=none"),
            std::string::npos);
}

TEST_F(StaticLayoutValueTest,
       ClockFailedDirtyWritebackKeepsNewestResidentValue) {
  HybridValueStore store(ClockConfig(root_, "clock-dirty-failure", 1, kCachedValueSize, true));
  const std::string a(kCachedValueSize, 'a'), b(kCachedValueSize, 'b'),
      updated(kCachedValueSize, 'u');
  const auto ha = store.AllocAndWrite(0, a.data(), a.size());
  const auto hb = store.AllocAndWrite(1, b.data(), b.size());
  std::string out;
  store.ReadValue(hb, out);
  store.ReadValue(hb, out);
  store.Write(hb, updated.data(), updated.size());
  store.ReadValue(ha, out);
  g_fail_next_write.store(true);
  for (int retry = 0; retry < 32 && g_fail_next_write.load(); ++retry)
    store.ReadValue(ha, out);
  EXPECT_EQ(out, a);
  EXPECT_FALSE(g_fail_next_write.load());
  EXPECT_EQ(TieredField(store.ExtraResultFields(), "tiered_clock_failures"), 1);
  store.ReadValue(hb, out);
  EXPECT_EQ(out, updated);
  for (int i = 0; i < 16; ++i)
    store.ReadValue(ha, out);
  store.ReadValue(hb, out);
  EXPECT_EQ(out, updated);
  store.Free(ha);
  store.Free(hb);
  EXPECT_EQ(
      TieredField(store.ExtraResultFields(), "tiered_clock_backed_values"), 0);
}
