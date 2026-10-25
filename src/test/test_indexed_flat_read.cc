// Standalone contract checks for fused indexed reads and allocator isolation.
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/kv_engine/engine_composite.h"

static void Check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

static BaseKVConfig Config(const std::string& name) {
  BaseKVConfig c;
  c.num_threads_ = 1;
  c.json_config_ = {
      {"capacity", 1024},
      {"index", {{"type", "DRAM_PET_HASH"}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", name},
        {"default_value_size_hint", 72},
        {"dram_allocator",
         {{"type", "CONCURRENT_SLAB_MEMORY_POOL"},
          {"capacity_bytes", 64ULL << 20}}}}}};
  return c;
}

int main(int argc, char** argv) {
  Check(argc == 2, "pass a unique output path prefix");
  const std::string root = argv[1];
  KVEngineComposite engine(Config(root + "_engine"));
  std::vector<uint64_t> keys{17, 91, 17};
  std::array<int64_t, 3> indices{};
  std::array<float, 48> values{};
  std::array<bool, 3> found{};
  BaseKV::BatchGetFlatStats stats;
  auto get = [&] {
    return engine.BatchGetIndexedFlat(
        base::ConstArray<uint64_t>(keys),
        indices.data(),
        values.data(),
        found.data(),
        3,
        16,
        0,
        &stats);
  };
  Check(get(), "all-miss read failed");
  Check(stats.missing_rows == 3 && indices[0] == -1 && !found[0],
        "miss semantics");
  std::array<float, 18> row{};
  const int64_t stable = 1234567890123LL;
  std::memcpy(row.data(), &stable, sizeof(stable));
  for (int i = 0; i < 16; ++i)
    row[i + 2] = static_cast<float>(i + 9);
  engine.Put(
      17,
      std::string(reinterpret_cast<const char*>(row.data()), sizeof(row)),
      0);
  Check(get(), "mixed read failed");
  Check(stats.missing_rows == 1 && indices[0] == stable && indices[2] == stable,
        "stable index or duplicate key changed");
  Check(found[0] && !found[1] && found[2], "found mask changed");
  for (int i = 0; i < 16; ++i) {
    Check(values[i] == i + 9 && values[16 + i] == 0 && values[32 + i] == i + 9,
          "value or missing zero fill changed");
  }
  const base::ConstArray<uint64_t> empty(nullptr, 0);
  Check(engine.BatchGetIndexedFlat(
            empty, nullptr, nullptr, nullptr, 0, 16, 0, &stats),
        "empty batch rejected");
  Check(stats.missing_rows == 0, "stale statistics after empty batch");
  Check(!engine.BatchGetIndexedFlat(
            base::ConstArray<uint64_t>(keys),
            indices.data(),
            values.data(),
            found.data(),
            3,
            15,
            0),
        "wrong row width accepted");
  Check(!engine.BatchGetIndexedFlat(
            base::ConstArray<uint64_t>(keys),
            nullptr,
            values.data(),
            found.data(),
            3,
            16,
            0),
        "null index output accepted");
  Check(!engine.BatchGetIndexedFlat(
            empty,
            nullptr,
            nullptr,
            nullptr,
            0,
            (std::numeric_limits<int64_t>::max)(),
            0),
        "overflow dimension accepted");

  // A size hint is not a proof that every stored row has that size. The fused
  // reader must reject a short allocation, just like the descriptor path.
  engine.Put(91, std::string(4, 'x'), 0);
  Check(!get(), "undersized row was read out of bounds");

  // Simultaneous pools of the same class must never share their TLS hot chunk.
  DramValueStore first(Config(root + "_first"));
  DramValueStore second(Config(root + "_second"));
  const auto h1 = first.AllocAndWrite(row.data(), sizeof(row));
  row[2]        = -99;
  const auto h2 = second.AllocAndWrite(row.data(), sizeof(row));
  int64_t one_index;
  bool one_found;
  std::array<float, 16> one_value;
  uint64_t misses;
  Check(first.ReadIndexedFixedRows(
            &h1, 1, &one_index, one_value.data(), &one_found, 64, &misses) &&
            one_value[0] == 9,
        "first allocator corrupted");
  Check(second.ReadIndexedFixedRows(
            &h2, 1, &one_index, one_value.data(), &one_found, 64, &misses) &&
            one_value[0] == -99,
        "second allocator corrupted");
  first.Retire(h1);
  second.Retire(h2);
  std::cout
      << "indexed flat contracts and independent allocator ownership: PASS\n";
}
