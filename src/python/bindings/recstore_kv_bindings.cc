#include "recstore_kv_bindings.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "base/factory.h"
#include "base/json.h"
#include "memory/shm_file.h"
#include "storage/kv_engine/engine_selector.h"

namespace recstore::python {

std::vector<uint64_t>
CheckIds(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
             ids) {
  auto info = ids.unchecked<1>();
  std::vector<uint64_t> keys(info.shape(0));
  for (py::ssize_t i = 0; i < info.shape(0); ++i) {
    keys[i] = static_cast<uint64_t>(info(i));
  }
  return keys;
}

base::ConstArray<uint64_t> Int64KeyView::AsConstArray() const {
  if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("ID batch exceeds RecStore array limit");
  }
  return base::ConstArray<uint64_t>(values, static_cast<int>(count));
}

Int64KeyView CheckIdsView(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        ids) {
  auto info = ids.unchecked<1>();
  // IDs are opaque 64-bit keys. Reusing the contiguous int64 representation
  // preserves the exact modulo-2^64 mapping used by CheckIds.
  return Int64KeyView{reinterpret_cast<const uint64_t*>(info.data(0)),
                      static_cast<size_t>(info.shape(0))};
}

std::unique_ptr<BaseKV> MakeLocalKV(
    int64_t capacity,
    int64_t row_bytes,
    const std::string& index_type,
    const std::string& value_dir,
    const std::string& allocator,
    int64_t allocator_capacity,
    int num_threads,
    int index_prefetch) {
  if (capacity <= 0 || row_bytes <= 0) {
    throw std::invalid_argument("capacity and row_bytes must be positive");
  }
  if (num_threads <= 0)
    num_threads = 1;
  std::filesystem::create_directories(value_dir);
  const uint64_t alloc_bytes =
      allocator_capacity > 0
          ? static_cast<uint64_t>(allocator_capacity)
          : std::max<uint64_t>(64ULL << 20,
                               static_cast<uint64_t>(capacity) *
                                   static_cast<uint64_t>(row_bytes + 8) * 2);

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(true);
  BaseKVConfig config;
  config.num_threads_ = static_cast<unsigned>(num_threads);
  config.json_config_ = {
      {"engine_type", "KVEngineComposite"},
      {"capacity", static_cast<uint64_t>(capacity)},
      {"index", {{"type", index_type}, {"prefetch_distance", index_prefetch}}},
      {"value",
       {{"type", "DRAM_VALUE_STORE"},
        {"path", value_dir},
        {"default_value_size_hint", row_bytes},
        {"dram_allocator",
         {{"type", allocator}, {"capacity_bytes", alloc_bytes}}}}}};
  auto resolved = base::ResolveEngine(config);
  std::unique_ptr<BaseKV> kv(
      base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
          resolved.engine, resolved.cfg));
  if (!kv)
    throw std::runtime_error("failed to create KVEngineComposite");
  return kv;
}

} // namespace recstore::python
