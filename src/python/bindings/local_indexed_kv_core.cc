#include "local_indexed_kv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <unistd.h>

#include "storage/index/utils/hash.h"

namespace recstore::python {

LocalIndexedKV::LocalIndexedKV(
    int64_t capacity_arg,
    int64_t embedding_dim,
    const std::string& index_type,
    const std::string& value_dir,
    const std::string& allocator,
    int64_t allocator_capacity,
    int num_threads,
    int index_prefetch)
    : capacity(capacity_arg),
      dim(embedding_dim),
      storage_dim(embedding_dim +
                  static_cast<int64_t>(sizeof(int64_t) / sizeof(float))),
      num_threads(std::max(1, num_threads)) {
  if (embedding_dim <= 0) {
    throw std::invalid_argument("embedding_dim must be positive");
  }
  kv = MakeLocalKV(
      capacity,
      storage_dim * sizeof(float),
      index_type,
      value_dir,
      allocator,
      allocator_capacity,
      num_threads,
      index_prefetch);
  worker_rows_scratch.resize(static_cast<size_t>(this->num_threads));
}

int LocalIndexedKV::EffectiveWorkers(size_t rows) const {
  constexpr size_t kMinRowsPerWorker = 4096;
  if (rows == 0)
    return 1;
  const size_t useful_workers =
      (rows + kMinRowsPerWorker - 1) / kMinRowsPerWorker;
  return std::max(1, std::min(num_threads, static_cast<int>(useful_workers)));
}

int64_t LocalIndexedKV::AllocateRowIndex() {
  if (!free_row_indices.empty()) {
    const int64_t row_index = free_row_indices.back();
    free_row_indices.pop_back();
    return row_index;
  }
  if (next_row_index >= capacity) {
    throw std::runtime_error("LocalIndexedKV row-index capacity exceeded");
  }
  return next_row_index++;
}

int64_t LocalIndexedKV::Size() {
  std::lock_guard<std::mutex> lock(mu);
  const uint64_t size = kv->ActiveKeyCount();
  if (size > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    throw std::runtime_error("LocalIndexedKV active key count is too large");
  }
  return static_cast<int64_t>(size);
}

py::tuple LocalIndexedKV::SnapshotIndexed() {
  std::lock_guard<std::mutex> lock(mu);
  const std::vector<uint64_t> keys = kv->SnapshotKeys();
  if (keys.size() >
      static_cast<size_t>((std::numeric_limits<py::ssize_t>::max)())) {
    throw std::runtime_error("LocalIndexedKV snapshot exceeds numpy limit");
  }

  const auto n = static_cast<py::ssize_t>(keys.size());
  py::array_t<int64_t> ids(n);
  py::array_t<int64_t> indices(n);
  py::array_t<float> values({n, static_cast<py::ssize_t>(dim)});
  if (keys.empty()) {
    return py::make_tuple(
        std::move(ids), std::move(indices), std::move(values));
  }
  static_assert(sizeof(uint64_t) == sizeof(int64_t));
  std::memcpy(ids.mutable_data(), keys.data(), keys.size() * sizeof(uint64_t));

  const int workers = EffectiveWorkers(keys.size());
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                       static_cast<size_t>(workers);
    if (begin == end)
      continue;
    auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    const bool ok    = kv->BatchGetDirectFixedRows(
        base::ConstArray<uint64_t>(
            keys.data() + begin, static_cast<int>(end - begin)),
        static_cast<int64_t>(end - begin),
        storage_dim,
        static_cast<unsigned>(worker),
        &local_rows);
    if (!ok || local_rows.size() != end - begin) {
      worker_ok[static_cast<size_t>(worker)] = 0;
      continue;
    }
  }
  if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
        return ok != 0;
      })) {
    throw std::runtime_error("LocalIndexedKV snapshot row lookup failed");
  }

  auto* index_ptr        = indices.mutable_data();
  auto* value_ptr        = values.mutable_data();
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  int invalid_row        = 0;
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(| : invalid_row)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      const size_t output_row = begin + local;
      const auto& row         = local_rows[local];
      if (row.missing || row.data == nullptr || row.size != row_bytes) {
        invalid_row = 1;
        continue;
      }
      std::memcpy(index_ptr + output_row, row.data, sizeof(int64_t));
      std::memcpy(value_ptr + output_row * static_cast<size_t>(dim),
                  row.data + sizeof(int64_t),
                  static_cast<size_t>(dim) * sizeof(float));
    }
  }
  if (invalid_row != 0) {
    throw std::runtime_error("LocalIndexedKV snapshot contains invalid rows");
  }
  return py::make_tuple(std::move(ids), std::move(indices), std::move(values));
}

int64_t LocalIndexedKV::DeleteKeysLocked(
    const std::vector<uint64_t>& requested_keys, bool reset_allocator) {
  std::vector<uint64_t> keys;
  keys.reserve(requested_keys.size());
  std::unordered_set<uint64_t> seen;
  seen.reserve(requested_keys.size());
  for (const uint64_t key : requested_keys) {
    if (seen.insert(key).second) {
      keys.push_back(key);
    }
  }

  std::vector<BaseKV::DirectFixedRow> rows;
  if (!keys.empty() &&
      !kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(keys),
          static_cast<int64_t>(keys.size()),
          storage_dim,
          0,
          &rows)) {
    throw std::runtime_error("LocalIndexedKV delete row lookup failed");
  }
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  std::vector<int64_t> deleted_indices;
  deleted_indices.reserve(keys.size());
  for (const auto& row : rows) {
    if (row.missing) {
      continue;
    }
    if (row.data == nullptr || row.size != row_bytes) {
      throw std::runtime_error("LocalIndexedKV delete found an invalid row");
    }
    int64_t row_index = -1;
    std::memcpy(&row_index, row.data, sizeof(int64_t));
    if (row_index < 0 || row_index >= next_row_index) {
      throw std::runtime_error(
          "LocalIndexedKV delete found an invalid stable index");
    }
    deleted_indices.push_back(row_index);
  }

  const uint64_t deleted =
      keys.empty() ? 0 : kv->BatchDelete(base::ConstArray<uint64_t>(keys), 0);
  if (deleted != deleted_indices.size()) {
    throw std::runtime_error(
        "LocalIndexedKV delete count does not match fetched rows");
  }
  for (const int64_t row_index : deleted_indices) {
    ClearAdamSlotRow(row_index);
  }
  if (reset_allocator) {
    if (kv->ActiveKeyCount() != 0) {
      throw std::runtime_error(
          "LocalIndexedKV clear left active keys in the engine");
    }
    next_row_index = 0;
    free_row_indices.clear();
  } else {
    free_row_indices.insert(
        free_row_indices.end(), deleted_indices.begin(), deleted_indices.end());
  }
  return static_cast<int64_t>(deleted);
}

int64_t LocalIndexedKV::DeleteIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        ids) {
  const auto keys = CheckIds(ids);
  std::lock_guard<std::mutex> lock(mu);
  return DeleteKeysLocked(keys, false);
}

int64_t LocalIndexedKV::ClearIndexed() {
  std::lock_guard<std::mutex> lock(mu);
  return DeleteKeysLocked(kv->SnapshotKeys(), true);
}

std::vector<int64_t> LocalIndexedKV::NormalizedFreeRows() {
  const std::vector<uint64_t> keys = kv->SnapshotKeys();
  if (keys.size() != kv->ActiveKeyCount()) {
    throw std::runtime_error("RecStore checkpoint key tracking/count mismatch");
  }
  std::vector<BaseKV::DirectFixedRow> rows;
  if (!keys.empty() &&
      !kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(keys),
          static_cast<int64_t>(keys.size()),
          storage_dim,
          0,
          &rows)) {
    throw std::runtime_error("RecStore checkpoint row lookup failed");
  }

  std::vector<uint8_t> occupied(static_cast<size_t>(next_row_index), 0);
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  for (const auto& row : rows) {
    if (row.missing || row.data == nullptr || row.size != row_bytes) {
      throw std::runtime_error("RecStore checkpoint found an invalid row");
    }
    int64_t row_index = -1;
    std::memcpy(&row_index, row.data, sizeof(int64_t));
    if (row_index < 0 || row_index >= next_row_index ||
        occupied[static_cast<size_t>(row_index)] != 0) {
      throw std::runtime_error(
          "RecStore checkpoint found an invalid stable index");
    }
    occupied[static_cast<size_t>(row_index)] = 1;
  }

  std::vector<uint8_t> already_free(static_cast<size_t>(next_row_index), 0);
  for (const int64_t row_index : free_row_indices) {
    if (row_index < 0 || row_index >= next_row_index ||
        occupied[static_cast<size_t>(row_index)] != 0 ||
        already_free[static_cast<size_t>(row_index)] != 0) {
      throw std::runtime_error(
          "RecStore checkpoint found an invalid free stable index");
    }
    already_free[static_cast<size_t>(row_index)] = 1;
  }

  // Reclaim any index reserved by a failed insert while preserving the LIFO
  // order of explicitly deleted rows at the back of the free list.
  std::vector<int64_t> normalized;
  normalized.reserve(static_cast<size_t>(next_row_index) - keys.size());
  for (int64_t row_index = 0; row_index < next_row_index; ++row_index) {
    if (occupied[static_cast<size_t>(row_index)] == 0 &&
        already_free[static_cast<size_t>(row_index)] == 0) {
      normalized.push_back(row_index);
    }
  }
  normalized.insert(
      normalized.end(), free_row_indices.begin(), free_row_indices.end());
  if (normalized.size() + keys.size() != static_cast<size_t>(next_row_index)) {
    throw std::runtime_error(
        "RecStore checkpoint allocator state is inconsistent");
  }
  return normalized;
}

void LocalIndexedKV::PutIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        indices,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        values) {
  const auto keys = CheckIdsView(ids);
  auto index_view = indices.unchecked<1>();
  auto value_view = values.unchecked<2>();
  if (index_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      value_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      value_view.shape(1) != dim) {
    throw std::invalid_argument(
        "put_indexed: ids, indices, and [num_ids, dim] values must align");
  }

  std::lock_guard<std::mutex> lock(mu);
  std::vector<float> packed(keys.size() * static_cast<size_t>(storage_dim));
  std::vector<base::ConstArray<float>> arrays(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    const int64_t row_index = index_view(static_cast<py::ssize_t>(i));
    if (row_index < 0 || row_index >= capacity) {
      throw std::invalid_argument("put_indexed: row index is out of range");
    }
    float* row = packed.data() + i * static_cast<size_t>(storage_dim);
    std::memcpy(row, &row_index, sizeof(int64_t));
    std::memcpy(row + sizeof(int64_t) / sizeof(float),
                value_view.data(static_cast<py::ssize_t>(i), 0),
                static_cast<size_t>(dim) * sizeof(float));
    arrays[i]      = base::ConstArray<float>(row, storage_dim);
    next_row_index = std::max(next_row_index, row_index + 1);
  }
  if (!keys.empty()) {
    kv->BatchPut(keys.AsConstArray(), &arrays, 0);
  }
}

py::dict LocalIndexedKV::BulkInsertIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
        indices,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& values,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  auto index_view        = indices.unchecked<1>();
  auto value_view        = values.unchecked<2>();
  if (index_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      value_view.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      value_view.shape(1) != dim) {
    throw std::invalid_argument(
        "bulk_insert_indexed: ids, indices, and [num_ids, dim] values "
        "must align");
  }
  const auto convert_end = clock::now();

  int invalid_index     = 0;
  int64_t max_row_index = -1;
  const int workers     = EffectiveWorkers(keys.size());
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(| : invalid_index) reduction(max : max_row_index)
  for (int64_t i = 0; i < static_cast<int64_t>(keys.size()); ++i) {
    const int64_t row_index = index_view(static_cast<py::ssize_t>(i));
    if (row_index < 0 || row_index >= capacity) {
      invalid_index = 1;
      continue;
    }
    max_row_index = std::max(max_row_index, row_index);
  }
  if (invalid_index != 0) {
    throw std::invalid_argument(
        "bulk_insert_indexed: row index is out of range");
  }
  const auto pack_end = clock::now();

  const auto engine_start = clock::now();
  std::vector<BaseKV::BulkLoadStats> worker_stats(static_cast<size_t>(workers));
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
  {
    std::lock_guard<std::mutex> lock(mu);
    auto insert = [&]() {
    // This path is intentionally restricted to unique keys that a
    // preceding lookup reserved as missing. Each worker owns a disjoint
    // range, so the engine can allocate values and build the index in
    // parallel without generic overwrite/retire bookkeeping.
#pragma omp parallel for num_threads(workers) schedule(static)
      for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = keys.size() * static_cast<size_t>(worker) /
                             static_cast<size_t>(workers);
        const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                           static_cast<size_t>(workers);
        if (begin == end)
          continue;
        worker_ok[static_cast<size_t>(worker)] =
            kv->BulkLoadIndexedFloatRange(
                base::ConstArray<uint64_t>(
                    keys.data() + begin, static_cast<int>(end - begin)),
                index_view.data(static_cast<py::ssize_t>(begin)),
                value_view.data(static_cast<py::ssize_t>(begin), 0),
                static_cast<int64_t>(end - begin),
                dim,
                static_cast<unsigned>(worker),
                &worker_stats[static_cast<size_t>(worker)])
                ? 1
                : 0;
      }
    };
    if (release_gil) {
      py::gil_scoped_release nogil;
      insert();
    } else {
      insert();
    }
    if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
          return ok != 0;
        })) {
      throw std::runtime_error(
          "BulkLoadIndexedFloatRange failed for local indexed rows");
    }
    next_row_index = std::max(next_row_index, max_row_index + 1);
  }
  const auto engine_end = clock::now();
  auto ns               = [](clock::duration d) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
  };
  py::dict profile;
  profile["arg_convert_ns"]   = ns(convert_end - total_start);
  profile["pack_ns"]          = ns(pack_end - convert_end);
  profile["engine_insert_ns"] = ns(engine_end - engine_start);
  for (const auto& field : {
           std::make_pair("prepare", &BaseKV::BulkLoadStats::prepare_ns),
           std::make_pair(
               "value_alloc", &BaseKV::BulkLoadStats::value_alloc_ns),
           std::make_pair(
               "handle_validate", &BaseKV::BulkLoadStats::handle_validate_ns),
           std::make_pair("index_put", &BaseKV::BulkLoadStats::index_put_ns),
           std::make_pair("key_track", &BaseKV::BulkLoadStats::key_track_ns),
           std::make_pair("worker_total", &BaseKV::BulkLoadStats::total_ns),
       }) {
    uint64_t cpu_ns = 0;
    uint64_t max_ns = 0;
    for (const auto& worker_stat : worker_stats) {
      const uint64_t value = worker_stat.*(field.second);
      cpu_ns += value;
      max_ns = std::max(max_ns, value);
    }
    profile[py::str(std::string("bulk_") + field.first + "_cpu_ns")] =
        static_cast<double>(cpu_ns);
    profile[py::str(std::string("bulk_") + field.first + "_max_ns")] =
        static_cast<double>(max_ns);
  }
  profile["total_ns"] = ns(engine_end - total_start);
  profile["rows"]     = static_cast<double>(keys.size());
  profile["workers"]  = static_cast<double>(workers);
  return profile;
}

py::dict LocalIndexedKV::StorageStats() {
  std::lock_guard<std::mutex> lock(mu);
  const uint64_t active_rows = kv->ActiveKeyCount();
  const uint64_t stable_rows = static_cast<uint64_t>(next_row_index);
  const uint64_t free_rows   = static_cast<uint64_t>(free_row_indices.size());
  const uint64_t embedding_row_bytes =
      static_cast<uint64_t>(dim) * sizeof(float);
  const uint64_t value_row_bytes =
      static_cast<uint64_t>(storage_dim) * sizeof(float);
  const uint64_t adam_bytes =
      static_cast<uint64_t>(adam_slot_blocks.size()) *
      static_cast<uint64_t>(kAdamSlotBlockRows) * static_cast<uint64_t>(dim) *
      sizeof(float) * 2;
  py::dict stats;
  stats["capacity"]            = static_cast<double>(capacity);
  stats["active_rows"]         = static_cast<double>(active_rows);
  stats["stable_rows"]         = static_cast<double>(stable_rows);
  stats["free_rows"]           = static_cast<double>(free_rows);
  stats["embedding_row_bytes"] = static_cast<double>(embedding_row_bytes);
  stats["value_row_bytes"]     = static_cast<double>(value_row_bytes);
  stats["adam_slot_bytes"]     = static_cast<double>(adam_bytes);
  stats["estimated_embedding_bytes"] =
      static_cast<double>(stable_rows * embedding_row_bytes);
  stats["estimated_value_bytes"] =
      static_cast<double>(active_rows * value_row_bytes);
  return stats;
}

} // namespace recstore::python
