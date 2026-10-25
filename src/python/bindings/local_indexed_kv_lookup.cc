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

py::tuple LocalIndexedKV::LookupIndexedImpl(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    bool allocate_missing,
    py::dict* profile) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  const auto convert_end = clock::now();
  const auto n           = static_cast<py::ssize_t>(keys.size());
  py::array_t<int64_t> indices(n);
  py::array_t<float> values({n, static_cast<py::ssize_t>(dim)});
  py::array_t<bool> found(n);
  auto* index_ptr      = indices.mutable_data();
  auto* value_ptr      = values.mutable_data();
  auto* found_ptr      = found.mutable_data();
  const auto alloc_end = clock::now();

  std::unique_lock<std::mutex> lock(mu);
  const auto lock_end = clock::now();
  const int workers   = EffectiveWorkers(keys.size());
  std::vector<BaseKV::BatchGetFlatStats> worker_stats(
      static_cast<size_t>(workers));
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
        &local_rows,
        &worker_stats[static_cast<size_t>(worker)]);
    if (!ok || local_rows.size() != end - begin) {
      worker_ok[static_cast<size_t>(worker)] = 0;
      continue;
    }
  }
  if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
        return ok != 0;
      })) {
    throw std::runtime_error("BatchGetDirectFixedRows failed");
  }
  const auto engine_end = clock::now();
  std::unordered_map<uint64_t, int64_t> new_indices;
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  int invalid_row        = 0;
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(| : invalid_row)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      const size_t i  = begin + local;
      const auto& row = local_rows[local];
      if (row.missing) {
        found_ptr[i] = false;
        std::memset(value_ptr + i * static_cast<size_t>(dim),
                    0,
                    static_cast<size_t>(dim) * sizeof(float));
        index_ptr[i] = -1;
        continue;
      }
      if (row.data == nullptr || row.size != row_bytes) {
        invalid_row = 1;
        continue;
      }
      found_ptr[i] = true;
      std::memcpy(index_ptr + i, row.data, sizeof(int64_t));
      std::memcpy(value_ptr + i * static_cast<size_t>(dim),
                  row.data + sizeof(int64_t),
                  static_cast<size_t>(dim) * sizeof(float));
    }
  }
  if (invalid_row != 0) {
    throw std::runtime_error("LocalIndexedKV stored row has invalid size");
  }
  if (allocate_missing) {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (found_ptr[i])
        continue;
      auto it = new_indices.find(keys[i]);
      if (it == new_indices.end()) {
        it = new_indices.emplace(keys[i], AllocateRowIndex()).first;
      }
      index_ptr[i] = it->second;
    }
  }
  const auto unpack_end = clock::now();
  if (profile != nullptr) {
    auto ns = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    (*profile)["arg_convert_ns"]  = ns(convert_end - total_start);
    (*profile)["output_alloc_ns"] = ns(alloc_end - convert_end);
    (*profile)["lock_wait_ns"]    = ns(lock_end - alloc_end);
    (*profile)["engine_get_ns"]   = ns(engine_end - lock_end);
    uint64_t index_lookup_ns      = 0;
    uint64_t missing_rows         = 0;
    for (const auto& stats : worker_stats) {
      index_lookup_ns += stats.index_lookup_ns;
      missing_rows += stats.missing_rows;
    }
    (*profile)["engine_index_lookup_ns"] = static_cast<double>(index_lookup_ns);
    (*profile)["unpack_copy_ns"]         = ns(unpack_end - engine_end);
    (*profile)["total_ns"]               = ns(unpack_end - total_start);
    (*profile)["missing_rows"]           = static_cast<double>(missing_rows);
    (*profile)["rows"]                   = static_cast<double>(keys.size());
    (*profile)["workers"]                = static_cast<double>(workers);
  }
  return py::make_tuple(
      std::move(indices), std::move(values), std::move(found));
}

py::tuple LocalIndexedKV::LookupIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    bool allocate_missing) {
  return LookupIndexedImpl(ids, allocate_missing, nullptr);
}

py::tuple LocalIndexedKV::LookupIndexedProfiled(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    bool allocate_missing) {
  py::dict profile;
  py::tuple result = LookupIndexedImpl(ids, allocate_missing, &profile);
  return py::make_tuple(result[0], result[1], result[2], std::move(profile));
}

py::dict LocalIndexedKV::LookupIndexedIntoRaw(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t indices_address,
    uintptr_t values_address,
    uintptr_t found_address,
    bool allocate_missing,
    bool release_gil,
    bool collect_profile) {
  return LookupIndexedIntoRawImpl(
      ids_address,
      num_rows,
      indices_address,
      values_address,
      found_address,
      allocate_missing,
      release_gil,
      collect_profile,
      true);
}

py::dict LocalIndexedKV::LookupIndexedLegacyIntoRaw(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t indices_address,
    uintptr_t values_address,
    uintptr_t found_address,
    bool allocate_missing,
    bool release_gil,
    bool collect_profile) {
  return LookupIndexedIntoRawImpl(
      ids_address,
      num_rows,
      indices_address,
      values_address,
      found_address,
      allocate_missing,
      release_gil,
      collect_profile,
      false);
}

py::dict LocalIndexedKV::LookupIndexedFusedIntoRaw(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t indices_address,
    uintptr_t values_address,
    uintptr_t found_address,
    bool allocate_missing,
    bool release_gil,
    bool collect_profile) {
  return LookupIndexedIntoRawImpl(
      ids_address,
      num_rows,
      indices_address,
      values_address,
      found_address,
      allocate_missing,
      release_gil,
      collect_profile,
      true);
}

py::dict LocalIndexedKV::LookupIndexedIntoRawImpl(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t indices_address,
    uintptr_t values_address,
    uintptr_t found_address,
    bool allocate_missing,
    bool release_gil,
    bool collect_profile,
    bool fused) {
  // This ABI-neutral entry point is intentionally internal. Framework
  // adapters validate dtype, device, contiguity, shape, and buffer lifetime
  // before passing the CPU tensor addresses used for this synchronous call.
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  if (num_rows < 0 ||
      static_cast<uint64_t>(num_rows) >
          static_cast<uint64_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("lookup_indexed_into_raw: invalid row count");
  }
  if (num_rows > 0 && (ids_address == 0 || indices_address == 0 ||
                       values_address == 0 || found_address == 0)) {
    throw std::invalid_argument(
        "lookup_indexed_into_raw: non-empty tensors require valid pointers");
  }

  const auto keys = Int64KeyView{reinterpret_cast<const uint64_t*>(ids_address),
                                 static_cast<size_t>(num_rows)};
  auto* index_ptr = reinterpret_cast<int64_t*>(indices_address);
  auto* value_ptr = reinterpret_cast<float*>(values_address);
  auto* found_ptr = reinterpret_cast<bool*>(found_address);
  const auto convert_end = clock::now();

  const int workers = EffectiveWorkers(keys.size());
  std::vector<BaseKV::BatchGetFlatStats> worker_stats(
      static_cast<size_t>(workers));
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
  std::vector<uint64_t> worker_get_ns(static_cast<size_t>(workers), 0);
  std::vector<uint64_t> worker_unpack_ns(static_cast<size_t>(workers), 0);
  clock::time_point lock_start;
  clock::time_point lock_end;
  clock::time_point engine_end;
  uint64_t missing_rows = 0;

  auto execute = [&]() {
    lock_start = clock::now();
    std::unique_lock<std::mutex> lock(mu);
    lock_end               = clock::now();
    const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(+ : missing_rows)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                         static_cast<size_t>(workers);
      if (begin == end)
        continue;
      if (fused) {
        auto& stats          = worker_stats[static_cast<size_t>(worker)];
        const auto get_start = clock::now();
        const bool ok        = kv->BatchGetIndexedFlat(
            base::ConstArray<uint64_t>(
                keys.data() + begin, static_cast<int>(end - begin)),
            index_ptr + begin,
            value_ptr + begin * static_cast<size_t>(dim),
            found_ptr + begin,
            static_cast<int64_t>(end - begin),
            dim,
            static_cast<unsigned>(worker),
            &stats,
            collect_profile);
        worker_get_ns[static_cast<size_t>(worker)] = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock::now() - get_start)
                .count());
        worker_ok[static_cast<size_t>(worker)] = ok ? 1 : 0;
        missing_rows += stats.missing_rows;
        continue;
      }
      auto& local_rows     = worker_rows_scratch[static_cast<size_t>(worker)];
      const auto get_start = clock::now();
      const bool ok        = kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(
              keys.data() + begin, static_cast<int>(end - begin)),
          static_cast<int64_t>(end - begin),
          storage_dim,
          static_cast<unsigned>(worker),
          &local_rows,
          collect_profile ? &worker_stats[static_cast<size_t>(worker)]
                                 : nullptr);
      const auto get_end                         = clock::now();
      worker_get_ns[static_cast<size_t>(worker)] = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              get_end - get_start)
              .count());
      if (!ok || local_rows.size() != end - begin) {
        worker_ok[static_cast<size_t>(worker)] = 0;
        continue;
      }

      const auto unpack_start = clock::now();
      for (size_t local = 0; local < local_rows.size(); ++local) {
        const size_t output_row = begin + local;
        const auto& row         = local_rows[local];
        if (row.missing) {
          found_ptr[output_row] = false;
          std::memset(value_ptr + output_row * static_cast<size_t>(dim),
                      0,
                      static_cast<size_t>(dim) * sizeof(float));
          index_ptr[output_row] = -1;
          ++missing_rows;
          continue;
        }
        if (row.data == nullptr || row.size != row_bytes) {
          worker_ok[static_cast<size_t>(worker)] = 0;
          break;
        }
        found_ptr[output_row] = true;
        std::memcpy(index_ptr + output_row, row.data, sizeof(int64_t));
        std::memcpy(value_ptr + output_row * static_cast<size_t>(dim),
                    row.data + sizeof(int64_t),
                    static_cast<size_t>(dim) * sizeof(float));
      }
      worker_unpack_ns[static_cast<size_t>(worker)] = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              clock::now() - unpack_start)
              .count());
    }
    if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
          return ok != 0;
        })) {
      throw std::runtime_error(
          "lookup_indexed_into_raw: engine lookup or row unpack failed");
    }

    if (allocate_missing && missing_rows != 0) {
      std::unordered_map<uint64_t, int64_t> new_indices;
      new_indices.reserve(static_cast<size_t>(missing_rows));
      for (size_t i = 0; i < keys.size(); ++i) {
        if (found_ptr[i])
          continue;
        auto it = new_indices.find(keys[i]);
        if (it == new_indices.end()) {
          it = new_indices.emplace(keys[i], AllocateRowIndex()).first;
        }
        index_ptr[i] = it->second;
      }
    }
    engine_end = clock::now();
  };

  if (release_gil) {
    py::gil_scoped_release nogil;
    execute();
  } else {
    execute();
  }
  const auto total_end = clock::now();

  py::dict profile;
  profile["missing_rows"] = static_cast<double>(missing_rows);
  profile["rows"]         = static_cast<double>(keys.size());
  profile["workers"]      = static_cast<double>(workers);
  if (!collect_profile)
    return profile;

  uint64_t get_cpu_ns          = 0;
  uint64_t get_max_ns          = 0;
  uint64_t unpack_cpu_ns       = 0;
  uint64_t unpack_max_ns       = 0;
  uint64_t index_lookup_cpu_ns = 0;
  uint64_t index_lookup_max_ns = 0;
  for (int worker = 0; worker < workers; ++worker) {
    const size_t index = static_cast<size_t>(worker);
    get_cpu_ns += worker_get_ns[index];
    get_max_ns = std::max(get_max_ns, worker_get_ns[index]);
    unpack_cpu_ns += worker_unpack_ns[index];
    unpack_max_ns = std::max(unpack_max_ns, worker_unpack_ns[index]);
    index_lookup_cpu_ns += worker_stats[index].index_lookup_ns;
    index_lookup_max_ns =
        std::max(index_lookup_max_ns, worker_stats[index].index_lookup_ns);
  }
  auto ns = [](clock::duration duration) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
  };
  profile["arg_convert_ns"]  = ns(convert_end - total_start);
  profile["output_alloc_ns"] = 0.0;
  profile["lock_wait_ns"]    = ns(lock_end - lock_start);
  // These are critical-path estimates inside the fused parallel region.
  profile["engine_get_ns"]     = static_cast<double>(get_max_ns);
  profile["engine_get_cpu_ns"] = static_cast<double>(get_cpu_ns);
  // `engine_index_lookup_ns` is retained as the aggregate CPU time for
  // compatibility.  The max value is the corresponding critical-path
  // estimate; summing worker times must not be compared with wall time.
  profile["engine_index_lookup_ns"] = static_cast<double>(index_lookup_cpu_ns);
  profile["engine_index_lookup_cpu_ns"] =
      static_cast<double>(index_lookup_cpu_ns);
  profile["engine_index_lookup_max_ns"] =
      static_cast<double>(index_lookup_max_ns);
  profile["unpack_copy_ns"]     = static_cast<double>(unpack_max_ns);
  profile["unpack_copy_cpu_ns"] = static_cast<double>(unpack_cpu_ns);
  if (fused) {
    uint64_t copy_cpu_ns = 0;
    uint64_t copy_max_ns = 0;
    for (const auto& stats : worker_stats) {
      copy_cpu_ns += stats.row_copy_ns;
      copy_max_ns = std::max(copy_max_ns, stats.row_copy_ns);
    }
    profile["row_copy_cpu_ns"] = static_cast<double>(copy_cpu_ns);
    profile["row_copy_max_ns"] = static_cast<double>(copy_max_ns);
  }
  profile["engine_get_unpack_ns"] = ns(engine_end - lock_end);
  profile["total_ns"]             = ns(total_end - total_start);
  return profile;
}

uint64_t LocalIndexedKV::LookupValuesIntoRawImpl(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t values_address,
    bool release_gil,
    py::dict* profile) {
  using clock                = std::chrono::steady_clock;
  const bool collect_profile = profile != nullptr;
  const auto total_start     = clock::now();
  if (num_rows < 0 ||
      static_cast<uint64_t>(num_rows) >
          static_cast<uint64_t>((std::numeric_limits<int>::max)())) {
    throw std::invalid_argument("lookup_values_into_raw: invalid row count");
  }
  if (num_rows > 0 && (ids_address == 0 || values_address == 0)) {
    throw std::invalid_argument(
        "lookup_values_into_raw: non-empty tensors require valid pointers");
  }

  const auto keys = Int64KeyView{reinterpret_cast<const uint64_t*>(ids_address),
                                 static_cast<size_t>(num_rows)};
  auto* value_ptr = reinterpret_cast<float*>(values_address);
  const int workers = EffectiveWorkers(keys.size());
  std::vector<BaseKV::BatchGetFlatStats> worker_stats(
      static_cast<size_t>(workers));
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);
  clock::time_point lock_start;
  clock::time_point lock_end;
  clock::time_point engine_end;

  auto execute = [&]() {
    lock_start = clock::now();
    std::unique_lock<std::mutex> lock(mu);
    lock_end = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
      const size_t begin = keys.size() * static_cast<size_t>(worker) /
                           static_cast<size_t>(workers);
      const size_t end = keys.size() * static_cast<size_t>(worker + 1) /
                         static_cast<size_t>(workers);
      if (begin == end)
        continue;
      worker_ok[static_cast<size_t>(worker)] =
          kv->BatchGetFlatRange(
              base::ConstArray<uint64_t>(
                  keys.data() + begin, static_cast<int>(end - begin)),
              value_ptr + begin * static_cast<size_t>(dim),
              static_cast<int64_t>(end - begin),
              storage_dim,
              static_cast<int64_t>(sizeof(int64_t) / sizeof(float)),
              dim,
              static_cast<unsigned>(worker),
              &worker_stats[static_cast<size_t>(worker)],
              collect_profile)
              ? 1
              : 0;
    }
    engine_end = clock::now();
  };

  if (release_gil) {
    py::gil_scoped_release nogil;
    execute();
  } else {
    execute();
  }
  if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
        return ok != 0;
      })) {
    throw std::runtime_error("lookup_values_into_raw: engine lookup failed");
  }
  const auto total_end = clock::now();

  uint64_t missing_rows        = 0;
  uint64_t index_lookup_cpu_ns = 0;
  uint64_t index_lookup_max_ns = 0;
  uint64_t row_copy_cpu_ns     = 0;
  uint64_t row_copy_max_ns     = 0;
  for (const auto& stats : worker_stats) {
    missing_rows += stats.missing_rows;
    index_lookup_cpu_ns += stats.index_lookup_ns;
    index_lookup_max_ns = std::max(index_lookup_max_ns, stats.index_lookup_ns);
    row_copy_cpu_ns += stats.row_copy_ns;
    row_copy_max_ns = std::max(row_copy_max_ns, stats.row_copy_ns);
  }

  if (!collect_profile)
    return missing_rows;

  auto ns = [](clock::duration duration) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
  };
  (*profile)["missing_rows"] = static_cast<double>(missing_rows);
  (*profile)["rows"]         = static_cast<double>(keys.size());
  (*profile)["workers"]      = static_cast<double>(workers);
  (*profile)["lock_wait_ns"] = ns(lock_end - lock_start);
  (*profile)["engine_index_lookup_cpu_ns"] =
      static_cast<double>(index_lookup_cpu_ns);
  (*profile)["engine_index_lookup_max_ns"] =
      static_cast<double>(index_lookup_max_ns);
  (*profile)["row_copy_cpu_ns"]      = static_cast<double>(row_copy_cpu_ns);
  (*profile)["row_copy_max_ns"]      = static_cast<double>(row_copy_max_ns);
  (*profile)["engine_get_unpack_ns"] = ns(engine_end - lock_end);
  (*profile)["total_ns"]             = ns(total_end - total_start);
  return missing_rows;
}

uint64_t LocalIndexedKV::LookupValuesIntoRaw(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t values_address,
    bool release_gil) {
  return LookupValuesIntoRawImpl(
      ids_address, num_rows, values_address, release_gil, nullptr);
}

py::dict LocalIndexedKV::LookupValuesIntoRawProfiled(
    uintptr_t ids_address,
    int64_t num_rows,
    uintptr_t values_address,
    bool release_gil) {
  py::dict profile;
  LookupValuesIntoRawImpl(
      ids_address, num_rows, values_address, release_gil, &profile);
  return profile;
}

} // namespace recstore::python
