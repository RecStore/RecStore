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

void LocalIndexedKV::ClearAdamSlotRow(int64_t row_index) {
  const size_t block = static_cast<size_t>(row_index / kAdamSlotBlockRows);
  if (block >= adam_slot_blocks.size()) {
    return;
  }
  std::memset(AdamSlotPtr(row_index, false),
              0,
              static_cast<size_t>(dim) * sizeof(float));
  std::memset(AdamSlotPtr(row_index, true),
              0,
              static_cast<size_t>(dim) * sizeof(float));
}

void LocalIndexedKV::EnsureAdamSlotRows(int64_t rows) {
  if (rows < 0 || rows > capacity) {
    throw std::runtime_error("Adam slot row count exceeds engine capacity");
  }
  const size_t required_blocks =
      static_cast<size_t>((rows + kAdamSlotBlockRows - 1) / kAdamSlotBlockRows);
  while (adam_slot_blocks.size() < required_blocks) {
    adam_slot_blocks.push_back(std::make_unique<AdamSlotBlock>(dim));
  }
}

float* LocalIndexedKV::AdamSlotPtr(int64_t row_index, bool squared) {
  const size_t block = static_cast<size_t>(row_index / kAdamSlotBlockRows);
  const size_t row   = static_cast<size_t>(row_index % kAdamSlotBlockRows);
  auto& values       = squared ? adam_slot_blocks[block]->exp_avg_sq
                               : adam_slot_blocks[block]->exp_avg;
  return values.data() + row * static_cast<size_t>(dim);
}

bool LocalIndexedKV::ApplySgdUpdateParallel(
    const Int64KeyView& keys, const float* grads, float lr) {
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
    worker_ok[static_cast<size_t>(worker)] =
        kv->ApplySgdUpdateFlatRange(
            base::ConstArray<uint64_t>(
                keys.data() + begin, static_cast<int>(end - begin)),
            grads + begin * static_cast<size_t>(dim),
            static_cast<int64_t>(end - begin),
            storage_dim,
            static_cast<int64_t>(sizeof(int64_t) / sizeof(float)),
            dim,
            lr,
            static_cast<unsigned>(worker))
            ? 1
            : 0;
  }
  return std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
    return ok != 0;
  });
}

bool LocalIndexedKV::ApplyAdamWTFUpdateParallel(
    const Int64KeyView& keys,
    const float* grads,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    AdamUpdateStats* stats) {
  using clock            = std::chrono::steady_clock;
  const int workers      = EffectiveWorkers(keys.size());
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  std::vector<BaseKV::BatchGetFlatStats> worker_stats(
      static_cast<size_t>(workers));
  std::vector<uint8_t> worker_ok(static_cast<size_t>(workers), 1);

  const auto get_start = clock::now();
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
  const auto get_end = clock::now();
  if (!std::all_of(worker_ok.begin(), worker_ok.end(), [](uint8_t ok) {
        return ok != 0;
      })) {
    return false;
  }

  auto& row_indices = row_indices_scratch;
  row_indices.resize(keys.size());
  int invalid_row       = 0;
  int64_t max_row_index = -1;
#pragma omp parallel for num_threads(workers) schedule(static)                 \
    reduction(| : invalid_row) reduction(max : max_row_index)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      const auto& row = local_rows[local];
      if (row.missing || row.data == nullptr || row.size != row_bytes) {
        invalid_row = 1;
        continue;
      }
      int64_t row_index = -1;
      std::memcpy(&row_index, row.data, sizeof(int64_t));
      if (row_index < 0 || row_index >= capacity) {
        invalid_row = 1;
        continue;
      }
      row_indices[begin + local] = row_index;
      max_row_index              = std::max(max_row_index, row_index);
    }
  }
  if (invalid_row != 0)
    return false;

  const auto alloc_start = clock::now();
  EnsureAdamSlotRows(max_row_index + 1);
  const auto alloc_end = clock::now();

  const float next_beta1_t = adam_beta1_t * static_cast<float>(beta1);
  const float next_beta2_t = adam_beta2_t * static_cast<float>(beta2);
  const float alpha        = -static_cast<float>(lr) / (1.0f - next_beta1_t) *
                      std::sqrt(1.0f - next_beta2_t);
  const double weight_scale = 1.0 - weight_decay;
  const auto math_start     = clock::now();
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = keys.size() * static_cast<size_t>(worker) /
                         static_cast<size_t>(workers);
    const auto& local_rows = worker_rows_scratch[static_cast<size_t>(worker)];
    for (size_t local = 0; local < local_rows.size(); ++local) {
      const size_t row_pos = begin + local;
      float* embedding     = reinterpret_cast<float*>(
          const_cast<char*>(local_rows[local].data) + sizeof(int64_t));
      float* exp_avg    = AdamSlotPtr(row_indices[row_pos], false);
      float* exp_avg_sq = AdamSlotPtr(row_indices[row_pos], true);
      const float* grad = grads + row_pos * static_cast<size_t>(dim);
      for (int64_t c = 0; c < dim; ++c) {
        const size_t col       = static_cast<size_t>(c);
        const float grad_value = grad[col];
        embedding[col]         = static_cast<float>(
            static_cast<double>(embedding[col]) * weight_scale);
        exp_avg[col] = static_cast<float>(
            beta1 * static_cast<double>(exp_avg[col]) +
            (1.0 - beta1) * static_cast<double>(grad_value));
        exp_avg_sq[col] = static_cast<float>(
            beta2 * static_cast<double>(exp_avg_sq[col]) +
            (1.0 - beta2) * static_cast<double>(grad_value) *
                static_cast<double>(grad_value));
        embedding[col] = static_cast<float>(
            static_cast<double>(embedding[col]) +
            static_cast<double>(alpha) * static_cast<double>(exp_avg[col]) /
                (static_cast<double>(std::sqrt(exp_avg_sq[col])) + eps));
      }
    }
  }
  const auto math_end = clock::now();
  adam_beta1_t        = next_beta1_t;
  adam_beta2_t        = next_beta2_t;
  ++adam_step;

  if (stats != nullptr) {
    stats->engine_get_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            get_end - get_start)
            .count());
    for (const auto& worker_stat : worker_stats) {
      stats->index_lookup_ns += worker_stat.index_lookup_ns;
      stats->index_lookup_max_ns =
          std::max(stats->index_lookup_max_ns, worker_stat.index_lookup_ns);
    }
    stats->slot_alloc_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            alloc_end - alloc_start)
            .count());
    stats->math_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            math_end - math_start)
            .count());
  }
  return true;
}

py::dict LocalIndexedKV::SgdUpdateIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
    double lr,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  const auto convert_end = clock::now();
  auto g                 = grads.unchecked<2>();
  if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      g.shape(1) != dim) {
    throw std::invalid_argument(
        "sgd_update_indexed: grads must be [num_ids, dim] float");
  }
  if (!(lr > 0.0)) {
    throw std::invalid_argument("sgd_update_indexed: lr must be positive");
  }

  const auto engine_start = clock::now();
  bool ok                 = true;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!keys.empty())
        ok = ApplySgdUpdateParallel(keys, g.data(0, 0), static_cast<float>(lr));
    } else if (!keys.empty()) {
      ok = ApplySgdUpdateParallel(keys, g.data(0, 0), static_cast<float>(lr));
    }
  }
  const auto engine_end = clock::now();
  if (!ok) {
    throw std::runtime_error(
        "ApplySgdUpdateFlatRange failed (all keys must already exist)");
  }
  auto ns = [](clock::duration d) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
  };
  py::dict profile;
  profile["arg_convert_ns"]   = ns(convert_end - total_start);
  profile["engine_update_ns"] = ns(engine_end - engine_start);
  profile["total_ns"]         = ns(engine_end - total_start);
  profile["rows"]             = static_cast<double>(keys.size());
  profile["workers"] = static_cast<double>(EffectiveWorkers(keys.size()));
  return profile;
}

py::dict LocalIndexedKV::AdamWTFUpdateIndexed(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& grads,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    bool release_gil) {
  using clock            = std::chrono::steady_clock;
  const auto total_start = clock::now();
  const auto keys        = CheckIdsView(ids);
  const auto convert_end = clock::now();
  auto g                 = grads.unchecked<2>();
  if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
      g.shape(1) != dim) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed: grads must be [num_ids, dim] float");
  }
  if (!(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0) ||
      eps < 0.0 || lr < 0.0 || weight_decay < 0.0) {
    throw std::invalid_argument(
        "adamw_tf_update_indexed: invalid optimizer parameters");
  }

  const auto lock_start = clock::now();
  bool ok               = true;
  AdamUpdateStats update_stats;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto lock_end = clock::now();
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!keys.empty()) {
        ok = ApplyAdamWTFUpdateParallel(
            keys,
            g.data(0, 0),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            &update_stats);
      }
    } else if (!keys.empty()) {
      ok = ApplyAdamWTFUpdateParallel(
          keys,
          g.data(0, 0),
          lr,
          beta1,
          beta2,
          eps,
          weight_decay,
          &update_stats);
    }
    if (!ok) {
      throw std::runtime_error(
          "AdamWTF update failed (keys must exist and stable indices must "
          "be valid)");
    }
    const auto engine_end = clock::now();
    auto ns               = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    py::dict profile;
    profile["arg_convert_ns"]   = ns(convert_end - total_start);
    profile["lock_wait_ns"]     = ns(lock_end - lock_start);
    profile["engine_update_ns"] = ns(engine_end - lock_end);
    profile["engine_get_ns"] = static_cast<double>(update_stats.engine_get_ns);
    profile["engine_index_lookup_ns"] =
        static_cast<double>(update_stats.index_lookup_ns);
    profile["engine_index_lookup_max_ns"] =
        static_cast<double>(update_stats.index_lookup_max_ns);
    profile["slot_alloc_ns"] = static_cast<double>(update_stats.slot_alloc_ns);
    profile["adam_math_ns"]  = static_cast<double>(update_stats.math_ns);
    profile["total_ns"]      = ns(engine_end - total_start);
    profile["rows"]          = static_cast<double>(keys.size());
    profile["workers"] = static_cast<double>(EffectiveWorkers(keys.size()));
    profile["step"]    = static_cast<double>(adam_step);
    return profile;
  }
}

py::dict LocalIndexedKV::OptimizerStateStats() const {
  py::dict stats;
  stats["adam_step"]        = static_cast<double>(adam_step);
  stats["adam_beta1_t"]     = static_cast<double>(adam_beta1_t);
  stats["adam_beta2_t"]     = static_cast<double>(adam_beta2_t);
  stats["adam_slot_blocks"] = static_cast<double>(adam_slot_blocks.size());
  stats["adam_slot_rows"]   = static_cast<double>(
      adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows));
  stats["adam_slot_bytes"] = static_cast<double>(
      adam_slot_blocks.size() * static_cast<size_t>(kAdamSlotBlockRows) *
      static_cast<size_t>(dim) * sizeof(float) * 2);
  return stats;
}

} // namespace recstore::python
