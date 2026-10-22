#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>

#include "recstore_kv_bindings.h"

namespace recstore::python {

struct LocalKV {
  std::unique_ptr<BaseKV> kv;
  int64_t dim;
  std::string value_path;
  uint64_t adam_step = 0;
  float adam_beta1_t = 1.0f;
  float adam_beta2_t = 1.0f;
  std::unordered_map<uint64_t, std::vector<float>> adam_m;
  std::unordered_map<uint64_t, std::vector<float>> adam_v;

  LocalKV(int64_t capacity,
          int64_t embedding_dim,
          const std::string& index_type,
          const std::string& value_dir,
          const std::string& allocator,
          int64_t allocator_capacity,
          int num_threads)
      : dim(embedding_dim), value_path(value_dir) {
    if (capacity <= 0 || embedding_dim <= 0) {
      throw std::invalid_argument(
          "capacity and embedding_dim must be positive");
    }
    kv = MakeLocalKV(
        capacity,
        embedding_dim * sizeof(float),
        index_type,
        value_dir,
        allocator,
        allocator_capacity,
        num_threads);
  }

  // Returns [num_ids, dim] float32 numpy array; missing keys read as zeros.
  py::array_t<float>
  Lookup(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
             ids) {
    const auto keys = CheckIds(ids);
    py::array_t<float> out(
        {static_cast<py::ssize_t>(keys.size()), static_cast<py::ssize_t>(dim)});
    float* out_ptr = out.mutable_data();
    if (keys.size()) {
      std::memset(out_ptr, 0, keys.size() * dim * sizeof(float));
      if (!kv->BatchGetFlat(
              base::ConstArray<uint64_t>(keys),
              out_ptr,
              static_cast<int64_t>(keys.size()),
              dim,
              0)) {
        throw std::runtime_error("BatchGetFlat failed");
      }
    }
    return out;
  }

  // Phase-profiled lookup: same work as Lookup but returns per-phase
  // nanoseconds so callers can attribute cost to each stage of the read path:
  //   arg_convert  numpy -> std::vector<uint64_t> (per-element cast loop)
  //   output_alloc py::array_t allocation + pre-memset
  //   index_lookup PetHash probe per key (HintPrefetch + GetInternal)
  //   row_copy     value store ReadFlatFixedRows (ptr translate + memcpy)
  //   zero_fill    memset of missing rows (0 when all hit)
  //   total        whole call
  std::pair<py::array_t<float>, py::dict> LookupProfiled(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids) {
    using clock     = std::chrono::steady_clock;
    const auto t0   = clock::now();
    const auto keys = CheckIds(ids);
    const auto t1   = clock::now();
    py::array_t<float> out(
        {static_cast<py::ssize_t>(keys.size()), static_cast<py::ssize_t>(dim)});
    float* out_ptr = out.mutable_data();
    std::memset(out_ptr, 0, keys.size() * dim * sizeof(float));
    const auto t2 = clock::now();
    BaseKV::BatchGetFlatStats stats;
    bool ok = true;
    if (keys.size()) {
      ok = kv->BatchGetFlat(
          base::ConstArray<uint64_t>(keys),
          out_ptr,
          static_cast<int64_t>(keys.size()),
          dim,
          0,
          &stats);
    }
    const auto t3 = clock::now();
    if (!ok)
      throw std::runtime_error("BatchGetFlat failed");
    auto ns = [](clock::duration d) {
      return static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    };
    py::dict ph;
    ph["arg_convert_ns"]  = ns(t1 - t0);
    ph["output_alloc_ns"] = ns(t2 - t1);
    ph["index_lookup_ns"] = static_cast<double>(stats.index_lookup_ns);
    ph["row_copy_ns"]     = static_cast<double>(stats.row_copy_ns);
    ph["zero_fill_ns"]    = static_cast<double>(stats.zero_fill_ns);
    ph["missing_rows"]    = static_cast<double>(stats.missing_rows);
    ph["total_ns"]        = ns(t3 - t0);
    return {std::move(out), std::move(ph)};
  }

  // Index probe only (no value copy): isolates PetHash lookup cost.
  py::dict IndexOnly(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids) {
    const auto keys = CheckIds(ids);
    BaseKV::BatchGetFlatStats stats;
    if (keys.size() &&
        !kv->BatchGetIndexOnly(base::ConstArray<uint64_t>(keys), 0, &stats)) {
      throw std::runtime_error("BatchGetIndexOnly failed");
    }
    py::dict ph;
    ph["missing_rows"] = static_cast<double>(stats.missing_rows);
    ph["num_keys"]     = static_cast<double>(keys.size());
    return ph;
  }

  // Lookup into a caller-provided [num_ids, dim] float32 buffer.
  // Lets the caller reuse one buffer across calls, skipping both the
  // per-call allocation and the pre-memset (ReadFlatFixedRows fills every
  // row, zero-filling misses itself).
  void LookupInto(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      py::array_t<float, py::array::c_style | py::array::forcecast> out) {
    const auto keys = CheckIds(ids);
    auto v          = out.unchecked<2>();
    if (v.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        v.shape(1) != static_cast<py::ssize_t>(dim)) {
      throw std::invalid_argument(
          "lookup_into: out must be [num_ids, dim] float32");
    }
    float* out_ptr = out.mutable_data();
    if (keys.size()) {
      // The engine call touches no Python objects: release the GIL so
      // concurrent lookups from multiple Python threads actually overlap.
      py::gil_scoped_release nogil;
      if (!kv->BatchGetFlat(
              base::ConstArray<uint64_t>(keys),
              out_ptr,
              static_cast<int64_t>(keys.size()),
              dim,
              0)) {
        throw std::runtime_error("BatchGetFlat failed");
      }
    }
  }

  void
  Put(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          values) {
    const auto keys = CheckIds(ids);
    auto vals       = values.unchecked<2>();
    if (vals.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        vals.shape(1) != dim) {
      throw std::invalid_argument("put: values must be [num_ids, dim] float");
    }
    const auto* data = vals.data(0, 0);
    std::vector<base::ConstArray<float>> arrays(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      arrays[i] = base::ConstArray<float>(data + i * dim, dim);
    }
    {
      py::gil_scoped_release nogil;
      kv->BatchPut(base::ConstArray<uint64_t>(keys), &arrays, 0);
    }
  }

  // release_gil=false (serial path): holding the GIL through a sub-millisecond
  // engine call avoids a costly GIL reacquisition after release — in a busy
  // process (e.g. a dataloader thread) the caller can wait several ms behind
  // another thread's GIL slice. release_gil=true (parallel path): worker
  // threads need the release so their engine calls actually overlap.
  void SgdUpdate(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          grads,
      double lr,
      bool release_gil) {
    const auto keys = CheckIds(ids);
    auto g          = grads.unchecked<2>();
    if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        g.shape(1) != dim) {
      throw std::invalid_argument(
          "sgd_update: grads must be [num_ids, dim] float");
    }
    if (release_gil) {
      py::gil_scoped_release nogil;
      if (!kv->ApplySgdUpdateFlat(
              base::ConstArray<uint64_t>(keys),
              g.data(0, 0),
              static_cast<int64_t>(keys.size()),
              dim,
              static_cast<float>(lr),
              0,
              0)) {
        throw std::runtime_error("ApplySgdUpdateFlat failed");
      }
    } else if (!kv->ApplySgdUpdateFlat(
                   base::ConstArray<uint64_t>(keys),
                   g.data(0, 0),
                   static_cast<int64_t>(keys.size()),
                   dim,
                   static_cast<float>(lr),
                   0,
                   0)) {
      throw std::runtime_error("ApplySgdUpdateFlat failed");
    }
  }

  void AdamWUpdate(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          grads,
      double lr,
      double beta1,
      double beta2,
      double eps,
      double weight_decay,
      bool release_gil) {
    const auto keys = CheckIds(ids);
    auto g          = grads.unchecked<2>();
    if (g.shape(0) != static_cast<py::ssize_t>(keys.size()) ||
        g.shape(1) != dim) {
      throw std::invalid_argument(
          "adamw_update: grads must be [num_ids, dim] float");
    }
    if (!(beta1 >= 0.0 && beta1 < 1.0 && beta2 >= 0.0 && beta2 < 1.0) ||
        eps <= 0.0 || lr <= 0.0 || weight_decay < 0.0) {
      throw std::invalid_argument("adamw_update: invalid optimizer parameters");
    }

    const auto* grad_ptr = g.data(0, 0);
    const size_t n       = keys.size();
    std::vector<float> rows(n * static_cast<size_t>(dim));
    std::vector<float> effective(n * static_cast<size_t>(dim));
    // release_gil=false: serial callers avoid the GIL reacquisition stall
    // (see SgdUpdate). release_gil=true: needed for overlap from worker
    // threads. The engine calls and the pure-C++ math below touch no Python
    // objects either way.
    std::unique_ptr<py::gil_scoped_release> nogil;
    if (release_gil)
      nogil = std::make_unique<py::gil_scoped_release>();
    if (n > 0 &&
        !kv->BatchGetFlat(
            base::ConstArray<uint64_t>(keys),
            rows.data(),
            static_cast<int64_t>(n),
            dim,
            0)) {
      throw std::runtime_error("BatchGetFlat failed");
    }

    ++adam_step;
    // Match SparseAdamWTF's float beta accumulators and alpha formula.
    adam_beta1_t *= static_cast<float>(beta1);
    adam_beta2_t *= static_cast<float>(beta2);
    const float alpha = -static_cast<float>(lr) / (1.0f - adam_beta1_t) *
                        std::sqrt(1.0f - adam_beta2_t);
    const double one_minus_weight_decay = 1.0 - weight_decay;
    for (size_t i = 0; i < n; ++i) {
      auto& m = adam_m[keys[i]];
      auto& v = adam_v[keys[i]];
      if (m.empty()) {
        m.assign(static_cast<size_t>(dim), 0.0f);
        v.assign(static_cast<size_t>(dim), 0.0f);
      }
      const float* grad = grad_ptr + i * static_cast<size_t>(dim);
      float* out        = effective.data() + i * static_cast<size_t>(dim);
      const float* row  = rows.data() + i * static_cast<size_t>(dim);
      for (int64_t c = 0; c < dim; ++c) {
        const size_t index     = static_cast<size_t>(c);
        const float grad_value = grad[index];
        m[index] =
            static_cast<float>(beta1 * m[index] + (1.0 - beta1) * grad_value);
        v[index] = static_cast<float>(
            beta2 * v[index] + (1.0 - beta2) * grad_value * grad_value);
        // ApplySgdUpdateFlat computes row -= lr * effective. Encode the
        // native TensorFlow-style AdamW result as that effective gradient:
        // row' = row * (1 - weight_decay) + alpha * m / (sqrt(v) + eps).
        float updated = static_cast<float>(
            one_minus_weight_decay * static_cast<double>(row[index]));
        const float sqrt_v = std::sqrt(v[index]);
        const double adam_term =
            static_cast<double>(alpha) * static_cast<double>(m[index]) /
            (static_cast<double>(sqrt_v) + eps);
        updated = static_cast<float>(static_cast<double>(updated) + adam_term);
        out[index] = static_cast<float>(
            (static_cast<double>(row[index]) - updated) / lr);
      }
    }

    if (n > 0 &&
        !kv->ApplySgdUpdateFlat(
            base::ConstArray<uint64_t>(keys),
            effective.data(),
            static_cast<int64_t>(n),
            dim,
            static_cast<float>(lr),
            0,
            0)) {
      throw std::runtime_error("ApplySgdUpdateFlat failed");
    }
  }
};
void BindLocalKV(py::module_& m) {
  py::class_<LocalKV>(m, "LocalKV")
      .def(py::init<int64_t,
                    int64_t,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    int64_t,
                    int>(),
           py::arg("capacity"),
           py::arg("embedding_dim"),
           py::arg("index_type"),
           py::arg("value_path"),
           py::arg("allocator")          = "CONCURRENT_SLAB_MEMORY_POOL",
           py::arg("allocator_capacity") = 0,
           py::arg("num_threads")        = 1)
      .def("lookup", &LocalKV::Lookup)
      .def("lookup_profiled", &LocalKV::LookupProfiled)
      .def("index_only", &LocalKV::IndexOnly)
      .def("lookup_into", &LocalKV::LookupInto)
      .def("put", &LocalKV::Put)
      .def("sgd_update",
           &LocalKV::SgdUpdate,
           py::arg("ids"),
           py::arg("grads"),
           py::arg("lr"),
           py::arg("release_gil") = true)
      .def("adamw_update",
           &LocalKV::AdamWUpdate,
           py::arg("ids"),
           py::arg("grads"),
           py::arg("lr"),
           py::arg("beta1"),
           py::arg("beta2"),
           py::arg("eps"),
           py::arg("weight_decay"),
           py::arg("release_gil") = true);
}

} // namespace recstore::python
