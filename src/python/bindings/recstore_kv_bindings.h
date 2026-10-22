#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/array.h"
#include "storage/kv_engine/base_kv.h"

namespace recstore::python {

namespace py = pybind11;

std::vector<uint64_t> CheckIds(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids);

struct Int64KeyView {
  const uint64_t* values;
  size_t count;

  const uint64_t* data() const { return values; }
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  uint64_t operator[](size_t index) const { return values[index]; }
  base::ConstArray<uint64_t> AsConstArray() const;
};

Int64KeyView CheckIdsView(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& ids);

std::unique_ptr<BaseKV> MakeLocalKV(
    int64_t capacity,
    int64_t row_bytes,
    const std::string& index_type,
    const std::string& value_dir,
    const std::string& allocator,
    int64_t allocator_capacity,
    int num_threads,
    int index_prefetch = 1);

void BindLocalKV(py::module_& module);
void BindLocalIndexedKV(py::module_& module);

} // namespace recstore::python
