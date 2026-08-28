#pragma once

#ifdef USE_TORCH
#  include <torch/torch.h>
namespace c10 {
using float16_t = c10::Half;
}
#endif

#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace base {

const int64_t EMBEDDING_DIMENSION_D = 128;

enum class DataType {
  UNKNOWN,
  UINT64,
  FLOAT32,
  FLOAT16,
  INT32,
  INT16,
  INT8,
};

inline std::string DataTypeToString(DataType dtype) {
  switch (dtype) {
  case DataType::UINT64:
    return "UINT64";
  case DataType::FLOAT32:
    return "FLOAT32";
  case DataType::FLOAT16:
    return "FLOAT16";
  case DataType::INT32:
    return "INT32";
  case DataType::INT16:
    return "INT16";
  case DataType::INT8:
    return "INT8";
  default:
    return "UNKNOWN";
  }
}

inline size_t DataTypeSize(DataType dtype) {
  switch (dtype) {
  case DataType::UINT64:
    return sizeof(uint64_t);
  case DataType::FLOAT32:
    return sizeof(float);
  case DataType::FLOAT16:
    return 2;
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT16:
    return sizeof(int16_t);
  case DataType::INT8:
    return sizeof(int8_t);
  default:
    throw std::runtime_error("Unsupported DataType for DataTypeSize.");
  }
}

class RecTensor {
public:
  RecTensor() = default;

  // Borrowed view: destructor does not free data_ptr.
  template <typename T>
  RecTensor(T* data, const std::vector<int64_t>& shape)
      : data_ptr_(static_cast<void*>(data)), shape_(shape) {
    if (std::is_same<T, uint64_t>::value) {
      dtype_ = DataType::UINT64;
    } else if (std::is_same<T, float>::value) {
      dtype_ = DataType::FLOAT32;
#ifdef USE_TORCH
    } else if (std::is_same<T, c10::float16_t>::value) {
      dtype_ = DataType::FLOAT16;
#endif
    } else if (std::is_same<T, int32_t>::value) {
      dtype_ = DataType::INT32;
    } else if (std::is_same<T, int16_t>::value) {
      dtype_ = DataType::INT16;
    } else if (std::is_same<T, int8_t>::value) {
      dtype_ = DataType::INT8;
    } else {
      dtype_ = DataType::UNKNOWN;
      throw std::runtime_error("Unsupported type for RecTensor constructor.");
    }
    recalculate_num_elements();
  }

  RecTensor(void* data, const std::vector<int64_t>& shape, DataType dtype)
      : data_ptr_(data), shape_(shape), dtype_(dtype) {
    recalculate_num_elements();
  }

  // Owned buffer: allocates num_elements * DataTypeSize(dtype) bytes.
  RecTensor(const std::vector<int64_t>& shape, DataType dtype)
      : shape_(shape), dtype_(dtype), owns_data_(true) {
    recalculate_num_elements();
    const size_t nbytes = num_elements_ * DataTypeSize(dtype);
    data_ptr_           = nbytes == 0 ? nullptr : new char[nbytes];
  }

  RecTensor(const RecTensor& other)
      : data_ptr_(other.data_ptr_),
        shape_(other.shape_),
        dtype_(other.dtype_),
        num_elements_(other.num_elements_),
        owns_data_(false) {}

  RecTensor(RecTensor&& other) noexcept
      : data_ptr_(other.data_ptr_),
        shape_(std::move(other.shape_)),
        dtype_(other.dtype_),
        num_elements_(other.num_elements_),
        owns_data_(other.owns_data_) {
    other.data_ptr_     = nullptr;
    other.owns_data_    = false;
    other.num_elements_ = 0;
    other.dtype_        = DataType::UNKNOWN;
  }

  RecTensor& operator=(const RecTensor& other) {
    if (this == &other) {
      return *this;
    }
    release();
    data_ptr_     = other.data_ptr_;
    shape_        = other.shape_;
    dtype_        = other.dtype_;
    num_elements_ = other.num_elements_;
    owns_data_    = false;
    return *this;
  }

  RecTensor& operator=(RecTensor&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release();
    data_ptr_           = other.data_ptr_;
    shape_              = std::move(other.shape_);
    dtype_              = other.dtype_;
    num_elements_       = other.num_elements_;
    owns_data_          = other.owns_data_;
    other.data_ptr_     = nullptr;
    other.owns_data_    = false;
    other.num_elements_ = 0;
    other.dtype_        = DataType::UNKNOWN;
    return *this;
  }

  ~RecTensor() { release(); }

  void* data() const { return data_ptr_; }

  template <typename T>
  T* data_as() const {
    if (dtype_ == DataType::UINT64 && !std::is_same<T, uint64_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is UINT64, accessed as different type.");
    }
    if (dtype_ == DataType::FLOAT32 && !std::is_same<T, float>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is FLOAT32, accessed as different type.");
    }
#ifdef USE_TORCH
    if (dtype_ == DataType::FLOAT16 &&
        !std::is_same<T, c10::float16_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is FLOAT16, accessed as different type.");
    }
#endif
    if (dtype_ == DataType::INT32 && !std::is_same<T, int32_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT32, accessed as different type.");
    }
    if (dtype_ == DataType::INT16 && !std::is_same<T, int16_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT16, accessed as different type.");
    }
    if (dtype_ == DataType::INT8 && !std::is_same<T, int8_t>::value) {
      throw std::runtime_error(
          "Type mismatch: Tensor is INT8, accessed as different type.");
    }
    return static_cast<T*>(data_ptr_);
  }

  const std::vector<int64_t>& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  size_t num_elements() const { return num_elements_; }
  size_t dim() const { return shape_.size(); }
  bool owns() const { return owns_data_; }
  int64_t shape(size_t i) const {
    if (i >= shape_.size())
      throw std::out_of_range("Shape index out of range.");
    return shape_[i];
  }

  void set_data(void* data) {
    release();
    data_ptr_ = data;
  }
  void set_shape(const std::vector<int64_t>& new_shape) {
    shape_ = new_shape;
    recalculate_num_elements();
  }
  void set_dtype(DataType new_dtype) { dtype_ = new_dtype; }

private:
  void release() {
    if (owns_data_) {
      delete[] static_cast<char*>(data_ptr_);
    }
    data_ptr_  = nullptr;
    owns_data_ = false;
  }

  void recalculate_num_elements() {
    if (shape_.empty()) {
      num_elements_ = 0;
    } else {
      for (long long dim_size : shape_) {
        if (dim_size < 0)
          throw std::runtime_error("Tensor dimension size cannot be negative.");
      }
      num_elements_ = std::accumulate(
          shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
    }
  }

  void* data_ptr_         = nullptr;
  std::vector<int64_t> shape_;
  DataType dtype_         = DataType::UNKNOWN;
  size_t num_elements_    = 0;
  bool owns_data_         = false;
};

} // namespace base
