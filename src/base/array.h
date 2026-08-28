#pragma once
#include <google/protobuf/repeated_field.h>

#include <string>
#include <vector>

namespace base {

template <typename T>
struct ConstArray {
  const T* list = nullptr;
  int size      = 0;

  ConstArray(const T* list, int size) : list(list), size(size) {}

  ConstArray(const std::vector<T>& vector)
      : list(vector.data()), size(vector.size()) {} // NOLINT

  ConstArray(const std::string& binary_data) // NOLINT
      : list(reinterpret_cast<const T*>(binary_data.data())),
        size(binary_data.size() / sizeof(T)) {}

  ConstArray(const google::protobuf::RepeatedField<T>& repeated_field) // NOLINT
      : list(repeated_field.data()), size(repeated_field.size()) {}

  ConstArray() : list(nullptr), size(0) {}

  // [start, end)
  ConstArray<T> SubArray(int start, int end) const {
    return ConstArray<T>(list + start, end - start);
  }

  void SetData(const void* data, int data_size) {
    if (data_size % sizeof(T) == 0) {
      list = reinterpret_cast<const T*>(data);
      size = data_size / sizeof(T);
    } else {
      list = nullptr;
      size = 0;
    }
  }

  void SetData(std::vector<T>& vector) {
    list = vector.data();
    size = vector.size();
  }

  const T& operator[](int n) const { return list[n]; }

  const T* Data() const { return list; }

  void SetData(const std::string& data) { SetData(data.data(), data.size()); }

  void AppendToVector(std::vector<T>* vector) const {
    for (int i = 0; i < size; ++i)
      vector->push_back(list[i]);
  }
  void CopyToVector(std::vector<T>* vector) const {
    vector->clear();
    AppendToVector(vector);
  }

  std::vector<T> ToVector() const {
    std::vector<T> vector;
    vector.reserve(size);
    AppendToVector(&vector);
    return vector;
  }

  void Set(const T* list_head, int list_size) {
    list = list_head;
    size = list_size;
  }

  auto begin() const { return list; }
  auto end() const { return list + size; }

  const T& front() const { return list[0]; }

  const T& back() const { return list[size - 1]; }

  int Size() const { return size; }

  const char* binary_data() const {
    return reinterpret_cast<const char*>(list);
  }
  int64_t binary_size() const { return sizeof(T) * size; }
  std::string as_string() const {
    return (list != nullptr) ? std::string(binary_data(), binary_size()) : "";
  }
  std::string Debug() const {
    std::string temp = "";
    for (int i = 0; i < size; i++)
      temp += std::to_string(list[i]) + ",";
    temp += "\n";
    return temp;
  }
};

template <typename T>
struct MutableArray {
  T* list  = nullptr;
  int size = 0;

  MutableArray(T* list, int size) : list(list), size(size) {}

  MutableArray(std::vector<T>& vector)
      : list(vector.data()), size(vector.size()) {} // NOLINT

  MutableArray(std::string& binary_data) // NOLINT
      : list(reinterpret_cast<T*>(binary_data.data())),
        size(binary_data.size() / sizeof(T)) {}

  MutableArray() : list(nullptr), size(0) {}

  T* Data() { return list; }

  void SetData(char* data, int data_size) {
    if (data_size % sizeof(T) == 0) {
      list = reinterpret_cast<T*>(data);
      size = data_size / sizeof(T);
    } else {
      *(int*)0 = 0;
      list     = nullptr;
      size     = 0;
    }
  }

  void SetData(std::vector<T>& vector) {
    list = vector.data();
    size = vector.size();
  }

  void SetData(std::string& data) { SetData(data.data(), data.size()); }

  void AppendToVector(std::vector<T>* vector) const {
    for (int i = 0; i < size; ++i)
      vector->push_back(list[i]);
  }
  void CopyToVector(std::vector<T>* vector) const {
    vector->clear();
    AppendToVector(vector);
  }

  void Set(T* list_head, int list_size) {
    list = list_head;
    size = list_size;
  }

  T& operator[](int n) { return list[n]; }

  const T& operator[](int n) const { return list[n]; }

  ConstArray<T> ToConstArray() { return ConstArray<T>(list, size); }

  auto begin() const { return list; }
  auto end() const { return list + size; }

  int Size() const { return size; }

  char* binary_data() const { return reinterpret_cast<char*>(list); }
  int64_t binary_size() const { return sizeof(T) * size; }
  std::string as_string() const {
    return (list != nullptr) ? std::string(binary_data(), binary_size()) : "";
  }
};
} // namespace base