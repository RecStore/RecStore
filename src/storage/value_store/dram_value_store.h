#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <glog/logging.h>

#include "base/factory.h"
#include "memory/malloc.h"
#include "memory/memory_factory.h"
#include "storage/value_store/value_store.h"

class DramValueStore : public ValueStore {
public:
  explicit DramValueStore(const BaseKVConfig& config) {
    const auto& j = config.json_config_;
    if (!j.contains("value") || !j.at("value").contains("dram_allocator")) {
      throw std::invalid_argument(
          "DramValueStore requires value.dram_allocator");
    }
    const auto& value = j.at("value");
    if (!value.contains("path") ||
        value.at("path").get<std::string>().empty()) {
      throw std::invalid_argument(
          "DramValueStore requires non-empty value.path");
    }
    const std::string path           = value.at("path").get<std::string>();
    const auto& dram                 = value.at("dram_allocator");
    const std::string allocator_type = dram.value("type", "R2_SLAB");
    const uint64_t capacity_bytes = dram.at("capacity_bytes").get<uint64_t>();
    std::vector<int> size_classes;
    if (dram.contains("size_classes")) {
      size_classes = dram.at("size_classes").get<std::vector<int>>();
      for (int slab_size : size_classes) {
        if (slab_size <= 0) {
          throw std::invalid_argument(
              "DramValueStore requires positive dram_allocator.size_classes");
        }
      }
    }

    using MF = base::
        Factory<base::MallocApi, const std::string&, int64, const std::string&>;
    using MFS =
        base::Factory<base::MallocApi,
                      const std::string&,
                      int64,
                      const std::string&,
                      const std::vector<int>&>;
    if (!size_classes.empty()) {
      // Only allocators that registered the sized factory accept an explicit
      // size-class list; everything else keeps the fixed-class constructor.
      if (MFS::creators().count(allocator_type) == 0) {
        throw std::invalid_argument(
            "allocator does not support size_classes: " + allocator_type);
      }
      allocator_.reset(MFS::NewInstance(
          allocator_type,
          path,
          static_cast<int64>(capacity_bytes),
          "DRAM",
          size_classes));
    } else {
      allocator_.reset(MF::NewInstance(
          allocator_type, path, static_cast<int64>(capacity_bytes), "DRAM"));
    }
    if (!allocator_) {
      throw std::runtime_error("failed to create DramValueStore allocator");
    }
    allocator_->Initialize();
    recycler_ = std::make_unique<base::ThreadSafeDelayedRecycle>(
        allocator_.get(), kRecycleDelayUs);
  }

  uint64_t Alloc(size_t size) override {
    char* data = allocator_->New(static_cast<int>(size));
    if (data == nullptr) {
      return kValueHandleNone;
    }
    return EncodeOffset(allocator_->GetMallocOffset(data));
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    char* dst = Ptr(handle);
    if (dst == nullptr) {
      return;
    }
    std::memcpy(dst, data, size);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    char* src = Ptr(handle);
    if (src == nullptr || out_buf == nullptr) {
      return 0;
    }
    const size_t n = std::min(
        buf_size,
        static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle))));
    std::memcpy(out_buf, src, n);
    return n;
  }

  void Free(uint64_t handle) override {
    char* data = Ptr(handle);
    if (data != nullptr) {
      allocator_->Free(data);
    }
  }

  void Retire(uint64_t handle) override {
    if (handle == kValueHandleNone) {
      return;
    }
    recycler_->Recycle(allocator_->GetMallocData(DecodeOffset(handle)));
  }

  const char* DirectPtr(uint64_t handle) const override {
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  char* RDMABackingData() const override { return allocator_->BackingData(); }

  size_t RDMABackingSize() const override {
    return static_cast<size_t>(allocator_->BackingSize());
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    return static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle)));
  }

  bool ReadFlatFixedRows(const uint64_t* handles,
                         size_t num_rows,
                         void* out_buf,
                         size_t row_bytes,
                         uint64_t* missing_rows) const override {
    if (handles == nullptr || out_buf == nullptr || row_bytes == 0) {
      return false;
    }
    if (!IsPowerOfTwo(row_bytes)) {
      return ReadFlatFixedRowsGeneric(
          handles, num_rows, out_buf, row_bytes, missing_rows);
    }
    uint64_t local_missing = 0;
    char* dst              = static_cast<char*>(out_buf);
    const unsigned row_shift = Log2PowerOfTwo(row_bytes);
    for (size_t row = 0; row < num_rows; ++row) {
      char* row_dst = dst + (row << row_shift);
      if (handles[row] == kValueHandleNone) {
        std::memset(row_dst, 0, row_bytes);
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      std::memcpy(row_dst, src, row_bytes);
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  bool ReadFlatFixedRowSlices(const uint64_t* handles,
                              size_t num_rows,
                              void* out_buf,
                              size_t stored_row_bytes,
                              size_t source_offset_bytes,
                              size_t output_row_bytes,
                              uint64_t* missing_rows) const override {
    if (handles == nullptr || out_buf == nullptr || output_row_bytes == 0 ||
        source_offset_bytes > stored_row_bytes ||
        output_row_bytes > stored_row_bytes - source_offset_bytes) {
      return false;
    }
    if (!IsPowerOfTwo(output_row_bytes)) {
      return ReadFlatFixedRowSlicesGeneric(
          handles, num_rows, out_buf, source_offset_bytes, output_row_bytes,
          missing_rows);
    }
    uint64_t local_missing = 0;
    char* dst              = static_cast<char*>(out_buf);
    const unsigned row_shift = Log2PowerOfTwo(output_row_bytes);
    for (size_t row = 0; row < num_rows; ++row) {
      char* row_dst = dst + (row << row_shift);
      if (handles[row] == kValueHandleNone) {
        std::memset(row_dst, 0, output_row_bytes);
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      std::memcpy(row_dst, src + source_offset_bytes, output_row_bytes);
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  bool GetDirectFixedRows(const uint64_t* handles,
                          size_t num_rows,
                          size_t row_bytes,
                          DirectFixedRow* rows,
                          uint64_t* missing_rows) const override {
    if (handles == nullptr || rows == nullptr || row_bytes == 0) {
      return false;
    }
    uint64_t local_missing = 0;
    for (size_t row = 0; row < num_rows; ++row) {
      if (handles[row] == kValueHandleNone) {
        rows[row] = DirectFixedRow{nullptr, row_bytes, true};
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      if (SlotCapacity(handles[row]) < row_bytes) {
        return false;
      }
      rows[row] = DirectFixedRow{src, row_bytes, false};
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  std::string GetInfo() const override { return allocator_->GetInfo(); }
  uint64_t TotalAllocCount() const { return allocator_->total_malloc(); }

private:
  [[gnu::noinline]] bool ReadFlatFixedRowsGeneric(
      const uint64_t* handles,
      size_t num_rows,
      void* out_buf,
      size_t row_bytes,
      uint64_t* missing_rows) const {
    uint64_t local_missing = 0;
    char* dst              = static_cast<char*>(out_buf);
    for (size_t row = 0; row < num_rows; ++row) {
      char* row_dst = dst + row * row_bytes;
      if (handles[row] == kValueHandleNone) {
        std::memset(row_dst, 0, row_bytes);
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      std::memcpy(row_dst, src, row_bytes);
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  [[gnu::noinline]] bool ReadFlatFixedRowSlicesGeneric(
      const uint64_t* handles,
      size_t num_rows,
      void* out_buf,
      size_t source_offset_bytes,
      size_t output_row_bytes,
      uint64_t* missing_rows) const {
    uint64_t local_missing = 0;
    char* dst              = static_cast<char*>(out_buf);
    for (size_t row = 0; row < num_rows; ++row) {
      char* row_dst = dst + row * output_row_bytes;
      if (handles[row] == kValueHandleNone) {
        std::memset(row_dst, 0, output_row_bytes);
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      std::memcpy(row_dst, src + source_offset_bytes, output_row_bytes);
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  static uint64_t EncodeOffset(int64 offset) {
    return static_cast<uint64_t>(offset) + 1;
  }

  static int64 DecodeOffset(uint64_t handle) {
    return static_cast<int64>(handle - 1);
  }

  char* Ptr(uint64_t handle) const {
    if (handle == kValueHandleNone) {
      return nullptr;
    }
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  static bool IsPowerOfTwo(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
  }

  static unsigned Log2PowerOfTwo(size_t value) {
    return static_cast<unsigned>(
        __builtin_ctzll(static_cast<unsigned long long>(value)));
  }

  std::unique_ptr<base::MallocApi> allocator_;
  std::unique_ptr<base::ThreadSafeDelayedRecycle> recycler_;
  static constexpr int64 kRecycleDelayUs = 1000;
};

FACTORY_REGISTER(
    ValueStore, DRAM_VALUE_STORE, DramValueStore, const BaseKVConfig&);
