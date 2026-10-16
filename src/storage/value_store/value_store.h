#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/index/index.h"

class ValueStore {
public:
  struct ReadResult {
    std::vector<char> data;
  };

  struct WriteSpec {
    const void* data;
    size_t size;
  };

  struct DirectFixedRow {
    const char* data = nullptr;
    size_t size      = 0;
    bool missing     = false;
  };

  virtual ~ValueStore() = default;

  virtual uint64_t Alloc(size_t size)                                  = 0;
  virtual void Write(uint64_t handle, const void* data, size_t size)   = 0;
  virtual uint64_t AllocAndWrite(const void* data, size_t size)        = 0;
  virtual size_t Read(uint64_t handle, void* out_buf, size_t buf_size) = 0;
  virtual void Free(uint64_t handle)                                   = 0;
  virtual void Retire(uint64_t handle) { Free(handle); }
  virtual const char* DirectPtr(uint64_t handle) const { return nullptr; }
  virtual char* RDMABackingData() const { return nullptr; }
  virtual size_t RDMABackingSize() const { return 0; }
  virtual size_t SlotCapacity(uint64_t handle) const = 0;
  virtual void BatchWrite(const std::vector<uint64_t>& handles,
                          const std::vector<WriteSpec>& specs) {
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("BatchWrite size mismatch");
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        continue;
      }
      Write(handles[i], specs[i].data, specs[i].size);
    }
  }

  virtual void BatchRead(const std::vector<uint64_t>& handles,
                         std::vector<ReadResult>& out_results) {
    out_results.resize(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        out_results[i].data.clear();
        continue;
      }
      const size_t cap = SlotCapacity(handles[i]);
      out_results[i].data.resize(cap);
      const size_t actual = Read(
          handles[i], out_results[i].data.data(), out_results[i].data.size());
      out_results[i].data.resize(actual);
    }
  }

  virtual bool ReadFlatFixedRows(
      const uint64_t* handles,
      size_t num_rows,
      void* out_buf,
      size_t row_bytes,
      uint64_t* missing_rows) const {
    (void)handles;
    (void)num_rows;
    (void)out_buf;
    (void)row_bytes;
    (void)missing_rows;
    return false;
  }

  virtual bool ReadFlatFixedRowSlices(
      const uint64_t* handles,
      size_t num_rows,
      void* out_buf,
      size_t stored_row_bytes,
      size_t source_offset_bytes,
      size_t output_row_bytes,
      uint64_t* missing_rows) const {
    (void)handles;
    (void)num_rows;
    (void)out_buf;
    (void)stored_row_bytes;
    (void)source_offset_bytes;
    (void)output_row_bytes;
    (void)missing_rows;
    return false;
  }

  virtual bool GetDirectFixedRows(
      const uint64_t* handles,
      size_t num_rows,
      size_t row_bytes,
      DirectFixedRow* rows,
      uint64_t* missing_rows) const {
    (void)handles;
    (void)num_rows;
    (void)row_bytes;
    (void)rows;
    (void)missing_rows;
    return false;
  }

  virtual std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteSpec>& specs) {
    std::vector<uint64_t> handles;
    handles.reserve(specs.size());
    for (const auto& spec : specs) {
      handles.push_back(AllocAndWrite(spec.data, spec.size));
    }
    return handles;
  }

  virtual std::string GetInfo() const { return "(no info)"; }
  virtual std::string ExtraResultFields() const { return ""; }
};
