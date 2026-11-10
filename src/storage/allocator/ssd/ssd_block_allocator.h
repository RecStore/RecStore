#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "base/factory.h"
#include "storage/kv_engine/base_kv.h"

class SsdBlockAllocator {
public:
  struct ReadEntry {
    uint64_t handle;
    void* out_buf;
  };

  struct WriteEntry {
    const void* data;
    size_t data_size;
  };

  virtual ~SsdBlockAllocator() = default;

  virtual uint64_t Alloc(size_t data_size)                                = 0;
  virtual void Free(uint64_t handle)                                      = 0;
  virtual void Write(uint64_t handle, const void* data, size_t data_size) = 0;
  virtual size_t Read(uint64_t handle, void* out_buf, size_t buf_size)    = 0;
  virtual uint64_t AllocAndWrite(const void* data, size_t data_size)      = 0;
  virtual size_t SlotCapacity(uint64_t handle) const                      = 0;

  virtual void BatchWrite(const std::vector<uint64_t>& handles,
                          const std::vector<WriteEntry>& entries) {
    if (handles.size() != entries.size())
      throw std::invalid_argument("SSD batch write size mismatch");
    for (size_t i = 0; i < handles.size(); ++i)
      if (handles[i] != kInvalidHandle)
        Write(handles[i], entries[i].data, entries[i].data_size);
  }

  virtual void BatchRead(const std::vector<ReadEntry>& entries,
                         std::vector<size_t>& out_sizes) {
    out_sizes.resize(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      out_sizes[i] = Read(entries[i].handle,
                          entries[i].out_buf,
                          SlotCapacity(entries[i].handle));
    }
  }

  virtual std::vector<uint64_t>
  BatchAllocAndWrite(const std::vector<WriteEntry>& entries) {
    std::vector<uint64_t> handles;
    handles.reserve(entries.size());
    for (const auto& entry : entries) {
      handles.push_back(AllocAndWrite(entry.data, entry.data_size));
    }
    return handles;
  }

  static constexpr uint64_t kInvalidHandle = UINT64_MAX;
};
