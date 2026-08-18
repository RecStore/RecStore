#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "base/tensor.h"
#include "framework/common/op_runtime_support.h"

namespace recstore {

constexpr const char* kHierKVBackendName = "hierkv";

bool IsHierKVBackendName(const std::string& backend_name);

class HierKVLocalRuntime {
public:
  bool InitEmbeddingTable(const std::string& table_name,
                          const EmbeddingTableConfig& config);
  void Write(const base::RecTensor& keys, const base::RecTensor& values);
  void Read(const base::RecTensor& keys, base::RecTensor& values);
  void Update(const std::string& table_name,
              const base::RecTensor& keys,
              const base::RecTensor& grads);
  uint64_t Prefetch(const base::RecTensor& keys, int64_t embedding_dim);
  bool IsPrefetchDone(uint64_t prefetch_id);
  void WaitForPrefetch(uint64_t prefetch_id);
  void ConsumePrefetch(uint64_t prefetch_id, base::RecTensor& values);

private:
  struct Impl;
  Impl& impl();
};

HierKVLocalRuntime& GetHierKVLocalRuntime();

} // namespace recstore
