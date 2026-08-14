#pragma once

#include <cstddef>
#include <cstdint>

namespace recstore::storage::plugin {

constexpr uint32_t kDynamicKVPluginAbiV1 = 1;
constexpr const char* kDynamicKVPluginEntryPoint =
    "dynamic_kv_plugin_api_v1";

struct DynamicKVPluginConfigV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t capacity;
  uint64_t value_size;
  uint64_t block_size;
};

struct DynamicKVPluginApiV1 {
  uint32_t abi_version;
  uint32_t struct_size;

  int (*create)(const DynamicKVPluginConfigV1* config,
                void** handle,
                char* error,
                size_t error_size);
  void (*destroy)(void* handle);
  int (*get)(void* handle,
             uint64_t key,
             void* value,
             size_t value_size,
             uint8_t* found,
             char* error,
             size_t error_size);
  int (*exists)(void* handle,
                uint64_t key,
                uint8_t* found,
                char* error,
                size_t error_size);
  int (*put)(void* handle,
             uint64_t key,
             const void* value,
             size_t value_size,
             char* error,
             size_t error_size);
  int (*batch_get)(void* handle,
                   const uint64_t* keys,
                   size_t count,
                   void* values,
                   size_t value_size,
                   uint8_t* found,
                   char* error,
                   size_t error_size);
  int (*batch_put)(void* handle,
                   const uint64_t* keys,
                   size_t count,
                   const void* values,
                   size_t value_size,
                   char* error,
                   size_t error_size);
  int (*clear)(void* handle, char* error, size_t error_size);
};

using GetDynamicKVPluginApiV1 = const DynamicKVPluginApiV1* (*)();

} // namespace recstore::storage::plugin
