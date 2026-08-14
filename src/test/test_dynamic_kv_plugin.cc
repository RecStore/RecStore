#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "storage/external/detable/dynamic_kv_plugin.h"

namespace {

using recstore::storage::plugin::DynamicKVPluginApiV1;
using recstore::storage::plugin::DynamicKVPluginConfigV1;

struct TestStore {
  size_t value_size;
  std::mutex mutex;
  std::unordered_map<uint64_t, std::string> rows;
};

void SetError(char* error, size_t error_size, const char* message) {
  if (error == nullptr || error_size == 0) {
    return;
  }
  std::strncpy(error, message, error_size - 1);
  error[error_size - 1] = '\0';
}

template <typename Fn>
int Guard(char* error, size_t error_size, Fn&& fn) {
  try {
    fn();
    return 0;
  } catch (const std::exception& e) {
    SetError(error, error_size, e.what());
  } catch (...) {
    SetError(error, error_size, "unknown test plugin error");
  }
  return -1;
}

int Create(const DynamicKVPluginConfigV1* config,
           void** handle,
           char* error,
           size_t error_size) {
  return Guard(error, error_size, [&] {
    if (config == nullptr || handle == nullptr || config->value_size == 0) {
      throw std::invalid_argument("invalid plugin configuration");
    }
    auto* store = new TestStore{static_cast<size_t>(config->value_size)};
    store->rows.reserve(config->capacity);
    *handle = store;
  });
}

void Destroy(void* handle) { delete static_cast<TestStore*>(handle); }

int Get(void* handle,
        uint64_t key,
        void* value,
        size_t value_size,
        uint8_t* found,
        char* error,
        size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    if (value == nullptr || found == nullptr || value_size != store.value_size) {
      throw std::invalid_argument("invalid get arguments");
    }
    std::lock_guard<std::mutex> lock(store.mutex);
    const auto it = store.rows.find(key);
    *found = it != store.rows.end();
    if (*found) {
      std::memcpy(value, it->second.data(), value_size);
    }
  });
}

int Exists(void* handle,
           uint64_t key,
           uint8_t* found,
           char* error,
           size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    if (found == nullptr) {
      throw std::invalid_argument("invalid exists arguments");
    }
    std::lock_guard<std::mutex> lock(store.mutex);
    *found = store.rows.count(key) != 0;
  });
}

int Put(void* handle,
        uint64_t key,
        const void* value,
        size_t value_size,
        char* error,
        size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    if (value == nullptr || value_size != store.value_size) {
      throw std::invalid_argument("invalid put arguments");
    }
    std::lock_guard<std::mutex> lock(store.mutex);
    store.rows[key].assign(static_cast<const char*>(value), value_size);
  });
}

int BatchGet(void* handle,
             const uint64_t* keys,
             size_t count,
             void* values,
             size_t value_size,
             uint8_t* found,
             char* error,
             size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    if ((count != 0 && (keys == nullptr || values == nullptr || found == nullptr)) ||
        value_size != store.value_size) {
      throw std::invalid_argument("invalid batch_get arguments");
    }
    std::lock_guard<std::mutex> lock(store.mutex);
    auto* output = static_cast<char*>(values);
    for (size_t i = 0; i < count; ++i) {
      const auto it = store.rows.find(keys[i]);
      found[i] = it != store.rows.end();
      if (found[i]) {
        std::memcpy(output + i * value_size, it->second.data(), value_size);
      }
    }
  });
}

int BatchPut(void* handle,
             const uint64_t* keys,
             size_t count,
             const void* values,
             size_t value_size,
             char* error,
             size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    if ((count != 0 && (keys == nullptr || values == nullptr)) ||
        value_size != store.value_size) {
      throw std::invalid_argument("invalid batch_put arguments");
    }
    std::lock_guard<std::mutex> lock(store.mutex);
    const auto* input = static_cast<const char*>(values);
    for (size_t i = 0; i < count; ++i) {
      store.rows[keys[i]].assign(input + i * value_size, value_size);
    }
  });
}

int Clear(void* handle, char* error, size_t error_size) {
  return Guard(error, error_size, [&] {
    auto& store = *static_cast<TestStore*>(handle);
    std::lock_guard<std::mutex> lock(store.mutex);
    store.rows.clear();
  });
}

const DynamicKVPluginApiV1 kApi = {
    recstore::storage::plugin::kDynamicKVPluginAbiV1,
    sizeof(DynamicKVPluginApiV1),
    Create,
    Destroy,
    Get,
    Exists,
    Put,
    BatchGet,
    BatchPut,
    Clear};

} // namespace

extern "C" __attribute__((visibility("default")))
const DynamicKVPluginApiV1* dynamic_kv_plugin_api_v1() {
  return &kApi;
}
