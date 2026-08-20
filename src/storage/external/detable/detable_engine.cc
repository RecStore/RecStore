#include <dlfcn.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "base/factory.h"
#include "storage/external/detable/dynamic_kv_plugin.h"
#include "storage/kv_engine/base_kv.h"

namespace {

using recstore::storage::plugin::DynamicKVPluginApiV1;
using recstore::storage::plugin::DynamicKVPluginConfigV1;

constexpr size_t kErrorBufferSize = 1024;

size_t ConfigValueSize(const BaseKVConfig& config) {
  const auto& j = config.json_config_;
  if (j.contains("value_size")) {
    return j.at("value_size").get<size_t>();
  }
  if (j.contains("value")) {
    return j.at("value").value("default_value_size_hint", 0);
  }
  return 0;
}

const json& PluginConfig(const BaseKVConfig& config) {
  if (!config.json_config_.contains("detable")) {
    throw std::invalid_argument("KVEngineDETable requires detable config");
  }
  return config.json_config_.at("detable");
}

void ValidateFloatAligned(size_t value_size, const char* operation) {
  if (value_size == 0 || value_size % sizeof(float) != 0) {
    throw std::invalid_argument(
        std::string(operation) +
        " requires a non-zero float-aligned value_size");
  }
}

std::runtime_error PluginError(const char* operation,
                               const std::array<char, kErrorBufferSize>& error) {
  const std::string detail = error[0] == '\0' ? "unknown provider error"
                                               : std::string(error.data());
  return std::runtime_error(std::string("KVEngineDETable ") + operation +
                            " failed: " + detail);
}

bool HasRequiredFunctions(const DynamicKVPluginApiV1& api) {
  return api.create != nullptr && api.destroy != nullptr &&
         api.get != nullptr && api.exists != nullptr && api.put != nullptr &&
         api.batch_get != nullptr && api.batch_put != nullptr &&
         api.clear != nullptr;
}

} // namespace

class KVEngineDETable : public BaseKV {
public:
  explicit KVEngineDETable(const BaseKVConfig& config)
      : BaseKV(config), value_size_(ConfigValueSize(config)) {
    if (value_size_ == 0) {
      throw std::invalid_argument("KVEngineDETable requires value_size");
    }

    const auto& plugin = PluginConfig(config);
    const std::string library_path =
        plugin.value("library_path", std::string());
    if (library_path.empty()) {
      throw std::invalid_argument(
          "KVEngineDETable requires detable.library_path");
    }

    library_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      throw std::runtime_error("KVEngineDETable failed to load provider: " +
                               std::string(dlerror()));
    }

    dlerror();
    auto entry = reinterpret_cast<
        recstore::storage::plugin::GetDynamicKVPluginApiV1>(
        dlsym(library_,
              recstore::storage::plugin::kDynamicKVPluginEntryPoint));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || entry == nullptr) {
      CloseLibrary();
      throw std::runtime_error(
          "KVEngineDETable provider entry point is unavailable: " +
          std::string(symbol_error == nullptr ? "unknown error" : symbol_error));
    }

    api_ = entry();
    if (api_ == nullptr ||
        api_->abi_version !=
            recstore::storage::plugin::kDynamicKVPluginAbiV1 ||
        api_->struct_size < sizeof(DynamicKVPluginApiV1) ||
        !HasRequiredFunctions(*api_)) {
      CloseLibrary();
      throw std::runtime_error("KVEngineDETable provider ABI mismatch");
    }

    DynamicKVPluginConfigV1 plugin_config{
        recstore::storage::plugin::kDynamicKVPluginAbiV1,
        sizeof(DynamicKVPluginConfigV1),
        config.json_config_.at("capacity").get<uint64_t>(),
        value_size_,
        plugin.value("block_size", uint64_t{10240})};
    std::array<char, kErrorBufferSize> error{};
    if (api_->create(&plugin_config, &handle_, error.data(), error.size()) != 0 ||
        handle_ == nullptr) {
      const auto exception = PluginError("create", error);
      if (handle_ != nullptr) {
        api_->destroy(handle_);
        handle_ = nullptr;
      }
      CloseLibrary();
      throw exception;
    }
  }

  ~KVEngineDETable() override {
    if (api_ != nullptr && handle_ != nullptr) {
      api_->destroy(handle_);
    }
    CloseLibrary();
  }

  void Get(uint64_t key, std::string& value, unsigned tid) override {
    (void)tid;
    value.resize(value_size_);
    uint8_t found = 0;
    std::array<char, kErrorBufferSize> error{};
    if (api_->get(handle_,
                  key,
                  value.data(),
                  value_size_,
                  &found,
                  error.data(),
                  error.size()) != 0) {
      throw PluginError("get", error);
    }
    if (!found) {
      value.clear();
    }
  }

  bool Exists(uint64_t key, unsigned tid) override {
    (void)tid;
    uint8_t found = 0;
    std::array<char, kErrorBufferSize> error{};
    if (api_->exists(
            handle_, key, &found, error.data(), error.size()) != 0) {
      throw PluginError("exists", error);
    }
    return found != 0;
  }

  void Put(uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    (void)tid;
    if (value.size() != value_size_) {
      throw std::invalid_argument("KVEngineDETable requires fixed-size Put");
    }
    std::array<char, kErrorBufferSize> error{};
    if (api_->put(handle_,
                  key,
                  value.data(),
                  value.size(),
                  error.data(),
                  error.size()) != 0) {
      throw PluginError("put", error);
    }
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "KVEngineDETable::BatchPut");
    if (values == nullptr || keys.Size() != static_cast<int>(values->size())) {
      throw std::invalid_argument("KVEngineDETable::BatchPut size mismatch");
    }
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    for (int i = 0; i < keys.Size(); ++i) {
      if ((*values)[i].Size() != floats_per_row) {
        throw std::invalid_argument("KVEngineDETable::BatchPut row size mismatch");
      }
      std::memcpy(flat.data() + static_cast<size_t>(i) * value_size_,
                  (*values)[i].Data(),
                  value_size_);
    }
    BatchPutRaw(keys, flat.data());
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "KVEngineDETable::BatchGet");
    if (values == nullptr) {
      throw std::invalid_argument("KVEngineDETable::BatchGet values is null");
    }
    thread_local std::vector<std::vector<float>> buffers;
    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    buffers.assign(keys.Size(), std::vector<float>(floats_per_row));
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    std::vector<uint8_t> found(keys.Size(), 0);
    BatchGetRaw(keys, flat.data(), found.data());

    values->resize(keys.Size());
    for (int i = 0; i < keys.Size(); ++i) {
      if (!found[i]) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      std::memcpy(buffers[i].data(),
                  flat.data() + static_cast<size_t>(i) * value_size_,
                  value_size_);
      (*values)[i] =
          base::ConstArray<float>(buffers[i].data(), buffers[i].size());
    }
  }

  bool BatchGetFlat(base::ConstArray<uint64_t> keys,
                    float* values,
                    int64_t num_rows,
                    int64_t embedding_dim,
                    unsigned tid,
                    BatchGetFlatStats* stats = nullptr) override {
    (void)tid;
    if (values == nullptr || num_rows != keys.Size() || num_rows < 0 ||
        embedding_dim <= 0 ||
        static_cast<size_t>(embedding_dim) * sizeof(float) != value_size_) {
      throw std::invalid_argument("KVEngineDETable::BatchGetFlat shape mismatch");
    }
    std::vector<uint8_t> found(keys.Size(), 0);
    BatchGetRaw(keys, values, found.data());
    uint64_t missing = 0;
    for (int i = 0; i < keys.Size(); ++i) {
      if (!found[i]) {
        std::memset(values + static_cast<size_t>(i) * embedding_dim,
                    0,
                    value_size_);
        ++missing;
      }
    }
    if (stats != nullptr) {
      stats->missing_rows += missing;
    }
    return true;
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    if (value == nullptr && keys.Size() > 0) {
      throw std::invalid_argument("KVEngineDETable::BulkLoad value is null");
    }
    BatchPutRaw(keys, value);
  }

  void clear() override {
    std::array<char, kErrorBufferSize> error{};
    if (api_->clear(handle_, error.data(), error.size()) != 0) {
      throw PluginError("clear", error);
    }
  }

private:
  void BatchGetRaw(base::ConstArray<uint64_t> keys,
                   void* values,
                   uint8_t* found) {
    std::array<char, kErrorBufferSize> error{};
    if (api_->batch_get(handle_,
                        keys.Data(),
                        keys.Size(),
                        values,
                        value_size_,
                        found,
                        error.data(),
                        error.size()) != 0) {
      throw PluginError("batch_get", error);
    }
  }

  void BatchPutRaw(base::ConstArray<uint64_t> keys, const void* values) {
    std::array<char, kErrorBufferSize> error{};
    if (api_->batch_put(handle_,
                        keys.Data(),
                        keys.Size(),
                        values,
                        value_size_,
                        error.data(),
                        error.size()) != 0) {
      throw PluginError("batch_put", error);
    }
  }

  void CloseLibrary() {
    if (library_ != nullptr) {
      dlclose(library_);
      library_ = nullptr;
    }
  }

  size_t value_size_ = 0;
  void* library_ = nullptr;
  const DynamicKVPluginApiV1* api_ = nullptr;
  void* handle_ = nullptr;
};

extern "C" void RecStoreForceLinkDETableEngine() {}

FACTORY_REGISTER(BaseKV, KVEngineDETable, KVEngineDETable, const BaseKVConfig&);
