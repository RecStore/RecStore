#include "framework/hugectr/hierkv_backend.h"
#include "framework/hugectr/op_hugectr.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "../third_party/HugeCTR/HugeCTR/include/tensor2.hpp"

class RawPtrBuffer : public HugeCTR::TensorBuffer2 {
private:
  void* ptr_;

public:
  explicit RawPtrBuffer(void* ptr) : ptr_(ptr) {}

  void* get_ptr() override { return ptr_; }

  bool allocated() const override { return ptr_ != nullptr; }
};

void check_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "CUDA Error: " + std::string(cudaGetErrorString(err)));
  }
}

bool cuda_runtime_available() {
  int device_count         = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    std::cerr << "Skipping HierKV CUDA selection test: "
              << cudaGetErrorString(status) << std::endl;
    cudaGetLastError();
    return false;
  }
  if (device_count == 0) {
    std::cerr << "Skipping HierKV CUDA selection test: no CUDA device"
              << std::endl;
    return false;
  }
  return true;
}

struct ScopedConfigOverride {
  explicit ScopedConfigOverride(const std::string& content)
      : temp_dir_(std::filesystem::temp_directory_path() /
                  ("recstore_hugectr_test_" +
                   std::to_string(static_cast<long long>(getpid())))),
        old_path_(std::filesystem::current_path()),
        path_(temp_dir_ / "recstore_config.json") {
    std::filesystem::create_directories(temp_dir_);
    std::filesystem::current_path(temp_dir_);
    std::ifstream input(path_);
    if (input.good()) {
      restore_ = true;
      original_.assign(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
    }
    input.close();

    std::ofstream output(path_, std::ios::trunc);
    output << content;
  }

  ~ScopedConfigOverride() noexcept {
    if (restore_) {
      std::ofstream output(path_, std::ios::trunc);
      output << original_;
    }
    std::error_code ec;
    std::filesystem::current_path(old_path_, ec);
    std::filesystem::remove_all(temp_dir_, ec);
  }

private:
  std::filesystem::path temp_dir_;
  std::filesystem::path old_path_;
  std::filesystem::path path_;
  std::string original_;
  bool restore_{false};
};

static void test_backend_parser() {
  using recstore::framework::HugeCTRBackendKind;
  using recstore::framework::ParseHugeCTRBackendKind;

  assert(ParseHugeCTRBackendKind(json::object()) ==
         HugeCTRBackendKind::RecStore);
  assert(
      ParseHugeCTRBackendKind(json{{"hugectr", {{"backend", "recstore"}}}}) ==
      HugeCTRBackendKind::RecStore);
  assert(ParseHugeCTRBackendKind(json{{"hugectr", {{"backend", "hierkv"}}}}) ==
         HugeCTRBackendKind::HierKV);

  bool invalid_backend_throws = false;
  try {
    (void)ParseHugeCTRBackendKind(json{{"hugectr", {{"backend", "invalid"}}}});
  } catch (const std::invalid_argument&) {
    invalid_backend_throws = true;
  }
  assert(invalid_backend_throws);
}

static void test_hierkv_backend_config_validation() {
  bool missing_block_throws = false;
  try {
    recstore::framework::HugeCTRHierKVBackend backend(json::object());
    (void)backend;
  } catch (const std::invalid_argument&) {
    missing_block_throws = true;
  }
  assert(missing_block_throws);

  const json valid_config = {
      {"hierkv",
       {{"max_capacity", 1024},
        {"max_hbm_for_vectors", 4096},
        {"dim", base::EMBEDDING_DIMENSION_D}}}};
  recstore::framework::HugeCTRHierKVBackend backend(valid_config);

  bool read_throws = false;
  try {
    HugeCTR::Tensor2<long long> dummy_keys(
        {(size_t)1}, std::make_shared<RawPtrBuffer>(nullptr));
    HugeCTR::Tensor2<float> dummy_values(
        {(size_t)1, (size_t)base::EMBEDDING_DIMENSION_D},
        std::make_shared<RawPtrBuffer>(nullptr));
    backend.Read(dummy_keys, dummy_values);
  } catch (const std::runtime_error&) {
    read_throws = true;
  }
  assert(read_throws);

  bool dim_mismatch_throws = false;
  try {
    HugeCTR::Tensor2<long long> dummy_keys(
        {(size_t)1}, std::make_shared<RawPtrBuffer>(nullptr));
    HugeCTR::Tensor2<float> bad_grads(
        {(size_t)1, (size_t)64}, std::make_shared<RawPtrBuffer>(nullptr));
    backend.Update(dummy_keys, bad_grads);
  } catch (const std::invalid_argument&) {
    dim_mismatch_throws = true;
  }
  assert(dim_mismatch_throws);
}

static void maybe_test_hierkv_selection() {
  ScopedConfigOverride cfg(R"JSON(
{
  "cache_ps": {
    "ps_type": "BRPC",
    "num_threads": 32,
    "num_shards": 1,
    "servers": [{"host": "127.0.0.1", "port": 15123, "shard": 0}],
    "base_kv_config": {
      "capacity": 1000,
      "index": {"type": "DRAM_EXTENDIBLE_HASH"},
      "value": {
        "type": "DRAM_VALUE_STORE",
        "path": "/dev/shm/recstore_data/value",
        "default_value_size_hint": 512,
        "dram_allocator": {
          "type": "PERSIST_LOOP_SLAB",
          "capacity_bytes": 512000
        }
      }
    }
  },
  "distributed_client": {
    "num_shards": 1,
    "hash_method": "city_hash",
    "servers": [{"host": "127.0.0.1", "port": 15123, "shard": 0}]
  },
  "client": {"host": "127.0.0.1", "port": 15123, "shard": 0},
  "hugectr": {"backend": "hierkv"},
  "hierkv": {
    "max_capacity": 1024,
    "max_hbm_for_vectors": 4096,
    "dim": 128
  }
}
)JSON");

  const int batch_size = 1;
  void* d_keys_ptr     = nullptr;
  void* d_grads_ptr    = nullptr;

  if (!cuda_runtime_available())
    return;

  try {
    std::vector<long long> h_keys(batch_size, 1);
    std::vector<float> h_grads(batch_size * base::EMBEDDING_DIMENSION_D, 0.1f);

    check_cuda_error(cudaMalloc(&d_keys_ptr, sizeof(long long)));
    check_cuda_error(cudaMalloc(&d_grads_ptr, sizeof(float) * h_grads.size()));
    check_cuda_error(cudaMemcpy(
        d_keys_ptr, h_keys.data(), sizeof(long long), cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(
        d_grads_ptr,
        h_grads.data(),
        sizeof(float) * h_grads.size(),
        cudaMemcpyHostToDevice));

    HugeCTR::Tensor2<long long> d_keys(
        {(size_t)batch_size}, std::make_shared<RawPtrBuffer>(d_keys_ptr));
    HugeCTR::Tensor2<float> d_grads(
        {(size_t)batch_size, (size_t)base::EMBEDDING_DIMENSION_D},
        std::make_shared<RawPtrBuffer>(d_grads_ptr));

    bool throws_not_impl = false;
    try {
      recstore::framework::emb_update_hugectr(d_keys, d_grads);
    } catch (const std::runtime_error& e) {
      throws_not_impl =
          std::string(e.what()).find("not implemented") != std::string::npos;
    }
    assert(throws_not_impl);
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("CUDA Error:") != std::string::npos) {
      if (d_keys_ptr)
        cudaFree(d_keys_ptr);
      if (d_grads_ptr)
        cudaFree(d_grads_ptr);
      std::cerr << "Skipping HierKV CUDA selection test: " << e.what()
                << std::endl;
      return;
    }
    if (d_keys_ptr)
      cudaFree(d_keys_ptr);
    if (d_grads_ptr)
      cudaFree(d_grads_ptr);
    throw;
  } catch (...) {
    if (d_keys_ptr)
      cudaFree(d_keys_ptr);
    if (d_grads_ptr)
      cudaFree(d_grads_ptr);
    throw;
  }

  cudaFree(d_keys_ptr);
  cudaFree(d_grads_ptr);
}

int main() {
  test_backend_parser();
  test_hierkv_backend_config_validation();
  maybe_test_hierkv_selection();
  std::cout << "HugeCTR runtime backend tests passed." << std::endl;
  return 0;
}
