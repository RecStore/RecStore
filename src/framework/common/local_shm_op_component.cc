#include "framework/common/local_shm_op_component.h"

#include <stdexcept>

namespace recstore {

namespace {

void ValidateKeys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " +
                                base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional, but has " +
                                std::to_string(keys.dim()) + " dimensions.");
  }
}

void ValidateEmbeddings(const base::RecTensor& embeddings,
                        const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " +
        base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " +
        std::to_string(embeddings.dim()) + " dimensions.");
  }
}

} // namespace

LocalShmPSClient* GetLocalShmClientOrThrow(BasePSClient* client,
                                           const std::string& backend_name,
                                           const char* api_name) {
  if (backend_name != "local_shm") {
    throw std::runtime_error(
        std::string(api_name) +
        " requires local_shm backend, but current backend is " + backend_name);
  }
  auto* local_client = dynamic_cast<LocalShmPSClient*>(client);
  if (local_client == nullptr) {
    throw std::runtime_error(
        std::string(api_name) + " requires LocalShmPSClient backend instance.");
  }
  return local_client;
}

void LocalShmLookupFlat(BasePSClient* client,
                        const std::string& backend_name,
                        const base::RecTensor& keys,
                        base::RecTensor& values) {
  ValidateKeys(keys);
  ValidateEmbeddings(values, "Values");

  const int64_t num_rows = keys.shape(0);
  if (values.shape(0) != num_rows) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(num_rows) +
        " but values has length " + std::to_string(values.shape(0)));
  }

  auto* local_client =
      GetLocalShmClientOrThrow(client, backend_name, "local_lookup_flat");
  base::ConstArray<uint64_t> key_array(keys.data_as<uint64_t>(), num_rows);
  const int64_t embedding_dim = values.shape(1);
  const int ret               = local_client->GetParameterFlat(
      key_array, values.data_as<float>(), num_rows, embedding_dim);
  if (ret != 0) {
    throw std::runtime_error(
        "Failed to read embeddings from local_shm PS client.");
  }
}

int SubmitLocalShmLookupFlat(
    BasePSClient* client,
    const std::string& backend_name,
    const base::RecTensor& keys,
    int64_t embedding_dim,
    LocalShmFlatGetHandle* handle) {
  ValidateKeys(keys);
  const int64_t num_rows = keys.shape(0);
  base::ConstArray<uint64_t> key_array(keys.data_as<uint64_t>(), num_rows);
  auto* local_client =
      GetLocalShmClientOrThrow(client, backend_name, "local_lookup_flat");
  return local_client->SubmitGetParameterFlat(
      key_array, num_rows, embedding_dim, handle);
}

int WaitLocalShmLookupFlat(BasePSClient* client,
                           const std::string& backend_name,
                           LocalShmFlatGetHandle* handle) {
  auto* local_client =
      GetLocalShmClientOrThrow(client, backend_name, "local_lookup_flat");
  return local_client->WaitGetParameterFlat(handle);
}

void ReleaseLocalShmLookupFlat(BasePSClient* client,
                               const std::string& backend_name,
                               LocalShmFlatGetHandle* handle) {
  auto* local_client =
      GetLocalShmClientOrThrow(client, backend_name, "local_lookup_flat");
  local_client->ReleaseGetParameterFlat(handle);
}

void LocalShmUpdateFlat(BasePSClient* client,
                        const std::string& backend_name,
                        const std::string& table_name,
                        const base::RecTensor& keys,
                        const base::RecTensor& grads) {
  ValidateKeys(keys);
  ValidateEmbeddings(grads, "Grads");

  const int64_t num_rows = keys.shape(0);
  if (grads.shape(0) != num_rows) {
    throw std::invalid_argument(
        "Dimension mismatch: Keys has length " + std::to_string(num_rows) +
        " but grads has length " + std::to_string(grads.shape(0)));
  }

  const int64_t embedding_dim = grads.shape(1);
  if (embedding_dim <= 0) {
    throw std::invalid_argument(
        "Invalid grad dimension D: " + std::to_string(embedding_dim));
  }

  auto* local_client =
      GetLocalShmClientOrThrow(client, backend_name, "local_update_flat");
  base::ConstArray<uint64_t> key_array(keys.data_as<uint64_t>(), num_rows);
  const int ret = local_client->UpdateParameter(table_name, key_array, grads);
  if (ret != 0) {
    throw std::runtime_error("Failed to update embeddings via PS client.");
  }
}

} // namespace recstore
