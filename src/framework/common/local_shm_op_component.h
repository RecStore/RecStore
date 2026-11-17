#pragma once

#include <string>

#include "base/tensor.h"
#include "ps/base/base_client.h"
#include "ps/local_shm/local_shm_client.h"

namespace recstore {

LocalShmPSClient* GetLocalShmClientOrThrow(BasePSClient* client,
                                           const std::string& backend_name,
                                           const char* api_name);

void LocalShmLookupFlat(BasePSClient* client,
                        const std::string& backend_name,
                        const base::RecTensor& keys,
                        base::RecTensor& values);

int SubmitLocalShmLookupFlat(
    BasePSClient* client,
    const std::string& backend_name,
    const base::RecTensor& keys,
    int64_t embedding_dim,
    LocalShmFlatGetHandle* handle);

int WaitLocalShmLookupFlat(BasePSClient* client,
                           const std::string& backend_name,
                           LocalShmFlatGetHandle* handle);

void ReleaseLocalShmLookupFlat(BasePSClient* client,
                               const std::string& backend_name,
                               LocalShmFlatGetHandle* handle);

void LocalShmUpdateFlat(
    BasePSClient* client,
    const std::string& backend_name,
    const std::string& table_name,
    const base::RecTensor& keys,
    const base::RecTensor& grads,
    float learning_rate);

} // namespace recstore
