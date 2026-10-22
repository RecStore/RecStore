#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "recstore_kv_bindings.h"

namespace recstore::python {

// In-process embedding store with stable row indices and optimizer slots.
struct LocalIndexedKV {
  static constexpr int64_t kAdamSlotBlockRows = 65536;
  struct AdamSlotBlock {
    explicit AdamSlotBlock(int64_t dim)
        : exp_avg(static_cast<size_t>(kAdamSlotBlockRows) *
                      static_cast<size_t>(dim),
                  0.0f),
          exp_avg_sq(static_cast<size_t>(kAdamSlotBlockRows) *
                         static_cast<size_t>(dim),
                     0.0f) {}

    std::vector<float> exp_avg;
    std::vector<float> exp_avg_sq;
  };
  struct AdamUpdateStats {
    uint64_t index_lookup_ns     = 0;
    uint64_t index_lookup_max_ns = 0;
    uint64_t engine_get_ns       = 0;
    uint64_t slot_alloc_ns       = 0;
    uint64_t math_ns             = 0;
  };
  struct RestoredIndexedState {
    int64_t next_row_index = 0;
    uint64_t active_rows   = 0;
    uint64_t adam_step     = 0;
    float adam_beta1_t     = 1.0f;
    float adam_beta2_t     = 1.0f;
    int64_t slot_rows      = 0;
    std::vector<int64_t> free_rows;
    std::vector<float> exp_avg;
    std::vector<float> exp_avg_sq;
  };
  std::unique_ptr<BaseKV> kv;
  int64_t capacity;
  int64_t dim;
  int64_t storage_dim;
  int num_threads;
  int64_t next_row_index = 0;
  std::vector<int64_t> free_row_indices;
  uint64_t adam_step = 0;
  float adam_beta1_t = 1.0f;
  float adam_beta2_t = 1.0f;
  std::vector<std::unique_ptr<AdamSlotBlock>> adam_slot_blocks;
  std::vector<std::vector<BaseKV::DirectFixedRow>> worker_rows_scratch;
  std::vector<int64_t> row_indices_scratch;
  std::mutex mu;
  LocalIndexedKV(int64_t capacity_arg,
                 int64_t embedding_dim,
                 const std::string& index_type,
                 const std::string& value_dir,
                 const std::string& allocator,
                 int64_t allocator_capacity,
                 int num_threads,
                 int index_prefetch);
  int EffectiveWorkers(size_t rows) const;
  int64_t AllocateRowIndex();
  void ClearAdamSlotRow(int64_t row_index);
  void EnsureAdamSlotRows(int64_t rows);
  float* AdamSlotPtr(int64_t row_index, bool squared);
  bool ApplySgdUpdateParallel(
      const Int64KeyView& keys, const float* grads, float lr);
  bool ApplyAdamWTFUpdateParallel(
      const Int64KeyView& keys,
      const float* grads,
      double lr,
      double beta1,
      double beta2,
      double eps,
      double weight_decay,
      AdamUpdateStats* stats);
  py::tuple LookupIndexedImpl(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      bool allocate_missing,
      py::dict* profile);
  py::tuple LookupIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      bool allocate_missing);
  py::tuple LookupIndexedProfiled(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      bool allocate_missing);
  py::dict LookupIndexedIntoRaw(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t indices_address,
      uintptr_t values_address,
      uintptr_t found_address,
      bool allocate_missing,
      bool release_gil,
      bool collect_profile);
  py::dict LookupIndexedLegacyIntoRaw(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t indices_address,
      uintptr_t values_address,
      uintptr_t found_address,
      bool allocate_missing,
      bool release_gil,
      bool collect_profile);
  py::dict LookupIndexedFusedIntoRaw(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t indices_address,
      uintptr_t values_address,
      uintptr_t found_address,
      bool allocate_missing,
      bool release_gil,
      bool collect_profile);
  py::dict LookupIndexedIntoRawImpl(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t indices_address,
      uintptr_t values_address,
      uintptr_t found_address,
      bool allocate_missing,
      bool release_gil,
      bool collect_profile,
      bool fused);
  uint64_t LookupValuesIntoRawImpl(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t values_address,
      bool release_gil,
      py::dict* profile);
  uint64_t LookupValuesIntoRaw(uintptr_t ids_address,
                               int64_t num_rows,
                               uintptr_t values_address,
                               bool release_gil);
  py::dict LookupValuesIntoRawProfiled(
      uintptr_t ids_address,
      int64_t num_rows,
      uintptr_t values_address,
      bool release_gil);
  int64_t Size();
  py::tuple SnapshotIndexed();
  int64_t DeleteKeysLocked(const std::vector<uint64_t>& requested_keys,
                           bool reset_allocator);
  int64_t DeleteIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids);
  int64_t ClearIndexed();
  std::vector<int64_t> NormalizedFreeRows();
  bool WriteIndexedState(const std::filesystem::path& state_path,
                         const std::string& metadata);
  RestoredIndexedState ReadIndexedState(const std::filesystem::path& state_path,
                                        const std::string& expected_metadata);
  void ValidateRestoredRows(const RestoredIndexedState& state);
  bool WriteIndexedManifest(
      const std::filesystem::path& checkpoint_path,
      const std::filesystem::path& temp_path,
      const std::string& metadata,
      const std::string& kv_filename,
      const std::string& state_filename);
  bool
  SaveCheckpoint(const std::string& checkpoint, const std::string& metadata);
  bool LoadCheckpoint(const std::string& checkpoint,
                      const std::string& expected_metadata);
  void PutIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          indices,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          values);
  py::dict BulkInsertIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          indices,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          values,
      bool release_gil);
  py::dict SgdUpdateIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          grads,
      double lr,
      bool release_gil);
  py::dict AdamWTFUpdateIndexed(
      const py::array_t<int64_t, py::array::c_style | py::array::forcecast>&
          ids,
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          grads,
      double lr,
      double beta1,
      double beta2,
      double eps,
      double weight_decay,
      bool release_gil);
  py::dict OptimizerStateStats() const;
  py::dict StorageStats();
};

} // namespace recstore::python
