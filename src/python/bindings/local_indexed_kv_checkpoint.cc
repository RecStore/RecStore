#include "local_indexed_kv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <unistd.h>

#include "storage/index/utils/hash.h"

namespace recstore::python {

namespace {

constexpr std::array<char, 8> kIndexedManifestMagic = {
    'R', 'S', 'I', 'D', 'X', 'M', '0', '1'};
constexpr std::array<char, 8> kIndexedStateMagic = {
    'R', 'S', 'I', 'D', 'X', 'S', '0', '1'};
constexpr uint32_t kIndexedCheckpointVersion      = 1;
constexpr uint64_t kIndexedCheckpointChecksumSeed = 0x6a09e667f3bcc909ULL;

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t* result) {
  if (result == nullptr ||
      left > (std::numeric_limits<uint64_t>::max)() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t* result) {
  if (result == nullptr ||
      (right != 0 && left > (std::numeric_limits<uint64_t>::max)() / right)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool IsSafeSidecarFilename(const std::string& filename) {
  if (filename.empty() || filename == "." || filename == "..") {
    return false;
  }
  const std::filesystem::path path(filename);
  return !path.is_absolute() && !path.has_parent_path() &&
         path.filename() == path;
}

bool WriteCheckpointBytes(
    std::ofstream& output, const void* data, size_t size) {
  if (size >
      static_cast<size_t>((std::numeric_limits<std::streamsize>::max)())) {
    return false;
  }
  if (size != 0) {
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(size));
  }
  return output.good();
}

template <typename T>
bool WriteCheckpointPod(std::ofstream& output, const T& value) {
  return WriteCheckpointBytes(output, &value, sizeof(value));
}

bool ReadCheckpointBytes(
    std::ifstream& input, void* data, uint64_t size, uint64_t* remaining) {
  if (remaining == nullptr || size > *remaining ||
      size > static_cast<uint64_t>(
                 (std::numeric_limits<std::streamsize>::max)())) {
    return false;
  }
  if (size != 0) {
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!input) {
      return false;
    }
  }
  *remaining -= size;
  return true;
}

template <typename T>
bool ReadCheckpointPod(std::ifstream& input, T* value, uint64_t* remaining) {
  return ReadCheckpointBytes(input, value, sizeof(*value), remaining);
}

void UpdateCheckpointChecksum(
    uint64_t* checksum, const void* data, size_t size) {
  if (size != 0) {
    *checksum = xxhash(data, size, *checksum);
  }
}

template <typename T>
void UpdateCheckpointChecksumPod(uint64_t* checksum, const T& value) {
  UpdateCheckpointChecksum(checksum, &value, sizeof(value));
}

struct IndexedCheckpointManifest {
  std::string metadata;
  std::string kv_filename;
  std::string state_filename;
};

IndexedCheckpointManifest
ReadIndexedCheckpointManifest(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("failed to inspect RecStore checkpoint manifest");
  }
  uint64_t remaining = static_cast<uint64_t>(file_size);
  std::ifstream input(path, std::ios::binary);
  std::array<char, kIndexedManifestMagic.size()> magic{};
  uint32_t version              = 0;
  uint64_t metadata_size        = 0;
  uint64_t kv_name_size         = 0;
  uint64_t state_name_size      = 0;
  uint64_t payload_size         = 0;
  uint64_t payload_and_checksum = 0;
  if (!input ||
      !ReadCheckpointBytes(input, magic.data(), magic.size(), &remaining) ||
      !ReadCheckpointPod(input, &version, &remaining) ||
      !ReadCheckpointPod(input, &metadata_size, &remaining) ||
      !ReadCheckpointPod(input, &kv_name_size, &remaining) ||
      !ReadCheckpointPod(input, &state_name_size, &remaining) ||
      magic != kIndexedManifestMagic || version != kIndexedCheckpointVersion ||
      metadata_size > remaining || kv_name_size > remaining - metadata_size ||
      state_name_size > remaining - metadata_size - kv_name_size ||
      metadata_size > std::numeric_limits<size_t>::max() ||
      kv_name_size > std::numeric_limits<size_t>::max() ||
      state_name_size > std::numeric_limits<size_t>::max() ||
      !CheckedAdd(metadata_size, kv_name_size, &payload_size) ||
      !CheckedAdd(payload_size, state_name_size, &payload_size) ||
      !CheckedAdd(payload_size, sizeof(uint64_t), &payload_and_checksum) ||
      payload_and_checksum != remaining) {
    throw std::runtime_error("invalid RecStore checkpoint manifest header");
  }

  IndexedCheckpointManifest manifest;
  manifest.metadata.resize(static_cast<size_t>(metadata_size));
  manifest.kv_filename.resize(static_cast<size_t>(kv_name_size));
  manifest.state_filename.resize(static_cast<size_t>(state_name_size));
  if (!ReadCheckpointBytes(
          input, manifest.metadata.data(), metadata_size, &remaining) ||
      !ReadCheckpointBytes(
          input, manifest.kv_filename.data(), kv_name_size, &remaining) ||
      !ReadCheckpointBytes(
          input, manifest.state_filename.data(), state_name_size, &remaining)) {
    throw std::runtime_error("truncated RecStore checkpoint manifest");
  }
  uint64_t saved_checksum = 0;
  if (!ReadCheckpointPod(input, &saved_checksum, &remaining) ||
      remaining != 0) {
    throw std::runtime_error("invalid RecStore checkpoint manifest trailer");
  }
  uint64_t checksum = kIndexedCheckpointChecksumSeed;
  UpdateCheckpointChecksum(&checksum, magic.data(), magic.size());
  UpdateCheckpointChecksumPod(&checksum, version);
  UpdateCheckpointChecksumPod(&checksum, metadata_size);
  UpdateCheckpointChecksumPod(&checksum, kv_name_size);
  UpdateCheckpointChecksumPod(&checksum, state_name_size);
  UpdateCheckpointChecksum(&checksum, manifest.metadata.data(), metadata_size);
  UpdateCheckpointChecksum(
      &checksum, manifest.kv_filename.data(), kv_name_size);
  UpdateCheckpointChecksum(
      &checksum, manifest.state_filename.data(), state_name_size);
  if (saved_checksum != checksum ||
      !IsSafeSidecarFilename(manifest.kv_filename) ||
      !IsSafeSidecarFilename(manifest.state_filename)) {
    throw std::runtime_error("invalid RecStore checkpoint manifest");
  }
  return manifest;
}

} // namespace

bool LocalIndexedKV::WriteIndexedState(const std::filesystem::path& state_path,
                                       const std::string& metadata) {
  const std::vector<int64_t> normalized_free = NormalizedFreeRows();
  const uint64_t active_rows                 = kv->ActiveKeyCount();
  const int64_t slot_rows =
      adam_slot_blocks.empty()
          ? 0
          : std::min<int64_t>(next_row_index,
                              static_cast<int64_t>(adam_slot_blocks.size()) *
                                  kAdamSlotBlockRows);
  const uint64_t metadata_size = static_cast<uint64_t>(metadata.size());
  const uint64_t free_count    = static_cast<uint64_t>(normalized_free.size());
  std::ofstream output(state_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  uint64_t checksum = kIndexedCheckpointChecksumSeed;
  auto write        = [&](const void* data, size_t size) {
    UpdateCheckpointChecksum(&checksum, data, size);
    return WriteCheckpointBytes(output, data, size);
  };
  bool ok =
      write(kIndexedStateMagic.data(), kIndexedStateMagic.size()) &&
      write(&kIndexedCheckpointVersion, sizeof(kIndexedCheckpointVersion)) &&
      write(&metadata_size, sizeof(metadata_size)) &&
      write(&capacity, sizeof(capacity)) && write(&dim, sizeof(dim)) &&
      write(&next_row_index, sizeof(next_row_index)) &&
      write(&active_rows, sizeof(active_rows)) &&
      write(&free_count, sizeof(free_count)) &&
      write(&adam_step, sizeof(adam_step)) &&
      write(&adam_beta1_t, sizeof(adam_beta1_t)) &&
      write(&adam_beta2_t, sizeof(adam_beta2_t)) &&
      write(&slot_rows, sizeof(slot_rows)) &&
      write(metadata.data(), metadata.size()) &&
      write(normalized_free.data(), normalized_free.size() * sizeof(int64_t));

  for (bool squared : {false, true}) {
    int64_t remaining_rows = slot_rows;
    for (size_t block = 0;
         ok && remaining_rows > 0 && block < adam_slot_blocks.size();
         ++block) {
      const int64_t rows =
          std::min<int64_t>(remaining_rows, kAdamSlotBlockRows);
      const auto& values = squared ? adam_slot_blocks[block]->exp_avg_sq
                                   : adam_slot_blocks[block]->exp_avg;
      const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(dim);
      ok                 = write(values.data(), count * sizeof(float));
      remaining_rows -= rows;
    }
    ok = ok && remaining_rows == 0;
  }
  ok = ok && WriteCheckpointPod(output, checksum);
  output.flush();
  ok = ok && output.good();
  output.close();
  return ok && !output.fail();
}

LocalIndexedKV::RestoredIndexedState
LocalIndexedKV::ReadIndexedState(const std::filesystem::path& state_path,
                                 const std::string& expected_metadata) {
  std::error_code error;
  const std::uintmax_t file_size =
      std::filesystem::file_size(state_path, error);
  if (error || file_size > std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("failed to inspect RecStore indexed state");
  }
  uint64_t remaining = static_cast<uint64_t>(file_size);
  std::ifstream input(state_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open RecStore indexed state");
  }
  uint64_t checksum = kIndexedCheckpointChecksumSeed;
  auto read         = [&](void* data, uint64_t size) {
    const bool ok = ReadCheckpointBytes(input, data, size, &remaining);
    if (ok) {
      UpdateCheckpointChecksum(&checksum, data, static_cast<size_t>(size));
    }
    return ok;
  };

  std::array<char, kIndexedStateMagic.size()> magic{};
  uint32_t version       = 0;
  uint64_t metadata_size = 0;
  int64_t saved_capacity = 0;
  int64_t saved_dim      = 0;
  uint64_t free_count    = 0;
  RestoredIndexedState state;
  if (!read(magic.data(), magic.size()) || !read(&version, sizeof(version)) ||
      !read(&metadata_size, sizeof(metadata_size)) ||
      !read(&saved_capacity, sizeof(saved_capacity)) ||
      !read(&saved_dim, sizeof(saved_dim)) ||
      !read(&state.next_row_index, sizeof(state.next_row_index)) ||
      !read(&state.active_rows, sizeof(state.active_rows)) ||
      !read(&free_count, sizeof(free_count)) ||
      !read(&state.adam_step, sizeof(state.adam_step)) ||
      !read(&state.adam_beta1_t, sizeof(state.adam_beta1_t)) ||
      !read(&state.adam_beta2_t, sizeof(state.adam_beta2_t)) ||
      !read(&state.slot_rows, sizeof(state.slot_rows)) ||
      magic != kIndexedStateMagic || version != kIndexedCheckpointVersion ||
      saved_capacity != capacity || saved_dim != dim ||
      state.next_row_index < 0 || state.next_row_index > capacity ||
      state.slot_rows < 0 || state.slot_rows > state.next_row_index ||
      state.active_rows > static_cast<uint64_t>(state.next_row_index) ||
      free_count > static_cast<uint64_t>(state.next_row_index) ||
      state.active_rows + free_count !=
          static_cast<uint64_t>(state.next_row_index) ||
      metadata_size > remaining || !std::isfinite(state.adam_beta1_t) ||
      !std::isfinite(state.adam_beta2_t)) {
    throw std::runtime_error("invalid RecStore indexed state header");
  }
  if (state.slot_rows != 0 &&
      static_cast<uint64_t>(state.slot_rows) >
          std::numeric_limits<size_t>::max() / static_cast<uint64_t>(dim)) {
    throw std::runtime_error("RecStore indexed optimizer state is too large");
  }

  uint64_t free_bytes      = 0;
  uint64_t slot_values_u64 = 0;
  uint64_t slot_bytes      = 0;
  uint64_t required_bytes  = 0;
  if (metadata_size > std::numeric_limits<size_t>::max() ||
      !CheckedMultiply(free_count, sizeof(int64_t), &free_bytes) ||
      !CheckedMultiply(static_cast<uint64_t>(state.slot_rows),
                       static_cast<uint64_t>(dim),
                       &slot_values_u64) ||
      !CheckedMultiply(slot_values_u64, sizeof(float), &slot_bytes) ||
      free_bytes > std::numeric_limits<size_t>::max() ||
      slot_values_u64 > std::numeric_limits<size_t>::max() ||
      slot_bytes > std::numeric_limits<size_t>::max() ||
      !CheckedAdd(metadata_size, free_bytes, &required_bytes) ||
      !CheckedAdd(required_bytes, slot_bytes, &required_bytes) ||
      !CheckedAdd(required_bytes, slot_bytes, &required_bytes) ||
      !CheckedAdd(required_bytes, sizeof(uint64_t), &required_bytes) ||
      required_bytes != remaining) {
    throw std::runtime_error("invalid RecStore indexed state size");
  }

  std::string metadata(static_cast<size_t>(metadata_size), '\0');
  state.free_rows.resize(static_cast<size_t>(free_count));
  const size_t slot_values = static_cast<size_t>(slot_values_u64);
  state.exp_avg.resize(slot_values);
  state.exp_avg_sq.resize(slot_values);
  if (!read(metadata.data(), metadata_size) ||
      !read(state.free_rows.data(), free_bytes) ||
      !read(state.exp_avg.data(), slot_bytes) ||
      !read(state.exp_avg_sq.data(), slot_bytes) ||
      metadata != expected_metadata) {
    throw std::runtime_error("invalid RecStore indexed state payload");
  }
  uint64_t saved_checksum = 0;
  if (!ReadCheckpointPod(input, &saved_checksum, &remaining) ||
      remaining != 0 || saved_checksum != checksum) {
    throw std::runtime_error("invalid RecStore indexed state checksum");
  }

  std::vector<uint8_t> seen(static_cast<size_t>(state.next_row_index), 0);
  for (const int64_t row_index : state.free_rows) {
    if (row_index < 0 || row_index >= state.next_row_index ||
        seen[static_cast<size_t>(row_index)] != 0) {
      throw std::runtime_error("invalid RecStore checkpoint free list");
    }
    seen[static_cast<size_t>(row_index)] = 1;
  }
  return state;
}

void LocalIndexedKV::ValidateRestoredRows(const RestoredIndexedState& state) {
  if (kv->ActiveKeyCount() != state.active_rows) {
    throw std::runtime_error("RecStore restored active-row count mismatch");
  }
  const std::vector<uint64_t> keys = kv->SnapshotKeys();
  std::vector<BaseKV::DirectFixedRow> rows;
  if (!keys.empty() &&
      !kv->BatchGetDirectFixedRows(
          base::ConstArray<uint64_t>(keys),
          static_cast<int64_t>(keys.size()),
          storage_dim,
          0,
          &rows)) {
    throw std::runtime_error("RecStore restored row lookup failed");
  }
  std::vector<uint8_t> seen(static_cast<size_t>(state.next_row_index), 0);
  for (const int64_t row_index : state.free_rows) {
    seen[static_cast<size_t>(row_index)] = 1;
  }
  const size_t row_bytes = static_cast<size_t>(storage_dim) * sizeof(float);
  for (const auto& row : rows) {
    if (row.missing || row.data == nullptr || row.size != row_bytes) {
      throw std::runtime_error("RecStore restored an invalid value row");
    }
    int64_t row_index = -1;
    std::memcpy(&row_index, row.data, sizeof(int64_t));
    if (row_index < 0 || row_index >= state.next_row_index ||
        seen[static_cast<size_t>(row_index)] != 0) {
      throw std::runtime_error("RecStore restored invalid stable indices");
    }
    seen[static_cast<size_t>(row_index)] = 1;
  }
  if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
    throw std::runtime_error("RecStore restored allocator has unowned rows");
  }
}

bool LocalIndexedKV::WriteIndexedManifest(
    const std::filesystem::path& checkpoint_path,
    const std::filesystem::path& temp_path,
    const std::string& metadata,
    const std::string& kv_filename,
    const std::string& state_filename) {
  const uint64_t metadata_size   = static_cast<uint64_t>(metadata.size());
  const uint64_t kv_name_size    = static_cast<uint64_t>(kv_filename.size());
  const uint64_t state_name_size = static_cast<uint64_t>(state_filename.size());
  std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return false;
  }
  uint64_t checksum = kIndexedCheckpointChecksumSeed;
  auto write        = [&](const void* data, size_t size) {
    UpdateCheckpointChecksum(&checksum, data, size);
    return WriteCheckpointBytes(output, data, size);
  };
  bool ok =
      write(kIndexedManifestMagic.data(), kIndexedManifestMagic.size()) &&
      write(&kIndexedCheckpointVersion, sizeof(kIndexedCheckpointVersion)) &&
      write(&metadata_size, sizeof(metadata_size)) &&
      write(&kv_name_size, sizeof(kv_name_size)) &&
      write(&state_name_size, sizeof(state_name_size)) &&
      write(metadata.data(), metadata.size()) &&
      write(kv_filename.data(), kv_filename.size()) &&
      write(state_filename.data(), state_filename.size()) &&
      WriteCheckpointPod(output, checksum);
  output.flush();
  ok = ok && output.good();
  output.close();
  if (!ok || output.fail()) {
    return false;
  }
  std::error_code error;
  std::filesystem::rename(temp_path, checkpoint_path, error);
  return !error;
}

bool LocalIndexedKV::SaveCheckpoint(const std::string& checkpoint,
                                    const std::string& metadata) {
  if (checkpoint.empty()) {
    throw std::invalid_argument("RecStore checkpoint path must be non-empty");
  }
  std::lock_guard<std::mutex> lock(mu);
  const std::filesystem::path checkpoint_path(checkpoint);
  const std::string base = checkpoint_path.filename().string();
  if (!IsSafeSidecarFilename(base)) {
    throw std::invalid_argument(
        "RecStore checkpoint path must name a regular manifest file");
  }
  const std::filesystem::path parent =
      checkpoint_path.parent_path().empty()
          ? std::filesystem::path(".")
          : checkpoint_path.parent_path();
  std::filesystem::create_directories(parent);
  IndexedCheckpointManifest previous_manifest;
  bool has_previous_manifest = false;
  std::error_code exists_error;
  if (std::filesystem::is_regular_file(checkpoint_path, exists_error) &&
      !exists_error) {
    try {
      previous_manifest     = ReadIndexedCheckpointManifest(checkpoint_path);
      has_previous_manifest = true;
    } catch (const std::exception&) {
      // A valid new generation may replace a corrupt old manifest. Its
      // unreferenced sidecars cannot be identified safely and are retained.
    }
  }
  static std::atomic<uint64_t> generation{0};
  const uint64_t nonce = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string suffix =
      "." + std::to_string(getpid()) + "." + std::to_string(nonce) + "." +
      std::to_string(generation.fetch_add(1));
  const std::string kv_filename          = base + suffix + ".kv";
  const std::string state_filename       = base + suffix + ".state";
  const std::filesystem::path kv_path    = parent / kv_filename;
  const std::filesystem::path state_path = parent / state_filename;
  const std::filesystem::path manifest_temp =
      parent / (base + suffix + ".manifest.tmp");

  if (!kv->SaveCheckpoint(kv_path.string(), metadata) ||
      !WriteIndexedState(state_path, metadata) ||
      !WriteIndexedManifest(
          checkpoint_path,
          manifest_temp,
          metadata,
          kv_filename,
          state_filename)) {
    std::error_code error;
    std::filesystem::remove(kv_path, error);
    std::filesystem::remove(state_path, error);
    std::filesystem::remove(manifest_temp, error);
    return false;
  }
  if (has_previous_manifest) {
    std::error_code cleanup_error;
    if (previous_manifest.kv_filename != kv_filename) {
      std::filesystem::remove(
          parent / previous_manifest.kv_filename, cleanup_error);
    }
    cleanup_error.clear();
    if (previous_manifest.state_filename != state_filename) {
      std::filesystem::remove(
          parent / previous_manifest.state_filename, cleanup_error);
    }
  }
  return true;
}

bool LocalIndexedKV::LoadCheckpoint(const std::string& checkpoint,
                                    const std::string& expected_metadata) {
  if (checkpoint.empty()) {
    throw std::invalid_argument("RecStore checkpoint path must be non-empty");
  }
  std::lock_guard<std::mutex> lock(mu);
  if (kv->ActiveKeyCount() != 0 || next_row_index != 0 ||
      !free_row_indices.empty()) {
    throw std::runtime_error(
        "RecStore checkpoint restore requires an empty indexed engine");
  }
  const std::filesystem::path checkpoint_path(checkpoint);
  const IndexedCheckpointManifest manifest =
      ReadIndexedCheckpointManifest(checkpoint_path);
  if (manifest.metadata != expected_metadata) {
    throw std::runtime_error("RecStore checkpoint metadata mismatch");
  }
  const std::filesystem::path parent =
      checkpoint_path.parent_path().empty()
          ? std::filesystem::path(".")
          : checkpoint_path.parent_path();
  const std::filesystem::path kv_path    = parent / manifest.kv_filename;
  const std::filesystem::path state_path = parent / manifest.state_filename;
  RestoredIndexedState state = ReadIndexedState(state_path, expected_metadata);

  std::vector<std::unique_ptr<AdamSlotBlock>> restored_slot_blocks;
  const size_t required_blocks = static_cast<size_t>(
      (state.slot_rows + kAdamSlotBlockRows - 1) / kAdamSlotBlockRows);
  restored_slot_blocks.reserve(required_blocks);
  for (size_t block = 0; block < required_blocks; ++block) {
    restored_slot_blocks.push_back(std::make_unique<AdamSlotBlock>(dim));
  }
  size_t offset          = 0;
  int64_t remaining_rows = state.slot_rows;
  for (size_t block = 0; remaining_rows > 0; ++block) {
    const int64_t rows = std::min<int64_t>(remaining_rows, kAdamSlotBlockRows);
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(dim);
    std::memcpy(restored_slot_blocks[block]->exp_avg.data(),
                state.exp_avg.data() + offset,
                count * sizeof(float));
    std::memcpy(restored_slot_blocks[block]->exp_avg_sq.data(),
                state.exp_avg_sq.data() + offset,
                count * sizeof(float));
    offset += count;
    remaining_rows -= rows;
  }
  if (!kv->LoadCheckpoint(kv_path.string(), expected_metadata)) {
    return false;
  }
  try {
    ValidateRestoredRows(state);
  } catch (...) {
    kv->clear();
    throw;
  }

  next_row_index   = state.next_row_index;
  free_row_indices = std::move(state.free_rows);
  adam_step        = state.adam_step;
  adam_beta1_t     = state.adam_beta1_t;
  adam_beta2_t     = state.adam_beta2_t;
  adam_slot_blocks = std::move(restored_slot_blocks);
  return true;
}

} // namespace recstore::python
