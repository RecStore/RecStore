#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "base/array.h"
#include "base/tensor.h"
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace recstore {

// Fixed embedding width shared by the v4 exact-owner replacement wire format.
inline constexpr size_t kCanonicalSparseDim = 128;

// Byte-unaligned finite scan over wire payloads (v4 rejects nonfinite rows).
inline bool CanonicalAllFinite(const void* values, size_t count) {
  const auto* bytes = static_cast<const char*>(values);
  for (size_t at = 0; at < count; ++at) {
    float value;
    std::memcpy(&value, bytes + at * sizeof(float), sizeof(value));
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

// Diagnostic-only, enabled before process startup. Disabled runs take no
// clocks.
inline bool SparseCommitProfileEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("RECSTORE_SPARSE_COMMIT_PROFILE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

inline uint64_t SparseCommitProfileNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Homogeneous-host wire format, shared by RPC, SHM and RDMA transports.
struct SparseGradientCommitHeader {
  uint64_t session    = 0;
  int64_t step        = 0;
  uint32_t rank       = 0;
  uint32_t world      = 1;
  uint32_t dim        = 0;
  uint32_t rows       = 0;
  uint32_t chunk      = 0;
  uint32_t chunks     = 1;
  uint32_t kind       = 0; // 0: submit, 1: query, 2: close, 3: abort
  float learning_rate = 0.01f;
  uint32_t divisor    = 1;
  uint32_t version    = 1;
};
static_assert(sizeof(SparseGradientCommitHeader) == 56,
              "SparseGradientCommitHeader wire size");

inline SparseGradientCommitHeader
DecodeSparseGradientCommit(const std::string& payload) {
  SparseGradientCommitHeader h;
  if (payload.size() < sizeof(h)) {
    throw std::invalid_argument("Truncated sparse gradient commit header");
  }
  std::memcpy(&h, payload.data(), sizeof(h));
  if ((h.version != 1 && h.version != 4) || !h.session || !h.world ||
      h.world > 65536 || h.rank >= h.world || !h.dim || h.dim > 65536 ||
      h.step < 0 || h.kind > 3 || !h.chunks || h.chunks > 1048576 ||
      h.chunk >= h.chunks || (h.divisor != 1 && h.divisor != h.world) ||
      !std::isfinite(h.learning_rate) || h.learning_rate < 0) {
    throw std::invalid_argument("Invalid sparse gradient commit metadata");
  }
  const uint64_t bytes =
      sizeof(h) + uint64_t(h.rows) * (8 + uint64_t(h.dim) * 4);
  if (bytes != payload.size() || (h.kind != 0 && h.rows != 0)) {
    throw std::invalid_argument("Invalid sparse gradient commit payload size");
  }
  if (h.version == 4) {
    if (h.world != 2 || h.dim != kCanonicalSparseDim || h.divisor != 1 ||
        h.chunks > 64 || payload.size() > (size_t{1} << 20))
      throw std::invalid_argument("Unsupported exact owner replacement bounds");
    uint64_t previous = 0;
    for (size_t row = 0; row < h.rows; ++row) {
      uint64_t key;
      std::memcpy(&key, payload.data() + sizeof(h) + row * 8, 8);
      if (row && key <= previous)
        throw std::invalid_argument(
            "Exact replacement keys must be strictly increasing");
      previous = key;
    }
    if (!CanonicalAllFinite(payload.data() + sizeof(h) + size_t(h.rows) * 8,
                            size_t(h.rows) * h.dim))
      throw std::invalid_argument("Non-finite exact replacement values");
  }
  return h;
}

inline std::string PackSparseGradientCommit(SparseGradientCommitHeader h,
                                            const uint64_t* keys = nullptr,
                                            const float* grads   = nullptr) {
  std::string result(
      sizeof(h) + size_t(h.rows) * (8 + size_t(h.dim) * 4), '\0');
  std::memcpy(result.data(), &h, sizeof(h));
  if (h.rows) {
    if (!keys || !grads)
      throw std::invalid_argument("Missing sparse gradient commit rows");
    std::memcpy(result.data() + sizeof(h), keys, size_t(h.rows) * 8);
    std::memcpy(result.data() + sizeof(h) + size_t(h.rows) * 8,
                grads,
                size_t(h.rows) * h.dim * 4);
  }
  DecodeSparseGradientCommit(result);
  return result;
}

class SparseGradientCommitClient {
public:
  virtual ~SparseGradientCommitClient() = default;
  virtual int64_t ExchangeSparseGradientCommit(const std::string& table,
                                               const std::string& payload) {
    throw std::runtime_error(
        "Sparse gradient commit protocol unsupported by this PS backend");
  }

  virtual size_t SparseGradientCommitPayloadLimit() const { return 256 * 1024; }

  virtual int64_t SparseGradientCommit(
      const std::string& table,
      SparseGradientCommitHeader h,
      const base::ConstArray<uint64_t>& keys,
      const base::RecTensor& grads) {
    if (h.kind != 0)
      return ExchangeSparseGradientCommit(table, PackSparseGradientCommit(h));
    if (grads.dim() != 2 ||
        grads.shape(0) != static_cast<int64_t>(keys.Size()) ||
        grads.shape(1) != h.dim || grads.dtype() != base::DataType::FLOAT32) {
      throw std::invalid_argument("Invalid sparse gradient shape");
    }
    const size_t limit     = SparseGradientCommitPayloadLimit();
    const size_t row_bytes = 8 + size_t(h.dim) * 4;
    if (limit < sizeof(h) + row_bytes)
      throw std::runtime_error("Gradient row exceeds transport slot");
    const size_t rows = (limit - sizeof(h)) / row_bytes;
    h.chunks          = std::max<size_t>(1, (keys.Size() + rows - 1) / rows);
    int64_t committed = -1;
    for (h.chunk = 0; h.chunk < h.chunks; ++h.chunk) {
      const size_t begin = size_t(h.chunk) * rows;
      h.rows             = std::min(rows, keys.Size() - begin);
      committed          = ExchangeSparseGradientCommit(
          table,
          PackSparseGradientCommit(
              h,
              h.rows ? keys.Data() + begin : nullptr,
              h.rows ? grads.data_as<float>() + begin * h.dim : nullptr));
    }
    return committed;
  }
};

// One in-flight step per table. Handlers never wait for another trainer.
class SparseGradientCommitCoordinator {
public:
  using Apply = std::function<void(
      const std::vector<uint64_t>&, const std::vector<float>&)>;

  // Wrapper validation can fail before Exchange sees a malformed request.
  // A recognizable different session must not poison the active one.
  void FailSession(const std::string& table, const std::string& payload) {
    uint64_t session = 0;
    const bool has_session =
        payload.size() >= sizeof(SparseGradientCommitHeader);
    if (has_session)
      std::memcpy(&session, payload.data(), sizeof(session));
    std::lock_guard<std::mutex> guard(mu_);
    const auto it = states_.find(table);
    if (it != states_.end() && it->second.header.version == 4 &&
        (!has_session || session == it->second.header.session))
      it->second.failed = true;
  }

  int64_t Exchange(const std::string& table,
                   std::string payload,
                   const Apply& apply,
                   int exact_owner_rank = -1) {
    const bool profile       = SparseCommitProfileEnabled();
    const auto request_start = profile ? SparseCommitProfileNowNs() : 0;
    SparseGradientCommitHeader h;
    try {
      h = DecodeSparseGradientCommit(payload);
      if (h.version == 4 &&
          (exact_owner_rank < 0 || exact_owner_rank >= 2 ||
           (h.rows && h.rank != static_cast<uint32_t>(exact_owner_rank))))
        throw std::invalid_argument(
            "Exact replacement requires the server-bound owner rank");
    } catch (...) {
      FailSession(table, payload);
      throw;
    }
    // ponytail: one lock per PS shard; split by table if independent jobs need
    // concurrent commits. The fused BagPipe table already requires ordering.
    std::lock_guard<std::mutex> guard(mu_);
    const auto locked = profile ? SparseCommitProfileNowNs() : 0;
    auto it           = states_.find(table);
    if (it == states_.end()) {
      if (h.kind != 0 || h.step != 0) {
        throw std::runtime_error("Unknown sparse gradient session");
      }
      State state;
      state.header           = h;
      state.exact_owner_rank = exact_owner_rank;
      state.ranks.resize(h.world);
      state.closed.resize(h.world, false);
      it = states_.emplace(table, std::move(state)).first;
    }
    auto& state         = it->second;
    const auto& initial = state.header;
    if (initial.session != h.session || initial.world != h.world ||
        initial.dim != h.dim || initial.learning_rate != h.learning_rate ||
        initial.divisor != h.divisor || initial.version != h.version ||
        (h.version == 4 && state.exact_owner_rank != exact_owner_rank)) {
      if (initial.session == h.session &&
          (initial.version == 4 || h.version == 4))
        state.failed = true;
      throw std::runtime_error(
          "Sparse gradient session configuration mismatch");
    }
    if (h.kind == 3)
      state.failed = true;
    if (state.failed)
      throw std::runtime_error(
          "Sparse gradient session aborted or commit failed");
    if (h.kind == 1)
      return state.committed;
    if (h.kind == 2) {
      if (state.committed < h.step ||
          (h.version == 4 &&
           std::any_of(state.ranks.begin(),
                       state.ranks.end(),
                       [](const auto& rank) { return !rank.empty(); })))
        throw std::runtime_error("Closing an uncommitted step");
      state.closed[h.rank] = true;
      const auto committed = state.committed;
      bool all_closed      = true;
      for (bool closed : state.closed)
        all_closed &= closed;
      if (all_closed)
        states_.erase(it);
      return committed;
    }
    // A transport retry must never apply an already committed step twice.
    if (h.step <= state.committed) {
      if (h.version == 4 &&
          (h.step != state.committed ||
           h.rank >= state.last_exact_packets.size() ||
           h.chunk >= state.last_exact_packets[h.rank].size() ||
           state.last_exact_packets[h.rank][h.chunk] != payload)) {
        state.failed = true;
        throw std::runtime_error(
            "Conflicting or unknown completed exact replacement retry");
      }
      return state.committed;
    }
    if (h.step != state.committed + 1 || state.closed[h.rank] ||
        (h.version == 4 &&
         std::any_of(state.closed.begin(), state.closed.end(), [](bool closed) {
           return closed;
         }))) {
      if (h.version == 4)
        state.failed = true;
      throw std::runtime_error(
          "Sparse gradient commits must be submitted in order");
    }
    if (h.version == 4) {
      try {
        return ExchangeExact(
            state,
            h,
            std::move(payload),
            apply,
            profile,
            request_start,
            locked);
      } catch (...) {
        // Exact replacement failures are sticky; never retry or fall back.
        state.failed = true;
        throw;
      }
    }
    auto& chunks = state.ranks[h.rank];
    if (chunks.empty())
      chunks.resize(h.chunks);
    if (chunks.size() != h.chunks)
      throw std::runtime_error("Gradient chunk count changed");
    if (!chunks[h.chunk].empty()) {
      if (chunks[h.chunk] != payload)
        throw std::runtime_error("Conflicting gradient chunk retry");
      return state.committed;
    }
    const auto payload_bytes = payload.size();
    chunks[h.chunk]          = std::move(payload);
    bool complete            = true;
    for (const auto& rank : state.ranks) {
      if (rank.empty()) {
        complete = false;
        break;
      }
      for (const auto& chunk : rank) {
        if (chunk.empty()) {
          complete = false;
          break;
        }
      }
      if (!complete)
        break;
    }
    const auto collected = profile ? SparseCommitProfileNowNs() : 0;
    if (profile) {
      if (!state.profile_first_ns)
        state.profile_first_ns = request_start;
      state.profile_lock_ns += locked - request_start;
      state.profile_collect_ns += collected - locked;
      state.profile_payload_bytes += payload_bytes;
      state.profile_rows += h.rows;
      ++state.profile_chunks;
    }
    if (!complete)
      return state.committed;
    std::vector<uint64_t> keys;
    std::vector<float> grads;
    std::unordered_map<uint64_t, size_t> positions;
    // Fixed rank/chunk order makes network arrival order irrelevant.
    for (const auto& rank : state.ranks) {
      for (const auto& chunk : rank) {
        const auto ch         = DecodeSparseGradientCommit(chunk);
        const char* key_data  = chunk.data() + sizeof(ch);
        const char* grad_data = key_data + size_t(ch.rows) * 8;
        for (size_t row = 0; row < ch.rows; ++row) {
          uint64_t key;
          std::memcpy(&key, key_data + row * 8, 8);
          auto entry = positions.emplace(key, keys.size());
          if (entry.second) {
            keys.push_back(key);
            grads.resize(keys.size() * h.dim, 0.0f);
          }
          float* dst = grads.data() + entry.first->second * h.dim;
          for (size_t col = 0; col < h.dim; ++col) {
            float value;
            std::memcpy(&value, grad_data + (row * h.dim + col) * 4, 4);
            dst[col] += value;
          }
        }
      }
    }
    if (h.divisor != 1)
      for (auto& value : grads)
        value /= h.divisor;
    const auto aggregated = profile ? SparseCommitProfileNowNs() : 0;
    try {
      if (!keys.empty())
        apply(keys, grads);
    } catch (...) {
      state.failed = true;
      throw;
    }
    const auto applied = profile ? SparseCommitProfileNowNs() : 0;
    if (profile) {
      state.profile_aggregate_ns += aggregated - collected;
      state.profile_apply_ns += applied - aggregated;
      state.profile_unique_rows += keys.size();
    }
    FinishStep(state, h, profile, request_start, collected, applied);
    return state.committed;
  }

private:
  struct State {
    SparseGradientCommitHeader header;
    int64_t committed = -1;
    bool failed       = false;
    std::vector<std::vector<std::string>> ranks;
    // V4 retains only the last completed step for bit-identical ACK retries.
    std::vector<std::vector<std::string>> last_exact_packets;
    int exact_owner_rank = -1;
    size_t exact_rows    = 0;
    std::vector<bool> closed;
    uint64_t profile_first_ns      = 0;
    uint64_t profile_lock_ns       = 0;
    uint64_t profile_collect_ns    = 0;
    uint64_t profile_payload_bytes = 0;
    uint64_t profile_rows          = 0;
    uint64_t profile_chunks        = 0;
    uint64_t profile_aggregate_ns = 0, profile_apply_ns = 0;
    size_t profile_unique_rows = 0;
  };

  // Packets are validated before storage and retained until commit for retries.
  static SparseGradientCommitHeader StoredHeader(const std::string& packet) {
    SparseGradientCommitHeader h;
    std::memcpy(&h, packet.data(), sizeof(h));
    return h;
  }
  int64_t ExchangeExact(
      State& state,
      const SparseGradientCommitHeader& h,
      std::string payload,
      const Apply& apply,
      bool profile,
      uint64_t request_start,
      uint64_t locked) {
    auto& packets = state.ranks[h.rank];
    if (packets.empty())
      packets.resize(h.chunks);
    if (packets.size() != h.chunks)
      throw std::runtime_error("Exact replacement chunk count changed");
    if (!packets[h.chunk].empty()) {
      if (packets[h.chunk] != payload)
        throw std::runtime_error("Conflicting exact replacement retry");
      return state.committed;
    }
    constexpr size_t max_rows = 2 * 26 * 1024;
    if (h.rows > max_rows - state.exact_rows)
      throw std::invalid_argument("Exact replacement step exceeds row bound");
    state.exact_rows += h.rows;
    const size_t payload_bytes = payload.size();
    packets[h.chunk]           = std::move(payload);
    bool complete              = true;
    for (const auto& rank : state.ranks) {
      if (rank.empty() ||
          std::any_of(rank.begin(), rank.end(), [](const auto& packet) {
            return packet.empty();
          })) {
        complete = false;
        break;
      }
    }
    const auto collected = profile ? SparseCommitProfileNowNs() : 0;
    if (profile) {
      if (!state.profile_first_ns)
        state.profile_first_ns = request_start;
      state.profile_lock_ns += locked - request_start;
      state.profile_collect_ns += collected - locked;
      state.profile_payload_bytes += payload_bytes;
      state.profile_rows += h.rows;
      ++state.profile_chunks;
    }
    if (!complete)
      return state.committed;

    std::vector<uint64_t> keys;
    std::vector<float> values;
    keys.reserve(state.exact_rows);
    values.resize(state.exact_rows * h.dim);
    // Only the server-bound owner can have rows. Retain its exact row/bit
    // order; no reduction, scaling, or optimizer arithmetic is performed in V4.
    for (const auto& packet : state.ranks[state.exact_owner_rank]) {
      const auto ch          = StoredHeader(packet);
      const char* key_data   = packet.data() + sizeof(ch);
      const char* value_data = key_data + size_t(ch.rows) * 8;
      const size_t begin     = keys.size();
      for (size_t row = 0; row < ch.rows; ++row) {
        uint64_t key;
        std::memcpy(&key, key_data + row * 8, 8);
        if (!keys.empty() && key <= keys.back())
          throw std::invalid_argument(
              "Exact replacement chunks are not strictly sorted");
        keys.push_back(key);
      }
      if (ch.rows)
        std::memcpy(values.data() + begin * h.dim,
                    value_data,
                    size_t(ch.rows) * h.dim * sizeof(float));
    }
    if (keys.size() != state.exact_rows)
      throw std::runtime_error("Exact replacement owner row count mismatch");
    const auto assembled = profile ? SparseCommitProfileNowNs() : 0;
    if (!keys.empty())
      apply(keys, values);
    const auto applied = profile ? SparseCommitProfileNowNs() : 0;
    if (profile) {
      state.profile_aggregate_ns += assembled - collected;
      state.profile_apply_ns += applied - assembled;
      state.profile_unique_rows += keys.size();
    }
    FinishStep(state, h, profile, request_start, collected, applied);
    return state.committed;
  }
  void FinishStep(State& state,
                  const SparseGradientCommitHeader& h,
                  bool profile,
                  uint64_t request_start,
                  uint64_t collected,
                  uint64_t applied) {
    state.committed = h.step;
    if (h.version == 4) {
      state.last_exact_packets = std::move(state.ranks);
      state.ranks.resize(h.world);
      state.exact_rows = 0;
    } else {
      for (auto& rank : state.ranks)
        rank.clear();
    }
    if (profile) {
      const auto released = SparseCommitProfileNowNs();
      // One stdio call keeps each diagnostic line intact across server threads.
      // Request timing excludes logging and destruction of local merge buffers.
      std::fprintf(
          stderr,
          "component=sparse_commit event=coordinator_phase session=%llu "
          "step=%lld "
          "world=%u dim=%u chunks=%llu input_rows=%llu unique_rows=%zu "
          "payload_bytes=%llu arrival_span_ms=%.6f lock_wait_ms=%.6f "
          "collect_cpu_ms=%.6f aggregate_ms=%.6f apply_ms=%.6f "
          "release_chunks_ms=%.6f last_request_ms=%.6f version=%u\n",
          static_cast<unsigned long long>(h.session),
          static_cast<long long>(h.step),
          h.world,
          h.dim,
          static_cast<unsigned long long>(state.profile_chunks),
          static_cast<unsigned long long>(state.profile_rows),
          state.profile_unique_rows,
          static_cast<unsigned long long>(state.profile_payload_bytes),
          (collected - state.profile_first_ns) / 1e6,
          state.profile_lock_ns / 1e6,
          state.profile_collect_ns / 1e6,
          state.profile_aggregate_ns / 1e6,
          state.profile_apply_ns / 1e6,
          (released - applied) / 1e6,
          (released - request_start) / 1e6,
          h.version);
      state.profile_first_ns      = 0;
      state.profile_lock_ns       = 0;
      state.profile_collect_ns    = 0;
      state.profile_payload_bytes = 0;
      state.profile_rows          = 0;
      state.profile_chunks        = 0;
      state.profile_aggregate_ns = state.profile_apply_ns = 0;
      state.profile_unique_rows                           = 0;
    }
  }
  std::mutex mu_;
  std::unordered_map<std::string, State> states_;
};
} // namespace recstore
