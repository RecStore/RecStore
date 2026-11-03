#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "city.h"

namespace recstore {

// Merged-gradient owner routing. Each union key carries one merged gradient row
// per source rank that observed it, so per-key vector counts are 0 or 1 and no
// reduction-tree derivation exists. Field layout mirrors CanonicalOwnerPlan so
// the GPU exchange, forward exchange and v4 exact writeback stay unchanged.
struct OwnerMergedPlan {
  std::vector<int64_t> permutation; // Local merged rows, destination 0 then 1.
  std::vector<int64_t> meta; // [key,c0,c1,type,offset0,nvec0,offset1,nvec1].
  std::vector<int64_t> keys; // Logical keys owned by this rank, ascending.
  std::array<int64_t, 2> send{}; // Merged rows by destination rank.
  std::array<int64_t, 2> recv{}; // Merged rows by source rank.
  std::vector<int64_t> forward_send_index;
  std::vector<int64_t> forward_reorder;
  std::array<int64_t, 2> forward_send{};
  std::array<int64_t, 2> forward_recv{};
  std::array<int64_t, 2> rows{}; // Local unique rows by source rank.
};

namespace owner_merged_detail {
inline void ValidateInput(
    int64_t n,
    const int64_t* keys,
    const int64_t* counts,
    int64_t features,
    const int64_t* boundaries) {
  // Merged semantics: counts only annotate presence (c0/c1 columns); the
  // per-feature occurrence-total equality of the canonical protocol does not
  // hold for merged rows and is intentionally not checked here.
  if (n < 0 || n > 26 * 1024 || (n && (!keys || !counts)))
    throw std::invalid_argument("Invalid merged owner key/count buffers");
  for (int64_t i = 0; i < n; ++i)
    if (keys[i] < boundaries[0] || keys[i] >= boundaries[features] ||
        (i && keys[i] <= keys[i - 1]) || counts[i] < 1 || counts[i] > 1024)
      throw std::invalid_argument("Invalid sorted logical owner keys/counts");
}
} // namespace owner_merged_detail

// Fixed deployment: two city_hash shards, each colocated with the same rank.
// boundaries has features+1 entries, including the exclusive final row bound.
// registered_tag is the actual backend namespace, not a table index guess.
// Both offsets in meta are relative to their separate source tensor. A missing
// source has zero rows and retains its current nonnegative prefix offset.
inline OwnerMergedPlan BuildOwnerMergedPlan(
    int64_t n0,
    const int64_t* keys0,
    const int64_t* counts0,
    int64_t n1,
    const int64_t* keys1,
    const int64_t* counts1,
    int64_t features,
    const int64_t* table_boundaries,
    const int64_t* table_types,
    int64_t registered_tag,
    int64_t rank,
    bool include_forward = false) {
  constexpr uint64_t kLogicalLimit = uint64_t{1} << 56;
  if (features < 1 || features > 26 || !table_boundaries || !table_types ||
      registered_tag < 0 || registered_tag > 255 || rank < 0 || rank > 1)
    throw std::invalid_argument("Invalid merged owner layout/tag/rank");
  if (table_boundaries[0] < 0 ||
      table_boundaries[features] > static_cast<int64_t>(kLogicalLimit))
    throw std::invalid_argument("Merged owner layout overlaps namespace bits");
  for (int64_t feature = 0; feature < features; ++feature)
    if (table_boundaries[feature + 1] <= table_boundaries[feature] ||
        table_types[feature] < 0 || table_types[feature] > 1)
      throw std::invalid_argument("Invalid merged owner table boundary/type");
  owner_merged_detail::ValidateInput(
      n0, keys0, counts0, features, table_boundaries);
  owner_merged_detail::ValidateInput(
      n1, keys1, counts1, features, table_boundaries);

  OwnerMergedPlan result;
  std::array<std::vector<int64_t>, 2> permutations;
  std::array<std::vector<int64_t>, 2> forward_sends, forward_local_positions;
  int64_t local_row = 0, local_position = 0;
  std::array<std::array<int64_t, 2>, 2> destination_rows{};
  result.keys.reserve(static_cast<size_t>(n0 + n1));
  result.meta.reserve(static_cast<size_t>(n0 + n1) * 8);
  int64_t i0 = 0, i1 = 0, feature = 0;
  while (i0 < n0 || i1 < n1) {
    const int64_t key =
        i1 == n1 || (i0 < n0 && keys0[i0] < keys1[i1]) ? keys0[i0] : keys1[i1];
    const int64_t c0 = i0 < n0 && keys0[i0] == key ? counts0[i0++] : 0;
    const int64_t c1 = i1 < n1 && keys1[i1] == key ? counts1[i1++] : 0;
    while (feature + 1 < features && key >= table_boundaries[feature + 1])
      ++feature;
    const auto type = static_cast<uint8_t>(table_types[feature]);
    const std::array<int64_t, 2> count{c0, c1};
    const std::array<int64_t, 2> present{c0 > 0, c1 > 0};
    const uint64_t physical = static_cast<uint64_t>(key) |
                              (static_cast<uint64_t>(registered_tag) << 56);
    // Identical to base/hash.h::GetHash on the encoded uint64 key.
    const int owner = static_cast<int>(
        ::CityHash64(reinterpret_cast<const char*>(&physical),
                     sizeof(physical)) %
        2);
    if (include_forward) {
      if (owner == rank) {
        for (int destination = 0; destination < 2; ++destination)
          if (present[destination])
            forward_sends[destination].push_back(result.keys.size());
      }
      if (present[rank])
        forward_local_positions[owner].push_back(local_row++);
    }
    if (owner == rank) {
      result.keys.push_back(key);
      const auto& prefix = destination_rows[owner];
      result.meta.insert(
          result.meta.end(),
          {key, c0, c1, type, prefix[0], present[0], prefix[1], present[1]});
    }
    if (present[rank])
      permutations[owner].push_back(local_position++);
    for (int source = 0; source < 2; ++source) {
      destination_rows[owner][source] += present[source];
      result.rows[source] += present[source];
    }
  }
  result.send = {destination_rows[0][rank], destination_rows[1][rank]};
  result.recv = destination_rows[rank];
  result.permutation.reserve(static_cast<size_t>(rank == 0 ? n0 : n1));
  for (const auto& part : permutations)
    result.permutation.insert(
        result.permutation.end(), part.begin(), part.end());
  if (static_cast<int64_t>(result.permutation.size()) != result.rows[rank] ||
      result.rows[0] != n0 || result.rows[1] != n1 ||
      result.send[0] + result.send[1] != result.rows[rank])
    throw std::logic_error(
        "Merged owner permutation does not cover local rows");
  if (include_forward) {
    if (local_row != (rank == 0 ? n0 : n1))
      throw std::logic_error(
          "Owner forward plan does not cover local unique keys");
    result.forward_reorder.resize(static_cast<size_t>(local_row));
    int64_t received_row = 0;
    for (int peer = 0; peer < 2; ++peer) {
      result.forward_send[peer] = forward_sends[peer].size();
      result.forward_recv[peer] = forward_local_positions[peer].size();
      result.forward_send_index.insert(
          result.forward_send_index.end(),
          forward_sends[peer].begin(),
          forward_sends[peer].end());
      for (const auto position : forward_local_positions[peer])
        result.forward_reorder[position] = received_row++;
    }
  }
  return result;
}

} // namespace recstore
