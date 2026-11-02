#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "city.h"
#include "ps/base/canonical_sparse_gradient.h"

namespace recstore {

struct CanonicalOwnerPlan {
  std::vector<int64_t> permutation;  // Local canonical vectors, destination 0 then 1.
  std::vector<int64_t> meta;         // [key,c0,c1,type,offset0,nvec0,offset1,nvec1].
  std::vector<int64_t> keys;         // Logical keys owned by this rank, ascending.
  std::array<int64_t, 2> send{};     // Vector rows by destination rank.
  std::array<int64_t, 2> recv{};     // Vector rows by source rank.
  std::vector<int64_t> forward_send_index;
  std::vector<int64_t> forward_reorder;
  std::array<int64_t, 2> forward_send{};
  std::array<int64_t, 2> forward_recv{};
  std::array<int64_t, 2> vectors{};  // Original packed vector counts by source rank.
};

namespace canonical_owner_detail {
inline int64_t ValidateInput(int64_t n, const int64_t* keys,
                             const int64_t* counts, int64_t features,
                             const int64_t* boundaries) {
  if (n < 0 || n > 26 * 1024 || (n && (!keys || !counts)))
    throw std::invalid_argument("Invalid canonical owner key/count buffers");
  std::array<int64_t, 26> feature_counts{};
  int64_t feature = 0, total = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (keys[i] < boundaries[0] || keys[i] >= boundaries[features] ||
        (i && keys[i] <= keys[i - 1]) || counts[i] < 1 || counts[i] > 1024)
      throw std::invalid_argument("Invalid sorted logical owner keys/counts");
    while (feature + 1 < features && keys[i] >= boundaries[feature + 1])
      ++feature;
    feature_counts[feature] += counts[i];
    if (feature_counts[feature] > 1024)
      throw std::invalid_argument("Canonical owner feature exceeds batch bound");
    total += counts[i];
  }
  for (int64_t feature_id = 1; feature_id < features; ++feature_id)
    if (feature_counts[feature_id] != feature_counts[0])
      throw std::invalid_argument("Canonical owner feature occurrence totals differ");
  return total;
}
}  // namespace canonical_owner_detail

// Fixed deployment: two city_hash shards, each colocated with the same rank.
// boundaries has features+1 entries, including the exclusive final row bound.
// registered_tag is the actual backend namespace, not a table index guess.
// Both offsets in meta are relative to their separate source tensor. A missing
// source has zero vectors and retains its current nonnegative prefix offset.
inline CanonicalOwnerPlan BuildCanonicalOwnerPlan(
    int64_t n0, const int64_t* keys0, const int64_t* counts0,
    int64_t n1, const int64_t* keys1, const int64_t* counts1,
    int64_t features, const int64_t* table_boundaries,
    const int64_t* table_types, int64_t registered_tag, int64_t rank,
    bool include_forward = false) {
  constexpr uint64_t kLogicalLimit = uint64_t{1} << 56;
  if (features < 1 || features > 26 || !table_boundaries || !table_types ||
      registered_tag < 0 || registered_tag > 255 || rank < 0 || rank > 1)
    throw std::invalid_argument("Invalid canonical owner layout/tag/rank");
  if (table_boundaries[0] < 0 ||
      table_boundaries[features] > static_cast<int64_t>(kLogicalLimit))
    throw std::invalid_argument("Canonical owner layout overlaps namespace bits");
  for (int64_t feature = 0; feature < features; ++feature)
    if (table_boundaries[feature + 1] <= table_boundaries[feature] ||
        table_types[feature] < 0 || table_types[feature] > 1)
      throw std::invalid_argument("Invalid canonical owner table boundary/type");
  const std::array<int64_t, 2> occurrences{
      canonical_owner_detail::ValidateInput(n0, keys0, counts0, features, table_boundaries),
      canonical_owner_detail::ValidateInput(n1, keys1, counts1, features, table_boundaries)};

  CanonicalOwnerPlan result;
  std::array<std::vector<int64_t>, 2> permutations;
  std::array<std::vector<int64_t>, 2> forward_sends, forward_local_positions;
  int64_t local_row = 0;
  std::array<std::array<int64_t, 2>, 2> destination_vectors{};
  result.keys.reserve(static_cast<size_t>(n0 + n1));
  result.meta.reserve(static_cast<size_t>(n0 + n1) * 8);
  int64_t i0 = 0, i1 = 0, feature = 0;
  while (i0 < n0 || i1 < n1) {
    const int64_t key = i1 == n1 || (i0 < n0 && keys0[i0] < keys1[i1])
        ? keys0[i0] : keys1[i1];
    const int64_t c0 = i0 < n0 && keys0[i0] == key ? counts0[i0++] : 0;
    const int64_t c1 = i1 < n1 && keys1[i1] == key ? counts1[i1++] : 0;
    while (feature + 1 < features && key >= table_boundaries[feature + 1])
      ++feature;
    const auto type = static_cast<uint8_t>(table_types[feature]);
    const std::array<int64_t, 2> count{c0, c1};
    const std::array<int64_t, 2> vectors{
        CanonicalExpectedVectors(c0, c1, type, 0),
        CanonicalExpectedVectors(c0, c1, type, 1)};
    for (int source = 0; source < 2; ++source)
      if (vectors[source] > count[source] || vectors[source] > 64)
        throw std::logic_error("Canonical owner vector count violates protocol bounds");
    const uint64_t physical = static_cast<uint64_t>(key) |
        (static_cast<uint64_t>(registered_tag) << 56);
    // Identical to base/hash.h::GetHash on the encoded uint64 key.
    const int owner = static_cast<int>(
        ::CityHash64(reinterpret_cast<const char*>(&physical), sizeof(physical)) % 2);
    if (include_forward) {
      if (owner == rank) {
        for (int destination = 0; destination < 2; ++destination)
          if (count[destination])
            forward_sends[destination].push_back(result.keys.size());
      }
      if (count[rank])
        forward_local_positions[owner].push_back(local_row++);
    }
    if (owner == rank) {
      result.keys.push_back(key);
      const auto& prefix = destination_vectors[owner];
      result.meta.insert(result.meta.end(),
          {key, c0, c1, type, prefix[0], vectors[0], prefix[1], vectors[1]});
    }
    auto& permutation = permutations[owner];
    for (int64_t vector = 0; vector < vectors[rank]; ++vector)
      permutation.push_back(result.vectors[rank] + vector);
    for (int source = 0; source < 2; ++source) {
      destination_vectors[owner][source] += vectors[source];
      result.vectors[source] += vectors[source];
    }
  }
  for (int source = 0; source < 2; ++source)
    if (result.vectors[source] > occurrences[source])
      throw std::logic_error("Canonical owner packed vector bound exceeded");
  result.send = {destination_vectors[0][rank], destination_vectors[1][rank]};
  result.recv = destination_vectors[rank];
  result.permutation.reserve(static_cast<size_t>(result.vectors[rank]));
  for (const auto& part : permutations)
    result.permutation.insert(result.permutation.end(), part.begin(), part.end());
  if (static_cast<int64_t>(result.permutation.size()) != result.vectors[rank])
    throw std::logic_error("Canonical owner permutation does not cover local vectors");
  if (include_forward) {
    if (local_row != (rank == 0 ? n0 : n1))
      throw std::logic_error("Owner forward plan does not cover local unique keys");
    result.forward_reorder.resize(static_cast<size_t>(local_row));
    int64_t received_row = 0;
    for (int peer = 0; peer < 2; ++peer) {
      result.forward_send[peer] = forward_sends[peer].size();
      result.forward_recv[peer] = forward_local_positions[peer].size();
      result.forward_send_index.insert(result.forward_send_index.end(),
          forward_sends[peer].begin(), forward_sends[peer].end());
      for (const auto position : forward_local_positions[peer])
        result.forward_reorder[position] = received_row++;
    }
  }
  return result;
}

}  // namespace recstore
