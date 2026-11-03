#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace recstore {

// Integer-only canonical ticket and feature-major inverse. The caller owns
// six non-overlapping output buffers, disjoint from inputs, each with
// capacity features*batch int64 values.
// Return the number of unique keys; positions and inverse use full capacity.
inline int64_t BuildCanonicalSparseTicket(
    const int64_t* ids,
    const int64_t* offsets,
    const int64_t* types,
    int64_t features,
    int64_t batch,
    int64_t* keys,
    int64_t* counts,
    int64_t* positions,
    int64_t* starts,
    int64_t* key_types,
    int64_t* inverse) {
  if (features < 1 || features > 26 || batch < 0 || batch > 1024)
    throw std::invalid_argument("Invalid fixed-workload dimensions");
  if (!offsets || !types ||
      (batch && (!ids || !keys || !counts || !positions || !starts ||
                 !key_types || !inverse)))
    throw std::invalid_argument("Null canonical ticket buffer");
  const auto maximum = std::numeric_limits<int64_t>::max();
  for (int64_t feature = 0; feature < features; ++feature) {
    if (offsets[feature] < 0 ||
        (feature && offsets[feature] <= offsets[feature - 1]) ||
        (types[feature] != 0 && types[feature] != 1))
      throw std::invalid_argument("Invalid ordered namespace or table type");
    for (int64_t sample = 0; sample < batch; ++sample) {
      const auto raw = ids[feature * batch + sample];
      if (raw < 0 || raw > maximum - offsets[feature])
        throw std::invalid_argument("Negative ID or fused int64 overflow");
      if (feature + 1 < features &&
          raw + offsets[feature] >= offsets[feature + 1])
        throw std::invalid_argument("Feature namespace overlap");
    }
  }
  if (!batch)
    return 0;
  struct Entry {
    uint64_t key;
    int64_t original;
  };
  const auto size = features * batch;
  std::vector<Entry> a(size), b(size);
  uint64_t largest = 0;
  for (int64_t feature = 0; feature < features; ++feature) {
    for (int64_t sample = 0; sample < batch; ++sample) {
      const auto original = feature * batch + sample;
      const auto key = static_cast<uint64_t>(ids[original] + offsets[feature]);
      a[original]    = {key, original};
      largest        = std::max(largest, key);
    }
  }
  unsigned passes = 1;
  for (auto remaining = largest >> 8; remaining; remaining >>= 8)
    ++passes;
  for (unsigned pass = 0; pass < passes; ++pass) {
    const auto shift = pass * 8;
    std::array<int64_t, 256> next{};
    for (const auto& entry : a)
      ++next[(entry.key >> shift) & 255];
    int64_t total = 0;
    for (auto& bucket : next) {
      const auto count = bucket;
      bucket           = total;
      total += count;
    }
    // Forward scatter preserves original occurrence order for all equal keys.
    for (const auto& entry : a)
      b[next[(entry.key >> shift) & 255]++] = entry;
    a.swap(b);
  }
  int64_t unique = 0;
  for (int64_t at = 0; at < size; ++at) {
    const auto key      = static_cast<int64_t>(a[at].key);
    const auto original = a[at].original;
    const auto feature  = original / batch;
    if (at == 0 || key != static_cast<int64_t>(a[at - 1].key)) {
      keys[unique]      = key;
      counts[unique]    = 0;
      starts[unique]    = at;
      key_types[unique] = types[feature];
      ++unique;
    }
    ++counts[unique - 1];
    positions[at]     = (original % batch) * features + feature;
    inverse[original] = unique - 1;
  }
  return unique;
}

} // namespace recstore
