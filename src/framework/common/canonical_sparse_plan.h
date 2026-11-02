#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace recstore {

constexpr int64_t kCanonicalSparsePlanMaxPositions = 26 * 1024;

struct CanonicalSparsePlanSizes {
  int64_t tasks;
  int64_t vectors;
};

// Integer-only canonical-v3 plan; no gradient arithmetic or namespace changes.
// The caller provides records[n * 5] and tasks[positions_n * 6]. Every task
// consumes at least one distinct local occurrence, so this capacity suffices.
inline CanonicalSparsePlanSizes BuildCanonicalSparsePlan(
    int64_t n, const int64_t* keys, const int64_t* counts,
    const int64_t* starts, const int64_t* types,
    int64_t peer_n, const int64_t* peer_keys, const int64_t* peer_counts,
    int64_t rank, int64_t positions_n, int64_t* records, int64_t* tasks) {
  if ((rank != 0 && rank != 1) || n < 0 || n > kCanonicalSparsePlanMaxPositions ||
      peer_n < 0 || peer_n > kCanonicalSparsePlanMaxPositions || positions_n < 0 ||
      positions_n > kCanonicalSparsePlanMaxPositions)
    throw std::invalid_argument("Invalid rank or fixed-workload bounds");
  if ((n && (!keys || !counts || !starts || !types || !records)) ||
      (peer_n && (!peer_keys || !peer_counts)) || (positions_n && !tasks))
    throw std::invalid_argument("Null canonical plan buffer");
  for (int64_t j = 0; j < peer_n; ++j) {
    if (peer_keys[j] < 0 || (j && peer_keys[j] <= peer_keys[j-1]) ||
        peer_counts[j] < 1 || peer_counts[j] > 1024)
      throw std::invalid_argument("Invalid sorted peer key/count payload");
  }
  int64_t occurrence_count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (keys[i] < 0 || (i && keys[i] <= keys[i-1]) ||
        counts[i] < 1 || counts[i] > 1024 || starts[i] != occurrence_count ||
        (types[i] != 0 && types[i] != 1))
      throw std::invalid_argument("Invalid local ticket");
    occurrence_count += counts[i];
  }
  if (occurrence_count != positions_n)
    throw std::invalid_argument("Local counts do not cover positions");
  int64_t next_peer = 0, next_task = 0, next_vector = 0;
  for (int64_t i = 0; i < n; ++i) {
    while (next_peer < peer_n && peer_keys[next_peer] < keys[i]) ++next_peer;
    const int64_t other = next_peer < peer_n && peer_keys[next_peer] == keys[i]
        ? peer_counts[next_peer] : 0;
    const int64_t c0 = rank == 0 ? counts[i] : other;
    const int64_t c1 = rank == 1 ? counts[i] : other;
    const int64_t first_vector = next_vector;
    auto emit = [&](int64_t kind, int64_t start, int64_t count) {
      // Every task consumes at least one distinct local occurrence.
      if (next_task >= positions_n || count <= 0 || start < 0 ||
          start + count > counts[i])
        throw std::logic_error("Invalid task capacity or occurrence interval");
      int64_t* task = tasks + 6 * next_task++;
      task[0] = kind;
      task[1] = starts[i] + start;
      task[2] = count;
      task[3] = next_vector;
      task[4] = types[i] == 0;
      task[5] = types[i] == 1;
      next_vector += kind == 2 ? count : 1;
    };
    if (types[i] == 1) {
      emit(counts[i] < 32 ? 0 : 1, 0, counts[i]);
    } else if (c0 + c1 < 32) {
      emit(rank == 0 || c0 == 0 ? 0 : 2, 0, counts[i]);
    } else {
      const int64_t total = c0 + c1;
      const int64_t own_start = rank == 0 ? 0 : c0;
      const int64_t own_end = rank == 0 ? c0 : total;
      for (int64_t base = 0; base < total; base += 1024) {
        const int64_t end = std::min(base + 1024, total);
        if (own_start <= base && end <= own_end) {
          emit(1, base - own_start, end - base);
          continue;
        }
        const int64_t width = (end - base + 31) / 32;
        for (int64_t leaf = 0; leaf < 32; ++leaf) {
          const int64_t lo = base + leaf * width;
          const int64_t hi = std::min(base + (leaf + 1) * width, end);
          const int64_t start = std::max(lo, own_start);
          const int64_t stop = std::min(hi, own_end);
          if (start >= stop) continue;
          emit(lo < c0 && c0 < hi && rank == 1 ? 2 : 0,
               start - own_start, stop - start);
        }
      }
    }
    int64_t* record = records + 5 * i;
    record[0] = keys[i];
    record[1] = c0;
    record[2] = c1;
    record[3] = next_vector - first_vector;
    record[4] = types[i];
  }
  return {next_task, next_vector};
}

} // namespace recstore
