#include <gtest/gtest.h>
#include <array>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include "base/sgd_update.h"
#include "framework/common/owner_merged_plan.h"
#include "ps/base/sparse_gradient_commit.h"
#include "framework/common/sparse_gradient_commit_runtime.h"

using recstore::PackSparseGradientCommit;
using recstore::SparseGradientCommitCoordinator;
using recstore::SparseGradientCommitHeader;

TEST(SparseGradientCommit, ChunksEmptyRankAndRetryCommitExactlyOnce) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session = 7;
  h.world   = 3;
  h.dim     = 2;
  h.divisor = 3;
  std::vector<uint64_t> applied_keys;
  std::vector<float> applied;
  int calls  = 0;
  auto apply = [&](const auto& keys, const auto& grads) {
    applied_keys = keys;
    applied      = grads;
    ++calls;
  };
  const uint64_t key = 9;
  const float g0[] = {3, 6}, g1[] = {6, 3};
  h.rows     = 1;
  h.chunks   = 2;
  auto first = PackSparseGradientCommit(h, &key, g0);
  EXPECT_EQ(coordinator.Exchange("table", first, apply), -1);
  EXPECT_EQ(coordinator.Exchange("table", first, apply), -1);
  EXPECT_THROW(coordinator.Exchange(
                   "table", PackSparseGradientCommit(h, &key, g1), apply),
               std::runtime_error);
  h.chunk = 1;
  h.rows  = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            -1);
  h.rank   = 1;
  h.chunk  = 0;
  h.chunks = 1;
  h.rows   = 1;
  EXPECT_EQ(coordinator.Exchange(
                "table", PackSparseGradientCommit(h, &key, g1), apply),
            -1);
  h.rank = 2;
  h.rows = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            0);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(applied_keys, std::vector<uint64_t>({9}));
  EXPECT_EQ(applied, std::vector<float>({3, 3}));
  EXPECT_EQ(coordinator.Exchange("table", first, apply), 0);
  EXPECT_EQ(calls, 1);
  h.kind = 1;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            0);
  h.kind = 0;
  h.step = 2;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.step = 1;
  for (h.rank = 0; h.rank < h.world; ++h.rank) {
    EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
              h.rank == 2 ? 1 : 0);
  }
  EXPECT_EQ(calls, 1); // All-empty steps still advance the commit frontier.
  h.kind = 2;
  for (h.rank = 0; h.rank < h.world; ++h.rank)
    coordinator.Exchange("table", PackSparseGradientCommit(h), apply);
  h.kind    = 0;
  h.step    = 0;
  h.session = 8;
  h.rank    = 0;
  EXPECT_EQ(coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
            -1);
}

TEST(SparseGradientCommit, FailedCommitCannotBeReplayed) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session    = 4;
  h.dim        = 1;
  h.rows       = 1;
  uint64_t key = 1;
  float grad   = 2;
  auto packet  = PackSparseGradientCommit(h, &key, &grad);
  int calls    = 0;
  auto fail    = [&](const auto&, const auto&) {
    ++calls;
    throw std::runtime_error("storage failed");
  };
  EXPECT_THROW(coordinator.Exchange("table", packet, fail), std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", packet, fail), std::runtime_error);
  EXPECT_EQ(calls, 1);
  h.rows = 0;
  h.kind = 1;
  EXPECT_THROW(coordinator.Exchange("table", PackSparseGradientCommit(h), fail),
               std::runtime_error);
  EXPECT_THROW(recstore::DecodeSparseGradientCommit(packet.substr(1)),
               std::invalid_argument);
  h.world = 0;
  EXPECT_THROW(PackSparseGradientCommit(h), std::invalid_argument);
}

TEST(SparseGradientCommit, ArrivalOrderAndRetriesPreserveAccumulationOrder) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session               = 12;
  h.world                 = 2;
  h.dim                   = 1;
  h.rows                  = 1;
  h.chunks                = 2;
  const uint64_t keys[]   = {9, 9, 9, 4};
  const float gradients[] = {1.0e20f, -1.0e20f, 3.0f, 5.0f};
  const int arrivals[][4] = {{3, 1, 0, 2}, {2, 0, 3, 1}};
  int calls               = 0;
  auto apply              = [&](const auto& actual_keys, const auto& grads) {
    ++calls;
    EXPECT_EQ(actual_keys, std::vector<uint64_t>({9, 4}));
    EXPECT_EQ(grads, std::vector<float>({3, 5}));
  };
  for (h.step = 0; h.step < 2; ++h.step) {
    std::vector<std::string> packets;
    for (int index = 0; index < 4; ++index) {
      h.rank  = index / 2;
      h.chunk = index % 2;
      packets.push_back(
          PackSparseGradientCommit(h, keys + index, gradients + index));
    }
    for (int received = 0; received < 4; ++received) {
      const auto& packet  = packets[arrivals[h.step][received]];
      const auto expected = received == 3 ? h.step : h.step - 1;
      EXPECT_EQ(coordinator.Exchange("table", packet, apply), expected);
      EXPECT_EQ(coordinator.Exchange("table", packet, apply), expected);
      EXPECT_EQ(calls, h.step + (received == 3));
    }
  }
}

TEST(SparseGradientCommit, ChangedChunkCountsAndEmptyStepsResetAggregation) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session  = 13;
  h.world    = 2;
  h.dim      = 1;
  h.divisor  = 2;
  int calls  = 0;
  auto apply = [&](const auto& keys, const auto& grads) {
    ++calls;
    EXPECT_EQ(keys, std::vector<uint64_t>({h.step == 0 ? 5u : 9u}));
    EXPECT_EQ(grads, std::vector<float>({h.step == 0 ? 7.0f : 2.0f}));
  };
  for (h.step = 0; h.step < 3; ++h.step) {
    for (h.rank = 0; h.rank < h.world; ++h.rank) {
      h.chunks = (h.step + h.rank) % 2 + 1;
      for (h.chunk = 0; h.chunk < h.chunks; ++h.chunk) {
        h.rows             = h.step != 1 && h.chunk == 0 ? 1 : 0;
        const uint64_t key = h.step == 0 ? 5 : 9;
        const float grad   = h.step == 0 ? 7 : 2;
        auto packet        = PackSparseGradientCommit(h, &key, &grad);
        const bool last    = h.rank + 1 == h.world && h.chunk + 1 == h.chunks;
        EXPECT_EQ(coordinator.Exchange("table", std::move(packet), apply),
                  last ? h.step : h.step - 1);
      }
    }
    EXPECT_EQ(calls, h.step == 0 ? 1 : h.step);
  }
}

TEST(SparseGradientCommit, AbortRejectsPendingCommitAndQueries) {
  SparseGradientCommitCoordinator coordinator;
  SparseGradientCommitHeader h;
  h.session          = 14;
  h.world            = 2;
  h.dim              = 1;
  h.rows             = 1;
  const uint64_t key = 3;
  const float grad   = 1;
  int calls          = 0;
  auto apply         = [&](const auto&, const auto&) { ++calls; };
  const auto packet  = PackSparseGradientCommit(h, &key, &grad);
  EXPECT_EQ(coordinator.Exchange("table", packet, apply), -1);
  h.rows = 0;
  h.kind = 3;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.kind = 0;
  h.rank = 1;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", packet, apply),
               std::runtime_error);
  h.kind = 1;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  h.kind = 2;
  EXPECT_THROW(
      coordinator.Exchange("table", PackSparseGradientCommit(h), apply),
      std::runtime_error);
  EXPECT_EQ(calls, 0);
}

namespace {
struct GradientRows {
  std::vector<uint64_t> keys;
  std::vector<float> grads;
};

std::vector<std::string>
MakePackets(const std::vector<std::vector<GradientRows>>& ranks,
            uint32_t dim,
            uint32_t divisor,
            uint32_t version = 1) {
  SparseGradientCommitHeader h;
  h.version = version;
  h.session = 21;
  h.world   = ranks.size();
  h.dim     = dim;
  h.divisor = divisor;
  std::vector<std::string> packets;
  for (h.rank = 0; h.rank < h.world; ++h.rank) {
    h.chunks = ranks[h.rank].size();
    for (h.chunk = 0; h.chunk < h.chunks; ++h.chunk) {
      const auto& rows = ranks[h.rank][h.chunk];
      h.rows           = rows.keys.size();
      EXPECT_EQ(rows.grads.size(), rows.keys.size() * dim);
      packets.push_back(
          PackSparseGradientCommit(h, rows.keys.data(), rows.grads.data()));
    }
  }
  return packets;
}

struct AggregatedRows {
  std::vector<uint64_t> keys;
  std::vector<float> grads;
  int calls = 0;
};

AggregatedRows
AggregatePackets(const std::vector<std::string>& packets, bool reserve) {
  SparseGradientCommitCoordinator coordinator(reserve);
  AggregatedRows result;
  auto apply = [&](const auto& keys, const auto& grads) {
    result.keys  = keys;
    result.grads = grads;
    ++result.calls;
  };
  for (size_t i = 0; i < packets.size(); ++i) {
    const int64_t expected = i + 1 == packets.size() ? 0 : -1;
    EXPECT_EQ(coordinator.Exchange("table", packets[i], apply), expected);
    // Both in-flight and post-commit duplicates must remain idempotent.
    EXPECT_EQ(coordinator.Exchange("table", packets[i], apply), expected);
  }
  return result;
}

std::map<uint64_t, std::vector<uint32_t>>
RowBits(const AggregatedRows& rows, size_t dim) {
  std::map<uint64_t, std::vector<uint32_t>> result;
  for (size_t row = 0; row < rows.keys.size(); ++row) {
    auto& bits = result[rows.keys[row]];
    bits.resize(dim);
    std::memcpy(bits.data(), rows.grads.data() + row * dim, dim * 4);
  }
  return result;
}
} // namespace

TEST(SparseGradientCommit, ReservePreservesCancellationAndDuplicateOrder) {
  const uint64_t max_key = std::numeric_limits<uint64_t>::max();
  const std::vector<std::vector<GradientRows>> ranks{
      {{{5, 5}, {1.0e20f, -1.0e20f}}, {{}, {}}, {{5, 9}, {3, 1.0e20f}}},
      {{{5, 9}, {1, -1.0e20f}}, {{9, max_key}, {5, -0.0f}}},
      {{{}, {}}, {{}, {}}},
      {{{1, 9}, {8, 7}}},
  };
  auto packets = MakePackets(ranks, 1, 4);
  std::reverse(packets.begin(), packets.end());
  const auto original = AggregatePackets(packets, false);
  const auto reserved = AggregatePackets(packets, true);
  EXPECT_EQ(original.calls, 1);
  EXPECT_EQ(reserved.calls, 1);
  EXPECT_EQ(original.keys, std::vector<uint64_t>({5, 9, max_key, 1}));
  EXPECT_EQ(reserved.keys, original.keys);
  EXPECT_EQ(reserved.grads, std::vector<float>({1, 3, 0, 2}));
  EXPECT_EQ(RowBits(reserved, 1), RowBits(original, 1));
}

TEST(SparseGradientCommit, ReserveHandlesWithinAndAcrossChunkDisorder) {
  for (bool split : {false, true}) {
    std::vector<std::vector<GradientRows>> ranks{
        {{{7, 2}, {3, 5}}}, {{{1, 9}, {7, 11}}}};
    if (split)
      ranks[0] = {{{7}, {3}}, {{}, {}}, {{2}, {5}}};
    const auto packets  = MakePackets(ranks, 1, 2);
    const auto original = AggregatePackets(packets, false);
    const auto reserved = AggregatePackets(packets, true);
    EXPECT_EQ(reserved.keys, original.keys);
    EXPECT_EQ(reserved.keys, std::vector<uint64_t>({7, 2, 1, 9}));
    EXPECT_EQ(RowBits(reserved, 1), RowBits(original, 1));
    EXPECT_EQ(reserved.calls, 1);
  }
}

TEST(SparseGradientCommit, ReserveRandomInputsMatchUnreservedPathBitForBit) {
  std::mt19937 rng(9741);
  const float values[] = {
      1.0e20f, -1.0e20f, 3, -7, 0, -0.0f, 1.0e-20f, -1.0e-20f};
  for (bool sorted : {false, true}) {
    for (int trial = 0; trial < 40; ++trial) {
      SCOPED_TRACE(::testing::Message()
                   << "sorted=" << sorted << " trial=" << trial);
      const uint32_t world   = 1u << (trial % 4);
      const uint32_t dim     = trial % 3 == 0 ? 128 : trial % 3 + 1;
      const uint32_t divisor = trial % 2 == 0 ? 1 : world;
      std::vector<std::vector<GradientRows>> ranks(world);
      for (auto& rank : ranks) {
        std::vector<uint64_t> keys(rng() % 65);
        for (auto& key : keys)
          key = rng() % 19;
        if (sorted)
          std::sort(keys.begin(), keys.end());
        rank.push_back({{}, {}});
        size_t begin = 0;
        while (begin < keys.size()) {
          const size_t end = std::min(keys.size(), begin + 1 + rng() % 7);
          GradientRows chunk;
          chunk.keys.assign(keys.begin() + begin, keys.begin() + end);
          chunk.grads.resize(chunk.keys.size() * dim);
          for (auto& value : chunk.grads)
            value = values[rng() % (sizeof(values) / sizeof(values[0]))];
          rank.push_back(std::move(chunk));
          if (rng() % 3 == 0)
            rank.push_back({{}, {}});
          begin = end;
        }
        rank.push_back({{}, {}});
      }
      auto packets = MakePackets(ranks, dim, divisor);
      std::shuffle(packets.begin(), packets.end(), rng);
      const auto original = AggregatePackets(packets, false);
      const auto reserved = AggregatePackets(packets, true);
      EXPECT_EQ(reserved.calls, original.calls);
      EXPECT_EQ(RowBits(reserved, dim), RowBits(original, dim));
      EXPECT_EQ(reserved.keys, original.keys);
    }
  }
}

TEST(SparseGradientCommit, ReserveEmptyStepAndLargeWorld) {
  std::vector<std::vector<GradientRows>> ranks(2);
  for (auto& rank : ranks)
    rank = {{{}, {}}, {{}, {}}};
  EXPECT_EQ(AggregatePackets(MakePackets(ranks, 1, 2), true).calls, 0);
  ranks.resize(9);
  for (size_t rank = 0; rank < ranks.size(); ++rank)
    ranks[rank] = {{{9 - rank}, {float(rank)}}};
  const auto packets  = MakePackets(ranks, 1, 9);
  const auto original = AggregatePackets(packets, false);
  const auto reserved = AggregatePackets(packets, true);
  EXPECT_EQ(reserved.keys, original.keys);
  EXPECT_EQ(RowBits(reserved, 1), RowBits(original, 1));
}

TEST(SparseGradientCommit, ReserveFailureAndConflictingRetryDoNotReapply) {
  SparseGradientCommitCoordinator coordinator(true);
  SparseGradientCommitHeader h;
  h.session          = 25;
  h.world            = 2;
  h.dim              = 1;
  h.rows             = 1;
  const uint64_t key = 2;
  const float first = 3, conflict = 7;
  const auto first_packet = PackSparseGradientCommit(h, &key, &first);
  int calls               = 0;
  auto fail               = [&](const auto&, const auto&) {
    ++calls;
    throw std::runtime_error("storage failed");
  };
  EXPECT_EQ(coordinator.Exchange("table", first_packet, fail), -1);
  EXPECT_THROW(coordinator.Exchange(
                   "table", PackSparseGradientCommit(h, &key, &conflict), fail),
               std::runtime_error);
  h.rank                 = 1;
  const auto last_packet = PackSparseGradientCommit(h, &key, &first);
  EXPECT_THROW(coordinator.Exchange("table", last_packet, fail),
               std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", last_packet, fail),
               std::runtime_error);
  EXPECT_THROW(coordinator.Exchange("table", first_packet, fail),
               std::runtime_error);
  EXPECT_EQ(calls, 1);
  h.rows = 0;
  for (h.kind = 1; h.kind <= 3; ++h.kind)
    EXPECT_THROW(
        coordinator.Exchange("table", PackSparseGradientCommit(h), fail),
        std::runtime_error);
}

namespace {
std::string ChangeHeader(const std::string& packet,
                         uint32_t version,
                         uint32_t kind = 0,
                         int64_t step  = -1) {
  auto h    = recstore::DecodeSparseGradientCommit(packet);
  h.version = version;
  h.kind    = kind;
  if (step >= 0)
    h.step = step;
  if (kind) {
    h.rows   = 0;
    h.chunk  = 0;
    h.chunks = 1; // Control messages do not repeat submit chunk metadata.
    return PackSparseGradientCommit(h);
  }
  auto result = packet;
  std::memcpy(result.data(), &h, sizeof(h));
  return result;
}

size_t CompareSortedPackets(const std::vector<std::string>& packets,
                            const std::vector<size_t>& order) {
  SparseGradientCommitCoordinator baseline(true), streaming(true);
  AggregatedRows expected, actual;
  const auto dim = recstore::DecodeSparseGradientCommit(packets[0]).dim;
  auto collect   = [&](const auto& keys, const auto& grads) {
    actual.keys.insert(actual.keys.end(), keys.begin(), keys.end());
    actual.grads.insert(actual.grads.end(), grads.begin(), grads.end());
    ++actual.calls;
  };
  size_t early = 0;
  for (size_t i = 0; i < order.size(); ++i) {
    const auto& packet = packets[order[i]];
    baseline.Exchange("t", packet, [&](const auto& keys, const auto& grads) {
      expected.keys  = keys;
      expected.grads = grads;
    });
    const auto v2        = ChangeHeader(packet, 2);
    const auto committed = i + 1 == order.size() ? 0 : -1;
    EXPECT_EQ(streaming.Exchange("t", v2, collect), committed);
    EXPECT_EQ(streaming.Exchange("t", v2, collect), committed);
    EXPECT_EQ(streaming.Exchange("t", ChangeHeader(v2, 2, 1), collect),
              committed);
    if (i + 1 < order.size())
      early = actual.keys.size();
  }
  const auto bits = RowBits(actual, dim);
  EXPECT_EQ(bits.size(), actual.keys.size()); // No key may be applied twice.
  EXPECT_EQ(bits, RowBits(expected, dim));
  return early;
}
} // namespace

TEST(SparseGradientCommit,
     SortedFrontierAllChunkArrivalsPreserveExactArithmetic) {
  const auto max     = std::numeric_limits<uint64_t>::max();
  const auto packets = MakePackets(
      {{{{0, 2}, {-0.f, 1e20f}}, {{2}, {-1e20f}}, {{2, max}, {3.25f, 7}}},
       {{{}, {}}, {{0, 2}, {-1e20f, 1}}, {{2, max}, {4, -7}}}},
      1,
      2);
  std::vector<size_t> order{0, 1, 2, 3, 4, 5};
  size_t early = 0;
  do {
    early += CompareSortedPackets(packets, order);
  } while (std::next_permutation(order.begin(), order.end()));
  EXPECT_GT(early, 0);
}

TEST(SparseGradientCommit, SortedFrontierRandomDuplicatesAndManyChunks) {
  std::mt19937 rng(281);
  const float values[] = {1e20f, -1e20f, 3, -7, 0, -0.f, 1e-20f};
  for (int trial = 0; trial < 20; ++trial) {
    SCOPED_TRACE(trial);
    const uint32_t dim   = trial % 2 ? 3 : 128;
    const uint32_t count = trial == 0 ? 64 : 4 + rng() % 8;
    std::vector<std::vector<GradientRows>> ranks(2);
    for (auto& rank : ranks) {
      std::vector<uint64_t> keys(count * 3);
      for (auto& key : keys)
        key = rng() % 23;
      std::sort(keys.begin(), keys.end());
      for (size_t chunk = 0; chunk < count; ++chunk) {
        GradientRows rows;
        if (chunk % 5) {
          rows.keys.assign(
              keys.begin() + chunk * 3, keys.begin() + chunk * 3 + 3);
          rows.grads.resize(rows.keys.size() * dim);
          for (auto& grad : rows.grads)
            grad = values[rng() % 7];
        }
        rank.push_back(std::move(rows));
      }
    }
    const auto packets = MakePackets(ranks, dim, trial % 2 ? 1 : 2);
    std::vector<size_t> order(packets.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    CompareSortedPackets(packets, order);
  }
}

TEST(SparseGradientCommit, SortedFrontierUnknownEmptyPrefixAndMaxKey) {
  const auto max = std::numeric_limits<uint64_t>::max();
  auto packets   = MakePackets({{{{max}, {3}}}, {{{}, {}}, {{}, {}}}}, 1, 2, 2);
  SparseGradientCommitCoordinator coordinator;
  int calls  = 0;
  auto apply = [&](const auto& keys, const auto& grads) {
    ++calls;
    EXPECT_EQ(keys, std::vector<uint64_t>{max});
    EXPECT_EQ(grads, std::vector<float>{1.5f});
  };
  EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), -1);
  EXPECT_EQ(coordinator.Exchange("t", packets[1], apply), -1);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(coordinator.Exchange("t", packets[2], apply), 0);
  EXPECT_EQ(calls, 1);
  auto empty = MakePackets({{{{}, {}}}, {{{}, {}}}}, 1, 2);
  EXPECT_EQ(CompareSortedPackets(empty, {1, 0}), 0);
}

TEST(SparseGradientCommit,
     SortedFrontierPrefixFailureIsStickyAndReadGateStaysClosed) {
  for (int mode = 0; mode < 10; ++mode) {
    SCOPED_TRACE(mode);
    auto packets = MakePackets(
        {{{{1, 5}, {1, 2}}, {{8}, {3}}}, {{{2, 6}, {4, 5}}, {{9}, {6}}}},
        1,
        2,
        2);
    SparseGradientCommitCoordinator coordinator;
    size_t applied = 0;
    auto apply     = [&](const auto& keys, const auto&) {
      applied += keys.size();
      if (mode == 0)
        throw std::runtime_error("partial storage apply");
    };
    EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), -1);
    if (mode == 0) {
      EXPECT_THROW(coordinator.Exchange("t", packets[2], apply),
                   std::runtime_error);
    } else {
      EXPECT_EQ(coordinator.Exchange("t", packets[2], apply), -1);
      EXPECT_EQ(applied, 2);
      EXPECT_EQ(
          coordinator.Exchange("t", ChangeHeader(packets[0], 2, 1), apply), -1);
      std::string bad;
      if (mode == 1)
        bad = MakePackets({{{{1}, {3}}}, {{{}, {}}}}, 1, 2, 2)[0];
      if (mode == 2) {
        bad = packets[0];
        bad.back() ^= 1;
      }
      if (mode == 3)
        bad = ChangeHeader(packets[1], 1);
      if (mode == 4)
        bad = ChangeHeader(packets[0], 2, 3);
      if (mode == 5)
        bad = "truncated";
      if (mode == 6) {
        bad          = packets[1];
        uint64_t key = 0;
        std::memcpy(bad.data() + sizeof(SparseGradientCommitHeader), &key, 8);
      }
      if (mode == 7) {
        bad             = packets[1];
        auto h          = recstore::DecodeSparseGradientCommit(bad);
        h.learning_rate = 0.5f;
        std::memcpy(bad.data(), &h, sizeof(h));
      }
      if (mode == 8)
        bad = ChangeHeader(packets[1], 2, 0, 2);
      if (mode == 9)
        bad = MakePackets(
            {{{{1, 5}, {1, 2}}, {{9, 8}, {3, 4}}}, {{{}, {}}}}, 1, 2, 2)[1];
      EXPECT_THROW(coordinator.Exchange("t", bad, apply), std::exception);
    }
    const auto before = applied;
    EXPECT_THROW(coordinator.Exchange("t", packets[0], apply),
                 std::runtime_error);
    EXPECT_THROW(coordinator.Exchange("t", packets[3], apply),
                 std::runtime_error);
    for (uint32_t kind = 1; kind <= 3; ++kind)
      EXPECT_THROW(
          coordinator.Exchange("t", ChangeHeader(packets[0], 2, kind), apply),
          std::runtime_error);
    EXPECT_EQ(applied, before);
  }
}

TEST(SparseGradientCommit, SortedFrontierBoundsAreCheckedBeforeAnyApply) {
  const auto packets = MakePackets({{{{1}, {2}}}, {{{2}, {3}}}}, 1, 2, 2);
  for (int mode = 0; mode < 4; ++mode) {
    SparseGradientCommitCoordinator coordinator;
    int calls  = 0;
    auto apply = [&](const auto&, const auto&) { ++calls; };
    EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), -1);
    auto bad = packets[1];
    auto h   = recstore::DecodeSparseGradientCommit(bad);
    if (mode == 0)
      h.chunks = 65;
    if (mode == 1)
      h.world = 3;
    if (mode == 2) {
      h.rows = 100000;
      bad.resize(sizeof(h) + h.rows * 12);
    }
    if (mode == 3)
      h.version = 1;
    std::memcpy(bad.data(), &h, sizeof(h));
    EXPECT_THROW(coordinator.Exchange("t", bad, apply), std::exception);
    EXPECT_THROW(coordinator.Exchange("t", packets[1], apply),
                 std::runtime_error);
    EXPECT_EQ(calls, 0);
  }
}

TEST(SparseGradientCommit, SortedFrontierNextStepCloseAndReopen) {
  auto packets = MakePackets({{{{1}, {2}}}, {{{}, {}}}}, 1, 2, 2);
  SparseGradientCommitCoordinator coordinator;
  int calls  = 0;
  auto apply = [&](const auto&, const auto&) { ++calls; };
  EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), -1);
  EXPECT_THROW(coordinator.Exchange("t", ChangeHeader(packets[0], 2, 2), apply),
               std::runtime_error);
  EXPECT_EQ(coordinator.Exchange("t", packets[1], apply), 0);
  EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), 0);
  EXPECT_EQ(calls, 1);
  coordinator.Exchange("t", ChangeHeader(packets[0], 2, 0, 1), apply);
  EXPECT_THROW(
      coordinator.Exchange("t", ChangeHeader(packets[0], 2, 2, 0), apply),
      std::runtime_error);
  coordinator.Exchange("t", ChangeHeader(packets[1], 2, 0, 1), apply);
  EXPECT_EQ(calls, 2);
  for (const auto& packet : packets)
    EXPECT_EQ(coordinator.Exchange("t", ChangeHeader(packet, 2, 2, 1), apply),
              1);
  EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), -1);
  EXPECT_EQ(coordinator.Exchange("t", packets[1], apply), 0);
  EXPECT_EQ(calls, 3);
}

TEST(SparseGradientCommit, SortedStreamHeaderOptInKeepsWorldOneOnV1) {
  const char* value  = std::getenv("RECSTORE_SPARSE_COMMIT_SORTED_STREAM");
  const bool existed = value != nullptr;
  const std::string original = value ? value : "";
  for (const char* enabled : {"0", "1"}) {
    setenv("RECSTORE_SPARSE_COMMIT_SORTED_STREAM", enabled, 1);
    for (int world : {1, 2, 3})
      for (int kind = 0; kind <= 3; ++kind) {
        const auto h = recstore::MakeSparseGradientCommitHeader(
            7, 0, 0, world, 1, .01, world, kind);
        EXPECT_EQ(h.version, world == 2 && enabled[0] == '1' ? 2 : 1);
      }
  }
  if (existed)
    setenv("RECSTORE_SPARSE_COMMIT_SORTED_STREAM", original.c_str(), 1);
  else
    unsetenv("RECSTORE_SPARSE_COMMIT_SORTED_STREAM");
}

TEST(SparseGradientCommit, SortedFrontierCannotStartStepAfterAnyRankClosed) {
  const auto packets = MakePackets({{{{1}, {2}}}, {{{}, {}}}}, 1, 2, 2);
  SparseGradientCommitCoordinator coordinator;
  int calls  = 0;
  auto apply = [&](const auto&, const auto&) { ++calls; };
  for (const auto& packet : packets)
    coordinator.Exchange("t", packet, apply);
  EXPECT_EQ(coordinator.Exchange("t", ChangeHeader(packets[0], 2, 2), apply),
            0);
  EXPECT_EQ(coordinator.Exchange("t", packets[0], apply), 0);
  EXPECT_THROW(
      coordinator.Exchange("t", ChangeHeader(packets[1], 2, 0, 1), apply),
      std::runtime_error);
  EXPECT_EQ(calls, 1);
}

TEST(SparseGradientCommit, SortedFailureHookIsScopedToTableAndSession) {
  auto packets = MakePackets({{{{1}, {2}}}, {{{2}, {3}}}}, 1, 2, 2);
  SparseGradientCommitCoordinator coordinator;
  int calls  = 0;
  auto apply = [&](const auto&, const auto&) { ++calls; };
  coordinator.Exchange("a", packets[0], apply);
  coordinator.Exchange("b", packets[0], apply);
  auto other = packets[1];
  auto h     = recstore::DecodeSparseGradientCommit(other);
  ++h.session;
  h.chunks = 65;
  std::memcpy(other.data(), &h, sizeof(h));
  coordinator.FailSortedSession("a", other);
  EXPECT_THROW(coordinator.Exchange("a", other, apply), std::invalid_argument);
  EXPECT_EQ(coordinator.Exchange("a", ChangeHeader(packets[0], 2, 1), apply),
            -1);
  coordinator.FailSortedSession("a", packets[0]);
  EXPECT_THROW(coordinator.Exchange("a", packets[1], apply),
               std::runtime_error);
  EXPECT_EQ(coordinator.Exchange("b", packets[1], apply), 0);
  EXPECT_EQ(calls, 1);
  coordinator.Exchange("v1", ChangeHeader(packets[0], 1), apply);
  coordinator.FailSortedSession("v1", packets[0]);
  EXPECT_EQ(coordinator.Exchange("v1", ChangeHeader(packets[1], 1), apply), 0);
}

TEST(SparseGradientCommit, OptionalFmaPreservesOffAndMatchesScalarFmaBitwise) {
  std::mt19937 rng(90271);
  const float edges[] = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      1e20f,
      -1e20f,
      1e-20f,
      -1e-20f,
      std::numeric_limits<float>::denorm_min()};
  for (size_t count : {0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129}) {
    for (bool initialize : {false, true}) {
      for (float lr : {0.0f, 0.01f, 0.1f, 1.0f}) {
        for (int trial = 0; trial < 30; ++trial) {
          std::vector<float> values(count + 2, 7), grads(count + 2, 9);
          for (size_t i = 1; i <= count; ++i) {
            auto finite = [&] {
              uint32_t bits = rng();
              bits = (bits & 0x807fffffu) | (uint32_t(rng() % 254) << 23);
              float value;
              std::memcpy(&value, &bits, 4);
              return value;
            };
            values[i] = trial % 2 ? finite() : edges[rng() % 9];
            grads[i]  = trial % 2 ? finite() : edges[rng() % 9];
          }
          auto expected = values, actual = values, scalar = values;
          for (size_t i = 1; i <= count; ++i) {
            if (recstore::SparseCommitFmaEnabled())
              expected[i] =
                  std::fma(-lr, grads[i], initialize ? 0.f : values[i]);
            else if (initialize)
              expected[i] = -lr * grads[i];
            else {
              volatile float product = lr * grads[i];
              expected[i]            = values[i] - product;
            }
          }
          recstore::ApplySgdRow(
              actual.data() + 1, grads.data() + 1, count, lr, initialize);
          EXPECT_EQ(
              std::memcmp(actual.data(), expected.data(), actual.size() * 4),
              0);
          recstore::SgdFmaScalar(
              scalar.data() + 1, grads.data() + 1, count, lr, initialize);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
          if (recstore::SgdFmaAvx2Supported()) {
            auto vectorized = values;
            recstore::SgdFmaAvx2(
                vectorized.data() + 1, grads.data() + 1, count, lr, initialize);
            EXPECT_EQ(std::memcmp(
                          vectorized.data(), scalar.data(), scalar.size() * 4),
                      0);
          }
#endif
        }
      }
    }
  }
  float value = 1, grad = 0;
  recstore::ApplySgdRow(&value, &grad, 1, 0.01f, true);
  EXPECT_EQ(std::signbit(value), !recstore::SparseCommitFmaEnabled());
}

TEST(SparseGradientCommit, FiniteScanUnalignedAndEveryBadPosition) {
  EXPECT_TRUE(recstore::CanonicalAllFinite(nullptr, 0));
  const uint32_t valid[] = {
      0,
      0x80000000u,
      1,
      0x80000001u,
      0x007fffffu,
      0x807fffffu,
      0x00800000u,
      0x80800000u,
      0x7f7fffffu,
      0xff7fffffu};
  const uint32_t invalid[] = {
      0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u};
  for (size_t alignment = 0; alignment < 8; ++alignment) {
    std::vector<char> bytes(129 * sizeof(float) + 8);
    char* raw = bytes.data() + alignment;
    for (size_t i = 0; i < 129; ++i)
      std::memcpy(raw + i * 4, &valid[i % 10], 4);
    for (size_t count = 0; count <= 129; ++count)
      EXPECT_TRUE(recstore::CanonicalAllFinite(raw, count));
    for (size_t at = 0; at < 129; ++at) {
      uint32_t original;
      std::memcpy(&original, raw + at * 4, 4);
      for (uint32_t value : invalid) {
        std::memcpy(raw + at * 4, &value, 4);
        EXPECT_FALSE(recstore::CanonicalAllFinite(raw, 129));
      }
      std::memcpy(raw + at * 4, &original, 4);
    }
  }
}

namespace {
SparseGradientCommitHeader ExactHeader(
    uint32_t rank, int64_t step = 0, uint32_t chunk = 0, uint32_t chunks = 1) {
  SparseGradientCommitHeader h;
  h.version = 4;
  h.session = 4401;
  h.world   = 2;
  h.dim     = 128;
  h.rank    = rank;
  h.step    = step;
  h.chunk   = chunk;
  h.chunks  = chunks;
  return h;
}
std::vector<float> ExactValues(size_t rows) {
  const uint32_t bits[] = {
      0,
      0x80000000u,
      1,
      0x80000001u,
      0x007fffffu,
      0x807fffffu,
      0x00800000u,
      0x80800000u,
      0x7f7fffffu,
      0xff7fffffu,
      0x3f800001u,
      0xbf800001u};
  std::vector<float> values(rows * 128);
  for (size_t i = 0; i < values.size(); ++i)
    std::memcpy(&values[i], &bits[i % (sizeof(bits) / sizeof(bits[0]))], 4);
  return values;
}
std::string ExactPacket(SparseGradientCommitHeader h,
                        const std::vector<uint64_t>& keys,
                        const std::vector<float>& values = {}) {
  h.rows    = keys.size();
  auto data = values.empty() ? ExactValues(keys.size()) : values;
  if (data.size() != keys.size() * 128)
    throw std::invalid_argument("Invalid test fixture shape");
  return PackSparseGradientCommit(h, keys.data(), data.data());
}
std::string ExactControl(uint32_t rank, uint32_t kind, int64_t step = 0) {
  auto h = ExactHeader(rank, step);
  h.kind = kind;
  return PackSparseGradientCommit(h);
}
} // namespace

TEST(SparseGradientCommit, ExactOwnerAllArrivalsWaitForWholeStepAndCopyBits) {
  const std::vector<uint64_t> expected_keys = {0, 1, 32, UINT64_MAX};
  const auto expected                       = ExactValues(expected_keys.size());
  for (int owner : {0, 1}) {
    std::vector<std::string> packets;
    for (int chunk = 0; chunk < 3; ++chunk) {
      const size_t begin = chunk == 0 ? 0 : size_t(chunk + 1);
      const size_t end   = chunk == 0 ? 2 : size_t(chunk + 2);
      packets.push_back(ExactPacket(
          ExactHeader(owner, 0, chunk, 3),
          {expected_keys.begin() + begin, expected_keys.begin() + end},
          {expected.begin() + begin * 128, expected.begin() + end * 128}));
    }
    packets.push_back(ExactPacket(ExactHeader(1 - owner), {}));
    std::vector<size_t> order = {0, 1, 2, 3};
    do {
      SparseGradientCommitCoordinator c(false);
      int calls  = 0;
      auto apply = [&](const auto& keys, const auto& values) {
        ++calls;
        EXPECT_EQ(keys, expected_keys);
        ASSERT_EQ(values.size(), expected.size());
        EXPECT_EQ(
            std::memcmp(values.data(), expected.data(), values.size() * 4), 0);
      };
      for (size_t i = 0; i < order.size(); ++i) {
        EXPECT_EQ(c.Exchange("e", packets[order[i]], apply, owner),
                  i == 3 ? 0 : -1);
        EXPECT_EQ(calls, i == 3 ? 1 : 0);
        EXPECT_EQ(c.Exchange("e", packets[order[i]], apply, owner),
                  i == 3 ? 0 : -1);
      }
      for (const auto& packet : packets)
        EXPECT_EQ(c.Exchange("e", packet, apply, owner), 0);
      EXPECT_EQ(calls, 1);
    } while (std::next_permutation(order.begin(), order.end()));
  }
}

TEST(SparseGradientCommit,
     ExactOwnerCodecRejectsBoundsDisorderAndEveryNonfinite) {
  auto h = ExactHeader(0);
  EXPECT_EQ(
      recstore::DecodeSparseGradientCommit(ExactPacket(h, {0, UINT64_MAX}))
          .version,
      4);
  EXPECT_THROW(ExactPacket(h, {1, 1}), std::invalid_argument);
  EXPECT_THROW(ExactPacket(h, {2, 1}), std::invalid_argument);
  for (int mode = 0; mode < 5; ++mode) {
    auto bad = h;
    if (mode == 0)
      bad.world = 1;
    if (mode == 1)
      bad.dim = 64;
    if (mode == 2)
      bad.divisor = 2;
    if (mode == 3)
      bad.chunks = 65;
    if (mode == 4) {
      bad.kind = 1;
      bad.rows = 1;
    }
    if (mode == 4) {
      auto values  = ExactValues(1);
      uint64_t key = 1;
      EXPECT_THROW(PackSparseGradientCommit(bad, &key, values.data()),
                   std::invalid_argument);
    } else
      EXPECT_THROW(PackSparseGradientCommit(bad), std::invalid_argument);
  }
  auto values                = ExactValues(1);
  const uint32_t nonfinite[] = {
      0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7f800001u};
  for (size_t col = 0; col < 128; ++col) {
    const float previous = values[col];
    for (uint32_t bits : nonfinite) {
      std::memcpy(&values[col], &bits, 4);
      EXPECT_THROW(ExactPacket(h, {1}, values), std::invalid_argument);
    }
    values[col] = previous;
  }
  std::vector<uint64_t> many(2048);
  std::iota(many.begin(), many.end(), 0);
  EXPECT_THROW(ExactPacket(h, many), std::invalid_argument); // More than 1 MiB.
}

TEST(SparseGradientCommit, ExactOwnerExplicitBindingAndPeerRowsFailClosed) {
  for (int mode = 0; mode < 4; ++mode) {
    SparseGradientCommitCoordinator c;
    int calls        = 0;
    auto apply       = [&](const auto&, const auto&) { ++calls; };
    const auto first = ExactPacket(ExactHeader(0, 0, 0, 2), {1});
    c.Exchange("e", first, apply, 0);
    if (mode == 0)
      EXPECT_THROW(c.Exchange("e", first, apply), std::invalid_argument);
    if (mode == 1)
      EXPECT_THROW(c.Exchange("e", ExactPacket(ExactHeader(1), {9}), apply, 0),
                   std::invalid_argument);
    if (mode == 2)
      EXPECT_THROW(c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 1),
                   std::runtime_error);
    if (mode == 3)
      EXPECT_THROW(c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 2),
                   std::invalid_argument);
    EXPECT_THROW(c.Exchange("e", ExactControl(0, 1), apply, 0),
                 std::runtime_error);
    EXPECT_EQ(calls, 0);
  }
}

TEST(SparseGradientCommit,
     ExactOwnerLastRetryUsesPayloadBitsAndRejectsUnknownOldStep) {
  for (int mode = 0; mode < 4; ++mode) {
    SparseGradientCommitCoordinator c;
    int calls        = 0;
    auto apply       = [&](const auto&, const auto&) { ++calls; };
    const auto first = ExactPacket(ExactHeader(0), {1});
    c.Exchange("e", first, apply, 0);
    c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 0);
    EXPECT_EQ(c.Exchange("e", first, apply, 0), 0);
    std::string bad = first;
    if (mode == 0) {
      uint32_t negative_zero = 0x80000000u;
      std::memcpy(bad.data() + sizeof(SparseGradientCommitHeader) + 8,
                  &negative_zero,
                  4);
    }
    if (mode == 1)
      bad = ExactPacket(ExactHeader(0), {2});
    if (mode == 2)
      bad = ExactPacket(ExactHeader(0, 0, 0, 2), {1});
    if (mode == 3) {
      c.Exchange("e", ExactPacket(ExactHeader(0, 1), {}), apply, 0);
      EXPECT_EQ(c.Exchange("e", ExactPacket(ExactHeader(1, 1), {}), apply, 0),
                1);
    }
    EXPECT_THROW(c.Exchange("e", bad, apply, 0), std::runtime_error);
    EXPECT_THROW(c.Exchange("e", ExactControl(0, 1), apply, 0),
                 std::runtime_error);
    EXPECT_EQ(calls, 1);
  }
}

TEST(SparseGradientCommit, ExactOwnerLateValidationFailureNeverAppliesPrefix) {
  for (int mode = 0; mode < 5; ++mode) {
    SparseGradientCommitCoordinator c;
    int calls  = 0;
    auto apply = [&](const auto&, const auto&) { ++calls; };
    c.Exchange("e", ExactPacket(ExactHeader(0, 0, 0, 2), {9}), apply, 0);
    c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 0);
    auto bad = ExactPacket(
        ExactHeader(0, 0, 1, 2),
        {mode == 0   ? 9u
         : mode == 1 ? 1u
                     : 12u});
    if (mode == 2) {
      uint32_t nan = 0x7fc00001u;
      std::memcpy(bad.data() + bad.size() - 4, &nan, 4);
    }
    if (mode == 3)
      bad.resize(bad.size() - 1);
    if (mode == 4)
      bad = ExactPacket(ExactHeader(0, 0, 1, 3), {12});
    EXPECT_ANY_THROW(c.Exchange("e", bad, apply, 0));
    EXPECT_THROW(c.Exchange("e", ExactControl(0, 1), apply, 0),
                 std::runtime_error);
    EXPECT_EQ(calls, 0);
  }
}

TEST(SparseGradientCommit, ExactOwnerEmptyStepsPendingCloseAndReopen) {
  SparseGradientCommitCoordinator c;
  int calls  = 0;
  auto apply = [&](const auto&, const auto&) { ++calls; };
  c.Exchange("e", ExactPacket(ExactHeader(0), {}), apply, 0);
  EXPECT_EQ(c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 0), 0);
  c.Exchange("e", ExactPacket(ExactHeader(0, 1), {}), apply, 0);
  EXPECT_THROW(c.Exchange("e", ExactControl(0, 2, 0), apply, 0),
               std::runtime_error);
  EXPECT_EQ(c.Exchange("e", ExactPacket(ExactHeader(1, 1), {}), apply, 0), 1);
  for (int rank = 0; rank < 2; ++rank)
    EXPECT_EQ(c.Exchange("e", ExactControl(rank, 2, 1), apply, 0), 1);
  for (int rank = 0; rank < 2; ++rank) {
    auto h = ExactHeader(rank);
    ++h.session;
    EXPECT_EQ(c.Exchange("e", ExactPacket(h, {}), apply, 1), rank ? 0 : -1);
  }
  EXPECT_EQ(calls, 0);
}

TEST(SparseGradientCommit, ExactOwnerFailureAbortAndVersionMismatchAreSticky) {
  for (int mode = 0; mode < 4; ++mode) {
    SparseGradientCommitCoordinator c;
    int calls  = 0;
    auto apply = [&](const auto&, const auto&) {
      ++calls;
      if (mode == 0)
        throw std::runtime_error("write failure");
    };
    c.Exchange("e", ExactPacket(ExactHeader(0), {1}), apply, 0);
    if (mode == 0)
      EXPECT_THROW(c.Exchange("e", ExactPacket(ExactHeader(1), {}), apply, 0),
                   std::runtime_error);
    if (mode == 1)
      EXPECT_THROW(c.Exchange("e", ExactControl(1, 3), apply, 0),
                   std::runtime_error);
    if (mode == 2) {
      auto h    = ExactHeader(1);
      h.version = 2;
      EXPECT_THROW(c.Exchange("e", PackSparseGradientCommit(h), apply, 0),
                   std::runtime_error);
    }
    if (mode == 3)
      EXPECT_THROW(
          c.Exchange("e", ExactPacket(ExactHeader(1, 2), {}), apply, 0),
          std::runtime_error);
    EXPECT_THROW(c.Exchange("e", ExactControl(0, 1), apply, 0),
                 std::runtime_error);
    EXPECT_EQ(calls, mode == 0 ? 1 : 0);
    EXPECT_EQ(c.Exchange("other", ExactPacket(ExactHeader(0), {}), apply, 0),
              -1);
  }
}

TEST(SparseGradientCommit, ExactOwnerWholeStepRowsAreBoundedBeforeApply) {
  SparseGradientCommitCoordinator c;
  int calls  = 0;
  auto apply = [&](const auto&, const auto&) { ++calls; };
  for (uint32_t chunk = 0; chunk < 52; ++chunk) {
    std::vector<uint64_t> keys(1024);
    std::iota(keys.begin(), keys.end(), uint64_t(chunk) * 1024);
    EXPECT_EQ(
        c.Exchange(
            "e", ExactPacket(ExactHeader(0, 0, chunk, 53), keys), apply, 0),
        -1);
  }
  EXPECT_THROW(
      c.Exchange(
          "e", ExactPacket(ExactHeader(0, 0, 52, 53), {53248}), apply, 0),
      std::invalid_argument);
  EXPECT_THROW(c.Exchange("e", ExactControl(0, 1), apply, 0),
               std::runtime_error);
  EXPECT_EQ(calls, 0);
}

// ---- Merged-gradient owner plan (owner_merged_plan.h) ----

namespace {
int MergedOracleOwner(uint64_t key, uint64_t tag) {
  const uint64_t physical = key | (tag << 56);
  return static_cast<int>(
      ::CityHash64(reinterpret_cast<const char*>(&physical), sizeof(physical)) %
      2);
}
} // namespace

TEST(OwnerMergedPlan, RoutingMatchesCityHashOracleForBothRanks) {
  std::mt19937_64 rng(20260330);
  const std::vector<int64_t> boundaries = {0, 100, 200};
  const std::vector<int64_t> types      = {0, 1};
  const int64_t features = 2, tag = 3;
  for (int trial = 0; trial < 300; ++trial) {
    std::vector<int64_t> keys[2], counts[2];
    for (int rank = 0; rank < 2; ++rank) {
      const int n = static_cast<int>(rng() % 40);
      std::set<int64_t> unique;
      for (int i = 0; i < n; ++i)
        unique.insert(static_cast<int64_t>(rng() % 200));
      for (int64_t key : unique) {
        keys[rank].push_back(key);
        // Deliberately arbitrary per-key counts: the merged per-feature
        // occurrence-total equality must not be required for merged rows.
        counts[rank].push_back(static_cast<int64_t>(rng() % 1024 + 1));
      }
    }
    for (int rank = 0; rank < 2; ++rank) {
      const auto plan = recstore::BuildOwnerMergedPlan(
          static_cast<int64_t>(keys[0].size()),
          keys[0].data(),
          counts[0].data(),
          static_cast<int64_t>(keys[1].size()),
          keys[1].data(),
          counts[1].data(),
          features,
          boundaries.data(),
          types.data(),
          tag,
          rank,
          true);
      // Independent oracle over the sorted union.
      size_t i0 = 0, i1 = 0, owned_index = 0, local_position = 0, offset0 = 0,
             offset1 = 0;
      std::vector<int64_t> expect_keys, expect_meta, expect_perm[2];
      std::vector<int64_t> expect_fsend[2], expect_frecv_pos[2];
      int64_t send[2] = {0, 0}, recv[2] = {0, 0}, fsend[2] = {0, 0},
              frecv[2]  = {0, 0};
      int64_t local_row = 0;
      while (i0 < keys[0].size() || i1 < keys[1].size()) {
        const int64_t key =
            i1 == keys[1].size() ||
                    (i0 < keys[0].size() && keys[0][i0] < keys[1][i1])
                ? keys[0][i0]
                : keys[1][i1];
        const int64_t c0 =
            i0 < keys[0].size() && keys[0][i0] == key ? counts[0][i0++] : 0;
        const int64_t c1 =
            i1 < keys[1].size() && keys[1][i1] == key ? counts[1][i1++] : 0;
        const int owner    = MergedOracleOwner(static_cast<uint64_t>(key), tag);
        const int64_t type = types[key >= boundaries[1] ? 1 : 0];
        const bool present[2] = {c0 > 0, c1 > 0};
        if (present[rank]) {
          expect_perm[owner].push_back(static_cast<int64_t>(local_position++));
          expect_frecv_pos[owner].push_back(local_row++);
          ++send[owner];
        }
        if (owner == rank) {
          expect_keys.push_back(key);
          expect_meta.insert(
              expect_meta.end(),
              {key,
               c0,
               c1,
               type,
               static_cast<int64_t>(offset0),
               present[0],
               static_cast<int64_t>(offset1),
               present[1]});
          if (present[0])
            ++offset0;
          if (present[1])
            ++offset1;
          for (int destination = 0; destination < 2; ++destination)
            if (present[destination])
              expect_fsend[destination].push_back(owned_index);
          ++owned_index;
          for (int source = 0; source < 2; ++source)
            if (present[source])
              ++recv[source];
        }
      }
      for (int peer = 0; peer < 2; ++peer) {
        fsend[peer] = static_cast<int64_t>(expect_fsend[peer].size());
        frecv[peer] = static_cast<int64_t>(expect_frecv_pos[peer].size());
      }
      std::vector<int64_t> expect_permutation = expect_perm[0];
      expect_permutation.insert(
          expect_permutation.end(),
          expect_perm[1].begin(),
          expect_perm[1].end());
      std::vector<int64_t> expect_fsend_index = expect_fsend[0];
      expect_fsend_index.insert(
          expect_fsend_index.end(),
          expect_fsend[1].begin(),
          expect_fsend[1].end());
      std::vector<int64_t> expect_reorder(local_row, -1);
      int64_t received = 0;
      for (int peer = 0; peer < 2; ++peer)
        for (const auto position : expect_frecv_pos[peer])
          expect_reorder[position] = received++;
      EXPECT_EQ(plan.keys, expect_keys);
      EXPECT_EQ(plan.meta, expect_meta);
      EXPECT_EQ(plan.permutation, expect_permutation);
      EXPECT_EQ(plan.send, (std::array<int64_t, 2>{send[0], send[1]}));
      EXPECT_EQ(plan.recv, (std::array<int64_t, 2>{recv[0], recv[1]}));
      EXPECT_EQ(plan.forward_send_index, expect_fsend_index);
      EXPECT_EQ(plan.forward_reorder, expect_reorder);
      EXPECT_EQ(plan.forward_send,
                (std::array<int64_t, 2>{fsend[0], fsend[1]}));
      EXPECT_EQ(plan.forward_recv,
                (std::array<int64_t, 2>{frecv[0], frecv[1]}));
      ASSERT_EQ(plan.meta.size() / 8, plan.keys.size());
      for (size_t row = 0; row < plan.keys.size(); ++row) {
        const auto* m = plan.meta.data() + row * 8;
        EXPECT_TRUE(m[5] == 0 || m[5] == 1);
        EXPECT_TRUE(m[7] == 0 || m[7] == 1);
        EXPECT_EQ(m[5] != 0, m[1] != 0);
        EXPECT_EQ(m[7] != 0, m[2] != 0);
        EXPECT_LE(m[4] + m[5], plan.recv[0]);
        EXPECT_LE(m[6] + m[7], plan.recv[1]);
      }
    }
  }
}

TEST(OwnerMergedPlan, CrossRankSplitsAndEmptySidesAgree) {
  const std::vector<int64_t> boundaries = {0, 100, 200};
  const std::vector<int64_t> types      = {0, 1};
  {
    std::vector<int64_t> k0 = {1, 5, 150}, c0 = {2, 1, 7}, k1 = {5, 199},
                         c1 = {3, 1};
    auto p0                 = recstore::BuildOwnerMergedPlan(
        3,
        k0.data(),
        c0.data(),
        2,
        k1.data(),
        c1.data(),
        2,
        boundaries.data(),
        types.data(),
        0,
        0);
    auto p1 = recstore::BuildOwnerMergedPlan(
        3,
        k0.data(),
        c0.data(),
        2,
        k1.data(),
        c1.data(),
        2,
        boundaries.data(),
        types.data(),
        0,
        1);
    EXPECT_EQ(p0.recv[1], p1.send[0]);
    EXPECT_EQ(p1.recv[0], p0.send[1]);
    EXPECT_EQ(p0.recv[0], p0.send[0]);
    EXPECT_EQ(p1.recv[1], p1.send[1]);
    EXPECT_EQ(p0.rows, (std::array<int64_t, 2>{3, 2}));
    EXPECT_EQ(p0.send[0] + p0.send[1], 3);
    EXPECT_EQ(p1.send[0] + p1.send[1], 2);
  }
  for (int rank = 0; rank < 2; ++rank) {
    auto empty_side = rank == 0;
    std::vector<int64_t>
        k0 = empty_side ? std::vector<int64_t>{7, 120} : std::vector<int64_t>{},
        c0 = empty_side ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{};
    std::vector<int64_t>
        k1 = empty_side ? std::vector<int64_t>{} : std::vector<int64_t>{7, 120},
        c1 = empty_side ? std::vector<int64_t>{} : std::vector<int64_t>{3, 1};
    auto plan = recstore::BuildOwnerMergedPlan(
        static_cast<int64_t>(k0.size()),
        k0.data(),
        c0.data(),
        static_cast<int64_t>(k1.size()),
        k1.data(),
        c1.data(),
        2,
        boundaries.data(),
        types.data(),
        0,
        rank,
        true);
    EXPECT_EQ(plan.rows[rank],
              static_cast<int64_t>((rank == 0 ? k0 : k1).size()));
    EXPECT_EQ(plan.rows[1 - rank], 0);
    EXPECT_EQ(plan.permutation.size(), static_cast<size_t>(plan.rows[rank]));
    EXPECT_EQ(plan.send[0] + plan.send[1], plan.rows[rank]);
    // Ownership still routes the populated side's keys; the empty rank owns
    // only whatever the hash assigns to it, never rows of its own.
    EXPECT_LE(plan.recv[0], plan.rows[0]);
    EXPECT_LE(plan.recv[1], plan.rows[1]);
    for (int64_t key : plan.keys) {
      const auto& side = rank == 0 ? k0 : k1;
      EXPECT_NE(std::find(side.begin(), side.end(), key), side.end());
    }
    std::vector<int64_t> covered(plan.permutation);
    std::sort(covered.begin(), covered.end());
    std::vector<int64_t> range(static_cast<size_t>(plan.rows[rank]));
    std::iota(range.begin(), range.end(), 0);
    EXPECT_EQ(covered, range);
    EXPECT_EQ(plan.forward_reorder.size(),
              static_cast<size_t>(plan.rows[rank]));
  }
  auto both = recstore::BuildOwnerMergedPlan(
      0,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      2,
      boundaries.data(),
      types.data(),
      0,
      0,
      true);
  EXPECT_TRUE(
      both.permutation.empty() && both.keys.empty() && both.meta.empty());
  EXPECT_EQ(both.forward_reorder.size(), 0u);
}

TEST(OwnerMergedPlan, AcceptsUnequalFeatureTotalsAndRejectsMalformedInputs) {
  const std::vector<int64_t> boundaries = {0, 100, 200};
  const std::vector<int64_t> types      = {0, 1};
  // Feature 1 has zero occurrences while feature 0 has three: the merged
  // occurrence-total equality is intentionally absent for merged rows.
  std::vector<int64_t> k0 = {5}, c0 = {3};
  std::vector<int64_t> empty;
  auto plan = recstore::BuildOwnerMergedPlan(
      1,
      k0.data(),
      c0.data(),
      0,
      nullptr,
      nullptr,
      2,
      boundaries.data(),
      types.data(),
      9,
      0);
  EXPECT_EQ(plan.meta, (std::vector<int64_t>{5, 3, 0, 0, 0, 1, 0, 0}));
  EXPECT_EQ(plan.rows, (std::array<int64_t, 2>{1, 0}));
  EXPECT_THROW(
      recstore::BuildOwnerMergedPlan(
          1,
          empty.data(),
          c0.data(),
          0,
          nullptr,
          nullptr,
          2,
          boundaries.data(),
          types.data(),
          9,
          0),
      std::invalid_argument);
  for (int mode = 0; mode < 8; ++mode) {
    std::vector<int64_t> keys = k0, counts = c0;
    std::vector<int64_t> bad_boundaries = boundaries, bad_types = types;
    int64_t tag = 0, rank = 0, features = 2;
    switch (mode) {
    case 0:
      keys   = {5, 3};
      counts = {1, 1};
      break; // unsorted
    case 1:
      keys   = {5, 5};
      counts = {1, 1};
      break; // duplicate
    case 2:
      counts = {0};
      break; // zero occurrence
    case 3:
      counts = {1025};
      break; // above batch bound
    case 4:
      keys = {200};
      break; // outside final boundary
    case 5:
      keys = {-1};
      break; // below first boundary
    case 6:
      tag = 256;
      break; // namespace overflow
    case 7:
      rank = 2;
      break; // shard out of range
    }
    EXPECT_THROW(
        recstore::BuildOwnerMergedPlan(
            static_cast<int64_t>(keys.size()),
            keys.data(),
            counts.data(),
            0,
            nullptr,
            nullptr,
            features,
            bad_boundaries.data(),
            bad_types.data(),
            tag,
            rank),
        std::invalid_argument)
        << mode;
  }
  EXPECT_THROW(
      recstore::BuildOwnerMergedPlan(
          1,
          k0.data(),
          c0.data(),
          0,
          nullptr,
          nullptr,
          2,
          std::vector<int64_t>{0, 100, 100}.data(),
          types.data(),
          0,
          0),
      std::invalid_argument);
  EXPECT_THROW(
      recstore::BuildOwnerMergedPlan(
          1,
          k0.data(),
          c0.data(),
          0,
          nullptr,
          nullptr,
          2,
          boundaries.data(),
          std::vector<int64_t>{0, 2}.data(),
          0,
          0),
      std::invalid_argument);
}
