#include <gtest/gtest.h>
#include <array>
#include <map>
#include <numeric>
#include <random>
#include <set>
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
      SparseGradientCommitCoordinator c;
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
      auto bad  = ExactPacket(h, {});
      h.version = 3;
      std::memcpy(bad.data(), &h, sizeof(h));
      EXPECT_THROW(c.Exchange("e", bad, apply, 0), std::invalid_argument);
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
          rank);
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
        rank);
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
      0);
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
