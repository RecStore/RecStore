#include "framework/cache/cache_aware_op.h"
#include "framework/common/sparse_gradient_commit_runtime.h"
#include <gtest/gtest.h>
#include <atomic>

using namespace recstore;
using namespace recstore::framework;

TEST(SparseCommitRuntime, WaitsForStagedInputsAndCommittedStep) {
  SparseGradientCommitRuntime runtime;
  auto h = MakeSparseGradientCommitHeader(1, 0, 0, 1, 1, .01, 1, 0);
  std::promise<void> release;
  auto gate  = release.get_future().share();
  auto owner = std::make_shared<std::pair<uint64_t, float>>(7, 2.0f);
  std::weak_ptr<std::pair<uint64_t, float>> weak = owner;
  base::RecTensor keys(&owner->first, {1}), grads(&owner->second, {1, 1});
  std::atomic<int> calls{0};
  auto id = runtime.Submit(
      [&](auto header, const auto& k, const auto& g) -> int64_t {
        ++calls;
        if (header.kind == 0) {
          EXPECT_EQ(k.template data_as<uint64_t>()[0], 7);
          EXPECT_EQ(g.template data_as<float>()[0], 2.0f);
          return -1;
        }
        EXPECT_EQ(header.kind, 1);
        EXPECT_EQ(k.num_elements(), 0);
        return header.step;
      },
      h,
      keys,
      grads,
      [owner, gate] { gate.wait(); });
  owner.reset();
  EXPECT_EQ(calls, 0);
  EXPECT_FALSE(weak.expired());
  release.set_value();
  runtime.Wait(id);
  EXPECT_EQ(calls, 2);
  EXPECT_TRUE(weak.expired());
  EXPECT_THROW(runtime.Wait(id), std::runtime_error);
}

TEST(SparseCommitRuntime, TimeoutAbortsWithoutResubmittingAndPreservesError) {
  SparseGradientCommitRuntime runtime;
  auto h = MakeSparseGradientCommitHeader(1, 0, 0, 1, 1, .01, 1, 0);
  base::RecTensor keys({0}, base::DataType::UINT64),
      grads({0, 1}, base::DataType::FLOAT32);
  std::vector<uint32_t> kinds;
  auto id = runtime.Submit(
      [&](auto header, const auto&, const auto&) -> int64_t {
        kinds.push_back(header.kind);
        if (header.kind == 3)
          throw std::runtime_error("abort also failed");
        return -1;
      },
      h,
      keys,
      grads,
      {},
      std::chrono::milliseconds(0));
  try {
    runtime.Wait(id);
    FAIL() << "timeout expected";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "Sparse gradient commit timed out");
  }
  EXPECT_EQ(kinds, (std::vector<uint32_t>{0, 3}));
  EXPECT_THROW(
      MakeSparseGradientCommitHeader(1, 0, 0, 2, 1, .01, 4294967297LL, 0),
      std::invalid_argument);
}

TEST(SparseCommitRuntime, StagingFailureAbortsBeforeSendingGradients) {
  SparseGradientCommitRuntime runtime;
  auto h = MakeSparseGradientCommitHeader(1, 0, 0, 1, 1, .01, 1, 0);
  base::RecTensor keys({0}, base::DataType::UINT64),
      grads({0, 1}, base::DataType::FLOAT32);
  std::vector<uint32_t> kinds;
  auto id = runtime.Submit(
      [&](auto header, const auto&, const auto&) -> int64_t {
        kinds.push_back(header.kind);
        return -1;
      },
      h,
      keys,
      grads,
      [] { throw std::runtime_error("staging failed"); });
  EXPECT_THROW(runtime.Wait(id), std::runtime_error);
  EXPECT_EQ(kinds, (std::vector<uint32_t>{3}));
}

TEST(CacheAwareOp, MissPipelineBypassAndRecoveryShareOnePolicy) {
  CacheAwareOp op;
  std::string order;
  int queries     = 0;
  int64_t missing = 2048;
  CacheLookupActions io{
      true,
      [&] {
        ++queries;
        order += 'q';
        return missing;
      },
      [&] { order += 'r'; },
      [&] { order += 'f'; },
      [&] { order += 's'; },
      [&] { order += 'b'; },
      [&] { order += 'c'; }};
  // A cold start must not disable the cache after one batch: the first two
  // low-hit batches still go through the cache and only then trip the latch.
  op.Lookup(2048, io);
  EXPECT_EQ(order, "qrfs");
  order.clear();
  op.Lookup(2048, io);
  EXPECT_EQ(order, "qrfs");
  EXPECT_TRUE(op.Bypassed());

  // Bypassed lookups read the backend and keep the resident rows for the
  // re-probe instead of clearing them.
  order.clear();
  op.Lookup(2048, io);
  EXPECT_EQ(order, "b");

  // The probe fires on exactly the 32nd bypassed lookup. A failed probe keeps
  // the cache off and re-arms the interval.
  for (int i = 0; i < 30; ++i)
    op.Lookup(2048, io);
  EXPECT_EQ(queries, 2);
  op.Lookup(2048, io);
  EXPECT_EQ(queries, 3);
  EXPECT_TRUE(op.Bypassed());

  // The latch is re-probed periodically; a good hit rate restores the cache.
  missing = 0;
  for (int i = 0; i < 64 && op.Bypassed(); ++i)
    op.Lookup(2048, io);
  EXPECT_FALSE(op.Bypassed());
  EXPECT_NE(order.find('q'), std::string::npos);
}

TEST(CacheAwareOp, UpdateDuringBypassStillInvalidates) {
  // Bypass no longer clears the cache, and the probe can serve resident rows
  // again. Every write that reaches the cache must therefore invalidate, even
  // while the latch is down.
  CacheAwareOp op;
  int invalidations = 0;
  CacheLookupActions lookup{
      true,
      [] { return int64_t{2048}; },
      [] {},
      [] {},
      [] {},
      [] {},
      [] { FAIL() << "bypass must not clear the cache"; }};
  op.Lookup(2048, lookup);
  op.Lookup(2048, lookup);
  ASSERT_TRUE(op.Bypassed());

  CacheUpdateActions update{
      [] { return true; },
      [] { return true; },
      [&] { ++invalidations; },
      [] { FAIL() << "invalidate is available, clear should not run"; }};
  op.Update(2048, update, [] {});
  EXPECT_EQ(invalidations, 1);
}

TEST(CacheAwareOp, FailedFillFallsBackButNoEvictionPathPropagates) {
  CacheAwareOp op;
  std::string order;
  CacheLookupActions io{
      true,
      [] { return 1; },
      [] {},
      [] { throw std::runtime_error("fill"); },
      [&] { order += 's'; },
      [&] { order += 'b'; },
      [&] { order += 'c'; }};
  op.Lookup(1, io);
  EXPECT_EQ(order, "cb");
  order.clear();
  EXPECT_THROW(op.Lookup(1, io, true), std::runtime_error);
  EXPECT_TRUE(order.empty());
  io.usable = false;
  EXPECT_THROW(op.Lookup(1, io, true), std::runtime_error);
}

TEST(CacheAwareOp, AsyncCompletionPrecedesInvalidationAndFailureClearsCache) {
  PendingCacheUpdates pending;
  g_cache_aware_op.SetBypassEnabled(true);
  g_cache_aware_op.Reset();
  std::string order;
  CacheUpdateActions actions{
      [] { return true; },
      [] { return true; },
      [&] { order += 'i'; },
      [&] { order += 'c'; }};
  pending.Add(1, 1, actions);
  EXPECT_TRUE(order.empty());
  pending.Wait(1, [&] { order += 'w'; });
  EXPECT_EQ(order, "wi");
  order.clear();
  pending.Add(2, 1, actions);
  EXPECT_THROW(pending.Wait(2, [] { throw std::runtime_error("backend"); }),
               std::runtime_error);
  EXPECT_EQ(order, "c");
  EXPECT_THROW(pending.Wait(2, [] {}), std::runtime_error);
  order.clear();
  g_cache_aware_op.SetBypassEnabled(false);
  g_cache_aware_op.AfterUpdate(1, actions);
  EXPECT_TRUE(order.empty());
  g_cache_aware_op.AfterUpdate(1, actions, true);
  EXPECT_EQ(order, "i");
  order.clear();
  CacheAwareOp::AfterReload(false, actions);
  EXPECT_TRUE(order.empty());
  CacheAwareOp::AfterReload(true, actions);
  EXPECT_EQ(order, "c");
  g_cache_aware_op.SetBypassEnabled(true);
}
