import queue
import unittest
from unittest import mock

import torch

from ..bagpipe_cache import comm as comm_module
from ..bagpipe_cache.comm import BagPipeCommMixin
from ..bagpipe_cache.grads import BagPipeGradMixin


class _DoneWork:
    def wait(self):
        return None


class _CommHarness(BagPipeCommMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self._global_id_to_index = {10: 0}
        self._shared_ids_tensor = torch.tensor([10], dtype=torch.int64)
        self._global_unique_count = 1
        self._stats = {
            "bagpipe_all_reduce_calls": 0,
            "bagpipe_all_reduce_ids": 0,
            "bagpipe_all_reduce_ms": 0,
        }

    def _is_distributed(self):
        return True


class _FakeClient:
    def __init__(self):
        self.updates = []
        self.invalidations = []
        self.best_effort_applies = []
        self.async_updates = []  # (handle, name, ids, grads) issued via update_async
        self.waits = []
        self._next_handle = 1000

    def apply_sgd_update_gpu_cache_best_effort(self, name, ids, grads, *,
                                               learning_rate):
        self.best_effort_applies.append(
            (name, ids.clone(), grads.clone(), learning_rate)
        )

    def update(self, name, ids, grads):
        self.updates.append((name, ids.clone(), grads.clone()))

    def update_async(self, name, ids, grads, learning_rate=0.0):
        self._next_handle += 1
        self.async_updates.append(
            (self._next_handle, name, ids.clone(), grads.clone())
        )
        return self._next_handle

    def wait(self, handle):
        self.waits.append(handle)
        return None

    def invalidate_gpu_cache(self, name, ids):
        self.invalidations.append((name, ids.clone()))

    def invalidate_gpu_cache_with_mask(self, name, ids):
        self.invalidations.append((name, ids.clone()))
        return torch.ones_like(ids, dtype=torch.bool)


class _DenseWork:
    def __init__(self, ids, grads):
        self.result = (ids, grads)

    def wait(self):
        return None


class _GradHarness(BagPipeGradMixin):
    def __init__(self, rank=0):
        self.device = torch.device("cpu")
        self._rank = rank
        self.kv_client = _FakeClient()
        self.embedding_module = type("EmbeddingModule", (), {})()
        self.master_table_name = "table"
        self.lr = 0.01
        # Flat-tensor bookkeeping (isomorphic to
        # controller._alloc_bookkeeping, fixed 64)
        self._latest_dev = torch.zeros(64, dtype=torch.int32)
        self._ttl_dev = torch.zeros(64, dtype=torch.int32)
        self._cached_dev = torch.zeros(64, dtype=torch.bool)
        self._dirty_dev = torch.zeros(64, dtype=torch.bool)
        self._cached_dev[torch.tensor([10, 20, 30])] = True
        self._insert_failures = 0
        self._sync_later_stream = None
        self._pending_sync_now_work = None
        self._pending_sync_now_lr = None
        self._ttl_margin = 256
        self._shared_ids = {10, 30}
        self._shared_ids_tensor = torch.tensor([10, 30], dtype=torch.int64)
        # The harness declares an oracle set (id 20 really is rank-local),
        # so ids outside the set may take the local-only fast path. The
        # incomplete-set scenario is covered separately by
        # test_unknown_ids_are_aggregated_when_set_is_incomplete.
        self._shared_ids_complete = True
        # Shared-set routing threshold (matches the controller default):
        # {10, 30} is 2 rows <= threshold -> dense routing by default; the
        # large-set routing is covered by the _dense_shared_max_rows=0 case.
        self._global_unique_count = 2
        self._dense_shared_max_rows = 4096
        # Oracle prescan sizing metadata is absent by default: the sparse
        # path then uses the count-exchange variant (same as the fallback
        # training mode).  Tests for the sync-free path set it explicitly.
        self._prescan_sparse_max = None
        # No GPU hot accumulators -> _hot_add falls back to host float
        # (straight into _stats)
        self._hot_stats_dev = {}
        self.dense_calls = []
        self.sparse_calls = []
        self.prescan_sparse_calls = []
        self._stats = {
            "bagpipe_update_ms": 0.0,
            "bagpipe_sgd_cache_success": 0.0,
            "bagpipe_sgd_cache_fallback": 0.0,
            "bagpipe_all_reduce_ms": 0.0,
            "bagpipe_sync_now_overlap_ms": 0.0,
            "bagpipe_sync_now_ids": 0.0,
            "bagpipe_sync_later_ids": 0.0,
            "bagpipe_no_sync_ids": 0.0,
            "bagpipe_unknown_aggregated_ids": 0.0,
        }

    def _is_distributed(self):
        return True

    def _get_rank(self):
        return self._rank

    # Identity mapping: tests use the fused id directly as the compact
    # index (fixed 64-slot bookkeeping)
    def _to_compact(self, ids):
        return ids

    def _to_fused(self, compact):
        return compact

    def _compact_in_range(self, compact):
        return compact[compact < self._cached_dev.numel()]

    def _record_shared_id_candidates(self, _ids):
        return None

    def _hot_add(self, key, count):
        # Test harness has no GPU hot accumulators: accumulate host floats
        # directly (isomorphic to the controller's fallback branch when
        # _hot_stats_dev is empty)
        if isinstance(count, torch.Tensor):
            count = float(count.item())
        self._stats[key] = self._stats.get(key, 0.0) + count

    def _dense_all_reduce_async(self, ids, grads, stream=None):
        self.dense_calls.append((ids.clone(), grads.clone(), stream))
        return ids, grads, _DenseWork(ids, grads)

    def _all_gather_sparse_async(self, ids, grads, stream=None):
        self.sparse_calls.append((ids.clone(), grads.clone(), stream))
        if ids.numel() == 0:
            return ids, grads, None
        # Aggregated result = this rank's contribution (single-process
        # harness has no peer)
        return ids, grads, _DenseWork(ids.clone(), grads.clone())

    def _all_gather_sparse_prescan_async(self, ids, grads, mask, max_n,
                                         stream=None):
        self.prescan_sparse_calls.append(
            (ids.clone(), grads.clone(), mask.clone(), int(max_n), stream)
        )
        # Mirror the real implementation's GPU-side compaction so tests can
        # assert on the padded payload.
        slot = torch.cumsum(mask, 0) - 1
        trash = torch.full_like(slot, max_n)
        dest = torch.where(mask, slot, trash)
        padded_ids = torch.full((max_n + 1,), -1, dtype=torch.int64)
        padded_ids[dest] = ids
        padded_ids = padded_ids[:max_n]
        if max_n == 0 or padded_ids.numel() == 0:
            return ids[:0], grads[:0], None
        return (
            ids, grads,
            _DenseWork(padded_ids.clone(),
                       torch.zeros(max_n, grads.size(1))),
        )


class TestBagPipeDenseReduce(unittest.TestCase):
    def test_dense_work_returns_only_ids_present_in_global_mapping(self):
        harness = _CommHarness()
        ids = torch.tensor([10, 99], dtype=torch.int64)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(comm_module.dist, "all_reduce", return_value=_DoneWork()):
            valid_ids, valid_grads, work = harness._dense_all_reduce_async(ids, grads)
            work.wait()

        self.assertEqual(valid_ids.tolist(), [10])
        self.assertEqual(valid_grads.tolist(), [[1.0, 2.0]])
        result_ids, result_grads = work.result
        self.assertEqual(result_ids.tolist(), [10])
        self.assertEqual(result_grads.tolist(), [[1.0, 2.0]])

    def test_dense_reduce_maps_via_searchsorted_positions(self):
        harness = _CommHarness()
        harness._shared_ids_tensor = torch.tensor([10, 20, 30], dtype=torch.int64)
        harness._global_unique_count = 3
        harness._global_id_to_index = {10: 0, 20: 1, 30: 2}
        ids = torch.tensor([30, 99, 10], dtype=torch.int64)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        captured = {}

        def fake_all_reduce(t, async_op=True):
            captured["dense"] = t.clone()
            return _DoneWork()

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(comm_module.dist, "all_reduce", side_effect=fake_all_reduce):
            valid_ids, _, work = harness._dense_all_reduce_async(ids, grads)
            work.wait()

        # valid rows are 30 (index 2) and 10 (index 0); 99 filtered out
        self.assertEqual(valid_ids.tolist(), [30, 10])
        dense = captured["dense"]
        self.assertEqual(dense[2].tolist(), [1.0, 2.0])
        self.assertEqual(dense[0].tolist(), [5.0, 6.0])
        self.assertEqual(dense[1].tolist(), [0.0, 0.0])

    def test_dense_work_applies_peer_rows_this_rank_did_not_contribute(self):
        """The dense apply set must be rank-uniform: the all_reduced buffer is
        identical on every rank, so every rank must apply its nonzero rows --
        including rows a peer contributed and this rank never saw.  Applying
        only own rows would fork the replicas for shared ids whose batch
        membership differs across ranks (no per-step PS push exists to
        repair them afterwards)."""
        harness = _CommHarness()
        harness._shared_ids_tensor = torch.tensor([10, 20, 30], dtype=torch.int64)
        harness._global_unique_count = 3
        harness._global_id_to_index = {10: 0, 20: 1, 30: 2}
        # This rank's batch touches only id 10.
        ids = torch.tensor([10], dtype=torch.int64)
        grads = torch.tensor([[1.0, 2.0]])

        def fake_all_reduce(t, async_op=True):
            # Simulate the peer's contribution: id 20 (row 1) carries a
            # gradient this rank did not contribute; id 30 (row 2) stays 0.
            t[1] += torch.tensor([7.0, 8.0])
            return _DoneWork()

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(comm_module.dist, "all_reduce", side_effect=fake_all_reduce):
            _, _, work = harness._dense_all_reduce_async(ids, grads)
            work.wait()

        result_ids, result_grads = work.result
        # Uniform nonzero set: own row 10 AND peer row 20; untouched row 30
        # is a no-op and must stay out (it would otherwise be marked dirty).
        self.assertEqual(result_ids.tolist(), [10, 20])
        self.assertEqual(result_grads.tolist(), [[1.0, 2.0], [7.0, 8.0]])


class TestCompositeWorkSentinelAggregation(unittest.TestCase):
    """wait() recovers each rank's contribution via the -1 sentinel mask,
    with no per-rank count synchronization."""

    def test_wait_masks_sentinel_rows_and_aggregates_duplicates(self):
        from ..bagpipe_cache.types import _CompositeWork

        # rank0 contributes [10] (second row is a sentinel); rank1
        # contributes [15, 10]
        ids_list = [torch.tensor([10, -1]), torch.tensor([15, 10])]
        grads_list = [
            # rank0's second row is a sentinel row and must be dropped
            torch.tensor([[1.0, 1.0], [9.0, 9.0]]),
            torch.tensor([[5.0, 5.0], [2.0, 2.0]]),
        ]
        work = _CompositeWork(
            _DoneWork(), _DoneWork(), ids_list, grads_list,
            world_size=2, dim=2, device=torch.device("cpu"), stats={},
        )
        work.wait()
        ids, grads = work.result
        self.assertEqual(ids.tolist(), [10, 15])
        # id 10 sums across ranks: [1,1]+[2,2]=[3,3]; the sentinel row's
        # [9,9] must not be counted
        self.assertEqual(grads.tolist(), [[3.0, 3.0], [5.0, 5.0]])

    def test_wait_all_sentinel_returns_empty(self):
        from ..bagpipe_cache.types import _CompositeWork

        ids_list = [torch.tensor([-1]), torch.tensor([-1])]
        grads_list = [
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        ]
        work = _CompositeWork(
            _DoneWork(), _DoneWork(), ids_list, grads_list,
            world_size=2, dim=2, device=torch.device("cpu"), stats={},
        )
        work.wait()
        ids, grads = work.result
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(grads.shape, (0, 2))


class TestBagPipeSparseAllGather(unittest.TestCase):
    def test_sparse_all_gather_pads_with_sentinel_and_aggregates(self):
        """2-rank fake all_gather: the peer contributes the same rows as
        this rank, so gradients are summed."""
        harness = _CommHarness()
        ids = torch.tensor([10], dtype=torch.int64)
        grads = torch.tensor([[1.0, 2.0]])

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            _, _, work = harness._all_gather_sparse_async(ids, grads)
            work.wait()

        agg_ids, agg_grads = work.result
        self.assertEqual(agg_ids.tolist(), [10])
        self.assertEqual(agg_grads.tolist(), [[2.0, 4.0]])

    def test_sparse_all_gather_early_returns_when_all_ranks_empty(self):
        harness = _CommHarness()

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            _, _, work = harness._all_gather_sparse_async(
                torch.empty(0, dtype=torch.int64), torch.zeros(0, 2)
            )

        self.assertIsNone(work)


class _SharedIdSetHarness(BagPipeCommMixin):
    """Minimal host for BagPipeCommMixin: only exercises shared-set
    construction / completeness flags."""

    def __init__(self):
        self.device = torch.device("cpu")
        self._shared_ids = None
        self._shared_ids_tensor = None
        self._shared_ids_complete = False
        self._global_id_to_index = None
        self._global_unique_count = 0
        self._init_unique_ids = set()
        self._prescan_done = False
        self._prescan_unique_ids = set()
        self._prescan_batch_ids = []
        self._prescan_batch_numels = []
        self._prescan_sparse_max = None
        self._dynamic_lookahead = 4
        self._stats = {
            "bagpipe_shared_ids": 0.0,
            "bagpipe_prescan_batches": 0.0,
            "bagpipe_prescan_ids": 0.0,
            "bagpipe_prescan_sparse_max_sum": 0.0,
        }

    @property
    def lookahead_value(self):
        return self._dynamic_lookahead

    def _is_distributed(self):
        return True


def _fake_two_rank_all_gather(out_list, tensor, async_op=False):
    """Fake 2-rank all_gather: rank1 contributes the same as rank0 (every
    id is shared)."""
    for out in out_list:
        out.copy_(tensor)
    return _DoneWork()


class TestBagPipeSharedIdSetBuild(unittest.TestCase):
    def test_candidates_accumulate_without_collective(self):
        harness = _SharedIdSetHarness()
        with mock.patch.object(comm_module.dist, "all_gather") as ag:
            harness._record_shared_id_candidates(
                torch.tensor([10, 20], dtype=torch.int64)
            )
            harness._record_shared_id_candidates(
                torch.tensor([20, 30], dtype=torch.int64)
            )
        ag.assert_not_called()
        self.assertEqual(harness._init_unique_ids, {10, 20, 30})

    def test_build_waits_for_the_step_boundary_and_stays_incomplete(self):
        harness = _SharedIdSetHarness()
        harness._record_shared_id_candidates(
            torch.tensor([10, 20], dtype=torch.int64)
        )
        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            harness._maybe_build_shared_id_set(3)
            self.assertIsNone(harness._shared_ids)  # 3 < max(lookahead, 2)
            harness._maybe_build_shared_id_set(4)
        self.assertEqual(harness._shared_ids, {10, 20})
        # The fallback set is only a prefix sample -> it cannot license
        # the local-only fast path
        self.assertFalse(harness._shared_ids_complete)
        self.assertEqual(harness._global_unique_count, 2)

    def test_finalize_prescan_marks_the_set_complete(self):
        harness = _SharedIdSetHarness()
        harness._prescan_unique_ids = {10, 20}
        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            harness.finalize_prescan()
        self.assertEqual(harness._shared_ids, {10, 20})
        self.assertTrue(harness._shared_ids_complete)

    def test_single_rank_set_is_complete_and_empty(self):
        harness = _SharedIdSetHarness()
        harness._is_distributed = lambda: False
        harness._maybe_build_shared_id_set(0)
        self.assertEqual(harness._shared_ids, set())
        self.assertTrue(harness._shared_ids_complete)


class TestCleanupTriggersSharedIdBuild(unittest.TestCase):
    def test_scheduler_builds_the_set_at_the_step_boundary(self):
        """The build point is scheduling (reached by every rank every step),
        not each rank's own batch counter."""
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin

        seen = []

        class _Harness(BagPipeEvictionMixin):
            def __init__(self):
                self._cached_dev = None
                self.cache_capacity = 0
                self._capacity_watermark = 0.9
                self.cleanup_interval = 8
                self._cleanup_pending = False
                self._cleanup_scan_ttl = False
                self._cleanup_batch = 0
                self._insert_failures = 0
                self._stats = {"bagpipe_cleanup_ms": 0.0}

            def _maybe_build_shared_id_set(self, current_step):
                seen.append(current_step)

            def _maybe_adjust_lookahead(self, current_batch):
                return None

            def _cached_count_upper_bound(self):
                return 0

        _Harness().schedule_cleanup(7)
        self.assertEqual(seen, [7])


class TestCleanupScheduling(unittest.TestCase):
    def _controller(self, *, cache_capacity=10):
        from ..bagpipe_cache.controller import BagPipeCacheController

        class KV:
            def contains_gpu_cache(self, ids):
                return torch.ones_like(ids, dtype=torch.bool)

            def invalidate_gpu_cache_with_mask(self, name, ids):
                return torch.ones_like(ids, dtype=torch.bool)

        return BagPipeCacheController(
            embedding_module=None,
            kv_client=KV(),
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=cache_capacity,
            embedding_dim=4,
            fuse_k=6,
            table_offsets={"a": 0},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 100},
        )

    def test_ttl_scan_is_scheduled_every_eight_steps(self):
        ctrl = self._controller()
        try:
            for step in range(7):
                ctrl.schedule_cleanup(step)
                self.assertFalse(ctrl._cleanup_pending)
            ctrl.schedule_cleanup(7)
            self.assertTrue(ctrl._cleanup_pending)
            self.assertTrue(ctrl._cleanup_scan_ttl)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_pressure_decision_uses_host_upper_bound(self):
        ctrl = self._controller(cache_capacity=1)
        try:
            ctrl._cached_count_exact = 58  # high watermark is 57/64 entries
            with mock.patch.object(
                ctrl, "_cached_count", side_effect=AssertionError("full scan")
            ):
                ctrl.schedule_cleanup(0)
            self.assertTrue(ctrl._cleanup_pending)
            self.assertFalse(ctrl._cleanup_scan_ttl)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_ttl_and_insert_failure_share_one_budget(self):
        ctrl = self._controller(cache_capacity=0)
        try:
            ids = torch.arange(20)
            ctrl._cached_dev[ids] = True
            ctrl._ttl_dev[ids] = 100
            ctrl._ttl_dev[:4] = 0
            ctrl._insert_failures = 10
            ctrl._max_evict_per_cleanup = 6

            ctrl._run_cleanup(8, scan_ttl=True)

            self.assertEqual(int(ctrl._cached_dev.sum().item()), 14)
            self.assertEqual(ctrl._stats["bagpipe_evicted_ids"], 6.0)
            self.assertTrue(ctrl._cleanup_pending)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_pressure_cleanup_does_not_evict_current_batch_hits(self):
        ctrl = self._controller(cache_capacity=1)
        try:
            ids = torch.arange(60)
            protected = ids[:5]
            ctrl._cached_dev[ids] = True
            ctrl._ttl_dev[ids] = 10
            ctrl._ttl_dev[protected] = 1

            ctrl._run_cleanup(
                8, scan_ttl=False, protected_ids=protected
            )

            self.assertTrue(bool(ctrl._cached_dev[protected].all().item()))
            self.assertEqual(int(ctrl._cached_dev.sum().item()), 57)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_prefill_waits_for_gradients_before_cleanup(self):
        ctrl = self._controller()
        try:
            order = []
            ctrl._wait_pending_sync_now = lambda: order.append("gradients")
            ctrl._run_scheduled_cleanup = lambda: order.append("cleanup")

            ctrl.prefill_cache(None, torch.device("cpu"))

            self.assertEqual(order, ["gradients", "cleanup"])
        finally:
            ctrl._cleanup_queue.put(None)

    def test_prefill_consumes_handle_before_cleanup(self):
        ctrl = self._controller()
        try:
            order = []
            ctrl.embedding_module = type("EmbeddingModule", (), {})()
            ctrl._lookahead_ids.append((0, torch.tensor([1])))
            ctrl._prepared_ids[0] = None
            ctrl._prefetch_handles[0] = mock.sentinel.slot
            ctrl._prefetch_all_hits[0] = False
            ctrl._wait_pending_sync_now = lambda: order.append("gradients")
            ctrl._fill_from_preissued = (
                lambda slot, device: order.append("prefill") or False
            )
            ctrl._run_scheduled_cleanup = (
                lambda protected_ids=None: order.append("cleanup")
            )

            ctrl.prefill_cache(None, torch.device("cpu"))

            self.assertEqual(order, ["gradients", "prefill", "cleanup"])
        finally:
            ctrl._cleanup_queue.put(None)


class TestBagPipeUnknownIdAggregation(unittest.TestCase):
    """While the set is incomplete, ids outside it are unknown-ownership
    and must not be treated as local-only.

    The fallback set covers only the first max(lookahead, 2) steps, so an
    id outside it may still appear on another rank: applying it
    immediately as local-only and marking it dirty would let the eviction
    value writeback overwrite other ranks' increments (last-writer-wins)
    and fork the replicas permanently.
    """

    def _incomplete_harness(self):
        harness = _GradHarness(rank=0)
        harness._shared_ids_complete = False
        return harness

    def test_unknown_ids_are_aggregated_when_set_is_incomplete(self):
        harness = self._incomplete_harness()
        ids = torch.tensor([10, 20], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        # id 20 is outside the known shared set -> no local apply, no
        # dirty mark (writebacks would overwrite each other)
        self.assertEqual(harness.kv_client.best_effort_applies, [])
        self.assertFalse(bool(harness._dirty_dev[20].item()))
        # it goes through the sparse aggregation path instead, consumed at
        # the same barrier as the dense path
        self.assertEqual(len(harness.sparse_calls), 1)
        s_ids, s_grads, _ = harness.sparse_calls[0]
        self.assertEqual(s_ids.tolist(), [20])
        self.assertEqual(s_grads.tolist(), [[2.0, 2.0]])
        self.assertEqual(harness._stats["bagpipe_unknown_aggregated_ids"], 1.0)
        self.assertEqual(harness._stats["bagpipe_no_sync_ids"], 0.0)
        # known-shared id 10 still goes dense
        d_ids, _, _ = harness.dense_calls[0]
        self.assertEqual(d_ids.tolist(), [10])

    def test_oracle_set_keeps_local_only_fast_path(self):
        harness = _GradHarness(rank=0)
        self.assertTrue(harness._shared_ids_complete)
        ids = torch.tensor([10, 20], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        # the oracle set proves id 20 appears only on this rank -> the
        # fast path is available again
        self.assertEqual(len(harness.kv_client.best_effort_applies), 1)
        self.assertTrue(bool(harness._dirty_dev[20].item()))
        self.assertEqual(harness.sparse_calls, [])

    def test_dense_and_sparse_collectives_are_both_issued(self):
        """An empty bucket must not skip the collective either: the peer
        enters it, and an early exit hangs it."""
        harness = self._incomplete_harness()
        ids = torch.tensor([20], dtype=torch.int64)
        grads = torch.tensor([[2.0, 2.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        # This rank has no shared ids (empty dense bucket) yet still issues
        # the collective (matching the peer's all_reduce); sparse carries
        # the unknown id.
        self.assertEqual(len(harness.dense_calls), 1)
        self.assertEqual(harness.dense_calls[0][0].numel(), 0)
        self.assertEqual(len(harness.sparse_calls), 1)
        self.assertEqual(harness.sparse_calls[0][0].tolist(), [20])

    def test_large_shared_set_routes_shared_ids_through_sparse(self):
        """With a shared set above the threshold, shared ids fold into the
        sparse all_gather (communicating only the rows that actually appear
        this step) instead of all-reducing the whole shared-set buffer."""
        harness = _GradHarness(rank=0)
        # Oracle set (the unknown path never triggers); the shared set is
        # "large" -> sparse routing
        harness._dense_shared_max_rows = 0
        ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        # dense is not issued; the shared ids (10, 30) go through a single
        # sparse aggregation
        self.assertEqual(harness.dense_calls, [])
        self.assertEqual(len(harness.sparse_calls), 1)
        s_ids, s_grads, _ = harness.sparse_calls[0]
        self.assertEqual(s_ids.tolist(), [10, 30])
        self.assertEqual(s_grads.tolist(), [[1.0, 1.0], [4.0, 4.0]])
        # the local-only id 20 is still applied immediately
        self.assertEqual(len(harness.kv_client.best_effort_applies), 1)
        self.assertEqual(harness.kv_client.best_effort_applies[0][1].tolist(), [20])

    def test_large_shared_set_sparse_issued_even_with_empty_bucket(self):
        """Under sparse routing an empty local aggregation bucket must not
        skip the all_gather: the peer may have shared ids to exchange, and
        an early exit would hang it."""
        harness = _GradHarness(rank=0)
        harness._dense_shared_max_rows = 0
        # This batch has only the local-only id 20 (oracle-proven); the
        # aggregation bucket is empty
        ids = torch.tensor([20], dtype=torch.int64)
        grads = torch.tensor([[2.0, 2.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        self.assertEqual(harness.dense_calls, [])
        self.assertEqual(len(harness.sparse_calls), 1)
        self.assertEqual(harness.sparse_calls[0][0].numel(), 0)

    def test_barrier_applies_dense_and_sparse_aggregates_together(self):
        harness = _GradHarness(rank=0)
        harness._pending_sync_now_work = [
            _DenseWork(
                torch.tensor([10], dtype=torch.int64),
                torch.tensor([[1.0, 1.0]]),
            ),
            _DenseWork(
                torch.tensor([20], dtype=torch.int64),
                torch.tensor([[2.0, 2.0]]),
            ),
        ]
        harness._pending_sync_now_lr = 0.25

        harness._wait_pending_sync_now()

        # The two aggregation results merge into one in-place apply; the
        # applied rows are dirty (their local value is now ahead of the PS).
        self.assertEqual(len(harness.kv_client.best_effort_applies), 1)
        _, a_ids, a_grads, a_lr = harness.kv_client.best_effort_applies[0]
        self.assertEqual(sorted(a_ids.tolist()), [10, 20])
        self.assertEqual(a_grads.shape, (2, 2))
        self.assertEqual(a_lr, 0.25)
        self.assertEqual(len(harness.kv_client.async_updates), 0)
        self.assertTrue(bool(harness._dirty_dev[10].item()))
        self.assertTrue(bool(harness._dirty_dev[20].item()))
        self.assertEqual(harness._pending_sync_now_work, None)


class TestBagPipeAggregatedApply(unittest.TestCase):
    def test_update_grads_local_only_best_effort_and_shared_dense(self):
        harness = _GradHarness(rank=0)
        ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        # local-only id 20: best-effort in-place apply, dirty-marked
        self.assertEqual(len(harness.kv_client.best_effort_applies), 1)
        name, l_ids, l_grads, l_lr = harness.kv_client.best_effort_applies[0]
        self.assertEqual(name, "table")
        self.assertEqual(l_ids.tolist(), [20])
        self.assertEqual(l_grads.tolist(), [[2.0, 2.0]])
        self.assertEqual(l_lr, 0.1)
        self.assertTrue(bool(harness._dirty_dev[20].item()))
        # Lookup backfill registration: resident + TTL renewed
        # (batch 5 + margin 256)
        self.assertTrue(bool(harness._cached_dev[20].item()))
        self.assertEqual(int(harness._ttl_dev[20].item()), 5 + 256)
        # shared ids must NOT be applied locally
        # dense all_reduce got the shared subset (10, 30)
        d_ids, d_grads, _ = harness.dense_calls[0]
        self.assertEqual(sorted(d_ids.tolist()), [10, 30])
        self.assertEqual(d_grads.shape, (2, 2))
        # no PS push from update_grads itself (persistence is at the barrier)
        self.assertEqual(harness.kv_client.updates, [])

    def test_update_grads_without_lookup_info_keeps_shared_mirror_and_dirty(self):
        """Production lookups never record residency info (``last_lookup``
        is None: ``_bagpipe_last_lookup_resident`` has no production
        writer).  update_grads must leave the mirror untouched then.

        Clearing it dropped every shared row from the mirror each step: the
        fill marks them, the clear unmarked them, and the deferred
        aggregated apply skipped their dirty mark -- so with eviction-based
        persistence (no per-step PS push, no anti-entropy repair) their
        increments never reached the PS.  This is the production 2-rank
        correctness path, which the residency-recording protocol above does
        not cover."""
        harness = _GradHarness(rank=0)
        ids = torch.tensor([10, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)
        harness._wait_pending_sync_now()

        # Shared rows stay mirror-resident through update_grads ...
        self.assertTrue(bool(harness._cached_dev[10].item()))
        self.assertTrue(bool(harness._cached_dev[30].item()))
        # ... and the barrier apply marks them dirty, so the eviction
        # writeback / shutdown flush persist their aggregated increments.
        self.assertTrue(bool(harness._dirty_dev[10].item()))
        self.assertTrue(bool(harness._dirty_dev[30].item()))
        self.assertEqual(
            len(harness.kv_client.best_effort_applies), 1
        )

    def test_update_grads_uses_authoritative_lookup_residency(self):
        harness = _GradHarness(rank=0)
        ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
        # ID 20 is local-only but its no-evict insert failed.
        harness.embedding_module._bagpipe_last_lookup_resident = (
            ids,
            torch.tensor([True, False, True]),
        )

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        self.assertEqual(harness.kv_client.best_effort_applies, [])
        self.assertFalse(bool(harness._cached_dev[20].item()))
        self.assertFalse(bool(harness._dirty_dev[20].item()))
        self.assertTrue(bool(harness._cached_dev[10].item()))
        self.assertTrue(bool(harness._cached_dev[30].item()))
        self.assertIsNone(
            harness.embedding_module._bagpipe_last_lookup_resident
        )

    def test_lookup_does_not_renew_expired_resident_before_cleanup(self):
        harness = _GradHarness(rank=0)
        ids = torch.tensor([10], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0]])
        harness._ttl_dev[10] = 1
        harness.embedding_module._bagpipe_last_lookup_resident = (
            ids,
            torch.tensor([True]),
        )

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        self.assertEqual(int(harness._ttl_dev[10].item()), 1)

    def test_backfill_insert_failure_feeds_eviction_backstop(self):
        """A no-evict insert that fails on the lookup-backfill path must reach
        ``_insert_failures`` -- it is the only signal the cleanup backstop
        uses to retire low-TTL rows and free slots.

        Only the prefill path used to count these, so a cache that filled up
        through backfill kept failing to insert with no eviction reaction.
        """
        harness = _GradHarness(rank=0)
        ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
        harness.embedding_module._bagpipe_last_lookup_resident = (
            ids,
            torch.tensor([True, False, True]),
        )

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        self.assertEqual(harness._insert_failures, 1)
        self.assertEqual(harness._stats["bagpipe_insert_failures"], 1.0)

    def test_update_grads_single_all_reduce_on_side_stream(self):
        """Merged now/later: all shared ids go through a single all_reduce
        (side stream), no longer split into two paths by the recurrence
        window (both paths had identical scheduling; the split only paid
        an extra 6MB all_reduce)."""
        harness = _GradHarness(rank=0)
        harness._sync_later_stream = mock.sentinel.side_stream
        # id 10 recurs at batch 6 (formerly now), id 30's last use is
        # this very batch (formerly later)
        harness._latest_dev[torch.tensor([10])] = 6
        harness._latest_dev[torch.tensor([30])] = 5
        ids = torch.tensor([10, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=5)

        self.assertEqual(len(harness.dense_calls), 1)
        call_ids, _, call_stream = harness.dense_calls[0]
        self.assertEqual(call_ids.tolist(), [10, 30])
        self.assertIs(call_stream, mock.sentinel.side_stream)
        self.assertEqual(harness._stats["bagpipe_sync_now_ids"], 2.0)

    def test_barrier_applies_aggregated_on_all_ranks_without_ps_push(self):
        """No rank pushes to the PS at the barrier anymore (official BagPipe
        semantics): every rank applies the aggregate in place and marks the
        rows dirty; persistence rides on the eviction writeback."""
        for rank in (0, 1):
            harness = _GradHarness(rank=rank)
            harness._pending_sync_now_work = _DenseWork(
                torch.tensor([10], dtype=torch.int64),
                torch.tensor([[1.0, 1.0]]),
            )
            harness._pending_sync_now_lr = 0.25

            harness._wait_pending_sync_now()

            # every rank applies the aggregated grads in place
            self.assertEqual(len(harness.kv_client.best_effort_applies), 1)
            _, a_ids, a_grads, a_lr = harness.kv_client.best_effort_applies[0]
            self.assertEqual(a_ids.tolist(), [10])
            self.assertEqual(a_lr, 0.25)
            # no rank invalidates the aggregated ids anymore
            self.assertEqual(harness.kv_client.invalidations, [])
            # no rank pushes to the PS (no per-step persistence traffic)
            self.assertEqual(harness.kv_client.async_updates, [])
            self.assertEqual(harness.kv_client.updates, [])
            # the applied row is dirty: local value != PS value
            self.assertTrue(bool(harness._dirty_dev[10].item()))
            # entries stay cached (in-place update keeps them valid)
            self.assertTrue(bool(harness._cached_dev[10].item()))

    def test_aggregated_apply_marks_only_resident_rows_dirty(self):
        """A row the mirror does not hold was skipped by the best-effort
        apply, so it must not be marked dirty -- the shutdown flush would
        otherwise read undefined bytes for it and corrupt the PS."""
        harness = _GradHarness(rank=0)
        harness._cached_dev[torch.tensor([10, 30])] = False
        harness._pending_sync_now_work = _DenseWork(
            torch.tensor([10, 30], dtype=torch.int64),
            torch.tensor([[1.0, 1.0], [4.0, 4.0]]),
        )
        harness._pending_sync_now_lr = 0.25

        harness._wait_pending_sync_now()

        self.assertFalse(bool(harness._dirty_dev[10].item()))
        self.assertFalse(bool(harness._dirty_dev[30].item()))


class TestCompactTranslation(unittest.TestCase):
    def test_consume_stats_snapshot_never_double_counts(self):
        """consume_stats(reset=False) is a read-only snapshot: the GPU hot
        accumulators are merged into the returned value and then zeroed;
        later calls must not count the same batch twice."""
        from ..bagpipe_cache.controller import BagPipeCacheController

        ctrl = BagPipeCacheController(
            embedding_module=None,
            kv_client=None,
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=1000,
            embedding_dim=4,
            fuse_k=30,
            table_offsets={"a": 0},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 100},
        )
        try:
            ctrl._hot_add("bagpipe_no_sync_ids", torch.tensor(5.0))
            snap1 = ctrl.consume_stats(reset=False)
            self.assertEqual(snap1["bagpipe_no_sync_ids"], 5.0)
            # Accumulators zeroed and _stats kept (reset=False accumulates):
            # with no new counts the snapshot value is unchanged -- the old
            # behavior (reset=False not zeroing the accumulators) would
            # have made it 10.0.
            snap2 = ctrl.consume_stats(reset=False)
            self.assertEqual(snap2["bagpipe_no_sync_ids"], 5.0)
            # reset=True must return the whole accumulated interval (5+3)
            # and then zero -- counts during the read-only snapshot period
            # are neither swallowed nor double-counted.
            ctrl._hot_add("bagpipe_no_sync_ids", torch.tensor(3.0))
            snap3 = ctrl.consume_stats(reset=True)
            self.assertEqual(snap3["bagpipe_no_sync_ids"], 8.0)
            snap4 = ctrl.consume_stats(reset=True)
            self.assertEqual(snap4["bagpipe_no_sync_ids"], 0.0)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_consume_stats_per_step_rows_and_live_state_snapshots(self):
        """consume_stats(reset=True) is the per-step export path:

        - state-snapshot keys (lookahead / cleanup_interval / shared_ids)
          are refreshed from live state on every call, so a mid-training
          change (dynamic lookahead, prescan building the shared set) is
          visible in the next row instead of the init-time value;
        - counts are per-step deltas, not cumulative (each call zeroes);
        - removed dead keys (write points deleted before stats export
          ever shipped) stay absent."""
        from ..bagpipe_cache.controller import BagPipeCacheController

        ctrl = BagPipeCacheController(
            embedding_module=None,
            kv_client=None,
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=1000,
            embedding_dim=4,
            fuse_k=30,
            table_offsets={"a": 0},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 100},
        )
        try:
            row1 = ctrl.consume_stats(reset=True)
            self.assertEqual(row1["bagpipe_lookahead"], 4.0)
            self.assertEqual(row1["bagpipe_shared_ids"], 0.0)
            # Dynamic lookahead changes mid-training (opt 9): the next
            # exported row must reflect the new value, not the frozen
            # init-time one that reset_stats seeding produced.
            ctrl._dynamic_lookahead = 8
            ctrl._shared_ids = {1, 2, 3}
            row2 = ctrl.consume_stats(reset=True)
            self.assertEqual(row2["bagpipe_lookahead"], 8.0)
            self.assertEqual(row2["bagpipe_shared_ids"], 3.0)
            # Per-step deltas: each reset=True call returns only the counts
            # accumulated since the previous one.
            ctrl._hot_add("bagpipe_no_sync_ids", torch.tensor(2.0))
            ctrl._hot_add("bagpipe_sync_now_ids", torch.tensor(7.0))
            row3 = ctrl.consume_stats(reset=True)
            self.assertEqual(row3["bagpipe_no_sync_ids"], 2.0)
            self.assertEqual(row3["bagpipe_sync_now_ids"], 7.0)
            row4 = ctrl.consume_stats(reset=True)
            self.assertEqual(row4["bagpipe_no_sync_ids"], 0.0)
            self.assertEqual(row4["bagpipe_sync_now_ids"], 0.0)
            # Dead keys (no write point since before stats export shipped)
            # must not resurface.
            for dead in (
                "bagpipe_prefetch_pruned",
                "bagpipe_prefetch_local_nosync_kept",
                "bagpipe_sync_later_ids",
                "bagpipe_sgd_cache_fallback",
                "bagpipe_eviction_stream_ms",
                "bagpipe_eviction_stream_overlap_ms",
            ):
                self.assertNotIn(dead, row4)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_fused_compact_roundtrip_and_space(self):
        """fused id (table<<fuse_k|row) -> compact (table<<shift|row)
        round-trips consistently; bookkeeping space = num_tables << shift
        (not blown up by the nominal fuse_k=30 space)."""
        from ..bagpipe_cache.controller import BagPipeCacheController

        ctrl = BagPipeCacheController(
            embedding_module=None,
            kv_client=None,
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=1000,
            embedding_dim=4,
            fuse_k=30,
            table_offsets={"a": 0, "b": 1 << 30},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 200000, "b": 100},
        )
        try:
            # stride = next_pow2(200000) = 262144 = 2^18
            self.assertEqual(ctrl._bk_stride, 262144)
            self.assertEqual(ctrl._latest_dev.numel(), 2 << 18)
            fused = torch.tensor(
                [5, (1 << 30) + 7, (1 << 30) + 99], dtype=torch.int64
            )
            compact = ctrl._to_compact(fused)
            self.assertEqual(
                compact.tolist(), [5, (1 << 18) + 7, (1 << 18) + 99]
            )
            self.assertEqual(ctrl._to_fused(compact).tolist(), fused.tolist())
            # An out-of-range row (>= stride) has its compact land in the
            # next table's segment; callers filter it
            oob = ctrl._to_compact(torch.tensor([(1 << 30) + 300000]))
            self.assertEqual(
                ctrl._compact_in_range(oob).numel(), 0
            )
        finally:
            ctrl._cleanup_queue.put(None)



class TestPrefetchResidencyValidation(unittest.TestCase):
    def _controller(self, kv_client):
        from ..bagpipe_cache.controller import BagPipeCacheController

        return BagPipeCacheController(
            embedding_module=None,
            kv_client=kv_client,
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=10,
            embedding_dim=4,
            fuse_k=6,
            table_offsets={"a": 0},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 100},
        )

    def test_stale_mirror_is_repaired_before_hit_only_lookup(self):
        class KV:
            def __init__(self):
                self.prefetched = []

            def contains_gpu_cache(self, keys):
                # ID 20 was evicted behind the Python mirror's back.
                return keys == 10

            def prefetch(self, ids):
                self.prefetched.append(ids.clone())
                return 777

        kv = KV()
        ctrl = self._controller(kv)
        try:
            ids = torch.tensor([10, 20], dtype=torch.int64)
            compact = ctrl._to_compact(ids)
            ctrl._latest_dev[compact] = 7
            ctrl._ttl_dev[compact] = 100
            ctrl._cached_dev[compact] = True

            ctrl._preissue_prefetch(7, ids, compact)

            self.assertFalse(bool(ctrl._cached_dev[20].item()))
            self.assertTrue(bool(ctrl._cached_dev[10].item()))
            self.assertFalse(ctrl._prefetch_all_hits[7])
            self.assertEqual(kv.prefetched[0].tolist(), [20])
            self.assertEqual(ctrl._prefetch_handles[7].ids_cpu.tolist(), [20])
        finally:
            ctrl._cleanup_queue.put(None)

    def test_validated_all_hits_skip_prefetch_and_enable_fast_lookup(self):
        class KV:
            def contains_gpu_cache(self, keys):
                return torch.ones_like(keys, dtype=torch.bool)

            def prefetch(self, ids):
                raise AssertionError("all-hit batch must not prefetch")

        ctrl = self._controller(KV())
        try:
            ids = torch.tensor([10, 20], dtype=torch.int64)
            compact = ctrl._to_compact(ids)
            ctrl._latest_dev[compact] = 7
            ctrl._ttl_dev[compact] = 100
            ctrl._cached_dev[compact] = True

            ctrl._preissue_prefetch(7, ids, compact)

            self.assertIsNone(ctrl._prefetch_handles[7])
            self.assertTrue(ctrl._prefetch_all_hits[7])
        finally:
            ctrl._cleanup_queue.put(None)

    def test_expired_dirty_resident_row_is_prefetched_for_anti_entropy(self):
        class KV:
            def contains_gpu_cache(self, keys):
                return torch.ones_like(keys, dtype=torch.bool)

            def prefetch(self, ids):
                return 123

        ctrl = self._controller(KV())
        try:
            ids = torch.tensor([10], dtype=torch.int64)
            compact = ctrl._to_compact(ids)
            ctrl._latest_dev[compact] = 7
            ctrl._ttl_dev[compact] = 1
            ctrl._cached_dev[compact] = True
            ctrl._dirty_dev[compact] = True

            ctrl._preissue_prefetch(7, ids, compact)

            self.assertEqual(ctrl._prefetch_handles[7].handle, 123)
            self.assertFalse(ctrl._prefetch_all_hits[7])
            self.assertEqual(ctrl._prefetch_handles[7].ids_cpu.tolist(), [10])
            self.assertEqual(int(ctrl._ttl_dev[compact].item()), 1)
            self.assertTrue(bool(ctrl._dirty_dev[compact].item()))
        finally:
            ctrl._cleanup_queue.put(None)

    def test_insert_upper_bound_counts_prefetch_targets_once(self):
        class KV:
            def prefetch(self, ids):
                return 123

        ctrl = self._controller(KV())
        try:
            ids = torch.tensor([10, 20], dtype=torch.int64)
            compact = ctrl._to_compact(ids)
            ctrl._latest_dev[compact] = 7
            ctrl._ttl_dev[10] = 100
            ctrl._cached_dev[10] = True

            ctrl._preissue_prefetch(7, ids, compact)

            self.assertEqual(ctrl._cached_insert_attempts, 1)
            self.assertEqual(ctrl._prefetch_handles[7].ids_cpu.tolist(), [20])
        finally:
            ctrl._cleanup_queue.put(None)

    def test_expired_resident_prefill_preserves_value_and_expiry(self):
        class KV:
            def wait_and_get(self, handle, embedding_dim, device):
                return torch.full((1, embedding_dim), -1.0)

            def prefill_gpu_cache_no_evict(self, name, ids, values):
                # The C++ contract reports post-fill residency.  A resident
                # row returns true even though its value was not replaced.
                return torch.ones_like(ids, dtype=torch.bool)

        from ..bagpipe_cache.types import PrefetchSlot

        ctrl = self._controller(KV())
        try:
            ctrl._cached_dev[10] = True
            ctrl._dirty_dev[10] = True
            ctrl._ttl_dev[10] = 1
            slot = PrefetchSlot(
                handle=123,
                ids_cpu=torch.tensor([10]),
                ttl=263,
                issue_ts=0.0,
                num_ids=1,
            )

            all_hits = ctrl._fill_from_preissued(slot, torch.device("cpu"))

            self.assertFalse(all_hits)
            self.assertEqual(int(ctrl._ttl_dev[10].item()), 1)
            self.assertTrue(bool(ctrl._dirty_dev[10].item()))
            self.assertEqual(ctrl._insert_failures, 0)
        finally:
            ctrl._cleanup_queue.put(None)

    def test_external_cache_reset_fails_loudly(self):
        class KV:
            def __init__(self):
                self.generation = 10

            def get_gpu_cache_generation(self):
                return self.generation

        kv = KV()
        ctrl = self._controller(kv)
        try:
            kv.generation = 11
            with self.assertRaisesRegex(RuntimeError, "generation changed"):
                ctrl._preissue_prefetch(
                    7,
                    torch.tensor([10], dtype=torch.int64),
                    torch.tensor([10], dtype=torch.int64),
                )
        finally:
            ctrl._cleanup_queue.put(None)



class _EvictionHarness:
    """Minimal host for BagPipeEvictionMixin: identity mapping + fixed
    64-slot bookkeeping."""

    def __init__(self, fused_ids):
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin  # noqa: F401

        self.device = torch.device("cpu")
        self._latest_dev = torch.zeros(64, dtype=torch.int32)
        self._ttl_dev = torch.zeros(64, dtype=torch.int32)
        self._cached_dev = torch.zeros(64, dtype=torch.bool)
        self._dirty_dev = torch.zeros(64, dtype=torch.bool)
        self._cached_dev[torch.tensor(fused_ids)] = True
        self._shared_ids_tensor = None
        self._eviction_stream = None
        self.embedding_dim = 2
        self._stats = {
            "bagpipe_writeback_ids": 0.0,
            "bagpipe_evicted_ids": 0.0,
            "bagpipe_evict_stale_mirror_ids": 0.0,
        }
        self.written = []

    def _is_distributed(self):
        return False

    def _get_rank(self):
        return 0

    def _compact_in_range(self, compact):
        return compact[compact < self._dirty_dev.numel()]

    def _to_compact(self, ids):
        return ids

    def _to_fused(self, compact):
        return compact

    def _wait_pending_sync_now(self):
        return None

    def _wait_prev_sync_later(self):
        return None


class TestShutdownFlush(unittest.TestCase):
    def test_flush_writes_all_dirty_rows_including_shared(self):
        """With no per-step PS push, a dirty shared row carries unpersisted
        aggregated increments and MUST be flushed at shutdown (single rank:
        the owner sharding keeps every row)."""
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin

        class _WithMixin(BagPipeEvictionMixin, _EvictionHarness):
            def _cleanup_loop(self):
                return None

        m = _WithMixin([5, 10, 20])
        m._dirty_dev[torch.tensor([5, 10, 20])] = True
        m._shared_ids_tensor = torch.tensor([10], dtype=torch.int64)
        m._cleanup_queue = queue.Queue()

        class _KV:
            def gpu_cache_lookup_flat(self, keys, embedding_dim):
                n = int(keys.numel())
                return torch.arange(n * embedding_dim, dtype=torch.float32).reshape(
                    n, embedding_dim
                )

            def emb_write_values(self, name, ids, vals):
                m.written.append((name, ids.clone()))

        m.kv_client = _KV()
        m.master_table_name = "t"

        m.shutdown()

        # All dirty rows are written back (5/20 local-only, 10 shared).
        self.assertEqual(sorted(i.tolist() for _, i in m.written), [[5, 10, 20]])
        self.assertFalse(bool(m._dirty_dev[5].item()))
        self.assertFalse(bool(m._dirty_dev[10].item()))
        self.assertFalse(bool(m._dirty_dev[20].item()))
        # The background-thread exit sentinel (None) was posted; no tasks
        # remain in the queue
        self.assertIsNone(m._cleanup_queue.get_nowait())


class TestAuthoritativeInvalidation(unittest.TestCase):
    def test_partial_removal_writes_back_only_removed_rows(self):
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin

        class _WithMixin(BagPipeEvictionMixin, _EvictionHarness):
            pass

        m = _WithMixin([5, 10, 20])
        m._dirty_dev[torch.tensor([5, 10, 20])] = True
        m._ttl_dev[torch.tensor([5, 10, 20])] = 100
        m._cleanup_queue = queue.Queue()

        class KV:
            def contains_gpu_cache(self, ids):
                return torch.tensor([True, False, True])

            def gpu_cache_lookup_flat(self, keys, embedding_dim):
                n = int(keys.numel())
                return torch.arange(
                    n * embedding_dim, dtype=torch.float32
                ).reshape(n, embedding_dim)

            def invalidate_gpu_cache_with_mask(self, name, ids):
                self.ids = ids.clone()
                self.name = name
                return torch.ones_like(ids, dtype=torch.bool)

            def emb_write_values(self, name, ids, vals):
                m.written.append((name, ids.clone()))

        m.kv_client = KV()
        m.master_table_name = "t"
        m._evict_entries(torch.tensor([5, 10, 20]))

        # 10 only ever existed in the mirror, so there is nothing to write
        # back for it; the genuinely resident rows are persisted first.
        self.assertEqual(m.written[0][1].tolist(), [5, 20])
        self.assertEqual(m._stats["bagpipe_writeback_ids"], 2.0)
        self.assertFalse(bool(m._cached_dev[5].item()))
        self.assertFalse(bool(m._cached_dev[10].item()))
        self.assertFalse(bool(m._cached_dev[20].item()))
        self.assertFalse(bool(m._dirty_dev[5].item()))
        self.assertFalse(bool(m._dirty_dev[10].item()))
        self.assertFalse(bool(m._dirty_dev[20].item()))
        self.assertEqual(int(m._ttl_dev[10].item()), 0)
        self.assertEqual(m._stats["bagpipe_evicted_ids"], 2.0)
        self.assertEqual(m._stats["bagpipe_evict_stale_mirror_ids"], 1.0)

    def test_stale_mirror_row_is_not_rewritten_or_reselected(self):
        """A row the mirror calls resident but the C++ cache does not hold has
        to leave the mirror.

        Otherwise it is never reported as removed, so it is never cleared, so
        it stays expired and the very next cleanup picks it again -- an
        unbounded re-eviction loop that also writes back whatever the miss
        path read for a row that is not there.
        """
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin

        class _WithMixin(BagPipeEvictionMixin, _EvictionHarness):
            pass

        m = _WithMixin([5, 10, 20])
        m._dirty_dev[torch.tensor([5, 10, 20])] = True
        m._ttl_dev[torch.tensor([5, 10, 20])] = 100
        m._cleanup_queue = queue.Queue()

        class KV:
            def contains_gpu_cache(self, ids):
                return torch.tensor([True, False, True])

            def gpu_cache_lookup_flat(self, keys, embedding_dim):
                n = int(keys.numel())
                return torch.arange(
                    n * embedding_dim, dtype=torch.float32
                ).reshape(n, embedding_dim)

            def invalidate_gpu_cache_with_mask(self, name, ids):
                # 10 is gone from the cache; only the mirror still lists it.
                return torch.ones_like(ids, dtype=torch.bool)

            def emb_write_values(self, name, ids, vals):
                m.written.append((name, ids.clone()))

        m.kv_client = KV()
        m.master_table_name = "t"
        m._evict_entries(torch.tensor([5, 10, 20]))

        self.assertEqual(m.written[0][1].tolist(), [5, 20])

        expired_again = (m._cached_dev & (m._ttl_dev < 101)).nonzero()
        self.assertEqual(expired_again.tolist(), [])


class TestPrescanSparseAllGather(unittest.TestCase):
    """_all_gather_sparse_prescan_async: GPU-side sentinel padding with
    oracle-provided sizing -- no count exchange, no .item()."""

    def test_padding_compacts_masked_rows_in_order(self):
        harness = _CommHarness()
        ids = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        # rows 0 and 2 are in the bucket; max_n = 3
        mask = torch.tensor([True, False, True, False])
        captured = {}

        def fake_all_gather(out_list, tensor, async_op=False):
            for out in out_list:
                out.copy_(tensor)
            captured.setdefault("inputs", []).append(tensor.clone())
            return _DoneWork()

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(comm_module.dist, "all_gather",
                               side_effect=fake_all_gather):
            _, _, work = harness._all_gather_sparse_prescan_async(
                ids, grads, mask, max_n=3
            )
            work.wait()

        padded_ids = captured["inputs"][0]
        # masked ids in cumsum order, remaining rows sentinel-padded
        self.assertEqual(padded_ids.tolist(), [10, 12, -1])
        # no count exchange: the payload all_gathers are the only collectives
        self.assertEqual(len(captured["inputs"]), 2)
        agg_ids, agg_grads = work.result
        self.assertEqual(agg_ids.tolist(), [10, 12])
        self.assertEqual(agg_grads.tolist(), [[2.0, 2.0], [6.0, 6.0]])

    def test_zero_max_skips_collectives(self):
        harness = _CommHarness()
        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(comm_module.dist, "all_gather") as ag:
            _, _, work = harness._all_gather_sparse_prescan_async(
                torch.tensor([10], dtype=torch.int64),
                torch.tensor([[1.0, 1.0]]),
                torch.tensor([True]),
                max_n=0,
            )
        ag.assert_not_called()
        self.assertIsNone(work)

    def test_empty_mask_is_all_sentinel(self):
        harness = _CommHarness()

        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            _, _, work = harness._all_gather_sparse_prescan_async(
                torch.tensor([10, 11], dtype=torch.int64),
                torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                torch.tensor([False, False]),
                max_n=2,
            )
            work.wait()

        agg_ids, _ = work.result
        self.assertEqual(agg_ids.numel(), 0)


class TestPrescanCountExchange(unittest.TestCase):
    """finalize_prescan also exchanges per-batch shared-ID counts so
    update_grads can size the sparse all_gather without communication."""

    def _harness_with_batches(self):
        harness = _SharedIdSetHarness()
        harness._prescan_unique_ids = {10, 20, 30}
        harness._prescan_batch_ids = [
            torch.tensor([10, 20], dtype=torch.int64),
            torch.tensor([20], dtype=torch.int64),
            torch.tensor([30], dtype=torch.int64),
        ]
        return harness

    def test_finalize_builds_sparse_max_per_batch(self):
        harness = self._harness_with_batches()
        # Fake 2-rank all_gather: the peer mirrors our contribution, so the
        # per-batch maximum equals the local |shared ∩ batch| counts.
        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            harness.finalize_prescan()
        # shared set from the fake exchange = {10, 20, 30} (both "ranks"
        # contribute the same ids); per-batch hits: [2, 1, 1]
        self.assertEqual(harness._shared_ids, {10, 20, 30})
        self.assertEqual(harness._prescan_sparse_max, [2, 1, 1])
        self.assertEqual(harness._prescan_batch_numels, [2, 1, 1])
        # per-batch tensors are released after the exchange
        self.assertIsNone(harness._prescan_batch_ids)

    def test_finalize_without_batches_leaves_metadata_empty(self):
        harness = _SharedIdSetHarness()
        harness._prescan_unique_ids = {10, 20}
        with mock.patch.object(comm_module.dist, "get_world_size", return_value=2), \
             mock.patch.object(
                 comm_module.dist, "all_gather",
                 side_effect=_fake_two_rank_all_gather,
             ):
            harness.finalize_prescan()
        self.assertIsNone(harness._prescan_sparse_max)
        self.assertEqual(harness._prescan_batch_numels, [])


class TestPrescanRoutingInUpdateGrads(unittest.TestCase):
    def test_oracle_path_routes_through_prescan_sizing(self):
        harness = _GradHarness(rank=0)
        harness._dense_shared_max_rows = 0  # force sparse routing
        harness._prescan_sparse_max = [2, 1, 4]
        ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=4)
        # batch_num 4 wraps to prescan index 4 % 3 = 1 -> max_n = 1
        self.assertEqual(len(harness.prescan_sparse_calls), 1)
        _, _, mask, max_n, _ = harness.prescan_sparse_calls[0]
        self.assertEqual(max_n, 1)
        self.assertEqual(mask.tolist(), [True, False, True])
        # the count-exchange path must not run alongside
        self.assertEqual(harness.sparse_calls, [])

    def test_fallback_path_used_without_prescan_metadata(self):
        harness = _GradHarness(rank=0)
        harness._dense_shared_max_rows = 0
        ids = torch.tensor([10, 30], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [4.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=0)

        self.assertEqual(harness.prescan_sparse_calls, [])
        self.assertEqual(len(harness.sparse_calls), 1)
        s_ids, _, _ = harness.sparse_calls[0]
        self.assertEqual(s_ids.tolist(), [10, 30])

    def test_incomplete_set_never_uses_prescan_sizing(self):
        """The unknown-ownership bucket includes ids outside the shared set,
        so its size is not bounded by the prescan counts."""
        harness = _GradHarness(rank=0)
        harness._shared_ids_complete = False
        harness._dense_shared_max_rows = 0
        harness._prescan_sparse_max = [2]
        ids = torch.tensor([10, 20], dtype=torch.int64)
        grads = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=0)

        self.assertEqual(harness.prescan_sparse_calls, [])
        self.assertEqual(len(harness.sparse_calls), 1)
        # shared 10 + unknown 20 both fold into the count-exchange path
        self.assertEqual(harness.sparse_calls[0][0].tolist(), [10, 20])


class TestPrescanAlignmentTripwire(unittest.TestCase):
    """enqueue fails loudly when the training order diverged from prescan."""

    def _controller(self):
        from ..bagpipe_cache.controller import BagPipeCacheController

        class KV:
            def contains_gpu_cache(self, keys):
                return torch.ones_like(keys, dtype=torch.bool)

        return BagPipeCacheController(
            embedding_module=None,
            kv_client=KV(),
            lookahead_value=4,
            cleanup_batch_proportion=0.25,
            cache_capacity=10,
            embedding_dim=4,
            fuse_k=6,
            table_offsets={"a": 0},
            master_table_name="t",
            device=torch.device("cpu"),
            id_extractor=lambda sf: sf,
            table_sizes={"a": 100},
        )

    def test_enqueue_rejects_diverged_batch(self):
        ctrl = self._controller()
        try:
            ctrl._prescan_batch_numels = [3, 2]
            with self.assertRaisesRegex(RuntimeError, "alignment broken"):
                ctrl.enqueue(torch.tensor([7, 8], dtype=torch.int64))
        finally:
            ctrl._cleanup_queue.put(None)

    def test_enqueue_accepts_matching_and_wrapped_batches(self):
        ctrl = self._controller()
        try:
            ctrl._prescan_batch_numels = [3, 2]
            ticket = ctrl.enqueue(torch.tensor([7, 8, 9], dtype=torch.int64))
            self.assertEqual(int(ticket[0].numel()), 3)
            ctrl.enqueue(torch.tensor([1, 2], dtype=torch.int64))
            # batch 2 wraps to prescan index 0 (numel 3)
            ctrl.enqueue(torch.tensor([1, 2, 3], dtype=torch.int64))
            self.assertEqual(ctrl._next_enqueue_batch, 3)
        finally:
            ctrl._cleanup_queue.put(None)


class TestSharedRowEvictionWriteback(unittest.TestCase):
    """Eviction writeback keys on dirty alone: a shared row advanced by the
    aggregated apply is persisted when evicted (no per-step push)."""

    def _harness(self):
        from ..bagpipe_cache.eviction import BagPipeEvictionMixin

        class _WithMixin(BagPipeEvictionMixin, _EvictionHarness):
            pass

        m = _WithMixin([5, 10, 20])
        m._ttl_dev[torch.tensor([5, 10, 20])] = 100
        m._cleanup_queue = queue.Queue()

        class KV:
            def contains_gpu_cache(self, ids):
                return torch.ones_like(ids, dtype=torch.bool)

            def gpu_cache_lookup_flat(self, keys, embedding_dim):
                n = int(keys.numel())
                return torch.arange(
                    n * embedding_dim, dtype=torch.float32
                ).reshape(n, embedding_dim)

            def emb_write_values(self, name, ids, vals):
                m.written.append((name, ids.clone()))

            def invalidate_gpu_cache_with_mask(self, name, ids):
                return torch.ones_like(ids, dtype=torch.bool)

        m.kv_client = KV()
        m.master_table_name = "t"
        return m

    def test_dirty_shared_row_is_written_back_at_eviction(self):
        m = self._harness()
        m._shared_ids_tensor = torch.tensor([10], dtype=torch.int64)
        # 10 is shared and dirty (aggregated apply advanced it); 5 is dirty
        # local-only; 20 is clean (never applied) -> no writeback needed.
        m._dirty_dev[torch.tensor([5, 10])] = True

        m._evict_entries(torch.tensor([5, 10, 20]))

        self.assertEqual(m.written[0][1].tolist(), [5, 10])
        self.assertFalse(bool(m._dirty_dev[5].item()))
        self.assertFalse(bool(m._dirty_dev[10].item()))

    def test_clean_rows_skip_writeback(self):
        m = self._harness()
        m._evict_entries(torch.tensor([5, 10, 20]))
        self.assertEqual(m.written, [])

    def test_writeback_failure_keeps_dirty_row_resident(self):
        m = self._harness()
        m._dirty_dev[5] = True

        def fail_write(name, ids, vals):
            raise RuntimeError("PS unavailable")

        m.kv_client.emb_write_values = fail_write
        removed = m._evict_entries(torch.tensor([5]))

        self.assertEqual(removed, 0)
        self.assertTrue(bool(m._cached_dev[5].item()))
        self.assertTrue(bool(m._dirty_dev[5].item()))
        self.assertEqual(int(m._ttl_dev[5].item()), 100)

    def test_partial_shard_failure_reconciles_completed_rows(self):
        m = self._harness()
        m._dirty_dev[torch.tensor([5, 10])] = True
        calls = 0

        def contains(ids):
            nonlocal calls
            calls += 1
            if calls == 1:
                return torch.ones_like(ids, dtype=torch.bool)
            return torch.tensor([False, True])

        m.kv_client.contains_gpu_cache = contains

        def fail_after_first_shard(name, ids, vals):
            raise RuntimeError("second shard unavailable")

        m.kv_client.emb_write_values = fail_after_first_shard
        removed = m._evict_entries(torch.tensor([5, 10]))

        self.assertEqual(removed, 1)
        self.assertFalse(bool(m._cached_dev[5].item()))
        self.assertFalse(bool(m._dirty_dev[5].item()))
        self.assertTrue(bool(m._cached_dev[10].item()))
        self.assertTrue(bool(m._dirty_dev[10].item()))


if __name__ == "__main__":
    unittest.main()
