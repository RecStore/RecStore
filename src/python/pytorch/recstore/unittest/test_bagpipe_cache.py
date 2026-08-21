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

    def update_async(self, name, ids, grads):
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
        self.master_table_name = "table"
        self.lr = 0.01
        # 扁平张量簿记 (与 controller._alloc_bookkeeping 同构, 固定 64 大小)
        self._latest_dev = torch.zeros(64, dtype=torch.int32)
        self._ttl_dev = torch.zeros(64, dtype=torch.int32)
        self._cached_dev = torch.zeros(64, dtype=torch.bool)
        self._dirty_dev = torch.zeros(64, dtype=torch.bool)
        self._cached_dev[torch.tensor([10, 20, 30])] = True
        self._sync_later_stream = None
        self._pending_sync_now_work = None
        self._pending_sync_now_lr = None
        self._pending_ps_push_handle = None
        self._anti_entropy_step = 0
        self._anti_entropy_interval = 50
        self._anti_entropy_ids = 200
        self._ttl_margin = 256
        self._shared_ids = {10, 30}
        self._shared_ids_tensor = torch.tensor([10, 30], dtype=torch.int64)
        # 无 GPU 热累加器 → _hot_add 回退 host float (直接进 _stats)
        self._hot_stats_dev = {}
        self.dense_calls = []
        self._stats = {
            "bagpipe_update_ms": 0.0,
            "bagpipe_sgd_cache_success": 0.0,
            "bagpipe_sgd_cache_fallback": 0.0,
            "bagpipe_all_reduce_ms": 0.0,
            "bagpipe_sync_now_overlap_ms": 0.0,
            "bagpipe_sync_now_ids": 0.0,
            "bagpipe_sync_later_ids": 0.0,
            "bagpipe_no_sync_ids": 0.0,
            "bagpipe_anti_entropy_ids": 0.0,
        }

    def _is_distributed(self):
        return True

    def _get_rank(self):
        return self._rank

    # 身份映射: 单测直接用 fused id 当 compact 索引 (固定 64 大小簿记)
    def _to_compact(self, ids):
        return ids

    def _to_fused(self, compact):
        return compact

    def _compact_in_range(self, compact):
        return compact[compact < self._cached_dev.numel()]

    def _maybe_build_shared_id_set(self, _ids):
        return None

    def _hot_add(self, key, count):
        # 单测 harness 无 GPU 热累加器: 直接累加 host float (与控制器
        # _hot_stats_dev 为空时的回退分支同构)
        if isinstance(count, torch.Tensor):
            count = float(count.item())
        self._stats[key] = self._stats.get(key, 0.0) + count

    def _dense_all_reduce_async(self, ids, grads, stream=None):
        self.dense_calls.append((ids.clone(), grads.clone(), stream))
        return ids, grads, _DenseWork(ids, grads)


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
        # lookup 回填登记: 驻留 + TTL 续期 (batch 5 + margin 256)
        self.assertTrue(bool(harness._cached_dev[20].item()))
        self.assertEqual(int(harness._ttl_dev[20].item()), 5 + 256)
        # shared ids must NOT be applied locally
        # dense all_reduce got the shared subset (10, 30)
        d_ids, d_grads, _ = harness.dense_calls[0]
        self.assertEqual(sorted(d_ids.tolist()), [10, 30])
        self.assertEqual(d_grads.shape, (2, 2))
        # no PS push from update_grads itself (persistence is at the barrier)
        self.assertEqual(harness.kv_client.updates, [])

    def test_update_grads_single_all_reduce_on_side_stream(self):
        """合并 now/later: 全部共享 id 走单次 all_reduce (侧流), 不再按
        重现窗口切两条路 (两条路调度相同, 双路只是多付一次 6MB all_reduce)。"""
        harness = _GradHarness(rank=0)
        harness._sync_later_stream = mock.sentinel.side_stream
        # id 10 将在批 6 重现 (原 now), id 30 最近一次就是本批 (原 later)
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

    def test_barrier_applies_aggregated_on_all_ranks_and_pushes_on_rank0(self):
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
            # rank0 persists via the depth-1 async pipeline, other ranks don't
            if rank == 0:
                self.assertEqual(len(harness.kv_client.async_updates), 1)
                self.assertIsNotNone(harness._pending_ps_push_handle)
            else:
                self.assertEqual(harness.kv_client.async_updates, [])
                self.assertIsNone(harness._pending_ps_push_handle)
            # entries stay cached (in-place update keeps them valid)
            self.assertTrue(bool(harness._cached_dev[10].item()))

    def test_anti_entropy_invalidates_sample_on_interval(self):
        harness = _GradHarness(rank=0)
        harness._anti_entropy_step = 49
        harness._anti_entropy_interval = 50
        harness._anti_entropy_ids = 2
        harness._shared_ids = {10, 20, 30}
        harness._shared_ids_tensor = torch.tensor([10, 20, 30], dtype=torch.int64)
        harness._pending_sync_now_work = None

        harness._wait_pending_sync_now()

        self.assertEqual(len(harness.kv_client.invalidations), 1)
        invalidated = harness.kv_client.invalidations[0][1].tolist()
        self.assertEqual(len(invalidated), 2)
        self.assertTrue(set(invalidated).issubset({10, 20, 30}))
        for fid in invalidated:
            self.assertFalse(bool(harness._cached_dev[fid].item()))
            self.assertFalse(bool(harness._dirty_dev[fid].item()))
            self.assertEqual(int(harness._ttl_dev[fid].item()), 0)
        # off-interval steps do nothing
        harness._anti_entropy_step = 50
        harness._wait_pending_sync_now()
        self.assertEqual(len(harness.kv_client.invalidations), 1)


class TestCompactTranslation(unittest.TestCase):
    def test_fused_compact_roundtrip_and_space(self):
        """fused id (table<<fuse_k|row) → compact (table<<shift|row) 往返一致,
        簿记空间 = num_tables << shift (不被 fuse_k=30 的名义空间撑爆)."""
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
            # 越界 row (>= stride) 的 compact 会落入下一表区段, 由调用侧过滤
            oob = ctrl._to_compact(torch.tensor([(1 << 30) + 300000]))
            self.assertEqual(
                ctrl._compact_in_range(oob).numel(), 0
            )
        finally:
            ctrl._cleanup_queue.put(None)


class _EvictionHarness:
    """BagPipeEvictionMixin 的最小宿主: 身份映射 + 固定 64 槽簿记."""

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
        self._stats = {"bagpipe_writeback_ids": 0.0}
        self.written = []

    def _to_compact(self, ids):
        return ids

    def _to_fused(self, compact):
        return compact

    def _wait_pending_sync_now(self):
        return None

    def _wait_prev_sync_later(self):
        return None


class TestShutdownFlush(unittest.TestCase):
    def test_flush_writes_dirty_non_shared_and_clears_flags(self):
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

        # 只有 dirty 且非 shared 的 5/20 被写回 (10 是 shared)
        self.assertEqual(sorted(i.tolist() for _, i in m.written), [[5, 20]])
        self.assertFalse(bool(m._dirty_dev[5].item()))
        self.assertFalse(bool(m._dirty_dev[20].item()))
        # shared 的 dirty 标记保留 (其持久化走聚合推送)
        self.assertTrue(bool(m._dirty_dev[10].item()))
        # 后台线程退出哨兵 (None) 已投递, 队列无残留任务
        self.assertIsNone(m._cleanup_queue.get_nowait())
        self.assertTrue(m._cleanup_queue.empty())


if __name__ == "__main__":
    unittest.main()
