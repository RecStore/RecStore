import unittest
from unittest import mock

import torch

from ..bagpipe_cache import comm as comm_module
from ..bagpipe_cache import grads as grads_module
from ..bagpipe_cache.comm import BagPipeCommMixin
from ..bagpipe_cache.grads import BagPipeGradMixin


class _DoneWork:
    def wait(self):
        return None


class _CommHarness(BagPipeCommMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self._global_id_to_index = {10: 0}
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

    def apply_sgd_update_gpu_cache(self, *_args, **_kwargs):
        return False

    def update(self, name, ids, grads):
        self.updates.append((name, ids.clone(), grads.clone()))

    def invalidate_gpu_cache(self, name, ids):
        self.invalidations.append((name, ids.clone()))


class _SparseWork:
    def __init__(self, ids, grads):
        self.result = (ids, grads)

    def wait(self):
        return None


class _GradHarness(BagPipeGradMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self.kv_client = _FakeClient()
        self.cache_entries = {10: object(), 20: object()}
        self.sync_later_grads = {}
        self._first_update = True
        self._sync_later_future = None
        self._sync_later_stream = None
        self._stats = {
            "bagpipe_update_ms": 0,
            "bagpipe_sgd_cache_fallback": 0,
            "bagpipe_all_reduce_ms": 0,
        }
        self.sparse_gather_calls = 0

    def _is_distributed(self):
        return True

    def _get_rank(self):
        return 0

    def _all_ranks_cache_update_succeeded(self, _local_success):
        return False

    def _all_gather_sparse_async(self, ids, grads, stream=None):
        self.sparse_gather_calls += 1
        return ids, grads, _SparseWork(ids, grads)

    def _dense_all_reduce_async(self, *_args, **_kwargs):
        raise AssertionError("fallback must not use the filtered dense path")


class TestBagPipeDistributedFallback(unittest.TestCase):
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

    def test_cache_update_status_uses_global_minimum(self):
        harness = _GradHarness()

        def mark_failed(status, op=None):
            self.assertIs(op, grads_module.dist.ReduceOp.MIN)
            status.zero_()
            return _DoneWork()

        with mock.patch.object(grads_module.dist, "all_reduce", side_effect=mark_failed):
            result = BagPipeGradMixin._all_ranks_cache_update_succeeded(harness, True)

        self.assertFalse(result)

    def test_failed_cache_update_uses_sparse_fallback_and_invalidates_rank_zero(self):
        harness = _GradHarness()
        ids = torch.tensor([10, 20], dtype=torch.int64)
        grads = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        harness.update_grads("table", ids, grads, lr=0.1, batch_num=0)

        self.assertEqual(harness.sparse_gather_calls, 1)
        self.assertEqual(len(harness.kv_client.updates), 1)
        pushed_ids = harness.kv_client.updates[0][1]
        pushed_grads = harness.kv_client.updates[0][2]
        self.assertEqual(pushed_ids.size(0), pushed_grads.size(0))
        self.assertEqual(len(harness.kv_client.invalidations), 1)


if __name__ == "__main__":
    unittest.main()
