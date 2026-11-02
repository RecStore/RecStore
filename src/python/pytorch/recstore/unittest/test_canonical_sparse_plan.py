"""CPU-only checks against the actual canonical planner op when it is loaded.

For native validation, load the intended library in a fresh process before
importing recstore. No PS, CUDA context, or synthetic native implementation is used.
"""
import dataclasses
import os
import unittest
from unittest import mock

import numpy as np
import torch

from .. import canonical_ps as planner


FLAG = 'RECSTORE_PS_CANONICAL_CPP_PLAN'


def single(count, kind, key=7):
    array = lambda value: np.asarray(value, np.int64)
    return planner.LocalTicket(array([key] if count else []), array([count] if count else []),
                               np.arange(count, dtype=np.int64), array([0] if count else []),
                               array([kind] if count else []), count, 1)


class TestCanonicalSparsePlanSelection(unittest.TestCase):
    def test_default_and_other_values_keep_python_branch(self):
        with mock.patch.object(planner, '_build_plan_cpp', side_effect=AssertionError('native called')):
            for value in (None, '0', 'true'):
                with self.subTest(value=value), mock.patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(FLAG, None)
                    else:
                        os.environ[FLAG] = value
                    plan = planner.build_plan(single(2, planner.TW), [7], [1], 0)
                    self.assertEqual(plan.records.tolist(), [[7, 2, 1, 1, planner.TW]])

    def test_opt_in_missing_op_fails_without_python_fallback(self):
        with mock.patch.dict(os.environ, {FLAG: '1'}), mock.patch.object(
                torch.ops.recstore_ops, 'canonical_sparse_plan', None, create=True):
            with self.assertRaisesRegex(RuntimeError, 'requires the canonical_sparse_plan native op'):
                planner.build_plan(single(1, planner.TW), [], [], 0)


@unittest.skipUnless(callable(getattr(torch.ops.recstore_ops, 'canonical_sparse_plan', None)),
                     'requires the rebuilt canonical_sparse_plan native op')
class TestCanonicalSparsePlanNative(unittest.TestCase):
    def compare(self, own, peer, rank):
        with mock.patch.dict(os.environ, {FLAG: '0'}):
            expected = planner.build_plan(own, peer.keys, peer.counts, rank)
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            actual = planner.build_plan(own, peer.keys, peer.counts, rank)
        for name in ('records', 'tasks', 'positions'):
            a, b = getattr(actual, name), getattr(expected, name)
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), name)
            self.assertEqual(a.tobytes(), b.tobytes(), name)
        for name in ('batch_size', 'features', 'rank', 'vector_count'):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)
        return actual

    def test_tw_dp_boundaries_empty_and_int64_max(self):
        counts = (0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 127, 511, 512, 513, 991, 992, 993, 1023, 1024)
        for c0 in counts:
            for c1 in counts:
                for kind in (planner.TW, planner.DP):
                    tickets = [single(c0, kind), single(c1, kind)]
                    for rank in (0, 1):
                        with self.subTest(c0=c0, c1=c1, kind=kind, rank=rank):
                            self.compare(tickets[rank], tickets[1-rank], rank)
        self.compare(single(17, planner.TW, np.iinfo(np.int64).max),
                     single(16, planner.TW, np.iinfo(np.int64).max), 1)

    def test_mixed_stable_positions_and_full_task_capacity(self):
        rng = np.random.default_rng(11020508)
        for _ in range(20):
            batch, features = int(rng.integers(1, 1025)), int(rng.integers(1, 27))
            types = rng.integers(0, 2, size=features, dtype=np.int64)
            offsets = np.arange(features, dtype=np.int64) << 30
            tickets = [planner.make_ticket(rng.integers(0, 40, size=(features, batch), dtype=np.int64),
                                           offsets, types) for _ in (0, 1)]
            for rank in (0, 1):
                self.compare(tickets[rank], tickets[1-rank], rank)
        offsets = np.arange(26, dtype=np.int64) << 30
        own = planner.make_ticket(np.tile(np.arange(1024, dtype=np.int64), (26, 1)), offsets, np.arange(26) % 2)
        peer = planner.make_ticket(np.empty((26, 0), np.int64), offsets, np.arange(26) % 2)
        self.assertEqual(len(self.compare(own, peer, 0).tasks), 26 * 1024)

    def test_cpu_tensor_ticket_and_peer(self):
        own = planner.make_ticket(np.array([[5, 2, 5], [1, 1, 0]], np.int64), [0, 100], [0, 1])
        tensor_ticket = dataclasses.replace(own, **{name: torch.from_numpy(getattr(own, name)) for name in
                                          ('keys', 'counts', 'starts', 'positions', 'table_types')})
        with mock.patch.dict(os.environ, {FLAG: '0'}):
            expected = planner.build_plan(own, [5, 101], [2, 1], 0)
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            actual = planner.build_plan(tensor_ticket, torch.tensor([5, 101]), torch.tensor([2, 1]), 0)
        for name in ('records', 'tasks', 'positions'):
            self.assertEqual(getattr(actual, name).tobytes(), getattr(expected, name).tobytes())

    def test_native_rejects_invalid_shapes_layout_counts_and_bounds(self):
        op = torch.ops.recstore_ops.canonical_sparse_plan
        base = [torch.tensor([7]), torch.tensor([1]), torch.tensor([0]), torch.tensor([0]),
                torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64), 1, 0]
        failures = [
            {0: torch.tensor([7.0])}, {0: torch.tensor([[7]])},
            {0: torch.tensor([7], device='meta')}, {1: torch.tensor([], dtype=torch.int64)},
            {0: torch.tensor([-1])}, {1: torch.tensor([0])}, {1: torch.tensor([1025])},
            {2: torch.tensor([1])}, {3: torch.tensor([2])}, {6: -1}, {6: 26625}, {7: 2},
            {4: torch.tensor([7]), 5: torch.tensor([], dtype=torch.int64)},
            {4: torch.tensor([7]), 5: torch.tensor([0])},
            {4: torch.tensor([7]), 5: torch.tensor([1025])},
            {4: torch.tensor([7, 7]), 5: torch.tensor([1, 1])},
            {4: torch.tensor([8, 7]), 5: torch.tensor([1, 1])},
            {0: torch.tensor([3, 1, 7, 2])[::2], 1: torch.tensor([1, 1]),
             2: torch.tensor([0, 1]), 3: torch.tensor([0, 0]), 6: 2},
            {0: torch.tensor([7, 7]), 1: torch.tensor([1, 1]),
             2: torch.tensor([0, 1]), 3: torch.tensor([0, 0]), 6: 2},
            {6: 0},
        ]
        for changes in failures:
            args = base.copy()
            for index, value in changes.items():
                args[index] = value
            with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError)):
                op(*args)

    def test_wrapper_rejects_invalid_source_positions(self):
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            for positions in (np.array([1], np.int64), np.array([-1], np.int64), np.array([0.0])):
                own = dataclasses.replace(single(1, planner.TW), positions=positions)
                with self.assertRaisesRegex(ValueError, 'source gradient positions'):
                    planner.build_plan(own, [], [], 0)


if __name__ == '__main__':
    unittest.main()
