"""CPU radix-ticket regression gates; native cases require the intended rebuilt op."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from .. import canonical_ps as planner

FLAG = 'RECSTORE_PS_CANONICAL_CPP_TICKET'
FIELDS = ('keys', 'counts', 'positions', 'starts', 'table_types')


class TestCanonicalSparseTicketSelection(unittest.TestCase):
    def test_default_and_other_values_keep_numpy(self):
        with mock.patch.object(torch.ops.recstore_ops, 'canonical_sparse_ticket',
                               side_effect=AssertionError('native called'), create=True):
            for value in (None, '0', 'true'):
                with self.subTest(value=value), mock.patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(FLAG, None)
                    else:
                        os.environ[FLAG] = value
                    ticket, inverse = planner.make_ticket_with_inverse(
                        np.array([[2, 1, 2], [1, 0, 0]], np.int64), [0, 100], [0, 1])
                    self.assertEqual(ticket.keys.tolist(), [1, 2, 100, 101])
                    self.assertEqual(inverse.tolist(), [1, 0, 1, 3, 2, 2])

    def test_opt_in_fails_in_actual_unloaded_process(self):
        env = os.environ.copy()
        env.update(RECSTORE_DEFER_OPS_LOAD='1', RECSTORE_PS_CANONICAL_CPP_TICKET='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        root = str(Path(planner.__file__).resolve().parents[1])
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, '-c', '''
import numpy as np
import torch
from recstore import canonical_ps
assert not callable(getattr(torch.ops.recstore_ops, 'canonical_sparse_ticket', None))
try:
    canonical_ps.make_ticket_with_inverse(np.array([[0]],np.int64),[0],[0])
except RuntimeError as exc:
    assert 'requires the canonical_sparse_ticket native op' in str(exc)
else:
    raise AssertionError('Missing native op silently fell back')
assert not torch.cuda.is_initialized()
'''], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(callable(getattr(torch.ops.recstore_ops, 'canonical_sparse_ticket', None)),
                     'requires the rebuilt canonical_sparse_ticket native op')
class TestCanonicalSparseTicketNative(unittest.TestCase):
    def compare(self, raw, offsets, types):
        # make_ticket retains the frozen v16 NumPy implementation. The native
        # path uses stable byte scatter, independent of this argsort reference.
        expected = planner.make_ticket(raw, offsets, types)
        inverse = np.empty(raw.size, np.int64)
        if raw.size:
            feature_major = ((expected.positions % expected.features) * expected.batch_size
                             + expected.positions // expected.features)
            inverse[feature_major] = np.repeat(np.arange(len(expected.keys), dtype=np.int64),
                                               expected.counts)
        before = raw.tobytes()
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            actual, actual_inverse = planner.make_ticket_with_inverse(raw, offsets, types)
        for name in FIELDS:
            a, b = getattr(actual, name), getattr(expected, name)
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), name)
            self.assertEqual(a.tobytes(), b.tobytes(), name)
        self.assertEqual((actual_inverse.dtype, actual_inverse.shape), (inverse.dtype, inverse.shape))
        self.assertEqual(actual_inverse.tobytes(), inverse.tobytes())
        self.assertEqual(raw.tobytes(), before)
        return actual, actual_inverse

    def test_frozen_stable_order_fixture_and_ownership(self):
        raw = np.array([[2, 1, 2], [1, 0, 0]], np.int64)
        ticket, inverse = self.compare(raw, [0, 100], [0, 1])
        # Literal fixture from the frozen v16 reference, including tied IDs.
        golden = ([1, 2, 100, 101], [1, 2, 2, 1], [2, 0, 4, 3, 5, 1],
                  [0, 1, 3, 5], [0, 0, 1, 1])
        for name, values in zip(FIELDS, golden):
            self.assertEqual(getattr(ticket, name).tobytes(), np.array(values, np.int64).tobytes())
        saved = [getattr(ticket, name).copy() for name in FIELDS] + [inverse.copy()]
        raw.fill(9)
        self.compare(np.zeros((26, 1024), np.int64), np.arange(26) << 30, np.arange(26) % 2)
        for actual, expected in zip([getattr(ticket, n) for n in FIELDS] + [inverse], saved):
            self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_empty_tail_full_duplicates_and_strides(self):
        rng = np.random.default_rng(11020537)
        for features in (1, 2, 26):
            for batch in (0, 1, 2, 31, 32, 33, 128, 1023, 1024):
                for cardinality in (1, 3, 1024):
                    raw = rng.integers(0, cardinality, size=(features, batch), dtype=np.int64)
                    offsets, types = np.arange(features) << 30, np.arange(features) % 2
                    for values in (raw, raw[:, ::-1]):
                        with self.subTest(features=features, batch=batch, cardinality=cardinality):
                            self.compare(values, offsets, types)

    def test_negative_strides_in_singleton_and_empty_numpy_arrays(self):
        offsets = np.array([0], np.int64)[::-1]
        types = np.array([1], np.int64)[::-1]
        for batch in (0, 1):
            raw = np.array([[7]], np.int64)[:, :batch][:, ::-1]
            self.assertTrue(any(stride < 0 for stride in offsets.strides))
            self.assertTrue(any(stride < 0 for stride in types.strides))
            self.compare(raw, offsets, types)

    def test_all_byte_boundaries_and_last_namespace(self):
        maximum = np.iinfo(np.int64).max
        for passes in range(1, 9):
            top = (1 << min(8 * passes, 63)) - 1
            low = 1 << (8 * (passes - 1))
            for kind in (0, 1):
                self.compare(np.array([[top, low, 0, top, low]], np.int64), [0], [kind])
        self.compare(np.array([[0, 1], [0, 2]], np.int64), [0, maximum - 2], [0, 1])
        self.compare(np.empty((2, 0), np.int64), [0, maximum], [0, 1])

    def test_native_rejects_invalid_inputs(self):
        op = torch.ops.recstore_ops.canonical_sparse_ticket
        maximum = np.iinfo(np.int64).max
        base = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 100]), torch.tensor([0, 1])]
        cases = [{0: base[0].float()}, {0: base[0].reshape(-1)},
                 {0: torch.empty((2, 2), dtype=torch.int64, device='meta')},
                 {0: base[0].t()}, {1: base[1].reshape(1, 2)},
                 {2: base[2].float()}, {1: torch.tensor([0])},
                 {2: torch.tensor([0, 2])}, {1: torch.tensor([-1, 100])},
                 {1: torch.tensor([0, 0])}, {1: torch.tensor([100, 0])},
                 {1: torch.tensor([0, 1])}, {0: torch.tensor([[-1, 0], [0, 1]])},
                 {1: torch.tensor([0, maximum])},
                 {0: torch.empty((2, 1025), dtype=torch.int64)},
                 {0: torch.empty((0, 1), dtype=torch.int64)},
                 {0: torch.empty((27, 0), dtype=torch.int64)},
                 {1: torch.tensor([0, 5, 100, 6])[::2]}]
        for changes in cases:
            args = base.copy()
            for index, value in changes.items():
                args[index] = value
            with self.subTest(changes=changes), self.assertRaises((RuntimeError, ValueError)):
                op(*args)
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            for raw, offsets, types in ((np.zeros((1, 1), np.float32), [0], [0]),
                                        (np.zeros((1, 1), np.int64), [], [0])):
                with self.assertRaises(ValueError):
                    planner.make_ticket_with_inverse(raw, offsets, types)


if __name__ == '__main__':
    unittest.main()
