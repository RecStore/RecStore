"""Merged-gradient owner plan op checks against an independent NumPy oracle.

The routing oracle calls the real CityHash64 from the cityhash library, the
same symbol the C++ planner uses on the encoded physical key. Native op tests
require the rebuilt library in a fresh process before importing recstore.
"""
import unittest

import numpy as np
import torch

from .. import canonical_ps as planner
from ..sharded_client import _city_hash64_of_uint64

OP = getattr(torch.ops.recstore_ops, 'owner_merged_plan', None)
FOP = getattr(torch.ops.recstore_ops, 'owner_merged_forward_plan', None)


def oracle(keys0, counts0, keys1, counts1, boundaries, types, tag, rank):
    i0 = i1 = 0
    keys, meta = [], []
    perm = [[], []]
    fsend, fpos = [[], []], [[], []]
    send = [0, 0]
    recv = [0, 0]
    local_position = local_row = 0
    feature = 0
    while i0 < len(keys0) or i1 < len(keys1):
        key = keys0[i0] if i1 == len(keys1) or (i0 < len(keys0) and keys0[i0] < keys1[i1]) \
            else keys1[i1]
        c0 = counts0[i0] if i0 < len(keys0) and keys0[i0] == key else 0
        c1 = counts1[i1] if i1 < len(keys1) and keys1[i1] == key else 0
        if i0 < len(keys0) and keys0[i0] == key:
            i0 += 1
        if i1 < len(keys1) and keys1[i1] == key:
            i1 += 1
        while feature + 1 < len(boundaries) - 1 and key >= boundaries[feature + 1]:
            feature += 1
        owner = _city_hash64_of_uint64(int(key) | (int(tag) << 56)) % 2
        present = [c0 > 0, c1 > 0]
        if present[rank]:
            perm[owner].append(local_position)
            local_position += 1
            fpos[owner].append(local_row)
            local_row += 1
            send[owner] += 1
        if owner == rank:
            offset0 = sum(1 for i, row in enumerate(meta) if row[5]) if meta else 0
            offset1 = sum(1 for i, row in enumerate(meta) if row[7]) if meta else 0
            meta.append([key, c0, c1, int(types[feature]), offset0, int(present[0]),
                         offset1, int(present[1])])
            for destination in (0, 1):
                if present[destination]:
                    fsend[destination].append(len(keys))
            keys.append(key)
            for source in (0, 1):
                if present[source]:
                    recv[source] += 1
    permutation = perm[0] + perm[1]
    fsend_index = fsend[0] + fsend[1]
    reorder = [None] * local_row
    received = 0
    for peer in (0, 1):
        for position in fpos[peer]:
            reorder[position] = received
            received += 1
    return {'permutation': permutation, 'keys': keys, 'meta': meta, 'send': send,
            'recv': recv, 'forward_send_index': fsend_index, 'forward_reorder': reorder,
            'forward_send': [len(fsend[p]) for p in (0, 1)],
            'forward_recv': [len(fpos[p]) for p in (0, 1)],
            'rows': [len(keys0), len(keys1)]}


def run_plan(keys0, counts0, keys1, counts1, boundaries, types, tag, rank, forward):
    args = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64))
            for x in (keys0, counts0, keys1, counts1, boundaries, types)]
    operation = FOP if forward else OP
    outputs = operation(*args, int(tag), int(rank))
    fields = ('permutation', 'keys', 'meta', 'send', 'recv')
    result = dict(zip(fields, [x.tolist() for x in outputs[:5]]))
    if forward:
        result.update(forward_send_index=outputs[5].tolist(),
                      forward_send=outputs[6].tolist(),
                      forward_recv=outputs[7].tolist(),
                      forward_reorder=outputs[8].tolist())
    return result


@unittest.skipUnless(callable(OP) and callable(FOP),
                     'requires the rebuilt owner_merged_plan native op')
class TestOwnerMergedPlanNative(unittest.TestCase):
    def compare(self, keys0, counts0, keys1, counts1, boundaries, types, tag):
        for rank in (0, 1):
            expected = oracle(keys0, counts0, keys1, counts1, boundaries, types, tag, rank)
            actual = run_plan(keys0, counts0, keys1, counts1, boundaries, types, tag,
                              rank, True)
            for name, value in expected.items():
                if name == 'rows':
                    self.assertEqual(actual['send'][0] + actual['send'][1], value[rank])
                    continue
                self.assertEqual(actual[name], value, (name, rank))
            for row in actual['meta']:
                self.assertIn(row[5], (0, 1))
                self.assertIn(row[7], (0, 1))
                self.assertEqual(bool(row[1]), bool(row[5]))
                self.assertEqual(bool(row[2]), bool(row[7]))
                self.assertLessEqual(row[4] + row[5], actual['recv'][0])
                self.assertLessEqual(row[6] + row[7], actual['recv'][1])

    def test_random_workloads_match_city_hash_oracle(self):
        rng = np.random.default_rng(20260330)
        for boundaries, types, tag in (([0, 100, 200], [0, 1], 3),
                                       ([5, 17, 18], [1, 0], 0),
                                       ([0, 2 ** 40], [1], 255)):
            for trial in range(60):
                arrays = []
                for _ in range(2):
                    count = int(rng.integers(0, 30))
                    keys = np.unique(rng.integers(boundaries[0], boundaries[-1],
                                                  size=count)).astype(np.int64)
                    # Arbitrary per-key counts: merged semantics must not require
                    # the canonical per-feature occurrence-total equality.
                    counts = rng.integers(1, 1025, size=len(keys)).astype(np.int64)
                    arrays.append((keys, counts))
                self.compare(arrays[0][0], arrays[0][1], arrays[1][0], arrays[1][1],
                             np.array(boundaries, dtype=np.int64),
                             np.array(types, dtype=np.int64), tag)

    def test_fixed_shapes_empty_and_single_side(self):
        boundaries, types, tag = np.array([0, 100, 200]), np.array([0, 1]), 9
        self.compare(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                     np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                     boundaries, types, tag)
        self.compare(np.array([7, 120], dtype=np.int64), np.array([1, 2]),
                     np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                     boundaries, types, tag)
        self.compare(np.array([5], dtype=np.int64), np.array([1024]),
                     np.array([5, 6], dtype=np.int64), np.array([1, 1]),
                     boundaries, types, tag)

    def test_malformed_inputs_fail_closed(self):
        boundaries, types = np.array([0, 100, 200]), np.array([0, 1])
        empty = np.array([], dtype=np.int64)
        cases = [
            (np.array([5, 3]), np.array([1, 1])),      # unsorted
            (np.array([5, 5]), np.array([1, 1])),      # duplicate
            (np.array([5]), np.array([0])),            # zero count
            (np.array([5]), np.array([1025])),         # above batch bound
            (np.array([200]), np.array([1])),          # outside final boundary
            (np.array([-1]), np.array([1])),           # below first boundary
        ]
        for keys, counts in cases:
            with self.subTest(keys=keys.tolist()):
                args = [torch.from_numpy(np.ascontiguousarray(x)) for x in
                        (keys, counts, empty, empty, boundaries, types)]
                with self.assertRaises((RuntimeError, ValueError)):
                    OP(*args, 0, 0)
        with self.assertRaises((RuntimeError, ValueError)):
            OP(torch.from_numpy(np.array([5], dtype=np.int64)),
               torch.from_numpy(np.array([1], dtype=np.int64)),
               torch.from_numpy(empty), torch.from_numpy(empty),
               torch.from_numpy(boundaries), torch.from_numpy(types), 256, 0)
        with self.assertRaises((RuntimeError, ValueError)):
            OP(torch.from_numpy(np.array([5], dtype=np.int64)),
               torch.from_numpy(np.array([1], dtype=np.int64)),
               torch.from_numpy(empty), torch.from_numpy(empty),
               torch.from_numpy(boundaries), torch.from_numpy(types), 0, 2)

    def test_unequal_feature_occurrence_totals_are_accepted(self):
        # Feature 1 has zero occurrences while feature 0 has three: the
        # canonical occurrence-total check must not run for merged rows.
        boundaries, types = np.array([0, 100, 200]), np.array([0, 1])
        actual = run_plan(np.array([5]), np.array([3]), np.array([], dtype=np.int64),
                          np.array([], dtype=np.int64), boundaries, types, 0, 0, False)
        self.assertEqual(actual['meta'], [[5, 3, 0, 0, 0, 1, 0, 0]])
        self.assertEqual(actual['send'], [1, 0])
        self.assertEqual(actual['recv'], [1, 0])


TICKET_OP = getattr(torch.ops.recstore_ops, 'canonical_sparse_ticket', None)
TICKET_FIELDS = ('keys', 'counts', 'positions', 'starts', 'table_types')


@unittest.skipUnless(callable(TICKET_OP), 'requires the rebuilt canonical_sparse_ticket native op')
class TestNativeTicket(unittest.TestCase):
    """The ticket op kept after the canonical removal: native vs NumPy ticket."""

    def compare(self, raw, offsets, types):
        expected = planner.make_ticket(raw, offsets, types)
        inverse = np.empty(raw.size, np.int64)
        if raw.size:
            feature_major = ((expected.positions % expected.features) * expected.batch_size
                             + expected.positions // expected.features)
            inverse[feature_major] = np.repeat(np.arange(len(expected.keys), dtype=np.int64),
                                               expected.counts)
        before = raw.tobytes()
        actual, actual_inverse = planner.make_ticket_with_inverse(raw, offsets, types, native=True)
        for name in TICKET_FIELDS:
            a, b = getattr(actual, name), getattr(expected, name)
            self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), name)
            self.assertEqual(a.tobytes(), b.tobytes(), name)
        self.assertEqual((actual_inverse.dtype, actual_inverse.shape),
                         (inverse.dtype, inverse.shape))
        self.assertEqual(actual_inverse.tobytes(), inverse.tobytes())
        self.assertEqual(raw.tobytes(), before)

    def test_frozen_stable_order_fixture(self):
        # Literal fixture from the frozen v16 reference, including tied IDs.
        raw = np.array([[2, 1, 2], [1, 0, 0]], np.int64)
        self.compare(raw, [0, 100], [0, 1])
        ticket = planner.make_ticket_with_inverse(raw, [0, 100], [0, 1], native=True)[0]
        golden = ([1, 2, 100, 101], [1, 2, 2, 1], [2, 0, 4, 3, 5, 1],
                  [0, 1, 3, 5], [0, 0, 1, 1])
        for name, values in zip(TICKET_FIELDS, golden):
            self.assertEqual(getattr(ticket, name).tobytes(),
                             np.array(values, np.int64).tobytes())

    def test_random_shapes_duplicates_and_strides(self):
        rng = np.random.default_rng(11020537)
        for features in (1, 2, 26):
            for batch in (0, 1, 2, 31, 32, 33, 128, 1023, 1024):
                for cardinality in (1, 3, 1024):
                    raw = rng.integers(0, cardinality, size=(features, batch), dtype=np.int64)
                    offsets, types = np.arange(features) << 30, np.arange(features) % 2
                    for values in (raw, raw[:, ::-1]):
                        with self.subTest(features=features, batch=batch,
                                          cardinality=cardinality):
                            self.compare(values, offsets, types)

    def test_native_rejects_invalid_inputs(self):
        maximum = np.iinfo(np.int64).max
        base = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 100]),
                torch.tensor([0, 1])]
        cases = [{0: base[0].float()}, {0: base[0].reshape(-1)},
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
                TICKET_OP(*args)


if __name__ == '__main__':
    unittest.main()
