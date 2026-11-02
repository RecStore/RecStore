"""CPU scheduling tests for the merged-gradient owner pipeline (fake transport).

The real owner_merged_plan native op performs routing; the exchange transport
and GPU arithmetic are explicit fakes. These tests cover the merged packing
branch (deterministic index_add_ rows), submission without canonical partial
records, and read/version ordering. No claim about NCCL or kernel arithmetic.
"""
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import torch

from .. import owner_merged_gpu
from ..ps_owner_pipeline import GPUOwnerPipeline
from . import test_ps_pipeline_canonical as fixtures
from .test_ps_owner_pipeline import OwnerPS

PLAN_OP = getattr(torch.ops.recstore_ops, 'owner_merged_plan', None)


class FakeMergedExchange:
    def __init__(self, permutation, keys, meta, send, recv, weights, **kwargs):
        self.keys, self.weights = keys, weights
        self.permutation, self.meta, self.send, self.recv = permutation, meta, send, recv
        self.packed = None
        self.drained = False

    def run(self, packed, lr):
        # Opaque candidates; the merged packing and plan wiring are the test
        # targets here. Arithmetic is covered by test_owner_merged_math.
        self.packed = packed.clone()
        return self.weights.clone(), torch.zeros(1, dtype=torch.int32)

    def drain(self):
        self.drained = True


def merged_routing(pipeline, plan, peers, sizes):
    columns = [peer[:size, column].contiguous()
               for peer, size in zip(peers, sizes) for column in (0, 1)]
    owner = PLAN_OP(*columns, pipeline._owner_boundaries, pipeline._owner_types,
                    pipeline._owner_tag, pipeline._rank)
    plan.owner = owner


def make_merged_owner(ps):
    pipeline = fixtures.make_pipeline(ps, True)
    pipeline.__class__ = GPUOwnerPipeline
    pipeline.unified_forward = False
    pipeline.candidate_handoff = False
    pipeline.verify_handoff = False
    pipeline._verified_owner_weights = None
    pipeline._owner_candidate = None
    pipeline.canonical_gpu_plan = False
    pipeline._canonical_gpu_staging = None
    pipeline._pending_owner_step = None
    pipeline._owner_failed = False
    pipeline._owner_plan = pipeline._owner_read = pipeline._owner_exchange = None
    # Attributes normally set by GPUOwnerPipeline.__init__ (bypassed by the
    # class swap): non-unified, no-handoff, no-lookahead, merged gradients.
    pipeline.owner_merged_grad = True
    pipeline.metadata_lookahead = 0
    pipeline._metadata_enqueued = pipeline._metadata_queue_hwm = 0
    pipeline._metadata_stopping = False
    pipeline._metadata_prepare_error = None
    pipeline.compute_ahead = False
    pipeline._retired_step = pipeline._native_submitted_step = pipeline._candidate_step = -1
    pipeline._commit_step = -1
    pipeline._compute_read_step = None
    pipeline._retiring_owner = pipeline._untracked_owner_commit = pipeline._compute_error = None
    pipeline._owner_boundaries = torch.tensor([0, 100, 200], dtype=torch.int64)
    pipeline._owner_types = torch.tensor([0, 1], dtype=torch.int64)
    pipeline._owner_tag = 0
    routing = lambda plan, peers, sizes: merged_routing(pipeline, plan, peers, sizes)
    pipeline._augment_canonical_plan = routing
    return pipeline


def pooled_gradient(step):
    """Distinct value per sample so index_add_ summation is observable."""
    samples = fixtures.local_batches[step].shape[0]
    return torch.arange(samples, dtype=torch.float32).reshape(-1, 1, 1).expand(
        -1, len(fixtures.types), 128).contiguous()


def expected_merged_rows(pipeline, pooled):
    rows = torch.zeros((pipeline._current_ticket[0].numel(), 128), dtype=torch.float32)
    rows.index_add_(0, pipeline._current_inverse, pooled.reshape(-1, 128))
    return rows


@unittest.skipUnless(callable(PLAN_OP), 'requires the rebuilt owner_merged_plan native op')
class TestGPUOwnerMergedPipelineScheduling(unittest.TestCase):
    def test_merged_packing_plan_and_submission_without_partial_records(self):
        with fixtures.context(fixtures.Collective()), mock.patch.object(
                owner_merged_gpu, 'MergedGPUOwnerExchange', FakeMergedExchange):
            ps = OwnerPS()
            pipeline = make_merged_owner(ps)
            optimizer = fixtures.pipeline_module.PSPipelineSGD(
                SimpleNamespace(reset_trace=lambda: None), pipeline)
            prepared = {}

            def prepare(step):
                row = {}
                ticket = pipeline.on_batch_prepared(
                    step, fixtures.Features(), fixtures.local_batches[step], row)
                prepared[step] = ticket, row

            prepare(0)
            lookup_worker = ThreadPoolExecutor(max_workers=1)
            try:
                for step in range(3):
                    if step + 1 < 3:
                        prepare(step + 1)  # Match the runner: prepare before lookup.
                    ticket, row = prepared[step]
                    if step == 1:
                        # The next owner read must wait for the global ACK.
                        reads_before = list(ps.reads)
                        future = lookup_worker.submit(pipeline.before_lookup, step, None, ticket, row)
                        self.assertFalse(future.done())
                        self.assertEqual(ps.reads, reads_before)
                        ps.release.set()
                        future.result(timeout=5)
                    else:
                        pipeline.before_lookup(step, None, ticket, row)
                    if step == 0:
                        pipeline._owner_read.result(timeout=5)
                        pipeline._split_clean.result(timeout=5)
                        ps.release.clear()
                    own = torch.from_numpy(fixtures.ticket(fixtures.local_batches[step]).keys)
                    peer = torch.from_numpy(fixtures.ticket(fixtures.peer_batches[step]).keys)
                    union = torch.unique(torch.cat([own, peer]))
                    permutation, keys, metadata, send, recv = pipeline._owner_plan[:5]
                    # Real city_hash routing: an ascending subset of the union.
                    self.assertTrue(torch.equal(keys, union[torch.isin(union, keys)]))
                    self.assertTrue(bool(torch.isin(keys, union).all()))
                    self.assertEqual(send[0] + send[1], own.numel())
                    self.assertEqual(sorted(permutation.tolist()), list(range(own.numel())))
                    pooled = pooled_gradient(step)
                    optimizer.record_pooled_grad(pooled)
                    self.assertNotIn(step, pipeline._canonical_metadata)
                    expected = expected_merged_rows(pipeline, pooled)
                    self.assertTrue(torch.equal(optimizer._grads, expected))
                    optimizer.step()
                    self.assertIsNone(pipeline._canonical_records)
                    self.assertTrue(torch.equal(ps.commits[-1][0], keys))
                    self.assertEqual(ps.commits[-1][2]['step'], step)
                    self.assertTrue(torch.equal(pipeline._owner_exchange.packed, expected))
                    if step == 0:
                        self.assertTrue(ps.waiting.wait(5))
                        self.assertFalse(pipeline._read.done())
                pipeline.shutdown()
                self.assertTrue(pipeline._owner_exchange is None)
                self.assertFalse(ps.closed[-1]['abort'])
            finally:
                ps.release.set()
                lookup_worker.shutdown(wait=True)
                pipeline.shutdown(flush=False)

    def test_failed_commit_prevents_next_owner_read_and_preserves_error(self):
        with fixtures.context(fixtures.Collective()), mock.patch.object(
                owner_merged_gpu, 'MergedGPUOwnerExchange', FakeMergedExchange):
            ps = OwnerPS()
            pipeline = make_merged_owner(ps)
            optimizer = fixtures.pipeline_module.PSPipelineSGD(
                SimpleNamespace(reset_trace=lambda: None), pipeline)
            first = pipeline.on_batch_prepared(0, fixtures.Features(), fixtures.local_batches[0], {})
            second = pipeline.on_batch_prepared(1, fixtures.Features(), fixtures.local_batches[1], {})
            pipeline.before_lookup(0, None, first, {})
            pipeline._owner_read.result(timeout=5)
            pipeline._split_clean.result(timeout=5)
            original = RuntimeError('merged owner writeback failed')
            ps.failure = original
            try:
                optimizer.record_pooled_grad(pooled_gradient(0))
                optimizer.step()
                reads = list(ps.reads)
                with self.assertRaises(RuntimeError) as caught:
                    pipeline.before_lookup(1, None, second, {})
                self.assertIs(caught.exception, original)
                self.assertEqual(ps.reads, reads)
                pipeline._owner_exchange.drain = mock.Mock(side_effect=ValueError('cleanup'))
                with self.assertRaises(RuntimeError) as caught:
                    pipeline.shutdown()
                self.assertIs(caught.exception, original)
                self.assertTrue(any('cleanup' in note for note in original.__notes__))
                self.assertTrue(ps.closed[-1]['abort'])
            finally:
                pipeline.shutdown(flush=False)


if __name__ == '__main__':
    unittest.main()
