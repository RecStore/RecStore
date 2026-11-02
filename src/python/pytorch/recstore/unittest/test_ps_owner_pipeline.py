"""CPU scheduling tests with explicit fake owner routing/CUDA/PS transports."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from .. import canonical_ps, owner_gpu
from ..ps_owner_pipeline import GPUOwnerPipeline
from . import test_ps_pipeline_canonical as fixtures


class OwnerPS(fixtures.FakePS):
    def sparse_canonical_gradient_commit_async(self, *args, **kwargs):
        raise AssertionError('Owner path submitted canonical gradients to CPU')

    def sparse_owner_replace_async(self, table, ids, values, status, **kwargs):
        self.commits.append((ids.clone(), values.clone(), kwargs))
        return kwargs['step']

    def sparse_owner_replace_close(self, table, **kwargs):
        self.closed.append(kwargs)


class FakeExchange:
    def __init__(self, permutation, keys, meta, send, recv, weights, **kwargs):
        self.keys, self.weights = keys, weights
        self.drained = False

    def run(self, vectors, lr):
        # Opaque unchanged candidates: arithmetic is tested by native/CUDA gates.
        return self.weights.clone(), torch.zeros(1, dtype=torch.int32)

    def drain(self):
        self.drained = True


def make_owner(ps):
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

    def routing(plan, peers, sizes):
        # This fake gives rank0 the entire union; it tests read/version ordering,
        # not city_hash or canonical vector routing.
        keys = torch.unique(torch.cat([p[:n, 0] for p, n in zip(peers, sizes)]))
        plan.owner = (torch.arange(plan.vector_count), keys,
                      torch.zeros((len(keys), 8), dtype=torch.int64),
                      torch.tensor([plan.vector_count, 0]), torch.tensor([plan.vector_count, 0]))
    pipeline._augment_canonical_plan = routing
    return pipeline


class TestGPUOwnerPipelineScheduling(unittest.TestCase):
    def test_owned_union_reads_after_global_ack_including_peer_only_rows(self):
        with fixtures.context(fixtures.Collective()), mock.patch.object(
                canonical_ps, 'packed_gpu_vectors', fixtures.fake_pack), mock.patch.object(
                owner_gpu, 'GPUOwnerExchange', FakeExchange):
            ps = OwnerPS()
            pipeline = make_owner(ps)
            optimizer = fixtures.pipeline_module.PSPipelineSGD(
                SimpleNamespace(reset_trace=lambda: None), pipeline)
            prepared = {}
            def prepare(step):
                row = {}
                ticket = pipeline.on_batch_prepared(step, fixtures.Features(), fixtures.local_batches[step], row)
                prepared[step] = ticket, row
            prepare(0)
            lookup_worker = ThreadPoolExecutor(max_workers=1)
            try:
                for step in range(3):
                    if step + 1 < 3:
                        prepare(step + 1)  # Match the runner: prepare before lookup.
                    ticket, row = prepared[step]
                    if step == 1:
                        reads_before = list(ps.reads)
                        future = lookup_worker.submit(pipeline.before_lookup, step, None, ticket, row)
                        self.assertFalse(future.done())
                        self.assertEqual(ps.reads, reads_before)
                        ps.release.set()
                        future.result(timeout=5)
                    else:
                        pipeline.before_lookup(step, None, ticket, row)
                    expected = torch.unique(torch.cat([
                        torch.from_numpy(fixtures.ticket(batch).keys)
                        for batch in (fixtures.local_batches[step], fixtures.peer_batches[step])]))
                    self.assertTrue(torch.equal(pipeline._owner_plan[1], expected))
                    if step == 0:
                        pipeline._owner_read.result(timeout=5)
                        pipeline._split_clean.result(timeout=5)
                        ps.release.clear()
                    optimizer.record_pooled_grad(torch.zeros((len(fixtures.local_batches[step]), 2, 128)))
                    optimizer.step()
                    self.assertTrue(torch.equal(ps.commits[-1][0], expected))
                    self.assertEqual(ps.commits[-1][2]['step'], step)
                    if step == 0:
                        self.assertTrue(ps.waiting.wait(5))
                        self.assertFalse(pipeline._read.done())
                exchange = pipeline._owner_exchange
                pipeline.shutdown()
                self.assertTrue(exchange.drained)
                self.assertFalse(ps.closed[-1]['abort'])
            finally:
                ps.release.set()
                lookup_worker.shutdown(wait=True)
                pipeline.shutdown(flush=False)

    def test_failed_commit_prevents_next_owner_read_and_preserves_error(self):
        with fixtures.context(fixtures.Collective()), mock.patch.object(
                canonical_ps, 'packed_gpu_vectors', fixtures.fake_pack), mock.patch.object(
                owner_gpu, 'GPUOwnerExchange', FakeExchange):
            ps = OwnerPS()
            pipeline = make_owner(ps)
            optimizer = fixtures.pipeline_module.PSPipelineSGD(
                SimpleNamespace(reset_trace=lambda: None), pipeline)
            first = pipeline.on_batch_prepared(0, fixtures.Features(), fixtures.local_batches[0], {})
            second = pipeline.on_batch_prepared(1, fixtures.Features(), fixtures.local_batches[1], {})
            pipeline.before_lookup(0, None, first, {})
            pipeline._owner_read.result(timeout=5)
            pipeline._split_clean.result(timeout=5)
            original = RuntimeError('owner writeback failed')
            ps.failure = original
            try:
                optimizer.record_pooled_grad(torch.zeros((2, 2, 128)))
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
