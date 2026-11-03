"""CPU scheduling tests for the merged-gradient owner pipeline (fake transport).

The real owner_merged_plan native op performs routing; the exchange transport
and GPU arithmetic are explicit fakes. These tests cover the merged packing
branch (deterministic index_add_ rows), submission without partial records,
read/version ordering, and count-exchange metadata validation. No claim about
NCCL or kernel arithmetic.
"""
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import torch

from .. import owner_merged_gpu
from ..ps_owner_pipeline import GPUOwnerPipeline, owner_options_from_env
from . import _owner_test_fixtures as fixtures

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


def make_merged_owner(ps, *, table_types=fixtures.types, **options):
    # Run both real constructors. Only the CUDA allocation/JIT boundary is
    # replaced for CPU scheduling tests; no private state is reconstructed.
    ps.ops = SimpleNamespace(owner_merged_plan=PLAN_OP or mock.Mock(),
                             owner_merged_forward_plan=getattr(torch.ops.recstore_ops, 'owner_merged_forward_plan', None) or mock.Mock(),
                             sparse_owner_replace_async=ps.sparse_owner_replace_async,
                             sparse_owner_replace_close=ps.sparse_owner_replace_close)
    ps.get_embedding_table_namespace = lambda table: (0, 0)
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop('device', None)
        return tensor(*args, **kwargs)

    with mock.patch.object(torch, 'tensor', side_effect=cpu_tensor), mock.patch.object(
            owner_merged_gpu, 'initialize_owner_merged_math'), mock.patch.object(
            fixtures.dist, 'get_backend', return_value='nccl'):
        pipeline = GPUOwnerPipeline(
            None, ps, table_offsets={'f0': 0, 'f1': 100}, master_table_name='master',
            embedding_dim=128, device='cuda', dependency_split=True, total_steps=3,
            owner_table_end=200, owner_table_types=table_types, **options)
    pipeline.device = torch.device('cpu')  # Fake exchanges below operate on CPU.
    return pipeline


class TestOwnerConstruction(unittest.TestCase):
    def test_real_constructor_preserves_owner_mode_and_full_options(self):
        with fixtures.context(fixtures.Collective()):
            pipeline = make_merged_owner(fixtures.OwnerPS(), table_types=None, unified_forward=True,
                candidate_handoff=True, compute_ahead=True, metadata_lookahead=4)
            try:
                self.assertTrue(pipeline.owner_updates)
                self.assertTrue(pipeline.owner_merged_grad)
                self.assertTrue(pipeline.compute_ahead)
                self.assertEqual(pipeline.metadata_lookahead, 4)
                self.assertEqual(pipeline.table_types, (0, 0))
                self.assertIsInstance(pipeline._layout_digest, int)
                self.assertEqual(pipeline._pending_commits, {})
            finally:
                with mock.patch.object(torch.cuda, 'synchronize'):
                    pipeline.shutdown(flush=False)

    def test_invalid_combinations_fail_before_gpu_initialization(self):
        cases = [dict(compute_ahead=True), dict(candidate_handoff=True),
                 dict(verify_handoff=True), dict(metadata_lookahead=4),
                 dict(unified_forward=True, metadata_lookahead=3)]
        for options in cases:
            with self.subTest(options=options), mock.patch.object(
                    owner_merged_gpu, 'initialize_owner_merged_math') as initialize:
                with self.assertRaises(ValueError):
                    GPUOwnerPipeline(None, None, owner_table_end=200, **options)
                initialize.assert_not_called()

    def test_owner_mode_does_not_require_redundant_merged_flag(self):
        self.assertIsNone(owner_options_from_env({}))
        options = owner_options_from_env({'RECSTORE_PS_OWNER_GPU': '1'})
        self.assertNotIn('owner_merged_grad', options)
        self.assertEqual(options, owner_options_from_env({
            'RECSTORE_PS_OWNER_GPU': '1', 'RECSTORE_PS_OWNER_MERGED_GRAD': '1'}))
        for env in ({'RECSTORE_PS_OWNER_FORWARD': '1'},
                    {'RECSTORE_PS_OWNER_GPU': '1', 'RECSTORE_PS_OWNER_MERGED_GRAD': '0'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                owner_options_from_env(env)


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
            ps = fixtures.OwnerPS()
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
                    expected = expected_merged_rows(pipeline, pooled)
                    self.assertTrue(torch.equal(optimizer._grads, expected))
                    optimizer.step()
                    self.assertNotIn(step, pipeline._owner_metadata)
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
            ps = fixtures.OwnerPS()
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

    def test_bad_count_exchange_fails_before_any_native_submission(self):
        for bad in ('layout', 'batch', 'width', 'counts', 'keys'):
            with self.subTest(bad=bad), fixtures.context(fixtures.Collective(bad)):
                ps = fixtures.OwnerPS()
                pipeline = make_merged_owner(ps)
                try:
                    with self.assertRaises((RuntimeError, ValueError)):
                        pipeline._exchange_key_counts(0, fixtures.ticket(fixtures.local_batches[0]))
                    self.assertFalse(ps.commits)
                finally:
                    pipeline.shutdown(flush=False)


class FakeForwardExchange(FakeMergedExchange):
    """Fake peer transport with real CUDA buffers/events for lifecycle tests."""
    def __init__(self, permutation, keys, meta, send, recv, weights, **kwargs):
        weights = weights.to(kwargs['device'])
        super().__init__(permutation, keys, meta, send, recv, weights, **kwargs)
        self._gpu = (meta, permutation, weights)
        self.row_count = kwargs['forward_reorder'].numel()

    def forward_rows(self):
        return torch.zeros((self.row_count, 128), device=self.weights.device)

    def run(self, packed, lr):
        self.packed = packed.clone()
        return self.weights + 1, torch.zeros(1, dtype=torch.int32, device=self.weights.device)


@unittest.skipUnless(callable(PLAN_OP) and torch.cuda.is_available(),
                     'requires native owner plan and CUDA')
class TestOwnerOptimizedLifecycle(unittest.TestCase):
    def test_full_options_finish_and_failed_wait_scheduling(self):
        for failed_step in (None, 0, 1):
            with self.subTest(failed_step=failed_step), fixtures.context(fixtures.Collective()), mock.patch.object(
                    owner_merged_gpu, 'ForwardMergedGPUOwnerExchange', FakeForwardExchange):
                ps = fixtures.OwnerPS()
                pipeline = make_merged_owner(ps, unified_forward=True,
                    candidate_handoff=True, compute_ahead=True, metadata_lookahead=4)
                pipeline.device = torch.device('cuda', torch.cuda.current_device())
                optimizer = fixtures.pipeline_module.PSPipelineSGD(
                    SimpleNamespace(reset_trace=lambda: None), pipeline)
                original_submit = pipeline._worker.submit
                original_error = RuntimeError('owner wait scheduling failed')

                def submit(fn, *args):
                    if failed_step == pipeline._accepted_step and fn.__name__ == '_wait_commit':
                        raise original_error
                    return original_submit(fn, *args)

                try:
                    handles = [pipeline.queue_cpu_metadata(i, batch)
                               for i, batch in enumerate(fixtures.local_batches)]
                    rows = [{}, {}, {}]
                    tickets = {0: pipeline.adopt_cpu_metadata(0, handles[0], rows[0])}
                    with mock.patch.object(pipeline._worker, 'submit', side_effect=submit), mock.patch.object(
                            ps, 'sparse_gradient_commit_wait', wraps=ps.sparse_gradient_commit_wait) as wait:
                        for step in range(3):
                            if step + 1 < 3:
                                tickets[step + 1] = pipeline.adopt_cpu_metadata(
                                    step + 1, handles[step + 1], rows[step + 1])
                            pipeline.before_lookup(step, None, tickets[step], rows[step])
                            optimizer.record_pooled_grad(pooled_gradient(step).to(pipeline.device))
                            if step == failed_step:
                                with self.assertRaises(RuntimeError) as caught:
                                    optimizer.step()
                                self.assertIs(caught.exception, original_error)
                                # Cleanup retains the original failure while consuming the native handle.
                                with self.assertRaises(RuntimeError) as caught:
                                    pipeline.shutdown(flush=False)
                                self.assertIs(caught.exception, original_error)
                                self.assertTrue(ps.closed[-1]['abort'])
                                self.assertEqual(wait.call_count, step + 1)
                                self.assertEqual(ps.closed[-1]['step'], step)
                                break
                            optimizer.step()
                            optimizer.zero_grad()
                        if failed_step is None:
                            pipeline.finish_training()
                            self.assertEqual(pipeline._retired_step, 2)
                            self.assertEqual(pipeline.prepared_steps, 3)
                            self.assertTrue(pipeline._metadata_stopping)
                            pipeline.shutdown()
                            self.assertFalse(ps.closed[-1]['abort'])
                            self.assertEqual(wait.call_count, 3)
                            self.assertGreater(rows[1]['ps_owner_handoff_rows'], 0)
                        self.assertFalse(pipeline._pending_commits)
                        self.assertTrue(pipeline._closed)
                finally:
                    ps.release.set()
                    pipeline.shutdown(flush=False)


class TestBasePipelineTicketModeGuards(unittest.TestCase):
    def test_base_pipeline_rejects_the_ticket_mode(self):
        with self.assertRaises(ValueError) as caught:
            fixtures.pipeline_module.PSPipeline(
                None, fixtures.OwnerPS(), table_offsets={'f0': 0, 'f1': 100},
                master_table_name='master', embedding_dim=128, device='cpu',
                dependency_split=True, total_steps=3,
                owner_table_types=[0, 1])
        self.assertIn('merged GPU owner pipeline', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
