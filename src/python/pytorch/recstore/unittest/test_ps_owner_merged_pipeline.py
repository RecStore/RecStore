"""CPU scheduling tests for the merged-gradient owner pipeline (fake transport).

The real owner_merged_forward_plan native op performs routing; the exchange transport
and GPU arithmetic are explicit fakes. These tests cover the merged packing
branch (deterministic index_add_ rows), submission without partial records,
read/version ordering, and count-exchange metadata validation. No claim about
NCCL or kernel arithmetic.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from .. import owner_forward_gpu, owner_merged_gpu
from ..ps_owner_pipeline import GPUOwnerPipeline, owner_options_from_env
from . import _owner_test_fixtures as fixtures

PLAN_OP = getattr(torch.ops.recstore_ops, 'owner_merged_forward_plan', None)


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


def make_merged_owner(ps, *, table_types=fixtures.types):
    # Run both real constructors. Only the CUDA allocation/JIT boundary is
    # replaced for CPU scheduling tests; no private state is reconstructed.
    ps.ops = SimpleNamespace(owner_merged_forward_plan=PLAN_OP or mock.Mock(),
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
            owner_table_end=200, owner_table_types=table_types)
    pipeline.device = torch.device('cpu')  # Fake exchanges below operate on CPU.
    return pipeline


class TestOwnerConstruction(unittest.TestCase):
    def test_real_constructor_preserves_owner_mode_and_full_options(self):
        with fixtures.context(fixtures.Collective()):
            pipeline = make_merged_owner(fixtures.OwnerPS(), table_types=None)
            try:
                self.assertTrue(pipeline.owner_updates)
                self.assertTrue(pipeline.owner_merged_grad)
                self.assertEqual(pipeline.metadata_lookahead, 4)
                self.assertEqual(pipeline.table_types, (0, 0))
                self.assertIsInstance(pipeline._layout_digest, int)
                self.assertEqual(pipeline._pending_commits, {})
            finally:
                with mock.patch.object(torch.cuda, 'synchronize'):
                    pipeline.shutdown(flush=False)

    def test_owner_mode_has_one_environment_switch(self):
        self.assertIsNone(owner_options_from_env({}))
        self.assertEqual(owner_options_from_env({'RECSTORE_PS_OWNER_GPU': '1'}), {})


def pooled_gradient(step):
    """Distinct value per sample so index_add_ summation is observable."""
    samples = fixtures.local_batches[step].shape[0]
    return torch.arange(samples, dtype=torch.float32).reshape(-1, 1, 1).expand(
        -1, len(fixtures.types), 128).contiguous()


def expected_merged_rows(pipeline, pooled):
    rows = torch.zeros((pipeline._current_ticket[0].numel(), 128), dtype=torch.float32)
    rows.index_add_(0, pipeline._current_inverse, pooled.reshape(-1, 128))
    return rows



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
                    owner_forward_gpu, 'ForwardGPUOwnerExchange', FakeForwardExchange):
                ps = fixtures.OwnerPS()
                pipeline = make_merged_owner(ps)
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
        with self.assertRaises(TypeError):
            fixtures.pipeline_module.PSPipeline(
                None, fixtures.OwnerPS(), table_offsets={'f0': 0, 'f1': 100},
                master_table_name='master', embedding_dim=128, device='cpu',
                dependency_split=True, total_steps=3,
                owner_table_types=[0, 1])


if __name__ == '__main__':
    unittest.main()
