import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

from ..ps_pipeline import PSPipeline, PSPipelineSGD


class _Features:
    def __init__(self, ids, keys=None):
        self.ids = torch.tensor(ids, dtype=torch.int64)
        self._keys = keys or ["f0"]

    def keys(self):
        return self._keys

    def values(self):
        return self.ids

    def lengths(self):
        return torch.ones_like(self.ids)


class _PS:
    def __init__(self):
        self.rows = torch.ones((16, 2))
        self.release = threading.Event()
        self.waiting = threading.Event()
        self.release.set()
        self.pending = {}
        self.reads = []
        self.commits = []
        self.closed = []
        self.fail = False
        self.fail_close = False

    def pull(self, name, ids):
        self.reads.append(ids.tolist())
        return self.rows[ids].clone()

    def sparse_gradient_commit_async(self, name, ids, grads, **kwargs):
        self.commits.append(kwargs)
        self.pending[kwargs["step"]] = ids.clone(), grads.clone(), kwargs
        return kwargs["step"]

    def sparse_gradient_commit_wait(self, step):
        self.waiting.set()
        if not self.release.wait(5):
            raise TimeoutError("fake commit was not released")
        if self.fail:
            raise OSError("failed PS commit")
        ids, grads, args = self.pending.pop(step)
        self.rows.index_add_(0, ids, grads, alpha=-args["lr"] / args["divisor"])

    def sparse_gradient_commit_close(self, *args, **kwargs):
        self.closed.append(kwargs)
        self.release.set()
        if self.fail_close:
            raise RuntimeError("abort acknowledged as failed session")


def _pipeline(ps, **kwargs):
    return PSPipeline(None, ps, table_offsets={"f0": 0}, master_table_name="table",
                      embedding_dim=2, device="cpu", lr=0.1, **kwargs)


def _consume(pipeline, step, ids):
    features = _Features(ids)
    row = {}
    ticket = pipeline.on_batch_prepared(step, features, None, row)
    pipeline.before_lookup(step, features, ticket, row)
    return ticket, pipeline.lookup(features).values(), row


class TestPSPipeline(unittest.TestCase):
    def test_native_acceptance_survives_wait_scheduling_failure(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        ticket, _, _ = _consume(pipeline, 0, [3])
        original = RuntimeError("wait scheduling failed")
        try:
            with mock.patch.object(pipeline._worker, "submit", side_effect=original):
                with self.assertRaises(RuntimeError) as caught:
                    pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
            self.assertIs(caught.exception, original)
            self.assertEqual(pipeline._submitted, 0)
            self.assertEqual(pipeline._accepted_step, 0)
            self.assertEqual(len(ps.pending), 1)
            pipeline.shutdown(flush=False)
            self.assertFalse(ps.pending)
            self.assertFalse(pipeline._pending_commits)
            self.assertEqual(ps.closed[-1]["step"], 0)
            self.assertTrue(ps.closed[-1]["abort"])
            pipeline.shutdown(flush=False)
            self.assertEqual(len(ps.closed), 1)
        finally:
            pipeline.shutdown(flush=False)

    def test_flush_after_scheduling_failure_aborts_and_preserves_error(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        ticket, _, _ = _consume(pipeline, 0, [3])
        original = RuntimeError("wait scheduling failed")
        try:
            with mock.patch.object(pipeline._worker, "submit", side_effect=original):
                with self.assertRaises(RuntimeError):
                    pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
            with self.assertRaises(RuntimeError) as caught:
                pipeline.shutdown()
            self.assertIs(caught.exception, original)
            self.assertTrue(ps.closed[-1]["abort"])
            self.assertFalse(ps.pending)
        finally:
            pipeline.shutdown(flush=False)

    def test_cancelled_wait_still_consumes_native_handle(self):
        from concurrent.futures import Future
        ps = _PS()
        pipeline = _pipeline(ps)
        ticket, _, _ = _consume(pipeline, 0, [3])
        cancelled = Future()
        cancelled.cancel()
        try:
            with mock.patch.object(pipeline._worker, "submit", return_value=cancelled):
                pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
            pipeline.shutdown(flush=False)
            self.assertFalse(ps.pending)
            self.assertFalse(pipeline._pending_commits)
        finally:
            pipeline.shutdown(flush=False)

    def test_next_read_failure_does_not_wait_native_handle_twice(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        ticket, _, _ = _consume(pipeline, 0, [3])
        try:
            with mock.patch.object(ps, "sparse_gradient_commit_wait",
                                   wraps=ps.sparse_gradient_commit_wait) as wait:
                with mock.patch.object(pipeline, "_queue_next_read", side_effect=RuntimeError("read enqueue")):
                    with self.assertRaisesRegex(RuntimeError, "read enqueue"):
                        pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
                pipeline.shutdown(flush=False)
                wait.assert_called_once_with(0)
            self.assertFalse(ps.pending)
        finally:
            pipeline.shutdown(flush=False)

    def test_owned_destination_read_waits_for_commit(self):
        class IntoPS(_PS):
            def pull(self, *args):
                raise AssertionError("owned read must not allocate a second result")

            def pull_into(self, name, ids, values):
                self.reads.append(ids.tolist())
                values.copy_(self.rows[ids])
                self.last_destination = values
                return values

        ps = IntoPS()
        pipeline = _pipeline(ps)
        try:
            ticket, values, _ = _consume(pipeline, 0, [2, 2, 5])
            self.assertIs(pipeline._current_rows, ps.last_destination)
            torch.testing.assert_close(values, torch.ones((3, 2)))
            ps.release.clear()
            pipeline.submit_sparse_grad(ticket[0], torch.tensor([[2., 2.], [3., 3.]]))
            self.assertTrue(ps.waiting.wait(2))
            features = _Features([2, 5])
            next_ticket = pipeline.on_batch_prepared(1, features, None, {})
            self.assertEqual(ps.reads, [[2, 5]])
            ps.release.set()
            pipeline.before_lookup(1, features, next_ticket, {})
            torch.testing.assert_close(pipeline.lookup(features).values(),
                                       torch.tensor([[.8, .8], [.7, .7]]))
            pipeline.submit_sparse_grad(next_ticket[0], torch.zeros((2, 2)))
        finally:
            ps.release.set()
            pipeline.shutdown()

    def test_next_read_waits_for_global_commit_and_expands_duplicates(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        pipeline._world = 2
        try:
            ticket, values, row = _consume(pipeline, 0, [3, 3, 7])
            torch.testing.assert_close(values, torch.ones((3, 2)))
            self.assertEqual(ticket[0].tolist(), [3, 7])
            self.assertEqual(ticket[1].tolist(), [0, 0, 1])
            self.assertEqual(row["ps_pipeline_cache_bytes"], 0)
            ps.release.clear()
            pipeline.submit_sparse_grad(ticket[0], torch.tensor([[2., 2.], [1., 1.]]))
            self.assertTrue(ps.waiting.wait(2))
            features = _Features([3, 7])
            next_ticket = pipeline.on_batch_prepared(1, features, None, {})
            self.assertEqual(ps.reads, [[3, 7]])
            self.assertFalse(pipeline._read.done())
            ps.release.set()
            pipeline.before_lookup(1, features, next_ticket, {})
            torch.testing.assert_close(pipeline.lookup(features).values(),
                                       torch.tensor([[.9, .9], [.95, .95]]))
            pipeline.submit_sparse_grad(next_ticket[0], torch.ones((2, 2)))
            pipeline.shutdown()
            torch.testing.assert_close(ps.rows[3], torch.tensor([.85, .85]))
            self.assertTrue(all(commit["divisor"] == 2 for commit in ps.commits))
            self.assertEqual(ps.closed[-1]["step"], 1)
            self.assertFalse(ps.closed[-1]["abort"])
        finally:
            ps.release.set()
            pipeline.shutdown(flush=False)

    def test_gradient_ready_hook_defers_next_read_and_submits_only_once(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        pipeline._world = 2
        module = SimpleNamespace(reset_trace=lambda: None)
        optimizer = PSPipelineSGD(module, pipeline)
        try:
            _, values, _ = _consume(pipeline, 0, [3, 3, 7])
            features = _Features([3, 7])
            ticket = pipeline.on_batch_prepared(1, features, None, {})
            self.assertIsNone(pipeline._read)
            self.assertEqual(ps.reads, [[3, 7]])
            ps.release.clear()
            leaf = values.reshape(3, 1, 2).detach().requires_grad_()
            leaf.register_hook(optimizer.submit_pooled_grad)
            (leaf * torch.tensor([1., 2., 3.]).reshape(3, 1, 1)).sum().backward()
            self.assertTrue(ps.waiting.wait(2))
            self.assertFalse(pipeline._read.done())
            optimizer.step()
            self.assertEqual(len(ps.commits), 1)
            self.assertEqual(ps.reads, [[3, 7]])
            self.assertEqual(pipeline._current_ticket[0].device.type, "cpu")
            ps.release.set()
            pipeline.before_lookup(1, features, ticket, {})
            torch.testing.assert_close(pipeline.lookup(features).values(), torch.full((2, 2), .85))
            optimizer.record_pooled_grad(torch.zeros((2, 1, 2)))
            optimizer.step()
        finally:
            ps.release.set()
            pipeline.shutdown()

    def test_cpu_plan_inverse_matches_sample_major_gradient(self):
        ps = _PS()
        ps.rows = torch.arange(32, dtype=torch.float32).reshape(16, 2)
        initial = ps.rows.clone()
        pipeline = PSPipeline(None, ps, table_offsets={"f0": 0, "f1": 8},
                              master_table_name="table", embedding_dim=2, device="cpu", lr=.1)
        optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
        try:
            features = _Features([3, 7, 3, 3], keys=["f0", "f1"])
            ticket = pipeline.on_batch_prepared(0, features,
                                                torch.tensor([[3, 3], [7, 3]]), {})
            pipeline.before_lookup(0, features, ticket, {})
            self.assertEqual(pipeline._current_inverse.tolist(), [0, 2, 1, 2])
            torch.testing.assert_close(pipeline.lookup(features).values(),
                                       initial[[3, 11, 7, 11]].reshape(2, 4))
            optimizer.record_pooled_grad(torch.tensor([[[1., 1.], [2., 2.]],
                                                       [[3., 3.], [4., 4.]]]))
            optimizer.step()
            pipeline._wait_pending()
            torch.testing.assert_close(ps.rows[[3, 7, 11]], initial[[3, 7, 11]] -
                                       torch.tensor([[.1, .1], [.3, .3], [.6, .6]]))
        finally:
            pipeline.shutdown()

    def test_gradient_reduction_restores_determinism_mode_even_on_failure(self):
        initial_mode = torch.get_deterministic_debug_mode()
        native_index_add = torch.Tensor.index_add_
        try:
            for mode in (0, 1, 2):
                for fail in (False, True):
                    with self.subTest(mode=mode, fail=fail):
                        torch.set_deterministic_debug_mode(mode)
                        pipeline = _pipeline(_PS())
                        optimizer = PSPipelineSGD(SimpleNamespace(reset_trace=lambda: None), pipeline)
                        try:
                            _consume(pipeline, 0, [3, 3, 7])
                            def checked_index_add(tensor, *args, **kwargs):
                                self.assertEqual(torch.get_deterministic_debug_mode(), 2)
                                if fail:
                                    raise RuntimeError("reduction failed")
                                return native_index_add(tensor, *args, **kwargs)
                            with mock.patch.object(torch.Tensor, "index_add_", checked_index_add):
                                if fail:
                                    with self.assertRaisesRegex(RuntimeError, "reduction failed"):
                                        optimizer.record_pooled_grad(torch.ones((3, 1, 2)))
                                else:
                                    optimizer.record_pooled_grad(torch.ones((3, 1, 2)))
                                    torch.testing.assert_close(optimizer._grads, torch.tensor([[2., 2.], [1., 1.]]))
                            self.assertEqual(torch.get_deterministic_debug_mode(), mode)
                        finally:
                            pipeline.shutdown(flush=False)
        finally:
            torch.set_deterministic_debug_mode(initial_mode)

    def test_empty_rank_submits_marker_and_final_step_drains(self):
        ps = _PS()
        pipeline = _pipeline(ps)
        module = SimpleNamespace(_trace=[], reset_trace=lambda: module._trace.clear())
        optimizer = PSPipelineSGD(module, pipeline)
        try:
            _, values, _ = _consume(pipeline, 0, [])
            self.assertEqual(values.shape, (0, 2))
            optimizer.step()
            optimizer.flush()
            pipeline.shutdown()
            self.assertEqual(len(ps.commits), 1)
            self.assertEqual(ps.reads, [])
            self.assertFalse(ps.pending)
        finally:
            pipeline.shutdown(flush=False)

    def test_future_failure_reaches_consumer_and_shutdown(self):
        ps = _PS()
        ps.fail = True
        ps.fail_close = True
        pipeline = _pipeline(ps)
        ticket, _, _ = _consume(pipeline, 0, [3])
        pipeline.submit_sparse_grad(ticket[0], torch.ones((1, 2)))
        features = _Features([3])
        ticket = pipeline.on_batch_prepared(1, features, None, {})
        with self.assertRaisesRegex(OSError, "failed PS commit"):
            pipeline.before_lookup(1, features, ticket, {})
        with self.assertRaisesRegex(OSError, "failed PS commit"):
            pipeline.shutdown()
        self.assertTrue(pipeline._closed)
        self.assertTrue(ps.closed[-1]["abort"])
        self.assertEqual(ps.reads, [[3]])

    def test_validation_exports_committed_touched_rows_without_plans(self):
        from model_zoo.rs_demo.runtime.validation import export_validation_state
        ps = _PS()
        pipeline = _pipeline(ps, validation_state=True)
        embedding = SimpleNamespace(_fusion_k=20, _embedding_bag_configs=[
            SimpleNamespace(name="f0", num_embeddings=16, embedding_dim=2)])
        try:
            for step, ids in enumerate(([7, 3], [7, 11])):
                ticket, _, _ = _consume(pipeline, step, ids)
                pipeline.submit_sparse_grad(ticket[0], torch.ones((2, 2)))
            with tempfile.TemporaryDirectory() as directory:
                path = export_validation_state(Path(directory) / "state.pt", embedding,
                                               torch.nn.Linear(2, 1), pipeline)
                state = torch.load(path, weights_only=True)
            self.assertEqual(state["embedding_kind"], "planned")
            self.assertEqual(state["global_ids"].tolist(), [3, 7, 11])
            torch.testing.assert_close(state["values"], ps.rows[[3, 7, 11]])
        finally:
            pipeline.shutdown()


if __name__ == "__main__":
    unittest.main()
