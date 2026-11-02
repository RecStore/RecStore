"""CPU checks for dependency classification and read-after-global-commit order."""
import json
import multiprocessing
import tempfile
import unittest
from contextlib import contextmanager
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist

from ..ps_pipeline import PSPipeline
from .test_ps_pipeline import _Features, _PS, _pipeline


@contextmanager
def _split(ps, total_steps=2, dirty=(1, 2)):
    with mock.patch.multiple(dist, is_initialized=lambda: True,
                             get_world_size=lambda: 2, get_rank=lambda: 0,
                             broadcast=lambda *a, **kw: None,
                             new_group=lambda **kw: object(),
                             destroy_process_group=lambda group: None):
        pipeline = _pipeline(ps, dependency_split=True, total_steps=total_steps)
        pipeline._exchange_dirty_ids = lambda step, ids: (
            torch.tensor(dirty, dtype=torch.int64), {"ps_pipeline_metadata_step": step})
        try:
            yield pipeline
        finally:
            ps.release.set()
            pipeline.shutdown(flush=False)


def _prepare(pipeline, step, ids):
    features = _Features(ids)
    row = {}
    ticket = pipeline.on_batch_prepared(step, features, None, row)
    return features, ticket, row


def _consume(pipeline, step, prepared):
    features, ticket, row = prepared
    pipeline.before_lookup(step, features, ticket, row)
    return pipeline.lookup(features).values(), row


class _CollectivePS(_PS):
    """Fake KV values; a real, separate Gloo collective gates global commit."""
    def sparse_gradient_commit_wait(self, step):
        self.waiting.set()
        if not self.release.wait(10):
            raise TimeoutError("test did not release commit")
        ids, grads, args = self.pending.pop(step)
        update = torch.zeros_like(self.rows)
        update.index_add_(0, ids, grads)
        dist.all_reduce(update)  # Default group, never the metadata group.
        self.rows.add_(update, alpha=-args["lr"] / args["divisor"])


def _gloo_rank(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "init"),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    ps = _CollectivePS()
    pipeline = _pipeline(ps, dependency_split=True, total_steps=3)
    # Duplicates, a remote-only dirty row, ID zero, and an empty final rank.
    batches = ([1, 1], [0, 1, 2, 4], [2]) if rank == 0 else ([2, 2], [0, 1, 2, 5], [])
    prepared = _prepare(pipeline, 0, batches[0])
    rows = []
    try:
        for step in range(3):
            next_prepared = _prepare(pipeline, step + 1, batches[step + 1]) if step < 2 else None
            values, row = _consume(pipeline, step, prepared)
            torch.testing.assert_close(values, ps.rows[torch.tensor(batches[step], dtype=torch.int64)],
                                       rtol=0, atol=0)
            rows.append(row)
            if step == 0:
                ps.release.clear()
            ids = prepared[1][0]
            pipeline.submit_sparse_grad(ids, torch.full((ids.numel(), 2), float(rank + 1)))
            if step == 0:
                assert ps.waiting.wait(10), "clean pull blocked on current global commit"
                assert ps.reads == [[rank + 1], [0, rank + 4]], ps.reads
                assert not pipeline._read.done(), "dirty pull must wait for global commit"
                ps.release.set()
            prepared = next_prepared
        pipeline.shutdown()
        assert ps.reads == ([[1], [0, 4], [1, 2], [2]] if rank == 0
                            else [[2], [0, 5], [1, 2]]), ps.reads
        assert rows[1]["ps_pipeline_metadata_sent_bytes"] == 32
        assert rows[2]["ps_pipeline_metadata_sent_bytes"] == 56
        assert rows[1]["ps_pipeline_global_dirty_rows"] == 2
        assert rows[2]["ps_pipeline_global_dirty_rows"] == 5
        assert not ps.closed[-1]["abort"] and not pipeline._metadata
        Path(directory, f"rank{rank}.json").write_text(json.dumps({"rows": rows, "reads": ps.reads}))
    finally:
        ps.release.set()
        pipeline.shutdown(flush=False)
        dist.destroy_process_group()


class TestDependencySplit(unittest.TestCase):
    def test_sorted_dependency_membership_preserves_large_ids_and_empty_sets(self):
        ps = _PS()
        cases = [([], []), ([0], []), ([0, 1, 2], [0, 2]),
                 ([0, 2**56, 2**63-2, 2**63-1], [2**56, 2**63-1])]
        with _split(ps) as pipeline:
            pipeline._pull = lambda ids, previous: (torch.empty((ids.numel(), 2)), 0.0, 0.0)
            previous = Future()
            previous.set_result(None)
            for query, dirty in cases:
                with self.subTest(query=query, dirty=dirty):
                    ids = torch.tensor(query, dtype=torch.int64)
                    metadata = Future()
                    metadata.set_result((torch.tensor(dirty, dtype=torch.int64), {}))
                    _, clean_pos, dirty_pos, _, _ = pipeline._read_clean(ids, previous, metadata)
                    self.assertEqual(ids[clean_pos].tolist(), [v for v in query if v not in dirty])
                    self.assertEqual(ids[dirty_pos].tolist(), [v for v in query if v in dirty])

    def test_split_merge_preserves_all_float_bits(self):
        # Copies must preserve signed zeros, subnormals, and finite extremes.
        bits = torch.tensor([[0, -2147483648], [1, -2147483647],
                             [2139095039, -8388609], [1065353216, -1082130432]], dtype=torch.int32)
        expected = bits.view(torch.float32)
        clean_pos = torch.tensor([0, 2]); dirty_pos = torch.tensor([1, 3])
        ps = _PS()
        with _split(ps) as pipeline:
            pipeline._pull = lambda ids, previous: (expected[dirty_pos].clone(), 0.0, 0.0)
            clean = Future(); commit = Future(); commit.set_result(0.0)
            clean.set_result((torch.arange(4), clean_pos, dirty_pos,
                              expected[clean_pos].clone(), {"ps_pipeline_clean_pull_ms": 0.0}))
            actual, _, _, _ = pipeline._finish_split_read(clean, commit)
            self.assertTrue(torch.equal(actual.view(torch.int32), bits))

    def test_remote_dirty_rows_wait_and_each_next_row_is_read_once(self):
        class RemotePS(_PS):
            def sparse_gradient_commit_wait(self, step):
                super().sparse_gradient_commit_wait(step)
                if step == 0:
                    self.rows[2].sub_(.25)

        ps = RemotePS()
        with _split(ps) as pipeline:
            first = _prepare(pipeline, 0, [1, 1])
            second = _prepare(pipeline, 1, [0, 1, 2, 4])
            _, row0 = _consume(pipeline, 0, first)
            ps.release.clear()
            pipeline.submit_sparse_grad(first[1][0], torch.ones((1, 2)))
            self.assertTrue(ps.waiting.wait(3))
            self.assertEqual(ps.reads, [[1], [0, 4]])
            self.assertFalse(pipeline._read.done())
            ps.release.set()
            values, row1 = _consume(pipeline, 1, second)
            torch.testing.assert_close(values, torch.tensor([[1., 1.], [.95, .95], [.75, .75], [1., 1.]]),
                                       rtol=0, atol=0)
            self.assertEqual(ps.reads, [[1], [0, 4], [1, 2]])
            self.assertEqual(row1["ps_pipeline_clean_rows"], 2)
            self.assertEqual(row1["ps_pipeline_dirty_rows"], 2)
            self.assertEqual(row1["ps_pipeline_read_bytes"], 32)
            self.assertEqual(set(row0), set(row1))
            pipeline.submit_sparse_grad(second[1][0], torch.zeros((4, 2)))
            pipeline.shutdown()
            self.assertEqual(len(ps.reads), 3)
            self.assertFalse(ps.closed[-1]["abort"])

    def test_empty_all_clean_and_all_dirty_partitions(self):
        for next_ids, dirty, expected in (([], (1,), [[1]]), ([4], (1,), [[1], [4]]),
                                          ([1], (1,), [[1], [1]])):
            with self.subTest(next_ids=next_ids), _split(_PS(), dirty=dirty) as pipeline:
                first = _prepare(pipeline, 0, [1])
                second = _prepare(pipeline, 1, next_ids)
                _consume(pipeline, 0, first)
                pipeline.submit_sparse_grad(first[1][0], torch.ones((1, 2)))
                values, _ = _consume(pipeline, 1, second)
                torch.testing.assert_close(values, pipeline.kv_client.rows[second[1][0]], rtol=0, atol=0)
                pipeline.submit_sparse_grad(second[1][0], torch.zeros((len(next_ids), 2)))
                pipeline.shutdown()
                self.assertEqual(pipeline.kv_client.reads, expected)

    def test_final_step_has_no_metadata_or_future_read(self):
        with _split(_PS(), total_steps=1) as pipeline:
            pipeline._exchange_dirty_ids = mock.Mock(side_effect=AssertionError("unneeded collective"))
            first = _prepare(pipeline, 0, [])
            _consume(pipeline, 0, first)
            pipeline.submit_sparse_grad(first[1][0], torch.empty((0, 2)))
            with self.assertRaises(RuntimeError):
                _prepare(pipeline, 1, [1])
            pipeline.shutdown()
            pipeline._exchange_dirty_ids.assert_not_called()
            self.assertEqual(pipeline.kv_client.reads, [])

    def test_missing_lookahead_rejected_before_native_submission(self):
        with _split(_PS()) as pipeline:
            first = _prepare(pipeline, 0, [1])
            _consume(pipeline, 0, first)
            with self.assertRaisesRegex(RuntimeError, "Prepare next"):
                pipeline.submit_sparse_grad(first[1][0], torch.ones((1, 2)))
            self.assertEqual(pipeline.kv_client.commits, [])

    def test_commit_failure_blocks_dirty_read_and_preserves_first_error(self):
        ps = _PS()
        ps.fail = ps.fail_close = True
        with _split(ps) as pipeline:
            first = _prepare(pipeline, 0, [1])
            second = _prepare(pipeline, 1, [1, 4])
            _consume(pipeline, 0, first)
            pipeline.submit_sparse_grad(first[1][0], torch.ones((1, 2)))
            with self.assertRaisesRegex(OSError, "failed PS commit"):
                _consume(pipeline, 1, second)
            with mock.patch.object(dist, "destroy_process_group", side_effect=ValueError("cleanup failed")):
                with self.assertRaisesRegex(OSError, "failed PS commit"):
                    pipeline.shutdown()
            self.assertEqual(ps.reads, [[1], [4]])
            self.assertTrue(pipeline._closed)
            self.assertTrue(ps.closed[-1]["abort"])

    def test_metadata_failure_never_reads_next_values_and_abort_cleans_up(self):
        with _split(_PS()) as pipeline:
            pipeline._exchange_dirty_ids = mock.Mock(side_effect=ValueError("bad metadata"))
            first = _prepare(pipeline, 0, [1])
            second = _prepare(pipeline, 1, [1, 4])
            _consume(pipeline, 0, first)
            pipeline.submit_sparse_grad(first[1][0], torch.ones((1, 2)))
            with self.assertRaisesRegex(ValueError, "bad metadata"):
                _consume(pipeline, 1, second)
            with mock.patch.object(dist, "destroy_process_group", side_effect=ValueError("cleanup failed")):
                pipeline.shutdown(flush=False)
            self.assertEqual(pipeline.kv_client.reads, [[1]])
            self.assertTrue(pipeline._closed)
            self.assertTrue(pipeline.kv_client.closed[-1]["abort"])

    def test_requires_two_ranks_and_known_positive_step_count(self):
        for steps in (None, 0, -1, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                _pipeline(_PS(), dependency_split=True, total_steps=steps)
        with self.assertRaisesRegex(ValueError, "world=2"):
            _pipeline(_PS(), dependency_split=True, total_steps=2)

    def test_constructor_failure_releases_only_created_resources_and_preserves_error(self):
        for failure in ("group", "metadata_worker", "io_worker"):
            with self.subTest(failure=failure):
                original = RuntimeError("construction failed")
                executor = mock.Mock()
                executor.shutdown.side_effect = OSError("worker cleanup failed")
                group = object()
                create_group = mock.Mock(side_effect=original) if failure == "group" else mock.Mock(return_value=group)
                destroy = mock.Mock(side_effect=OSError("group cleanup failed"))
                effects = [executor, original] if failure == "io_worker" else [original]
                with mock.patch.multiple(dist, is_initialized=lambda: True,
                                         get_world_size=lambda: 2, get_rank=lambda: 0,
                                         broadcast=lambda *a, **kw: None,
                                         new_group=create_group, destroy_process_group=destroy), \
                     mock.patch("recstore.ps_pipeline.ThreadPoolExecutor", side_effect=effects) as create_worker:
                    with self.assertRaises(RuntimeError) as raised:
                        _pipeline(_PS(), dependency_split=True, total_steps=2)
                    self.assertIs(raised.exception, original)
                self.assertEqual(create_worker.call_count, {"group": 0, "metadata_worker": 1, "io_worker": 2}[failure])
                if failure == "group":
                    destroy.assert_not_called()
                else:
                    destroy.assert_called_once_with(group)
                if failure == "io_worker":
                    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
                else:
                    executor.shutdown.assert_not_called()

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo unavailable")
    def test_two_rank_cpu_gloo_metadata_and_global_commit(self):
        with tempfile.TemporaryDirectory(prefix="recstore_split_gloo_") as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_gloo_rank, args=(rank, directory)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(35)
                self.assertTrue(all(not process.is_alive() for process in processes), "Gloo worker stalled")
                self.assertEqual([p.exitcode for p in processes], [0, 0])
                self.assertTrue(all(Path(directory, f"rank{rank}.json").exists() for rank in range(2)))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)


if __name__ == "__main__":
    unittest.main()
