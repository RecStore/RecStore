from __future__ import annotations

import types
import unittest

import torch

from ..bagpipe_cache.manager import (
    BagPipeWindowScheduler,
    attach_or_refetch_with_bagpipe_policy,
    notify_sparse_update,
)


class _FakeConsumeDecision:
    def __init__(
        self,
        stale_ids: list[int],
        valid_ids: list[int],
        *,
        stale_cached_ids: list[int] | None = None,
        stale_refetch_ids: list[int] | None = None,
    ) -> None:
        self.stale_ids = torch.tensor(stale_ids, dtype=torch.int64)
        self.valid_prefetch_ids = torch.tensor(valid_ids, dtype=torch.int64)
        self.stale_cached_ids = torch.tensor(
            stale_cached_ids if stale_cached_ids is not None else [],
            dtype=torch.int64,
        )
        self.stale_refetch_ids = torch.tensor(
            stale_refetch_ids if stale_refetch_ids is not None else stale_ids,
            dtype=torch.int64,
        )

    @property
    def requires_refetch(self) -> bool:
        return bool(self.stale_refetch_ids.numel() > 0)


class _FakeBagPipePolicy:
    def __init__(self, decision: _FakeConsumeDecision) -> None:
        self.decision = decision
        self.consume_steps: list[int] = []

    def on_consume(self, step: int) -> _FakeConsumeDecision:
        self.consume_steps.append(int(step))
        return self.decision


class _FakeBagPipeUpdatePolicy:
    def __init__(self) -> None:
        self.update_calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def on_update(self, step: int, ids, *, cache_updated_ids=()) -> None:
        self.update_calls.append(
            (
                int(step),
                ids.detach().cpu().clone(),
                cache_updated_ids.detach().cpu().clone(),
            )
        )


class _FakeCachedSubsetPolicy(_FakeBagPipeUpdatePolicy):
    def __init__(self, cached_ids: list[int]) -> None:
        super().__init__()
        self.cached_ids = set(int(v) for v in cached_ids)

    def cached_tensor_for(self, ids) -> torch.Tensor:
        values = [
            int(v)
            for v in ids.detach().to(dtype=torch.int64, device="cpu").flatten().tolist()
            if int(v) in self.cached_ids
        ]
        return torch.tensor(values, dtype=torch.int64)


class _FakePrefetcher:
    def __init__(self, *, discard_result: bool = True) -> None:
        self.discard_result = bool(discard_result)
        self.discard_calls = 0
        self.attach_calls = 0
        self.attach_kwargs: list[dict[str, object]] = []

    def discard_next_ready(self) -> bool:
        self.discard_calls += 1
        return self.discard_result

    def attach_next(self, **kwargs) -> bool:
        self.attach_calls += 1
        self.attach_kwargs.append(dict(kwargs))
        return True


class _FakeEmbeddingModule:
    def __init__(self) -> None:
        self.issue_fused_prefetch_calls: list[object] = []
        self.cache_insert_ids: list[torch.Tensor] = []
        self.cache_invalidate_ids: list[torch.Tensor] = []

    def issue_fused_prefetch(self, sparse_features):
        self.issue_fused_prefetch_calls.append(sparse_features)

    def prefill_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        self.cache_insert_ids.append(fused_ids.detach().cpu().clone())
        return True

    def invalidate_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        self.cache_invalidate_ids.append(fused_ids.detach().cpu().clone())
        return True


class _FakeLookaheadPrefetcher:
    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self.enqueued_ids: list[torch.Tensor] = []
        self.advance_calls = 0
        self.stats = {
            "prefetch_issued_batches": 0.0,
            "prefetch_total_ids": 0.0,
            "prefetch_issue_ms": 0.0,
        }

    def consume_stats(self, *, reset: bool = True):
        del reset
        return dict(self.stats)

    def enqueue_fused_ids(self, fused_ids: torch.Tensor) -> None:
        self.enqueued_ids.append(fused_ids.detach().cpu().clone())
        self.stats["prefetch_issued_batches"] += 1.0
        self.stats["prefetch_total_ids"] += float(fused_ids.numel())
        self.stats["prefetch_issue_ms"] += 0.5

    def advance(self) -> bool:
        self.advance_calls += 1
        return True


class _RealPrefetchModule:
    def __init__(self) -> None:
        self.next_handle = 1
        self.issued_fused_ids: dict[int, torch.Tensor] = {}
        self.attached: list[tuple[int, list[int]]] = []

    def issue_fused_id_prefetch(
        self,
        fused_ids: torch.Tensor,
        *,
        record_handle: bool = True,
    ):
        del record_handle
        handle = self.next_handle
        self.next_handle += 1
        fused_ids_cpu = fused_ids.detach().to(dtype=torch.int64, device="cpu").flatten()
        self.issued_fused_ids[handle] = fused_ids_cpu.clone()
        return handle, int(fused_ids_cpu.numel()), 0.0, fused_ids_cpu, None

    def set_fused_prefetch_handle(
        self,
        handle: int,
        *,
        num_ids: int,
        issue_ts: float,
        fused_ids_cpu: torch.Tensor,
        fused_inverse=None,
        invalid_fused_ids_cpu=None,
    ) -> None:
        del num_ids, issue_ts, fused_inverse, invalid_fused_ids_cpu
        self.attached.append((int(handle), fused_ids_cpu.tolist()))

    def prefill_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        del fused_ids
        return True

    def invalidate_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
        del fused_ids
        return True


class TestBagPipeCacheManager(unittest.TestCase):
    def test_window_scheduler_issues_future_nonempty_prefetches_after_step_barrier(self) -> None:
        from ..bagpipe_cache import BagPipeCachePolicy

        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        prefetcher = _FakeLookaheadPrefetcher(depth=2)
        module = _FakeEmbeddingModule()
        scheduler = BagPipeWindowScheduler(
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            read_before_update=True,
            read_mode="prefetch",
        )
        prepared_batches = [
            (0, {}, torch.tensor([4], dtype=torch.int64)),
            (1, {}, torch.tensor([4, 5], dtype=torch.int64)),
            (2, {}, torch.tensor([4, 5], dtype=torch.int64)),
            (3, {}, torch.tensor([5], dtype=torch.int64)),
        ]

        for step, _, fused_ids in prepared_batches:
            scheduler.observe_batch(step, fused_ids)
        scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)
        scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)

        rows = [row for _, row, _ in prepared_batches]
        self.assertEqual(prefetcher.enqueued_ids, [])
        self.assertEqual(prefetcher.advance_calls, 0)
        end_row: dict[str, object] = {}
        scheduler.on_step_end(0, end_row)
        self.assertEqual(prefetcher.enqueued_ids, [])
        scheduler.issue_prefetches_ready_after_update(current_step=0, row=end_row)
        self.assertEqual([ids.tolist() for ids in prefetcher.enqueued_ids], [[5]])
        self.assertEqual(prefetcher.advance_calls, 1)
        self.assertEqual(rows[0]["bagpipe_prefetch_ids"], 1)
        self.assertEqual(rows[0]["bagpipe_cache_insert_ids"], 1)
        self.assertEqual(rows[0]["bagpipe_cache_insert_success_ids"], 0)
        self.assertEqual(rows[1]["bagpipe_prefetch_ids"], 1)
        self.assertEqual(rows[2]["bagpipe_prefetch_ids"], 0)
        self.assertEqual([ids.tolist() for ids in module.cache_insert_ids], [])
        self.assertNotIn("bagpipe_prefetch_ids", rows[3])
        self.assertEqual(end_row["prefetch_issued_batches"], 1.0)
        self.assertEqual(end_row["prefetch_total_ids"], 1.0)
        self.assertEqual(end_row["prefetch_issue_ms"], 0.5)

    def test_window_scheduler_real_prefetcher_attaches_handle_for_matching_step(self) -> None:
        from model_zoo.rs_demo.runtime.prefetch import LookaheadPrefetcher
        from ..bagpipe_cache import BagPipeCachePolicy

        module = _RealPrefetchModule()
        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=8)
        prefetcher = LookaheadPrefetcher(module, depth=2, embedding_dim=4)
        scheduler = BagPipeWindowScheduler(
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            read_before_update=True,
            read_mode="prefetch",
        )
        prepared_batches = [
            (0, {}, torch.tensor([100], dtype=torch.int64)),
            (1, {}, torch.tensor([101], dtype=torch.int64)),
            (2, {}, torch.tensor([102], dtype=torch.int64)),
            (3, {}, torch.tensor([103], dtype=torch.int64)),
        ]
        for step, _, fused_ids in prepared_batches:
            scheduler.observe_batch(step, fused_ids)

        scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)
        row0: dict[str, object] = {}
        scheduler.on_step_end(0, row0)
        scheduler.issue_prefetches_ready_after_update(current_step=0, row=row0)

        self.assertTrue(prefetcher.attach_next())
        self.assertEqual(module.attached, [(1, [101])])

        scheduler.plan_ready(current_step=1, prepared_batches=prepared_batches)
        row1: dict[str, object] = {}
        scheduler.on_step_end(1, row1)
        scheduler.issue_prefetches_ready_after_update(current_step=1, row=row1)

        self.assertTrue(prefetcher.attach_next())
        self.assertEqual(module.attached, [(1, [101]), (2, [102])])

    def test_window_scheduler_clamps_large_depth_issue_window(self) -> None:
        from model_zoo.rs_demo.runtime.prefetch import LookaheadPrefetcher
        from ..bagpipe_cache import BagPipeCachePolicy

        module = _RealPrefetchModule()
        policy = BagPipeCachePolicy(lookahead_depth=5, cache_capacity=16)
        prefetcher = LookaheadPrefetcher(module, depth=5, embedding_dim=4)
        scheduler = BagPipeWindowScheduler(
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            read_before_update=True,
            read_mode="prefetch",
            prefetch_issue_depth=2,
        )
        prepared_batches = [
            (step, {}, torch.tensor([100 + step], dtype=torch.int64))
            for step in range(6)
        ]
        for step, _, fused_ids in prepared_batches:
            scheduler.observe_batch(step, fused_ids)

        scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)
        row0: dict[str, object] = {}
        scheduler.on_step_end(0, row0)
        scheduler.issue_prefetches_ready_after_update(current_step=0, row=row0)

        self.assertEqual(
            [ids.tolist() for ids in module.issued_fused_ids.values()],
            [[101], [102]],
        )
        self.assertNotIn("bagpipe_prefetch_ids", prepared_batches[3][1])

    def test_window_scheduler_surfaces_capacity_overflow(self) -> None:
        from ..bagpipe_cache import BagPipeCachePolicy

        policy = BagPipeCachePolicy(lookahead_depth=2, cache_capacity=1)
        prefetcher = _FakeLookaheadPrefetcher(depth=1)
        module = _FakeEmbeddingModule()
        scheduler = BagPipeWindowScheduler(
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            read_before_update=True,
            read_mode="prefetch",
        )
        prepared_batches = [
            (0, {}, torch.tensor([4], dtype=torch.int64)),
            (1, {}, torch.tensor([4, 5], dtype=torch.int64)),
            (2, {}, torch.tensor([5], dtype=torch.int64)),
        ]

        for step, _, fused_ids in prepared_batches:
            scheduler.observe_batch(step, fused_ids)

        with self.assertRaisesRegex(RuntimeError, "BagPipe cache capacity exceeded"):
            scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)
        self.assertEqual(module.cache_invalidate_ids, [])

    def test_window_scheduler_invalidates_step_end_expired_gpu_cache_rows(self) -> None:
        from ..bagpipe_cache import BagPipeCachePolicy

        policy = BagPipeCachePolicy(lookahead_depth=1, cache_capacity=8)
        prefetcher = _FakeLookaheadPrefetcher(depth=1)
        module = _FakeEmbeddingModule()
        scheduler = BagPipeWindowScheduler(
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            read_before_update=True,
            read_mode="prefetch",
        )
        prepared_batches = [
            (0, {}, torch.tensor([4], dtype=torch.int64)),
            (1, {}, torch.tensor([4], dtype=torch.int64)),
        ]

        for step, _, fused_ids in prepared_batches:
            scheduler.observe_batch(step, fused_ids)
        scheduler.plan_ready(current_step=0, prepared_batches=prepared_batches)
        row: dict[str, object] = {}

        evicted = scheduler.on_step_end(2, row)

        self.assertEqual(evicted.tolist(), [4])
        self.assertEqual([ids.tolist() for ids in module.cache_invalidate_ids], [[4]])
        self.assertEqual(row["bagpipe_step_end_evict_ids"], 1)
        self.assertEqual(row["bagpipe_step_end_gpu_cache_evict_success_ids"], 1)
        self.assertEqual(row["bagpipe_step_end_gpu_cache_evict_failures"], 0)

    def test_valid_prefetch_attaches_ready_handle(self) -> None:
        policy = _FakeBagPipePolicy(_FakeConsumeDecision([], [7]))
        prefetcher = _FakePrefetcher()
        module = _FakeEmbeddingModule()
        row: dict[str, object] = {}

        attach_or_refetch_with_bagpipe_policy(
            prefetch_depth=1,
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            sparse_features=object(),
            row=row,
            step=2,
        )

        self.assertEqual(policy.consume_steps, [2])
        self.assertEqual(prefetcher.attach_calls, 1)
        self.assertEqual(prefetcher.discard_calls, 0)
        self.assertEqual(module.issue_fused_prefetch_calls, [])
        self.assertEqual(row["bagpipe_stale_ids"], 0)
        self.assertEqual(row["bagpipe_valid_prefetch_ids"], 1)
        self.assertEqual(row["bagpipe_discarded_stale_handle"], 0)

    def test_all_stale_refetch_discards_handle_and_refetches_same_batch(self) -> None:
        policy = _FakeBagPipePolicy(_FakeConsumeDecision([7], []))
        prefetcher = _FakePrefetcher()
        module = _FakeEmbeddingModule()
        features = object()
        row: dict[str, object] = {}

        attach_or_refetch_with_bagpipe_policy(
            prefetch_depth=1,
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            sparse_features=features,
            row=row,
            step=2,
        )

        self.assertEqual(prefetcher.discard_calls, 1)
        self.assertEqual(prefetcher.attach_calls, 0)
        self.assertEqual(module.issue_fused_prefetch_calls, [features])
        self.assertEqual(row["bagpipe_stale_ids"], 1)
        self.assertEqual(row["bagpipe_stale_cached_ids"], 0)
        self.assertEqual(row["bagpipe_stale_refetch_ids"], 1)
        self.assertEqual(row["bagpipe_discarded_stale_handle"], 1)

    def test_partial_stale_refetch_repairs_invalid_ids_without_discard(self) -> None:
        policy = _FakeBagPipePolicy(_FakeConsumeDecision([7], [8]))
        prefetcher = _FakePrefetcher()
        module = _FakeEmbeddingModule()
        row: dict[str, object] = {}

        attach_or_refetch_with_bagpipe_policy(
            prefetch_depth=1,
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            sparse_features=object(),
            row=row,
            step=2,
        )

        self.assertEqual(prefetcher.discard_calls, 0)
        self.assertEqual(prefetcher.attach_calls, 1)
        self.assertEqual(
            prefetcher.attach_kwargs[0]["invalid_fused_ids"].tolist(),
            [7],
        )
        self.assertEqual(module.issue_fused_prefetch_calls, [])
        self.assertEqual(row["bagpipe_discarded_stale_handle"], 0)

    def test_stale_cached_discards_handle_without_refetch(self) -> None:
        policy = _FakeBagPipePolicy(
            _FakeConsumeDecision(
                [7],
                [8],
                stale_cached_ids=[7],
                stale_refetch_ids=[],
            )
        )
        prefetcher = _FakePrefetcher()
        module = _FakeEmbeddingModule()
        row: dict[str, object] = {}

        attach_or_refetch_with_bagpipe_policy(
            prefetch_depth=1,
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            sparse_features=object(),
            row=row,
            step=2,
        )

        self.assertEqual(prefetcher.discard_calls, 0)
        self.assertEqual(prefetcher.attach_calls, 1)
        self.assertEqual(
            prefetcher.attach_kwargs[0]["invalid_fused_ids"].tolist(),
            [7],
        )
        self.assertEqual(module.issue_fused_prefetch_calls, [])
        self.assertEqual(row["bagpipe_stale_ids"], 1)
        self.assertEqual(row["bagpipe_stale_cached_ids"], 1)
        self.assertEqual(row["bagpipe_stale_refetch_ids"], 0)
        self.assertEqual(row["bagpipe_discarded_stale_handle"], 0)

    def test_depth_zero_issues_direct_prefetch_and_sets_bagpipe_counters(self) -> None:
        policy = _FakeBagPipePolicy(_FakeConsumeDecision([7], [8]))
        prefetcher = _FakePrefetcher()
        module = _FakeEmbeddingModule()
        features = object()
        row: dict[str, object] = {}

        attach_or_refetch_with_bagpipe_policy(
            prefetch_depth=0,
            bagpipe_policy=policy,
            lookahead_prefetcher=prefetcher,
            embedding_module=module,
            sparse_features=features,
            row=row,
            step=2,
        )

        self.assertEqual(policy.consume_steps, [])
        self.assertEqual(prefetcher.attach_calls, 0)
        self.assertEqual(prefetcher.discard_calls, 0)
        self.assertEqual(module.issue_fused_prefetch_calls, [features])
        self.assertEqual(row["bagpipe_stale_ids"], 0)
        self.assertEqual(row["bagpipe_stale_cached_ids"], 0)
        self.assertEqual(row["bagpipe_stale_refetch_ids"], 0)
        self.assertEqual(row["bagpipe_valid_prefetch_ids"], 0)
        self.assertEqual(row["bagpipe_discarded_stale_handle"], 0)

    def test_sparse_update_marks_gpu_cache_updated_ids_for_policy(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls = []

            def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate):
                self.calls.append((name, ids.clone(), grads.clone(), float(learning_rate)))
                return True

        client = _Client()
        module = types.SimpleNamespace(kv_client=client)
        optimizer = types.SimpleNamespace(
            last_update_payloads=lambda: [
                {
                    "module": module,
                    "name": "table0",
                    "ids": torch.tensor([4, 5], dtype=torch.int64),
                    "grads": torch.ones((2, 4), dtype=torch.float32),
                    "lr": 0.25,
                }
            ]
        )
        policy = _FakeBagPipeUpdatePolicy()
        row: dict[str, object] = {}

        notify_sparse_update(
            bagpipe_policy=policy,
            sparse_optimizer=optimizer,
            fallback_updated_ids=torch.tensor([4, 5], dtype=torch.int64),
            row=row,
            step=3,
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], "table0")
        self.assertEqual(client.calls[0][3], 0.25)
        self.assertEqual(row["bagpipe_gpu_cache_update_ids"], 2)
        self.assertEqual(row["bagpipe_policy_cached_update_ids"], 0)
        self.assertEqual(row["bagpipe_gpu_cache_update_failures"], 0)
        self.assertEqual(policy.update_calls[0][1].tolist(), [4, 5])
        self.assertEqual(policy.update_calls[0][2].tolist(), [4, 5])

    def test_sparse_update_failure_does_not_mark_gpu_cache_updated(self) -> None:
        class _Client:
            def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate):
                del name, ids, grads, learning_rate
                return False

        module = types.SimpleNamespace(kv_client=_Client())
        optimizer = types.SimpleNamespace(
            last_update_payloads=lambda: [
                {
                    "module": module,
                    "name": "table0",
                    "ids": torch.tensor([4, 5], dtype=torch.int64),
                    "grads": torch.ones((2, 4), dtype=torch.float32),
                    "lr": 0.25,
                }
            ]
        )
        policy = _FakeBagPipeUpdatePolicy()
        row: dict[str, object] = {}

        notify_sparse_update(
            bagpipe_policy=policy,
            sparse_optimizer=optimizer,
            fallback_updated_ids=torch.tensor([4, 5], dtype=torch.int64),
            row=row,
            step=3,
        )

        self.assertEqual(row["bagpipe_gpu_cache_update_ids"], 0)
        self.assertEqual(row["bagpipe_policy_cached_update_ids"], 0)
        self.assertEqual(row["bagpipe_gpu_cache_update_failures"], 1)
        self.assertEqual(policy.update_calls[0][1].tolist(), [4, 5])
        self.assertEqual(policy.update_calls[0][2].tolist(), [])

    def test_sparse_update_filters_payload_to_cached_policy_ids(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls = []

            def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate):
                self.calls.append((name, ids.clone(), grads.clone(), float(learning_rate)))
                return True

        client = _Client()
        module = types.SimpleNamespace(kv_client=client)
        optimizer = types.SimpleNamespace(
            last_update_payloads=lambda: [
                {
                    "module": module,
                    "name": "table0",
                    "ids": torch.tensor([4, 5, 6], dtype=torch.int64),
                    "grads": torch.arange(12, dtype=torch.float32).view(3, 4),
                    "lr": 0.25,
                }
            ]
        )
        policy = _FakeCachedSubsetPolicy([5])
        row: dict[str, object] = {}

        notify_sparse_update(
            bagpipe_policy=policy,
            sparse_optimizer=optimizer,
            fallback_updated_ids=torch.tensor([4, 5, 6], dtype=torch.int64),
            row=row,
            step=3,
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1].tolist(), [5])
        self.assertEqual(client.calls[0][2].tolist(), [[4.0, 5.0, 6.0, 7.0]])
        self.assertEqual(row["bagpipe_gpu_cache_update_ids"], 1)
        self.assertEqual(row["bagpipe_policy_cached_update_ids"], 1)
        self.assertEqual(row["bagpipe_gpu_cache_update_failures"], 0)
        self.assertEqual(policy.update_calls[0][1].tolist(), [4, 5, 6])
        self.assertEqual(policy.update_calls[0][2].tolist(), [5])

    def test_sparse_update_repairs_missing_cached_row_before_policy_update(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.calls = 0

            def apply_sgd_update_gpu_cache(self, name, ids, grads, *, learning_rate):
                del name, ids, grads, learning_rate
                self.calls += 1
                return self.calls > 1

        class _Module:
            def __init__(self) -> None:
                self.kv_client = _Client()
                self.repaired_ids: list[torch.Tensor] = []

            def prefill_gpu_cache_for_fused_ids(self, fused_ids: torch.Tensor) -> bool:
                self.repaired_ids.append(fused_ids.detach().cpu().clone())
                return True

        module = _Module()
        optimizer = types.SimpleNamespace(
            last_update_payloads=lambda: [
                {
                    "module": module,
                    "name": "table0",
                    "ids": torch.tensor([5], dtype=torch.int64),
                    "grads": torch.ones((1, 4), dtype=torch.float32),
                    "lr": 0.25,
                }
            ]
        )
        policy = _FakeCachedSubsetPolicy([5])
        row: dict[str, object] = {}

        notify_sparse_update(
            bagpipe_policy=policy,
            sparse_optimizer=optimizer,
            fallback_updated_ids=torch.tensor([5], dtype=torch.int64),
            row=row,
            step=3,
        )

        self.assertEqual(module.kv_client.calls, 2)
        self.assertEqual([ids.tolist() for ids in module.repaired_ids], [[5]])
        self.assertEqual(row["bagpipe_gpu_cache_update_ids"], 1)
        self.assertEqual(row["bagpipe_policy_cached_update_ids"], 1)
        self.assertEqual(row["bagpipe_gpu_cache_update_failures"], 0)
        self.assertEqual(policy.update_calls[0][2].tolist(), [5])


if __name__ == "__main__":
    unittest.main()
