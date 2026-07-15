from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _bool_int(flag: bool) -> int:
    return 1 if flag else 0


@dataclass(frozen=True)
class BagPipeConsumeResult:
    stale_ids: int
    stale_cached_ids: int
    stale_refetch_ids: int
    valid_prefetch_ids: int
    discarded_stale_handle: int


@dataclass(frozen=True)
class BagPipeUpdateResult:
    updated_ids: torch.Tensor
    cache_updated_ids: torch.Tensor
    gpu_cache_update_failures: int


class BagPipeWindowScheduler:
    """Coordinates BagPipe oracle planning and lookahead prefetch enqueue."""

    def __init__(
        self,
        *,
        bagpipe_policy: Any,
        lookahead_prefetcher: Any,
        embedding_module: Any | None = None,
        read_before_update: bool,
        read_mode: str,
        prefetch_issue_depth: int | None = None,
    ) -> None:
        self.bagpipe_policy = bagpipe_policy
        self.lookahead_prefetcher = lookahead_prefetcher
        self.embedding_module = embedding_module
        self.read_before_update = bool(read_before_update)
        self.read_mode = str(read_mode)
        if prefetch_issue_depth is None:
            self.prefetch_issue_depth = self.depth
        else:
            requested_issue_depth = max(0, int(prefetch_issue_depth))
            self.prefetch_issue_depth = (
                self.depth if requested_issue_depth == 0 else min(self.depth, requested_issue_depth)
            )
        self._planned_steps: set[int] = set()
        self._pending_prefetch: dict[int, torch.Tensor] = {}

    @property
    def depth(self) -> int:
        return int(getattr(self.lookahead_prefetcher, "depth", 0))

    def observe_batch(self, step: int, fused_ids: torch.Tensor) -> None:
        self.bagpipe_policy.observe_batch(int(step), fused_ids)

    def plan_ready(
        self,
        *,
        current_step: int,
        prepared_batches: Any,
    ) -> None:
        plan_until = int(current_step) + self.prefetch_issue_depth
        for item in prepared_batches:
            batch_step, row = self._extract_step_and_row(item)
            if batch_step > plan_until or batch_step in self._planned_steps:
                continue
            bagpipe_plan = self.bagpipe_policy.plan_for_step(batch_step)
            self._record_plan(row, bagpipe_plan)
            cache_insert_ids = self._cache_insert_ids_not_prefetched(
                bagpipe_plan.cache_insert_ids,
                bagpipe_plan.prefetch_ids,
            )
            self._insert_planned_cache_rows(row, cache_insert_ids)
            self._invalidate_evicted_cache_rows(
                row,
                bagpipe_plan.evict_ids,
                prefix="bagpipe_gpu_cache_evict",
            )
            self._planned_steps.add(batch_step)
            if not (self.read_before_update and self.read_mode == "prefetch"):
                continue
            if batch_step <= int(current_step):
                continue
            self._pending_prefetch[batch_step] = bagpipe_plan.prefetch_ids

    def on_step_end(self, step: int, row: dict[str, Any]) -> torch.Tensor:
        evicted = self.bagpipe_policy.on_step_end(int(step))
        row["bagpipe_step_end_evict_ids"] = int(evicted.numel())
        self._invalidate_evicted_cache_rows(
            row,
            evicted,
            prefix="bagpipe_step_end_gpu_cache_evict",
        )
        return evicted

    def issue_prefetches_ready_after_update(
        self,
        *,
        current_step: int,
        row: dict[str, Any],
    ) -> None:
        if not (self.read_before_update and self.read_mode == "prefetch"):
            return
        ready_until = int(current_step) + self.prefetch_issue_depth
        issued_nonempty = False
        for batch_step in sorted(list(self._pending_prefetch.keys())):
            if batch_step > ready_until:
                continue
            prefetch_ids = self._pending_prefetch.pop(batch_step)
            if prefetch_ids.numel() == 0:
                issued = getattr(self.bagpipe_policy, "on_prefetch_issued", None)
                if callable(issued):
                    issued(batch_step, prefetch_ids)
                continue
            self._enqueue_planned_prefetch(row, prefetch_ids)
            issued_nonempty = True
            issued = getattr(self.bagpipe_policy, "on_prefetch_issued", None)
            if callable(issued):
                issued(batch_step, prefetch_ids)
        if issued_nonempty:
            advance_all = getattr(self.lookahead_prefetcher, "advance_all", None)
            if callable(advance_all):
                advance_all()

    @staticmethod
    def _extract_step_and_row(item: Any) -> tuple[int, dict[str, Any]]:
        batch_step = int(item[0])
        row = item[1]
        if not isinstance(row, dict):
            raise TypeError("prepared batch item must expose a row dict at index 1")
        return batch_step, row

    @staticmethod
    def _record_plan(row: dict[str, Any], bagpipe_plan: Any) -> None:
        row["bagpipe_prefetch_ids"] = int(bagpipe_plan.prefetch_ids.numel())
        row["bagpipe_cache_insert_ids"] = int(bagpipe_plan.cache_insert_ids.numel())
        row["bagpipe_evict_ids"] = int(bagpipe_plan.evict_ids.numel())
        row["bagpipe_sync_now_ids"] = int(bagpipe_plan.sync_now_ids.numel())
        row["bagpipe_sync_later_ids"] = int(bagpipe_plan.sync_later_ids.numel())
        row["bagpipe_no_sync_ids"] = int(bagpipe_plan.no_sync_ids.numel())

    def _enqueue_planned_prefetch(
        self,
        row: dict[str, Any],
        prefetch_ids: torch.Tensor,
    ) -> None:
        if prefetch_ids.numel() == 0:
            return
        prefetch_issue_before = self.lookahead_prefetcher.consume_stats(reset=False)
        self.lookahead_prefetcher.enqueue_fused_ids(prefetch_ids)
        self.lookahead_prefetcher.advance()
        prefetch_issue_after = self.lookahead_prefetcher.consume_stats(reset=False)
        for key in (
            "prefetch_issued_batches",
            "prefetch_total_ids",
            "prefetch_issue_ms",
        ):
            row[key] = float(prefetch_issue_after.get(key, 0.0)) - float(
                prefetch_issue_before.get(key, 0.0)
            )

    @staticmethod
    def _cache_insert_ids_not_prefetched(
        cache_insert_ids: torch.Tensor,
        prefetch_ids: torch.Tensor,
    ) -> torch.Tensor:
        if cache_insert_ids.numel() == 0 or prefetch_ids.numel() == 0:
            return cache_insert_ids
        prefetch_set = set(
            int(v)
            for v in prefetch_ids.detach().to(dtype=torch.int64, device="cpu").flatten().tolist()
        )
        values = [
            int(v)
            for v in cache_insert_ids.detach().to(dtype=torch.int64, device="cpu").flatten().tolist()
            if int(v) not in prefetch_set
        ]
        if not values:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor(values, dtype=torch.int64)

    def _insert_planned_cache_rows(
        self,
        row: dict[str, Any],
        cache_insert_ids: torch.Tensor,
    ) -> None:
        row["bagpipe_cache_insert_success_ids"] = 0
        row["bagpipe_cache_insert_failures"] = 0
        if cache_insert_ids.numel() == 0:
            return
        inserter = getattr(self.embedding_module, "prefill_gpu_cache_for_fused_ids", None)
        if not callable(inserter):
            row["bagpipe_cache_insert_failures"] = 1
            row["bagpipe_cache_insert_failure_reason"] = (
                "missing prefill_gpu_cache_for_fused_ids"
            )
            return
        try:
            if bool(inserter(cache_insert_ids)):
                row["bagpipe_cache_insert_success_ids"] = int(cache_insert_ids.numel())
            else:
                row["bagpipe_cache_insert_failures"] = 1
                row["bagpipe_cache_insert_failure_reason"] = (
                    "prefill_gpu_cache_for_fused_ids returned False"
                )
        except Exception as exc:
            row["bagpipe_cache_insert_failures"] = 1
            row["bagpipe_cache_insert_failure_reason"] = f"{type(exc).__name__}: {exc}"

    def _invalidate_evicted_cache_rows(
        self,
        row: dict[str, Any],
        evict_ids: torch.Tensor,
        *,
        prefix: str,
    ) -> None:
        row[f"{prefix}_success_ids"] = 0
        row[f"{prefix}_failures"] = 0
        if evict_ids.numel() == 0:
            return
        invalidator = getattr(self.embedding_module, "invalidate_gpu_cache_for_fused_ids", None)
        if not callable(invalidator):
            row[f"{prefix}_failures"] = 1
            row[f"{prefix}_failure_reason"] = (
                "missing invalidate_gpu_cache_for_fused_ids"
            )
            return
        try:
            if bool(invalidator(evict_ids)):
                row[f"{prefix}_success_ids"] = int(evict_ids.numel())
            else:
                row[f"{prefix}_failures"] = 1
                row[f"{prefix}_failure_reason"] = (
                    "invalidate_gpu_cache_for_fused_ids returned False"
                )
        except Exception as exc:
            row[f"{prefix}_failures"] = 1
            row[f"{prefix}_failure_reason"] = f"{type(exc).__name__}: {exc}"


def attach_or_refetch_with_bagpipe_policy(
    *,
    prefetch_depth: int,
    bagpipe_policy: Any,
    lookahead_prefetcher: Any,
    embedding_module: Any,
    sparse_features: Any,
    row: dict[str, Any],
    step: int,
) -> BagPipeConsumeResult:
    """Attach a valid lookahead handle or repair a stale BagPipe prefetch."""
    if int(prefetch_depth) <= 0:
        row["bagpipe_stale_ids"] = 0
        row["bagpipe_stale_cached_ids"] = 0
        row["bagpipe_stale_refetch_ids"] = 0
        row["bagpipe_valid_prefetch_ids"] = 0
        row["bagpipe_discarded_stale_handle"] = 0
        embedding_module.issue_fused_prefetch(sparse_features)
        return BagPipeConsumeResult(0, 0, 0, 0, 0)

    consume_decision = bagpipe_policy.on_consume(int(step))
    stale_ids = int(consume_decision.stale_ids.numel())
    stale_cached_ids = int(
        getattr(
            consume_decision,
            "stale_cached_ids",
            torch.empty((0,), dtype=torch.int64),
        ).numel()
    )
    stale_refetch_ids = int(
        getattr(
            consume_decision,
            "stale_refetch_ids",
            consume_decision.stale_ids,
        ).numel()
    )
    valid_prefetch_ids = int(consume_decision.valid_prefetch_ids.numel())
    row["bagpipe_stale_ids"] = stale_ids
    row["bagpipe_stale_cached_ids"] = stale_cached_ids
    row["bagpipe_stale_refetch_ids"] = stale_refetch_ids
    row["bagpipe_valid_prefetch_ids"] = valid_prefetch_ids

    can_repair_from_valid_prefetch = valid_prefetch_ids > 0 and stale_ids > 0
    if stale_refetch_ids > 0 and not can_repair_from_valid_prefetch:
        discarded_stale_handle = _bool_int(lookahead_prefetcher.discard_next_ready())
    else:
        discarded_stale_handle = 0
    row["bagpipe_discarded_stale_handle"] = discarded_stale_handle

    if consume_decision.requires_refetch and not can_repair_from_valid_prefetch:
        embedding_module.issue_fused_prefetch(sparse_features)
    else:
        attach_kwargs = {}
        if stale_ids > 0:
            attach_kwargs["invalid_fused_ids"] = consume_decision.stale_ids
        try:
            lookahead_prefetcher.attach_next(**attach_kwargs)
        except TypeError:
            lookahead_prefetcher.attach_next()

    return BagPipeConsumeResult(
        stale_ids=stale_ids,
        stale_cached_ids=stale_cached_ids,
        stale_refetch_ids=stale_refetch_ids,
        valid_prefetch_ids=valid_prefetch_ids,
        discarded_stale_handle=discarded_stale_handle,
    )


def notify_sparse_update(
    *,
    bagpipe_policy: Any,
    sparse_optimizer: Any,
    fallback_updated_ids: torch.Tensor,
    row: dict[str, Any],
    step: int,
) -> BagPipeUpdateResult:
    """Apply sparse optimizer payloads to GPU cache before policy invalidation."""
    updated_ids = fallback_updated_ids.detach().to(dtype=torch.int64, device="cpu")
    cache_updated_chunks: list[torch.Tensor] = []
    failures = 0
    failure_reasons: list[str] = []
    attempted_cache_update_ids = 0
    policy_cached_update_ids = 0
    payloads_fn = getattr(sparse_optimizer, "last_update_payloads", None)
    payloads = payloads_fn() if callable(payloads_fn) else []
    for payload in payloads:
        ids = payload.get("ids")
        grads = payload.get("grads")
        module = payload.get("module")
        name = payload.get("name")
        if not isinstance(ids, torch.Tensor) or not isinstance(grads, torch.Tensor):
            continue
        if not name:
            continue
        cached_for = getattr(bagpipe_policy, "cached_tensor_for", None)
        if callable(cached_for):
            cached_ids = cached_for(ids)
            policy_cached_update_ids += int(cached_ids.numel())
            if cached_ids.numel() == 0:
                continue
            ids_cpu = ids.detach().to(dtype=torch.int64, device="cpu").flatten()
            cached_set = set(int(v) for v in cached_ids.tolist())
            positions = [
                index for index, emb_id in enumerate(ids_cpu.tolist()) if int(emb_id) in cached_set
            ]
            if len(positions) != int(ids_cpu.numel()):
                position_tensor = torch.tensor(
                    positions,
                    dtype=torch.long,
                    device=ids.device,
                )
                ids = ids.index_select(0, position_tensor)
                grads = grads.index_select(0, position_tensor.to(device=grads.device))
        attempted_cache_update_ids += int(ids.numel())
        kv_client = getattr(module, "kv_client", None)
        updater = getattr(kv_client, "apply_sgd_update_gpu_cache", None)
        if not callable(updater):
            failures += 1
            failure_reasons.append("missing apply_sgd_update_gpu_cache")
            continue
        try:
            if updater(
                name,
                ids,
                grads,
                learning_rate=float(payload.get("lr", 0.0)),
            ):
                cache_updated_chunks.append(ids.detach().to(dtype=torch.int64, device="cpu"))
            else:
                repair = getattr(module, "prefill_gpu_cache_for_fused_ids", None)
                if callable(repair) and bool(repair(ids.detach().to(dtype=torch.int64, device="cpu"))):
                    if updater(
                        name,
                        ids,
                        grads,
                        learning_rate=float(payload.get("lr", 0.0)),
                    ):
                        cache_updated_chunks.append(
                            ids.detach().to(dtype=torch.int64, device="cpu")
                        )
                        continue
                failures += 1
                failure_reasons.append("apply_sgd_update_gpu_cache returned False")
        except Exception as exc:
            failures += 1
            failure_reasons.append(f"{type(exc).__name__}: {exc}")

    cache_updated_ids = (
        torch.unique(torch.cat(cache_updated_chunks), sorted=True)
        if cache_updated_chunks
        else torch.empty((0,), dtype=torch.int64)
    )
    row["bagpipe_gpu_cache_update_ids"] = int(cache_updated_ids.numel())
    row["bagpipe_gpu_cache_update_attempt_ids"] = int(attempted_cache_update_ids)
    row["bagpipe_policy_cached_update_ids"] = int(policy_cached_update_ids)
    row["bagpipe_gpu_cache_update_failures"] = int(failures)
    row["bagpipe_gpu_cache_update_failure_reason"] = "; ".join(failure_reasons[:3])
    bagpipe_policy.on_update(
        step=int(step),
        ids=updated_ids,
        cache_updated_ids=cache_updated_ids,
    )
    return BagPipeUpdateResult(
        updated_ids=updated_ids,
        cache_updated_ids=cache_updated_ids,
        gpu_cache_update_failures=int(failures),
    )
