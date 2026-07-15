from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch


def _ids_to_tuple(ids: torch.Tensor | Iterable[int]) -> tuple[int, ...]:
    if isinstance(ids, torch.Tensor):
        if ids.numel() == 0:
            return ()
        ids_cpu = ids.detach().to(dtype=torch.int64, device="cpu").flatten()
        return tuple(int(item) for item in torch.unique(ids_cpu, sorted=True).tolist())
    return tuple(sorted({int(item) for item in ids}))


def _tensor_from_ids(ids: Iterable[int]) -> torch.Tensor:
    values = list(ids)
    if not values:
        return torch.empty((0,), dtype=torch.int64)
    return torch.tensor(values, dtype=torch.int64)


@dataclass(frozen=True)
class BagPipeStepPlan:
    step: int
    prefetch_ids: torch.Tensor
    cache_insert_ids: torch.Tensor
    evict_ids: torch.Tensor
    sync_now_ids: torch.Tensor
    sync_later_ids: torch.Tensor
    no_sync_ids: torch.Tensor
    ttl_updates: dict[int, int]


@dataclass(frozen=True)
class BagPipeConsumeDecision:
    step: int
    valid_prefetch_ids: torch.Tensor
    stale_ids: torch.Tensor
    stale_cached_ids: torch.Tensor
    stale_refetch_ids: torch.Tensor

    @property
    def requires_refetch(self) -> bool:
        return bool(self.stale_refetch_ids.numel() > 0)


class BagPipeCachePolicy:
    """BagPipe-style oracle TTL and stale-prefetch policy.

    This class is intentionally backend-agnostic. It mirrors the BagPipe oracle
    decisions needed by RecStore: future-use TTLs, cache residency planning, and
    invalidation of prefetched rows when a sparse update lands before consume.
    """

    def __init__(self, lookahead_depth: int, cache_capacity: int) -> None:
        self.lookahead_depth = max(0, int(lookahead_depth))
        self.cache_capacity = max(0, int(cache_capacity))
        self._batches: dict[int, tuple[int, ...]] = {}
        self._cache_ttl: OrderedDict[int, int] = OrderedDict()
        self._planned_prefetch: dict[int, set[int]] = {}
        self._updated_after_plan: dict[int, int] = {}
        self._cache_updated_after_plan: dict[int, int] = {}

    def observe_batch(self, step: int, ids: torch.Tensor | Iterable[int]) -> None:
        self._batches[int(step)] = _ids_to_tuple(ids)

    def cached_ids(self) -> set[int]:
        return set(self._cache_ttl.keys())

    def cached_tensor_for(self, ids: torch.Tensor | Iterable[int]) -> torch.Tensor:
        cached = self.cached_ids()
        return _tensor_from_ids(emb_id for emb_id in _ids_to_tuple(ids) if emb_id in cached)

    def plan_for_step(self, step: int) -> BagPipeStepPlan:
        step = int(step)
        self._evict_expired_before(step)
        current_ids = set(self._batches.get(step, ()))
        next_ids = set(self._batches.get(step + 1, ()))
        future_last_use = self._future_last_use(step)

        cache_before_batch = set(self._cache_ttl.keys())
        prefetch_ids = sorted(current_ids - cache_before_batch)
        cache_insert = {
            emb_id
            for emb_id in current_ids
            if emb_id not in cache_before_batch
            and step < future_last_use.get(emb_id, step) <= step + self.lookahead_depth
        }
        ttl_updates = {
            emb_id: future_last_use[emb_id]
            for emb_id in sorted(current_ids)
            if step < future_last_use.get(emb_id, step) <= step + self.lookahead_depth
        }

        for emb_id in sorted(ttl_updates):
            self._touch_cache(emb_id, ttl_updates[emb_id])

        if self.cache_capacity <= 0 and self._cache_ttl:
            raise RuntimeError(
                "BagPipe cache capacity exceeded: capacity=0 "
                f"required={len(self._cache_ttl)} at step={step}"
            )
        if self.cache_capacity > 0 and len(self._cache_ttl) > self.cache_capacity:
            raise RuntimeError(
                "BagPipe cache capacity exceeded: "
                f"capacity={self.cache_capacity} live={len(self._cache_ttl)} "
                f"at step={step}"
            )
        evict_ids = torch.empty((0,), dtype=torch.int64)
        self._planned_prefetch.setdefault(step, set())

        sync_now = sorted(current_ids & next_ids)
        sync_later_candidates = current_ids - next_ids
        no_sync = [
            emb_id
            for emb_id in sorted(sync_later_candidates)
            if future_last_use.get(emb_id, step) == step
        ]
        no_sync_set = set(no_sync)
        sync_later = sorted(sync_later_candidates - no_sync_set)

        return BagPipeStepPlan(
            step=step,
            prefetch_ids=_tensor_from_ids(prefetch_ids),
            cache_insert_ids=_tensor_from_ids(sorted(cache_insert)),
            evict_ids=evict_ids,
            sync_now_ids=_tensor_from_ids(sync_now),
            sync_later_ids=_tensor_from_ids(sync_later),
            no_sync_ids=_tensor_from_ids(no_sync),
            ttl_updates=ttl_updates,
        )

    def on_update(
        self,
        step: int,
        ids: torch.Tensor | Iterable[int],
        *,
        cache_updated_ids: torch.Tensor | Iterable[int] = (),
    ) -> None:
        cache_updated = set(_ids_to_tuple(cache_updated_ids))
        for emb_id in _ids_to_tuple(ids):
            if emb_id in cache_updated:
                self._cache_updated_after_plan[emb_id] = max(
                    int(step),
                    self._cache_updated_after_plan.get(emb_id, -1),
                )
            else:
                self._updated_after_plan[emb_id] = max(
                    int(step),
                    self._updated_after_plan.get(emb_id, -1),
                )
                self._cache_ttl.pop(emb_id, None)

    def on_prefetch_issued(self, step: int, ids: torch.Tensor | Iterable[int]) -> None:
        issued = set(_ids_to_tuple(ids))
        self._planned_prefetch[int(step)] = issued
        for emb_id in issued:
            self._updated_after_plan.pop(emb_id, None)
            self._cache_updated_after_plan.pop(emb_id, None)

    def on_consume(self, step: int) -> BagPipeConsumeDecision:
        step = int(step)
        planned = set(self._planned_prefetch.get(step, set()))
        stale = {
            emb_id
            for emb_id in planned
            if self._updated_after_plan.get(emb_id, -1) < step
            and self._updated_after_plan.get(emb_id, -1) >= 0
        }
        cached_now = set(self._cache_ttl.keys())
        stale_cached = {
            emb_id
            for emb_id in stale
            if emb_id in cached_now
            and self._cache_updated_after_plan.get(emb_id, -1)
            == self._updated_after_plan.get(emb_id, -2)
        }
        stale_refetch = stale - stale_cached
        valid = sorted(planned - stale)
        return BagPipeConsumeDecision(
            step=step,
            valid_prefetch_ids=_tensor_from_ids(valid),
            stale_ids=_tensor_from_ids(sorted(stale)),
            stale_cached_ids=_tensor_from_ids(sorted(stale_cached)),
            stale_refetch_ids=_tensor_from_ids(sorted(stale_refetch)),
        )

    def on_step_end(self, step: int) -> torch.Tensor:
        return self._evict_expired_through(int(step))

    def _future_last_use(self, step: int) -> dict[int, int]:
        last_use: dict[int, int] = {}
        end = step + self.lookahead_depth
        for future_step in range(step, end + 1):
            for emb_id in self._batches.get(future_step, ()):
                last_use[emb_id] = future_step
        return last_use

    def _touch_cache(self, emb_id: int, ttl: int) -> None:
        if emb_id in self._cache_ttl:
            self._cache_ttl.move_to_end(emb_id)
        self._cache_ttl[emb_id] = int(ttl)

    def _evict_expired_before(self, step: int) -> torch.Tensor:
        expired = [
            emb_id
            for emb_id, ttl in list(self._cache_ttl.items())
            if ttl < int(step)
        ]
        for emb_id in expired:
            self._cache_ttl.pop(emb_id, None)
        return _tensor_from_ids(expired)

    def _evict_expired_through(self, step: int) -> torch.Tensor:
        expired = [
            emb_id
            for emb_id, ttl in list(self._cache_ttl.items())
            if ttl <= int(step)
        ]
        for emb_id in expired:
            self._cache_ttl.pop(emb_id, None)
        return _tensor_from_ids(expired)
