from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefetchSlot:
    handle: int
    num_ids: int
    issue_ts: float
    fused_ids_cpu: Any
    fused_inverse: Any


class LookaheadPrefetcher:
    """Owns lookahead prefetch scheduling and overlap accounting."""

    def __init__(
        self,
        embedding_module: Any,
        depth: int,
        *,
        embedding_dim: int,
        value_bytes: int = 4,
    ) -> None:
        self._embedding_module = embedding_module
        self._depth = max(0, int(depth))
        self._embedding_dim = max(0, int(embedding_dim))
        self._value_bytes = max(1, int(value_bytes))
        self._pending: deque[PrefetchSlot] = deque()
        self._ready: deque[PrefetchSlot] = deque()
        self._stats: dict[str, float] = {}
        self.reset_stats()

    @property
    def depth(self) -> int:
        return self._depth

    def reset_stats(self) -> None:
        self._stats = {
            "prefetch_depth": float(self._depth),
            "prefetch_issued_batches": 0.0,
            "prefetch_consumed_batches": 0.0,
            "prefetch_discarded_batches": 0.0,
            "prefetch_pending_batches": float(len(self._pending)),
            "prefetch_ready_batches": float(len(self._ready)),
            "prefetch_total_ids": 0.0,
            "prefetch_consumed_total_ids": 0.0,
            "prefetch_discarded_total_ids": 0.0,
            "prefetch_issue_to_consume_ms": 0.0,
            "prefetch_window_live_ids": float(self.live_ids),
            "prefetch_window_live_bytes": float(self.live_bytes),
            "prefetch_window_peak_live_ids": float(self.live_ids),
            "prefetch_window_peak_live_bytes": float(self.live_bytes),
        }

    @property
    def live_ids(self) -> int:
        return sum(slot.num_ids for slot in self._pending) + sum(
            slot.num_ids for slot in self._ready
        )

    @property
    def live_bytes(self) -> int:
        return int(self.live_ids) * int(self._embedding_dim) * int(self._value_bytes)

    def _refresh_window_stats(self) -> None:
        live_ids = float(self.live_ids)
        live_bytes = float(self.live_bytes)
        self._stats["prefetch_pending_batches"] = float(len(self._pending))
        self._stats["prefetch_ready_batches"] = float(len(self._ready))
        self._stats["prefetch_window_live_ids"] = live_ids
        self._stats["prefetch_window_live_bytes"] = live_bytes
        self._stats["prefetch_window_peak_live_ids"] = max(
            float(self._stats.get("prefetch_window_peak_live_ids", 0.0)),
            live_ids,
        )
        self._stats["prefetch_window_peak_live_bytes"] = max(
            float(self._stats.get("prefetch_window_peak_live_bytes", 0.0)),
            live_bytes,
        )

    def enqueue(self, sparse_features: Any) -> None:
        if self._depth <= 0:
            return
        result = self._embedding_module.issue_fused_prefetch(
            sparse_features,
            record_handle=False,
        )
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result
        self._pending.append(
            PrefetchSlot(
                handle=int(handle),
                num_ids=int(num_ids),
                issue_ts=float(issue_ts),
                fused_ids_cpu=fused_ids_cpu,
                fused_inverse=fused_inverse,
            )
        )
        self._stats["prefetch_issued_batches"] += 1.0
        self._stats["prefetch_total_ids"] += float(num_ids)
        self._refresh_window_stats()

    def enqueue_fused_ids(self, fused_ids: Any) -> None:
        if self._depth <= 0:
            return
        issue = getattr(self._embedding_module, "issue_fused_id_prefetch", None)
        if not callable(issue):
            raise RuntimeError(
                "BagPipe fused-id prefetch requires issue_fused_id_prefetch()."
            )
        result = issue(fused_ids, record_handle=False)
        handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = result
        self._pending.append(
            PrefetchSlot(
                handle=int(handle),
                num_ids=int(num_ids),
                issue_ts=float(issue_ts),
                fused_ids_cpu=fused_ids_cpu,
                fused_inverse=fused_inverse,
            )
        )
        self._stats["prefetch_issued_batches"] += 1.0
        self._stats["prefetch_total_ids"] += float(num_ids)
        self._refresh_window_stats()

    def advance(self) -> bool:
        if self._depth <= 0 or len(self._pending) <= self._depth:
            self._refresh_window_stats()
            return False
        self._ready.append(self._pending.popleft())
        self._refresh_window_stats()
        return True

    def advance_all(self) -> int:
        moved = 0
        while self._pending:
            self._ready.append(self._pending.popleft())
            moved += 1
        self._refresh_window_stats()
        return moved

    def attach_next(self, *, invalid_fused_ids: Any = None) -> bool:
        if self._depth <= 0 or not self._ready:
            self._refresh_window_stats()
            return False
        slot = self._ready.popleft()
        self._embedding_module.set_fused_prefetch_handle(
            slot.handle,
            num_ids=slot.num_ids,
            issue_ts=slot.issue_ts,
            fused_ids_cpu=slot.fused_ids_cpu,
            fused_inverse=slot.fused_inverse,
            invalid_fused_ids_cpu=invalid_fused_ids,
        )
        self._stats["prefetch_consumed_batches"] += 1.0
        self._stats["prefetch_consumed_total_ids"] += float(slot.num_ids)
        self._stats["prefetch_issue_to_consume_ms"] += max(
            0.0,
            (time.perf_counter() - slot.issue_ts) * 1e3,
        )
        self._refresh_window_stats()
        return True

    def discard_next_ready(self) -> bool:
        if self._depth <= 0 or not self._ready:
            self._refresh_window_stats()
            return False
        slot = self._ready.popleft()
        self._stats["prefetch_discarded_batches"] += 1.0
        self._stats["prefetch_discarded_total_ids"] += float(slot.num_ids)
        self._refresh_window_stats()
        return True

    def consume_stats(
        self,
        *,
        reset: bool = True,
        dense_compute_ms: float = 0.0,
        wait_ms: float = 0.0,
    ) -> dict[str, float]:
        self._refresh_window_stats()
        stats = dict(self._stats)
        issue_to_consume = float(stats.get("prefetch_issue_to_consume_ms", 0.0))
        stats["prefetch_dense_compute_ms"] = float(dense_compute_ms)
        stats["prefetch_network_wait_ms"] = float(wait_ms)
        stats["prefetch_exposed_network_ms"] = max(
            0.0,
            float(wait_ms) - float(dense_compute_ms),
        )
        stats["prefetch_dense_cover_ratio"] = (
            min(1.0, float(dense_compute_ms) / float(wait_ms)) if wait_ms > 0 else 1.0
        )
        stats["prefetch_issue_to_consume_cover_ratio"] = (
            min(1.0, float(dense_compute_ms) / issue_to_consume)
            if issue_to_consume > 0
            else 1.0
        )
        if reset:
            self.reset_stats()
        return stats
