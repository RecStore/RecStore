"""Embedding lookup read strategies for training loops.

``read_mode`` axes (orthogonal to fusion layout on the embedding module):

- ``direct``: synchronous pull inside forward (accuracy baseline).
- ``prefetch``: async get with optional lookahead window ``prefetch_depth``.
  Does **not** wait for in-flight sparse updates; may observe stale values.
- ``bagpipe``: async get that stalls conflicting reads until updates land
  (same accuracy as ``direct``).  Delegates lifecycle to a
  :class:`~recstore.optim.plugin.OptimizationPlugin`.

Fusion on/off only affects which module APIs are used to encode ids; it must
not rewrite ``read_mode`` semantics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol
import torch


@dataclass(frozen=True)
class PrefetchSlot:
    handle: int
    num_ids: int
    issue_ts: float
    fused_ids_cpu: Any
    fused_inverse: Any
    full_batch: bool


@dataclass(frozen=True)
class PreparedTicket:
    """Ticket returned by ``PrefetchReadPath.on_batch_prepared``.

    Carries pre-built unique fused ids so the runner can record id stats
    without a second ``torch.unique`` pass, and ``PrefetchReadPath`` can
    issue a prepared prefetch in ``before_lookup``.
    """

    unique_ids: Any  # torch.Tensor or None
    raw_count: int


class LookaheadPrefetcher:
    """Owns cross-step fused prefetch scheduling for ``read_mode=prefetch``."""

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

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def live_ids(self) -> int:
        return sum(slot.num_ids for slot in self._pending) + sum(
            slot.num_ids for slot in self._ready
        )

    @property
    def live_bytes(self) -> int:
        return int(self.live_ids) * int(self._embedding_dim) * int(self._value_bytes)

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
                full_batch=True,
            )
        )

    def enqueue_fused_ids(self, fused_ids: Any) -> None:
        if self._depth <= 0:
            return
        issue = getattr(self._embedding_module, "issue_fused_id_prefetch", None)
        if not callable(issue):
            raise RuntimeError(
                "fused-id prefetch requires issue_fused_id_prefetch()."
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
                full_batch=False,
            )
        )

    def advance(self) -> bool:
        if self._depth <= 0 or len(self._pending) <= self._depth:
            return False
        self._ready.append(self._pending.popleft())
        return True

    def advance_all(self) -> int:
        moved = 0
        while self._pending:
            self._ready.append(self._pending.popleft())
            moved += 1
        return moved

    def attach_next(self, *, invalid_fused_ids: Any = None) -> bool:
        if self._depth <= 0 or not self._ready:
            return False
        slot = self._ready.popleft()
        self._embedding_module.set_fused_prefetch_handle(
            slot.handle,
            num_ids=slot.num_ids,
            issue_ts=slot.issue_ts,
            fused_ids_cpu=slot.fused_ids_cpu,
            fused_inverse=slot.fused_inverse,
            invalid_fused_ids_cpu=invalid_fused_ids,
            full_batch=slot.full_batch,
        )
        return True

    def discard_next_ready(self) -> bool:
        if self._depth <= 0 or not self._ready:
            return False
        self._ready.popleft()
        return True


def prepare_fused_ids_from_sparse_batch(
    sparse_batch: Any,
    feature_offsets: Any,
) -> tuple[Any, Any, int]:
    """CPU unique fused ids from a dense sparse batch (fusion-EBC optimization)."""
    if sparse_batch.ndim != 2 or sparse_batch.shape[1] != feature_offsets.numel():
        raise ValueError("sparse batch shape does not match feature offsets")
    fused_ids = (
        sparse_batch.to(dtype=torch.int64, device="cpu") + feature_offsets
    ).T.reshape(-1)
    unique_ids, inverse = torch.unique(fused_ids, return_inverse=True)
    return unique_ids, inverse, int(fused_ids.numel())


class EmbeddingReadPath(Protocol):
    @property
    def depth(self) -> int: ...

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        """Prepare-phase hook. Return a ticket consumed by ``before_lookup``."""

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        """Issue/attach async handle before ``embedding_module(...)``."""

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        """Post-update hook (window advance / future bagpipe wait)."""

    def advance_all(self) -> int:
        """Issue reads for all recorded-but-unissued batches (end-of-run drain)."""


class DirectReadPath:
    """Synchronous pull inside forward; no async issue."""

    @property
    def depth(self) -> int:
        return 0

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        del step, sparse_features, sparse_batch, row
        return None

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        del step, sparse_features, ticket, row

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        del step, sparse_features, sparse_optimizer, row

    def advance_all(self) -> int:
        return 0


class PrefetchReadPath:
    """Async embedding read with a trainer-clock lookahead window.

    With ``prefetch_depth > 0`` the read for step ``i + depth`` is issued at
    step ``i`` right after the sparse update; the first batches (which have no
    earlier update hook) are issued at their own lookup as a bootstrap.
    Consumption attaches the handle issued for the current step.

    Overlaps gets with later work and does **not** block on in-flight sparse
    updates, so values may be stale relative to ``direct`` / ``bagpipe``.
    """

    def __init__(
        self,
        embedding_module: Any,
        *,
        prefetch_depth: int,
        feature_offsets: Any | None = None,
    ) -> None:
        if not bool(getattr(embedding_module, "_enable_fusion", False)):
            raise RuntimeError(
                "read_mode=prefetch currently requires a fusion-enabled "
                "embedding module (non-fused async APIs are not wired yet)"
            )
        if not callable(getattr(embedding_module, "issue_fused_prefetch", None)):
            raise RuntimeError(
                "read_mode=prefetch requires embedding_module.issue_fused_prefetch"
            )
        self._module = embedding_module
        self._feature_offsets = feature_offsets
        self._pending_inverse: Any = None
        self._depth = max(0, int(prefetch_depth))
        # step -> sparse_features recorded at prepare time, oldest first
        self._recorded: dict[int, Any] = {}
        # step -> (handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse)
        self._slots: dict[int, Any] = {}

    @property
    def depth(self) -> int:
        return self._depth

    def _ensure_issued(self, through_step: int) -> int:
        """Issue reads for recorded batches up to ``through_step`` (in order)."""
        issued = 0
        for step in sorted(list(self._recorded)):
            if step > through_step:
                break
            self._slots[step] = self._module.issue_fused_prefetch(
                self._recorded.pop(step),
                record_handle=False,
            )
            issued += 1
        return issued

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        if self._depth > 0:
            # Record only; issuing is driven by the trainer step clock so the
            # read for step ``i + depth`` goes out exactly at step ``i``.
            self._recorded[int(step)] = sparse_features
            return None

        del step
        # Same-step async get: deduplicate on the KJT device.  The backend
        # still receives CPU unique IDs, while the inverse stays on GPU for
        # the lookup and pooled-gradient paths.
        prepare_fused_prefetch = getattr(self._module, "prepare_fused_prefetch", None)
        if sparse_features is not None and callable(prepare_fused_prefetch):
            fused_id_start = time.perf_counter()
            ticket = prepare_fused_prefetch(sparse_features)
            row["lookup_ids_build_ms"] = (time.perf_counter() - fused_id_start) * 1e3
            return ticket
        # Keep the old CPU helper as a compatibility fallback for lightweight
        # embedding fakes and legacy modules that only expose issue_prepared.
        if (
            sparse_batch is not None
            and self._feature_offsets is not None
            and callable(getattr(self._module, "issue_prepared_fused_prefetch", None))
        ):
            fused_id_start = time.perf_counter()
            unique_ids, inverse, raw_count = prepare_fused_ids_from_sparse_batch(
                sparse_batch, self._feature_offsets
            )
            row["lookup_ids_build_ms"] = (time.perf_counter() - fused_id_start) * 1e3
            self._pending_inverse = inverse
            return PreparedTicket(unique_ids=unique_ids, raw_count=raw_count)
        return "issue_on_lookup"

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        del row
        if self._depth > 0:
            step = int(step)
            # Bootstrap: only the earliest steps reach here without a prior
            # after_sparse_update having issued their read.
            self._ensure_issued(step)
            slot = self._slots.pop(step, None)
            if slot is None:
                raise RuntimeError(
                    f"prefetch slot missing for step {step}; "
                    "on_batch_prepared was not called for this step"
                )
            handle, num_ids, issue_ts, fused_ids_cpu, fused_inverse = slot
            self._module.set_fused_prefetch_handle(
                handle,
                num_ids=num_ids,
                issue_ts=issue_ts,
                fused_ids_cpu=fused_ids_cpu,
                fused_inverse=fused_inverse,
                full_batch=True,
            )
            return
        del step
        if isinstance(ticket, PreparedTicket):
            self._module.issue_prepared_fused_prefetch(
                ticket.unique_ids, self._pending_inverse, ticket.raw_count
            )
            return
        # ticket = deduplicated(IDs)
        if ticket is not None and ticket != "issue_on_lookup":
            self._module.issue_prepared_fused_prefetch(*ticket)
            return
        self._module.issue_fused_prefetch(sparse_features)

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        del sparse_features, sparse_optimizer, row
        if self._depth > 0:
            self._ensure_issued(int(step) + self._depth)
        # Intentionally no stale repair — that is bagpipe's job.

    def advance_all(self) -> int:
        if self._depth <= 0 or not self._recorded:
            return 0
        return self._ensure_issued(max(self._recorded))


class BagPipeReadPath:
    """BagPipe read path: async gets with update-aware cache.

    Delegates all lifecycle hooks to a :class:`BagPipePlugin` instance.
    The plugin manages its own internal prefetch pipeline, so this read
    path returns ``None`` tickets and reports ``advance_all() == 0``.
    """

    def __init__(self, plugin: Any, *, device: Any) -> None:
        self._plugin = plugin
        self._device = device

    @property
    def depth(self) -> int:
        return self._plugin.lookahead_depth

    @property
    def desired_buffer_size(self) -> int:
        return self._plugin.lookahead_depth * 2

    def on_batch_prepared(
        self,
        step: int,
        sparse_features: Any,
        sparse_batch: Any,
        row: dict[str, Any],
    ) -> Any:
        del step, sparse_batch, row
        # ticket = (unique_ids, inverse, raw_count) 由 controller 的 enqueue
        # 产生 (设备端), 训练循环把它透传给 record_pooled_grad 的 prepared
        # 路径, 免去第二次 unique。
        return self._plugin.on_prepare(sparse_features)

    def before_lookup(
        self,
        step: int,
        sparse_features: Any,
        ticket: Any,
        row: dict[str, Any],
    ) -> None:
        del step, ticket, row
        self._plugin.on_consume(sparse_features, self._device)

    def after_sparse_update(
        self,
        step: int,
        sparse_features: Any,
        sparse_optimizer: Any,
        row: dict[str, Any],
    ) -> None:
        del sparse_features, sparse_optimizer
        self._plugin.on_step_end(step, row)

    def advance_all(self) -> int:
        return 0  # bagpipe manages its own pipeline internally


def build_embedding_read_path(
    read_mode: str,
    *,
    embedding_module: Any,
    prefetch_depth: int = 0,
    embedding_dim: int = 0,
    feature_offsets: Any | None = None,
    plugin: Any | None = None,
    device: Any | None = None,
) -> EmbeddingReadPath:
    mode = str(read_mode).strip().lower()
    if mode == "direct":
        return DirectReadPath()
    if mode == "prefetch":
        return PrefetchReadPath(
            embedding_module,
            prefetch_depth=prefetch_depth,
            feature_offsets=feature_offsets,
        )
    if mode == "bagpipe":
        if plugin is None:
            raise RuntimeError(
                "read_mode=bagpipe requires a plugin instance; "
                "create one via OptimizationPluginRegistry.create('bagpipe', ...)"
            )
        return BagPipeReadPath(plugin, device=device)
    raise RuntimeError(
        f"unsupported read_mode={read_mode!r}; expected direct|prefetch|bagpipe"
    )
