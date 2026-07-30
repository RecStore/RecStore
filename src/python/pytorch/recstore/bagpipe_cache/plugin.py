"""BagPipePlugin — high-level lifecycle wrapper for BagPipe cache.

Encapsulates :class:`BagPipeCacheController` and :class:`BagPipeSparseSGD`
into three lifecycle hooks (``on_prepare``, ``on_consume``, ``on_step_end``)
so that training loops can integrate BagPipe without scattered if-branches.

When disabled, every method is a no-op and ``create_optimizer`` returns a
plain ``SparseSGD``, so the caller code path is identical regardless of
whether BagPipe is active.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import torch

from .controller import BagPipeCacheController
from .optimizer import BagPipeSparseSGD


class BagPipePlugin:
    """High-level BagPipe lifecycle wrapper.

    Usage::

        bagpipe = BagPipePlugin.create(
            enabled=cfg.enable_bagpipe_cache,
            embedding_module=embedding_module,
            kv_client=client,
            lookahead=4,
            cache_capacity=160000,
            embedding_dim=64,
            table_offsets=table_offsets,
            master_table_name="table",
            device=device,
            id_extractor=extractor,
        )
        sparse_optimizer = bagpipe.create_optimizer(
            embedding_module, lr=0.01
        )

        for step in range(steps):
            bagpipe.on_prepare(sparse_features)
            bagpipe.on_consume(sparse_features, device)
            # ... forward, backward ...
            sparse_optimizer.step()
            sparse_optimizer.flush()
            bagpipe.on_step_end(step, row)

        bagpipe.shutdown()
    """

    def __init__(
        self,
        controller: BagPipeCacheController | None,
        *,
        embedding_module: Any,
        lr: float,
    ) -> None:
        self._controller = controller
        self._embedding_module = embedding_module
        self._lr = float(lr)

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        embedding_module: Any,
        kv_client: Any,
        lookahead: int,
        cleanup_batch_proportion: float,
        cache_capacity: int,
        embedding_dim: int,
        fuse_k: int,
        table_offsets: Dict[str, int],
        master_table_name: str,
        device: torch.device,
        lr: float = 0.01,
        id_extractor: Callable[[Any], torch.Tensor],
    ) -> "BagPipePlugin":
        """Create a BagPipePlugin.

        When *enabled* is False, returns a no-op plugin whose methods are
        all safe to call.
        """
        if not enabled:
            return cls(None, embedding_module=embedding_module, lr=lr)

        controller = BagPipeCacheController(
            embedding_module,
            kv_client,
            lookahead_value=lookahead,
            cleanup_batch_proportion=cleanup_batch_proportion,
            cache_capacity=cache_capacity,
            embedding_dim=embedding_dim,
            fuse_k=fuse_k,
            table_offsets=table_offsets,
            master_table_name=master_table_name,
            device=device,
            lr=lr,
            id_extractor=id_extractor,
        )
        return cls(controller, embedding_module=embedding_module, lr=lr)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Whether BagPipe cache is active."""
        return self._controller is not None

    @property
    def lookahead_value(self) -> int:
        """Dynamic lookahead for batch preparation depth.

        Returns 0 when disabled so the caller can fall back to its own
        prefetcher depth.
        """
        if self._controller is None:
            return 0
        return self._controller.lookahead_value

    # ------------------------------------------------------------------
    #  Optimizer
    # ------------------------------------------------------------------

    def create_optimizer(self, modules, lr: float):
        """Return :class:`BagPipeSparseSGD` (enabled) or ``SparseSGD`` (disabled)."""
        if self._controller is not None:
            return BagPipeSparseSGD(modules, lr=lr, controller=self._controller)
        from python.pytorch.recstore.optimizer import SparseSGD
        return SparseSGD(modules, lr=lr)

    # ------------------------------------------------------------------
    #  Lifecycle hooks
    # ------------------------------------------------------------------

    def on_prepare(self, sparse_features: Any) -> None:
        """Batch preparation hook: enqueue + pre-issue PS prefetch.

        Called during ``prepare_next_batch`` — ``lookahead`` steps before
        the batch is consumed, so the PS has time to respond.
        """
        if self._controller is not None:
            self._controller.enqueue(sparse_features)

    def on_consume(
        self,
        sparse_features: Any,
        device: torch.device,
    ) -> None:
        """Consume hook: prefill GPU cache from the pre-issued prefetch.

        Called just before the forward pass.  ``wait_and_get`` is
        near-instant because the PS had ``lookahead`` steps to respond.
        """
        if self._controller is not None:
            self._controller.prefill_cache(sparse_features, device)

    def on_step_end(self, step: int, row: Dict[str, Any]) -> None:
        """Step-end hook: TTL eviction + writeback + stats.

        Called after ``sparse_optimizer.step()`` and ``flush()``.
        """
        if self._controller is not None:
            cleanup_start = time.perf_counter()
            self._controller.cleanup(step)
            row["bagpipe_cleanup_step_ms"] = (
                time.perf_counter() - cleanup_start
            ) * 1e3
        else:
            row.setdefault("bagpipe_gpu_cache_update_ids", 0)
            row.setdefault("bagpipe_gpu_cache_update_attempt_ids", 0)
            row.setdefault("bagpipe_gpu_cache_update_failures", 0)
            row.setdefault("bagpipe_gpu_cache_update_failure_reason", "")

    def consume_stats_into(self, row: Dict[str, Any]) -> None:
        """Merge BagPipe cumulative stats into *row* (does not reset)."""
        if self._controller is not None:
            row.update(self._controller.consume_stats(reset=False))

    def shutdown(self) -> None:
        """Wait for async work and stop the background cleanup thread."""
        if self._controller is not None:
            self._controller.shutdown()
