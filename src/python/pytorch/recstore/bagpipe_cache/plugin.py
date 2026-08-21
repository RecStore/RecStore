"""BagPipePlugin — high-level lifecycle wrapper for BagPipe cache.

Encapsulates :class:`BagPipeCacheController` and :class:`BagPipeSparseSGD`
into three lifecycle hooks (``on_prepare``, ``on_consume``, ``on_step_end``)
so that training loops can integrate BagPipe without scattered if-branches.

Inherits from :class:`~recstore.optim.plugin.OptimizationPlugin` so it can
be registered with :class:`~recstore.optim.registry.OptimizationPluginRegistry`
and created uniformly via ``OptimizationPluginRegistry.create("bagpipe", ...)``.

When disabled (controller is None), every method is a no-op and
``create_optimizer`` returns a plain ``SparseSGD``, so the caller code path
is identical regardless of whether BagPipe is active.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

import torch

from ..optim.plugin import OptimizationPlugin

from .controller import BagPipeCacheController
from .optimizer import BagPipeSparseSGD


class BagPipePlugin(OptimizationPlugin):
    """High-level BagPipe lifecycle wrapper.

    Usage::

        # Via registry (preferred):
        from recstore.optim import OptimizationPluginRegistry
        bagpipe = OptimizationPluginRegistry.create(
            "bagpipe",
            embedding_module=embedding_module,
            kv_client=client,
            lookahead=4,
            cleanup_proportion=0.25,
            cache_capacity=160000,
            embedding_dim=64,
            fuse_k=30,
            table_offsets=table_offsets,
            master_table_name="table",
            device=device,
            lr=0.01,
            id_extractor=extractor,
        )
        sparse_optimizer = bagpipe.create_sparse_optimizer(
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
        embedding_module: Any,
        kv_client: Any,
        lookahead: int = 0,
        cleanup_proportion: float | None = None,
        cleanup_batch_proportion: float | None = None,
        cache_capacity: int = 0,
        embedding_dim: int = 128,
        fuse_k: int = 30,
        table_offsets: Dict[str, int] | None = None,
        table_sizes: Dict[str, int] | None = None,
        master_table_name: str = "",
        device: torch.device | None = None,
        lr: float = 0.01,
        id_extractor: Callable[[Any], torch.Tensor] | None = None,
        enabled: bool = True,
        **_: Any,
    ) -> "BagPipePlugin":
        """Create a BagPipePlugin.

        When *enabled* is False, returns a no-op plugin whose methods are
        all safe to call.

        Accepts ``**_`` to swallow extra kwargs from the registry (e.g.
        ``plugin_config``) that BagPipe does not use.

        ``cleanup_proportion`` (from :class:`OptimizationConfig`) and
        ``cleanup_batch_proportion`` (legacy name) are aliases; if both are
        given, ``cleanup_proportion`` wins.
        """
        if not enabled:
            return cls(None, embedding_module=embedding_module, lr=lr)

        # Resolve cleanup proportion from either alias.
        if cleanup_proportion is not None:
            cleanup_batch = float(cleanup_proportion)
        elif cleanup_batch_proportion is not None:
            cleanup_batch = float(cleanup_batch_proportion)
        else:
            cleanup_batch = 0.25

        controller = BagPipeCacheController(
            embedding_module,
            kv_client,
            lookahead_value=lookahead,
            cleanup_batch_proportion=cleanup_batch,
            cache_capacity=cache_capacity,
            embedding_dim=embedding_dim,
            fuse_k=fuse_k,
            table_offsets=table_offsets or {},
            table_sizes=table_sizes,
            master_table_name=master_table_name,
            device=device if device is not None else torch.device("cpu"),
            lr=lr,
            id_extractor=id_extractor if id_extractor is not None else (lambda sf: sf),
        )
        return cls(controller, embedding_module=embedding_module, lr=lr)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Whether BagPipe cache is active."""
        return self._controller is not None

    # ------------------------------------------------------------------
    #  Optimizer
    # ------------------------------------------------------------------

    def create_sparse_optimizer(self, modules, lr: float):
        """Return :class:`BagPipeSparseSGD` (enabled) or ``SparseSGD`` (disabled).

        Implements the :meth:`OptimizationPlugin.create_sparse_optimizer`
        abstract method.
        """
        if self._controller is not None:
            return BagPipeSparseSGD(modules, lr=lr, controller=self._controller)
        from ..optimizer import SparseSGD
        return SparseSGD(modules, lr=lr)

    # Backwards-compatible alias.
    create_optimizer = create_sparse_optimizer

    # ------------------------------------------------------------------
    #  Lifecycle hooks
    # ------------------------------------------------------------------

    def on_prepare(self, sparse_features: Any):
        """Batch preparation hook: enqueue + pre-issue PS prefetch.

        Called during ``prepare_next_batch`` — ``lookahead`` steps before
        the batch is consumed, so the PS has time to respond.  Returns the
        controller's ``(unique_ids, inverse, raw_count)`` ticket so the
        training loop can reuse it as prepared-ids metadata (skips a second
        unique pass in the pooled-gradient path), or ``None``.
        """
        if self._controller is not None:
            return self._controller.enqueue(sparse_features)
        return None

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
            row.update(self._controller.consume_stats(reset=True))
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

    # ------------------------------------------------------------------
    #  OptimizationPlugin optional hooks
    # ------------------------------------------------------------------

    @property
    def lookahead_depth(self) -> int:
        """Dynamic lookahead for batch preparation depth.

        Returns 0 when disabled so the caller can fall back to its own
        prefetcher depth.
        """
        if self._controller is None:
            return 0
        return self._controller.lookahead_value

    def config_schema(self) -> Dict[str, Any]:
        return {
            "lookahead": {"type": "int", "range": (1, 16)},
            "cleanup_proportion": {"type": "float", "range": (0.05, 1.0)},
            "cache_capacity": {"type": "int", "range": (10000, 1000000)},
        }
