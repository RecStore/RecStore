"""LookaheadPlugin — adapts :class:`LookaheadPrefetcher` to OptimizationPlugin.

``LookaheadPrefetcher`` has a different interface (``enqueue`` / ``advance`` /
``attach_next``) from :class:`OptimizationPlugin`.  This adapter bridges the
two so that lookahead prefetch can be used through the unified plugin
interface.
"""

from __future__ import annotations

from typing import Any, Dict

from ..embedding_read_path import LookaheadPrefetcher

from .plugin import OptimizationPlugin


class LookaheadPlugin(OptimizationPlugin):
    """Adapt :class:`LookaheadPrefetcher` to the :class:`OptimizationPlugin` interface."""

    def __init__(
        self,
        prefetcher: LookaheadPrefetcher,
        embedding_module: Any,
    ) -> None:
        self._prefetcher = prefetcher
        self._embedding_module = embedding_module

    @classmethod
    def create(
        cls,
        *,
        embedding_module: Any,
        lookahead: int,
        embedding_dim: int,
        **_: Any,
    ) -> "LookaheadPlugin":
        """Factory matching the registry convention.

        Extra kwargs (``kv_client``, ``cache_capacity``, etc.) are accepted
        but ignored — lookahead only needs the embedding module.
        """
        prefetcher = LookaheadPrefetcher(
            embedding_module,
            depth=lookahead,
            embedding_dim=embedding_dim,
        )
        return cls(prefetcher, embedding_module)

    # -- OptimizationPlugin interface -----------------------------------

    def create_sparse_optimizer(self, modules: Any, lr: float) -> Any:
        from ..optimizer import SparseSGD
        return SparseSGD(modules, lr=lr)

    def on_prepare(self, sparse_features: Any) -> None:
        self._prefetcher.enqueue(sparse_features)
        self._prefetcher.advance()

    def on_consume(self, sparse_features: Any, device: Any) -> None:
        self._prefetcher.attach_next()

    def on_step_end(self, step: int, row: Dict[str, Any]) -> None:
        pass

    def consume_stats_into(self, row: Dict[str, Any]) -> None:
        # LookaheadPrefetcher doesn't expose stats directly; surface live
        # id / byte counts as cost-model features.
        row["lookahead_live_ids"] = self._prefetcher.live_ids
        row["lookahead_live_bytes"] = self._prefetcher.live_bytes
        row["lookahead_depth"] = self._prefetcher.depth

    @property
    def lookahead_depth(self) -> int:
        return self._prefetcher.depth * 2

    def shutdown(self) -> None:
        pass

    # -- accessors ------------------------------------------------------

    @property
    def prefetcher(self) -> LookaheadPrefetcher:
        """Direct access to the wrapped prefetcher (for advanced callers)."""
        return self._prefetcher

    def config_schema(self) -> Dict[str, Any]:
        return {
            "lookahead": {"type": "int", "range": (1, 16)},
            "embedding_dim": {"type": "int", "range": (1, 4096)},
        }
