"""OptimizationPlugin — abstract base class for macro optimization strategies.

All optimization strategies (Bagpipe, Lookahead, FusedRec, etc.) implement
this interface.  The training loop calls through this interface uniformly,
without knowing which strategy is active.

Lifecycle (per training step)::

    on_prepare(sparse_features)          # batch prep: prefetch / enqueue
    on_consume(sparse_features, device)  # consume: prefill cache / attach
    # ... forward, backward, sparse_optimizer.step() ...
    on_step_end(step, row)               # step end: cleanup / stats / writeback

At startup::

    plugin = OptimizationPluginRegistry.create(name, ...)
    sparse_optimizer = plugin.create_sparse_optimizer(modules, lr)

At shutdown::

    plugin.shutdown()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class OptimizationPlugin(ABC):
    """Abstract base class for macro optimization strategy plugins.

    Subclasses must implement all abstract methods.  Optional hooks
    (``lookahead_depth``, ``config_schema``) have default no-op / zero
    implementations.
    """

    # ------------------------------------------------------------------
    #  Factory convention
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, **kwargs: Any) -> "OptimizationPlugin":
        """Factory called by :class:`OptimizationPluginRegistry`.

        Subclasses override this to accept keyword arguments from the
        registry's ``create(**kwargs)`` call.  The default implementation
        passes all kwargs through to ``__init__``.
        """
        return cls(**kwargs)

    # ------------------------------------------------------------------
    #  Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def create_sparse_optimizer(self, modules: Any, lr: float) -> Any:
        """Return the sparse optimizer for this strategy.

        Implementations typically return :class:`~recstore.optimizer.SparseSGD`
        or a subclass thereof (e.g. ``BagPipeSparseSGD``).
        """
        ...

    @abstractmethod
    def on_prepare(self, sparse_features: Any) -> None:
        """Batch preparation hook — prefetch / enqueue future batches.

        Called ``lookahead_depth`` steps before the batch is consumed,
        giving the PS time to respond.
        """
        ...

    @abstractmethod
    def on_consume(self, sparse_features: Any, device: Any) -> None:
        """Consume hook — prefill GPU cache / attach prefetch handle.

        Called just before the forward pass.
        """
        ...

    @abstractmethod
    def on_step_end(self, step: int, row: Dict[str, Any]) -> None:
        """Step-end hook — cleanup, eviction, writeback, stats.

        Called after ``sparse_optimizer.step()`` and ``flush()``.
        """
        ...

    @abstractmethod
    def consume_stats_into(self, row: Dict[str, Any]) -> None:
        """Merge cumulative plugin stats into *row* (does not reset)."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Wait for async work and release resources."""
        ...

    # ------------------------------------------------------------------
    #  Optional hooks
    # ------------------------------------------------------------------

    @property
    def lookahead_depth(self) -> int:
        """Current lookahead depth — guides the prepare-queue depth.

        Returns 0 by default.  Plugins with lookahead override this.
        """
        return 0

    def config_schema(self) -> Dict[str, Any]:
        """Configuration schema for this plugin (for auto-research search spaces).

        Returns an empty dict by default.  Plugins with tunable hyperparameters
        override this to expose their search space.
        """
        return {}


class NullOptimizationPlugin(OptimizationPlugin):
    """No-op plugin — all methods are safe to call but do nothing.

    ``create_sparse_optimizer`` returns a plain :class:`SparseSGD`.
    This is the null-object pattern for the ``"none"`` strategy.
    """

    def create_sparse_optimizer(self, modules: Any, lr: float) -> Any:
        from ..optimizer import SparseSGD
        return SparseSGD(modules, lr=lr)

    def on_prepare(self, sparse_features: Any) -> None:
        pass

    def on_consume(self, sparse_features: Any, device: Any) -> None:
        pass

    def on_step_end(self, step: int, row: Dict[str, Any]) -> None:
        pass

    def consume_stats_into(self, row: Dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        pass
