"""Macro optimization layer for RecStore.

This package provides a unified plugin interface for optimization strategies
(Bagpipe, Lookahead, FusedRec, etc.) so that the training loop can invoke
them without scattered if/elif branches.

Public API
----------

.. autosummary::

   OptimizationPlugin
   NullOptimizationPlugin
   OptimizationPluginRegistry
   OptimizationConfig
   KVClientProtocol
   EmbeddingModuleProtocol
   LookaheadPlugin

How to add a new optimization strategy
--------------------------------------

1.  Create a new module (e.g. ``optim/fusedrec.py``) with a class that
    inherits from :class:`OptimizationPlugin`::

        class FusedRecPlugin(OptimizationPlugin):
            @classmethod
            def create(cls, *, embedding_module, kv_client, lookahead, **_):
                ...

            def create_sparse_optimizer(self, modules, lr):
                ...

            def on_prepare(self, sparse_features): ...
            def on_consume(self, sparse_features, device): ...
            def on_step_end(self, step, row): ...
            def consume_stats_into(self, row): ...
            def shutdown(self): ...

2.  Register it in this file::

        OptimizationPluginRegistry.register("fusedrec", FusedRecPlugin.create)

3.  (Optional) Expose a config schema for auto-research::

        def config_schema(self):
            return {"fusion_k": {"type": "int", "range": (1, 64)}}

The training loop can then use ``--optimization-plugin fusedrec`` without
any code changes to the loop itself.
"""

from __future__ import annotations

from .config import OptimizationConfig
from .lookahead import LookaheadPlugin
from .plugin import NullOptimizationPlugin, OptimizationPlugin
from .protocols import EmbeddingModuleProtocol, KVClientProtocol
from .registry import OptimizationPluginRegistry

# ---------------------------------------------------------------------------
# Built-in plugin registration
# ---------------------------------------------------------------------------
# BagPipePlugin is registered lazily to avoid importing bagpipe_cache
# (which pulls in the controller / optimizer) when only the base classes
# are needed.

_BUILTIN_REGISTERED = False


def _register_builtins() -> None:
    global _BUILTIN_REGISTERED
    if _BUILTIN_REGISTERED:
        return
    _BUILTIN_REGISTERED = True

    # Lookahead is lightweight — register eagerly.
    OptimizationPluginRegistry.register("lookahead", LookaheadPlugin.create)

    # BagPipe pulls in controller/optimizer; register on first use.
    try:
        from ..bagpipe_cache.plugin import BagPipePlugin
        OptimizationPluginRegistry.register("bagpipe", BagPipePlugin.create)
    except ImportError:
        pass


_register_builtins()


__all__ = [
    "OptimizationPlugin",
    "NullOptimizationPlugin",
    "OptimizationPluginRegistry",
    "OptimizationConfig",
    "KVClientProtocol",
    "EmbeddingModuleProtocol",
    "LookaheadPlugin",
]
