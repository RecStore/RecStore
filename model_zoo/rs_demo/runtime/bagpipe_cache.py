"""Backward-compatibility shim — re-exports from the standalone package.

The full BagPipe cache implementation now lives in
``src/python/pytorch/recstore/bagpipe_cache/`` as a model-agnostic package.
This shim preserves the original import path
(``from ..runtime.bagpipe_cache import BagPipeCacheController``) so existing
code continues to work without changes.
"""

from python.pytorch.recstore.bagpipe_cache import (  # noqa: F401
    BagPipeCacheController,
    BagPipeCachePolicy,
    BagPipeConsumeDecision,
    BagPipeConsumeResult,
    BagPipeSparseSGD,
    BagPipeStepPlan,
    BagPipeUpdateResult,
    BagPipeWindowScheduler,
    CacheEntry,
    PrefetchSlot,
    attach_or_refetch_with_bagpipe_policy,
    notify_sparse_update,
)

# Re-export LookaheadPrefetcher for backward compat (was re-exported by the
# original monolithic file via ``from .prefetch import LookaheadPrefetcher``).
from .prefetch import LookaheadPrefetcher  # noqa: F401

__all__ = [
    "BagPipeCacheController",
    "BagPipeCachePolicy",
    "BagPipeConsumeDecision",
    "BagPipeConsumeResult",
    "BagPipeSparseSGD",
    "BagPipeStepPlan",
    "BagPipeUpdateResult",
    "BagPipeWindowScheduler",
    "CacheEntry",
    "LookaheadPrefetcher",
    "PrefetchSlot",
    "attach_or_refetch_with_bagpipe_policy",
    "notify_sparse_update",
]
