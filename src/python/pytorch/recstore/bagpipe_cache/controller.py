"""BagPipeCacheController — the orchestrator combining all BagPipe mixins.

Manages GPU cache lifecycle with BagPipe-style TTL eviction and writeback.
This is a drop-in replacement for LookaheadPrefetcher when enable_bagpipe_cache
is set.  All cross-cutting logic (comm, prefetch, eviction, gradient update)
is provided by the corresponding mixins.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set, Tuple

import torch

from .comm import BagPipeCommMixin
from .eviction import BagPipeEvictionMixin
from .grads import BagPipeGradMixin
from .prefetch import BagPipePrefetchMixin
from .types import CacheEntry, PrefetchSlot

logger = logging.getLogger(__name__)


class BagPipeCacheController(
    BagPipeCommMixin,
    BagPipePrefetchMixin,
    BagPipeEvictionMixin,
    BagPipeGradMixin,
):
    """Manages GPU cache lifecycle with BagPipe-style TTL eviction and writeback.

    Replaces :class:`LookaheadPrefetcher` when ``enable_bagpipe_cache`` is set.
    Works with the local_shm fast path + GPU cache enabled.
    """

    def __init__(
        self,
        embedding_module: Any,
        kv_client: Any,
        *,
        lookahead_value: int,
        cleanup_batch_proportion: float,
        cache_capacity: int,
        embedding_dim: int,
        fuse_k: int,
        table_offsets: Dict[str, int],
        master_table_name: str,
        device: torch.device,
        lr: float = 0.01,
        id_extractor: Callable[[Any], torch.Tensor],
    ) -> None:
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self._base_lookahead = max(1, int(lookahead_value))
        self._dynamic_lookahead = self._base_lookahead
        self._max_lookahead = max(self._base_lookahead, 16)
        self._additive_increase = 1
        self._multiplicative_decrease = 0.5
        self._pressure_low = 0.70
        self._pressure_high = 0.90
        self._lookahead_adjust_interval = 10
        self.cleanup_batch_proportion = float(cleanup_batch_proportion)
        self.cleanup_interval = max(1, int(self.cleanup_batch_proportion * self._dynamic_lookahead))
        self.cache_capacity = int(cache_capacity)
        self.embedding_dim = int(embedding_dim)
        self.fuse_k = int(fuse_k)
        self.table_offsets = dict(table_offsets)
        self.master_table_name = master_table_name
        self.device = device
        self.lr = float(lr)

        # Model-agnostic ID extractor (caller must provide)
        self._id_extractor = id_extractor

        # ---- Oracle tracking (sparse, Python-level) ----
        self.latest_tracker: Dict[int, int] = {}
        self.cache_entries: Dict[int, CacheEntry] = {}
        self.sync_later_grads: Dict[int, torch.Tensor] = {}

        # ---- Lookahead buffer (future batch unique ID sets) ----
        self._lookahead_ids: Deque[Tuple[int, torch.Tensor]] = deque()
        self._next_enqueue_batch = 0
        self._current_batch = 0

        # ---- Prefetch batching ----
        self._batched_prefetch_ids: Set[int] = set()
        self._batched_prefetch_ttl: Dict[int, int] = {}
        self._batched_count = 0
        self._pending_prefetch: Optional[PrefetchSlot] = None

        # ---- Stats ----
        self._stats: Dict[str, float] = {}
        self.reset_stats()

        # ---- Background cleanup thread ----
        self._cleanup_queue: queue.Queue = queue.Queue()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="bagpipe-cleanup"
        )
        self._cleanup_thread.start()

        # ---- Overlap state: async all_reduce on separate CUDA streams ----
        self._sync_later_stream: Optional[torch.cuda.Stream] = None
        self._eviction_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._sync_later_stream = torch.cuda.Stream(device=self.device)
            self._eviction_stream = torch.cuda.Stream(device=self.device)
        self._sync_later_future = None
        self._sync_later_ids: Optional[torch.Tensor] = None
        self._sync_later_grads_buf: Optional[torch.Tensor] = None
        self._first_update = True
        self._pending_sync_now_work = None
        self._pending_sync_now_ids = None

        # ---- no_sync: shared vs local-only IDs ----
        self._shared_ids: Optional[Set[int]] = None
        self._global_id_to_index: Optional[Dict[int, int]] = None
        self._global_unique_count: int = 0
        self._init_unique_ids: Set[int] = set()
        self._init_batches_seen: int = 0
        self._prescan_done: bool = False
        self._prescan_unique_ids: Set[int] = set()
        self._stats["bagpipe_no_sync_ids"] = 0.0
        self._stats["bagpipe_shared_ids"] = 0.0

    # ------------------------------------------------------------------
    #  Public API (compatible with LookaheadPrefetcher where possible)
    # ------------------------------------------------------------------

    @property
    def lookahead_value(self) -> int:
        """Dynamic lookahead, adjusted based on cache pressure (opt 9)."""
        return self._dynamic_lookahead

    @property
    def depth(self) -> int:
        return self.lookahead_value

    def reset_stats(self) -> None:
        self._stats = {
            "bagpipe_lookahead": float(self.lookahead_value),
            "bagpipe_cleanup_interval": float(self.cleanup_interval),
            "bagpipe_cache_entries": 0.0,
            "bagpipe_dirty_entries": 0.0,
            "bagpipe_prefetch_batches": 0.0,
            "bagpipe_prefetch_ids": 0.0,
            "bagpipe_prefetch_skip_cached": 0.0,
            "bagpipe_prefetch_pruned": 0.0,
            "bagpipe_prefetch_local_nosync_kept": 0.0,
            "bagpipe_sync_now_overlap_ms": 0.0,
            "bagpipe_sync_now_ids": 0.0,
            "bagpipe_sync_later_ids": 0.0,
            # __init__ seeds these two separately; omitting them here made
            # reset_stats() (called every step via consume_stats) drop the
            # keys and the next update_grads raise KeyError.
            "bagpipe_no_sync_ids": 0.0,
            "bagpipe_shared_ids": 0.0,
            "bagpipe_evicted_ids": 0.0,
            "bagpipe_writeback_ids": 0.0,
            "bagpipe_sgd_cache_success": 0.0,
            "bagpipe_sgd_cache_fallback": 0.0,
            "bagpipe_all_reduce_calls": 0.0,
            "bagpipe_all_reduce_ids": 0.0,
            "bagpipe_all_reduce_ms": 0.0,
            "bagpipe_prefill_ms": 0.0,
            "bagpipe_update_ms": 0.0,
            "bagpipe_cleanup_ms": 0.0,
            "bagpipe_eviction_stream_ms": 0.0,
            "bagpipe_eviction_stream_overlap_ms": 0.0,
            "bagpipe_prescan_batches": 0.0,
            "bagpipe_prescan_ids": 0.0,
            "bagpipe_dynamic_lookahead": 0.0,
            "bagpipe_cache_pressure": 0.0,
        }

    def consume_stats(self, *, reset: bool = True) -> Dict[str, float]:
        self._stats["bagpipe_cache_entries"] = float(len(self.cache_entries))
        self._stats["bagpipe_dirty_entries"] = float(
            sum(1 for e in self.cache_entries.values() if e.dirty)
        )
        stats = dict(self._stats)
        if reset:
            self.reset_stats()
        return stats
