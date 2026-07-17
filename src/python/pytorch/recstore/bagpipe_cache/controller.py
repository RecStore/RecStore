"""BagPipeCacheController — the orchestrator combining all BagPipe mixins.

Manages GPU cache lifecycle with BagPipe-style TTL eviction and writeback.
This is a drop-in replacement for LookaheadPrefetcher when enable_bagpipe_cache
is set.  All cross-cutting logic (comm, prefetch, eviction) is provided by
the corresponding mixins.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist

from .comm import BagPipeCommMixin
from .eviction import BagPipeEvictionMixin
from .prefetch import BagPipePrefetchMixin
from .types import CacheEntry, PrefetchSlot

logger = logging.getLogger(__name__)


def _default_kjt_id_extractor(table_offsets: Dict[str, int]) -> Callable[[Any], torch.Tensor]:
    """Create the default fused-ID extractor for KeyedJaggedTensor batches."""
    from ..data.dlrm_source import convert_kjt_ids_to_fused_ids  # type: ignore

    def extract(sparse_features: Any) -> torch.Tensor:
        return convert_kjt_ids_to_fused_ids(sparse_features, table_offsets)

    return extract


class BagPipeCacheController(
    BagPipeCommMixin, BagPipePrefetchMixin, BagPipeEvictionMixin
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
        id_extractor: Optional[Callable[[Any], torch.Tensor]] = None,
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

        # Model-agnostic ID extractor (opt: decouple from model_zoo)
        self._id_extractor = id_extractor or _default_kjt_id_extractor(table_offsets)

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

    # ------------------------------------------------------------------
    #  Overlap barriers (sync_later + sync_now)
    # ------------------------------------------------------------------

    def _wait_prev_sync_later(self) -> None:
        """Wait for the previous iteration's async sync_later all_reduce."""
        if self._first_update or self._sync_later_future is None:
            self._first_update = False
            return

        t_start = time.perf_counter()
        if self._sync_later_stream is not None:
            self._sync_later_stream.synchronize()
        self._sync_later_future.wait()
        sl_ids, sl_grads = self._sync_later_future.result
        if sl_ids is None:
            sl_ids = self._sync_later_ids
            sl_grads = self._sync_later_grads_buf

        if sl_ids is not None and sl_ids.numel() > 0:
            if not self._is_distributed() or self._get_rank() == 0:
                try:
                    self.kv_client.update(self.master_table_name, sl_ids, sl_grads)
                except Exception as exc:
                    logger.warning("[BagPipe] sync_later deferred push failed: %s", exc)
            if self._is_distributed() and self._get_rank() != 0:
                try:
                    self.kv_client.invalidate_gpu_cache(self.master_table_name, sl_ids)
                except Exception as exc:
                    logger.warning("[BagPipe] sync_later deferred invalidate failed: %s", exc)

        self._sync_later_future = None
        self._sync_later_ids = None
        self._sync_later_grads_buf = None
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

    def _wait_pending_sync_now(self) -> None:
        """Wait for the previous step's sync_now all_reduce and push to PS (opt 11)."""
        work = self._pending_sync_now_work
        if work is None:
            return
        self._pending_sync_now_work = None
        now_ids_list = self._pending_sync_now_ids or []
        self._pending_sync_now_ids = None

        t_start = time.perf_counter()
        if work is not None:
            work.wait()
            agg_ids, agg_grads = work.result
        else:
            return
        if not self._is_distributed() or self._get_rank() == 0:
            try:
                self.kv_client.update(self.master_table_name, agg_ids, agg_grads)
            except Exception as exc:
                logger.warning("[BagPipe] sync_now deferred push failed: %s", exc)
        if self._is_distributed() and self._get_rank() != 0:
            try:
                self.kv_client.invalidate_gpu_cache(self.master_table_name, agg_ids)
            except Exception as exc:
                logger.warning("[BagPipe] sync_now deferred invalidate failed: %s", exc)
        for fid in now_ids_list:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)
        self._stats["bagpipe_sync_now_overlap_ms"] += (time.perf_counter() - t_start) * 1e3

    # ------------------------------------------------------------------
    #  Gradient update (sync_now / sync_later / no_sync split)
    # ------------------------------------------------------------------

    def update_grads(
        self,
        table_name: str,
        unique_ids: torch.Tensor,
        summed_grads: torch.Tensor,
        lr: float,
        batch_num: int,
    ) -> None:
        """Apply SGD update to GPU cache + sync_now/sync_later split."""
        t_start = time.perf_counter()

        self._wait_prev_sync_later()

        if unique_ids.numel() == 0:
            self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
            return

        ids_cuda = unique_ids.to(self.device, dtype=torch.int64)
        grads_cuda = summed_grads.to(self.device, dtype=torch.float32)
        if grads_cuda.dim() == 1:
            grads_cuda = grads_cuda.unsqueeze(1)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        if not grads_cuda.is_contiguous():
            grads_cuda = grads_cuda.contiguous()

        try:
            success = self.kv_client.apply_sgd_update_gpu_cache(
                table_name, ids_cuda, grads_cuda, learning_rate=lr
            )
        except Exception as exc:
            logger.warning("[BagPipe] apply_sgd_update_gpu_cache raised: %s", exc)
            success = False

        if not success:
            self._stats["bagpipe_sgd_cache_fallback"] += 1
            _, _, work = self._dense_all_reduce_async(ids_cuda, grads_cuda)
            if work is not None:
                work.wait()
                agg_ids, agg_grads = work.result
            else:
                agg_ids, agg_grads = ids_cuda, grads_cuda
            if not self._is_distributed() or self._get_rank() == 0:
                try:
                    self.kv_client.update(table_name, agg_ids, agg_grads)
                except Exception as exc:
                    logger.warning("[BagPipe] fallback push failed: %s", exc)
            if self._is_distributed() and self._get_rank() != 0:
                try:
                    self.kv_client.invalidate_gpu_cache(table_name, ids_cuda)
                except Exception:
                    pass
            for fid in ids_cuda.tolist():
                self.cache_entries.pop(fid, None)
                self.sync_later_grads.pop(fid, None)
            self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
            return

        self._stats["bagpipe_sgd_cache_success"] += 1

        self._maybe_build_shared_id_set(ids_cuda)

        id_list = ids_cuda.tolist()
        shared = self._shared_ids or set()

        no_sync_ids: list[int] = []
        no_sync_grads_indices: list[int] = []
        sync_now_ids: list[int] = []
        sync_now_grads_indices: list[int] = []
        sync_later_ids: list[int] = []
        sync_later_grads_indices: list[int] = []
        for j, fid in enumerate(id_list):
            if fid not in shared:
                no_sync_ids.append(fid)
                no_sync_grads_indices.append(j)
            elif self.latest_tracker.get(fid, batch_num) <= batch_num:
                sync_now_ids.append(fid)
                sync_now_grads_indices.append(j)
            else:
                sync_later_ids.append(fid)
                sync_later_grads_indices.append(j)

        no_sync_count = len(no_sync_ids)
        sync_now_count = len(sync_now_ids)
        sync_later_count = len(sync_later_ids)
        self._stats["bagpipe_sync_now_ids"] += float(sync_now_count)
        self._stats["bagpipe_sync_later_ids"] += float(sync_later_count)
        self._stats["bagpipe_no_sync_ids"] += float(no_sync_count)

        # ---- no_sync: local-only IDs ----
        if no_sync_count > 0:
            if self._shared_ids is None:
                ns_indices = torch.tensor(no_sync_grads_indices, dtype=torch.long,
                                           device=self.device)
                ns_ids = ids_cuda[ns_indices].contiguous()
                ns_grads = grads_cuda[ns_indices].contiguous()
                for j, fid in enumerate(no_sync_ids):
                    if fid in self.sync_later_grads:
                        ns_grads[j] += self.sync_later_grads[fid].to(self.device)
                try:
                    self.kv_client.update(self.master_table_name, ns_ids, ns_grads)
                except Exception as exc:
                    logger.warning("[BagPipe] no_sync push failed: %s", exc)
                for fid in no_sync_ids:
                    self.cache_entries.pop(fid, None)
                    self.sync_later_grads.pop(fid, None)
            else:
                for fid in no_sync_ids:
                    self.sync_later_grads.pop(fid, None)
                    entry = self.cache_entries.get(fid)
                    if entry is not None:
                        entry.dirty = True

        # ---- sync_now: dense async all_reduce, deferred wait (opt 11) ----
        if sync_now_count > 0:
            now_indices = torch.tensor(sync_now_grads_indices, dtype=torch.long,
                                        device=self.device)
            now_ids = ids_cuda[now_indices].contiguous()
            now_grads = grads_cuda[now_indices].contiguous()
            if self.sync_later_grads:
                now_ids_list = now_ids.tolist()
                for j, fid in enumerate(now_ids_list):
                    if fid in self.sync_later_grads:
                        now_grads[j] += self.sync_later_grads[fid].to(self.device)
            _, _, work = self._dense_all_reduce_async(now_ids, now_grads)
            self._pending_sync_now_work = work
            self._pending_sync_now_ids = now_ids.tolist()
        else:
            self._pending_sync_now_work = None
            self._pending_sync_now_ids = None

        # ---- sync_later: launch async all_reduce on dedicated stream ----
        if sync_later_count > 0:
            later_indices = torch.tensor(sync_later_grads_indices, dtype=torch.long,
                                          device=self.device)
            later_ids = ids_cuda[later_indices].contiguous()
            later_grads = grads_cuda[later_indices].clone().contiguous()
            later_ids_list = later_ids.tolist()
            for j, fid in enumerate(later_ids_list):
                if fid in self.sync_later_grads:
                    later_grads[j] += self.sync_later_grads[fid].to(self.device)
            _, grads_buf, work = self._dense_all_reduce_async(
                later_ids, later_grads, stream=self._sync_later_stream
            )
            self._sync_later_future = work
            self._sync_later_ids = later_ids
            self._sync_later_grads_buf = grads_buf
            for fid in later_ids_list:
                entry = self.cache_entries.get(fid)
                if entry is not None:
                    entry.dirty = True

        self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3

    def _flush_sync_now(
        self,
        sync_now_ids: list[int],
        grads_cpu: torch.Tensor,
        all_id_list: list[int],
    ) -> None:
        """Legacy flush path — thin wrapper around dense all_reduce."""
        id_to_idx = {fid: i for i, fid in enumerate(all_id_list)}
        now_indices = torch.tensor(
            [id_to_idx[fid] for fid in sync_now_ids], dtype=torch.long
        )
        now_ids_cpu = torch.tensor(sync_now_ids, dtype=torch.int64)
        now_grads_cpu = grads_cpu.index_select(0, now_indices).clone()
        for j, fid in enumerate(sync_now_ids):
            if fid in self.sync_later_grads:
                now_grads_cpu[j] += self.sync_later_grads[fid]

        _, _, work = self._dense_all_reduce_async(now_ids_cpu, now_grads_cpu)
        if work is not None:
            work.wait()
            agg_ids, agg_grads = work.result
        else:
            agg_ids, agg_grads = now_ids_cpu, now_grads_cpu
        if not self._is_distributed() or self._get_rank() == 0:
            try:
                self.kv_client.update(self.master_table_name, agg_ids, agg_grads)
            except Exception as exc:
                logger.warning("[BagPipe] sync_now push failed: %s", exc)
        if self._is_distributed() and self._get_rank() != 0:
            try:
                self.kv_client.invalidate_gpu_cache(self.master_table_name, agg_ids.to(self.device))
            except Exception as exc:
                logger.warning("[BagPipe] sync_now invalidate failed: %s", exc)
        for fid in sync_now_ids:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)
