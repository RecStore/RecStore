"""BagPipe-style GPU cache controller for RecStore.

Implements the core BagPipe architecture (SOSP'23) on top of RecStore's
existing GPU cache and prefetch infrastructure:

  * Oracle lookahead: track future access patterns to decide what to cache
    and when to evict (TTL-based eviction).
  * Smart prefetch: only prefetch IDs that are not cached or have expired TTL.
  * sync_now / sync_later: split gradient updates into immediate (IDs that
    will not be reused) vs deferred (IDs that will be reused within the
    lookahead window).  Deferred grads are accumulated and flushed at cleanup.
  * TTL-based eviction with writeback: expired dirty entries are written
    back to the parameter server before eviction.
  * Batched prefetch: accumulate prefetch IDs across ``cleanup_interval``
    batches to reduce RPC count.

The controller is a drop-in replacement for :class:`LookaheadPrefetcher`
when BagPipe mode is enabled.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist

from .prefetch import LookaheadPrefetcher  # noqa: F401  (re-export compat)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """Per-entry metadata for a fused ID currently residing in the GPU cache."""
    ttl: int            # batch number at which this entry expires
    dirty: bool = False # whether it received an SGD update (needs writeback)


@dataclass
class PrefetchSlot:
    """A batched prefetch in flight."""
    handle: int
    ids_cpu: torch.Tensor       # unique fused IDs prefetched
    ttl_map: Dict[int, int]     # fused_id → expiry batch
    issue_ts: float
    num_ids: int


# ---------------------------------------------------------------------------
#  BagPipeCacheController
# ---------------------------------------------------------------------------

class BagPipeCacheController:
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
    ) -> None:
        self.embedding_module = embedding_module
        self.kv_client = kv_client
        self.lookahead_value = max(1, int(lookahead_value))
        self.cleanup_interval = max(1, int(cleanup_batch_proportion * self.lookahead_value))
        self.cache_capacity = int(cache_capacity)
        self.embedding_dim = int(embedding_dim)
        self.fuse_k = int(fuse_k)
        self.table_offsets = dict(table_offsets)
        self.master_table_name = master_table_name
        self.device = device
        self.lr = float(lr)

        # ---- Oracle tracking (sparse, Python-level) ----
        # fused_id → last batch number in which this ID appears
        self.latest_tracker: Dict[int, int] = {}

        # ---- Cache entry tracking ----
        # fused_id → CacheEntry(ttl, dirty)
        self.cache_entries: Dict[int, CacheEntry] = {}

        # ---- sync_now / sync_later ----
        # fused_id → accumulated gradient tensor (on CPU)
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

    # ------------------------------------------------------------------
    #  Public API (compatible with LookaheadPrefetcher where possible)
    # ------------------------------------------------------------------

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
    #  Oracle: fused ID computation
    # ------------------------------------------------------------------

    def _compute_unique_fused_ids(self, sparse_features: Any) -> torch.Tensor:
        """Extract unique fused IDs from a KeyedJaggedTensor sparse batch."""
        from ..data.dlrm_source import convert_kjt_ids_to_fused_ids

        fused_all = convert_kjt_ids_to_fused_ids(sparse_features, self.table_offsets)
        if fused_all.numel() == 0:
            return fused_all
        unique_ids = torch.unique(fused_all)
        return unique_ids

    # ------------------------------------------------------------------
    #  Enqueue (called during batch preparation, ahead of consumption)
    # ------------------------------------------------------------------

    def enqueue(self, sparse_features: Any) -> None:
        """Record a batch's unique IDs into the oracle lookahead buffer.

        Called when a batch is *prepared* (ahead of consumption), mirroring
        ``LookaheadPrefetcher.enqueue``.  Updates ``latest_tracker`` so that
        the controller knows when each ID will next be used.
        """
        unique_ids = self._compute_unique_fused_ids(sparse_features)
        batch_num = self._next_enqueue_batch
        self._next_enqueue_batch += 1
        self._lookahead_ids.append((batch_num, unique_ids))

        # Update latest_tracker: for each ID, record the latest batch it appears in
        id_set = unique_ids.tolist() if unique_ids.numel() > 0 else []
        for fid in id_set:
            self.latest_tracker[fid] = max(
                self.latest_tracker.get(fid, -1), batch_num
            )

    # ------------------------------------------------------------------
    #  Consume (called when a batch is about to be used)
    # ------------------------------------------------------------------

    def prefill_cache(
        self,
        sparse_features: Any,
        compute_device: torch.device,
    ) -> None:
        """Issue smart prefetch + fill GPU cache with TTL tracking.

        Called at consumption time (replaces ``attach_next``).  Only prefetches
        IDs that are not cached or have expired TTL, then fills the GPU cache
        and records TTL for each newly cached entry.
        """
        t_start = time.perf_counter()

        # Pop the oldest entry from the lookahead buffer
        if not self._lookahead_ids:
            return
        batch_num, unique_ids = self._lookahead_ids.popleft()
        self._current_batch = batch_num

        if unique_ids.numel() == 0:
            return

        id_list = unique_ids.tolist()
        id_set = set(id_list)

        # ---- Determine prefetch targets (not cached or expired) ----
        prefetch_targets: list[int] = []
        for fid in id_list:
            entry = self.cache_entries.get(fid)
            if entry is None or entry.ttl < batch_num:
                prefetch_targets.append(fid)
            else:
                self._stats["bagpipe_prefetch_skip_cached"] += 1

        # ---- Compute TTL for all IDs in this batch ----
        ttl_map: Dict[int, int] = {}
        for fid in id_list:
            last_use = self.latest_tracker.get(fid, batch_num)
            ttl_map[fid] = last_use

        # ---- Batched prefetch accumulation ----
        self._batched_prefetch_ids.update(prefetch_targets)
        self._batched_prefetch_ttl.update(
            {fid: ttl_map[fid] for fid in prefetch_targets if fid in ttl_map}
        )
        self._batched_count += 1

        # Issue batched prefetch when cleanup_interval batches accumulated
        if self._batched_count >= self.cleanup_interval:
            self._issue_batched_prefetch()

        # If we have a pending prefetch, consume it and fill the cache
        if self._pending_prefetch is not None:
            self._fill_cache_from_pending(compute_device)

        self._stats["bagpipe_prefill_ms"] += (time.perf_counter() - t_start) * 1e3

    def _issue_batched_prefetch(self) -> None:
        """Issue a single batched prefetch for accumulated IDs."""
        if not self._batched_prefetch_ids:
            self._batched_count = 0
            return

        ids_cpu = torch.tensor(
            sorted(self._batched_prefetch_ids), dtype=torch.int64
        )
        issue_ts = time.perf_counter()

        try:
            handle = self.kv_client.prefetch(ids_cpu)
        except Exception as exc:
            logger.warning("[BagPipe] prefetch issue failed: %s", exc)
            self._batched_prefetch_ids.clear()
            self._batched_prefetch_ttl.clear()
            self._batched_count = 0
            return

        self._pending_prefetch = PrefetchSlot(
            handle=handle,
            ids_cpu=ids_cpu,
            ttl_map=dict(self._batched_prefetch_ttl),
            issue_ts=issue_ts,
            num_ids=int(ids_cpu.numel()),
        )
        self._stats["bagpipe_prefetch_batches"] += 1
        self._stats["bagpipe_prefetch_ids"] += float(ids_cpu.numel())

        self._batched_prefetch_ids.clear()
        self._batched_prefetch_ttl.clear()
        self._batched_count = 0

    def _fill_cache_from_pending(self, compute_device: torch.device) -> None:
        """Wait for the pending prefetch result and fill the GPU cache."""
        slot = self._pending_prefetch
        if slot is None:
            return
        self._pending_prefetch = None

        try:
            values = self.kv_client.wait_and_get(
                slot.handle,
                self.embedding_dim,
                device=compute_device,
            )
        except Exception as exc:
            logger.warning("[BagPipe] prefetch wait failed: %s", exc)
            return

        ids_cuda = slot.ids_cpu.to(device=compute_device, dtype=torch.int64)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        if not values.is_contiguous():
            values = values.contiguous()

        try:
            self.kv_client.prefill_gpu_cache(self.master_table_name, ids_cuda, values)
        except Exception as exc:
            logger.warning("[BagPipe] GPU cache prefill failed: %s", exc)
            return

        # Update cache_entries with TTL
        for fid in slot.ids_cpu.tolist():
            ttl = slot.ttl_map.get(fid, self._current_batch)
            self.cache_entries[fid] = CacheEntry(ttl=ttl, dirty=False)

    # ------------------------------------------------------------------
    #  Cross-GPU gradient synchronization (all_reduce)
    # ------------------------------------------------------------------

    def _is_distributed(self) -> bool:
        """Check if torch.distributed is initialized with > 1 GPU."""
        try:
            return dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    def _get_rank(self) -> int:
        """Get the current process rank (0 if not distributed)."""
        try:
            return dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            return 0

    def _all_reduce_sparse_grads(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce sparse gradients across GPUs via all_gather + aggregate.

        Each GPU may have a different set of (id, grad) pairs.  After this
        call, each GPU has the aggregated grads (summed across all GPUs for
        each unique ID).

        Only rank 0 should push the result to the PS to avoid double-counting.

        Returns (unique_ids, aggregated_grads) on the controller's device.
        """
        if not self._is_distributed():
            return ids, grads  # Single GPU: no all_reduce needed

        t_start = time.perf_counter()
        world_size = dist.get_world_size()
        n = ids.numel()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        # Step 1: Gather N (num IDs) from all GPUs
        n_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, n_tensor)
        max_n = max(int(ni.item()) for ni in n_list)

        if max_n == 0:
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids, grads

        # Step 2: Pad ids and grads to max_n on device
        ids_dev = ids.to(self.device, dtype=torch.int64)
        grads_dev = grads.to(self.device, dtype=torch.float32)
        if not ids_dev.is_contiguous():
            ids_dev = ids_dev.contiguous()
        if not grads_dev.is_contiguous():
            grads_dev = grads_dev.contiguous()

        padded_ids = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        padded_ids[:n] = ids_dev
        padded_grads = torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
        padded_grads[:n] = grads_dev

        # Step 3: All-gather padded tensors
        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                       for _ in range(world_size)]
        dist.all_gather(ids_list, padded_ids)
        dist.all_gather(grads_list, padded_grads)

        # Step 4: Unpad and aggregate (sum grads for same ID across GPUs)
        all_ids_chunks = []
        all_grads_chunks = []
        for r in range(world_size):
            nr = int(n_list[r].item())
            if nr > 0:
                all_ids_chunks.append(ids_list[r][:nr])
                all_grads_chunks.append(grads_list[r][:nr])

        if not all_ids_chunks:
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids, grads

        cat_ids = torch.cat(all_ids_chunks)
        cat_grads = torch.cat(all_grads_chunks)

        # Aggregate by unique ID (sum grads)
        unique_ids, inverse = torch.unique(cat_ids, return_inverse=True)
        aggregated = torch.zeros(
            (len(unique_ids), dim), dtype=cat_grads.dtype, device=cat_grads.device
        )
        aggregated.index_add_(0, inverse, cat_grads)

        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ids"] += float(len(unique_ids))
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

        return unique_ids, aggregated

    # ------------------------------------------------------------------
    #  Gradient update (sync_now / sync_later)
    # ------------------------------------------------------------------

    def update_grads(
        self,
        table_name: str,
        unique_ids: torch.Tensor,
        summed_grads: torch.Tensor,
        lr: float,
        batch_num: int,
    ) -> None:
        """Apply SGD update to GPU cache + sync_now/sync_later split.

        Consistency model:
          1. SGD applied in-place on GPU cache (cache_value -= grad * lr)
          2. sync_now (last use): push grads to PS via emb_update_table
             (PS_value -= grad * lr, invalidates only those keys from cache)
          3. sync_later (will reuse): accumulate grads, defer PS push to eviction
          4. At eviction: emb_update_table flushes accumulated grads to PS

        Since cache was prefetched from PS, both have the same base value.
        After SGD on both, cache_value == PS_value.  Consistent.
        """
        t_start = time.perf_counter()
        if unique_ids.numel() == 0:
            return

        ids_cpu = unique_ids.to("cpu") if unique_ids.is_cuda else unique_ids
        grads_cpu = summed_grads.to("cpu") if summed_grads.is_cuda else summed_grads
        id_list = ids_cpu.tolist()

        # Try in-place SGD update on the GPU cache
        ids_cuda = ids_cpu.to(self.device)
        grads_cuda = grads_cpu.to(self.device)

        try:
            success = self.kv_client.apply_sgd_update_gpu_cache(
                ids_cuda, grads_cuda, lr
            )
        except Exception as exc:
            logger.warning("[BagPipe] apply_sgd_update_gpu_cache raised: %s", exc)
            success = False

        if success:
            self._stats["bagpipe_sgd_cache_success"] += 1

            # Classify sync_now / sync_later
            sync_now_ids: list[int] = []
            sync_later_ids: list[int] = []
            for fid in id_list:
                last_use = self.latest_tracker.get(fid, batch_num)
                if last_use > batch_num:
                    sync_later_ids.append(fid)
                else:
                    sync_now_ids.append(fid)

            # ---- sync_now: push grads to PS immediately ----
            # emb_update_table applies SGD to PS AND invalidates only those
            # specific keys from the GPU cache (NOT a full clear like push).
            if sync_now_ids:
                self._flush_sync_now(sync_now_ids, grads_cpu, id_list)
                self._stats["bagpipe_sync_now_ids"] += float(len(sync_now_ids))

            # ---- sync_later: accumulate grads for deferred flush ----
            for i, fid in enumerate(id_list):
                if fid in sync_later_ids:
                    grad = grads_cpu[i]
                    if fid not in self.sync_later_grads:
                        self.sync_later_grads[fid] = grad.clone()
                    else:
                        self.sync_later_grads[fid] += grad
                    # Mark dirty so eviction knows to flush
                    entry = self.cache_entries.get(fid)
                    if entry is not None:
                        entry.dirty = True
            self._stats["bagpipe_sync_later_ids"] += float(len(sync_later_ids))
        else:
            # Fallback: some keys not in cache.  Apply grads directly to PS.
            # all_reduce first to aggregate across GPUs, then only rank 0 pushes.
            self._stats["bagpipe_sgd_cache_fallback"] += 1
            # Fold in accumulated sync_later grads (same rationale as
            # _flush_sync_now) so deferred grads are not silently dropped.
            fallback_grads_cpu = grads_cpu.clone()
            for j, fid in enumerate(id_list):
                if fid in self.sync_later_grads:
                    fallback_grads_cpu[j] += self.sync_later_grads[fid]
            agg_ids, agg_grads = self._all_reduce_sparse_grads(
                ids_cpu, fallback_grads_cpu)
            if not self._is_distributed() or self._get_rank() == 0:
                try:
                    self.kv_client.update(table_name, agg_ids, agg_grads)
                except Exception as exc:
                    logger.warning("[BagPipe] emb_update_table fallback failed: %s", exc)
            if self._is_distributed() and self._get_rank() != 0:
                try:
                    self.kv_client.invalidate_gpu_cache(ids_cuda)
                except Exception as exc:
                    logger.warning("[BagPipe] invalidate fallback failed: %s", exc)
            # Clear tracking for keys not in cache
            for fid in id_list:
                self.cache_entries.pop(fid, None)
                self.sync_later_grads.pop(fid, None)

        self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3

    def _flush_sync_now(
        self,
        sync_now_ids: list[int],
        grads_cpu: torch.Tensor,
        all_id_list: list[int],
    ) -> None:
        """Push gradients for sync_now IDs to the PS with cross-GPU all_reduce.

        Flow:
          1. all_reduce local sync_now grads across GPUs (sparse all_gather)
          2. Only rank 0 pushes aggregated grads to PS via emb_update_table
             (avoids double-counting: PS_value -= agg_grad * lr, applied once)
          3. All ranks invalidate their own sync_now IDs from local cache
          4. Clear Python tracking

        Uses emb_update_table (NOT push/emb_write) because emb_write clears
        the ENTIRE GPU cache, while emb_update_table only invalidates specific
        keys.
        """
        # Build index mapping: which rows of grads correspond to sync_now IDs
        id_to_idx = {fid: i for i, fid in enumerate(all_id_list)}
        now_indices = torch.tensor(
            [id_to_idx[fid] for fid in sync_now_ids], dtype=torch.long
        )
        now_ids_cpu = torch.tensor(sync_now_ids, dtype=torch.int64)
        now_grads_cpu = grads_cpu.index_select(0, now_indices)

        # Fold in any previously accumulated sync_later grads for these IDs.
        # An ID may have been sync_later in earlier steps (deferred push) and
        # now becomes sync_now (last use).  Without this, the accumulated
        # deferred grads would be silently dropped when sync_later_grads is
        # cleared below, causing a permanent gradient loss on the PS.
        now_grads_cpu = now_grads_cpu.clone()
        for j, fid in enumerate(sync_now_ids):
            if fid in self.sync_later_grads:
                now_grads_cpu[j] += self.sync_later_grads[fid]

        # Step 1: all_reduce across GPUs
        agg_ids, agg_grads = self._all_reduce_sparse_grads(now_ids_cpu, now_grads_cpu)

        # Step 2: Only rank 0 pushes to PS (avoids double-counting)
        if not self._is_distributed() or self._get_rank() == 0:
            try:
                self.kv_client.update(
                    self.master_table_name, agg_ids, agg_grads
                )
            except Exception as exc:
                logger.warning("[BagPipe] emb_update_table sync_now failed: %s", exc)

        # Step 3: Non-rank-0 GPUs invalidate their local cache
        # (rank 0's cache already invalidated by emb_update_table)
        if self._is_distributed() and self._get_rank() != 0:
            now_ids_cuda = now_ids_cpu.to(self.device)
            if not now_ids_cuda.is_contiguous():
                now_ids_cuda = now_ids_cuda.contiguous()
            try:
                self.kv_client.invalidate_gpu_cache(now_ids_cuda)
            except Exception as exc:
                logger.warning("[BagPipe] invalidate_gpu_cache sync_now failed: %s", exc)

        # Step 4: Clear tracking and accumulated sync_later grads
        for fid in sync_now_ids:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)

    # ------------------------------------------------------------------
    #  Cleanup (TTL-based eviction + grad flush)
    # ------------------------------------------------------------------

    def cleanup(self, current_batch: int) -> None:
        """Evict expired cache entries and flush deferred sync_later grads.

        Called at ``cleanup_interval`` boundaries.  For each expired entry:
          1. If it has accumulated sync_later grads, flush them to PS
             (emb_update_table applies SGD to PS + invalidates specific keys)
          2. Invalidate from GPU cache (if not already invalidated by step 1)
          3. Remove from Python tracking

        No writeback of cache values is needed — gradients are the source of
        truth for PS updates, and the cache value mirrors the PS value after
        both have SGD applied.
        """
        t_start = time.perf_counter()

        # Find expired entries
        expired: list[int] = []
        for fid, entry in list(self.cache_entries.items()):
            if entry.ttl < current_batch:
                expired.append(fid)

        if not expired:
            self._stats["bagpipe_cleanup_ms"] += (time.perf_counter() - t_start) * 1e3
            return

        self._evict_entries(expired)
        self._stats["bagpipe_cleanup_ms"] += (time.perf_counter() - t_start) * 1e3

    def _evict_entries(self, expired_ids: list[int]) -> None:
        """Evict expired entries: flush sync_later grads + invalidate from cache."""
        if not expired_ids:
            return

        self._stats["bagpipe_evicted_ids"] += float(len(expired_ids))

        # ---- Flush sync_later grads for expired dirty entries ----
        # Only flush for entries that have accumulated deferred grads.
        # emb_update_table applies grads to PS + invalidates specific keys.
        dirty_expired = [
            fid for fid in expired_ids
            if fid in self.sync_later_grads
        ]
        if dirty_expired:
            self._flush_sync_later(dirty_expired)
            self._stats["bagpipe_writeback_ids"] += float(len(dirty_expired))

        # ---- Invalidate remaining expired entries from GPU cache ----
        # Some may already be invalidated by emb_update_table above.
        # invalidate_gpu_cache on already-removed keys is a no-op.
        remaining = [
            fid for fid in expired_ids
            if fid in self.cache_entries
        ]
        if remaining:
            remaining_cuda = torch.tensor(
                remaining, dtype=torch.int64, device=self.device
            )
            if not remaining_cuda.is_contiguous():
                remaining_cuda = remaining_cuda.contiguous()
            try:
                self.kv_client.invalidate_gpu_cache(remaining_cuda)
            except Exception as exc:
                logger.warning("[BagPipe] invalidate_gpu_cache failed: %s", exc)

        # ---- Remove from tracking ----
        for fid in expired_ids:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)

    def _flush_sync_later(self, ids: list[int]) -> None:
        """Flush accumulated sync_later gradients to PS with cross-GPU all_reduce.

        Flow:
          1. all_reduce accumulated local grads across GPUs (sparse all_gather)
          2. Only rank 0 pushes aggregated grads to PS via emb_update_table
          3. Non-rank-0 GPUs invalidate their local cache for expired IDs

        Uses emb_update_table (NOT push) to avoid clearing the entire GPU cache.
        The accumulated raw gradients are applied to PS as SGD updates.
        """
        sl_ids_cpu = torch.tensor(ids, dtype=torch.int64)
        sl_grads_cpu = torch.stack([
            self.sync_later_grads[fid] for fid in ids
        ])

        # Step 1: all_reduce across GPUs
        agg_ids, agg_grads = self._all_reduce_sparse_grads(sl_ids_cpu, sl_grads_cpu)

        # Step 2: Only rank 0 pushes to PS
        if not self._is_distributed() or self._get_rank() == 0:
            try:
                self.kv_client.update(
                    self.master_table_name, agg_ids, agg_grads
                )
            except Exception as exc:
                logger.warning("[BagPipe] emb_update_table sync_later failed: %s", exc)

        # Step 3: Non-rank-0 GPUs invalidate their local cache
        if self._is_distributed() and self._get_rank() != 0:
            sl_ids_cuda = sl_ids_cpu.to(self.device)
            if not sl_ids_cuda.is_contiguous():
                sl_ids_cuda = sl_ids_cuda.contiguous()
            try:
                self.kv_client.invalidate_gpu_cache(sl_ids_cuda)
            except Exception as exc:
                logger.warning("[BagPipe] invalidate_gpu_cache sync_later failed: %s", exc)

    # ------------------------------------------------------------------
    #  Background cleanup thread
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background thread for deferred cleanup tasks."""
        while True:
            try:
                task = self._cleanup_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            # Currently cleanup is synchronous; this hook exists for future
            # async writeback without blocking the training loop.
            pass

    def shutdown(self) -> None:
        """Signal the background thread to exit."""
        self._cleanup_queue.put(None)


# ---------------------------------------------------------------------------
#  BagPipeSparseSGD — optimizer that routes through the controller
# ---------------------------------------------------------------------------

class BagPipeSparseSGD:
    """Drop-in replacement for SparseSGD that uses BagPipeCacheController.

    Instead of pushing gradients to the PS via ``update_async``/``wait``,
    this optimizer delegates to ``BagPipeCacheController.update_grads()``
    which applies SGD in-place on the GPU cache and splits updates into
    sync_now (immediate) / sync_later (deferred).
    """

    def __init__(self, params, lr: float, controller: "BagPipeCacheController"):
        self.param_groups = [{"params": params, "lr": float(lr)}]
        self.controller = controller
        self._batch_num = 0
        self._last_step_profile: Dict[str, float] = {}
        self._perf_stats: Dict[str, float] = {}
        self.reset_perf_stats()

    # -- perf stats interface (compatible with SparseOptimizer) --
    def reset_perf_stats(self) -> None:
        self._perf_stats = {
            "update_trace_merge_ms": 0.0,
            "update_owner_exchange_ms": 0.0,
            "update_local_apply_ms": 0.0,
            "update_async_enqueue_ms": 0.0,
            "update_flush_wait_ms": 0.0,
        }

    def _perf_add(self, key: str, delta_ms: float) -> None:
        self._perf_stats[key] = self._perf_stats.get(key, 0.0) + float(delta_ms)

    def consume_perf_stats(self, reset: bool = True) -> Dict[str, float]:
        stats = dict(self._perf_stats)
        if reset:
            self.reset_perf_stats()
        return stats

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for mod in group["params"]:
                if hasattr(mod, "reset_trace"):
                    mod.reset_trace()

    def step(self) -> None:
        import time as _time
        from python.pytorch.recstore.optimizer import _collect_traces_by_name

        with torch.no_grad():
            self._last_step_profile = {}
            for group in self.param_groups:
                lr = group["lr"]
                for mod in group["params"]:
                    if not hasattr(mod, "_trace") or not mod._trace:
                        continue

                    t_merge_start = _time.perf_counter()
                    traces_by_name = _collect_traces_by_name(mod)
                    self._perf_add(
                        "update_trace_merge_ms",
                        (_time.perf_counter() - t_merge_start) * 1e3,
                    )

                    for name, entries in traces_by_name.items():
                        if not entries:
                            continue
                        all_ids = torch.cat(
                            [ids for ids, _ in entries], dim=0
                        )
                        all_grads = torch.cat(
                            [grads for _, grads in entries], dim=0
                        )

                        unique_ids, inverse_indices = torch.unique(
                            all_ids, return_inverse=True
                        )
                        summed_grads = torch.zeros(
                            (len(unique_ids), all_grads.size(1)),
                            device=all_grads.device,
                            dtype=all_grads.dtype,
                        )
                        summed_grads.index_add_(0, inverse_indices, all_grads)

                        self.controller.update_grads(
                            name,
                            unique_ids,
                            summed_grads,
                            lr,
                            self._batch_num,
                        )

                    if hasattr(mod, "reset_trace"):
                        mod.reset_trace()

            self._batch_num += 1

    def flush(self) -> None:
        """No-op: BagPipe controller handles writeback at cleanup time."""
        pass
