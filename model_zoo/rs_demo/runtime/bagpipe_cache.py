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

        # ---- Overlap state: async all_reduce on a separate CUDA stream ----
        # Mirrors the original BagPipe bagcache.py design: sync_now is
        # all_reduced immediately (async), sync_later is all_reduced on a
        # dedicated stream and overlapped with the next iteration's
        # forward+backward.  The result is waited at the start of the
        # *following* update_grads call.
        self._sync_later_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._sync_later_stream = torch.cuda.Stream(device=self.device)
        # Pending sync_later future (dist.Work) + its IDs/grads for rank-0 push
        self._sync_later_future = None
        self._sync_later_ids: Optional[torch.Tensor] = None
        self._sync_later_grads_buf: Optional[torch.Tensor] = None
        # Whether this is the first step (skip waiting for prior sync_later)
        self._first_update = True

        # ---- no_sync: local-only IDs that don't need cross-GPU all_reduce ----
        # Set of fused_ids that appear on >1 rank (shared IDs needing all_reduce)
        self._shared_ids: Optional[Set[int]] = None
        # Global ID → dense index mapping (for dense all_reduce optimization)
        self._global_id_to_index: Optional[Dict[int, int]] = None
        self._global_unique_count: int = 0
        # Accumulate unique IDs during first lookahead window for one-time all_gather
        self._init_unique_ids: Set[int] = set()
        self._init_batches_seen: int = 0
        # Stats
        self._stats["bagpipe_no_sync_ids"] = 0.0
        self._stats["bagpipe_shared_ids"] = 0.0

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
            "bagpipe_prefetch_pruned": 0.0,
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
        """Issue a single batched prefetch for accumulated IDs.

        Prefetch pruning: skip IDs that are no_sync (local-only, not shared
        across ranks).  These IDs don't need cross-GPU sync, so prefetching
        them for sync purposes is wasteful.  They are still fetched on-demand
        during lookup if not in cache.
        """
        if not self._batched_prefetch_ids:
            self._batched_count = 0
            return

        # Prefetch pruning: remove no_sync (local-only) IDs from prefetch list
        # if the shared ID set has been built.  In the original BagPipe, this
        # is done via others_no_sync pruning in fill_prefetch_cache.
        shared = self._shared_ids
        if shared is not None:
            pruned_ids = set()
            pruned_ttl = {}
            skipped = 0
            for fid in self._batched_prefetch_ids:
                if fid in shared:
                    pruned_ids.add(fid)
                    pruned_ttl[fid] = self._batched_prefetch_ttl.get(fid, 0)
                else:
                    skipped += 1
            self._stats["bagpipe_prefetch_pruned"] = self._stats.get("bagpipe_prefetch_pruned", 0.0) + float(skipped)
            self._batched_prefetch_ids = pruned_ids
            self._batched_prefetch_ttl = pruned_ttl

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


    def _load_balanced_push(
        self, table_name: str, ids: torch.Tensor, grads: torch.Tensor
    ) -> None:
        """Push aggregated gradients to the PS, load-balanced across ranks.

        Each rank pushes a disjoint contiguous slice of the (already
        all_reduced) gradients, relieving rank-0 which also hosts the PS
        server.  Mirrors the original BagPipe ``bagcache.py`` eviction
        writeback that splits global IDs by ``chunk_size = len // world_size``.

        ``kv_client.update`` routes through ``emb_update_table`` whose C++
        cache-maintain path invalidates the pushed keys from the local GPU
        cache, so callers should still invalidate the *non-pushed* keys
        separately.
        """
        if ids.numel() == 0:
            return
        if not self._is_distributed():
            try:
                self.kv_client.update(table_name, ids, grads)
            except Exception as exc:
                logger.warning("[BagPipe] load-balanced push failed: %s", exc)
            return
        rank = self._get_rank()
        world = dist.get_world_size()
        n = int(ids.size(0))
        chunk = n // world
        start = rank * chunk
        end = n if rank == world - 1 else (rank + 1) * chunk
        if start >= end:
            return
        my_ids = ids[start:end]
        my_grads = grads[start:end]
        if not my_ids.is_contiguous():
            my_ids = my_ids.contiguous()
        if not my_grads.is_contiguous():
            my_grads = my_grads.contiguous()
        try:
            self.kv_client.update(table_name, my_ids, my_grads)
        except Exception as exc:
            logger.warning("[BagPipe] load-balanced push failed: %s", exc)

    def _maybe_build_shared_id_set(self, unique_ids: torch.Tensor) -> None:
        """One-time all_gather of unique IDs to determine which IDs are
        shared across ranks (need all_reduce) vs local-only (no_sync).

        Called during the first few update_grads calls.  Accumulates unique
        IDs from the first ``lookahead_value`` batches, then does a single
        all_gather to build the shared ID set.
        """
        if self._shared_ids is not None:
            return  # Already built
        if not self._is_distributed():
            self._shared_ids = set()
            return

        # Accumulate unique IDs from current batch
        id_list = unique_ids.tolist() if unique_ids.numel() > 0 else []
        self._init_unique_ids.update(id_list)
        self._init_batches_seen += 1

        # Wait for enough batches to capture the ID distribution
        if self._init_batches_seen < max(self.lookahead_value, 2):
            return

        # One-time all_gather: collect unique ID sets from all ranks
        local_ids = torch.tensor(sorted(self._init_unique_ids),
                                  dtype=torch.int64, device=self.device)
        local_n = torch.tensor([local_ids.numel()], dtype=torch.int64,
                                device=self.device)
        world_size = dist.get_world_size()

        # Gather counts
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, local_n)
        max_n = max(int(ni.item()) for ni in n_list)

        # Pad and gather IDs
        padded = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        padded[:local_ids.numel()] = local_ids
        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        dist.all_gather(ids_list, padded)

        # Build frequency map: ID → number of ranks that have it
        id_rank_count: Dict[int, int] = {}
        for r in range(world_size):
            nr = int(n_list[r].item())
            for fid in ids_list[r][:nr].tolist():
                id_rank_count[fid] = id_rank_count.get(fid, 0) + 1

        # Shared = appears on >1 rank
        self._shared_ids = {fid for fid, cnt in id_rank_count.items() if cnt > 1}

        # Build global ID → dense index mapping (for dense all_reduce)
        all_shared = sorted(self._shared_ids)
        self._global_id_to_index = {fid: i for i, fid in enumerate(all_shared)}
        self._global_unique_count = len(all_shared)

        logger.info(
            "[BagPipe] no_sync: %d shared IDs (appear on >1 rank), "
            "%d local-only IDs (skip all_reduce)",
            len(self._shared_ids),
            len(id_rank_count) - len(self._shared_ids),
        )
        # Update stats with shared ID count
        self._stats["bagpipe_shared_ids"] = float(len(self._shared_ids))
        # Free init memory
        self._init_unique_ids = set()

    def _wait_prev_sync_later(self) -> None:
        """Wait for the previous iteration's async sync_later all_reduce and
        push the result to PS (rank 0) / invalidate cache (non-rank-0).

        Called at the *start* of update_grads so the sync_later communication
        overlaps with the forward+backward of the current step, exactly as in
        the original BagPipe bagcache.py cache_sync().
        """
        if self._first_update or self._sync_later_future is None:
            self._first_update = False
            return

        t_start = time.perf_counter()
        # Ensure the sync_later stream's all_reduce has finished
        if self._sync_later_stream is not None:
            self._sync_later_stream.synchronize()
        # Wait for the NCCL work handle and get aggregated result
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

    def _all_gather_sparse_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Sparse all_gather + aggregate gradients across GPUs (async).

        Each rank may have a different set of (id, grad) pairs.  After this
        call completes (via work.wait()), the returned tensors contain the
        aggregated grads (summed across all GPUs for each unique ID).

        Uses dist.all_gather with async_op=True on the given CUDA stream so
        the communication can overlap with computation.
        """
        if not self._is_distributed():
            return ids, grads, None

        t_start = time.perf_counter()
        world_size = dist.get_world_size()
        n = ids.numel()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        # Step 1: Gather N from all ranks (sync, small)
        n_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, n_tensor)
        max_n = max(int(ni.item()) for ni in n_list)

        if max_n == 0:
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

        # Step 2: Pad to max_n on device
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

        # Step 3: All-gather padded tensors (async on stream)
        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                       for _ in range(world_size)]

        ctx_stream = stream
        if ctx_stream is not None:
            with torch.cuda.stream(ctx_stream):
                work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
                work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)
        else:
            work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
            work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)

        # We need to store the lists for later aggregation after wait()
        # Return a composite work handle
        class _CompositeWork:
            def __init__(self, work_ids, work_grads, ids_list, grads_list,
                         n_list, world_size, dim, device, stats):
                self._work_ids = work_ids
                self._work_grads = work_grads
                self._ids_list = ids_list
                self._grads_list = grads_list
                self._n_list = n_list
                self._world_size = world_size
                self._dim = dim
                self._device = device
                self._stats = stats
                self._result = None

            def wait(self):
                self._work_ids.wait()
                self._work_grads.wait()
                # Aggregate: unpad + unique + sum
                all_ids = []
                all_grads = []
                for r in range(self._world_size):
                    nr = int(self._n_list[r].item())
                    if nr > 0:
                        all_ids.append(self._ids_list[r][:nr])
                        all_grads.append(self._grads_list[r][:nr])
                if not all_ids:
                    self._result = (torch.tensor([], dtype=torch.int64, device=self._device),
                                    torch.zeros(0, self._dim, device=self._device))
                    return
                cat_ids = torch.cat(all_ids)
                cat_grads = torch.cat(all_grads)
                unique_ids, inverse = torch.unique(cat_ids, return_inverse=True)
                aggregated = torch.zeros(
                    (len(unique_ids), self._dim), dtype=cat_grads.dtype, device=cat_grads.device)
                aggregated.index_add_(0, inverse, cat_grads)
                self._result = (unique_ids, aggregated)

            @property
            def result(self):
                return self._result

        work = _CompositeWork(work_ids, work_grads, ids_list, grads_list,
                              n_list, world_size, dim, self.device, self._stats)
        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
        return ids_dev, grads_dev, work

    # ------------------------------------------------------------------
    #  Gradient update (sync_now / sync_later)
    # ------------------------------------------------------------------

    def _dense_all_reduce_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Dense all_reduce of shared-ID gradients using a global index mapping.

        Replaces the all_gather + unique + index_add_ approach with a single
        dist.all_reduce on a dense tensor indexed by a pre-built global ID →
        index mapping.  This mirrors the original BagPipe's approach of
        all_reducing dense grad tensors extracted via local cache indices.

        Requires _global_id_to_index to be built (via _maybe_build_shared_id_set).
        Falls back to _all_gather_sparse_async if the mapping is not available.
        """
        if not self._is_distributed():
            return ids, grads, None

        if self._global_id_to_index is None or self._global_unique_count == 0:
            # Fallback to sparse all_gather if mapping not built
            return self._all_gather_sparse_async(ids, grads, stream)

        t_start = time.perf_counter()
        world_size = dist.get_world_size()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        # Create dense grad tensor indexed by global ID position
        dense_grads = torch.zeros(
            (self._global_unique_count, dim),
            dtype=torch.float32, device=self.device,
        )

        # Map local IDs to global indices and scatter grads
        id_list = ids.tolist()
        global_indices = []
        valid_mask = []
        for fid in id_list:
            gidx = self._global_id_to_index.get(fid, -1)
            global_indices.append(gidx)
            valid_mask.append(gidx >= 0)

        if not any(valid_mask):
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

        valid_indices = torch.tensor(
            [gi for gi in global_indices if gi >= 0],
            dtype=torch.long, device=self.device,
        )
        valid_grads = grads[
            torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        ].to(self.device, dtype=torch.float32).contiguous()

        dense_grads.index_put_((valid_indices,), valid_grads)

        # Single all_reduce on the dense tensor (async)
        if stream is not None:
            with torch.cuda.stream(stream):
                work = dist.all_reduce(dense_grads, async_op=True)
        else:
            work = dist.all_reduce(dense_grads, async_op=True)

        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ids"] += float(len(valid_indices))
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

        # Return a composite work that extracts the reduced grads for our IDs
        class _DenseWork:
            def __init__(self, work, dense_grads, valid_indices, ids, dim, device):
                self._work = work
                self._dense_grads = dense_grads
                self._valid_indices = valid_indices
                self._ids = ids
                self._dim = dim
                self._device = device
                self._result = None

            def wait(self):
                self._work.wait()
                # Extract reduced grads for our valid IDs
                reduced = self._dense_grads.index_select(0, self._valid_indices)
                self._result = (self._ids.to(self._device), reduced)

            @property
            def result(self):
                return self._result

        work_obj = _DenseWork(work, dense_grads, valid_indices, ids, dim, self.device)
        return ids.to(self.device), grads.to(self.device), work_obj

    def update_grads(
        self,
        table_name: str,
        unique_ids: torch.Tensor,
        summed_grads: torch.Tensor,
        lr: float,
        batch_num: int,
    ) -> None:
        """Apply SGD update to GPU cache + sync_now/sync_later split.

        Mirrors the original BagPipe bagcache.py cache_sync() overlap design:
          0. Wait for the *previous* step's async sync_later all_reduce (this
             overlapped with the current step's forward+backward).
          1. SGD applied in-place on GPU cache (cache_value -= grad * lr).
          2. sync_now (last use): dense async all_reduce + rank-0 PS push.
          3. sync_later (will reuse): launch dense async all_reduce on a
             dedicated CUDA stream (overlaps with next step's compute).

        All tensors stay on GPU — no CPU round-trip.
        """
        t_start = time.perf_counter()

        # Step 0: wait for previous sync_later (overlap barrier)
        self._wait_prev_sync_later()

        if unique_ids.numel() == 0:
            self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
            return

        # Keep everything on GPU
        ids_cuda = unique_ids.to(self.device, dtype=torch.int64)
        grads_cuda = summed_grads.to(self.device, dtype=torch.float32)
        if grads_cuda.dim() == 1:
            grads_cuda = grads_cuda.unsqueeze(1)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        if not grads_cuda.is_contiguous():
            grads_cuda = grads_cuda.contiguous()

        # In-place SGD on GPU cache
        try:
            success = self.kv_client.apply_sgd_update_gpu_cache(
                table_name, ids_cuda, grads_cuda, learning_rate=lr
            )
        except Exception as exc:
            logger.warning("[BagPipe] apply_sgd_update_gpu_cache raised: %s", exc)
            success = False

        if not success:
            # Fallback: push directly to PS with sync all_reduce
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

        # Build shared ID set (one-time, first few batches)
        self._maybe_build_shared_id_set(ids_cuda)

        # Classify into no_sync / sync_now / sync_later
        # no_sync: local-only ID (not shared across ranks) → no all_reduce
        # sync_now: shared AND last use → all_reduce + push to PS
        # sync_later: shared AND will reuse → defer all_reduce
        id_list = ids_cuda.tolist()
        shared = self._shared_ids or set()

        no_sync_ids = []
        no_sync_grads_indices = []
        sync_now_ids = []
        sync_now_grads_indices = []
        sync_later_ids = []
        sync_later_grads_indices = []
        for j, fid in enumerate(id_list):
            if fid not in shared:
                # no_sync: local-only, skip all_reduce
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

        # ---- no_sync: local-only IDs. SGD already applied to the GPU cache
        # via apply_sgd_update_gpu_cache above.  Two paths:
        #  * Warmup (shared-ID set not yet built): push grads directly to PS
        #    so the PS stays correct before the no_sync classification is
        #    available.
        #  * Steady state: DEFER the PS update to eviction-time value
        #    writeback (BagPipe model: local-only updates stay in the cache
        #    until the entry is evicted, then the cache value is written back
        #    to the PS).  Avoids a per-step PS push for the majority of IDs. ----
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
                # Deferred value-writeback path (opt 6).  The GPU cache
                # already holds the locally-updated value; mark the entry
                # dirty so _evict_entries writes it back to the PS at TTL
                # expiry instead of pushing grads every step.
                for fid in no_sync_ids:
                    self.sync_later_grads.pop(fid, None)
                    entry = self.cache_entries.get(fid)
                    if entry is not None:
                        entry.dirty = True

        # ---- sync_now: dense async all_reduce, then rank-0 push ----
        if sync_now_count > 0:
            now_indices = torch.tensor(sync_now_grads_indices, dtype=torch.long,
                                        device=self.device)
            now_ids = ids_cuda[now_indices].contiguous()
            now_grads = grads_cuda[now_indices].contiguous()
            # Fold in previously accumulated sync_later grads for these IDs
            if self.sync_later_grads:
                now_ids_list = now_ids.tolist()
                for j, fid in enumerate(now_ids_list):
                    if fid in self.sync_later_grads:
                        now_grads[j] += self.sync_later_grads[fid].to(self.device)
            _, _, work = self._dense_all_reduce_async(now_ids, now_grads)
            if work is not None:
                work.wait()  # sync_now must complete before PS push
                agg_ids, agg_grads = work.result
            else:
                agg_ids, agg_grads = now_ids, now_grads
            if not self._is_distributed() or self._get_rank() == 0:
                try:
                    self.kv_client.update(self.master_table_name, agg_ids, agg_grads)
                except Exception as exc:
                    logger.warning("[BagPipe] sync_now push failed: %s", exc)
            if self._is_distributed() and self._get_rank() != 0:
                try:
                    self.kv_client.invalidate_gpu_cache(self.master_table_name, agg_ids)
                except Exception as exc:
                    logger.warning("[BagPipe] sync_now invalidate failed: %s", exc)
            for fid in now_ids.tolist():
                self.cache_entries.pop(fid, None)
                self.sync_later_grads.pop(fid, None)

        # ---- sync_later: launch async all_reduce on dedicated stream ----
        # This overlaps with the NEXT step's forward+backward.
        if sync_later_count > 0:
            later_indices = torch.tensor(sync_later_grads_indices, dtype=torch.long,
                                          device=self.device)
            later_ids = ids_cuda[later_indices].contiguous()
            later_grads = grads_cuda[later_indices].clone().contiguous()
            # Accumulate into prior sync_later grads
            later_ids_list = later_ids.tolist()
            for j, fid in enumerate(later_ids_list):
                if fid in self.sync_later_grads:
                    later_grads[j] += self.sync_later_grads[fid].to(self.device)
            # Launch async all_reduce on the dedicated stream
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
        """Legacy flush path — thin wrapper around dense all_reduce.

        The main update_grads path handles sync_now inline on GPU.
        """
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

    def cleanup(self, current_batch: int) -> None:
        """Evict expired cache entries (opt 5: GPU work here, PS write async).

        GPU-cache reads and invalidations run synchronously on the training
        thread (they touch the GPU and must not contend with a background
        thread).  Only the PS value-writeback (network) is offloaded to the
        background ``_cleanup_loop`` thread via ``_cleanup_queue``.  Mirrors
        the original BagPipe ``bagcache.py`` ``launch_cache_cleanup`` thread
        that performs ``cache_eviction_update`` concurrently with training.
        """
        t_start = time.perf_counter()

        # Find expired entries
        expired: list[int] = []
        for fid, entry in list(self.cache_entries.items()):
            if entry.ttl < current_batch:
                expired.append(fid)

        if expired:
            self._evict_entries(expired)

        self._stats["bagpipe_cleanup_ms"] += (time.perf_counter() - t_start) * 1e3

    def _read_cache_values(self, ids: list[int]) -> tuple:
        """Read current GPU cache values for local-only (no_sync) IDs.

        Runs on the training thread (GPU op).  The values already have local
        SGD applied via ``apply_sgd_update_gpu_cache``; for no_sync IDs the
        local gradient is the full gradient, so the value equals the correct
        PS value.  The actual PS write is offloaded to the background thread
        (opt 5).
        """
        ids_cuda = torch.tensor(ids, dtype=torch.int64, device=self.device)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        try:
            values = self.kv_client.gpu_cache_lookup_flat(
                ids_cuda, self.embedding_dim
            )
            if values.device != self.device:
                values = values.to(self.device)
            if not values.is_contiguous():
                values = values.contiguous()
            return ids_cuda, values
        except Exception as exc:
            logger.warning("[BagPipe] value read for writeback failed: %s", exc)
            return ids_cuda, None

    def _evict_entries(self, expired_ids: list[int]) -> None:
        """Evict expired entries: write back values / flush grads + invalidate.

        Two writeback paths, mirroring the original BagPipe bagcache.py
        ``clean_up_caches``:
          * Deferred no_sync (local-only) entries: write cache *values* back
            to the PS (opt 6).  These had SGD applied locally but their PS
            update was deferred.
          * Deferred sync_later (shared) entries: flush aggregated *gradients*
            to the PS (load-balanced, opt 4) then invalidate.
        Remaining (non-dirty) entries are simply invalidated.
        """
        if not expired_ids:
            return

        self._stats["bagpipe_evicted_ids"] += float(len(expired_ids))

        shared = self._shared_ids or set()

        # ---- Value writeback for deferred no_sync (local-only) entries ----
        # Read cache values on this (training) thread, then offload only the
        # PS write to the background thread (opt 5).  Correct for no_sync IDs
        # because local grad == full grad.
        deferred_value = [
            fid for fid in expired_ids
            if fid in self.cache_entries
            and self.cache_entries[fid].dirty
            and fid not in shared
            and fid not in self.sync_later_grads
        ]
        if deferred_value:
            wb_ids, wb_vals = self._read_cache_values(deferred_value)
            if wb_vals is not None and wb_ids.numel() > 0:
                self._cleanup_queue.put((wb_ids, wb_vals))
            self._stats["bagpipe_writeback_ids"] += float(len(deferred_value))

        # ---- Grad flush for deferred sync_later (shared) entries ----
        dirty_expired = [
            fid for fid in expired_ids
            if fid in self.sync_later_grads
        ]
        if dirty_expired:
            self._flush_sync_later(dirty_expired)
            self._stats["bagpipe_writeback_ids"] += float(len(dirty_expired))

        # ---- Invalidate remaining expired entries from GPU cache ----
        # Some may already be invalidated by emb_write_values /
        # emb_update_table above.  invalidate_gpu_cache on already-removed
        # keys is a no-op.
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
                self.kv_client.invalidate_gpu_cache(self.master_table_name, remaining_cuda)
            except Exception as exc:
                logger.warning("[BagPipe] invalidate_gpu_cache failed: %s", exc)

        # ---- Remove from tracking ----
        for fid in expired_ids:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)

    def _flush_sync_later(self, ids: list[int]) -> None:
        """Flush accumulated sync_later gradients (eviction-time fallback).

        In the overlap path, sync_later grads are all_reduced async and pushed
        at the next update_grads call.  This method handles the case where
        entries are evicted before the async push completes — it does a
        synchronous dense all_reduce and push.
        """
        if not ids:
            return
        sl_ids_cpu = torch.tensor(ids, dtype=torch.int64)
        sl_grads_cpu = torch.stack([
            self.sync_later_grads[fid] for fid in ids
        ])

        _, _, work = self._dense_all_reduce_async(sl_ids_cpu, sl_grads_cpu)
        if work is not None:
            work.wait()
            agg_ids, agg_grads = work.result
        else:
            agg_ids, agg_grads = sl_ids_cpu, sl_grads_cpu
        self._load_balanced_push(self.master_table_name, agg_ids, agg_grads)
        if self._is_distributed():
            try:
                self.kv_client.invalidate_gpu_cache(self.master_table_name, agg_ids.to(self.device))
            except Exception as exc:
                logger.warning("[BagPipe] sync_later invalidate failed: %s", exc)

    # ------------------------------------------------------------------
    #  Background cleanup thread
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background thread for asynchronous PS value writeback (opt 5).

        Processes enqueued (ids, values) pairs and writes them to the PS via
        ``emb_write_values`` (network only).  GPU-cache reads and
        invalidations are done on the training thread (in ``_evict_entries``)
        to avoid GPU mutex contention; only the PS write is offloaded here.
        Mirrors the original BagPipe ``launch_cache_cleanup`` thread that
        performs ``cache_eviction_update`` concurrently with training.
        """
        while True:
            try:
                task = self._cleanup_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            wb_ids, wb_vals = task
            if wb_vals is None:
                continue
            try:
                self.kv_client.emb_write_values(
                    self.master_table_name, wb_ids, wb_vals
                )
            except Exception as exc:
                logger.warning("[BagPipe] async value writeback failed: %s", exc)

    def shutdown(self) -> None:
        """Signal the background thread to exit and flush pending work."""
        # Flush any pending async sync_later before shutting down
        try:
            self._wait_prev_sync_later()
        except Exception:
            pass
        # Drain pending async PS writebacks synchronously so deferred no_sync
        # values reach the PS before the process exits (opt 5).
        try:
            while True:
                try:
                    task = self._cleanup_queue.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    break
                wb_ids, wb_vals = task
                if wb_vals is not None:
                    self.kv_client.emb_write_values(
                        self.master_table_name, wb_ids, wb_vals
                    )
        except Exception:
            pass
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
