"""Prefetch lifecycle for BagPipe cache controller.

Handles enqueue (batch preparation), oracle prescan, smart prefetch
issuing with pressure-aware pruning, and GPU cache fill (opt 3, 8, 10).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import torch

from .types import CacheEntry, PrefetchSlot

logger = logging.getLogger(__name__)


class BagPipePrefetchMixin:
    """Mixin providing prefetch and prescan methods.

    Expects the host class to provide: ``device``, ``kv_client``,
    ``embedding_dim``, ``master_table_name``, ``cache_entries``,
    ``latest_tracker``, ``_lookahead_ids``, ``_next_enqueue_batch``,
    ``_current_batch``, ``_batched_prefetch_ids``, ``_batched_prefetch_ttl``,
    ``_batched_count``, ``_pending_prefetch``, ``cleanup_interval``,
    ``_shared_ids``, ``cache_capacity``, ``_stats``, ``_id_extractor``.
    """

    # ------------------------------------------------------------------
    #  Fused ID extraction (model-agnostic, injected via _id_extractor)
    # ------------------------------------------------------------------

    def _compute_unique_fused_ids(self, sparse_features: Any) -> torch.Tensor:
        """Extract unique fused IDs from a sparse batch via the injected extractor."""
        fused_all = self._id_extractor(sparse_features)
        if fused_all.numel() == 0:
            return fused_all
        return torch.unique(fused_all)

    # ------------------------------------------------------------------
    #  Enqueue (called during batch preparation, ahead of consumption)
    # ------------------------------------------------------------------

    def enqueue(self, sparse_features: Any) -> None:
        """Record a batch's unique IDs into the oracle lookahead buffer."""
        unique_ids = self._compute_unique_fused_ids(sparse_features)
        batch_num = self._next_enqueue_batch
        self._next_enqueue_batch += 1
        self._lookahead_ids.append((batch_num, unique_ids))

        id_set = unique_ids.tolist() if unique_ids.numel() > 0 else []
        for fid in id_set:
            self.latest_tracker[fid] = max(
                self.latest_tracker.get(fid, -1), batch_num
            )

    # ------------------------------------------------------------------
    #  Oracle prescan (opt 8)
    # ------------------------------------------------------------------

    def prescan_batch(self, batch_num: int, sparse_features: Any) -> None:
        """Record a batch's unique IDs during a full-dataset pre-scan."""
        unique_ids = self._compute_unique_fused_ids(sparse_features)
        id_set = unique_ids.tolist() if unique_ids.numel() > 0 else []
        for fid in id_set:
            self.latest_tracker[fid] = max(
                self.latest_tracker.get(fid, -1), batch_num
            )
            self._prescan_unique_ids.add(fid)
        self._stats["bagpipe_prescan_batches"] += 1
        self._stats["bagpipe_prescan_ids"] += float(len(id_set))

    # ------------------------------------------------------------------
    #  Consume (called when a batch is about to be used)
    # ------------------------------------------------------------------

    def prefill_cache(
        self,
        sparse_features: Any,
        compute_device: torch.device,
    ) -> None:
        """Issue smart prefetch + fill GPU cache with TTL tracking.

        First waits for the previous step's sync_now all_reduce (opt 11:
        overlaps with the gap between steps and the subsequent prefetch
        fill), then prefetches IDs that are not cached or have expired TTL.
        """
        t_start = time.perf_counter()
        self._wait_pending_sync_now()

        if not self._lookahead_ids:
            return
        batch_num, unique_ids = self._lookahead_ids.popleft()
        self._current_batch = batch_num

        if unique_ids.numel() == 0:
            return

        id_list = unique_ids.tolist()

        prefetch_targets: list[int] = []
        for fid in id_list:
            entry = self.cache_entries.get(fid)
            if entry is None or entry.ttl < batch_num:
                prefetch_targets.append(fid)
            else:
                self._stats["bagpipe_prefetch_skip_cached"] += 1

        ttl_map: Dict[int, int] = {}
        for fid in id_list:
            last_use = self.latest_tracker.get(fid, batch_num)
            ttl_map[fid] = last_use

        self._batched_prefetch_ids.update(prefetch_targets)
        self._batched_prefetch_ttl.update(
            {fid: ttl_map[fid] for fid in prefetch_targets if fid in ttl_map}
        )
        self._batched_count += 1

        if self._batched_count >= self.cleanup_interval:
            self._issue_batched_prefetch()

        if self._pending_prefetch is not None:
            self._fill_cache_from_pending(compute_device)

        self._stats["bagpipe_prefill_ms"] += (time.perf_counter() - t_start) * 1e3

    def _issue_batched_prefetch(self) -> None:
        """Issue a single batched prefetch with pressure-aware pruning (opt 3, 10)."""
        if not self._batched_prefetch_ids:
            self._batched_count = 0
            return

        shared = self._shared_ids
        if shared is not None:
            pressure = len(self.cache_entries) / self.cache_capacity if self.cache_capacity > 0 else 0.0
            keep_local_nosync = pressure > 0.50
            if keep_local_nosync:
                self._stats["bagpipe_prefetch_local_nosync_kept"] = \
                    self._stats.get("bagpipe_prefetch_local_nosync_kept", 0.0) + \
                    float(sum(1 for fid in self._batched_prefetch_ids if fid not in shared))
            else:
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

        for fid in slot.ids_cpu.tolist():
            ttl = slot.ttl_map.get(fid, self._current_batch)
            self.cache_entries[fid] = CacheEntry(ttl=ttl, dirty=False)
