"""Cross-GPU gradient communication for BagPipe cache controller.

Handles shared-ID classification, dense/sparse all_reduce, and the
one-time all_gather that builds the shared-ID set (opt 1, 2, 8).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .types import _CompositeWork, _DenseWork

logger = logging.getLogger(__name__)


class BagPipeCommMixin:
    """Mixin providing cross-GPU gradient synchronization primitives.

    Expects the host class to provide: ``device``, ``_shared_ids``,
    ``_global_id_to_index``, ``_global_unique_count``, ``_init_unique_ids``,
    ``_init_batches_seen``, ``_prescan_done``, ``_prescan_unique_ids``,
    ``_stats``, ``lookahead_value``.
    """

    def _is_distributed(self) -> bool:
        try:
            return dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    def _get_rank(self) -> int:
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
        server.  Mirrors the original BagPipe eviction writeback that splits
        global IDs by ``chunk_size = len // world_size``.
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

    # ------------------------------------------------------------------
    #  Shared-ID set construction (opt 1, opt 8 prescan)
    # ------------------------------------------------------------------

    def _maybe_build_shared_id_set(self, unique_ids: torch.Tensor) -> None:
        """One-time all_gather of unique IDs to determine which IDs are
        shared across ranks (need all_reduce) vs local-only (no_sync).

        Skipped if oracle prescan (opt 8) already built the complete set.
        """
        if self._shared_ids is not None:
            return
        if not self._is_distributed():
            self._shared_ids = set()
            return

        id_list = unique_ids.tolist() if unique_ids.numel() > 0 else []
        self._init_unique_ids.update(id_list)
        self._init_batches_seen += 1

        if self._init_batches_seen < max(self.lookahead_value, 2):
            return

        self._build_shared_id_set_from(self._init_unique_ids)
        logger.info(
            "[BagPipe] no_sync: %d shared IDs (appear on >1 rank), "
            "%d local-only IDs (skip all_reduce)",
            len(self._shared_ids),
            len(self._init_unique_ids) - len(self._shared_ids),
        )
        self._stats["bagpipe_shared_ids"] = float(len(self._shared_ids))
        self._init_unique_ids = set()

    def _build_shared_id_set_from(self, local_id_set: set) -> None:
        """Build the shared-ID set + global index mapping via all_gather."""
        if not self._is_distributed():
            self._shared_ids = set()
            self._global_id_to_index = {}
            self._global_unique_count = 0
            return

        local_ids = torch.tensor(sorted(local_id_set),
                                  dtype=torch.int64, device=self.device)
        local_n = torch.tensor([local_ids.numel()], dtype=torch.int64,
                                device=self.device)
        world_size = dist.get_world_size()
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, local_n)
        max_n = max(int(ni.item()) for ni in n_list)

        padded = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        padded[:local_ids.numel()] = local_ids
        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        dist.all_gather(ids_list, padded)

        id_rank_count: Dict[int, int] = {}
        for r in range(world_size):
            nr = int(n_list[r].item())
            for fid in ids_list[r][:nr].tolist():
                id_rank_count[fid] = id_rank_count.get(fid, 0) + 1

        self._shared_ids = {fid for fid, cnt in id_rank_count.items() if cnt > 1}
        all_shared = sorted(self._shared_ids)
        self._global_id_to_index = {fid: i for i, fid in enumerate(all_shared)}
        self._global_unique_count = len(all_shared)

    def finalize_prescan(self) -> None:
        """After all batches pre-scanned, build the complete shared-ID set."""
        if self._prescan_done:
            return
        self._build_shared_id_set_from(self._prescan_unique_ids)
        self._prescan_done = True
        self._stats["bagpipe_shared_ids"] = float(len(self._shared_ids))
        total_unique = len(self._prescan_unique_ids)
        logger.info(
            "[BagPipe] oracle prescan: %d shared IDs, %d local-only, "
            "%d total unique across %d batches",
            len(self._shared_ids),
            total_unique - len(self._shared_ids),
            total_unique,
            int(self._stats["bagpipe_prescan_batches"]),
        )
        self._prescan_unique_ids = set()

    # ------------------------------------------------------------------
    #  All-reduce primitives (opt 2)
    # ------------------------------------------------------------------

    def _all_gather_sparse_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Sparse all_gather + aggregate gradients across GPUs (async)."""
        if not self._is_distributed():
            return ids, grads, None

        t_start = time.perf_counter()
        world_size = dist.get_world_size()
        n = ids.numel()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        n_tensor = torch.tensor([n], dtype=torch.int64, device=self.device)
        n_list = [torch.zeros(1, dtype=torch.int64, device=self.device)
                   for _ in range(world_size)]
        dist.all_gather(n_list, n_tensor)
        max_n = max(int(ni.item()) for ni in n_list)

        if max_n == 0:
            self._stats["bagpipe_all_reduce_calls"] += 1
            self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
            return ids[:0], grads[:0], None

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

        ids_list = [torch.zeros(max_n, dtype=torch.int64, device=self.device)
                     for _ in range(world_size)]
        grads_list = [torch.zeros(max_n, dim, dtype=torch.float32, device=self.device)
                       for _ in range(world_size)]

        if stream is not None:
            with torch.cuda.stream(stream):
                work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
                work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)
        else:
            work_ids = dist.all_gather(ids_list, padded_ids, async_op=True)
            work_grads = dist.all_gather(grads_list, padded_grads, async_op=True)

        work = _CompositeWork(work_ids, work_grads, ids_list, grads_list,
                              n_list, world_size, dim, self.device, self._stats)
        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3
        return ids_dev, grads_dev, work

    def _dense_all_reduce_async(
        self,
        ids: torch.Tensor,
        grads: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Dense all_reduce of shared-ID gradients using a global index mapping."""
        if not self._is_distributed():
            return ids, grads, None

        if self._global_id_to_index is None or self._global_unique_count == 0:
            return self._all_gather_sparse_async(ids, grads, stream)

        t_start = time.perf_counter()
        world_size = dist.get_world_size()

        if grads.dim() == 1:
            grads = grads.unsqueeze(1)
        dim = grads.size(1)

        dense_grads = torch.zeros(
            (self._global_unique_count, dim),
            dtype=torch.float32, device=self.device,
        )

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

        if stream is not None:
            with torch.cuda.stream(stream):
                work = dist.all_reduce(dense_grads, async_op=True)
        else:
            work = dist.all_reduce(dense_grads, async_op=True)

        self._stats["bagpipe_all_reduce_calls"] += 1
        self._stats["bagpipe_all_reduce_ids"] += float(len(valid_indices))
        self._stats["bagpipe_all_reduce_ms"] += (time.perf_counter() - t_start) * 1e3

        work_obj = _DenseWork(work, dense_grads, valid_indices, ids, dim, self.device)
        return ids.to(self.device), grads.to(self.device), work_obj
