"""BagPipeGradMixin — gradient update logic with sync_now / sync_later / no_sync split.

Extracted from controller.py.  The overlap barriers (_wait_prev_sync_later
and _wait_pending_sync_now) are tightly coupled to update_grads because they
wait for the async all_reduce operations launched in the previous step's
update_grads call.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class BagPipeGradMixin:
    """Provides update_grads and the associated overlap barriers.

    Expects the host class to provide (via other mixins or __init__):
        - self.device, self.kv_client, self.master_table_name
        - self.cache_entries, self.sync_later_grads, self.latest_tracker
        - self._sync_later_stream, self._sync_later_future
        - self._sync_later_ids, self._sync_later_grads_buf
        - self._first_update, self._pending_sync_now_work, self._pending_sync_now_ids
        - self._shared_ids, self._stats
        - self._dense_all_reduce_async(), _maybe_build_shared_id_set()
        - self._is_distributed(), _get_rank()
    """

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
