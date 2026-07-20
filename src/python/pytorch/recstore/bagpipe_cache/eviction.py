"""Cache eviction, writeback, and dynamic lookahead for BagPipe controller.

Handles TTL-based eviction, value/gradient writeback to PS, async
background thread, and dynamic lookahead adjustment (opt 4, 5, 6, 7, 9).
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class BagPipeEvictionMixin:
    """Mixin providing eviction, writeback, and cleanup-thread methods.

    Expects the host class to provide: ``device``, ``kv_client``,
    ``embedding_dim``, ``master_table_name``, ``cache_entries``,
    ``cache_capacity``, ``sync_later_grads``, ``_shared_ids``,
    ``_eviction_stream``, ``_cleanup_queue``, ``_cleanup_thread``,
    ``_stats``, ``_base_lookahead``, ``_dynamic_lookahead``,
    ``_max_lookahead``, ``_additive_increase``,
    ``_multiplicative_decrease``, ``_pressure_low``, ``_pressure_high``,
    ``_lookahead_adjust_interval``, ``cleanup_batch_proportion``,
    ``cleanup_interval``, ``_dense_all_reduce_async``,
    ``_load_balanced_push``, ``_is_distributed``, ``_get_rank``.
    """

    def cleanup(self, current_batch: int) -> None:
        """Evict expired cache entries + dynamic lookahead adjustment."""
        t_start = time.perf_counter()

        expired: list[int] = []
        for fid, entry in list(self.cache_entries.items()):
            if entry.ttl < current_batch:
                expired.append(fid)

        if expired:
            self._evict_entries(expired)

        self._maybe_adjust_lookahead(current_batch)

        self._stats["bagpipe_cleanup_ms"] += (time.perf_counter() - t_start) * 1e3

    def _maybe_adjust_lookahead(self, current_batch: int) -> None:
        """Adjust lookahead based on cache pressure (opt 9)."""
        if current_batch % self._lookahead_adjust_interval != 0:
            return
        if self.cache_capacity <= 0:
            return
        pressure = len(self.cache_entries) / self.cache_capacity
        old_la = self._dynamic_lookahead
        if pressure > self._pressure_high:
            self._dynamic_lookahead = max(1, int(self._dynamic_lookahead * self._multiplicative_decrease))
        elif pressure < self._pressure_low:
            self._dynamic_lookahead = min(self._max_lookahead, self._dynamic_lookahead + self._additive_increase)
        if self._dynamic_lookahead != old_la:
            self.cleanup_interval = max(1, int(self.cleanup_batch_proportion * self._dynamic_lookahead))
            logger.info(
                "[BagPipe] dynamic lookahead: %d -> %d (pressure=%.2f, cache=%d/%d)",
                old_la, self._dynamic_lookahead, pressure,
                len(self.cache_entries), self.cache_capacity,
            )
        self._stats["bagpipe_dynamic_lookahead"] = float(self._dynamic_lookahead)
        self._stats["bagpipe_cache_pressure"] = pressure

    def _read_cache_values(self, ids: list[int]) -> tuple:
        """Read GPU cache values for no_sync IDs on the eviction stream (opt 7)."""
        ids_cuda = torch.tensor(ids, dtype=torch.int64, device=self.device)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        stream = self._eviction_stream
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    values = self.kv_client.gpu_cache_lookup_flat(
                        ids_cuda, self.embedding_dim
                    )
                event = torch.cuda.Event()
                event.record(stream)
            else:
                values = self.kv_client.gpu_cache_lookup_flat(
                    ids_cuda, self.embedding_dim
                )
                event = None
            if values.device != self.device:
                values = values.to(self.device)
            if not values.is_contiguous():
                values = values.contiguous()
            return ids_cuda, values, event
        except Exception as exc:
            logger.warning("[BagPipe] value read for writeback failed: %s", exc)
            return ids_cuda, None, None

    def _evict_entries(self, expired_ids: list[int]) -> None:
        """Evict expired entries: write back values / flush grads + invalidate (opt 4, 6)."""
        if not expired_ids:
            return

        self._stats["bagpipe_evicted_ids"] += float(len(expired_ids))

        shared = self._shared_ids or set()

        deferred_value = [
            fid for fid in expired_ids
            if fid in self.cache_entries
            and self.cache_entries[fid].dirty
            and fid not in shared
            and fid not in self.sync_later_grads
        ]
        if deferred_value:
            wb_ids, wb_vals, wb_event = self._read_cache_values(deferred_value)
            if wb_vals is not None and wb_ids.numel() > 0:
                self._cleanup_queue.put((wb_ids, wb_vals, wb_event))
            self._stats["bagpipe_writeback_ids"] += float(len(deferred_value))

        dirty_expired = [
            fid for fid in expired_ids
            if fid in self.sync_later_grads
        ]
        if dirty_expired:
            self._flush_sync_later(dirty_expired)
            self._stats["bagpipe_writeback_ids"] += float(len(dirty_expired))

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

        for fid in expired_ids:
            self.cache_entries.pop(fid, None)
            self.sync_later_grads.pop(fid, None)

    def _flush_sync_later(self, ids: list[int]) -> None:
        """Flush accumulated sync_later gradients (eviction-time fallback)."""
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
    #  Background cleanup thread (opt 5)
    # ------------------------------------------------------------------

    def _cleanup_loop(self) -> None:
        """Background thread for asynchronous PS value writeback (opt 5)."""
        while True:
            try:
                task = self._cleanup_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            wb_ids, wb_vals, wb_event = task
            if wb_vals is None:
                continue
            if wb_event is not None:
                wb_event.synchronize()
            try:
                self.kv_client.emb_write_values(
                    self.master_table_name, wb_ids, wb_vals
                )
            except Exception as exc:
                logger.warning("[BagPipe] async value writeback failed: %s", exc)

    def shutdown(self) -> None:
        """Signal the background thread to exit and flush pending work."""
        try:
            self._wait_pending_sync_now()
        except Exception:
            pass
        try:
            self._wait_prev_sync_later()
        except Exception:
            pass
        try:
            while True:
                try:
                    task = self._cleanup_queue.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    break
                wb_ids, wb_vals, wb_event = task
                if wb_event is not None:
                    wb_event.synchronize()
                if wb_vals is not None:
                    self.kv_client.emb_write_values(
                        self.master_table_name, wb_ids, wb_vals
                    )
        except Exception:
            pass
        self._cleanup_queue.put(None)
