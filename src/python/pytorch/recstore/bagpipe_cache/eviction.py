"""Cache eviction, writeback, and dynamic lookahead for BagPipe controller.

Handles TTL-based eviction, value writeback to PS, async background thread,
and dynamic lookahead adjustment (opt 4, 5, 6, 7, 9).

驱逐判定是扁平张量上的向量化全空间掩码扫描 (~5.2M 元素, ~0.1ms),
与原版 BagPipe 的 numpy 数组语义同构; 没有任何逐 fid 的 Python 循环。
"""

from __future__ import annotations

import logging
import queue
import time

import torch

logger = logging.getLogger(__name__)


class BagPipeEvictionMixin:
    """Mixin providing eviction, writeback, and cleanup-thread methods.

    Expects the host class to provide: ``device``, ``kv_client``,
    ``embedding_dim``, ``master_table_name``, ``_cached_dev``, ``_ttl_dev``,
    ``_dirty_dev``, ``_shared_ids_tensor``, ``cache_capacity``,
    ``_eviction_stream``, ``_cleanup_queue``, ``_cleanup_thread``,
    ``_stats``, ``_base_lookahead``, ``_dynamic_lookahead``,
    ``_max_lookahead``, ``_additive_increase``,
    ``_multiplicative_decrease``, ``_pressure_low``, ``_pressure_high``,
    ``_lookahead_adjust_interval``, ``cleanup_batch_proportion``,
    ``cleanup_interval``, ``_cached_count()``, ``_is_distributed()``,
    ``_get_rank()``, ``_capacity_watermark``, ``_max_evict_per_cleanup``,
    ``_to_compact()``, ``_to_fused()``.
    """

    def cleanup(self, current_batch: int) -> None:
        """Evict expired cache entries + dynamic lookahead adjustment."""
        t_start = time.perf_counter()

        if self._cached_dev is not None:
            # 到期判定: 驻留且 ttl < 当前批 (向量化全空间扫描)
            expired_mask = self._cached_dev & (self._ttl_dev < current_batch)
            expired_ids = expired_mask.nonzero(as_tuple=False).squeeze(1)
            if expired_ids.numel() > 0:
                # 驱逐上限: 超出的条目保持旧 ttl, 下一次 cleanup 再扫出
                n_evict = min(
                    int(expired_ids.numel()), self._max_evict_per_cleanup
                )
                if n_evict < expired_ids.numel():
                    # 优先逐 ttl 最低的
                    order = self._ttl_dev[expired_ids].argsort()
                    expired_ids = expired_ids[order[:n_evict]].contiguous()
                else:
                    expired_ids = expired_ids.contiguous()
                self._evict_entries(expired_ids)

            # 容量兜底: 超过高水位时按 ttl 最低(最久未用)继续驱逐到水位以下
            capacity = self.cache_capacity
            if capacity > 0:
                high_water = int(capacity * self._capacity_watermark)
                overflow = self._cached_count() - high_water
                if overflow > 0:
                    ttl_eff = torch.where(
                        self._cached_dev,
                        self._ttl_dev,
                        torch.iinfo(torch.int32).max,
                    )
                    overflow_ids = torch.topk(
                        ttl_eff, k=min(overflow, self._max_evict_per_cleanup),
                        largest=False,
                    ).indices.contiguous()
                    self._evict_entries(overflow_ids)

        self._maybe_adjust_lookahead(current_batch)

        self._stats["bagpipe_cleanup_ms"] += (time.perf_counter() - t_start) * 1e3

    def _maybe_adjust_lookahead(self, current_batch: int) -> None:
        """Adjust lookahead based on cache pressure (opt 9)."""
        if current_batch % self._lookahead_adjust_interval != 0:
            return
        if self.cache_capacity <= 0:
            return
        pressure = self._cached_count() / self.cache_capacity
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
                self._cached_count(), self.cache_capacity,
            )
        self._stats["bagpipe_dynamic_lookahead"] = float(self._dynamic_lookahead)
        self._stats["bagpipe_cache_pressure"] = pressure

    def _read_cache_values(self, ids: torch.Tensor) -> tuple:
        """Read GPU cache values for no_sync IDs on the eviction stream (opt 7)."""
        ids_cuda = ids.to(device=self.device, dtype=torch.int64)
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

    def _evict_entries(self, expired_ids: torch.Tensor) -> None:
        """Evict expired entries: write back values + invalidate (opt 4, 6).

        ``expired_ids`` 是 compact 索引 (来自簿记张量的 nonzero);
        kv_client / GPU cache 的 key 空间是 fused id, 调用前统一转换。
        只有 dirty 且非 shared 的条目需要值写回 (no_sync 的 PS 持久化);
        shared 条目已由 barrier 的聚合推送持久化, 本地副本直接失效即可。
        """
        if expired_ids.numel() == 0:
            return

        self._stats["bagpipe_evicted_ids"] += float(expired_ids.numel())

        dirty_m = self._dirty_dev[expired_ids]
        shared_t = self._shared_ids_tensor
        if shared_t is not None and shared_t.numel() > 0:
            shared_m = torch.isin(expired_ids, self._to_compact(shared_t))
        else:
            shared_m = torch.zeros_like(dirty_m)
        wb_mask = dirty_m & ~shared_m
        wb_ids = expired_ids[wb_mask]
        if wb_ids.numel() > 0:
            wb_ids, wb_vals, wb_event = self._read_cache_values(
                self._to_fused(wb_ids)
            )
            if wb_vals is not None and wb_ids.numel() > 0:
                self._cleanup_queue.put((wb_ids, wb_vals, wb_event))
            self._stats["bagpipe_writeback_ids"] += float(wb_ids.numel())

        try:
            self.kv_client.invalidate_gpu_cache(
                self.master_table_name, self._to_fused(expired_ids)
            )
        except Exception as exc:
            logger.warning("[BagPipe] invalidate_gpu_cache failed: %s", exc)

        # 清簿记 (compact 索引)
        self._cached_dev[expired_ids] = False
        self._dirty_dev[expired_ids] = False
        self._ttl_dev[expired_ids] = 0

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
        # rank0 深度-1 流水线上未完成的持久化推送
        pending_push = getattr(self, "_pending_ps_push_handle", None)
        if pending_push is not None:
            try:
                self.kv_client.wait(pending_push)
            except Exception:
                pass
            self._pending_ps_push_handle = None
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
        self._flush_dirty_at_shutdown()
        self._cleanup_queue.put(None)

    def _flush_dirty_at_shutdown(self) -> None:
        """训练结束时把仍在 cache 的 no_sync (dirty 且非 shared) 值写回 PS。

        容量充裕时 TTL 几乎不会在训练中途到期 (margin 覆盖重现间隔,
        eviction 是唯一的常规写回触发点), 剩余 dirty 条目必须在此
        统一落盘, 否则 no_sync 的本地 SGD 增量在进程退出时丢失。
        分块读写以约束峰值显存 (~32MB/块)。
        """
        if self._dirty_dev is None:
            return
        try:
            dirty_mask = self._dirty_dev.clone()
        except Exception:
            return
        shared_t = self._shared_ids_tensor
        if shared_t is not None and shared_t.numel() > 0:
            dirty_mask[self._to_compact(shared_t)] = False
        chunk = 65536
        total = int(dirty_mask.sum().item())
        if total == 0:
            return
        logger.info("[BagPipe] shutdown: flushing %d dirty entries to PS", total)
        done = 0
        while done < total:
            remaining = dirty_mask.sum().item()
            if int(remaining) == 0:
                break
            ids = dirty_mask.nonzero(as_tuple=False).squeeze(1)[:chunk]
            ids_fused, vals, event = self._read_cache_values(self._to_fused(ids))
            if vals is not None and ids_fused.numel() > 0:
                if event is not None:
                    event.synchronize()
                try:
                    self.kv_client.emb_write_values(
                        self.master_table_name, ids_fused, vals
                    )
                except Exception as exc:
                    logger.warning("[BagPipe] shutdown writeback failed: %s", exc)
                self._stats["bagpipe_writeback_ids"] += float(ids_fused.numel())
                done += int(ids_fused.numel())
            # 无论写回成败都清标记, 防止死循环
            dirty_mask[ids] = False
            self._dirty_dev[ids] = False
