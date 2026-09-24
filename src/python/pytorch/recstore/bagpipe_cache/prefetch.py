"""Prefetch lifecycle for BagPipe cache controller.

Handles enqueue (batch preparation + PS prefetch pre-issue), oracle
prescan, and GPU cache fill (opt 3, 8, 10).  The pre-issue design comes
from the benchmark branch (codex/bench-bagpipe-buffer-pool): the PS
prefetch is issued at enqueue time, ``lookahead`` steps before the batch
is consumed, so ``wait_and_get`` at consume is near-instant.

全部 per-id 决策走 controller 的扁平 GPU 张量 (_latest_dev/_ttl_dev/
_cached_dev), 与原版 BagPipe oracle 的 numpy 数组向量化同构 —— enqueue
路径没有任何逐 fid 的 Python 循环。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from .types import PrefetchSlot

logger = logging.getLogger(__name__)


class BagPipePrefetchMixin:
    """Mixin providing prefetch and prescan methods.

    Expects the host class to provide: ``device``, ``kv_client``,
    ``embedding_dim``, ``master_table_name``, ``_lookahead_ids``,
    ``_next_enqueue_batch``, ``_current_batch``, ``_prefetch_handles``,
    ``_shared_ids``, ``cache_capacity``, ``_stats``, ``_id_extractor``,
    ``_latest_dev``, ``_ttl_dev``, ``_cached_dev``, ``_to_compact()``,
    ``_to_fused()``, ``_compact_in_range()``,
    ``_ttl_margin``, ``_max_inflight_prefetch``.
    """

    # ------------------------------------------------------------------
    #  Fused ID extraction (model-agnostic, injected via _id_extractor)
    # ------------------------------------------------------------------

    def _compute_unique_fused_ids(
        self, sparse_features: Any
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Extract unique fused IDs + inverse from a sparse batch.

        返回 ``(unique_ids, inverse, raw_count)``; unique_ids/inverse 留在
        提取器所在设备 (生产路径为 GPU), inverse 供 record_pooled_grad 的
        prepared 路径复用, 免去 runner 侧第二次 unique。
        """
        fused_all = self._id_extractor(sparse_features)
        if fused_all.numel() == 0:
            return fused_all, fused_all, 0
        unique_ids, inverse = torch.unique(fused_all, return_inverse=True)
        return unique_ids, inverse, int(fused_all.numel())

    # ------------------------------------------------------------------
    #  Enqueue (called during batch preparation, ahead of consumption)
    # ------------------------------------------------------------------

    def enqueue(self, sparse_features: Any):
        """Record a batch's unique IDs and pre-issue its PS prefetch.

        Pre-issuing at enqueue (lookahead steps before consume) lets the PS
        process the request while the main stream runs dense compute, so
        ``wait_and_get`` at consume is near-instant.

        Returns ``(unique_ids, inverse, raw_count)`` on the controller device
        — the runner threads it through as the prepared-ids ticket so
        ``record_pooled_grad`` skips its repeat_interleave + re-unique pass,
        or ``None`` for an empty batch (caller falls back to the ordinary
        length-aware path).
        """
        unique_ids, inverse, raw_count = self._compute_unique_fused_ids(
            sparse_features
        )
        batch_num = self._next_enqueue_batch
        self._next_enqueue_batch += 1

        if unique_ids.numel() == 0:
            self._lookahead_ids.append((batch_num, unique_ids))
            return None

        ids_dev = unique_ids.to(self.device, dtype=torch.int64)
        inverse_dev = inverse.to(self.device)
        self._lookahead_ids.append((batch_num, ids_dev))
        self._prepared_ids[batch_num] = (
            ids_dev,
            inverse_dev,
            raw_count,
        )
        compact = self._compact_in_range(self._to_compact(ids_dev))
        # enqueue 批号严格递增, 直接 scatter 即等价于 max(旧值, batch)
        self._latest_dev[compact] = batch_num
        self._preissue_prefetch(batch_num, ids_dev, compact)
        return (ids_dev, inverse_dev, raw_count)

    # ------------------------------------------------------------------
    #  Oracle prescan (opt 8)
    # ------------------------------------------------------------------

    def prescan_batch(self, batch_num: int, sparse_features: Any) -> None:
        """Record a batch's unique IDs during a full-dataset pre-scan."""
        unique_ids, _, _ = self._compute_unique_fused_ids(sparse_features)
        id_set = unique_ids.tolist() if unique_ids.numel() > 0 else []
        self._prescan_unique_ids.update(id_set)
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
        """Fill GPU cache from the pre-issued prefetch handle with TTL tracking.

        First consumes the deferred aggregated-gradient barrier (the previous
        step's dense all_reduce: reduced shared-ID gradients are applied in
        place to the local cache on every rank, and rank0 issues the PS
        persistence push), then fills the cache from the handle pre-issued at
        enqueue time.  Because the PS had ``lookahead`` steps to respond,
        ``wait_and_get`` is near-instant.

        The barrier MUST run here — before the forward lookup — so the
        aggregated apply lands before the next read of those entries.
        """
        t_start = time.perf_counter()
        self._wait_pending_sync_now()
        self._check_gpu_cache_generation()

        if not self._lookahead_ids:
            return
        batch_num, unique_ids = self._lookahead_ids.popleft()
        self._current_batch = batch_num
        if self.embedding_module is not None:
            self.embedding_module._bagpipe_current_prepared_ids = (
                self._prepared_ids.pop(batch_num, None)
            )
            self.embedding_module._bagpipe_cache_generation = self._cache_generation
            self.embedding_module._bagpipe_cache_generation_mismatch = False

        slot = self._prefetch_handles.pop(batch_num, None)
        all_hits = bool(self._prefetch_all_hits.pop(batch_num, False))
        if slot is not None:
            all_hits = self._fill_from_preissued(slot, compute_device)
        elif all_hits and unique_ids.numel() > 0:
            # The all-hit decision was taken at enqueue time, but TTL/capacity
            # eviction, anti-entropy, and the writeback thread can invalidate
            # these ids before they are consumed.  Re-validate immediately
            # ahead of the lock-free hit-only lookup; a stale flag would make
            # the lookup read rows that are no longer resident.
            resident = self.kv_client.contains_gpu_cache(unique_ids)
            if not bool(resident.all().item()):
                all_hits = False
                self._stats["bagpipe_all_hit_revalidations"] += 1.0
                compact = self._to_compact(unique_ids)
                in_range = compact < self._latest_dev.numel()
                compact = compact[in_range]
                self._cached_dev[compact[~resident[in_range]]] = False
        self.embedding_module._bagpipe_all_cache_hits = all_hits
        probe_path = os.environ.get("RS_DEMO_BAGPIPE_PROBE_LOG")
        if probe_path:
            try:
                with open(probe_path, "a", encoding="utf-8") as probe_file:
                    probe_file.write(
                        "[BagPipe-probe] prefill "
                        f"batch={batch_num} slot={slot is not None} "
                        f"all_hits={all_hits}\n"
                    )
            except OSError:
                pass

        self._stats["bagpipe_prefill_ms"] += (time.perf_counter() - t_start) * 1e3

    def _preissue_prefetch(self, batch_num: int, ids_dev: torch.Tensor,
                           compact: torch.Tensor) -> None:
        """Pre-issue the PS prefetch for a batch at enqueue time.

        Determines uncached/expired targets (vectorized mask over the flat
        bookkeeping tensors, indexed by compact id) and issues
        ``kv_client.prefetch`` (non-blocking, fused-id space).  The handle is
        stashed in ``_prefetch_handles`` for ``prefill_cache`` to consume
        ``lookahead`` steps later.

        NOTE: the historical shared-id pruning (dropping local-only ids from
        the prefetch once the shared set was built) is intentionally removed:
        it starved the prefetch of most of its targets (the majority of ids
        are local-only), so the GPU cache never got warmed ahead of the
        lookup and every miss had to be backfilled synchronously.

        Throttled by _max_inflight_prefetch: each pre-issued prefetch is
        chunked by the RDMA client (MaxGetKeysPerRpc keys per RPC, per shard)
        and every chunk holds an RC write slot until the batch is consumed,
        so unbounded pre-issue drained the 64-slot pool (2 shards × 32 QPs ×
        1 slot) and starved the synchronous lookup backfill into "no idle RC
        write slot available".  In steady state the cap does not shorten the
        pipeline: one handle is retired and one issued per step, so batches
        are still pre-issued at enqueue with the full lookahead lead time.
        """
        self._check_gpu_cache_generation()
        if ids_dev.numel() == 0:
            self._prefetch_handles[batch_num] = None
            self._prefetch_all_hits[batch_num] = True
            self._prepared_ids[batch_num] = (
                ids_dev,
                ids_dev,
                0,
            )
            return

        # 向量化命中判定: 驻留且未过期 (compact 索引)。控制流只用 numel
        # 元数据判断, 计数走 GPU 热累加器 —— 全程无 .item() 同步。
        hit = self._cached_dev[compact] & (self._ttl_dev[compact] >= batch_num)
        hit_compact = compact[hit]
        if hit_compact.numel() > 0:
            self._hot_add("bagpipe_prefetch_skip_cached", hit.sum())
            # 命中续期到「最近使用 + margin」(latest 单调不减, 直接赋值即 max)
            self._ttl_dev[hit_compact] = (
                self._latest_dev[hit_compact] + self._ttl_margin
            )

        targets = compact[~hit]
        if targets.numel() == 0:
            # The flat mirror is policy state, not proof of C++ residency.
            # Validate it before allowing the lock-free hit-only lookup.
            contains = getattr(self.kv_client, "contains_gpu_cache", None)
            if not callable(contains):
                raise RuntimeError(
                    "BagPipe requires contains_gpu_cache to validate residency"
                )
            if compact.numel() == ids_dev.numel():
                resident = contains(ids_dev)
                if bool(resident.all().item()):
                    self._prefetch_handles[batch_num] = None
                    self._prefetch_all_hits[batch_num] = True
                    return
                # Keep the mirror convergent and prefetch only real misses.
                self._cached_dev[compact[~resident]] = False
                hit_compact = hit_compact[resident]
                targets = compact[~resident]
            else:
                raise RuntimeError(
                    "BagPipe bookkeeping dropped an in-range fused ID"
                )
            self._prefetch_handles[batch_num] = None
            self._prefetch_all_hits[batch_num] = False
            if targets.numel() == 0:
                return
        self._prefetch_all_hits[batch_num] = False

        outstanding = sum(
            1 for h in self._prefetch_handles.values() if h is not None
        )
        if outstanding >= self._max_inflight_prefetch:
            # 槽池饱和: 该批不预取, 未命中 id 由 lookup 的 C++ miss 回填补齐
            self._prefetch_handles[batch_num] = None
            self._stats["bagpipe_prefetch_throttled"] += 1.0
            probe_path = os.environ.get("RS_DEMO_BAGPIPE_PROBE_LOG")
            if probe_path:
                try:
                    with open(probe_path, "a", encoding="utf-8") as probe_file:
                        probe_file.write(
                            "[BagPipe-probe] throttle "
                            f"batch={batch_num} outstanding={outstanding}\n"
                        )
                except OSError:
                    pass
            return

        # Preserve target/ttl ordering; the RDMA client does not require sorted
        # keys.  This batch is the latest use, so its TTL is batch + margin.
        targets = targets.contiguous()
        ttl = batch_num + self._ttl_margin
        # kv_client / GPU cache 的 key 空间是 fused id
        ids_cpu = self._to_fused(targets).cpu()
        issue_ts = time.perf_counter()

        try:
            handle = self.kv_client.prefetch(ids_cpu)
        except Exception as exc:
            logger.warning("[BagPipe] prefetch pre-issue failed: %s", exc)
            self._prefetch_handles[batch_num] = None
            return

        self._prefetch_handles[batch_num] = PrefetchSlot(
            handle=handle,
            ids_cpu=ids_cpu,
            ttl=ttl,
            issue_ts=issue_ts,
            num_ids=int(ids_cpu.numel()),
        )
        self._stats["bagpipe_prefetch_batches"] += 1
        self._stats["bagpipe_prefetch_ids"] += float(ids_cpu.numel())

    def _fill_from_preissued(
        self, slot: PrefetchSlot, compute_device: torch.device
    ) -> bool:
        """Wait for the pre-issued prefetch result and fill the GPU cache."""
        try:
            values = self.kv_client.wait_and_get(
                slot.handle,
                self.embedding_dim,
                device=compute_device,
            )
        except Exception as exc:
            logger.warning("[BagPipe] prefetch wait failed: %s", exc)
            return False

        ids_cuda = slot.ids_cpu.to(device=compute_device, dtype=torch.int64)
        if not ids_cuda.is_contiguous():
            ids_cuda = ids_cuda.contiguous()
        if not values.is_contiguous():
            values = values.contiguous()

        try:
            prefill_no_evict = getattr(
                self.kv_client, "prefill_gpu_cache_no_evict", None
            )
            if not callable(prefill_no_evict):
                raise RuntimeError(
                    "BagPipe requires prefill_gpu_cache_no_evict to own "
                    "cache eviction"
                )
            inserted = prefill_no_evict(
                self.master_table_name, ids_cuda, values
            )
        except Exception as exc:
            logger.warning("[BagPipe] GPU cache prefill failed: %s", exc)
            return False

        # Only rows that the no-evict kernel actually inserted are resident.
        # This is the authoritative transition into the flat mirror.
        compact = self._to_compact(ids_cuda)
        in_range = compact < self._latest_dev.numel()
        compact = compact[in_range]
        inserted = inserted[in_range]
        compact = compact[inserted]
        self._ttl_dev[compact] = slot.ttl
        self._cached_dev[compact] = True
        inserted_count = int(inserted.sum().item())
        if inserted_count < int(inserted.numel()):
            self._insert_failures += int(inserted.numel()) - inserted_count
            self._stats["bagpipe_insert_failures"] = float(
                self._insert_failures
            )
        return (
            inserted_count == int(inserted.numel())
            and int(inserted.numel()) == int(slot.num_ids)
        )
