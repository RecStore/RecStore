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
from .types import PrefetchSlot

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

    # nv_gpu_cache.hpp: SET_ASSOCIATIVITY=2, SLAB_SIZE=32 → 64 槽/set
    _GPU_CACHE_SET_SLOTS = 64

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
        table_sizes: Optional[Dict[str, int]] = None,
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
        # C++ gpu cache 的 capacity 单位是 slab set, 每 set 持有
        # SET_ASSOCIATIVITY(2) × SLAB_SIZE(32) = 64 个 entry 槽
        # (nv_gpu_cache.hpp / nv_gpu_cache.cu num_slot_ 计算)。簿记的
        # 容量压力判定必须换算成 entry 数, 否则会在 1/64 真实容量处
        # 提前触发容量驱逐, 造成无谓的 writeback churn (实测 92.8K →
        # 76.1K samples/s)。
        self.cache_capacity = int(cache_capacity) * self._GPU_CACHE_SET_SLOTS
        self.embedding_dim = int(embedding_dim)
        self.fuse_k = int(fuse_k)
        self.table_offsets = dict(table_offsets)
        self.master_table_name = master_table_name
        self.device = device
        self.lr = float(lr)

        # Model-agnostic ID extractor (caller must provide)
        self._id_extractor = id_extractor

        # ---- Oracle tracking: flat tensors indexed by compact ID ----
        # 原版 BagPipe 的 oracle 用 numpy 扁平数组做全部 per-id 决策
        # (latest_tracker[table][ids] 向量化); 这里用同构的 GPU 张量,
        # 消灭逐 fid 的 Python dict 循环 (实测 12K id/步在 enqueue 路径
        # 花掉 ~15ms)。四张平行张量按 compact id 直接索引:
        #   _latest_dev  int32  最近一次 enqueue 见到该 id 的批次号
        #   _ttl_dev     int32  缓存条目过期批次号 (0 = 过期)
        #   _cached_dev  bool   是否驻留在 GPU cache
        #   _dirty_dev   bool   本地 SGD 已应用、尚未写回 PS
        #
        # fused id = (table_idx << fuse_k) | row (fuse_k=30, 26 张表 →
        # 名义空间 ~2^35, 直接按 fused id 建扁平张量会 OOM)。实际占用
        # 是每表 [t<<fuse_k, t<<fuse_k + cap_t), 因此映射到紧凑的
        # 表内偏移: compact = (table_idx << _bk_shift) | row,
        # stride = next_pow2(max cap), 总空间 = num_tables << shift
        # (~6.8M, 四张张量共 ~68MB)。
        caps = [int(v) for v in (table_sizes or {}).values()] or [1 << 20]
        max_cap = max(caps)
        self._bk_stride = 1 << max(1, max_cap - 1).bit_length()
        self._bk_shift = self._bk_stride.bit_length() - 1
        num_tables = max(1, len(table_offsets))
        self._latest_dev: Optional[torch.Tensor] = None
        self._ttl_dev: Optional[torch.Tensor] = None
        self._cached_dev: Optional[torch.Tensor] = None
        self._dirty_dev: Optional[torch.Tensor] = None
        self._alloc_bookkeeping(num_tables << self._bk_shift)

        # ---- Lookahead buffer (future batch unique ID sets) ----
        self._lookahead_ids: Deque[Tuple[int, torch.Tensor]] = deque()
        self._next_enqueue_batch = 0
        self._current_batch = 0

        # ---- Prefetch pre-issue (opt 3, 10) ----
        # Per-batch prefetch handles pre-issued at enqueue time so that
        # wait_and_get at consume is near-instant (the PS had `lookahead`
        # steps to respond while the main stream ran dense compute).
        # Cap on concurrent in-flight pre-issued prefetches: each prefetch is
        # chunked by the RDMA client (1600 keys/RPC at dim=128, ×2 shards) and
        # every chunk holds an RC write slot until the batch is consumed —
        # one 12K-id prefetch ≈ 8 slots, so the 64-slot pool (2 shards ×
        # 32 QPs × 1 slot) supports only a handful concurrently.  The cap
        # does NOT shorten the pipeline: in steady state one handle is
        # retired (consumed) and one issued per step, so every batch is still
        # pre-issued at enqueue, lookahead steps ahead of its consumption.
        self._prefetch_handles: Dict[int, Optional[PrefetchSlot]] = {}
        self._max_inflight_prefetch = 3
        # TTL margin beyond the last-seen batch: ids in this dataset recur
        # only every ~epoch_length batches, so a TTL of exactly "last use"
        # expires every entry between recurrences and the cache collapses to
        # the current window (measured: mean 1116 entries at capacity 160000,
        # i.e. every id re-fetched from the PS on each recurrence).  The
        # margin keeps entries resident across one recurrence gap; capacity
        # is protected by a high-watermark fallback eviction in cleanup().
        self._ttl_margin = 256
        self._capacity_watermark = 0.9
        # 单次 cleanup 的到期驱逐上限: 防止冷启动后一次性弹出数万条目
        # 造成 eviction 尖峰 (writeback 队列也会随之堆积)。超出的条目
        # 保持旧 ttl, 下一次 cleanup 自然再被扫出。
        self._max_evict_per_cleanup = 4096

        # ---- Stats ----
        self._stats: Dict[str, float] = {}
        # 热路径计数 (每步多次累加) 用 GPU 标量张量持有, 避免在
        # update_grads / _preissue_prefetch 中途 .item() 强制排空设备队列;
        # consume_stats 在 timer.finish() (device drain) 之后统一取回。
        self._hot_stats_dev: Dict[str, torch.Tensor] = {
            k: torch.zeros((), dtype=torch.float64, device=device)
            for k in (
                "bagpipe_no_sync_ids",
                "bagpipe_sync_now_ids",
                "bagpipe_prefetch_skip_cached",
            )
        }
        self.reset_stats()

        # ---- Background cleanup thread ----
        self._cleanup_queue: queue.Queue = queue.Queue()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="bagpipe-cleanup"
        )
        self._cleanup_thread.start()

        # ---- Overlap state: async all_reduce on a separate CUDA stream ----
        # 聚合梯度的 dense all_reduce 统一在侧流上发射 (与主流计算重叠),
        # 下一步 prefill 的 barrier 处等待并原位落 cache。
        self._sync_later_stream: Optional[torch.cuda.Stream] = None
        self._eviction_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._sync_later_stream = torch.cuda.Stream(device=self.device)
            self._eviction_stream = torch.cuda.Stream(device=self.device)
        self._pending_sync_now_work = None
        self._pending_sync_now_lr: Optional[float] = None
        # rank0 聚合推送 PS 的深度-1 流水线句柄: 持久化不在一致性关键路径
        # 上, 但必须保序 (等上一条完成再发下一条)。
        self._pending_ps_push_handle: Optional[int] = None

        # ---- Anti-entropy backstop (fill-vs-push ordering race) ----
        self._anti_entropy_step = 0
        self._anti_entropy_interval = 50
        self._anti_entropy_ids = 200

        # ---- no_sync: shared vs local-only IDs ----
        self._shared_ids: Optional[Set[int]] = None
        self._shared_ids_tensor: Optional[torch.Tensor] = None
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
            "bagpipe_prefetch_throttled": 0.0,
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
            "bagpipe_anti_entropy_ids": 0.0,
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
        # 热计数器在此取回 (.item() 位于 timer.finish() 的 device drain 之后,
        # 不在关键路径上引入同步)
        for key, acc in self._hot_stats_dev.items():
            self._stats[key] = self._stats.get(key, 0.0) + float(acc.item())
            if reset:
                acc.zero_()
        self._stats["bagpipe_cache_entries"] = float(self._cached_count())
        self._stats["bagpipe_dirty_entries"] = float(self._dirty_count())
        stats = dict(self._stats)
        if reset:
            self.reset_stats()
        return stats

    # ------------------------------------------------------------------
    #  Flat-tensor bookkeeping (原版 oracle 的 numpy 扁平数组同构)
    # ------------------------------------------------------------------

    def _alloc_bookkeeping(self, size: int) -> None:
        """(Re)allocate the four parallel bookkeeping tensors, copying old contents."""
        dev = self.device
        latest = torch.zeros(size, dtype=torch.int32, device=dev)
        ttl = torch.zeros(size, dtype=torch.int32, device=dev)
        cached = torch.zeros(size, dtype=torch.bool, device=dev)
        dirty = torch.zeros(size, dtype=torch.bool, device=dev)
        if self._latest_dev is not None:
            n = self._latest_dev.numel()
            latest[:n] = self._latest_dev
            ttl[:n] = self._ttl_dev
            cached[:n] = self._cached_dev
            dirty[:n] = self._dirty_dev
        self._latest_dev = latest
        self._ttl_dev = ttl
        self._cached_dev = cached
        self._dirty_dev = dirty

    def _to_compact(self, fused: torch.Tensor) -> torch.Tensor:
        """fused id → compact 表内偏移索引 (向量化, 保序)."""
        row = fused & ((1 << self.fuse_k) - 1)
        return ((fused >> self.fuse_k) << self._bk_shift) + row

    def _to_fused(self, compact: torch.Tensor) -> torch.Tensor:
        """compact 索引 → fused id (向量化, 保序; _to_compact 的逆)."""
        row = compact & (self._bk_stride - 1)
        return ((compact >> self._bk_shift) << self.fuse_k) + row

    def _compact_in_range(self, compact: torch.Tensor) -> torch.Tensor:
        """过滤超出簿记空间的 compact id (未知表/越界 row, 不应出现)。"""
        return compact[compact < self._latest_dev.numel()]

    def _hot_add(self, key: str, count: torch.Tensor) -> None:
        """热路径计数累加: 优先 GPU 标量 (无同步), 否则回退 host float。"""
        acc = self._hot_stats_dev.get(key)
        if acc is not None:
            acc += count.to(dtype=acc.dtype)
        else:
            self._stats[key] = self._stats.get(key, 0.0) + float(count)

    def _cached_count(self) -> int:
        if self._cached_dev is None:
            return 0
        return int(self._cached_dev.sum().item())

    def _dirty_count(self) -> int:
        if self._dirty_dev is None:
            return 0
        return int(self._dirty_dev.sum().item())
