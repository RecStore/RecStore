"""BagPipeGradMixin — gradient update logic with shared / no_sync split.

Extracted from controller.py.  The overlap barrier (_wait_pending_sync_now)
is tightly coupled to update_grads because it waits for the async all_reduce
launched in the previous step's update_grads call.

协议（P0b 重构后, now/later 已合并）:

- local-only id:  立即 best-effort 原位 SGD 落本地 cache（无 Query、无同步、
                   缺 key 静默跳过），PS 持久化走 dirty 标记 + eviction 写回。
- shared id:      本地不 apply（避免本地梯度与聚合梯度双重计入）；梯度进
                   单次 dense all_reduce（侧流发射，与主流计算重叠），
                   **所有 rank** 在下一步 prefill 的 barrier 处用聚合梯度
                   best-effort 原位落 cache。副本收敛 =
                   同一 PS 初值 + 同一聚合增量序列。rank0 推 PS 降级为
                   持久化（深度-1 异步流水线），不在一致性关键路径上。
- 失效:           每步不再失效共享 id（原实现 rank≠0 每步 invalidate ~3.6K
                   热门共享 id，是 embed_lookup 41ms 的主因）。barrier 先于
                   prefill 执行，填充读到的已是推送后的 PS 值；残余的 RDMA
                   服务端排序竞态由反熵兜底（_anti_entropy_maybe：每
                   _anti_entropy_interval 步随机失效少量共享缓存 id 强制重拉）。

旧的 all-or-nothing apply_sgd_update_gpu_cache（内部 Query 强制
cudaStreamSynchronize + 全 rank 成功投票 all_reduce）连同 fallback 路径一并
移除：best-effort op 永不失败，缺失的 key 下一步 prefill 自然补齐。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BagPipeGradMixin:
    """Provides update_grads and the associated overlap barrier.

    Expects the host class to provide (via other mixins or __init__):
        - self.device, self.kv_client, self.master_table_name, self.lr
        - self._latest_dev, self._ttl_dev, self._cached_dev, self._dirty_dev
        - self._to_compact(), _to_fused(), _compact_in_range()
        - self._sync_later_stream, self._pending_ps_push_handle
        - self._pending_sync_now_work, self._pending_sync_now_lr
        - self._anti_entropy_step, _anti_entropy_interval, _anti_entropy_ids
        - self._shared_ids, self._shared_ids_tensor, self._stats
        - self._dense_all_reduce_async(), _maybe_build_shared_id_set()
        - self._hot_add(), self._is_distributed(), _get_rank()
    """

    # ------------------------------------------------------------------
    #  Aggregated-gradient apply (all ranks converge on the same deltas)
    # ------------------------------------------------------------------

    def _apply_aggregated(self, agg_ids: torch.Tensor,
                          agg_grads: torch.Tensor, lr: float) -> None:
        """All ranks: apply aggregated shared-ID gradients to the local cache.

        The replicas converge because every rank starts from the same PS value
        and applies the same sequence of aggregated increments.  rank0 also
        pushes the aggregate to the PS, but only as persistence — no rank
        invalidates on this path.  持久化推送走深度-1 异步流水线 (等上一条
        完成再发下一条, 保 PS 侧顺序), 不在一致性关键路径上同步等待。
        """
        agg_ids = agg_ids.to(self.device, dtype=torch.int64)
        agg_grads = agg_grads.to(self.device, dtype=torch.float32)
        if agg_grads.dim() == 1:
            agg_grads = agg_grads.unsqueeze(1)
        if not agg_ids.is_contiguous():
            agg_ids = agg_ids.contiguous()
        if not agg_grads.is_contiguous():
            agg_grads = agg_grads.contiguous()
        if agg_ids.numel() == 0:
            return

        try:
            self.kv_client.apply_sgd_update_gpu_cache_best_effort(
                self.master_table_name, agg_ids, agg_grads,
                learning_rate=lr if lr is not None else self.lr,
            )
        except Exception as exc:
            logger.warning("[BagPipe] aggregated best-effort apply failed: %s", exc)

        if not self._is_distributed() or self._get_rank() == 0:
            prev = getattr(self, "_pending_ps_push_handle", None)
            if prev is not None:
                try:
                    self.kv_client.wait(prev)
                except Exception as exc:
                    logger.warning("[BagPipe] previous PS push wait failed: %s", exc)
            try:
                self._pending_ps_push_handle = self.kv_client.update_async(
                    self.master_table_name, agg_ids, agg_grads
                )
            except Exception as exc:
                self._pending_ps_push_handle = None
                logger.warning("[BagPipe] aggregated PS push failed: %s", exc)

    # ------------------------------------------------------------------
    #  Anti-entropy backstop
    # ------------------------------------------------------------------

    def _anti_entropy_maybe(self) -> None:
        """Every _anti_entropy_interval steps, invalidate a small random sample
        of cached shared IDs so any replica that diverged through the
        fill-vs-push RDMA ordering race is forced to refetch from the PS.

        Expected divergence lifetime is bounded by interval * (shared / sample)
        steps; cost is a few hundred invalidations + refetches per interval,
        amortized to ~nothing per step.
        """
        self._anti_entropy_step += 1
        if self._anti_entropy_step % self._anti_entropy_interval != 0:
            return
        shared_t = self._shared_ids_tensor
        if shared_t is None or shared_t.numel() == 0:
            return
        # 从 shared 集随机采样并过滤出已驻留的 (向量化; 簿记是 compact 索引)
        k = min(self._anti_entropy_ids, int(shared_t.numel()))
        perm = torch.randperm(
            int(shared_t.numel()), device=shared_t.device
        )[: k * 4]
        cand = shared_t[perm]
        cand_compact = self._compact_in_range(self._to_compact(cand))
        cand_compact = cand_compact[self._cached_dev[cand_compact]]
        if cand_compact.numel() == 0:
            return
        sample = cand_compact[:k].contiguous()
        try:
            self.kv_client.invalidate_gpu_cache(
                self.master_table_name, self._to_fused(sample)
            )
        except Exception as exc:
            logger.warning("[BagPipe] anti-entropy invalidate failed: %s", exc)
            return
        self._cached_dev[sample] = False
        self._dirty_dev[sample] = False
        self._ttl_dev[sample] = 0
        self._stats["bagpipe_anti_entropy_ids"] += float(sample.numel())

    # ------------------------------------------------------------------
    #  Overlap barrier (aggregated all_reduce of the previous step)
    # ------------------------------------------------------------------

    def _wait_pending_sync_now(self) -> None:
        """Wait for the previous step's aggregated-gradient all_reduce.

        Consumed at the top of prefill_cache, before the forward lookup: the
        aggregated shared-ID gradients are applied in place on every rank, so
        the entries stay cached and valid — no invalidation, no refetch.
        all_reduce 在侧流 (_sync_later_stream) 上发射, 与主流计算重叠;
        apply 是主流 kernel, 需先同步侧流再读聚合结果。
        """
        self._anti_entropy_maybe()

        work = self._pending_sync_now_work
        if work is None:
            self._pending_sync_now_lr = None
            return
        self._pending_sync_now_work = None
        lr = self._pending_sync_now_lr
        self._pending_sync_now_lr = None

        t_start = time.perf_counter()
        work.wait()
        if self._sync_later_stream is not None:
            self._sync_later_stream.synchronize()
        agg_ids, agg_grads = work.result
        self._apply_aggregated(agg_ids, agg_grads, lr)
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
        """Split gradients into local-only (immediate best-effort SGD) and
        shared (dense all_reduce, aggregated apply at the next barrier).

        Note: the deferred barrier is consumed in prefill_cache (before the
        forward lookup), NOT here.

        热路径上没有任何 ``.item()``/``.cpu()`` —— 控制流只依赖 numel 元数据,
        计数走 GPU 热累加器 (``_hot_add``), 在 consume_stats 的 device-drain
        点统一取回。原 now/later 双路 all_reduce 已合并为单次: 两条路都在
        下一步 prefill 才被消费, 调度完全相同, 区分只剩一次多余的 6MB
        all_reduce 与逐 id 判定开销。
        """
        t_start = time.perf_counter()

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

        self._stats["bagpipe_sgd_cache_success"] += 1

        self._maybe_build_shared_id_set(ids_cuda)

        # ---- shared / local-only 向量化切分 (sorted 张量 + searchsorted,
        #      与 _dense_all_reduce_async 同构, 无 .item()) ----
        shared_t = self._shared_ids_tensor
        if shared_t is not None and shared_t.numel() > 0:
            pos = torch.searchsorted(shared_t, ids_cuda)
            pos_clamped = pos.clamp(max=shared_t.numel() - 1)
            shared_mask = shared_t[pos_clamped] == ids_cuda
        else:
            shared_mask = torch.zeros_like(ids_cuda, dtype=torch.bool)
        local_mask = ~shared_mask

        # ---- no_sync (local-only): 立即 best-effort 原位 SGD ----
        self._hot_add("bagpipe_no_sync_ids", local_mask.sum())
        local_ids = ids_cuda[local_mask]
        if local_ids.numel() > 0:
            local_grads = grads_cuda[local_mask]
            try:
                self.kv_client.apply_sgd_update_gpu_cache_best_effort(
                    table_name, local_ids, local_grads, learning_rate=lr
                )
            except Exception as exc:
                logger.warning("[BagPipe] local best-effort apply failed: %s", exc)
            if self._shared_ids is None:
                # 共享集合尚未构建（前 lookahead 个 step）：直接推 PS
                try:
                    self.kv_client.update(
                        self.master_table_name, local_ids, local_grads
                    )
                except Exception as exc:
                    logger.warning("[BagPipe] no_sync push failed: %s", exc)
                local_compact = self._compact_in_range(
                    self._to_compact(local_ids)
                )
                self._cached_dev[local_compact] = False
                self._dirty_dev[local_compact] = False
            else:
                # PS 持久化走 dirty 张量 + eviction 值写回 (scatter, 无 tolist)。
                # 本批 id 刚经过 lookup (C++ miss 路径会回填 GPU cache,
                # op_torch.cc gpu_cache_lookup_flat), best-effort apply 必然
                # 命中; 此处同步登记驻留 + 续期 TTL —— 簿记对 lookup 回填
                # 是盲的, 不登记则这些条目既不参与命中判定也永不写回。
                local_compact = self._compact_in_range(
                    self._to_compact(local_ids)
                )
                self._dirty_dev[local_compact] = True
                self._cached_dev[local_compact] = True
                self._ttl_dev[local_compact] = batch_num + self._ttl_margin

        # ---- shared: 本地不 apply；单次 dense all_reduce (侧流, 与主流
        #      计算重叠), 聚合后全 rank 在下一步 prefill 落 cache ----
        shared_ids = ids_cuda[shared_mask]
        if shared_ids.numel() == 0:
            self._pending_sync_now_work = None
            self._pending_sync_now_lr = None
            self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
            return
        self._hot_add("bagpipe_sync_now_ids", shared_mask.sum())

        shared_grads = grads_cuda[shared_mask]
        _, _, work = self._dense_all_reduce_async(
            shared_ids, shared_grads, stream=self._sync_later_stream
        )
        self._pending_sync_now_work = work
        self._pending_sync_now_lr = lr

        self._stats["bagpipe_update_ms"] += (time.perf_counter() - t_start) * 1e3
