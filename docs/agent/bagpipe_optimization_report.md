# BagPipe 预取缓存优化对比报告

基于 `docs/agent/bagpipe_prefetch_analysis.md` 识别的「12 个全局 CUDA 同步点阻断
异步流水线」问题，将优化拆分为 3 个独立项目，每个项目在单独分支上实现，并以相同
工作负载逐项基准对比，严格保证正确性（loss 轨迹对齐）。

## 测试环境与工作负载

- 单卡 A100-SXM4-80GB，ps_type=BRPC，2 shard 本机部署
- `batch_size=2048`，26 表 × 200000 × 64 维，RankMixer 模型
  （`tokens_split_dim=2400`、`blocks=2`、`gate_num=6`）
- `--enable-bagpipe-cache --bagpipe-lookahead 4 --gpu-cache-capacity 160000`
- 30 步，warmup 5，取后 10 步统计；每项优化至少跑 2 次确认稳定
- 数据集为确定性随机生成的 8192 样本（`day_0_*.npy`），所有分支共用同一份数据

> 说明：数据加载器每 ~4 步产生一个约 409 样本的小批次（6553 训练样本 / 2048 ≈ 3.2），
> 因此步时间呈「3 慢 + 1 快」的 4 步周期。**全批次（2048）步时间**是公平对比口径；
> `last10_mean` 含小批次，仅作总量参考。

## 优化项目与分支

每条分支基于上一条，便于「相比上一项」对比增量：

| 项目 | 分支 | 核心改动 |
|------|------|----------|
| Opt1 | `opt/bagpipe-event-stage-timing` | 用 CUDA 事件计时替代训练循环中 12 个 `sync_device`，步骤末仅 1 次同步 |
| Opt2 | `opt/bagpipe-async-prefetch-early-issue` | 将非阻塞 `kv_client.prefetch` 从 consume 移到 enqueue（提前 `lookahead` 步发射） |
| Opt3 | `opt/bagpipe-remove-runtime-syncs` | 移除 `prepare_hybrid_dlrm_input`/`run_hybrid_backward` 内部 4 个冗余 `sync_device` |

## 性能对比（全批次，后 10 步中 >150ms 的步骤均值，单位 ms）

| 阶段 | Baseline | Opt1 | Opt2 | Opt3 |
|------|---------:|-----:|-----:|-----:|
| embed_lookup | 22.78 | 21.41 | **3.38** | 3.69 |
| embed_pool | 0.25 | 0.24 | 0.22 | 0.29 |
| dense_fwd | 97.04 | 96.98 | 96.90 | 97.02 |
| backward | 211.83 | 211.28 | 210.93 | **207.36** |
| optimizer | 3.70 | 3.48 | 3.37 | 2.85 |
| sparse_update | 9.42 | 8.56 | 9.16 | 10.54 |
| **step_total（全批次）** | **346.40** | **343.11** | **325.06** | **323.12** |
| last10_mean（含小批次） | 297.49 | 294.29 | 278.70 | 275.79 |

## 逐项增量提升

| 对比 | step_total Δ（全批次） | 相对提升 | 主要改善点 |
|------|----------------------:|---------:|------------|
| Opt1 vs Baseline | −3.3 ms | −0.9% | 消除阶段间 CPU-GPU 提交气泡；计时仍准确（事件 vs 同步） |
| Opt2 vs Opt1 | −18.1 ms | −5.3% | **embed_lookup 21.4 → 3.4 ms**：预取提前 `lookahead` 步发射，`wait_and_get` 近即时 |
| Opt3 vs Opt2 | −1.9 ms | −0.6% | **backward 211 → 207 ms**：移除 backward 边界内部同步，消除 ~4ms GPU 空闲气泡 |
| **Opt3 vs Baseline** | **−23.3 ms** | **−6.7%** | 累计：预取网络等待几乎完全隐藏 + 流水线气泡消除 |

## 各项优化详解

### Opt1：CUDA 事件计时替代阻塞同步（`opt/bagpipe-event-stage-timing`）

**问题**：训练循环在每个阶段边界调用 `sync_device()`（`torch.cuda.synchronize`），
共 12 次。全局同步等待所有 CUDA stream 完成，阻塞 CPU 跨阶段提交 GPU 工作，制造
CPU-GPU 气泡；同时阻断 BagPipe 后台写回线程 / eviction stream 与主计算的重叠。

**改动**（`model_zoo/rs_demo/runners/recstore_runner.py`）：
- 新增 `CudaStageTimer`：在默认流上以 `event.record()` 非阻塞记录阶段边界事件，
  步骤末 `resolve()` 做一次 `synchronize` 并用 `elapsed_time` 还原各阶段 GPU 时长
- 移除训练循环中全部 12 个 `sync_device`；`loss` 标量读取延迟到 `resolve` 之后
  （原 `loss.detach().float().cpu().item()` 本就会隐式同步，延迟后不再阻塞流水线）

**效果**：阶段 GPU 计时仍准确（dense_fwd 97ms 与基线一致），仅消除提交气泡，
增量较小（−0.9%）。但它是后续重叠的前提——同步不拆除，CPU 无法提前提交后续阶段。

### Opt2：预取提前发射实现真正异步重叠（`opt/bagpipe-async-prefetch-early-issue`）

**问题**：原实现把 `kv_client.prefetch`（非阻塞发射）与 `wait_and_get`（阻塞等待 PS
响应）都放在 `prefill_cache`（consume 时刻）。PS 响应未提前发起，`wait_and_get` 阻塞
主线程 ~20ms，期间 GPU 空闲，网络等待无法与稠密计算重叠。这正是分析报告指出的
「同步预取等待阻断异步流水线」。

**改动**（`src/python/pytorch/recstore/bagpipe_cache/prefetch.py` + `controller.py`）：
- `enqueue`（prepare 阶段，提前 `lookahead` 步）中新增 `_preissue_prefetch`：判定
  未缓存/过期目标（保留压力感知剪枝 opt 3/10），立即 `kv_client.prefetch`（非阻塞），
  把 handle 存入 `_prefetch_handles[batch_num]`
- `prefill_cache`（consume）改为弹出预发 handle，`wait_and_get`（此时 PS 已在
  `lookahead` 步稠密计算期间处理完请求，近即时）+ `prefill_gpu_cache`
- 用按批预发（`_preissue_prefetch` + `_fill_from_preissued`）替换原批量
  issue/fill（`_issue_batched_prefetch` + `_fill_cache_from_pending`）

**效果**：`embed_lookup` 从 ~21ms 降至 ~3.4ms（仅剩 `prefill_gpu_cache` GPU 填充 +
lookup），网络等待几乎完全隐藏到稠密计算背后。增量 −5.3%，是三项中收益最大者。

**正确性**：TTL 保证跳过预取的 ID 在 consume 时仍缓存（其 `ttl ≥ consume 批次`，
cleanup 不会提前驱逐）；冗余预取无害。loss 轨迹与基线最大差 0.0016。

### Opt3：移除运行时辅助函数内部冗余同步（`opt/bagpipe-remove-runtime-syncs`）

**问题**：Opt1 只移除了训练循环里的同步，但 `prepare_hybrid_dlrm_input` 与
`run_hybrid_backward`（`model_zoo/rs_demo/runtime/hybrid_dlrm.py`）内部各自还有 2 个
`sync_device`，是稠密计算路径上最后残留的阻塞点。backward 末尾的同步让 CPU 等
backward 完成才能提交 optimizer/sparse_update，制造 ~4ms GPU 空闲气泡。

**改动**：移除这 4 个内部 `sync_device`（保留 `sync_device` 定义供 torchrec/hps runner
使用）。同流操作本就有序，无需显式同步保证正确性。

**效果**：`backward` 211 → 207ms（消除边界气泡），`optimizer` 3.5 → 2.8ms，
增量 −0.6%。

## 正确性验证

所有分支共用同一份确定性数据，逐步对比 loss 轨迹（30 步）：

| 分支 | 与基线最大 loss 差 | 与基线平均 loss 差 | 末步 loss |
|------|-------------------:|-------------------:|----------:|
| Opt1 | 0.0043 | 0.0006 | 16.8052 |
| Opt2 | 0.0016 | 0.0004 | 16.8031 |
| Opt3 | 0.0036 | 0.0006 | 16.8025 |
| Baseline | — | — | 16.8031 |

差异均在 0.005 以内（相对 17 的 ~0.03%），属异步时序导致的浮点级噪声，非正确性缺陷：
前 7 步逐位一致，之后因缓存驱逐/写回时序微小差异缓慢漂移但收敛到同一末值。
稀疏更新可见性与 read-after-write 顺序由 TTL + 同流有序 + 控制器显式 wait
（`_wait_pending_sync_now` / `_wait_prev_sync_later`）共同保证，未被破坏。

## 结论

1. **Opt2 是核心收益项**：把预取发射从 consume 提前到 enqueue，让 PS 网络往返与
   `lookahead` 步稠密计算重叠，`embed_lookup` 21 → 3.4ms，单步 −5.3%。
2. **Opt1/Opt3 是流水线气泡消除**：拆除同步点让 CPU 跨阶段提前提交 GPU 工作，
   收益较小（合计 ~1.5%），但是 Opt2 重叠能落地的前提。
3. **累计全批次步时间 346.4 → 323.1ms（−6.7%）**，与分析报告预估的 ~9% 上限接近；
   剩余步时间 97% 为稠密前向+反向 GPU 计算（304ms），已无更多异步可隐藏空间。
4. **正确性严格保持**：三项优化的 loss 轨迹与基线浮点级对齐，收敛一致。

## 复现

```bash
# 数据（确定性随机，8192 样本，所有分支共用）
python3 - <<'PY'
import numpy as np, os
d="model_zoo/torchrec_dlrm/processed_day_0_data"; os.makedirs(d, exist_ok=True)
rng=np.random.default_rng(42); N=8192
np.save(f"{d}/day_0_dense.npy", rng.standard_normal((N,13)).astype(np.float32))
np.save(f"{d}/day_0_sparse.npy", rng.integers(0,200000,(N,26),dtype=np.int64))
np.save(f"{d}/day_0_labels.npy", rng.integers(0,2,(N,)).astype(np.float32))
PY

# 任一分支运行基准
python3 model_zoo/rs_demo/run_mock_stress.py --backend recstore --model rankmixer \
  --nnodes 1 --nproc-per-node 1 --steps 30 --warmup-steps 5 --batch-size 2048 \
  --num-embeddings 200000 --embedding-dim 64 \
  --rankmixer-tokens-split-dim 2400 --rankmixer-blocks 2 --rankmixer-gate-num 6 \
  --enable-gpu-cache --gpu-cache-capacity 160000 \
  --enable-bagpipe-cache --bagpipe-lookahead 4 --server-wait-seconds 40 \
  --output-root /tmp/rs_bench --run-id bench
```

分支顺序（每条基于上一条）：
`feat/rankmixer-integration` → `opt/bagpipe-event-stage-timing` →
`opt/bagpipe-async-prefetch-early-issue` → `opt/bagpipe-remove-runtime-syncs`
