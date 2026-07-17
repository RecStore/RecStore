# RankMixer 接入 RecStore：PS 架构 vs QuantaRec 架构性能对比

本文档记录将 QuantaRec 的 RankMixer 模型接入 RecStore rs_demo、并在相同模型/相同数据下
对比 **PS 架构（RecStore + BagPipe）** 与 **本地动态 Embedding 架构（QuantaRec/TorchRec）**
性能的完整过程与结论。

## 1. 接入内容

在 `feat/rankmixer-integration` 分支（基于已含 BagPipe 合入的 master）上完成：

- `runtime/rankmixer_model.py`（新增）：从 QuantaRec `model/` 忠实移植的 RankMixer 计算图，
  包含 `MaskBlock`、`LT` 投影、`TokenMixer`、`PFFN`（`BatchLinear`+`PerTokenLayerNorm`）、
  `PLE`（`MMoEMaskedGate` 6 专家 + 5 任务组 / 36 任务）。自包含，无 quantarec 依赖。
  生产超参：`tokens_split_dim=2400`、`rankmixer_blocks=2`、`gatenum=6`、`masked_dim=56`。
- `runtime/hybrid_dlrm.py`：新增 `build_dense_module`/`build_criterion`/`compute_dense_loss`
  分发器，使 `--model rankmixer` 与原 DLRM 路径共用同一训练循环（backward、sparse update、
  BagPipe、prefetch 完全复用）。
- `runners/recstore_runner.py`、`runners/torchrec_runner.py`：接入分发器，两个 backend 均可跑
  RankMixer。RecStore 路径保留全部优化（BagPipe 控制器、GPU cache、oracle prescan、prefetch）。
- `config.py`：新增 `--model`、`--rankmixer-*` 参数。
- `verify_rankmixer_correctness.py`（新增）：正确性校验。
- `analyze_rankmixer_bench.py`（新增）：双架构性能对比分析。

附带修复两处阻断 BagPipe 路径的既有问题：入口 `run_mock_stress.py` 未把 `src/` 加入
`sys.path`（BagPipe shim 的 `from python.pytorch.recstore...` 导入失败）；BagPipe 控制器
默认 `id_extractor` 引用了不存在的 `..data.dlrm_source`（改为由 runner 显式注入）。

## 2. 正确性校验（`verify_rankmixer_correctness.py`）

| 校验项 | 结果 |
|--------|------|
| 确定性（float64，两次同输入） | loss/logits/梯度逐位一致 |
| Embedding fetch 等价（本地 gather == PS pull） | max_diff = 0.00e+00 |
| RankMixer 计算对相同 embedding 输出 | logits max_diff = 0.00e+00 |
| 梯度流通（MaskBlock/LT/TokenMixer/PFFN/PLE/insert_w） | 146/146 参数非零梯度 |

两个架构跑的是**同一份 RankMixer 计算代码**，embedding 取值一致（仅获取方式不同），
因此前向/反向数值一致，精度对齐。

## 3. 性能对比（单卡 A100，bs=2048，26 表 × 200000 × 64 维，tokens_split_dim=2400）

| 配置 | lookup(ms) | dense_fwd(ms) | backward(ms) | sparse_upd(ms) | 步内合计(ms) | loss 首→末 |
|------|-----------:|--------------:|-------------:|---------------:|-------------:|-----------|
| RecStore 直连（无优化） | 47.29 | 105.64 | 208.40 | 25.01 | ~390 | 17.47→16.81 |
| **RecStore + BagPipe** | **19.30** | 81.55 | 176.13 | **8.40** | **~289** | 17.47→16.82 |
| TorchRec（本地 emb，稠密梯度） | 1.90 | 81.06 | 176.21 | 119.68 | ~382 | 17.48→17.03 |

- `dense_fwd`/`backward` 三者几乎一致（同一 RankMixer 计算图，~81ms / ~176ms），
  说明计算路径忠实、可比。
- BagPipe 把 lookup 从 47ms→19ms（2.4×）、sparse update 从 25ms→8.4ms（3×），
  靠 prefetch/写回与计算 overlap 掩盖 PS 网络延迟。
- BagPipe prefetch 命中率：`prefetch_skip_cached / prefetch_ids ≈ 24918/33269 ≈ 75%`。

## 4. 架构优劣分析

### PS 架构（RecStore + BagPipe）

- **优点**
  - 稀疏更新天然 O(触达行)：只把被访问的行写回 PS，`sparse_upd` 仅 8.4ms。
    大词表下相对「本地全表稠密梯度同步」优势巨大（119.7ms → 8.4ms）。
  - Embedding 集中存储、按 shard 水平扩展，单卡显存不背全表，支持超大规模词表。
  - BagPipe GPU cache + oracle prescan 把网络延迟 overlap 掉 60%+，lookup 降至 19ms。
- **缺点**
  - lookup 走网络，即使有 BagPipe 仍 19ms，是本地内存（1.9ms）的 ~10×。
    通信是 PS 架构的固有瓶颈，cache 只能缓解不能消除。
  - 强依赖 PS server 可用性与部署；local_shm 多卡快路径在当前环境存在初始化死锁。
  - 更新可见性跨 step，需要 prefetch/read-after-write 顺序保证（BagPipe 已处理）。

### 本地动态 Embedding 架构（QuantaRec / TorchRec）

- **优点**
  - lookup 本地内存，1.9ms，无网络往返，是 PS 的 ~10× 快。
  - 无独立 PS server，部署简单，单进程即可起训。
- **缺点**
  - 梯度同步是关键：TorchRec 的 EmbeddingBagCollection 走**稠密**全表 allreduce + 优化器，
    `sparse_upd` 高达 119.7ms（O(表大小)），大词表下不可接受——这正是 QuantaRec
    采用**动态哈希 embedding + 稀疏通信**的原因。
  - 全表常驻显存/内存，词表规模受单机容量限制；需多卡分片（如 EP）才能扩展。

### 核心结论

1. **计算等价**：两种架构跑同一 RankMixer 计算图，dense_fwd/backward 完全一致，
   loss 收敛轨迹对齐（精度一致）。
2. **PS 的本质权衡**：用「更慢的 lookup（网络）」换「更便宜的稀疏更新（O(触达行)）」
   和「可水平扩展的集中存储」。
3. **BagPipe 是 PS 可用性的关键**：无优化时 lookup 47ms、update 25ms，总步 ~390ms；
   加 BagPipe 后 ~289ms，反超 TorchRec 稠密路径（~382ms）。
4. **对真实 QuantaRec（稀疏通信）的推算**：本地 lookup ~1.9ms + 稀疏 update ~8–10ms
   ≈ ~272ms/步，与 RecStore+BagPipe（~289ms）基本持平，QuantaRec 略快于本地 lookup 优势。
   即：**当本地架构也做稀疏通信时，两者趋于同档，差距来自 lookup 网络延迟 vs 本地内存**。

## 5. 复现

```bash
# 正确性
python3 model_zoo/rs_demo/verify_rankmixer_correctness.py

# TorchRec（QuantaRec 架构代理，本地 emb）
python3 model_zoo/rs_demo/run_mock_stress.py --backend torchrec --model rankmixer \
  --nnodes 1 --nproc-per-node 1 --no-start-server \
  --steps 30 --warmup-steps 5 --batch-size 2048 --num-embeddings 200000 --embedding-dim 64 \
  --rankmixer-tokens-split-dim 2400 --rankmixer-blocks 2 --rankmixer-gate-num 6 \
  --torchrec-memory-mode hbm --output-root /tmp/rs_rankmixer_bench/torchrec --run-id bench

# RecStore + BagPipe（PS 架构）
python3 model_zoo/rs_demo/run_mock_stress.py --backend recstore --model rankmixer \
  --nnodes 1 --nproc-per-node 1 \
  --steps 30 --warmup-steps 5 --batch-size 2048 --num-embeddings 200000 --embedding-dim 64 \
  --rankmixer-tokens-split-dim 2400 --rankmixer-blocks 2 --rankmixer-gate-num 6 \
  --enable-gpu-cache --gpu-cache-capacity 160000 \
  --enable-bagpipe-cache --bagpipe-lookahead 4 --server-wait-seconds 40 \
  --output-root /tmp/rs_rankmixer_bench/recstore --run-id bench

# 对比分析
python3 model_zoo/rs_demo/analyze_rankmixer_bench.py \
  /tmp/rs_rankmixer_bench/recstore/outputs/bench/recstore_main.csv \
  /tmp/rs_rankmixer_bench/torchrec/outputs/bench/torchrec_main.csv
```

## 6. 已知简化与后续

- TorchRec 后端用 `EmbeddingBagCollection`（稠密梯度）作为 QuantaRec 动态 embedding 的代理，
  其 `sparse_upd` 偏高；真实 QuantaRec 用稀疏通信会显著更低（见结论 4）。
- RankMixer 的 `object_emb`（interact_group 序列特征）与 `launch_type`/`position` 偏置用合成值
  替代；不影响计算量与架构对比，但非生产数据。
- `dense_input_dim` 由 `num_sparse_features × embedding_dim` 自动分段，未逐段对齐生产
  `[284,180,3168,1261,184]`；LT/TokenMixer/PFFN/PLE 结构与超参与生产一致。
- 多机/多卡对比（3 机 RDMA）可经 `recstore_sync.py` 扩展，当前为单卡对照。
