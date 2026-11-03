# Embedding 与 Optimizer

计算层的主入口是 `RecStoreEmbeddingBagCollection`。`DistEmbedding` 是单表封装，适合简单实验和兼容旧训练代码；它的更新路径和 EBC 不一样。

| 模块 | 文件 | 使用场景 |
|------|------|----------|
| `RecStoreEmbeddingBagCollection` | `src/python/pytorch/torchrec_kv/EmbeddingBag.py` | TorchRec 风格多表、多特征模型。当前模型接入优先看它。 |
| `DistEmbedding` | `src/python/pytorch/recstore/DistEmb.py` | 单个 embedding table。通过 `DistTensor` 读写，optimizer 在 Python 侧做 SGD 写回。 |

## Embedding 模块

构造示例与输入说明见 [Embedding API](../api/embedding.md)。

## fused id

开启 fusion 后，表前缀写进高位：`fused_id = raw_id + (table_idx << fusion_k)`。

改 `fusion_k` 时要确认 raw id 不会覆盖表前缀。

## 预取

```python
handle = module.issue_fused_prefetch(features)
module.set_fused_prefetch_handle(handle)
out = module(features)
```

`issue_fused_prefetch` 去重 fused ids 并返回 handle。forward 看到 `_fused_prefetch_handle` 后等待结果，再按 inverse index 恢复原始顺序。没有 handle 时走同步读取。

## 梯度和 optimizer

Python 侧 optimizer 先消费模块 `_trace`。EBC 和 `DistEmbedding` 后续更新位置不同。

| 顺序 | 操作 | 文件 |
|------|------|------|
| 1 | forward 读取 embedding。 | `torchrec_kv/EmbeddingBag.py`、`recstore/DistEmb.py` |
| 2 | backward 写入 `module._trace`。 | 同上 |
| 3 | `SparseOptimizer.step()` 读取并聚合 trace。 | `recstore/optimizer.py` |
| 4 | EBC 调 `RecStoreClient.update_async(table, ids, grads)`；`DistEmbedding` 读当前权重并在 Python 侧计算新值。 | `recstore/optimizer.py` |
| 5 | EBC 进入 `KVClientOp::EmbUpdate` 和 PS optimizer；`DistEmbedding` 写回 `DistTensor`。 | `src/framework/op.cc`、`src/optimizer/optimizer.cpp`、`recstore/DistTensor.py` |

| 模块 | trace item |
|------|------------|
| `RecStoreEmbeddingBagCollection` | `(table_name, ids, grads)` |
| `DistEmbedding` | `(ids, grads)` |

`SparseOptimizer.step()` 处理后会清理 trace。跳过这一步会让下一轮 batch 混入旧梯度。

### EBC 更新

`SparseSGD` 聚合 `_trace` 后调用 `kv_client.update_async(name, ids, grads)`。学习率和优化器公式由后端处理。local_shm / HierKV 快路径会走 `local_update_flat`。

### DistEmbedding 更新

`SparseSGD` 会合并重复 id，读取 `mod.weight[unique_ids]`，在 Python 侧执行 `current - lr * grad`，再写回 `DistTensor`。这条路径没有使用后端 optimizer 状态。

??? note "后端 optimizer"

    后端实现位于 `src/optimizer/optimizer.cpp`。SGD 无额外状态；AdaGrad 和 RowWiseAdaGrad 的累积状态都在后端维护。EBC 的远端更新应保持这个边界。

## 修改检查

改 trace 结构：同步更新 `SparseOptimizer` 和相关单测。

改后端 optimizer：补跑 EBC / PS 更新路径测试，确认 read-modify-write 顺序。

改 `DistEmbedding`：检查 `DistTensor.__getitem__` / `__setitem__` 和 Python 侧 SGD 聚合。

改本地快路径：检查 `local_update_flat` 和 RPC fallback。
