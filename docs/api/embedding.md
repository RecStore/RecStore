# Embedding 模块

计算层的主入口是 `RecStoreEmbeddingBagCollection`。`DistEmbedding` 是单表封装，适合简单实验和兼容旧训练代码；它的更新路径和 EBC 不一样。

| 模块 | 文件 | 使用场景 |
|------|------|----------|
| `RecStoreEmbeddingBagCollection` | `src/python/pytorch/torchrec_kv/EmbeddingBag.py` | TorchRec 风格多表、多特征模型。当前模型接入优先看它。 |
| `DistEmbedding` | `src/python/pytorch/recstore/DistEmb.py` | 单个 embedding table。通过 `DistTensor` 读写，optimizer 在 Python 侧做 SGD 写回。 |

### RecStoreEmbeddingBagCollection

```python
module = RecStoreEmbeddingBagCollection(
    embedding_bag_configs=configs,
    lr=0.01,
    enable_fusion=True,
    fusion_k=30,
)
```

输入是 `KeyedJaggedTensor`。模块按 feature 找到 table，读取 ids 对应的 embedding，再用 `torch.nn.functional.embedding_bag` 做 sum pooling。

### DistEmbedding

```python
emb = DistEmbedding(
    num_embeddings=100000,
    embedding_dim=64,
    name="user_embedding",
    init_func=None,
    lr=0.01,
)
```

`forward(ids)` 通过 `DistTensor.__getitem__` 调 `client.pull`。backward 会对重复 id 做 `torch.unique + index_add_`，把聚合后的 `(ids, grads)` 写入 `_trace`。

实现与更新流程见 [Embedding 与 Optimizer](../calc/embedding_optimizer.md)。
