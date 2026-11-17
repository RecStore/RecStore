# Embedding 模块

模块划分与选择（`RecStoreEmbeddingBagCollection` vs `DistEmbedding`）见 [计算层：Embedding 与 Optimizer](../calc/embedding_optimizer.md)。本文只列 Python API。

## RecStoreEmbeddingBagCollection

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
