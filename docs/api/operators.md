# PyTorch Operators

!!! note "待重写"

    按当前算子签名更新参数表；init_embedding_table 示例仍是旧的三参数形式。

文件：`src/framework/pytorch/op_torch.cc`

| op | 作用 |
|----|------|
| `emb_read(keys, embedding_dim)` | 同步读取，返回 `[N, D]`。 |
| `emb_write(keys, values)` | 同步写入。 |
| `emb_update_table(table, keys, grads)` | 带表名更新。 |
| `init_embedding_table(name, num_embeddings, embedding_dim)` | 创建后端表。 |
| `emb_prefetch(keys)` / `emb_wait_result(handle, embedding_dim)` | 预取。 |
| `set_ps_config(host, port)` / `set_ps_backend(backend)` | 连接与后端选择。 |
| `local_lookup_flat(keys, embedding_dim)` | 本地快路径读取。 |
| `local_update_flat(table, keys, grads)` | 本地快路径更新。 |

张量要求：

| 参数 | 要求 |
|------|------|
| `keys` | int64，1D，contiguous。 |
| `values` | float32，2D，contiguous，行数等于 `keys.numel()`。 |
| `grads` | float32，2D，contiguous。 |
| `embedding_dim` | 大于 0。 |

CUDA tensor 会复制到 CPU 后调用 `CommonOp`。读取结果按输入设备返回。
