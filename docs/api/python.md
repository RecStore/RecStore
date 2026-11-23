# RecStoreClient

文件：`src/python/pytorch/recstore/KVClient.py`

```python
from recstore.KVClient import RecStoreClient

client = RecStoreClient(role="default")
```

`RecStoreClient` 按 role 缓存实例，底层 `lib_recstore_ops.so` 按构建目录约定查找。测试辅助类 `src/test/framework/pytorch/client.py` 另有 `library_path=` 参数，生产路径不使用。

| API | 作用 |
|-----|------|
| `init_data(name, shape, dtype, init_func=None)` | 初始化后端表，记录本地元数据，写入初始值。可选 `mask`（长度为 `shape[0]` 的布尔张量）只写入选中的行；这是覆盖写，不是 insert-if-absent，掩码内的 live 行会被改掉。 |
| `pull(name, ids)` | 读取 `[N, D]` embedding。 |
| `pull_with_found(name, ids)` | 读取并返回 `(values, found)`；`found` 标记每一行是否存在。后端不支持时返回 `(values, None)`。 |
| `init_missing(name, ids, found, init_func=None)` | 给 `pull_with_found` 报为缺失的行写初值，返回写入行数；已存在的行不动。是覆盖写，不是原子 insert-if-absent。 |
| `push(name, ids, data)` | 写入 `[N, D]` embedding。 |
| `prefetch(ids)` / `wait_and_get(prefetch_id, embedding_dim)` | 异步预取；`prefetch` 返回 id，`wait_and_get` 取回 `[N, D]`。也可用 `wait(handle)` 只等待完成。 |
| `update(name, ids, grads)` | 发送 sparse gradients。 |
| `save_checkpoint(path, metadata)` | 先把排队中的异步更新 flush 掉，再逐分片存档；`path` 是目录。metadata 必须含非空 `checkpoint_id` 和对象型 `identity`，恢复时要用同一份。 |
| `load_checkpoint(path, metadata)` | 从存档恢复，只能在新建且未写入的 PS 上调用；有未完成的异步更新或 identity 不一致会报错。 |
| `set_ps_config(host, port)` | 重建底层 PS client。 |
| `set_ps_backend(backend)` | 切换 RPC、本地共享内存或 HierKV 路径。 |
| `delete_data` | 未实现，调用直接抛异常；后端数据不受影响。 |

缺失行的语义：

- `found` 与 `values` 来自同一次读取，`True` 表示该行在这次读取时存在。不能按“值全零”判断缺失：所有实现了这条通道的后端都会把缺失行填零（copy 路径清零输出缓冲，RDMA direct 路径从注册的零区取行）。
- 目前 `local_shm`、`brpc`、`gRPC`、`RDMA` 实现了该通道；其他后端返回 `(values, None)`，Python 侧回落到 `pull`。
- `init_missing` 只写 `found == False` 的行，但它是普通覆盖写，不是原子 insert-if-absent：查找与写入之间由其他写入者创建的行会被覆盖。
- 通道只回传每 key 一字节的掩码，不回传紧凑位置。引擎侧的 `BatchGetFlatRangeWithMissing` 会给出缺失行的位置，但那只对 C++ 调用方有意义：掩码是 1 字节/key，位置是 8 字节/缺失行，走 RPC 反而更大。

定期存档与启动恢复：

```python
from recstore.checkpoint import PeriodicSparseCheckpoint

saver = PeriodicSparseCheckpoint(
    client, "outputs/run/sparse_checkpoints", identity, every_n_steps=100
)
saver.maybe_save(step)  # 只在 step % every_n_steps == 0 时写一版
```

`save_sparse_checkpoint()` 写 `<root>/<checkpoint_id>/` 和 `manifest.json`，先落分片再发布 manifest；读端忽略没有 manifest 的目录，写失败不会破坏上一版。要恢复时把 `cache_ps.checkpoint.restore_dir` 指向同一个 root（或在训练侧读取 manifest 后调 `load_checkpoint`），语义见 [PS 存档与恢复](../parameter_server/base.md#checkpoint-restore)。
