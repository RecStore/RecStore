# RecStoreClient

文件：`src/python/pytorch/recstore/KVClient.py`

```python
from recstore.KVClient import RecStoreClient

client = RecStoreClient(role="default")
```

`RecStoreClient` 按 role 缓存实例，底层 `lib_recstore_ops.so` 按构建目录约定查找。测试辅助类 `src/test/framework/pytorch/client.py` 另有 `library_path=` 参数，生产路径不使用。

| API | 作用 |
|-----|------|
| `init_data(name, shape, dtype, init_func=None)` | 初始化后端表，记录本地元数据，写入初始值。 |
| `pull(name, ids)` | 读取 `[N, D]` embedding。 |
| `push(name, ids, data)` | 写入 `[N, D]` embedding。 |
| `prefetch(ids)` / `wait_and_get(prefetch_id, embedding_dim)` | 异步预取；`prefetch` 返回 id，`wait_and_get` 取回 `[N, D]`。也可用 `wait(handle)` 只等待完成。 |
| `update(name, ids, grads)` | 发送 sparse gradients。 |
| `set_ps_config(host, port)` | 重建底层 PS client。 |
| `set_ps_backend(backend)` | 切换 RPC、本地共享内存或 HierKV 路径。 |
| `delete_data` | 清理 Python 本地元数据，后端数据保留。 |
