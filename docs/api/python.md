# RecStoreClient

文件：`src/python/pytorch/recstore/KVClient.py`

```python
from recstore.KVClient import RecStoreClient

client = RecStoreClient(library_path="/path/to/lib_recstore_ops.so")
```

`RecStoreClient` 按 role 缓存实例。未传 `library_path` 时，代码按构建目录约定查找 `lib_recstore_ops.so`。

| API | 作用 |
|-----|------|
| `init_data(name, shape, dtype, init_func=None)` | 初始化后端表，记录本地元数据，写入初始值。 |
| `pull(name, ids)` | 读取 `[N, D]` embedding。 |
| `push(name, ids, data)` | 写入 `[N, D]` embedding。 |
| `prefetch(ids)` / `wait_for_prefetch(handle)` | 异步预取。 |
| `update(name, ids, grads)` | 发送 sparse gradients。 |
| `set_ps_config(host, port)` | 重建底层 PS client。 |
| `set_ps_backend(backend)` | 切换 RPC、本地共享内存或 HierKV 路径。 |

??? warning "delete_data 的边界"

    `delete_data` 只清理 Python 本地元数据，不删除后端数据。不要把它当成存储清理接口。
