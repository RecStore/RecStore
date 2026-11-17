# Python 客户端基础测试

**代码位置**: `src/test/framework/pytorch/test_client.py`

## 简介
该测试验证 Client 端与 Parameter Server 的基础交互语义。它不依赖 PyTorch 的 `nn.Module`，而是直接调用 `RecstoreClient` 的底层接口。

## 流程与测试点

测试流程按顺序进行，每一个步骤都验证了 PS 交互的一个核心环节。

| 步骤 | 测试内容 | 预期行为 |
| :--- | :--- | :--- |
| **Test 0** | **Embedding 表初始化** | 建表成功时返回 `True`。 |
| **Test 1** | **基础同步读写** | 写入的数据必须与读回的数据完全一致（`torch.allclose`）。 |
| **Test 2** | **异步预取** | 通过 Request ID（`pid`）取回正确的数据。 |
| **Test 3** | **Table-aware 更新** | Server 端应用 SGD 更新；客户端手动计算预期值比对，容忍微小精度误差。 |

### Test 0：初始化

```python
ok = client.init_embedding_table("default", 10000, 128)
assert ok
```

### Test 1：同步读写

```python
# Write [1001, 1002, 1003]
client.emb_write(keys_to_write, values)

# Read back
read_values = client.emb_read(keys_to_write, 128)
```

### Test 2：异步预取

```python
pid = client.emb_prefetch(keys)  # 发送请求，非阻塞

# ... 其他计算任务 ...

result = client.emb_wait_result(pid, 128)  # 阻塞等待结果
```

### Test 3：Table-aware 更新

```python
# 模拟梯度更新: param = param - lr * grad
client.emb_update_table("default", keys, grads)
```
