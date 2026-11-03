# Python 客户端基础测试

!!! note "待重写"

    更新建表参数，并将表格内的代码示例改为独立代码块。

**代码位置**: `src/test/framework/pytorch/test_client.py`

## 简介
该测试验证 Client 端与 Parameter Server 的基础交互语义。它不依赖 PyTorch 的 `nn.Module`，而是直接调用 `RecstoreClient` 的底层接口。

## 流程与测试点

测试流程按顺序进行，每一个步骤都验证了 PS 交互的一个核心环节。

| 步骤 | 测试内容 | 关键代码逻辑 | 预期行为 |
| :--- | :--- | :--- | :--- |
| **Test 0** | **Embedding 表初始化** | ```python\nok = client.init_embedding_table("default", 10000, 128)\nassert ok\n``` | 只要 Server 正常运行，返回 `True` 表示建表成功。 |
| **Test 1** | **基础同步读写** | ```python\n# Write [1001, 1002, 1003]\nclient.emb_write(keys_to_write, values)\n# Read back\nread_values = client.emb_read(keys_to_write, 128)\n``` | 写入的数据必须与读回的数据完全一致 (`torch.allclose`)。 |
| **Test 2** | **异步预取 (Async Prefetch)** | ```python\npid = client.emb_prefetch(keys) # 发送请求，非阻塞\n# ... 其他计算任务 ...\nresult = client.emb_wait_result(pid, 128) # 阻塞等待结果\n``` | 通过 Request ID (`pid`) 能够取回正确的数据。 |
| **Test 3** | **Table-aware 更新** | ```python\n# 模拟梯度更新: param = param - lr * grad\nclient.emb_update_table("default", keys, grads)\n``` | Server 端应用 SGD 更新。客户端手动计算预期值进行比对，容忍微小精度误差。 |
