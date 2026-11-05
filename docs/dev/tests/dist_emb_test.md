# DistEmbedding 模块测试

**代码位置**: `src/python/pytorch/recstore/unittest/test_dist_emb.py`

## 简介
`DistEmbedding` 是 RecStore 的核心构建块，它是 `nn.Module` 的直接子类。此测试验证它完全遵循 PyTorch 的 Autograd 机制，并能无缝接入 Parameter Server。

## 流程与测试点

| 测试方法 | 描述 | 图解/逻辑 | 预期结果 |
| :--- | :--- | :--- | :--- |
| `test_initialization_and_forward` | **初始化、元数据注册与 forward** | `DistEmbedding(...)` -> 连接 Client -> 注册 Table -> forward | Client 内部元数据结构中包含该 Table 的 Shape 和 Dtype；forward 输出形状正确。 |
| `test_backward_and_optimizer_step` | **梯度追踪与优化器更新** | `loss.backward()` -> Hook 记录 `{Keys, Grads}` -> `SparseSGD.step()` -> `client.push` | backward 只记录 Trace，不更新权重；`step()` 后权重按 SGD 公式变化。 |
| `test_update_with_duplicate_ids` | **重复 ID 的梯度聚合** | 输入 Batch: `[ID 1, ID 1, ID 1]`，backward 后 step | Server 端收到的 ID 1 梯度为 3 次梯度的总和。 |
| `test_persistence_with_same_name` | **数据持久化** | `Emb(name="A")` 退出 -> 重新创建 `Emb(name="A")` | 新实例能读取到上一个实例写入的数据。 |
