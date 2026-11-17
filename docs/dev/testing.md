# 开发测试概览

本部分主要索引开发过程中使用的各类测试脚本。详细的测试逻辑请参阅下方对应的子文档。

## 客户端层 (Client Layer)

| 模块 | 测试说明 | 详细文档 |
| :--- | :--- | :--- |
| **Python Client Binding** | 验证客户端 C++ 到 Python 的绑定及基础通信能力。 | [Python 客户端基础测试](tests/client_test.md) |

## PyTorch 框架层 (Framework Layer)

| 模块 | 测试说明 | 详细文档 |
| :--- | :--- | :--- |
| **EBC Precision** | 验证 TorchRec 兼容层 (`RecStoreEBC`) 与原生的数值对齐。 | [EBC 精度对齐测试](tests/ebc_precision.md) |
| **DistEmbedding** | 验证独立分布式 Embedding 模块的各项功能（如梯度聚合、持久化）。 | [DistEmbedding 模块测试](tests/dist_emb_test.md) |

## 工具与覆盖率

| 模块 | 测试说明 | 详细文档 |
| :--- | :--- | :--- |
| **测试辅助工具** | `ps_server` 启动器、环境变量与 runner 辅助函数。 | [测试用工具](tests/tools.md) |
| **代码覆盖率** | C++ / Python 覆盖率报告入口。 | [代码覆盖率](tests/coverage.md) |
