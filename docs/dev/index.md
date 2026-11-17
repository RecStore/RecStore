# 开发指南

- [系统架构](../architecture/overview.md)
- [计算层](../calc/index.md)
- [参数服务器](../parameter_server/index.md)
- [存储层](../storage/index.md)
- [添加算子](feature/add_operator.md)
- [扩展存储组件](feature/storage.md)
- [性能埋点](instrumentation.md)
- [测试与正确性](testing.md)
- [文档维护](documentation.md)

## 开发入门

环境准备见 [Docker 开发镜像](zero_config_run.md)与[从源码构建](../quickstart.md)，测试入口见[测试概览](testing.md)。

## 代码地图

| 目录 | 内容 |
| --- | --- |
| `src/framework/` | `CommonOp` / `KVClientOp` 与 PyTorch binding（`pytorch/`） |
| `src/ps/` | 参数服务器：`base/`、`grpc/`、`brpc/`、`rdma/`、`proto/` |
| `src/storage/` | KVEngine、index、value store、allocator、IO backend |
| `src/memory/` | 持久化与并发 slab allocator |
| `src/python/` | Python 客户端与 TorchRec 兼容层 |
| `src/test/` | 单测、集成测试与测试脚本 |
| `tools/benchmarks/` | storage-only / PS / E2E 基准 runner |
| `model_zoo/` | DLRM 与 rs_demo 示例 |

修改前先看[测试概览](testing.md)，文档约定见[文档维护](documentation.md)。
