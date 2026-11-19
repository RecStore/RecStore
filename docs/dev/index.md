# 开发指南

本页是开发入口，按「环境 → 测试 → 代码结构 → 扩展 → 提交前检查」的顺序组织。每一节只说明该做什么、去哪里看，具体步骤放在对应的子页面里。

## 环境与首次运行

RecStore 提供两种环境准备方式：不需要编译时可以使用 [Docker 开发镜像](zero_config_run.md)，需要调试源码时按 [从源码构建](../quickstart.md) 进行编译。无论选择哪一种，都建议先按 [快速运行](../quickrun.md) 完成一次端到端运行；环境相关的问题可以在 [常见问题](../faq.md) 中查找。

## 测试

[测试概览](testing.md) 按客户端、框架与存储三层索引了现有的测试脚本。编译完成后，可以在构建目录下执行 `ctest --output-on-failure` 运行全部用例；如果只关注某一类，可以通过标签过滤，例如 `ctest -L python` 只运行 Python 绑定相关用例，`-L rdma_integration` 与 `-L perf` 分别对应 RDMA 集成与性能用例。RDMA 用例依赖真实设备，设备选择与启动器用法见 [测试辅助工具](tests/tools.md)。

## 代码结构

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

各层的设计说明分别位于 [系统架构](../architecture/overview.md)、[计算层](../calc/index.md)、[参数服务器](../parameter_server/index.md) 与 [存储层](../storage/index.md)。

## 扩展开发

- [添加算子](feature/add_operator.md)：以 Optimizer 为例，说明一个新算子从接口到测试的完整改造路径。
- [扩展存储组件](feature/storage.md)：说明如何新增 Index 与 value store。
- [性能埋点](instrumentation.md)：说明 `report()` 入口与 sink 分发机制。

## 提交前检查

提交前建议完成以下检查：运行与改动相关的测试；如果改动触及存储或 PS 层，先确认单测通过，再做吞吐测量；文档改动需要跑一次 `mkdocs build --strict`，写法约定见 [文档维护](documentation.md)。涉及跨机部署或多 rank 的改动，可参考 [分布式训练](distributed.md)，以及 [性能分析](../performance/index.md) 与 [KVEngine 存储基准](ycsb.md) 中的测量口径。
