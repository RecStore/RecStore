# 文档维护

缺页用「待补充」标记；需要更新的旧文用「待重写」标记。

## 待补充

- [配置指南](../usage/configuration.md)
- [DLRM / TorchRec 集成](../usage/integration.md)
- [测试范围与指标](../performance/methodology.md)
- [端到端训练基准](../performance/e2e.md)
- [参数服务器与传输基准](../performance/ps.md)
- [命令行与环境变量](../api/cli.md)

## 待重写

| 页面 | 后续修改 |
| --- | --- |
| [如何添加新算子](../dev/feature/add_operator.md) | 示例仍按单个 op_torch.cc 组织，需更新 binding 文件划分和建表参数。 |
| [从源码构建](../quickstart.md) | 与 Docker 开发镜像页面合并重复的环境准备步骤。 |
| [Docker 开发镜像](../dev/zero_config_run.md) | 统一镜像来源、标签和构建步骤，消除与源码构建指南的重复。 |
| [监控与 Grafana](../performance/monitoring.md) | 补齐服务安装与看板导入步骤。 |
| [性能报告](../performance/reports.md) | 补齐环境、版本和复现命令。 |
| [性能埋点](instrumentation.md) | 整理已折叠的旧版 txt 埋点说明。 |

## 接口文档核对

`EmbWriteAsync`、`IsWriteDone`、`WaitForWrite`、`SaveToFile`、`LoadFromFile` 在现有文档中列为占位接口，后续按实现进度更新 API 说明。

新增接口时，同步修改骨架与各 binding 文件（`op.h` / `op.cc` / `op_base.cc` /
`op_flat.cc` / `op_prefetch.cc` / `cache/cache_bindings.cc`）以及 mock 路径；文件划分随接口核对一并更新。

## 性能文档维护

RDMA 模块页维护架构与参数语义，单次运行的吞吐数据和矩阵结果放在性能报告中。benchmark 输出、日志和 `summary.md` 放在独立的 `results/` 目录。

性能比较标注 storage-only、PS/network 或 PyTorch/model 层，指标解释见[性能比较背景](../agent/perf.md)。

## 本地预览

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

构建检查：

```bash
mkdocs build --strict
```
