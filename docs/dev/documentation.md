# 文档维护

页面内不再使用「待补充 / 待重写」横幅；未完成的内容统一记在本页，按主题追加。新增或修改页面后，至少跑一次 `mkdocs build --strict`。

## 内容待补充

| 页面 | 后续修改 |
| --- | --- |
| [监控与 Grafana](../performance/monitoring.md) | 补 ClickHouse / Grafana 的安装细节与看板导入步骤。 |
| [性能报告](../performance/reports.md) | 补历史结果的版本、硬件与拓扑。 |
| [性能埋点](instrumentation.md) | 把折叠的旧版 txt 埋点并入当前 report 体系。 |
| [如何添加新算子](../dev/feature/add_operator.md) | 按拆分后的 binding 文件补完整示例；文件划分本身不在这里重复。 |
| [DLRM / TorchRec 集成](../usage/integration.md) | 补一个只依赖 `recstore` 包的最小训练 demo（放到 `examples/`）。 |
| [Docker 开发镜像](zero_config_run.md) | 目前只有 CPU 镜像；可用的 GPU 镜像还在做，完成后补 GPU 训练路径。 |
| [配置指南](../usage/configuration.md) | 场景与加载顺序已收敛到本页，字段表留在 [配置字段参考](../config.md)；后续把 `config.md` 里重复的示例收敛成 tab。 |
| [命令行与环境变量](../api/cli.md) | 常用编译参数与 RDMA 变量已补；后续可自动导出 gflags 与 runner 参数。 |
| [KVEngine 存储基准](ycsb.md) | 页面已按"先正确性、再 YCSB、再 batch lookup、再读结果"重写；补 SSD 分档与外部引擎（FASTERKV / HPS）的对照结果。 |

## 接口文档核对

`EmbWriteAsync`、`IsWriteDone`、`WaitForWrite`、`SaveToFile`、`LoadFromFile` 在现有文档中列为占位接口，后续按实现进度更新 API 说明。

新增接口时，同步修改骨架与各 binding 文件（`op.h` / `op.cc` / `op_base.cc` / `op_flat.cc` / `op_prefetch.cc` / `cache/cache_bindings.cc`）以及 mock 路径；文件划分随接口核对一并更新。

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
