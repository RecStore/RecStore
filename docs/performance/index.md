# 性能指南

- [测试范围与指标](methodology.md)
- [端到端训练基准](e2e.md)
- [参数服务器与传输基准](ps.md)
- [KVEngine 存储基准](../dev/ycsb.md)
- [Profiling 与瓶颈分析](profiling.md)
- [监控与 Grafana](monitoring.md)
- [性能报告](reports.md)

## 基准入口 {#benchmark-entry}

仓库在.agents目录下内置了一系列测试skills，可以以对话的形式方便地启动性能测试，并自动整理给出报告。

用 Claude Code/Cursor/Codex 跑性能基准时，先按**测试层级**选 skill（定义在仓库 `.agents/skills/`）。


| 测试层级                    | 测什么                                       | 使用 skill                                                                   | 主要入口                                               |
| ----------------------- | ----------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| **storage-only**        | KVEngine（纯本地，无 PS 传输）                     | `rs-benchmark-kvengine`     | `tools/benchmarks/run_kvengine_compare.py`         |
| **PS/network**          | 本机或指定拓扑上的 GRPC / BRPC / RDMA 等传输          | `rs-benchmark-ps`                 | `tools/benchmarks/run_benchmark_ps.py`             |
| **端到端**                 | 端到端                                       | `rs-benchmark-e2e` | `python3 -m tools.benchmarks.e2e.custom.cli`       |
