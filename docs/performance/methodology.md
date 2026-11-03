# 测试范围与指标

RecStore 的基准分三层，指标不能跨层比较：

| 层级 | 测什么 | 典型指标 | 入口 |
| --- | --- | --- | --- |
| **storage-only** | KVEngine 纯本地读写，不含 PS 与网络 | keys/s、ops/s | `tools/benchmarks/run_kvengine_compare.py` |
| **PS/network** | 指定拓扑上的 GRPC / BRPC / RDMA 传输 | keys/s、p50/p99 延迟 | `tools/benchmarks/run_benchmark_ps.py` |
| **端到端** | 数据加载、forward/backward、sparse update、dense optimizer 全链路 | samples/s、ms/step | `python3 -m tools.benchmarks.e2e.custom.cli` |

口径约定：

- storage-only 与 PS/network 用吞吐（keys/s 或 ops/s）；端到端用 samples/s 与
  每步耗时（ms/step）。
- 同一份代码在不同机器上可差 10% 以上；小于 15% 的差异要用同一二进制交错 A/B。
- 引用数字时注明层级、`read_mode`、`batch_keys` 和是否开启性能埋点。

各层的具体运行方式见[性能指南](index.md)的基准入口。
