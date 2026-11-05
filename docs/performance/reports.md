# 性能报告

以下结果分别对应 storage-only 和 PS/network 两层：存储端使用
`BatchGetFlat(500 random keys)`，RDMA GET 使用 `batch_keys=500`。

测试时 PS 使用 16 核，RDMA 网卡为 200 Gbps。各次测试的机器配置不同，数字仅用于同层对比；层级定义与指标口径见[测试范围与指标](methodology.md)，复现入口见[性能指南](index.md#benchmark-entry)。

| 层级           | 场景                                       | 吞吐                                                |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| storage-only | `DRAM_PET_HASH, DRAM_VALUE_STORE (Ours)` | `51.96M keys/s`                                   |
| PS/network   | `RDMA(staging-copy)`                     | stable `~41.64M keys/s` avg, best `45.62M keys/s` |
