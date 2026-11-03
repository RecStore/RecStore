# 性能报告

!!! note "待重写"

    现有数据需补齐测试版本、硬件、拓扑和复现命令。

以下结果分别对应 storage-only 和 PS/network 两层：存储端使用
`BatchGetFlat(500 random keys)`，RDMA GET 使用 `batch_keys=500`。

测试时 PS 使用 16 核，RDMA 网卡为 200 Gbps。各次测试的机器配置不同。

| 层级           | 场景                                       | 吞吐                                                |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| storage-only | `DRAM_PET_HASH, DRAM_VALUE_STORE (Ours)` | `51.96M keys/s`                                   |
| PS/network   | `RDMA(staging-copy)`                     | stable `~41.64M keys/s` avg, best `45.62M keys/s` |
