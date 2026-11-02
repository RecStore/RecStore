# 性能报告

!!! note "待重写"

    现有数据需补齐测试版本、硬件、拓扑和复现命令。

不要把某一层的结果直接扩展成整体架构结论。当前 RDMA GET 的阶段性结论对齐了
storage-only 的 `BatchGetFlat(500 random keys)` 和 PS/network 的
RDMA GET `batch_keys=500`，结论是：


目前的性能数据（由于机器配置不同），仅供参考，PS启动了16核，RDMA网卡为200 Gbps：


| 层级           | 场景                                       | 吞吐                                                |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| storage-only | `DRAM_PET_HASH, DRAM_VALUE_STORE (Ours)` | `51.96M keys/s`                                   |
| PS/network   | `RDMA(staging-copy)`                     | stable `~41.64M keys/s` avg, best `45.62M keys/s` |
