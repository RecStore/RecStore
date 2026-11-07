# 性能报告

以下结果按层分组：storage-only 使用 `BatchGetFlat(500 random keys)`，PS/network 的 RDMA GET
使用 `batch_keys=500`，端到端使用 DLRM 全链路。各次测试的机器配置不同，数字仅用于同层对比；
层级定义与指标口径见[测试范围与指标](methodology.md)，复现入口见[性能指南](index.md#benchmark-entry)。

| 层级           | 场景                                       | 吞吐                                                |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| storage-only | `DRAM_PET_HASH, DRAM_VALUE_STORE (Ours)` | `51.96M keys/s`                                   |
| PS/network   | `RDMA(staging-copy)`                     | stable `~41.64M keys/s` avg, best `45.62M keys/s` |

## 混合存储（DRAM+SSD）

`TIERED_VALUE_STORE` 在双机 RDMA、`ps_pipeline` 下的 SSD 比例扫描（node190 PS + node191 client，
200G IB，DLRM batch 1024，64 steps，3 次重复）：

| 访问走 SSD | samples/s | 相对 DRAM-only | step ms（均值） | 其中 commit ms |
| ---: | ---: | ---: | ---: | ---: |
| 0.0% | 37.3K | 100% | 27.6 | 20.6 |
| 16.3% | 18.7K | 50% | 55.1 | 34.2 |
| 54.1% | 9.7K | 26% | 105.7 | 68.6 |
| 97.2% | 6.3K | 17% | 162.5 | 114.4 |

完整表格、拆分口径、放置策略与复现参数见[混合存储（DRAM+SSD）端到端基准](tiered-e2e.md)。
