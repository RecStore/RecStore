# 测试范围与指标

本页说明三层基准各自验证什么、跑完怎么读结果。层级选择和入口总表在[性能指南](index.md)，这里不重复。

## 每个基准负责验证什么

### storage-only —— `rs-benchmark-kvengine`

- 覆盖：`benchmark_kv_engine` 的 YCSB A/B/C、`batch_get_flat` 随机批量读，以及 `test_kvengine` 正确性。
- 不覆盖：PS 协议、多客户端并发、网络与 RDMA。
- 回答的问题：某个 index/value 组合的本地读写上限是多少；PS 层的吞吐是否已经撞到这个上限。
- 开跑前会确认：线程数、SSD 数据路径、workload、repeat、record-count、`read_mode`、`batch_keys`。
- 产出：`summary.csv`、`kvengine_workload_summary.csv`、吞吐 SVG，以及中文 `summary.md`（Workload 说明 + Run throughput 等三张表）。

### PS/network —— `rs-benchmark-ps`

- 覆盖：`ps_transport_benchmark` 在 GRPC / BRPC / RDMA 上的吞吐与延迟，local 与 cross-host 拓扑，以及 RDMA 模块自检。
- 不覆盖：训练侧 forward/backward、sparse update 语义。
- 开跑前会确认：test mode、transports、IP 列表与角色映射、workload profile（`smoke` / `rpc-small` / `rdma-capacity`）、进程与线程数、runtime、repeat。
- 产出：生成的配置、日志、`summary.csv`、中文 `summary.md`（成功 / 跳过 / 失败行 + 实际命令）。

### 端到端 —— `rs-benchmark-e2e`

- 覆盖：DLRM / TorchRec 全链路（数据加载 → forward/backward → sparse update → dense optimizer），并以 TorchRec-HBM 为对照。
- 开跑前只需给出 client 列表、PS 列表、输出目录；其余用默认口径（batch 1024、dim 128、steps 80 + warmup 5、repeat 3、prefetch depth 0、`DRAM_PET_HASH`）。
- 产出：`summary_e2e.csv`、中文 `summary.md`（Workload 说明 + E2E throughput + latency breakdown）。

## 怎么读 skill 的输出

1. 先读 `summary.md` 里的「Workload 说明」。模型、batch、步数、prefetch、`read_mode`、PS 类型、是否开埋点都在这里；口径不同的数字不能直接比。
2. 再看表格：storage-only 与 PS/network 用吞吐（keys/s、ops/s）和 p50/p99 延迟；端到端用 samples/s 与 ms/step。
3. 最后看「跳过 / 失败」行和日志路径。被 skip 的组合等于没测，不能算通过。
4. 需要引用数字时给产物路径（`results/<run>/summary.md`），不要在文档里复述易过期的数字。

## 口径纪律

- 三层数字不能跨层比较；同层也要对齐 transport、`read_mode`、`batch_keys`、线程数。
- 小于 15% 的差异必须用同一二进制交错 A/B；跨构建对比先确认源码一致。
- 只有对外报数才开性能埋点，并在结论里注明「开 / 关」。
