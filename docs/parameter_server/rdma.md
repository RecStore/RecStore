# RDMA 模块说明

本文档说明 RecStore Parameter Server 的 RDMA 主路径、入口边界和运行时约束。
可执行的构建、正确性测试、benchmark 命令、默认值和报告规则统一维护在
[benchmark-ps skill](../../.agents/skills/benchmark-ps/SKILL.md)；不要在本页复制测试命令或临时性能数字。
面向真实跨机部署的配置、IB/RoCE v1/RoCE v2 选择和改造边界，见内部设计草稿
`tmp/rdma_architecture/`；草稿不纳入正式文档导航。

默认工作目录为仓库根目录：

```bash
cd /app/RecStore
```

## 1. 适用范围

当前文档只覆盖 Parameter Server 的 RDMA 主路径，不讨论 gRPC / bRPC 的通用网络栈。

当前实现的两个重要边界：

- RDMA 数据面使用仓内 verbs RC-write slot transport，不再依赖 Mayfly `RawMessage`。
- RDMA 控制面由 shard 0 的 `petps_server` 内置 gRPC 控制面承担，不再依赖 memcached。

RecStore 有两条 RDMA 入口：

| 入口 | 主要组件 | 用途 |
| --- | --- | --- |
| PetPS RDMA | `petps_server` + `PetPSClient` | RDMA 数据面、协议验证和专项 transport benchmark |
| Op-layer RDMA | `RDMAPSClientAdapter` + `KVClientOp` | 通过统一 op 接口接入 RDMA 后端 |

两条入口复用 RDMA transport，但初始化和参数来源不同：

- PetPS integration / benchmark 主要使用 C++ gflags 和对应 runner。
- Op-layer / Python client 使用环境变量和测试配置。
- 不要将某个 runner 的参数复制到另一条入口。

Op-layer RDMA 目前不是 gRPC/bRPC 的完整替代：`AsyncGetParameter` 已复用统一的
`RdmaDistributedExecutor`，`Command` 当前只通过首个 shard 的 `Barrier` 提供同步语义；
它主要用于 correctness / integration，而不是完整性能替代路径。

## 2. 架构

### 2.1 RC-write slot transport

- client 先将 `RequestDescriptor` 和 payload 写入 server request slot。
- `CommitWord` 最后写入，server 通过轮询 commit word 发现请求。
- server 写入 client response payload 后，最后回写 `StatusWord`。
- client 轮询 status word 判断请求完成。

同一条 QP 可以复用多个逻辑 slot。单 client 到单 shard 的在途上限是：

```text
qps-per-client-per-shard * slots-per-qp
```

因此 `qps-per-client-per-shard` 是 QP 池规模，不是单独的吞吐结论。

### 2.2 代码分工

| 文件 | 职责 |
| --- | --- |
| `src/ps/rdma/raw_verbs_transport.*` | 设备、MR、QP、CQ、控制面 metadata 交换和 verbs completion |
| `src/ps/rdma/rc_transport.*` | slot 布局、shard/client/lane offset、请求提交和 profile |
| `src/ps/rdma/petps_client.*` | PetPS 请求映射、QP 选择和 client 在途管理 |
| `src/ps/rdma/petps_server.*` | slot 扫描、协议处理和 response 完成 |

`raw_verbs_transport` 直接走真实 verbs RC 路径，不是 `shm_open + mmap` baseline。
`GlobalAddress`、QP metadata exchange 和 ready 协调由 RecStore 自维护。

### 2.3 控制面

`global_id=0` 的 `petps_server` 在启动时监听 TCP 控制面端口。client 和其他
server 通过它交换 `RawVerbsNodeMeta` 并等待 ready；控制面只负责启动期协调，
不参与热路径 GET/PUT/UPDATE。

常用字段：

| 参数 | 说明 |
| --- | --- |
| `--rdma-namespace` | RDMA metadata namespace，默认由 runner 自动生成 |
| `--rdma-control-plane-host` | shard 0 控制面地址 |
| `--rdma-control-plane-port` | shard 0 控制面端口；跨主机运行时应为每次运行选择唯一端口 |

## 3. 运行时约束

- `async_stream` 要求 `qps-per-client-per-shard * slots-per-qp >= async-depth`。
- QP 资源不足时，`RawVerbsTransport` 会在初始化阶段失败；常见错误是
  `ibv_create_qp failed` 或 `no idle RC write slot available`。
- `slots-per-qp` 默认是 `1`，增加它主要提高在途深度，不增加 QP 数。
- `read` 和 `push` 是不同的 PUT-v2 payload 路径，结果不能直接混合比较。
- `rdma-get-response-mode=auto` 会按 index/value layout 选择 GET response path；
  当前 `DRAM_PET_HASH` 使用 `staging_copy`，其他 index 默认使用 `direct_sg`。
- Debug/O0 适合调试，不应作为吞吐结果；benchmark 报告使用 Release 构建。
- 源码修改后必须重编对应的 `petps_server`、benchmark、integration 或 op-layer target，
  否则旧 binary 可能掩盖修改结果。

## 4. 关键参数

以下只保留容易影响语义或排障的参数；当前 benchmark 默认值和完整参数组合见
[benchmark-ps skill](../../.agents/skills/benchmark-ps/SKILL.md)。

| 参数 | 含义 |
| --- | --- |
| `--qps-per-client-per-shard` | 每个 client 到每个 shard 的 QP 数 |
| `--slots-per-qp` | 每条 QP 的逻辑 slot 数 |
| `--async-depth` | `async_stream` 单 client 的在途请求深度 |
| `--rdma-put-protocol-version` | PUT 协议版本；`2` 是当前主路径 |
| `--rdma-put-v2-transfer-mode` | PUT-v2 payload 方式：`read` 或 `push` |
| `--rdma-wait-timeout-ms` | RDMA 请求等待超时 |
| `--rdma-get-response-mode` | GET response 路径：`auto`、`direct_sg` 或 `staging_copy` |
| `--server-rdma-threads` | generic PS runner 的 server polling thread 数 |
| `--rdma-rc-server-get-workers` | GET payload worker 数；`0` 表示 poller 同步处理 |
| `--rdma-rc-server-coroutines-per-thread` | 每个 polling thread 的 scanner coroutine 数 |

拓扑参数要按入口理解：generic PS runner 使用 `--server-shard-ips`、
`--client-ips` 和 `--client-processes-per-ip`；专项 RC runner 使用
`--server-count`、`--client-count` 和 `--thread-num`。不要混用两组参数。

## 5. Profile 解读

Profile 主要分为三层，先确认 benchmark 所属层级，再解释数字。

### Client：`component=rdma_rc_client_profile`

重点字段：`submit_request_ns`、`wait_status_ns`、`copy_response_ns`、
`revoke_resource_ns`、`pending_rpc_peak`、`acquire_qp_failures`。

提交、等待或 response copy 分别偏高时，对应 client 固定开销、completion/服务端处理、
或数据搬运可能是瓶颈；`acquire_qp_failures` 表示资源不足，不是性能变慢。

### Server：`component=rdma_rc_server_profile`

重点字段：`poll_loop_ns`、`scan_rounds`、`ready_slots`、`empty_scan_rounds`、
`handle_get_ns`、`get_batch_get_ns`、`get_zero_fill_ns`、`get_row_copy_ns`、
`complete_response_ns`。

`empty_scan_rounds` 高通常表示低负载空轮询过重；GET 的 batch lookup、zero-fill、
row copy 和 response completion 可分别定位查找、清零、搬运和回写成本。

### Transport：`component=rdma_rc_transport_profile`

重点字段：`submit_request_ns`、`drain_pending_submit_ns`、`complete_response_ns`、
`drain_pending_response_ns` 以及各类 write count。drain 时间高通常说明提交或 response
存在背压。

## 6. 排障入口

按以下顺序缩小范围：

1. 确认 binary 是最新构建，并且使用了正确的入口参数。
2. 检查 `/dev/infiniband`、`ibv_devices` 和 shard 0 控制面日志。
3. 若卡在 `control-plane-wait` 或 `startup-wait`，先查看 shard 0 日志和控制面端口。
4. 若出现 `librdkafka.so.1` 缺失，确认 `build/lib` 在运行时库路径中：
   `LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}`。
5. 若出现 `unknown command line flag 'rdma_transport_mode'` 或旧的
   `--use-local-memcached`，优先检查是否运行了旧 binary 或旧 runner。
6. 若请求卡住，检查 MR 注册、QP metadata 的 shard/lane 匹配、CQ 消费线程以及
   client/server transport mode。

构建、RDMA verbs 检查、PetPS integration、op-layer 测试、脚本单测和最小真实 RDMA
闭环命令见 [benchmark-ps skill](../../.agents/skills/benchmark-ps/SKILL.md)。

## 7. 维护边界

- 本页维护稳定的架构事实和参数语义，不维护某次运行的吞吐数字、矩阵结果或路线图。
- benchmark 输出、日志和 `summary.md` 放在独立的 `results/` 目录。
- 性能比较必须标注 storage-only、PS/network 或 PyTorch/model 层；解释规则见
  [`docs/agent/perf.md`](../agent/perf.md)。
