# RDMA 模块运行手册

更新时间：2026-06-03

本文档整理当前 RecStore RDMA 主路径的边界、参数、验证入口、已知限制和下一步路线图。默认工作目录为仓库根目录：

```bash
cd /app/RecStore
```

## 1. 适用范围

当前文档只覆盖 Parameter Server 里的 RDMA 主路径，不讨论 gRPC / bRPC 的通用网络栈。

当前主路径已经完成两次硬切换：

- RDMA 数据面不再依赖历史 Mayfly `RawMessage` 路径。
- RDMA 控制面不再依赖 `memcached`，统一由 shard 0 `petps_server` 内置 gRPC 控制面承担。

RecStore 现在有两条 RDMA 入口：

| 层级 | 入口 | 用途 |
|------|------|------|
| PetPS RDMA | `petps_server` + `PetPSClient` | RDMA 数据面、协议验证、transport benchmark |
| Op-layer RDMA | `RDMAPSClientAdapter` + `KVClientOp` | 通过统一 op 接口验证 RDMA 后端 |

两条入口复用同一套 RDMA 传输，但初始化和调用方式不同：

- PetPS integration 和 benchmark 主要通过 C++ gflags 传参。
- Op-layer / Python client 主要通过环境变量和测试配置传参。
- 不要把脚本参数、C++ gflag 和环境变量混用到错误入口。

Op-layer RDMA 目前不是 gRPC/bRPC 的完整替代：

- `AsyncGetParameter` 和 `Command` 还未实现。
- `UpdateParameter` 走同步 read-modify-write。
- 更适合作为 correctness / integration 路径，而不是完整性能替代路径。

## 2. 当前 RDMA 架构

### 2.1 传输模式

当前 PetPS RDMA 入口使用 RC-write slot transport：

- `RequestDescriptor + payload` 先写入 server request slot。
- `CommitWord` 最后写入，server 通过轮询 commit word 发现请求。
- server 处理后先写 client response payload。
- `StatusWord` 最后写回，client 通过轮询 status word 判断完成。

当前实现还支持 `slots_per_qp` 这一层逻辑复用：同一条 QP lane 上可以挂多个逻辑 slot，实际的 in-flight 上限是 `qps-per-client-per-shard * slots-per-qp`，不是只看 QP 数。

### 2.2 代码分工

`src/ps/rdma/raw_verbs_transport.*` 已经不再是 `shm_open + mmap` baseline，而是直接走真实 verbs RC 路径：

- 打开 RDMA 设备
- 注册本地 MR
- 创建 QP / CQ
- 通过 shard 0 gRPC 控制面交换 metadata
- 轮询 RDMA write completion

这条路径当前已经不再链接或包含 Mayfly 控制面代码；`GlobalAddress`、QP metadata exchange 和 ready 协调都在仓内自维护。

`src/ps/rdma/rc_transport.*` 负责：

- 固定 slot 布局
- shard / client / lane offset 计算
- request / response 提交流程
- 连接和回写的 profile 统计

`src/ps/rdma/petps_client.*` 和 `src/ps/rdma/petps_server.*` 负责：

- 把 RDMA slot 映射成 PetPS 的 request / response
- client QP 选择与 in-flight 管理
- server slot 扫描、协议处理、response 完成

### 2.3 现在的关键约束

- `qps-per-client-per-shard` 不是“吞吐参数”，而是 client 侧可用的 QP 池规模。
- `slots-per-qp` 是每条 QP 上的逻辑 slot 数，默认是 `1`，主要用来扩展 in-flight 上限而不是增加 QP 数。
- `async_stream` 下，`qps-per-client-per-shard * slots-per-qp` 不能小于 `async-depth`。
- 如果硬件 QP 资源不足，即使参数合法，`RawVerbsTransport` 也会在 client 初始化阶段直接报错并拒绝启动。
- 这比“跑到一半再卡住”更容易定位，也更符合当前的 fail-fast 目标。

## 3. 构建

常用 RDMA 目标：

```bash
cmake --build ./build --target \
  ps_transport_benchmark \
  rdma_rc_transport_benchmark \
  petps_server \
  petps_integration_test \
  recstore_torch_ops \
  test_allshards_ps_client \
  -j
```

吞吐测试优先使用 Release/O3 构建：

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target ps_transport_benchmark petps_server -j
```

ssh runner 通过 `--build-dir build_release` 和 `--remote-build-dir build_release`
选择 Release 二进制；local runner 的 client 和 server 也会跟随 `--build-dir`。
Debug/O0 适合调试，不适合作为吞吐上限；2026-06-03 的 p4/t3/q16/d16 测试中，
跨机 Release/O3 为 `45.238-45.523 M keys/s`，本机 Release/O3 为
`45.720 M keys/s`。

如果刚改过 `src/ps/rdma/*`、`src/test/scripts/*rdma*` 或 op-layer 相关代码，先重编对应目标再判断行为。旧的 `petps_server`、`ps_transport_benchmark`、`rdma_rc_transport_benchmark` 或 `recstore_torch_ops` 二进制很容易造成“源码已改但测试仍卡住”的假象。

如果 `petps_server` 启动时报下面的动态库错误：

```text
error while loading shared libraries: librdkafka.so.1: cannot open shared object file
```

当前 CMake 会把 HugeCTR vendored `librdkafka` 纳入构建，正常情况下 `petps_server` 会解析到：

```text
/app/RecStore/build/lib/librdkafka.so.1
```

如果旧构建目录或手工启动环境仍然找不到动态库，先把本地构建出的 `build/lib` 加到运行时库路径，再启动 integration 或 benchmark：

```bash
export LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}
```

这只是 HPS 依赖的运行时动态库路径问题，不代表 RDMA verbs、QP 或控制面已经失败。

## 4. 参数说明

下面这张表只列 RDMA benchmark 和 runtime 中最常见、最容易误解的参数。

| 参数 | 含义 | 备注 |
|------|------|------|
| `--iterations` | 每个 round 内执行的请求次数 | 影响单轮总请求数，通常和 `batch-keys` 一起看 |
| `--rounds` | 计入统计的测量轮数 | 最终吞吐由这些轮次聚合得到 |
| `--warmup-rounds` / `--rdma-warmup-rounds` | 热身轮数 | 不计入结果，只用于预热连接、缓存和内存路径 |
| `--batch-keys` / `--batch_keys` | 每次请求携带的 key 数 | 决定单次 RPC 的负载大小 |
| `--thread-num` / `--server-rdma-threads` | server 侧 RDMA polling thread 数 | `--thread-num` 属于 RC 专项入口；`--server-rdma-threads` 属于 generic PS runner |
| `--client-count` / `--client-processes-per-ip` | benchmark client 进程数 | `--client-count` 属于 RC 专项入口；generic PS runner 按 client IP 展开进程数 |
| `--server-count` / `--server-shard-ips` / `--num_shards` | server / shard 数量 | generic PS runner 中 `--server-shard-ips` 的列表长度就是 shard 数 |
| `--qps-per-client-per-shard` | 每个 client 到每个 shard 的 QP 数 | 单独只表示 QP 池规模；真实 in-flight 上限还要乘 `--slots-per-qp` |
| `--slots-per-qp` | 每条 QP 上可复用的逻辑 slot 数 | 默认 `1`；适合在不增加 QP 数的情况下提高在途深度 |
| `--async-depth` | `async_stream` 里单 client 的在途请求深度 | 低负载上限测试的关键参数 |
| `--rdma-put-protocol-version` | PUT 协议版本 | `1` 是 legacy，`2` 是当前主路径 |
| `--rdma-put-v2-transfer-mode` | PUT-v2 payload 传输方式 | `read` 表示 server 读 payload，`push` 表示 client 主动写 payload |
| `--rdma-wait-timeout-ms` | RDMA 请求等待超时 | 过短会导致 benchmark 误判为超时失败 |
| `--profile-interval-ms` | RDMA RC 统计输出间隔 | `0` 表示不做周期性 profiling，仅输出 benchmark 结果 |
| `--rdma-rc-server-get-workers` | generic PS runner 的 server GET payload worker 数 | `0` 表示 poller 同步处理 GET；`>0` 会把 payload 读取/填充 offload 给 worker |
| `--rdma-rc-server-coroutines-per-thread` | generic PS runner 每个 polling thread 内的 scanner coroutine 数 | 当前作为调度实验维度；`C=1` 是推荐 benchmark 默认 |
| `--server-coroutines-per-thread` | 每个 polling thread 上的 server 协程数 | 值越大越偏向 coop 扫描，不代表一定更快 |
| `--fake-get-mode` | benchmark-only fake GET 行为 | `none`、`status_only`、`index_only`、`payload_memset` |
| `--skip-client-copy` | 是否跳过 client 端 GET payload 拷贝 | 只用于 benchmark 排查，不适合作为默认配置 |
| `--rdma-get-response-mode` | RDMA GET response payload 路径 | `auto`、`direct_sg`、`staging_copy`；generic PS runner 参数 |
| `--build-dir` | ssh runner 的本地构建目录 | 默认 `build`；Release 压测设为 `build_release` |
| `--remote-build-dir` | ssh runner 的远端构建目录 | 默认 `build`；Release 压测设为 `build_release` |

### 4.1 重要解读

- `batch-keys` 是请求粒度，不是总样本数。
- `iterations * rounds * batch-keys` 才是测量期的 key 总量。
- `read` 和 `push` 的结果不能直接混成一个吞吐结论。
- `qps-per-client-per-shard` 变大通常会增加连接并发，但也会增加资源占用和初始化成本。
- `slots-per-qp` 变大通常会增加单条 QP 的并发深度，但也会增加每条 lane 的本地状态和回写压力。
- `profile-interval-ms` 只影响周期性统计输出，不应和吞吐提升直接画等号。
- `rdma-get-response-mode=auto` 当前按 index/value layout 选择 GET response path：
  `DRAM_PET_HASH` 走 `staging_copy`，其他 index 默认走 `direct_sg`。

### 4.2 启动前硬约束

当前 benchmark 已经加入启动前校验：

- `async_stream` 要求 `qps-per-client-per-shard * slots-per-qp >= async-depth`
- 否则直接拒绝启动，避免把 client slot 池打满后在运行中报 `no idle RC write slot available`

如果你要做低负载上限测试，建议先确保这个约束满足，再看真实瓶颈。

## 5. Profile 统计怎么读

当前 RDMA profile 已经拆成三层，足够定位大部分低负载瓶颈。

### 5.1 Client 侧

来自 `component=rdma_rc_client_profile`，重点看这些字段：

- `submit_request_ns`
- `wait_status_ns`
- `copy_response_ns`
- `revoke_resource_ns`
- `pending_rpc_peak`
- `acquire_qp_count`
- `acquire_qp_failures`

解读方式：

- `submit_request_ns` 高，说明一次请求发出去的固定开销重。
- `wait_status_ns` 高，说明 completion / 回写链路慢，或者 server 处理慢。
- `copy_response_ns` 高，说明 response copy 成为低负载瓶颈。
- `pending_rpc_peak` 高，说明在途请求池压力大。
- `acquire_qp_failures` 不是“慢”，而是“资源不够”。

### 5.2 Server 侧

来自 `component=rdma_rc_server_profile`，重点看这些字段：

- `poll_loop_ns`
- `scan_rounds`
- `scanned_slots`
- `ready_slots`
- `empty_scan_rounds`
- `handle_get_ns`
- `get_batch_get_ns`
- `get_zero_fill_ns`
- `get_row_copy_ns`
- `handle_put_ns`
- `handle_update_ns`
- `handle_init_ns`
- `complete_response_ns`

解读方式：

- `poll_loop_ns` 高，通常说明空轮询太多，是低负载上限的第一嫌疑。
- `empty_scan_rounds` 高，说明扫描很多次但没命中请求。
- `get_batch_get_ns` 高，说明 batch 查找本身重。
- `get_zero_fill_ns` 高，说明缺失值或响应缓冲清零成本偏高。
- `get_row_copy_ns` 高，说明数据搬运是瓶颈。
- `complete_response_ns` 高，说明回写完成链路重。

### 5.3 Transport 侧

来自 `component=rdma_rc_transport_profile`，重点看这些字段：

- `submit_request_ns`
- `drain_pending_submit_ns`
- `complete_response_ns`
- `drain_pending_response_ns`
- `submit_descriptor_write_count`
- `submit_commit_write_count`
- `response_payload_write_count`
- `response_status_write_count`

解读方式：

- `submit_request_ns` 高，说明 client 侧 verbs 提交开销偏大。
- `drain_pending_submit_ns` 高，说明上一笔请求没有及时完成，出现提交背压。
- `complete_response_ns` 高，说明 server 侧回写成本偏大。
- `drain_pending_response_ns` 高，说明 response path 也有背压。

## 6. Benchmark 建议

### 6.1 Benchmark 入口选择

当前有两个 benchmark 二进制，它们不是重复目标：

| 目标 | Runner | 适用场景 |
|------|--------|----------|
| `ps_transport_benchmark` | `tools/benchmarks/run_rdma_transport_benchmarks.py` | 通用 PS transport 对比入口，可跑 RDMA / gRPC / bRPC；RDMA 模式主要用于验证 PUT-v2 `read` 和 `push` 两条路径。 |
| `rdma_rc_transport_benchmark` | `tools/benchmarks/run_rdma_rc_transport_benchmark.py` | RDMA RC 专项压测入口，支持 `client-count`、`async_stream`、QP 池、server coroutine、fake get、skip client copy 等 RC 诊断参数。 |

选择规则：

- 想比较 RDMA 和 gRPC / bRPC，或验证 PUT-v2 `read` / `push` 传输模式，用 `ps_transport_benchmark`。
- 想测 RDMA RC 本身的单 shard、多 client、低负载上限或 async pipeline，用 `rdma_rc_transport_benchmark`。
- 不要把两个入口的参数混用。`--rdma-put-v2-transfer-mode` 属于通用入口；`--qps-per-client-per-shard`、`--async-depth`、`--client-count` 属于 RC 专项入口。

### 6.2 通用 PS Transport 入口

建议先跑 PetPS RC-write correctness 基线，再做 benchmark：

```bash
python3 tools/benchmarks/run_rdma_transport_benchmarks.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --iterations 300 \
  --batch-keys 500 \
  --rounds 20 \
  --rdma-warmup-rounds 10 \
  --report-mode summary \
  --rdma-only \
  --rdma-thread-num 1 \
  --rdma-put-protocol-version 2 \
  --rdma-put-v2-transfer-mode read \
  --rdma-wait-timeout-ms 20000 \
  --rdma-client-timeout-sec 60 \
  --show-runner-logs \
  --rdma-control-plane-host 127.0.0.1
```

summary 表中的 `put_v2` 列用于确认 PUT-v2 payload transfer mode；`read` 和 `push` 的结果不能直接混比。

如果你想在不继续增加 QP 数的情况下抬高 `async_stream` 深度，可以优先调 `--slots-per-qp`，再看 `qps-per-client-per-shard` 是否还需要一起放大。

`transactions/fetch` 路径使用 `tools/benchmarks/run_benchmark_ps.py`。这个 runner 的拓扑参数采用显式命名：

- `--server-shard-ips`：每个 shard 对应一个 server IP，列表长度就是 shard 数。
- `--client-ips`：要 ssh / 本地启动 client 的 IP 列表。
- `--client-processes-per-ip`：每个 client IP 上启动多少个 benchmark 进程。
- `--client-threads-per-process`：每个 benchmark 进程内的线程数。
- `--server-worker-threads`：PS/KV worker 线程数。
- `--server-rdma-threads`：RDMA server polling 线程数。
- `--rdma-rc-server-get-workers`：server GET payload worker 线程数；`0` 表示 poller 同步处理 GET。
- `--rdma-rc-server-coroutines-per-thread`：每个 polling thread 内的 scanner coroutine 数；当前作为调度实验维度，不作为性能默认。
- `--rdma-get-response-mode`：GET response payload 路径；推荐默认用 `auto`。

本地单 shard、4 client 进程的 RDMA fetch profile 示例：

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports rdma \
  --server-shard-ips 127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 4 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 1024 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 1 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-profile-interval-ms 1000 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_profile_0529/rdma_1s4c_profile_clean
```

#### 6.2.1 当前验证状态与边界

截至 `2026-05-29`，`run_benchmark_ps.py` 的 RDMA `transactions/fetch` 路径已验证：

- `--server-shard-ips 127.0.0.1`
- `--server-shard-ips 127.0.0.1,127.0.0.1`
- `--client-processes-per-ip 1`、`2`、`4` 和 `8`
- `--client-threads-per-process 1`
- `--client-load-threads-per-process 1`
- benchmark 进程内复用同一个 RDMA client 做 load 和 run

这个路径依赖 `RDMAPSClientAdapter` 保留 runner 传入的 `global_id`、`num_server_processes` 和 `num_client_processes`。如果 adapter 覆盖这些拓扑参数，多 client generic benchmark 会在控制面 metadata exchange 阶段超时，例如曾经出现过：

```text
get_meta timeout key=2:0->0:0
```

通用 PS benchmark 的 RDMA 并发建议通过 `--client-ips` 和 `--client-processes-per-ip` 增加 client 进程数。单个 benchmark 进程内仍建议保持 `--client-threads-per-process=1`，除非已经重新验证同进程多线程下的 RDMA client/lane 语义。

如果要复现当前 GET worker / poller 调度实验，建议显式写出 `T/N/C` 三个维度，并把它们编码进结果目录名：

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 8 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 8 \
  --repeat 3 \
  --execution-backend local \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 240 \
  --cluster-timeout 80 \
  --transaction-profile \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 8 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --output-dir results/benchmark_ps_profile_<date>_n8_p8_t16_c1_repeat3
```

当前本机结果里，`p8/N8/T16/C1` repeat=3 平均 `14.45M keys/s`；同条件 `C4` 平均 `12.17M keys/s`。因此 `C=1` 是当前推荐 benchmark 默认，`C>1` 只能作为 coroutine scanner 调度实验，不应默认为优化项。

公平比较 RDMA / GRPC / BRPC 时，必须对齐 `--client-threads-per-process` 和 `--client-load-threads-per-process`。如果 RPC 使用 16 线程而 RDMA 使用 1 线程，这只能称为 mixed-concurrency capacity check，不能作为公平 transport 对比。

#### 6.2.2 2026-05-29 本地公平矩阵结论

本轮本地矩阵固定：

- `record_count=200000`
- `value_size=512`
- `batch_keys=500`
- `client_threads_per_process=1`
- `client_load_threads_per_process=1`
- `runtime_seconds=3`
- `repeat=1`
- `server_worker_threads=32`
- `server_rdma_threads=1`
- `rdma_rc_qps_per_client_per_shard=16`

raw results 保存在：

```text
results/benchmark_ps_matrix_0529
```

这组历史矩阵仅作为本页内的早期基线快照保留；新的吞吐结论以本页后续
`DRAM_PET_HASH + auto/staging_copy` 路径和 `rs-benchmark-ps` 执行手册为准。

总吞吐矩阵如下，单位为 `M keys/s`：

| server_shards | client_processes | client_threads_per_process | GRPC | BRPC | RDMA |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.467 | 0.333 | 2.947 |
| 1 | 2 | 1 | 0.638 | 0.519 | 2.977 |
| 1 | 4 | 1 | 0.859 | 0.998 | 2.976 |
| 1 | 8 | 1 | 1.170 | 1.229 | 3.006 |
| 2 | 1 | 1 | 0.465 | 0.470 | 1.398 |
| 2 | 2 | 1 | 0.620 | 0.683 | 1.626 |
| 2 | 4 | 1 | 0.640 | 0.954 | 1.487 |
| 2 | 8 | 1 | 0.785 | 1.210 | 1.817 |

解读边界：

- 单 shard 下 RDMA 稳定在 `~3.0 M keys/s`，明显高于同线程配置下的 GRPC / BRPC。
- 从 `client_processes_per_ip=1` 增加到 `8` 后，RDMA 总吞吐基本不再提升，说明当前本地单 shard 瓶颈不在 client 线程数，而更可能在 server 处理、RDMA PS benchmark 调度/拷贝、polling 或本机资源竞争。
- 2 shards 全部在 `127.0.0.1` 上启动时，RDMA 反而下降到 `~1.4-1.8 M keys/s`。这是本地多 server 进程压力结果，不是跨机器 shard 扩容结论。
- 多 client 的 per-client 吞吐有明显离散度，尤其 RDMA `p=8`。报告总吞吐时也要保留 per-client min/max，否则会掩盖公平性和调度问题。
- 这组数据属于 PS/network 层，不应被解释成 storage-only、RDMA RC transport-only 或 PyTorch/model 层结论。

#### 6.2.3 为什么 generic benchmark 和 dedicated RC benchmark 可能结论不同

`ps_transport_benchmark` 和 `rdma_rc_transport_benchmark` 不是同一条调用路径：

- `rdma_rc_transport_benchmark` 主要验证 `PetPSClient` 的专项 RC 闭环
- `ps_transport_benchmark` 走的是更上层的 `RDMAPSClientAdapter`
- `transactions` 模式下还会经历 preload、thread 创建、client 生命周期切换等额外行为

因此：

- dedicated RC benchmark 成功，不能自动推出 generic PS benchmark 也成功
- generic PS benchmark 失败，也不能直接证明底层 RC transport 已坏

排障时先用 dedicated RC benchmark 建立 transport baseline，再看 generic benchmark 是否是更上层生命周期或调用方式问题。

#### 6.2.4 2026-05-31 RDMA GET response path 固化

本轮对齐了 storage-only 与 PS/network 两层：

- storage-only：`benchmark_kv_engine --read_mode=batch_get_flat --batch_keys=500`
- PS/network：`run_benchmark_ps.py --batch-keys 500`

关键结论：

| 层级 | index | response path | 吞吐 |
|---|---|---|---:|
| storage-only | `DRAM_EXTENDIBLE_HASH` | `BatchGetFlat(500 random keys)` | `19.45M keys/s` |
| PS/network | `DRAM_EXTENDIBLE_HASH` | direct-SG | `19.37M keys/s` |
| storage-only | `DRAM_PET_HASH` | `BatchGetFlat(500 random keys)` | `51.96M keys/s` |
| PS/network | `DRAM_PET_HASH` | direct-SG | `14.93M keys/s` |
| PS/network | `DRAM_PET_HASH` | staging-copy, stable `qps=16` | `41.64M keys/s` avg, best `45.62M keys/s` |
| PS/network | `DRAM_PET_HASH` | staging-copy, tuning `qps=20` | up to `46.69M keys/s` |

这修正了 2026-05-30 的阶段性判断：`direct-SG` 不是所有 index 的默认最优路径。
`DRAM_EXTENDIBLE_HASH` 的主要瓶颈已经是 index lookup；`DRAM_PET_HASH` 的 lookup
更快，但随机 row refs 更离散，direct-SG 需要组织更多 SGE/WR，反而慢于把 value
先聚合到连续 response buffer 的 staging-copy。

推荐默认：

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --index-type DRAM_PET_HASH \
  --rdma-get-response-mode auto \
  --batch-keys 500 \
  --value-size 512
```

`auto` 当前解析为：

| index | response path |
|---|---|
| `DRAM_PET_HASH` | `staging_copy` |
| 其他 index | `direct_sg` |

如果需要回归 direct-SG 本身，显式使用 `--rdma-get-response-mode direct_sg`；如果
需要验证连续 response buffer 路径，显式使用 `--rdma-get-response-mode staging_copy`。

### 6.3 RDMA RC 专项入口

当前 `run_rdma_rc_transport_benchmark.py` 使用 shard 0 `petps_server` 内置控制面，不再接受旧的 `--use-local-memcached` 参数。看到 `unrecognized arguments: --use-local-memcached` 时，删除该参数即可；不要改回 memcached 控制面。

最小真实 RDMA RC 闭环命令：

```bash
export LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}
python3 tools/benchmarks/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 2 \
  --rounds 1 \
  --warmup-rounds 0 \
  --batch-keys 4 \
  --value-size 16 \
  --op get \
  --report-mode summary \
  --qps-per-client-per-shard 1 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 60 \
  --cluster-timeout 30 \
  --rdma-control-plane-host 127.0.0.1
```

单 client `async_stream` smoke 可用下面的命令，适合在调大吞吐前确认 RC 专项路径能完整跑通：

```bash
export LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}
python3 tools/benchmarks/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 100 \
  --rounds 10 \
  --warmup-rounds 3 \
  --batch-keys 16 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 300 \
  --cluster-timeout 60
```

多 client 最小解析验证：

```bash
export LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}
python3 tools/benchmarks/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 2 \
  --thread-num 1 \
  --iterations 2 \
  --rounds 1 \
  --warmup-rounds 0 \
  --batch-keys 4 \
  --value-size 16 \
  --op get \
  --report-mode summary \
  --qps-per-client-per-shard 1 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 60 \
  --cluster-timeout 30 \
  --rdma-control-plane-host 127.0.0.1 \
  --quiet
```

判定标准：

- 输出必须包含 `phase=measure summary`。
- 单 client 表必须出现 `RDMA RC Benchmark Summary` 和 `RDMA RC Aggregate Summary`。
- 多 client `--quiet` 模式至少要在 aggregate 表中看到 `clients` 等于实际 client 数。
- 如果只看到原始 benchmark 行但没有 summary 表，优先检查 runner 的 summary 正则和多 client stdout 行边界。

### 6.4 当前更重要的 benchmark 目标

现在的重点不是继续把 `client-count` 往上堆，而是：

- 低负载下提高单 shard 的 req qps 上限
- 看清哪些是固定开销，哪些是资源上限
- 找出可以稳定提升上限的最小改动

因此，建议优先用这些组合看上限：

- `op=get`
- `op=async_get`
- 小 `batch-keys`
- 小到中等 `async-depth`
- `client-count=1/2`

### 6.5 低负载上限测试建议

建议把这些参数固定住，只改一个维度：

- `value_size`
- `batch-keys`
- `iterations`
- `rounds`
- `thread-num`
- `server-count`

优先扫描的维度：

- `async-depth`
- `server-coroutines-per-thread`
- `qps-per-client-per-shard`
- `profile-interval-ms`

## 7. 当前状态

当前 RDMA RC 主路径已经不是“单 QP 单 slot”的旧模型了，而是按 `qp_index + slot_in_qp` 共同寻址：

- client 侧会为每个 `qp_index` 分配一组逻辑 slot。
- server 侧会按 `client_id / qp_index / slot_in_qp` 反解请求槽位。
- `slots_per_qp` 已经贯通到 client、server、runner、测试和 benchmark 参数解析。

当前仓内的活跃 RDMA 路径还有几个明确事实：

- 运行时不再依赖 `memcached`。
- 顶层构建不再引入 `third_party/Mayfly-main`。
- `src/ps/base/Postoffice.*` 只保留进程角色和 shard/client 元信息，不再负责 memcached 读写。
- `src/benchmark/perf_sgl.cc` 已移除，不再作为维护中的 RDMA 路径。
- RDMA 目标现在显式链接 `ibverbs`，不再依赖旧链路隐式带出 verbs 符号。

这意味着文档里判断并发能力时，不能再只看 `qps-per-client-per-shard`，而要看 `qps-per-client-per-shard * slots-per-qp`。

如果你要补 benchmark 数字，建议把结果放到单独的 `results/` 目录，不要把一次临时跑出来的吞吐值长期固化在主文档里。

### 7.1 最近验证状态

以 2026-05-29 本地仓库状态为准，下面这些验证已经通过：

- `cmake -S . -B build`
- `cmake --build build --target ps_transport_benchmark ps_server petps_server -j`
- `python3 -m unittest src/test/scripts/test_run_benchmark_ps.py`
- `ctest -R 'grpc_ps_client_test|dist_grpc_ps_client_test|brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_transport_benchmark|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure`
- `run_benchmark_ps.py` 公平矩阵：`server_shards=1,2`，`client_processes_per_ip=1,2,4,8`，`client_threads_per_process=1`

这组验证说明当前“无 Mayfly / 无 memcached”版本至少在：

- 构建
- runner 参数拼接
- 控制面 metadata/ready 协调
- PS runner 参数拼接
- 单机多 client PS/network benchmark 闭环
- RDMA / GRPC / BRPC 同线程矩阵对比

这些层面没有明显回归。

## 8. 当前路线图

当前主线已经从“单纯追更高并发”调整为“按 KVEngine 和 value layout 选择正确的
GET response path”。早期阶段从 `~2M keys/s` 提升到 `qps=16` repeat 平均
`~41.64M keys/s`、单轮最高 `45.62M keys/s`，以及 `qps=20` tuning profile
最高 `~46.69M keys/s`；这些数字只作为历史性能基线，新的 benchmark 复现方式以
本页和 `.agents/skills/rs-benchmark-ps/SKILL.md` 为准。

### 8.1 第一优先级：固化 PET staging-copy 主线

现阶段推荐把下面的组合视为 RDMA GET 默认性能 case：

- `DRAM_PET_HASH`
- `--rdma-get-response-mode auto`
- `rdma_rc_qps_per_client_per_shard=16`
- `batch_keys=500`
- `value_size=512`

后续优化应先保持这条路径可复现，再观察是否接近当前 RDMA transport/device 观测上限
`~48.7M keys/s`。

`rdma_rc_qps_per_client_per_shard=20` 可作为冲高 tuning profile 保留记录，但不作为
默认参数。2026-06-01 的复测显示，`qps=20` 可以单轮冲到 `46.69M keys/s`，
但 repeat 结果更常落在 `41-43M keys/s`。增加 client 进程数、用
`slots_per_qp=2` 加深单 client outstanding、round-robin QP acquisition、client
wait-spin 都是负结果，不应作为默认优化方向。

### 8.2 第二优先级：保留 EH direct-SG 回归

`DRAM_EXTENDIBLE_HASH + direct-SG` 仍然有价值，因为它验证 SG response path 本身，
也能作为 index lookup 瓶颈的稳定对照。当前 EH direct-SG 与 EH storage-only
batch lookup 都在 `~19M keys/s` 附近，说明继续优化 EH 需要先优化 index lookup。

### 8.3 第三优先级：减少 server 空转

先看 `petps_server` 的：

- `poll_loop_ns`
- `empty_scan_rounds`
- `scan_hit_pct`

如果低负载下空扫太多，说明当前轮询模式过重。优先方向是：

- 减少无效扫描
- 让“没有活跃请求”的时候更轻
- 避免 polling thread 大量空转抢 CPU

### 8.4 第四优先级：压低 client 提交和等待成本

再看 `PetPSClient` 的：

- `submit_request_ns`
- `wait_status_ns`
- `copy_response_ns`
- `pending_rpc_peak`

如果低负载下这些值仍然偏高，说明单请求固定开销太大。优先方向是：

- 减少锁争用
- 减少 QP 选择和状态维护开销
- 减少 response copy 和 slot 清理成本

### 8.5 第五优先级：压薄 transport 提交/完成路径

再看 `rc_transport` 的：

- `submit_avg_ns`
- `complete_avg_ns`
- `drain_submit_avg_ns`
- `drain_response_avg_ns`

如果这里高，说明 verbs 提交 / 完成本身已经是瓶颈。优先方向是：

- 减少不必要的 write / complete 往返
- 让提交和完成路径更短
- 避免让背压在低负载下提前出现

### 8.6 不作为主路线的方向

- 继续加 `client-count`
- 继续抬高 `async-depth` 到超过 QP 池承受范围
- 把吞吐增长归因于“更多并发”而不是“更低的固定开销”

## 9. 已知限制和失败模式

### 9.1 QP 资源不足

如果 `client-count` 或 `qps-per-client-per-shard` 太大，可能出现：

- `ibv_create_qp failed`
- `no idle RC write slot available`
- client 初始化阶段直接报错退出

如果是 `slots-per-qp` 太小，更常见的是启动前被 `async_stream` 的容量校验拦住，或者运行时更早触发 `no idle RC write slot available`。

这是资源不足，不是正常性能退化。

### 9.2 参数不合法

`async_stream` 下如果 `qps-per-client-per-shard * slots-per-qp < async-depth`，benchmark 会直接拒绝启动。

### 9.3 旧二进制

如果看到了看似“不对劲”的行为，先确认 binary 是最新构建的目标。RDMA 这条链路对旧二进制非常敏感。

### 9.4 shard 0 控制面

RDMA runner 不再依赖 `memcached`，也不再依赖 Mayfly 控制面。当前做法是：

- `global_id=0` 的 `petps_server` 在启动时监听一个本地 TCP 控制面端口。
- client 和其他 shard 通过这个控制面交换 `RawVerbsNodeMeta`，并等待所有 server ready。
- runner 默认自动生成 `rdma_rc_namespace`，并在未显式指定时自动分配 `rdma_control_plane_port`。
- 控制面只承担启动期元数据交换和 ready 协调，不参与热路径 GET/PUT/UPDATE 数据面。

常用参数：

| 参数 | 说明 |
|------|------|
| `--rdma-namespace` | RDMA 元数据命名空间；默认 `auto` |
| `--rdma-control-plane-host` | shard 0 控制面监听地址 |
| `--rdma-control-plane-port` | shard 0 控制面监听端口；不传则自动分配 |

## 10. 验证入口

### 10.1 PetPS Integration

单分片：

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.UpdateGetRoundTripSingleShard \
  --rdma-control-plane-host=127.0.0.1 \
  --show-runner-logs \
  --client-timeout=20 \
  --cluster-timeout=35
```

多分片：

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --client-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --rdma-control-plane-host=127.0.0.1 \
  --show-runner-logs \
  --client-timeout=25 \
  --cluster-timeout=45
```

多分片排障时优先检查：

- `distributed_client.num_shards`
- `distributed_client.servers`
- `server-count`
- `num_server_processes`
- key 到 shard 的路由是否一致

当前 integration 覆盖的场景：

- `PutGetRoundTripSingleShard`
- `UpdateGetRoundTripSingleShard`
- `PutGetRoundTripMultiShard`

这些场景由 `run_petps_integration.py` 驱动，实际是否通过仍要看当次运行结果；文档这里只记录覆盖范围，不把一次运行结论长期固化在正文里。

### 10.2 Op-layer RDMA

Op-layer RDMA 使用配置：

```text
src/test/configs/recstore_config.op_rdma.json
```

常用 ctest：

```bash
ctest --test-dir ./build -R "^test_op_runtime_support$|^test_op$" -VV
ctest --test-dir ./build -R "^pytorch_client_test_rdma_basic$" -VV
ctest --test-dir ./build -R "^pytorch_client_test_rdma$|^pytorch_client_test_rdma_auto$" -VV
```

也可以手工运行基本 RDMA 客户端测试：

```bash
RECSTORE_CONFIG=./src/test/configs/recstore_config.op_rdma.json \
RECSTORE_CLIENT_TEST_PHASE=basic \
RECSTORE_RDMA_CONTROL_PLANE_HOST=127.0.0.1 \
python3 src/test/framework/pytorch/test_client.py ./build/lib/lib_recstore_ops.so
```

这些测试的 `SKIP_RETURN_CODE` 是 `77`。skip 只说明 helper 的前置检查没有通过，不等价于真实 RDMA benchmark 一定不可运行。

当前 op-layer RDMA 覆盖：

- `test_op_runtime_support`
- `test_op`
- `pytorch_client_test_rdma_basic`
- `pytorch_client_test_rdma`

这些测试覆盖 PyTorch custom op 到 `RDMAPSClientAdapter -> PetPSClient -> verbs RC` 的 init/write/read、prefetch 和 table-aware update roundtrip；如果 helper 前置条件不满足，`SKIP_RETURN_CODE` 仍然是 `77`。

## 11. 单测和脚本测试

协议 helper 和 wrapper：

```bash
ctest --test-dir ./build -R "^test_allshards_ps_client$" -VV
```

runner 参数拼接：

```bash
python3 -m unittest src/test/scripts/test_petps_cluster_runner.py
python3 -m unittest src/test/scripts/test_run_rdma_rc_transport_benchmark.py
python3 -m unittest src/test/scripts/test_run_rdma_transport_benchmarks.py
```

这些测试不证明 RDMA 数据面可用，只证明协议编码、分片 wrapper 和脚本 plumbing 没有明显回归。其中 `test_run_rdma_rc_transport_benchmark.py` 覆盖：

- `--quiet` 模式只输出 summary / aggregate，不输出 progress 噪声。
- 多 client 流式 stdout 保留行边界，避免多条 `phase=measure summary` 粘连后只解析到第一条。
- 旧的 `--use-local-memcached` 参数不应再出现在生成命令里。

真实 RDMA 数据面至少跑一个最小 benchmark 闭环，推荐先用 `6.3` 的单 client 命令，再用 2-client `--quiet` 命令确认聚合表中的 `clients=2`。

## 12. 排障顺序

1. 确认二进制是最新构建的目标，尤其是 `petps_server`、`ps_transport_benchmark`、`petps_integration_test` 和 `recstore_torch_ops`。
2. 确认参数传到了正确入口：runner 用 RDMA 专项参数，C++ binary 读对应 gflags，op-layer 读 `RECSTORE_CONFIG` 和相关环境变量。
3. 确认 RDMA 设备和 shard 0 控制面可用：检查 `/dev/infiniband`、`ibv_devices`、`ss -ltnp | grep ':25100'` 或 runner 输出的实际控制面端口。
4. 如果 runner 卡在 `control-plane-wait` 或 `startup-wait`，先看 runner 捕获的 shard 0 日志。
5. 如果看到 `unknown command line flag 'rdma_transport_mode'`，通常是跑到了旧 binary，或者当前目标没有链接 RDMA client 相关对象。
6. 如果看到 `error while loading shared libraries: librdkafka.so.1`，先确认 `ldd build/bin/ps_server` 是否能解析到 `/app/RecStore/build/lib/librdkafka.so.1`；必要时设置 `LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}` 再重跑。
7. 如果看到 `unknown argument --use-local-memcached` 或还在传 `RECSTORE_MEMCACHED_*`，说明还在使用旧脚本或旧命令。
8. 如果 RDMA 路径卡住，重点检查 raw verbs buffer 是否注册、QP metadata 是否按 shard/lane 匹配、CQ 是否被错误线程消费、server/client mode 是否一致。

最小日常验证顺序：

```bash
cmake --build ./build --target ps_transport_benchmark rdma_rc_transport_benchmark petps_server recstore_torch_ops -j
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.UpdateGetRoundTripSingleShard \
  --rdma-control-plane-host=127.0.0.1
ctest --test-dir ./build -R "^pytorch_client_test_rdma_basic$" -VV
```
