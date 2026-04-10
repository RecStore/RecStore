# 性能分析

RecStore 内置了完善的性能埋点与分析机制，涵盖了从 PyTorch OP 层、C++ 客户端到 gRPC 服务端及底层存储的完整链路。

## 1. 快速使用

### Grafana

编译时开启性能上报宏：

```bash linenums="1" hl_lines="2 3"
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_PERF_REPORT=ON
```

=== "本地分析"

    该模式无需依赖 ClickHouse/Grafana，可直接在模型训练时采集 `embupdate_stages` 真实链路数据，分析 update 在 OP 层、gRPC 客户端、gRPC 服务端各阶段耗时。

    ```bash title="开启本地结构化事件落盘（JSONL）"
    export RECSTORE_REPORT_MODE=local
    export RECSTORE_REPORT_LOCAL_SINK=both
    export RECSTORE_REPORT_JSONL_PATH=/tmp/recstore_report_events.jsonl
    ```

    启动参数服务器与模型训练（例如 DLRM）后，使用分析脚本：

    ```bash title="分析模型层真实 update 数据"
    python3 src/test/scripts/analyze_embupdate_stages.py \
      --input /tmp/recstore_report_events.jsonl \
      --group-by-prefix \
      --export-csv /tmp/embupdate_real.csv \
      --top 30
    ```

    脚本会输出：
    - 各阶段统计（mean / p50 / p95 / p99 / max）
    - 近似网络开销：`client_rpc_us - server_total_us`
    - 慢请求 TopN（按同一 trace 聚合）
    - 可直接二次分析的 CSV（如 `/tmp/embupdate_real.csv`）

    ??? tip "DLRM 同源 mock demo"

        目录：`model_zoo/rs_demo`

        用途：
        - 复用 DLRM 相同数据入口：`model_zoo/torchrec_dlrm/processed_day_0_data`
        - 复用 DLRM 相同稀疏组织：26 特征 -> KJT
        - 复用 DLRM 相同 fused id 规则：`(table_idx << fuse_k) + id`
        - 产出 `op_client / brpc_client / brpc_server` 三段 `embupdate_stages`

        最小运行：

        ```bash
        python3 model_zoo/rs_demo/run_mock_stress.py \
          --steps 60 \
          --batch-size 4096 \
          --num-embeddings 200000 \
          --embedding-dim 128 \
          --jsonl /tmp/rs_demo_events.jsonl \
          --csv /tmp/rs_demo_embupdate.csv
        ```

        必看输出：
        - JSONL：`/tmp/rs_demo_events.jsonl`
        - CSV：`/tmp/rs_demo_embupdate.csv`
        - 服务端日志：`/tmp/rs_demo_ps_server.log`

=== "Grafana 在线看板（ClickHouse）"

    运行时确保 `recstore_config.json` 中 `report_API` 指向可写入 ClickHouse 的 report 服务，然后启动训练/服务端即可自动上报，可在 Grafana 端口查看看板。

    ??? info "启用 Grafana 和相关上报命令"

        ```bash title="启动 ClickHouse"
        systemctl is-active --quiet clickhouse-server || sudo systemctl start clickhouse-server; echo "✅ ClickHouse is running."
        ```

        ```bash title="启动 Grafana"
        systemctl is-active --quiet grafana-server || sudo systemctl start grafana-server; echo "✅ Grafana is running (http://localhost:3000)."
        ```

        ```bash title="启动 Report Service（端口需要和 recstore_config.json 对齐）"
        pkill -f "uvicorn report_service:app" 2>/dev/null; nohup uvicorn report_service:app --host 127.0.0.1 --port 8080 > report_service.log 2>&1 & echo "✅ Report Service is running (http://127.0.0.1:8080)."
        ```

### bRPC 内置 CPU Profiler

RecStore 支持直接用 bRPC builtin profiler 进行热点分析，适合 Get/Update 等多种 RPC 混合流量场景。

#### 编译

需在 CMake 配置时打开 `USE_BRPC_CPU_PROFILER`：

```bash linenums="1" hl_lines="4"
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_PERF_REPORT=ON \
    -DUSE_BRPC_CPU_PROFILER=ON \
    -DUSE_GPERF_PROFILING=ON
```

#### 启动

??? info "安装 FlameGraph 来生成火焰图"
    
    ```bash
    cd /tmp
    git clone https://github.com/brendangregg/FlameGraph.git
    ```

启动时需要使用 bRPC 作为网络层通讯，同时不要设置 `CPUPROFILE`，否则会和 builtin profiler 冲突：

```bash title="带参数启用参数服务器"
unset CPUPROFILE
export FLAMEGRAPH_PL_PATH=/tmp/FlameGraph/flamegraph.pl
./build/bin/ps_server --config_path ./recstore_config.json
```

假设服务监听端口为 15000（通常在终端输出地址），可访问：

- `http://127.0.0.1:15000/hotspots/cpu?seconds=10`
- `http://127.0.0.1:15000/pprof/profile?seconds=10`

如需远程访问，可先做端口转发。


???+ info "采样频率与时长"

    采样频率由 `CPUPROFILE_FREQUENCY` 环境变量控制，默认 100（每秒 100 次）。
    采样时长通过 URL 参数 `?seconds=5` 控制。
    其他信息请参考 [bRPC 官方 profiler 文档](https://github.com/apache/brpc/blob/master/docs/cn/cpu_profiler.md)

### 运行性能统计

**服务端 (PS Server)**

可以通过环境变量 `CPUPROFILE` 开启 CPU Profiler，并通过 `--perf_report_path` 指定性能报告输出路径：

```bash
CPUPROFILE=/tmp/ps_cpu.prof ./build/bin/ps_server --perf_report_path=/tmp/ps_perf.log
```

**训练端 (DLRM Trainer)**

使用 `run_single_day.sh` 脚本一键运行性能测试：

```bash
export RECSTORE_PERF_REPORT_PATH=/tmp/trainer_perf.log
export RECSTORE_PERF_INTERVAL_MS=5000
bash run_single_day.sh --custom --dataset-size 4096 --epochs 10
```

???+ note "关于run_single_day 脚本"
    这是对于模型层 DLRM 的小型测试脚本，可以使用单天数据进行测试，同时限制了数据量和嵌入表的大小。
    运行 `bash run_single_day.sh --help` 可查看更多参数说明。

### 查看结果

**日志分析**

`perf_report_path` 指定的文件会实时记录每个时间窗口（如 5000ms）内的耗时统计（P50/P99）：

```bash
# 查看服务端性能概览
tail -n 50 /tmp/ps_perf.log

# 查看训练端性能概览
tail -n 50 /tmp/trainer_perf.log
```

**CPU Profiling 分析**

使用 `google-pprof` 分析 CPU 热点（需要安装 `gperftools`）：

```bash
google-pprof --text build/bin/grpc_ps_server /tmp/ps_cpu.prof
# 或者导出 pdf/svg
google-pprof --pdf build/bin/grpc_ps_server /tmp/ps_cpu.prof > ps_cpu.pdf
```

**模型层统计**

```bash
tensorboard --logdir=./logs --port=6006
```

## 2. 性能埋点关键字

??? warning "本段落内容在最新版本可能被弃用，请参考下面 Grafana report"

    RecStore 使用 `xmh::Timer` (位于 `src/base/timer.h`) 进行全链路的耗时打点。数据流向如下：

    ### 写路径自顶向下

    | 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
    | :--- | :--- | :--- | :--- |
    | **PyTorch OP** | `OP.EmbWrite.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
    | | `OP.EmbWrite.Call` | 核心逻辑调用耗时 | |
    | **C++ Client** | `ClientOp.EmbWrite.Total` | 客户端操作总耗时 | `src/framework/op.cc` |
    | | `ClientOp.EmbWrite.BuildVector` | Tensor 转 C++ Vector 的开销 | |
    | **gRPC Client** | `Client.PutParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
    | | `Client.PutParameter.Serialize` | 序列化 Data Request 耗时 | |
    | | `Client.PutParameter.RPC` | 网络传输 + 服务端处理总耗时 | |
    | **Server** | `PS.PutParameter.Handle` | 服务端处理总耗时 | `src/ps/grpc/grpc_ps_server.cpp` |
    | | `PS.PutParameter.KVPutAll` | 写入底层 KV 存储的耗时 | |

    ### 读路径自顶向下

    | 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
    | :--- | :--- | :--- | :--- |
    | **PyTorch OP** | `OP.EmbRead.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
    | | `OP.EmbRead.ToCPUKeys` | Embedding Key 从 GPU 拷贝到 CPU 的耗时 | |
    | **gRPC Client** | `Client.GetParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
    | | `Client.GetParameter.AsyncWait` | 异步等待网络回包的耗时（反映服务端处理+网络延迟） | |
    | **Server** | `KV BatchGet` | 底层 KV 引擎批量读取耗时 | `src/storage/kv_engines.md` |

    ??? tip "什么是 Timer?"
        `xmh::Timer` 是一个高性能的 C++ 计时器工具类。它通过 `Timer::Start("Name")` 和 `Timer::Stop("Name")` 记录代码块耗时，并自动统计 P50、P99 等分位数值，定期输出到日志文件中。


## 3. Grafana report

RecStore 的 Grafana 看板基于 `report()` 上报链路，完整链路为：

`RecStore (C++/Python 调用 report)` -> `HTTP report_API` -> `ClickHouse` -> `Grafana`

#### 上报协议与入口

- 上报接口由 `recstore_config.json` 中 `report_API` 指定（默认：`http://127.0.0.1:8081/report`）。
- `report()` 会构造 JSON 并异步发送，核心字段如下：

| 字段名 | 含义 | 示例 |
| :--- | :--- | :--- |
| `table_name` | 目标报表/逻辑表名 | `op_latency` |
| `unique_id` | 单次请求或链路标识 | `EmbRead|1711111111000000` |
| `metric_name` | 指标名称 | `recstore_us` |
| `metric_value` | 指标数值（double） | `532.0` |

对应实现可参考：

- `src/base/report/report_client.cpp`
- `src/base/report/report_client.h`

#### 常见 report 表与指标（来自当前代码）

| 表名 | 常见指标名 | 说明 |
| :--- | :--- | :--- |
| `op_latency` | `recstore_us`、`recserver_us` | 端到端或服务端处理耗时 |
| `embread_stages` | `duration_us`、`request_size`、`cache_lookup_us`、`deserialize_duration_us` | EmbRead 细分阶段耗时与规模 |s
| `ps_client_latency` | `latency_us` | PS 客户端侧延迟 |
| `ps_server_latency` | `latency_us` | PS 服务端侧延迟 |
| `emb_read_flame_map` | `level`、`value`、`self`、`start` | 火焰图分层结构数据 |

典型调用位置：

- `src/framework/op.cc`
- `src/ps/grpc/grpc_ps_server.cpp`
- `src/ps/brpc/brpc_ps_client.cpp`
- `src/ps/brpc/brpc_ps_server.cpp`
- `src/ps/grpc/grpc_ps_client.cpp`
- `src/ps/base/cache_ps_impl.h`
