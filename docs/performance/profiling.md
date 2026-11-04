# Profiling 与瓶颈分析

性能上报的编译选项见[监控与 Grafana](monitoring.md)。

## 本地分析

该模式无需依赖 ClickHouse/Grafana，可直接在模型训练时采集 `embupdate_stages` 真实链路数据，分析 update 在 OP 层、gRPC 客户端、gRPC 服务端各阶段耗时。

```bash title="开启本地结构化事件落盘（JSONL）"
export RECSTORE_REPORT_MODE=local
export RECSTORE_REPORT_LOCAL_SINK=jsonl
export RECSTORE_REPORT_JSONL_PATH=/tmp/recstore_report_events.jsonl
export RECSTORE_REPORT_FLUSH_EVERY_N=256
```

接下来启动参数服务器与模型训练，可以使用随机数据集的DLRM：

```bash title="运行 DLRM 模型（随机数据），自动拉起ps_server"
cd model_zoo/torchrec_dlrm/
bash ./run_single_day.sh --ps --random-dataset --dataset-size 256
```

使用分析脚本：

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

### DLRM 同源 mock demo

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

## 运行性能统计

### 服务端 (PS Server)

服务端的结构化统计通过 `RECSTORE_REPORT_*` 上报（见[监控与 Grafana](monitoring.md)），编译时打开 `-DUSE_PERF_REPORT=ON`。

CPU Profiler 通过 `CPUPROFILE` 开启，需要带 `-DCMAKE_BUILD_TYPE=Debug
-DUSE_GPERF_PROFILING=ON` 构建：

```bash
CPUPROFILE=/tmp/ps_cpu.prof ./build/bin/ps_server
```

### 训练端 (DLRM Trainer)

使用 `run_single_day.sh` 脚本一键运行模型层测试：

```bash
bash run_single_day.sh --custom --dataset-size 4096 --epochs 10
```

???+ note "关于run_single_day 脚本"
    这是对于模型层 DLRM 的小型测试脚本，可以使用单天数据进行测试，同时限制了数据量和嵌入表的大小。
    运行 `bash run_single_day.sh --help` 可查看更多参数说明。

### CPU Profiling 分析

使用 `google-pprof` 分析 CPU 热点（需要安装 `gperftools`）：

```bash
google-pprof --text build/bin/ps_server /tmp/ps_cpu.prof
# 或者导出 pdf/svg
google-pprof --pdf build/bin/ps_server /tmp/ps_cpu.prof > ps_cpu.pdf
```

### 模型层统计

```bash
tensorboard --logdir=./logs --port=6006
```
