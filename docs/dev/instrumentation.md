# 性能埋点

所有埋点走同一个入口：调用点调 `report(table_name, unique_id, metric_name, value)`，由
`src/base/report/report_client.cpp` 组装成一条 `MetricRecord`，再按配置分发给各个 sink。

```text
report()
  └─ MetricRecord（附带 host / pid / run_id / rank / shard 维度）
      ├─ remote  → 异步 HTTP → report_API → ClickHouse → Grafana
      ├─ csv     → 长表追加（每行一个指标）
      ├─ jsonl   → 一行一个事件
      ├─ glog    → REPORT_LOCAL_EVENT 日志行
      └─ MetricSink → RegisterMetricSink() 注册的自定义 exporter
```

内置 sink 由 `observability` 配置选择，配置字段和环境变量见[配置字段参考](../config.md) 的「观测上报配置」一节；
自定义 sink 走 C++ 接口，不经过配置。sink 的选择不影响调用点，也不改变远端 payload 的格式。

## MetricRecord 字段

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `timestamp_us` | 采集时刻 | Unix 微秒 |
| `host` | `gethostname()` | 节点名 |
| `pid` | `getpid()` | 进程号 |
| `run_id` | `RECSTORE_RUN_ID` | 一次运行的标识，每次 `report()` 重读 |
| `rank` | `RANK`，回退 `LOCAL_RANK` | 训练 rank，未设置时为 `-1` |
| `shard` | `RECSTORE_SHARD_ID`，回退 `SHARD_ID` | PS 分片，未设置时为 `-1` |
| `table_name` / `unique_id` / `metric_name` | 调用点 | 指标定位 |
| `metric_value` | 调用点 | 指标数值 |
| `source` | 调用点 | 默认 `report` |

远端上报只发送 `table_name`、`unique_id`、`metric_name`、`metric_value`、`timestamp_us`、`source`
六个字段，与历史 payload 保持兼容，避免部署方的 `report_service` 因未知字段拒绝；`host` / `pid` /
`run_id` / `rank` / `shard` 只出现在 CSV 和自定义 sink 里。

## CSV 形态

长表，一行一个指标，列固定为：

```csv
timestamp_us,host,pid,run_id,rank,shard,table_name,unique_id,metric_name,metric_value,source
```

追加写，文件为空时写一次表头；字段含逗号、引号或换行时按 RFC 4180 加引号。默认路径
`recstore_metrics_{host}_{pid}.csv`，`{host}` / `{pid}` 用来把 PS server 和各训练 rank 分开。
离线 pivot 可以直接用 `tools/benchmarks/aggregate.py`。

## 常见 report 表与指标

| 表名 | 常见指标名 | 说明 |
| --- | --- | --- |
| `op_latency` | `recstore_us`、`recserver_us` | 端到端或服务端处理耗时 |
| `embread_stages` | `duration_us`、`request_size`、`cache_lookup_us`、`deserialize_duration_us` | EmbRead 细分阶段耗时与规模 |
| `ps_client_latency` | `latency_us` | PS 客户端侧延迟 |
| `ps_server_latency` | `latency_us` | PS 服务端侧延迟 |
| `emb_read_flame_map` | `level`、`value`、`self`、`start` | 火焰图分层结构数据 |

## 自定义 sink

```cpp
class MyExporter : public recstore::MetricSink {
public:
  void Emit(const recstore::MetricRecord& record) override {
    // record 只在本次调用内有效，需要保留就拷贝。
  }
  void Flush() override {}
};

recstore::RegisterMetricSink(std::make_unique<MyExporter>());
```

`Emit` 会被多个线程并发调用，实现必须自己保证线程安全。注册发生在第一次 `report()` 之前才能收到全部样本，
之后再注册只收到后续事件。自定义 sink 与内置 sink 并行，不受 `observability.sinks` 影响。

## 与 stdout 周期表的关系

`xmh::Reporter::StartReportThread()` 每 5 s 往 stdout 打一张 P50/P99 文本表，由 `ps_server` 和
`recstore_pytorch` 启动。`report()` 里 `metric_name` 为 `duration_ns` / `latency_ns` / `duration_us` /
`latency_us` 的样本会同时通过 `xmh::Timer::ManualRecordNs` 计入这张表，其余指标只走 sink。
关掉 sink 不影响 stdout 表，反过来也一样。

## 构建开关

`USE_PERF_REPORT`（默认 `OFF`）控制 `ENABLE_PERF_REPORT` 宏：打开后 Release 构建也会保留各处的
`report()` 调用点并链接 `report_client`，不再要求 Debug；关闭时调用点被编译掉，不会产生样本。
