# 监控与 Grafana

编译时开启性能上报宏，Debug / Release 都可以：

```bash linenums="1" hl_lines="3"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_PERF_REPORT=ON
```

运行时在 `recstore_config.json` 的 `observability.sinks` 里加 `remote`，并让 `observability.remote.endpoint`
（或等价的旧字段 `report_API`）指向接收端，然后启动训练/服务端即可自动上报。字段与本地 CSV 见[配置字段参考](../config.md)。

## 启动接收端

仓库自带一个只依赖标准库的参考实现 `tools/observability/report_service.py`：它校验六个字段的 payload，
批量写入 ClickHouse（`JSONEachRow` over HTTP）或追加 JSONL。先建表，再起服务：

```bash title="创建 ClickHouse 表"
clickhouse-client --multiquery < tools/observability/schema.sql
```

```bash title="启动 Report Service（端口和 observability.remote.endpoint 对齐）"
python3 tools/observability/report_service.py \
    --listen 127.0.0.1:8081 \
    --clickhouse http://127.0.0.1:8123 --database recstore --table metrics
```

没有 ClickHouse 时用 `--jsonl results/report_events.jsonl` 落文件，可以先验证客户端是否接通。
`GET /healthz` 返回 `accepted` / `written` / `failed` / `buffered` 计数。ClickHouse 与 Grafana 的启动方式
取决于部署环境（systemd、容器或直接下载二进制均可）；未配置 endpoint 时客户端默认上报到
`http://127.0.0.1:8081/report`。

客户端只要求接收端对 `POST /report` 返回 2xx，可以换成任何遵守同样 payload 的实现。

## Grafana 看板

表结构与指标定义见[性能埋点](../dev/instrumentation.md)。`tools/observability/grafana_dashboard.json`
提供示例看板（采样率、按指标的 p50/p99、最近指标表），导入后把看板的 `datasource` 变量绑到
ClickHouse 数据源即可。
