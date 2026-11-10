# 监控与 Grafana

编译时开启性能上报宏，Debug / Release 都可以：

```bash linenums="1" hl_lines="3"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_PERF_REPORT=ON
```

运行时在 `recstore_config.json` 的 `observability.sinks` 里加 `remote`，并让 `observability.remote.endpoint`
（或等价的旧字段 `report_API`）指向可写入 ClickHouse 的 report 服务，然后启动训练/服务端即可自动上报，
可在 Grafana 端口看板查看。字段与本地 CSV 见[配置字段参考](../config.md)。

## 启用服务


```bash title="启动 ClickHouse"
systemctl is-active --quiet clickhouse-server || sudo systemctl start clickhouse-server; echo "✅ ClickHouse is running."
```

```bash title="启动 Grafana"
systemctl is-active --quiet grafana-server || sudo systemctl start grafana-server; echo "✅ Grafana is running (http://localhost:3000)."
```

```bash title="启动 Report Service（端口需要和 recstore_config.json 对齐）"
pkill -f "uvicorn report_service:app" 2>/dev/null; nohup uvicorn report_service:app --host 127.0.0.1 --port 8081 > report_service.log 2>&1 & echo "✅ Report Service is running (http://127.0.0.1:8081)."
```

`report_service` 不在本仓库内，由部署方自行提供；未配置 `report_API` 时客户端默认上报到 `http://127.0.0.1:8081/report`。

## Grafana 看板

ClickHouse 表与指标定义见[性能埋点](../dev/instrumentation.md)。看板 JSON 不在仓库内，需要按这些表自行导入或配置。
