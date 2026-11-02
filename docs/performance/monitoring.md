# 监控与 Grafana

!!! note "待重写"

    整理 ClickHouse、Grafana 与 report service 的安装和看板导入步骤。

编译时开启性能上报宏：

```bash linenums="1" hl_lines="2 3"
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_PERF_REPORT=ON
```

运行时确保 `recstore_config.json` 中 `report_API` 指向可写入 ClickHouse 的 report 服务，然后启动训练/服务端即可自动上报，可在 Grafana 端口查看看板。

## 启用服务


```bash title="启动 ClickHouse"
systemctl is-active --quiet clickhouse-server || sudo systemctl start clickhouse-server; echo "✅ ClickHouse is running."
```

```bash title="启动 Grafana"
systemctl is-active --quiet grafana-server || sudo systemctl start grafana-server; echo "✅ Grafana is running (http://localhost:3000)."
```

```bash title="启动 Report Service（端口需要和 recstore_config.json 对齐）"
pkill -f "uvicorn report_service:app" 2>/dev/null; nohup uvicorn report_service:app --host 127.0.0.1 --port 8080 > report_service.log 2>&1 & echo "✅ Report Service is running (http://127.0.0.1:8080)."
```
