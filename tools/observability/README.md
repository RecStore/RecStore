# Observability deployment assets

Reference pieces for the `remote` sink described in
[`docs/dev/instrumentation.md`](../../docs/dev/instrumentation.md). The C++
client only needs an HTTP endpoint that answers 2xx for `POST /report`; the
files here make that endpoint runnable and queryable.

| File | Purpose |
| --- | --- |
| `report_service.py` | Standard-library HTTP receiver: validates the six-field payload, buffers rows, inserts into ClickHouse (`JSONEachRow` over HTTP) or appends JSONL |
| `schema.sql` | ClickHouse database and table for the payload |
| `grafana_dashboard.json` | Dashboard with sample rate, p50/p99 latency and a recent-metrics table |
| `test_report_service.py` | Unit and end-to-end tests (`python3 -m unittest tools.observability.test_report_service`) |

## Local run without ClickHouse

```bash
python3 tools/observability/report_service.py \
    --listen 127.0.0.1:8081 --jsonl results/report_events.jsonl
```

Point the client at it (the endpoint is config-file only; there is no
environment variable for it):

```json
"observability": {
  "sinks": ["remote"],
  "remote": {"endpoint": "http://127.0.0.1:8081/report"}
}
```

`RECSTORE_REPORT_SINKS=remote` overrides `observability.sinks` when you do not
want to edit the file.

## Local run with ClickHouse

```bash
clickhouse-client --multiquery < tools/observability/schema.sql
python3 tools/observability/report_service.py \
    --listen 127.0.0.1:8081 \
    --clickhouse http://127.0.0.1:8123 --database recstore --table metrics
```

`GET /healthz` reports `accepted` / `written` / `failed` / `buffered` counters,
which is the quickest way to confirm the client is reaching the service.
Import `grafana_dashboard.json` after adding the ClickHouse datasource and bind
the dashboard's `datasource` variable to it.

## Notes

- The payload is intentionally six fields wide. Dimensions such as `host`,
  `pid`, `run_id`, `rank` and `shard` stay in the CSV sink and in custom
  `MetricSink` implementations, so existing report services keep working.
- Inserts are batched (`--batch`, `--flush-ms`) because the C++ client posts
  one object per sample. A failed batch is counted in `/healthz` and dropped
  rather than retried, which keeps the service from growing without bound.
- Any other service that accepts the same payload works too; nothing in the
  client depends on this implementation.
