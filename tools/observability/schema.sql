-- Schema for the ClickHouse table behind the `remote` report sink.
--
-- The client posts one JSON object per sample with six fields; the physical
-- table keeps exactly those columns, so `report_service.py` can insert with
-- FORMAT JSONEachRow and Grafana can filter by table_name / metric_name.
--
--   clickhouse-client --multiquery < tools/observability/schema.sql

CREATE DATABASE IF NOT EXISTS recstore;

CREATE TABLE IF NOT EXISTS recstore.metrics
(
    timestamp_us Int64,
    table_name   LowCardinality(String),
    unique_id    String,
    metric_name  LowCardinality(String),
    metric_value Float64,
    source       LowCardinality(String),
    ingested_at  DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(toDateTime(intDiv(timestamp_us, 1000000)))
ORDER BY (table_name, metric_name, timestamp_us)
TTL toDateTime(intDiv(timestamp_us, 1000000)) + INTERVAL 30 DAY;
