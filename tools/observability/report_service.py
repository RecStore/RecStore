#!/usr/bin/env python3
"""Reference receiver for the RecStore ``remote`` report sink.

The C++ client (``src/base/report/report_client.cpp``) posts one JSON object
per sample to ``POST /report``::

    {"table_name": "...", "unique_id": "...", "metric_name": "...",
     "metric_value": 1.0, "timestamp_us": 0, "source": "..."}

This service validates the payload, buffers rows and writes them either to
ClickHouse (``JSONEachRow`` over the HTTP interface) or to a JSONL file. It
depends on the Python standard library only, so it can run next to the
benchmarks without an extra virtualenv.

Example::

    python3 tools/observability/report_service.py \
        --listen 127.0.0.1:8081 \
        --clickhouse http://127.0.0.1:8123 \
        --database recstore --table metrics

    python3 tools/observability/report_service.py \
        --listen 127.0.0.1:8081 --jsonl recstore_metrics.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

REQUIRED_FIELDS = (
    "table_name",
    "unique_id",
    "metric_name",
    "metric_value",
    "timestamp_us",
    "source",
)

STRING_FIELDS = ("table_name", "unique_id", "metric_name", "source")


class BadPayload(ValueError):
    """Raised when a request body is not a valid metric payload."""


def parse_payload(body: bytes) -> list[dict[str, Any]]:
    """Validate a request body and return rows with the six wire fields."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadPayload(f"invalid JSON: {exc}") from exc

    items = payload if isinstance(payload, list) else [payload]
    if not items:
        raise BadPayload("empty payload")
    return [_normalize_row(item, index) for index, item in enumerate(items)]


def _normalize_row(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise BadPayload(f"row {index} is not a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise BadPayload(f"row {index} is missing {', '.join(missing)}")

    row: dict[str, Any] = {}
    for field in STRING_FIELDS:
        value = item[field]
        if not isinstance(value, str):
            raise BadPayload(f"row {index} field {field} must be a string")
        row[field] = value

    value = item["metric_value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadPayload(f"row {index} field metric_value must be a number")
    row["metric_value"] = float(value)

    timestamp = item["timestamp_us"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise BadPayload(f"row {index} field timestamp_us must be an integer")
    row["timestamp_us"] = timestamp
    return row


class RowBuffer:
    """Thread-safe hand-off between request handlers and the flusher."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.accepted = 0
        self.written = 0
        self.failed = 0

    def add(self, rows: Iterable[dict[str, Any]]) -> int:
        batch = list(rows)
        with self._lock:
            self._rows.extend(batch)
            self.accepted += len(batch)
            return len(self._rows)

    def take(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if not self._rows:
                return []
            batch = self._rows[:limit]
            del self._rows[:limit]
            return batch

    def pending(self) -> int:
        with self._lock:
            return len(self._rows)

    def mark_written(self, count: int) -> None:
        with self._lock:
            self.written += count

    def mark_failed(self, count: int) -> None:
        with self._lock:
            self.failed += count

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "accepted": self.accepted,
                "written": self.written,
                "failed": self.failed,
                "buffered": len(self._rows),
            }


class JsonlWriter:
    """Append rows as JSON lines; used for local runs and tests."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def write(self, rows: list[dict[str, Any]]) -> None:
        payload = "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        )
        with self._lock, open(self._path, "a", encoding="utf-8") as handle:
            handle.write(payload)


class ClickHouseWriter:
    """Insert rows through the ClickHouse HTTP interface."""

    def __init__(
        self,
        url: str,
        database: str,
        table: str,
        user: str = "",
        password: str = "",
        timeout: float = 5.0,
    ) -> None:
        for name, value in (("database", database), ("table", table)):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise SystemExit(f"invalid {name} name: {value!r}")
        self._endpoint = (
            f"{url.rstrip('/')}/?query=INSERT%20INTO%20{database}.{table}"
            "%20FORMAT%20JSONEachRow"
        )
        self._timeout = timeout
        self._auth = None
        if user or password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self._auth = f"Basic {token}"

    def write(self, rows: list[dict[str, Any]]) -> None:
        body = "\n".join(
            json.dumps(row, separators=(",", ":")) for row in rows
        ).encode("utf-8")
        headers = {"Content-Type": "application/x-ndjson"}
        if self._auth:
            headers["Authorization"] = self._auth
        request = urllib.request.Request(
            self._endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                if response.status != 200:
                    detail = response.read().decode("utf-8", "replace").strip()
                    raise RuntimeError(f"ClickHouse returned {response.status}: {detail}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise RuntimeError(f"ClickHouse returned {exc.code}: {detail}") from exc


class Flusher(threading.Thread):
    """Move buffered rows to the writer, periodically and on demand."""

    def __init__(
        self,
        buffer: RowBuffer,
        writer: Any,
        batch: int = 200,
        interval: float = 1.0,
        log: Any = print,
    ) -> None:
        super().__init__(name="report-flusher", daemon=True)
        self._buffer = buffer
        self._writer = writer
        self._batch = max(1, batch)
        self._interval = max(0.01, interval)
        self._log = log
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            self.flush_once()
        self.flush_once()

    def stop(self) -> None:
        self._stop.set()

    def flush_once(self) -> int:
        written = 0
        while True:
            rows = self._buffer.take(self._batch)
            if not rows:
                return written
            try:
                self._writer.write(rows)
            except Exception as exc:  # noqa: BLE001 - keep the service alive
                self._buffer.mark_failed(len(rows))
                self._log(f"report_service: write failed ({len(rows)} rows): {exc}")
            else:
                self._buffer.mark_written(len(rows))
                written += len(rows)


class ReportServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        buffer: RowBuffer,
        flusher: Flusher,
        eager_flush_rows: int = 2000,
    ) -> None:
        super().__init__(address, ReportHandler)
        self.buffer = buffer
        self.flusher = flusher
        self.eager_flush_rows = max(1, eager_flush_rows)

    def server_close(self) -> None:
        self.flusher.stop()
        self.flusher.flush_once()
        super().server_close()


class ReportHandler(BaseHTTPRequestHandler):
    server: ReportServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?", 1)[0] not in ("/healthz", "/"):
            self._reply(404, {"error": "not found"})
            return
        status = self.server.buffer.snapshot()
        status["status"] = "ok"
        self._reply(200, status)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.split("?", 1)[0] not in ("/report", "/"):
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._reply(400, {"error": "invalid Content-Length"})
            return
        if length <= 0:
            self._reply(400, {"error": "empty body"})
            return
        try:
            rows = parse_payload(self.rfile.read(length))
        except BadPayload as exc:
            self._reply(400, {"error": str(exc)})
            return
        buffered = self.server.buffer.add(rows)
        if buffered >= self.server.eager_flush_rows:
            self.server.flusher.flush_once()
        self._reply(202, {"accepted": len(rows)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            f"{time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}\n"
        )

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_writer(options: argparse.Namespace) -> Any:
    if options.jsonl:
        return JsonlWriter(options.jsonl)
    return ClickHouseWriter(
        url=options.clickhouse,
        database=options.database,
        table=options.table,
        user=options.clickhouse_user,
        password=options.clickhouse_password,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listen",
        default="127.0.0.1:8081",
        help="host:port to bind (default: 127.0.0.1:8081)",
    )
    parser.add_argument(
        "--clickhouse",
        default="http://127.0.0.1:8123",
        help="ClickHouse HTTP endpoint (default: http://127.0.0.1:8123)",
    )
    parser.add_argument("--database", default="recstore")
    parser.add_argument("--table", default="metrics")
    parser.add_argument(
        "--jsonl",
        default="",
        help="append rows to this JSONL file instead of writing to ClickHouse",
    )
    parser.add_argument(
        "--clickhouse-user",
        default=os.environ.get("CLICKHOUSE_USER", ""),
        help="ClickHouse user (default: $CLICKHOUSE_USER, else no authentication)",
    )
    parser.add_argument(
        "--clickhouse-password",
        default=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        help="ClickHouse password (default: $CLICKHOUSE_PASSWORD)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=200,
        help="rows per insert (default: 200)",
    )
    parser.add_argument(
        "--flush-ms",
        type=int,
        default=1000,
        help="buffer flush interval in milliseconds (default: 1000)",
    )
    return parser.parse_args(argv)


def parse_listen(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"--listen expects host:port, got {value!r}")
    return host, int(port)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    buffer = RowBuffer()
    flusher = Flusher(
        buffer,
        build_writer(options),
        batch=options.batch,
        interval=options.flush_ms / 1000.0,
    )
    server = ReportServer(parse_listen(options.listen), buffer, flusher)
    flusher.start()
    target = "ClickHouse " + options.clickhouse if not options.jsonl else options.jsonl
    print(f"report_service: listening on {options.listen}, writing to {target}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"report_service: stopped {buffer.snapshot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
