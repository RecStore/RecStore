"""Tests for tools/observability/report_service.py."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from tools.observability import report_service as service


def sample_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "table_name": "op_latency",
        "unique_id": "recstore_us",
        "metric_name": "duration_us",
        "metric_value": 12.5,
        "timestamp_us": 1_700_000_000_000_000,
        "source": "report",
    }
    row.update(overrides)
    return row


class ParsePayloadTest(unittest.TestCase):
    def test_single_object_drops_unknown_fields(self) -> None:
        body = json.dumps(sample_row(host="node190", rank=0)).encode()
        rows = service.parse_payload(body)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(service.REQUIRED_FIELDS))
        self.assertEqual(rows[0]["metric_value"], 12.5)

    def test_array_payload(self) -> None:
        body = json.dumps([sample_row(), sample_row(unique_id="other")]).encode()
        rows = service.parse_payload(body)
        self.assertEqual([row["unique_id"] for row in rows], ["recstore_us", "other"])

    def test_integer_metric_value_is_coerced(self) -> None:
        rows = service.parse_payload(json.dumps(sample_row(metric_value=3)).encode())
        self.assertIsInstance(rows[0]["metric_value"], float)

    def test_rejects_invalid_json(self) -> None:
        with self.assertRaises(service.BadPayload):
            service.parse_payload(b"{not json")

    def test_rejects_empty_array(self) -> None:
        with self.assertRaises(service.BadPayload):
            service.parse_payload(b"[]")

    def test_rejects_non_object_row(self) -> None:
        with self.assertRaises(service.BadPayload):
            service.parse_payload(b"[1, 2]")

    def test_rejects_missing_field(self) -> None:
        row = sample_row()
        del row["source"]
        with self.assertRaisesRegex(service.BadPayload, "missing source"):
            service.parse_payload(json.dumps(row).encode())

    def test_rejects_non_numeric_value(self) -> None:
        with self.assertRaisesRegex(service.BadPayload, "metric_value"):
            service.parse_payload(json.dumps(sample_row(metric_value="12.5")).encode())

    def test_rejects_bool_value(self) -> None:
        with self.assertRaisesRegex(service.BadPayload, "metric_value"):
            service.parse_payload(json.dumps(sample_row(metric_value=True)).encode())

    def test_rejects_non_integer_timestamp(self) -> None:
        with self.assertRaisesRegex(service.BadPayload, "timestamp_us"):
            service.parse_payload(json.dumps(sample_row(timestamp_us=1.5)).encode())


class RowBufferTest(unittest.TestCase):
    def test_take_returns_fifo_batches(self) -> None:
        buffer = service.RowBuffer()
        buffer.add([sample_row(unique_id="a"), sample_row(unique_id="b")])
        first = buffer.take(1)
        second = buffer.take(10)
        self.assertEqual([row["unique_id"] for row in first], ["a"])
        self.assertEqual([row["unique_id"] for row in second], ["b"])
        self.assertEqual(buffer.pending(), 0)


class JsonlWriterTest(unittest.TestCase):
    def test_appends_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            writer = service.JsonlWriter(str(path))
            writer.write([sample_row()])
            writer.write([sample_row(unique_id="second")])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["unique_id"], "second")


class ClickHouseWriterTest(unittest.TestCase):
    def test_insert_uses_json_each_row_over_http(self) -> None:
        writer = service.ClickHouseWriter(
            "http://127.0.0.1:8123/", "recstore", "metrics", user="u", password="p"
        )
        response = mock.MagicMock()
        response.status = 200
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        with mock.patch.object(
            service.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            writer.write([sample_row(), sample_row(unique_id="second")])

        request = urlopen.call_args.args[0]
        body = request.data.decode("utf-8").splitlines()
        self.assertEqual(len(body), 2)
        self.assertEqual(json.loads(body[1])["unique_id"], "second")
        self.assertIn("INSERT%20INTO%20recstore.metrics", request.full_url)
        self.assertIn("FORMAT%20JSONEachRow", request.full_url)
        self.assertEqual(
            request.get_header("Authorization"), "Basic dTpw"
        )

    def test_reports_clickhouse_error(self) -> None:
        writer = service.ClickHouseWriter("http://127.0.0.1:8123", "recstore", "metrics")
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8123", 500, "boom", {}, None
        )
        error.read = mock.Mock(return_value=b"Code: 60. Table doesn't exist")
        with mock.patch.object(service.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Table doesn't exist"):
                writer.write([sample_row()])


class FlusherTest(unittest.TestCase):
    def test_flush_writes_all_rows(self) -> None:
        buffer = service.RowBuffer()
        writer = mock.Mock()
        flusher = service.Flusher(buffer, writer, batch=2)
        buffer.add([sample_row() for _ in range(3)])

        written = flusher.flush_once()

        self.assertEqual(written, 3)
        self.assertEqual(writer.write.call_count, 2)
        self.assertEqual(buffer.snapshot(), {
            "accepted": 3, "written": 3, "failed": 0, "buffered": 0,
        })

    def test_failed_batch_is_counted_and_dropped(self) -> None:
        buffer = service.RowBuffer()
        writer = mock.Mock()
        writer.write.side_effect = RuntimeError("down")
        flusher = service.Flusher(buffer, writer, log=lambda _message: None)
        buffer.add([sample_row()])

        flusher.flush_once()

        self.assertEqual(buffer.snapshot()["failed"], 1)
        self.assertEqual(buffer.snapshot()["buffered"], 0)


class ServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "events.jsonl"
        buffer = service.RowBuffer()
        flusher = service.Flusher(
            buffer, service.JsonlWriter(str(self.path)), interval=60.0
        )
        self.server = service.ReportServer(("127.0.0.1", 0), buffer, flusher)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()

    def post(self, payload: object) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            f"{self.base}/report",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_post_then_flush_then_healthz(self) -> None:
        status, body = self.post(sample_row())
        self.assertEqual(status, 202)
        self.assertEqual(body, {"accepted": 1})

        self.server.flusher.flush_once()
        rows = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["table_name"], "op_latency")

        with urllib.request.urlopen(f"{self.base}/healthz", timeout=5) as response:
            health = json.loads(response.read())
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["written"], 1)

    def test_rejects_bad_payload(self) -> None:
        status, body = self.post({"table_name": "op_latency"})
        self.assertEqual(status, 400)
        self.assertIn("missing", str(body["error"]))

    def test_unknown_path_returns_404(self) -> None:
        request = urllib.request.Request(f"{self.base}/nope", method="POST", data=b"{}")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
