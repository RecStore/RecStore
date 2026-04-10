from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.torchrec_trace_report import (
    TRACE_CSV_FIELDS,
    summarize_trace_dir,
    summarize_trace_file,
    write_trace_csv,
)


class TestTorchRecTraceReport(unittest.TestCase):
    def test_summarize_trace_file_classifies_events(self) -> None:
        trace = {
            "traceEvents": [
                {"name": "cudaStreamSynchronize", "dur": 1000},
                {"name": "ncclAlltoall", "dur": 2000},
                {"name": "torch/distributed/distributed_c10d.py(123): all_reduce", "dur": 3000},
                {"name": "threading.py(324): wait", "dur": 700},
                {"name": "input_dist pack", "dur": 4000},
                {"name": "output_dist unpack", "dur": 5000},
                {"name": "other event", "dur": 6000},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "sample.pt.trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            row = summarize_trace_file(trace_path)

        self.assertEqual(row["trace_path"], str(trace_path))
        self.assertEqual(row["rank"], 0)
        self.assertEqual(row["step"], -1)
        self.assertAlmostEqual(row["cuda_stream_sync_ms"], 1.0)
        self.assertAlmostEqual(row["unknown_sync_ms"], 0.7)
        self.assertAlmostEqual(row["nccl_kernel_ms"], 2.0)
        self.assertAlmostEqual(row["unclassified_collective_ms"], 3.0)
        self.assertAlmostEqual(row["input_dist_pack_ms"], 4.0)
        self.assertAlmostEqual(row["output_dist_unpack_ms"], 5.0)
        self.assertAlmostEqual(row["collective_total_ms"], 5.0)

    def test_summarize_trace_dir_returns_one_row_per_file(self) -> None:
        trace = {"traceEvents": [{"name": "cudaStreamSynchronize", "dur": 1000}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            first = dir_path / "a.pt.trace.json"
            second = dir_path / "b.pt.trace.json"
            first.write_text(json.dumps(trace), encoding="utf-8")
            second.write_text(json.dumps(trace), encoding="utf-8")

            rows = summarize_trace_dir(dir_path)

        self.assertEqual(len(rows), 2)
        paths = {row["trace_path"] for row in rows}
        self.assertEqual(paths, {str(first), str(second)})

    def test_write_trace_csv_writes_header_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trace.csv"
            write_trace_csv(csv_path, [])
            lines = csv_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines, [",".join(TRACE_CSV_FIELDS)])

    def test_write_trace_csv_writes_rows(self) -> None:
        trace = {"traceEvents": [{"name": "cudaStreamSynchronize", "dur": 1000}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "worker-rank3-step12.pt.trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            row = summarize_trace_file(trace_path)
            csv_path = Path(tmpdir) / "trace.csv"

            write_trace_csv(csv_path, [row])
            with csv_path.open("r", encoding="utf-8") as f:
                line = next(csv.DictReader(f))

        self.assertEqual(line["trace_path"], str(trace_path))
        self.assertEqual(line["cuda_stream_sync_ms"], "1.0")
        self.assertEqual(line["unknown_sync_ms"], "0.0")
        self.assertEqual(line["rank"], "3")
        self.assertEqual(line["step"], "12")


if __name__ == "__main__":
    unittest.main()
