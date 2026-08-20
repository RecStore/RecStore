from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from python.pytorch.recstore.analysis.aggregate import (
    aggregate_torchrec_main_csv,
    write_aggregate_csv,
)


def _by_metric(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int | str]]:
    return {str(row["metric"]): row for row in rows}


class TestTorchRecAggregate(unittest.TestCase):
    def test_aggregate_main_csv_outputs_basic_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "main.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step_total_ms",
                        "collective_total_ms",
                        "embed_transport_ms",
                        "kv_local_only_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step_total_ms": 10.0,
                        "collective_total_ms": 1.0,
                        "embed_transport_ms": 1.5,
                        "kv_local_only_ms": 2.0,
                    }
                )
                writer.writerow(
                    {
                        "step_total_ms": 20.0,
                        "collective_total_ms": 3.0,
                        "embed_transport_ms": 3.5,
                        "kv_local_only_ms": 4.0,
                    }
                )

            agg = _by_metric(aggregate_torchrec_main_csv(path))

        self.assertEqual(agg["row_count"]["mean"], 2.0)
        self.assertEqual(agg["row_count"]["p50"], 2.0)
        self.assertEqual(agg["row_count"]["p95"], 2.0)
        self.assertEqual(agg["row_count"]["max"], 2.0)
        self.assertEqual(agg["step_total_ms"]["mean"], 15.0)
        self.assertEqual(agg["step_total_ms"]["p50"], 15.0)
        self.assertEqual(agg["step_total_ms"]["p95"], 19.5)
        self.assertEqual(agg["step_total_ms"]["max"], 20.0)
        self.assertEqual(agg["embed_transport_ms"]["mean"], 2.5)
        self.assertEqual(agg["embed_transport_ms"]["p50"], 2.5)
        self.assertEqual(agg["embed_transport_ms"]["p95"], 3.4)
        self.assertEqual(agg["embed_transport_ms"]["max"], 3.5)

    def test_aggregate_main_csv_includes_throughput_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "main.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step_total_ms",
                        "samples_per_sec",
                        "batches_per_sec",
                        "batch_raw_ids",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step_total_ms": 10.0,
                        "samples_per_sec": 100.0,
                        "batches_per_sec": 10.0,
                        "batch_raw_ids": 20,
                    }
                )
                writer.writerow(
                    {
                        "step_total_ms": 20.0,
                        "samples_per_sec": 200.0,
                        "batches_per_sec": 20.0,
                        "batch_raw_ids": 40,
                    }
                )

            agg = _by_metric(aggregate_torchrec_main_csv(path))

        self.assertEqual(agg["samples_per_sec"]["mean"], 150.0)
        self.assertEqual(agg["batches_per_sec"]["mean"], 15.0)
        self.assertEqual(agg["batch_raw_ids"]["mean"], 30.0)

    def test_aggregate_csv_to_wide_pivots_long_format(self) -> None:
        from python.pytorch.recstore.analysis.aggregate import aggregate_csv_to_wide

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agg.csv"
            write_aggregate_csv(
                path,
                [
                    {
                        "metric": "row_count",
                        "mean": 2,
                        "p50": 2,
                        "p95": 2,
                        "max": 2,
                    },
                    {
                        "metric": "step_total_ms",
                        "mean": 15.0,
                        "p50": 15.0,
                        "p95": 19.5,
                        "max": 20.0,
                    },
                ],
            )
            wide = aggregate_csv_to_wide(path)

        self.assertEqual(wide["row_count_mean"], "2.00")
        self.assertEqual(wide["step_total_ms_p95"], "19.50")
        self.assertEqual(wide["step_total_ms_max"], "20.00")


if __name__ == "__main__":
    unittest.main()
