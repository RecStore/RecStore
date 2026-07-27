from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from python.pytorch.recstore.analysis.aggregate import (
    aggregate_torchrec_main_csv,
    write_aggregate_csv,
)


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

            agg = aggregate_torchrec_main_csv(path)

        self.assertEqual(agg["row_count"], 2)
        self.assertEqual(agg["step_total_ms_mean"], 15.0)
        self.assertEqual(agg["step_total_ms_p50"], 15.0)
        self.assertEqual(agg["step_total_ms_p95"], 19.5)
        self.assertEqual(agg["step_total_ms_max"], 20.0)
        self.assertEqual(agg["embed_transport_ms_mean"], 2.5)
        self.assertEqual(agg["embed_transport_ms_p50"], 2.5)
        self.assertEqual(agg["embed_transport_ms_p95"], 3.4)
        self.assertEqual(agg["embed_transport_ms_max"], 3.5)

    def test_aggregate_main_csv_includes_prefetch_counter_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "main.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step_total_ms",
                        "prefetch_depth",
                        "prefetch_issued_batches",
                        "prefetch_consumed_batches",
                        "prefetch_pending_batches",
                        "prefetch_ready_batches",
                        "prefetch_total_ids",
                        "prefetch_consumed_total_ids",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step_total_ms": 10.0,
                        "prefetch_depth": 2,
                        "prefetch_issued_batches": 1,
                        "prefetch_consumed_batches": 0,
                        "prefetch_pending_batches": 1,
                        "prefetch_ready_batches": 0,
                        "prefetch_total_ids": 10,
                        "prefetch_consumed_total_ids": 0,
                    }
                )
                writer.writerow(
                    {
                        "step_total_ms": 20.0,
                        "prefetch_depth": 2,
                        "prefetch_issued_batches": 1,
                        "prefetch_consumed_batches": 1,
                        "prefetch_pending_batches": 2,
                        "prefetch_ready_batches": 1,
                        "prefetch_total_ids": 14,
                        "prefetch_consumed_total_ids": 14,
                    }
                )

            agg = aggregate_torchrec_main_csv(path)

        self.assertEqual(agg["prefetch_depth_mean"], 2.0)
        self.assertEqual(agg["prefetch_issued_batches_mean"], 1.0)
        self.assertEqual(agg["prefetch_consumed_batches_mean"], 0.5)
        self.assertEqual(agg["prefetch_pending_batches_mean"], 1.5)
        self.assertEqual(agg["prefetch_ready_batches_mean"], 0.5)
        self.assertEqual(agg["prefetch_total_ids_mean"], 12.0)
        self.assertEqual(agg["prefetch_consumed_total_ids_mean"], 7.0)

    def test_write_aggregate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "agg.csv"
            write_aggregate_csv(
                out_path,
                {
                    "row_count": 1,
                    "step_total_ms_mean": 10.0,
                },
            )

            with out_path.open("r", encoding="utf-8") as f:
                row = next(csv.DictReader(f))

        self.assertEqual(row["row_count"], "1")
        self.assertEqual(row["step_total_ms_mean"], "10.0")


if __name__ == "__main__":
    unittest.main()
