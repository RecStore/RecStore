from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.torchrec_aggregate import (
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
                        "kv_local_only_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step_total_ms": 10.0,
                        "collective_total_ms": 1.0,
                        "kv_local_only_ms": 2.0,
                    }
                )
                writer.writerow(
                    {
                        "step_total_ms": 20.0,
                        "collective_total_ms": 3.0,
                        "kv_local_only_ms": 4.0,
                    }
                )

            agg = aggregate_torchrec_main_csv(path)

        self.assertEqual(agg["row_count"], 2)
        self.assertEqual(agg["step_total_ms_mean"], 15.0)
        self.assertEqual(agg["step_total_ms_p50"], 15.0)
        self.assertEqual(agg["step_total_ms_p95"], 19.5)
        self.assertEqual(agg["step_total_ms_max"], 20.0)

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
