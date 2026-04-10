from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.torchrec_compare import (
    build_compare_rows,
    write_compare_csv,
)


class TestTorchRecCompare(unittest.TestCase):
    def test_build_compare_rows_aligned_stage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            with recstore_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "emb_stage_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                        "sparse_update_ms",
                        "step_total_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "emb_stage_ms": 12.0,
                        "dense_fwd_ms": 4.0,
                        "backward_ms": 5.0,
                        "optimizer_ms": 6.0,
                        "sparse_update_ms": 7.0,
                        "step_total_ms": 30.0,
                    }
                )

            with torchrec_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "emb_stage_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                        "sparse_update_ms",
                        "step_total_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "emb_stage_ms": 10.0,
                        "dense_fwd_ms": 3.0,
                        "backward_ms": 4.0,
                        "optimizer_ms": 5.0,
                        "sparse_update_ms": 6.0,
                        "step_total_ms": 25.0,
                    }
                )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["emb_stage"]["recstore_ms"], 12.0)
        self.assertEqual(by_metric["emb_stage"]["torchrec_ms"], 10.0)
        self.assertEqual(by_metric["dense_fwd"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["backward"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["optimizer"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["sparse_update"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["step_total"]["delta_ms"], 5.0)

    def test_build_compare_rows_prefers_measured_rows_over_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            with recstore_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "warmup_excluded",
                        "emb_stage_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                        "sparse_update_ms",
                        "step_total_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "warmup_excluded": 1,
                        "emb_stage_ms": 100.0,
                        "dense_fwd_ms": 40.0,
                        "backward_ms": 50.0,
                        "optimizer_ms": 60.0,
                        "sparse_update_ms": 70.0,
                        "step_total_ms": 300.0,
                    }
                )
                writer.writerow(
                    {
                        "warmup_excluded": 0,
                        "emb_stage_ms": 12.0,
                        "dense_fwd_ms": 4.0,
                        "backward_ms": 5.0,
                        "optimizer_ms": 6.0,
                        "sparse_update_ms": 7.0,
                        "step_total_ms": 30.0,
                    }
                )

            with torchrec_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "warmup_excluded",
                        "emb_stage_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                        "sparse_update_ms",
                        "step_total_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "warmup_excluded": 1,
                        "emb_stage_ms": 80.0,
                        "dense_fwd_ms": 30.0,
                        "backward_ms": 40.0,
                        "optimizer_ms": 50.0,
                        "sparse_update_ms": 60.0,
                        "step_total_ms": 250.0,
                    }
                )
                writer.writerow(
                    {
                        "warmup_excluded": 0,
                        "emb_stage_ms": 10.0,
                        "dense_fwd_ms": 3.0,
                        "backward_ms": 4.0,
                        "optimizer_ms": 5.0,
                        "sparse_update_ms": 6.0,
                        "step_total_ms": 25.0,
                    }
                )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["dense_fwd"]["recstore_ms"], 4.0)
        self.assertEqual(by_metric["dense_fwd"]["torchrec_ms"], 3.0)
        self.assertEqual(by_metric["step_total"]["recstore_ms"], 30.0)
        self.assertEqual(by_metric["step_total"]["torchrec_ms"], 25.0)

    def test_build_compare_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            with recstore_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "network_transport_us",
                        "storage_backend_update_us",
                        "server_total_us",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "network_transport_us": 2000,
                        "storage_backend_update_us": 3000,
                        "server_total_us": 4000,
                    }
                )

            with torchrec_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "collective_total_ms",
                        "kv_local_only_ms",
                        "kv_extended_ms",
                        "network_proxy_torchrec_extended_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "collective_total_ms": 1.0,
                        "kv_local_only_ms": 2.0,
                        "kv_extended_ms": 3.0,
                        "network_proxy_torchrec_extended_ms": 1.5,
                    }
                )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["network_main"]["recstore_ms"], 2.0)
        self.assertEqual(by_metric["network_main"]["torchrec_ms"], 1.0)
        self.assertEqual(by_metric["kv_strict"]["recstore_ms"], 3.0)
        self.assertEqual(by_metric["kv_strict"]["torchrec_ms"], 2.0)

    def test_write_compare_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "compare.csv"
            write_compare_csv(
                out_path,
                [
                    {
                        "metric": "network_main",
                        "recstore_ms": 2.0,
                        "torchrec_ms": 1.0,
                        "delta_ms": 1.0,
                        "delta_ratio": 1.0,
                    }
                ],
            )

            with out_path.open("r", encoding="utf-8") as f:
                row = next(csv.DictReader(f))

        self.assertEqual(row["metric"], "network_main")
        self.assertEqual(row["delta_ms"], "1.0")


if __name__ == "__main__":
    unittest.main()
