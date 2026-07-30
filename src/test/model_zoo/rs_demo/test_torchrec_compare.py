from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from python.pytorch.recstore.analysis.compare import (
    build_exposed_gap_rows,
    build_compare_rows,
)


def _write_csv(path: Path, fieldnames: list[str], *rows: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestTorchRecCompare(unittest.TestCase):
    def test_build_exposed_gap_rows_splits_raw_and_exposed_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            _write_csv(
                recstore_csv,
                [
                    "warmup_excluded",
                    "step_total_ms",
                    "emb_stage_ms",
                    "lookup_total_ms",
                    "embed_lookup_local_ms",
                    "lookup_wait_ms",
                    "dense_compute_ms",
                    "sparse_optimizer_ms",
                    "dense_fwd_ms",
                    "backward_ms",
                    "dense_optimizer_ms",
                ],
                {
                    "warmup_excluded": 0,
                    "step_total_ms": 12.0,
                    "emb_stage_ms": 3.0,
                    "lookup_total_ms": 2.8,
                    "embed_lookup_local_ms": 3.0,
                    "lookup_wait_ms": 0.2,
                    "dense_compute_ms": 0.5,
                    "sparse_optimizer_ms": 4.0,
                    "dense_fwd_ms": 1.0,
                    "backward_ms": 1.5,
                    "dense_optimizer_ms": 0.5,
                },
            )

            _write_csv(
                torchrec_csv,
                [
                    "warmup_excluded",
                    "step_total_ms",
                    "emb_stage_ms",
                    "embed_lookup_local_ms",
                    "sparse_optimizer_ms",
                    "dense_fwd_ms",
                    "backward_ms",
                    "dense_optimizer_ms",
                ],
                {
                    "warmup_excluded": 0,
                    "step_total_ms": 9.0,
                    "emb_stage_ms": 1.5,
                    "embed_lookup_local_ms": 1.4,
                    "sparse_optimizer_ms": 1.0,
                    "dense_fwd_ms": 1.0,
                    "backward_ms": 1.4,
                    "dense_optimizer_ms": 0.4,
                },
            )

            rows = build_exposed_gap_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertAlmostEqual(by_metric["prefetch_network"]["recstore_raw_ms"], 0.2)
        self.assertAlmostEqual(by_metric["prefetch_network"]["recstore_exposed_ms"], 0.0)
        self.assertAlmostEqual(by_metric["prefetch_network"]["torchrec_exposed_ms"], 0.0)
        self.assertAlmostEqual(by_metric["prefetch_network"]["delta_exposed_ms"], 0.0)
        self.assertEqual(by_metric["prefetch_network"]["bottleneck"], "raw_only")
        self.assertAlmostEqual(by_metric["embedding_lookup"]["delta_raw_ms"], 1.4)
        self.assertAlmostEqual(by_metric["sparse_optimizer"]["delta_raw_ms"], 3.0)
        self.assertAlmostEqual(by_metric["step_total"]["delta_raw_ms"], 3.0)

    def test_build_compare_rows_aligned_stage_metrics(self) -> None:
        stage_fields = [
            "emb_stage_ms",
            "dense_fwd_ms",
            "backward_ms",
            "dense_optimizer_ms",
            "sparse_optimizer_ms",
            "step_total_ms",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            _write_csv(
                recstore_csv,
                stage_fields,
                {
                    "emb_stage_ms": 12.0,
                    "dense_fwd_ms": 4.0,
                    "backward_ms": 5.0,
                    "dense_optimizer_ms": 6.0,
                    "sparse_optimizer_ms": 7.0,
                    "step_total_ms": 30.0,
                },
            )
            _write_csv(
                torchrec_csv,
                stage_fields,
                {
                    "emb_stage_ms": 10.0,
                    "dense_fwd_ms": 3.0,
                    "backward_ms": 4.0,
                    "dense_optimizer_ms": 5.0,
                    "sparse_optimizer_ms": 6.0,
                    "step_total_ms": 25.0,
                },
            )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["emb_stage"]["recstore_ms"], 12.0)
        self.assertEqual(by_metric["emb_stage"]["torchrec_ms"], 10.0)
        self.assertEqual(by_metric["dense_fwd"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["backward"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["dense_optimizer"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["sparse_optimizer"]["delta_ms"], 1.0)
        self.assertEqual(by_metric["step_total"]["delta_ms"], 5.0)

    def test_build_compare_rows_prefers_measured_rows_over_warmup(self) -> None:
        stage_fields = [
            "warmup_excluded",
            "emb_stage_ms",
            "dense_fwd_ms",
            "backward_ms",
            "dense_optimizer_ms",
            "sparse_optimizer_ms",
            "step_total_ms",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            _write_csv(
                recstore_csv,
                stage_fields,
                {
                    "warmup_excluded": 1,
                    "emb_stage_ms": 100.0,
                    "dense_fwd_ms": 40.0,
                    "backward_ms": 50.0,
                    "dense_optimizer_ms": 60.0,
                    "sparse_optimizer_ms": 70.0,
                    "step_total_ms": 300.0,
                },
                {
                    "warmup_excluded": 0,
                    "emb_stage_ms": 12.0,
                    "dense_fwd_ms": 4.0,
                    "backward_ms": 5.0,
                    "dense_optimizer_ms": 6.0,
                    "sparse_optimizer_ms": 7.0,
                    "step_total_ms": 30.0,
                },
            )
            _write_csv(
                torchrec_csv,
                stage_fields,
                {
                    "warmup_excluded": 1,
                    "emb_stage_ms": 80.0,
                    "dense_fwd_ms": 30.0,
                    "backward_ms": 40.0,
                    "dense_optimizer_ms": 50.0,
                    "sparse_optimizer_ms": 60.0,
                    "step_total_ms": 250.0,
                },
                {
                    "warmup_excluded": 0,
                    "emb_stage_ms": 10.0,
                    "dense_fwd_ms": 3.0,
                    "backward_ms": 4.0,
                    "dense_optimizer_ms": 5.0,
                    "sparse_optimizer_ms": 6.0,
                    "step_total_ms": 25.0,
                },
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

            _write_csv(
                recstore_csv,
                [
                    "network_transport_us",
                    "storage_backend_update_us",
                    "server_total_us",
                ],
                {
                    "network_transport_us": 2000,
                    "storage_backend_update_us": 3000,
                    "server_total_us": 4000,
                },
            )
            _write_csv(
                torchrec_csv,
                [
                    "embed_transport_ms",
                    "kv_local_only_ms",
                    "kv_extended_ms",
                    "network_proxy_torchrec_extended_ms",
                ],
                {
                    "embed_transport_ms": 1.0,
                    "kv_local_only_ms": 2.0,
                    "kv_extended_ms": 3.0,
                    "network_proxy_torchrec_extended_ms": 1.5,
                },
            )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["network_main"]["recstore_ms"], 2.0)
        self.assertEqual(by_metric["network_main"]["torchrec_ms"], 1.0)
        self.assertEqual(by_metric["kv_strict"]["recstore_ms"], 3.0)
        self.assertEqual(by_metric["kv_strict"]["torchrec_ms"], 2.0)

    def test_build_compare_rows_falls_back_to_collective_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            _write_csv(
                recstore_csv,
                [
                    "network_transport_us",
                    "storage_backend_update_us",
                    "server_total_us",
                ],
                {
                    "network_transport_us": 2000,
                    "storage_backend_update_us": 3000,
                    "server_total_us": 4000,
                },
            )
            _write_csv(
                torchrec_csv,
                [
                    "collective_total_ms",
                    "kv_local_only_ms",
                    "kv_extended_ms",
                    "input_pack_ms",
                    "output_unpack_ms",
                ],
                {
                    "collective_total_ms": 1.0,
                    "kv_local_only_ms": 2.0,
                    "kv_extended_ms": 3.0,
                    "input_pack_ms": 0.25,
                    "output_unpack_ms": 0.25,
                },
            )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["network_main"]["torchrec_ms"], 1.0)
        self.assertEqual(by_metric["network_extended"]["torchrec_ms"], 1.5)


if __name__ == "__main__":
    unittest.main()
