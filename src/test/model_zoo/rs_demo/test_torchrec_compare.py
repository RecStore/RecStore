from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from python.pytorch.recstore.analysis.compare import (
    build_exposed_gap_rows,
    build_compare_rows,
    write_compare_csv,
)
from model_zoo.rs_demo.analyze_torchrec_gap import write_markdown


class TestTorchRecCompare(unittest.TestCase):
    def test_build_exposed_gap_rows_splits_raw_and_exposed_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recstore_csv = Path(tmpdir) / "recstore_main.csv"
            torchrec_csv = Path(tmpdir) / "torchrec.csv"

            with recstore_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "warmup_excluded",
                        "step_total_ms",
                        "emb_stage_ms",
                        "lookup_total_ms",
                        "embed_lookup_local_ms",
                        "lookup_wait_ms",
                        "planned_gpu_cache_prefill_wait_ms",
                        "prefetch_network_wait_ms",
                        "dense_compute_ms",
                        "lookup_gpu_cache_query_ms",
                        "lookup_gpu_cache_fill_ms",
                        "planned_gpu_cache_prefill_ms",
                        "sparse_update_ms",
                        "update_gpu_cache_invalidate_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "warmup_excluded": 0,
                        "step_total_ms": 12.0,
                        "emb_stage_ms": 3.0,
                        "lookup_total_ms": 2.8,
                        "embed_lookup_local_ms": 3.0,
                        "lookup_wait_ms": 0.2,
                        "planned_gpu_cache_prefill_wait_ms": 0.6,
                        "prefetch_network_wait_ms": 0.8,
                        "dense_compute_ms": 0.5,
                        "lookup_gpu_cache_query_ms": 1.1,
                        "lookup_gpu_cache_fill_ms": 0.2,
                        "planned_gpu_cache_prefill_ms": 0.9,
                        "sparse_update_ms": 4.0,
                        "update_gpu_cache_invalidate_ms": 0.7,
                        "dense_fwd_ms": 1.0,
                        "backward_ms": 1.5,
                        "optimizer_ms": 0.5,
                    }
                )

            with torchrec_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "warmup_excluded",
                        "step_total_ms",
                        "emb_stage_ms",
                        "embed_lookup_local_ms",
                        "sparse_update_ms",
                        "dense_fwd_ms",
                        "backward_ms",
                        "optimizer_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "warmup_excluded": 0,
                        "step_total_ms": 9.0,
                        "emb_stage_ms": 1.5,
                        "embed_lookup_local_ms": 1.4,
                        "sparse_update_ms": 1.0,
                        "dense_fwd_ms": 1.0,
                        "backward_ms": 1.4,
                        "optimizer_ms": 0.4,
                    }
                )

            rows = build_exposed_gap_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertAlmostEqual(by_metric["prefetch_network"]["recstore_raw_ms"], 0.8)
        self.assertAlmostEqual(by_metric["prefetch_network"]["recstore_exposed_ms"], 0.3)
        self.assertAlmostEqual(by_metric["prefetch_network"]["torchrec_exposed_ms"], 0.0)
        self.assertAlmostEqual(by_metric["prefetch_network"]["delta_exposed_ms"], 0.3)
        self.assertEqual(by_metric["prefetch_network"]["bottleneck"], "exposed")
        self.assertAlmostEqual(by_metric["embedding_lookup"]["delta_raw_ms"], 1.4)
        self.assertAlmostEqual(by_metric["gpu_cache_query"]["delta_raw_ms"], 1.1)
        self.assertAlmostEqual(by_metric["sparse_update"]["delta_raw_ms"], 3.0)
        self.assertAlmostEqual(by_metric["step_total"]["delta_raw_ms"], 3.0)

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
                        "embed_transport_ms",
                        "kv_local_only_ms",
                        "kv_extended_ms",
                        "network_proxy_torchrec_extended_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "embed_transport_ms": 1.0,
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

    def test_build_compare_rows_falls_back_to_collective_total(self) -> None:
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
                        "input_pack_ms",
                        "output_unpack_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "collective_total_ms": 1.0,
                        "kv_local_only_ms": 2.0,
                        "kv_extended_ms": 3.0,
                        "input_pack_ms": 0.25,
                        "output_unpack_ms": 0.25,
                    }
                )

            rows = build_compare_rows(recstore_csv, torchrec_csv)

        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["network_main"]["torchrec_ms"], 1.0)
        self.assertEqual(by_metric["network_extended"]["torchrec_ms"], 1.5)

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

    def test_write_exposed_gap_markdown_mentions_cache_and_update_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "gap.md"
            write_markdown(
                out_path,
                [
                    {
                        "metric": "step_total",
                        "recstore_raw_ms": 12.0,
                        "recstore_exposed_ms": 12.0,
                        "torchrec_raw_ms": 9.0,
                        "torchrec_exposed_ms": 9.0,
                        "delta_raw_ms": 3.0,
                        "delta_exposed_ms": 3.0,
                        "bottleneck": "exposed",
                        "note": "step",
                    },
                    {
                        "metric": "embedding_lookup",
                        "recstore_raw_ms": 2.8,
                        "recstore_exposed_ms": 2.8,
                        "torchrec_raw_ms": 1.4,
                        "torchrec_exposed_ms": 1.4,
                        "delta_raw_ms": 1.4,
                        "delta_exposed_ms": 1.4,
                        "bottleneck": "exposed",
                        "note": "lookup",
                    },
                    {
                        "metric": "prefetch_network",
                        "recstore_raw_ms": 0.8,
                        "recstore_exposed_ms": 0.3,
                        "torchrec_raw_ms": 0.0,
                        "torchrec_exposed_ms": 0.0,
                        "delta_raw_ms": 0.8,
                        "delta_exposed_ms": 0.3,
                        "bottleneck": "exposed",
                        "note": "prefetch",
                    },
                    {
                        "metric": "gpu_cache_query",
                        "recstore_raw_ms": 1.1,
                        "recstore_exposed_ms": 1.1,
                        "torchrec_raw_ms": 0.0,
                        "torchrec_exposed_ms": 0.0,
                        "delta_raw_ms": 1.1,
                        "delta_exposed_ms": 1.1,
                        "bottleneck": "exposed",
                        "note": "query",
                    },
                    {
                        "metric": "gpu_cache_prefill",
                        "recstore_raw_ms": 0.9,
                        "recstore_exposed_ms": 0.9,
                        "torchrec_raw_ms": 0.0,
                        "torchrec_exposed_ms": 0.0,
                        "delta_raw_ms": 0.9,
                        "delta_exposed_ms": 0.9,
                        "bottleneck": "exposed",
                        "note": "prefill",
                    },
                    {
                        "metric": "gpu_cache_invalidate",
                        "recstore_raw_ms": 0.7,
                        "recstore_exposed_ms": 0.7,
                        "torchrec_raw_ms": 0.0,
                        "torchrec_exposed_ms": 0.0,
                        "delta_raw_ms": 0.7,
                        "delta_exposed_ms": 0.7,
                        "bottleneck": "exposed",
                        "note": "invalidate",
                    },
                    {
                        "metric": "sparse_update",
                        "recstore_raw_ms": 4.0,
                        "recstore_exposed_ms": 4.0,
                        "torchrec_raw_ms": 1.0,
                        "torchrec_exposed_ms": 1.0,
                        "delta_raw_ms": 3.0,
                        "delta_exposed_ms": 3.0,
                        "bottleneck": "exposed",
                        "note": "update",
                    },
                ],
                recstore_csv=Path("/tmp/rec.csv"),
                torchrec_csv=Path("/tmp/tr.csv"),
            )

            text = out_path.read_text(encoding="utf-8")

        self.assertIn("Paper-Style Exposed Time", text)
        self.assertIn("GPU cache overhead", text)
        self.assertIn("GPU cache signals", text)
        self.assertIn("Sparse update gap", text)
        self.assertIn("nsys profile", text)


if __name__ == "__main__":
    unittest.main()
