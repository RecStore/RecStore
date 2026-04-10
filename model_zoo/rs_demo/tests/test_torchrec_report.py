from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo.runtime.report import finalize_torchrec_row, write_stage_csv
from model_zoo.rs_demo.runners.torchrec_runner import _merge_rank_outputs


class TestTorchRecReport(unittest.TestCase):
    def test_write_stage_csv_includes_kv_columns(self) -> None:
        row = finalize_torchrec_row(
            {
                "backend": "torchrec",
                "batch_size": 256,
                "step": 5,
                "warmup_excluded": 0,
                "collective_mode": "not_measured_single_process",
                "collective_measured": 0,
                "step_total_ms": 10.0,
                "batch_prepare_ms": 1.0,
                "input_pack_ms": 0.5,
                "embed_lookup_local_ms": 2.0,
                "embed_pool_local_ms": 1.0,
                "collective_launch_ms": 0.0,
                "collective_wait_ms": 0.0,
                "output_unpack_ms": 0.7,
                "dense_fwd_ms": 1.1,
                "backward_ms": 1.8,
                "optimizer_ms": 0.9,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "main.csv"
            write_stage_csv(out_path, [row])
            with out_path.open("r", encoding="utf-8") as f:
                line = next(csv.DictReader(f))

        self.assertEqual(line["collective_total_ms"], "0.0")
        self.assertEqual(line["collective_mode"], "not_measured_single_process")
        self.assertEqual(line["collective_measured"], "0")
        self.assertEqual(line["kv_local_only_ms"], "3.0")
        self.assertEqual(line["kv_extended_ms"], "4.2")

    def test_merge_rank_outputs_preserves_rank_and_step_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = Path(tmpdir) / "rank0.csv"
            path1 = Path(tmpdir) / "rank1.csv"
            out_path = Path(tmpdir) / "merged.csv"
            write_stage_csv(
                path0,
                [
                    {
                        "backend": "torchrec",
                        "rank": 1,
                        "step": 2,
                        "collective_mode": "measured_distributed",
                        "collective_measured": 1,
                        "step_total_ms": 20.0,
                    }
                ],
            )
            write_stage_csv(
                path1,
                [
                    {
                        "backend": "torchrec",
                        "rank": 0,
                        "step": 1,
                        "collective_mode": "measured_distributed",
                        "collective_measured": 1,
                        "step_total_ms": 10.0,
                    }
                ],
            )

            rows = _merge_rank_outputs([path0, path1], out_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rank"], 0)
        self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(rows[1]["rank"], 1)
        self.assertEqual(rows[1]["step"], 2)


if __name__ == "__main__":
    unittest.main()
