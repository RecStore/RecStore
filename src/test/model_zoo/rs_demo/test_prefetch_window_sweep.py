from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_zoo.rs_demo import run_prefetch_window_sweep


class TestPrefetchWindowSweep(unittest.TestCase):
    def test_depth_zero_uses_prefetch_same_batch_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            class Args:
                repo_root = Path("/repo")
                output_root = Path(tmpdir)
                run_id_prefix = "prefwin"
                steps = 8
                warmup_steps = 1
                batch_size = 64
                num_embeddings = 5000
                embedding_dim = 128
                ps_type = "BRPC"
                ps_kv_backend = "recstore_dram"
                recstore_index_type = "DRAM_EXTENDIBLE_HASH"
                no_start_server = False
                library_path = ""
                data_dir = ""

            cmd, run_id = run_prefetch_window_sweep._run_cmd(
                Args,
                depth=0,
                capacity=0,
            )

        self.assertEqual(run_id, "prefwin-d0-c0")
        read_mode_idx = cmd.index("--read-mode")
        self.assertEqual(cmd[read_mode_idx + 1], "prefetch")
        prefetch_depth_idx = cmd.index("--prefetch-depth")
        self.assertEqual(cmd[prefetch_depth_idx + 1], "0")


if __name__ == "__main__":
    unittest.main()
