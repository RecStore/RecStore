import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from tools.benchmarks.run_hierkv_recstore_mixed_benchmark import (  # noqa: E402
    build_recstore_server_cmd,
    build_hierkv_cmd,
    build_recstore_cmd,
    collect_summary_rows,
    print_summary_table,
)


class TestRunHierKVRecStoreMixedBenchmark(unittest.TestCase):
    def test_collect_summary_rows_parses_recstore_and_hierkv(self):
        sample = (
            "system=RecStore transport=GRPC phase=init summary rounds=1 "
            "iterations=1 batch_keys=0 num_embeddings=1000000 elapsed_us_mean=5000 "
            "elapsed_us_p50=5000 elapsed_us_p95=5000 elapsed_us_p99=5000 "
            "ops_per_sec=200000 key_ops_per_sec=200000\n"
            "system=RecStore transport=GRPC phase=measure summary rounds=3 "
            "iterations=100 batch_keys=128 num_embeddings=1000000 elapsed_us_mean=1000 "
            "elapsed_us_p50=900 elapsed_us_p95=1200 elapsed_us_p99=1300 "
            "ops_per_sec=200000 key_ops_per_sec=25600000\n"
            "system=HierKV transport=LOCAL_GPU phase=measure summary rounds=3 "
            "iterations=100 batch_keys=128 num_embeddings=1000000 elapsed_us_mean=200 "
            "elapsed_us_p50=180 elapsed_us_p95=240 elapsed_us_p99=260 "
            "ops_per_sec=1000000 key_ops_per_sec=128000000\n"
        )
        rows = collect_summary_rows(sample)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["system"], "RecStore")
        self.assertEqual(rows[0]["phase"], "init")
        self.assertEqual(rows[2]["system"], "HierKV")
        self.assertEqual(rows[0]["num_embeddings"], 1000000)

    def test_print_summary_table_renders_both_systems(self):
        rows = [
            {
                "system": "RecStore",
                "transport": "GRPC",
                "phase": "init",
                "rounds": 1,
                "iterations": 1,
                "batch_keys": 0,
                "num_embeddings": 1000000,
                "mean": 5000.0,
                "p50": 5000.0,
                "p95": 5000.0,
                "p99": 5000.0,
                "ops": 200000.0,
                "key_ops": 200000.0,
            },
            {
                "system": "RecStore",
                "transport": "GRPC",
                "phase": "measure",
                "rounds": 3,
                "iterations": 100,
                "batch_keys": 128,
                "num_embeddings": 1000000,
                "mean": 1000.0,
                "p50": 900.0,
                "p95": 1200.0,
                "p99": 1300.0,
                "ops": 200000.0,
                "key_ops": 25600000.0,
            },
            {
                "system": "HierKV",
                "transport": "LOCAL_GPU",
                "phase": "measure",
                "rounds": 3,
                "iterations": 100,
                "batch_keys": 128,
                "num_embeddings": 1000000,
                "mean": 200.0,
                "p50": 180.0,
                "p95": 240.0,
                "p99": 260.0,
                "ops": 1000000.0,
                "key_ops": 128000000.0,
            },
        ]
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(rows)
        text = out.getvalue()
        self.assertIn("=== Mixed Benchmark Summary ===", text)
        self.assertIn("| system", text)
        self.assertIn("| HierKV", text)
        self.assertIn("| RecStore", text)
        self.assertIn("| init", text)

    def test_build_recstore_cmd_contains_mixed_parameters(self):
        args = type(
            "Args",
            (),
            {
                "recstore_binary": "/tmp/recstore_mixed_benchmark",
                "transport": "brpc",
                "host": "127.0.0.1",
                "port": 15000,
                "iterations": 100,
                "rounds": 3,
                "warmup_rounds": 1,
                "batch_keys": 128,
                "embedding_dim": 64,
                "num_embeddings": 1000000,
                "init_chunk_size": 8192,
                "report_mode": "summary",
                "update_scale": 0.001,
                "brpc_timeout_ms": 20000,
            },
        )()
        cmd = build_recstore_cmd(args)
        self.assertIn("--embedding_dim=64", cmd)
        self.assertIn("--batch_keys=128", cmd)
        self.assertIn("--update_scale=0.001", cmd)
        self.assertIn("--num_embeddings=1000000", cmd)
        self.assertIn("--init_chunk_size=8192", cmd)
        self.assertIn("--brpc_timeout_ms=20000", cmd)

    def test_build_recstore_server_cmd_uses_brpc_binary_and_flag(self):
        args = type(
            "Args",
            (),
            {
                "recstore_server_binary": "/tmp/brpc_ps_server",
                "recstore_config": "recstore_config.json",
                "transport": "brpc",
            },
        )()
        cmd = build_recstore_server_cmd(args, Path("/app/RecStore"))
        self.assertEqual(cmd[0], "/tmp/brpc_ps_server")
        self.assertEqual(cmd[1], "--brpc_config_path")
        self.assertTrue(cmd[2].endswith("/app/RecStore/recstore_config.json"))

    def test_build_hierkv_cmd_contains_capacity_parameters(self):
        args = type(
            "Args",
            (),
            {
                "hierkv_binary": "/tmp/mixed_benchmark",
                "iterations": 100,
                "rounds": 3,
                "warmup_rounds": 1,
                "batch_keys": 128,
                "embedding_dim": 64,
                "num_embeddings": 1000000,
                "init_chunk_size": 8192,
                "report_mode": "summary",
                "update_scale": 0.001,
                "init_capacity": 0,
                "max_capacity": 0,
                "max_hbm_for_vectors": 4096,
            },
        )()
        cmd = build_hierkv_cmd(args)
        self.assertIn("--init_capacity=1000000", cmd)
        self.assertIn("--max_capacity=1000000", cmd)
        self.assertIn("--max_hbm_for_vectors=4096", cmd)
        self.assertIn("--num_embeddings=1000000", cmd)


if __name__ == "__main__":
    unittest.main()
