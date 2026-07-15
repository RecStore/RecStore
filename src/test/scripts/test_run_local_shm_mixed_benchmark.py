import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from tools.benchmarks.run_local_shm_mixed_benchmark import (  # noqa: E402
    build_benchmark_cmd,
    build_local_shm_server_cmd,
    build_runtime_config,
    collect_summary_rows,
    resolve_kv_path,
    print_summary_table,
)


class TestRunLocalShmMixedBenchmark(unittest.TestCase):
    def test_build_runtime_config_contains_local_shm_block(self):
        config = build_runtime_config(
            region_name="bench_region",
            slot_count=8,
            ready_queue_count=4,
            ready_queue_burst_limit=16,
            slot_buffer_bytes=4096,
            client_timeout_ms=2000,
            kv_path="/tmp/bench_kv",
            capacity=2048,
            value_size=64,
        )
        self.assertEqual(config["cache_ps"]["ps_type"], "LOCAL_SHM")
        self.assertEqual(config["local_shm"]["region_name"], "bench_region")
        self.assertEqual(config["local_shm"]["slot_count"], 8)
        self.assertEqual(config["local_shm"]["ready_queue_count"], 4)
        self.assertEqual(config["local_shm"]["ready_queue_burst_limit"], 16)
        self.assertEqual(
            config["cache_ps"]["base_kv_config"]["value"]["path"],
            "/dev/shm/bench_kv/value"
        )

    def test_resolve_kv_path_moves_tmp_runtime_under_dev_shm(self):
        path = resolve_kv_path(Path("/tmp/recstore_local_shm_bench_abc"))
        self.assertEqual(path, Path("/dev/shm/recstore_local_shm_bench_abc/kv_store"))

    def test_resolve_kv_path_keeps_dev_shm_runtime(self):
        path = resolve_kv_path(Path("/dev/shm/recstore_local_shm_bench_abc"))
        self.assertEqual(path, Path("/dev/shm/recstore_local_shm_bench_abc/kv_store"))

    def test_build_local_shm_server_cmd_contains_config_path(self):
        cmd = build_local_shm_server_cmd(
            Path("/tmp/local_shm_ps_server"), Path("/tmp/local_shm_config.json")
        )
        self.assertEqual(cmd[0], "/tmp/local_shm_ps_server")
        self.assertIn("--config_path=/tmp/local_shm_config.json", cmd)

    def test_build_benchmark_cmd_uses_local_shm_transport(self):
        cmd = build_benchmark_cmd(
            benchmark_binary=Path("/tmp/recstore_mixed_benchmark"),
            iterations=10,
            rounds=2,
            warmup_rounds=1,
            batch_keys=32,
            embedding_dim=16,
            num_embeddings=2048,
            report_mode="summary",
            update_scale=0.125,
            table_name="bench_table",
        )
        self.assertEqual(cmd[0], "/tmp/recstore_mixed_benchmark")
        self.assertIn("--transport=local_shm", cmd)
        self.assertIn("--table_name=bench_table", cmd)
        self.assertIn("--embedding_dim=16", cmd)
        self.assertIn("--num_embeddings=2048", cmd)

    def test_collect_summary_rows_keeps_all_summary_phases(self):
        sample = (
            "system=RecStore transport=LOCAL_SHM phase=init summary rounds=1 "
            "iterations=1 batch_keys=0 num_embeddings=2048 "
            "elapsed_us_mean=300 elapsed_us_p50=300 "
            "elapsed_us_p95=300 elapsed_us_p99=300 ops_per_sec=50 key_ops_per_sec=50\n"
            "system=RecStore transport=LOCAL_SHM phase=warmup summary rounds=1 "
            "iterations=10 batch_keys=32 num_embeddings=2048 "
            "elapsed_us_mean=200 elapsed_us_p50=200 "
            "elapsed_us_p95=200 elapsed_us_p99=200 ops_per_sec=100 key_ops_per_sec=3200\n"
            "system=RecStore transport=LOCAL_SHM phase=measure summary rounds=2 "
            "iterations=10 batch_keys=32 num_embeddings=2048 "
            "elapsed_us_mean=100 elapsed_us_p50=90 "
            "elapsed_us_p95=120 elapsed_us_p99=130 ops_per_sec=200 key_ops_per_sec=6400\n"
        )
        rows = collect_summary_rows(sample)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["transport"], "LOCAL_SHM")
        self.assertEqual(rows[0]["phase"], "init")
        self.assertEqual(rows[2]["phase"], "measure")
        self.assertEqual(rows[2]["num_embeddings"], 2048)
        self.assertEqual(rows[2]["rounds"], 2)

    def test_print_summary_table_renders_local_shm(self):
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(
                [
                    {
                        "system": "RecStore",
                        "transport": "LOCAL_SHM",
                        "phase": "init",
                        "rounds": 2,
                        "iterations": 10,
                        "batch_keys": 32,
                        "num_embeddings": 2048,
                        "mean": 100.0,
                        "p50": 90.0,
                        "p95": 120.0,
                        "p99": 130.0,
                        "ops": 200.0,
                        "key_ops": 6400.0,
                    }
                ]
            )
        text = out.getvalue()
        self.assertIn("LOCAL_SHM", text)
        self.assertIn("=== Local SHM Mixed Benchmark Summary ===", text)

    def test_runtime_config_is_json_serializable(self):
        config = build_runtime_config(
            region_name="bench_region",
            slot_count=8,
            ready_queue_count=2,
            ready_queue_burst_limit=8,
            slot_buffer_bytes=4096,
            client_timeout_ms=2000,
            kv_path="/dev/shm/bench_kv",
            capacity=2048,
            value_size=64,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["local_shm"]["region_name"], "bench_region")
        self.assertEqual(loaded["local_shm"]["ready_queue_count"], 2)
        self.assertEqual(loaded["local_shm"]["ready_queue_burst_limit"], 8)


if __name__ == "__main__":
    unittest.main()
