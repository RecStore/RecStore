import sys
import unittest
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
)
from tools.benchmarks.run_rdma_transport_benchmarks import (  # noqa: E402
    build_benchmark_cmd,
    build_rdma_runner,
    collect_summary_rows,
    is_benchmark_noise_line,
    load_client_endpoint,
    print_summary_table,
)


class TestRunRDMATransportBenchmarks(unittest.TestCase):
    def test_rdma_runner_uses_rdma_specific_config(self):
        args = SimpleNamespace(
            show_runner_logs=False,
            batch_keys=500,
            rdma_thread_num=4,
            rdma_namespace="bench-ns",
            rdma_control_plane_host="127.0.0.2",
            rdma_control_plane_port=25001,
            rdma_put_protocol_version=1,
            rdma_put_v2_transfer_mode="read",
            rdma_put_v2_push_slot_bytes=262144,
            rdma_put_v2_push_slots_per_client=16,
            rdma_put_v2_push_region_offset=73400320,
            rdma_put_client_send_arena_bytes=123456,
            rdma_put_server_scratch_bytes=654321,
            rdma_wait_timeout_ms=15000,
            rdma_rc_server_get_workers=2,
        )

        runner = build_rdma_runner(args)

        expected = (Path("/app/RecStore") / DEFAULT_RDMA_SINGLE_SHARD_CONFIG).resolve()
        self.assertEqual(runner.config_path, expected)
        self.assertEqual(runner.thread_num, 4)
        self.assertEqual(runner.max_kv_num_per_request, 500)
        self.assertEqual(runner.rdma_namespace, "bench-ns")
        self.assertEqual(runner.rdma_control_plane_host, "127.0.0.2")
        self.assertEqual(runner.rdma_control_plane_port, 25001)
        self.assertEqual(runner.rdma_put_protocol_version, 1)
        self.assertEqual(runner.rdma_put_v2_transfer_mode, "read")
        self.assertEqual(runner.rdma_put_v2_push_slot_bytes, 262144)
        self.assertEqual(runner.rdma_put_v2_push_slots_per_client, 16)
        self.assertEqual(runner.rdma_put_v2_push_region_offset, 73400320)
        self.assertEqual(runner.rdma_put_client_send_arena_bytes, 123456)
        self.assertEqual(runner.rdma_put_server_scratch_bytes, 654321)
        self.assertEqual(runner.rdma_wait_timeout_ms, 15000)
        self.assertEqual(runner.rdma_server_get_workers, 2)

    def test_load_client_endpoint_for_default_grpc_config(self):
        host, port = load_client_endpoint(DEFAULT_GRPC_MAIN_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 15000)

    def test_load_client_endpoint_for_brpc_config(self):
        host, port = load_client_endpoint(DEFAULT_BRPC_BENCHMARK_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 25000)

    def test_build_benchmark_cmd_includes_report_mode(self):
        cmd = build_benchmark_cmd(
            benchmark_binary="./build/bin/ps_transport_benchmark",
            transport="rdma",
            iterations=20,
            rounds=50,
            warmup_rounds=5,
            report_mode="summary",
            batch_keys=500,
            host=None,
            port=None,
            num_shards=1,
        )

        self.assertIn("--report_mode=summary", cmd)
        self.assertIn("--warmup_rounds=5", cmd)
        self.assertIn("--batch_keys=500", cmd)

    def test_build_benchmark_cmd_does_not_forward_server_only_rdma_flags(self):
        cmd = build_benchmark_cmd(
            benchmark_binary="./build/bin/ps_transport_benchmark",
            transport="rdma",
            iterations=20,
            rounds=50,
            warmup_rounds=5,
            report_mode="summary",
            batch_keys=500,
            host=None,
            port=None,
            num_shards=1,
        )

        rendered = " ".join(cmd)
        self.assertNotIn("rdma_put_protocol_version", rendered)
        self.assertNotIn("rdma_put_v2_transfer_mode", rendered)

    def test_filters_common_rdma_noise_lines(self):
        self.assertTrue(is_benchmark_noise_line("I open mlx5_0 :)"))
        self.assertFalse(is_benchmark_noise_line("transport=RDMA phase=measure"))

    def test_collect_summary_rows_parses_measure_summary(self):
        sample = (
            "transport=RDMA op=put phase=warmup summary rounds=5 iterations=20 "
            "batch_keys=4 "
            "elapsed_us_mean=300 elapsed_us_p50=290 elapsed_us_p95=350 "
            "elapsed_us_p99=360 ops_per_sec=100000 key_ops_per_sec=400000\n"
            "transport=RDMA op=put phase=measure summary rounds=50 iterations=20 "
            "batch_keys=4 "
            "elapsed_us_mean=250 elapsed_us_p50=248 elapsed_us_p95=264 "
            "elapsed_us_p99=270 ops_per_sec=159413.35 key_ops_per_sec=637653.4\n"
            "transport=RDMA op=get phase=measure summary rounds=50 iterations=20 "
            "batch_keys=4 "
            "elapsed_us_mean=180 elapsed_us_p50=175 elapsed_us_p95=195 "
            "elapsed_us_p99=205 ops_per_sec=210000 key_ops_per_sec=840000\n"
        )
        rows = collect_summary_rows(sample)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["transport"], "RDMA")
        self.assertEqual(rows[0]["op"], "put")
        self.assertEqual(rows[0]["rounds"], 50)
        self.assertEqual(rows[0]["batch_keys"], 4)
        self.assertAlmostEqual(rows[0]["ops"], 159413.35)
        self.assertEqual(rows[1]["op"], "get")
        self.assertAlmostEqual(rows[1]["mean"], 180.0)

    def test_print_summary_table_renders_markdown_style_table(self):
        rows = [
            {
                "transport": "RDMA",
                "put_v2_transfer_mode": "read",
                "op": "put",
                "rounds": 50,
                "iterations": 20,
                "batch_keys": 4,
                "mean": 250.0,
                "p50": 248.0,
                "p95": 264.0,
                "p99": 270.0,
                "ops": 159413.35,
                "key_ops": 637653.4,
            },
            {
                "transport": "RDMA",
                "put_v2_transfer_mode": "read",
                "op": "get",
                "rounds": 50,
                "iterations": 20,
                "batch_keys": 4,
                "mean": 180.0,
                "p50": 175.0,
                "p95": 195.0,
                "p99": 205.0,
                "ops": 210000.0,
                "key_ops": 840000.0,
            },
        ]
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(rows)
        text = out.getvalue()
        self.assertIn("=== Benchmark Summary (measure phase) ===", text)
        self.assertIn("| transport", text)
        self.assertIn("| put_v2", text)
        self.assertIn("| read", text)
        self.assertIn("| op ", text)
        self.assertIn("| RDMA", text)
        self.assertIn("| put", text)
        self.assertIn("| get", text)

    def test_help_contains_rdma_only_switch(self):
        script = Path(__file__).resolve().parents[3] / "tools" / "benchmarks" / "run_rdma_transport_benchmarks.py"
        completed = subprocess.run(
            ["python3", str(script), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--rdma-only", completed.stdout)
        self.assertIn("--rdma-put-protocol-version", completed.stdout)
        self.assertIn("--rdma-put-v2-transfer-mode", completed.stdout)
        self.assertIn("--rdma-control-plane-host", completed.stdout)
        self.assertNotIn("--use-local-memcached", completed.stdout)


if __name__ == "__main__":
    unittest.main()
