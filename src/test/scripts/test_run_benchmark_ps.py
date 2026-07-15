import json
import argparse
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.benchmarks.run_benchmark_ps import (  # noqa: E402
    apply_interactive_prompts,
    build_benchmark_cmd,
    build_rdma_runner,
    build_remote_exec_cmd,
    build_rpc_server_cmd,
    build_runtime_config,
    build_topology_plan,
    collect_ps_result_rows,
    collect_summary_rows,
    normalize_transport_list,
    parse_args,
    parse_csv_list,
    parse_client_plan,
    parse_server_plan,
    recommended_dram_capacity_bytes,
    recommended_ssd_capacity_bytes,
    replace_config_path_arg,
    resolve_base_port,
    resolve_rdma_get_response_mode,
    write_summary_csv,
)


class TestRunBenchmarkPS(unittest.TestCase):
    def test_parse_csv_list_strips_empty_items(self):
        self.assertEqual(parse_csv_list("grpc, brpc,,rdma"), ["grpc", "brpc", "rdma"])

    def test_normalize_transport_list_deduplicates_and_uppercases(self):
        self.assertEqual(
            normalize_transport_list("grpc, BRPC, grpc, rdma"),
            ["GRPC", "BRPC", "RDMA"],
        )

    def test_build_topology_plan_expands_shards_and_client_processes(self):
        topology = build_topology_plan(
            "GRPC",
            server_shard_ips=["server-a", "server-b"],
            client_ips=["client-a", "client-b"],
            client_processes_per_ip=2,
            base_port=15000,
        )
        self.assertEqual(len(topology.server_plan), 2)
        self.assertEqual(topology.server_plan[1].host, "server-b")
        self.assertEqual(topology.server_plan[1].ssh_host, "server-b")
        self.assertEqual(topology.server_plan[1].shard, 1)
        self.assertEqual(topology.server_plan[1].port, 15001)
        self.assertEqual(len(topology.client_plan), 4)
        self.assertEqual([client.host for client in topology.client_plan], [
            "client-a",
            "client-a",
            "client-b",
            "client-b",
        ])

    def test_build_topology_plan_rejects_invalid_client_processes_per_ip(self):
        with self.assertRaisesRegex(ValueError, "client_processes_per_ip"):
            build_topology_plan(
                "GRPC",
                server_shard_ips=["server-a"],
                client_ips=["client-a"],
                client_processes_per_ip=0,
                base_port=15000,
            )

    def test_parse_server_plan_supports_explicit_shard_binding(self):
        servers = parse_server_plan(
            "1:server-b:26000:8,0:server-a:25000:3",
            "RDMA",
        )
        self.assertEqual([server.server_index for server in servers], [0, 1])
        self.assertEqual(servers[0].host, "server-a")
        self.assertEqual(servers[0].ssh_host, "server-a")
        self.assertEqual(servers[0].port, 25000)
        self.assertEqual(servers[0].shard, 3)
        self.assertEqual(servers[1].shard, 8)

    def test_parse_client_plan_supports_repeated_hosts(self):
        clients = parse_client_plan("1:client-b,0:client-a,2:client-a", "GRPC")
        self.assertEqual([client.client_index for client in clients], [0, 1, 2])
        self.assertEqual(clients[2].host, "client-a")
        self.assertEqual(clients[2].ssh_host, "client-a")

    def test_explicit_plans_split_ssh_target_from_endpoint_host(self):
        topology = build_topology_plan(
            "RDMA",
            server_shard_ips=["ignored"],
            client_ips=["ignored"],
            client_processes_per_ip=1,
            base_port=25000,
            server_plan="0:xieminhui@10.0.2.190:25000:0",
            client_plan="0:xieminhui@10.0.2.191",
        )
        self.assertEqual(topology.server_plan[0].host, "10.0.2.190")
        self.assertEqual(topology.server_plan[0].ssh_host, "xieminhui@10.0.2.190")
        self.assertEqual(topology.client_plan[0].host, "10.0.2.191")
        self.assertEqual(topology.client_plan[0].ssh_host, "xieminhui@10.0.2.191")

    def test_build_topology_plan_uses_explicit_server_and_client_plan(self):
        topology = build_topology_plan(
            "BRPC",
            server_shard_ips=["ignored"],
            client_ips=["ignored"],
            client_processes_per_ip=1,
            base_port=25000,
            server_plan="0:server-a:25010:4,1:server-b:25011:9",
            client_plan="0:client-a,1:client-b",
        )
        self.assertEqual(len(topology.server_plan), 2)
        self.assertEqual(topology.server_plan[0].shard, 4)
        self.assertEqual(topology.server_plan[1].port, 25011)
        self.assertEqual(len(topology.client_plan), 2)
        self.assertEqual(topology.client_plan[1].host, "client-b")

    def test_build_topology_plan_allows_server_plan_with_default_clients(self):
        topology = build_topology_plan(
            "GRPC",
            server_shard_ips=["unused"],
            client_ips=["client-a"],
            client_processes_per_ip=3,
            base_port=15000,
            server_plan="server-a:25000:0,server-a:25001:1",
        )
        self.assertEqual(len(topology.server_plan), 2)
        self.assertEqual(len(topology.client_plan), 3)
        self.assertEqual(topology.client_plan[2].host, "client-a")

    def test_build_runtime_config_writes_explicit_shards(self):
        topology = build_topology_plan(
            "BRPC",
            server_shard_ips=["server-a", "server-b"],
            client_ips=["client-a"],
            client_processes_per_ip=1,
            base_port=25000,
        )
        config = build_runtime_config(
            transport="BRPC",
            topology=topology,
            capacity=4096,
            value_size=128,
            max_keys_per_request=256,
            num_threads=8,
            index_type="DRAM_PET_HASH",
            value_store_type="DRAM_VALUE_STORE",
            dram_allocator="PERSIST_LOOP_SLAB",
            data_root="/tmp/bench/value",
            ssd_data_root="/tmp/bench/ssd",
            ssd_capacity_bytes=268435456,
            ssd_io_backend="IOURING",
            ssd_queue_depth=512,
        )
        self.assertEqual(config["cache_ps"]["ps_type"], "BRPC")
        self.assertEqual(config["distributed_client"]["servers"][1]["host"], "server-b")
        self.assertEqual(config["distributed_client"]["servers"][1]["shard"], 1)
        self.assertEqual(config["client"]["port"], 25000)

    def test_build_runtime_config_uses_rdma_ps_type(self):
        topology = build_topology_plan(
            "RDMA",
            server_shard_ips=["rdma-a"],
            client_ips=["client-a"],
            client_processes_per_ip=1,
            base_port=25000,
        )
        config = build_runtime_config(
            transport="RDMA",
            topology=topology,
            capacity=1024,
            value_size=64,
            max_keys_per_request=64,
            num_threads=2,
            index_type="DRAM_EXTENDIBLE_HASH",
            value_store_type="DRAM_VALUE_STORE",
            dram_allocator="PERSIST_LOOP_SLAB",
            data_root="/tmp/rdma/value",
            ssd_data_root="/tmp/rdma/ssd",
            ssd_capacity_bytes=268435456,
            ssd_io_backend="IOURING",
            ssd_queue_depth=512,
        )
        self.assertEqual(config["cache_ps"]["ps_type"], "RDMA")

    def test_build_runtime_config_adds_slab_metadata_capacity(self):
        topology = build_topology_plan(
            "GRPC",
            server_shard_ips=["127.0.0.1"],
            client_ips=["127.0.0.1"],
            client_processes_per_ip=1,
            base_port=15000,
        )
        config = build_runtime_config(
            transport="GRPC",
            topology=topology,
            capacity=1024,
            value_size=128,
            max_keys_per_request=64,
            num_threads=2,
            index_type="DRAM_EXTENDIBLE_HASH",
            value_store_type="DRAM_VALUE_STORE",
            dram_allocator="PERSIST_LOOP_SLAB",
            data_root="/tmp/grpc/value",
            ssd_data_root="/tmp/grpc/ssd",
            ssd_capacity_bytes=268435456,
            ssd_io_backend="IOURING",
            ssd_queue_depth=512,
        )
        self.assertEqual(
            config["cache_ps"]["base_kv_config"]["value"]["dram_allocator"]["capacity_bytes"],
            1 << 20,
        )

    def test_build_runtime_config_supports_tiered_value_store(self):
        topology = build_topology_plan(
            "GRPC",
            server_shard_ips=["127.0.0.1"],
            client_ips=["127.0.0.1"],
            client_processes_per_ip=1,
            base_port=15000,
        )
        config = build_runtime_config(
            transport="GRPC",
            topology=topology,
            capacity=1024,
            value_size=128,
            max_keys_per_request=64,
            num_threads=2,
            index_type="DRAM_EXTENDIBLE_HASH",
            value_store_type="TIERED_VALUE_STORE",
            dram_allocator="PERSIST_LOOP_SLAB",
            data_root="/tmp/grpc/value",
            ssd_data_root="/tmp/grpc/ssd",
            ssd_capacity_bytes=268435456,
            ssd_io_backend="IOURING",
            ssd_queue_depth=512,
        )
        value = config["cache_ps"]["base_kv_config"]["value"]
        self.assertEqual(value["type"], "TIERED_VALUE_STORE")
        self.assertNotIn("path", value)
        self.assertEqual(value["dram_allocator"]["path"], "/tmp/grpc/value/dram")
        self.assertEqual(value["ssd_allocator"]["path"], "/tmp/grpc/ssd/ssd.db")
        self.assertEqual(value["ssd_allocator"]["io"]["type"], "IOURING")

    def test_recommended_dram_capacity_bytes_adds_slab_headroom(self):
        self.assertEqual(
            recommended_dram_capacity_bytes(
                capacity=1_000_000,
                value_size=512,
                dram_allocator="PERSIST_LOOP_SLAB",
            ),
            596 * (1 << 20),
        )

    def test_recommended_ssd_capacity_bytes_has_minimum(self):
        self.assertEqual(
            recommended_ssd_capacity_bytes(capacity=1024, value_size=128),
            256 * 1024 * 1024,
        )

    def test_recommended_dram_capacity_bytes_keeps_non_slab_exact(self):
        self.assertEqual(
            recommended_dram_capacity_bytes(
                capacity=100,
                value_size=64,
                dram_allocator="R2_ALLOC",
            ),
            6400,
        )

    def test_resolve_base_port_uses_brpc_single_shard_default(self):
        self.assertEqual(resolve_base_port("BRPC", 25000, 1), 15000)
        self.assertEqual(resolve_base_port("BRPC", 25000, 2), 25000)

    def test_resolve_rdma_get_response_mode_auto_binds_pet_hash_to_staging(self):
        self.assertEqual(
            resolve_rdma_get_response_mode("DRAM_PET_HASH", "auto"),
            "staging_copy",
        )
        self.assertEqual(
            resolve_rdma_get_response_mode("DRAM_EXTENDIBLE_HASH", "auto"),
            "direct_sg",
        )

    def test_build_benchmark_cmd_includes_transactions_args(self):
        topology = build_topology_plan(
            "RDMA",
            server_shard_ips=["127.0.0.1"],
            client_ips=["127.0.0.1"],
            client_processes_per_ip=1,
            base_port=15000,
        )
        cmd = build_benchmark_cmd(
            benchmark_binary="/tmp/ps_transport_benchmark",
            transport="RDMA",
            topology=topology,
            config_path="/tmp/config.json",
            record_count=1000,
            runtime_seconds=5,
            client_threads_per_process=16,
            load_threads=8,
            batch_keys=64,
            value_size=256,
            distribution="uniform",
            zipfian_alpha=0.9,
            read_ratio=100,
            mode="fetch",
            report_mode="summary",
            prefetch_depth=4,
            rdma_adapter_skip_prefetch_result_copy=True,
            rdma_get_response_mode="staging_copy",
            verify_deterministic_values=True,
        )
        self.assertIn("--workload=transactions", cmd)
        self.assertIn("--record_count=1000", cmd)
        self.assertIn("--config_path=/tmp/config.json", cmd)
        self.assertIn("--value_size=256", cmd)
        self.assertIn("--prefetch_depth=4", cmd)
        self.assertIn("--rdma_adapter_skip_prefetch_result_copy=true", cmd)
        self.assertIn("--rdma_get_response_mode=staging_copy", cmd)
        self.assertIn("--verify_deterministic_values=true", cmd)

    def test_build_rpc_server_cmd_passes_brpc_config_and_port(self):
        cmd = build_rpc_server_cmd(
            "/tmp/ps_server",
            "BRPC",
            "/tmp/config.json",
            shard=3,
            port=25000,
        )
        self.assertIn("--config_path=/tmp/config.json", cmd)
        self.assertIn("--brpc_config_path=/tmp/config.json", cmd)
        self.assertIn("--local_shard_id=3", cmd)
        self.assertIn("--brpc_server_port=25000", cmd)

    def test_build_rpc_server_cmd_leaves_grpc_single_shard_default_port(self):
        cmd = build_rpc_server_cmd(
            "/tmp/ps_server",
            "GRPC",
            "/tmp/config.json",
            shard=0,
            port=25000,
        )
        self.assertIn("--config_path=/tmp/config.json", cmd)
        self.assertIn("--grpc_local_shard_id=0", cmd)
        self.assertNotIn("--brpc_server_port=25000", cmd)

    def test_parse_args_rejects_rdma_prefetch_depth_above_slot_capacity(self):
        argv = [
            "run_benchmark_ps.py",
            "--transports",
            "rdma",
            "--rdma-rc-qps-per-client-per-shard",
            "4",
            "--rdma-rc-slots-per-qp",
            "1",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_sets_rdma_fetch_qp_default(self):
        argv = [
            "run_benchmark_ps.py",
            "--transports",
            "rdma",
            "--mode",
            "fetch",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.rdma_rc_qps_per_client_per_shard, 16)

    def test_parse_args_accepts_build_dirs(self):
        argv = [
            "run_benchmark_ps.py",
            "--build-dir",
            "build_release",
            "--remote-build-dir",
            "build_release",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.build_dir, "build_release")
        self.assertEqual(args.remote_build_dir, "build_release")

    def test_parse_args_accepts_tiered_dram_capacity_override(self):
        argv = [
            "run_benchmark_ps.py",
            "--value-store-type",
            "TIERED_VALUE_STORE",
            "--tiered-dram-capacity-bytes",
            "1048576",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.value_store_type, "TIERED_VALUE_STORE")
        self.assertEqual(args.tiered_dram_capacity_bytes, 1048576)

    def test_parse_args_sets_local_rdma_bind_core_defaults(self):
        argv = [
            "run_benchmark_ps.py",
            "--transports",
            "rdma",
            "--server-rdma-threads",
            "16",
            "--client-threads-per-process",
            "1",
            "--client-load-threads-per-process",
            "1",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.rdma_server_bind_core_offset, 0)
        self.assertEqual(args.rdma_rc_server_numa_id, 0)
        self.assertEqual(args.rdma_rc_client_numa_id, 1)
        self.assertEqual(args.rdma_client_bind_core_offset, 0)
        self.assertEqual(args.rdma_client_bind_core_stride, 2)

    def test_build_rdma_runner_forwards_profile_interval(self):
        args = argparse.Namespace(
            server_rdma_threads=1,
            cluster_timeout=45,
            startup_delay=2.0,
            output_dir=Path("/tmp/bench"),
            show_runner_logs=False,
            rdma_wait_timeout_ms=20000,
            rdma_rc_qps_per_client_per_shard=16,
            rdma_rc_slots_per_qp=1,
            rdma_rc_profile_interval_ms=250,
            rdma_rc_server_coroutines_per_thread=4,
            rdma_rc_server_get_workers=2,
            rdma_rc_inline_bytes=64,
            rdma_rc_client_numa_id=1,
            rdma_rc_server_numa_id=0,
            rdma_rc_fake_get_mode="status_only",
            rdma_rc_skip_client_copy=True,
            rdma_server_bind_core_offset=0,
            rdma_client_bind_core_offset=4,
            rdma_client_bind_core_stride=1,
        )
        runner = build_rdma_runner(
            args,
            config_path="/tmp/config.json",
            server_binary="/tmp/petps_server",
            server_shards=2,
            client_processes=2,
            value_size=512,
            max_keys_per_request=1024,
            rdma_namespace="profile-test",
            rdma_control_plane_host="127.0.0.1",
            rdma_control_plane_port=25000,
        )
        self.assertEqual(runner.rdma_rc_profile_interval_ms, 250)
        self.assertIn(
            "--rdma_rc_profile_interval_ms=250",
            runner.build_server_cmd(0),
        )
        self.assertIn(
            "--rdma_rc_server_get_workers=2",
            runner.build_server_cmd(0),
        )
        self.assertIn(
            "--rdma_rc_profile_interval_ms=250",
            runner.build_client_cmd(["/tmp/client"], client_index=0),
        )
        self.assertIn("--numa_id=0", runner.build_server_cmd(0))
        self.assertIn(
            "--rdma_rc_client_numa_id=1",
            runner.build_client_cmd(["/tmp/client"], client_index=0),
        )
        self.assertIn("--rdma_rc_fake_get_mode=status_only", runner.build_server_cmd(0))
        self.assertIn(
            "--rdma_rc_skip_client_copy=true",
            runner.build_client_cmd(["/tmp/client"], client_index=0),
        )

    def test_collect_ps_result_rows_parses_load_and_run(self):
        text = (
            "PS_BENCHMARK_RESULT phase=load transport=GRPC mode=fetch "
            "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=64 "
            "records=1000 runtime_s=1.0 batches=10 key_ops=640 "
            "throughput_batches_sec=10 throughput_keys_sec=6400\n"
            "PS_BENCHMARK_RESULT phase=run transport=GRPC mode=fetch "
            "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=64 "
            "records=1000 runtime_s=1.0 batches=20 key_ops=1280 "
            "throughput_batches_sec=20 throughput_keys_sec=12800\n"
        )
        rows = collect_ps_result_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["phase"], "load")
        self.assertEqual(rows[1]["throughput_keys_sec"], 12800.0)

    def test_collect_summary_rows_keeps_measure_phase(self):
        text = (
            "transport=GRPC op=get phase=warmup summary rounds=1 iterations=10 "
            "batch_keys=64 elapsed_us_mean=200 elapsed_us_p50=180 "
            "elapsed_us_p95=220 elapsed_us_p99=230 ops_per_sec=100 key_ops_per_sec=6400\n"
            "transport=GRPC op=get phase=measure summary rounds=2 iterations=10 "
            "batch_keys=64 elapsed_us_mean=100 elapsed_us_p50=90 "
            "elapsed_us_p95=110 elapsed_us_p99=120 ops_per_sec=200 key_ops_per_sec=12800\n"
        )
        rows = collect_summary_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "measure")
        self.assertEqual(rows[0]["mean"], 100.0)

    def test_build_remote_exec_cmd_wraps_container_when_present(self):
        cmd = build_remote_exec_cmd(
            host="worker-a",
            remote_repo="/app/RecStore",
            remote_container="recstore-dev",
            shell_command="echo ready",
        )
        self.assertEqual(cmd[:4], ["ssh", "worker-a", "bash", "-lc"])
        self.assertIn("docker exec recstore-dev bash -lc", cmd[-1])
        self.assertIn("cd /app/RecStore && echo ready", cmd[-1])

    def test_build_remote_exec_cmd_quotes_remote_shell_command(self):
        cmd = build_remote_exec_cmd(
            host="worker-a",
            remote_repo="/app/RecStore",
            remote_container=None,
            shell_command="mkdir -p /tmp/recstore sync",
        )
        self.assertEqual(cmd[:4], ["ssh", "worker-a", "bash", "-lc"])
        self.assertEqual(cmd[-1], "'cd /app/RecStore && mkdir -p /tmp/recstore sync'")

    def test_replace_config_path_arg_rewrites_existing_flag(self):
        cmd = replace_config_path_arg(
            [
                "/tmp/ps_transport_benchmark",
                "--transport=grpc",
                "--config_path=/tmp/old.json",
            ],
            "/tmp/new.json",
        )
        self.assertIn("--config_path=/tmp/new.json", cmd)
        self.assertNotIn("--config_path=/tmp/old.json", cmd)

    def test_write_summary_csv_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            write_summary_csv(
                [
                    {
                        "transport": "GRPC",
                        "status": "success",
                        "phase": "run",
                        "client_index": 0,
                        "repeat_index": 0,
                        "server_shards": 1,
                        "client_processes": 1,
                        "server_shard_ips": "127.0.0.1",
                        "client_ips": "127.0.0.1",
                        "record_count": 1000,
                        "value_size": 128,
                        "batch_keys": 64,
                        "client_threads_per_process": 16,
                        "runtime_seconds": 1,
                        "distribution": "uniform",
                        "mode": "fetch",
                        "ops_per_sec": 10.0,
                        "key_ops_per_sec": 640.0,
                        "mean_us": "",
                        "p50_us": "",
                        "p95_us": "",
                        "p99_us": "",
                        "log_path": "/tmp/log",
                        "message": "",
                    }
                ],
                csv_path,
            )
            text = csv_path.read_text(encoding="utf-8")
        self.assertIn("transport,status,phase,client_index", text)
        self.assertIn("GRPC,success,run,0", text)

    def test_apply_interactive_prompts_updates_prompted_values(self):
        args = argparse.Namespace(
            transports="rdma,grpc,brpc",
            client_ips="127.0.0.1",
            server_shard_ips="127.0.0.1",
            client_processes_per_ip=1,
            server_plan="",
            client_plan="",
            record_count=1000000,
            value_size=512,
            batch_keys=1024,
            client_threads_per_process=16,
            runtime_seconds=5,
            repeat=1,
            execution_backend="local",
            output_dir=Path("/tmp/default"),
        )
        answers = iter(
            [
                "grpc",
                "client-a",
                "server-a,server-b",
                "1",
                "0:server-a:15000:0,1:server-b:15001:3",
                "",
                "4096",
                "128",
                "64",
                "8",
                "2",
                "3",
                "ssh",
                "/tmp/bench-out",
            ]
        )
        with mock.patch("builtins.input", side_effect=lambda _prompt: next(answers)):
            apply_interactive_prompts(args)
        self.assertEqual(args.transports, "grpc")
        self.assertEqual(args.server_shard_ips, "server-a,server-b")
        self.assertEqual(args.client_processes_per_ip, 1)
        self.assertEqual(args.server_plan, "0:server-a:15000:0,1:server-b:15001:3")
        self.assertEqual(args.record_count, 4096)
        self.assertEqual(args.execution_backend, "ssh")
        self.assertEqual(args.output_dir, Path("/tmp/bench-out"))

    def test_help_mentions_remote_and_topology_args(self):
        script = Path(__file__).resolve().parents[3] / "tools" / "benchmarks" / "run_benchmark_ps.py"
        completed = subprocess.run(
            ["python3", str(script), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--execution-backend", completed.stdout)
        self.assertIn("--remote-sync", completed.stdout)
        self.assertIn("--client-ips", completed.stdout)
        self.assertIn("--server-shard-ips", completed.stdout)
        self.assertIn("--client-processes-per-ip", completed.stdout)
        self.assertIn("--client-threads-per-process", completed.stdout)
        self.assertIn("--client-load-threads-per-process", completed.stdout)
        self.assertIn("--server-worker-threads", completed.stdout)
        self.assertIn("--server-rdma-threads", completed.stdout)
        self.assertIn("--server-plan", completed.stdout)
        self.assertIn("--client-plan", completed.stdout)
        self.assertIn("--interactive", completed.stdout)
        self.assertIn("--prefetch-depth", completed.stdout)
        self.assertIn("--rdma-rc-profile-interval-ms", completed.stdout)
        self.assertIn("--rdma-rc-fake-get-mode", completed.stdout)
        self.assertIn("--rdma-rc-skip-client-copy", completed.stdout)
        self.assertNotIn("--client-hosts", completed.stdout)
        self.assertNotIn("--server-hosts", completed.stdout)
        self.assertNotIn("--server-count", completed.stdout)
        self.assertNotIn("--client-count", completed.stdout)
        self.assertNotIn("--threads", completed.stdout)
        self.assertNotIn("--load-threads", completed.stdout)
        self.assertNotIn("--server-num-threads", completed.stdout)
        self.assertNotIn("--rdma-thread-num", completed.stdout)


if __name__ == "__main__":
    unittest.main()
