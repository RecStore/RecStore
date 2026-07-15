import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from tools.benchmarks.run_ps_dram_transport_benchmark import (  # noqa: E402
    BenchmarkCaseError,
    build_benchmark_cmd,
    build_runtime_config,
    build_failure_row,
    collect_case_rows,
    collect_ps_result_rows,
    collect_summary_rows,
    is_port_open,
    parse_csv_list,
    prepare_runtime_paths,
    print_summary_table,
    resolve_case_base_port,
    resolve_failure_stage,
    resolve_case_load_threads,
    resolve_local_shm_ready_queue_count,
    write_csv,
)


class TestRunPSDramTransportBenchmark(unittest.TestCase):
    def test_build_runtime_config_uses_dram_index_and_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="BRPC",
                backend_alias="dram_pet_dram",
                runtime_dir=Path(tmpdir),
                num_shards=2,
                base_port=25000,
                capacity=4096,
                value_size=128,
                max_keys_per_request=256,
                num_threads=8,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        cache_ps = config["cache_ps"]
        self.assertEqual(cache_ps["ps_type"], "BRPC")
        self.assertEqual(cache_ps["num_shards"], 2)
        self.assertEqual(
            cache_ps["base_kv_config"]["index"]["type"], "DRAM_PET_HASH"
        )
        self.assertEqual(
            cache_ps["base_kv_config"]["value"]["type"], "DRAM_VALUE_STORE"
        )
        self.assertEqual(config["client"]["port"], 25000)
        self.assertEqual(config["distributed_client"]["servers"][1]["shard"], 1)
        self.assertNotIn("local_shm", config)

    def test_build_runtime_config_keeps_benchmark_base_port_for_single_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="BRPC",
                backend_alias="dram_pet_dram",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=25000,
                capacity=4096,
                value_size=128,
                max_keys_per_request=256,
                num_threads=8,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        self.assertEqual(config["client"]["port"], 25000)
        self.assertEqual(config["cache_ps"]["servers"][0]["port"], 25000)

    def test_local_shm_config_contains_transport_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="LOCAL_SHM",
                backend_alias="dram_eh_dram",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=0,
                capacity=1024,
                value_size=64,
                max_keys_per_request=128,
                num_threads=1,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="bench_region",
                local_shm_slot_count=32,
                local_shm_ready_queue_count=2,
                local_shm_ready_queue_burst_limit=4,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=2000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        self.assertEqual(config["cache_ps"]["ps_type"], "LOCAL_SHM")
        self.assertEqual(config["local_shm"]["region_name"], "bench_region")
        self.assertEqual(config["local_shm"]["ready_queue_count"], 2)
        self.assertFalse(config["local_shm"]["thread_ready_queue_sharding"])

    def test_build_runtime_config_uses_hps_hash_map_external_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="GRPC",
                backend_alias="hps_hash_map",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=15000,
                capacity=2048,
                value_size=512,
                max_keys_per_request=256,
                num_threads=8,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        base_kv = config["cache_ps"]["base_kv_config"]
        self.assertEqual(base_kv["external_engine_type"], "KVEngineHPSHashMap")
        self.assertEqual(base_kv["value_size"], 512)
        self.assertEqual(base_kv["capacity"], 2048)
        self.assertNotIn("index", base_kv)
        self.assertNotIn("value", base_kv)

    def test_build_runtime_config_uses_hps_rocksdb_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="BRPC",
                backend_alias="hps_rocksdb",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=25000,
                capacity=2048,
                value_size=512,
                max_keys_per_request=256,
                num_threads=4,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        base_kv = config["cache_ps"]["base_kv_config"]
        self.assertEqual(base_kv["external_engine_type"], "KVEngineHPSRocksDB")
        self.assertIn("hps_rocksdb", base_kv["path"])
        self.assertEqual(base_kv["rocksdb_path"], base_kv["path"])

    def test_build_runtime_config_uses_minimum_ssd_allocator_capacity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="GRPC",
                backend_alias="dram_eh_ssd",
                runtime_dir=Path(tmpdir),
                num_shards=1,
                base_port=15000,
                capacity=2048,
                value_size=128,
                max_keys_per_request=128,
                num_threads=4,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=4096,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
        ssd = config["cache_ps"]["base_kv_config"]["value"]["ssd_allocator"]
        self.assertGreaterEqual(ssd["capacity_bytes"], 256 * 1024 * 1024)

    def test_build_benchmark_cmd_includes_transport_and_value_size(self):
        cmd = build_benchmark_cmd(
            benchmark_binary=Path("/tmp/ps_transport_benchmark"),
            transport="GRPC",
            host="127.0.0.1",
            port=15100,
            num_shards=2,
            config_path=Path("/tmp/config.json"),
            mode="fetch",
            record_count=1000,
            runtime_seconds=5,
            threads=16,
            load_threads=0,
            batch_size=64,
            value_size=256,
            distribution="uniform",
            zipfian_alpha=0.9,
            read_ratio=100,
            report_mode="summary",
            seed=1234,
        )
        self.assertEqual(cmd[0], "/tmp/ps_transport_benchmark")
        self.assertIn("--transport=grpc", cmd)
        self.assertIn("--port=15100", cmd)
        self.assertIn("--config_path=/tmp/config.json", cmd)
        self.assertIn("--workload=transactions", cmd)
        self.assertIn("--value_size=256", cmd)
        self.assertIn("--seed=1234", cmd)

    def test_build_benchmark_cmd_can_use_preload_batch_size(self):
        cmd = build_benchmark_cmd(
            benchmark_binary=Path("/tmp/ps_transport_benchmark"),
            transport="GRPC",
            host="127.0.0.1",
            port=15000,
            num_shards=1,
            config_path=Path("/tmp/config.json"),
            mode="mixed",
            record_count=1000,
            runtime_seconds=5,
            threads=8,
            load_threads=1,
            batch_size=64,
            value_size=512,
            distribution="uniform",
            zipfian_alpha=0.9,
            read_ratio=95,
            report_mode="summary",
            seed=1234,
            phase="load",
        )
        self.assertIn("--batch_keys=64", cmd)
        self.assertIn("--load_only=true", cmd)

    def test_collect_summary_rows_keeps_measure_rows(self):
        sample = (
            "transport=GRPC op=put phase=warmup summary rounds=1 iterations=10 "
            "batch_keys=64 elapsed_us_mean=200 elapsed_us_p50=200 "
            "elapsed_us_p95=200 elapsed_us_p99=200 ops_per_sec=100 key_ops_per_sec=6400\n"
            "transport=GRPC op=get phase=measure summary rounds=3 iterations=10 "
            "batch_keys=64 elapsed_us_mean=100 elapsed_us_p50=90 "
            "elapsed_us_p95=120 elapsed_us_p99=130 ops_per_sec=200 key_ops_per_sec=12800\n"
        )
        rows = collect_summary_rows(sample)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "GRPC")
        self.assertEqual(rows[0]["op"], "get")
        self.assertEqual(rows[0]["batch_keys"], 64)

    def test_print_summary_table_renders_index_type(self):
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(
                [
                    {
                        "backend_alias": "dram_eh_dram",
                        "backend_layer": "PS/network",
                        "index_type": "DRAM_EXTENDIBLE_HASH",
                        "transport": "BRPC",
                        "mode": "fetch",
                        "phase": "run",
                        "threads": 16,
                        "batch_size": 1024,
                        "records": 1000000,
                        "throughput_keys_sec": 12800000.0,
                    }
                ]
            )
        text = out.getvalue()
        self.assertIn("PS Backend Transport Benchmark Summary", text)
        self.assertIn("dram_eh_dram", text)
        self.assertIn("DRAM_EXTENDIBLE_HASH", text)
        self.assertIn("BRPC", text)

    def test_print_summary_table_handles_failure_rows_without_throughput(self):
        out = StringIO()
        with redirect_stdout(out):
            print_summary_table(
                [
                    {
                        "backend_alias": "dram_eh_dram",
                        "backend_layer": "PS/network",
                        "index_type": "DRAM_EXTENDIBLE_HASH",
                        "transport": "LOCAL_SHM",
                        "mode": "fetch",
                        "phase": "run",
                        "threads": 8,
                        "client_processes": 1,
                        "process_id": "all",
                        "aggregate": "true",
                        "batch_size": 256,
                        "records": 20000,
                        "throughput_keys_sec": "",
                        "status": "failed",
                    }
                ]
            )
        text = out.getvalue()
        self.assertIn("LOCAL_SHM", text)
        self.assertIn("failed", text)

    def test_write_csv_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "out.csv"
            write_csv(
                [
                    {
                        "backend_alias": "dram_map_dram",
                        "backend_layer": "PS/network",
                        "index_type": "DRAM_UNORDERED_MAP",
                        "value_store_type": "DRAM_VALUE_STORE",
                        "value_size": 512,
                        "capacity": 1024,
                        "transport": "LOCAL_SHM",
                        "phase": "measure",
                        "mode": "fetch",
                        "read_ratio": 100,
                        "threads": 16,
                        "batch_size": 1024,
                        "records": 1024,
                        "distribution": "uniform",
                        "zipfian_alpha": 0.9,
                        "runtime_s": 5.0,
                        "batches": 1,
                        "key_ops": 1024,
                        "throughput_batches_sec": 1.0,
                        "throughput_keys_sec": 1024.0,
                    }
                ],
                csv_path,
            )
            rows = csv_path.read_text(encoding="utf-8")
        self.assertIn("backend_alias,backend_layer,index_type,value_store_type", rows)
        self.assertIn("dram_map_dram,PS/network,DRAM_UNORDERED_MAP,DRAM_VALUE_STORE", rows)

    def test_write_csv_preserves_failure_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "out.csv"
            write_csv(
                [
                    build_failure_row(
                        backend_alias="hps_rocksdb",
                        transport="GRPC",
                        mode="mixed",
                        read_ratio=95,
                        threads=8,
                        client_processes=4,
                        batch_size=1024,
                        value_size=512,
                        capacity=100000,
                        distribution="uniform",
                        zipfian_alpha=0.9,
                        failure_stage="load",
                        error_tail="Deadline Exceeded",
                    )
                ],
                csv_path,
            )
            rows = csv_path.read_text(encoding="utf-8")
        self.assertIn("status,failure_stage,error_tail", rows)
        self.assertIn("hps_rocksdb,PS/network,,HPS_ROCKSDB", rows)
        self.assertIn("failed,load,Deadline Exceeded", rows)

    def test_resolve_failure_stage_uses_benchmark_case_error_stage(self):
        self.assertEqual(
            resolve_failure_stage(BenchmarkCaseError("load", "preload failed")),
            "load",
        )
        self.assertEqual(resolve_failure_stage(RuntimeError("boom")), "run_one_case")

    def test_config_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_runtime_config(
                transport="GRPC",
                backend_alias="dram_map_dram",
                runtime_dir=Path(tmpdir),
                num_shards=2,
                base_port=15100,
                capacity=1024,
                value_size=64,
                max_keys_per_request=128,
                num_threads=2,
                dram_allocator="PERSIST_LOOP_SLAB",
                local_shm_region="unused",
                local_shm_slot_count=64,
                local_shm_ready_queue_count=1,
                local_shm_ready_queue_burst_limit=8,
                local_shm_slot_buffer_bytes=8192,
                local_shm_client_timeout_ms=1000,
                local_shm_thread_ready_queue_sharding=False,
                dram_capacity_multiplier=2.0,
            )
            loaded = json.loads(json.dumps(config))
        self.assertEqual(loaded["cache_ps"]["base_kv_config"]["capacity"], 1024)

    def test_prepare_runtime_paths_creates_ssd_parents_without_touching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir)
            ssd_value = runtime_dir / "ssd" / "value.db"
            tiered_ssd = runtime_dir / "tiered" / "ssd.db"
            config = {
                "cache_ps": {
                    "base_kv_config": {
                        "value": {
                            "type": "TIERED_VALUE_STORE",
                            "path": str(ssd_value),
                            "ssd_allocator": {"path": str(tiered_ssd)},
                        }
                    }
                }
            }

            prepare_runtime_paths(config)

            self.assertTrue(ssd_value.parent.is_dir())
            self.assertTrue(tiered_ssd.parent.is_dir())
            self.assertFalse(ssd_value.exists())
            self.assertFalse(tiered_ssd.exists())

    def test_parse_csv_list_normalizes_values(self):
        self.assertEqual(parse_csv_list("grpc, BRPC"), ["GRPC", "BRPC"])

    def test_resolve_case_base_port_uses_grpc_single_shard_default(self):
        class Args:
            grpc_base_port = 18100
            brpc_base_port = 28100

        self.assertEqual(resolve_case_base_port("GRPC", 1, Args()), 15000)
        self.assertEqual(resolve_case_base_port("GRPC", 2, Args()), 18100)
        self.assertEqual(resolve_case_base_port("BRPC", 1, Args()), 28100)

    def test_resolve_case_load_threads_serializes_hps_rocksdb_default(self):
        class Args:
            load_threads = 0
            hps_rocksdb_load_threads = 1

        self.assertEqual(resolve_case_load_threads("hps_rocksdb", Args()), 1)
        self.assertEqual(resolve_case_load_threads("hps_hash_map", Args()), 0)

    def test_resolve_local_shm_ready_queue_count_uses_manual_value(self):
        self.assertEqual(resolve_local_shm_ready_queue_count(8, 16), 8)

    def test_resolve_local_shm_ready_queue_count_auto_from_threads(self):
        self.assertEqual(resolve_local_shm_ready_queue_count(0, 16), 16)
        self.assertEqual(resolve_local_shm_ready_queue_count(0, 1), 1)

    def test_collect_ps_result_rows_parses_transactions(self):
        text = (
            "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
            "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
            "records=1000000 runtime_s=5.0 batches=10 key_ops=10240 "
            "throughput_batches_sec=2 throughput_keys_sec=2048\n"
        )
        rows = collect_ps_result_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "BRPC")
        self.assertEqual(rows[0]["threads"], 16)
        self.assertEqual(rows[0]["throughput_keys_sec"], 2048.0)

    def test_collect_ps_result_rows_ignores_local_shm_profile_lines(self):
        text = (
            "PS_LOCAL_SHM_PROFILE phase=run samples=100 acquire_slot_us_mean=1 "
            "enqueue_us_mean=2 wait_us_mean=3 release_us_mean=4 "
            "request_total_us_mean=5 server_queue_wait_us_mean=6 "
            "server_backend_us_mean=7 opcode=GET\n"
            "PS_LOCAL_SHM_PROFILE_OPCODE phase=run opcode=PUT samples=50 "
            "acquire_slot_us_mean=1 enqueue_us_mean=2 wait_us_mean=3 "
            "release_us_mean=4 request_total_us_mean=5 "
            "server_queue_wait_us_mean=6 server_backend_us_mean=7\n"
            "PS_BENCHMARK_RESULT phase=run transport=LOCAL_SHM mode=mixed "
            "distribution=uniform zipfian_alpha=0.9 threads=8 batch_size=256 "
            "records=20000 runtime_s=5.0 batches=10 key_ops=2560 "
            "throughput_batches_sec=2 throughput_keys_sec=512\n"
        )
        rows = collect_ps_result_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transport"], "LOCAL_SHM")
        self.assertEqual(rows[0]["throughput_keys_sec"], 512.0)

    def test_collect_case_rows_adds_process_metadata_and_aggregate(self):
        outputs = [
            (
                0,
                "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
                "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
                "records=1000000 runtime_s=5.0 batches=10 key_ops=10240 "
                "throughput_batches_sec=2 throughput_keys_sec=2048\n",
            ),
            (
                1,
                "PS_BENCHMARK_RESULT phase=run transport=BRPC mode=fetch "
                "distribution=uniform zipfian_alpha=0.9 threads=16 batch_size=1024 "
                "records=1000000 runtime_s=5.0 batches=20 key_ops=20480 "
                "throughput_batches_sec=4 throughput_keys_sec=4096\n",
            ),
        ]
        rows = collect_case_rows(
            outputs,
            backend_alias="dram_pet_dram",
            value_size=512,
            capacity=1000000,
            read_ratio=100,
            client_processes=2,
        )
        per_process = [row for row in rows if row["aggregate"] == "false"]
        aggregate = [row for row in rows if row["aggregate"] == "true"]
        self.assertEqual(len(per_process), 2)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(per_process[1]["process_id"], 1)
        self.assertEqual(aggregate[0]["process_id"], "all")
        self.assertEqual(aggregate[0]["threads"], 32)
        self.assertEqual(aggregate[0]["client_processes"], 2)
        self.assertEqual(aggregate[0]["throughput_keys_sec"], 6144.0)

    def test_collect_case_rows_adds_single_process_aggregate(self):
        outputs = [
            (
                0,
                "PS_BENCHMARK_RESULT phase=run transport=GRPC mode=fetch "
                "distribution=uniform zipfian_alpha=0.9 threads=4 batch_size=128 "
                "records=2048 runtime_s=1.0 batches=10 key_ops=1280 "
                "throughput_batches_sec=10 throughput_keys_sec=1280\n",
            )
        ]
        rows = collect_case_rows(
            outputs,
            backend_alias="hps_hash_map",
            value_size=128,
            capacity=2048,
            read_ratio=100,
            client_processes=1,
        )
        aggregate = [row for row in rows if row["aggregate"] == "true"]
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["backend_alias"], "hps_hash_map")
        self.assertEqual(aggregate[0]["process_id"], "all")
        self.assertEqual(aggregate[0]["throughput_keys_sec"], 1280.0)

    def test_is_port_open_returns_false_for_unused_port(self):
        self.assertFalse(is_port_open("127.0.0.1", 1, timeout_s=0.01))


if __name__ == "__main__":
    unittest.main()
