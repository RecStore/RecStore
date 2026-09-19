#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks._legacy_scripts import ensure_legacy_test_script_path
from tools.benchmarks.render import fmt_num, per_request_us, print_markdown_table
from tools.benchmarks.summary import collect_ps_transport_summary_rows

ensure_legacy_test_script_path()

from petps_cluster_runner import PetPSClusterRunner, REPO_ROOT
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_rdma_integration_config,
)


CONTROL_PLANE_NOISE_PATTERNS = (
    "[petps-control-plane]",
    "component=rdma_control_plane",
)

def is_runner_noise_line(line):
    return any(pattern in line for pattern in CONTROL_PLANE_NOISE_PATTERNS)


def is_summary_line(line):
    return " phase=measure summary " in line or " phase=warmup summary " in line


def print_filtered_output(text, show_runner_logs, quiet):
    for line in text.splitlines():
        if quiet and not is_summary_line(line):
            continue
        if not show_runner_logs and is_runner_noise_line(line):
            continue
        print(line)


def collect_summary_rows(text):
    rows = collect_ps_transport_summary_rows(text)
    for row in rows:
        row.pop("phase", None)
        row["client_index"] = None
    return rows


def print_summary_table(rows):
    if not rows:
        print("[summary] no parsed measure summary rows found")
        return

    header = [
        "client",
        "transport",
        "op",
        "rounds",
        "iterations",
        "batch_keys",
        "mean_req_us",
        "p50_req_us",
        "p95_req_us",
        "p99_req_us",
        "key_ops/s",
    ]
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            [
                str(row.get("client_index", "")),
                row["transport"],
                row["op"],
                str(row["rounds"]),
                str(row["iterations"]),
                str(row["batch_keys"]),
                fmt_num(per_request_us(row, "mean")),
                fmt_num(per_request_us(row, "p50")),
                fmt_num(per_request_us(row, "p95")),
                fmt_num(per_request_us(row, "p99")),
                fmt_num(row["key_ops"]),
            ]
        )
    print_markdown_table("RDMA RC Benchmark Summary (measure phase)", header, rendered_rows)


def print_aggregate_table(rows):
    if not rows:
        return

    grouped = {}
    for row in rows:
        key = (row["transport"], row["op"], row["batch_keys"])
        grouped.setdefault(key, []).append(row)

    header = [
        "transport",
        "op",
        "clients",
        "batch_keys",
        "agg_ops/s",
        "agg_key_ops/s",
        "mean_req_us_avg",
    ]
    rendered_rows = []
    for (transport, op, batch_keys), group in sorted(grouped.items()):
        agg_ops = sum(row["ops"] for row in group)
        agg_key_ops = sum(row["key_ops"] for row in group)
        avg_req_us = sum(per_request_us(row, "mean") for row in group) / len(group)
        rendered_rows.append(
            [
                transport,
                op,
                str(len(group)),
                str(batch_keys),
                fmt_num(agg_ops),
                fmt_num(agg_key_ops),
                fmt_num(avg_req_us),
            ]
        )
    print_markdown_table("RDMA RC Aggregate Summary (measure phase)", header, rendered_rows)


def build_benchmark_cmd(args):
    cmd = [
        args.benchmark_binary,
        f"--config_path={args.config_path}",
        f"--num_shards={args.server_count}",
        f"--iterations={args.iterations}",
        f"--rounds={args.rounds}",
        f"--warmup_rounds={args.warmup_rounds}",
        f"--batch_keys={args.batch_keys}",
        f"--op={args.op}",
        f"--get_ratio={args.get_ratio}",
        f"--async_depth={args.async_depth}",
        f"--report_mode={args.report_mode}",
    ]
    if args.qps_per_client_per_shard is not None:
        cmd.append(
            "--rdma_rc_qps_per_client_per_shard="
            f"{args.qps_per_client_per_shard}"
        )
    if args.slots_per_qp is not None:
        cmd.append("--rdma_rc_slots_per_qp=" f"{args.slots_per_qp}")
    if args.rdma_wait_timeout_ms is not None:
        cmd.append(f"--rdma_wait_timeout_ms={args.rdma_wait_timeout_ms}")
    if args.profile_interval_ms is not None:
        cmd.append(
            "--rdma_rc_profile_interval_ms="
            f"{args.profile_interval_ms}"
        )
    if args.inline_bytes is not None:
        cmd.append("--rdma_rc_inline_bytes=" f"{args.inline_bytes}")
    if args.client_numa_id is not None:
        cmd.append("--rdma_rc_client_numa_id=" f"{args.client_numa_id}")
    if args.server_numa_id is not None:
        cmd.append("--rdma_rc_server_numa_id=" f"{args.server_numa_id}")
    if args.verify_values:
        cmd.append("--verify_values=true")
        cmd.append(
            f"--verify_value_row_stride={args.verify_value_row_stride}"
        )
    return cmd


def parse_client_numa_ids(value, client_count):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids = [int(part) for part in parts]
    if len(ids) != client_count:
        raise ValueError(
            "--client-numa-ids must provide exactly one device id per client"
        )
    return ids


def local_numa_node_count():
    sys_node_root = "/sys/devices/system/node"
    if not os.path.exists(sys_node_root):
        return 1
    count = 0
    for name in os.listdir(sys_node_root):
        path = os.path.join(sys_node_root, name)
        if name.startswith("node") and name[4:].isdigit() and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "cpulist")):
                count += 1
    return max(1, count)


def write_runtime_config(args, source_config_path, runtime_dir):
    with open(source_config_path) as fh:
        config = json.load(fh)

    cache_ps = config.setdefault("cache_ps", {})
    source_base_kv = cache_ps.get("base_kv_config", {})
    capacity = int(source_base_kv.get("capacity", 1000000))
    base_path = (
        f"/dev/shm/recstore_rdma_rc_benchmark_{os.getpid()}_{time.time_ns()}"
    )
    cache_ps["base_kv_config"] = {
        "capacity": capacity,
        "index": {"type": "DRAM_PET_HASH"},
        "value": {
            "type": "DRAM_VALUE_STORE",
            "path": f"{base_path}/value",
            "default_value_size_hint": args.value_size,
            "dram_allocator": {
                "type": "CONCURRENT_SLAB_MEMORY_POOL",
                "capacity_bytes": capacity * args.value_size,
            },
        },
    }
    deployment = config.get("rdma_deployment")
    if not isinstance(deployment, dict):
        raise ValueError("RDMA benchmark config requires rdma_deployment")
    nodes = deployment.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("rdma_deployment.nodes must be an array")
    server_count = int(args.server_count)
    client_count = int(args.client_count)
    server_nodes = [node for node in nodes if node.get("role") == "server"]
    client_nodes = [node for node in nodes if node.get("role") == "client"]
    if len(server_nodes) != server_count:
        raise ValueError(
            "rdma_deployment server node count does not match --server-count"
        )
    if [node.get("node_id") for node in server_nodes] != list(range(server_count)):
        raise ValueError(
            "rdma_deployment server node ids must be dense 0..num_shards-1"
        )
    template = client_nodes[0] if client_nodes else {
        "role": "client",
        "device": "mlx5_0",
        "port": 1,
        "gid_index": 0,
        "mode": "ib",
    }
    deployment["num_clients"] = client_count
    deployment["nodes"] = server_nodes + [
        {**template, "node_id": server_count + client_index, "role": "client"}
        for client_index in range(client_count)
    ]

    runtime_config_path = (
        f"{runtime_dir}/recstore_config.rdma_runtime.json"
    )
    with open(runtime_config_path, "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    return runtime_config_path


def terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _stream_process_output(
    client_index,
    stream,
    sink,
    show_runner_logs,
    is_stderr,
    log_path=None,
):
    prefix = f"[rdma-rc-client:{client_index}] "
    if is_stderr:
        prefix = f"[rdma-rc-client:{client_index}:stderr] "
    log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for raw_line in iter(stream.readline, ""):
            line = raw_line.rstrip()
            sink.append(raw_line)
            if log_handle is not None:
                log_handle.write(raw_line)
                log_handle.flush()
            if show_runner_logs:
                print(prefix + line)
    finally:
        if log_handle is not None:
            log_handle.close()


def run_benchmark_clients(runner, args):
    if args.client_count == 1:
        completed = runner.run_client(
            build_benchmark_cmd(args),
            timeout=args.client_timeout,
            stream_output=args.show_runner_logs,
        )
        if not args.show_runner_logs:
            print_filtered_output(completed.stdout, args.show_runner_logs, args.quiet)
            print_filtered_output(completed.stderr, args.show_runner_logs, args.quiet)
        rows = collect_summary_rows(completed.stdout)
        for row in rows:
            row["client_index"] = 0
        return completed.returncode, rows

    processes = []
    stdout_buffers = {}
    stderr_buffers = {}
    stream_threads = []
    log_dir = tempfile.mkdtemp(prefix="rdma_rc_client_logs_")
    deadline = time.monotonic() + args.client_timeout
    for client_index in range(args.client_count):
        cmd = runner.build_client_cmd(
            build_benchmark_cmd(args), client_index=client_index
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=runner.build_client_env(client_index),
        )
        processes.append((client_index, process))
        stdout_buffers[client_index] = []
        stderr_buffers[client_index] = []
        stdout_log_path = os.path.join(log_dir, f"client_{client_index}.stdout.log")
        stderr_log_path = os.path.join(log_dir, f"client_{client_index}.stderr.log")
        stdout_thread = threading.Thread(
            target=_stream_process_output,
            args=(
                client_index,
                process.stdout,
                stdout_buffers[client_index],
                args.show_runner_logs,
                False,
                stdout_log_path,
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_process_output,
            args=(
                client_index,
                process.stderr,
                stderr_buffers[client_index],
                args.show_runner_logs,
                True,
                stderr_log_path,
            ),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        stream_threads.extend([stdout_thread, stderr_thread])

    exit_codes = {}
    timed_out = False
    for client_index, process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            break
        exit_codes[client_index] = process.returncode

    if timed_out:
        for _client_index, process in processes:
            terminate_process(process)

    for thread in stream_threads:
        thread.join(timeout=5)

    rows = []
    return_code = 124 if timed_out else 0
    for client_index, _process in processes:
        stdout = "".join(stdout_buffers.get(client_index, []))
        stderr = "".join(stderr_buffers.get(client_index, []))
        rc = exit_codes.get(client_index, 124 if timed_out else 0)
        if not args.show_runner_logs:
            print_filtered_output(stdout, args.show_runner_logs, args.quiet)
            print_filtered_output(stderr, args.show_runner_logs, args.quiet)
        parsed = collect_summary_rows(stdout)
        for row in parsed:
            row["client_index"] = client_index
        rows.extend(parsed)
        if rc != 0:
            return_code = rc
            print(f"[rdma-rc-client:{client_index}] exited with code {rc}")

    if timed_out:
        print(f"[rdma-rc-client] timed out after {args.client_timeout} seconds")
    return return_code, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
    parser.add_argument("--server-count", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=1)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--batch-keys", type=int, default=16)
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "put",
            "get",
            "async_get",
            "async_stream",
            "mixed",
        ],
        default="all",
    )
    parser.add_argument("--get-ratio", type=int, default=95)
    parser.add_argument("--async-depth", type=int, default=1)
    parser.add_argument(
        "--report-mode",
        choices=["summary", "per_round", "both"],
        default="summary",
    )
    parser.add_argument("--config-path")
    parser.add_argument("--max-kv-num-per-request", type=int)
    parser.add_argument("--client-timeout", type=int, default=120)
    parser.add_argument("--cluster-timeout", type=int, default=35)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument(
        "--qps-per-client-per-shard",
        "--rdma-rc-qps-per-client-per-shard",
        dest="qps_per_client_per_shard",
        type=int,
    )
    parser.add_argument(
        "--slots-per-qp",
        "--rdma-rc-slots-per-qp",
        dest="slots_per_qp",
        type=int,
    )
    parser.add_argument("--rdma-wait-timeout-ms", type=int)
    parser.add_argument(
        "--profile-interval-ms",
        "--rdma-rc-profile-interval-ms",
        dest="profile_interval_ms",
        type=int,
    )
    parser.add_argument(
        "--server-coroutines-per-thread",
        "--rdma-rc-server-coroutines-per-thread",
        dest="server_coroutines_per_thread",
        type=int,
    )
    parser.add_argument(
        "--server-get-workers",
        "--rdma-rc-server-get-workers",
        dest="server_get_workers",
        type=int,
    )
    parser.add_argument(
        "--inline-bytes",
        "--rdma-rc-inline-bytes",
        dest="inline_bytes",
        type=int,
    )
    parser.add_argument(
        "--client-numa-id",
        "--rdma-rc-client-numa-id",
        dest="client_numa_id",
        type=int,
    )
    parser.add_argument(
        "--client-numa-ids",
        dest="client_numa_ids",
        help="comma-separated RDMA device ids, one per client process",
    )
    parser.add_argument(
        "--server-numa-id",
        "--rdma-rc-server-numa-id",
        dest="server_numa_id",
        type=int,
    )
    parser.add_argument("--server-bind-core-offset", type=int)
    parser.add_argument("--client-bind-core-offset", type=int)
    parser.add_argument("--client-bind-core-stride", type=int)
    parser.add_argument(
        "--fake-get-mode",
        "--rdma-rc-fake-get-mode",
        dest="fake_get_mode",
        choices=["none", "status_only", "index_only", "payload_memset"],
    )
    parser.add_argument(
        "--skip-client-copy",
        "--rdma-rc-skip-client-copy",
        dest="skip_client_copy",
        action="store_true",
    )
    parser.add_argument("--verify-values", action="store_true")
    parser.add_argument("--verify-value-row-stride", type=int, default=1)
    parser.add_argument("--show-runner-logs", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress logs and print only the final aggregate summary",
    )
    args = parser.parse_args()

    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    if args.server_count <= 0:
        raise ValueError("--server-count must be positive")
    if args.client_count <= 0:
        raise ValueError("--client-count must be positive")
    if args.client_numa_id is not None and args.client_numa_ids is not None:
        raise ValueError(
            "--client-numa-id and --client-numa-ids are mutually exclusive"
        )
    if args.slots_per_qp is not None and args.slots_per_qp <= 0:
        raise ValueError("--slots-per-qp must be positive")
    client_numa_ids = parse_client_numa_ids(args.client_numa_ids, args.client_count)
    if args.server_numa_id is None:
        args.server_numa_id = 0
    if (
        args.client_numa_id is None
        and args.client_numa_ids is None
        and local_numa_node_count() > 1
    ):
        args.client_numa_id = 1
    if args.op == "async_stream" and args.qps_per_client_per_shard is not None:
        slots_per_qp = args.slots_per_qp if args.slots_per_qp is not None else 1
        capacity = args.qps_per_client_per_shard * slots_per_qp
        if capacity < args.async_depth:
            raise ValueError(
                "async_stream requires qps_per_client_per_shard * slots_per_qp "
                ">= async_depth"
            )
    if args.server_bind_core_offset is None:
        args.server_bind_core_offset = 0
    if args.client_bind_core_offset is None:
        if args.client_numa_id != args.server_numa_id:
            args.client_bind_core_offset = 0
        else:
            args.client_bind_core_offset = args.server_count * max(
                1, args.thread_num + (args.server_get_workers or 0)
            )
    if args.client_bind_core_stride is None:
        args.client_bind_core_stride = 1

    source_config_path = resolve_rdma_integration_config(
        args.server_count, args.config_path
    )
    if args.server_count == 1 and source_config_path is None:
        source_config_path = DEFAULT_RDMA_SINGLE_SHARD_CONFIG
    if args.server_count > 1 and source_config_path is None:
        source_config_path = DEFAULT_RDMA_MULTI_SHARD_CONFIG

    max_kv_num_per_request = (
        args.max_kv_num_per_request
        if args.max_kv_num_per_request is not None
        else max(1, args.batch_keys)
    )

    with tempfile.TemporaryDirectory(prefix="recstore_rdma_rc_benchmark_") as tmpdir:
        config_path = write_runtime_config(args, source_config_path, tmpdir)
        args.config_path = config_path
        runner = PetPSClusterRunner(
            config_path=config_path,
            num_servers=args.server_count,
            num_clients=args.client_count,
            thread_num=args.thread_num,
            value_size=args.value_size,
            max_kv_num_per_request=max_kv_num_per_request,
            timeout=args.cluster_timeout,
            verbose=args.show_runner_logs,
            show_status_logs=args.show_runner_logs,
            show_control_plane_logs=args.show_runner_logs,
            rdma_namespace=args.rdma_namespace,
            rdma_control_plane_host=args.rdma_control_plane_host,
            rdma_control_plane_port=args.rdma_control_plane_port,
            rdma_qps_per_client_per_shard=args.qps_per_client_per_shard,
            rdma_slots_per_qp=args.slots_per_qp,
            rdma_wait_timeout_ms=args.rdma_wait_timeout_ms,
            rdma_profile_interval_ms=args.profile_interval_ms,
            rdma_server_coroutines_per_thread=args.server_coroutines_per_thread,
            rdma_server_get_workers=args.server_get_workers,
            rdma_inline_bytes=args.inline_bytes,
            rdma_client_numa_id=args.client_numa_id,
            rdma_client_numa_ids=client_numa_ids,
            rdma_server_numa_id=args.server_numa_id,
            rdma_server_bind_core_offset=args.server_bind_core_offset,
            rdma_client_bind_core_offset=args.client_bind_core_offset,
            rdma_client_bind_core_stride=args.client_bind_core_stride,
            rdma_fake_get_mode=args.fake_get_mode,
            rdma_skip_client_copy=args.skip_client_copy,
        )

        summary_rows = []
        with runner.run():
            returncode, rows = run_benchmark_clients(runner, args)
            summary_rows.extend(rows)
            if returncode != 0:
                return returncode

    if not args.quiet:
        print_summary_table(summary_rows)
        print_aggregate_table(summary_rows)
    else:
        print_aggregate_table(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
