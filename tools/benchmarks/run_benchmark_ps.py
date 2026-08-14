#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks._legacy_scripts import ensure_legacy_test_script_path

ensure_legacy_test_script_path()

from petps_cluster_runner import PetPSClusterRunner, REPO_ROOT
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_server_runner import PSServerRunner


SUMMARY_RE = re.compile(
    r"transport=(?P<transport>\S+) "
    r"op=(?P<op>\S+) "
    r"phase=(?P<phase>\S+) "
    r"summary "
    r"rounds=(?P<rounds>\d+) "
    r"iterations=(?P<iterations>\d+) "
    r"batch_keys=(?P<batch_keys>\d+) "
    r"elapsed_us_mean=(?P<mean>[0-9.eE+-]+) "
    r"elapsed_us_p50=(?P<p50>[0-9.eE+-]+) "
    r"elapsed_us_p95=(?P<p95>[0-9.eE+-]+) "
    r"elapsed_us_p99=(?P<p99>[0-9.eE+-]+) "
    r"ops_per_sec=(?P<ops>[0-9.eE+-]+) "
    r"key_ops_per_sec=(?P<key_ops>[0-9.eE+-]+)"
)
PS_RESULT_PREFIX = "PS_BENCHMARK_RESULT "
RUNNER_VERSION = 1
DEFAULT_REMOTE_REPO = "/app/RecStore"
DEFAULT_REMOTE_RUNTIME_ROOT = "/tmp/recstore_benchmark_ps"
DEFAULT_LOCAL_DATA_ROOT = "/tmp/recstore_benchmark_ps_data"
DEFAULT_RESULT_PREFIX = "benchmark_ps_"
DEFAULT_RDMA_FETCH_QPS_PER_CLIENT_PER_SHARD = 16
MIN_SSD_CAPACITY_BYTES = 256 * 1024 * 1024
SUCCESS_STATUS = "success"
SKIPPED_STATUS = "skipped"
SLAB_ALLOCATOR_CHUNK_BYTES = 1 << 20
SLAB_ALLOCATOR_METADATA_BYTES = 8
SLAB_ALLOCATOR_CAPACITY_HEADROOM_NUMERATOR = 6
SLAB_ALLOCATOR_CAPACITY_HEADROOM_DENOMINATOR = 5
SINGLE_SHARD_PORT_OVERRIDES = {
    "BRPC": 15000,
}


@dataclass(frozen=True)
class TransportSpec:
    name: str
    server_binary: str
    base_port: int
    uses_rdma_cluster: bool


@dataclass(frozen=True)
class ServerPlan:
    server_index: int
    host: str
    ssh_host: str
    shard: int
    transport: str
    port: int


@dataclass(frozen=True)
class ClientPlan:
    client_index: int
    host: str
    ssh_host: str
    transport: str


@dataclass(frozen=True)
class TopologyPlan:
    transport: str
    server_plan: list[ServerPlan]
    client_plan: list[ClientPlan]


@dataclass(frozen=True)
class ClientProcessSpec:
    client_index: int
    host: str
    cmd: list[str]
    cwd: str
    env: dict[str, str] | None
    stdout_log_path: Path
    stderr_log_path: Path


@dataclass(frozen=True)
class ClientProcessResult:
    client_index: int
    host: str
    returncode: int
    stdout: str
    stderr: str
    stdout_log_path: Path
    stderr_log_path: Path


TRANSPORT_SPECS = {
    "GRPC": TransportSpec("GRPC", "ps_server", 15000, False),
    "BRPC": TransportSpec("BRPC", "ps_server", 25000, False),
    "RDMA": TransportSpec("RDMA", "petps_server", 25000, True),
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_transport_list(value: str) -> list[str]:
    transports = []
    for item in parse_csv_list(value):
        transport = item.upper()
        if transport not in TRANSPORT_SPECS:
            raise ValueError(f"unsupported transport: {item}")
        if transport not in transports:
            transports.append(transport)
    if not transports:
        raise ValueError("at least one transport is required")
    return transports


def normalize_host_list(value: str, field_name: str) -> list[str]:
    hosts = parse_csv_list(value)
    if not hosts:
        raise ValueError(f"{field_name} must not be empty")
    return hosts


def split_ssh_endpoint_host(host: str) -> tuple[str, str]:
    if "@" not in host:
        return host, host
    user, endpoint = host.rsplit("@", 1)
    if not user or not endpoint:
        raise ValueError(f"invalid ssh endpoint host: {host}")
    return endpoint, host


def local_numa_node_count() -> int:
    node_root = Path("/sys/devices/system/node")
    if not node_root.exists():
        return 1
    count = sum(
        1
        for path in node_root.glob("node[0-9]*")
        if path.is_dir() and (path / "cpulist").exists()
    )
    return max(1, count)


def parse_server_plan(value: str, transport: str) -> list[ServerPlan]:
    servers = []
    for server_index, item in enumerate(parse_csv_list(value)):
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(
                "server_plan entries must be host:port:shard or "
                "server_index:host:port:shard"
            )
        if len(parts) == 3:
            host, port, shard = parts
            parsed_server_index = server_index
        else:
            parsed_server_index_raw, host, port, shard = parts
            parsed_server_index = int(parsed_server_index_raw)
        if not host:
            raise ValueError("server_plan host must not be empty")
        endpoint_host, ssh_host = split_ssh_endpoint_host(host)
        servers.append(
            ServerPlan(
                server_index=parsed_server_index,
                host=endpoint_host,
                ssh_host=ssh_host,
                shard=int(shard),
                transport=transport,
                port=int(port),
            )
        )
    if not servers:
        raise ValueError("server_plan must not be empty")
    if len({server.server_index for server in servers}) != len(servers):
        raise ValueError("server_plan server_index values must be unique")
    if len({server.shard for server in servers}) != len(servers):
        raise ValueError("server_plan shard values must be unique")
    return sorted(servers, key=lambda server: server.server_index)


def parse_client_plan(value: str, transport: str) -> list[ClientPlan]:
    clients = []
    for client_index, item in enumerate(parse_csv_list(value)):
        parts = item.split(":")
        if len(parts) == 1:
            parsed_client_index = client_index
            host = parts[0]
        elif len(parts) == 2:
            parsed_client_index = int(parts[0])
            host = parts[1]
        else:
            raise ValueError("client_plan entries must be host or client_index:host")
        if not host:
            raise ValueError("client_plan host must not be empty")
        endpoint_host, ssh_host = split_ssh_endpoint_host(host)
        clients.append(
            ClientPlan(
                client_index=parsed_client_index,
                host=endpoint_host,
                ssh_host=ssh_host,
                transport=transport,
            )
        )
    if not clients:
        raise ValueError("client_plan must not be empty")
    if len({client.client_index for client in clients}) != len(clients):
        raise ValueError("client_plan client_index values must be unique")
    return sorted(clients, key=lambda client: client.client_index)


def build_topology_plan(
    transport: str,
    server_shard_ips: list[str],
    client_ips: list[str],
    client_processes_per_ip: int,
    base_port: int,
    server_plan: str = "",
    client_plan: str = "",
) -> TopologyPlan:
    if server_plan:
        parsed_server_plan = parse_server_plan(server_plan, transport)
    else:
        parsed_server_plan = []
    if client_plan:
        parsed_client_plan = parse_client_plan(client_plan, transport)
    else:
        parsed_client_plan = []
    if parsed_server_plan or parsed_client_plan:
        if not parsed_server_plan:
            parsed_server_plan = build_topology_plan(
                transport,
                server_shard_ips,
                client_ips,
                client_processes_per_ip,
                base_port,
            ).server_plan
        if not parsed_client_plan:
            parsed_client_plan = build_topology_plan(
                transport,
                [server.host for server in parsed_server_plan],
                client_ips,
                client_processes_per_ip,
                base_port,
            ).client_plan
        return TopologyPlan(
            transport=transport,
            server_plan=parsed_server_plan,
            client_plan=parsed_client_plan,
        )

    if not server_shard_ips:
        raise ValueError("server_shard_ips must not be empty")
    if not client_ips:
        raise ValueError("client_ips must not be empty")
    if client_processes_per_ip <= 0:
        raise ValueError("client_processes_per_ip must be positive")

    server_plan = []
    for server_index, host in enumerate(server_shard_ips):
        endpoint_host, ssh_host = split_ssh_endpoint_host(host)
        server_plan.append(
            ServerPlan(
                server_index=server_index,
                host=endpoint_host,
                ssh_host=ssh_host,
                shard=server_index,
                transport=transport,
                port=base_port + server_index,
            )
        )

    client_plan = []
    for host in client_ips:
        endpoint_host, ssh_host = split_ssh_endpoint_host(host)
        for _ in range(client_processes_per_ip):
            client_plan.append(
                ClientPlan(
                    client_index=len(client_plan),
                    host=endpoint_host,
                    ssh_host=ssh_host,
                    transport=transport,
                )
            )

    return TopologyPlan(
        transport=transport,
        server_plan=server_plan,
        client_plan=client_plan,
    )


def build_runtime_config(
    transport: str,
    topology: TopologyPlan,
    capacity: int,
    value_size: int,
    max_keys_per_request: int,
    num_threads: int,
    index_type: str,
    value_store_type: str,
    dram_allocator: str,
    data_root: str,
    ssd_data_root: str,
    ssd_capacity_bytes: int,
    ssd_io_backend: str,
    ssd_queue_depth: int,
) -> dict:
    capacity_bytes = recommended_dram_capacity_bytes(
        capacity=capacity,
        value_size=value_size,
        dram_allocator=dram_allocator,
    )
    value_config = build_value_store_config(
        value_store_type=value_store_type,
        value_size=value_size,
        dram_allocator=dram_allocator,
        dram_capacity_bytes=capacity_bytes,
        data_root=data_root,
        ssd_data_root=ssd_data_root,
        ssd_capacity_bytes=ssd_capacity_bytes,
        ssd_io_backend=ssd_io_backend,
        ssd_queue_depth=ssd_queue_depth,
    )
    servers = [
        {"host": server.host, "port": server.port, "shard": server.shard}
        for server in topology.server_plan
    ]
    ps_type = "RDMA" if transport == "RDMA" else transport
    config = {
        "cache_ps": {
            "ps_type": ps_type,
            "max_batch_keys_size": max_keys_per_request,
            "num_threads": num_threads,
            "num_shards": len(topology.server_plan),
            "servers": servers,
            "base_kv_config": {
                "capacity": capacity,
                "index": {"type": index_type},
                "value": value_config,
            },
        },
        "distributed_client": {
            "num_shards": len(topology.server_plan),
            "hash_method": "city_hash",
            "max_keys_per_request": max_keys_per_request,
            "servers": servers,
        },
        "client": {
            "host": topology.server_plan[0].host,
            "port": topology.server_plan[0].port,
            "shard": topology.server_plan[0].shard,
        },
    }
    return config


def recommended_ssd_capacity_bytes(*, capacity: int, value_size: int) -> int:
    return max(
        capacity * max(value_size + SLAB_ALLOCATOR_METADATA_BYTES, 128) * 2,
        MIN_SSD_CAPACITY_BYTES,
    )


def build_ssd_allocator_config(
    *, capacity_bytes: int, ssd_io_backend: str, ssd_queue_depth: int
) -> dict:
    return {
        "type": "SSD_SLAB",
        "capacity_bytes": capacity_bytes,
        "min_block_size": 128,
        "max_block_size": 4096,
        "io": {
            "type": ssd_io_backend,
            "queue_depth": ssd_queue_depth,
            "base_offset_bytes": 4096,
        },
    }


def build_value_store_config(
    *,
    value_store_type: str,
    value_size: int,
    dram_allocator: str,
    dram_capacity_bytes: int,
    data_root: str,
    ssd_data_root: str,
    ssd_capacity_bytes: int,
    ssd_io_backend: str,
    ssd_queue_depth: int,
) -> dict:
    if value_store_type == "DRAM_VALUE_STORE":
        return {
            "type": "DRAM_VALUE_STORE",
            "default_value_size_hint": value_size,
            "dram_allocator": {
                "type": dram_allocator,
                "capacity_bytes": dram_capacity_bytes,
            },
            "path": data_root,
        }
    if value_store_type == "TIERED_VALUE_STORE":
        ssd_allocator = build_ssd_allocator_config(
            capacity_bytes=ssd_capacity_bytes,
            ssd_io_backend=ssd_io_backend,
            ssd_queue_depth=ssd_queue_depth,
        )
        ssd_allocator["path"] = str(Path(ssd_data_root) / "ssd.db")
        return {
            "type": "TIERED_VALUE_STORE",
            "default_value_size_hint": value_size,
            "dram_allocator": {
                "type": dram_allocator,
                "capacity_bytes": dram_capacity_bytes,
                "path": str(Path(data_root) / "dram"),
            },
            "ssd_allocator": ssd_allocator,
            "tiering": {"cache_policy": "LRU"},
        }
    raise ValueError(f"unsupported value_store_type: {value_store_type}")


def resolve_base_port(transport: str, requested_base_port: int, server_count: int) -> int:
    if server_count == 1 and transport in SINGLE_SHARD_PORT_OVERRIDES:
        return SINGLE_SHARD_PORT_OVERRIDES[transport]
    return requested_base_port


def recommended_dram_capacity_bytes(
    *, capacity: int, value_size: int, dram_allocator: str
) -> int:
    per_value_bytes = value_size
    if dram_allocator in {"PERSIST_LOOP_SLAB", "CONCURRENT_SLAB_MEMORY_POOL"}:
        per_value_bytes += SLAB_ALLOCATOR_METADATA_BYTES
        raw_capacity = int(capacity * per_value_bytes)
        raw_capacity = (
            raw_capacity
            * SLAB_ALLOCATOR_CAPACITY_HEADROOM_NUMERATOR
            + SLAB_ALLOCATOR_CAPACITY_HEADROOM_DENOMINATOR
            - 1
        ) // SLAB_ALLOCATOR_CAPACITY_HEADROOM_DENOMINATOR
        return (
            (raw_capacity + SLAB_ALLOCATOR_CHUNK_BYTES - 1)
            // SLAB_ALLOCATOR_CHUNK_BYTES
        ) * SLAB_ALLOCATOR_CHUNK_BYTES
    return int(capacity * per_value_bytes)


def build_benchmark_cmd(
    benchmark_binary: str,
    transport: str,
    topology: TopologyPlan,
    config_path: str,
    record_count: int,
    runtime_seconds: int,
    client_threads_per_process: int,
    load_threads: int,
    batch_keys: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    read_ratio: int,
    mode: str,
    report_mode: str,
    prefetch_depth: int = 0,
    transaction_profile: bool = False,
    rdma_direct_async_fetch: bool = False,
    rdma_adapter_skip_prefetch_result_copy: bool = False,
    rdma_get_response_mode: str = "direct_sg",
    skip_load: bool = False,
    verify_deterministic_values: bool = False,
) -> list[str]:
    cmd = [
        benchmark_binary,
        f"--transport={transport.lower()}",
        f"--host={topology.server_plan[0].host}",
        f"--port={topology.server_plan[0].port}",
        f"--num_shards={len(topology.server_plan)}",
        f"--config_path={config_path}",
        "--workload=transactions",
        f"--mode={mode}",
        f"--record_count={record_count}",
        f"--running_seconds={runtime_seconds}",
        f"--thread_num={client_threads_per_process}",
        f"--load_thread_num={load_threads}",
        f"--batch_keys={batch_keys}",
        f"--value_size={value_size}",
        f"--distribution={distribution}",
        f"--zipfian_alpha={zipfian_alpha}",
        f"--read_ratio={read_ratio}",
        f"--report_mode={report_mode}",
    ]
    if prefetch_depth > 0:
        cmd.append(f"--prefetch_depth={prefetch_depth}")
    if transaction_profile:
        cmd.append("--transaction_profile=true")
    if rdma_direct_async_fetch:
        cmd.append("--rdma_direct_async_fetch=true")
    if rdma_adapter_skip_prefetch_result_copy:
        cmd.append("--rdma_adapter_skip_prefetch_result_copy=true")
    if transport == "RDMA":
        cmd.append(f"--rdma_get_response_mode={rdma_get_response_mode}")
    if skip_load:
        cmd.append("--skip_load=true")
    if verify_deterministic_values:
        cmd.append("--verify_deterministic_values=true")
    return cmd


def replace_config_path_arg(argv: list[str], config_path: str) -> list[str]:
    replaced = []
    prefix = "--config_path="
    found = False
    for arg in argv:
        if arg.startswith(prefix):
            replaced.append(f"{prefix}{config_path}")
            found = True
        else:
            replaced.append(arg)
    if not found:
        replaced.append(f"{prefix}{config_path}")
    return replaced


def collect_summary_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match is None or match.group("phase") != "measure":
            continue
        rows.append(
            {
                "transport": match.group("transport"),
                "op": match.group("op"),
                "phase": match.group("phase"),
                "rounds": int(match.group("rounds")),
                "iterations": int(match.group("iterations")),
                "batch_keys": int(match.group("batch_keys")),
                "mean": float(match.group("mean")),
                "p50": float(match.group("p50")),
                "p95": float(match.group("p95")),
                "p99": float(match.group("p99")),
                "ops": float(match.group("ops")),
                "key_ops": float(match.group("key_ops")),
            }
        )
    return rows


def collect_ps_result_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith(PS_RESULT_PREFIX):
            continue
        row: dict[str, str | int | float] = {}
        for part in line.strip().split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            row[key] = value
        for key in ("threads", "batch_size", "records", "batches", "key_ops"):
            if key in row:
                row[key] = int(str(row[key]))
        for key in (
            "zipfian_alpha",
            "runtime_s",
            "throughput_batches_sec",
            "throughput_keys_sec",
        ):
            if key in row:
                row[key] = float(str(row[key]))
        rows.append(row)
    return rows


def is_port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


def wait_tcp_ports_ready(servers: list[ServerPlan], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(is_port_open(server.host, server.port) for server in servers):
            return
        time.sleep(0.2)
    ports = [f"{server.host}:{server.port}" for server in servers]
    raise TimeoutError(f"timed out waiting for tcp ports: {ports}")


def wait_rdma_control_plane_ready(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_port_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(
        f"timed out waiting for RDMA control plane: {host}:{port}"
    )


def format_log_paths(*paths: Path) -> str:
    return ";".join(str(path) for path in paths)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%m%d%H%M")
    return (REPO_ROOT / "results" / f"{DEFAULT_RESULT_PREFIX}{stamp}").resolve()


def resolve_output_dir(value: str) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path
    return default_output_dir()


def resolve_rdma_get_response_mode(index_type: str, requested_mode: str) -> str:
    mode = requested_mode.lower()
    if mode == "auto":
        return "staging_copy" if index_type == "DRAM_PET_HASH" else "direct_sg"
    if mode not in {"direct_sg", "staging_copy"}:
        raise ValueError(
            "rdma_get_response_mode must be auto, direct_sg, or staging_copy"
        )
    return mode


def prompt_value(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def apply_interactive_prompts(args: argparse.Namespace) -> None:
    print("Benchmark PS interactive setup. Press Enter to keep defaults.")
    args.transports = prompt_value("transports", args.transports)
    args.client_ips = prompt_value("client IPs", args.client_ips)
    args.server_shard_ips = prompt_value("server shard IPs", args.server_shard_ips)
    args.client_processes_per_ip = int(
        prompt_value(
            "client processes per IP", str(args.client_processes_per_ip)
        )
    )
    args.server_plan = prompt_value("server plan", args.server_plan)
    args.client_plan = prompt_value("client plan", args.client_plan)
    args.record_count = int(prompt_value("record count", str(args.record_count)))
    args.value_size = int(prompt_value("value size", str(args.value_size)))
    args.batch_keys = int(prompt_value("batch keys", str(args.batch_keys)))
    args.client_threads_per_process = int(
        prompt_value(
            "client threads per process", str(args.client_threads_per_process)
        )
    )
    args.runtime_seconds = int(
        prompt_value("runtime seconds", str(args.runtime_seconds))
    )
    args.repeat = int(prompt_value("repeat count", str(args.repeat)))
    args.execution_backend = prompt_value(
        "execution backend (local/ssh)", args.execution_backend
    )
    args.output_dir = resolve_output_dir(
        prompt_value("result output directory", str(args.output_dir))
    )


def quote_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def build_remote_exec_cmd(
    host: str,
    remote_repo: str,
    remote_container: str | None,
    shell_command: str,
) -> list[str]:
    repo_command = f"cd {shlex.quote(remote_repo)} && {shell_command}"
    if remote_container:
        container_command = (
            f"docker exec {shlex.quote(remote_container)} "
            f"bash -lc {shlex.quote(repo_command)}"
        )
        return ["ssh", host, "bash", "-lc", shlex.quote(container_command)]
    return ["ssh", host, "bash", "-lc", shlex.quote(repo_command)]


def run_command(
    cmd: list[str],
    *,
    capture_output: bool = True,
    check: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=capture_output,
        check=False,
        cwd=cwd,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def ensure_remote_path(
    host: str,
    path: str,
    remote_repo: str,
    remote_container: str | None,
) -> None:
    cmd = build_remote_exec_cmd(
        host,
        remote_repo,
        remote_container,
        f"mkdir -p {shlex.quote(path)}",
    )
    run_command(cmd, capture_output=True, check=True)


def sync_file_to_remote(
    local_path: Path,
    host: str,
    remote_path: str,
    remote_repo: str,
    remote_container: str | None,
) -> None:
    if remote_container:
        host_tmp_dir = f"/tmp/recstore_benchmark_ps_sync/{os.getpid()}"
        host_tmp_path = f"{host_tmp_dir}/{local_path.name}"
        run_command(
            [
                "ssh",
                host,
                "bash",
                "-lc",
                shlex.quote(f"mkdir -p {shlex.quote(host_tmp_dir)}"),
            ],
            capture_output=True,
            check=True,
        )
        run_command(
            ["scp", str(local_path), f"{host}:{host_tmp_path}"],
            capture_output=True,
            check=True,
        )
        remote_dir = os.path.dirname(remote_path)
        host_copy_command = " && ".join(
            [
                (
                    f"docker exec {shlex.quote(remote_container)} "
                    f"mkdir -p {shlex.quote(remote_dir)}"
                ),
                (
                    f"docker cp {shlex.quote(host_tmp_path)} "
                    f"{shlex.quote(remote_container)}:{shlex.quote(remote_path)}"
                ),
            ]
        )
        run_command(
            ["ssh", host, "bash", "-lc", shlex.quote(host_copy_command)],
            capture_output=True,
            check=True,
        )
        return

    remote_dir = os.path.dirname(remote_path)
    run_command(
        [
            "ssh",
            host,
            "bash",
            "-lc",
            shlex.quote(f"mkdir -p {shlex.quote(remote_dir)}"),
        ],
        capture_output=True,
        check=True,
    )
    run_command(
        ["scp", str(local_path), f"{host}:{remote_path}"],
        capture_output=True,
        check=True,
    )


def fetch_remote_text_file(
    host: str,
    remote_path: str,
    local_path: Path,
    remote_repo: str,
    remote_container: str | None,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_remote_exec_cmd(
        host,
        remote_repo,
        remote_container,
        f"cat {shlex.quote(remote_path)}",
    )
    completed = run_command(cmd, capture_output=True, check=False)
    if completed.returncode != 0:
        local_path.write_text(
            (
                f"[fetch_remote_text_file] failed for {host}:{remote_path}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}\n"
            ),
            encoding="utf-8",
        )
        return
    local_path.write_text(completed.stdout, encoding="utf-8")


def ensure_remote_binary_state(
    local_binary: Path,
    host: str,
    remote_binary_path: str,
    remote_repo: str,
    remote_container: str | None,
    remote_sync: str,
) -> None:
    if remote_sync == "none":
        return
    if remote_sync == "check":
        cmd = build_remote_exec_cmd(
            host,
            remote_repo,
            remote_container,
            f"test -x {shlex.quote(remote_binary_path)}",
        )
        run_command(cmd, capture_output=True, check=True)
        return
    if remote_sync != "rsync":
        raise ValueError(f"unsupported remote_sync mode: {remote_sync}")
    if remote_container is None and shutil.which("rsync"):
        remote_dir = os.path.dirname(remote_binary_path)
        run_command(
            [
                "ssh",
                host,
                "bash",
                "-lc",
                shlex.quote(f"mkdir -p {shlex.quote(remote_dir)}"),
            ],
            capture_output=True,
            check=True,
        )
        run_command(
            ["rsync", "-a", str(local_binary), f"{host}:{remote_binary_path}"],
            capture_output=True,
            check=True,
        )
    else:
        if not shutil.which("scp"):
            raise RuntimeError("scp is required for --remote-sync=rsync")
        sync_file_to_remote(
            local_binary,
            host,
            remote_binary_path,
            remote_repo,
            remote_container,
        )
    cmd = build_remote_exec_cmd(
        host,
        remote_repo,
        remote_container,
        f"chmod +x {shlex.quote(remote_binary_path)}",
    )
    run_command(cmd, capture_output=True, check=True)


def check_remote_rdma_available(
    host: str,
    remote_repo: str,
    remote_container: str | None,
) -> str | None:
    cmd = build_remote_exec_cmd(
        host,
        remote_repo,
        remote_container,
        "test -d /dev/infiniband && ls /dev/infiniband/uverbs* >/dev/null 2>&1",
    )
    completed = run_command(cmd, capture_output=True, check=False)
    if completed.returncode == 0:
        return None
    return f"remote RDMA verbs devices are unavailable on host {host}"


def get_git_metadata() -> dict[str, str | bool]:
    head = run_command(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    status = run_command(
        ["git", "status", "--short"],
        capture_output=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_base_result_row(
    *,
    transport: str,
    status: str,
    repeat_index: int,
    client_index: int,
    topology: TopologyPlan,
    record_count: int,
    value_size: int,
    batch_keys: int,
    client_threads_per_process: int,
    runtime_seconds: int,
    distribution: str,
    mode: str,
    log_path: str,
    message: str = "",
    phase: str = "",
) -> dict[str, str | int | float]:
    return {
        "transport": transport,
        "status": status,
        "phase": phase,
        "client_index": client_index,
        "repeat_index": repeat_index,
        "server_shards": len(topology.server_plan),
        "client_processes": len(topology.client_plan),
        "server_shard_ips": ",".join(server.host for server in topology.server_plan),
        "client_ips": ",".join(sorted({client.host for client in topology.client_plan})),
        "record_count": record_count,
        "value_size": value_size,
        "batch_keys": batch_keys,
        "client_threads_per_process": client_threads_per_process,
        "runtime_seconds": runtime_seconds,
        "distribution": distribution,
        "mode": mode,
        "ops_per_sec": "",
        "key_ops_per_sec": "",
        "payload_gbps": "",
        "mean_us": "",
        "p50_us": "",
        "p95_us": "",
        "p99_us": "",
        "log_path": log_path,
        "message": message,
    }


def result_rows_from_client_output(
    *,
    transport: str,
    repeat_index: int,
    topology: TopologyPlan,
    record_count: int,
    value_size: int,
    batch_keys: int,
    client_threads_per_process: int,
    runtime_seconds: int,
    distribution: str,
    mode: str,
    client_result: ClientProcessResult,
) -> list[dict[str, str | int | float]]:
    result_rows = []
    ps_rows = collect_ps_result_rows(client_result.stdout)
    summary_rows = collect_summary_rows(client_result.stdout)
    summary_by_op = {row["op"]: row for row in summary_rows}
    for row in ps_rows:
        phase = str(row.get("phase", ""))
        phase_summary = summary_by_op.get("get") if phase == "run" else None
        result = make_base_result_row(
            transport=transport,
            status=SUCCESS_STATUS,
            repeat_index=repeat_index,
            client_index=client_result.client_index,
            topology=topology,
            record_count=record_count,
            value_size=value_size,
            batch_keys=batch_keys,
            client_threads_per_process=client_threads_per_process,
            runtime_seconds=runtime_seconds,
            distribution=distribution,
            mode=mode,
            log_path=format_log_paths(
                client_result.stdout_log_path, client_result.stderr_log_path
            ),
            phase=phase,
        )
        result["ops_per_sec"] = row.get("throughput_batches_sec", "")
        result["key_ops_per_sec"] = row.get("throughput_keys_sec", "")
        if result["key_ops_per_sec"] not in ("", None):
            result["payload_gbps"] = (
                float(result["key_ops_per_sec"]) * value_size / 1e9
            )
        if phase_summary is not None:
            result["mean_us"] = phase_summary.get("mean", "")
            result["p50_us"] = phase_summary.get("p50", "")
            result["p95_us"] = phase_summary.get("p95", "")
            result["p99_us"] = phase_summary.get("p99", "")
        result_rows.append(result)
    return result_rows


def write_summary_csv(rows: list[dict[str, str | int | float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transport",
        "status",
        "phase",
        "client_index",
        "repeat_index",
        "server_shards",
        "client_processes",
        "server_shard_ips",
        "client_ips",
        "record_count",
        "value_size",
        "batch_keys",
        "client_threads_per_process",
        "runtime_seconds",
        "distribution",
        "mode",
        "ops_per_sec",
        "key_ops_per_sec",
        "payload_gbps",
        "mean_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "log_path",
        "message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _format_metric(value: str | int | float) -> str:
    if value in ("", None):
        return "-"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, int):
        return str(value)
    return f"{value:,.2f}"


def _format_mkeys(value: str | int | float) -> str:
    if value in ("", None):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric / 1e6:,.3f}"


def aggregate_transport_payload_gbps(
    rows: list[dict[str, str | int | float]],
) -> dict[str, float]:
    per_repeat: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("status") != SUCCESS_STATUS or row.get("phase") != "run":
            continue
        payload_gbps = row.get("payload_gbps", "")
        if payload_gbps in ("", None):
            continue
        key = (str(row["transport"]), int(row["repeat_index"]))
        per_repeat[key] = per_repeat.get(key, 0.0) + float(payload_gbps)
    by_transport: dict[str, list[float]] = {}
    for (transport, _), payload_gbps in per_repeat.items():
        by_transport.setdefault(transport, []).append(payload_gbps)
    return {
        transport: sum(values) / len(values)
        for transport, values in by_transport.items()
    }


def write_summary_markdown(
    rows: list[dict[str, str | int | float]],
    summary_path: Path,
    args: argparse.Namespace,
    run_config: dict,
) -> None:
    success_rows = [row for row in rows if row["status"] == SUCCESS_STATUS]
    issue_rows = [row for row in rows if row["status"] != SUCCESS_STATUS]

    lines = [
        "# Benchmark PS Summary",
        "",
        "## Workload 说明",
        (
            "本次结果属于 PS/network 层。"
            f"transports={args.transports}，execution_backend={args.execution_backend}，"
            f"client_ips={args.client_ips}，server_shard_ips={args.server_shard_ips}，"
            f"client_plan={args.client_plan or '-'}，server_plan={args.server_plan or '-'}，"
            f"server_shards={len(args.server_shard_ips.split(','))}，"
            f"client_processes_per_ip={args.client_processes_per_ip}，"
            f"record_count={args.record_count}，value_size={args.value_size}，"
            f"batch_keys={args.batch_keys}，"
            f"client_threads_per_process={args.client_threads_per_process}，"
            f"client_load_threads_per_process="
            f"{args.client_load_threads_per_process if args.client_load_threads_per_process > 0 else args.client_threads_per_process}，"
            f"server_worker_threads={args.server_worker_threads}，"
            f"server_rdma_threads={args.server_rdma_threads}，"
            f"index_type={args.index_type}，value_store_type={args.value_store_type}，"
            f"dram_allocator={args.dram_allocator}，"
            f"tiered_dram_capacity_bytes={args.tiered_dram_capacity_bytes or '-'}，"
            f"ssd_capacity_bytes={args.ssd_capacity_bytes or '-'}，"
            f"ssd_io_backend={args.ssd_io_backend}，"
            f"ssd_queue_depth={args.ssd_queue_depth}，"
            f"runtime_seconds={args.runtime_seconds}，repeat={args.repeat}，"
            f"distribution={args.distribution}，mode={args.mode}。"
        ),
        "payload GB/s = key_ops/s × value_size，仅表示有效 value payload，不含协议头、key 请求、索引查询、staging 或拷贝开销。",
        f"运行配置见 `{run_config['run_config_path']}`，原始结果见 `{run_config['summary_csv_path']}`。",
        "",
        "## 成功结果",
    ]

    if success_rows:
        lines.extend(
            [
                "| transport | repeat | phase | client | M keys/s | payload GB/s | mean_us | log_path |",
                "|-|-|-|-|-|-|-|-|",
            ]
        )
        for row in success_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["transport"]),
                        str(row["repeat_index"]),
                        str(row["phase"]),
                        str(row["client_index"]),
                        _format_mkeys(row["key_ops_per_sec"]),
                        _format_metric(row["payload_gbps"]),
                        _format_metric(row["mean_us"]),
                        str(row["log_path"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("无成功结果。")

    payload_summary = aggregate_transport_payload_gbps(success_rows)
    if payload_summary:
        brpc_payload = payload_summary.get("BRPC")
        lines.extend(
            [
                "",
                "## Transport 对比",
                "| transport | 有效 value payload GB/s | 相对 BRPC |",
                "|-|-:|-:|",
            ]
        )
        for transport in sorted(payload_summary):
            payload_gbps = payload_summary[transport]
            ratio = (
                f"{payload_gbps / brpc_payload:.2f}x"
                if brpc_payload is not None and brpc_payload > 0
                else "-"
            )
            lines.append(f"| {transport} | {payload_gbps:.3f} | {ratio} |")

    lines.extend(["", "## Skip / Failure"])
    if issue_rows:
        lines.extend(
            [
                "| transport | status | repeat | client | message | log_path |",
                "|-|-|-|-|-|-|",
            ]
        )
        for row in issue_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["transport"]),
                        str(row["status"]),
                        str(row["repeat_index"]),
                        str(row["client_index"]),
                        str(row["message"]),
                        str(row["log_path"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("无 skip 或 failure。")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_client_process_specs(
    *,
    topology: TopologyPlan,
    benchmark_cmd: list[str],
    client_timeout: int,
    client_log_dir: Path,
    env_builder,
    command_builder,
    cwd: str,
) -> list[ClientProcessSpec]:
    specs = []
    for client in topology.client_plan:
        stdout_log_path = client_log_dir / f"client_{client.client_index}.stdout.log"
        stderr_log_path = client_log_dir / f"client_{client.client_index}.stderr.log"
        specs.append(
            ClientProcessSpec(
                client_index=client.client_index,
                host=client.ssh_host,
                cmd=command_builder(benchmark_cmd, client),
                cwd=cwd,
                env=env_builder(client),
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
            )
        )
    return specs


def run_client_process_group(
    specs: list[ClientProcessSpec],
    timeout: int,
) -> list[ClientProcessResult]:
    processes = []
    for spec in specs:
        spec.stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = spec.stdout_log_path.open("w", encoding="utf-8")
        stderr_handle = spec.stderr_log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            spec.cmd,
            cwd=spec.cwd,
            env=spec.env,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        processes.append((spec, process, stdout_handle, stderr_handle))

    deadline = time.monotonic() + timeout if timeout > 0 else None
    results = []
    try:
        for spec, process, _stdout_handle, _stderr_handle in processes:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(spec.cmd, timeout)
            process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        for _spec, process, _stdout_handle, _stderr_handle in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        raise
    finally:
        for _spec, _process, stdout_handle, stderr_handle in processes:
            stdout_handle.close()
            stderr_handle.close()

    for spec, process, _stdout_handle, _stderr_handle in processes:
        stdout = spec.stdout_log_path.read_text(encoding="utf-8")
        stderr = spec.stderr_log_path.read_text(encoding="utf-8")
        results.append(
            ClientProcessResult(
                client_index=spec.client_index,
                host=spec.host,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                stdout_log_path=spec.stdout_log_path,
                stderr_log_path=spec.stderr_log_path,
            )
        )
    return results


def build_rpc_server_cmd(
    server_binary: str,
    transport: str,
    config_path: str,
    shard: int,
    port: int,
) -> list[str]:
    cmd = [
        server_binary,
        f"--config_path={config_path}",
        f"--brpc_config_path={config_path}",
        f"--grpc_local_shard_id={shard}",
        f"--local_shard_id={shard}",
    ]
    if transport == "BRPC":
        cmd.append(f"--brpc_server_port={port}")
    return cmd


def build_rdma_runner(
    args: argparse.Namespace,
    *,
    config_path: str,
    server_binary: str,
    server_shards: int,
    client_processes: int,
    value_size: int,
    max_keys_per_request: int,
    rdma_namespace: str,
    rdma_control_plane_host: str,
    rdma_control_plane_port: int | None,
) -> PetPSClusterRunner:
    return PetPSClusterRunner(
        server_path=server_binary,
        config_path=config_path,
        num_servers=server_shards,
        num_clients=client_processes,
        logical_clients_per_process=getattr(
            args, "client_threads_per_process", 1
        ),
        thread_num=args.server_rdma_threads,
        value_size=value_size,
        max_kv_num_per_request=max_keys_per_request,
        timeout=args.cluster_timeout,
        startup_delay=args.startup_delay,
        log_dir=str(args.output_dir / "logs"),
        verbose=args.show_runner_logs,
        show_status_logs=args.show_runner_logs,
        show_control_plane_logs=args.show_runner_logs,
        rdma_namespace=rdma_namespace,
        rdma_control_plane_host=rdma_control_plane_host,
        rdma_control_plane_port=rdma_control_plane_port,
        rdma_control_plane_timeout_ms=max(args.cluster_timeout * 1000, 30000),
        rdma_wait_timeout_ms=args.rdma_wait_timeout_ms,
        rdma_rc_qps_per_client_per_shard=args.rdma_rc_qps_per_client_per_shard,
        rdma_rc_slots_per_qp=args.rdma_rc_slots_per_qp,
        rdma_rc_profile_interval_ms=args.rdma_rc_profile_interval_ms,
        rdma_rc_server_coroutines_per_thread=args.rdma_rc_server_coroutines_per_thread,
        rdma_rc_server_get_workers=args.rdma_rc_server_get_workers,
        rdma_rc_inline_bytes=args.rdma_rc_inline_bytes,
        rdma_rc_client_numa_id=args.rdma_rc_client_numa_id,
        rdma_rc_server_numa_id=args.rdma_rc_server_numa_id,
        rdma_rc_fake_get_mode=args.rdma_rc_fake_get_mode,
        rdma_rc_skip_client_copy=args.rdma_rc_skip_client_copy,
        rdma_server_bind_core_offset=args.rdma_server_bind_core_offset,
        rdma_client_bind_core_offset=args.rdma_client_bind_core_offset,
        rdma_client_bind_core_stride=args.rdma_client_bind_core_stride,
    )


def dump_petps_server_logs(runner: PetPSClusterRunner, server_log_dir: Path) -> list[Path]:
    server_log_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for global_id, lines in sorted(runner.process_logs.items()):
        log_path = server_log_dir / f"server_{global_id}.log"
        log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        paths.append(log_path)
    return paths


def resolve_local_build_dir(args: argparse.Namespace) -> Path:
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = REPO_ROOT / build_dir
    return build_dir


def run_local_rpc_case(
    args: argparse.Namespace,
    *,
    transport: str,
    topology: TopologyPlan,
    config_path: Path,
    benchmark_cmd: list[str],
    repeat_index: int,
) -> tuple[list[dict[str, str | int | float]], bool]:
    client_log_dir = args.output_dir / "logs" / transport.lower() / f"repeat_{repeat_index}"
    server_log_dir = client_log_dir / "server"
    runner = PSServerRunner(
        launcher_cli=str(
            (resolve_local_build_dir(args) / "bin" / "ps_server_launcher_cli").resolve()
        ),
        server_path=str(
            (
                resolve_local_build_dir(args)
                / "bin"
                / TRANSPORT_SPECS[transport].server_binary
            ).resolve()
        ),
        config_path=str(config_path),
        log_dir=str(server_log_dir),
        timeout=args.cluster_timeout,
        num_shards=len(topology.server_plan),
        verbose=args.show_runner_logs,
        startup_delay=args.startup_delay,
    )

    try:
        with runner.run():
            specs = build_client_process_specs(
                topology=topology,
                benchmark_cmd=benchmark_cmd,
                client_timeout=args.client_timeout,
                client_log_dir=client_log_dir,
                env_builder=lambda _client: None,
                command_builder=lambda base_cmd, _client: list(base_cmd),
                cwd=str(REPO_ROOT),
            )
            results = run_client_process_group(specs, args.client_timeout)
    except subprocess.TimeoutExpired:
        server_logs = sorted(server_log_dir.glob("*.log"))
        issue = make_base_result_row(
            transport=transport,
            status="timeout",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path=";".join(str(path) for path in server_logs),
            message=f"client timeout after {args.client_timeout}s",
        )
        return [issue], False
    except Exception as exc:
        server_logs = sorted(server_log_dir.glob("*.log"))
        issue = make_base_result_row(
            transport=transport,
            status="startup_failure",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path=";".join(str(path) for path in server_logs),
            message=str(exc),
        )
        return [issue], False

    rows = []
    ok = True
    for result in results:
        if result.returncode != 0:
            rows.append(
                make_base_result_row(
                    transport=transport,
                    status="client_failure",
                    repeat_index=repeat_index,
                    client_index=result.client_index,
                    topology=topology,
                    record_count=args.record_count,
                    value_size=args.value_size,
                    batch_keys=args.batch_keys,
                    client_threads_per_process=args.client_threads_per_process,
                    runtime_seconds=args.runtime_seconds,
                    distribution=args.distribution,
                    mode=args.mode,
                    log_path=format_log_paths(
                        result.stdout_log_path, result.stderr_log_path
                    ),
                    message=f"client exited with code {result.returncode}",
                )
            )
            ok = False
            continue
        rows.extend(
            result_rows_from_client_output(
                transport=transport,
                repeat_index=repeat_index,
                topology=topology,
                record_count=args.record_count,
                value_size=args.value_size,
                batch_keys=args.batch_keys,
                client_threads_per_process=args.client_threads_per_process,
                runtime_seconds=args.runtime_seconds,
                distribution=args.distribution,
                mode=args.mode,
                client_result=result,
            )
        )
    return rows, ok


def run_local_rdma_case(
    args: argparse.Namespace,
    *,
    topology: TopologyPlan,
    config_path: Path,
    benchmark_cmd: list[str],
    repeat_index: int,
    rdma_namespace: str,
    rdma_control_plane_host: str,
    rdma_control_plane_port: int | None,
) -> tuple[list[dict[str, str | int | float]], bool]:
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        issue = make_base_result_row(
            transport="RDMA",
            status=SKIPPED_STATUS,
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path="",
            message=skip_reason,
        )
        return [issue], True

    max_keys_per_request = max(args.batch_keys, args.max_keys_per_request)
    runner = build_rdma_runner(
        args,
        config_path=str(config_path),
        server_binary=str(
            (resolve_local_build_dir(args) / "bin" / "petps_server").resolve()
        ),
        server_shards=len(topology.server_plan),
        client_processes=len(topology.client_plan),
        value_size=args.value_size,
        max_keys_per_request=max_keys_per_request,
        rdma_namespace=rdma_namespace,
        rdma_control_plane_host=rdma_control_plane_host,
        rdma_control_plane_port=rdma_control_plane_port,
    )
    client_log_dir = args.output_dir / "logs" / "rdma" / f"repeat_{repeat_index}"
    server_log_dir = client_log_dir / "server"

    try:
        with runner.run():
            specs = build_client_process_specs(
                topology=topology,
                benchmark_cmd=benchmark_cmd,
                client_timeout=args.client_timeout,
                client_log_dir=client_log_dir,
                env_builder=lambda client: runner.build_client_env(
                    client.client_index
                ),
                command_builder=lambda base_cmd, client: runner.build_client_cmd(
                    list(base_cmd), client_index=client.client_index
                ),
                cwd=str(REPO_ROOT),
            )
            results = run_client_process_group(specs, args.client_timeout)
            server_logs = dump_petps_server_logs(runner, server_log_dir)
    except subprocess.TimeoutExpired:
        issue = make_base_result_row(
            transport="RDMA",
            status="timeout",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path="",
            message=f"client timeout after {args.client_timeout}s",
        )
        return [issue], False
    except Exception as exc:
        issue = make_base_result_row(
            transport="RDMA",
            status="startup_failure",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path="",
            message=str(exc),
        )
        return [issue], False

    rows = []
    ok = True
    server_log_path = ";".join(str(path) for path in server_logs)
    for result in results:
        if result.returncode == RDMA_SKIP_EXIT_CODE:
            rows.append(
                make_base_result_row(
                    transport="RDMA",
                    status=SKIPPED_STATUS,
                    repeat_index=repeat_index,
                    client_index=result.client_index,
                    topology=topology,
                    record_count=args.record_count,
                    value_size=args.value_size,
                    batch_keys=args.batch_keys,
                    client_threads_per_process=args.client_threads_per_process,
                    runtime_seconds=args.runtime_seconds,
                    distribution=args.distribution,
                    mode=args.mode,
                    log_path=format_log_paths(
                        result.stdout_log_path, result.stderr_log_path
                    ),
                    message=result.stdout.strip() or result.stderr.strip(),
                )
            )
            continue
        if result.returncode != 0:
            rows.append(
                make_base_result_row(
                    transport="RDMA",
                    status="client_failure",
                    repeat_index=repeat_index,
                    client_index=result.client_index,
                    topology=topology,
                    record_count=args.record_count,
                    value_size=args.value_size,
                    batch_keys=args.batch_keys,
                    client_threads_per_process=args.client_threads_per_process,
                    runtime_seconds=args.runtime_seconds,
                    distribution=args.distribution,
                    mode=args.mode,
                    log_path=format_log_paths(
                        result.stdout_log_path, result.stderr_log_path
                    )
                    + (f";{server_log_path}" if server_log_path else ""),
                    message=f"client exited with code {result.returncode}",
                )
            )
            ok = False
            continue
        rows.extend(
            result_rows_from_client_output(
                transport="RDMA",
                repeat_index=repeat_index,
                topology=topology,
                record_count=args.record_count,
                value_size=args.value_size,
                batch_keys=args.batch_keys,
                client_threads_per_process=args.client_threads_per_process,
                runtime_seconds=args.runtime_seconds,
                distribution=args.distribution,
                mode=args.mode,
                client_result=result,
            )
        )
    return rows, ok


def build_remote_background_server_cmd(
    host: str,
    remote_repo: str,
    remote_container: str | None,
    argv: list[str],
    remote_log_path: str,
) -> list[str]:
    remote_log_dir = os.path.dirname(remote_log_path)
    shell_command = " && ".join(
        [
            f"mkdir -p {shlex.quote(remote_log_dir)}",
            (
                f"(nohup {quote_argv(argv)} > {shlex.quote(remote_log_path)} "
                "2>&1 < /dev/null & echo $!)"
            ),
        ]
    )
    return build_remote_exec_cmd(host, remote_repo, remote_container, shell_command)


def run_remote_case(
    args: argparse.Namespace,
    *,
    transport: str,
    topology: TopologyPlan,
    config_path: Path,
    benchmark_cmd: list[str],
    repeat_index: int,
    rdma_namespace: str,
    rdma_control_plane_host: str,
    rdma_control_plane_port: int | None,
) -> tuple[list[dict[str, str | int | float]], bool]:
    remote_run_root = (
        f"{args.remote_runtime_root}/run_{os.getpid()}_{int(time.time())}"
        f"/{transport.lower()}/repeat_{repeat_index}"
    )
    remote_config_path = f"{remote_run_root}/config.json"
    remote_server_log_dir = f"{remote_run_root}/logs/server"
    remote_client_log_dir = f"{remote_run_root}/logs/client"
    unique_hosts = sorted(
        {server.ssh_host for server in topology.server_plan}
        | {client.ssh_host for client in topology.client_plan}
    )

    for host in unique_hosts:
        ensure_remote_path(host, remote_run_root, args.remote_repo, args.remote_container)

    spec = TRANSPORT_SPECS[transport]
    local_build_dir = Path(args.build_dir)
    if not local_build_dir.is_absolute():
        local_build_dir = REPO_ROOT / local_build_dir
    remote_build_dir = args.remote_build_dir.strip("/")
    local_server_binary = local_build_dir / "bin" / spec.server_binary
    local_benchmark_binary = local_build_dir / "bin" / "ps_transport_benchmark"
    remote_server_binary = (
        f"{args.remote_repo}/{remote_build_dir}/bin/{spec.server_binary}"
    )
    remote_benchmark_binary = (
        f"{args.remote_repo}/{remote_build_dir}/bin/ps_transport_benchmark"
    )

    try:
        for host in unique_hosts:
            ensure_remote_binary_state(
                local_server_binary,
                host,
                remote_server_binary,
                args.remote_repo,
                args.remote_container,
                args.remote_sync,
            )
            ensure_remote_binary_state(
                local_benchmark_binary,
                host,
                remote_benchmark_binary,
                args.remote_repo,
                args.remote_container,
                args.remote_sync,
            )
            sync_file_to_remote(
                config_path,
                host,
                remote_config_path,
                args.remote_repo,
                args.remote_container,
            )
            if transport == "RDMA":
                reason = check_remote_rdma_available(
                    host, args.remote_repo, args.remote_container
                )
                if reason:
                    issue = make_base_result_row(
                        transport=transport,
                        status=SKIPPED_STATUS,
                        repeat_index=repeat_index,
                        client_index=-1,
                        topology=topology,
                        record_count=args.record_count,
                        value_size=args.value_size,
                        batch_keys=args.batch_keys,
                        client_threads_per_process=args.client_threads_per_process,
                        runtime_seconds=args.runtime_seconds,
                        distribution=args.distribution,
                        mode=args.mode,
                        log_path="",
                        message=reason,
                    )
                    return [issue], True
    except Exception as exc:
        issue = make_base_result_row(
            transport=transport,
            status="preflight_failure",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path=str(config_path),
            message=str(exc),
        )
        return [issue], False

    server_specs = []
    if transport == "RDMA":
        max_keys_per_request = max(args.batch_keys, args.max_keys_per_request)
        rdma_builder = build_rdma_runner(
            args,
            config_path=remote_config_path,
            server_binary=remote_server_binary,
            server_shards=len(topology.server_plan),
            client_processes=len(topology.client_plan),
            value_size=args.value_size,
            max_keys_per_request=max_keys_per_request,
            rdma_namespace=rdma_namespace,
            rdma_control_plane_host=rdma_control_plane_host,
            rdma_control_plane_port=rdma_control_plane_port,
        )
        for server in topology.server_plan:
            remote_log_path = f"{remote_server_log_dir}/server_{server.server_index}.log"
            server_specs.append(
                (
                    server,
                    build_remote_background_server_cmd(
                        server.ssh_host,
                        args.remote_repo,
                        args.remote_container,
                        rdma_builder.build_server_cmd(server.server_index),
                        remote_log_path,
                    ),
                    remote_log_path,
                )
            )
    else:
        for server in topology.server_plan:
            remote_log_path = f"{remote_server_log_dir}/server_{server.server_index}.log"
            server_specs.append(
                (
                    server,
                    build_remote_background_server_cmd(
                        server.ssh_host,
                        args.remote_repo,
                        args.remote_container,
                        build_rpc_server_cmd(
                            remote_server_binary,
                            transport,
                            remote_config_path,
                            server.shard,
                            server.port,
                        ),
                        remote_log_path,
                    ),
                    remote_log_path,
                )
            )

    remote_processes = []
    local_server_log_dir = (
        args.output_dir / "logs" / transport.lower() / f"repeat_{repeat_index}" / "server"
    )
    local_server_log_dir.mkdir(parents=True, exist_ok=True)

    try:
        for server, cmd, remote_log_path in server_specs:
            completed = run_command(cmd, capture_output=True, check=True)
            pid = completed.stdout.strip().splitlines()[-1].strip()
            remote_processes.append((server, pid, remote_log_path))

        if transport == "RDMA":
            wait_rdma_control_plane_ready(
                rdma_control_plane_host,
                rdma_control_plane_port if rdma_control_plane_port is not None else 25100,
                args.cluster_timeout,
            )
        else:
            wait_tcp_ports_ready(topology.server_plan, args.cluster_timeout)

        if transport == "RDMA":
            max_keys_per_request = max(args.batch_keys, args.max_keys_per_request)
            rdma_builder = build_rdma_runner(
                args,
                config_path=remote_config_path,
                server_binary=remote_server_binary,
                server_shards=len(topology.server_plan),
                client_processes=len(topology.client_plan),
                value_size=args.value_size,
                max_keys_per_request=max_keys_per_request,
                rdma_namespace=rdma_namespace,
                rdma_control_plane_host=rdma_control_plane_host,
                rdma_control_plane_port=rdma_control_plane_port,
            )
            specs = build_client_process_specs(
                topology=topology,
                benchmark_cmd=benchmark_cmd,
                client_timeout=args.client_timeout,
                client_log_dir=args.output_dir
                / "logs"
                / transport.lower()
                / f"repeat_{repeat_index}",
                env_builder=lambda _client: None,
                command_builder=lambda base_cmd, client: build_remote_exec_cmd(
                    client.ssh_host,
                    args.remote_repo,
                    args.remote_container,
                    quote_argv(
                        rdma_builder.build_client_cmd(
                            replace_config_path_arg(
                                [remote_benchmark_binary, *base_cmd[1:]],
                                remote_config_path,
                            ),
                            client_index=client.client_index,
                        )
                    ),
                ),
                cwd=str(REPO_ROOT),
            )
        else:
            remote_cmd = replace_config_path_arg(
                [remote_benchmark_binary, *benchmark_cmd[1:]],
                remote_config_path,
            )
            specs = build_client_process_specs(
                topology=topology,
                benchmark_cmd=remote_cmd,
                client_timeout=args.client_timeout,
                client_log_dir=args.output_dir
                / "logs"
                / transport.lower()
                / f"repeat_{repeat_index}",
                env_builder=lambda _client: None,
                command_builder=lambda base_cmd, client: build_remote_exec_cmd(
                    client.ssh_host,
                    args.remote_repo,
                    args.remote_container,
                    quote_argv(base_cmd),
                ),
                cwd=str(REPO_ROOT),
            )

        results = run_client_process_group(specs, args.client_timeout)
    except subprocess.TimeoutExpired:
        issue = make_base_result_row(
            transport=transport,
            status="timeout",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path="",
            message=f"client timeout after {args.client_timeout}s",
        )
        results = None
        ok = False
        rows = [issue]
    except Exception as exc:
        issue = make_base_result_row(
            transport=transport,
            status="startup_failure",
            repeat_index=repeat_index,
            client_index=-1,
            topology=topology,
            record_count=args.record_count,
            value_size=args.value_size,
            batch_keys=args.batch_keys,
            client_threads_per_process=args.client_threads_per_process,
            runtime_seconds=args.runtime_seconds,
            distribution=args.distribution,
            mode=args.mode,
            log_path="",
            message=str(exc),
        )
        results = None
        ok = False
        rows = [issue]
    else:
        rows = []
        ok = True
        for result in results:
            if result.returncode != 0:
                rows.append(
                    make_base_result_row(
                        transport=transport,
                        status="client_failure",
                        repeat_index=repeat_index,
                        client_index=result.client_index,
                        topology=topology,
                        record_count=args.record_count,
                        value_size=args.value_size,
                        batch_keys=args.batch_keys,
                        client_threads_per_process=args.client_threads_per_process,
                        runtime_seconds=args.runtime_seconds,
                        distribution=args.distribution,
                        mode=args.mode,
                        log_path=format_log_paths(
                            result.stdout_log_path, result.stderr_log_path
                        ),
                        message=f"client exited with code {result.returncode}",
                    )
                )
                ok = False
                continue
            rows.extend(
                result_rows_from_client_output(
                    transport=transport,
                    repeat_index=repeat_index,
                    topology=topology,
                    record_count=args.record_count,
                    value_size=args.value_size,
                    batch_keys=args.batch_keys,
                    client_threads_per_process=args.client_threads_per_process,
                    runtime_seconds=args.runtime_seconds,
                    distribution=args.distribution,
                    mode=args.mode,
                    client_result=result,
                )
            )
    finally:
        for server, pid, remote_log_path in remote_processes:
            stop_cmd = build_remote_exec_cmd(
                server.ssh_host,
                args.remote_repo,
                args.remote_container,
                f"kill {shlex.quote(pid)} >/dev/null 2>&1 || true",
            )
            run_command(stop_cmd, capture_output=True, check=False)
            fetch_remote_text_file(
                server.ssh_host,
                remote_log_path,
                local_server_log_dir / f"server_{server.server_index}.log",
                args.remote_repo,
                args.remote_container,
            )

    return rows, ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PS/network benchmark across rdma, grpc, and brpc transports."
        )
    )
    parser.add_argument("--benchmark-binary", default="")
    parser.add_argument(
        "--build-dir",
        default="build",
        help=(
            "Local build directory used for ssh backend binary state checks. "
            "Defaults to build."
        ),
    )
    parser.add_argument("--transports", default="rdma,grpc,brpc")
    parser.add_argument(
        "--client-ips",
        default="127.0.0.1",
        help="Comma-separated client hosts. Each host starts client-processes-per-ip processes.",
    )
    parser.add_argument(
        "--server-shard-ips",
        default="127.0.0.1",
        help="Comma-separated server hosts, one entry per shard. The list length is the shard count.",
    )
    parser.add_argument(
        "--server-plan",
        default="",
        help=(
            "Explicit server topology as comma-separated host:port:shard or "
            "server_index:host:port:shard entries. Overrides "
            "--server-shard-ips mapping when set."
        ),
    )
    parser.add_argument(
        "--client-plan",
        default="",
        help=(
            "Explicit client topology as comma-separated host or "
            "client_index:host entries. Overrides --client-ips/"
            "--client-processes-per-ip mapping when set."
        ),
    )
    parser.add_argument("--client-processes-per-ip", type=int, default=1)
    parser.add_argument("--record-count", type=int, default=1000000)
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--batch-keys", type=int, default=1024)
    parser.add_argument("--client-threads-per-process", type=int, default=16)
    parser.add_argument("--client-load-threads-per-process", type=int, default=0)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument(
        "--distribution", choices=["uniform", "zipfian"], default="uniform"
    )
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument(
        "--mode",
        choices=["fetch", "insert", "mixed", "fetch_insert"],
        default="fetch",
    )
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--max-keys-per-request", type=int, default=500)
    parser.add_argument("--server-worker-threads", type=int, default=32)
    parser.add_argument("--index-type", default="DRAM_EXTENDIBLE_HASH")
    parser.add_argument(
        "--value-store-type",
        choices=["DRAM_VALUE_STORE", "TIERED_VALUE_STORE"],
        default="DRAM_VALUE_STORE",
    )
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument(
        "--tiered-dram-capacity-bytes",
        type=int,
        default=0,
        help=(
            "Override TIERED_VALUE_STORE DRAM tier capacity. "
            "Use a small value to force SSD-tier fallback coverage."
        ),
    )
    parser.add_argument("--ssd-capacity-bytes", type=int, default=0)
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument(
        "--report-mode",
        choices=["summary", "per_round", "both"],
        default="summary",
    )
    parser.add_argument(
        "--execution-backend", choices=["local", "ssh"], default="local"
    )
    parser.add_argument(
        "--remote-sync", choices=["check", "rsync", "none"], default="check"
    )
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument(
        "--remote-build-dir",
        default="build",
        help=(
            "Build directory under --remote-repo used by ssh backend. "
            "Defaults to build."
        ),
    )
    parser.add_argument("--remote-container", default="")
    parser.add_argument("--remote-runtime-root", default=DEFAULT_REMOTE_RUNTIME_ROOT)
    parser.add_argument("--local-data-root", default=DEFAULT_LOCAL_DATA_ROOT)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--client-timeout", type=int, default=120)
    parser.add_argument("--cluster-timeout", type=int, default=45)
    parser.add_argument("--show-runner-logs", action="store_true")
    parser.add_argument("--server-rdma-threads", type=int, default=1)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument("--rdma-wait-timeout-ms", type=int)
    parser.add_argument("--rdma-rc-qps-per-client-per-shard", type=int)
    parser.add_argument("--rdma-rc-slots-per-qp", type=int)
    parser.add_argument("--rdma-rc-profile-interval-ms", type=int)
    parser.add_argument("--rdma-rc-server-coroutines-per-thread", type=int)
    parser.add_argument("--rdma-rc-server-get-workers", type=int)
    parser.add_argument("--rdma-server-bind-core-offset", type=int)
    parser.add_argument("--rdma-client-bind-core-offset", type=int)
    parser.add_argument("--rdma-client-bind-core-stride", type=int)
    parser.add_argument("--rdma-rc-client-numa-id", type=int)
    parser.add_argument("--rdma-rc-server-numa-id", type=int)
    parser.add_argument("--rdma-rc-inline-bytes", type=int)
    parser.add_argument(
        "--rdma-rc-fake-get-mode",
        choices=["none", "status_only", "index_only", "payload_memset"],
    )
    parser.add_argument("--rdma-rc-skip-client-copy", action="store_true")
    parser.add_argument("--transaction-profile", action="store_true")
    parser.add_argument("--verify-deterministic-values", action="store_true")
    parser.add_argument("--rdma-direct-async-fetch", action="store_true")
    parser.add_argument("--rdma-adapter-skip-prefetch-result-copy", action="store_true")
    parser.add_argument(
        "--rdma-get-response-mode",
        choices=["auto", "direct_sg", "staging_copy"],
        default="auto",
        help=(
            "RDMA GET response path. auto uses staging_copy for "
            "DRAM_PET_HASH and direct_sg otherwise."
        ),
    )
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for benchmark parameters before execution.",
    )
    args = parser.parse_args()
    args.output_dir = resolve_output_dir(args.output_dir)
    args.remote_container = args.remote_container or None
    if args.interactive:
        apply_interactive_prompts(args)
        args.remote_container = args.remote_container or None
    args.rdma_get_response_mode = resolve_rdma_get_response_mode(
        args.index_type, args.rdma_get_response_mode
    )
    transports = normalize_transport_list(args.transports)
    if "RDMA" in transports and args.mode == "fetch":
        qps_per_client_per_shard = (
            args.rdma_rc_qps_per_client_per_shard
            or DEFAULT_RDMA_FETCH_QPS_PER_CLIENT_PER_SHARD
        )
        slots_per_qp = args.rdma_rc_slots_per_qp or 1
        prefetch_depth = args.prefetch_depth or 16
        if qps_per_client_per_shard <= 0:
            parser.error("--rdma-rc-qps-per-client-per-shard must be positive")
        if slots_per_qp <= 0:
            parser.error("--rdma-rc-slots-per-qp must be positive")
        slot_capacity = qps_per_client_per_shard * slots_per_qp
        if prefetch_depth > slot_capacity:
            parser.error(
                "RDMA fetch prefetch depth exceeds RC slot capacity: "
                f"prefetch_depth={prefetch_depth}, "
                f"rdma_rc_qps_per_client_per_shard={qps_per_client_per_shard}, "
                f"rdma_rc_slots_per_qp={slots_per_qp}"
            )
        if args.rdma_rc_qps_per_client_per_shard is None:
            args.rdma_rc_qps_per_client_per_shard = qps_per_client_per_shard
    if "RDMA" in transports and args.execution_backend == "local":
        if args.rdma_rc_server_numa_id is None:
            args.rdma_rc_server_numa_id = 0
        if args.rdma_rc_client_numa_id is None:
            args.rdma_rc_client_numa_id = args.rdma_rc_server_numa_id
        server_shards = (
            len(parse_server_plan(args.server_plan, "RDMA"))
            if args.server_plan
            else len(normalize_host_list(args.server_shard_ips, "server_shard_ips"))
        )
        get_workers = args.rdma_rc_server_get_workers or 0
        server_bind_core_stride = max(
            1, args.server_rdma_threads + get_workers
        )
        if args.rdma_server_bind_core_offset is None:
            args.rdma_server_bind_core_offset = 0
        if args.rdma_client_bind_core_offset is None:
            if args.rdma_rc_client_numa_id != args.rdma_rc_server_numa_id:
                args.rdma_client_bind_core_offset = 0
            else:
                args.rdma_client_bind_core_offset = (
                    server_shards * server_bind_core_stride
                )
        if args.rdma_client_bind_core_stride is None:
            args.rdma_client_bind_core_stride = max(
                args.client_threads_per_process,
                args.client_load_threads_per_process,
                1,
            )
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transports = normalize_transport_list(args.transports)
    server_shard_ips = normalize_host_list(
        args.server_shard_ips, "server_shard_ips"
    )
    client_ips = normalize_host_list(args.client_ips, "client_ips")
    benchmark_binary = Path(
        args.benchmark_binary
        or (Path(args.build_dir) / "bin" / "ps_transport_benchmark")
    )
    if not benchmark_binary.is_absolute():
        benchmark_binary = (REPO_ROOT / benchmark_binary).resolve()
    if not benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {benchmark_binary}")

    capacity = args.capacity if args.capacity > 0 else args.record_count
    dram_capacity_bytes = recommended_dram_capacity_bytes(
        capacity=capacity,
        value_size=args.value_size,
        dram_allocator=args.dram_allocator,
    )
    if (
        args.value_store_type == "TIERED_VALUE_STORE"
        and args.tiered_dram_capacity_bytes > 0
    ):
        dram_capacity_bytes = args.tiered_dram_capacity_bytes
    ssd_capacity_bytes = args.ssd_capacity_bytes or recommended_ssd_capacity_bytes(
        capacity=capacity,
        value_size=args.value_size,
    )
    load_threads = (
        args.client_load_threads_per_process
        if args.client_load_threads_per_process > 0
        else args.client_threads_per_process
    )
    git_metadata = get_git_metadata()
    run_config = {
        "benchmark_ps_runner_version": RUNNER_VERSION,
        "run_config_path": str((args.output_dir / "run_config.json").resolve()),
        "summary_csv_path": str((args.output_dir / "summary.csv").resolve()),
        "summary_md_path": str((args.output_dir / "summary.md").resolve()),
        "git": git_metadata,
        "args": vars(args).copy(),
        "cases": [],
    }
    run_config["args"]["output_dir"] = str(args.output_dir)

    all_rows = []
    overall_ok = True
    for transport in transports:
        spec = TRANSPORT_SPECS[transport]
        base_port = resolve_base_port(
            transport,
            spec.base_port,
            len(server_shard_ips),
        )
        topology = build_topology_plan(
            transport,
            server_shard_ips,
            client_ips,
            args.client_processes_per_ip,
            base_port,
            server_plan=args.server_plan,
            client_plan=args.client_plan,
        )
        rdma_control_plane_host = (
            args.rdma_control_plane_host or topology.server_plan[0].host
        )
        rdma_control_plane_port = args.rdma_control_plane_port
        if transport == "RDMA" and rdma_control_plane_port is None:
            if args.execution_backend == "local":
                rdma_control_plane_port = None
            else:
                rdma_control_plane_port = 25100

        for repeat_index in range(args.repeat):
            run_id = f"{transport.lower()}_repeat_{repeat_index}"
            rdma_namespace = args.rdma_namespace
            if rdma_namespace == "auto":
                rdma_namespace = f"benchmark-ps-{os.getpid()}-{repeat_index}"
            case_data_root = (
                Path(args.local_data_root) / run_id / "value"
            ).as_posix()
            case_ssd_data_root = (
                Path(args.local_data_root) / run_id / "ssd"
            ).as_posix()
            Path(case_data_root).mkdir(parents=True, exist_ok=True)
            Path(case_ssd_data_root).mkdir(parents=True, exist_ok=True)
            config = build_runtime_config(
                transport=transport,
                topology=topology,
                capacity=capacity,
                value_size=args.value_size,
                max_keys_per_request=max(args.batch_keys, args.max_keys_per_request),
                num_threads=args.server_worker_threads,
                index_type=args.index_type,
                value_store_type=args.value_store_type,
                dram_allocator=args.dram_allocator,
                data_root=case_data_root,
                ssd_data_root=case_ssd_data_root,
                ssd_capacity_bytes=ssd_capacity_bytes,
                ssd_io_backend=args.ssd_io_backend,
                ssd_queue_depth=args.ssd_queue_depth,
            )
            config["cache_ps"]["base_kv_config"]["value"]["dram_allocator"][
                "capacity_bytes"
            ] = dram_capacity_bytes
            config_path = args.output_dir / "configs" / f"{run_id}.json"
            write_json(config_path, config)

            benchmark_cmd = build_benchmark_cmd(
                benchmark_binary=str(benchmark_binary),
                transport=transport,
                topology=topology,
                config_path=str(config_path),
                record_count=args.record_count,
                runtime_seconds=args.runtime_seconds,
                client_threads_per_process=args.client_threads_per_process,
                load_threads=load_threads,
                batch_keys=args.batch_keys,
                value_size=args.value_size,
                distribution=args.distribution,
                zipfian_alpha=args.zipfian_alpha,
                read_ratio=args.read_ratio,
                mode=args.mode,
                report_mode=args.report_mode,
                prefetch_depth=args.prefetch_depth,
                transaction_profile=args.transaction_profile,
                rdma_direct_async_fetch=args.rdma_direct_async_fetch,
                rdma_adapter_skip_prefetch_result_copy=(
                    args.rdma_adapter_skip_prefetch_result_copy
                ),
                rdma_get_response_mode=args.rdma_get_response_mode,
                skip_load=args.skip_load,
                verify_deterministic_values=args.verify_deterministic_values,
            )

            run_config["cases"].append(
                {
                    "transport": transport,
                    "repeat_index": repeat_index,
                    "config_path": str(config_path),
                    "topology": asdict(topology),
                    "rdma_namespace": rdma_namespace,
                    "rdma_control_plane_host": rdma_control_plane_host,
                    "rdma_control_plane_port": rdma_control_plane_port,
                }
            )

            if args.execution_backend == "local":
                if transport == "RDMA":
                    rows, ok = run_local_rdma_case(
                        args,
                        topology=topology,
                        config_path=config_path,
                        benchmark_cmd=benchmark_cmd,
                        repeat_index=repeat_index,
                        rdma_namespace=rdma_namespace,
                        rdma_control_plane_host=rdma_control_plane_host,
                        rdma_control_plane_port=rdma_control_plane_port,
                    )
                else:
                    rows, ok = run_local_rpc_case(
                        args,
                        transport=transport,
                        topology=topology,
                        config_path=config_path,
                        benchmark_cmd=benchmark_cmd,
                        repeat_index=repeat_index,
                    )
            else:
                rows, ok = run_remote_case(
                    args,
                    transport=transport,
                    topology=topology,
                    config_path=config_path,
                    benchmark_cmd=benchmark_cmd,
                    repeat_index=repeat_index,
                    rdma_namespace=rdma_namespace,
                    rdma_control_plane_host=rdma_control_plane_host,
                    rdma_control_plane_port=rdma_control_plane_port,
                )

            all_rows.extend(rows)
            overall_ok = overall_ok and ok

    summary_csv_path = args.output_dir / "summary.csv"
    write_summary_csv(all_rows, summary_csv_path)
    summary_md_path = args.output_dir / "summary.md"
    write_summary_markdown(all_rows, summary_md_path, args, run_config)
    write_json(args.output_dir / "run_config.json", run_config)
    print(f"[output] run_config={args.output_dir / 'run_config.json'}")
    print(f"[output] summary_csv={summary_csv_path}")
    print(f"[output] summary_md={summary_md_path}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
