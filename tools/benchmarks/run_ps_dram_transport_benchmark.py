#!/usr/bin/env python3

import argparse
import csv
import json
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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

DEFAULT_BACKENDS = (
    "dram_eh_dram",
    "dram_map_dram",
    "dram_pet_dram",
    "hps_hash_map",
    "hps_rocksdb",
)
DEFAULT_TRANSPORTS = ("GRPC", "BRPC", "LOCAL_SHM")
DEFAULT_DRAM_DATA_ROOT = Path("/dev/shm/recstore_ps_dram_bench")
DEFAULT_SSD_DATA_ROOT = Path("/tmp/recstore_ps_backend_bench")
MIN_SSD_CAPACITY_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TransportSpec:
    transport: str
    server_binary: str
    base_port: int


@dataclass(frozen=True)
class BackendSpec:
    alias: str
    backend_layer: str
    external_engine_type: str
    index_type: str
    value_store_type: str
    uses_ssd: bool = False


class BenchmarkCaseError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


TRANSPORT_SPECS = {
    "GRPC": TransportSpec("GRPC", "ps_server", 15000),
    "BRPC": TransportSpec("BRPC", "ps_server", 25000),
    "LOCAL_SHM": TransportSpec("LOCAL_SHM", "local_shm_ps_server", 0),
}

BACKEND_SPECS = {
    "dram_eh_dram": BackendSpec(
        "dram_eh_dram", "PS/network", "", "DRAM_EXTENDIBLE_HASH", "DRAM_VALUE_STORE"
    ),
    "dram_map_dram": BackendSpec(
        "dram_map_dram", "PS/network", "", "DRAM_UNORDERED_MAP", "DRAM_VALUE_STORE"
    ),
    "dram_pet_dram": BackendSpec(
        "dram_pet_dram", "PS/network", "", "DRAM_PET_HASH", "DRAM_VALUE_STORE"
    ),
    "dram_eh_ssd": BackendSpec(
        "dram_eh_ssd",
        "PS/network",
        "",
        "DRAM_EXTENDIBLE_HASH",
        "SSD_VALUE_STORE",
        True,
    ),
    "dram_pet_ssd": BackendSpec(
        "dram_pet_ssd",
        "PS/network",
        "",
        "DRAM_PET_HASH",
        "SSD_VALUE_STORE",
        True,
    ),
    "dram_eh_tiered": BackendSpec(
        "dram_eh_tiered",
        "PS/network",
        "",
        "DRAM_EXTENDIBLE_HASH",
        "TIERED_VALUE_STORE",
        True,
    ),
    "hps_hash_map": BackendSpec(
        "hps_hash_map", "PS/network", "KVEngineHPSHashMap", "", "HPS_HASH_MAP"
    ),
    "hps_rocksdb": BackendSpec(
        "hps_rocksdb",
        "PS/network",
        "KVEngineHPSRocksDB",
        "",
        "HPS_ROCKSDB",
        True,
    ),
}


def build_runtime_config(
    transport: str,
    backend_alias: str,
    runtime_dir: Path,
    num_shards: int,
    base_port: int,
    capacity: int,
    value_size: int,
    max_keys_per_request: int,
    num_threads: int,
    dram_allocator: str,
    local_shm_region: str,
    local_shm_slot_count: int,
    local_shm_ready_queue_count: int,
    local_shm_ready_queue_burst_limit: int,
    local_shm_slot_buffer_bytes: int,
    local_shm_client_timeout_ms: int,
    local_shm_thread_ready_queue_sharding: bool,
    dram_capacity_multiplier: float,
    ssd_capacity_multiplier: float = 2.0,
    ssd_io_backend: str = "IOURING",
    ssd_queue_depth: int = 512,
) -> dict:
    normalized_transport = transport.upper()
    backend = resolve_backend_spec(backend_alias)
    servers = [
        {"host": "127.0.0.1", "port": base_port + shard, "shard": shard}
        for shard in range(num_shards)
    ]
    case_slug = f"{backend.alias}_{normalized_transport.lower()}"
    data_root = (
        DEFAULT_DRAM_DATA_ROOT
        / runtime_dir.name
        / case_slug
    )
    ssd_root = DEFAULT_SSD_DATA_ROOT / runtime_dir.name / case_slug
    base_kv_config = build_base_kv_config(
        backend=backend,
        data_root=data_root,
        ssd_root=ssd_root,
        capacity=capacity,
        value_size=value_size,
        dram_allocator=dram_allocator,
        dram_capacity_multiplier=dram_capacity_multiplier,
        ssd_capacity_multiplier=ssd_capacity_multiplier,
        ssd_io_backend=ssd_io_backend,
        ssd_queue_depth=ssd_queue_depth,
    )
    config = {
        "cache_ps": {
            "ps_type": normalized_transport,
            "max_batch_keys_size": max_keys_per_request,
            "num_threads": num_threads,
            "num_shards": num_shards,
            "servers": servers,
            "base_kv_config": base_kv_config,
        },
        "distributed_client": {
            "num_shards": num_shards,
            "hash_method": "city_hash",
            "max_keys_per_request": max_keys_per_request,
            "servers": servers,
        },
        "client": {
            "host": "127.0.0.1",
            "port": base_port,
            "shard": 0,
        },
    }
    if normalized_transport == "LOCAL_SHM":
        config["local_shm"] = {
            "region_name": local_shm_region,
            "slot_count": local_shm_slot_count,
            "ready_queue_count": local_shm_ready_queue_count,
            "ready_queue_burst_limit": local_shm_ready_queue_burst_limit,
            "slot_buffer_bytes": local_shm_slot_buffer_bytes,
            "client_timeout_ms": local_shm_client_timeout_ms,
            "thread_ready_queue_sharding": local_shm_thread_ready_queue_sharding,
        }
    return config


def resolve_backend_spec(alias: str) -> BackendSpec:
    normalized = alias.strip().lower()
    if normalized.upper().startswith("DRAM_"):
        normalized = {
            "dram_extendible_hash": "dram_eh_dram",
            "dram_unordered_map": "dram_map_dram",
            "dram_pet_hash": "dram_pet_dram",
        }.get(normalized, normalized)
    try:
        return BACKEND_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported backend alias: {alias}") from exc


def build_base_kv_config(
    backend: BackendSpec,
    data_root: Path,
    ssd_root: Path,
    capacity: int,
    value_size: int,
    dram_allocator: str,
    dram_capacity_multiplier: float,
    ssd_capacity_multiplier: float,
    ssd_io_backend: str,
    ssd_queue_depth: int,
) -> dict:
    if backend.external_engine_type:
        path = ssd_root / backend.alias if backend.alias == "hps_rocksdb" else data_root
        config = {
            "external_engine_type": backend.external_engine_type,
            "capacity": capacity,
            "path": str(path),
            "value_size": value_size,
            "num_threads": 1 if backend.alias == "hps_rocksdb" else 0,
            "max_batch_size": 65536,
        }
        if backend.alias == "hps_rocksdb":
            config["rocksdb_path"] = str(path)
        return config

    dram_capacity = int(capacity * value_size * dram_capacity_multiplier)
    ssd_capacity = max(
        int(capacity * max(value_size + 8, 128) * ssd_capacity_multiplier),
        MIN_SSD_CAPACITY_BYTES,
    )
    config = {
        "capacity": capacity,
        "index": {"type": backend.index_type},
        "value": {
            "type": backend.value_store_type,
            "default_value_size_hint": value_size,
        },
    }
    if backend.value_store_type == "DRAM_VALUE_STORE":
        config["value"]["path"] = str(data_root / "value")
        config["value"]["dram_allocator"] = {
            "type": dram_allocator,
            "capacity_bytes": dram_capacity,
        }
    elif backend.value_store_type == "SSD_VALUE_STORE":
        config["value"]["path"] = str(ssd_root / "value.db")
        config["value"]["ssd_allocator"] = build_ssd_allocator(
            ssd_capacity, ssd_io_backend, ssd_queue_depth
        )
    elif backend.value_store_type == "TIERED_VALUE_STORE":
        config["value"]["dram_allocator"] = {
            "type": dram_allocator,
            "capacity_bytes": dram_capacity,
            "path": str(data_root / "dram"),
        }
        config["value"]["ssd_allocator"] = build_ssd_allocator(
            ssd_capacity, ssd_io_backend, ssd_queue_depth
        )
        config["value"]["ssd_allocator"]["path"] = str(ssd_root / "ssd.db")
        config["value"]["tiering"] = {"cache_policy": "none"}
    else:
        raise ValueError(f"unsupported value_store_type: {backend.value_store_type}")
    return config


def build_ssd_allocator(
    capacity_bytes: int, ssd_io_backend: str, ssd_queue_depth: int
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


def prepare_runtime_paths(config: dict) -> None:
    base_kv = config.get("cache_ps", {}).get("base_kv_config", {})
    if not isinstance(base_kv, dict):
        return
    if base_kv.get("external_engine_type") == "KVEngineHPSRocksDB":
        for key in ("path", "rocksdb_path"):
            path = base_kv.get(key)
            if path:
                Path(str(path)).mkdir(parents=True, exist_ok=True)

    value = base_kv.get("value", {})
    if not isinstance(value, dict):
        return
    for path in (
        value.get("path"),
        value.get("dram_allocator", {}).get("path")
        if isinstance(value.get("dram_allocator"), dict)
        else None,
        value.get("ssd_allocator", {}).get("path")
        if isinstance(value.get("ssd_allocator"), dict)
        else None,
    ):
        if path:
            Path(str(path)).parent.mkdir(parents=True, exist_ok=True)


def build_benchmark_cmd(
    benchmark_binary: Path,
    transport: str,
    host: str,
    port: int,
    num_shards: int,
    config_path: Path,
    mode: str,
    record_count: int,
    runtime_seconds: int,
    threads: int,
    load_threads: int,
    batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    read_ratio: int,
    report_mode: str,
    seed: int,
    phase: Literal["load", "run", "load_and_run"] = "load_and_run",
) -> list[str]:
    cmd = [
        str(benchmark_binary),
        f"--transport={transport.lower()}",
        f"--host={host}",
        f"--port={port}",
        f"--num_shards={num_shards}",
        f"--config_path={config_path}",
        "--workload=transactions",
        f"--mode={mode}",
        f"--record_count={record_count}",
        f"--running_seconds={runtime_seconds}",
        f"--thread_num={threads}",
        f"--load_thread_num={load_threads}",
        f"--batch_keys={batch_size}",
        f"--value_size={value_size}",
        f"--distribution={distribution}",
        f"--zipfian_alpha={zipfian_alpha}",
        f"--read_ratio={read_ratio}",
        f"--report_mode={report_mode}",
        f"--seed={seed}",
    ]
    if phase == "load":
        cmd.append("--load_only=true")
    elif phase == "run":
        cmd.append("--skip_load=true")
    return cmd


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


def collect_case_rows(
    process_outputs: list[tuple[int, str]],
    backend_alias: str,
    value_size: int,
    capacity: int,
    read_ratio: int,
    client_processes: int,
) -> list[dict[str, str | int | float]]:
    backend = resolve_backend_spec(backend_alias)
    rows: list[dict[str, str | int | float]] = []
    for process_id, stdout in process_outputs:
        for row in collect_ps_result_rows(stdout):
            row["status"] = "ok"
            row["failure_stage"] = ""
            row["error_tail"] = ""
            row["backend_alias"] = backend.alias
            row["backend_layer"] = backend.backend_layer
            row["index_type"] = backend.index_type
            row["value_store_type"] = backend.value_store_type
            row["value_size"] = value_size
            row["capacity"] = capacity
            row["read_ratio"] = read_ratio
            row["client_processes"] = client_processes
            row["process_id"] = process_id
            row["aggregate"] = "false"
            rows.append(row)

    aggregate_groups: dict[tuple[str, str, str, str], list[dict[str, str | int | float]]] = {}
    for row in rows:
        key = (
            str(row.get("phase", "")),
            str(row.get("transport", "")),
            str(row.get("mode", "")),
            str(row.get("distribution", "")),
        )
        aggregate_groups.setdefault(key, []).append(row)

    for group_rows in aggregate_groups.values():
        first = group_rows[0]
        aggregate = dict(first)
        aggregate["process_id"] = "all"
        aggregate["aggregate"] = "true"
        aggregate["threads"] = sum(int(row["threads"]) for row in group_rows)
        aggregate["client_processes"] = len(group_rows)
        aggregate["runtime_s"] = max(float(row["runtime_s"]) for row in group_rows)
        aggregate["batches"] = sum(int(row["batches"]) for row in group_rows)
        aggregate["key_ops"] = sum(int(row["key_ops"]) for row in group_rows)
        aggregate["throughput_batches_sec"] = sum(
            float(row["throughput_batches_sec"]) for row in group_rows
        )
        aggregate["throughput_keys_sec"] = sum(
            float(row["throughput_keys_sec"]) for row in group_rows
        )
        rows.append(aggregate)

    return rows


def build_failure_row(
    backend_alias: str,
    transport: str,
    mode: str,
    read_ratio: int,
    threads: int,
    client_processes: int,
    batch_size: int,
    value_size: int,
    capacity: int,
    distribution: str,
    zipfian_alpha: float,
    failure_stage: str,
    error_tail: str,
) -> dict[str, str | int | float]:
    backend = resolve_backend_spec(backend_alias)
    return {
        "status": "failed",
        "failure_stage": failure_stage,
        "error_tail": " ".join(error_tail.split())[:1000],
        "backend_alias": backend.alias,
        "backend_layer": backend.backend_layer,
        "index_type": backend.index_type,
        "value_store_type": backend.value_store_type,
        "value_size": value_size,
        "capacity": capacity,
        "transport": transport,
        "phase": failure_stage,
        "mode": mode,
        "read_ratio": read_ratio,
        "threads": threads * client_processes,
        "client_processes": client_processes,
        "process_id": "all",
        "aggregate": "true",
        "batch_size": batch_size,
        "records": capacity,
        "distribution": distribution,
        "zipfian_alpha": zipfian_alpha,
        "runtime_s": "",
        "batches": "",
        "key_ops": "",
        "throughput_batches_sec": "",
        "throughput_keys_sec": "",
    }


def is_port_open(host: str, port: int, timeout_s: float = 0.2) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def wait_process_ready(process: subprocess.Popen[str], delay_s: float) -> None:
    time.sleep(delay_s)
    if process.poll() is not None:
        raise RuntimeError(f"ps server exited early with code {process.returncode}")


def wait_tcp_ports_ready(
    process: subprocess.Popen[str],
    servers: list[dict],
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ps server exited early with code {process.returncode}")
        if all(is_port_open(str(server["host"]), int(server["port"])) for server in servers):
            return
        time.sleep(0.1)
    ports = [f"{server['host']}:{server['port']}" for server in servers]
    raise RuntimeError(f"timed out waiting for ps server ports: {ports}")


def run_one_case(
    repo_root: Path,
    server_binary: Path,
    benchmark_binary: Path,
    config_path: Path,
    server_log_path: Path,
    transport: str,
    num_shards: int,
    mode: str,
    record_count: int,
    runtime_seconds: int,
    threads: int,
    load_threads: int,
    batch_size: int,
    load_batch_size: int,
    value_size: int,
    distribution: str,
    zipfian_alpha: float,
    read_ratio: int,
    report_mode: str,
    startup_delay: float,
    client_timeout_s: int,
    client_processes: int,
    seed: int,
    output_dir: Path,
    split_load_phase: bool = True,
) -> list[tuple[int, str, str]]:
    with server_log_path.open("w", encoding="utf-8") as server_log:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        server_cmd = [str(server_binary), f"--config_path={config_path}"]
        if transport.upper() == "BRPC" and num_shards == 1:
            server_cmd.append(
                f"--brpc_server_port={int(config['cache_ps']['servers'][0]['port'])}"
            )
        server = subprocess.Popen(
            server_cmd,
            cwd=str(repo_root),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            if transport.upper() == "LOCAL_SHM":
                wait_process_ready(server, startup_delay)
            else:
                wait_tcp_ports_ready(
                    server,
                    config["cache_ps"]["servers"],
                    timeout_s=max(startup_delay, 1.0) + 30.0,
                )
            client = config["client"]
            load_cmd = build_benchmark_cmd(
                benchmark_binary=benchmark_binary,
                transport=transport,
                host=client["host"],
                port=int(client["port"]),
                num_shards=num_shards,
                config_path=config_path,
                mode=mode,
                record_count=record_count,
                runtime_seconds=runtime_seconds,
                threads=threads,
                load_threads=load_threads,
                batch_size=batch_size,
                value_size=value_size,
                distribution=distribution,
                zipfian_alpha=zipfian_alpha,
                read_ratio=read_ratio,
                report_mode=report_mode,
                seed=seed,
                phase="load" if split_load_phase else "load_and_run",
            )
            if split_load_phase and load_batch_size != batch_size:
                load_cmd = build_benchmark_cmd(
                    benchmark_binary=benchmark_binary,
                    transport=transport,
                    host=client["host"],
                    port=int(client["port"]),
                    num_shards=num_shards,
                    config_path=config_path,
                    mode=mode,
                    record_count=record_count,
                    runtime_seconds=runtime_seconds,
                    threads=threads,
                    load_threads=load_threads,
                    batch_size=load_batch_size,
                    value_size=value_size,
                    distribution=distribution,
                    zipfian_alpha=zipfian_alpha,
                    read_ratio=read_ratio,
                    report_mode=report_mode,
                    seed=seed,
                    phase="load",
                )
            else:
                load_cmd = build_benchmark_cmd(
                    benchmark_binary=benchmark_binary,
                    transport=transport,
                    host=client["host"],
                    port=int(client["port"]),
                    num_shards=num_shards,
                    config_path=config_path,
                    mode=mode,
                    record_count=record_count,
                    runtime_seconds=runtime_seconds,
                    threads=threads,
                    load_threads=load_threads,
                    batch_size=batch_size,
                    value_size=value_size,
                    distribution=distribution,
                    zipfian_alpha=zipfian_alpha,
                    read_ratio=read_ratio,
                    report_mode=report_mode,
                    seed=seed,
                    phase="load" if split_load_phase else "load_and_run",
                )
            if split_load_phase:
                try:
                    load = subprocess.run(
                        load_cmd,
                        cwd=str(repo_root),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        timeout=client_timeout_s if client_timeout_s > 0 else None,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise BenchmarkCaseError(
                        "load",
                        f"benchmark preload command timed out after {client_timeout_s} seconds\n"
                        f"cmd={' '.join(load_cmd)}\n"
                        f"server_log={server_log_path}",
                    ) from exc
                load_stdout_path = output_dir / f"{config_path.stem}_load.stdout"
                load_stderr_path = output_dir / f"{config_path.stem}_load.stderr"
                load_stdout_path.write_text(load.stdout, encoding="utf-8")
                load_stderr_path.write_text(load.stderr, encoding="utf-8")
                if load.returncode != 0:
                    raise BenchmarkCaseError(
                        "load",
                        "benchmark preload command failed\n"
                        f"cmd={' '.join(load_cmd)}\n"
                        f"stdout:\n{load.stdout}\n"
                        f"stderr:\n{load.stderr}\n"
                        f"server_log={server_log_path}",
                    )
            processes = [
                subprocess.Popen(
                    build_benchmark_cmd(
                        benchmark_binary=benchmark_binary,
                        transport=transport,
                        host=client["host"],
                        port=int(client["port"]),
                        num_shards=num_shards,
                        config_path=config_path,
                        mode=mode,
                        record_count=record_count,
                        runtime_seconds=runtime_seconds,
                        threads=threads,
                        load_threads=load_threads,
                        batch_size=batch_size,
                        value_size=value_size,
                        distribution=distribution,
                        zipfian_alpha=zipfian_alpha,
                        read_ratio=read_ratio,
                        report_mode=report_mode,
                        seed=seed + 1000003 * (process_id + 1),
                        phase="run" if split_load_phase else "load_and_run",
                    ),
                    cwd=str(repo_root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for process_id in range(client_processes)
            ]
            deadline = (
                time.time() + client_timeout_s if client_timeout_s > 0 else None
            )
            outputs: list[tuple[int, str, str]] = []
            for process_id, process in enumerate(processes):
                timeout = None
                if deadline is not None:
                    timeout = max(0.1, deadline - time.time())
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    for pending in processes:
                        if pending.poll() is None:
                            pending.kill()
                    for pending in processes:
                        pending.wait()
                    raise BenchmarkCaseError(
                        "run",
                        f"benchmark run command timed out after {client_timeout_s} seconds",
                    )
                if process.returncode != 0:
                    raise BenchmarkCaseError(
                        "run",
                        "benchmark command failed\n"
                        f"process_id={process_id}\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}\n"
                        f"server_log={server_log_path}",
                    )
                (output_dir / f"{config_path.stem}_client_{process_id}.stdout").write_text(
                    stdout, encoding="utf-8"
                )
                (output_dir / f"{config_path.stem}_client_{process_id}.stderr").write_text(
                    stderr, encoding="utf-8"
                )
                outputs.append((process_id, stdout, stderr))
            return outputs
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def print_summary_table(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        print("[summary] no parsed measure summary rows found")
        return

    header = [
        "index_type",
        "backend_alias",
        "transport",
        "mode",
        "phase",
        "threads",
        "client_processes",
        "process_id",
        "aggregate",
        "batch_size",
        "records",
        "M keys/s",
    ]
    table = [header]
    for row in rows:
        if str(row.get("phase")) != "run":
            continue
        if str(row.get("aggregate", "true")) == "false":
            continue
        throughput_keys_sec = row.get("throughput_keys_sec", "")
        status = str(row.get("status", "success"))
        throughput_label = status
        if throughput_keys_sec not in ("", None):
            throughput_label = f"{float(throughput_keys_sec) / 1e6:,.3f}"
        table.append(
            [
                str(row["index_type"]),
                str(row.get("backend_alias", "")),
                str(row["transport"]),
                str(row["mode"]),
                str(row["phase"]),
                str(row["threads"]),
                str(row.get("client_processes", 1)),
                str(row.get("process_id", "all")),
                str(row.get("aggregate", "true")),
                str(row["batch_size"]),
                str(row["records"]),
                throughput_label,
            ]
        )

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    separator = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    print("\n=== PS Backend Transport Benchmark Summary ===")
    print(render(table[0]))
    print(separator)
    for row in table[1:]:
        print(render(row))


def write_csv(rows: list[dict[str, str | int | float]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "failure_stage",
        "error_tail",
        "backend_alias",
        "backend_layer",
        "index_type",
        "value_store_type",
        "value_size",
        "capacity",
        "transport",
        "phase",
        "mode",
        "read_ratio",
        "threads",
        "client_processes",
        "process_id",
        "aggregate",
        "batch_size",
        "records",
        "distribution",
        "zipfian_alpha",
        "runtime_s",
        "batches",
        "key_ops",
        "throughput_batches_sec",
        "throughput_keys_sec",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_csv_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def parse_backend_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def resolve_case_base_port(transport: str, num_shards: int, args: argparse.Namespace) -> int:
    if transport == "LOCAL_SHM":
        return TRANSPORT_SPECS[transport].base_port
    if transport == "GRPC" and num_shards == 1:
        return TRANSPORT_SPECS[transport].base_port
    if transport == "GRPC":
        return args.grpc_base_port
    if transport == "BRPC":
        return args.brpc_base_port
    return TRANSPORT_SPECS[transport].base_port


def resolve_case_load_threads(backend_alias: str, args: argparse.Namespace) -> int:
    if args.load_threads > 0:
        return args.load_threads
    if resolve_backend_spec(backend_alias).alias == "hps_rocksdb":
        return args.hps_rocksdb_load_threads
    return args.load_threads


def resolve_local_shm_ready_queue_count(
    configured_ready_queue_count: int, benchmark_threads: int
) -> int:
    if configured_ready_queue_count > 0:
        return configured_ready_queue_count
    return max(1, benchmark_threads)


def resolve_failure_stage(exc: Exception) -> str:
    if isinstance(exc, BenchmarkCaseError):
        return exc.stage
    return "run_one_case"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/app/RecStore")
    parser.add_argument("--benchmark-binary", default="/app/RecStore/build/bin/ps_transport_benchmark")
    parser.add_argument("--server-bin-dir", default="/app/RecStore/build/bin")
    parser.add_argument("--transports", default=",".join(DEFAULT_TRANSPORTS))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument(
        "--index-types",
        default="",
        help="Deprecated: use --backends. Values are mapped to matching DRAM aliases.",
    )
    parser.add_argument("--mode", choices=["fetch", "insert", "mixed", "fetch_insert"], default="fetch")
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--hps-rocksdb-load-threads", type=int, default=1)
    parser.add_argument("--client-processes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--hps-rocksdb-load-batch-size",
        type=int,
        default=64,
        help="preload batch size for hps_rocksdb; <=0 uses --batch-size",
    )
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=1000000)
    parser.add_argument("--distribution", choices=["uniform", "zipfian"], default="uniform")
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--grpc-base-port", type=int, default=15000)
    parser.add_argument("--brpc-base-port", type=int, default=25000)
    parser.add_argument("--max-keys-per-request", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--dram-capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--ssd-capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--combined-load-and-run", action="store_true", default=False)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--client-timeout-s", type=int, default=120)
    parser.add_argument("--report-mode", choices=["summary", "per_round", "both"], default="summary")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--keep-runtime-dir", action="store_true", default=False)
    parser.add_argument("--local-shm-region", default="recstore_local_ps")
    parser.add_argument("--local-shm-slot-count", type=int, default=64)
    parser.add_argument(
        "--local-shm-ready-queue-count",
        type=int,
        default=0,
        help="0 means auto: use --threads",
    )
    parser.add_argument("--local-shm-ready-queue-burst-limit", type=int, default=8)
    parser.add_argument("--local-shm-slot-buffer-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--local-shm-client-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--local-shm-thread-ready-queue-sharding",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    benchmark_binary = Path(args.benchmark_binary).resolve()
    server_bin_dir = Path(args.server_bin_dir).resolve()
    if not benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {benchmark_binary}")
    if args.client_processes <= 0:
        raise ValueError("--client-processes must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    transports = parse_csv_list(args.transports)
    backend_aliases = (
        parse_backend_list(args.index_types)
        if args.index_types
        else parse_backend_list(args.backends)
    )
    for transport in transports:
        if transport not in TRANSPORT_SPECS:
            raise ValueError(f"unsupported transport: {transport}")
    for backend_alias in backend_aliases:
        resolve_backend_spec(backend_alias)

    if args.output_dir:
        runtime_dir = Path(args.output_dir).resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
    else:
        runtime_dir = Path(tempfile.mkdtemp(prefix="recstore_ps_dram_bench_"))

    rows = []
    try:
        for backend_alias in backend_aliases:
            backend = resolve_backend_spec(backend_alias)
            for transport in transports:
                spec = TRANSPORT_SPECS[transport]
                server_binary = server_bin_dir / spec.server_binary
                if not server_binary.exists():
                    raise FileNotFoundError(f"server binary not found: {server_binary}")
                case_num_shards = 1 if transport == "LOCAL_SHM" else args.num_shards
                case_local_shm_ready_queue_count = (
                    resolve_local_shm_ready_queue_count(
                        args.local_shm_ready_queue_count, args.threads
                    )
                    if transport == "LOCAL_SHM"
                    else args.local_shm_ready_queue_count
                )
                base_port = resolve_case_base_port(
                    transport, case_num_shards, args
                )
                config = build_runtime_config(
                    transport=transport,
                    backend_alias=backend.alias,
                    runtime_dir=runtime_dir,
                    num_shards=case_num_shards,
                    base_port=base_port,
                    capacity=args.capacity,
                    value_size=args.value_size,
                    max_keys_per_request=args.max_keys_per_request,
                    num_threads=args.num_threads,
                    dram_allocator=args.dram_allocator,
                    local_shm_region=args.local_shm_region,
                    local_shm_slot_count=args.local_shm_slot_count,
                    local_shm_ready_queue_count=case_local_shm_ready_queue_count,
                    local_shm_ready_queue_burst_limit=args.local_shm_ready_queue_burst_limit,
                    local_shm_slot_buffer_bytes=args.local_shm_slot_buffer_bytes,
                    local_shm_client_timeout_ms=args.local_shm_client_timeout_ms,
                    local_shm_thread_ready_queue_sharding=args.local_shm_thread_ready_queue_sharding,
                    dram_capacity_multiplier=args.dram_capacity_multiplier,
                    ssd_capacity_multiplier=args.ssd_capacity_multiplier,
                    ssd_io_backend=args.ssd_io_backend,
                    ssd_queue_depth=args.ssd_queue_depth,
                )
                case_slug = f"{backend.alias}_{transport.lower()}"
                config_path = runtime_dir / f"{case_slug}.json"
                log_path = runtime_dir / f"{case_slug}_server.log"
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                prepare_runtime_paths(config)
                print(f"[case] backend={backend.alias} transport={transport} config={config_path}")
                load_batch_size = (
                    args.hps_rocksdb_load_batch_size
                    if backend.alias == "hps_rocksdb"
                    and args.hps_rocksdb_load_batch_size > 0
                    else args.batch_size
                )
                try:
                    outputs = run_one_case(
                        repo_root=repo_root,
                        server_binary=server_binary,
                        benchmark_binary=benchmark_binary,
                        config_path=config_path,
                        server_log_path=log_path,
                        transport=transport,
                        num_shards=case_num_shards,
                        mode=args.mode,
                        record_count=args.capacity,
                        runtime_seconds=args.runtime_seconds,
                        threads=args.threads,
                        load_threads=resolve_case_load_threads(backend.alias, args),
                        batch_size=args.batch_size,
                        load_batch_size=load_batch_size,
                        value_size=args.value_size,
                        distribution=args.distribution,
                        zipfian_alpha=args.zipfian_alpha,
                        read_ratio=args.read_ratio,
                        report_mode=args.report_mode,
                        startup_delay=args.startup_delay,
                        client_timeout_s=args.client_timeout_s,
                        client_processes=args.client_processes,
                        seed=args.seed,
                        output_dir=runtime_dir,
                        split_load_phase=not args.combined_load_and_run,
                    )
                except Exception as exc:
                    failure_stage = resolve_failure_stage(exc)
                    print(f"[case-failed] backend={backend.alias} transport={transport}: {exc}")
                    rows.append(
                        build_failure_row(
                            backend_alias=backend.alias,
                            transport=transport,
                            mode=args.mode,
                            read_ratio=args.read_ratio,
                            threads=args.threads,
                            client_processes=args.client_processes,
                            batch_size=args.batch_size,
                            value_size=args.value_size,
                            capacity=args.capacity,
                            distribution=args.distribution,
                            zipfian_alpha=args.zipfian_alpha,
                            failure_stage=failure_stage,
                            error_tail=str(exc),
                        )
                    )
                    continue
                process_outputs = []
                for process_id, stdout, stderr in outputs:
                    print(f"[client] process_id={process_id}")
                    print(stdout, end="" if stdout.endswith("\n") else "\n")
                    if stderr:
                        print(stderr, end="" if stderr.endswith("\n") else "\n")
                    process_outputs.append((process_id, stdout))
                rows.extend(
                    collect_case_rows(
                        process_outputs,
                        backend_alias=backend.alias,
                        value_size=args.value_size,
                        capacity=args.capacity,
                        read_ratio=args.read_ratio,
                        client_processes=args.client_processes,
                    )
                )

        print_summary_table(rows)
        csv_path = Path(args.csv_path).resolve() if args.csv_path else runtime_dir / "ps_dram_transport_benchmark.csv"
        write_csv(rows, csv_path)
        print(f"[output] csv={csv_path}")
        print(f"[output] runtime_dir={runtime_dir}")
    finally:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
