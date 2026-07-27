from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

_LOCAL_SHM_READY_DELAY_S = 0.5
_MIN_SSD_CAPACITY_BYTES = 256 * 1024 * 1024


def _resolve_tiered_ssd_path(output_root: str, run_id: str, path_suffix: str) -> str:
    del output_root
    return str(
        Path("/tmp")
        / "rs_demo_kv"
        / run_id
        / f"kv_{path_suffix}"
        / "tiered_ssd.db"
    )


def _resolve_hps_data_path(output_root: str, run_id: str, path_suffix: str) -> str:
    del output_root
    return str(
        Path("/tmp")
        / "rs_demo_hps"
        / run_id
        / f"kv_{path_suffix}"
        / "hps"
    )


def resolve_kv_data_path(
    output_root: str,
    run_id: str,
    path_suffix: str,
    allocator: str,
) -> str:
    del output_root, allocator
    # DRAM_VALUE_STORE rejects filesystem-backed paths outside /dev/shm.
    # Keep KV backing files on tmpfs while logs, configs, and reports stay under output_root.
    return str(Path("/dev/shm") / "rs_demo_kv" / run_id / f"kv_{path_suffix}")


def wait_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            time.sleep(0.2)
        finally:
            sock.close()
    return False


def is_port_bindable(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pick_free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def choose_available_ports(host: str, preferred0: int, preferred1: int) -> tuple[int, int]:
    if preferred0 != preferred1 and is_port_bindable(host, preferred0) and is_port_bindable(host, preferred1):
        return preferred0, preferred1
    p0 = pick_free_port(host)
    p1 = pick_free_port(host)
    while p1 == p0:
        p1 = pick_free_port(host)
    return p0, p1


def normalize_allocator_type(allocator: str) -> str:
    allocator_upper = allocator.upper()
    if allocator_upper in {"PERSISTLOOPSHMMALLOC", "PERSIST_LOOP_SHM_MALLOC", "PERSIST_LOOP_SLAB"}:
        return "PERSIST_LOOP_SLAB"
    if allocator_upper in {"R2SHMMALLOC", "R2_SHM_MALLOC", "R2_SLAB"}:
        return "R2_SLAB"
    return allocator


def _build_ssd_allocator(capacity_bytes: int) -> dict:
    return {
        "type": "SSD_SLAB",
        "capacity_bytes": max(int(capacity_bytes), _MIN_SSD_CAPACITY_BYTES),
        "min_block_size": 128,
        "max_block_size": 4096,
        "io": {
            "type": "IOURING",
            "queue_depth": 512,
            "base_offset_bytes": 4096,
        },
    }


def _prepare_base_kv_paths(base_kv: dict) -> None:
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
        if not path:
            continue
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)


def _build_recstore_base_kv_config(
    *,
    ps_kv_backend: str,
    kv_data_path: str,
    tiered_ssd_path: str,
    capacity: int,
    value_size_hint: int,
    allocator: str,
    index_type: str,
    tiered_dram_capacity_multiplier: float,
) -> dict:
    dram_capacity = capacity * value_size_hint * 2
    value: dict = {
        "type": "DRAM_VALUE_STORE",
        "path": f"{kv_data_path}/value",
        "default_value_size_hint": value_size_hint,
        "dram_allocator": {
            "type": normalize_allocator_type(allocator),
            "capacity_bytes": dram_capacity,
        },
    }
    if ps_kv_backend == "recstore_tiered":
        tiered_dram_capacity = int(
            capacity * value_size_hint * float(tiered_dram_capacity_multiplier)
        )
        ssd_capacity = max(
            capacity * max(value_size_hint + 8, 128) * 2,
            _MIN_SSD_CAPACITY_BYTES,
        )
        value = {
            "type": "TIERED_VALUE_STORE",
            "default_value_size_hint": value_size_hint,
            "dram_allocator": {
                "type": normalize_allocator_type(allocator),
                "capacity_bytes": tiered_dram_capacity,
                "path": f"{kv_data_path}/dram",
            },
            "ssd_allocator": _build_ssd_allocator(ssd_capacity),
            "tiering": {"cache_policy": "LRU"},
        }
        value["ssd_allocator"]["path"] = tiered_ssd_path
    return {
        "capacity": capacity,
        "index": {"type": index_type},
        "value": value,
    }


def _build_hps_base_kv_config(
    *,
    ps_kv_backend: str,
    hps_data_path: str,
    capacity: int,
    value_size_hint: int,
) -> dict:
    if value_size_hint <= 0:
        raise ValueError("HPS BaseKV backend requires a positive value_size_bytes")
    external_engine_type = {
        "hps_hash_map": "KVEngineHPSHashMap",
        "hps_rocksdb": "KVEngineHPSRocksDB",
    }[ps_kv_backend]
    cfg = {
        "external_engine_type": external_engine_type,
        "capacity": capacity,
        "path": hps_data_path,
        "value_size": value_size_hint,
        "num_threads": 1 if ps_kv_backend == "hps_rocksdb" else 0,
        "max_batch_size": 65536,
    }
    if ps_kv_backend == "hps_rocksdb":
        cfg["rocksdb_path"] = hps_data_path
    return cfg


def build_base_kv_config_for_backend(
    *,
    ps_kv_backend: str,
    output_root: str,
    run_id: str,
    path_suffix: str,
    allocator: str,
    capacity: int,
    value_size_hint: int,
    index_type: str,
    tiered_dram_capacity_multiplier: float = 2.0,
) -> dict:
    backend = ps_kv_backend.strip().lower()
    kv_data_path = resolve_kv_data_path(
        output_root=output_root,
        run_id=run_id,
        path_suffix=path_suffix,
        allocator=allocator,
    )
    if backend in {"recstore_dram", "recstore_tiered"}:
        return _build_recstore_base_kv_config(
            ps_kv_backend=backend,
            kv_data_path=kv_data_path,
            tiered_ssd_path=_resolve_tiered_ssd_path(output_root, run_id, path_suffix),
            capacity=capacity,
            value_size_hint=value_size_hint,
            allocator=allocator,
            index_type=index_type,
            tiered_dram_capacity_multiplier=tiered_dram_capacity_multiplier,
        )
    if backend in {"hps_hash_map", "hps_rocksdb"}:
        return _build_hps_base_kv_config(
            ps_kv_backend=backend,
            hps_data_path=_resolve_hps_data_path(output_root, run_id, path_suffix),
            capacity=capacity,
            value_size_hint=value_size_hint,
        )
    raise ValueError(f"unsupported ps_kv_backend: {ps_kv_backend}")


def wait_server_ready(
    proc: subprocess.Popen,
    host: str,
    port0: int,
    port1: int,
    timeout_s: float,
    ps_type: str = "BRPC",
) -> bool:
    if str(ps_type).upper() == "LOCAL_SHM":
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            time.sleep(_LOCAL_SHM_READY_DELAY_S)
            return proc.poll() is None
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        ok0 = wait_port(host, port0, timeout_s=0.5)
        ok1 = wait_port(host, port1, timeout_s=0.5)
        if ok0 and ok1:
            return True
    return False


def build_runtime_config(
    base_cfg: dict,
    host: str,
    port0: int,
    port1: int,
    allocator: str,
    path_suffix: str,
    ps_type: str,
    output_root: str,
    run_id: str,
    kv_capacity: int | None = None,
    value_size_bytes: int | None = None,
    index_type: str = "DRAM_EXTENDIBLE_HASH",
    ps_kv_backend: str = "recstore_dram",
    tiered_dram_capacity_multiplier: float = 2.0,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("cache_ps", {})
    cfg["cache_ps"]["ps_type"] = ps_type.upper()

    cfg.setdefault("client", {})
    cfg["client"]["host"] = host
    cfg["client"]["port"] = port0
    cfg["client"]["shard"] = 0

    cfg.setdefault("distributed_client", {})
    if cfg["cache_ps"]["ps_type"] == "LOCAL_SHM":
        servers = [{"host": host, "port": port0, "shard": 0}]
    else:
        servers = [
            {"host": host, "port": port0, "shard": 0},
            {"host": host, "port": port1, "shard": 1},
        ]
    cfg["distributed_client"]["num_shards"] = len(servers)
    cfg["distributed_client"]["servers"] = list(servers)

    cfg["cache_ps"]["num_shards"] = len(servers)
    cfg["cache_ps"]["servers"] = list(servers)
    if cfg["cache_ps"]["ps_type"] == "LOCAL_SHM":
        cfg["local_shm"] = {
            "region_name": f"recstore_rs_demo_{run_id}_{path_suffix}",
            "slot_count": 256,
            "ready_queue_count": 2,
            "ready_queue_burst_limit": 16,
            "slot_buffer_bytes": 8 * 1024 * 1024,
            "client_timeout_ms": 30000,
        }

    previous_base_kv = cfg["cache_ps"].get("base_kv_config", {})
    if not isinstance(previous_base_kv, dict):
        previous_base_kv = {}
    capacity = int(kv_capacity or previous_base_kv.get("capacity", 1_000_000))
    previous_value = previous_base_kv.get("value", {})
    if not isinstance(previous_value, dict):
        previous_value = {}
    value_size_hint = int(
        value_size_bytes
        or previous_base_kv.get("value_size", 0)
        or previous_value.get("default_value_size_hint", 512)
    )
    base_kv = build_base_kv_config_for_backend(
        ps_kv_backend=ps_kv_backend,
        output_root=output_root,
        run_id=run_id,
        path_suffix=path_suffix,
        allocator=allocator,
        capacity=capacity,
        value_size_hint=value_size_hint,
        index_type=index_type,
        tiered_dram_capacity_multiplier=tiered_dram_capacity_multiplier,
    )
    _prepare_base_kv_paths(base_kv)
    cfg["cache_ps"]["base_kv_config"] = base_kv
    return cfg


def resolve_default_ports(base_cfg: dict) -> tuple[int, int]:
    distributed_servers = (
        base_cfg.get("distributed_client", {}).get("servers", [])
        or base_cfg.get("cache_ps", {}).get("servers", [])
    )
    if len(distributed_servers) >= 2:
        return int(distributed_servers[0]["port"]), int(distributed_servers[1]["port"])
    if "client" in base_cfg and "port" in base_cfg["client"]:
        p0 = int(base_cfg["client"]["port"])
        return p0, p0 + 1
    return 15000, 15001


def start_server(repo_root: Path, cfg_path: Path, log_path: Path) -> subprocess.Popen:
    with cfg_path.open("r", encoding="utf-8") as f:
        runtime_cfg = json.load(f)
    ps_type = str(runtime_cfg.get("cache_ps", {}).get("ps_type", "BRPC")).upper()
    if ps_type == "LOCAL_SHM":
        server_bin = repo_root / "build/bin/local_shm_ps_server"
    else:
        server_bin = repo_root / "build/bin/ps_server"
    if not server_bin.exists():
        raise FileNotFoundError(f"ps_server not found: {server_bin}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(server_bin), "--config_path", str(cfg_path)],
        cwd=str(repo_root),
        stdout=fout,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    proc._rs_demo_log_file = fout  # type: ignore[attr-defined]
    return proc


def start_rdma_server_cluster(
    repo_root: Path,
    cfg_path: Path,
    log_path: Path,
    *,
    num_clients: int = 1,
    thread_num: int = 1,
    value_size: int = 512,
    max_kv_num_per_request: int = 512,
    timeout_s: float = 60.0,
    rdma_control_plane_host: str = "127.0.0.1",
):
    scripts_dir = repo_root / "src/test/scripts"
    import_path = str(scripts_dir)
    import sys

    if import_path not in sys.path:
        sys.path.insert(0, import_path)
    from petps_cluster_runner import PetPSClusterRunner  # type: ignore

    with cfg_path.open("r", encoding="utf-8") as f:
        runtime_cfg = json.load(f)
    distributed_cfg = runtime_cfg.get("distributed_client", {})
    cache_ps_cfg = runtime_cfg.get("cache_ps", {})
    server_count = int(
        distributed_cfg.get("num_shards")
        or cache_ps_cfg.get("num_shards")
        or len(distributed_cfg.get("servers", []))
        or len(cache_ps_cfg.get("servers", []))
        or 1
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner = PetPSClusterRunner(
        server_path=str(repo_root / "build/bin/petps_server"),
        config_path=str(cfg_path),
        num_servers=server_count,
        num_clients=max(1, int(num_clients)),
        thread_num=int(thread_num),
        value_size=int(value_size),
        max_kv_num_per_request=int(max_kv_num_per_request),
        timeout=max(1, int(timeout_s)),
        log_dir=str(log_path.parent),
        verbose=True,
        show_status_logs=True,
        show_control_plane_logs=True,
        rdma_namespace=f"rs-demo-{cfg_path.parent.name}",
        rdma_control_plane_host=str(rdma_control_plane_host),
        rdma_wait_timeout_ms=20000,
        rdma_qps_per_client_per_shard=32,
        rdma_slots_per_qp=1,
        rdma_server_coroutines_per_thread=1,
        rdma_server_get_workers=0,
        rdma_profile_interval_ms=1000,
    )
    with log_path.open("w", encoding="utf-8") as f:
        f.write(
            "petps_server cluster\n"
            f"config_path={cfg_path}\n"
            f"server_count={server_count}\n"
            f"value_size={value_size}\n"
            f"max_kv_num_per_request={max_kv_num_per_request}\n"
            f"rdma_namespace={runner.rdma_namespace}\n"
            f"rdma_control_plane={runner.rdma_control_plane_host}:{runner.rdma_control_plane_port}\n"
        )
    runner._rs_demo_log_path = log_path  # type: ignore[attr-defined]
    runner.start()
    return runner


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    log_f = getattr(proc, "_rs_demo_log_file", None)
    if log_f is not None:
        log_f.close()


def stop_rdma_server_cluster(runner) -> None:
    if runner is None:
        return
    log_path = getattr(runner, "_rs_demo_log_path", None)
    process_logs = {
        global_id: list(lines)
        for global_id, lines in getattr(runner, "process_logs", {}).items()
    }
    try:
        runner.stop()
    finally:
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write("\npetps_server captured output\n")
                if not process_logs:
                    f.write("(no captured petps_server output)\n")
                for global_id, lines in sorted(process_logs.items()):
                    f.write(f"\n[petps_server:{global_id}]\n")
                    for line in lines:
                        f.write(f"{line}\n")


def make_runtime_dir(
    base_cfg: dict,
    host: str,
    port0: int,
    port1: int,
    allocator: str,
    output_root: str,
    run_id: str,
    ps_type: str = "BRPC",
    kv_capacity: int | None = None,
    value_size_bytes: int | None = None,
    index_type: str = "DRAM_EXTENDIBLE_HASH",
    ps_kv_backend: str = "recstore_dram",
    tiered_dram_capacity_multiplier: float = 2.0,
) -> tuple[Path, Path]:
    unique_tag = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    output_root_path = Path(output_root).resolve()
    runtime_cfg = build_runtime_config(
        base_cfg=base_cfg,
        host=host,
        port0=port0,
        port1=port1,
        allocator=allocator,
        path_suffix=unique_tag,
        ps_type=ps_type,
        output_root=str(output_root_path),
        run_id=run_id,
        kv_capacity=kv_capacity,
        value_size_bytes=value_size_bytes,
        index_type=index_type,
        ps_kv_backend=ps_kv_backend,
        tiered_dram_capacity_multiplier=tiered_dram_capacity_multiplier,
    )
    runtime_dir = output_root_path / "runtime" / run_id / unique_tag
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg_path = runtime_dir / "recstore_config.json"
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        json.dump(runtime_cfg, f, indent=2)
    return runtime_dir, runtime_cfg_path
