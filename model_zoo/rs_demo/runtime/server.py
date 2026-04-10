from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path


def resolve_kv_data_path(
    output_root: str,
    run_id: str,
    path_suffix: str,
    allocator: str,
) -> str:
    allocator_upper = allocator.upper()
    if allocator_upper == "R2SHMMALLOC":
        # R2ShmMalloc can hang during ps_server init on the NAS mount.
        # Keep its backing path on the local filesystem and move logs/configs to NAS.
        return str(Path("/tmp") / "rs_demo_kv" / run_id / f"kv_{path_suffix}")
    return str(Path(output_root) / "runtime" / run_id / f"kv_{path_suffix}")


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


def wait_server_ready(
    proc: subprocess.Popen,
    host: str,
    port0: int,
    port1: int,
    timeout_s: float,
) -> bool:
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
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("cache_ps", {})
    cfg["cache_ps"]["ps_type"] = ps_type.upper()

    cfg.setdefault("client", {})
    cfg["client"]["host"] = host
    cfg["client"]["port"] = port0
    cfg["client"]["shard"] = 0

    cfg.setdefault("distributed_client", {})
    cfg["distributed_client"]["num_shards"] = 2
    cfg["distributed_client"]["servers"] = [
        {"host": host, "port": port0, "shard": 0},
        {"host": host, "port": port1, "shard": 1},
    ]

    cfg["cache_ps"]["num_shards"] = 2
    cfg["cache_ps"]["servers"] = [
        {"host": host, "port": port0, "shard": 0},
        {"host": host, "port": port1, "shard": 1},
    ]

    base_kv = cfg["cache_ps"].setdefault("base_kv_config", {})
    base_kv["value_memory_management"] = allocator
    base_kv["path"] = resolve_kv_data_path(
        output_root=output_root,
        run_id=run_id,
        path_suffix=path_suffix,
        allocator=allocator,
    )
    base_kv["index_type"] = "DRAM"
    if kv_capacity is not None:
        base_kv["capacity"] = int(kv_capacity)
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
) -> tuple[Path, Path]:
    unique_tag = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    runtime_cfg = build_runtime_config(
        base_cfg=base_cfg,
        host=host,
        port0=port0,
        port1=port1,
        allocator=allocator,
        path_suffix=unique_tag,
        ps_type=ps_type,
        output_root=output_root,
        run_id=run_id,
        kv_capacity=kv_capacity,
    )
    runtime_dir = Path(output_root) / "runtime" / run_id / unique_tag
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg_path = runtime_dir / "recstore_config.json"
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        json.dump(runtime_cfg, f, indent=2)
    return runtime_dir, runtime_cfg_path
