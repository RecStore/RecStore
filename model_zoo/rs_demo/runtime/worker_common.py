from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.pytorch.recstore.benchmark.report import write_stage_csv


@dataclass(frozen=True)
class WorkerContext:
    rank: int
    local_rank: int
    world_size: int
    output_dir: Path


def is_worker_process(backend: str) -> bool:
    env_prefix = f"RS_DEMO_{backend.upper().replace('-', '_')}_"
    return os.environ.get(f"{env_prefix}WORKER") == "1"


def read_worker_context(backend: str, default_world_size: int) -> WorkerContext | None:
    env_prefix = f"RS_DEMO_{backend.upper().replace('-', '_')}_"
    if not is_worker_process(backend):
        return None

    output_dir_value = os.environ.get(f"{env_prefix}WORKER_DIR")
    if not output_dir_value:
        raise RuntimeError(f"{env_prefix}WORKER_DIR is required for worker runs")
    return WorkerContext(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", str(default_world_size))),
        output_dir=Path(output_dir_value),
    )


def build_worker_env(backend: str, output_dir: Path) -> dict[str, str]:
    env_prefix = f"RS_DEMO_{backend.upper().replace('-', '_')}_"
    env = os.environ.copy()
    env[f"{env_prefix}WORKER"] = "1"
    env[f"{env_prefix}WORKER_DIR"] = str(output_dir)
    socket_ifname = pick_socket_ifname()
    if socket_ifname:
        env.setdefault("NCCL_SOCKET_IFNAME", socket_ifname)
        env.setdefault("GLOO_SOCKET_IFNAME", socket_ifname)
    env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    return env


def bool_int(flag: bool) -> int:
    return 1 if flag else 0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_stage_csv(path, rows)


def pick_socket_ifname() -> str | None:
    preferred = ("eno1", "eno8303")
    try:
        available = set(os.listdir("/sys/class/net"))
    except OSError:
        return None
    for name in preferred:
        if name in available:
            return name
    return None


def barrier_for_step_alignment(dist, device, local_rank: int, use_dist: bool) -> None:
    if not use_dist:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def merge_rank_outputs(paths: list[Path], out_path: Path) -> list[dict[str, Any]]:
    """Merge per-rank CSV outputs into a single sorted file."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        for row in load_rows(path):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                    continue
                if key in {"backend", "collective_mode"}:
                    normalized[key] = value
                    continue
                try:
                    if "." in value:
                        normalized[key] = float(value)
                    else:
                        normalized[key] = int(value)
                except (TypeError, ValueError):
                    normalized[key] = value
            merged.append(normalized)
    if any(str(row.get("torchrec_dist_mode", "")) == "fair_remote" for row in merged):
        merged = [row for row in merged if int(row.get("torchrec_is_trainer", 1)) == 1]
    merged.sort(key=lambda row: (int(row.get("rank", 0)), int(row.get("step", 0))))
    write_rows(out_path, merged)
    return merged


def parse_nccl_transport_log(log_path: Path | None) -> str:
    """Extract NCCL transport info (NET/IB HCA, interface) from a log file."""
    if log_path is None or not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    import re
    # Look for NET/IB lines
    for line in text.splitlines():
        if "NET/IB" in line or "NCCL INFO" in line and ("mlx" in line or "ens" in line or "eth" in line):
            return line.strip()
    return ""
