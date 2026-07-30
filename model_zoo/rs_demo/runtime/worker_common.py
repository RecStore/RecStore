from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from python.pytorch.recstore.benchmark.report import write_stage_csv


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


def parse_nccl_transport_log(log_path: Path | None) -> str:
    if log_path is None or not log_path.exists():
        return "unknown"
    match = re.search(
        r"NCCL INFO NET/(IB|Socket)\s*:\s*Using",
        log_path.read_text(errors="replace"),
    )
    if not match:
        return "unknown"
    return "RDMA" if match.group(1) == "IB" else "TCP"


def barrier_for_step_alignment(dist, device, local_rank: int, use_dist: bool) -> None:
    if not use_dist:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
