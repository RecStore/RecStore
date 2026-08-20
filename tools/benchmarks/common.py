from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/tmp/rs_demo")
DEFAULT_REMOTE_HOST = "shq@10.0.2.68"
DEFAULT_REMOTE_REPO = "/tmp/RecStore"
DEFAULT_REMOTE_CONTAINER = "recstore"
DEFAULT_REMOTE_RUNTIME_ROOT = DEFAULT_OUTPUT_ROOT / "runtime"
DEFAULT_WARMUP_STEPS = 2
DEFAULT_STEPS = 12
DEFAULT_NUM_EMBEDDINGS = 10000
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_SEED = 20260330
DEFAULT_MASTER_ADDR = "10.0.2.196"
DEFAULT_REMOTE_SERVER_HOST = "10.0.2.68"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    column: str
    transform: str = "mean"


@dataclass(frozen=True)
class LaneSpec:
    name: str
    slug: str
    backend: str
    steps: int
    warmup_steps: int
    batch_size: int
    num_embeddings: int
    embedding_dim: int
    nnodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    master_addr: str = DEFAULT_MASTER_ADDR
    master_port: int = 29500
    rdzv_id: str = ""
    no_start_server: bool = False
    server_host: str | None = None
    server_port0: int | None = None
    server_port1: int | None = None
    recstore_runtime_dir: str | None = None
    torchrec_dist_mode: str | None = None
    ps_type: str | None = None
    allocator: str | None = None
    remote_worker: bool = False
    needs_remote_ps: bool = False
    needs_remote_sync: bool = False
    remote_container: str = DEFAULT_REMOTE_CONTAINER
    metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    lanes: tuple[LaneSpec, ...]


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
