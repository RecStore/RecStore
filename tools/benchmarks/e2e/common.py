from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path("/nas/home/shq/docker/rs_demo/bench_e2e")
DEFAULT_DAY0 = Path("/nas/home/shq/RecStore_/model_zoo/torchrec_dlrm/partial_data/day_0.bak")
SPARSE_FEATURES_PER_SAMPLE = 26

@dataclass(frozen=True)
class E2ELane:
    slug: str
    label: str
    backend: str
    ps_type: str = ""
    recstore_index_type: str = ""
    ps_kv_backend: str = ""
    allocator: str = "R2ShmMalloc"
    nproc_per_node: int = 1
    torchrec_memory_mode: str = "hbm"
    prefetch_depth: int = 0
    enable_single_node_fast_path: bool = False
    single_node_ps_backend: str = "local_shm"
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentPlan:
    profile: str
    output_root: Path
    data_rows: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    embedding_dims: tuple[int, ...]
    steps: int
    warmup_steps: int
    repeat: int
    lanes: tuple[E2ELane, ...]


@dataclass(frozen=True)
class PlanOverrides:
    data_rows: tuple[int, ...] = ()
    batch_sizes: tuple[int, ...] = ()
    num_embeddings: tuple[int, ...] = ()
    embedding_dims: tuple[int, ...] = ()
    steps: int | None = None
    warmup_steps: int | None = None
    repeat: int | None = None
    only_lanes: tuple[str, ...] = ()
    include_ablation_lanes: bool = False


@dataclass(frozen=True)
class ExecutionContext:
    remote_train_host: str = ""
    remote_ssh_port: int = 22
    remote_repo_root: Path = ROOT
    python_bin: str = sys.executable
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    external_recstore_runtime_dir: Path | None = None
    no_start_recstore_server: bool = False
    server_host: str = ""
    server_port0: int | None = None
    server_port1: int | None = None


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    title: str
    xlabel: str
    ylabel: str
    series: dict[str, list[tuple[float, float]]]
    xmode_log: bool = False
    description: str = ""


@dataclass(frozen=True)
class FigureSection:
    title: str
    purpose: str
    figures: tuple[FigureSpec, ...]


def _has_rdma() -> bool:
    infiniband = Path("/dev/infiniband")
    return infiniband.exists() and any(infiniband.glob("uverbs*"))


def _gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _run_text(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    except Exception:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _collect_environment_metadata() -> dict[str, Any]:
    env: dict[str, Any] = {
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "hostname": _run_text(["hostname"]),
        "kernel": _run_text(["uname", "-srmo"]),
    }
    nvidia = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        env["nvidia_smi_gpu"] = nvidia
    try:
        import torch

        env["torch_version"] = getattr(torch, "__version__", "")
        env["torch_cuda"] = getattr(torch.version, "cuda", "")
        env["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        env["torch_version"] = ""
        env["torch_cuda"] = ""
        env["cudnn_version"] = ""
    try:
        import torchrec

        env["torchrec_version"] = getattr(torchrec, "__version__", "installed")
    except Exception:
        env["torchrec_version"] = ""
    return env


def _dense_arch_for_embedding_dim(embedding_dim: int) -> str:
    if int(embedding_dim) >= 128:
        return "512,256,128"
    return f"512,256,{int(embedding_dim)}"


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(value <= 0 for value in values):
        raise ValueError(f"all values must be positive: {raw}")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in _read_csv(path)]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _cached_text_file(path_text: str, max_bytes: int = 262_144) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except OSError:
        return ""


def _status_reason(row: dict[str, Any]) -> str:
    explicit = str(row.get("skip_reason") or row.get("message") or "").strip()
    if explicit:
        return explicit
    if row.get("status") == "failed":
        log_text = _cached_text_file(str(row.get("log_path") or ""))
        lower_log = log_text.lower()
        if "out of memory" in lower_log or "oom" in lower_log:
            return "OOM during server/client startup; see per-run log"
        if "wait_server timeout" in log_text or "RdmaControlPlaneClient::WaitServer" in log_text:
            return "RDMA server did not publish ready before client init; see per-run log"
        if "SIGSEGV" in log_text or "SIGABRT" in log_text:
            return "native process crashed; see per-run log"
        return "benchmark failed; see per-run log"
    return str(row.get("log_path") or row.get("status") or "")
