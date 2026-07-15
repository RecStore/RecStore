from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..common import ROOT

DEFAULT_OUTPUT_DIR = Path("results") / time.strftime("brpc_e2e_%m%d%H%M")


@dataclass(frozen=True)
class ClientSpec:
    ssh_host: str = "local"
    ssh_port: int = 22
    repo_root: Path = ROOT
    ip: str = "127.0.0.1"
    gpu_id: int = 0
    node_rank: int = 0
    nproc_per_node: int = 1


@dataclass(frozen=True)
class ServerSpec:
    ssh_host: str = "local"
    ssh_port: int = 22
    repo_root: Path = ROOT
    ip: str = "127.0.0.1"
    port: int = 15000
    shard_id: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    clients: tuple[ClientSpec, ...]
    servers: tuple[ServerSpec, ...]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    runtime_dir: Path | None = None
    dataset_path: Path = Path("model_zoo/torchrec_dlrm/processed_day_0_data")
    model: str = "dlrm"
    batch_size: int = 1024
    embedding_dim: int = 128
    num_embeddings: int = 200000
    init_rows: int = 50000
    steps: int = 80
    warmup_steps: int = 5
    repeat: int = 3
    read_mode: str = "prefetch"
    prefetch_depth: int = 0
    index_type: str = "DRAM_PET_HASH"
    torchrec_baselines: tuple[str, ...] = ("hbm",)
    master_port: int = 29500
    python_bin: str = sys.executable
    skip_build: bool = False
    skip_tests: bool = False

    @property
    def resolved_runtime_dir(self) -> Path:
        return self.runtime_dir or (self.output_dir / "runtime")


def _parse_key_values(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value item in spec: {part!r}")
        key, value = part.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def _ssh_from_values(values: dict[str, str]) -> tuple[str, int]:
    ssh_host = values.get("ssh", values.get("ssh_host", "local")).strip()
    ssh_port_raw = values.get("ssh_port", values.get("ssh-port"))
    ssh_port = 22 if ssh_port_raw is None else int(ssh_port_raw)
    return ssh_host, ssh_port


def parse_client_spec(raw: str) -> ClientSpec:
    values = _parse_key_values(raw)
    ssh_host, ssh_port = _ssh_from_values(values)
    return ClientSpec(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        repo_root=Path(values.get("repo", values.get("repo_root", str(ROOT)))),
        ip=values.get("ip", "127.0.0.1"),
        gpu_id=int(values.get("gpu", values.get("gpu_id", "0"))),
        node_rank=int(values.get("node_rank", values.get("rank", "0"))),
        nproc_per_node=int(values.get("nproc", values.get("nproc_per_node", "1"))),
    )


def parse_server_spec(raw: str) -> ServerSpec:
    values = _parse_key_values(raw)
    ssh_host, ssh_port = _ssh_from_values(values)
    return ServerSpec(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        repo_root=Path(values.get("repo", values.get("repo_root", str(ROOT)))),
        ip=values.get("ip", "127.0.0.1"),
        port=int(values.get("port", "15000")),
        shard_id=int(values.get("shard", values.get("shard_id", "0"))),
    )


def infer_client_deployment(clients: tuple[ClientSpec, ...]) -> str:
    ips = {client.ip for client in clients}
    ranks = {client.node_rank for client in clients}
    return "distributed" if len(ips) > 1 or len(ranks) > 1 else "single-node"


def infer_ps_deployment(servers: tuple[ServerSpec, ...]) -> str:
    return "sharded-ps" if len(servers) > 1 else "single-ps"


def parse_transports(raw: str) -> tuple[str, ...]:
    transports = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not transports:
        return ("BRPC",)
    unsupported = [item for item in transports if item not in {"BRPC", "GRPC", "RDMA"}]
    if unsupported:
        raise ValueError(f"unsupported transports: {unsupported}")
    return transports


def parse_torchrec_baselines(raw: str) -> tuple[str, ...]:
    baselines = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unsupported = [item for item in baselines if item not in {"hbm", "uvm_caching"}]
    if unsupported:
        raise ValueError(f"unsupported TorchRec baselines: {unsupported}")
    return baselines


def torchrec_label(memory_mode: str) -> str:
    if memory_mode == "uvm_caching":
        return "TorchRec-UVMCache"
    return "TorchRec-HBM"
