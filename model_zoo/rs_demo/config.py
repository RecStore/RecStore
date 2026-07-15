from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_NUM_EMBEDDINGS_PER_FEATURE = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]


def parse_num_embeddings_per_feature(value: str | Sequence[int] | None) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        values = [int(part) for part in parts]
    else:
        values = [int(item) for item in value]
    if len(values) != len(DEFAULT_NUM_EMBEDDINGS_PER_FEATURE):
        raise ValueError(
            "num_embeddings_per_feature must contain exactly "
            f"{len(DEFAULT_NUM_EMBEDDINGS_PER_FEATURE)} values"
        )
    if any(item <= 0 for item in values):
        raise ValueError("num_embeddings_per_feature values must be positive")
    return values


def cap_default_num_embeddings_per_feature(cap: int) -> list[int]:
    cap = int(cap)
    if cap <= 0:
        raise ValueError("num_embeddings cap must be positive")
    return [min(int(vocab), cap) for vocab in DEFAULT_NUM_EMBEDDINGS_PER_FEATURE]


def resolve_num_embeddings_per_feature(
    num_embeddings: int,
    override: str | Sequence[int] | None = None,
) -> list[int]:
    parsed = parse_num_embeddings_per_feature(override)
    if parsed:
        return parsed
    return cap_default_num_embeddings_per_feature(int(num_embeddings))


def format_num_embeddings_per_feature(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def total_num_embeddings_per_feature(values: Sequence[int]) -> int:
    return sum(int(value) for value in values)


def ensure_shared_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o777)
    except OSError:
        pass


@dataclass
class RunConfig:
    num_embeddings: int = 200000
    num_embeddings_per_feature: str = ""
    embedding_dim: int = 128
    batch_size: int = 4096
    steps: int = 80
    warmup_steps: int = 5
    seed: int = 20260330
    table_name: str = "mock_perf_table"
    init_rows: int = 50000
    read_before_update: bool = True
    read_mode: str = "prefetch"
    prefetch_depth: int = 0
    prefetch_issue_depth: int = 20
    recstore_enable_fusion: bool = True
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port0: int | None = None
    server_port1: int | None = None
    server_wait_seconds: float = 20.0
    allocator: str = "R2ShmMalloc"
    output_root: str = "/nas/home/shq/docker/rs_demo"
    run_id: str = ""
    jsonl: str = ""
    csv: str = ""
    local_shm_server_csv: str = ""
    recstore_main_csv: str = ""
    recstore_main_agg_csv: str = ""
    library_path: str = ""
    recstore_runtime_dir: str = ""
    server_log: str = ""
    data_dir: str = "model_zoo/torchrec_dlrm/processed_day_0_data"
    train_ratio: float = 0.8
    fuse_k: int = 30
    dense_arch_layer_sizes: str = "512,256,128"
    over_arch_layer_sizes: str = "1024,1024,512,256,1"
    backend: str = "recstore"
    nproc: int = 1
    nnodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    enable_single_node_distributed_fast_path: bool = False
    single_node_ps_backend: str = "local_shm"
    single_node_owner_policy: str = "hash_mod_world_size"
    enable_gpu_cache: bool = False
    gpu_cache_capacity: int = 0
    disable_gpu_cache_lookup_bypass: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    rdzv_backend: str = "c10d"
    rdzv_id: str = ""
    ps_type: str = "BRPC"
    recstore_index_type: str = "DRAM_EXTENDIBLE_HASH"
    ps_kv_backend: str = "recstore_dram"
    tiered_dram_capacity_multiplier: float = 2.0
    torchrec_profiler: bool = False
    torchrec_dist_mode: str = "replicated"
    torchrec_memory_mode: str = "hbm"
    torchrec_timing_sync_mode: str = "stage"
    torchrec_align_recstore_init: bool = False
    torchrec_profiler_warmup: int = 0
    torchrec_profiler_active: int = 2
    torchrec_profiler_repeat: int = 1
    torchrec_trace_dir: str = ""
    torchrec_main_csv: str = ""
    torchrec_main_agg_csv: str = ""
    torchrec_trace_csv: str = ""
    torchrec_compare_recstore_csv: str = ""
    torchrec_compare_csv: str = ""
    hps_torch_model_name: str = "recstore_hps_torch"
    hps_torch_config_file: str = ""
    hps_torch_model_dir: str = ""
    hps_torch_main_csv: str = ""
    hps_torch_main_agg_csv: str = ""
    hps_torch_key_offset_mode: str = "cumulative"
    hps_torch_materialize_embeddings: bool = True
    hps_torch_force_materialize: bool = False
    hps_torch_gpucache: bool = True
    hps_torch_gpucacheper: float = 1.0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular benchmark demo based on DLRM-style data path."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="recstore",
        choices=["recstore", "torchrec", "hps_torch"],
    )
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument(
        "--enable-single-node-distributed-fast-path",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--single-node-ps-backend",
        type=str,
        default="local_shm",
        choices=["local_shm", "hierkv"],
    )
    parser.add_argument(
        "--single-node-owner-policy",
        type=str,
        default="hash_mod_world_size",
        choices=["hash_mod_world_size"],
    )
    parser.add_argument(
        "--enable-gpu-cache",
        action="store_true",
        default=False,
        help="Enable RecStore GPU read/write training cache for local fast path.",
    )
    parser.add_argument(
        "--gpu-cache-capacity",
        type=int,
        default=0,
        help="Number of embedding rows to keep in the RecStore GPU cache.",
    )
    parser.add_argument(
        "--disable-gpu-cache-lookup-bypass",
        action="store_true",
        default=False,
        help=(
            "Keep querying the RecStore GPU cache for large low-hit lookups. "
            "Useful for planned/lookahead cache experiments."
        ),
    )
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rdzv-backend", type=str, default="c10d")
    parser.add_argument("--rdzv-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default="/nas/home/shq/docker/rs_demo")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--ps-type",
        type=str,
        default="BRPC",
        choices=["BRPC", "GRPC", "LOCAL_SHM", "RDMA"],
    )
    parser.add_argument(
        "--recstore-index-type",
        type=str,
        default="DRAM_EXTENDIBLE_HASH",
        choices=["DRAM_UNORDERED_MAP", "DRAM_EXTENDIBLE_HASH", "DRAM_PET_HASH"],
    )
    parser.add_argument(
        "--ps-kv-backend",
        type=str,
        default="recstore_dram",
        choices=["recstore_dram", "recstore_tiered", "hps_hash_map", "hps_rocksdb"],
        help=(
            "Server-side BaseKV backend used by the RecStore PyTorch runner. "
            "HPS options route the model through RecStore PS with an HPS KV engine."
        ),
    )
    parser.add_argument(
        "--tiered-dram-capacity-multiplier",
        type=float,
        default=2.0,
        help=(
            "DRAM allocator bytes for recstore_tiered as "
            "kv_capacity * value_size_bytes * multiplier."
        ),
    )
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument(
        "--num-embeddings-per-feature",
        type=str,
        default="",
        help=(
            "Comma-separated cardinalities for the 26 sparse tables. "
            "When omitted, --num-embeddings is treated as a per-table cap "
            "over the default Criteo DLRM table sizes."
        ),
    )
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--table-name", type=str, default="mock_perf_table")
    parser.add_argument("--init-rows", type=int, default=50000)
    parser.add_argument("--read-before-update", action="store_true", default=True)
    parser.add_argument("--no-read-before-update", action="store_true")
    parser.add_argument(
        "--read-mode",
        type=str,
        default="prefetch",
        choices=["prefetch", "direct"],
        help="read path mode when read-before-update is enabled",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=0,
        help=(
            "Number of future batches to issue fused embedding prefetches ahead. "
            "0 keeps the legacy issue-and-immediate-wait path."
        ),
    )
    parser.add_argument(
        "--prefetch-issue-depth",
        type=int,
        default=20,
        help=(
            "Maximum future batches with live issued prefetch handles. "
            "The oracle may still observe --prefetch-depth batches, but this "
            "caps outstanding network/GPU-cache pressure for large windows. "
            "Use 0 to match --prefetch-depth."
        ),
    )
    parser.add_argument(
        "--disable-recstore-fusion",
        action="store_true",
        default=False,
        help="Disable RecStore fused table id path for ablation runs.",
    )
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port0", type=int, default=None)
    parser.add_argument("--server-port1", type=int, default=None)
    parser.add_argument("--server-wait-seconds", type=float, default=20.0)
    parser.add_argument("--allocator", type=str, default="R2ShmMalloc")
    parser.add_argument("--jsonl", type=str, default="")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--local-shm-server-csv", type=str, default="")
    parser.add_argument("--recstore-main-csv", type=str, default="")
    parser.add_argument("--recstore-main-agg-csv", type=str, default="")
    parser.add_argument("--library-path", type=str, default="")
    parser.add_argument("--recstore-runtime-dir", type=str, default="")
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="model_zoo/torchrec_dlrm/processed_day_0_data",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--fuse-k", type=int, default=30)
    parser.add_argument(
        "--dense-arch-layer-sizes",
        type=str,
        default="512,256,128",
    )
    parser.add_argument(
        "--over-arch-layer-sizes",
        type=str,
        default="1024,1024,512,256,1",
    )
    parser.add_argument("--torchrec-profiler", action="store_true", default=False)
    parser.add_argument(
        "--torchrec-dist-mode",
        type=str,
        default="replicated",
        choices=["replicated", "fair_remote"],
    )
    parser.add_argument(
        "--torchrec-memory-mode",
        type=str,
        default="hbm",
        choices=["hbm", "uvm_caching"],
        help="TorchRec embedding memory mode. hbm keeps the current GPU-resident baseline; uvm_caching uses TorchRec/FBGEMM fused UVM caching when available.",
    )
    parser.add_argument(
        "--torchrec-timing-sync-mode",
        type=str,
        default="stage",
        choices=["stage", "step", "none"],
        help="TorchRec CUDA synchronization policy for benchmark timing. stage synchronizes inside each measured stage; step synchronizes only at step boundaries; none avoids explicit timing synchronizes.",
    )
    parser.add_argument(
        "--torchrec-align-recstore-init",
        action="store_true",
        default=False,
        help=(
            "Validation mode: zero TorchRec embeddings and reset dense-module RNG "
            "to match RecStore's zero-initialized PS path."
        ),
    )
    parser.add_argument("--torchrec-profiler-warmup", type=int, default=0)
    parser.add_argument("--torchrec-profiler-active", type=int, default=2)
    parser.add_argument("--torchrec-profiler-repeat", type=int, default=1)
    parser.add_argument("--torchrec-trace-dir", type=str, default="")
    parser.add_argument("--torchrec-main-csv", type=str, default="")
    parser.add_argument(
        "--torchrec-main-agg-csv",
        type=str,
        default="",
    )
    parser.add_argument("--torchrec-trace-csv", type=str, default="")
    parser.add_argument(
        "--torchrec-compare-recstore-csv",
        type=str,
        default="",
        help="If provided, generate RecStore vs TorchRec comparison csv from this RecStore csv.",
    )
    parser.add_argument(
        "--torchrec-compare-csv",
        type=str,
        default="",
    )
    parser.add_argument("--hps-torch-model-name", type=str, default="recstore_hps_torch")
    parser.add_argument("--hps-torch-config-file", type=str, default="")
    parser.add_argument("--hps-torch-model-dir", type=str, default="")
    parser.add_argument("--hps-torch-main-csv", type=str, default="")
    parser.add_argument("--hps-torch-main-agg-csv", type=str, default="")
    parser.add_argument(
        "--hps-torch-key-offset-mode",
        type=str,
        default="cumulative",
        choices=["cumulative", "none"],
        help=(
            "How HPS table keys are written. cumulative gives each table a disjoint "
            "key range, matching HPS table-fusion requirements."
        ),
    )
    parser.add_argument(
        "--hps-torch-no-materialize-embeddings",
        action="store_true",
        default=False,
        help="Reuse existing HPS key/emb_vector files instead of generating them.",
    )
    parser.add_argument(
        "--hps-torch-force-materialize",
        action="store_true",
        default=False,
        help="Regenerate HPS key/emb_vector files even if metadata matches.",
    )
    parser.add_argument(
        "--hps-torch-disable-gpucache",
        action="store_true",
        default=False,
    )
    parser.add_argument("--hps-torch-gpucacheper", type=float, default=1.0)
    return parser


def parse_config(argv: list[str] | None = None) -> RunConfig:
    ns = build_parser().parse_args(argv)
    cfg_kwargs = vars(ns).copy()
    cfg_kwargs.pop("no_read_before_update", None)
    cfg_kwargs.pop("no_start_server", None)
    disable_recstore_fusion = bool(cfg_kwargs.pop("disable_recstore_fusion", False))
    hps_no_materialize = bool(cfg_kwargs.pop("hps_torch_no_materialize_embeddings", False))
    hps_disable_gpucache = bool(cfg_kwargs.pop("hps_torch_disable_gpucache", False))
    if cfg_kwargs["nproc_per_node"] is None:
        cfg_kwargs["nproc_per_node"] = cfg_kwargs.get("nproc", 1)
    cfg = RunConfig(**cfg_kwargs)
    if ns.no_read_before_update:
        cfg.read_before_update = False
    if disable_recstore_fusion:
        cfg.recstore_enable_fusion = False
    if ns.no_start_server:
        cfg.start_server = False
    if hps_no_materialize:
        cfg.hps_torch_materialize_embeddings = False
    if hps_disable_gpucache:
        cfg.hps_torch_gpucache = False
    return cfg


def validate_hps_torch_config(cfg: RunConfig) -> None:
    if cfg.backend != "hps_torch":
        return
    resolve_num_embeddings_per_feature(cfg.num_embeddings, cfg.num_embeddings_per_feature)
    if cfg.nnodes != 1:
        raise RuntimeError("hps_torch backend currently supports single-node runs only.")
    if cfg.nproc_per_node <= 0:
        raise RuntimeError("--nproc-per-node must be greater than 0.")
    if cfg.node_rank != 0:
        raise RuntimeError("hps_torch single-node runs require --node-rank=0.")
    if cfg.hps_torch_gpucacheper < 0.0 or cfg.hps_torch_gpucacheper > 1.0:
        raise RuntimeError("--hps-torch-gpucacheper must be within [0, 1].")


def validate_torchrec_config(cfg: RunConfig) -> None:
    if cfg.backend != "torchrec":
        return
    resolve_num_embeddings_per_feature(cfg.num_embeddings, cfg.num_embeddings_per_feature)

    if cfg.nnodes <= 0:
        raise RuntimeError("--nnodes must be greater than 0.")
    if cfg.nproc_per_node <= 0:
        raise RuntimeError("--nproc-per-node must be greater than 0.")
    if cfg.node_rank < 0 or cfg.node_rank >= cfg.nnodes:
        raise RuntimeError("--node-rank must be within [0, nnodes).")

    profiler_subargs_nondefault = any(
        [
            cfg.torchrec_profiler_warmup != 0,
            cfg.torchrec_profiler_active != 2,
            cfg.torchrec_profiler_repeat != 1,
        ]
    )

    if profiler_subargs_nondefault and not cfg.torchrec_profiler:
        raise RuntimeError(
            "TorchRec profiler sub-arguments require --torchrec-profiler."
        )
    if cfg.torchrec_dist_mode == "fair_remote":
        world_size = cfg.nnodes * cfg.nproc_per_node
        if world_size <= 1:
            raise RuntimeError("fair_remote requires world_size greater than 1.")


def validate_recstore_config(cfg: RunConfig) -> None:
    if cfg.backend != "recstore":
        return
    resolve_num_embeddings_per_feature(cfg.num_embeddings, cfg.num_embeddings_per_feature)

    if cfg.nnodes <= 0:
        raise RuntimeError("--nnodes must be greater than 0.")
    if cfg.nproc_per_node <= 0:
        raise RuntimeError("--nproc-per-node must be greater than 0.")
    if cfg.node_rank < 0 or cfg.node_rank >= cfg.nnodes:
        raise RuntimeError("--node-rank must be within [0, nnodes).")
    if cfg.enable_gpu_cache and cfg.gpu_cache_capacity <= 0:
        raise RuntimeError(
            "--gpu-cache-capacity must be positive when --enable-gpu-cache is set"
        )
    if cfg.prefetch_depth < 0:
        raise RuntimeError("--prefetch-depth must be non-negative")
    if cfg.prefetch_issue_depth < 0:
        raise RuntimeError("--prefetch-issue-depth must be non-negative")
    if cfg.tiered_dram_capacity_multiplier < 0:
        raise RuntimeError("--tiered-dram-capacity-multiplier must be non-negative")
    if cfg.enable_single_node_distributed_fast_path:
        if cfg.nnodes != 1:
            raise RuntimeError(
                "RecStore single-node distributed fast path requires --nnodes=1."
            )
        if cfg.nproc_per_node <= 1:
            raise RuntimeError(
                "RecStore single-node distributed fast path requires --nproc-per-node greater than 1."
            )
        if cfg.single_node_ps_backend not in {"local_shm", "hierkv"}:
            raise RuntimeError(
                "RecStore single-node distributed fast path only supports --single-node-ps-backend=local_shm or hierkv."
            )
        if cfg.single_node_owner_policy != "hash_mod_world_size":
            raise RuntimeError(
                "RecStore single-node distributed fast path only supports --single-node-owner-policy=hash_mod_world_size."
            )
    if cfg.nnodes > 1 and not cfg.recstore_runtime_dir:
        raise RuntimeError(
            "RecStore multi-node requires --recstore-runtime-dir pointing to a shared runtime directory."
        )


def ensure_run_id(cfg: RunConfig) -> None:
    if cfg.run_id:
        return
    cfg.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def populate_default_paths(cfg: RunConfig) -> None:
    ensure_run_id(cfg)
    cfg.output_root = str(Path(cfg.output_root).resolve())
    outputs_base = Path(cfg.output_root) / "outputs" / cfg.run_id
    logs_base = Path(cfg.output_root) / "logs" / cfg.run_id

    if not cfg.jsonl:
        cfg.jsonl = str(outputs_base / "recstore_events.jsonl")
    if not cfg.csv:
        cfg.csv = str(outputs_base / "recstore_embupdate.csv")
    if not cfg.local_shm_server_csv:
        cfg.local_shm_server_csv = str(outputs_base / "recstore_local_shm_server.csv")
    if not cfg.recstore_main_csv:
        cfg.recstore_main_csv = str(outputs_base / "recstore_main.csv")
    if not cfg.recstore_main_agg_csv:
        cfg.recstore_main_agg_csv = str(outputs_base / "recstore_main_agg.csv")
    if not cfg.server_log:
        cfg.server_log = str(logs_base / "ps_server.log")
    if not cfg.torchrec_trace_dir:
        cfg.torchrec_trace_dir = str(outputs_base / "torchrec_traces")
    if not cfg.torchrec_main_csv:
        cfg.torchrec_main_csv = str(outputs_base / "torchrec_main.csv")
    if not cfg.torchrec_main_agg_csv:
        cfg.torchrec_main_agg_csv = str(outputs_base / "torchrec_main_agg.csv")
    if not cfg.torchrec_trace_csv:
        cfg.torchrec_trace_csv = str(outputs_base / "torchrec_trace.csv")
    if not cfg.torchrec_compare_csv:
        cfg.torchrec_compare_csv = str(outputs_base / "recstore_torchrec_compare.csv")
    if not cfg.hps_torch_model_dir:
        cfg.hps_torch_model_dir = str(outputs_base / "hps_torch_model")
    if not cfg.hps_torch_config_file:
        cfg.hps_torch_config_file = str(outputs_base / "hps_torch.json")
    if not cfg.hps_torch_main_csv:
        cfg.hps_torch_main_csv = str(outputs_base / "hps_torch_main.csv")
    if not cfg.hps_torch_main_agg_csv:
        cfg.hps_torch_main_agg_csv = str(outputs_base / "hps_torch_main_agg.csv")

    cfg.recstore_main_csv = str(Path(cfg.recstore_main_csv).resolve())
    cfg.recstore_main_agg_csv = str(Path(cfg.recstore_main_agg_csv).resolve())


def ensure_parent_dirs(cfg: RunConfig) -> None:
    ensure_shared_dir(Path(cfg.jsonl).parent)
    ensure_shared_dir(Path(cfg.csv).parent)
    ensure_shared_dir(Path(cfg.local_shm_server_csv).parent)
    ensure_shared_dir(Path(cfg.recstore_main_csv).parent)
    ensure_shared_dir(Path(cfg.recstore_main_agg_csv).parent)
    ensure_shared_dir(Path(cfg.server_log).parent)
    ensure_shared_dir(Path(cfg.torchrec_trace_dir))
    ensure_shared_dir(Path(cfg.torchrec_main_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_main_agg_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_trace_csv).parent)
    ensure_shared_dir(Path(cfg.torchrec_compare_csv).parent)
    ensure_shared_dir(Path(cfg.hps_torch_config_file).parent)
    ensure_shared_dir(Path(cfg.hps_torch_model_dir))
    ensure_shared_dir(Path(cfg.hps_torch_main_csv).parent)
    ensure_shared_dir(Path(cfg.hps_torch_main_agg_csv).parent)
