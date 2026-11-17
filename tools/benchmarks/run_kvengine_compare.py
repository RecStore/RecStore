"""Run native benchmark_kv_engine YCSB comparisons.

Typical usage::

    python tools/benchmarks/run_kvengine_compare.py \\
        --engines petkv dram_pet_dram fasterkv \\
        --workloads a b c \\
        --distributions uniform zipfian \\
        --threads 16 --record-count 10000000 \\
        --output-dir results/ycsb

Draw-only mode (re-render charts from an existing summary.csv)::

    python tools/benchmarks/run_kvengine_compare.py --draw --output-dir results/ycsb
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

# ── Constants & defaults ──────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_BIN = ROOT / "build/bin/benchmark_kv_engine"
DEFAULT_DRAM_ROOT = Path("/dev/shm/recstore")
DEFAULT_SSD_ROOT = Path("/mnt/nvme1n1_recstore/recstore")
SLAB_ALLOCATOR_CHUNK_BYTES = 1 << 20
SLAB_ALLOCATOR_METADATA_BYTES = 8
SLAB_ALLOCATOR_LOAD_FACTOR = 1.2
DISTRIBUTION_LABELS = {
    "uniform": "uniform",
    "zipfian": "zipfian(alpha=0.9)",
}

DEFAULT_RUN_ENGINES = [
    "petkv",
    "dram_pet_dram",
    # "dram_eh_dram",
    # "dram_eh_ssd",
    # "dram_pet_ssd",
    # "dram_eh_tiered",
    "dram_pet_tiered",
]
THIRD_PARTY_ENGINES = {"fasterkv", "fasterkv_ssd", "hps_rocksdb"}

SUMMARY_FIELDS = [
    "workload",
    "engine",
    "index_type",
    "value_store_type",
    "repeat",
    "record_count",
    "operation_count",
    "threads",
    "distribution",
    "zipfian_alpha",
    "read_mode",
    "batch_keys",
    "phase",
    "exit_code",
    "load_runtime_sec",
    "load_operations",
    "load_throughput_ops_sec",
    "run_runtime_sec",
    "run_operations",
    "run_throughput_ops_sec",
    "run_read_operations",
    "run_update_operations",
    "data_path",
    "log_path",
    "raw_log_path",
    "error_tail",
]

AGGREGATE_FIELDS = [
    "distribution",
    "distribution_label",
    "workload",
    "engine",
    "engine_label",
    "runs",
    "successes",
    "avg_load_ops_sec",
    "avg_run_ops_sec",
    "avg_run_operations",
]

_WORKLOAD_ALIASES: dict[str, str] = {
    "workloada": "a", "workloadb": "b", "workloadc": "c",
    "a": "a", "b": "b", "c": "c",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def gflag(name: str, value: object) -> str:
    return f"--{name}={value}"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def normalize_workload(workload: str) -> str:
    try:
        return _WORKLOAD_ALIASES[workload.lower()]
    except KeyError:
        raise ValueError(f"unknown workload '{workload}', expected a/b/c")


def slab_entries_per_chunk(slab_size: int) -> int:
    """Mirror ConcurrentSlabMemoryPool's per-chunk entry calculation."""
    if slab_size <= 0:
        raise ValueError("slab_size must be positive")
    entries = SLAB_ALLOCATOR_CHUNK_BYTES // slab_size
    entries -= entries % 64
    # ChunkHeader is 8 bytes and BitMap is 4 bytes plus one uint64 per 64 slots.
    while 12 + (entries // 64) * 8 + entries * slab_size > SLAB_ALLOCATOR_CHUNK_BYTES:
        entries -= 64
    if entries <= 0:
        raise ValueError(f"slab_size {slab_size} is too large for one slab chunk")
    return entries


def recommended_slab_capacity_bytes(*, record_count: int, slab_size: int) -> int:
    target_records = int(record_count * SLAB_ALLOCATOR_LOAD_FACTOR)
    entries_per_chunk = slab_entries_per_chunk(slab_size)
    chunks = (target_records + entries_per_chunk - 1) // entries_per_chunk
    return chunks * SLAB_ALLOCATOR_CHUNK_BYTES


def recommended_dram_capacity_bytes(
    *, value_store_type: str | None, args: argparse.Namespace
) -> int:
    if args.dram_capacity_bytes > 0:
        return args.dram_capacity_bytes
    if value_store_type not in {"DRAM_VALUE_STORE", "TIERED_VALUE_STORE"}:
        return 0
    per_value_bytes = args.value_size
    if args.dram_allocator in {"PERSIST_LOOP_SLAB", "CONCURRENT_SLAB_MEMORY_POOL"}:
        per_value_bytes += args.dram_capacity_metadata_bytes
        return recommended_slab_capacity_bytes(
            record_count=args.record_count,
            slab_size=per_value_bytes,
        )
    return args.record_count * per_value_bytes


# ── RunEnv ────────────────────────────────────────────────────────────────────


@dataclass
class RunEnv:
    """Holds all benchmark-environment parameters shared across every engine.

    Constructed once per (workload, distribution, engine) invocation via
    ``from_args``. Provides ``common_gflags()`` so individual engines only need
    to declare their own flags.
    """

    workload: str
    distribution: str
    record_count: int
    threads: int
    load_threads: int
    runtime_seconds: int
    value_size: int
    read_mode: str
    batch_keys: int
    skip_load: bool
    skip_run: bool
    dram_allocator: str
    ssd_io_backend: str
    ssd_queue_depth: int
    dram_capacity_bytes: int
    ssd_capacity_bytes: int
    dram_path: Path
    ssd_path: Path
    zipfian_alpha: float

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        workload: str,
        distribution: str,
        dram_path: Path,
        ssd_path: Path,
    ) -> "RunEnv":
        """Construct from parsed CLI args plus the per-run resolved paths."""
        value_store_type = getattr(args, "_current_value_store_type", None)
        return cls(
            workload=workload,
            distribution=distribution,
            record_count=args.record_count,
            threads=args.threads,
            load_threads=args.load_threads,
            runtime_seconds=args.runtime_seconds,
            value_size=args.value_size,
            read_mode=args.read_mode,
            batch_keys=args.batch_keys,
            skip_load=args.skip_load,
            skip_run=args.skip_run,
            dram_allocator=args.dram_allocator,
            ssd_io_backend=args.ssd_io_backend,
            ssd_queue_depth=args.ssd_queue_depth,
            dram_capacity_bytes=recommended_dram_capacity_bytes(
                value_store_type=value_store_type,
                args=args,
            ),
            ssd_capacity_bytes=args.ssd_capacity_bytes,
            dram_path=dram_path,
            ssd_path=ssd_path,
            zipfian_alpha=args.zipfian_alpha,
        )

    def common_gflags(self) -> list[str]:
        """Return gflags common to every engine invocation."""
        flags = [
            gflag("dram_path", self.dram_path),
            gflag("ssd_path", self.ssd_path),
            gflag("record_count", self.record_count),
            gflag("workload", self.workload),
            gflag("distribution", self.distribution),
            gflag("zipfian_alpha", self.zipfian_alpha),
            gflag("thread_num", self.threads),
            gflag("load_thread_num", self.load_threads),
            gflag("running_seconds", self.runtime_seconds),
            gflag("value_size", self.value_size),
            gflag("read_mode", self.read_mode),
            gflag("batch_keys", self.batch_keys),
            gflag("load", str(not self.skip_load).lower()),
            gflag("run", str(not self.skip_run).lower()),
            gflag("dram_allocator", self.dram_allocator),
            gflag("ssd_io_backend", self.ssd_io_backend),
            gflag("ssd_queue_depth", self.ssd_queue_depth),
        ]
        if self.dram_capacity_bytes:
            flags.append(gflag("dram_capacity_bytes", self.dram_capacity_bytes))
        if self.ssd_capacity_bytes:
            flags.append(gflag("ssd_capacity_bytes", self.ssd_capacity_bytes))
        return flags


# ── Engine hierarchy ──────────────────────────────────────────────────────────


class BaseEngine(ABC):
    """Abstract base class for all KV engine wrappers.

    Subclasses declare ``engine_class`` and implement ``run_cmds()`` to return
    only the engine-specific gflags. ``index_type`` and ``value_store_type`` are
    intentionally absent here; they belong to ``RecStoreCompositeEngine``.
    """

    engine_class: ClassVar[str]

    def __init__(
        self,
        args: argparse.Namespace,
        engine_props: dict[str, str] | None = None,
    ) -> None:
        self._args = args
        # Copy to avoid aliasing the caller's dict.
        self._engine_props: dict[str, str] = dict(engine_props) if engine_props else {}

    @abstractmethod
    def run_cmds(self) -> list[str]:
        """Return the gflags specific to this engine."""

    def _prop_overrides(self) -> list[str]:
        """Convert engine_props dict to gflags."""
        return [gflag(key, value) for key, value in self._engine_props.items()]


class RecStoreCompositeEngine(BaseEngine):
    """RecStore built-in engine combining an index type and a value store.

    Covers all DRAM_PET_HASH / DRAM_EXTENDIBLE_HASH × DRAM / SSD / TIERED
    combinations via constructor parameters. No further subclasses needed.
    """

    engine_class = "KVEngine"

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        index_type: str,
        value_store_type: str,
        engine_props: dict[str, str] | None = None,
    ) -> None:
        super().__init__(args, engine_props)
        self.index_type = index_type
        self.value_store_type = value_store_type

    def run_cmds(self) -> list[str]:
        return [
            gflag("engine_class", self.engine_class),
            gflag("index_type", self.index_type),
            gflag("value_store_type", self.value_store_type),
        ] + self._prop_overrides()


class PetKVEngine(BaseEngine):
    """Standalone PetKV engine (no index_type / value_store_type)."""

    engine_class = "KVEnginePetKV"

    def run_cmds(self) -> list[str]:
        return [gflag("engine_class", self.engine_class)] + self._prop_overrides()


class FasterKVEngine(BaseEngine):
    """FasterKV engine; handles all fasterkv_* gflags in one place."""

    engine_class = "KVEngineFasterKV"

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        default_storage: str = "memory",
        engine_props: dict[str, str] | None = None,
    ) -> None:
        super().__init__(args, engine_props)
        self._default_storage = default_storage

    def run_cmds(self) -> list[str]:
        a = self._args
        # CLI --fasterkv-storage wins; otherwise fall back to per-instance default.
        storage = a.fasterkv_storage if a.fasterkv_storage is not None else self._default_storage
        flags = [
            gflag("engine_class", self.engine_class),
            gflag("fasterkv_storage", storage),
        ]
        if a.fasterkv_log_path:
            flags.append(gflag("fasterkv_log_path", a.fasterkv_log_path))
        if a.fasterkv_hlog_memory_bytes:
            flags.append(gflag("fasterkv_hlog_memory_bytes", a.fasterkv_hlog_memory_bytes))
        if a.fasterkv_mutable_fraction > 0.0:
            flags.append(gflag("fasterkv_mutable_fraction", a.fasterkv_mutable_fraction))
        if a.fasterkv_read_cache_bytes:
            flags.append(gflag("fasterkv_read_cache_bytes", a.fasterkv_read_cache_bytes))
        return flags + self._prop_overrides()


class HPSRocksDBEngine(BaseEngine):
    """HPS RocksDB engine."""

    engine_class = "KVEngineHPSRocksDB"

    def run_cmds(self) -> list[str]:
        return [gflag("engine_class", self.engine_class)] + self._prop_overrides()


def make_engine(
    name: str,
    args: argparse.Namespace,
    engine_props: dict[str, str] | None = None,
) -> BaseEngine:
    """Factory: map an engine alias to a concrete BaseEngine instance."""
    match name:
        case "petkv":
            return PetKVEngine(args, engine_props=engine_props)
        case "dram_pet_dram":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_PET_HASH", value_store_type="DRAM_VALUE_STORE",
                engine_props=engine_props)
        case "dram_eh_dram":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_EXTENDIBLE_HASH", value_store_type="DRAM_VALUE_STORE",
                engine_props=engine_props)
        case "dram_eh_ssd":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_EXTENDIBLE_HASH", value_store_type="SSD_VALUE_STORE",
                engine_props=engine_props)
        case "dram_pet_ssd":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_PET_HASH", value_store_type="SSD_VALUE_STORE",
                engine_props=engine_props)
        case "dram_eh_tiered":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_EXTENDIBLE_HASH", value_store_type="TIERED_VALUE_STORE",
                engine_props=engine_props)
        case "dram_pet_tiered":
            return RecStoreCompositeEngine(
                args, index_type="DRAM_PET_HASH", value_store_type="TIERED_VALUE_STORE",
                engine_props=engine_props)
        case "fasterkv":
            return FasterKVEngine(args, default_storage="memory", engine_props=engine_props)
        case "fasterkv_ssd":
            return FasterKVEngine(args, default_storage="ssd", engine_props=engine_props)
        case "hps_rocksdb":
            return HPSRocksDBEngine(args, engine_props=engine_props)
        case _:
            raise ValueError(f"Unknown engine alias '{name}'")


# ── Command building ──────────────────────────────────────────────────────────


def build_cmd(env: RunEnv, engine: BaseEngine, extra_args: list[str]) -> list[str]:
    """Assemble the full benchmark invocation command."""
    return [str(BENCHMARK_BIN)] + env.common_gflags() + engine.run_cmds() + extra_args


# ── Output parsing & IO ───────────────────────────────────────────────────────


def parse_result_line(line: str) -> dict[str, str]:
    parts = line.strip().split()
    out: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def extract_metrics(output: str) -> tuple[dict[str, str], dict[str, str]]:
    load: dict[str, str] = {}
    run: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("YCSB_LOAD_RESULT "):
            load = parse_result_line(line)
        elif line.startswith("YCSB_RESULT "):
            run = parse_result_line(line)
    return load, run


def error_tail(output: str, exit_code: int) -> str:
    if exit_code == 0:
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-5:])[:1000]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_summary(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Aggregation & rendering ───────────────────────────────────────────────────


def write_aggregate(rows: list[dict[str, object]], path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["distribution"]), str(row["workload"]), str(row["engine"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (distribution, workload, engine), group in sorted(grouped.items()):
        ok = [r for r in group if str(r["exit_code"]) == "0"]
        run_values = [float(r["run_throughput_ops_sec"]) for r in ok if r["run_throughput_ops_sec"]]
        load_values = [float(r["load_throughput_ops_sec"]) for r in ok if r["load_throughput_ops_sec"]]
        run_ops = [float(r["run_operations"]) for r in ok if r["run_operations"]]
        out.append({
            "distribution": distribution,
            "distribution_label": DISTRIBUTION_LABELS.get(distribution, distribution),
            "workload": workload,
            "engine": engine,
            "engine_label": engine,
            "runs": len(group),
            "successes": len(ok),
            "avg_load_ops_sec": sum(load_values) / len(load_values) if load_values else "",
            "avg_run_ops_sec": sum(run_values) / len(run_values) if run_values else "",
            "avg_run_operations": sum(run_ops) / len(run_ops) if run_ops else "",
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(out)
    return out


def format_ops(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def render_chart(
    rows: list[dict[str, object]], svg_path: Path, *, required: bool
) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        if required:
            raise SystemExit(
                "matplotlib is required to render the YCSB chart. "
                "Install it with: python3 -m pip install matplotlib"
            ) from exc
        print(
            "warning: matplotlib is not installed; summary CSV files were generated, "
            "but kvengine_ycsb_run_throughput.svg was skipped."
        )
        return False

    categories = sorted({(str(r["distribution"]), str(r["workload"])) for r in rows})
    engines = [e for e in DEFAULT_RUN_ENGINES if any(str(r["engine"]) == e for r in rows)]
    engines.extend(sorted({str(r["engine"]) for r in rows} - set(engines)))
    values = {
        (str(r["distribution"]), str(r["workload"]), str(r["engine"])): float(r["avg_run_ops_sec"] or 0)
        for r in rows
    }

    x = list(range(len(categories)))
    width = 0.82 / max(len(engines), 1)
    fig_w = max(14, 2.4 * len(categories))
    fig, ax = plt.subplots(figsize=(fig_w, 7), constrained_layout=True)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]

    for idx, engine in enumerate(engines):
        offset = (idx - (len(engines) - 1) / 2) * width
        heights = [values.get((dist, wl, engine), 0.0) for dist, wl in categories]
        bars = ax.bar(
            [pos + offset for pos in x],
            heights,
            width=width,
            label=engine,
            color=colors[idx % len(colors)],
        )
        ax.bar_label(
            bars,
            labels=[format_ops(v) if v > 0 else "" for v in heights],
            rotation=75,
            padding=3,
            fontsize=8,
        )

    ax.set_title(
        "YCSB timed-run throughput by KVEngine and key distribution",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("YCSB workload / key distribution")
    ax.set_ylabel("Run throughput (ops/sec)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{wl}\n{DISTRIBUTION_LABELS.get(dist, dist)}" for dist, wl in categories],
        fontsize=9,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_ops(v)))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, fontsize=9)

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return True


# ── Benchmark execution ───────────────────────────────────────────────────────


def split_engine_props(items: list[str]) -> dict[str, dict[str, str]]:
    """Parse --engine-prop overrides into a per-engine dict.

    Each item must be in the form ``engine:key=value``.
    Returns ``{engine_name: {key: value, ...}, ...}``.
    """
    out: dict[str, dict[str, str]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--engine-prop must use engine:key=value: {item}")
        engine, prop = item.split(":", 1)
        if "=" not in prop:
            raise ValueError(f"--engine-prop must include key=value: {item}")
        key, value = prop.split("=", 1)
        out.setdefault(engine, {})[key] = value
    return out


def ensure_build(enable_third_party_engines: bool) -> None:
    subprocess.run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            "build",
            "-DRECSTORE_ENABLE_THIRD_PARTY_ENGINES="
            + ("ON" if enable_third_party_engines else "OFF"),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--target", "benchmark_kv_engine", "-j"],
        cwd=ROOT,
        check=True,
    )


def run_one(
    *,
    engine_name: str,
    workload: str,
    repeat: int,
    distribution: str,
    args: argparse.Namespace,
    output_dir: Path,
    engine_props: dict[str, str],
) -> dict[str, object]:
    run_name = f"{sanitize(workload)}_{sanitize(distribution)}_{sanitize(engine_name)}_r{repeat}"
    data_path = output_dir / "data" / run_name
    dram_path = DEFAULT_DRAM_ROOT / run_name
    ssd_path = DEFAULT_SSD_ROOT / run_name
    log_path = output_dir / "logs" / f"{run_name}.log"
    raw_log_path = output_dir / "logs" / f"{run_name}.raw.log"

    if not args.keep_data:
        shutil.rmtree(dram_path, ignore_errors=True)
        shutil.rmtree(ssd_path, ignore_errors=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DRAM_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_SSD_ROOT.mkdir(parents=True, exist_ok=True)
    dram_path.mkdir(parents=True, exist_ok=True)
    ssd_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    args._current_value_store_type = (
        engine_props.get("value_store_type")
        if "value_store_type" in engine_props
        else (
            "DRAM_VALUE_STORE" if engine_name == "petkv"
            else None
        )
    )
    if engine_name in {
        "dram_pet_dram",
        "dram_eh_dram",
    }:
        args._current_value_store_type = "DRAM_VALUE_STORE"
    elif engine_name in {
        "dram_pet_ssd",
        "dram_eh_ssd",
    }:
        args._current_value_store_type = "SSD_VALUE_STORE"
    elif engine_name in {
        "dram_pet_tiered",
        "dram_eh_tiered",
    }:
        args._current_value_store_type = "TIERED_VALUE_STORE"

    env = RunEnv.from_args(
        args,
        workload=workload,
        distribution=distribution,
        dram_path=dram_path,
        ssd_path=ssd_path,
    )
    engine = make_engine(engine_name, args, engine_props)
    cmd = build_cmd(env, engine, args.extra_arg)

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start

    full_log = " ".join(cmd) + "\n\n" + proc.stdout + f"\nwall_runtime_sec={elapsed}\n"
    log_path.write_text(full_log, encoding="utf-8")
    # Keep an unmodified benchmark output file for post-run inspection/parsing.
    raw_log_path.write_text(proc.stdout, encoding="utf-8")

    load, run = extract_metrics(proc.stdout)

    row: dict[str, object] = {
        "workload": workload,
        "engine": engine_name,
        # index_type / value_store_type are RecStoreCompositeEngine-specific; default to "None".
        "index_type": getattr(engine, "index_type", "None"),
        "value_store_type": getattr(engine, "value_store_type", "None"),
        "repeat": repeat,
        "record_count": args.record_count,
        "operation_count": args.operation_count,
        "threads": args.threads,
        "distribution": distribution,
        "zipfian_alpha": args.zipfian_alpha,
        "read_mode": args.read_mode,
        "batch_keys": args.batch_keys,
        "phase": "load-run",
        "exit_code": proc.returncode,
        "load_runtime_sec": load.get("seconds", ""),
        "load_operations": load.get("ops", ""),
        "load_throughput_ops_sec": load.get("throughput_ops_sec", ""),
        "run_runtime_sec": run.get("runtime_s", ""),
        "run_operations": run.get("ops", ""),
        "run_throughput_ops_sec": run.get("throughput_ops_sec", ""),
        "run_read_operations": run.get("read_ops", ""),
        "run_update_operations": run.get("update_ops", ""),
        "data_path": str(data_path),
        "log_path": str(log_path),
        "raw_log_path": str(raw_log_path),
        "error_tail": error_tail(proc.stdout, proc.returncode),
    }

    if not args.keep_data:
        shutil.rmtree(dram_path, ignore_errors=True)
        shutil.rmtree(ssd_path, ignore_errors=True)
    return row


def run_suite(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    distribution: str,
    engine_props: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Run all (repeat × workload × engine) combinations for one distribution."""
    rows: list[dict[str, object]] = []
    summary_path = output_dir / "summary.csv"
    for repeat in range(args.repeat):
        for workload_arg in args.workloads:
            workload = normalize_workload(workload_arg)
            for engine_name in args.engines:
                props = engine_props.get(engine_name, {})
                row = run_one(
                    engine_name=engine_name,
                    workload=workload,
                    repeat=repeat,
                    distribution=distribution,
                    args=args,
                    output_dir=output_dir,
                    engine_props=props,
                )
                rows.append(row)
                print(
                    f"{workload} {distribution} {engine_name} r{repeat}: "
                    f"exit={row['exit_code']} load={row['load_throughput_ops_sec']} "
                    f"run={row['run_throughput_ops_sec']}"
                )
                write_summary(summary_path, rows)
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native benchmark_kv_engine YCSB comparisons."
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(DEFAULT_RUN_ENGINES),
        help=(
            "KV engines to run. Built-in RecStore combos default to "
            f"{DEFAULT_RUN_ENGINES}; external engines: fasterkv, "
            "fasterkv_ssd, hps_rocksdb."
        ),
    )
    parser.add_argument("--workloads", nargs="+", default=["a", "b", "c"])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "zipfian", "trace"],
        default=["uniform"],
        help="Run one or more distributions in one command.",
    )
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--record-count", type=int, default=10 * 1_000_000)
    parser.add_argument(
        "--operation-count",
        type=int,
        default=0,
        help="Kept for CSV compatibility; timed benchmark ignores this value.",
    )
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--value-size", type=int, default=128)
    parser.add_argument(
        "--read-mode",
        choices=["exists", "get", "batch_get_flat"],
        default="get",
    )
    parser.add_argument("--batch-keys", type=int, default=500)
    parser.add_argument("--bulk-load", action="store_true")
    parser.add_argument("--dram-allocator", default="CONCURRENT_SLAB_MEMORY_POOL")
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--dram-capacity-bytes", type=int, default=0)
    parser.add_argument(
        "--dram-capacity-metadata-bytes",
        type=int,
        default=SLAB_ALLOCATOR_METADATA_BYTES,
        help="Per-value allocator metadata bytes reserved for slab-based DRAM allocators.",
    )
    parser.add_argument("--ssd-capacity-bytes", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results/ycsb_kvengine"),
        help="Directory to write results (default: ../../results/ycsb_kvengine).",
    )
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra benchmark_kv_engine argument, e.g. --extra-arg=--print_util=true.",
    )
    parser.add_argument(
        "--engine-prop",
        action="append",
        default=[],
        help="engine:gflag=value override, e.g. dram_eh_dram:dram_capacity_bytes=...",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw-only mode: only render aggregate CSV/chart from existing summary.csv.",
    )
    parser.add_argument(
        "--fasterkv-storage",
        choices=["memory", "ssd"],
        default=None,
        help="Override fasterkv_storage for fasterkv/fasterkv_ssd engines.",
    )
    parser.add_argument("--fasterkv-log-path", default="")
    parser.add_argument("--fasterkv-hlog-memory-bytes", type=int, default=0)
    parser.add_argument("--fasterkv-mutable-fraction", type=float, default=0.0)
    parser.add_argument("--fasterkv-read-cache-bytes", type=int, default=0)
    return parser.parse_args(argv)


# ── Entry point ───────────────────────────────────────────────────────────────


def _cmd_draw(args: argparse.Namespace) -> int:
    """Re-render aggregate CSV and chart from an existing summary.csv."""
    summary_path = args.output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist; run benchmark first")
    all_rows = load_summary(summary_path)
    aggregate_path = args.output_dir / "kvengine_workload_summary.csv"
    chart_svg_path = args.output_dir / "kvengine_ycsb_run_throughput.svg"
    aggregate_rows = write_aggregate(all_rows, aggregate_path)
    render_chart(aggregate_rows, chart_svg_path, required=True)
    print(f"summary: {summary_path}")
    print(f"aggregate: {aggregate_path}")
    print(f"chart: {chart_svg_path}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the full benchmark suite and emit results."""
    if args.build:
        ensure_build(bool(set(args.engines) & THIRD_PARTY_ENGINES))
    if not BENCHMARK_BIN.exists():
        raise FileNotFoundError(
            f"{BENCHMARK_BIN} does not exist; build target benchmark_kv_engine first"
        )

    engine_props = split_engine_props(args.engine_prop)
    distributions = args.distributions
    all_rows: list[dict[str, object]] = []
    for distribution in distributions:
        suite_output_dir = (
            args.output_dir / distribution if len(distributions) > 1 else args.output_dir
        )
        rows = run_suite(
            args=args,
            output_dir=suite_output_dir,
            distribution=distribution,
            engine_props=engine_props,
        )
        all_rows.extend(rows)

    summary_path = args.output_dir / "summary.csv"
    if len(distributions) > 1:
        write_summary(summary_path, all_rows)
    aggregate_path = args.output_dir / "kvengine_workload_summary.csv"
    chart_svg_path = args.output_dir / "kvengine_ycsb_run_throughput.svg"
    aggregate_rows = write_aggregate(all_rows, aggregate_path)
    chart_rendered = render_chart(aggregate_rows, chart_svg_path, required=False)
    print(f"summary: {summary_path}")
    print(f"aggregate: {aggregate_path}")
    if chart_rendered:
        print(f"chart: {chart_svg_path}")
    return 0 if all(int(row["exit_code"]) == 0 for row in all_rows) else 1


def main() -> int:
    args = parse_args()
    return _cmd_draw(args) if args.draw else _cmd_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
