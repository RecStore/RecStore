from __future__ import annotations

import socket
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..common import ROOT, SPARSE_FEATURES_PER_SAMPLE, _read_csv
from .config import BenchmarkConfig, infer_client_deployment, infer_ps_deployment


def _warm_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [row for row in rows if str(row.get("warmup_excluded", "0")) not in {"1", "true", "True"}]


def _mean(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"}]
    return statistics.fmean(vals) if vals else 0.0


LATENCY_BREAKDOWN_COLUMNS = (
    "batch_prepare_ms",
    "input_pack_ms",
    "lookup_ids_build_ms",
    "embed_lookup_ms",
    "dense_fwd_ms",
    "backward_ms",
    "dense_optimizer_ms",
    "sparse_backward_replay_ms",
    "sparse_optimizer_step_ms",
    "update_overlap_prepare_ms",
    "sparse_optimizer_flush_ms",
    "sparse_optimizer_ms",
    "step_total_ms",
)


def _p95(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = sorted(float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"})
    if not vals:
        return 0.0
    return vals[int(round((len(vals) - 1) * 0.95))]


def _group_by_step(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("step", "")), []).append(row)
    return grouped


def _job_samples_per_sec(rows: list[dict[str, str]], batch_size: int) -> float:
    per_step_totals = []
    for step_rows in _group_by_step(rows).values():
        latencies = [
            float(row["step_total_ms"])
            for row in step_rows
            if row.get("step_total_ms", "") not in {"", "nan", "NaN"}
        ]
        if latencies and max(latencies) > 0.0:
            per_step_totals.append(batch_size * len(step_rows) * 1000.0 / max(latencies))
    return statistics.fmean(per_step_totals) if per_step_totals else 0.0


def _job_rows_per_sec(rows: list[dict[str, str]], batch_size: int, latency_column: str) -> float:
    sparse_rows = batch_size * SPARSE_FEATURES_PER_SAMPLE
    per_step_totals = []
    for step_rows in _group_by_step(rows).values():
        total = 0.0
        for row in step_rows:
            raw = row.get(latency_column, "")
            if raw in ("", "nan", "NaN"):
                continue
            latency_ms = float(raw)
            if latency_ms > 0.0:
                total += sparse_rows / (latency_ms / 1000.0) / 1e6
        per_step_totals.append(total)
    return statistics.fmean(per_step_totals) if per_step_totals else 0.0


def collect_summary_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_csv: set[str] = set()
    for item in manifest:
        csv_key = str(item.get("main_csv", ""))
        if csv_key in seen_csv:
            # One manifest row per client node shares the same main_csv (it already
            # aggregates all ranks) -- processing it more than once double counts the run.
            continue
        seen_csv.add(csv_key)
        path = Path(csv_key)
        if not path.exists():
            continue
        warm = _warm_rows(path)
        batch_size = int(item["batch_size"])
        latency_means = {
            column: _mean(warm, column)
            if any(row.get(column, "") not in {"", "nan", "NaN"} for row in warm)
            else ""
            for column in LATENCY_BREAKDOWN_COLUMNS
        }
        out.append(
            {
                **item,
                "p95_step_total_ms": _p95(warm, "step_total_ms"),
                "samples_per_sec": _job_samples_per_sec(warm, batch_size),
                "lookup_mrows_per_sec": _job_rows_per_sec(warm, batch_size, "embed_lookup_ms"),
                "update_mrows_per_sec": _job_rows_per_sec(warm, batch_size, "sparse_optimizer_ms"),
                **{f"mean_{column}": value for column, value in latency_means.items()},
            }
        )
    return out


def _unit(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3f}K"
    return f"{value:.3f}"


def _repeat_stats(rows: list[dict[str, Any]], metric: str) -> tuple[float, float, int]:
    vals = [float(row.get(metric, 0.0) or 0.0) for row in rows if float(row.get(metric, 0.0) or 0.0) > 0.0]
    if not vals:
        return 0.0, 0.0, 0
    mean = statistics.fmean(vals)
    cv = statistics.pstdev(vals) / mean if len(vals) >= 2 and mean > 0.0 else 0.0
    return mean, cv, len(vals)


def _per_repeat_samples(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row.get("repeat_index", 0)))
    parts = []
    for row in ordered:
        value = float(row.get("samples_per_sec", 0.0) or 0.0)
        parts.append(_unit(value) if value > 0.0 else "-")
    return ", ".join(parts) if parts else "-"


def _git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def render_summary_md(cfg: BenchmarkConfig, rows: list[dict[str, Any]]) -> str:
    header = [_git_commit_hash(), socket.gethostname(), ""]
    clients = "; ".join(
        f"{client.ip}/gpu{client.gpu_id}/rank{client.node_rank}/nproc{client.nproc_per_node}"
        for client in cfg.clients
    )
    servers = "; ".join(f"{server.ip}:{server.port}/shard{server.shard_id}" for server in cfg.servers)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("lane", row.get("transport", ""))), []).append(row)
    systems = sorted(grouped) or ["-"]
    transports = sorted(
        {str(row["transport"]) for row in rows if row.get("backend") == "recstore" and row.get("transport")}
    )

    def table(metrics: list[tuple[str, list[str]]]) -> list[str]:
        return [
            f"| 指标 | {' | '.join(systems)} |",
            f"| --- | {' | '.join(['---:'] * len(systems))} |",
            *(f"| {label} | {' | '.join(values)} |" for label, values in metrics),
        ]

    def format_mean(group: list[dict[str, Any]], metric: str) -> str:
        values = [row[metric] for row in group if row.get(metric, "") not in {"", "nan", "NaN"}]
        return f"{statistics.fmean(float(value) for value in values):.3f}" if values else "-"

    lines = [
        *header,
        "## Workload 说明",
        "",
        (
            f"本次测试模型为 {cfg.model}，client 部署为 {infer_client_deployment(cfg.clients)}，"
            f"PS 部署为 {infer_ps_deployment(cfg.servers)}，RecStore transport={','.join(transports) or '-'}，"
            f"client=[{clients}]，PS=[{servers}]。"
            f"batch_size={cfg.batch_size}，embedding_dim={cfg.embedding_dim}，"
            f"num_embeddings={cfg.num_embeddings}，steps={cfg.steps}，warmup_steps={cfg.warmup_steps}，"
            f"repeat={cfg.repeat}，read_mode={cfg.read_mode}，prefetch_depth={cfg.prefetch_depth}，"
            f"index_type={cfg.index_type}，TorchRec baseline={','.join(cfg.torchrec_baselines) or 'disabled'}，"
            f"dataset={cfg.dataset_path}，runtime={cfg.resolved_runtime_dir}，"
            f"output={cfg.output_dir}。"
        ),
        "",
    ]
    lines.extend(table([
        ("backend", [str(grouped[name][0].get("backend", "-")) if name in grouped else "-" for name in systems]),
        ("batch_size", [str(grouped[name][0].get("batch_size", "-")) if name in grouped else "-" for name in systems]),
        ("embedding_dim", [str(grouped[name][0].get("embedding_dim", "-")) if name in grouped else "-" for name in systems]),
    ]))

    lines.extend(
        [
            "",
            "## E2E 吞吐（samples/s，...）",
            "",
        ]
    )
    repeat_stats = {
        name: _repeat_stats(group, "samples_per_sec") for name, group in grouped.items()
    }
    lines.extend(table([
        ("repeat_n", [str(repeat_stats[name][2]) if name in grouped else "0" for name in systems]),
        ("samples/s 各次", [_per_repeat_samples(grouped[name]) if name in grouped else "-" for name in systems]),
        ("samples/s 均值", [_unit(repeat_stats[name][0]) if name in grouped else "0.000" for name in systems]),
        ("CV", [f"{repeat_stats[name][1] * 100:.2f}%" if name in grouped else "0.00%" for name in systems]),
        ("lookup M rows/s", [f"{_mean(grouped[name], 'lookup_mrows_per_sec'):.3f}" if name in grouped else "0.000" for name in systems]),
        ("update M rows/s", [f"{_mean(grouped[name], 'update_mrows_per_sec'):.3f}" if name in grouped else "0.000" for name in systems]),
    ]))

    lines.extend(
        [
            "",
            "## E2E 延迟分解（ms，warmup 已剔除，跨 rank 与 repeat 取均值）",
            "",
        ]
    )
    lines.extend(table([
        *[
            (column, [format_mean(grouped[name], f"mean_{column}") if name in grouped else "-" for name in systems])
            for column in LATENCY_BREAKDOWN_COLUMNS
        ],
        ("p95 step_total", [f"{_mean(grouped[name], 'p95_step_total_ms'):.3f}" if name in grouped else "0.000" for name in systems]),
    ]))
    lines.append("")
    return "\n".join(lines)
