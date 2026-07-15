from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SUMMARY_FIELDS = (
    "step_total_ms",
    "samples_per_sec",
    "batches_per_sec",
    "dense_compute_ms",
    "embed_lookup_local_ms",
    "lookup_total_ms",
    "lookup_wait_ms",
    "lookup_local_lookup_ms",
    "lookup_gpu_cache_pull_ms",
    "lookup_fallback_pull_ms",
    "lookup_partial_merge_ms",
    "lookup_partial_prefetched_ids",
    "lookup_partial_missing_ids",
    "prefetch_issue_ms",
    "prefetch_issue_to_consume_ms",
    "prefetch_wait_share_of_lookup",
    "prefetch_dense_compute_ms",
    "prefetch_network_wait_ms",
    "prefetch_exposed_network_ms",
    "prefetch_dense_cover_ratio",
    "prefetch_issue_to_consume_cover_ratio",
    "prefetch_window_live_ids",
    "prefetch_window_live_bytes",
    "prefetch_window_peak_live_ids",
    "prefetch_window_peak_live_bytes",
    "prefetch_window_live_cache_capacity_ratio",
    "prefetch_window_peak_cache_capacity_ratio",
    "lookup_gpu_cache_request_count",
    "lookup_gpu_cache_hit_count",
    "lookup_gpu_cache_miss_count",
    "lookup_gpu_cache_hit_rate",
    "lookup_gpu_cache_query_ms",
    "lookup_gpu_cache_fill_ms",
    "planned_gpu_cache_prefill_batches",
    "planned_gpu_cache_prefill_wait_ms",
    "planned_gpu_cache_prefill_ids",
    "planned_gpu_cache_prefill_successes",
    "planned_gpu_cache_prefill_fallbacks",
    "planned_gpu_cache_prefill_result_size_mismatches",
    "batch_raw_ids",
    "batch_unique_ids",
    "batch_dedup_ratio",
    "gpu_cache_clear_count",
    "update_gpu_cache_invalidate_ms",
    "sparse_update_ms",
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _mean_numeric(rows: list[dict[str, str]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return sum(values) / len(values) if values else 0.0


def _load_measured_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    measured = [row for row in rows if row.get("warmup_excluded", "") != "1"]
    return measured or rows


def _run_cmd(args: argparse.Namespace, *, depth: int, capacity: int) -> tuple[list[str], str]:
    run_id = f"{args.run_id_prefix}-d{depth}-c{capacity}"
    cmd = [
        sys.executable,
        str(args.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "recstore",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-embeddings",
        str(args.num_embeddings),
        "--embedding-dim",
        str(args.embedding_dim),
        "--read-mode",
        "prefetch",
        "--prefetch-depth",
        str(depth),
        "--run-id",
        run_id,
        "--output-root",
        str(args.output_root),
        "--ps-type",
        str(args.ps_type),
        "--ps-kv-backend",
        str(args.ps_kv_backend),
        "--recstore-index-type",
        str(args.recstore_index_type),
    ]
    if args.no_start_server:
        cmd.append("--no-start-server")
    if args.library_path:
        cmd.extend(["--library-path", str(args.library_path)])
    if args.data_dir:
        cmd.extend(["--data-dir", str(args.data_dir)])
    if capacity > 0:
        cmd.extend(
            [
                "--enable-gpu-cache",
                "--gpu-cache-capacity",
                str(capacity),
                "--disable-gpu-cache-lookup-bypass",
            ]
        )
    return cmd, run_id


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "depth",
        "gpu_cache_capacity",
        "run_id",
        "csv",
        "recommended_by_step",
    ] + [f"{field}_mean" for field in SUMMARY_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep lookahead depth and GPU cache capacity for RecStore prefetch experiments."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/recstore_prefetch_window_sweep"))
    parser.add_argument("--run-id-prefix", type=str, default="prefetch-window")
    parser.add_argument("--depths", type=str, default="0,1,2,4,8")
    parser.add_argument("--gpu-cache-capacities", type=str, default="0,4096,8192,16384")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ps-type", type=str, default="LOCAL_SHM")
    parser.add_argument("--ps-kv-backend", type=str, default="recstore_dram")
    parser.add_argument("--recstore-index-type", type=str, default="DRAM_EXTENDIBLE_HASH")
    parser.add_argument("--library-path", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--no-start-server", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    depths = _parse_ints(args.depths)
    capacities = _parse_ints(args.gpu_cache_capacities)
    summary_rows: list[dict[str, object]] = []
    for depth in depths:
        for capacity in capacities:
            if depth == 0 and capacity == 0:
                pass
            cmd, run_id = _run_cmd(args, depth=depth, capacity=capacity)
            print(" ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=str(args.repo_root), check=True)
            csv_path = args.output_root / "outputs" / run_id / "recstore_main.csv"
            rows = _load_measured_rows(csv_path)
            summary: dict[str, object] = {
                "depth": depth,
                "gpu_cache_capacity": capacity,
                "run_id": run_id,
                "csv": str(csv_path),
                "recommended_by_step": 0,
            }
            for field in SUMMARY_FIELDS:
                summary[f"{field}_mean"] = _mean_numeric(rows, field)
            summary_rows.append(summary)

    if args.dry_run:
        return
    valid_rows = [
        row
        for row in summary_rows
        if float(row.get("planned_gpu_cache_prefill_result_size_mismatches_mean", 0.0)) == 0.0
    ]
    if valid_rows:
        best = min(valid_rows, key=lambda row: float(row.get("step_total_ms_mean", 0.0)))
        best["recommended_by_step"] = 1
    summary_path = args.output_root / "prefetch_window_sweep_summary.csv"
    _write_summary(summary_path, summary_rows)
    print(f"summary_csv={summary_path}")
    if valid_rows:
        print(
            "recommended="
            f"depth={best['depth']} "
            f"gpu_cache_capacity={best['gpu_cache_capacity']} "
            f"step_total_ms_mean={best['step_total_ms_mean']}"
        )


if __name__ == "__main__":
    main()
