#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rs_demo RecStore vs TorchRec comparison over batch/nproc grid."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--num-embeddings", type=int, default=10000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/rs_demo_nproc_compare",
    )
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--torchrec-profiler", action="store_true", default=False)
    parser.add_argument("--torchrec-profiler-active", type=int, default=1)
    parser.add_argument("--torchrec-profiler-repeat", type=int, default=1)
    return parser


def _run(cmd: list[str], cwd: Path) -> None:
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )


def _is_completed(compare_csv: Path, torchrec_agg_csv: Path) -> bool:
    return compare_csv.exists() and torchrec_agg_csv.exists()


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"expected rows in csv: {path}")
    return rows[0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | int | float]] = []
    compare_rows: list[dict[str, str | int | float]] = []

    run_idx = 0
    for batch_size in args.batch_sizes:
        for nproc in args.nprocs:
            for repeat_idx in range(args.repeat):
                run_idx += 1
                tag = f"b{batch_size}_n{nproc}_r{repeat_idx}"
                recstore_csv = output_dir / f"{tag}_recstore.csv"
                recstore_jsonl = output_dir / f"{tag}_recstore.jsonl"
                recstore_log = output_dir / f"{tag}_ps_server.log"
                torchrec_csv = output_dir / f"{tag}_torchrec.csv"
                torchrec_agg_csv = output_dir / f"{tag}_torchrec_agg.csv"
                torchrec_trace_dir = output_dir / f"{tag}_torchrec_traces"
                torchrec_trace_csv = output_dir / f"{tag}_torchrec_trace.csv"
                compare_csv = output_dir / f"{tag}_compare.csv"

                if _is_completed(compare_csv, torchrec_agg_csv):
                    print(f"[rs_demo] skip completed {tag}")
                    agg_row = _read_single_row(torchrec_agg_csv)
                    agg_row["batch_size"] = batch_size
                    agg_row["nproc"] = nproc
                    agg_row["repeat"] = repeat_idx
                    summary_rows.append(agg_row)
                    for row in _read_rows(compare_csv):
                        row["batch_size"] = batch_size
                        row["nproc"] = nproc
                        row["repeat"] = repeat_idx
                        compare_rows.append(row)
                    continue

                common_args = [
                    "--steps",
                    str(args.steps),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--batch-size",
                    str(batch_size),
                    "--num-embeddings",
                    str(args.num_embeddings),
                    "--embedding-dim",
                    str(args.embedding_dim),
                ]

                recstore_cmd = [
                    sys.executable,
                    str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
                    *common_args,
                    "--jsonl",
                    str(recstore_jsonl),
                    "--csv",
                    str(recstore_csv),
                    "--server-log",
                    str(recstore_log),
                ]
                _run(recstore_cmd, repo_root)

                torchrec_cmd = [
                    sys.executable,
                    str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
                    "--backend",
                    "torchrec",
                    "--nproc",
                    str(nproc),
                    "--master-port",
                    str(args.master_port_base + run_idx),
                    *common_args,
                    "--no-start-server",
                    "--torchrec-main-csv",
                    str(torchrec_csv),
                    "--torchrec-main-agg-csv",
                    str(torchrec_agg_csv),
                    "--torchrec-trace-dir",
                    str(torchrec_trace_dir),
                    "--torchrec-trace-csv",
                    str(torchrec_trace_csv),
                    "--torchrec-compare-recstore-csv",
                    str(recstore_csv),
                    "--torchrec-compare-csv",
                    str(compare_csv),
                ]
                if args.torchrec_profiler:
                    torchrec_cmd.extend(
                        [
                            "--torchrec-profiler",
                            "--torchrec-profiler-active",
                            str(args.torchrec_profiler_active),
                            "--torchrec-profiler-repeat",
                            str(args.torchrec_profiler_repeat),
                        ]
                    )
                _run(torchrec_cmd, repo_root)

                agg_row = _read_single_row(torchrec_agg_csv)
                agg_row["batch_size"] = batch_size
                agg_row["nproc"] = nproc
                agg_row["repeat"] = repeat_idx
                summary_rows.append(agg_row)

                with compare_csv.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        row["batch_size"] = batch_size
                        row["nproc"] = nproc
                        row["repeat"] = repeat_idx
                        compare_rows.append(row)

    summary_path = output_dir / "torchrec_grid_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    compare_path = output_dir / "recstore_torchrec_grid_compare.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "batch_size",
            "nproc",
            "repeat",
            "metric",
            "recstore_ms",
            "torchrec_ms",
            "delta_ms",
            "delta_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"[rs_demo] output_dir: {output_dir}")
    print(f"[rs_demo] torchrec_grid_summary: {summary_path}")
    print(f"[rs_demo] recstore_torchrec_grid_compare: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
