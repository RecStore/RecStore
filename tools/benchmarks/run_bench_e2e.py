from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmarks.e2e.common import (
    DEFAULT_DAY0,
    DEFAULT_OUTPUT_ROOT,
    ExecutionContext,
    E2ELane,
    PlanOverrides,
    ROOT,
    _collect_environment_metadata,
    _gpu_count,
    _has_rdma,
    _load_manifest,
    _parse_int_tuple,
    _parse_str_tuple,
    _read_csv,
    _write_csv,
)
from tools.benchmarks.e2e.commands import build_rs_demo_command, wrap_remote_command
from tools.benchmarks.e2e.figures import build_figure_specs, write_svg_figures
from tools.benchmarks.e2e.plan import build_plan
from tools.benchmarks.e2e.report import render_latex_report
from tools.benchmarks.e2e.runner import run_e2e_plan, run_rdma_ps_calibration
from tools.benchmarks.e2e.summary import (
    build_result_insights,
    build_gap_summary,
    collect_e2e_summary,
    combine_existing_roots,
)

__all__ = [
    "ExecutionContext",
    "E2ELane",
    "PlanOverrides",
    "build_figure_specs",
    "build_gap_summary",
    "build_plan",
    "build_result_insights",
    "build_rs_demo_command",
    "collect_e2e_summary",
    "combine_existing_roots",
    "main",
    "render_latex_report",
    "run_e2e_plan",
    "run_rdma_ps_calibration",
    "wrap_remote_command",
    "write_svg_figures",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RecStore vs TorchRec E2E benchmark matrix.")
    parser.add_argument("--profile", choices=["smoke", "pilot", "stress", "full"], default="pilot")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_DAY0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-rdma-ps", action="store_true")
    parser.add_argument("--data-rows", default="", help="Comma-separated override for dataset row counts.")
    parser.add_argument("--batch-sizes", default="", help="Comma-separated override for batch sizes.")
    parser.add_argument("--num-embeddings", default="", help="Comma-separated override for embedding table rows.")
    parser.add_argument("--embedding-dims", default="", help="Comma-separated override for embedding dimensions.")
    parser.add_argument("--steps", type=int, default=None, help="Override measured steps per run.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps per run.")
    parser.add_argument("--repeat", type=int, default=None, help="Override repeats per configuration.")
    parser.add_argument(
        "--only-lanes",
        default="",
        help="Comma-separated lane slugs to run after lane construction.",
    )
    parser.add_argument(
        "--include-ablation-lanes",
        action="store_true",
        help="Include extra 1P RecStore backend/transport/prefetch ablation lanes.",
    )
    parser.add_argument(
        "--remote-train-host",
        default="",
        help="Run E2E benchmark commands through ssh on this host/container.",
    )
    parser.add_argument(
        "--remote-ssh-port",
        type=int,
        default=22,
        help="SSH port for --remote-train-host (default: 22).",
    )
    parser.add_argument(
        "--remote-repo-root",
        type=Path,
        default=ROOT,
        help="Repository path visible from --remote-train-host.",
    )
    parser.add_argument(
        "--remote-python-bin",
        default=sys.executable,
        help="Python executable used on --remote-train-host.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="torchrun/rs_demo node count.")
    parser.add_argument("--node-rank", type=int, default=0, help="torchrun/rs_demo node rank for this runner.")
    parser.add_argument("--master-addr", default="127.0.0.1", help="torchrun master address.")
    parser.add_argument(
        "--external-recstore-runtime-dir",
        type=Path,
        default=None,
        help="Existing RecStore runtime dir/config visible to the train host.",
    )
    parser.add_argument(
        "--no-start-recstore-server",
        action="store_true",
        help="Pass --no-start-server for RecStore lanes so training uses an external PS.",
    )
    parser.add_argument("--server-host", default="", help="RecStore PS host passed to RecStore lanes.")
    parser.add_argument("--server-port0", type=int, default=None, help="RecStore shard-0 port.")
    parser.add_argument("--server-port1", type=int, default=None, help="RecStore shard-1 port.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate summary CSV and LaTeX from existing manifest/results.",
    )
    parser.add_argument(
        "--combine-roots",
        type=Path,
        nargs="+",
        default=[],
        help="Combine existing E2E output roots before regenerating summaries.",
    )
    args = parser.parse_args(argv)

    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.repeat is not None and args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.nnodes <= 0:
        raise ValueError("--nnodes must be positive")
    if args.node_rank < 0:
        raise ValueError("--node-rank must be non-negative")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = PlanOverrides(
        data_rows=_parse_int_tuple(args.data_rows),
        batch_sizes=_parse_int_tuple(args.batch_sizes),
        num_embeddings=_parse_int_tuple(args.num_embeddings),
        embedding_dims=_parse_int_tuple(args.embedding_dims),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeat=args.repeat,
        only_lanes=_parse_str_tuple(args.only_lanes),
        include_ablation_lanes=args.include_ablation_lanes,
    )
    plan = build_plan(args.profile, output_root, overrides)
    context = ExecutionContext(
        remote_train_host=args.remote_train_host,
        remote_ssh_port=args.remote_ssh_port,
        remote_repo_root=args.remote_repo_root,
        python_bin=args.remote_python_bin,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        external_recstore_runtime_dir=args.external_recstore_runtime_dir,
        no_start_recstore_server=args.no_start_recstore_server,
        server_host=args.server_host,
        server_port0=args.server_port0,
        server_port1=args.server_port1,
    )
    metadata = {
        "profile": plan.profile,
        "output_root": str(output_root),
        "input_file": str(args.input_file),
        "gpu_count": _gpu_count(),
        "rdma_available": _has_rdma(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_rows": list(plan.data_rows),
        "batch_sizes": list(plan.batch_sizes),
        "num_embeddings": list(plan.num_embeddings),
        "embedding_dims": list(plan.embedding_dims),
        "steps": plan.steps,
        "warmup_steps": plan.warmup_steps,
        "repeat": plan.repeat,
        "lanes": [lane.slug for lane in plan.lanes],
        "remote_train_host": context.remote_train_host,
        "remote_repo_root": str(context.remote_repo_root),
        "nnodes": context.nnodes,
        "node_rank": context.node_rank,
        "master_addr": context.master_addr,
        "external_recstore_runtime_dir": (
            str(context.external_recstore_runtime_dir)
            if context.external_recstore_runtime_dir is not None
            else ""
        ),
        "no_start_recstore_server": context.no_start_recstore_server,
        "server_host": context.server_host,
        "server_port0": context.server_port0 if context.server_port0 is not None else "",
        "server_port1": context.server_port1 if context.server_port1 is not None else "",
    }
    metadata.update(_collect_environment_metadata())
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.combine_roots:
        manifest, combined_ps_rows = combine_existing_roots(args.combine_roots, output_root)
    elif args.aggregate_only:
        manifest = _load_manifest(output_root / "manifest.csv")
        combined_ps_rows = []
    else:
        manifest = (
            []
            if args.skip_e2e
            else run_e2e_plan(plan, input_file=args.input_file, context=context, dry_run=args.dry_run)
        )
        combined_ps_rows = []
    _write_csv(output_root / "manifest.csv", manifest)
    summary_rows = collect_e2e_summary(manifest=manifest, output_root=output_root)
    _write_csv(output_root / "summary_e2e.csv", summary_rows)
    gap_rows = build_gap_summary(summary_rows)
    _write_csv(output_root / "summary_gap.csv", gap_rows)

    if args.combine_roots:
        ps_rows = combined_ps_rows
    elif args.skip_rdma_ps:
        ps_rows = _read_csv(output_root / "summary_ps_network.csv") if (output_root / "summary_ps_network.csv").exists() else []
    else:
        ps_rows = run_rdma_ps_calibration(
            output_root=output_root,
            profile=plan.profile,
            dry_run=args.dry_run,
        )
        _write_csv(output_root / "summary_ps_network.csv", ps_rows)

    svg_paths = write_svg_figures(
        output_root,
        summary_rows=summary_rows,
        gap_rows=gap_rows,
        ps_rows=ps_rows,
    )
    metadata["svg_figures"] = [str(path.relative_to(output_root)) for path in svg_paths]
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = render_latex_report(
        summary_rows=summary_rows,
        ps_rows=ps_rows,
        gap_rows=gap_rows,
        metadata=metadata,
    )
    report_path = output_root / "bench_e2e_report.tex"
    report_path.write_text(report, encoding="utf-8")
    print(f"[bench-e2e] output_root={output_root}")
    print(f"[bench-e2e] manifest={output_root / 'manifest.csv'}")
    print(f"[bench-e2e] e2e_summary={output_root / 'summary_e2e.csv'}")
    print(f"[bench-e2e] gap_summary={output_root / 'summary_gap.csv'}")
    print(f"[bench-e2e] ps_summary={output_root / 'summary_ps_network.csv'}")
    print(f"[bench-e2e] latex={report_path}")
    print(f"[bench-e2e] svg_figures={output_root / 'figures'} ({len(svg_paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
