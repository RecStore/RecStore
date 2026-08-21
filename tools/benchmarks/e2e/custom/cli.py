from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    BenchmarkConfig,
    ClientSpec,
    ServerSpec,
    infer_client_deployment,
    infer_ps_deployment,
    parse_client_spec,
    parse_server_spec,
    parse_torchrec_baselines,
    parse_transports,
)
from .report import collect_summary_rows, render_summary_md
from .runner import run_custom_benchmark
from .runtime import build_client_command, build_runtime_config, build_torchrec_command


def _build_config_from_args(args: argparse.Namespace) -> tuple[BenchmarkConfig, tuple[str, ...]]:
    clients = tuple(parse_client_spec(raw) for raw in args.client) if args.client else (ClientSpec(),)
    servers = tuple(parse_server_spec(raw) for raw in args.ps) if args.ps else (ServerSpec(),)
    output_dir = Path(args.output_dir)
    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else output_dir / "runtime"
    cfg = BenchmarkConfig(
        clients=clients,
        servers=servers,
        output_dir=output_dir,
        runtime_dir=runtime_dir,
        dataset_path=Path(args.data_dir),
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        init_rows=args.init_rows,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeat=args.repeat,
        read_mode=args.read_mode,
        prefetch_depth=args.prefetch_depth,
        optimization_cache_capacity=args.optimization_cache_capacity,
        index_type=args.index_type,
        torchrec_baselines=() if args.no_torchrec else parse_torchrec_baselines(args.torchrec_baselines),
        master_port=args.master_port,
        python_bin=args.python_bin,
        skip_build=args.skip_build,
        skip_tests=args.skip_tests,
    )
    return cfg, parse_transports(args.transports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RecStore BRPC/GRPC E2E benchmarks.")
    parser.add_argument("--client", action="append", default=[])
    parser.add_argument("--ps", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--transports", default="brpc")
    parser.add_argument("--data-dir", default="model_zoo/torchrec_dlrm/processed_day_0_data")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument("--init-rows", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--torchrec-baselines", default="hbm")
    parser.add_argument("--no-torchrec", action="store_true")
    parser.add_argument(
        "--read-mode",
        choices=["direct", "prefetch", "bagpipe"],
        default="prefetch",
    )
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument(
        "--optimization-cache-capacity",
        type=int,
        default=0,
        help="GPU cache capacity for optimization plugins (e.g. bagpipe); "
        "0 keeps the rs_demo default (bagpipe: 160000).",
    )
    parser.add_argument(
        "--index-type",
        choices=["DRAM_PET_HASH", "DRAM_EXTENDIBLE_HASH", "DRAM_UNORDERED_MAP"],
        default="DRAM_PET_HASH",
    )
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    cfg, transports = _build_config_from_args(args)
    return run_custom_benchmark(
        cfg,
        transports,
        dry_run=args.dry_run,
        aggregate_only=args.aggregate_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
