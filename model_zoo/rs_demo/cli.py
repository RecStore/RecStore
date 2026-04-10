#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .config import (
    RunConfig,
    ensure_parent_dirs,
    parse_config,
    populate_default_paths,
    validate_recstore_config,
    validate_torchrec_config,
)
from .runtime.report import analyze_embupdate, setup_local_report_env
from .runtime.torchrec_aggregate import aggregate_torchrec_main_csv, write_aggregate_csv
from .runtime.torchrec_compare import build_compare_rows, write_compare_csv
from .runtime.torchrec_trace_report import summarize_trace_dir, write_trace_csv
from .runtime.server import (
    choose_available_ports,
    make_runtime_dir,
    resolve_default_ports,
    start_server,
    stop_server,
    wait_server_ready,
)


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def estimate_recstore_kv_capacity(num_embeddings: int, table_count: int = 26) -> int:
    # Keep the runtime capacity close to the benchmark scale so ps_server
    # initialization does not reserve the oversized repository default.
    return max(int(num_embeddings) * int(table_count) * 2, 100_000)


def build_runner(cfg: RunConfig, runtime_dir: Path):
    if cfg.backend == "recstore":
        from .runners.recstore_runner import RecStoreRunner

        return RecStoreRunner(runtime_dir)
    if cfg.backend == "torchrec":
        try:
            from .runners.torchrec_runner import TorchRecRunner, ensure_torchrec_available
        except ModuleNotFoundError as exc:
            raise RuntimeError("TorchRec runner is not available") from exc
        ensure_torchrec_available()
        return TorchRecRunner(runtime_dir)
    raise ValueError(f"Unsupported backend: {cfg.backend}")


def main(argv: list[str] | None = None) -> int:
    cfg = parse_config(argv)
    populate_default_paths(cfg)
    validate_recstore_config(cfg)
    validate_torchrec_config(cfg)
    is_torchrec_worker = os.environ.get("RS_DEMO_TORCHREC_WORKER") == "1"
    is_recstore_worker = os.environ.get("RS_DEMO_RECSTORE_WORKER") == "1"
    if cfg.backend == "torchrec" and cfg.torchrec_profiler and not is_torchrec_worker:
        run_dir = Path(cfg.torchrec_trace_dir) / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S_%f"
        )
        cfg.torchrec_trace_dir = str(run_dir)
    ensure_parent_dirs(cfg)
    if cfg.backend == "recstore":
        setup_local_report_env(cfg.jsonl)

    if cfg.backend != "recstore":
        cfg.start_server = False

    repo_root = repo_root_from_this_file()
    with open(repo_root / "recstore_config.json", "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    p0_default, p1_default = resolve_default_ports(base_cfg)
    if cfg.server_port0 is None:
        cfg.server_port0 = p0_default
    if cfg.server_port1 is None:
        cfg.server_port1 = p1_default
    server_needed = cfg.backend == "recstore" and cfg.start_server
    if server_needed:
        cfg.server_port0, cfg.server_port1 = choose_available_ports(
            cfg.server_host, cfg.server_port0, cfg.server_port1
        )

    runtime_dir = Path(cfg.output_root) / "runtime" / cfg.run_id
    runtime_cfg_path = runtime_dir / "recstore_config.json"
    if cfg.backend == "recstore":
        runtime_dir, runtime_cfg_path = make_runtime_dir(
            base_cfg=base_cfg,
            host=cfg.server_host,
            port0=cfg.server_port0,
            port1=cfg.server_port1,
            allocator=cfg.allocator,
            output_root=cfg.output_root,
            run_id=cfg.run_id,
            ps_type=cfg.ps_type,
            kv_capacity=estimate_recstore_kv_capacity(cfg.num_embeddings),
        )

    proc = None
    try:
        if server_needed:
            print(f"[rs_demo] starting server ({cfg.ps_type}) with {runtime_cfg_path}")
            proc = start_server(repo_root, runtime_cfg_path, Path(cfg.server_log))
            if not wait_server_ready(
                proc=proc,
                host=cfg.server_host,
                port0=cfg.server_port0,
                port1=cfg.server_port1,
                timeout_s=cfg.server_wait_seconds,
            ):
                raise RuntimeError(
                    f"server failed to become ready: {cfg.server_host}:{cfg.server_port0},{cfg.server_port1}; "
                    f"log={cfg.server_log}"
                )
            print("[rs_demo] server is ready")

        runner = build_runner(cfg, runtime_dir)
        _run_result = runner.run(repo_root, cfg)
        if cfg.backend == "torchrec":
            if is_torchrec_worker:
                return 0
            print(f"[rs_demo] torchrec main csv: {cfg.torchrec_main_csv}")
            agg = aggregate_torchrec_main_csv(Path(cfg.torchrec_main_csv))
            write_aggregate_csv(Path(cfg.torchrec_main_agg_csv), agg)
            print(f"[rs_demo] torchrec main aggregate csv: {cfg.torchrec_main_agg_csv}")
            if cfg.torchrec_profiler:
                rows = summarize_trace_dir(Path(cfg.torchrec_trace_dir))
                write_trace_csv(Path(cfg.torchrec_trace_csv), rows)
                print(f"[rs_demo] torchrec trace csv: {cfg.torchrec_trace_csv}")
            if cfg.torchrec_compare_recstore_csv:
                compare_rows = build_compare_rows(
                    Path(cfg.torchrec_compare_recstore_csv),
                    Path(cfg.torchrec_main_csv),
                )
                write_compare_csv(Path(cfg.torchrec_compare_csv), compare_rows)
                print(f"[rs_demo] torchrec compare csv: {cfg.torchrec_compare_csv}")
            return 0

        if is_recstore_worker:
            return 0
        print("[rs_demo] analyzing embupdate stages...")
        extra_inputs: list[str] = []
        server_log_path = Path(cfg.server_log)
        if server_log_path.exists():
            extra_inputs.append(str(server_log_path))
        analyze_output = analyze_embupdate(
            repo_root,
            cfg.jsonl,
            cfg.csv,
            top_n=20,
            extra_inputs=extra_inputs,
        )
        print(analyze_output)
        agg = aggregate_torchrec_main_csv(Path(cfg.recstore_main_csv))
        write_aggregate_csv(Path(cfg.recstore_main_agg_csv), agg)
        print(f"[rs_demo] recstore main csv: {cfg.recstore_main_csv}")
        print(f"[rs_demo] recstore main aggregate csv: {cfg.recstore_main_agg_csv}")

        print(f"[rs_demo] jsonl: {cfg.jsonl}")
        print(f"[rs_demo] csv:   {cfg.csv}")
        if server_needed:
            print(f"[rs_demo] server log: {cfg.server_log}")
        return 0
    finally:
        stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
