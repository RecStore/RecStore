from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .commands import _run, build_rs_demo_command, format_command, prepare_dataset_slice, wrap_remote_command
from .common import ExecutionContext, ExperimentPlan, ROOT, _gpu_count, _has_rdma, _read_csv


def run_e2e_plan(
    plan: ExperimentPlan,
    *,
    input_file: Path,
    context: ExecutionContext | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    context = context or ExecutionContext()
    manifest: list[dict[str, Any]] = []
    run_index = 0
    detected_gpus = _gpu_count()
    for rows in plan.data_rows:
        dataset_dir = prepare_dataset_slice(
            input_file=input_file,
            rows=rows,
            output_root=plan.output_root,
            dry_run=dry_run,
        )
        for batch_size in plan.batch_sizes:
            train_rows = int(rows * 0.8)
            if batch_size > train_rows:
                continue
            for num_embeddings in plan.num_embeddings:
                for embedding_dim in plan.embedding_dims:
                    for repeat in range(1, plan.repeat + 1):
                        for lane in plan.lanes:
                            if lane.ps_type.upper() == "RDMA" and not _has_rdma():
                                manifest.append(
                                    {
                                        "run_id": "",
                                        "label": lane.label,
                                        "slug": lane.slug,
                                        "backend": lane.backend,
                                        "ps_type": lane.ps_type,
                                        "recstore_index_type": lane.recstore_index_type,
                                        "ps_kv_backend": lane.ps_kv_backend,
                                        "torchrec_memory_mode": lane.torchrec_memory_mode,
                                        "prefetch_depth": lane.prefetch_depth,
                                        "rows": rows,
                                        "batch_size": batch_size,
                                        "num_embeddings": num_embeddings,
                                        "embedding_dim": embedding_dim,
                                        "nproc_per_node": lane.nproc_per_node,
                                        "repeat": repeat,
                                        "status": "skipped",
                                        "exit_code": "",
                                        "skip_reason": "RDMA verbs devices are not available",
                                        "main_csv": "",
                                        "log_path": "",
                                        "command": "",
                                    }
                                )
                                continue
                            if lane.nproc_per_node > max(detected_gpus, 1):
                                manifest.append(
                                    {
                                        "run_id": "",
                                        "label": lane.label,
                                        "slug": lane.slug,
                                        "backend": lane.backend,
                                        "ps_type": lane.ps_type,
                                        "recstore_index_type": lane.recstore_index_type,
                                        "ps_kv_backend": lane.ps_kv_backend,
                                        "torchrec_memory_mode": lane.torchrec_memory_mode,
                                        "prefetch_depth": lane.prefetch_depth,
                                        "rows": rows,
                                        "batch_size": batch_size,
                                        "num_embeddings": num_embeddings,
                                        "embedding_dim": embedding_dim,
                                        "nproc_per_node": lane.nproc_per_node,
                                        "repeat": repeat,
                                        "status": "skipped",
                                        "exit_code": "",
                                        "skip_reason": (
                                            f"requires {lane.nproc_per_node} local CUDA devices, "
                                            f"detected {detected_gpus}"
                                        ),
                                        "main_csv": "",
                                        "log_path": "",
                                        "command": "",
                                    }
                                )
                                continue
                            run_index += 1
                            run_id = (
                                f"bench-{plan.profile}-{lane.slug}-"
                                f"r{rows}-b{batch_size}-n{num_embeddings}-"
                                f"d{embedding_dim}-rep{repeat}"
                            )
                            main_csv = (
                                plan.output_root
                                / "outputs"
                                / run_id
                                / ("torchrec_main.csv" if lane.backend == "torchrec" else "recstore_main.csv")
                            )
                            cmd = build_rs_demo_command(
                                lane=lane,
                                context=context,
                                run_id=run_id,
                                data_dir=dataset_dir,
                                output_root=plan.output_root,
                                rows=rows,
                                batch_size=batch_size,
                                steps=plan.steps,
                                warmup_steps=plan.warmup_steps,
                                num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                master_port=29600 + (run_index % 1000),
                            )
                            run_cmd = (
                                wrap_remote_command(
                                    cmd,
                                    context.remote_train_host,
                                    cwd=context.remote_repo_root,
                                    ssh_port=context.remote_ssh_port,
                                )
                                if context.remote_train_host
                                else cmd
                            )
                            log_path = plan.output_root / "logs" / "e2e" / f"{run_id}.log"
                            status = "dry_run" if dry_run else "ok"
                            exit_code = _run(run_cmd, log_path=log_path, dry_run=dry_run)
                            if not dry_run and (exit_code != 0 or not main_csv.exists()):
                                status = "failed"
                            manifest.append(
                                {
                                    "run_id": run_id,
                                    "label": lane.label,
                                    "slug": lane.slug,
                                    "backend": lane.backend,
                                    "ps_type": lane.ps_type,
                                    "recstore_index_type": lane.recstore_index_type,
                                    "ps_kv_backend": lane.ps_kv_backend,
                                    "torchrec_memory_mode": lane.torchrec_memory_mode,
                                    "prefetch_depth": lane.prefetch_depth,
                                    "rows": rows,
                                    "batch_size": batch_size,
                                    "num_embeddings": num_embeddings,
                                    "embedding_dim": embedding_dim,
                                    "nproc_per_node": lane.nproc_per_node,
                                    "repeat": repeat,
                                    "status": status,
                                    "exit_code": exit_code,
                                    "main_csv": str(main_csv),
                                    "log_path": str(log_path),
                                    "command": format_command(run_cmd),
                                    "remote_train_host": context.remote_train_host,
                                    "server_host": context.server_host,
                                    "recstore_runtime_dir": (
                                        str(context.external_recstore_runtime_dir)
                                        if context.external_recstore_runtime_dir is not None
                                        else ""
                                    ),
                                }
                            )
    return manifest


def run_rdma_ps_calibration(
    *,
    output_root: Path,
    profile: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if not _has_rdma():
        return [{"layer": "PS/network", "status": "skipped", "reason": "no RDMA verbs device"}]
    runtime_seconds = "1" if profile == "smoke" else "5"
    record_count = "10000" if profile == "smoke" else "1000000"
    batch_keys = "64" if profile == "smoke" else "500"
    client_processes = "1" if profile == "smoke" else "6"
    server_threads = "1" if profile == "smoke" else "16"
    out_dir = output_root / "ps_network" / f"rdma_{profile}"
    cmd = [
        sys.executable,
        str(ROOT / "tools/benchmarks/run_benchmark_ps.py"),
        "--transports",
        "rdma",
        "--client-ips",
        "127.0.0.1",
        "--server-shard-ips",
        "127.0.0.1",
        "--client-processes-per-ip",
        client_processes,
        "--record-count",
        record_count,
        "--value-size",
        "512",
        "--batch-keys",
        batch_keys,
        "--index-type",
        "DRAM_PET_HASH",
        "--client-threads-per-process",
        "1",
        "--client-load-threads-per-process",
        "1",
        "--runtime-seconds",
        runtime_seconds,
        "--repeat",
        "1",
        "--execution-backend",
        "local",
        "--prefetch-depth",
        "16",
        "--rdma-rc-qps-per-client-per-shard",
        "16",
        "--rdma-rc-slots-per-qp",
        "1",
        "--server-rdma-threads",
        server_threads,
        "--rdma-rc-server-get-workers",
        "0",
        "--rdma-rc-server-coroutines-per-thread",
        "1",
        "--rdma-get-response-mode",
        "auto",
        "--output-dir",
        str(out_dir),
    ]
    log_path = output_root / "logs" / "ps_network" / f"rdma_{profile}.log"
    exit_code = _run(cmd, log_path=log_path, dry_run=dry_run)
    summary_csv = out_dir / "summary.csv"
    if exit_code != 0 or (not dry_run and not summary_csv.exists()):
        return [
            {
                "layer": "PS/network",
                "status": "failed",
                "exit_code": exit_code,
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "command": " ".join(cmd),
            }
        ]
    rows = _read_csv(summary_csv) if not dry_run else []
    for row in rows:
        row["layer"] = "PS/network"
        row["status"] = "ok"
        row["summary_csv"] = str(summary_csv)
        row["log_path"] = str(log_path)
        row["command"] = " ".join(cmd)
    return rows
