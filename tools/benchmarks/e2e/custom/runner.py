from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..commands import _run, format_command
from ..common import ROOT, _has_rdma, _load_manifest, _write_csv
from .config import BenchmarkConfig, infer_client_deployment, infer_ps_deployment, torchrec_label
from .report import collect_summary_rows, render_summary_md
from .runtime import (
    build_client_command,
    build_runtime_config,
    build_server_command,
    build_torchrec_command,
    start_rdma_ps_cluster,
    stop_rdma_ps_cluster,
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_deployment(path: Path, cfg: BenchmarkConfig, transports: tuple[str, ...]) -> None:
    lines = [
        "# Benchmark E2E Deployment",
        "",
        f"模型: {cfg.model}",
        f"RecStore 传输: {','.join(transports)}",
        f"TorchRec baseline: {','.join(cfg.torchrec_baselines) or 'disabled'}",
        "client:",
    ]
    for client in cfg.clients:
        lines.append(
            f"  - ssh_host={client.ssh_host}, ssh_port={client.ssh_port}, repo={client.repo_root}, "
            f"ip={client.ip}, gpu={client.gpu_id}, node_rank={client.node_rank}, "
            f"nproc_per_node={client.nproc_per_node}"
        )
    lines.append("ps:")
    for server in cfg.servers:
        lines.append(
            f"  - ssh_host={server.ssh_host}, ssh_port={server.ssh_port}, repo={server.repo_root}, "
            f"ip={server.ip}, ps_port={server.port}, shard={server.shard_id}"
        )
    lines.extend(
        [
            f"client 部署: {infer_client_deployment(cfg.clients)}",
            f"PS 部署: {infer_ps_deployment(cfg.servers)}",
            f"分片: {len(cfg.servers)}, hash_method=city_hash, max_keys_per_request=65536",
            f"dataset: {cfg.dataset_path}",
            f"runtime: {cfg.resolved_runtime_dir}",
            f"output: {cfg.output_dir}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _checked_run(cmd: list[str], *, cwd: Path) -> None:
    code = _run(cmd, cwd=cwd)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _start_process(cmd: list[str], *, log_path: Path, cwd: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)


def _stop_processes(processes: list[subprocess.Popen[Any]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_client_group(
    entries: list[dict[str, Any]],
    *,
    dry_run: bool,
    commands: list[str],
    manifest: list[dict[str, Any]],
) -> None:
    for entry in entries:
        commands.append(format_command(entry["cmd"]))
    if dry_run:
        for entry in entries:
            manifest.append({**entry["row"], "status": "dry_run", "exit_code": ""})
        return

    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    try:
        for entry in entries:
            log_path = Path(entry["row"]["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                entry["cmd"],
                cwd=str(entry["cwd"]),
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, log, entry["row"]))
        for proc, log, row in running:
            returncode = proc.wait()
            log.close()
            manifest.append(
                {
                    **row,
                    "status": "ok" if returncode == 0 else "failed",
                    "exit_code": returncode,
                }
            )
    finally:
        for proc, log, _row in running:
            if proc.poll() is None:
                proc.terminate()
            if not log.closed:
                log.close()


def run_custom_benchmark(cfg: BenchmarkConfig, transports: tuple[str, ...], *, dry_run: bool, aggregate_only: bool) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.output_dir / "logs"
    _write_deployment(cfg.output_dir / "deployment.md", cfg, transports)

    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest: list[dict[str, Any]] = []

    if aggregate_only:
        manifest = _load_manifest(cfg.output_dir / "manifest.csv")
        summary_rows = collect_summary_rows(manifest)
        _write_csv(cfg.output_dir / "summary_e2e.csv", summary_rows)
        (cfg.output_dir / "summary.md").write_text(render_summary_md(cfg, summary_rows), encoding="utf-8")
        return 0

    if not cfg.skip_build and not dry_run:
        _checked_run(["cmake", "-S", ".", "-B", "build"], cwd=ROOT)
        build_targets = ["ps_server"]
        if any(transport.upper() == "RDMA" for transport in transports):
            build_targets.append("petps_server")
        _checked_run(["cmake", "--build", "build", "--target", *build_targets, "-j"], cwd=ROOT)
    if not cfg.skip_tests and not dry_run:
        if any(transport.upper() == "RDMA" for transport in transports):
            _checked_run(
                [
                    "ctest",
                    "-R",
                    "test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter|test_allshards_ps_client",
                    "--output-on-failure",
                ],
                cwd=ROOT,
            )
        else:
            _checked_run(
                [
                    "ctest",
                    "-R",
                    "brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client",
                    "--output-on-failure",
                ],
                cwd=ROOT,
            )

    for transport in transports:
        transport_lower = transport.lower()
        runtime_dir = cfg.resolved_runtime_dir / transport_lower
        config_path = runtime_dir / "recstore_config.json"
        runtime = build_runtime_config(
            cfg,
            transport=transport,
            value_path=runtime_dir / "value",
        )
        _write_json(config_path, runtime)
        processes: list[subprocess.Popen[Any]] = []
        rdma_runner = None
        try:
            if transport.upper() != "RDMA":
                for server in cfg.servers:
                    server_cmd = build_server_command(server=server, runtime_config=config_path, transport=transport)
                    commands.append(format_command(server_cmd))
                    if not dry_run:
                        processes.append(
                            _start_process(
                                server_cmd,
                                log_path=logs_dir / f"{transport_lower}_server_shard{server.shard_id}.log",
                                cwd=server.repo_root,
                            )
                        )
                if processes:
                    time.sleep(3.0)
            for repeat_index in range(cfg.repeat):
                if transport.upper() == "RDMA":
                    if not _has_rdma():
                        raise RuntimeError("RDMA verbs devices are not available on the PS host")
                    first_server = sorted(cfg.servers, key=lambda item: item.shard_id)[0]
                    if not dry_run:
                        stop_rdma_ps_cluster(rdma_runner)
                        rdma_runner = start_rdma_ps_cluster(
                            cfg=cfg,
                            config_path=config_path,
                            log_path=logs_dir / f"{transport_lower}_server_shard0_r{repeat_index}.log",
                            control_plane_host=first_server.ip,
                        )
                        _write_json(
                            runtime_dir / "rdma_env.json",
                            {
                                "namespace": rdma_runner.rdma_namespace,
                                "control_plane_host": first_server.ip,
                                "control_plane_port": rdma_runner.rdma_control_plane_port,
                            },
                        )
                        time.sleep(2.0)
                group_run_id = (
                    f"{transport_lower}_b{cfg.batch_size}_d{cfg.embedding_dim}"
                    f"_r{repeat_index}"
                )
                client_entries: list[dict[str, Any]] = []
                for client in cfg.clients:
                    client_cmd = build_client_command(
                        cfg=cfg,
                        transport=transport,
                        client=client,
                        run_id=group_run_id,
                        rdzv_id=group_run_id,
                        rdma_runner=rdma_runner,
                    )
                    log_path = logs_dir / f"{group_run_id}_n{client.node_rank}.log"
                    main_csv = cfg.output_dir / "outputs" / group_run_id / "recstore_main.csv"
                    row = {
                        "run_id": group_run_id,
                        "lane": transport,
                        "backend": "recstore",
                        "transport": transport,
                        "torchrec_memory_mode": "",
                        "client_ip": client.ip,
                        "server_count": len(cfg.servers),
                        "batch_size": cfg.batch_size,
                        "embedding_dim": cfg.embedding_dim,
                        "num_embeddings": cfg.num_embeddings,
                        "repeat_index": repeat_index,
                        "main_csv": str(main_csv),
                        "log_path": str(log_path),
                    }
                    client_entries.append({"cmd": client_cmd, "cwd": client.repo_root, "row": row})
                _run_client_group(
                    client_entries,
                    dry_run=dry_run,
                    commands=commands,
                    manifest=manifest,
                )
        finally:
            stop_rdma_ps_cluster(rdma_runner)
            _stop_processes(processes)

    for memory_mode in cfg.torchrec_baselines:
        lane = torchrec_label(memory_mode)
        mode_slug = memory_mode.replace("_", "")
        for repeat_index in range(cfg.repeat):
            group_run_id = (
                f"torchrec_{mode_slug}_b{cfg.batch_size}_d{cfg.embedding_dim}"
                f"_r{repeat_index}"
            )
            client_entries = []
            for client in cfg.clients:
                client_cmd = build_torchrec_command(
                    cfg=cfg,
                    memory_mode=memory_mode,
                    client=client,
                    run_id=group_run_id,
                    rdzv_id=group_run_id,
                )
                log_path = logs_dir / f"{group_run_id}_n{client.node_rank}.log"
                main_csv = cfg.output_dir / "outputs" / group_run_id / "torchrec_main.csv"
                row = {
                    "run_id": group_run_id,
                    "lane": lane,
                    "backend": "torchrec",
                    "transport": "",
                    "torchrec_memory_mode": memory_mode,
                    "client_ip": client.ip,
                    "server_count": 0,
                    "batch_size": cfg.batch_size,
                    "embedding_dim": cfg.embedding_dim,
                    "num_embeddings": cfg.num_embeddings,
                    "repeat_index": repeat_index,
                    "main_csv": str(main_csv),
                    "log_path": str(log_path),
                }
                client_entries.append({"cmd": client_cmd, "cwd": client.repo_root, "row": row})
            _run_client_group(
                client_entries,
                dry_run=dry_run,
                commands=commands,
                manifest=manifest,
            )

    (cfg.output_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    os.chmod(cfg.output_dir / "commands.sh", 0o755)
    _write_csv(cfg.output_dir / "manifest.csv", manifest)
    summary_rows = collect_summary_rows(manifest)
    _write_csv(cfg.output_dir / "summary_e2e.csv", summary_rows)
    (cfg.output_dir / "summary.md").write_text(render_summary_md(cfg, summary_rows), encoding="utf-8")
    print(f"[benchmark-e2e] output={cfg.output_dir}")
    print(f"[benchmark-e2e] deployment={cfg.output_dir / 'deployment.md'}")
    print(f"[benchmark-e2e] summary={cfg.output_dir / 'summary.md'}")
    failed = [row for row in manifest if str(row.get("status", "")) == "failed"]
    if failed:
        print(f"[benchmark-e2e] failed_runs={len(failed)} manifest={cfg.output_dir / 'manifest.csv'}")
        return 1
    return 0
