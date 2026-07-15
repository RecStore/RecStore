from __future__ import annotations

import shlex
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import DEFAULT_DAY0, ExecutionContext, E2ELane, ROOT, _dense_arch_for_embedding_dim


def _run(cmd: list[str], *, cwd: Path = ROOT, log_path: Path | None = None, dry_run: bool = False) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("$ " + format_command(cmd) + "\n")
    if dry_run:
        return 0
    start = time.time()
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as sink:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
            )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[exit_code] {proc.returncode}\n[elapsed_s] {time.time() - start:.3f}\n")
    return int(proc.returncode)


def prepare_dataset_slice(
    *,
    input_file: Path,
    rows: int,
    output_root: Path,
    dry_run: bool = False,
) -> Path:
    out_dir = output_root / "datasets" / f"criteo_day0_rows{rows}"
    marker = out_dir / "metadata.json"
    if marker.exists():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if int(metadata.get("accepted_rows", 0)) >= int(rows):
                return out_dir
        except json.JSONDecodeError:
            pass
    raw_dir = output_root / "raw_slices" / f"rows{rows}"
    cmd = [
        sys.executable,
        str(ROOT / "model_zoo/torchrec_dlrm/scripts/slice_preprocess_single_day.py"),
        "--input-file",
        str(input_file),
        "--output-raw-dir",
        str(raw_dir),
        "--output-dir",
        str(out_dir),
        "--rows",
        str(rows),
        "--seed",
        "20260330",
        "--progress-interval",
        "100000",
    ]
    log_path = output_root / "logs" / "dataset" / f"slice_rows{rows}.log"
    code = _run(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"dataset slicing failed for rows={rows}; log={log_path}")
    return out_dir


def build_rs_demo_command(
    *,
    lane: E2ELane,
    context: ExecutionContext,
    run_id: str,
    data_dir: Path,
    output_root: Path,
    rows: int,
    batch_size: int,
    steps: int,
    warmup_steps: int,
    num_embeddings: int,
    embedding_dim: int,
    master_port: int,
) -> list[str]:
    del rows
    cmd = [
        context.python_bin,
        str(context.remote_repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        lane.backend,
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--data-dir",
        str(data_dir),
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
        "--batch-size",
        str(batch_size),
        "--num-embeddings",
        str(num_embeddings),
        "--embedding-dim",
        str(embedding_dim),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(embedding_dim),
        "--nnodes",
        str(context.nnodes),
        "--node-rank",
        str(context.node_rank),
        "--nproc-per-node",
        str(lane.nproc_per_node),
        "--master-addr",
        context.master_addr,
        "--master-port",
        str(master_port),
        "--rdzv-id",
        run_id,
    ]
    if lane.backend == "torchrec":
        cmd.extend(["--no-start-server", "--torchrec-memory-mode", lane.torchrec_memory_mode])
    else:
        cmd.extend(
            [
                "--ps-type",
                lane.ps_type,
                "--recstore-index-type",
                lane.recstore_index_type,
                "--ps-kv-backend",
                lane.ps_kv_backend,
                "--allocator",
                lane.allocator,
                "--prefetch-depth",
                str(lane.prefetch_depth),
            ]
        )
        if lane.enable_single_node_fast_path:
            cmd.extend(
                [
                    "--enable-single-node-distributed-fast-path",
                    "--single-node-ps-backend",
                    lane.single_node_ps_backend,
                ]
            )
        if context.external_recstore_runtime_dir is not None:
            cmd.extend(["--recstore-runtime-dir", str(context.external_recstore_runtime_dir)])
        if context.no_start_recstore_server:
            cmd.append("--no-start-server")
        if context.server_host:
            cmd.extend(["--server-host", context.server_host])
        if context.server_port0 is not None:
            cmd.extend(["--server-port0", str(context.server_port0)])
        if context.server_port1 is not None:
            cmd.extend(["--server-port1", str(context.server_port1)])
    cmd.extend(lane.extra_args)
    return cmd


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def wrap_remote_command(
    cmd: list[str],
    host: str,
    *,
    cwd: Path,
    ssh_port: int = 22,
) -> list[str]:
    remote = "cd {cwd} && {cmd}".format(
        cwd=shlex.quote(str(cwd)),
        cmd=" ".join(shlex.quote(part) for part in cmd),
    )
    ssh_cmd = ["ssh"]
    if ssh_port != 22:
        ssh_cmd.extend(["-p", str(ssh_port)])
    ssh_cmd.extend([host.strip(), remote])
    return ssh_cmd
