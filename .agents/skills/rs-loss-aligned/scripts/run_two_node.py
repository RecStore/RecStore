#!/usr/bin/env python3
"""Default two-node RecStore / TorchRec loss-aligned run.

Must run from 10.0.2.192. Injects --seed and --torchrec-align-recstore-init;
do not replace this with tools.benchmarks.e2e.custom.cli as-is.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CLIENTS = (
    "ssh=root@10.0.2.191,ssh_port=22222,repo={repo},"
    "ip=10.0.2.191,gpu=0,node_rank=0,nproc=1",
    "ssh=local,ssh_port=22222,repo={repo},"
    "ip=10.0.2.192,gpu=1,node_rank=1,nproc=1",
)
DEFAULT_SERVERS = (
    "ssh=local,ssh_port=22222,repo={repo},ip=10.0.2.192,port=15000,shard=0",
    "ssh=root@10.0.2.191,ssh_port=22222,repo={repo},ip=10.0.2.191,port=15000,shard=1",
)
CONTROL_PLANE_HOST = "10.0.2.192"
REMOTE_HOST = "root@10.0.2.191"


def local_ips() -> set[str]:
    found = {"127.0.0.1", "localhost"}
    try:
        found.update(subprocess.check_output(["hostname", "-I"], text=True).split())
    except (OSError, subprocess.SubprocessError):
        pass
    return found


def enable_sshpass(password: str) -> str:
    real_ssh = shutil.which("ssh")
    if real_ssh is None:
        raise RuntimeError("ssh not found")
    if shutil.which("sshpass") is None:
        raise RuntimeError("sshpass not found")
    shim_dir = tempfile.mkdtemp(prefix="loss-aligned-ssh-")
    shim = Path(shim_dir) / "ssh"
    shim.write_text(
        "#!/bin/sh\n"
        f"exec sshpass -e {shlex.quote(real_ssh)} "
        '-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o700)
    os.environ["SSHPASS"] = password
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ['PATH']}"
    return shim_dir


def inject_args(cmd: list[str], extra: list[str]) -> list[str]:
    extra_s = " ".join(shlex.quote(part) for part in extra)
    if cmd and cmd[0] == "ssh":
        out = list(cmd)
        out[-1] = f"{out[-1]} {extra_s}"
        return out
    return cmd + extra


def kill_hosts(repo: Path, ssh_port: int) -> None:
    body = (repo / "tools/benchmarks/kill_bench_procs.sh").read_text(encoding="utf-8")
    for host in ("local", REMOTE_HOST):
        cmd = ["bash", "-s"] if host == "local" else ["ssh", "-p", str(ssh_port), host, "bash -s"]
        proc = subprocess.run(cmd, input=body, text=True, cwd=str(repo))
        if proc.returncode != 0:
            raise RuntimeError(f"kill_bench_procs failed on {host}: {proc.returncode}")


def run_group(entries: list[dict], extra: list[str], cwd: Path) -> int:
    running = []
    try:
        for entry in entries:
            cmd = inject_args(entry["cmd"], extra)
            log_path = Path(entry["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((proc, log, entry["log"]))
        codes = []
        for proc, log, log_path in running:
            code = proc.wait()
            log.write(f"\n[exit_code] {code}\n")
            log.close()
            codes.append(code)
            print(f"[loss-aligned] log={log_path} exit={code}", flush=True)
        return 0 if all(code == 0 for code in codes) else 1
    finally:
        for proc, log, _ in running:
            if proc.poll() is None:
                proc.terminate()
            if not log.closed:
                log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="two-node")
    parser.add_argument("--repo", default="")
    parser.add_argument("--data-dir", default="model_zoo/torchrec_dlrm/processed_day_0_data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument(
        "--read-mode",
        choices=["direct"],
        default="direct",
        help="Loss-aligned two-node path is direct only.",
    )
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--ssh-port", type=int, default=22222)
    parser.add_argument("--ssh-pass", default=os.environ.get("SSHPASS", "1234"))
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path.cwd()
    if not (repo / "model_zoo/rs_demo/run_mock_stress.py").exists():
        raise RuntimeError(f"not a RecStore checkout: {repo}")
    if CONTROL_PLANE_HOST not in local_ips():
        raise RuntimeError(
            f"run this script on {CONTROL_PLANE_HOST} (RDMA control-plane / PS shard0)"
        )

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (repo / out).resolve()

    sys.path.insert(0, str(repo))
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo / "src/python/pytorch"),
            str(repo / "src"),
            *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []),
        ]
    )

    from tools.benchmarks.e2e.common import _has_rdma
    from tools.benchmarks.e2e.custom.config import (
        BenchmarkConfig,
        parse_client_spec,
        parse_server_spec,
    )
    from tools.benchmarks.e2e.custom.runtime import (
        build_client_command,
        build_runtime_config,
        build_torchrec_command,
        start_rdma_ps_cluster,
        stop_rdma_ps_cluster,
    )

    if not _has_rdma():
        raise RuntimeError("RDMA verbs devices are not available on this host")

    shim = enable_sshpass(args.ssh_pass)
    repo_s = str(repo)
    clients = tuple(parse_client_spec(spec.format(repo=repo_s)) for spec in DEFAULT_CLIENTS)
    servers = tuple(parse_server_spec(spec.format(repo=repo_s)) for spec in DEFAULT_SERVERS)
    cfg = BenchmarkConfig(
        clients=clients,
        servers=servers,
        output_dir=out,
        runtime_dir=out / "runtime",
        dataset_path=Path(args.data_dir),
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeat=1,
        read_mode="direct",
        prefetch_depth=0,
        index_type="DRAM_PET_HASH",
        torchrec_baselines=("hbm",),
        master_port=args.master_port,
        python_bin=args.python_bin,
        skip_build=True,
        skip_tests=True,
    )
    extra_common = ["--seed", str(args.seed)]
    extra_torchrec = extra_common + ["--torchrec-align-recstore-init"]

    logs_dir = out / "logs"
    runtime_dir = cfg.resolved_runtime_dir / "rdma"
    config_path = runtime_dir / "recstore_config.json"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            build_runtime_config(cfg, transport="RDMA", value_path=runtime_dir / "value"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[loss-aligned] kill leftover processes", flush=True)
    kill_hosts(repo, args.ssh_port)

    rdma_runner = None
    recstore_rc = 1
    torchrec_rc = 1
    try:
        print("[loss-aligned] start RDMA PS", flush=True)
        rdma_runner = start_rdma_ps_cluster(
            cfg=cfg,
            config_path=config_path,
            log_path=logs_dir / "rdma_server.log",
            control_plane_host=CONTROL_PLANE_HOST,
        )
        (runtime_dir / "rdma_env.json").write_text(
            json.dumps(
                {
                    "namespace": rdma_runner.rdma_namespace,
                    "control_plane_host": CONTROL_PLANE_HOST,
                    "control_plane_port": rdma_runner.rdma_control_plane_port,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        recstore_entries = [
            {
                "cmd": build_client_command(
                    cfg=cfg,
                    transport="RDMA",
                    client=client,
                    run_id=args.run_id,
                    rdzv_id=args.run_id,
                    rdma_runner=rdma_runner,
                ),
                "log": str(logs_dir / f"{args.run_id}_recstore_n{client.node_rank}.log"),
            }
            for client in clients
        ]
        print("[loss-aligned] RecStore 2-node", flush=True)
        recstore_rc = run_group(recstore_entries, extra_common, repo)
    finally:
        stop_rdma_ps_cluster(rdma_runner)
        kill_hosts(repo, args.ssh_port)

    if recstore_rc != 0:
        print("[loss-aligned] RecStore failed; skip TorchRec", flush=True)
        shutil.rmtree(shim, ignore_errors=True)
        return recstore_rc

    try:
        torchrec_entries = [
            {
                "cmd": build_torchrec_command(
                    cfg=cfg,
                    memory_mode="hbm",
                    client=client,
                    run_id=args.run_id,
                    rdzv_id=args.run_id,
                ),
                "log": str(logs_dir / f"{args.run_id}_torchrec_n{client.node_rank}.log"),
            }
            for client in clients
        ]
        print("[loss-aligned] TorchRec 2-node", flush=True)
        torchrec_rc = run_group(torchrec_entries, extra_torchrec, repo)
    finally:
        kill_hosts(repo, args.ssh_port)
        shutil.rmtree(shim, ignore_errors=True)

    return torchrec_rc


if __name__ == "__main__":
    raise SystemExit(main())
