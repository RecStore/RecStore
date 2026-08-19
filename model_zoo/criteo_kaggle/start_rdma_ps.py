#!/usr/bin/env python3
"""Start two-node petps RDMA cluster and wait until SIGTERM."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.benchmarks.e2e.custom.config import BenchmarkConfig, ClientSpec, ServerSpec
from tools.benchmarks.e2e.custom.runtime import (
    _rdma_client_env,
    start_rdma_ps_cluster,
    stop_rdma_ps_cluster,
)


def _spec() -> BenchmarkConfig:
    repo = Path(os.environ.get("REPO_ROOT", str(REPO_ROOT)))
    remote_ssh = os.environ.get("CRITEO_REMOTE_SSH", "root@10.0.2.191")
    remote_port = int(os.environ.get("CRITEO_REMOTE_SSH_PORT", "22222"))
    local_ip = os.environ.get("CRITEO_LOCAL_IP", "10.0.2.192")
    remote_ip = os.environ.get("CRITEO_REMOTE_IP", "10.0.2.191")
    ps_port = int(os.environ.get("CRITEO_PS_PORT", "15000"))
    return BenchmarkConfig(
        clients=(
            ClientSpec(
                ssh_host=remote_ssh,
                ssh_port=remote_port,
                repo_root=repo,
                ip=remote_ip,
                gpu_id=int(os.environ.get("CRITEO_REMOTE_GPU", "0")),
                node_rank=0,
                nproc_per_node=1,
            ),
            ClientSpec(
                ssh_host="local",
                ssh_port=remote_port,
                repo_root=repo,
                ip=local_ip,
                gpu_id=int(os.environ.get("CRITEO_LOCAL_GPU", "3")),
                node_rank=1,
                nproc_per_node=1,
            ),
        ),
        servers=(
            ServerSpec(
                ssh_host="local",
                ssh_port=remote_port,
                repo_root=repo,
                ip=local_ip,
                port=ps_port,
                shard_id=0,
            ),
            ServerSpec(
                ssh_host=remote_ssh,
                ssh_port=remote_port,
                repo_root=repo,
                ip=remote_ip,
                port=ps_port,
                shard_id=1,
            ),
        ),
        batch_size=int(os.environ.get("CRITEO_BATCH_SIZE", "2048")),
        embedding_dim=128,
        output_dir=Path(os.environ.get("CRITEO_OUTPUT_DIR", str(repo / "results" / "criteo_kaggle_train"))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-out", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    cfg = _spec()
    runner = start_rdma_ps_cluster(
        cfg=cfg,
        config_path=Path(args.config),
        log_path=Path(args.log),
        control_plane_host=os.environ.get("CRITEO_LOCAL_IP", "10.0.2.192"),
    )
    env = _rdma_client_env(runner, response_mode="staging_copy")
    env_path = Path(args.env_out)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "".join(f"export {key}={value}\n" for key, value in env.items()),
        encoding="utf-8",
    )

    stop = False

    def _handle(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    try:
        while not stop:
            time.sleep(0.5)
    finally:
        stop_rdma_ps_cluster(runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
