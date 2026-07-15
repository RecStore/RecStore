#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from petps_cluster_runner import PetPSClusterRunner
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def infer_server_count(config_path):
    with resolve_repo_path(config_path).open() as fh:
        config = json.load(fh)
    distributed = config.get("distributed_client", {})
    cache_ps = config.get("cache_ps", {})
    return (
        distributed.get("num_shards")
        or cache_ps.get("num_shards")
        or len(distributed.get("servers", []))
        or len(cache_ps.get("servers", []))
        or 1
    )


def main():
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    parser = argparse.ArgumentParser(
        description=(
            "Run RDMA petps_server cluster with shard-0 control-plane lifecycle. "
            "Designed to align with ps_server-style config_path startup."
        )
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--server-count", type=int)
    parser.add_argument("--client-count", type=int, default=1)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument("--max-kv-num-per-request", type=int, default=64)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--rdma-per-thread-response-limit-bytes", type=int)
    parser.add_argument("--rdma-client-receive-arena-bytes", type=int)
    parser.add_argument("--validate-routing", action="store_true")
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show control-plane/status logs from PetPSClusterRunner",
    )
    args = parser.parse_args()

    server_count = args.server_count or infer_server_count(args.config_path)
    runner = PetPSClusterRunner(
        config_path=args.config_path,
        num_servers=server_count,
        num_clients=args.client_count,
        thread_num=args.thread_num,
        value_size=args.value_size,
        max_kv_num_per_request=args.max_kv_num_per_request,
        timeout=args.timeout,
        verbose=args.show_runner_logs,
        show_status_logs=args.show_runner_logs,
        show_control_plane_logs=args.show_runner_logs,
        rdma_namespace=args.rdma_namespace,
        rdma_control_plane_host=args.rdma_control_plane_host,
        rdma_control_plane_port=args.rdma_control_plane_port,
        rdma_per_thread_response_limit_bytes=args.rdma_per_thread_response_limit_bytes,
        rdma_client_receive_arena_bytes=args.rdma_client_receive_arena_bytes,
        validate_routing=args.validate_routing,
    )

    print(
        f"[petps-launch] config={args.config_path} servers={server_count} "
        f"clients={args.client_count} control_plane="
        f"{runner.rdma_control_plane_host}:{runner.rdma_control_plane_port}",
        flush=True,
    )
    runner.start()
    print(
        "[petps-launch] petps server cluster is running. Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[petps-launch] stopping petps server cluster...", flush=True)
    finally:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
