#!/usr/bin/env python3

import argparse
import tempfile

from petps_cluster_runner import PetPSClusterRunner
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    materialize_rdma_config,
    resolve_rdma_integration_config,
)
CONTROL_PLANE_NOISE_PATTERNS = (
    "[petps-control-plane]",
    "component=rdma_control_plane",
)


def normalize_timeout(timeout_seconds, field_name):
    if timeout_seconds <= 0:
        raise ValueError(f"{field_name} must be > 0, got {timeout_seconds}")
    return timeout_seconds


def is_runner_noise_line(line):
    return any(pattern in line for pattern in CONTROL_PLANE_NOISE_PATTERNS)


def print_filtered_output(text, show_runner_logs):
    for line in text.splitlines():
        if not show_runner_logs and is_runner_noise_line(line):
            continue
        print(line)


def main():
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-count", type=int, required=True)
    parser.add_argument("--test-binary", required=True)
    parser.add_argument("--gtest-filter", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument("--max-kv-num-per-request", type=int, default=64)
    parser.add_argument("--client-count", type=int, default=1)
    parser.add_argument("--client-timeout", type=int, default=15)
    parser.add_argument("--cluster-timeout", type=int, default=15)
    parser.add_argument("--status-refresh-interval", type=float, default=2.0)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument("--rdma-per-thread-response-limit-bytes", type=int)
    parser.add_argument("--rdma-client-receive-arena-bytes", type=int)
    parser.add_argument("--rdma-rc-server-coroutines-per-thread", type=int)
    parser.add_argument("--rdma-rc-server-get-workers", type=int)
    parser.add_argument("--validate-routing", action="store_true")
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show control-plane/status logs from runner and integration binary",
    )
    args = parser.parse_args()
    source_config_path = resolve_rdma_integration_config(args.server_count, args.config_path)
    client_timeout = normalize_timeout(args.client_timeout, "client-timeout")
    cluster_timeout = normalize_timeout(args.cluster_timeout, "cluster-timeout")

    with tempfile.TemporaryDirectory(prefix="recstore_petps_integration_") as runtime_dir:
        config_path = materialize_rdma_config(
            source_config_path, args.server_count, args.client_count, runtime_dir
        )
        runner = PetPSClusterRunner(
            config_path=config_path,
            num_servers=args.server_count,
            num_clients=args.client_count,
            thread_num=1,
            value_size=args.value_size,
            max_kv_num_per_request=args.max_kv_num_per_request,
            timeout=cluster_timeout,
            verbose=args.show_runner_logs,
            status_refresh_interval=args.status_refresh_interval,
            show_status_logs=args.show_runner_logs,
            show_control_plane_logs=args.show_runner_logs,
            rdma_namespace=args.rdma_namespace,
            rdma_control_plane_host=args.rdma_control_plane_host,
            rdma_control_plane_port=args.rdma_control_plane_port,
            rdma_per_thread_response_limit_bytes=args.rdma_per_thread_response_limit_bytes,
            rdma_client_receive_arena_bytes=args.rdma_client_receive_arena_bytes,
            rdma_rc_server_coroutines_per_thread=(
                args.rdma_rc_server_coroutines_per_thread
            ),
            rdma_server_get_workers=args.rdma_rc_server_get_workers,
            validate_routing=args.validate_routing,
        )

        with runner.run():
            completed = runner.run_client(
                [args.test_binary, f"--gtest_filter={args.gtest_filter}"],
                timeout=client_timeout,
                stream_output=False,
            )
            print_filtered_output(completed.stdout, args.show_runner_logs)
            print_filtered_output(completed.stderr, args.show_runner_logs)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
