#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
import time

from petps_cluster_runner import PetPSClusterRunner, REPO_ROOT, _to_text
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import materialize_rdma_config, resolve_rdma_integration_config
from run_petps_integration import (
    is_runner_noise_line,
    normalize_timeout,
)


def print_filtered_output(text, show_runner_logs, client_index):
    for line in text.splitlines():
        if not show_runner_logs and is_runner_noise_line(line):
            continue
        print(f"[petps-client:{client_index}] {line}")


def terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_clients_concurrently(runner, argv, client_count, timeout, show_runner_logs):
    env = runner.build_env()
    processes = []
    deadline = time.monotonic() + timeout
    for client_index in range(client_count):
        cmd = runner.build_client_cmd(argv, client_index=client_index)
        processes.append(
            (
                client_index,
                subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                ),
            )
        )

    results = []
    timed_out = False
    for client_index, process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr)
            results.append((client_index, 124, stdout, stderr))
            break
        results.append((client_index, process.returncode, stdout, stderr))

    if timed_out:
        for _client_index, process in processes:
            terminate_process(process)
        seen = {client_index for client_index, *_rest in results}
        for client_index, process in processes:
            if client_index not in seen:
                stdout, stderr = process.communicate()
                results.append((client_index, 124, stdout, stderr))

    for client_index, returncode, stdout, stderr in results:
        print_filtered_output(stdout, show_runner_logs, client_index)
        print_filtered_output(stderr, show_runner_logs, client_index)
        if returncode != 0:
            print(
                f"[petps-client:{client_index}] exited with code {returncode}"
            )

    if timed_out:
        print(f"[petps-client] timed out after {timeout} seconds")
        return 124

    return max((returncode for _idx, returncode, _out, _err in results), default=0)


def main():
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-count", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=2)
    parser.add_argument("--test-binary", required=True)
    parser.add_argument(
        "--gtest-filter",
        default="PetPSIntegrationTest.RepeatedPutGetStressSingleShard",
    )
    parser.add_argument("--config-path")
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument("--max-kv-num-per-request", type=int, default=64)
    parser.add_argument("--client-timeout", type=int, default=45)
    parser.add_argument("--cluster-timeout", type=int, default=25)
    parser.add_argument("--status-refresh-interval", type=float, default=2.0)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show control-plane/status logs from runner and integration binary",
    )
    args = parser.parse_args()

    if args.client_count <= 1:
        raise ValueError("--client-count must be > 1 for multi-client stress")

    source_config_path = resolve_rdma_integration_config(args.server_count, args.config_path)
    client_timeout = normalize_timeout(args.client_timeout, "client-timeout")
    cluster_timeout = normalize_timeout(args.cluster_timeout, "cluster-timeout")

    with tempfile.TemporaryDirectory(prefix="recstore_petps_stress_") as runtime_dir:
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
        )

        with runner.run():
            return run_clients_concurrently(
                runner,
                [args.test_binary, f"--gtest_filter={args.gtest_filter}"],
                args.client_count,
                client_timeout,
                args.show_runner_logs,
            )


if __name__ == "__main__":
    raise SystemExit(main())
