#!/usr/bin/env python3

import argparse
import contextlib
import os
import re
import tempfile
import time
from pathlib import Path

from petps_cluster_runner import PetPSClusterRunner
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    REPO_ROOT,
    materialize_rdma_config,
    resolve_rdma_integration_config,
)
CONTROL_PLANE_NOISE_PATTERNS = (
    "[petps-control-plane]",
    "component=rdma_control_plane",
)
FENCE_STATS_EVENT = "event=rebuild_fence_stats"
FENCE_EVENTS_PATTERN = re.compile(r"fence_events=(\d+)")


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


def max_fence_events(server_lines):
    """Highest cumulative rebuild-fence count reported by shard-0 so far."""
    best = 0
    for line in server_lines:
        if FENCE_STATS_EVENT not in line:
            continue
        match = FENCE_EVENTS_PATTERN.search(line)
        if match is not None:
            best = max(best, int(match.group(1)))
    return best


def fence_report_lines(server_lines):
    """How many `rebuild_fence_stats` reports shard-0 actually emitted.

    Distinguishes "the fence did not fire" (reports present, count 0) from
    "the counter was never measured" (no reports at all, e.g. the server was
    started without the opt-in report interval).
    """
    return sum(1 for line in server_lines if FENCE_STATS_EVENT in line)


def wait_for_fence_events(runner, timeout):
    """Shard-0 fence events, waiting up to `timeout` for the periodic report.

    The counter line is emitted on a fixed interval by the server poll thread
    (the runner SIGKILLs the server, so there is no exit-time flush). A short
    wait after the client finishes lets the last interval land before we read.
    """
    deadline = time.monotonic() + timeout
    while True:
        events = max_fence_events(runner.process_logs.get(0, []))
        if events > 0 or time.monotonic() >= deadline:
            return events
        time.sleep(0.25)


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
    parser.add_argument(
        "--logical-clients-per-process",
        type=int,
        help="logical RC lanes per client process (default: 1)",
    )
    parser.add_argument(
        "--rdma-qps-per-client-per-shard",
        type=int,
        help="RC write QPs per client per shard (default: binary default)",
    )
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
    parser.add_argument(
        "--rdma-rc-server-get-worker-hold-us",
        type=int,
        help=(
            "test-only: hold each GET payload worker for this many "
            "microseconds so a lane rebuild provably straddles an in-flight "
            "task and the response-epoch fence fires"
        ),
    )
    parser.add_argument(
        "--rdma-fake-get-mode",
        help=(
            "test-only: server-side GET fault injection "
            "(none/status_only/index_only/payload_memset/throw)"
        ),
    )
    parser.add_argument(
        "--rdma-lane-rebuild",
        action="store_true",
        help="enable lane/QP rebuild on both server and client",
    )
    parser.add_argument("--validate-routing", action="store_true")
    parser.add_argument(
        "--client-ssh",
        default="",
        help=(
            "run the client on this SSH target (user@host) instead of locally; "
            "sshpass must be available and SSHPASS set in the environment"
        ),
    )
    parser.add_argument(
        "--client-ssh-port",
        type=int,
        default=22222,
        help="SSH port for --client-ssh",
    )
    parser.add_argument(
        "--runtime-dir",
        help=(
            "directory for the materialized RDMA config; required to be "
            "readable by every host (the remote client reads it via "
            "RECSTORE_CONFIG). Defaults to a temporary directory locally, or "
            "<repo>/.cross-run when --client-ssh is used"
        ),
    )
    parser.add_argument(
        "--expect-fence-events",
        action="store_true",
        help=(
            "fail unless shard-0 reports at least one rebuild-fence event "
            "(fence_rejected + dropped_stale > 0); use on reruns that must "
            "prove the rebuild fence fired"
        ),
    )
    parser.add_argument(
        "--fence-stats-interval-ms",
        type=int,
        default=1000,
        help=(
            "how often the server reports its rebuild-fence counters "
            "(--rdma-rc-server-fence-stats-interval-ms; 0 disables the report, "
            "which is the production default). Only applied with "
            "--rdma-lane-rebuild"
        ),
    )
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show control-plane/status logs from runner and integration binary",
    )
    args = parser.parse_args()
    source_config_path = resolve_rdma_integration_config(args.server_count, args.config_path)
    client_timeout = normalize_timeout(args.client_timeout, "client-timeout")
    cluster_timeout = normalize_timeout(args.cluster_timeout, "cluster-timeout")
    test_binary = Path(args.test_binary)
    if not test_binary.is_absolute():
        test_binary = (REPO_ROOT / test_binary).resolve()
    test_binary = str(test_binary)

    if args.client_ssh:
        runtime_path = (
            Path(args.runtime_dir)
            if args.runtime_dir
            else REPO_ROOT / ".cross-run"
        )
        runtime_path.mkdir(parents=True, exist_ok=True)
        runtime_context = contextlib.nullcontext(str(runtime_path))
    else:
        runtime_context = tempfile.TemporaryDirectory(
            prefix="recstore_petps_integration_"
        )

    with runtime_context as runtime_dir:
        config_path = materialize_rdma_config(
            source_config_path,
            args.server_count,
            args.client_count,
            runtime_dir,
            device=os.environ.get("RECSTORE_RDMA_TEST_DEVICE"),
        )
        runner = PetPSClusterRunner(
            config_path=config_path,
            num_servers=args.server_count,
            num_clients=args.client_count,
            thread_num=1,
            value_size=args.value_size,
            max_kv_num_per_request=args.max_kv_num_per_request,
            timeout=cluster_timeout,
            logical_clients_per_process=args.logical_clients_per_process or 1,
            rdma_qps_per_client_per_shard=args.rdma_qps_per_client_per_shard,
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
            rdma_rc_server_get_worker_hold_us=(
                args.rdma_rc_server_get_worker_hold_us
            ),
            rdma_lane_rebuild_enabled=args.rdma_lane_rebuild or None,
            rdma_rc_fake_get_mode=args.rdma_fake_get_mode,
            rdma_rc_server_fence_stats_interval_ms=(
                args.fence_stats_interval_ms if args.rdma_lane_rebuild else None
            ),
            validate_routing=args.validate_routing,
            client_ssh=args.client_ssh or None,
            client_ssh_port=args.client_ssh_port,
        )

        with runner.run():
            client_argv = [
                test_binary,
                f"--gtest_filter={args.gtest_filter}",
            ]
            if args.rdma_rc_server_get_worker_hold_us:
                # Test-only: the server stalls its GET payload workers for this
                # lane, so the integration test sees the same value and skips
                # its GET-throughput expectation (the driver asserts the fence
                # counters instead).
                client_argv.append(
                    "--rdma_rc_server_get_worker_hold_us="
                    f"{args.rdma_rc_server_get_worker_hold_us}"
                )
            completed = runner.run_client(
                client_argv,
                timeout=client_timeout,
                stream_output=False,
            )
            print_filtered_output(completed.stdout, args.show_runner_logs)
            print_filtered_output(completed.stderr, args.show_runner_logs)
            fence_events = max_fence_events(runner.process_logs.get(0, []))
            report_lines = fence_report_lines(runner.process_logs.get(0, []))
            if args.expect_fence_events and fence_events == 0:
                fence_events = wait_for_fence_events(
                    runner, timeout=min(5.0, float(client_timeout))
                )
                report_lines = fence_report_lines(runner.process_logs.get(0, []))
            print(
                f"[petps-fence] shard0_report_lines={report_lines} "
                f"shard0_fence_events={fence_events}"
            )
            if args.expect_fence_events and report_lines == 0:
                print(
                    "[petps-fence] ERROR: shard-0 emitted no fence-stats report, "
                    "so the counter was never measured (is "
                    "--fence-stats-interval-ms 0?)"
                )
                return 1
            if args.expect_fence_events and fence_events == 0:
                print(
                    "[petps-fence] ERROR: expected the rebuild fence to fire, "
                    "but shard-0 reported no fence_rejected/dropped_stale events"
                )
                return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
