#!/usr/bin/env python3
"""Parent process for the env-gated RDMA fault-injection integration tests.

Three PetPSIntegrationTest cases need an external parent that faults shard 1 at
a precise moment: the test prints a warm-up marker on stdout and then expects
the parent to freeze or kill the peer shard. The test bodies assert the
semantics (verdict, latency, isolation); this script only orchestrates the
2-shard loopback cluster (shard 0 hosts the control plane) and the fault.

  fast_fail  SIGSTOP shard 1, wait until every lane is retired, then SIGCONT it
  rebuild    SIGSTOP shard 1, hold it for --freeze-seconds, then SIGCONT it
  killed     SIGKILL shard 1

The scenarios are registered as ctest tests behind ENABLE_RDMA_INTEGRATION_TESTS
(see src/ps/rdma/CMakeLists.txt).

Exit codes: 0 = passed, 1 = failed, 77 = RDMA unavailable on this host.
"""

import argparse
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from petps_cluster_runner import REPO_ROOT, PetPSClusterRunner
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import materialize_rdma_config, resolve_rdma_integration_config

SERVER_COUNT = 2
FAULT_SHARD = 1
CLIENT_INDEX = 0
MARKER_POLL_INTERVAL = 0.1
WARM_MARKER_TIMEOUT = 90
DIRTY_MARKER_TIMEOUT = 90
LOG_TAIL_LINES = 40


class Scenario:
    """One env-gated test and the fault that has to be injected for it."""

    def __init__(
        self,
        name,
        gtest_filter,
        env_var,
        warm_marker,
        fault,
        lanes,
        wait_timeout_ms,
        server_flags,
        client_flags,
        dirty_marker=None,
        lane_rebuild=False,
    ):
        self.name = name
        self.gtest_filter = gtest_filter
        self.env_var = env_var
        self.warm_marker = warm_marker
        self.dirty_marker = dirty_marker
        self.fault = fault
        self.lanes = lanes
        self.wait_timeout_ms = wait_timeout_ms
        self.server_flags = server_flags
        self.client_flags = client_flags
        self.lane_rebuild = lane_rebuild


SCENARIOS = {
    "fast_fail": Scenario(
        name="fast_fail",
        gtest_filter=(
            "PetPSIntegrationTest.ShardUnreachableRetiresLanesAndFailsFast"
        ),
        env_var="RECSTORE_RDMA_FAST_FAIL_TEST",
        warm_marker="RECSTORE_FAST_FAIL_WARM_READY",
        dirty_marker="RECSTORE_FAST_FAIL_ALL_LANES_DIRTY",
        fault="freeze_until_dirty",
        lanes=4,
        # The slow path has to beat the L3 budget by a wide margin.
        wait_timeout_ms=20000,
        server_flags=[
            "--logtostderr=1",
            "--rdma_heartbeat_interval_ms=500",
            "--rdma_health_stale_ms=1500",
            "--rdma_health_dead_ms=3000",
        ],
        client_flags=[
            "--logtostderr=1",
            "--v=1",
            "--rdma-health-probe-after-ms=500",
            "--rdma-health-probe-interval-ms=500",
            "--rdma-health-probe-timeout-ms=1000",
        ],
    ),
    "rebuild": Scenario(
        name="rebuild",
        gtest_filter="PetPSIntegrationTest.LaneRebuildRecoversAfterFreeze",
        env_var="RECSTORE_RDMA_REBUILD_RECOVERY_TEST",
        warm_marker="RECSTORE_REBUILD_WARM_READY",
        fault="freeze_then_thaw",
        lanes=2,
        wait_timeout_ms=8000,
        lane_rebuild=True,
        server_flags=["--logtostderr=1", "--rdma_heartbeat_interval_ms=500"],
        client_flags=[
            "--logtostderr=1",
            "--v=1",
            "--rdma-lane-rebuild-max-attempts=50",
            "--rdma-lane-rebuild-wait-ms=800",
            "--rdma-lane-rebuild-cooldown-ms=500",
        ],
    ),
    "killed": Scenario(
        name="killed",
        gtest_filter=(
            "PetPSIntegrationTest.KilledShardRetiresLanesThroughCompletionError"
        ),
        env_var="RECSTORE_RDMA_DEVICE_FAIL_TEST",
        warm_marker="RECSTORE_KILLED_SHARD_WARM_READY",
        dirty_marker="RECSTORE_KILLED_SHARD_ALL_LANES_DIRTY",
        fault="kill",
        lanes=4,
        # Product-default L3: the device layer has to beat it by a wide margin.
        wait_timeout_ms=60000,
        server_flags=[
            "--logtostderr=1",
            "--rdma_heartbeat_interval_ms=500",
            "--rdma_health_stale_ms=1500",
            "--rdma_health_dead_ms=3000",
        ],
        client_flags=[
            "--logtostderr=1",
            "--v=1",
            # Disable the slow path so only the device layer can fail the request.
            "--rdma-health-probe-after-ms=0",
            "--rdma-cq-harvest-interval-ms=50",
        ],
    ),
}


def wait_for_marker(log_path, marker, timeout):
    if marker is None:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if marker in log_path.read_text(errors="replace"):
                return True
        except FileNotFoundError:
            pass
        time.sleep(MARKER_POLL_INTERVAL)
    return False


def inject_fault(spec, shard1, log_path, freeze_seconds):
    if spec.fault == "kill":
        print(f"[fault-injection] SIGKILL shard {FAULT_SHARD} pid={shard1.pid}")
        shard1.kill()
        return True

    print(f"[fault-injection] SIGSTOP shard {FAULT_SHARD} pid={shard1.pid}")
    shard1.send_signal(signal.SIGSTOP)
    if spec.fault == "freeze_until_dirty":
        # Only unfreeze once the client retired every lane, otherwise the test
        # would re-arm lanes that the server may still read.
        if not wait_for_marker(log_path, spec.dirty_marker, DIRTY_MARKER_TIMEOUT):
            return False
    else:
        time.sleep(freeze_seconds)
    print(f"[fault-injection] SIGCONT shard {FAULT_SHARD} pid={shard1.pid}")
    shard1.send_signal(signal.SIGCONT)
    return True


def start_client(runner, spec, test_binary, log_path):
    argv = [test_binary, f"--gtest_filter={spec.gtest_filter}"]
    argv.extend(spec.client_flags)
    env = runner.build_client_env(CLIENT_INDEX)
    env[spec.env_var] = "1"
    handle = open(log_path, "w")
    process = subprocess.Popen(
        runner.build_client_cmd(argv, client_index=CLIENT_INDEX),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def print_log_tail(log_path):
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return
    print(f"[fault-injection] --- tail of {log_path} ---")
    for line in lines[-LOG_TAIL_LINES:]:
        print(line)


def report(spec, checks, log_path):
    passed = all(ok for _name, ok, _detail in checks)
    for name, ok, detail in checks:
        suffix = f" {detail}" if detail else ""
        print(
            f"[fault-injection] {'PASS' if ok else 'FAIL'} "
            f"scenario={spec.name}: {name}{suffix}"
        )
    if not passed:
        print_log_tail(log_path)
    return passed


def run_scenario(spec, args, config_path, log_dir):
    log_path = log_dir / f"{spec.name}.client.log"
    print(
        f"[fault-injection] scenario={spec.name} fault={spec.fault} "
        f"lanes={spec.lanes} log={log_path}"
    )
    runner = PetPSClusterRunner(
        config_path=config_path,
        num_servers=SERVER_COUNT,
        num_clients=1,
        thread_num=1,
        timeout=args.cluster_timeout,
        verbose=args.show_runner_logs,
        status_refresh_interval=0,
        show_status_logs=False,
        show_control_plane_logs=args.show_runner_logs,
        rdma_rc_qps_per_client_per_shard=spec.lanes,
        rdma_rc_slots_per_qp=1,
        rdma_wait_timeout_ms=spec.wait_timeout_ms,
        rdma_lane_rebuild_enabled=spec.lane_rebuild,
        server_command_wrapper=lambda _global_id, cmd: cmd + spec.server_flags,
    )
    client = None
    client_handle = None
    shard1 = None
    try:
        runner.start()
        # start() appends the server processes in global_id order.
        shard1 = runner.processes[FAULT_SHARD][0]
        client, client_handle = start_client(
            runner, spec, args.test_binary, log_path
        )

        if not wait_for_marker(log_path, spec.warm_marker, WARM_MARKER_TIMEOUT):
            return report(
                spec, [("client reached its warm-up marker", False, "")], log_path
            )
        if not inject_fault(spec, shard1, log_path, args.freeze_seconds):
            return report(
                spec, [("the injected fault retired a lane", False, "")], log_path
            )

        try:
            returncode = client.wait(timeout=args.client_timeout)
        except subprocess.TimeoutExpired:
            return report(
                spec,
                [("client finished", False, f"timeout={args.client_timeout}s")],
                log_path,
            )

        text = log_path.read_text(errors="replace")
        checks = [
            ("client finished", True, f"rc={returncode}"),
            ("client exited cleanly", returncode == 0, f"rc={returncode}"),
            ("gtest reported a pass", "[  PASSED  ] 1 test." in text, ""),
            ("shard 0 stayed up", runner.processes[0][0].poll() is None, ""),
        ]
        return report(spec, checks, log_path)
    finally:
        if client_handle is not None:
            client_handle.close()
        if client is not None and client.poll() is None:
            client.kill()
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if shard1 is not None and shard1.poll() is None:
            # A stopped shard cannot be reaped cleanly; resume it before stop().
            shard1.send_signal(signal.SIGCONT)
        runner.stop()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--test-binary", required=True)
    parser.add_argument("--config-path")
    parser.add_argument(
        "--log-dir",
        help="keep the per-scenario client logs here instead of a temp dir",
    )
    parser.add_argument("--client-timeout", type=int, default=150)
    parser.add_argument("--cluster-timeout", type=int, default=60)
    parser.add_argument(
        "--freeze-seconds",
        type=float,
        default=15.0,
        help="how long the rebuild scenario holds shard 1 frozen",
    )
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="stream the control-plane and petps_server logs",
    )
    return parser.parse_args()


def main():
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    args = parse_args()
    spec = SCENARIOS[args.scenario]

    with tempfile.TemporaryDirectory(prefix="recstore_petps_fault_") as runtime_dir:
        config_path = materialize_rdma_config(
            resolve_rdma_integration_config(SERVER_COUNT, args.config_path),
            SERVER_COUNT,
            1,
            runtime_dir,
        )
        log_dir = Path(args.log_dir) if args.log_dir else Path(runtime_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        passed = run_scenario(spec, args, config_path, log_dir)

    print(
        f"[fault-injection] scenario={spec.name} "
        f"VERDICT={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
