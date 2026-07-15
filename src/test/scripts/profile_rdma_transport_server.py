#!/usr/bin/env python3

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from petps_cluster_runner import PetPSClusterRunner
from ps_test_config import DEFAULT_RDMA_SINGLE_SHARD_CONFIG
from tools.benchmarks.run_rdma_transport_benchmarks import build_benchmark_cmd


def prepend_ld_preload(env, library):
    if not library:
        return
    current = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = library if not current else f"{library}:{current}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the RDMA transport benchmark while profiling petps_server."
    )
    parser.add_argument("--benchmark-binary", default="./build/bin/ps_transport_benchmark")
    parser.add_argument("--server-binary", default="./build/bin/petps_server")
    parser.add_argument("--profile-path", required=True)
    parser.add_argument(
        "--profiler-lib",
        default=os.environ.get("RECSTORE_PROFILER_LIB", ""),
        help="Optional libprofiler.so path to prepend to LD_PRELOAD.",
    )
    parser.add_argument(
        "--rdma-put-v2-transfer-mode",
        choices=["read", "push"],
        default="read",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--batch-keys", type=int, default=500)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--profile-signal", type=int, default=signal.SIGUSR2)
    parser.add_argument("--profile-frequency", type=int, default=1000)
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument("--show-runner-logs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    profile_path = Path(args.profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CPUPROFILE"] = str(profile_path)
    os.environ["CPUPROFILESIGNAL"] = str(int(args.profile_signal))
    os.environ["CPUPROFILE_FREQUENCY"] = str(args.profile_frequency)
    prepend_ld_preload(os.environ, args.profiler_lib)

    runner = PetPSClusterRunner(
        server_path=args.server_binary,
        config_path=DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        num_servers=1,
        num_clients=1,
        thread_num=args.thread_num,
        max_kv_num_per_request=args.batch_keys,
        verbose=args.show_runner_logs,
        show_status_logs=args.show_runner_logs,
        show_control_plane_logs=args.show_runner_logs,
        rdma_namespace=args.rdma_namespace,
        rdma_control_plane_host=args.rdma_control_plane_host,
        rdma_control_plane_port=args.rdma_control_plane_port,
        rdma_put_protocol_version=2,
        rdma_put_v2_transfer_mode=args.rdma_put_v2_transfer_mode,
        rdma_wait_timeout_ms=30000,
    )

    with runner.run():
        server_process = runner.processes[0][0]
        print(f"[server-profile] start pid={server_process.pid} path={profile_path}")
        os.kill(server_process.pid, int(args.profile_signal))
        time.sleep(0.2)
        completed = runner.run_client(
            build_benchmark_cmd(
                benchmark_binary=args.benchmark_binary,
                transport="rdma",
                iterations=args.iterations,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                report_mode="summary",
                batch_keys=args.batch_keys,
                num_shards=1,
            ),
            stream_output=True,
            timeout=args.timeout_sec,
        )
        print(f"[server-profile] stop pid={server_process.pid} path={profile_path}")
        os.kill(server_process.pid, int(args.profile_signal))
        time.sleep(0.5)
        if completed.returncode != 0:
            return completed.returncode

    if not profile_path.exists() or profile_path.stat().st_size == 0:
        print(
            "[server-profile] profile file was not created or is empty; "
            "check that petps_server is linked with gperftools or pass "
            "--profiler-lib/RECSTORE_PROFILER_LIB.",
            file=sys.stderr,
        )
        return 2
    print(f"[server-profile] wrote {profile_path} bytes={profile_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
