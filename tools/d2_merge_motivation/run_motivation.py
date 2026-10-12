#!/usr/bin/env python3
"""Motivation experiment for D2: request-side merge of multi-step lookups.

Question
--------
Does coalescing the lookups of several training steps into fewer, larger RDMA
GETs ("D2") pay off, and how much of what it targets is already delivered by the
current prefetch pipeline?

Method (PS/network layer, local RDMA loopback, single shard)
-----------------------------------------------------------
1. ``latency_*`` lanes: depth=1, single thread. The per-batch wait is a pure
   round trip, so fitting ``L(B) = a + b*B`` yields the fixed per-RPC cost ``a``
   (dispatch + RTT + status + copy setup) and the marginal cost per key ``b``.
   ``a`` is the ceiling on what removing one RPC can save.
2. ``pipe_*`` lanes: constant in-flight key volume, varying only request
   granularity. If coarser requests raise throughput, the path is per-RPC bound
   and D2 can pay.
3. ``cap_*`` lanes: batch sizes past the per-RPC response budget
   (``mtu * rdma_rc_target_response_mtu / value_size``) to show the client-side
   chunking ceiling: merging past the budget does not reduce request count.
4. ``agg_*`` lanes: 8 threads, same per-thread in-flight volume, to check
   whether coarsening still helps once the client is parallel.

Outputs go to ``results/d2_merge_motivation_<MMDDHHMM>/`` by default; override
with ``--output-dir``. The script is resumable: lanes that already produced a
client log are reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools/benchmarks/run_benchmark_ps.py"
BUILD_DIR = "build_release"
VALUE_SIZE = 512
RECORD_COUNT = 200000
MAX_KEYS_PER_REQUEST = 1600
RUNTIME_SECONDS = 5
SERVER_THREADS = 4
# Local RoCE loopback: server node 0 and client node 1 use the same device/GID.
FABRIC = "0:mlx5_3:1:7:rocev2:4:0:0,1:mlx5_3:1:7:rocev2:4:0:0"

# (lane_name, threads, batch_keys, prefetch_depth, extra_cli)
LANES = [
    # 1. depth=1 latency probe -> L(B)
    ("latency_d1_b64",   1,   64, 1, []),
    ("latency_d1_b256",  1,  256, 1, []),
    ("latency_d1_b1024", 1, 1024, 1, []),
    ("latency_d1_b1600", 1, 1600, 1, []),
    # 2. constant in-flight volume (1024 keys), varying granularity
    ("pipe_b64_d16",   1,   64, 16, []),
    ("pipe_b128_d8",   1,  128,  8, []),
    ("pipe_b256_d4",   1,  256,  4, []),
    ("pipe_b512_d2",   1,  512,  2, []),
    ("pipe_b1024_d1",  1, 1024,  1, []),
    # 3. constant in-flight volume past the 1600-key response budget
    ("cap_b1600_d4",   1, 1600,  4, []),
    ("cap_b3200_d2",   1, 3200,  2, []),
    # 4. 8-thread aggregate, constant in-flight volume per thread.
    #    8 x 32 QP/client is more than this host tolerates (client SIGKILL),
    #    so the aggregate lanes use 16 QP/client.
    ("agg_b64_d16_t8",  8,   64, 16, ["--rdma-rc-qps-per-client-per-shard", "16"]),
    ("agg_b256_d4_t8",  8,  256,  4, ["--rdma-rc-qps-per-client-per-shard", "16"]),
    ("agg_b1024_d1_t8", 8, 1024,  1, ["--rdma-rc-qps-per-client-per-shard", "16"]),
]

RESULT_RE = re.compile(
    r"PS_BENCHMARK_RESULT phase=run .*?threads=(?P<threads>\d+) "
    r"batch_size=(?P<batch>\d+).*?throughput_batches_sec=(?P<batches>[0-9.eE+-]+) "
    r"throughput_keys_sec=(?P<keys>[0-9.eE+-]+)"
)
PROFILE_RE = re.compile(
    r"PS_BENCHMARK_PROFILE phase=run .*?prefetch_depth=(?P<depth>\d+) "
    r"batches=(?P<batches>\d+) make_keys_avg_ns=(?P<make>[0-9.eE+-]+) "
    r"submit_avg_ns=(?P<submit>[0-9.eE+-]+) consume_avg_ns=(?P<consume>[0-9.eE+-]+) "
    r"wait_plus_result_avg_ns=(?P<wait>[0-9.eE+-]+)"
)


def lane_cmd(out_dir: Path, threads: int, batch: int, depth: int,
             extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, str(RUNNER),
        "--transports", "rdma",
        "--mode", "fetch",
        "--client-ips", "127.0.0.1",
        "--server-shard-ips", "127.0.0.1",
        "--client-processes-per-ip", "1",
        "--record-count", str(RECORD_COUNT),
        "--value-size", str(VALUE_SIZE),
        "--batch-keys", str(batch),
        "--client-threads-per-process", str(threads),
        "--runtime-seconds", str(RUNTIME_SECONDS),
        "--repeat", "1",
        "--prefetch-depth", str(depth),
        "--transaction-profile",
        "--distribution", "uniform",
        "--index-type", "DRAM_PET_HASH",
        "--max-keys-per-request", str(MAX_KEYS_PER_REQUEST),
        "--rdma-wait-timeout-ms", "20000",
        "--rdma-rc-qps-per-client-per-shard", "32",
        "--rdma-rc-slots-per-qp", "1",
        "--rdma-get-response-mode", "auto",
        "--server-rdma-threads", str(SERVER_THREADS),
        "--rdma-rc-server-get-workers", "0",
        "--rdma-rc-server-coroutines-per-thread", "1",
        "--rdma-fabric-plan", FABRIC,
        "--execution-backend", "local",
        "--build-dir", BUILD_DIR,
        "--output-dir", str(out_dir),
    ]
    return cmd + extra


def client_stdout(run_dir: Path, lane: str) -> Path | None:
    root = run_dir / "lanes" / lane / "logs" / "rdma" / "repeat_0"
    logs = sorted(root.glob("client_*.stdout.log"))
    return logs[0] if logs else None


def parse_lane(run_dir: Path, lane: str, threads: int, batch: int,
               depth: int) -> dict | None:
    log = client_stdout(run_dir, lane)
    if log is None:
        return None
    text = log.read_text(errors="replace")
    res = RESULT_RE.search(text)
    prof = PROFILE_RE.search(text)
    if res is None or prof is None:
        return None
    return {
        "lane": lane,
        "threads": threads,
        "batch_keys": batch,
        "prefetch_depth": depth,
        "in_flight_keys": batch * depth,
        "keys_per_sec": float(res.group("keys")),
        "M_keys_per_sec": float(res.group("keys")) / 1e6,
        "batches_per_sec": float(res.group("batches")),
        "make_keys_us": float(prof.group("make")) / 1e3,
        "submit_us": float(prof.group("submit")) / 1e3,
        "consume_us": float(prof.group("consume")) / 1e3,
        "wait_us": float(prof.group("wait")) / 1e3,
        "submit_us_per_key": float(prof.group("submit")) / 1e3 / batch,
        "wait_us_per_key": float(prof.group("wait")) / 1e3 / batch,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated lane name substrings")
    ap.add_argument("--force", action="store_true",
                    help="re-run lanes that already have client logs")
    ap.add_argument("--output-dir", default="",
                    help="run directory; default results/d2_merge_motivation_<MMDDHHMM>")
    args = ap.parse_args()
    filters = [f for f in args.only.split(",") if f]

    run_dir = (Path(args.output_dir).resolve() if args.output_dir
               else REPO / "results" / f"d2_merge_motivation_{time.strftime('%m%d%H%M')}")
    (run_dir / "lanes").mkdir(parents=True, exist_ok=True)
    print(f"[out ] {run_dir}")

    rows = []
    for lane, threads, batch, depth, extra in LANES:
        if filters and not any(f in lane for f in filters):
            continue
        if client_stdout(run_dir, lane) is not None and not args.force:
            row = parse_lane(run_dir, lane, threads, batch, depth)
            if row is not None:
                print(f"[skip] {lane} (already has results)")
                rows.append(row)
                continue
        print(f"[run ] {lane} threads={threads} batch={batch} depth={depth}", flush=True)
        subprocess.run([str(REPO / "tools/benchmarks/kill_bench_procs.sh")],
                       cwd=REPO, capture_output=True)
        proc = subprocess.run(lane_cmd(run_dir / "lanes" / lane, threads, batch, depth, extra),
                              cwd=REPO, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"[fail] {lane}: rc={proc.returncode}\n"
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        row = parse_lane(run_dir, lane, threads, batch, depth)
        if row is None:
            print(f"[warn] {lane}: could not parse results")
            continue
        rows.append(row)

    if not rows:
        return 1
    out = run_dir / "motivation.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {out} ({len(rows)} lanes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
