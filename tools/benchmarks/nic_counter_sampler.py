#!/usr/bin/env python3
"""Sample RDMA port counters so NIC bandwidth can be reconstructed offline.

Reads /sys/class/infiniband/<dev>/ports/1/counters/port_{xmit,rcv}_data on a
fixed interval and appends rows to a CSV. The raw counters are reported in
4-byte data units on mlx5; pass --data-unit-bytes 1 for pure byte counters.

Usage:
    python3 tools/benchmarks/nic_counter_sampler.py \
        --out results/<run>/nic_counters_<host>.csv --interval-ms 10

Stop with SIGTERM/SIGINT; the file is flushed on every sample.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import signal
import sys
import time

COUNTERS = ("port_xmit_data", "port_rcv_data")


def read_counter(device: str, counter: str) -> int | None:
    path = f"/sys/class/infiniband/{device}/ports/1/counters/{counter}"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval-ms", type=int, default=10)
    parser.add_argument(
        "--devices",
        default="",
        help="Comma-separated device list; default = every /sys/class/infiniband/mlx5_*.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--data-unit-bytes", type=int, default=4)
    args = parser.parse_args(argv)

    if args.devices.strip():
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    else:
        devices = sorted(os.path.basename(path) for path in glob.glob("/sys/class/infiniband/*"))
    if not devices:
        print("no RDMA devices found", file=sys.stderr)
        return 1

    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    interval = max(args.interval_ms, 1) / 1000.0
    deadline = time.monotonic() + args.duration_s if args.duration_s > 0 else None
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "mono_ns",
                "host",
                "device",
                "counter",
                "raw",
                "bytes",
            ]
        )
        host = os.uname().nodename
        next_tick = time.monotonic()
        while not stop:
            epoch = time.time()
            mono_ns = time.monotonic_ns()
            for device in devices:
                for counter in COUNTERS:
                    raw = read_counter(device, counter)
                    if raw is None:
                        continue
                    writer.writerow(
                        [
                            f"{epoch:.6f}",
                            mono_ns,
                            host,
                            device,
                            counter,
                            raw,
                            raw * args.data_unit_bytes,
                        ]
                    )
            handle.flush()
            if deadline is not None and time.monotonic() >= deadline:
                break
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    print(f"stopped, samples in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
