#!/usr/bin/env python3
"""Split 2n1g NIC traffic into the PS RDMA flow and the NCCL flow.

The RDMA/PS flow is rebuilt from the opt-in wire trace
(``RECSTORE_RDMA_WIRE_TRACE``) that every PS server and every client writes:
one JSONL record per posted work request, tagged with the writer and the peer
global node id.  Because the whole cluster shares absolute paths, the two
hosts write their own file; this tool merges them.

The NCCL flow comes from a torch profiler chrome trace, where c10d collectives
are paired with the NCCL kernels that carry them.

Clocks: the wire trace uses CLOCK_MONOTONIC nanoseconds, which has a different
origin per host.  ``epoch = mono + offset`` with the per-host offset measured
by ``time.time_ns() - time.monotonic_ns()`` (or read from the runner's event
log, which records both).

Usage:
    python3 tools/benchmarks/analyze_wire_trace.py \\
        --wire-local results/<run>/wire/rdma_wire.jsonl \\
        --wire-remote results/<run>/wire.remote/rdma_wire.jsonl \\
        --offset-local-ns 1778060703981148743 \\
        --offset-remote-ns 1780896799010428869 \\
        --trace results/<run>/outputs/rdma_*/traces/*.pt.trace.json \\
        --events results/<run>/outputs/rdma_*/events.rank0.jsonl \\
        --out-dir results/<run>/analysis --bucket-ms 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_wire(path: Path, offset_ns: int) -> list[dict]:
    records: list[dict] = []
    if not path or not Path(path).is_file():
        return records
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["epoch"] = (record["t"] + offset_ns) / 1e9
            records.append(record)
    return records


def load_events(path: Path | None) -> list[dict]:
    if not path:
        return []
    events: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def host_offsets_from_events(events: list[dict]) -> float | None:
    for event in events:
        if event.get("event") == "step_start":
            return float(event["epoch"]) - int(event["mono_ns"]) / 1e9
    return None


def step_windows(events: list[dict]) -> list[dict]:
    windows: list[dict] = []
    current: dict = {}
    for event in events:
        name = event.get("event")
        if name == "step_start":
            current = {"step": int(event["step"]), "start": float(event["epoch"])}
        elif name == "step_end" and current:
            current["end"] = float(event["epoch"])
            windows.append(current)
            current = {}
    return windows


def load_nccl_series(trace_path: Path | None, reference: tuple[float, int] | None):
    """Return [(epoch, duration_s, bytes)] for NCCL kernels in a chrome trace."""
    if not trace_path or not Path(trace_path).is_file():
        return []
    trace = json.loads(Path(trace_path).read_bytes())
    events = trace.get("traceEvents", [])
    epoch_ref, mono_ns = reference if reference else (0.0, 0)
    kernels = []
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event or "ts" not in event:
            continue
        name = str(event.get("name", ""))
        if "nccl" not in name.lower() and "AllReduce" not in name:
            continue
        epoch = epoch_ref + (event["ts"] / 1e6 - mono_ns / 1e9)
        kernels.append(
            {
                "epoch": epoch,
                "ts_us": event["ts"],
                "dur_s": event["dur"] / 1e6,
            }
        )
    kernels.sort(key=lambda item: item["ts_us"])
    ops = sorted(
        (
            event
            for event in events
            if event.get("cat") == "cpu_op"
            and str(event.get("name", "")).startswith("c10d::")
        ),
        key=lambda item: item["ts"],
    )
    dtype_bytes = {"float": 4, "float32": 4, "half": 2, "bfloat16": 2, "int64": 8}
    series = []
    for op, kernel in zip(ops, kernels):
        dims = op.get("args", {}).get("Input Dims") or []
        elements = 0
        if dims and dims[0]:
            value = dims[0][0]
            while isinstance(value, (list, tuple)):
                value = value[0] if value else 0
            try:
                elements = int(value)
            except (TypeError, ValueError):
                elements = 0
        size = elements * dtype_bytes.get(str(op.get("args", {}).get("Input type")), 4)
        series.append((kernel["epoch"], kernel["dur_s"], size))
    return series


def bucketize(records, bucket_s: float, key_fn):
    buckets: dict[int, float] = {}
    for record in records:
        index = int(record["epoch"] / bucket_s)
        buckets[index] = buckets.get(index, 0.0) + key_fn(record)
    return {index: value / bucket_s for index, value in buckets.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire-local", default="")
    parser.add_argument("--wire-remote", default="")
    parser.add_argument("--offset-local-ns", type=int, default=0)
    parser.add_argument("--offset-remote-ns", type=int, default=0)
    parser.add_argument("--events", default="", help="event log from either rank")
    parser.add_argument("--trace", default="", help="chrome trace used for NCCL bytes")
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket-ms", type=float, default=5.0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_wire(Path(args.wire_local), args.offset_local_ns)
    records += load_wire(Path(args.wire_remote), args.offset_remote_ns)
    if not records:
        raise SystemExit("no wire records; check --wire-local/--wire-remote and offsets")

    events = load_events(Path(args.events)) if args.events else []
    reference = None
    if events:
        entry = events[0]
        reference = (float(entry["epoch"]), int(entry["mono_ns"]))
    if args.offset_local_ns == 0 and events:
        derived = host_offsets_from_events(events)
        if derived:
            args.offset_local_ns = int(derived * 1e9)
            records = load_wire(Path(args.wire_local), args.offset_local_ns)
            records += load_wire(Path(args.wire_remote), args.offset_remote_ns)

    def writer_host(record: dict) -> int:
        return int(record["self"]) % args.num_servers

    def peer_host(record: dict) -> int:
        return int(record["node"]) % args.num_servers

    def is_wire(record: dict) -> bool:
        return writer_host(record) != peer_host(record)

    # Which side posts: node ids below num_servers belong to servers.
    def is_server(record: dict) -> bool:
        return int(record["self"]) < args.num_servers

    wire_tx = [r for r in records if is_wire(r) and not is_server(r)]
    wire_rx = [r for r in records if is_wire(r) and is_server(r)]
    loopback = [r for r in records if not is_wire(r)]

    bucket_s = args.bucket_ms / 1000.0
    tx_rate = bucketize(wire_tx, bucket_s, lambda r: float(r["bytes"]))
    rx_rate = bucketize(wire_rx, bucket_s, lambda r: float(r["bytes"]))
    loop_rate = bucketize(loopback, bucket_s, lambda r: float(r["bytes"]))

    nccl_series = load_nccl_series(Path(args.trace) if args.trace else None, reference)
    nccl_rate: dict[int, float] = {}
    for epoch, duration, size in nccl_series:
        if duration <= 0:
            continue
        nccl_rate[int(epoch / bucket_s)] = nccl_rate.get(int(epoch / bucket_s), 0.0) + size / duration

    windows = step_windows(events)
    steps_of_interest = [w for w in windows if (w.get("end", 0) - w.get("start", 0)) > 0]
    # Steady state: drop the warmup steps the profiler window lands after.
    if len(steps_of_interest) > 12:
        steps_of_interest = steps_of_interest[10:]

    def window_stats(start: float, end: float) -> dict:
        def total(selection, bucket_map):
            return sum(
                value * bucket_s
                for index, value in bucket_map.items()
                if start <= index * bucket_s < end
            )

        return {
            "ps_wire_client_to_server_MB": total(wire_tx, tx_rate) / 1e6,
            "ps_wire_server_to_client_MB": total(wire_rx, rx_rate) / 1e6,
            "ps_loopback_MB": total(loopback, loop_rate) / 1e6,
            "nccl_MB": sum(size for epoch, _, size in nccl_series if start <= epoch < end) / 1e6,
        }

    per_step = [
        {"step": window["step"], **window_stats(window["start"], window["end"])}
        for window in steps_of_interest
    ]

    summary = {
        "wire_records": len(records),
        "ps_wire_client_to_server_MB": sum(float(r["bytes"]) for r in wire_tx) / 1e6,
        "ps_wire_server_to_client_MB": sum(float(r["bytes"]) for r in wire_rx) / 1e6,
        "ps_loopback_MB": sum(float(r["bytes"]) for r in loopback) / 1e6,
        "nccl_MB": sum(size for _, _, size in nccl_series) / 1e6,
        "per_step": per_step,
        "by_tag": {},
    }
    tags: dict[str, dict[str, float]] = {}
    for record in records:
        entry = tags.setdefault(record["tag"], {"records": 0, "MB": 0.0})
        entry["records"] += 1
        entry["MB"] += float(record["bytes"]) / 1e6
    summary["by_tag"] = tags

    indices = sorted(set(tx_rate) | set(rx_rate) | set(loop_rate) | set(nccl_rate))
    with (out_dir / "nic_flows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "ps_wire_tx_MBps",
                "ps_wire_rx_MBps",
                "ps_loopback_MBps",
                "nccl_MBps",
            ]
        )
        for index in indices:
            writer.writerow(
                [
                    f"{index * bucket_s:.6f}",
                    f"{tx_rate.get(index, 0.0) / 1e6:.4f}",
                    f"{rx_rate.get(index, 0.0) / 1e6:.4f}",
                    f"{loop_rate.get(index, 0.0) / 1e6:.4f}",
                    f"{nccl_rate.get(index, 0.0) / 1e6:.4f}",
                ]
            )

    (out_dir / "wire_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_step"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
