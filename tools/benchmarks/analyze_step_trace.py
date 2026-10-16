#!/usr/bin/env python3
"""Reconstruct per-step GPU busy/gap timelines and NIC bandwidth from traces.

Inputs:
- torch profiler chrome trace(s) written by --recstore-profiler / --torchrec-profiler
- the JSONL event log written by --recstore-event-log (epoch + monotonic clocks)
- optional NIC counter CSVs from tools/benchmarks/nic_counter_sampler.py

The trace clock (CUPTI) is CLOCK_MONOTONIC microseconds; the event log records
both epoch and monotonic times, so traces can be placed on the wall clock and
compared with the NIC samples.

Usage:
    python3 tools/benchmarks/analyze_step_trace.py \
        --trace <rank0.pt.trace.json> --events <events.rank0.jsonl> \
        --nic-local <nic_local.csv> --nic-remote <nic_remote.csv> \
        --step 32 --out-dir results/<run>/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


NCCL_PATTERNS = (
    "nccl",
    "AllReduce",
    "AllGather",
    "ReduceScatter",
    "Broadcast",
    "SendRecv",
)


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if any(pattern.lower() in lowered for pattern in NCCL_PATTERNS):
        return "nccl"
    if "sgemm" in lowered or "gemm" in lowered or "cutlass" in lowered or "nvjet" in lowered:
        return "matmul"
    if "radix" in lowered or "sort" in lowered:
        return "sort"
    if "index" in lowered or "gather" in lowered:
        return "gather"
    if "copy" in lowered or "catarray" in lowered or "memcpy" in lowered:
        return "copy"
    if "elementwise" in lowered or "vectorized" in lowered or "reduce_kernel" in lowered:
        return "elementwise"
    if "memset" in lowered or "fill" in lowered:
        return "memset"
    return "other"


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def clock_reference(events: list[dict]) -> tuple[float, int]:
    for event in events:
        if event.get("event") == "profiler_begin":
            return float(event["epoch"]), int(event["mono_ns"])
    first = events[0]
    return float(first["epoch"]), int(first["mono_ns"])


def trace_us_to_epoch(ts_us: float, reference: tuple[float, int]) -> float:
    epoch, mono_ns = reference
    return epoch + (ts_us / 1e6 - mono_ns / 1e9)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def total_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def load_nic(path: Path) -> dict[str, list[tuple[float, int]]]:
    series: dict[str, list[tuple[float, int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['device']}:{row['counter']}"
            series.setdefault(key, []).append((float(row["epoch"]), int(row["bytes"])))
    for values in series.values():
        values.sort()
    return series


def nic_rate(series: list[tuple[float, int]], t0: float, t1: float) -> float:
    """Bytes/second on the counters covered by [t0, t1]; counters are cumulative."""
    inside = [(t, v) for t, v in series if t0 <= t <= t1]
    if len(inside) < 2:
        return 0.0
    bytes_delta = inside[-1][1] - inside[0][1]
    seconds = inside[-1][0] - inside[0][0]
    if seconds <= 0:
        return 0.0
    return bytes_delta / seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--events", default="")
    parser.add_argument("--nic-local", default="")
    parser.add_argument("--nic-remote", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Step index to analyze; -1 picks the last complete step in the trace.",
    )
    parser.add_argument("--gap-us", type=float, default=100.0, help="Minimum reported gap.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = json.loads(Path(args.trace).read_bytes())
    trace_events = trace["traceEvents"]

    events = load_events(Path(args.events)) if args.events else []
    reference = clock_reference(events) if events else (0.0, 0)

    kernels = []
    nccl_bytes_by_kernel: dict[int, int] = {}
    for index, event in enumerate(trace_events):
        if event.get("cat") != "kernel" or "dur" not in event or "ts" not in event:
            continue
        epochs = (
            trace_us_to_epoch(event["ts"], reference) if events else event["ts"] / 1e6
        )
        kernels.append(
            {
                "ts_us": event["ts"],
                "dur_us": event["dur"],
                "epoch": epochs,
                "name": event.get("name", ""),
                "kind": kernel_kind(event.get("name", "")),
                "stream": event.get("args", {}).get("stream"),
                "correlation": event.get("args", {}).get("correlation"),
            }
        )
    kernels.sort(key=lambda item: item["ts_us"])
    if not kernels:
        raise SystemExit("no kernel events in trace")

    # Pair c10d collectives with NCCL kernels to recover transferred bytes.
    c10d_ops = sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "cpu_op" and str(event.get("name", "")).startswith("c10d::")
        ),
        key=lambda item: item["ts"],
    )
    nccl_kernels = [kernel for kernel in kernels if kernel["kind"] == "nccl"]
    for op, kernel in zip(c10d_ops, nccl_kernels):
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
        dtype = op.get("args", {}).get("Input type", "float")
        dtype_bytes = {"float": 4, "float32": 4, "half": 2, "bfloat16": 2, "int64": 8}.get(
            str(dtype), 4
        )
        nccl_bytes_by_kernel[kernel["ts_us"]] = elements * dtype_bytes

    # Step windows come from the event log; otherwise infer from the largest gaps.
    steps: list[dict] = []
    if events:
        current: dict = {}
        for event in events:
            name = event.get("event")
            if name == "step_start":
                current = {"step": int(event["step"]), "start": float(event["epoch"])}
            elif name == "step_end" and current:
                current["end"] = float(event["epoch"])
                steps.append(current)
                current = {}

    if args.step >= 0:
        window = next((item for item in steps if item["step"] == args.step), None)
    else:
        window = steps[-2] if len(steps) > 1 else (steps[-1] if steps else None)
    if window is None and not events:
        # No event log: infer one step window from the largest inter-kernel gap.
        iv = merge_intervals(
            [
                (kernel["epoch"], kernel["epoch"] + kernel["dur_us"] / 1e6)
                for kernel in kernels
            ]
        )
        if len(iv) > 1:
            ranked = sorted(
                ((iv[i][0] - iv[i - 1][1], iv[i][0]) for i in range(1, len(iv))),
                reverse=True,
            )
            window = {"step": -1, "start": ranked[0][1], "end": iv[-1][1]}
    if window is None:
        raise SystemExit(f"step {args.step} not found in event log")

    t0, t1 = window["start"], window["end"]
    in_step = [kernel for kernel in kernels if t0 <= kernel["epoch"] < t1]
    busy = merge_intervals(
        [(kernel["epoch"], kernel["epoch"] + kernel["dur_us"] / 1e6) for kernel in in_step]
    )
    nccl_busy = merge_intervals(
        [
            (kernel["epoch"], kernel["epoch"] + kernel["dur_us"] / 1e6)
            for kernel in in_step
            if kernel["kind"] == "nccl"
        ]
    )
    compute_busy = merge_intervals(
        [
            (kernel["epoch"], kernel["epoch"] + kernel["dur_us"] / 1e6)
            for kernel in in_step
            if kernel["kind"] != "nccl"
        ]
    )

    gaps = []
    merged = merge_intervals([(t0, t0)] + busy + [(t1, t1)])
    for (_, end), (start, _) in zip(merged, merged[1:]):
        duration = start - end
        if duration * 1e6 >= args.gap_us:
            gaps.append({"start": end, "end": start, "ms": duration * 1e3})

    step_seconds = t1 - t0
    summary = {
        "step": window["step"],
        "window_s": step_seconds,
        "kernel_count": len(in_step),
        "gpu_busy_ms": total_length(busy) * 1e3,
        "gpu_busy_pct": total_length(busy) / step_seconds * 100,
        "nccl_busy_ms": total_length(nccl_busy) * 1e3,
        "compute_busy_ms": total_length(compute_busy) * 1e3,
        "gap_count": len(gaps),
        "gap_total_ms": sum(gap["ms"] for gap in gaps),
        "gap_max_ms": max((gap["ms"] for gap in gaps), default=0.0),
        "kernel_ms_by_kind": {
            kind: sum(
                kernel["dur_us"] for kernel in in_step if kernel["kind"] == kind
            )
            / 1e3
            for kind in sorted({kernel["kind"] for kernel in in_step})
        },
        "nccl_bytes_total": sum(
            nccl_bytes_by_kernel.get(kernel["ts_us"], 0)
            for kernel in in_step
            if kernel["kind"] == "nccl"
        ),
        "c10d_op_count": len(c10d_ops),
        "nccl_kernel_count": len(nccl_kernels),
    }

    with (out_dir / f"step_{window['step']}_gaps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start", "end", "ms"])
        writer.writeheader()
        writer.writerows(gaps)

    with (out_dir / f"step_{window['step']}_kernels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "ts_us", "dur_us", "kind", "stream", "name"],
        )
        writer.writeheader()
        for kernel in in_step:
            writer.writerow(
                {
                    "epoch": f"{kernel['epoch']:.9f}",
                    "ts_us": kernel["ts_us"],
                    "dur_us": kernel["dur_us"],
                    "kind": kernel["kind"],
                    "stream": kernel["stream"],
                    "name": kernel["name"],
                }
            )

    if args.nic_local or args.nic_remote:
        nic_summary = {}
        for label, path in (("local", args.nic_local), ("remote", args.nic_remote)):
            if not path:
                continue
            series = load_nic(Path(path))
            devices = sorted({key.split(":")[0] for key in series})
            nic_summary[label] = {
                device: {
                    "xmit_MBps": nic_rate(
                        series.get(f"{device}:port_xmit_data", []), t0, t1
                    )
                    / 1e6,
                    "rcv_MBps": nic_rate(
                        series.get(f"{device}:port_rcv_data", []), t0, t1
                    )
                    / 1e6,
                }
                for device in devices
            }
        summary["nic"] = nic_summary

    (out_dir / f"step_{window['step']}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
