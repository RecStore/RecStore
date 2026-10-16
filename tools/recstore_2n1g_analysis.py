#!/usr/bin/env python3
"""Build the derived tables behind the 2n1g RecStore/TorchRec figures.

Inputs (all produced by one run of tools/benchmarks/e2e/custom/cli.py):
  outputs/<lane>/events.rank{0,1}.jsonl   runner phase timeline (epoch + mono)
  outputs/<lane>/traces/run_*/rank0.pt.trace.json   chrome trace (CUPTI clock)
  wire/rdma_wire.jsonl, wire.remote/rdma_wire.jsonl  RDMA wire trace, one file
                                                     per host

Outputs (written to --out-dir):
  phase_steps.csv     per-step phase durations for both ranks
  gpu_timeline.csv    kernels in the profiled window with phase/op attribution
  nic_flows.csv       per-bucket RDMA/PS and NCCL bandwidth over that window
  analysis.json       headline numbers used by the figures and the write-up
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path


PHASES = (
    "prepare",
    "consume",
    "pack",
    "fwd",
    "bwd",
    "dense_opt",
    "sparse",
    "barrier",
)

NCCL_PATTERNS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast")


def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL, tolerating torn lines.

    Several processes append to one wire-trace file with buffered ``fwrite``,
    so a kill in the middle of a flush can splice two records together.  Those
    lines are dropped and counted instead of aborting the analysis.
    """
    records: list[dict] = []
    torn = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    if torn:
        print(f"warning: {path}: dropped {torn} torn JSONL lines")
    return records


def step_records(events: list[dict]) -> list[dict]:
    """Group the runner's phase events into per-step windows."""
    steps: list[dict] = []
    current: dict | None = None
    for event in events:
        name = event.get("event", "")
        if name == "step_start":
            current = {"step": int(event["step"]), "start": float(event["epoch"]), "phases": {}}
        elif name == "step_end" and current is not None:
            current["end"] = float(event["epoch"])
            steps.append(current)
            current = None
        elif current is not None and name != "profiler_begin":
            head, _, tail = name.rpartition("_")
            if head and tail in {"start", "end"}:
                current["phases"].setdefault(head, {})[tail] = float(event["epoch"])
    return steps


def profiler_steps(trace_events: list[dict]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for event in trace_events:
        name = str(event.get("name", ""))
        if not name.startswith("ProfilerStep#") or "dur" not in event:
            continue
        index = int(name.split("#", 1)[1])
        if index not in found or event["dur"] < found[index]["dur"]:
            found[index] = event
    return found


def trace_offset(trace_events: list[dict], steps: list[dict]) -> float:
    """Seconds to add to trace ts/1e6 to land in epoch time."""
    found = profiler_steps(trace_events)
    diffs = [
        step["start"] - found[step["step"]]["ts"] / 1e6
        for step in steps
        if step["step"] in found
    ]
    if not diffs:
        raise SystemExit("no ProfilerStep matches the event log; clocks cannot be aligned")
    baseline = st.median(diffs)
    close = [value for value in diffs if abs(value - baseline) < 1.0]
    return st.mean(close)


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if any(pattern in lowered for pattern in NCCL_PATTERNS):
        return "nccl"
    if "sgemm" in lowered or "gemm" in lowered or "cutlass" in lowered or "nvjet" in lowered:
        return "matmul"
    if "index" in lowered or "gather" in lowered or "radix" in lowered or "sort" in lowered:
        return "gather"
    if "embedding" in lowered:
        return "gather"
    if "copy" in lowered or "catarray" in lowered or "memcpy" in lowered:
        return "copy"
    if "elementwise" in lowered or "vectorized" in lowered:
        return "elementwise"
    if "reduce_kernel" in lowered:
        return "reduce"
    if "memset" in lowered or "fill" in lowered:
        return "memset"
    if "adam" in lowered or "muadam" in lowered or "sgd" in lowered:
        return "optimizer"
    return "other"


def load_trace_kernels(path: Path, offset: float) -> list[dict]:
    trace = json.loads(Path(path).read_bytes())
    events = trace.get("traceEvents", [])
    launcher: dict[str, dict] = {}
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        key = str(event.get("args", {}).get("External id"))
        if key != "None":
            launcher.setdefault(key, event)
    kernels: list[dict] = []
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        owner = launcher.get(str(event.get("args", {}).get("External id")))
        kernels.append(
            {
                "ts": event["ts"],
                "epoch": event["ts"] / 1e6 + offset,
                "dur_ms": event["dur"] / 1e3,
                "name": event.get("name", ""),
                "kind": kernel_kind(event.get("name", "")),
                "stream": event.get("args", {}).get("stream"),
                "op": owner.get("name", "") if owner else "",
                "tid": owner.get("tid") if owner else None,
                "launch_ts": owner.get("ts") if owner else None,
                "launch_dur_us": owner.get("dur") if owner else None,
            }
        )
    kernels.sort(key=lambda item: item["ts"])
    return kernels


def load_nccl_bytes(trace_path: Path, offset: float) -> list[dict]:
    """Attribute collective bytes to the NCCL kernel that carries them.

    Newer PyTorch traces no longer carry sizes on ``c10d::*`` cpu_ops, but the
    DDP hook still records ``record_param_comms`` with ``In msg nelems``, so the
    payload size comes from there and is then hung on the first NCCL kernel
    launched at or after the comm op.
    """
    trace = json.loads(Path(trace_path).read_bytes())
    events = trace.get("traceEvents", [])
    dtype_bytes = {"Byte": 1, "Float": 4, "float": 4, "Half": 2, "half": 2,
                   "Bfloat16": 2, "bfloat16": 2, "Long": 8, "int64": 8, "Int": 4}
    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel"
            and "dur" in event
            and "nccl" in str(event.get("name", "")).lower()
        ),
        key=lambda item: item["ts"],
    )
    if not kernels:
        return []
    ops = []
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        args = event.get("args", {})
        if str(event.get("name", "")) == "record_param_comms":
            elements = max(
                int(args.get("In msg nelems") or 0),
                int(args.get("Out msg nelems") or 0),
            )
            size = elements * dtype_bytes.get(str(args.get("dtype", "")), 1)
        elif "c10d::" in str(event.get("name", "")):
            dims = args.get("Input Dims") or []
            elements = 0
            if dims and dims[0]:
                value = dims[0][0]
                while isinstance(value, (list, tuple)):
                    value = value[0] if value else 0
                try:
                    elements = int(value)
                except (TypeError, ValueError):
                    elements = 0
            size = elements * dtype_bytes.get(str(args.get("Input type", "float")), 4)
        else:
            continue
        if size > 0 and "dur" in event:
            ops.append({"ts": event["ts"], "bytes": size, "name": event["name"]})
    per_kernel: dict[int, int] = collections.defaultdict(int)
    for op in sorted(ops, key=lambda item: item["ts"]):
        target = next((kernel for kernel in kernels if kernel["ts"] >= op["ts"]), None)
        if target is None:
            continue
        per_kernel[id(target)] += op["bytes"]

    rows = []
    for kernel in kernels:
        rows.append(
            {
                "ts": kernel["ts"],
                "epoch": kernel["ts"] / 1e6 + offset,
                "dur_ms": kernel["dur"] / 1e3,
                "bytes": per_kernel.get(id(kernel), 0),
                "collective": "",
                "op_epoch": kernel["ts"] / 1e6 + offset,
                "op_dur_ms": 0.0,
            }
        )
    return rows


def phase_of(epoch: float, phases: dict) -> str:
    for name in PHASES:
        window = phases.get(name)
        if window and "start" in window and "end" in window:
            if window["start"] <= epoch < window["end"]:
                return name
    return "other"


def load_wire(paths: list[tuple[Path, float]]) -> list[dict]:
    records: list[dict] = []
    for path, offset in paths:
        if not Path(path).is_file():
            continue
        for record in load_jsonl(Path(path)):
            record["epoch"] = record["t"] / 1e9 + offset
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repeat", default="r0", choices=["r0", "r1"])
    parser.add_argument("--lane", default="recstore")
    parser.add_argument("--torchrec-lane", default="torchrec_hbm")
    parser.add_argument("--offset-local-ns", type=int, default=0)
    parser.add_argument("--offset-remote-ns", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket-ms", type=float, default=1.0)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_dir = run_dir / "outputs" / f"rdma_b3072_d128_{args.repeat}"
    events = load_jsonl(lane_dir / "events.rank0.jsonl")
    events_rank1 = load_jsonl(lane_dir / "events.rank1.jsonl")
    steps = step_records(events)
    steps_rank1 = step_records(events_rank1)
    traces = sorted((lane_dir / "traces").glob("*/rank0.pt.trace.json"))
    if not traces:
        raise SystemExit(f"no rank0 trace under {lane_dir}")
    rank0_trace = traces[0]
    trace_events = json.loads(rank0_trace.read_bytes())["traceEvents"]
    offset = trace_offset(trace_events, steps)

    kernels = load_trace_kernels(rank0_trace, offset)
    profiled = sorted(profiler_steps(trace_events))
    window_start = min(kernels[0]["epoch"], steps[profiled[0]]["start"])
    window_stop = max(steps[profiled[-1]]["end"], kernels[-1]["epoch"])

    for kernel in kernels:
        kernel["phase"] = "outside"
    for step in steps:
        for kernel in kernels:
            if step["start"] <= kernel["epoch"] < step["end"]:
                kernel["phase"] = phase_of(kernel["epoch"], step["phases"])
                kernel["profiled_step"] = step["step"]

    nccl = load_nccl_bytes(rank0_trace, offset)
    wire = load_wire(
        [
            (run_dir / "wire" / "rdma_wire.jsonl", args.offset_local_ns / 1e9),
            (run_dir / "wire.remote" / "rdma_wire.jsonl", args.offset_remote_ns / 1e9),
        ]
    )
    for record in wire:
        record["is_server"] = int(record["self"]) < 2
        record["cross_host"] = (int(record["self"]) % 2) != (int(record["node"]) % 2)

    bucket_s = args.bucket_ms / 1000.0

    def rate_series(records: list[dict]) -> dict[int, float]:
        buckets: dict[int, float] = collections.defaultdict(float)
        for record in records:
            buckets[int(record["epoch"] / bucket_s)] += float(record["bytes"])
        return {key: value / bucket_s for key, value in buckets.items()}

    window = [step for step in steps if step["step"] in profiled]
    labels = [f"#{index}" for index in range(len(window))]

    nic_rows = []
    for index, step in enumerate(window):
        start, stop = step["start"], step["end"]
        inside = [r for r in wire if start <= r["epoch"] < stop]
        ps_tx = sum(r["bytes"] for r in inside if r["cross_host"] and not r["is_server"])
        ps_rx = sum(r["bytes"] for r in inside if r["cross_host"] and r["is_server"])
        loop = sum(r["bytes"] for r in inside if not r["cross_host"])
        kern = [k for k in kernels if start <= k["epoch"] < stop]
        busy = sum(k["dur_ms"] for k in kern)
        nccl_in = [n for n in nccl if start <= n["epoch"] < stop]
        nic_rows.append(
            {
                "step": step["step"],
                "label": labels[index],
                "duration_ms": (stop - start) * 1e3,
                "ps_wire_client_to_server_MB": ps_tx / 1e6,
                "ps_wire_server_to_client_MB": ps_rx / 1e6,
                "ps_loopback_MB": loop / 1e6,
                "kernel_count": len(kern),
                "kernel_busy_ms": busy,
                "kernel_busy_pct": busy / ((stop - start) * 1e3) * 100,
                "nccl_kernel_count": len(nccl_in),
                "nccl_MB": sum(n["bytes"] for n in nccl_in) / 1e6,
                "nccl_busy_ms": sum(n["dur_ms"] for n in nccl_in),
            }
        )

    with (out_dir / "nic_flows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(nic_rows)

    rate_rows = []
    for index in range(int(window_start / bucket_s), int(window_stop / bucket_s) + 1):
        start = index * bucket_s
        step_index = next(
            (i for i, step in enumerate(window) if step["start"] <= start < step["end"]), None
        )
        if step_index is None:
            continue
        ps_tx = rate_series([r for r in wire if r["cross_host"] and not r["is_server"]])
        rate_rows.append(
            {
                "label": labels[step_index],
                "offset_ms": (start - window[step_index]["start"]) * 1e3,
                "ps_wire_tx_MBps": ps_tx.get(index, 0.0) / 1e6,
            }
        )
    with (out_dir / "nic_rate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "offset_ms", "ps_wire_tx_MBps"])
        writer.writeheader()
        writer.writerows(rate_rows)

    phase_rows = []
    for lane_name, lane_steps in (("rank0", steps), ("rank1", steps_rank1)):
        for step in lane_steps:
            row = {"rank": lane_name, "step": step["step"], "total_ms": (step["end"] - step["start"]) * 1e3}
            for name in PHASES:
                info = step["phases"].get(name)
                row[f"{name}_ms"] = (
                    (info["end"] - info["start"]) * 1e3
                    if info and "start" in info and "end" in info
                    else 0.0
                )
            phase_rows.append(row)
    with (out_dir / "phase_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase_rows)

    kernel_rows = []
    for kernel in kernels:
        if not (window_start <= kernel["epoch"] < window_stop):
            continue
        kernel_rows.append(
            {
                "epoch": f"{kernel['epoch']:.6f}",
                "name": kernel["name"],
                "kind": kernel["kind"],
                "phase": kernel.get("phase", "outside"),
                "stream": kernel["stream"],
                "tid": kernel["tid"],
                "op": kernel["op"],
                "dur_ms": f"{kernel['dur_ms']:.6f}",
            }
        )
    with (out_dir / "gpu_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kernel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(kernel_rows)

    summary = {
        "rank0_trace": str(rank0_trace),
        "trace_offset_s": offset,
        "profiled_steps": profiled,
        "window_ms": (window_stop - window_start) * 1e3,
        "steps": nic_rows,
        "phase_mean_ms": {
            name: {
                lane_name: st.mean(
                    row[f"{name}_ms"] for row in phase_rows if row["rank"] == lane_name and row["step"] >= 20
                )
                for lane_name in ("rank0", "rank1")
            }
            for name in PHASES
        },
    }
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["steps"], indent=2))
    print("window_ms %.1f  trace_offset %.6f" % (summary["window_ms"], offset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
