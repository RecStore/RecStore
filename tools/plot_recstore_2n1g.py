#!/usr/bin/env python3
"""Figures for the 2n1g bs=3072 RecStore-BagPipe run.

Two figures, both driven by the same run directory:

  recstore_2n1g_step_timeline.png
      1. client GPU (rank0) kernel timeline, coloured by runner phase
      2. host phases per step + the BagPipe prefetch pipeline (lookahead=4)
      3. inter-node collective bandwidth (NCCL allreduce / all_gather)
      4. PS data-path bandwidth over RDMA, split by direction and record kind
      5. TorchRec-HBM reference lane

  recstore_2n1g_kernel_breakdown.png
      kernel kinds (count vs time), duration distribution, launcher threads and
      the forward/backward sub-op breakdown.

Inputs are the run's own artefacts: events.rank{0,1}.jsonl, the CUPTI trace,
the RDMA wire trace from both hosts and the TorchRec lane trace.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


PHASE_ORDER = ("consume", "pack", "fwd", "bwd", "dense_opt", "sparse", "barrier", "prepare")
PHASE_COLORS = {
    "prepare": "#8e44ad",
    "consume": "#2980b9",
    "pack": "#16a085",
    "fwd": "#27ae60",
    "bwd": "#e67e22",
    "dense_opt": "#f1c40f",
    "sparse": "#c0392b",
    "barrier": "#7f8c8d",
    "other": "#bdc3c7",
    "outside": "#ecf0f1",
}
PHASE_LABELS = {
    "consume": "consume（查表/wait）",
    "pack": "pack（拼 batch）",
    "fwd": "fwd（稠密前向）",
    "bwd": "bwd（反向+DDP）",
    "dense_opt": "dense_opt",
    "sparse": "sparse（改表回写）",
    "barrier": "barrier",
    "prepare": "prepare（为 N+4 预取）",
}
KIND_ORDER = ("matmul", "elementwise", "gather", "copy", "reduce", "memset", "nccl", "other")
KIND_LABELS = {
    "matmul": "matmul / sgemm",
    "elementwise": "elementwise",
    "gather": "index / sort / gather",
    "copy": "copy / cat",
    "reduce": "reduce",
    "memset": "fill / memset",
    "nccl": "NCCL 集合通信",
    "other": "其他",
}
KIND_COLORS = {
    "matmul": "#2980b9",
    "gather": "#16a085",
    "elementwise": "#e67e22",
    "copy": "#8e44ad",
    "reduce": "#d35400",
    "memset": "#95a5a6",
    "nccl": "#2c3e50",
    "other": "#bdc3c7",
}
NCCL_PATTERNS = ("nccl", "allreduce", "allgather", "reducescatter", "broadcast")
LINK_MBPS = 12500.0  # 100 Gb/s per port, the figure's reference line


def setup_fonts() -> None:
    for candidate in (
        Path.home() / ".local/share/fonts/noto-sc/NotoSansSC-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False


def load_jsonl(path: Path) -> list[dict]:
    records = []
    torn = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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
        return "reduce"
    return "other"


def load_kernels(trace_path: Path, offset: float) -> list[dict]:
    events = json.loads(Path(trace_path).read_bytes())["traceEvents"]
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
                "launch_us": (owner.get("dur", 0.0) if owner else 0.0),
            }
        )
    kernels.sort(key=lambda item: item["ts"])
    return kernels


def load_collective_bytes(trace_path: Path, offset: float) -> list[dict]:
    """Payload bytes per NCCL kernel.

    c10d cpu_ops carry no sizes in this PyTorch build, but the DDP hooks emit
    ``record_param_comms`` with ``In msg nelems``, so the payload is taken from
    there and hung on the first NCCL kernel launched at or after the op.
    """
    events = json.loads(Path(trace_path).read_bytes())["traceEvents"]
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
        if event.get("cat") != "cpu_op" or "dur" not in event:
            continue
        args = event.get("args", {})
        name = str(event.get("name", ""))
        if name == "record_param_comms":
            collective = str(args.get("Collective name", ""))
            if collective in ("wait", "barrier", ""):
                continue
            elements = max(int(args.get("In msg nelems") or 0), int(args.get("Out msg nelems") or 0))
            if "all_gather" in collective or "allgather" in collective:
                elements = int(args.get("In msg nelems") or 0)
            size = elements * dtype_bytes.get(str(args.get("dtype", "")), 4)
            if size > 0:
                ops.append({"ts": event["ts"], "bytes": size, "name": collective})
    per_kernel: dict[int, int] = collections.defaultdict(int)
    for op in sorted(ops, key=lambda item: item["ts"]):
        target = next((kernel for kernel in kernels if kernel["ts"] >= op["ts"]), None)
        if target is not None:
            per_kernel[id(target)] += op["bytes"]
    rows = []
    for kernel in kernels:
        rows.append(
            {
                "ts": kernel["ts"],
                "epoch": kernel["ts"] / 1e6 + offset,
                "dur_ms": kernel["dur"] / 1e3,
                "bytes": per_kernel.get(id(kernel), 0),
            }
        )
    return rows


def union_ms(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def series_from_records(records: list[dict], bucket_s: float) -> dict[int, float]:
    buckets: dict[int, float] = collections.defaultdict(float)
    for record in records:
        buckets[int(record["epoch"] / bucket_s)] += float(record["bytes"])
    return {index: value / bucket_s for index, value in buckets.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repeat", default="r0")
    parser.add_argument("--offset-local-ns", type=int, required=True)
    parser.add_argument("--offset-remote-ns", type=int, required=True)
    parser.add_argument("--torchrec-lane", default="torchrec_hbm_b3072_d128")
    parser.add_argument("--bucket-ms", type=float, default=0.5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    setup_fonts()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lane = run_dir / "outputs" / f"rdma_b3072_d128_{args.repeat}"

    events = load_jsonl(lane / "events.rank0.jsonl")
    steps = step_records(events)
    trace_path = sorted((lane / "traces").glob("*/rank0.pt.trace.json"))[0]
    trace_events = json.loads(trace_path.read_bytes())["traceEvents"]
    prof = profiler_steps(trace_events)
    profiled = sorted(prof)

    # CUPTI clock -> epoch, anchored on the ProfilerStep that matches each step.
    diffs = [
        step["start"] - prof[step["step"]]["ts"] / 1e6
        for step in steps
        if step["step"] in prof
    ]
    baseline = st.median(diffs)
    offset = st.mean([value for value in diffs if abs(value - baseline) < 1.0])
    print(f"clock alignment: offset={offset:.6f}s spread={(max(diffs) - min(diffs)) * 1e3:.3f}ms")

    window = [step for step in steps if step["step"] in prof]
    window_start = window[0]["start"]
    window_stop = window[-1]["end"]

    kernels = load_kernels(trace_path, offset)
    for kernel in kernels:
        kernel["phase"] = "outside"
        kernel["profiled_step"] = None
        for step in steps:
            if step["start"] <= kernel["epoch"] < step["end"]:
                owner = step
                kernel["profiled_step"] = step["step"]
                for name in PHASE_ORDER:
                    span = step["phases"].get(name)
                    if span and "start" in span and "end" in span and span["start"] <= kernel["epoch"] < span["end"]:
                        kernel["phase"] = name
                        break
                else:
                    kernel["phase"] = "other"
                break
    window_kernels = [
        kernel for kernel in kernels if window_start <= kernel["epoch"] < window_stop
    ]

    nccl_rows = load_collective_bytes(trace_path, offset)
    wire: list[dict] = []
    for path, offset_ns in (
        (run_dir / "wire" / "rdma_wire.jsonl", args.offset_local_ns),
        (run_dir / "wire.remote" / "rdma_wire.jsonl", args.offset_remote_ns),
    ):
        if not Path(path).is_file():
            continue
        for record in load_jsonl(Path(path)):
            record["epoch"] = record["t"] / 1e9 + offset_ns / 1e9
            record["is_server"] = int(record["self"]) < 2
            record["cross_host"] = (int(record["self"]) % 2) != (int(record["node"]) % 2)
            wire.append(record)
    wire_window = [record for record in wire if window_start <= record["epoch"] < window_stop]

    bucket_s = args.bucket_ms / 1000.0

    def ps(record: dict, *, up: bool, cross: bool, bulk: bool | None) -> bool:
        if record["is_server"] == up:  # up: client -> server, so is_server must be False
            return False
        if record["cross_host"] != cross:
            return False
        if int(record["bytes"]) <= 1024:
            return False
        if bulk is None:
            return True
        return (int(record["bytes"]) > 200_000) == bulk

    flow_defs = [
        ("ps_up_get", "PS 上行·查表请求（跨机）", lambda r: ps(r, up=True, cross=True, bulk=False), "#c0392b", "-", None),
        ("ps_up_bulk", "PS 上行·写回（跨机，rank0 独有）", lambda r: ps(r, up=True, cross=True, bulk=True), "#e67e22", "-", None),
        ("ps_down", "PS 下行·查表结果（跨机）", lambda r: ps(r, up=False, cross=True, bulk=None), "#2980b9", "-", None),
        ("ps_loop_up", "PS 同机环回（不跨机链路）", lambda r: ps(r, up=True, cross=False, bulk=None), "#95a5a6", "--", None),
        ("ps_loop_down", "PS 同机环回（下行）", lambda r: ps(r, up=False, cross=False, bulk=None), "#bdc3c7", ":", None),
    ]
    flow_series = {
        key: series_from_records([r for r in wire_window if pred(r)], bucket_s)
        for key, _, pred, _, _, _ in flow_defs
    }
    nccl_series: dict[int, float] = collections.defaultdict(float)
    for row in nccl_rows:
        if not (window_start <= row["epoch"] < window_stop) or row["bytes"] == 0:
            continue
        rate = row["bytes"] / max(row["dur_ms"] / 1e3, 1e-6)
        index = int(row["epoch"] / bucket_s)
        span = max(1, int(round(row["dur_ms"] / (args.bucket_ms))))
        for slot in range(index, index + span):
            nccl_series[slot] = max(nccl_series[slot], rate)
    nccl_series = {index: value / 1e6 for index, value in nccl_series.items()}
    for key in flow_series:
        flow_series[key] = {index: value / 1e6 for index, value in flow_series[key].items()}

    # ---------------------------------------------------------------- figure 1
    fig = plt.figure(figsize=(17, 17.5))
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[2.0, 3.4, 1.25, 1.6, 1.25],
        hspace=0.42,
        left=0.075, right=0.985, top=0.955, bottom=0.045,
    )
    ax_gpu = fig.add_subplot(gs[0])
    ax_host = fig.add_subplot(gs[1], sharex=ax_gpu)
    ax_nccl = fig.add_subplot(gs[2], sharex=ax_gpu)
    ax_ps = fig.add_subplot(gs[3], sharex=ax_gpu)
    ax_cmp = fig.add_subplot(gs[4])

    def x_of(epoch: float) -> float:
        return (epoch - window_start) * 1e3

    max_x = x_of(window_stop)

    # --- row 1: GPU kernels by phase -------------------------------------
    busy = union_ms([(k["epoch"], k["epoch"] + k["dur_ms"] / 1e3) for k in window_kernels])
    span_ms = (window_stop - window_start) * 1e3
    bin_ms = 0.25
    n_bins = int(span_ms / bin_ms) + 1
    bin_phase: list[str | None] = [None] * n_bins
    bin_weight: list[float] = [0.0] * n_bins
    weight = collections.defaultdict(float)
    for kernel in window_kernels:
        start = x_of(kernel["epoch"])
        first = max(0, int(start / bin_ms))
        last = min(n_bins - 1, int((start + kernel["dur_ms"]) / bin_ms))
        for index in range(first, last + 1):
            weight[(index, kernel["phase"])] += kernel["dur_ms"]
    for index in range(n_bins):
        best, best_weight = None, 0.0
        for key, value in weight.items():
            if key[0] == index and value > best_weight:
                best, best_weight = key[1], value
        bin_phase[index] = best
    for index in range(n_bins):
        if bin_phase[index] is None:
            continue
        ax_gpu.add_patch(
            Rectangle(
                (index * bin_ms, 0.62), bin_ms, 0.36,
                color=PHASE_COLORS.get(bin_phase[index], "#bdc3c7"), lw=0,
            )
        )
    for kernel in window_kernels:
        x0 = x_of(kernel["epoch"])
        ax_gpu.add_patch(
            Rectangle(
                (x0, 0.40), max(kernel["dur_ms"], 0.012), 0.18,
                color=PHASE_COLORS.get(kernel["phase"], "#bdc3c7"), alpha=0.95, lw=0,
            )
        )
    ax_gpu.set_ylim(0, 1.0)
    ax_gpu.set_yticks([])
    ax_gpu.set_title(
        "① RecStore-RDMA（rank0）GPU 时间轴：上带 = GPU 占用（白 = 无核），中带 = 逐核条码（0.012 ms 起），"
        "下方标尺 = 阶段窗口"
        f"（窗口 {span_ms:.0f} ms，GPU 忙 {busy * 1e3:.0f} ms = {busy * 1e3 / span_ms * 100:.0f}%）",
        fontsize=12, pad=22,
    )
    for step in window:
        ax_gpu.axvline(x_of(step["start"]), color="#34495e", lw=0.6, ls="--", alpha=0.55)
    # phase ruler
    for step in window:
        for name in PHASE_ORDER:
            span = step["phases"].get(name)
            if not span or "start" not in span or "end" not in span:
                continue
            y = 0.0 if name != "prepare" else 0.14
            height = 0.13 if name != "prepare" else 0.12
            ax_gpu.add_patch(
                Rectangle(
                    (x_of(span["start"]), y), (span["end"] - span["start"]) * 1e3, height,
                    color=PHASE_COLORS[name], alpha=0.92, lw=0,
                )
            )
    ax_gpu.legend(
        handles=[Patch(facecolor=PHASE_COLORS[name], label=PHASE_LABELS[name]) for name in PHASE_ORDER],
        loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=8, fontsize=8.5, framealpha=0.9,
    )

    # --- row 2: host phases + bagpipe pipeline ---------------------------
    ax_host.set_ylim(-0.6, len(window) - 0.4)
    ax_host.invert_yaxis()
    ax_host.set_yticks(range(len(window)))
    ax_host.set_yticklabels([f"step {step['step']}" for step in window], fontsize=8.5)
    for step in window:
        for name in PHASE_ORDER:
            span = step["phases"].get(name)
            if not span or "start" not in span or "end" not in span:
                continue
            ax_host.add_patch(
                Rectangle(
                    (x_of(span["start"]), step["step"] - window[0]["step"]),
                    (span["end"] - span["start"]) * 1e3, 1.0,
                    color=PHASE_COLORS[name], alpha=0.45 if name == "prepare" else 0.9, lw=0,
                )
            )
    for step in window:
        ax_host.axvline(x_of(step["start"]), color="#34495e", lw=0.6, ls="--", alpha=0.55)
    # bagpipe lead-time arrows: prepare(N+4) -> consume(N+4)
    lookahead = 4
    lead_times = []
    window_ids = {step["step"] for step in window}
    for step in window:
        target_step = step["step"] + lookahead
        target = next((item for item in steps if item["step"] == target_step), None)
        if target is None or target_step not in window_ids:
            continue
        prep = step["phases"].get("prepare")
        consume = target["phases"].get("consume")
        if not prep or "start" not in prep or not consume or "start" not in consume:
            continue
        lead_times.append(consume["start"] - prep["end"])
        ax_host.add_patch(
            FancyArrowPatch(
                (x_of(prep["end"]), step["step"] - window[0]["step"] + 0.95),
                (x_of(consume["start"]), target["step"] - window[0]["step"] + 0.05),
                arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#8e44ad", alpha=0.75,
                connectionstyle="arc3,rad=-0.25",
            )
        )
    lead_ms = st.mean(lead_times) * 1e3 if lead_times else 0.0
    ax_host.set_title(
        "② 主机侧阶段 + BagPipe 预取流水：紫箭头 = 「step N 的 prepare 为 step N+4 预取」→ step N+4 的 consume"
        f"（平均提前 {lead_ms:.0f} ms 发出，正好 4 步）",
        fontsize=12,
    )
    ax_host.text(
        0.005, 0.985,
        "prepare 段 = 在稀疏更新期间为 N+4 步入队（每步实测 40 条 RDMA GET，2 shard × 20，全部落在该窗口内）；"
        "consume 段 = wait_and_get 取回，而响应在 0.3 ms 后就已经回写，所以 wait 基本不阻塞主链",
        transform=ax_host.transAxes, fontsize=8.5, va="top", color="#2c3e50",
    )

    # --- row 3: NCCL ------------------------------------------------------
    items = sorted(nccl_series.items())
    xs = [(index * bucket_s - window_start) * 1e3 for index, _ in items]
    nccl_per_step = sum(nccl_series.values()) * bucket_s / len(window)
    ax_nccl.plot(
        xs, [value for _, value in items], color="#2c3e50", lw=1.1,
        label=f"① 两卡通信 NCCL（DDP allreduce + emb all_gather）  {nccl_per_step:.1f} MB/步（本 rank 发送）",
    )
    ps_cross: dict[int, float] = collections.defaultdict(float)
    for key in ("ps_up_get", "ps_up_bulk", "ps_down"):
        for index, value in flow_series[key].items():
            ps_cross[index] += value
    ps_per_step = sum(ps_cross.values()) * bucket_s / len(window)
    ax_nccl.plot(
        [(index * bucket_s - window_start) * 1e3 for index in sorted(ps_cross)],
        [ps_cross[index] for index in sorted(ps_cross)],
        color="#e67e22", lw=1.1,
        label=f"② PS 数据面 RDMA（跨机查表收发 + 写回）  {ps_per_step:.1f} MB/步",
    )
    ax_nccl.axhline(LINK_MBPS, color="#95a5a6", lw=0.9, ls=":")
    ax_nccl.text(max_x * 0.995, LINK_MBPS * 1.05, "100 Gb/s 线速参考 12 500 MB/s", ha="right", va="bottom", fontsize=8, color="#7f8c8d")
    ax_nccl.set_ylabel("MB/s\n(symlog)")
    ax_nccl.set_yscale("symlog", linthresh=100.0)
    ax_nccl.set_ylim(0, LINK_MBPS * 1.6)
    ax_nccl.set_title(
        "③ 网口总览（两类流量共用同一个 mlx5_3 物理口，分开画）：NCCL 两卡通信 vs PS 数据面 RDMA",
        fontsize=12,
    )
    ax_nccl.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    cross_active: set[int] = set()
    for row in nccl_rows:
        if row["dur_ms"] <= 0 or row["bytes"] == 0:
            continue
        start = max(row["epoch"], window_start)
        stop = min(row["epoch"] + row["dur_ms"] / 1e3, window_stop)
        if stop <= start:
            continue
        for slot in range(int(start / bucket_s), int(stop / bucket_s) + 1):
            cross_active.add(slot)
    for record in wire_window:
        if record["cross_host"] and int(record["bytes"]) > 1024:
            cross_active.add(int(record["epoch"] / bucket_s))
    busy_ms = len(cross_active) * args.bucket_ms
    busy_frac = busy_ms / ((window_stop - window_start) * 1e3)
    ax_nccl.text(
        0.005, 0.985,
        "NCCL 峰值 ~10 GB/s 贴着线速，但两类跨机流量都不连续：0.5 ms 桶并集口径下，跨机链路有流量的时间 "
        f"{busy_ms:.0f} ms / {max_x:.0f} ms = {busy_frac*100:.0f}%，其余 {100-busy_frac*100:.0f}% 空闲——卡住的是主机串行，不是带宽",
        transform=ax_nccl.transAxes, fontsize=8.5, va="top", color="#2c3e50",
    )

    # --- row 4: PS RDMA ---------------------------------------------------
    for key, label, _, color, style, _ in flow_defs:
        data = flow_series[key]
        if not data:
            continue
        xs = [(index * bucket_s - window_start) * 1e3 for index in sorted(data)]
        ys = [data[index] * (-1 if key == "ps_down" else 1) for index in sorted(data)]
        per_step = sum(data.values()) * bucket_s / len(window)
        ax_ps.plot(xs, ys, style, color=color, lw=1.1, label=f"{label}  {per_step:.2f} MB/步")
    ax_ps.axhline(0, color="#7f8c8d", lw=0.6)
    ax_ps.set_yscale("symlog", linthresh=100.0)
    ax_ps.set_ylabel("上行 + / 下行 −\nMB/s（symlog）")
    total_up = sum(sum(v.values()) for k, v in flow_series.items() if k.startswith("ps_up"))
    total_down = sum(sum(v.values()) for k, v in flow_series.items() if k == "ps_down")
    ax_ps.set_title(
        "④ PS 数据面 RDMA 流量（按记录类型拆分）：查表请求 / rank0 写回 / 查表结果，虚线为同机环回（不跨机链路）",
        fontsize=12,
    )
    ax_ps.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax_ps.set_xlabel("时间（ms，profiled 窗口起点 = 0）", fontsize=10)

    # --- row 5: TorchRec reference lane ----------------------------------
    torchrec_trace = sorted((run_dir / "outputs" / f"{args.torchrec_lane}_{args.repeat}" / "torchrec_traces").glob("*/rank0.pt.trace.json"))
    rec_rows = list(csv.DictReader((lane / "recstore_main.csv").open()))
    rec_steady = [row for row in rec_rows if int(row["step"]) >= 20]
    rec_step_ms = st.mean(float(row["step_total_ms"]) for row in rec_steady)
    world = 2  # 2 clients x 1 GPU; the step time covers both ranks
    rec_samples = st.mean(
        float(row["batch_size"]) * world * 1000.0 / float(row["step_total_ms"]) for row in rec_steady
    )
    rec_busy_ms = st.mean(
        union_ms(
            sorted(
                (kernel["epoch"], kernel["epoch"] + kernel["dur_ms"] / 1e3)
                for kernel in window_kernels
                if kernel["profiled_step"] == step["step"]
            )
        ) * 1e3
        for step in window
    )
    tr_summary = ""
    if torchrec_trace:
        tr_events = json.loads(torchrec_trace[0].read_bytes())["traceEvents"]
        tr_prof = profiler_steps(tr_events)
        tr_ids = sorted(tr_prof)
        tr_durations = [tr_prof[index]["dur"] / 1e3 for index in tr_ids]
        clean = [value for value in tr_durations if value < 3 * st.median(tr_durations)]
        tr_step_ms = st.mean(clean)
        tr_kernels = [
            event
            for event in tr_events
            if event.get("cat") == "kernel"
            and "dur" in event
            and not (event["ts"] < tr_prof[tr_ids[0]]["ts"] + 5e5 and event["dur"] / 1e3 > 50)
        ]
        tr_window = [
            event
            for event in tr_events
            if event.get("cat") == "kernel"
            and "dur" in event
            and tr_prof[tr_ids[1]]["ts"] <= event["ts"] <= tr_prof[tr_ids[-1]]["ts"] + tr_prof[tr_ids[-1]]["dur"]
        ]
        tr_busy_ms = union_ms(
            sorted(
                (event["ts"] / 1e6, (event["ts"] + event["dur"]) / 1e6)
                for event in tr_window
            )
        ) * 1e3 / max(len(tr_ids) - 1, 1)
        tr_count_ms = len(tr_window) / max(len(tr_ids) - 1, 1)
        tr_comm = sum(
            max(int(event.get("args", {}).get("In msg nelems") or 0), int(event.get("args", {}).get("Out msg nelems") or 0))
            * 4
            for event in tr_events
            if event.get("name") == "record_param_comms"
            and str(event.get("args", {}).get("Collective name", "")).startswith("all_to_all")
        )
        tr_samples_rows = list(
            csv.DictReader(
                (run_dir / "outputs" / f"{args.torchrec_lane}_{args.repeat}" / "torchrec_main.csv").open()
            )
        )
        tr_steady = [row for row in tr_samples_rows if int(row["step"]) >= 20]
        tr_csv_step = st.mean(float(row["step_total_ms"]) for row in tr_steady)
        tr_samples = st.mean(
            float(row["batch_size"]) * 2 * 1000.0 / float(row["step_total_ms"]) for row in tr_steady
        )
    else:
        tr_step_ms = tr_csv_step = tr_samples = tr_comm = tr_busy_ms = tr_count_ms = 0.0

    labels = ["每步耗时（ms）", "GPU 核忙碌（ms/步）", "每步 GPU 核数", "跨机通信量（MB/步）"]
    rec_kernel_per_step = len(window_kernels) / len(window)
    ps_cross_per_step = sum(
        sum(flow_series[key].values()) * bucket_s
        for key in ("ps_up_get", "ps_up_bulk", "ps_down")
    ) / len(window)
    rec_comm_per_step = (
        sum(row["bytes"] for row in nccl_rows if window_start <= row["epoch"] < window_stop) / len(window) / 1e6
        + ps_cross_per_step
    )
    tr_comm_per_step = tr_comm / max(len(clean), 1) / 1e6
    rec_values = [rec_step_ms, rec_busy_ms, rec_kernel_per_step, rec_comm_per_step]
    tr_values = [tr_csv_step, tr_busy_ms, tr_count_ms, tr_comm_per_step]
    index = range(len(labels))
    width = 0.36
    ax_cmp.bar([i - width / 2 for i in index], rec_values, width, color="#c0392b", label=f"RecStore-RDMA（{rec_samples / 1e3:.1f}K samples/s，2 卡合计）")
    ax_cmp.bar([i + width / 2 for i in index], tr_values, width, color="#2980b9", label=f"TorchRec-HBM（{tr_samples / 1e3:.1f}K samples/s，2 卡合计）")
    for i, (a, b) in enumerate(zip(rec_values, tr_values)):
        ax_cmp.text(i - width / 2, a, f"{a:,.1f}", ha="center", va="bottom", fontsize=8.5)
        ax_cmp.text(i + width / 2, b, f"{b:,.1f}", ha="center", va="bottom", fontsize=8.5)
    ax_cmp.set_xticks(list(index))
    ax_cmp.set_xticklabels(labels, fontsize=9.5)
    ax_cmp.set_yscale("log")
    ax_cmp.set_ylabel("数值（log）")
    ax_cmp.set_title("⑤ 对照 lane：TorchRec-HBM（同 bs=3072、同 2n1g、同 profiled 步）", fontsize=12)
    ax_cmp.legend(loc="upper left", fontsize=9)

    fig.savefig(out_dir / "recstore_2n1g_step_timeline.png", dpi=150)
    fig.savefig(out_dir / "recstore_2n1g_step_timeline.svg")
    plt.close(fig)
    print("wrote", out_dir / "recstore_2n1g_step_timeline.png")

    # ---------------------------------------------------------------- figure 2
    fig2 = plt.figure(figsize=(16.5, 11.5))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.34, wspace=0.24, left=0.085, right=0.98, top=0.93, bottom=0.07)

    kind_count = collections.Counter(kernel["kind"] for kernel in window_kernels)
    kind_time = collections.Counter()
    for kernel in window_kernels:
        kind_time[kernel["kind"]] += kernel["dur_ms"]
    total_count = sum(kind_count.values())
    total_time = sum(kind_time.values())
    kinds = [kind for kind in KIND_ORDER if kind_count[kind]]
    kinds.sort(key=lambda kind: -kind_time[kind])
    ax_b = fig2.add_subplot(gs2[0,0])
    y_positions = [0, 1]
    left = 0.0
    for kind in kinds:
        share = kind_count[kind] / total_count
        ax_b.barh(y_positions[0], share, left=left, color=KIND_COLORS[kind], height=0.55)
        if share > 0.04:
            ax_b.text(left + share / 2, y_positions[0], f"{share * 100:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += share
    left = 0.0
    for kind in kinds:
        share = kind_time[kind] / total_time
        ax_b.barh(y_positions[1], share, left=left, color=KIND_COLORS[kind], height=0.55)
        if share > 0.05:
            ax_b.text(left + share / 2, y_positions[1], f"{share * 100:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += share
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([f"核数占比\n({total_count} 个)", f"GPU 时间占比\n({total_time:.0f} ms)"], fontsize=9.5)
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax_b.legend(
        handles=[Patch(facecolor=KIND_COLORS[kind], label=f"{KIND_LABELS[kind]}") for kind in kinds],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8,
    )
    ax_b.set_title(
        f"B. GPU 在干什么：核数占比 vs GPU 时间占比（时间按核时长求和，多流重叠会双重计入；"
        f"窗口并集 {busy * 1e3:.0f} ms）",
        fontsize=11.5,
    )

    ax_c = fig2.add_subplot(gs2[0,1])
    durations = sorted(kernel["dur_ms"] for kernel in window_kernels)
    bins = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1, 3e1, 1e2]
    counts = [0] * (len(bins) - 1)
    times = [0.0] * (len(bins) - 1)
    for value in durations:
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                counts[i] += 1
                times[i] += value
                break
    centres = [(bins[i] * bins[i + 1]) ** 0.5 for i in range(len(bins) - 1)]
    ax_c.bar(centres, counts, width=[(bins[i + 1] - bins[i]) * 0.7 for i in range(len(bins) - 1)], color="#7f8c8d", label="核数")
    ax_c.set_xscale("log")
    ax_c.set_ylabel("核数", color="#7f8c8d")
    ax_c.set_xlabel("单个 kernel 时长（ms，log）")
    ax_c2 = ax_c.twinx()
    cumulative = []
    running = 0.0
    for value in times:
        running += value
        cumulative.append(running / total_time)
    ax_c2.plot(centres, cumulative, color="#c0392b", marker="o", ms=3.5, lw=1.4, label="累计 GPU 时间占比")
    ax_c2.set_ylabel("累计 GPU 时间占比", color="#c0392b")
    ax_c2.set_ylim(0, 1.02)
    big = sum(1 for value in durations if value > 0.5)
    big_time = sum(value for value in durations if value > 0.5)
    ax_c.set_title(
        f"C. kernel 时长分布：>0.5 ms 的 {big} 个核（{big / total_count * 100:.1f}%）占 {big_time / total_time * 100:.1f}% GPU 时间",
        fontsize=12,
    )
    ax_c2.legend(loc="lower right", fontsize=8.5)
    ax_c.legend(loc="upper left", fontsize=8.5)

    ax_d = fig2.add_subplot(gs2[1,0])
    launcher_stats: dict[int, dict] = collections.defaultdict(lambda: {"count": 0, "launch_us": 0.0, "gpu_ms": 0.0})
    for kernel in window_kernels:
        tid = kernel["tid"] if kernel["tid"] is not None else -1
        launcher_stats[tid]["count"] += 1
        launcher_stats[tid]["launch_us"] += kernel["launch_us"]
        launcher_stats[tid]["gpu_ms"] += kernel["dur_ms"]
    ordered = sorted(launcher_stats.items(), key=lambda item: -item[1]["count"])
    main_tid = ordered[0][0]
    names = []
    for tid, _ in ordered[:4]:
        label = "主线程" if tid == main_tid else ("autograd 线程" if tid != -1 else "无归属")
        names.append(f"{label}\ntid={tid}")
    count_share = [stats["count"] / total_count for _, stats in ordered[:4]]
    launch_total = sum(stats["launch_us"] for _, stats in ordered)
    launch_share = [stats["launch_us"] / launch_total for _, stats in ordered[:4]]
    positions = range(len(count_share))
    ax_d.bar([p - 0.2 for p in positions], count_share, 0.4, color="#2980b9", label="下发核数占比")
    ax_d.bar([p + 0.2 for p in positions], launch_share, 0.4, color="#c0392b", label="下发 CPU 时间占比")
    for i, (a, b) in enumerate(zip(count_share, launch_share)):
        ax_d.text(i - 0.2, a, f"{a * 100:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax_d.text(i + 0.2, b, f"{b * 100:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax_d.set_xticks(list(positions))
    ax_d.set_xticklabels(names, fontsize=9)
    ax_d.set_ylabel("占比")
    ax_d.legend(fontsize=8.5)
    detail = "  ".join(
        f"{'主线程' if tid == main_tid else 'autograd'}: 均值 {stats['launch_us'] / max(stats['count'], 1):.1f} µs/核"
        for tid, stats in ordered[:2]
    )
    ax_d.set_title(f"D. 谁下发这些核（D1 阶段 + D2 线程合并）：{detail}", fontsize=12)

    ax_e = fig2.add_subplot(gs2[1,1])
    sub = collections.defaultdict(lambda: collections.Counter())
    for kernel in window_kernels:
        if kernel["phase"] in ("fwd", "bwd"):
            sub[kernel["phase"]][kernel["op"] or kernel["name"].split("(")[0][:38]] += 1
    rows = []
    for phase in ("fwd", "bwd"):
        for op, count in sub[phase].most_common(7):
            rows.append((phase, op, count))
    labels = [f"{op[:34]}" for _, op, _ in rows]
    values = [count for _, _, count in rows]
    colors = ["#27ae60" if phase == "fwd" else "#e67e22" for phase, _, _ in rows]
    ax_e.barh(range(len(rows)), values, color=colors)
    ax_e.set_yticks(range(len(rows)))
    ax_e.set_yticklabels(labels, fontsize=8)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("窗口内 kernel 数")
    fwd_total = sum(sub["fwd"].values())
    bwd_total = sum(sub["bwd"].values())
    ax_e.legend(handles=[Patch(facecolor="#27ae60", label=f"fwd 共 {fwd_total} 个核"),
                         Patch(facecolor="#e67e22", label=f"bwd 共 {bwd_total} 个核")], fontsize=8.5)
    phase_count = collections.Counter(kernel["phase"] for kernel in window_kernels)
    phase_ms = collections.Counter()
    for kernel in window_kernels:
        phase_ms[kernel["phase"]] += kernel["dur_ms"]
    phase_note = "  ".join(
        f"{name} {phase_count[name]}个/{phase_ms[name]:.0f}ms"
        for name in ("sparse", "consume", "bwd", "fwd")
        if phase_count[name]
    )
    ax_e.set_title(
        f"E. 前向/反向内部：哪个 sub-op 发了这么多核\n"
        f"（窗口内全部阶段：{phase_note}）",
        fontsize=11.5,
    )

    fig2.savefig(out_dir / "recstore_2n1g_kernel_breakdown.png", dpi=150)
    fig2.savefig(out_dir / "recstore_2n1g_kernel_breakdown.svg")
    plt.close(fig2)
    print("wrote", out_dir / "recstore_2n1g_kernel_breakdown.png")

    # ---------------------------------------------------------------- tables
    with (run_dir / "analysis" / "nic_flows_window.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flow", "MB_per_window", "MB_per_step"])
        for key, label, _, _, _, _ in flow_defs:
            total = sum(flow_series[key].values()) * args.bucket_ms / 1000.0
            writer.writerow([label, f"{total:.3f}", f"{total / len(window):.3f}"])
        writer.writerow(["NCCL 两卡通信", f"{sum(nccl_series.values()) * args.bucket_ms / 1000:.3f}", f"{sum(nccl_series.values()) * args.bucket_ms / 1000 / len(window):.3f}"])
    with (run_dir / "analysis" / "kernel_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "count", "count_share", "gpu_ms", "time_share"])
        for kind in kinds:
            writer.writerow([kind, kind_count[kind], f"{kind_count[kind] / total_count:.4f}", f"{kind_time[kind]:.3f}", f"{kind_time[kind] / total_time:.4f}"])
    with (run_dir / "analysis" / "phase_kernels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "count", "gpu_ms"])
        by_phase = collections.Counter(kernel["phase"] for kernel in window_kernels)
        phase_ms = collections.Counter()
        for kernel in window_kernels:
            phase_ms[kernel["phase"]] += kernel["dur_ms"]
        for phase, count in by_phase.most_common():
            writer.writerow([phase, count, f"{phase_ms[phase]:.3f}"])

    print(json.dumps({
        "window_ms": span_ms,
        "gpu_busy_ms": busy * 1e3,
        "gpu_busy_pct": busy * 1e3 / span_ms * 100,
        "kernels": len(window_kernels),
        "ps_flows_MB_per_step": {
            label: round(sum(flow_series[key].values()) * args.bucket_ms / 1000.0 / len(window), 2)
            for key, label, _, _, _, _ in flow_defs
        },
        "nccl_MB_per_step": round(sum(nccl_series.values()) * args.bucket_ms / 1000.0 / len(window), 2),
        "lead_time_ms": round(lead_ms, 1),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
