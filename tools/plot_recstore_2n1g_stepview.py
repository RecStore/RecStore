#!/usr/bin/env python3
"""A 图（Quanta figA 风格）：2n1g bs=3072，一步内 GPU 在忙什么 + 网卡什么时候有流量。

横轴 = 一步内的时间。每个 arm 两块共用 x 轴：
  上：GPU 占用（堆叠：前向 / 反向 / NCCL / 稀疏更新 / BagPipe 取数 / 其他），
      背景斜纹 = 空窗；顶上一条是主机阶段标尺。
  下：同一时间轴的跨机网卡带宽，两条流量分开堆叠——PS 数据面 RDMA（上行/下行）
      与两卡 NCCL。PS 侧 wire trace 只有发起时刻、没有单条时长，画的是「每桶字节 /
      桶宽」的等效速率（上界），口径写在脚注里。

arm ① RecStore-RDMA（BagPipe），取 profiled 窗口里步时居中的那一步
arm ② TorchRec-HBM 对照（同 bs、同 2n1g，同样取步时居中的 profiled 步）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

BASE_PATH = Path(__file__).resolve().parent / "plot_recstore_2n1g.py"
_spec = importlib.util.spec_from_file_location("recstore_base", BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

C = {
    "fwd": "#6AA84F",
    "bwd": "#38761D",
    "ar": "#2A9D8F",
    "ag": "#8AB17D",
    "sparse": "#CC4125",
    "consume": "#3D85C6",
    "other": "#9E9E9E",
    "idle": "#F7F7F7",
    "idle_e": "#C9C9C9",
    "ps_up": "#E67E22",
    "ps_down": "#F5B041",
    "get": "#8E44AD",
    "warn": "#CC4125",
    "ink": "#222222",
}

PHASE_CAT = {
    "fwd": "fwd",
    "bwd": "bwd",
    "sparse": "sparse",
    "consume": "consume",
    "prepare": "consume",
    "pack": "other",
    "dense_opt": "other",
    "barrier": "other",
    "other": "other",
}
PHASE_CN = {
    "consume": "取数 + 查表（BagPipe wait / id 处理）",
    "pack": "拼 batch",
    "fwd": "dense 前向",
    "bwd": "反向 + DDP 规约",
    "dense_opt": "稠密优化器",
    "sparse": "稀疏更新（改表回写）",
    "prepare": "预取（为 step N+4 入队）",
    "barrier": "barrier",
}
CAT_LABEL = {
    "fwd": "前向 kernel",
    "bwd": "反向 kernel",
    "ar": "NCCL allreduce / 集合通信",
    "ag": "NCCL all_gather / all_to_all",
    "sparse": "稀疏更新 kernel",
    "consume": "BagPipe 取数 / id 处理 kernel",
    "other": "其他 kernel",
}


def setup() -> None:
    for candidate in (
        Path.home() / ".local/share/fonts/noto-sc/NotoSansSC-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans SC",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.9,
            "text.color": C["ink"],
            "axes.labelcolor": C["ink"],
            "xtick.color": "#555555",
            "ytick.color": "#555555",
        }
    )


def n_bins_of(step: dict, bin_ms: float) -> int:
    return max(1, int(round(step["dur"] * 1e3 / bin_ms)))


def gpu_coverage(step: dict, kernels: list[dict], cats: list[str], bin_ms: float) -> np.ndarray:
    """每个 bin 里各类 kernel 覆盖的时间占比（bin 宽归一化）。"""
    total = n_bins_of(step, bin_ms)
    arr = np.zeros((total, len(cats)))
    index_of = {name: i for i, name in enumerate(cats)}
    for kernel in kernels:
        if kernel.get("cat") not in index_of:
            continue
        start = max(kernel["epoch"], step["start"])
        stop = min(kernel["epoch"] + kernel["dur_ms"] / 1e3, step["end"])
        if stop <= start:
            continue
        first = int((start - step["start"]) * 1e3 / bin_ms)
        last = int((stop - step["start"]) * 1e3 / bin_ms)
        for index in range(max(0, first), min(total, last + 1)):
            lo = max(start, step["start"] + index * bin_ms / 1e3)
            hi = min(stop, step["start"] + (index + 1) * bin_ms / 1e3)
            arr[index, index_of[kernel["cat"]]] += (hi - lo) * 1e3 / bin_ms
    return np.clip(arr, 0.0, 1.0)


def nic_bytes(step: dict, records: list[dict], bin_ms: float) -> np.ndarray:
    """每条记录把 bytes 放进它所在的 bin，返回 GB/s（每桶字节 / 桶宽）。"""
    total = n_bins_of(step, bin_ms)
    arr = np.zeros(total)
    for record in records:
        if not (step["start"] <= record["epoch"] < step["end"]):
            continue
        index = int((record["epoch"] - step["start"]) * 1e3 / bin_ms)
        if 0 <= index < total:
            arr[index] += float(record["bytes"])
    return arr / (bin_ms / 1e3) / 1e9


def ps_sliding_rate(step: dict, records: list[dict], bin_ms: float, window_ms: float = 2.0) -> np.ndarray:
    """PS 侧只有发起时刻，单条时长未知：用 ±window/2 的滑窗把字节摊开，画「滑窗等效速率」。"""
    total = n_bins_of(step, bin_ms)
    arr = np.zeros(total)
    half = window_ms / 2.0
    for record in records:
        if not (step["start"] <= record["epoch"] < step["end"]):
            continue
        centre = (record["epoch"] - step["start"]) * 1e3
        first = int((centre - half) / bin_ms)
        last = int((centre + half) / bin_ms)
        for index in range(max(0, first), min(total, last + 1)):
            lo = max(centre - half, index * bin_ms)
            hi = min(centre + half, (index + 1) * bin_ms)
            if hi > lo:
                arr[index] += float(record["bytes"]) * (hi - lo) / window_ms
    return arr / (bin_ms / 1e3) / 1e9


def load_collectives(trace_path: Path) -> list[dict]:
    """和 base.load_collective_bytes 一样解析 record_param_comms，但保留集合通信类型。"""
    events = json.loads(Path(trace_path).read_bytes())["traceEvents"]
    dtype_bytes = {"Byte": 1, "Float": 4, "float": 4, "Half": 2, "half": 2,
                   "Bfloat16": 2, "bfloat16": 2, "Long": 8, "int64": 8, "Int": 4}
    kernels = sorted(
        (event for event in events
         if event.get("cat") == "kernel" and "dur" in event
         and "nccl" in str(event.get("name", "")).lower()),
        key=lambda item: item["ts"],
    )
    ops = []
    for event in events:
        if event.get("cat") != "cpu_op" or "dur" not in event:
            continue
        if str(event.get("name", "")) != "record_param_comms":
            continue
        args = event.get("args", {})
        collective = str(args.get("Collective name", ""))
        if collective in ("wait", "barrier", ""):
            continue
        elements = max(int(args.get("In msg nelems") or 0), int(args.get("Out msg nelems") or 0))
        if "all_gather" in collective or "allgather" in collective:
            elements = int(args.get("In msg nelems") or 0)
        size = elements * dtype_bytes.get(str(args.get("dtype", "")), 4)
        if size > 0:
            ops.append({"ts": event["ts"], "bytes": size, "collective": collective})
    per_kernel: dict[int, dict] = {}
    for op in sorted(ops, key=lambda item: item["ts"]):
        target = next((kernel for kernel in kernels if kernel["ts"] >= op["ts"]), None)
        if target is None:
            continue
        entry = per_kernel.setdefault(id(target), {"bytes": 0, "collectives": set()})
        entry["bytes"] += op["bytes"]
        entry["collectives"].add(op["collective"])
    rows = []
    for kernel in kernels:
        entry = per_kernel.get(id(kernel))
        rows.append(
            {
                "epoch": kernel["ts"] / 1e6,
                "dur_ms": kernel["dur"] / 1e3,
                "bytes": entry["bytes"] if entry else 0,
                "collectives": entry["collectives"] if entry else set(),
            }
        )
    return rows


def nccl_rate(step: dict, rows: list[dict], bin_ms: float) -> np.ndarray:
    """NCCL 内核的字节按内核时长摊到 bin 里：真实速率（GB/s）。"""
    total = n_bins_of(step, bin_ms)
    arr = np.zeros(total)
    for item in rows:
        if item["bytes"] == 0 or item["dur_ms"] <= 0:
            continue
        start = max(item["epoch"], step["start"])
        stop = min(item["epoch"] + item["dur_ms"] / 1e3, step["end"])
        if stop <= start:
            continue
        rate = item["bytes"] / item["dur_ms"] / 1e6  # MB/ms == GB/s
        first = int((start - step["start"]) * 1e3 / bin_ms)
        last = int((stop - step["start"]) * 1e3 / bin_ms)
        for index in range(max(0, first), min(total, last + 1)):
            lo = max(start, step["start"] + index * bin_ms / 1e3)
            hi = min(stop, step["start"] + (index + 1) * bin_ms / 1e3)
            arr[index] += rate * (hi - lo) * 1e3 / bin_ms
    return arr


def traffic_ms(step: dict, records: list[dict], bin_ms: float = 0.5) -> tuple[float, float]:
    """网卡有流量的时间（0.5 ms 粒度的桶，桶里字节 > 1 KB 才算）。"""
    slots: set[int] = set()
    total_bytes = 0
    for record in records:
        if not (step["start"] <= record["epoch"] < step["end"]):
            continue
        if int(record["bytes"]) <= 1024:
            continue
        slots.add(int((record["epoch"] - step["start"]) * 1e3 / bin_ms))
        total_bytes += int(record["bytes"])
    return len(slots) * bin_ms, total_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repeat", default="r0")
    parser.add_argument("--offset-local-ns", type=int, required=True)
    parser.add_argument("--offset-remote-ns", type=int, required=True)
    parser.add_argument("--torchrec-lane", default="torchrec_hbm_b3072_d128")
    parser.add_argument("--bin-ms", type=float, default=0.2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    setup()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lane = run_dir / "outputs" / f"rdma_b3072_d128_{args.repeat}"
    bin_ms = args.bin_ms

    # ---------------------------------------------------------- arm 1 data
    events = base.load_jsonl(lane / "events.rank0.jsonl")
    steps_all = base.step_records(events)
    trace_path = sorted((lane / "traces").glob("*/rank0.pt.trace.json"))[0]
    trace_events = json.loads(trace_path.read_bytes())["traceEvents"]
    prof = base.profiler_steps(trace_events)
    diffs = [s["start"] - prof[s["step"]]["ts"] / 1e6 for s in steps_all if s["step"] in prof]
    median_diff = st.median(diffs)
    offset = st.mean([d for d in diffs if abs(d - median_diff) < 1.0])
    print(f"clock alignment: offset={offset:.6f}s spread={(max(diffs) - min(diffs)) * 1e3:.3f}ms")

    kernels = base.load_kernels(trace_path, offset)
    for kernel in kernels:
        kernel["cat"] = "outside"
        for step in steps_all:
            if step["start"] <= kernel["epoch"] < step["end"]:
                for name in base.PHASE_ORDER:
                    span = step["phases"].get(name)
                    if span and "start" in span and "end" in span and span["start"] <= kernel["epoch"] < span["end"]:
                        kernel["cat"] = PHASE_CAT.get(name, "other")
                        break
                else:
                    kernel["cat"] = "other"
                break
        if kernel["kind"] == "nccl":
            kernel["cat"] = "ar" if "allreduce" in kernel["name"].lower() else "ag"

    profiled = [step for step in steps_all if step["step"] in prof]
    for step in profiled:
        step["dur"] = step["end"] - step["start"]
    median_dur = st.median(step["dur"] for step in profiled)
    step1 = min(profiled, key=lambda item: abs(item["dur"] - median_dur))
    step = step1

    nccl_rows = base.load_collective_bytes(trace_path, offset)
    wire: list[dict] = []
    for path, offset_ns in (
        (run_dir / "wire" / "rdma_wire.jsonl", args.offset_local_ns),
        (run_dir / "wire.remote" / "rdma_wire.jsonl", args.offset_remote_ns),
    ):
        for record in base.load_jsonl(path):
            record["epoch"] = record["t"] / 1e9 + offset_ns / 1e9
            record["is_server"] = int(record["self"]) < 2
            record["cross_host"] = (int(record["self"]) % 2) != (int(record["node"]) % 2)
            wire.append(record)

    ps_up = [r for r in wire if r["cross_host"] and not r["is_server"] and int(r["bytes"]) > 1024]
    ps_down = [r for r in wire if r["cross_host"] and r["is_server"] and int(r["bytes"]) > 1024]
    # 预取请求：客户端发起的跨机 12 992 B RDMA GET（64 B 头 + 1616 个 int64 key）
    ps_get = [r for r in wire if r["cross_host"] and not r["is_server"] and 4096 < int(r["bytes"]) <= 200_000]

    cats1 = ["fwd", "bwd", "sparse", "consume", "ar", "ag", "other"]
    cov1 = gpu_coverage(step, kernels, cats1, bin_ms)
    up1 = ps_sliding_rate(step, ps_up, bin_ms)
    down1 = ps_sliding_rate(step, ps_down, bin_ms)
    nccl1 = nccl_rate(step, nccl_rows, bin_ms)
    kernels1 = [k for k in kernels if step["start"] <= k["epoch"] < step["end"]]
    gpu_busy1 = base.union_ms(
        sorted((k["epoch"], k["epoch"] + k["dur_ms"] / 1e3) for k in kernels1)
    ) * 1e3
    nic_busy1, _ = traffic_ms(step, ps_up + ps_down)
    up_bytes1 = sum(int(r["bytes"]) for r in ps_up if step["start"] <= r["epoch"] < step["end"])
    down_bytes1 = sum(int(r["bytes"]) for r in ps_down if step["start"] <= r["epoch"] < step["end"])
    nccl_bytes1 = sum(
        row["bytes"] for row in nccl_rows if row["bytes"] > 0 and step["start"] <= row["epoch"] < step["end"]
    )
    nccl_busy1 = base.union_ms(
        [
            (row["epoch"], row["epoch"] + row["dur_ms"] / 1e3)
            for row in nccl_rows
            if row["bytes"] > 0 and step["start"] <= row["epoch"] < step["end"]
        ]
    ) * 1e3

    # 10 步窗口口径，供脚注引用
    window_start = profiled[0]["start"]
    window_stop = profiled[-1]["end"]
    window_kernels = [k for k in kernels if window_start <= k["epoch"] < window_stop]
    window_busy = base.union_ms(
        sorted((k["epoch"], k["epoch"] + k["dur_ms"] / 1e3) for k in window_kernels)
    ) * 1e3
    window_ms = (window_stop - window_start) * 1e3

    # ---------------------------------------------------------- arm 2 data
    tr_trace = sorted(
        (run_dir / "outputs" / f"{args.torchrec_lane}_{args.repeat}" / "torchrec_traces").glob("*/rank0.pt.trace.json")
    )[0]
    tr_events = json.loads(tr_trace.read_bytes())["traceEvents"]
    tr_steps_by_id: dict[int, dict] = {}
    for event in tr_events:
        name = str(event.get("name", ""))
        if not name.startswith("ProfilerStep#") or "dur" not in event:
            continue
        index = int(name.split("#", 1)[1])
        # 同名事件有两个线程各发一份，主线程那份时长更长
        if index not in tr_steps_by_id or event["dur"] > tr_steps_by_id[index]["dur_us"]:
            tr_steps_by_id[index] = {
                "start": event["ts"] / 1e6,
                "end": (event["ts"] + event["dur"]) / 1e6,
                "dur": event["dur"] / 1e6,
                "dur_us": event["dur"],
            }
    tr_profiled = sorted(tr_steps_by_id.values(), key=lambda item: item["start"])
    tr_median_dur = st.median(item["dur"] for item in tr_profiled)
    tr_step = min(tr_profiled, key=lambda item: abs(item["dur"] - tr_median_dur))

    back_intervals = [
        (event["ts"], event["ts"] + event.get("dur", 0.0))
        for event in tr_events
        if event.get("cat") == "cpu_op" and "evaluate_function" in str(event.get("name", ""))
    ]
    tr_ops = {
        str(event.get("args", {}).get("External id")): event
        for event in tr_events
        if event.get("cat") == "cpu_op" and event.get("dur") is not None
    }
    first_ts = min(item["start"] for item in tr_profiled) * 1e6
    tr_kernels: list[dict] = []
    for event in tr_events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        if event["ts"] < first_ts + 5e5 and event["dur"] / 1e3 > 50:
            continue
        owner = tr_ops.get(str(event.get("args", {}).get("External id")))
        name = str(event.get("name", ""))
        if "nccl" in name.lower():
            cat = "ar"
        elif owner is not None and (
            any(lo <= owner["ts"] <= hi for lo, hi in back_intervals)
            or "Backward" in str(owner.get("name", ""))
            or "AccumulateGrad" in str(owner.get("name", ""))
        ):
            cat = "bwd"
        else:
            cat = "fwd"
        tr_kernels.append({"epoch": event["ts"] / 1e6, "dur_ms": event["dur"] / 1e3, "cat": cat, "name": name})

    cats2 = ["fwd", "bwd", "ar"]
    cov2 = gpu_coverage(tr_step, tr_kernels, cats2, bin_ms)
    tr_nccl_rows = load_collectives(tr_trace)
    world = 2
    for row in tr_nccl_rows:
        if any("all_to_all" in name for name in row["collectives"]):
            # all_to_all 的 In msg nelems 是全 rank 的 payload，本 rank 上线字节只有 (N-1)/N
            row["bytes"] = int(row["bytes"] * (world - 1) / world)
    nccl2 = nccl_rate(tr_step, tr_nccl_rows, bin_ms)
    kernels2 = [k for k in tr_kernels if tr_step["start"] <= k["epoch"] < tr_step["end"]]
    gpu_busy2 = base.union_ms(sorted((k["epoch"], k["epoch"] + k["dur_ms"] / 1e3) for k in kernels2)) * 1e3
    nccl_busy2 = base.union_ms(
        [
            (row["epoch"], row["epoch"] + row["dur_ms"] / 1e3)
            for row in tr_nccl_rows
            if row["bytes"] > 0 and tr_step["start"] <= row["epoch"] < tr_step["end"]
        ]
    ) * 1e3
    tr_nccl_bytes = sum(
        row["bytes"] for row in tr_nccl_rows if row["bytes"] > 0 and tr_step["start"] <= row["epoch"] < tr_step["end"]
    )

    # ------------------------------------------------------------- figure
    total1 = n_bins_of(step, bin_ms)
    total2 = n_bins_of(tr_step, bin_ms)
    x1 = (np.arange(total1) + 0.5) * bin_ms
    x2 = (np.arange(total2) + 0.5) * bin_ms
    step1_ms = step["dur"] * 1e3
    step2_ms = tr_step["dur"] * 1e3

    fig = plt.figure(figsize=(17, 13.5))
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[2.15, 1.2, 0.85, 1.75, 1.2],
        hspace=1.02,
        left=0.075, right=0.985, top=0.876, bottom=0.098,
    )
    ax1_gpu = fig.add_subplot(gs[0])
    ax1_nic = fig.add_subplot(gs[1], sharex=ax1_gpu)
    ax1_lead = fig.add_subplot(gs[2])
    ax2_gpu = fig.add_subplot(gs[3])
    ax2_nic = fig.add_subplot(gs[4], sharex=ax2_gpu)

    # ---- arm 1: GPU
    ax1_gpu.bar(x1, np.clip(1.0 - cov1.sum(axis=1), 0, 1), width=bin_ms, color=C["idle"],
                edgecolor=C["idle_e"], linewidth=0.2, hatch="////", zorder=1)
    bottom = np.zeros(total1)
    for cat in cats1:
        ax1_gpu.bar(x1, cov1[:, cats1.index(cat)], bottom=bottom, width=bin_ms,
                    color=C[cat], linewidth=0, zorder=3)
        bottom += cov1[:, cats1.index(cat)]
    ax1_gpu.set_ylim(0, 1.0)
    ax1_gpu.set_yticks([0, 0.5, 1.0])
    ax1_gpu.set_yticklabels(["0", "50%", "100%"], fontsize=9)
    ax1_gpu.set_ylabel("GPU 占用", fontsize=10.5)
    plt.setp(ax1_gpu.get_xticklabels(), visible=False)
    ax1_gpu.set_xlim(0, total1 * bin_ms)

    def phase_span(name: str) -> tuple[float, float] | None:
        span = step["phases"].get(name)
        if not span or "start" not in span or "end" not in span:
            return None
        return (span["start"] - step["start"]) * 1e3, (span["end"] - step["start"]) * 1e3

    # 阶段条（Quanta 风格）：主条 = 步内主机阶段，子条 = 预取窗口
    ruler_y, ruler_h = 1.10, 0.115
    ax1_gpu.add_patch(Rectangle((0, ruler_y), step1_ms, ruler_h, facecolor="#F4F4F4",
                                edgecolor="#CCCCCC", lw=0.6, clip_on=False, zorder=1))
    narrow: list[tuple[float, float, str]] = []
    for name in base.PHASE_ORDER:
        if name == "prepare":
            continue
        span = phase_span(name)
        if span is None:
            continue
        left, right = span
        width = right - left
        ax1_gpu.add_patch(Rectangle((left, ruler_y), width, ruler_h, color=base.PHASE_COLORS[name],
                                    alpha=0.95, lw=0, clip_on=False, zorder=2))
        label = f"{PHASE_CN[name]} {width:.1f}"
        if width >= 7.0:
            ax1_gpu.text(left + width / 2, ruler_y + ruler_h / 2, label, ha="center", va="center",
                         fontsize=9, color="white", clip_on=False, zorder=3)
        else:
            narrow.append((left + width / 2, width, label))
    ms_per_pt = step1_ms / (0.91 * 17 * 72)

    def est_width_ms(text: str) -> float:
        points = sum(8.0 if ord(ch) > 0x2000 else 4.4 for ch in text) + 4.0
        return points * ms_per_pt

    row_edges: list[float] = []
    for centre, width, label in sorted(narrow):
        half = max(est_width_ms(label) / 2, width / 2) + 0.3
        row = next((i for i, edge in enumerate(row_edges) if centre - half > edge), None)
        if row is None:
            row = len(row_edges)
            row_edges.append(0.0)
        row_edges[row] = centre + half
        y = 1.238 + 0.052 * row
        ax1_gpu.annotate(
            label, xy=(centre, ruler_y + ruler_h), xycoords=("data", "axes fraction"),
            xytext=(centre, y), textcoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=8, color="#333333", clip_on=False,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7, shrinkA=0, shrinkB=0),
        )
    legend_anchor = 1.28 + 0.055 * len(row_edges)
    ax1_gpu.text(-0.4, ruler_y + ruler_h / 2, "主机阶段", ha="right", va="center",
                 fontsize=8.5, color="#555555", clip_on=False)
    ax1_gpu.text(-0.4, 1.04, "预取", ha="right", va="center",
                 fontsize=8.5, color=base.PHASE_COLORS["prepare"], clip_on=False)

    ax1_gpu.text(
        0.004, 0.985,
        f"① RecStore-RDMA（BagPipe）· step {step['step']} 一步内 GPU 占用：步时 {step1_ms:.1f} ms，"
        f"GPU 有核 {gpu_busy1:.1f} ms（{gpu_busy1 / step1_ms * 100:.0f}%），空 {step1_ms - gpu_busy1:.1f} ms"
        f"（10 步窗口并集口径：{window_busy:.0f} ms / {window_ms:.0f} ms = {window_busy / window_ms * 100:.0f}%）",
        transform=ax1_gpu.transAxes, ha="left", va="top", fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", lw=0.8, alpha=0.92),
    )
    consume = step["phases"].get("consume")
    if consume and "start" in consume and "end" in consume:
        lo = int((consume["start"] - step["start"]) * 1e3 / bin_ms)
        hi = int((consume["end"] - step["start"]) * 1e3 / bin_ms)
        idle_in_consume = float(np.clip(1.0 - cov1.sum(axis=1), 0, 1)[lo:hi].mean()) * (consume["end"] - consume["start"]) * 1e3
        ax1_gpu.annotate(
            f"最大空窗就在这里：consume 段 {(consume['end'] - consume['start']) * 1e3:.1f} ms 里 GPU 空 {idle_in_consume:.1f} ms",
            xy=(((consume["start"] + consume["end"]) / 2 - step["start"]) * 1e3, 0.66),
            xytext=(0.42, 0.78), textcoords="axes fraction",
            fontsize=9.5, color=C["warn"],
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=C["warn"], lw=0.9),
            arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.1),
        )
    shared_handles = [Patch(facecolor=C["idle"], edgecolor=C["idle_e"], hatch="////", label="GPU 空窗")]
    shared_handles += [Patch(facecolor=C[cat], label=CAT_LABEL[cat]) for cat in cats1]
    fig.legend(handles=shared_handles, loc="upper center", bbox_to_anchor=(0.5, 0.978),
               ncol=8, fontsize=9.5, framealpha=0.95)

    # ---- arm 1: NIC
    ax1_nic.bar(x1, up1, width=bin_ms, color=C["ps_up"], linewidth=0,
                label=f"PS 数据面 RDMA · 上行（请求 + 写回）{up_bytes1 / 1e6:.1f} MB/步")
    ax1_nic.bar(x1, down1, bottom=up1, width=bin_ms, color=C["ps_down"], linewidth=0,
                label=f"PS 数据面 RDMA · 下行（查表结果）{down_bytes1 / 1e6:.1f} MB/步")
    ax1_nic.bar(x1, nccl1, bottom=up1 + down1, width=bin_ms, color=C["ar"], linewidth=0,
                label=f"两卡 NCCL（allreduce / all_gather）{nccl_bytes1 / 1e6:.1f} MB/步")
    ax1_nic.axhline(base.LINK_MBPS / 1000.0, color="#95a5a6", lw=0.9, ls=":")
    ax1_nic.text(total1 * bin_ms * 0.995, base.LINK_MBPS / 1000.0 * 1.04,
                 "100 Gb/s 线速参考 12.5 GB/s", ha="right", va="bottom", fontsize=8.5, color="#7f8c8d")
    get_times = [(r["epoch"] - step["start"]) * 1e3 for r in ps_get if step["start"] <= r["epoch"] < step["end"]]
    ax1_nic.set_ylim(0, max(base.LINK_MBPS / 1000.0 * 1.3, float((up1 + down1 + nccl1).max()) * 1.3))
    ax1_nic.vlines(get_times, 0, ax1_nic.get_ylim()[1] * 0.97, color=C["get"], lw=0.6, alpha=0.35, ls="--",
                   label=f"BagPipe 预取 RDMA GET（这一步 {len(get_times)} 条，都是给 step {step['step'] + 4} 入队）")
    total_link_ms = nic_busy1 + nccl_busy1
    ax1_nic.set_ylabel("GB/s", fontsize=10)
    ax1_nic.set_title(
        f"网卡 · 同一时间轴上的两条跨机流量：PS 数据面有流量 {nic_busy1:.1f} ms、NCCL 有流量 {nccl_busy1:.1f} ms"
        f"（合计 {total_link_ms / step1_ms * 100:.0f}% 的时间，其余空转——卡住的是主机串行，不是带宽）",
        fontsize=11.5,
    )
    ax1_nic.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax1_nic.set_xlabel("一步内的时间（ms）", fontsize=10.5)

    # ---- arm 1: BagPipe lead time inset
    lead_rows = []
    by_id = {item["step"]: item for item in steps_all}
    window_ids = {item["step"] for item in profiled}
    for item in profiled:
        target = by_id.get(item["step"] + 4)
        prep = item["phases"].get("prepare")
        if target is None or item["step"] + 4 not in window_ids or not prep or "end" not in prep:
            continue
        consume_span = target["phases"].get("consume")
        if not consume_span or "start" not in consume_span:
            continue
        lead_rows.append((item["step"], (consume_span["start"] - prep["end"]) * 1e3))
    if lead_rows:
        values = [v for _, v in lead_rows]
        labels = [f"step {step_id} → {step_id + 4}" for step_id, _ in lead_rows]
        ax1_lead.barh(range(len(values)), values, color=C["get"], alpha=0.75, height=0.6)
        ax1_lead.set_yticks(range(len(values)))
        ax1_lead.set_yticklabels(labels, fontsize=8.5)
        ax1_lead.invert_yaxis()
        ax1_lead.set_xlim(0, max(values) * 1.18)
        ax1_lead.set_xlabel("提前量（ms）", fontsize=9)
        ax1_lead.set_title(
            f"①′ BagPipe 预取重叠的证据：prepare 段入队 → 目标步 consume 取用，平均提前 {st.mean(values):.0f} ms"
            f"（prepare 结束到目标步 consume 开始 ≈ {st.mean(values) / (st.mean(item['dur'] for item in profiled) * 1e3):.1f} 个步时）；"
            "响应 0.3 ms 就回写，consume 的 wait 不是主链瓶颈",
            fontsize=11.5,
        )

    # ---- arm 2: GPU
    ax2_gpu.bar(x2, np.clip(1.0 - cov2.sum(axis=1), 0, 1), width=bin_ms, color=C["idle"],
                edgecolor=C["idle_e"], linewidth=0.2, hatch="////", zorder=1)
    bottom = np.zeros(total2)
    for cat in cats2:
        ax2_gpu.bar(x2, cov2[:, cats2.index(cat)], bottom=bottom, width=bin_ms, color=C[cat], linewidth=0, zorder=3)
        bottom += cov2[:, cats2.index(cat)]
    ax2_gpu.set_ylim(0, 1.0)
    ax2_gpu.set_yticks([0, 0.5, 1.0])
    ax2_gpu.set_yticklabels(["0", "50%", "100%"], fontsize=9)
    ax2_gpu.set_ylabel("GPU 占用", fontsize=10.5)
    ax2_gpu.set_xlim(0, total2 * bin_ms)
    plt.setp(ax2_gpu.get_xticklabels(), visible=False)
    ax2_gpu.text(
        0.004, 0.985,
        f"② 对照 lane：TorchRec-HBM · 一步内 GPU 占用：步时 {step2_ms:.1f} ms，"
        f"GPU 有核 {gpu_busy2:.1f} ms（{gpu_busy2 / step2_ms * 100:.0f}%），空 {step2_ms - gpu_busy2:.1f} ms",
        transform=ax2_gpu.transAxes, ha="left", va="top", fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", lw=0.8, alpha=0.92),
    )

    # ---- arm 2: NIC
    ax2_nic.bar(x2, nccl2, width=bin_ms, color=C["ar"], linewidth=0,
                label=f"两卡 NCCL（all_to_all 主 + allreduce）{tr_nccl_bytes / 1e6 / (step2_ms / 1e3):.1f} MB/步")
    ax2_nic.axhline(base.LINK_MBPS / 1000.0, color="#95a5a6", lw=0.9, ls=":")
    ax2_nic.text(total2 * bin_ms * 0.995, base.LINK_MBPS / 1000.0 * 1.04,
                 "100 Gb/s 线速参考 12.5 GB/s", ha="right", va="bottom", fontsize=8.5, color="#7f8c8d")
    ax2_nic.set_ylabel("GB/s", fontsize=10)
    ax2_nic.set_ylim(0, max(base.LINK_MBPS / 1000.0 * 1.3, float(nccl2.max()) * 1.3))
    ax2_nic.set_title(
        f"网卡 · TorchRec 这档没有 PS，跨机流量只有集合通信：有流量 {nccl_busy2:.1f} ms / {step2_ms:.1f} ms"
        f"（{nccl_busy2 / step2_ms * 100:.0f}%），峰值 {float(nccl2.max()):.1f} GB/s 贴着线速",
        fontsize=11.5,
    )
    ax2_nic.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2_nic.set_xlabel("一步内的时间（ms）", fontsize=10.5)

    fig.suptitle(
        "A · 2n1g bs=3072：一步里 GPU 在忙什么、网卡什么时候有流量（RecStore-RDMA/BagPipe vs TorchRec-HBM）",
        fontsize=15, y=0.994,
    )
    fig.text(
        0.075, 0.022,
        "上排 = GPU kernel 占用（堆叠；斜纹 = 空窗）+ 主机阶段条（彩条上方标出各阶段时长）；下排 = 同一时间轴的跨机网卡带宽，"
        "PS 数据面 RDMA 与两卡 NCCL 分开堆叠。x 轴 = 一步内的时间，每一步各自以 step_start 为 0。\n"
        "口径：NCCL 带宽 = 内核字节 / 内核时长（字节取自 record_param_comms 的 In msg nelems）；PS 侧 wire trace 只记了发起时刻、"
        "没有单条时长，画的是 ±1 ms 滑窗的等效速率（上界）。同机环回不跨机，没画。",
        fontsize=9, color="#444444", va="bottom",
    )

    path = out_dir / "recstore_2n1g_stepview.png"
    fig.savefig(path, dpi=140)
    fig.savefig(path.with_suffix(".svg"))
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.svg')}")
    print(
        json.dumps(
            {
                "arm1_step": step["step"],
                "arm1_step_ms": round(step1_ms, 2),
                "arm1_gpu_busy_ms": round(gpu_busy1, 2),
                "arm1_gpu_busy_pct": round(gpu_busy1 / step1_ms * 100, 1),
                "arm1_ps_traffic_ms": round(nic_busy1, 2),
                "arm1_nccl_traffic_ms": round(nccl_busy1, 2),
                "arm1_ps_up_GBps_peak": round(float(up1.max()), 2),
                "arm1_ps_down_GBps_peak": round(float(down1.max()), 2),
                "arm1_nccl_GBps_peak": round(float(nccl1.max()), 2),
                "arm1_get_per_step": len(get_times),
                "arm1_up_MB": round(up_bytes1 / 1e6, 2),
                "arm1_down_MB": round(down_bytes1 / 1e6, 2),
                "arm1_nccl_MB": round(nccl_bytes1 / 1e6, 2),
                "arm1_bagpipe_lead_ms": round(st.mean([v for _, v in lead_rows]), 1) if lead_rows else None,
                "arm2_step_ms": round(step2_ms, 2),
                "arm2_gpu_busy_ms": round(gpu_busy2, 2),
                "arm2_gpu_busy_pct": round(gpu_busy2 / step2_ms * 100, 1),
                "arm2_nccl_traffic_ms": round(nccl_busy2, 2),
                "arm2_nccl_MB_per_step": round(tr_nccl_bytes / 1e6, 2),
                "arm2_nccl_GBps_peak": round(float(nccl2.max()), 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
