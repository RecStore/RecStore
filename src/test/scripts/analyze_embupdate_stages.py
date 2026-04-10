#!/usr/bin/env python3
"""
Analyze embupdate stage metrics from local report events.

Supports:
1) glog text logs containing lines like:
   REPORT_LOCAL_EVENT {"table_name":"embupdate_stages", ...}
2) pure JSONL where each line is the event JSON

Usage:
  python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/log
  python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/a --input /path/to/b --top 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List, Tuple


GLOG_EVENT_PREFIX = "REPORT_LOCAL_EVENT "
STAGE_TABLE = "embupdate_stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embupdate stage metrics.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input log/JSONL file path. Can be used multiple times.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top slow traces to print (default: 10).",
    )
    parser.add_argument(
        "--trace-prefix",
        default="",
        help="Only include traces whose unique_id starts with this prefix.",
    )
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help="Group traces by unique_id prefix before '|' and print group stats.",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Export per-trace merged metrics to CSV path.",
    )
    return parser.parse_args()


def read_events(paths: List[str]) -> List[dict]:
    events: List[dict] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                event = parse_event_line(line)
                if event is not None:
                    events.append(event)
    return events


def parse_event_line(line: str) -> dict | None:
    if line.startswith("{"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    idx = line.find(GLOG_EVENT_PREFIX)
    if idx >= 0:
        payload = line[idx + len(GLOG_EVENT_PREFIX) :].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    w = rank - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def summarize_metric(values: List[float]) -> str:
    if not values:
        return "count=0"
    return (
        f"count={len(values)} "
        f"mean={statistics.fmean(values):.2f}us "
        f"p50={percentile(values, 50):.2f}us "
        f"p95={percentile(values, 95):.2f}us "
        f"p99={percentile(values, 99):.2f}us "
        f"max={max(values):.2f}us"
    )


def non_negative(v: float) -> float:
    return v if v >= 0 else 0.0


def derive_chain_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Derive node-level latency decomposition for one merged request trace.
    """
    op_total = metrics.get("op_total_us")
    op_validate = metrics.get("op_validate_us")
    client_total = metrics.get("client_total_us")
    client_serialize = metrics.get("client_serialize_us")
    client_rpc = metrics.get("client_rpc_us")
    server_total = metrics.get("server_total_us")
    server_backend = metrics.get("server_backend_update_us")

    out: Dict[str, float] = {}
    if op_total is not None:
        out["compute_op_total_us"] = op_total
    if op_validate is not None:
        out["compute_op_validate_us"] = op_validate
    if op_total is not None and op_validate is not None:
        out["compute_op_non_validate_us"] = non_negative(op_total - op_validate)

    if client_serialize is not None:
        out["client_serialize_us"] = client_serialize
    if client_total is not None and client_serialize is not None and client_rpc is not None:
        out["client_framework_overhead_us"] = non_negative(
            client_total - client_serialize - client_rpc
        )

    if client_rpc is not None and server_total is not None:
        out["network_transport_us"] = non_negative(client_rpc - server_total)
    if server_total is not None and server_backend is not None:
        out["server_framework_overhead_us"] = non_negative(server_total - server_backend)
    if server_backend is not None:
        out["storage_backend_update_us"] = server_backend

    if op_total is not None and client_total is not None:
        out["op_wrapper_overhead_us"] = non_negative(op_total - client_total)
    return out


def build_trace_map(events: List[dict], trace_prefix: str) -> Dict[str, Dict[str, float]]:
    by_trace: Dict[str, Dict[str, float]] = {}
    for e in events:
        if e.get("table_name") != STAGE_TABLE:
            continue
        unique_id = str(e.get("unique_id", ""))
        if trace_prefix and not unique_id.startswith(trace_prefix):
            continue
        metric = str(e.get("metric_name", ""))
        try:
            value = float(e.get("metric_value", 0.0))
        except (TypeError, ValueError):
            continue
        trace = by_trace.setdefault(unique_id, {})
        trace[metric] = value
    return by_trace


def trace_token(unique_id: str) -> str:
    if "|" in unique_id:
        return unique_id.split("|", 1)[1]
    return unique_id


def _sort_trace_tokens(tokens: List[str]) -> List[str]:
    def key(token: str) -> tuple[int, object]:
        if token.isdigit():
            return (0, int(token))
        return (1, token)

    return sorted(tokens, key=key)


def build_merged_request_map(by_trace: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for unique_id, metrics in by_trace.items():
        token = trace_token(unique_id)
        dst = merged.setdefault(token, {})
        for k, v in metrics.items():
            dst[k] = v

    client_only_tokens = [
        token
        for token, metrics in merged.items()
        if (metrics.get("op_total_us") is not None or metrics.get("client_rpc_us") is not None)
        and metrics.get("server_total_us") is None
    ]
    server_only_tokens = [
        token
        for token, metrics in merged.items()
        if metrics.get("server_total_us") is not None
        and metrics.get("op_total_us") is None
        and metrics.get("client_rpc_us") is None
    ]

    for client_token, server_token in zip(
        _sort_trace_tokens(client_only_tokens),
        _sort_trace_tokens(server_only_tokens),
    ):
        client_metrics = merged.get(client_token)
        server_metrics = merged.get(server_token)
        if client_metrics is None or server_metrics is None:
            continue
        for k, v in server_metrics.items():
            client_metrics.setdefault(k, v)
        del merged[server_token]

    return merged


def print_overall(by_trace: Dict[str, Dict[str, float]]) -> None:
    all_metrics: Dict[str, List[float]] = {}
    for metrics in by_trace.values():
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    print(f"Traces: {len(by_trace)}")
    print("")
    print("Stage Metrics Summary:")
    for metric in sorted(all_metrics.keys()):
        if metric.endswith("_us"):
            print(f"  - {metric}: {summarize_metric(all_metrics[metric])}")
    print("")


def print_breakdown_ratios(by_trace: Dict[str, Dict[str, float]]) -> None:
    network_like: List[float] = []
    backend_like: List[float] = []
    serialize_like: List[float] = []

    for metrics in by_trace.values():
        client_rpc = metrics.get("client_rpc_us")
        server_total = metrics.get("server_total_us")
        backend = metrics.get("server_backend_update_us")
        serialize = metrics.get("client_serialize_us")
        if serialize is not None:
            serialize_like.append(serialize)
        if backend is not None:
            backend_like.append(backend)
        if client_rpc is not None and server_total is not None:
            network_like.append(max(0.0, client_rpc - server_total))

    print("Approx Breakdown:")
    print(f"  - serialize_us: {summarize_metric(serialize_like)}")
    print(f"  - backend_update_us: {summarize_metric(backend_like)}")
    print(f"  - rpc_minus_server_us (network/framework approx): {summarize_metric(network_like)}")
    print("")


def print_top_slow(by_trace: Dict[str, Dict[str, float]], top_n: int) -> None:
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for trace, metrics in by_trace.items():
        score = metrics.get("op_total_us")
        if score is None:
            score = metrics.get("client_total_us")
        if score is None:
            score = metrics.get("server_total_us")
        if score is None:
            continue
        ranked.append((trace, score, metrics))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {min(top_n, len(ranked))} Slow Traces:")
    for idx, (trace, score, metrics) in enumerate(ranked[:top_n], 1):
        serialize = metrics.get("client_serialize_us", float("nan"))
        rpc = metrics.get("client_rpc_us", float("nan"))
        server = metrics.get("server_total_us", float("nan"))
        backend = metrics.get("server_backend_update_us", float("nan"))
        op_total = metrics.get("op_total_us", float("nan"))
        print(
            f"  {idx:2d}. {trace} total={score:.2f}us "
            f"(op={op_total:.2f}, ser={serialize:.2f}, rpc={rpc:.2f}, server={server:.2f}, backend={backend:.2f})"
        )


def trace_group_name(unique_id: str) -> str:
    if "|" in unique_id:
        return unique_id.split("|", 1)[0]
    return unique_id


def print_group_stats(by_trace: Dict[str, Dict[str, float]]) -> None:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for trace, metrics in by_trace.items():
        g = trace_group_name(trace)
        g_metrics = grouped.setdefault(g, {})
        for k, v in metrics.items():
            if k.endswith("_us"):
                g_metrics.setdefault(k, []).append(v)

    if not grouped:
        return

    print("")
    print("Grouped Summary (by trace prefix before '|'):")
    for group in sorted(grouped.keys()):
        m = grouped[group]
        total = m.get("op_total_us") or m.get("client_total_us") or m.get("server_total_us") or []
        print(f"  - {group}: traces={len(total) if total else 0}")
        for key in ("op_total_us", "client_total_us", "server_total_us", "client_rpc_us", "server_backend_update_us"):
            vals = m.get(key, [])
            if vals:
                print(f"      {key}: {summarize_metric(vals)}")


def print_request_view(merged: Dict[str, Dict[str, float]], top_n: int) -> None:
    if not merged:
        return
    print("")
    print(f"Merged Request View (by trace token): {len(merged)} requests")
    print_breakdown_ratios(merged)
    print(f"Top {min(top_n, len(merged))} Slow Requests:")
    ranked: List[Tuple[str, float, Dict[str, float]]] = []
    for token, metrics in merged.items():
        score = metrics.get("op_total_us")
        if score is None:
            score = metrics.get("client_total_us")
        if score is None:
            score = metrics.get("server_total_us")
        if score is None:
            continue
        ranked.append((token, score, metrics))
    ranked.sort(key=lambda x: x[1], reverse=True)
    for idx, (token, score, m) in enumerate(ranked[:top_n], 1):
        net = float("nan")
        if m.get("client_rpc_us") is not None and m.get("server_total_us") is not None:
            net = max(0.0, m["client_rpc_us"] - m["server_total_us"])
        print(
            f"  {idx:2d}. trace={token} total={score:.2f}us "
            f"(op={m.get('op_total_us', float('nan')):.2f}, "
            f"ser={m.get('client_serialize_us', float('nan')):.2f}, "
            f"rpc={m.get('client_rpc_us', float('nan')):.2f}, "
            f"server={m.get('server_total_us', float('nan')):.2f}, "
            f"backend={m.get('server_backend_update_us', float('nan')):.2f}, "
            f"net≈{net:.2f})"
        )


def print_chain_breakdown(merged: Dict[str, Dict[str, float]]) -> None:
    if not merged:
        return
    buckets: Dict[str, List[float]] = {}
    for metrics in merged.values():
        derived = derive_chain_metrics(metrics)
        for k, v in derived.items():
            buckets.setdefault(k, []).append(v)

    print("")
    print("链路节点耗时拆分（按请求聚合）:")
    ordered = [
        "compute_op_total_us",
        "compute_op_validate_us",
        "compute_op_non_validate_us",
        "op_wrapper_overhead_us",
        "client_serialize_us",
        "client_framework_overhead_us",
        "network_transport_us",
        "server_framework_overhead_us",
        "storage_backend_update_us",
    ]
    for key in ordered:
        vals = buckets.get(key, [])
        if vals:
            print(f"  - {key}: {summarize_metric(vals)}")


def export_csv(by_trace: Dict[str, Dict[str, float]], output_path: str) -> None:
    columns = [
        "unique_id",
        "group",
        "op_total_us",
        "op_validate_us",
        "client_total_us",
        "client_serialize_us",
        "client_rpc_us",
        "server_total_us",
        "server_backend_update_us",
        "client_request_size",
        "server_request_size",
        "network_framework_us_approx",
        "compute_op_total_us",
        "compute_op_validate_us",
        "compute_op_non_validate_us",
        "op_wrapper_overhead_us",
        "client_framework_overhead_us",
        "network_transport_us",
        "server_framework_overhead_us",
        "storage_backend_update_us",
    ]

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for trace in sorted(by_trace.keys()):
            metrics = by_trace[trace]
            client_rpc = metrics.get("client_rpc_us")
            server_total = metrics.get("server_total_us")
            network_approx = ""
            if client_rpc is not None and server_total is not None:
                network_approx = max(0.0, client_rpc - server_total)
            row = {
                "unique_id": trace,
                "group": trace_group_name(trace),
                "op_total_us": metrics.get("op_total_us", ""),
                "op_validate_us": metrics.get("op_validate_us", ""),
                "client_total_us": metrics.get("client_total_us", ""),
                "client_serialize_us": metrics.get("client_serialize_us", ""),
                "client_rpc_us": metrics.get("client_rpc_us", ""),
                "server_total_us": metrics.get("server_total_us", ""),
                "server_backend_update_us": metrics.get("server_backend_update_us", ""),
                "client_request_size": metrics.get("client_request_size", ""),
                "server_request_size": metrics.get("server_request_size", ""),
                "network_framework_us_approx": network_approx,
                "compute_op_total_us": "",
                "compute_op_validate_us": "",
                "compute_op_non_validate_us": "",
                "op_wrapper_overhead_us": "",
                "client_framework_overhead_us": "",
                "network_transport_us": "",
                "server_framework_overhead_us": "",
                "storage_backend_update_us": "",
            }
            derived = derive_chain_metrics(metrics)
            for dk in (
                "compute_op_total_us",
                "compute_op_validate_us",
                "compute_op_non_validate_us",
                "op_wrapper_overhead_us",
                "client_framework_overhead_us",
                "network_transport_us",
                "server_framework_overhead_us",
                "storage_backend_update_us",
            ):
                if dk in derived:
                    row[dk] = derived[dk]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    events = read_events(args.input)
    by_trace = build_trace_map(events, args.trace_prefix)
    if not by_trace:
        print("No embupdate stage events found. Check input paths and logging mode.")
        return
    print_overall(by_trace)
    print_breakdown_ratios(by_trace)
    print_top_slow(by_trace, args.top)
    merged = build_merged_request_map(by_trace)
    print_request_view(merged, args.top)
    print_chain_breakdown(merged)
    if args.group_by_prefix:
        print_group_stats(by_trace)
    if args.export_csv:
        export_csv(by_trace, args.export_csv)
        print("")
        print(f"CSV exported: {args.export_csv}")


if __name__ == "__main__":
    main()
