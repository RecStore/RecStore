#!/usr/bin/env python3
"""Compare tiered cache policies at fixed value DRAM capacity using native YCSB."""

import argparse
import csv
import itertools
import json
import math
import random
import socket
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import run_kvengine_compare as bench


def write_summary(out, rows, options):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("variant", "ssd_ratio", "distribution", "policy", "budget"))
        groups[key].append(row)
    lines = [subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.ROOT, text=True).strip(),
             socket.gethostname(), "", "## Workload 说明", "",
             f"storage-only；当前工作树构建，源码改动见 source.patch。参数：`{json.dumps(vars(options), default=str, ensure_ascii=False)}`。"
             " SSD 比例指关闭缓存时的目标；各预算使用相同 value DRAM 容量，索引和元数据另计。"
             " CLOCK 共享 DRAM 高水位，不预留 LRU 缓存预算。"
             " IOURING/O_DIRECT、queue_depth=512、CONCURRENT_SLAB_MEMORY_POOL、SSD_SLAB。"
             " Zipfian alpha=0.9，开启 scramble；吞吐取成功运行的中位数。", "",
             "| 读取接口 | workload | 值大小 | 批大小 |", "|---|---|---:|---:|",
             f"| {options.read_mode} | c（纯读） | {options.value_size} B | {options.batch_keys} |", "",
             "## Run 吞吐（M keys/s，中位数）", "",
             "| 版本 | SSD 目标 | 分布 | 策略 | 预算 | 完成/尝试 | 中位数 | 最小～最大 | SSD miss 比例 |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    loads = []
    for key, group in sorted(groups.items()):
        ok = [r for r in group if r["exit_code"] == 0 and r["verified"]]
        budget = {"clock": "共享高水位", "adaptive": "按溢出量选择"}.get(key[3], f"{key[4]:.0%}")
        prefix = f"| {key[0]} | {key[1]:.2%} | {key[2]} | {key[3]} | {budget} |"
        if not ok:
            lines.append(prefix + f" 0/{len(group)} | 失败 | — | — |")
            continue
        rates = [float(r["run_throughput_ops_sec"]) / 1e6 for r in ok]
        misses = sum(r["measured_misses"] for r in ok)
        ops = sum(int(r["run_operations"]) for r in ok)
        miss_label = f"{misses / ops:.2%}" if key[3] != "none" and any(r["cache_enabled"] for r in ok) else "—"
        lines.append(prefix + f" {len(ok)}/{len(group)} | {statistics.median(rates):.3f} | {min(rates):.3f}～{max(rates):.3f} | {miss_label} |")
        loads.append(prefix + f" {statistics.median(float(r['load_throughput_ops_sec']) for r in ok) / 1e6:.3f} |")
    lines += ["", "## Load 吞吐（M keys/s，中位数）", "",
              "| 版本 | SSD 目标 | 分布 | 策略 | 预算 | 装载吞吐 |",
              "|---|---:|---|---|---:|---:|", *loads]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=bench.ROOT / "results" / datetime.now().strftime("benchmark_kvengine_cache_%m%d%H%M"))
    parser.add_argument("--ssd-root", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--ssd-ratios", type=float, nargs="+", default=[.7, .9])
    parser.add_argument("--cache-budget-ratios", type=float, nargs="+", default=[.05, .15, .3, .5, 1.0])
    parser.add_argument("--policies", choices=["none", "lru", "clock", "adaptive"], nargs="+", default=["lru"])
    parser.add_argument("--distributions", choices=["uniform", "zipfian"], nargs="+", default=["zipfian"])
    parser.add_argument("--record-count", type=int, default=1000000)
    parser.add_argument("--value-size", type=int, default=128)
    parser.add_argument("--dram-slot-bytes", type=int, required=True,
                        help="Actual DRAM slot capacity per value, including size-class rounding")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--runtime-seconds", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=int, default=5)
    parser.add_argument("--completion-poll-spins", type=int, default=8192)
    parser.add_argument("--read-mode", choices=["get", "batch_get_flat"], default="get")
    parser.add_argument("--batch-keys", type=int, default=500)
    parser.add_argument("--disable-fit-check", action="store_true")
    options = parser.parse_args()
    if (any(not 0 <= r < 1 for r in options.ssd_ratios)
            or any(not 0 < b <= 1 for b in options.cache_budget_ratios)
            or min(options.record_count, options.value_size, options.threads, options.repeat,
                   options.runtime_seconds, options.batch_keys) <= 0
            or options.warmup_seconds < 0 or options.dram_slot_bytes < options.value_size):
        parser.error("invalid capacity, timing, or workload parameters")
    out = options.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(vars(options), default=str, indent=2) + "\n")
    (out / "source.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--", "src", "tools", "docs"], cwd=bench.ROOT))
    args = bench.parse_args([])
    for name in ("record_count", "value_size", "threads", "runtime_seconds", "read_mode", "batch_keys"):
        setattr(args, name, getattr(options, name))
    args.load_threads = options.threads
    args.ssd_capacity_bytes = max(512 << 20, options.record_count * options.dram_slot_bytes * 2)
    args.extra_arg = ["--zipfian_scramble=true", f"--warmup_seconds={options.warmup_seconds}"]
    bench.DEFAULT_SSD_ROOT = options.ssd_root.resolve() / out.name
    bench.DEFAULT_DRAM_ROOT = out / "runtime_dram"
    optimized_binary = bench.BENCHMARK_BIN
    cases = []
    for repeat, ratio, distribution, policy in itertools.product(
            range(options.repeat), options.ssd_ratios, options.distributions, options.policies):
        for budget in options.cache_budget_ratios if policy == "lru" else [0.05]:
            cases.append((repeat, ratio, distribution, policy, budget, "optimized"))
        if options.baseline_binary:
            cases.append((repeat, ratio, distribution, policy, .05, "baseline"))
    random.Random(343).shuffle(cases)
    rows = []
    for number, (repeat, ratio, distribution, policy, budget, variant) in enumerate(cases, 1):
        case = f"{variant}_ssd{ratio:g}_{distribution}_{policy}_budget{budget:g}_r{repeat}"
        print(f"START {number}/{len(cases)} {case}", flush=True)
        bench.BENCHMARK_BIN = options.baseline_binary.resolve() if variant == "baseline" else optimized_binary
        args.dram_capacity_bytes = math.ceil(options.record_count * options.dram_slot_bytes * (1 - ratio) / .85)
        props = {"tiered_cache_policy": policy, "tiered_cache_budget_ratio": str(budget)}
        if variant == "optimized":
            props.update(tiered_cache_expected_dram_bytes=str(0 if options.disable_fit_check else options.record_count * options.dram_slot_bytes),
                         ssd_completion_poll_spins=str(options.completion_poll_spins))
        row = bench.run_one(engine_name="dram_pet_tiered", workload="c", repeat=repeat,
                            distribution=distribution, args=args, output_dir=out / case, engine_props=props)
        raw = Path(row["raw_log_path"]).read_text()
        snapshots = {line.split()[0]: bench.parse_result_line(line) for line in raw.splitlines()
                     if line.startswith(("YCSB_BEFORE_RUN ", "YCSB_AFTER_RUN "))}
        before, after = snapshots.get("YCSB_BEFORE_RUN", {}), snapshots.get("YCSB_AFTER_RUN", {})
        row.update(after)
        row.update(variant=variant, ssd_ratio=ratio, policy=policy, budget=budget,
                   configured_dram_bytes=args.dram_capacity_bytes,
                   verified="YCSB_VERIFY samples=1024 ok=1" in raw,
                   cache_enabled=after.get("tiered_cache_policy") in {"lru", "clock"})
        prefix = "tiered_clock" if after.get("tiered_cache_policy") == "clock" else "tiered_cache"
        for label, field in {"hits": prefix + "_hits", "misses": prefix + "_misses",
                             "pages": "layout_page_reads", "bytes": "layout_bytes_read"}.items():
            row["measured_" + label] = int(after.get(field, 0)) - int(before.get(field, 0))
        rows.append(row)
        with (out / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
        write_summary(out, rows, options)
        print(f"DONE {case} exit={row['exit_code']} ops/s={row['run_throughput_ops_sec']}", flush=True)
        if row["exit_code"] or not row["verified"]:
            raise RuntimeError(f"Benchmark failed: {row['log_path']}")


if __name__ == "__main__":
    main()
