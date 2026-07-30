from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.pytorch.recstore.analysis.compare import (
    build_exposed_gap_rows,
    write_compare_csv,
)


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _measured_rows(path: Path) -> list[dict[str, str]]:
    rows = _load_rows(path)
    measured = [row for row in rows if row.get("warmup_excluded", "") != "1"]
    return measured or rows


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


def write_markdown(path: Path, rows: list[dict[str, str | float]], *, recstore_csv: Path, torchrec_csv: Path) -> None:
    rec_rows = _measured_rows(recstore_csv) if recstore_csv.exists() else []
    by_metric = {str(row["metric"]): row for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: abs(float(row.get("delta_exposed_ms", 0.0))),
        reverse=True,
    )
    raw_ordered = sorted(
        rows,
        key=lambda row: abs(float(row.get("delta_raw_ms", 0.0))),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# RecStore vs TorchRec HBM Exposed Gap Analysis")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- RecStore CSV: `{recstore_csv}`")
    lines.append(f"- TorchRec HBM CSV: `{torchrec_csv}`")
    lines.append("")
    lines.append("## Paper-Style Exposed Time")
    lines.append("")
    lines.append(
        "`raw_ms` 是阶段实测耗时；`exposed_ms` 是扣除可重叠窗口后仍暴露在 step 上的耗时。"
        "对 RecStore prefetch 网络等待，当前脚手架使用 "
        "`max(0, prefetch_network_wait_ms - dense_compute_ms)`。"
        "TorchRec HBM lane 没有显式 prefetch network wait，因此该项暴露时间记为 0。"
    )
    lines.append("")
    lines.append("| metric | RecStore raw ms | RecStore exposed ms | TorchRec raw ms | TorchRec exposed ms | delta raw ms | delta exposed ms | bottleneck |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| {metric} | {rec_raw} | {rec_exp} | {tr_raw} | {tr_exp} | {d_raw} | {d_exp} | {bottleneck} |".format(
                metric=row["metric"],
                rec_raw=_fmt(row["recstore_raw_ms"]),
                rec_exp=_fmt(row["recstore_exposed_ms"]),
                tr_raw=_fmt(row["torchrec_raw_ms"]),
                tr_exp=_fmt(row["torchrec_exposed_ms"]),
                d_raw=_fmt(row["delta_raw_ms"]),
                d_exp=_fmt(row["delta_exposed_ms"]),
                bottleneck=row["bottleneck"],
            )
        )
    lines.append("")

    step = by_metric.get("step_total", {})
    lookup = by_metric.get("embedding_lookup", {})
    prefetch = by_metric.get("prefetch_network", {})
    cache_query = by_metric.get("gpu_cache_query", {})
    cache_prefill = by_metric.get("gpu_cache_prefill", {})
    update = by_metric.get("sparse_update", {})
    invalidate = by_metric.get("gpu_cache_invalidate", {})

    lines.append("## Diagnosis")
    lines.append("")
    lines.append(
        f"- End-to-end gap: RecStore step delta is `{_fmt(step.get('delta_raw_ms', 0.0))} ms`."
    )
    lines.append(
        f"- Lookup raw gap: RecStore lookup delta is `{_fmt(lookup.get('delta_raw_ms', 0.0))} ms`; "
        f"prefetch exposed network delta is `{_fmt(prefetch.get('delta_exposed_ms', 0.0))} ms`."
    )
    lines.append(
        f"- GPU cache overhead: query `{_fmt(cache_query.get('recstore_raw_ms', 0.0))} ms`, "
        f"prefill `{_fmt(cache_prefill.get('recstore_raw_ms', 0.0))} ms`, "
        f"invalidate `{_fmt(invalidate.get('recstore_raw_ms', 0.0))} ms`."
    )
    lines.append(
        f"- GPU cache signals: hit_rate=`{_fmt(_mean(rec_rows, 'lookup_gpu_cache_hit_rate'))}`, "
        f"prefill_success=`{_fmt(_mean(rec_rows, 'planned_gpu_cache_prefill_successes'))}`, "
        f"fallback=`{_fmt(_mean(rec_rows, 'planned_gpu_cache_prefill_fallbacks'))}`, "
        f"no_cuda=`{_fmt(_mean(rec_rows, 'planned_gpu_cache_prefill_no_cuda'))}`, "
        f"no_api=`{_fmt(_mean(rec_rows, 'planned_gpu_cache_prefill_no_api'))}`, "
        f"mismatch=`{_fmt(_mean(rec_rows, 'planned_gpu_cache_prefill_result_size_mismatches'))}`."
    )
    lines.append(
        f"- Sparse update gap: `{_fmt(update.get('delta_raw_ms', 0.0))} ms`; "
        "if this dominates the table, GPU cache alone cannot close the TorchRec HBM gap."
    )
    lines.append("")
    lines.append("Top exposed gaps:")
    for row in ordered[:5]:
        lines.append(
            f"- `{row['metric']}`: delta_exposed_ms=`{_fmt(row.get('delta_exposed_ms', 0.0))}`, "
            f"raw delta=`{_fmt(row.get('delta_raw_ms', 0.0))}`"
        )
    lines.append("")
    lines.append("Top raw gaps:")
    for row in raw_ordered[:5]:
        lines.append(
            f"- `{row['metric']}`: delta_raw_ms=`{_fmt(row.get('delta_raw_ms', 0.0))}`, "
            f"exposed delta=`{_fmt(row.get('delta_exposed_ms', 0.0))}`"
        )
    lines.append("")
    lines.append("## Profiler Follow-Up")
    lines.append("")
    lines.append(
        "CSV 级分析用于确定 gap 属于 lookup、cache query/prefill/invalidate、sparse update 还是 dense。"
        "若要继续定位具体 kernel/API，建议下一步使用 `--torchrec-profiler` 生成 PyTorch profiler trace，"
        "或用 `nsys profile --trace=cuda,nvtx,osrt` 包住相同命令查看 CPU/GPU 时间线。"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RecStore GPU-cache/pretech lanes with TorchRec HBM using raw and exposed time.",
    )
    parser.add_argument("--recstore-csv", type=Path, required=True)
    parser.add_argument("--torchrec-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = build_exposed_gap_rows(args.recstore_csv, args.torchrec_csv)
    write_compare_csv(args.output_csv, rows)
    write_markdown(
        args.output_md,
        rows,
        recstore_csv=args.recstore_csv,
        torchrec_csv=args.torchrec_csv,
    )
    print(f"[rs_demo] exposed gap csv: {args.output_csv}")
    print(f"[rs_demo] exposed gap report: {args.output_md}")
    print(f"[rs_demo] rows: {len(_load_rows(args.output_csv))}")


if __name__ == "__main__":
    main()
