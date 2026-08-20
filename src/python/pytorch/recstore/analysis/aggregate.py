from __future__ import annotations

import csv
import math
from pathlib import Path

# Non-_ms numeric columns worth aggregating when present in the main CSV.
EXTRA_NUMERIC_COLUMNS = {
    "batch_raw_ids",
    "batch_unique_ids",
    "batch_dedup_ratio",
    "samples_per_sec",
    "batches_per_sec",
    "loss",
    "prefetch_depth",
    "prefetch_issued_batches",
    "prefetch_consumed_batches",
    "prefetch_pending_batches",
    "prefetch_ready_batches",
    "prefetch_total_ids",
    "prefetch_consumed_total_ids",
}

AGG_FIELDNAMES = ["metric", "mean", "p50", "p95", "max"]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    weight = rank - lo
    return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_measured_rows(path: Path) -> list[dict[str, str]]:
    """CSV rows with warmup rows (warmup_excluded == '1') dropped; all rows if none measured."""
    rows = _load_csv_rows(path)
    measured = [row for row in rows if row.get("warmup_excluded", "") != "1"]
    return measured or rows


def mean_column(rows: list[dict[str, str]], field: str) -> float:
    """Mean of a numeric CSV column, skipping blanks/non-numeric; 0.0 if empty."""
    values = [v for v in (_to_float(row.get(field, "")) for row in rows) if v is not None]
    return sum(values) / len(values) if values else 0.0


def _round2(value: float) -> float:
    return round(value, 2)


def _stats_row(metric: str, values: list[float]) -> dict[str, float | int | str]:
    return {
        "metric": metric,
        "mean": _round2(sum(values) / len(values)),
        "p50": _round2(_percentile(values, 50.0)),
        "p95": _round2(_percentile(values, 95.0)),
        "max": _round2(max(values)),
    }


def aggregate_torchrec_main_csv(path: Path) -> list[dict[str, float | int | str]]:
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError(f"no rows found in torchrec main csv: {path}")

    numeric_columns = [
        name
        for name in rows[0].keys()
        if name.endswith("_ms") or name in EXTRA_NUMERIC_COLUMNS
    ]

    n = float(len(rows))
    result: list[dict[str, float | int | str]] = [
        {
            "metric": "row_count",
            "mean": _round2(n),
            "p50": _round2(n),
            "p95": _round2(n),
            "max": _round2(n),
        }
    ]
    for column in sorted(numeric_columns):
        values = []
        for row in rows:
            parsed = _to_float(row.get(column, ""))
            if parsed is not None:
                values.append(parsed)
        if not values:
            continue
        result.append(_stats_row(column, values))
    return result


def write_aggregate_csv(
    path: Path,
    aggregate: list[dict[str, float | int | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDNAMES)
        writer.writeheader()
        for row in aggregate:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "mean": f"{float(row['mean']):.2f}",
                    "p50": f"{float(row['p50']):.2f}",
                    "p95": f"{float(row['p95']):.2f}",
                    "max": f"{float(row['max']):.2f}",
                }
            )


def aggregate_csv_to_wide(path: Path) -> dict[str, str]:
    """Pivot long-format agg CSV back to wide keys for grid summaries."""
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError(f"expected rows in csv: {path}")
    # Legacy wide format: one row with metric_mean / metric_p50 columns.
    if "metric" not in rows[0]:
        return rows[0]
    wide: dict[str, str] = {}
    for row in rows:
        metric = row["metric"]
        for stat in ("mean", "p50", "p95", "max"):
            wide[f"{metric}_{stat}"] = row[stat]
    return wide
