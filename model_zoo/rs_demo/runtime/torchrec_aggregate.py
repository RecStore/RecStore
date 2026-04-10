from __future__ import annotations

import csv
import math
from pathlib import Path


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


def aggregate_torchrec_main_csv(path: Path) -> dict[str, float | int]:
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError(f"no rows found in torchrec main csv: {path}")

    result: dict[str, float | int] = {"row_count": len(rows)}
    numeric_columns = [name for name in rows[0].keys() if name.endswith("_ms")]

    for column in numeric_columns:
        values = []
        for row in rows:
            parsed = _to_float(row.get(column, ""))
            if parsed is not None:
                values.append(parsed)
        if not values:
            continue
        result[f"{column}_mean"] = sum(values) / len(values)
        result[f"{column}_p50"] = _percentile(values, 50.0)
        result[f"{column}_p95"] = _percentile(values, 95.0)
        result[f"{column}_max"] = max(values)

    return result


def write_aggregate_csv(path: Path, aggregate: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(aggregate.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(aggregate)
